"""
Model Backends for Text Generation

Supports multiple backends:
1. Local HuggingFace models
2. SiliconFlow API (OpenAI-compatible)
3. OpenAI API

Environment variables:
- BEYONDSTYLE_BACKEND: "local", "siliconflow", or "openai"
- BEYONDSTYLE_MODEL_PATH: Local model path (for local backend)
- BEYONDSTYLE_MODEL_NAME: Model name for API backends
- SILICONFLOW_API_KEY: API key for SiliconFlow
- SILICONFLOW_BASE_URL: SiliconFlow API base URL (default: https://api.siliconflow.cn/v1)
- OPENAI_API_KEY: API key for OpenAI
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Dict, Any


class BaseGenerator(ABC):
    """Abstract base class for text generators."""
    
    @abstractmethod
    def generate(
        self, 
        prompt: str, 
        max_new_tokens: int = 150, 
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate continuation for a prompt."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name/identifier."""
        pass
    
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        **kwargs
    ) -> List[str]:
        """Generate continuations for multiple prompts. Override for batch optimization."""
        return [self.generate(p, max_new_tokens, temperature, **kwargs) for p in prompts]


class LocalModelGenerator(BaseGenerator):
    """Generator using local HuggingFace models."""
    
    def __init__(
        self, 
        model_path: str, 
        device: str = "auto",
        torch_dtype: str = "float16",
        use_flash_attn: bool = True,
        use_bettertransformer: bool = False,
    ):
        """
        Initialize local model generator with acceleration options.
        
        Args:
            model_path: Path to HuggingFace model
            device: Device map ("auto", "cuda", "cpu")
            torch_dtype: Data type ("float16", "bfloat16", "float32")
            use_flash_attn: Use Flash Attention 2 or SDPA (recommended)
            use_bettertransformer: Use BetterTransformer optimization
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Handle relative paths
        model_path = str(Path(model_path).expanduser().resolve())
        
        print(f"Loading local model from {model_path}...")
        
        # Determine dtype
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype_val = dtype_map.get(torch_dtype, torch.float16)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, 
            trust_remote_code=True,
            local_files_only=True,
            padding_side="left"  # Required for batch generation with decoder-only models
        )
        
        # Model loading kwargs
        model_kwargs = {
            "torch_dtype": torch_dtype_val,
            "device_map": device,
            "trust_remote_code": True,
            "local_files_only": True,
        }
        
        # Try to enable Flash Attention 2 or SDPA
        if use_flash_attn:
            # First try flash_attention_2
            try:
                model_kwargs["attn_implementation"] = "flash_attention_2"
                self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
                print("✓ Using Flash Attention 2")
            except Exception as e:
                # Fall back to SDPA (PyTorch native)
                try:
                    model_kwargs["attn_implementation"] = "sdpa"
                    self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
                    print("✓ Using SDPA (Scaled Dot Product Attention)")
                except Exception as e2:
                    # Fall back to eager
                    model_kwargs.pop("attn_implementation", None)
                    self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
                    print("⚠ Using default attention (no acceleration)")
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        
        # Optional: BetterTransformer (alternative optimization)
        if use_bettertransformer:
            try:
                self.model = self.model.to_bettertransformer()
                print("✓ Using BetterTransformer")
            except Exception as e:
                print(f"⚠ BetterTransformer not available: {e}")
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self._model_name = Path(model_path).name
        print(f"Model loaded: {self._model_name}")
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    def generate(
        self, 
        prompt: str, 
        max_new_tokens: int = 150, 
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        import torch
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except:
            text = prompt
        
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        return response.strip()
    
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        **kwargs
    ) -> List[str]:
        """Batch generation with padding for better GPU utilization."""
        import torch
        
        # Format all prompts
        texts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            try:
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except:
                text = prompt
            texts.append(text)
        
        # Tokenize with padding
        inputs = self.tokenizer(
            texts, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        # Decode each output
        results = []
        for i, output in enumerate(outputs):
            input_len = inputs['input_ids'][i].ne(self.tokenizer.pad_token_id).sum()
            response = self.tokenizer.decode(
                output[input_len:], 
                skip_special_tokens=True
            )
            results.append(response.strip())
        
        return results


class APIGenerator(BaseGenerator):
    """Generator using OpenAI-compatible APIs (SiliconFlow, OpenAI, etc.)."""
    
    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str = "https://api.siliconflow.cn/v1",
        timeout: int = 60
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
            max_retries=5,
        )
        self._model_name = model_name
        self.base_url = base_url
        
        print(f"API Generator initialized: {model_name} @ {base_url}")
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    def generate(
        self, 
        prompt: str, 
        max_new_tokens: int = 150, 
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API error: {e}")
            raise

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate using proper chat messages format (system/user/assistant roles)."""
        try:
            response = self.client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"API error: {e}")
            raise


class SiliconFlowGenerator(APIGenerator):
    """Generator specifically for SiliconFlow API."""
    
    # Popular models on SiliconFlow
    POPULAR_MODELS = [
        "Qwen/Qwen2.5-7B-Instruct",
        "Qwen/Qwen2.5-14B-Instruct",
        "Qwen/Qwen2.5-32B-Instruct",
        "deepseek-ai/DeepSeek-V2.5",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "meta-llama/Meta-Llama-3.1-70B-Instruct",
    ]
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "Qwen/Qwen2.5-7B-Instruct",
        timeout: int = 180
    ):
        api_key = api_key or os.environ.get("SILICONFLOW_API_KEY")
        if not api_key:
            raise ValueError(
                "SiliconFlow API key required. "
                "Set SILICONFLOW_API_KEY environment variable or pass api_key parameter."
            )
        
        base_url = os.environ.get(
            "SILICONFLOW_BASE_URL", 
            "https://api.siliconflow.cn/v1"
        )
        
        super().__init__(
            api_key=api_key,
            model_name=model_name,
            base_url=base_url,
            timeout=timeout
        )


def create_generator(
    backend: Optional[str] = None,
    model_path: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs
) -> BaseGenerator:
    """
    Factory function to create appropriate generator based on configuration.
    
    Priority (highest to lowest):
    1. Explicit parameters
    2. Environment variables
    3. Defaults
    
    Args:
        backend: "local", "siliconflow", or "openai"
        model_path: Path to local model (for local backend)
        model_name: Model name (for API backends)
        api_key: API key (for API backends)
        **kwargs: Additional arguments passed to generator
    
    Returns:
        Configured generator instance
    
    Examples:
        # Using environment variables
        export BEYONDSTYLE_BACKEND=siliconflow
        export SILICONFLOW_API_KEY=your_key
        export BEYONDSTYLE_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct
        generator = create_generator()
        
        # Explicit local model
        generator = create_generator(backend="local", model_path="../../models/Qwen2.5-3B-Instruct")
        
        # Explicit API
        generator = create_generator(backend="siliconflow", api_key="...", model_name="...")
    """
    # Determine backend
    backend = backend or os.environ.get("BEYONDSTYLE_BACKEND", "local")
    backend = backend.lower()
    
    if backend == "local":
        # Get model path
        model_path = model_path or os.environ.get("BEYONDSTYLE_MODEL_PATH")
        if not model_path:
            raise ValueError(
                "Local model path required. "
                "Set BEYONDSTYLE_MODEL_PATH environment variable or pass model_path parameter."
            )
        return LocalModelGenerator(model_path, **kwargs)
    
    elif backend == "siliconflow":
        api_key = api_key or os.environ.get("SILICONFLOW_API_KEY")
        model_name = model_name or os.environ.get(
            "BEYONDSTYLE_MODEL_NAME", 
            "Qwen/Qwen2.5-7B-Instruct"
        )
        return SiliconFlowGenerator(api_key=api_key, model_name=model_name, **kwargs)
    
    elif backend == "openai":
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        model_name = model_name or os.environ.get("BEYONDSTYLE_MODEL_NAME", "gpt-4o-mini")
        base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
        return APIGenerator(
            api_key=api_key, 
            model_name=model_name, 
            base_url=base_url,
            **kwargs
        )
    
    else:
        raise ValueError(f"Unknown backend: {backend}. Use 'local', 'siliconflow', or 'openai'.")


def get_config_from_env() -> Dict[str, Any]:
    """Get all configuration from environment variables."""
    return {
        "backend": os.environ.get("BEYONDSTYLE_BACKEND", "local"),
        "model_path": os.environ.get("BEYONDSTYLE_MODEL_PATH"),
        "model_name": os.environ.get("BEYONDSTYLE_MODEL_NAME"),
        "siliconflow_api_key": os.environ.get("SILICONFLOW_API_KEY"),
        "siliconflow_base_url": os.environ.get("SILICONFLOW_BASE_URL"),
        "openai_api_key": os.environ.get("OPENAI_API_KEY"),
        "openai_base_url": os.environ.get("OPENAI_BASE_URL"),
    }


def print_config_help():
    """Print help message for configuration."""
    help_text = """
BeyondStyle Model Configuration
===============================

Environment Variables:
----------------------
BEYONDSTYLE_BACKEND      Backend type: "local", "siliconflow", or "openai" (default: local)
BEYONDSTYLE_MODEL_PATH   Path to local model (required for local backend)
BEYONDSTYLE_MODEL_NAME   Model name for API backends

For SiliconFlow:
SILICONFLOW_API_KEY      Your SiliconFlow API key
SILICONFLOW_BASE_URL     API base URL (default: https://api.siliconflow.cn/v1)

For OpenAI:
OPENAI_API_KEY           Your OpenAI API key
OPENAI_BASE_URL          API base URL (default: https://api.openai.com/v1)

Example .env file:
------------------
# Local model
BEYONDSTYLE_BACKEND=local
BEYONDSTYLE_MODEL_PATH=/home/user/models/Qwen2.5-3B-Instruct

# Or SiliconFlow API
# BEYONDSTYLE_BACKEND=siliconflow
# SILICONFLOW_API_KEY=your_api_key_here
# BEYONDSTYLE_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct

Popular SiliconFlow Models:
---------------------------
- Qwen/Qwen2.5-7B-Instruct
- Qwen/Qwen2.5-14B-Instruct
- Qwen/Qwen2.5-32B-Instruct
- deepseek-ai/DeepSeek-V2.5
- meta-llama/Meta-Llama-3.1-8B-Instruct
- meta-llama/Meta-Llama-3.1-70B-Instruct
"""
    print(help_text)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print_config_help()
    else:
        config = get_config_from_env()
        print("Current configuration:")
        for k, v in config.items():
            if "key" in k.lower() and v:
                v = v[:8] + "..." if len(v) > 8 else "***"
            print(f"  {k}: {v}")
