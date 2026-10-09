"""
Model Backends for Text Generation

Supports multiple backends:
1. Local HuggingFace models
2. A generic OpenAI-compatible API

Environment variables:
- GEN_BACKEND: "local" or "api"
- LOCAL_MODEL_PATH: Local model path (for local backend)
- API_MODEL: Model name for API backends
- API_MODELS: Optional comma-separated model pool used by paired-data synthesis
- API_KEY: Shared key for the OpenAI-compatible generation API
- API_BASE_URL: Shared base URL for the OpenAI-compatible generation API
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, List, Dict, Any

from src.config import get_api_base_url, get_api_key, get_api_models


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
    """Unsteered local generation using the shared optimized HF runtime."""
    
    def __init__(
        self, 
        model_path: str, 
        device: str = "auto",
        torch_dtype: str = "auto",
        use_flash_attn: bool = True,
        use_bettertransformer: bool = False,
        attention_backend: str = "auto",
        allow_tf32: bool = True,
    ):
        if use_bettertransformer:
            raise ValueError(
                "BetterTransformer is no longer a supported runtime path; "
                "use SDPA or FlashAttention 2"
            )
        from .steered import SteeredGenerator

        resolved_path = Path(model_path).expanduser().resolve()
        if not resolved_path.is_dir():
            raise FileNotFoundError(f"Local model directory not found: {resolved_path}")
        if not use_flash_attn and attention_backend == "auto":
            attention_backend = "sdpa"
        self._engine = SteeredGenerator(
            str(resolved_path),
            device=device,
            torch_dtype=torch_dtype,
            attention_backend=attention_backend,
            allow_tf32=allow_tf32,
        )
        self._model_name = resolved_path.name

    @property
    def model(self):
        return self._engine.model

    @property
    def tokenizer(self):
        return self._engine.tokenizer
    
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
        return self.generate_batch(
            [prompt],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            **kwargs,
        )[0]
    
    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 150,
        temperature: float = 0.7,
        **kwargs
    ) -> List[str]:
        """Batch generation with KV cache and correct left-padding slicing."""
        from .steered import SteeredGenerationConfig

        texts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            try:
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except (AttributeError, TypeError, ValueError):
                text = prompt
            texts.append(text)
        config = SteeredGenerationConfig(
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=float(kwargs.pop("top_p", 0.95)),
            seed=kwargs.pop("seed", 8236),
            use_cache=True,
            max_prompt_tokens=int(kwargs.pop("max_prompt_tokens", 2048)),
        )
        if kwargs:
            raise TypeError(
                f"Unsupported local generation options: {sorted(kwargs)}"
            )
        return [
            text.strip()
            for text in self._engine.generate_batch(
                texts,
                steering=None,
                config=config,
                show_progress=False,
            )
        ]


class APIGenerator(BaseGenerator):
    """Generator using any OpenAI-compatible chat-completions API."""
    
    def __init__(
        self,
        api_key: str,
        model_name: str,
        base_url: str,
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
    
    def _extract_text(self, response) -> str:
        choices = getattr(response, "choices", None)
        if not choices:
            raise RuntimeError(
                f"{self._model_name}: API response contained no choices"
            )
        msg = choices[0].message
        text = (getattr(msg, "content", None) or "").strip()
        if not text:
            raise RuntimeError(
                f"{self._model_name}: API response contained no final content"
            )
        return text

    def _inject_no_thinking(self, kwargs: dict) -> dict:
        # MiniMax-M2.5 on the configured compatible endpoint only permits
        # enable_thinking=true.  Omitting provider-specific controls lets that
        # model use its supported default.  Other configured families accept
        # the no-thinking controls below.
        if "minimax" in self._model_name.casefold():
            return dict(kwargs)
        eb = dict(kwargs.get("extra_body") or {})
        eb.setdefault("enable_thinking", False)
        eb.setdefault("thinking", {"type": "disabled"})
        eb.setdefault("chat_template_kwargs", {"enable_thinking": False})
        kwargs = dict(kwargs)
        kwargs["extra_body"] = eb
        return kwargs

    def generate(
        self, 
        prompt: str, 
        max_new_tokens: int = 150, 
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        try:
            kwargs = self._inject_no_thinking(kwargs)
            response = self.client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs
            )
            return self._extract_text(response)
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
            kwargs = self._inject_no_thinking(kwargs)
            response = self.client.chat.completions.create(
                model=self._model_name,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                **kwargs
            )
            return self._extract_text(response)
        except Exception as e:
            print(f"API error: {e}")
            raise


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
        backend: "local" or "api"
        model_path: Path to local model (for local backend)
        model_name: Model name (for API backends)
        api_key: API key (for API backends)
        **kwargs: Additional arguments passed to generator
    
    Returns:
        Configured generator instance
    
    Examples:
        # Using environment variables
        export GEN_BACKEND=api
        export API_KEY=your_key
        export API_BASE_URL=https://api.example.com/v1
        export API_MODEL=teacher-model
        generator = create_generator()
        
        # Explicit local model
        generator = create_generator(backend="local", model_path="../../models/Qwen2.5-3B-Instruct")
        
        # Explicit OpenAI-compatible API
        generator = create_generator(backend="api", api_key="...", model_name="...")
    """
    # Determine backend
    backend = backend or os.environ.get("GEN_BACKEND", "local")
    backend = backend.lower()
    
    if backend == "local":
        # Get model path
        model_path = model_path or os.environ.get("LOCAL_MODEL_PATH")
        if not model_path:
            raise ValueError(
                "Local model path required. "
                "Set LOCAL_MODEL_PATH environment variable or pass model_path parameter."
            )
        return LocalModelGenerator(model_path, **kwargs)
    
    elif backend == "api":
        api_key = api_key or get_api_key()
        model_name = model_name or os.environ.get("API_MODEL", "gpt-4o-mini")
        base_url = get_api_base_url()
        return APIGenerator(
            api_key=api_key, 
            model_name=model_name, 
            base_url=base_url,
            **kwargs
        )
    
    else:
        raise ValueError(
            f"Unknown backend: {backend}. Use 'local' or 'api'."
        )


def get_config_from_env() -> Dict[str, Any]:
    """Get all configuration from environment variables."""
    return {
        "backend": os.environ.get("GEN_BACKEND", "local"),
        "model_path": os.environ.get("LOCAL_MODEL_PATH"),
        "model_name": os.environ.get("API_MODEL"),
        "model_pool": get_api_models(),
        "api_key": get_api_key(required=False),
        "api_base_url": get_api_base_url(),
    }


def print_config_help():
    """Print help message for configuration."""
    help_text = """
Model Configuration
===================

Environment Variables:
----------------------
GEN_BACKEND          Backend type: "local" or "api" (default: local)
LOCAL_MODEL_PATH     Path to local model (required for local backend)
API_MODEL            Model name for API backends
API_MODELS           Optional comma-separated pool used by paired-data synthesis
API_KEY              Key for the OpenAI-compatible generation API
API_BASE_URL         Base URL for the OpenAI-compatible generation API

Example .env file:
------------------
# Local model
GEN_BACKEND=local
LOCAL_MODEL_PATH=/path/to/models/Qwen3-4B

# Or an OpenAI-compatible API
# GEN_BACKEND=api
# API_KEY=your_api_key_here
# API_BASE_URL=https://api.example.com/v1
# API_MODELS=teacher-a,teacher-b
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
