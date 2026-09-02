"""
Steered Text Generation.

Generates text with steering applied to hidden states.
Provides a unified interface for generation with any steering method.
"""

import os
import time
import torch
from typing import Dict, List, Optional
from dataclasses import dataclass, replace

from .generator import GenerationConfig
from ..runtime import (
    configure_torch_runtime,
    input_device,
    model_load_kwargs,
    seeded_rng,
    stable_seed,
)


@dataclass
class SteeredGenerationConfig(GenerationConfig):
    """Configuration for steered generation."""
    steering_strength: float = 1.0
    steering_direction: str = "human"  # "human" or "ai"
    apply_to_positions: str = "decode_only"
    strength_unit: str = "projection_std"
    seed: Optional[int] = 8236
    paired_sampling: bool = True
    use_cache: bool = True
    cache_implementation: str = "dynamic"
    max_prompt_tokens: int = 2048
    pad_to_multiple_of: Optional[int] = None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.steering_strength < 0:
            raise ValueError("steering_strength must be non-negative")
        if self.steering_direction not in {"human", "ai"}:
            raise ValueError("steering_direction must be 'human' or 'ai'")
        if self.apply_to_positions not in {"decode_only", "all", "last"}:
            raise ValueError("unsupported steering position mode")
        if self.strength_unit not in {"projection_std", "absolute"}:
            raise ValueError("unsupported steering strength unit")
        if self.max_prompt_tokens < 1:
            raise ValueError("max_prompt_tokens must be positive")
        if self.cache_implementation not in {"dynamic", "static"}:
            raise ValueError(
                "cache_implementation must be 'dynamic' or 'static'"
            )
        if (
            self.pad_to_multiple_of is not None
            and self.pad_to_multiple_of < 1
        ):
            raise ValueError("pad_to_multiple_of must be positive or None")


class SteeredGenerator:
    """
    Generator with steering applied during generation.
    
    This class wraps a HuggingFace model and applies steering to hidden states
    during the forward pass using hooks.
    
    Example:
        from src.steering import MeanDiffSteering, SteeringConfig
        from src.generation import SteeredGenerator
        
        steering = MeanDiffSteering.load("path/to/model.pt")
        generator = SteeredGenerator("Qwen/Qwen2.5-3B-Instruct")
        
        # Generate with steering
        text = generator.generate(
            "The weather today is",
            steering=steering,
            config=SteeredGenerationConfig(
                steering_strength=1.0,
                steering_direction="human"
            )
        )
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        torch_dtype: str = "auto",
        attention_backend: str = "auto",
        allow_tf32: bool = True,
    ):
        """
        Args:
            model_name: HuggingFace model name
            device: Device to use
            torch_dtype: ``auto``, ``bfloat16``, ``float16``, or ``float32``
            attention_backend: ``auto``, ``sdpa``, ``flash_attention_2``,
                or ``eager``. Auto falls back to SDPA when FlashAttention is
                unavailable.
        """
        self.model_name = model_name
        self.device = device
        self.torch_dtype = torch_dtype
        self.attention_backend = attention_backend
        self.allow_tf32 = allow_tf32
        
        self._model = None
        self._tokenizer = None
        self.model_load_info: Dict = {}
        self.last_generation_stats: Dict = {}
        self.last_comparison_stats: Dict[float, Dict] = {}
    
    @property
    def model(self):
        """Lazy load model."""
        if self._model is None:
            self._load_model()
        return self._model
    
    @property
    def tokenizer(self):
        """Lazy load tokenizer."""
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer
    
    def _load_model(self):
        """Load model and tokenizer."""
        from transformers import AutoModelForCausalLM, AutoTokenizer

        configure_torch_runtime(allow_tf32=self.allow_tf32)
        extra_kw = {}
        if os.path.isdir(self.model_name):
            extra_kw["local_files_only"] = True
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            **extra_kw,
        )
        if self._tokenizer.pad_token_id is None:
            if self._tokenizer.eos_token_id is None:
                raise ValueError(
                    f"Tokenizer for {self.model_name} has neither pad nor EOS token"
                )
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

        load_kwargs, load_info = model_load_kwargs(
            device=self.device,
            dtype=self.torch_dtype,
            attention_backend=self.attention_backend,
            trust_remote_code=True,
        )
        try:
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs,
                **extra_kw,
            )
        except Exception as error:
            can_fallback = (
                self.attention_backend == "auto"
                and load_info["resolved_attention_backend"]
                == "flash_attention_2"
                and (
                    "flash" in str(error).lower()
                    or isinstance(error, (ImportError, ModuleNotFoundError))
                )
            )
            if not can_fallback:
                raise
            load_kwargs["attn_implementation"] = "sdpa"
            load_info["resolved_attention_backend"] = "sdpa"
            load_info["attention_fallback_reason"] = (
                f"{type(error).__name__}: {str(error)[:240]}"
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                **load_kwargs,
                **extra_kw,
            )
        self._model.eval()
        self.model_load_info = load_info

    def _prepare_inputs(self, prompts: List[str], config: SteeredGenerationConfig):
        if not prompts or any(
            not isinstance(prompt, str) or not prompt
            for prompt in prompts
        ):
            raise ValueError("prompts must be a non-empty list of non-empty strings")
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.max_prompt_tokens,
            pad_to_multiple_of=config.pad_to_multiple_of,
        )
        device = input_device(self.model)
        return {key: value.to(device) for key, value in inputs.items()}

    def _steering_device(self, layer: int) -> torch.device:
        from ..steering.hooks import get_layer_module

        module = get_layer_module(self.model, layer)
        try:
            return next(module.parameters()).device
        except StopIteration:
            return input_device(self.model)

    def _do_generate(
        self,
        inputs,
        config: SteeredGenerationConfig,
        *,
        seed: Optional[int],
    ):
        """Run one deterministic or sampled batch and record throughput."""
        device = inputs["input_ids"].device
        do_sample = config.temperature is not None and config.temperature > 0
        generation_kwargs = {
            **inputs,
            "max_new_tokens": config.max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "use_cache": config.use_cache,
        }
        if config.use_cache:
            generation_kwargs["cache_implementation"] = (
                config.cache_implementation
            )
        if do_sample:
            generation_kwargs.update({
                "temperature": config.temperature,
                "top_p": config.top_p,
            })

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with seeded_rng(seed, device), torch.inference_mode():
            outputs = self.model.generate(**generation_kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        generated_tokens = int(
            sum(
                (row != self.tokenizer.pad_token_id).sum().item()
                for row in outputs[:, inputs["input_ids"].shape[1]:]
            )
        )
        self.last_generation_stats = {
            "batch_size": int(inputs["input_ids"].shape[0]),
            "prompt_width_tokens": int(inputs["input_ids"].shape[1]),
            "generated_tokens": generated_tokens,
            "elapsed_seconds": elapsed,
            "tokens_per_second": generated_tokens / max(elapsed, 1e-9),
            "seed": seed,
            "use_cache": config.use_cache,
            "cache_implementation": (
                config.cache_implementation if config.use_cache else None
            ),
            "pad_to_multiple_of": config.pad_to_multiple_of,
        }
        return outputs

    def _generate_batch(
        self,
        prompts: List[str],
        *,
        steering,
        config: SteeredGenerationConfig,
        seed: Optional[int],
    ) -> List[str]:
        from ..steering.base import SteeringConfig
        from ..steering.hooks import steering_applied

        inputs = self._prepare_inputs(prompts, config)
        hook_stats = None
        if steering is not None and config.steering_strength != 0:
            steering = steering.to(self._steering_device(steering.layer))
            steering_config = SteeringConfig(
                layer=steering.layer,
                strength=config.steering_strength,
                direction=config.steering_direction,
                strength_unit=config.strength_unit,
            )
            with steering_applied(
                self.model,
                steering,
                steering_config,
                config.apply_to_positions,
            ) as hook:
                outputs = self._do_generate(inputs, config, seed=seed)
                hook_stats = dict(hook.stats)
        else:
            outputs = self._do_generate(inputs, config, seed=seed)

        self.last_generation_stats["steering"] = {
            "enabled": hook_stats is not None,
            "layer": getattr(steering, "layer", None),
            "strength": config.steering_strength,
            "direction": config.steering_direction,
            "strength_unit": config.strength_unit,
            "position_mode": config.apply_to_positions,
            "hook_stats": hook_stats,
        }
        new_tokens = outputs[:, inputs["input_ids"].shape[1]:]
        return self.tokenizer.batch_decode(
            new_tokens,
            skip_special_tokens=True,
        )

    def generate(
        self,
        prompt: str,
        steering=None,
        config: Optional[SteeredGenerationConfig] = None,
    ) -> str:
        """
        Generate text with optional steering.
        
        Args:
            prompt: Input prompt
            steering: Steering method (BaseSteering subclass)
            config: Generation config
            
        Returns:
            Generated continuation text
        """
        config = config or SteeredGenerationConfig()
        return self._generate_batch(
            [prompt],
            steering=steering,
            config=config,
            seed=config.seed,
        )[0]

    def generate_comparison(
        self,
        prompt: str,
        steering,
        strengths: Optional[List[float]] = None,
        config: Optional[SteeredGenerationConfig] = None,
    ) -> Dict[float, str]:
        """
        Generate text at different steering strengths.
        
        Args:
            prompt: Input prompt
            steering: Steering method
            strengths: List of strength values to try
            config: Generation config (strength will be overridden)
            
        Returns:
            Dict mapping strength -> generated text
        """
        if strengths is None:
            strengths = [-2.0, -1.0, 0.0, 1.0, 2.0]
        batched = self.generate_batch_comparison(
            [prompt],
            steering,
            strengths,
            config,
        )
        return {strength: texts[0] for strength, texts in batched.items()}

    def generate_batch_comparison(
        self,
        prompts: List[str],
        steering,
        strengths: Optional[List[float]] = None,
        config: Optional[SteeredGenerationConfig] = None,
    ) -> Dict[float, List[str]]:
        """Generate a prompt batch over strengths with paired RNG streams."""
        config = config or SteeredGenerationConfig()
        if strengths is None:
            strengths = [-2.0, -1.0, 0.0, 1.0, 2.0]
        strengths = list(dict.fromkeys(float(value) for value in strengths))
        if not strengths:
            raise ValueError("strengths must contain at least one condition")
        results: Dict[float, List[str]] = {}
        stats: Dict[float, Dict] = {}
        for condition_index, strength in enumerate(strengths):
            direction = "human" if strength < 0 else "ai"
            condition_seed = (
                config.seed
                if config.paired_sampling
                else stable_seed(config.seed or 0, condition_index, strength)
            )
            condition_config = replace(
                config,
                steering_strength=abs(strength),
                steering_direction=direction,
                seed=condition_seed,
            )
            results[strength] = self._generate_batch(
                prompts,
                steering=steering if strength != 0 else None,
                config=condition_config,
                seed=condition_seed,
            )
            stats[strength] = dict(self.last_generation_stats)
        self.last_comparison_stats = stats
        return results

    def generate_batch(
        self,
        prompts: List[str],
        steering=None,
        config: Optional[SteeredGenerationConfig] = None,
        show_progress: bool = True,
    ) -> List[str]:
        """
        Generate text for multiple prompts.
        
        Args:
            prompts: List of input prompts
            steering: Optional steering method
            config: Generation config
            show_progress: Show progress bar
            
        Returns:
            List of generated texts
        """
        del show_progress  # Batching replaces the old serial progress loop.
        config = config or SteeredGenerationConfig()
        return self._generate_batch(
            prompts,
            steering=steering,
            config=config,
            seed=config.seed,
        )
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "hidden_size": getattr(
                getattr(self.model.config, "text_config", self.model.config),
                "hidden_size",
            ),
            "num_layers": getattr(
                getattr(self.model.config, "text_config", self.model.config),
                "num_hidden_layers",
            ),
            "device": str(input_device(self.model)),
            "dtype": str(self.model.dtype),
            "load": self.model_load_info,
        }


def quick_steer_demo(
    model_name: Optional[str] = None,
    steering_path: Optional[str] = None,
    prompt: str = "The meaning of life is",
    strengths: List[float] = [-2.0, -1.0, 0.0, 1.0, 2.0],
):
    """
    Quick demo of steered generation.
    
    Args:
        model_name: Model to use (default: from LOCAL_MODEL_PATH in .env)
        steering_path: Path to saved steering model
        prompt: Test prompt
        strengths: Steering strengths to test
    """
    from ..steering import MeanDiffSteering

    if model_name is None:
        from ..config import get_model_path
        model_name = get_model_path()
    
    print(f"Loading model: {model_name}")
    generator = SteeredGenerator(model_name)
    
    print(f"Model info: {generator.get_model_info()}")
    
    if steering_path:
        print(f"Loading steering from: {steering_path}")
        steering = MeanDiffSteering.load(steering_path)
    else:
        print("No steering model provided, generating baseline only")
        steering = None
    
    print(f"\nPrompt: {prompt}")
    print("-" * 50)
    
    if steering:
        results = generator.generate_comparison(prompt, steering, strengths)
        for strength, text in sorted(results.items()):
            direction = "human" if strength < 0 else ("baseline" if strength == 0 else "ai")
            print(f"\n[{direction}, strength={strength}]")
            print(text[:200] + "..." if len(text) > 200 else text)
    else:
        text = generator.generate(prompt)
        print(f"\n[baseline]")
        print(text)
