"""
Steered Text Generation.

Generates text with steering applied to hidden states.
Provides a unified interface for generation with any steering method.
"""

import torch
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path

from .generator import BaseGenerator, GenerationConfig


@dataclass
class SteeredGenerationConfig(GenerationConfig):
    """Configuration for steered generation."""
    steering_strength: float = 1.0
    steering_direction: str = "human"  # "human" or "ai"
    apply_to_positions: str = "all"  # "all", "last"


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
        torch_dtype: str = "float16",
    ):
        """
        Args:
            model_name: HuggingFace model name
            device: Device to use
            torch_dtype: Model dtype ("float16", "bfloat16", "float32")
        """
        self.model_name = model_name
        self.device = device
        self.torch_dtype = getattr(torch, torch_dtype)
        
        self._model = None
        self._tokenizer = None
    
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
        import os
        from transformers import AutoModelForCausalLM, AutoTokenizer

        extra_kw = {}
        if os.path.isdir(self.model_name):
            extra_kw["local_files_only"] = True
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, **extra_kw)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.torch_dtype,
            device_map=self.device,
            **extra_kw,
        )
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        steering = None,
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
        from ..steering.base import SteeringConfig
        from ..steering.hooks import steering_applied
        
        config = config or SteeredGenerationConfig()
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        if steering is not None:
            steering = steering.to(self.model.device)
            steering_config = SteeringConfig(
                layer=steering.layer,
                strength=config.steering_strength,
                direction=config.steering_direction,
            )
            
            with steering_applied(
                self.model, 
                steering, 
                steering_config,
                config.apply_to_positions
            ):
                outputs = self._do_generate(inputs, config)
        else:
            outputs = self._do_generate(inputs, config)
        
        # Decode new tokens only
        new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    def _do_generate(self, inputs, config):
        """Internal generate call."""
        return self.model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id,
        )
    
    @torch.no_grad()
    def generate_comparison(
        self,
        prompt: str,
        steering,
        strengths: List[float] = [-2.0, -1.0, 0.0, 1.0, 2.0],
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
        from ..steering.base import SteeringConfig
        from ..steering.hooks import steering_applied
        
        config = config or SteeredGenerationConfig()
        results = {}
        
        steering = steering.to(self.model.device)
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        for strength in strengths:
            if strength == 0:
                # Baseline - no steering
                outputs = self._do_generate(inputs, config)
            else:
                direction = "human" if strength < 0 else "ai"
                steering_config = SteeringConfig(
                    layer=steering.layer,
                    strength=abs(strength),
                    direction=direction,
                )
                
                with steering_applied(
                    self.model, 
                    steering, 
                    steering_config,
                    config.apply_to_positions
                ):
                    outputs = self._do_generate(inputs, config)
            
            new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            results[strength] = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        
        return results
    
    @torch.no_grad()
    def generate_batch(
        self,
        prompts: List[str],
        steering = None,
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
        from tqdm import tqdm
        
        results = []
        iterator = tqdm(prompts, desc="Generating") if show_progress else prompts
        
        for prompt in iterator:
            text = self.generate(prompt, steering=steering, config=config)
            results.append(text)
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        return {
            "model_name": self.model_name,
            "hidden_size": self.model.config.hidden_size,
            "num_layers": self.model.config.num_hidden_layers,
            "device": str(self.model.device),
            "dtype": str(self.model.dtype),
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
