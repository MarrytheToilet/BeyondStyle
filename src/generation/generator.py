"""Continuation generation with LLMs."""

import os
from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from tqdm import tqdm

from .prompts import PromptTemplate, get_prompt_templates


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 200
    num_samples: int = 3


@dataclass
class Continuation:
    """A single continuation sample."""
    prefix_id: str
    prefix_text: str
    condition: str
    continuation_text: str
    sample_idx: int
    language: str
    model: str
    metadata: Dict = field(default_factory=dict)


class BaseGenerator(ABC):
    """Base class for text generators."""
    
    @abstractmethod
    def generate(self, prompt: str, config: GenerationConfig) -> str:
        """Generate continuation for a prompt."""
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """Get the name of the model."""
        pass


class OpenAIGenerator(BaseGenerator):
    """Generator using OpenAI API."""
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client
    
    def generate(self, prompt: str, config: GenerationConfig) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.max_new_tokens,
        )
        return response.choices[0].message.content
    
    def get_model_name(self) -> str:
        return self.model


class AnthropicGenerator(BaseGenerator):
    """Generator using Anthropic API."""
    
    def __init__(self, model: str = "claude-3-sonnet-20240229"):
        self.model = model
        self._client = None
    
    @property
    def client(self):
        if self._client is None:
            from anthropic import Anthropic
            self._client = Anthropic()
        return self._client
    
    def generate(self, prompt: str, config: GenerationConfig) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=config.max_new_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
    
    def get_model_name(self) -> str:
        return self.model


class HuggingFaceGenerator(BaseGenerator):
    """Generator using local HuggingFace models."""
    
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None
    
    def _load_model(self):
        if self._model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16,
                device_map=self.device,
            )
    
    def generate(self, prompt: str, config: GenerationConfig) -> str:
        self._load_model()
        
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        
        outputs = self._model.generate(
            **inputs,
            max_new_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            do_sample=True,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        
        # Decode only the new tokens
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    def get_model_name(self) -> str:
        return self.model_name.split("/")[-1]


class ContinuationGenerator:
    """Generate continuations across conditions."""
    
    def __init__(
        self,
        generator: BaseGenerator,
        config: Optional[GenerationConfig] = None,
    ):
        self.generator = generator
        self.config = config or GenerationConfig()
    
    def generate_for_prefix(
        self,
        prefix_id: str,
        prefix_text: str,
        language: str,
    ) -> List[Continuation]:
        """Generate continuations for a single prefix across all conditions."""
        templates = get_prompt_templates(language)
        continuations = []
        
        for template in templates:
            prompt = template.format(prefix_text)
            
            for sample_idx in range(self.config.num_samples):
                text = self.generator.generate(prompt, self.config)
                
                cont = Continuation(
                    prefix_id=prefix_id,
                    prefix_text=prefix_text,
                    condition=template.name,
                    continuation_text=text,
                    sample_idx=sample_idx,
                    language=language,
                    model=self.generator.get_model_name(),
                )
                continuations.append(cont)
        
        return continuations
    
    def generate_batch(
        self,
        prefixes: List[Dict],
        language: str,
        show_progress: bool = True,
    ) -> List[Continuation]:
        """Generate continuations for a batch of prefixes."""
        all_continuations = []
        
        iterator = tqdm(prefixes, desc="Generating") if show_progress else prefixes
        
        for prefix in iterator:
            conts = self.generate_for_prefix(
                prefix_id=prefix["id"],
                prefix_text=prefix["text"],
                language=language,
            )
            all_continuations.extend(conts)
        
        return all_continuations


def create_generator(
    model_type: str,
    model_name: str,
    **kwargs
) -> BaseGenerator:
    """Factory function to create a generator."""
    if model_type == "openai":
        return OpenAIGenerator(model=model_name)
    elif model_type == "anthropic":
        return AnthropicGenerator(model=model_name)
    elif model_type == "huggingface":
        return HuggingFaceGenerator(model_name=model_name, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
