"""
Hidden States Extraction from LLM

Core functionality for extracting hidden states from transformer layers.
Supports continuation-specific extraction and multiple pooling strategies.
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ExtractionResult:
    """Result of hidden state extraction."""
    hidden_states: Dict[int, torch.Tensor]  # layer -> tensor
    token_count: int
    prefix_token_count: Optional[int] = None
    continuation_token_count: Optional[int] = None


class HiddenStatesExtractor:
    """
    Extract hidden states from a causal LM.
    
    This is the foundational class for all representation-based operations.
    
    Example:
        extractor = HiddenStatesExtractor("../../models/Llama-3.1-8B-Instruct")
        result = extractor.extract_continuation(
            prefix="The weather is",
            continuation=" sunny and warm today.",
            pooling="full_mean",
            layers=[16, 24, 32]
        )
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model: Optional[AutoModelForCausalLM] = None,
        tokenizer: Optional[AutoTokenizer] = None,
    ):
        """
        Initialize extractor.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use
            dtype: Model dtype
            model: Optionally pass existing model (avoids reloading)
            tokenizer: Optionally pass existing tokenizer
        """
        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
        else:
            print(f"Loading model: {model_name}")
            # Detect local paths so newer huggingface_hub doesn't treat them as repo IDs
            extra_kw = {}
            if os.path.isdir(model_name):
                extra_kw["local_files_only"] = True
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, **extra_kw
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device,
                trust_remote_code=True,
                **extra_kw,
            )
        
        self.model.eval()
        self.num_layers = self.model.config.num_hidden_layers
        self.hidden_size = self.model.config.hidden_size
        print(f"Model ready: {self.num_layers} layers, hidden_size={self.hidden_size}")
    
    @torch.no_grad()
    def extract(
        self,
        text: str,
        aggregation: str = "mean",
        layers: Optional[List[int]] = None,
    ) -> Dict[int, torch.Tensor]:
        """
        Extract hidden states for a single text.
        
        Args:
            text: Input text
            aggregation: "mean", "last", "first", "max", or "none"
            layers: Which layers to extract (None = all)
            
        Returns:
            Dict mapping layer index to hidden state tensor
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=2048
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        
        if layers is None:
            layers = list(range(self.num_layers + 1))
        
        result = {}
        for layer_idx in layers:
            if layer_idx > len(hidden_states) - 1:
                continue
            
            hs = hidden_states[layer_idx][0]  # (seq_len, hidden_size)
            
            if aggregation == "mean":
                agg = hs.mean(dim=0)
            elif aggregation == "last":
                agg = hs[-1]
            elif aggregation == "first":
                agg = hs[0]
            elif aggregation == "max":
                agg = hs.max(dim=0).values
            elif aggregation == "none":
                agg = hs
            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")
            
            result[layer_idx] = agg.cpu()
        
        return result
    
    @torch.no_grad()
    def extract_continuation(
        self,
        prefix: str,
        continuation: str,
        pooling: str = "full_mean",
        layers: Optional[List[int]] = None,
    ) -> ExtractionResult:
        """
        Extract hidden states for continuation region only (teacher forcing).
        
        Key extraction strategy:
        - Tokenize prefix + continuation together
        - Run forward pass (teacher forcing)
        - Only extract hidden states from continuation tokens
        
        Args:
            prefix: The prefix text
            continuation: The continuation text
            pooling: "full_mean", "middle", "last", "first", or "none"
            layers: Which layers to extract (None = all)
            
        Returns:
            ExtractionResult with hidden states for continuation
        """
        # Find continuation boundary
        prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=True)
        full_text = prefix + continuation
        
        continuation_start = len(prefix_tokens) - 1  # Adjust for overlap
        
        # Forward pass
        inputs = self.tokenizer(
            full_text, return_tensors="pt", truncation=True, max_length=2048
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        
        seq_len = inputs["input_ids"].shape[1]
        continuation_end = seq_len
        
        if layers is None:
            layers = list(range(self.num_layers + 1))
        
        result_states = {}
        for layer_idx in layers:
            if layer_idx > len(hidden_states) - 1:
                continue
            
            hs = hidden_states[layer_idx][0]  # (seq_len, hidden_size)
            cont_hs = hs[continuation_start:continuation_end]
            
            pooled = self._apply_pooling(cont_hs, pooling)
            result_states[layer_idx] = pooled.cpu()
        
        return ExtractionResult(
            hidden_states=result_states,
            token_count=seq_len,
            prefix_token_count=continuation_start,
            continuation_token_count=continuation_end - continuation_start,
        )
    
    def _apply_pooling(self, hidden_states: torch.Tensor, pooling: str) -> torch.Tensor:
        """Apply pooling to token-level hidden states."""
        seq_len = hidden_states.shape[0]
        
        if pooling == "full_mean":
            return hidden_states.mean(dim=0)
        elif pooling == "middle":
            start = int(seq_len * 0.3)
            end = int(seq_len * 0.7)
            if end <= start:
                end = start + 1
            return hidden_states[start:end].mean(dim=0)
        elif pooling == "last":
            return hidden_states[-1]
        elif pooling == "first":
            return hidden_states[0]
        elif pooling == "none":
            return hidden_states
        else:
            raise ValueError(f"Unknown pooling: {pooling}")
