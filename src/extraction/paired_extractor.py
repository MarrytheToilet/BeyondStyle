"""
Paired Hidden States Extraction

Extract and manage paired hidden states from human/AI continuations.
This is the core data structure for steering vector learning.
"""

import torch
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
import json

from .hidden_states import HiddenStatesExtractor


@dataclass
class PairedHiddenStates:
    """
    Container for paired human/AI hidden states.
    
    Attributes:
        human: (N, hidden_size) human continuation representations
        ai: (N, hidden_size) AI continuation representations
        layer: Layer index these states come from
        n_samples: Number of sample pairs
        metadata: Optional metadata dict
    """
    human: torch.Tensor  # (N, hidden_size)
    ai: torch.Tensor     # (N, hidden_size)
    layer: int
    n_samples: int
    metadata: Optional[dict] = None
    
    def __post_init__(self):
        assert self.human.shape == self.ai.shape, \
            f"Shape mismatch: human={self.human.shape}, ai={self.ai.shape}"
        assert self.human.shape[0] == self.n_samples, \
            f"n_samples mismatch: {self.human.shape[0]} vs {self.n_samples}"
    
    @property
    def hidden_size(self) -> int:
        return self.human.shape[1]
    
    def to(self, device: str) -> "PairedHiddenStates":
        """Move tensors to device."""
        return PairedHiddenStates(
            human=self.human.to(device),
            ai=self.ai.to(device),
            layer=self.layer,
            n_samples=self.n_samples,
            metadata=self.metadata,
        )
    
    def save(self, path: Path):
        """Save to file."""
        torch.save({
            "human": self.human,
            "ai": self.ai,
            "layer": self.layer,
            "n_samples": self.n_samples,
            "metadata": self.metadata,
        }, path)
        print(f"✓ Saved paired hidden states: {path}")
    
    @classmethod
    def load(cls, path: Path) -> "PairedHiddenStates":
        """Load from file."""
        data = torch.load(path, weights_only=False)
        return cls(
            human=data["human"],
            ai=data["ai"],
            layer=data["layer"],
            n_samples=data["n_samples"],
            metadata=data.get("metadata"),
        )
    
    def compute_mean_difference(self, normalize: bool = True) -> torch.Tensor:
        """
        Compute mean difference direction: v = mean(AI) - mean(Human).
        
        This is the simplest steering vector.
        
        Args:
            normalize: Whether to L2-normalize the direction
            
        Returns:
            Direction vector (hidden_size,)
        """
        diff = self.ai - self.human  # (N, hidden_size)
        direction = diff.mean(dim=0)  # (hidden_size,)
        
        if normalize:
            direction = direction / (direction.norm() + 1e-8)
        
        return direction


class PairedExtractor:
    """
    Extract paired hidden states from a dataset of human/AI continuations.
    
    Example:
        extractor = PairedExtractor("../../models/Llama-3.1-8B-Instruct")
        paired_states = extractor.extract_from_samples(
            samples=dataset["samples"],
            layer=16,
            pooling="full_mean"
        )
        paired_states.save("outputs/steering/Llama-3.1-8B-Instruct/states/paired_states_layer16.pt")
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ):
        self.extractor = HiddenStatesExtractor(
            model_name=model_name,
            device=device,
            dtype=dtype,
        )
        self.device = device
    
    def extract_from_samples(
        self,
        samples: List[dict],
        layer: int,
        pooling: str = "full_mean",
        show_progress: bool = True,
    ) -> PairedHiddenStates:
        """
        Extract paired hidden states from a list of samples.
        
        Args:
            samples: List of dicts with keys: prefix, human_continuation, ai_continuation
            layer: Layer to extract from
            pooling: Pooling strategy
            show_progress: Show progress bar
            
        Returns:
            PairedHiddenStates container
        """
        human_states = []
        ai_states = []
        
        iterator = tqdm(samples, desc=f"Extracting layer {layer}") if show_progress else samples
        
        for sample in iterator:
            prefix = sample["prefix"]
            
            # Extract human continuation
            human_result = self.extractor.extract_continuation(
                prefix=prefix,
                continuation=sample["human_continuation"],
                pooling=pooling,
                layers=[layer],
            )
            human_states.append(human_result.hidden_states[layer].float())
            
            # Extract AI continuation (supports both field names)
            ai_text = sample.get("ai_continuation") or sample.get("ai_continuation_chat", "")
            ai_result = self.extractor.extract_continuation(
                prefix=prefix,
                continuation=ai_text,
                pooling=pooling,
                layers=[layer],
            )
            ai_states.append(ai_result.hidden_states[layer].float())
        
        human_tensor = torch.stack(human_states)  # (N, hidden_size)
        ai_tensor = torch.stack(ai_states)        # (N, hidden_size)
        
        return PairedHiddenStates(
            human=human_tensor,
            ai=ai_tensor,
            layer=layer,
            n_samples=len(samples),
            metadata={
                "model": self.extractor.model_name,
                "pooling": pooling,
                "hidden_size": self.extractor.hidden_size,
            }
        )
    
    def extract_multiple_layers(
        self,
        samples: List[dict],
        layers: List[int],
        pooling: str = "full_mean",
        show_progress: bool = True,
    ) -> Dict[int, PairedHiddenStates]:
        """
        Extract paired hidden states from multiple layers efficiently.
        One forward pass per sample extracts all requested layers.
        """
        per_layer_human = {ly: [] for ly in layers}
        per_layer_ai = {ly: [] for ly in layers}

        iterator = tqdm(samples, desc="Extracting (multi-layer)") if show_progress else samples
        for sample in iterator:
            prefix = sample["prefix"]
            ai_text = sample.get("ai_continuation") or sample.get("ai_continuation_chat", "")

            h_result = self.extractor.extract_continuation(
                prefix=prefix, continuation=sample["human_continuation"],
                pooling=pooling, layers=layers,
            )
            a_result = self.extractor.extract_continuation(
                prefix=prefix, continuation=ai_text,
                pooling=pooling, layers=layers,
            )
            for ly in layers:
                per_layer_human[ly].append(h_result.hidden_states[ly].float())
                per_layer_ai[ly].append(a_result.hidden_states[ly].float())

        results = {}
        for ly in layers:
            results[ly] = PairedHiddenStates(
                human=torch.stack(per_layer_human[ly]),
                ai=torch.stack(per_layer_ai[ly]),
                layer=ly,
                n_samples=len(samples),
                metadata={
                    "model": self.extractor.model_name,
                    "pooling": pooling,
                    "hidden_size": self.extractor.hidden_size,
                },
            )
        return results
