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
from ..layer_indexing import LAYER_CONVENTION
from ..data.paired_io import generator_model, plan_serialization_share
from ..runtime import model_source_fingerprint


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
        dtype: torch.dtype | str = "auto",
        attention_backend: str = "auto",
        allow_tf32: bool = True,
    ):
        self.extractor = HiddenStatesExtractor(
            model_name=model_name,
            device=device,
            dtype=dtype,
            attention_backend=attention_backend,
            allow_tf32=allow_tf32,
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
        human_token_counts = []
        ai_token_counts = []
        
        iterator = tqdm(samples, desc=f"Extracting layer {layer}") if show_progress else samples
        
        for sample in iterator:
            prefix = sample["prefix"]
            language = sample.get("language")
            separator = (
                " " if language == "en"
                else "" if language == "zh"
                else None
            )
            
            # Extract human continuation
            human_result = self.extractor.extract_continuation(
                prefix=prefix,
                continuation=sample["human_continuation"],
                pooling=pooling,
                layers=[layer],
                separator=separator,
            )
            human_states.append(human_result.hidden_states[layer].float())
            human_token_counts.append(human_result.continuation_token_count)
            
            # Extract AI continuation (supports both field names)
            ai_text = sample.get("ai_continuation") or sample.get("ai_continuation_chat", "")
            ai_result = self.extractor.extract_continuation(
                prefix=prefix,
                continuation=ai_text,
                pooling=pooling,
                layers=[layer],
                separator=separator,
            )
            ai_states.append(ai_result.hidden_states[layer].float())
            ai_token_counts.append(ai_result.continuation_token_count)
        
        human_tensor = torch.stack(human_states)  # (N, hidden_size)
        ai_tensor = torch.stack(ai_states)        # (N, hidden_size)
        
        return PairedHiddenStates(
            human=human_tensor,
            ai=ai_tensor,
            layer=layer,
            n_samples=len(samples),
            metadata={
                "model": self.extractor.model_name,
                "model_identity": model_source_fingerprint(
                    self.extractor.model_name
                ),
                "pooling": pooling,
                "hidden_size": self.extractor.hidden_size,
                "layer_convention": LAYER_CONVENTION,
                "sample_ids": [sample["sample_id"] for sample in samples],
                "sources": [sample.get("source") for sample in samples],
                "generator_models": [
                    generator_model(sample) for sample in samples
                ],
                "human_continuation_tokens": human_token_counts,
                "ai_continuation_tokens": ai_token_counts,
                "plan_serialization_share": [
                    plan_serialization_share(sample) for sample in samples
                ],
            }
        )
    
    def extract_multiple_layers(
        self,
        samples: List[dict],
        layers: List[int],
        pooling: str = "full_mean",
        show_progress: bool = True,
        batch_size: int = 4,
    ) -> Dict[int, PairedHiddenStates]:
        """
        Extract paired hidden states from multiple layers efficiently.
        Batched human/AI forward passes extract all requested layers.
        """
        return self.extract_multiple_layers_for_poolings(
            samples=samples,
            layers=layers,
            poolings=[pooling],
            show_progress=show_progress,
            batch_size=batch_size,
        )[pooling]

    def extract_multiple_layers_for_poolings(
        self,
        samples: List[dict],
        layers: List[int],
        poolings: List[str],
        show_progress: bool = True,
        batch_size: int = 4,
    ) -> Dict[str, Dict[int, PairedHiddenStates]]:
        """Extract both representations in batched human+AI forward passes."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if not samples:
            raise ValueError("samples must be non-empty")
        poolings = list(dict.fromkeys(poolings))
        per_pooling_human = {
            pooling: {layer: [] for layer in layers}
            for pooling in poolings
        }
        per_pooling_ai = {
            pooling: {layer: [] for layer in layers}
            for pooling in poolings
        }
        human_token_counts = []
        ai_token_counts = []

        progress = (
            tqdm(total=len(samples), desc="Extracting (batched pairs)")
            if show_progress else None
        )
        offset = 0
        active_batch_size = min(batch_size, len(samples))
        while offset < len(samples):
            batch = samples[offset: offset + active_batch_size]
            prefixes = [sample["prefix"] for sample in batch]
            human_continuations = [
                sample["human_continuation"] for sample in batch
            ]
            ai_continuations = [
                sample.get("ai_continuation")
                or sample.get("ai_continuation_chat", "")
                for sample in batch
            ]
            separators = [
                (
                    " " if sample.get("language") == "en"
                    else "" if sample.get("language") == "zh"
                    else None
                )
                for sample in batch
            ]
            try:
                combined_results = (
                    self.extractor.extract_continuation_poolings_batch(
                        prefixes=prefixes + prefixes,
                        continuations=(
                            human_continuations + ai_continuations
                        ),
                        poolings=poolings,
                        layers=layers,
                        separators=separators + separators,
                    )
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or (
                    "out of memory" in str(error).lower()
                )
                if not is_oom or active_batch_size == 1:
                    if progress is not None:
                        progress.close()
                    raise
                active_batch_size = max(1, active_batch_size // 2)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            human_results = combined_results[:len(batch)]
            ai_results = combined_results[len(batch):]
            first_pooling = poolings[0]
            for human_result, ai_result in zip(
                human_results,
                ai_results,
            ):
                human_token_counts.append(
                    human_result[first_pooling].continuation_token_count
                )
                ai_token_counts.append(
                    ai_result[first_pooling].continuation_token_count
                )
                for pooling in poolings:
                    for layer in layers:
                        per_pooling_human[pooling][layer].append(
                            human_result[pooling]
                            .hidden_states[layer]
                            .float()
                        )
                        per_pooling_ai[pooling][layer].append(
                            ai_result[pooling]
                            .hidden_states[layer]
                            .float()
                        )
            offset += len(batch)
            if progress is not None:
                progress.update(len(batch))
        if progress is not None:
            progress.close()

        metadata_base = {
            "model": self.extractor.model_name,
            "model_identity": model_source_fingerprint(
                self.extractor.model_name
            ),
            "hidden_size": self.extractor.hidden_size,
            "layer_convention": LAYER_CONVENTION,
            "sample_ids": [sample["sample_id"] for sample in samples],
            "sources": [sample.get("source") for sample in samples],
            "generator_models": [
                generator_model(sample) for sample in samples
            ],
            "human_continuation_tokens": human_token_counts,
            "ai_continuation_tokens": ai_token_counts,
            "plan_serialization_share": [
                plan_serialization_share(sample) for sample in samples
            ],
            "requested_extraction_batch_size": batch_size,
            "model_load": self.extractor.model_load_info,
        }
        return {
            pooling: {
                layer: PairedHiddenStates(
                    human=torch.stack(per_pooling_human[pooling][layer]),
                    ai=torch.stack(per_pooling_ai[pooling][layer]),
                    layer=layer,
                    n_samples=len(samples),
                    metadata={**metadata_base, "pooling": pooling},
                )
                for layer in layers
            }
            for pooling in poolings
        }
