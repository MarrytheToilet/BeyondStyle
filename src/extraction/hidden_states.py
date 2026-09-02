"""
Hidden States Extraction from LLM

Core functionality for extracting hidden states from transformer layers.
Supports continuation-specific extraction and multiple pooling strategies.
"""

import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

from ..steering.hooks import get_layer_module
from ..runtime import (
    configure_torch_runtime,
    input_device,
    model_load_kwargs,
)


CJK_RE = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")


def continuation_separator(prefix: str, continuation: str) -> str:
    """Return the natural separator for independently stored text segments."""
    if not prefix or not continuation:
        return ""
    if prefix[-1].isspace() or continuation[0].isspace():
        return ""
    if CJK_RE.search(prefix[-16:]) or CJK_RE.search(continuation[:16]):
        return ""
    return " "


@dataclass
class ExtractionResult:
    """Result of hidden state extraction."""
    hidden_states: Dict[int, torch.Tensor]  # layer -> tensor
    token_count: int
    prefix_token_count: Optional[int] = None
    continuation_token_count: Optional[int] = None


@dataclass
class TokenExtractionResult:
    """Unpooled continuation-token states from one teacher-forced sequence."""

    hidden_states: Dict[int, torch.Tensor]  # layer -> (continuation, hidden)
    token_ids: torch.Tensor  # (continuation,)
    token_count: int
    prefix_token_count: int
    continuation_token_count: int
    separator: str


class HiddenStatesExtractor:
    """
    Extract hidden states from a causal LM.

    Public layer numbers are one-based decoder-block outputs. States are
    captured directly from decoder module ``layers[L - 1]`` so extraction and
    intervention refer to the identical residual-stream site, including the
    final decoder block before any model-level final norm.
    
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
        dtype: torch.dtype | str = "auto",
        attention_backend: str = "auto",
        allow_tf32: bool = True,
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
        self.attention_backend = attention_backend
        self.model_load_info = {}
        configure_torch_runtime(allow_tf32=allow_tf32)
        
        if model is not None and tokenizer is not None:
            self.model = model
            self.tokenizer = tokenizer
            self.model_load_info = {"source": "injected_model"}
        else:
            print(f"Loading model: {model_name}")
            # Detect local paths so newer huggingface_hub doesn't treat them as repo IDs
            extra_kw = {}
            if os.path.isdir(model_name):
                extra_kw["local_files_only"] = True
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True, **extra_kw
            )
            load_kwargs, load_info = model_load_kwargs(
                device=device,
                dtype=dtype,
                attention_backend=attention_backend,
                trust_remote_code=True,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **load_kwargs,
                **extra_kw,
            )
            self.model_load_info = load_info
        
        self.model.eval()
        self.device = input_device(self.model)
        text_config = getattr(
            self.model.config,
            "text_config",
            self.model.config,
        )
        self.num_layers = text_config.num_hidden_layers
        self.hidden_size = text_config.hidden_size
        # Run the text backbone rather than the task head. Hooks observe the
        # same decoder blocks, while avoiding an unnecessary vocabulary-sized
        # logits tensor (especially large for Gemma 4).
        if hasattr(self.model, "model") and hasattr(
            self.model.model,
            "language_model",
        ):
            self.forward_model = self.model.model.language_model
        elif hasattr(self.model, "model") and hasattr(
            self.model.model,
            "layers",
        ):
            self.forward_model = self.model.model
        else:
            self.forward_model = self.model
        print(f"Model ready: {self.num_layers} layers, hidden_size={self.hidden_size}")

    def _capture_layer_outputs(
        self,
        inputs: Dict[str, torch.Tensor],
        layers: List[int],
    ) -> Dict[int, torch.Tensor]:
        """Run one forward pass and capture exact decoder-block outputs."""
        captures = {}
        handles = []

        def make_hook(layer):
            def hook(_module, _hook_inputs, output):
                hidden = output[0] if isinstance(output, tuple) else output
                captures[layer] = hidden

            return hook

        for layer in layers:
            # get_layer_module validates the public one-based layer number.
            module = get_layer_module(self.model, layer)
            handles.append(module.register_forward_hook(make_hook(layer)))

        try:
            self.forward_model(**inputs, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()

        missing = sorted(set(layers) - set(captures))
        if missing:
            raise RuntimeError(f"Failed to capture decoder layers: {missing}")
        return captures
    
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
            layers: One-based decoder layers to extract (None = all)
            
        Returns:
            Dict mapping layer index to hidden state tensor
        """
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=2048
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        if layers is None:
            layers = list(range(1, self.num_layers + 1))
        captured_states = self._capture_layer_outputs(inputs, layers)
        
        result = {}
        for layer in layers:
            hs = captured_states[layer][0]  # (seq_len, hidden_size)
            
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
            
            result[layer] = agg.cpu()
        
        return result
    
    @torch.no_grad()
    def extract_continuation(
        self,
        prefix: str,
        continuation: str,
        pooling: str = "full_mean",
        layers: Optional[List[int]] = None,
        separator: Optional[str] = None,
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
            pooling: ``"last"`` for the final continuation token or
                ``"full_mean"`` for the mean over continuation tokens
            layers: One-based decoder layers to extract (None = all)
            
        Returns:
            ExtractionResult with hidden states for continuation
        """
        return self.extract_continuation_poolings(
            prefix=prefix,
            continuation=continuation,
            poolings=[pooling],
            layers=layers,
            separator=separator,
        )[pooling]

    @torch.no_grad()
    def extract_continuation_poolings(
        self,
        prefix: str,
        continuation: str,
        poolings: List[str],
        layers: Optional[List[int]] = None,
        separator: Optional[str] = None,
    ) -> Dict[str, ExtractionResult]:
        """Extract several continuation representations in one forward pass."""
        return self.extract_continuation_poolings_batch(
            prefixes=[prefix],
            continuations=[continuation],
            poolings=poolings,
            layers=layers,
            separators=[separator],
        )[0]

    @torch.no_grad()
    def extract_continuation_poolings_batch(
        self,
        prefixes: List[str],
        continuations: List[str],
        poolings: List[str],
        layers: Optional[List[int]] = None,
        separators: Optional[List[Optional[str]]] = None,
    ) -> List[Dict[str, ExtractionResult]]:
        """Extract a padded batch while preserving every continuation boundary."""
        if len(prefixes) != len(continuations):
            raise ValueError("prefixes and continuations must have equal length")
        if not prefixes:
            raise ValueError("batch must be non-empty")
        if any(not continuation for continuation in continuations):
            raise ValueError("continuation must be non-empty")
        if not poolings:
            raise ValueError("poolings must be non-empty")
        poolings = list(dict.fromkeys(poolings))
        unsupported = sorted(set(poolings) - {"last", "full_mean"})
        if unsupported:
            raise ValueError(
                f"Unknown continuation pooling(s): {unsupported}; "
                "choose 'last' and/or 'full_mean'"
            )
        inputs, encoded = self._prepare_continuation_batch(
            prefixes,
            continuations,
            separators,
        )
        if layers is None:
            layers = list(range(1, self.num_layers + 1))
        captured_states = self._capture_layer_outputs(inputs, layers)

        batch_results = []
        for row_index, row in enumerate(encoded):
            result_states = {pooling: {} for pooling in poolings}
            for layer in layers:
                hidden = captured_states[layer][row_index]
                continuation_hidden = hidden[
                    row["continuation_start"]: row["continuation_end"]
                ]
                for pooling in poolings:
                    result_states[pooling][layer] = self._apply_pooling(
                        continuation_hidden,
                        pooling,
                    ).cpu()
            batch_results.append({
                pooling: ExtractionResult(
                    hidden_states=result_states[pooling],
                    token_count=row["continuation_end"],
                    prefix_token_count=row["continuation_start"],
                    continuation_token_count=(
                        row["continuation_end"]
                        - row["continuation_start"]
                    ),
                )
                for pooling in poolings
            })
        return batch_results

    @torch.no_grad()
    def extract_continuation_tokens(
        self,
        prefix: str,
        continuation: str,
        layers: Optional[List[int]] = None,
        separator: Optional[str] = None,
        *,
        to_cpu: bool = True,
    ) -> TokenExtractionResult:
        """Extract every continuation-token state in one forward pass."""
        return self.extract_continuation_tokens_batch(
            prefixes=[prefix],
            continuations=[continuation],
            layers=layers,
            separators=[separator],
            to_cpu=to_cpu,
        )[0]

    @torch.no_grad()
    def extract_continuation_tokens_batch(
        self,
        prefixes: List[str],
        continuations: List[str],
        layers: Optional[List[int]] = None,
        separators: Optional[List[Optional[str]]] = None,
        *,
        to_cpu: bool = True,
    ) -> List[TokenExtractionResult]:
        """Extract unpooled continuation states for a padded batch.

        ``to_cpu=False`` is intended for callers that immediately reduce the
        states on the accelerator, for example by projecting them onto one or
        more directions. It avoids transferring the full hidden vectors.
        """
        inputs, encoded = self._prepare_continuation_batch(
            prefixes,
            continuations,
            separators,
        )
        if layers is None:
            layers = list(range(1, self.num_layers + 1))
        captured_states = self._capture_layer_outputs(inputs, layers)

        results = []
        for row_index, row in enumerate(encoded):
            start = row["continuation_start"]
            end = row["continuation_end"]
            states = {
                layer: captured_states[layer][row_index, start:end]
                for layer in layers
            }
            token_ids = inputs["input_ids"][row_index, start:end]
            if to_cpu:
                states = {
                    layer: hidden.cpu()
                    for layer, hidden in states.items()
                }
                token_ids = token_ids.cpu()
            results.append(
                TokenExtractionResult(
                    hidden_states=states,
                    token_ids=token_ids,
                    token_count=end,
                    prefix_token_count=start,
                    continuation_token_count=end - start,
                    separator=row["separator"],
                )
            )
        return results

    def _prepare_continuation_batch(
        self,
        prefixes: List[str],
        continuations: List[str],
        separators: Optional[List[Optional[str]]],
        *,
        max_length: int = 2048,
    ) -> Tuple[Dict[str, torch.Tensor], List[dict]]:
        """Tokenize stored prefix/continuation segments with exact boundaries."""
        if len(prefixes) != len(continuations):
            raise ValueError("prefixes and continuations must have equal length")
        if not prefixes:
            raise ValueError("batch must be non-empty")
        if any(not continuation for continuation in continuations):
            raise ValueError("continuation must be non-empty")
        if separators is None:
            separators = [None] * len(prefixes)
        if len(separators) != len(prefixes):
            raise ValueError("separators must match the batch length")

        encoded = []
        for prefix, continuation, separator in zip(
            prefixes,
            continuations,
            separators,
        ):
            if separator is None:
                separator = continuation_separator(prefix, continuation)
            # Separate tokenization keeps the continuation boundary exact.
            prefix_ids = self.tokenizer.encode(
                prefix,
                add_special_tokens=True,
            )
            continuation_ids = self.tokenizer.encode(
                separator + continuation,
                add_special_tokens=False,
            )
            if not continuation_ids:
                raise ValueError("continuation produced no tokens")
            combined = prefix_ids + continuation_ids
            if len(combined) > max_length:
                raise ValueError(
                    f"prefix + continuation has {len(combined)} tokens; "
                    f"the extraction limit is {max_length}"
                )
            encoded.append({
                "ids": combined,
                "continuation_start": len(prefix_ids),
                "continuation_end": len(combined),
                "separator": separator,
            })

        maximum_length = max(len(row["ids"]) for row in encoded)
        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_token_id is None:
            pad_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if pad_token_id is None:
            pad_token_id = 0
        input_ids = torch.full(
            (len(encoded), maximum_length),
            int(pad_token_id),
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        for row_index, row in enumerate(encoded):
            length = len(row["ids"])
            input_ids[row_index, :length] = torch.tensor(
                row["ids"],
                dtype=torch.long,
                device=self.device,
            )
            attention_mask[row_index, :length] = 1
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }, encoded
    
    def _apply_pooling(self, hidden_states: torch.Tensor, pooling: str) -> torch.Tensor:
        """Apply pooling to token-level hidden states."""
        seq_len = hidden_states.shape[0]
        
        if pooling == "full_mean":
            return hidden_states.mean(dim=0)
        elif pooling == "last":
            return hidden_states[-1]
        else:
            raise ValueError(
                f"Unknown continuation pooling: {pooling!r}; "
                "choose 'last' or 'full_mean'"
            )
