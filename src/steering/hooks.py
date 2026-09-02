"""Activation hooks for residual-stream steering.

The safe generation default is ``decode_only``: the first forward pass
(prompt prefill) is left untouched and only the newest token is modified on
subsequent cached decoding calls.  This keeps the prompt KV cache identical
to the unsteered baseline.
"""

import torch
from typing import Dict, List, Any, Union, Tuple
from contextlib import contextmanager
from .base import BaseSteering, SteeringConfig
from ..layer_indexing import decoder_block_index


POSITION_MODES = ("decode_only", "all", "last")


class ActivationHook:
    """
    A hook that applies steering to hidden states during forward pass.
    
    Usage:
        hook = ActivationHook(steering, config)
        handle = layer_module.register_forward_hook(hook)
        # ... run generation ...
        handle.remove()
    """
    
    def __init__(
        self,
        steering: BaseSteering,
        config: SteeringConfig,
        apply_to_positions: Union[str, List[int]] = "decode_only",
    ):
        """
        Args:
            steering: The steering method to apply
            config: Steering configuration
            apply_to_positions: Which token positions to apply steering to.
                ``decode_only`` skips the prompt-prefill forward and modifies
                only the newest token on later forwards.
                - "all": All positions
                - "last": Only last token
                - list of ints: Specific positions
        """
        if isinstance(apply_to_positions, str):
            if apply_to_positions not in POSITION_MODES:
                raise ValueError(
                    f"Unknown steering position mode: {apply_to_positions!r}; "
                    f"choose one of {POSITION_MODES}"
                )
        elif not (
            isinstance(apply_to_positions, (list, tuple))
            and all(isinstance(position, int) for position in apply_to_positions)
        ):
            raise TypeError(
                "apply_to_positions must be a supported mode or a list of ints"
            )
        self.steering = steering
        self.config = config
        self.apply_to_positions = apply_to_positions
        self.reset()

    def reset(self) -> None:
        """Reset per-generation call counters."""
        self.forward_calls = 0
        self.prefill_calls_skipped = 0
        self.steered_calls = 0
        self.steered_token_positions = 0

    @property
    def stats(self) -> Dict[str, int]:
        """Return auditable intervention counters for the last generation."""
        return {
            "forward_calls": self.forward_calls,
            "prefill_calls_skipped": self.prefill_calls_skipped,
            "steered_calls": self.steered_calls,
            "steered_token_positions": self.steered_token_positions,
        }

    def __call__(
        self,
        module,
        input: Tuple,
        output: Union[torch.Tensor, Tuple],
    ) -> Union[torch.Tensor, Tuple]:
        """
        Forward hook function.
        
        Args:
            module: The layer module
            input: Layer input
            output: Layer output (hidden_states, ...) or hidden_states
            
        Returns:
            Modified output
        """
        self.forward_calls += 1

        # Extract hidden states from output.
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None

        # Hugging Face cached generation performs prompt prefill on the first
        # model forward.  Returning the exact original object here ensures
        # both prompt activations and the resulting KV cache remain unchanged.
        if (
            self.apply_to_positions == "decode_only"
            and self.forward_calls == 1
        ):
            self.prefill_calls_skipped += 1
            return output

        # Compute the delta only when an intervention will actually be used.
        delta = self.steering.compute_delta(hidden_states, self.config)

        # Apply to specified positions.
        if self.apply_to_positions == "all":
            hidden_states = hidden_states + delta
            token_positions = (
                hidden_states.shape[1] if hidden_states.dim() == 3 else 1
            )
        elif self.apply_to_positions in ("last", "decode_only"):
            if hidden_states.dim() == 3:
                hidden_states = hidden_states.clone()
                hidden_states[:, -1, :] = hidden_states[:, -1, :] + delta[:, -1, :]
            else:
                hidden_states = hidden_states + delta
            token_positions = 1
        else:
            # List of positions
            hidden_states = hidden_states.clone()
            for pos in self.apply_to_positions:
                if hidden_states.dim() == 3:
                    hidden_states[:, pos, :] = hidden_states[:, pos, :] + delta[:, pos, :]
                else:
                    hidden_states = hidden_states + delta
            token_positions = len(self.apply_to_positions)

        self.steered_calls += 1
        self.steered_token_positions += token_positions

        if rest is not None:
            return (hidden_states,) + rest
        return hidden_states


def _get_decoder_layers(model):
    """Return the decoder block collection for common HF model layouts."""
    candidates = (
        ("model.layers", lambda m: m.model.layers),
        ("model.language_model.layers", lambda m: m.model.language_model.layers),
        ("language_model.layers", lambda m: m.language_model.layers),
        ("transformer.h", lambda m: m.transformer.h),
        ("layers", lambda m: m.layers),
    )
    for _, getter in candidates:
        try:
            layers = getter(model)
        except AttributeError:
            continue
        if layers is not None:
            return layers
    raise ValueError(f"Unknown model architecture: {type(model)}")


def get_layer_module(model, layer_idx: int):
    """
    Get the decoder module for a public one-based layer number.
    
    Public layer ``L`` denotes the output of decoder block ``L`` and therefore
    maps to the zero-based module ``layers[L - 1]``. This matches extraction
    from Hugging Face ``outputs.hidden_states[L]``.
    """
    layers = _get_decoder_layers(model)
    block_idx = decoder_block_index(layer_idx, len(layers))
    return layers[block_idx]


def install_steering_hook(
    model,
    steering: BaseSteering,
    config: SteeringConfig,
    apply_to_positions: Union[str, List[int]] = "decode_only",
) -> Any:
    """
    Install a steering hook on the model.
    
    Args:
        model: The language model
        steering: Steering method
        config: Steering configuration using one-based decoder layers
        apply_to_positions: Token positions to apply steering
        
    Returns:
        Hook handle (call handle.remove() to uninstall)
    """
    layer_module = get_layer_module(model, config.layer)
    hook = ActivationHook(steering, config, apply_to_positions)
    handle = layer_module.register_forward_hook(hook)
    # Keep the historical return type (a removable PyTorch handle) while
    # exposing counters for experiment manifests and tests.
    handle.steering_hook = hook
    return handle


class SteeringContext:
    """
    Context manager for applying steering during generation.
    
    Usage:
        with SteeringContext(model, steering, config) as ctx:
            outputs = model.generate(...)
    """
    
    def __init__(
        self,
        model,
        steering: BaseSteering,
        config: SteeringConfig,
        apply_to_positions: Union[str, List[int]] = "decode_only",
    ):
        self.model = model
        self.steering = steering
        self.config = config
        self.apply_to_positions = apply_to_positions
        self.handle = None
        self.hook = None
    
    def __enter__(self):
        self.handle = install_steering_hook(
            self.model,
            self.steering,
            self.config,
            self.apply_to_positions
        )
        self.hook = self.handle.steering_hook
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
            self.hook = None
        return False


@contextmanager
def steering_applied(
    model,
    steering: BaseSteering,
    config: SteeringConfig,
    apply_to_positions: Union[str, List[int]] = "decode_only",
):
    """
    Context manager for applying steering during forward pass.
    
    Usage:
        with steering_applied(model, steering, config):
            outputs = model.generate(...)
    """
    handle = install_steering_hook(model, steering, config, apply_to_positions)
    try:
        yield handle.steering_hook
    finally:
        handle.remove()


class MultiSteering:
    """
    Apply multiple steering methods to different layers.
    
    Useful for:
    - Applying steering to multiple layers
    - Combining different steering methods
    """
    
    def __init__(self, model):
        self.model = model
        self.steerings: Dict[
            int,
            Tuple[
                BaseSteering,
                SteeringConfig,
                Union[str, List[int]],
            ],
        ] = {}
        self.handles: List[Any] = []
    
    def add(
        self,
        steering: BaseSteering,
        config: SteeringConfig,
        apply_to_positions: Union[str, List[int]] = "decode_only",
    ):
        """Add a steering method for a specific layer."""
        layer = config.layer
        self.steerings[layer] = (steering, config, apply_to_positions)
    
    def install(self):
        """Install all steering hooks."""
        self.remove()  # Remove existing hooks
        
        for layer, (steering, config, positions) in self.steerings.items():
            handle = install_steering_hook(
                self.model, steering, config, positions
            )
            self.handles.append(handle)
    
    def remove(self):
        """Remove all steering hooks."""
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
    
    def __enter__(self):
        self.install()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()
        return False
