"""
Activation Hooks for Steering.

This module provides hook-based steering that can be applied during
model forward pass. The hooks intercept hidden states and apply
the steering delta.
"""

import torch
from typing import Dict, List, Callable, Any, Optional, Union, Tuple
from contextlib import contextmanager
from .base import BaseSteering, SteeringConfig


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
        apply_to_positions: str = "all"
    ):
        """
        Args:
            steering: The steering method to apply
            config: Steering configuration
            apply_to_positions: Which token positions to apply steering to
                - "all": All positions
                - "last": Only last token
                - list of ints: Specific positions
        """
        self.steering = steering
        self.config = config
        self.apply_to_positions = apply_to_positions
    
    def __call__(
        self, 
        module, 
        input: Tuple, 
        output: Union[torch.Tensor, Tuple]
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
        # Extract hidden states from output
        if isinstance(output, tuple):
            hidden_states = output[0]
            rest = output[1:]
        else:
            hidden_states = output
            rest = None
        
        # Compute delta
        delta = self.steering.compute_delta(hidden_states, self.config)
        
        # Apply to specified positions
        if self.apply_to_positions == "all":
            hidden_states = hidden_states + delta
        elif self.apply_to_positions == "last":
            if hidden_states.dim() == 3:
                hidden_states = hidden_states.clone()
                hidden_states[:, -1, :] = hidden_states[:, -1, :] + delta[:, -1, :]
            else:
                hidden_states = hidden_states + delta
        else:
            # List of positions
            hidden_states = hidden_states.clone()
            for pos in self.apply_to_positions:
                if hidden_states.dim() == 3:
                    hidden_states[:, pos, :] = hidden_states[:, pos, :] + delta[:, pos, :]
                else:
                    hidden_states = hidden_states + delta
        
        if rest is not None:
            return (hidden_states,) + rest
        return hidden_states


def get_layer_module(model, layer_idx: int):
    """
    Get the layer module for a specific layer index.
    
    Handles different model architectures.
    """
    # Qwen, LLaMA style
    if hasattr(model, 'model'):
        if hasattr(model.model, 'layers'):
            return model.model.layers[layer_idx]
    
    # GPT-2 style
    if hasattr(model, 'transformer'):
        if hasattr(model.transformer, 'h'):
            return model.transformer.h[layer_idx]
    
    # Mistral, other styles
    if hasattr(model, 'layers'):
        return model.layers[layer_idx]
    
    raise ValueError(f"Unknown model architecture: {type(model)}")


def install_steering_hook(
    model,
    steering: BaseSteering,
    config: SteeringConfig,
    apply_to_positions: str = "all"
) -> Any:
    """
    Install a steering hook on the model.
    
    Args:
        model: The language model
        steering: Steering method
        config: Steering configuration
        apply_to_positions: Token positions to apply steering
        
    Returns:
        Hook handle (call handle.remove() to uninstall)
    """
    layer_module = get_layer_module(model, config.layer)
    hook = ActivationHook(steering, config, apply_to_positions)
    return layer_module.register_forward_hook(hook)


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
        apply_to_positions: str = "all"
    ):
        self.model = model
        self.steering = steering
        self.config = config
        self.apply_to_positions = apply_to_positions
        self.handle = None
    
    def __enter__(self):
        self.handle = install_steering_hook(
            self.model,
            self.steering,
            self.config,
            self.apply_to_positions
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        return False


@contextmanager
def steering_applied(
    model,
    steering: BaseSteering,
    config: SteeringConfig,
    apply_to_positions: str = "all"
):
    """
    Context manager for applying steering during forward pass.
    
    Usage:
        with steering_applied(model, steering, config):
            outputs = model.generate(...)
    """
    handle = install_steering_hook(model, steering, config, apply_to_positions)
    try:
        yield
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
        self.steerings: Dict[int, Tuple[BaseSteering, SteeringConfig]] = {}
        self.handles: List[Any] = []
    
    def add(
        self,
        steering: BaseSteering,
        config: SteeringConfig,
        apply_to_positions: str = "all"
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
