"""
Base classes for steering methods.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class SteeringConfig:
    """Configuration for steering intervention."""
    layer: int  # Which layer to intervene on
    strength: float = 1.0  # Scaling factor
    direction: str = "human"  # "human" (subtract) or "ai" (add)
    token_positions: str = "all"  # "all", "last", or list of indices
    
    def get_sign(self) -> float:
        """Get sign multiplier based on direction."""
        return -1.0 if self.direction == "human" else 1.0


class BaseSteering(ABC, nn.Module):
    """
    Abstract base class for all steering methods.
    
    All steering methods must implement:
    - compute_delta: Given hidden state, compute the modification delta
    - fit: Train/fit the steering method from paired data
    
    The steering formula is: h' = h + delta
    Where delta = sign * strength * transform(h)
    """
    
    def __init__(self, hidden_size: int, layer: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer = layer
    
    @abstractmethod
    def compute_delta(
        self, 
        h: torch.Tensor,
        config: SteeringConfig
    ) -> torch.Tensor:
        """
        Compute the steering delta for given hidden states.
        
        Args:
            h: Hidden states [batch, seq_len, hidden_size] or [batch, hidden_size]
            config: Steering configuration
            
        Returns:
            delta: Same shape as h, the amount to add to h
        """
        pass
    
    @abstractmethod
    def fit(
        self,
        human_states: torch.Tensor,
        ai_states: torch.Tensor,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Train/fit the steering method from paired data.
        
        Args:
            human_states: Human text hidden states [N, hidden_size]
            ai_states: AI text hidden states [N, hidden_size]
            
        Returns:
            metrics: Training metrics dict
        """
        pass
    
    def steer(
        self,
        h: torch.Tensor,
        config: SteeringConfig
    ) -> torch.Tensor:
        """
        Apply steering to hidden states.
        
        Args:
            h: Hidden states
            config: Steering configuration
            
        Returns:
            Modified hidden states
        """
        delta = self.compute_delta(h, config)
        return h + delta
    
    def save(self, path: Path, metadata: Optional[Dict] = None):
        """Save steering model to file."""
        save_dict = {
            "state_dict": self.state_dict(),
            "class_name": self.__class__.__name__,
            "hidden_size": self.hidden_size,
            "layer": self.layer,
            "metadata": metadata or {}
        }
        torch.save(save_dict, path)
    
    @classmethod
    def load(cls, path: Path, **override_kwargs) -> "BaseSteering":
        """Load steering model from file."""
        data = torch.load(path, map_location="cpu")
        
        # Merge saved params with overrides
        kwargs = {
            "hidden_size": data["hidden_size"],
            "layer": data["layer"],
        }
        kwargs.update(override_kwargs)
        
        model = cls(**kwargs)
        model.load_state_dict(data["state_dict"])
        return model
    
    def __repr__(self):
        return f"{self.__class__.__name__}(layer={self.layer}, hidden_size={self.hidden_size})"
