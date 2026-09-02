"""
Mean Difference Steering (Baseline).

The simplest steering approach: compute mean difference between AI and human
hidden states, then add/subtract this direction during generation.

Formula: h' = h + sign * strength * v_src
Where v_src = mean(AI_states) - mean(Human_states)
"""

import torch
from typing import Dict, Any
from .base import BaseSteering, SteeringConfig


class MeanDiffSteering(BaseSteering):
    """
    Steering using mean difference direction.
    
    This is the baseline method that learns a single direction v_src
    representing the difference between AI and human text representations.
    
    Pros:
    - Simple, no training required beyond mean computation
    - Interpretable as a single direction
    
    Cons:
    - Direction may mix style and content information
    - Same delta applied regardless of input
    """
    
    BACKWARD_COMPATIBLE_MISSING_KEYS = {"projection_std"}

    def __init__(self, hidden_size: int, layer: int, normalize: bool = True):
        """
        Args:
            hidden_size: Dimension of hidden states
            layer: Layer index for this steering
            normalize: Whether to normalize the direction to unit vector
        """
        super().__init__(hidden_size, layer)
        self.normalize = normalize
        
        # Learnable steering direction
        self.register_buffer("v_src", torch.zeros(hidden_size))
        self.register_buffer("is_fitted", torch.tensor(False))
        # Pooled within-class standard deviation along v_src.  Multiplying a
        # unit direction by this value makes beta comparable across layers and
        # model families.  Old checkpoints load with the neutral value 1.
        self.register_buffer("projection_std", torch.tensor(1.0))
    
    def compute_delta(
        self,
        h: torch.Tensor,
        config: SteeringConfig
    ) -> torch.Tensor:
        """
        Compute steering delta (same for all inputs).
        
        The delta is sign * strength * v_src, broadcast to match h shape.
        """
        if not self.is_fitted:
            raise RuntimeError("MeanDiffSteering not fitted. Call fit() first.")
        
        # Compute delta vector (match input dtype and device)
        sign = config.get_sign()
        v_src = self.v_src.to(dtype=h.dtype, device=h.device)
        if config.strength_unit == "projection_std":
            calibration = self.projection_std.to(
                dtype=h.dtype,
                device=h.device,
            )
        else:
            calibration = torch.ones((), dtype=h.dtype, device=h.device)
        delta = sign * config.strength * calibration * v_src
        
        # Broadcast to match input shape
        if h.dim() == 2:
            # [batch, hidden_size]
            return delta.unsqueeze(0).expand_as(h)
        elif h.dim() == 3:
            # [batch, seq_len, hidden_size]
            return delta.view(1, 1, -1).expand_as(h)
        else:
            raise ValueError(f"Unexpected h shape: {h.shape}")
    
    def fit(
        self,
        human_states: torch.Tensor,
        ai_states: torch.Tensor,
        method: str = "paired",
        denoise: str = "raw",
        null_k: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Fit the mean difference direction.
        
        Args:
            human_states: [N, hidden_size]
            ai_states: [N, hidden_size]
            method: "paired" (paired difference) or "mean" (global mean difference)
            denoise: "raw" | "pca" | "nullspace" | "combined"
            null_k: number of content PCs to remove (for nullspace methods)
        """
        human_states = human_states.float()
        ai_states = ai_states.float()
        
        if denoise != "raw":
            from .denoise import compute_denoised_direction
            v_src, denoise_info = compute_denoised_direction(
                human_states, ai_states, method=denoise, null_k=null_k
            )
        elif method == "paired":
            diff = ai_states - human_states
            v_src = diff.mean(dim=0)
            denoise_info = None
        else:
            v_src = ai_states.mean(dim=0) - human_states.mean(dim=0)
            denoise_info = None
        
        # Optionally normalize (denoised vectors are already unit)
        raw_norm = v_src.norm().item()
        if self.normalize and denoise == "raw":
            v_src = v_src / v_src.norm()
        
        self.v_src.copy_(v_src)
        self.is_fitted.fill_(True)

        with torch.no_grad():
            human_scores = human_states @ self.v_src
            ai_scores = ai_states @ self.v_src
            degrees_of_freedom = max(
                int(human_scores.numel() + ai_scores.numel() - 2),
                1,
            )
            pooled_variance = (
                ((human_scores - human_scores.mean()) ** 2).sum()
                + ((ai_scores - ai_scores.mean()) ** 2).sum()
            ) / degrees_of_freedom
            projection_std = pooled_variance.sqrt().clamp_min(1e-6)
            self.projection_std.copy_(projection_std)
        
        # Compute metrics
        metrics = {
            "raw_norm": raw_norm,
            "final_norm": self.v_src.norm().item(),
            "method": method,
            "denoise": denoise,
            "n_samples": len(human_states),
            "projection_std": self.projection_std.item(),
        }
        if denoise_info:
            metrics["denoise_info"] = denoise_info
        
        # Separation metrics
        with torch.no_grad():
            human_proj = (human_states @ self.v_src).mean().item()
            ai_proj = (ai_states @ self.v_src).mean().item()
            metrics["mean_separation"] = ai_proj - human_proj
            metrics["human_proj_mean"] = human_proj
            metrics["ai_proj_mean"] = ai_proj
            metrics["standardized_separation"] = (
                (ai_proj - human_proj) / self.projection_std.item()
            )
        
        return metrics
    
    def get_direction(self) -> torch.Tensor:
        """Get the steering direction vector."""
        return self.v_src.clone()
    
    def analyze_separation(
        self,
        human_states: torch.Tensor,
        ai_states: torch.Tensor
    ) -> Dict[str, float]:
        """
        Analyze how well the direction separates human and AI states.
        
        Returns metrics like separation magnitude, overlap, etc.
        """
        human_proj = (human_states.float() @ self.v_src)
        ai_proj = (ai_states.float() @ self.v_src)
        
        return {
            "human_mean": human_proj.mean().item(),
            "human_std": human_proj.std().item(),
            "ai_mean": ai_proj.mean().item(),
            "ai_std": ai_proj.std().item(),
            "separation": (ai_proj.mean() - human_proj.mean()).item(),
            "overlap": self._compute_overlap(human_proj, ai_proj),
        }
    
    def _compute_overlap(
        self,
        human_proj: torch.Tensor,
        ai_proj: torch.Tensor
    ) -> float:
        """Compute overlap between human and AI projection distributions."""
        # Simple metric: fraction of human that overlaps with AI range
        ai_min, ai_max = ai_proj.min().item(), ai_proj.max().item()
        human_in_ai_range = ((human_proj >= ai_min) & (human_proj <= ai_max)).float().mean()
        
        human_min, human_max = human_proj.min().item(), human_proj.max().item()
        ai_in_human_range = ((ai_proj >= human_min) & (ai_proj <= human_max)).float().mean()
        
        return (human_in_ai_range.item() + ai_in_human_range.item()) / 2
