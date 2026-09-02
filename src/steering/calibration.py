"""Shared calibration of steering strength across layers and model families."""

from __future__ import annotations

from pathlib import Path

import torch

from .mean_diff import MeanDiffSteering


def pooled_projection_std(
    human: torch.Tensor,
    ai: torch.Tensor,
    direction: torch.Tensor,
) -> float:
    """Pooled within-class standard deviation along a unit direction."""
    human_scores = human.float() @ direction.float()
    ai_scores = ai.float() @ direction.float()
    denominator = max(human_scores.numel() + ai_scores.numel() - 2, 1)
    variance = (
        ((human_scores - human_scores.mean()) ** 2).sum()
        + ((ai_scores - ai_scores.mean()) ** 2).sum()
    ) / denominator
    return float(variance.sqrt().clamp_min(1e-6).item())


def load_calibrated_mean_diff(
    checkpoint_path: Path,
    states_path: Path,
) -> tuple[MeanDiffSteering, dict]:
    """Load a direction and derive calibration from its cached training states."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    steering = MeanDiffSteering.load(checkpoint_path)
    calibration_source = "checkpoint"
    if "projection_std" not in checkpoint["state_dict"]:
        states = torch.load(
            states_path,
            map_location="cpu",
            weights_only=False,
        )
        value = pooled_projection_std(
            states["human"],
            states["ai"],
            steering.get_direction(),
        )
        steering.projection_std.fill_(value)
        calibration_source = "derived_from_cached_training_states"
    return steering, {
        "projection_std": steering.projection_std.item(),
        "source": calibration_source,
        "checkpoint_metadata": checkpoint.get("metadata", {}),
    }
