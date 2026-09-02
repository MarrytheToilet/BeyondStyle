"""
Steering Module - Methods for modifying LLM hidden states.

This module provides additive steering with the within-class nullspace
direction as the primary estimator. Raw mean difference remains available as
an explicit control.
"""

from .base import BaseSteering, SteeringConfig
from .mean_diff import MeanDiffSteering
from .calibration import load_calibrated_mean_diff, pooled_projection_std
from .hooks import ActivationHook, install_steering_hook
from .metrics import pca_hull_centroid_distance, compute_layer_geometry
from .denoise import (
    compute_denoised_direction,
    null_space_projection,
    cross_validated_auc,
    DENOISE_METHODS,
)

__all__ = [
    "BaseSteering",
    "SteeringConfig",
    "MeanDiffSteering",
    "load_calibrated_mean_diff",
    "pooled_projection_std",
    "ActivationHook",
    "install_steering_hook",
    "pca_hull_centroid_distance",
    "compute_layer_geometry",
    "compute_denoised_direction",
    "null_space_projection",
    "cross_validated_auc",
    "DENOISE_METHODS",
]
