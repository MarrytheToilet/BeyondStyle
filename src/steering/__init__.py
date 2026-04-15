"""
Steering Module - Methods for modifying LLM hidden states.

This module provides additive steering via mean-difference direction:
- MeanDiffSteering: Compute v_src = mean(AI) - mean(Human), steer with α·v_src
- Denoising: PairedPCA, NullSpaceProjection, Combined for purer v_src extraction
"""

from .base import BaseSteering, SteeringConfig
from .mean_diff import MeanDiffSteering
from .hooks import ActivationHook, install_steering_hook
from .metrics import pca_hull_centroid_distance, fisher_ratio, compute_layer_geometry
from .denoise import (
    compute_denoised_direction,
    null_space_projection,
    fisher_direction,
    cross_validated_auc,
    DENOISE_METHODS,
)

__all__ = [
    "BaseSteering",
    "SteeringConfig",
    "MeanDiffSteering",
    "ActivationHook",
    "install_steering_hook",
    "pca_hull_centroid_distance",
    "fisher_ratio",
    "compute_layer_geometry",
    "compute_denoised_direction",
    "null_space_projection",
    "fisher_direction",
    "cross_validated_auc",
    "DENOISE_METHODS",
]
