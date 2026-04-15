"""BeyondStyle: Source Subspace Steering in LLM Generation

Core modules:
- extraction: Hidden state extraction from LLMs
- steering: Additive mean-difference steering (v_src)
- generation: Text generation with optional steering
- data: Dataset construction (corpus loading, splitting, AI continuation)
- visualization: Analysis figures and plots
"""

__version__ = "0.1.0"

# Core exports
from .extraction import HiddenStatesExtractor, PairedExtractor, PairedHiddenStates
from .steering import (
    BaseSteering, SteeringConfig,
    MeanDiffSteering,
    ActivationHook, install_steering_hook
)
from .generation import SteeredGenerator, SteeredGenerationConfig

__all__ = [
    # Extraction
    "HiddenStatesExtractor",
    "PairedExtractor", 
    "PairedHiddenStates",
    # Steering
    "BaseSteering",
    "SteeringConfig",
    "MeanDiffSteering",
    "ActivationHook",
    "install_steering_hook",
    # Generation
    "SteeredGenerator",
    "SteeredGenerationConfig",
]
