"""AI-like discourse direction steering and interpretability.

Source code for the paper *"What Makes Text Feel AI-Generated? Steering and
Interpreting AI-Like Discourse Directions in Hidden States."*

Core modules:
- extraction: Hidden state extraction from LLMs
- steering: Primary within-class nullspace steering with raw mean-difference control
- generation: Text generation with optional steering
- data: Dataset construction (corpus loading, splitting, AI continuation)
- linguistics: Fixed lexical, syntactic, discourse, and nuisance measurements
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
