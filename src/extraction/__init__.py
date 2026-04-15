"""
Extraction module for hidden states from LLMs.

This module provides tools for:
1. Extracting hidden states from any layer during forward pass
2. Extracting paired hidden states from human/AI continuations
"""

from .hidden_states import HiddenStatesExtractor, ExtractionResult
from .paired_extractor import PairedExtractor, PairedHiddenStates

__all__ = [
    "HiddenStatesExtractor",
    "ExtractionResult", 
    "PairedExtractor",
    "PairedHiddenStates",
]
