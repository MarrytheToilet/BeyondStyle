"""Canonical model-independent data construction and paired-data helpers."""

from .natural_window import NaturalLanguageWindow, NaturalLanguageWindowSplitter
from .paired_io import (
    ai_continuation,
    generator_model,
    load_paired_samples,
    plan_serialization_share,
)

__all__ = [
    "NaturalLanguageWindow",
    "NaturalLanguageWindowSplitter",
    "ai_continuation",
    "generator_model",
    "load_paired_samples",
    "plan_serialization_share",
]
