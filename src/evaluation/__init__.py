"""Evaluation module.

Provides three evaluation pipelines:
1. lm-eval benchmark (MMLU, HellaSwag, ARC, WikiText perplexity)
2. Winston AI detection scoring
3. Human AI-feel scoring workflow
"""

from .steered_model import (
    SteeredHFLM,
    apply_steering_hooks,
    load_hflm,
    _load_steering_model,
)

__all__ = ["SteeredHFLM", "apply_steering_hooks", "load_hflm"]
