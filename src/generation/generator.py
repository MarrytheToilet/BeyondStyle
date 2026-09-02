"""Shared generation configuration.

Concrete generation backends live in ``backends.py`` and optimized local
steering lives in ``steered.py``. Keeping only the configuration dataclass
here avoids maintaining a second, slower Hugging Face inference stack.
"""

from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Common text-generation controls."""

    temperature: float = 0.7
    top_p: float = 0.95
    max_new_tokens: int = 200
    num_samples: int = 3

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.num_samples < 1:
            raise ValueError("num_samples must be positive")
