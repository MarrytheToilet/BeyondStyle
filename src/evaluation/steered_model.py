"""Steered model wrapper for lm-eval integration.

Extends lm-eval's HFLM to apply steering hooks during all forward passes,
allowing benchmark evaluation of steered models (MMLU, WikiText, etc.).

Usage (programmatic):
    from src.evaluation import SteeredHFLM
    model = SteeredHFLM(
        pretrained="/path/to/model",
        steering_path="outputs/steering/.../models/mean_diff_fisher_layer14.pt",
        steering_alpha=1.0,
    )
    # Then use with lm_eval.simple_evaluate(model=model, tasks=[...])

Usage (via eval_lm_benchmark.py script):
    python scripts/eval_lm_benchmark.py --alpha 0.0 1.0 -1.0 --tasks mmlu hellaswag
"""

import sys
import types
import torch
from pathlib import Path
from typing import Optional, Union, List
from contextlib import contextmanager


def _patch_lm_eval_imports():
    """Patch lm-eval to work with transformers>=5.x (missing AutoModelForVision2Seq)."""
    if "lm_eval" not in sys.modules:
        sys.modules["lm_eval.models.hf_vlms"] = types.ModuleType("lm_eval.models.hf_vlms")


def _load_steering_model(path: str, layer_override: Optional[int] = None, device="cuda"):
    """Load a MeanDiffSteering model from a .pt file."""
    root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(root))
    from src.steering import MeanDiffSteering

    sm = MeanDiffSteering.load(path)
    if layer_override is not None:
        sm.layer = layer_override
    sm = sm.to(device)
    return sm


def apply_steering_hooks(model, steering_model, alpha: float = 1.0,
                         direction: str = "ai", positions: str = "all"):
    """Install steering hooks on a HuggingFace model.

    Args:
        model: HuggingFace CausalLM model
        steering_model: MeanDiffSteering instance (with .layer and .get_direction())
        alpha: steering strength (positive = toward AI, negative = toward Human)
        direction: "ai" or "human" — combined with alpha sign
        positions: "all" or "last"

    Returns:
        Hook handle (call handle.remove() to uninstall)
    """
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root.parent))

    from src.steering.base import SteeringConfig
    from src.steering.hooks import install_steering_hook

    if alpha < 0:
        actual_dir = "human"
        actual_strength = abs(alpha)
    else:
        actual_dir = "ai"
        actual_strength = abs(alpha)

    config = SteeringConfig(
        layer=steering_model.layer,
        strength=actual_strength,
        direction=actual_dir,
    )
    return install_steering_hook(model, steering_model, config, positions)


def load_hflm(pretrained: str, **kwargs):
    """Load an HFLM instance (lm-eval HuggingFace Language Model).

    This is a convenience function that handles the transformers 5.x patch.
    """
    _patch_lm_eval_imports()
    from lm_eval.models.huggingface import HFLM
    return HFLM(pretrained=pretrained, **kwargs)


class SteeredHFLM:
    """lm-eval compatible model wrapper with steering hooks.

    Delegates to an underlying HFLM instance and is registered as a virtual
    subclass of lm_eval.api.model.LM so it passes isinstance checks.

    For simple benchmark usage, prefer loading HFLM directly with load_hflm()
    and using apply_steering_hooks() on its ._model.
    """

    def __init__(
        self,
        pretrained: str = None,
        steering_path: Optional[str] = None,
        steering_alpha: float = 0.0,
        steering_layer: Optional[int] = None,
        steering_direction: str = "ai",
        steering_positions: str = "all",
        _hflm: object = None,
        **hflm_kwargs,
    ):
        self._hook_handle = None
        self._steering_model = None
        self._alpha = steering_alpha

        _patch_lm_eval_imports()
        from lm_eval.models.huggingface import HFLM

        if _hflm is not None:
            self._hflm = _hflm
        else:
            self._hflm = HFLM(pretrained=pretrained, **hflm_kwargs)

        if steering_path and steering_alpha != 0.0:
            self._install_steering(
                steering_path, steering_alpha, steering_layer,
                steering_direction, steering_positions,
            )

    def _install_steering(self, path, alpha, layer_override, direction, positions):
        """Load steering model and install hooks."""
        sm = _load_steering_model(path, layer_override, device=self._hflm.device)
        self._steering_model = sm
        self._hook_handle = apply_steering_hooks(
            self._hflm._model, sm, alpha=alpha,
            direction=direction, positions=positions,
        )

    def remove_hooks(self):
        """Remove installed steering hooks."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def swap_steering(self, steering_path: str, alpha: float,
                      layer_override: Optional[int] = None,
                      direction: str = "ai", positions: str = "all"):
        """Swap steering vector without reloading the model."""
        self.remove_hooks()
        self._alpha = alpha
        self._steering_model = None
        if steering_path and alpha != 0.0:
            self._install_steering(steering_path, alpha, layer_override, direction, positions)

    def __del__(self):
        try:
            self.remove_hooks()
        except Exception:
            pass

    # ── delegate everything to underlying HFLM ──
    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._hflm, name)

    # Explicit delegation for key lm-eval API methods
    @property
    def rank(self):
        return self._hflm.rank

    @property
    def world_size(self):
        return self._hflm.world_size

    @property
    def device(self):
        return self._hflm.device

    @property
    def model(self):
        return self._hflm._model

    @property
    def tokenizer(self):
        return self._hflm.tokenizer

    @property
    def eot_token_id(self):
        return self._hflm.eot_token_id

    @property
    def max_length(self):
        return self._hflm.max_length

    @property
    def batch_size(self):
        return self._hflm.batch_size

    @property
    def config(self):
        return self._hflm.config

    def loglikelihood(self, requests, *args, **kwargs):
        return self._hflm.loglikelihood(requests, *args, **kwargs)

    def loglikelihood_rolling(self, requests, *args, **kwargs):
        return self._hflm.loglikelihood_rolling(requests, *args, **kwargs)

    def generate_until(self, requests, *args, **kwargs):
        return self._hflm.generate_until(requests, *args, **kwargs)

    def tok_encode(self, *args, **kwargs):
        return self._hflm.tok_encode(*args, **kwargs)

    def tok_decode(self, *args, **kwargs):
        return self._hflm.tok_decode(*args, **kwargs)

    def __repr__(self):
        alpha_str = f", α={self._alpha}" if self._alpha != 0 else ""
        sm = self._steering_model
        layer_str = f", L{sm.layer}" if sm else ""
        return f"SteeredHFLM({self._hflm.model.name_or_path}{layer_str}{alpha_str})"


# Register as virtual subclass of lm_eval.api.model.LM so isinstance checks pass
_patch_lm_eval_imports()
try:
    from lm_eval.api.model import LM as _LM
    _LM.register(SteeredHFLM)
except Exception:
    pass
