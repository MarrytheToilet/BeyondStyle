"""Raw and within-class-nullspace human/AI direction estimators."""

import torch
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score


# ── helpers ──────────────────────────────────────────────

def _to_float(t: torch.Tensor) -> torch.Tensor:
    return t.float() if t.dtype != torch.float32 else t


def _unit(v: torch.Tensor) -> torch.Tensor:
    n = v.norm()
    return v / n if n > 1e-12 else v


def _ensure_sign(v: torch.Tensor, h: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Ensure v points from human toward AI (positive AI projection)."""
    if (a @ v).mean() < (h @ v).mean():
        v = -v
    return v


# ── NullSpace (within-class scatter projection) ─────────

def null_space_projection(
    v: torch.Tensor,
    human_states: torch.Tensor,
    ai_states: torch.Tensor,
    k: int = 10,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Project v into the null-space of the top-k *within-class* PCs.

    Within-class scatter = Cov(H) + Cov(A).  Its top PCs capture content
    variance (topics, entities) that varies *within* each class — NOT the
    class-separation direction.
    """
    h = _to_float(human_states)
    a = _to_float(ai_states)
    v = _to_float(v)

    h_centered = h - h.mean(dim=0)
    a_centered = a - a.mean(dim=0)
    within = torch.cat([h_centered, a_centered], dim=0)

    _, S_w, Vt_w = torch.linalg.svd(within, full_matrices=False)
    content_basis = Vt_w[:k]

    proj_coeffs = content_basis @ v
    v_content = proj_coeffs @ content_basis
    v_residual = v - v_content

    residual_norm = v_residual.norm().item()
    v_proj = _unit(v_residual)
    cosine_shift = (v @ v_proj).item()

    eigenvalues = (S_w ** 2) / (len(within) - 1)
    total_var = eigenvalues.sum().item()
    removed_var = eigenvalues[:k].sum().item()

    info = {
        "method": "null_space",
        "k": k,
        "residual_norm": residual_norm,
        "cosine_shift": cosine_shift,
        "content_variance_removed": removed_var / total_var,
    }
    return v_proj, info


# ── Cross-validated AUC ──────────────────────────────────

def cross_validated_auc(
    v: torch.Tensor,
    human_states: torch.Tensor,
    ai_states: torch.Tensor,
    n_folds: int = 5,
) -> Dict[str, float]:
    """Compute train and CV AUC for a fixed direction v."""
    h = _to_float(human_states)
    a = _to_float(ai_states)
    v = _to_float(v)

    hp = (h @ v).numpy()
    ap = (a @ v).numpy()

    scores = np.concatenate([hp, ap])
    labels = np.concatenate([np.zeros(len(hp)), np.ones(len(ap))])
    train_auc = float(roc_auc_score(labels, scores))

    X = torch.cat([h, a], dim=0).numpy()
    y = labels
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    cv_aucs = []
    for train_idx, test_idx in skf.split(X, y):
        test_scores = X[test_idx] @ v.numpy()
        try:
            cv_aucs.append(float(roc_auc_score(y[test_idx], test_scores)))
        except ValueError:
            pass

    return {
        "train_auc": train_auc,
        "cv_auc": float(np.mean(cv_aucs)) if cv_aucs else train_auc,
        "cv_auc_std": float(np.std(cv_aucs)) if cv_aucs else 0.0,
    }


# ── Unified interface ────────────────────────────────────

DENOISE_METHODS = {
    "raw": None,
    "nullspace": "null_space",
}


def compute_denoised_direction(
    human_states: torch.Tensor,
    ai_states: torch.Tensor,
    method: str = "raw",
    null_k: int = 10,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Unified entry point for denoising methods.

    Returns:
        v_src:  (D,) unit direction vector
        info:   dict with method-specific diagnostics + cross-validated AUC
    """
    h = _to_float(human_states)
    a = _to_float(ai_states)

    if method == "raw":
        diff = a - h
        v_src = _unit(diff.mean(dim=0))
        info = {"method": "raw", "n_pairs": len(h)}

    elif method == "nullspace":
        raw_v = _unit((a - h).mean(dim=0))
        v_src, info = null_space_projection(raw_v, h, a, k=null_k)
        v_src = _ensure_sign(v_src, h, a)

    else:
        raise ValueError(f"Unknown denoise method: {method!r}. "
                         f"Choose from {list(DENOISE_METHODS)}")

    if method != "raw":
        raw_v = _unit((a - h).mean(dim=0))
        info["cosine_vs_raw"] = float((v_src @ raw_v).item())

    cv = cross_validated_auc(v_src, h, a)
    info.update(cv)

    return v_src, info
