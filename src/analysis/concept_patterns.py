"""Covariance-aware activation patterns for linguistic concept steering.

The functions here deliberately distinguish a decoding/readout weight from an
activation pattern.  A scalar concept pattern is the hidden-state covariance
with a standardized concept score, divided by the score variance.  This is
the forward, generative association used for intervention; it is not the
coefficient of a backward probe.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


EPS = 1e-12


def unit_vector(value: np.ndarray) -> np.ndarray:
    """Return a float64 unit vector and reject a degenerate input."""
    vector = np.asarray(value, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= EPS:
        raise ValueError("Cannot normalize a degenerate vector")
    return vector / norm


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    """Finite Pearson correlation with a zero-variance guard."""
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    if left.shape != right.shape:
        raise ValueError("Correlation inputs must have the same shape")
    left = left - left.mean()
    right = right - right.mean()
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(left @ right / denominator) if denominator > EPS else 0.0


def signature_weights(
    human_features: np.ndarray,
    ai_features: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Orient a feature group along its standardized paired AI−Human shift."""
    human = np.asarray(human_features, dtype=np.float64)
    ai = np.asarray(ai_features, dtype=np.float64)
    if human.shape != ai.shape or human.ndim != 2:
        raise ValueError("Feature matrices must be equally shaped 2D arrays")
    pooled_scale = np.sqrt(
        (
            human.var(axis=0, ddof=1)
            + ai.var(axis=0, ddof=1)
        )
        / 2.0
    )
    pooled_scale[pooled_scale <= EPS] = 1.0
    standardized_shift = (ai - human).mean(axis=0) / pooled_scale
    weights = unit_vector(standardized_shift)
    return weights, {
        "standardized_shift_norm": float(
            np.linalg.norm(standardized_shift)
        ),
        "positive_feature_fraction": float(np.mean(weights > 0)),
    }


def covariance_activation_pattern(
    hidden: np.ndarray,
    concept_scores: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit the forward hidden pattern associated with one concept score."""
    hidden = np.asarray(hidden, dtype=np.float64)
    scores = np.asarray(concept_scores, dtype=np.float64).reshape(-1)
    if hidden.ndim != 2 or len(hidden) != len(scores):
        raise ValueError("Hidden states and scores must be row aligned")
    centered_hidden = hidden - hidden.mean(axis=0, keepdims=True)
    centered_scores = scores - scores.mean()
    score_sum_squares = float(centered_scores @ centered_scores)
    if score_sum_squares <= EPS:
        raise ValueError("Concept score has no variation")
    raw_pattern = (
        centered_hidden.T @ centered_scores
    ) / score_sum_squares
    pattern = unit_vector(raw_pattern)
    return pattern, {
        "raw_pattern_norm": float(np.linalg.norm(raw_pattern)),
        "score_std": float(scores.std(ddof=1)),
        "fitted_score_correlation": pearson(
            centered_hidden @ pattern,
            centered_scores,
        ),
    }


def principal_probe_pattern(
    hidden: np.ndarray,
    concept_scores: np.ndarray,
    *,
    components: int = 64,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit a truncated, covariance-normalized linear readout direction.

    Principal-component truncation regularizes the high-dimensional readout.
    Whitening the retained coordinates makes the mapped-back direction
    equivalent to principal-components regression up to a positive scalar.
    """
    hidden = np.asarray(hidden, dtype=np.float64)
    scores = np.asarray(concept_scores, dtype=np.float64).reshape(-1)
    if hidden.ndim != 2 or len(hidden) != len(scores):
        raise ValueError("Hidden states and scores must be row aligned")
    centered_hidden = hidden - hidden.mean(axis=0, keepdims=True)
    centered_scores = scores - scores.mean()
    rank = min(
        max(int(components), 1),
        len(hidden) - 1,
        hidden.shape[1],
    )
    gram = centered_hidden @ centered_hidden.T
    eigenvalues, left = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order[:rank]], 0.0)
    left = left[:, order[:rank]]
    active = eigenvalues > EPS
    if not active.any():
        raise ValueError("Hidden states have no nonzero principal components")
    eigenvalues = eigenvalues[active]
    left = left[:, active]
    principal_axes = (
        left.T @ centered_hidden
    ) / np.sqrt(eigenvalues)[:, None]
    latent = centered_hidden @ principal_axes.T
    latent_scale = latent.std(axis=0, ddof=1)
    latent_scale[latent_scale <= EPS] = 1.0
    whitened = latent / latent_scale
    coefficients = whitened.T @ centered_scores
    raw_pattern = principal_axes.T @ (coefficients / latent_scale)
    pattern = unit_vector(raw_pattern)
    return pattern, {
        "retained_components": float(len(eigenvalues)),
        "retained_variance_fraction": float(
            eigenvalues.sum() / max(float(np.trace(gram)), EPS)
        ),
        "fitted_score_correlation": pearson(
            centered_hidden @ pattern,
            centered_scores,
        ),
    }


def cross_source_consensus(
    human_pattern: np.ndarray,
    ai_pattern: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Average semantically oriented Human/AI patterns without sign flipping."""
    human = unit_vector(human_pattern)
    ai = unit_vector(ai_pattern)
    cosine = float(human @ ai)
    consensus = unit_vector(human + ai)
    return consensus, {
        "human_ai_pattern_cosine": cosine,
        "consensus_human_cosine": float(consensus @ human),
        "consensus_ai_cosine": float(consensus @ ai),
    }


def orthonormal_basis(
    vectors: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, list[str]]:
    """Return an orthonormal column basis in deterministic input order."""
    names = list(vectors)
    if not names:
        raise ValueError("At least one vector is required")
    matrix = np.column_stack(
        [unit_vector(vectors[name]) for name in names]
    )
    basis, triangular = np.linalg.qr(matrix, mode="reduced")
    active = np.abs(np.diag(triangular)) > 1e-8
    return basis[:, active], [
        name for name, keep in zip(names, active, strict=True) if keep
    ]


def decompose_source_direction(
    source: np.ndarray,
    concept_patterns: Mapping[str, np.ndarray],
    *,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, float | list[str]]]:
    """Build source, concept, residual and orthogonal-random directions."""
    source = unit_vector(source)
    concept_basis, active_names = orthonormal_basis(concept_patterns)
    projection = concept_basis @ (concept_basis.T @ source)
    residual_raw = source - projection
    residual = unit_vector(residual_raw)

    forbidden = np.column_stack([source, concept_basis])
    forbidden, _ = np.linalg.qr(forbidden, mode="reduced")
    rng = np.random.default_rng(seed)
    random = rng.normal(size=len(source))
    random = random - forbidden @ (forbidden.T @ random)
    random = unit_vector(random)

    directions = {
        "source": source,
        **{
            name: unit_vector(pattern)
            for name, pattern in concept_patterns.items()
        },
        "residual": residual,
        "random": random,
    }
    return directions, {
        "active_concept_patterns": active_names,
        "source_concept_projection_norm": float(
            np.linalg.norm(projection)
        ),
        "source_concept_projection_energy": float(projection @ projection),
        "source_residual_norm_before_normalization": float(
            np.linalg.norm(residual_raw)
        ),
        "residual_concept_max_abs_cosine": float(
            np.max(np.abs(concept_basis.T @ residual))
        ),
        "random_source_abs_cosine": float(abs(random @ source)),
        "random_concept_max_abs_cosine": float(
            np.max(np.abs(concept_basis.T @ random))
        ),
    }
