"""Cross-fitted linguistic reconstruction of paired hidden source shifts."""

from __future__ import annotations

from typing import Iterable

import numpy as np


EPS = 1e-12


def fit_within_class_nullspace_basis(
    human: np.ndarray,
    ai: np.ndarray,
    k: int = 10,
) -> np.ndarray:
    """Return the exact top-k within-class right singular vectors as rows.

    The steering checkpoints use an exact SVD.  Reusing the same estimator here
    prevents the linguistic decomposition from drifting to a slightly different
    nullspace.
    """
    human = np.asarray(human, dtype=np.float32)
    ai = np.asarray(ai, dtype=np.float32)
    within = np.concatenate(
        [human - human.mean(axis=0), ai - ai.mean(axis=0)],
        axis=0,
    )
    _, _, right = np.linalg.svd(within, full_matrices=False)
    return np.asarray(right[: min(k, len(right))], dtype=np.float32)


def project_out_basis(matrix: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Remove row-basis components from one or more hidden-state vectors."""
    matrix = np.asarray(matrix, dtype=np.float32)
    basis = np.asarray(basis, dtype=np.float32)
    return matrix - (matrix @ basis.T) @ basis


def residualize_preserving_mean(
    train: np.ndarray,
    test: np.ndarray,
    train_nuisance: np.ndarray,
    test_nuisance: np.ndarray,
    ridge: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove nuisance slopes learned on train without removing mean shifts."""
    train = np.asarray(train, dtype=np.float32)
    test = np.asarray(test, dtype=np.float32)
    train_nuisance = np.asarray(train_nuisance, dtype=np.float64)
    test_nuisance = np.asarray(test_nuisance, dtype=np.float64)
    nuisance_mean = train_nuisance.mean(axis=0, keepdims=True)
    train_design = train_nuisance - nuisance_mean
    test_design = test_nuisance - nuisance_mean
    gram = train_design.T @ train_design
    penalty = ridge * max(float(np.trace(gram)) / max(len(gram), 1), 1.0)
    coefficients = np.linalg.solve(
        gram + penalty * np.eye(gram.shape[0]),
        train_design.T @ train.astype(np.float64),
    )
    return (
        (train - train_design @ coefficients).astype(np.float32),
        (test - test_design @ coefficients).astype(np.float32),
    )


def scale_features_preserving_mean(
    train: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale by train standard deviations while retaining source mean shifts."""
    train = np.asarray(train, dtype=np.float32)
    test = np.asarray(test, dtype=np.float32)
    scales = train.std(axis=0, ddof=1, keepdims=True)
    scales[scales <= EPS] = 1.0
    return train / scales, test / scales, scales.ravel()


def fit_ridge_reconstruction(
    hidden_differences: np.ndarray,
    feature_differences: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Fit the paired source-signature map ΔH ≈ ΔF @ coefficients.

    The no-intercept constraint makes zero linguistic source contrast map to
    zero reconstructed hidden source contrast.  Unlike an in-sample
    uncentered covariance projection, its validity is assessed only on unseen
    generators through nested cross-fitting.
    """
    hidden = np.asarray(hidden_differences, dtype=np.float64)
    features = np.asarray(feature_differences, dtype=np.float64)
    gram = features.T @ features
    coefficients = np.linalg.solve(
        gram + float(alpha) * np.eye(gram.shape[0]),
        features.T @ hidden,
    )
    return coefficients.astype(np.float32)


def linguistic_mean_direction(
    features: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Return the hidden source-mean contribution predicted by ΔF."""
    features = np.asarray(features, dtype=np.float64)
    coefficients = np.asarray(coefficients, dtype=np.float64)
    return (features.mean(axis=0) @ coefficients).astype(np.float32)


def reconstruction_metrics(
    total_direction: np.ndarray,
    linguistic_direction: np.ndarray,
    *,
    source_axis: np.ndarray | None = None,
) -> dict[str, float]:
    """Evaluate held-out linguistic reconstruction of a hidden source shift."""
    total = np.asarray(total_direction, dtype=np.float64)
    linguistic = np.asarray(linguistic_direction, dtype=np.float64)
    residual = total - linguistic
    total_norm = float(np.linalg.norm(total))
    linguistic_norm = float(np.linalg.norm(linguistic))
    if source_axis is None:
        axis = total / max(total_norm, EPS)
    else:
        axis = np.asarray(source_axis, dtype=np.float64)
        axis = axis / max(float(np.linalg.norm(axis)), EPS)
    axis_total = float(total @ axis)
    axis_linguistic = float(linguistic @ axis)
    cosine = (
        float(total @ linguistic) / max(total_norm * linguistic_norm, EPS)
    )
    vector_r2 = 1.0 - float(residual @ residual) / max(
        float(total @ total),
        EPS,
    )
    aligned_fraction = float(total @ linguistic) / max(
        float(total @ total),
        EPS,
    )
    return {
        "source_axis_total": axis_total,
        "source_axis_linguistic": axis_linguistic,
        "source_axis_reconstructed_fraction": (
            axis_linguistic / axis_total if abs(axis_total) > EPS else 0.0
        ),
        "direction_cosine": cosine,
        "aligned_fraction": aligned_fraction,
        "linguistic_to_total_norm_ratio": linguistic_norm
        / max(total_norm, EPS),
        "vector_reconstruction_r2": vector_r2,
        "residual_to_total_norm_ratio": float(np.linalg.norm(residual))
        / max(total_norm, EPS),
    }


def select_ridge_alpha(
    scores: dict[float, Iterable[float]],
) -> tuple[float, dict[float, float]]:
    """Select the alpha with best mean held-out vector reconstruction."""
    means = {
        float(alpha): float(np.mean(list(values)))
        for alpha, values in scores.items()
    }
    best = max(means.values())
    # Prefer stronger regularization on exact ties.
    selected = max(alpha for alpha, value in means.items() if value == best)
    return selected, means


def orthonormal_union(vectors: Iterable[np.ndarray]) -> np.ndarray:
    """Return an orthonormal basis spanning non-zero input vectors."""
    columns: list[np.ndarray] = []
    for vector in vectors:
        value = np.asarray(vector, dtype=np.float64)
        if value.ndim == 1:
            value = value[:, None]
        if value.size and np.linalg.norm(value) > EPS:
            columns.append(value)
    if not columns:
        raise ValueError("At least one non-zero vector is required")
    concatenated = np.concatenate(columns, axis=1)
    q, r = np.linalg.qr(concatenated, mode="reduced")
    active = np.abs(np.diag(r)) > 1e-8
    return q[:, active].astype(np.float32)
