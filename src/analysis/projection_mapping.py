"""Projection-first mapping between linguistic measurements and source axes.

The central object is a scalar coordinate

    z(x, layer) = h(x, layer)^T v_source(layer)

rather than a reconstruction of the full hidden state.  This module contains
the train-only preprocessing and validation primitives used to test whether
surface linguistic measurements map to that coordinate within a source and
transfer across human/machine sources.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy import linalg
from scipy.stats import rankdata


EPS = 1e-12


def _as_2d(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[:, None] if array.ndim == 1 else array


def unit_vector(vector: np.ndarray) -> np.ndarray:
    """Return a float64 unit vector, preserving an all-zero input."""
    value = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(value))
    return value / norm if norm > EPS else value


def fit_within_class_basis(
    human: np.ndarray,
    ai: np.ndarray,
    *,
    k: int = 10,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit the exact top-k within-class right-singular basis on given rows."""
    human = np.asarray(human, dtype=np.float64)
    ai = np.asarray(ai, dtype=np.float64)
    if human.shape != ai.shape or human.ndim != 2:
        raise ValueError("human and ai must be equally shaped 2D arrays")
    within = np.concatenate(
        [human - human.mean(axis=0), ai - ai.mean(axis=0)],
        axis=0,
    )
    rank = min(max(int(k), 0), min(within.shape))
    if not rank:
        return np.empty((0, within.shape[1]), dtype=np.float64), {
            "null_k": 0.0,
            "within_variance_removed": 0.0,
        }
    gram = within @ within.T
    start = max(len(gram) - rank, 0)
    eigenvalues, left = linalg.eigh(
        gram,
        subset_by_index=(start, len(gram) - 1),
        check_finite=False,
    )
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    left = left[:, order]
    active = eigenvalues > EPS
    basis = (
        left[:, active].T @ within
    ) / np.sqrt(eigenvalues[active])[:, None]
    total = float(np.sum(within * within))
    return basis, {
        "null_k": float(len(basis)),
        "within_variance_removed": float(eigenvalues[active].sum())
        / max(total, EPS),
    }


def project_out_basis(
    values: np.ndarray,
    basis: np.ndarray,
) -> np.ndarray:
    """Remove components in a row-orthonormal basis."""
    values = np.asarray(values, dtype=np.float64)
    basis = np.asarray(basis, dtype=np.float64)
    return values - (values @ basis.T) @ basis


def fit_nullspace_source_direction(
    human: np.ndarray,
    ai: np.ndarray,
    *,
    k: int = 10,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit an AI-oriented mean direction after removing within-class PCs.

    Only the supplied rows are used.  The right-singular subspace is recovered
    through the sample Gram matrix, which is exact for the requested top-k
    subspace but substantially cheaper than a full SVD when hidden size is
    much larger than sample count.
    """
    human = np.asarray(human, dtype=np.float64)
    ai = np.asarray(ai, dtype=np.float64)
    if human.shape != ai.shape or human.ndim != 2:
        raise ValueError("human and ai must be equally shaped 2D arrays")
    raw = unit_vector((ai - human).mean(axis=0))
    basis, basis_info = fit_within_class_basis(human, ai, k=k)
    residual = raw - (raw @ basis.T) @ basis
    direction = unit_vector(residual)
    if float(ai.mean(axis=0) @ direction) < float(
        human.mean(axis=0) @ direction
    ):
        direction = -direction
    return direction, {
        **basis_info,
        "residual_norm": float(np.linalg.norm(residual)),
        "cosine_vs_raw": float(direction @ raw),
    }


def project_source_coordinate(
    hidden: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """Project one hidden-state matrix onto a unit source direction."""
    return np.asarray(hidden, dtype=np.float64) @ unit_vector(direction)


@dataclass(frozen=True)
class NuisanceResidualizer:
    """Train-fitted linear nuisance model with standardized covariates."""

    nuisance_mean: np.ndarray
    nuisance_scale: np.ndarray
    coefficients: np.ndarray

    def transform(
        self,
        values: np.ndarray,
        nuisance: np.ndarray,
    ) -> np.ndarray:
        values_2d = _as_2d(values)
        nuisance = _as_2d(nuisance)
        standardized = (
            nuisance - self.nuisance_mean
        ) / self.nuisance_scale
        design = np.column_stack([np.ones(len(standardized)), standardized])
        residual = values_2d - design @ self.coefficients
        return residual[:, 0] if np.asarray(values).ndim == 1 else residual


def fit_nuisance_residualizer(
    values: np.ndarray,
    nuisance: np.ndarray,
    *,
    ridge: float = 1e-6,
) -> NuisanceResidualizer:
    """Fit a nuisance regression on training rows only."""
    values_2d = _as_2d(values)
    nuisance = _as_2d(nuisance)
    nuisance_mean = nuisance.mean(axis=0, keepdims=True)
    nuisance_scale = nuisance.std(axis=0, ddof=1, keepdims=True)
    nuisance_scale[nuisance_scale <= EPS] = 1.0
    standardized = (nuisance - nuisance_mean) / nuisance_scale
    design = np.column_stack([np.ones(len(standardized)), standardized])
    gram = design.T @ design
    penalty = np.eye(gram.shape[0]) * float(ridge)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(
        gram + penalty,
        design.T @ values_2d,
    )
    return NuisanceResidualizer(
        nuisance_mean=nuisance_mean,
        nuisance_scale=nuisance_scale,
        coefficients=coefficients,
    )


@dataclass(frozen=True)
class PreparedMapping:
    """Nuisance-residualized, train-standardized mapping arrays."""

    train_features: np.ndarray
    test_features: np.ndarray
    train_scores: np.ndarray
    test_scores: np.ndarray
    feature_scale: np.ndarray
    score_scale: float


def prepare_mapping_split(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_scores: np.ndarray,
    test_scores: np.ndarray,
    train_nuisance: np.ndarray,
    test_nuisance: np.ndarray,
) -> PreparedMapping:
    """Fit every preprocessing operation on train and apply it to test."""
    feature_residualizer = fit_nuisance_residualizer(
        train_features,
        train_nuisance,
    )
    score_residualizer = fit_nuisance_residualizer(
        train_scores,
        train_nuisance,
    )
    train_x = feature_residualizer.transform(
        train_features,
        train_nuisance,
    )
    test_x = feature_residualizer.transform(
        test_features,
        test_nuisance,
    )
    train_y = score_residualizer.transform(train_scores, train_nuisance)
    test_y = score_residualizer.transform(test_scores, test_nuisance)

    feature_scale = train_x.std(axis=0, ddof=1)
    feature_scale[feature_scale <= EPS] = 1.0
    score_scale = float(train_y.std(ddof=1))
    if score_scale <= EPS:
        score_scale = 1.0
    return PreparedMapping(
        train_features=train_x / feature_scale,
        test_features=test_x / feature_scale,
        train_scores=train_y / score_scale,
        test_scores=test_y / score_scale,
        feature_scale=feature_scale,
        score_scale=score_scale,
    )


def fit_ridge_projection_map(
    features: np.ndarray,
    scores: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    """Fit a no-intercept ridge map from surface features to projection."""
    features = np.asarray(features, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    gram = features.T @ features
    return np.linalg.solve(
        gram + float(alpha) * np.eye(gram.shape[0]),
        features.T @ scores,
    )


def safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
    """Pearson correlation with a defined zero for degenerate inputs."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.std(ddof=1) <= EPS or right.std(ddof=1) <= EPS:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def safe_spearman(left: np.ndarray, right: np.ndarray) -> float:
    """Spearman correlation with a defined zero for degenerate inputs."""
    return safe_pearson(rankdata(left), rankdata(right))


def projection_mapping_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> dict[str, float]:
    """Return scale-aware and scale-free projection mapping metrics."""
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    pearson = safe_pearson(actual, predicted)
    centered_actual = actual - actual.mean()
    centered_predicted = predicted - predicted.mean()
    denominator = float(centered_actual @ centered_actual)
    centered_r2 = 1.0 - float(
        np.sum((centered_actual - centered_predicted) ** 2)
    ) / max(denominator, EPS)
    actual_sign = np.sign(actual - np.median(actual))
    predicted_sign = np.sign(predicted - np.median(predicted))
    comparable = (actual_sign != 0) & (predicted_sign != 0)
    sign_agreement = (
        float(np.mean(actual_sign[comparable] == predicted_sign[comparable]))
        if comparable.any()
        else 0.0
    )
    return {
        "pearson": pearson,
        "spearman": safe_spearman(actual, predicted),
        "correlation_geometry": pearson * pearson,
        "centered_r2": centered_r2,
        "sign_agreement": sign_agreement,
    }


def select_ridge_alpha(
    features: np.ndarray,
    scores: np.ndarray,
    nuisance: np.ndarray,
    environments: np.ndarray,
    alphas: Iterable[float],
) -> tuple[float, dict[float, float]]:
    """Select ridge strength by leave-one-environment-out train MSE."""
    features = np.asarray(features, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    nuisance = np.asarray(nuisance, dtype=np.float64)
    environments = np.asarray(environments)
    values = [float(alpha) for alpha in alphas]
    losses = {alpha: [] for alpha in values}
    unique = np.unique(environments)
    if len(unique) < 2:
        return max(values), {alpha: float("nan") for alpha in values}
    for heldout in unique:
        train = environments != heldout
        test = environments == heldout
        prepared = prepare_mapping_split(
            features[train],
            features[test],
            scores[train],
            scores[test],
            nuisance[train],
            nuisance[test],
        )
        for alpha in values:
            coefficients = fit_ridge_projection_map(
                prepared.train_features,
                prepared.train_scores,
                alpha=alpha,
            )
            prediction = prepared.test_features @ coefficients
            losses[alpha].append(
                float(np.mean((prepared.test_scores - prediction) ** 2))
            )
    means = {
        alpha: float(np.mean(fold_losses))
        for alpha, fold_losses in losses.items()
    }
    best_loss = min(means.values())
    selected = max(
        alpha for alpha, loss in means.items() if loss == best_loss
    )
    return selected, means


def permute_within_environments(
    values: np.ndarray,
    environments: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Permute labels within environments to retain environment marginals."""
    values = np.asarray(values)
    environments = np.asarray(environments)
    result = values.copy()
    for environment in np.unique(environments):
        indices = np.flatnonzero(environments == environment)
        result[indices] = values[rng.permutation(indices)]
    return result
