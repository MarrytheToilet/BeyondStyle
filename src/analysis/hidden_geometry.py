"""Geometry diagnostics for paired human/AI hidden states.

The functions in this module deliberately operate on paired displacements
``AI - Human``.  That keeps the shared prefix and much of the item-specific
content as a matched-pair control instead of treating the 1,000 texts as
independent observations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from sklearn.model_selection import KFold


EPS = 1e-12


def _as_float_tensor(values: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.detach().float()
    return torch.as_tensor(values, dtype=torch.float32)


def unit_vector(vector: torch.Tensor) -> torch.Tensor:
    """Return a unit vector, preserving an all-zero input."""
    vector = _as_float_tensor(vector)
    norm = vector.norm()
    return vector / norm if norm > EPS else vector


def row_cosines(matrix: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Cosine of every matrix row with ``direction``."""
    matrix = _as_float_tensor(matrix)
    direction = unit_vector(direction).to(matrix.device)
    norms = matrix.norm(dim=1).clamp_min(EPS)
    return (matrix @ direction) / norms


def normalize_rows(matrix: torch.Tensor | np.ndarray) -> torch.Tensor:
    """L2-normalize rows while preserving all-zero rows."""
    matrix = _as_float_tensor(matrix)
    return matrix / matrix.norm(dim=1, keepdim=True).clamp_min(EPS)


def centered_gram(features: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Return a centered, unit-Frobenius sample Gram matrix.

    Normalizing here makes the Frobenius inner product between two returned
    matrices equal to linear centered-kernel alignment (CKA).  Gram matrices
    allow comparisons across models with different hidden sizes.
    """
    features = _as_float_tensor(features)
    centered = features - features.mean(dim=0, keepdim=True)
    gram = centered @ centered.T
    if gram.shape[0]:
        gram = (
            gram
            - gram.mean(dim=0, keepdim=True)
            - gram.mean(dim=1, keepdim=True)
            + gram.mean()
        )
    norm = gram.norm()
    return gram / norm if norm > EPS else gram


def leading_residual_scores(
    paired_displacements: torch.Tensor | np.ndarray,
) -> torch.Tensor:
    """Scores on the leading PC after removing the mean displacement.

    The sign of a principal component is arbitrary. Downstream association
    analyses should therefore use absolute correlations or otherwise orient
    the score by a prespecified variable.
    """
    displacements = _as_float_tensor(paired_displacements)
    centered = displacements - displacements.mean(dim=0, keepdim=True)
    if centered.shape[0] < 2:
        return torch.zeros(
            centered.shape[0],
            dtype=centered.dtype,
            device=centered.device,
        )
    gram = centered @ centered.T / (centered.shape[0] - 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    leading_value = eigenvalues[-1].clamp_min(0)
    return eigenvectors[:, -1] * torch.sqrt(
        leading_value * (centered.shape[0] - 1)
    )


def linear_cka_from_grams(
    first_gram: torch.Tensor | np.ndarray,
    second_gram: torch.Tensor | np.ndarray,
) -> float:
    """Linear CKA between centered, normalized sample Gram matrices."""
    first = _as_float_tensor(first_gram)
    second = _as_float_tensor(second_gram).to(first.device)
    if first.shape != second.shape:
        raise ValueError(
            f"CKA Gram matrices must have identical shapes, got "
            f"{tuple(first.shape)} and {tuple(second.shape)}"
        )
    first_norm = first.norm()
    second_norm = second.norm()
    if first_norm <= EPS or second_norm <= EPS:
        return 0.0
    value = (first * second).sum() / (first_norm * second_norm)
    return float(value.clamp(0.0, 1.0).item())


def _positive_eigenvalues(features: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Non-negative covariance eigenvalues obtained through the sample Gram."""
    features = _as_float_tensor(features)
    centered = features - features.mean(dim=0, keepdim=True)
    if centered.shape[0] < 2:
        return torch.zeros(0, device=centered.device)
    gram = centered @ centered.T / (centered.shape[0] - 1)
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0)
    tolerance = max(float(eigenvalues.max().item()) * 1e-8, EPS)
    return eigenvalues[eigenvalues > tolerance]


def participation_ratio(eigenvalues: torch.Tensor | np.ndarray) -> float:
    """Effective dimensionality ``(sum λ)^2 / sum λ^2``."""
    values = _as_float_tensor(eigenvalues).clamp_min(0)
    denominator = (values.square()).sum()
    if denominator <= EPS:
        return 0.0
    return float((values.sum().square() / denominator).item())


def effective_rank(eigenvalues: torch.Tensor | np.ndarray) -> float:
    """Entropy-based effective rank, ``exp(H(λ / sum λ))``."""
    values = _as_float_tensor(eigenvalues).clamp_min(0)
    total = values.sum()
    if total <= EPS:
        return 0.0
    probabilities = values / total
    entropy = -(probabilities * probabilities.clamp_min(EPS).log()).sum()
    return float(entropy.exp().item())


def spectral_summary(
    paired_displacements: torch.Tensor | np.ndarray,
    cumulative_cutoffs: Iterable[int] = (1, 5, 10, 20, 50),
) -> dict[str, Any]:
    """Summarize the residual spectrum after removing the mean displacement."""
    eigenvalues = _positive_eigenvalues(paired_displacements)
    descending = torch.flip(eigenvalues, dims=[0])
    total = descending.sum().clamp_min(EPS)
    cumulative = torch.cumsum(descending, dim=0) / total
    result: dict[str, Any] = {
        "participation_ratio": participation_ratio(descending),
        "effective_rank": effective_rank(descending),
        "nonzero_rank": int(descending.numel()),
        "eigenvalues": descending.detach().cpu().numpy(),
    }
    for cutoff in cumulative_cutoffs:
        index = min(cutoff, len(cumulative)) - 1
        result[f"variance_top_{cutoff}"] = (
            float(cumulative[index].item()) if index >= 0 else 0.0
        )
    return result


@dataclass(frozen=True)
class PairAlignment:
    """Cross-validated agreement of pair displacements with a source direction."""

    scores: np.ndarray
    cosines: np.ndarray
    fold_ids: np.ndarray
    metrics: dict[str, float]


def cross_validated_pair_alignment(
    paired_displacements: torch.Tensor | np.ndarray,
    n_folds: int = 5,
    seed: int = 20260724,
) -> PairAlignment:
    """Fit the raw mean direction on training pairs and score held-out pairs.

    Each sample receives a signed projection and a cosine against a direction
    that was fitted without that sample.  This prevents the pair-concordance
    diagnostic from being inflated by using each displacement to define the
    direction against which it is evaluated.
    """
    displacements = _as_float_tensor(paired_displacements).cpu()
    n_pairs = len(displacements)
    if n_pairs < n_folds:
        raise ValueError(
            f"Need at least {n_folds} pairs for {n_folds}-fold alignment, "
            f"got {n_pairs}"
        )

    scores = np.empty(n_pairs, dtype=np.float64)
    cosines = np.empty(n_pairs, dtype=np.float64)
    fold_ids = np.empty(n_pairs, dtype=np.int64)
    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    all_indices = np.arange(n_pairs)
    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(all_indices)
    ):
        direction = unit_vector(displacements[train_indices].mean(dim=0))
        heldout = displacements[test_indices]
        scores[test_indices] = (heldout @ direction).numpy()
        cosines[test_indices] = row_cosines(heldout, direction).numpy()
        fold_ids[test_indices] = fold

    score_std = float(scores.std(ddof=1))
    metrics = {
        "oof_positive_rate": float(np.mean(scores > 0)),
        "oof_projection_mean": float(scores.mean()),
        "oof_projection_std": score_std,
        "oof_paired_d": float(scores.mean() / score_std)
        if score_std > EPS
        else 0.0,
        "oof_cosine_mean": float(cosines.mean()),
        "oof_cosine_median": float(np.median(cosines)),
        "oof_cosine_q10": float(np.quantile(cosines, 0.10)),
        "oof_cosine_q90": float(np.quantile(cosines, 0.90)),
    }
    return PairAlignment(
        scores=scores,
        cosines=cosines,
        fold_ids=fold_ids,
        metrics=metrics,
    )


def bootstrap_direction_stability(
    paired_displacements: torch.Tensor | np.ndarray,
    n_boot: int = 200,
    seed: int = 20260724,
) -> dict[str, float]:
    """Bootstrap stability of the mean source direction.

    Multinomial bootstrap counts avoid materializing a ``B × N × D`` tensor.
    The reported cosine is between every resampled direction and the direction
    fitted on all pairs.
    """
    displacements = _as_float_tensor(paired_displacements)
    n_pairs = len(displacements)
    if not n_pairs:
        raise ValueError("Cannot bootstrap an empty displacement matrix")
    generator = np.random.default_rng(seed)
    counts = generator.multinomial(
        n_pairs,
        np.full(n_pairs, 1.0 / n_pairs),
        size=n_boot,
    )
    count_tensor = torch.as_tensor(
        counts,
        dtype=displacements.dtype,
        device=displacements.device,
    )
    bootstrap_means = (count_tensor @ displacements) / n_pairs
    bootstrap_means /= bootstrap_means.norm(dim=1, keepdim=True).clamp_min(EPS)
    full_direction = unit_vector(displacements.mean(dim=0))
    cosines = (bootstrap_means @ full_direction).detach().cpu().numpy()
    return {
        "bootstrap_direction_cosine_mean": float(cosines.mean()),
        "bootstrap_direction_cosine_lo": float(np.quantile(cosines, 0.025)),
        "bootstrap_direction_cosine_hi": float(np.quantile(cosines, 0.975)),
        "n_boot": int(n_boot),
    }


def _group_between_fraction(
    displacements: torch.Tensor,
    group_codes: np.ndarray,
) -> float:
    global_mean = displacements.mean(dim=0)
    centered = displacements - global_mean
    total_energy = centered.square().sum()
    if total_energy <= EPS:
        return 0.0

    between_energy = torch.zeros(
        (),
        dtype=displacements.dtype,
        device=displacements.device,
    )
    for code in np.unique(group_codes):
        mask = torch.as_tensor(
            group_codes == code,
            dtype=torch.bool,
            device=displacements.device,
        )
        group_mean = displacements[mask].mean(dim=0)
        between_energy += mask.sum() * (group_mean - global_mean).square().sum()
    return float((between_energy / total_energy).item())


def group_direction_geometry(
    paired_displacements: torch.Tensor | np.ndarray,
    groups: Iterable[str],
    n_perm: int = 200,
    seed: int = 20260724,
) -> dict[str, Any]:
    """Decompose displacement geometry into global and group-specific parts.

    ``between_fraction`` is a high-dimensional one-way ANOVA energy ratio.
    A label permutation test evaluates whether named generators or corpora
    explain more displacement heterogeneity than equally sized random groups.
    """
    displacements = _as_float_tensor(paired_displacements)
    group_array = np.asarray(list(groups), dtype=object)
    if len(group_array) != len(displacements):
        raise ValueError(
            f"Expected {len(displacements)} groups, got {len(group_array)}"
        )

    labels, group_codes = np.unique(group_array, return_inverse=True)
    global_direction = unit_vector(displacements.mean(dim=0))
    group_directions: dict[str, torch.Tensor] = {}
    group_cosines: dict[str, float] = {}
    group_sizes: dict[str, int] = {}
    for code, label in enumerate(labels):
        mask = torch.as_tensor(
            group_codes == code,
            dtype=torch.bool,
            device=displacements.device,
        )
        direction = unit_vector(displacements[mask].mean(dim=0))
        name = str(label)
        group_directions[name] = direction
        group_cosines[name] = float((direction @ global_direction).item())
        group_sizes[name] = int(mask.sum().item())

    pairwise_cosines: list[float] = []
    pairwise_cosine_map: dict[str, float] = {}
    for first_index, first in enumerate(labels):
        for second in labels[first_index + 1 :]:
            cosine = float(
                (
                    group_directions[str(first)]
                    @ group_directions[str(second)]
                ).item()
            )
            pairwise_cosines.append(cosine)
            pairwise_cosine_map[f"{first}|||{second}"] = cosine

    observed = _group_between_fraction(displacements, group_codes)
    random = np.random.default_rng(seed)
    permutation_values = np.empty(n_perm, dtype=np.float64)
    for index in range(n_perm):
        permutation_values[index] = _group_between_fraction(
            displacements,
            random.permutation(group_codes),
        )
    p_value = float(
        (1 + np.count_nonzero(permutation_values >= observed)) / (n_perm + 1)
    )

    return {
        "between_fraction": observed,
        "permutation_p": p_value,
        "permutation_mean": float(permutation_values.mean()),
        "group_to_global_cosine_mean": float(
            np.mean(list(group_cosines.values()))
        ),
        "group_to_global_cosine_min": float(
            np.min(list(group_cosines.values()))
        ),
        "pairwise_group_cosine_mean": float(np.mean(pairwise_cosines))
        if pairwise_cosines
        else 1.0,
        "pairwise_group_cosine_min": float(np.min(pairwise_cosines))
        if pairwise_cosines
        else 1.0,
        "group_sizes": group_sizes,
        "group_to_global_cosines": group_cosines,
        "pairwise_group_cosines": pairwise_cosine_map,
    }


def coherent_displacement_fraction(
    paired_displacements: torch.Tensor | np.ndarray,
) -> float:
    """Fraction of mean squared displacement carried by the shared mean."""
    displacements = _as_float_tensor(paired_displacements)
    coherent_energy = displacements.mean(dim=0).square().sum()
    total_energy = displacements.square().sum(dim=1).mean()
    if total_energy <= EPS:
        return 0.0
    return float((coherent_energy / total_energy).item())
