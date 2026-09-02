"""Group-level causal alignment between steering changes and source signatures."""

from __future__ import annotations

import numpy as np


EPS = 1e-12


def set_overlap_metrics(
    reference: set[str],
    candidate: set[str],
) -> tuple[float, float, bool]:
    """Return reference recall, candidate precision, and opportunity flag."""
    reference = {value for value in reference if value}
    candidate = {value for value in candidate if value}
    opportunity = bool(reference)
    recall = (
        len(reference & candidate) / len(reference)
        if opportunity
        else 1.0
    )
    precision = (
        len(reference & candidate) / len(candidate)
        if candidate
        else float(not opportunity)
    )
    return float(recall), float(precision), opportunity


def source_signature_projection(
    observational_differences: np.ndarray,
    intervention_differences: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Project intervention effects onto the standardized AI−Human signature."""
    observational = np.asarray(observational_differences, dtype=np.float64)
    intervention = np.asarray(intervention_differences, dtype=np.float64)
    scales = observational.std(axis=0, ddof=1)
    active = scales > EPS
    if not np.any(active):
        return np.zeros(len(intervention)), 0.0
    observational = observational[:, active] / scales[active]
    intervention = intervention[:, active] / scales[active]
    direction = observational.mean(axis=0)
    norm = np.linalg.norm(direction)
    if norm <= EPS:
        return np.zeros(len(intervention)), 0.0
    direction /= norm
    projections = intervention @ direction
    intervention_mean = intervention.mean(axis=0)
    mean_norm = np.linalg.norm(intervention_mean)
    cosine = (
        float(intervention_mean @ direction / mean_norm)
        if mean_norm > EPS
        else 0.0
    )
    return projections, cosine


def paired_mean_inference(
    values: np.ndarray,
    *,
    bootstrap: int,
    permutations: int,
    seed: int,
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    observed = float(values.mean())
    sample_sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    if bootstrap:
        indices = rng.integers(0, len(values), size=(bootstrap, len(values)))
        bootstrap_means = values[indices].mean(axis=1)
        ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
    else:
        ci_low = ci_high = observed
    if permutations:
        signs = rng.choice(
            np.array([-1.0, 1.0]),
            size=(permutations, len(values)),
        )
        null_means = (signs * values).mean(axis=1)
        p_value = (1 + np.sum(np.abs(null_means) >= abs(observed))) / (
            permutations + 1
        )
    else:
        p_value = 1.0
    return {
        "mean": observed,
        "median": float(np.median(values)),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "paired_d": observed / sample_sd if sample_sd > EPS else 0.0,
        "positive_rate": float(np.mean(values > 0)),
        "sign_flip_p": float(p_value),
    }


def axis_ensemble_calibration(
    learned_values: np.ndarray,
    random_values: np.ndarray,
) -> dict[str, float]:
    """Calibrate one signed learned effect against random axis-level means.

    ``learned_values`` contains one value per prompt. ``random_values`` has
    shape ``(n_random_axes, n_prompts)``. Random-axis poles are arbitrary, so
    the finite-ensemble comparison is two-sided and uses absolute axis means.
    The +1 correction yields a finite-ensemble tail fraction.  It is a
    descriptive rank calibration, not an inferential p-value: the learned
    axis is target-fitted and is not exchangeable with random axes.
    """
    learned = np.asarray(learned_values, dtype=np.float64).reshape(-1)
    random = np.asarray(random_values, dtype=np.float64)
    if random.ndim != 2 or random.shape[1] != len(learned):
        raise ValueError(
            "random_values must have shape (n_axes, n_learned_prompts)"
        )
    if len(learned) < 1 or len(random) < 2:
        raise ValueError("calibration needs prompts and at least two axes")
    learned_mean = float(learned.mean())
    random_means = random.mean(axis=1)
    learned_magnitude = abs(learned_mean)
    random_magnitudes = np.abs(random_means)
    exceedances = int(np.sum(random_magnitudes >= learned_magnitude))
    rank = 1 + int(np.sum(random_magnitudes > learned_magnitude))
    return {
        "learned_signed_mean": learned_mean,
        "learned_abs_axis_mean": learned_magnitude,
        "random_signed_mean": float(random_means.mean()),
        "random_abs_axis_mean": float(random_magnitudes.mean()),
        "random_abs_axis_median": float(np.median(random_magnitudes)),
        "random_abs_axis_min": float(random_magnitudes.min()),
        "random_abs_axis_max": float(random_magnitudes.max()),
        "learned_abs_rank": float(rank),
        "n_random_axes": float(len(random)),
        "finite_ensemble_tail_fraction": float(
            (1 + exceedances) / (1 + len(random))
        ),
    }
