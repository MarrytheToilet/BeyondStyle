"""Nullspace-constrained pullback directions for causal concept steering.

An activation pattern is an encoding direction: it identifies where a concept
is readable. It is not generally the input perturbation that most efficiently
changes a downstream concept. For an input-layer perturbation ``delta`` and a
downstream concept readout ``v``, the first-order causal direction is

    E_x[J(x)^T v],

where ``J`` is the Jacobian from the intervention site to the downstream
residual stream. This module contains the estimator-side operations applied to
gradients collected from a frozen language model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .concept_patterns import cross_source_consensus
from .projection_mapping import project_out_basis, unit_vector


EPS = 1e-12


def layer_output_dir(root: Path, layer: int) -> Path:
    """Return the canonical directory for a one-based decoder layer."""
    if layer < 1:
        raise ValueError("layer must be a positive one-based index")
    return root / f"layer{layer:02d}"


def weighted_gradient_mean(
    gradients: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Return a weighted mean over gradient rows with strict validation."""
    gradients = np.asarray(gradients, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if gradients.ndim != 2:
        raise ValueError("gradients must be a 2D matrix")
    if weights.shape != (len(gradients),):
        raise ValueError("weights must contain one value per gradient row")
    if np.any(weights < 0) or float(weights.sum()) <= EPS:
        raise ValueError("weights must be nonnegative with positive total")
    return np.average(gradients, axis=0, weights=weights)


def stratified_permutation(
    strata: np.ndarray,
    *,
    seed: int,
) -> np.ndarray:
    """Permute row correspondence independently inside each environment."""
    strata = np.asarray(strata)
    if strata.ndim != 1 or len(strata) < 2:
        raise ValueError("strata must be a one-dimensional nontrivial array")
    rng = np.random.default_rng(seed)
    permutation = np.arange(len(strata))
    for value in np.unique(strata):
        indices = np.flatnonzero(strata == value)
        if len(indices) > 1:
            permutation[indices] = rng.permutation(indices)
    return permutation


def normalized_curve_area(
    layers: np.ndarray,
    values: np.ndarray,
) -> float:
    """Return a depth-normalized trapezoidal area for one layerwise curve.

    ``values`` should already use the scientifically meaningful scale (for
    example, absolute transport capacity). Decoder-layer indices are used as
    the integration coordinate, so irregularly sampled layers do not receive
    equal weight accidentally. Dividing by the covered depth makes the result
    an average curve height and therefore comparable across model depths.
    """
    layers = np.asarray(layers, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if layers.ndim != 1 or values.ndim != 1:
        raise ValueError("layers and values must be one-dimensional")
    if layers.shape != values.shape or len(layers) < 2:
        raise ValueError("a curve requires at least two aligned points")
    if not np.all(np.isfinite(layers)) or not np.all(np.isfinite(values)):
        raise ValueError("curve coordinates must be finite")
    order = np.argsort(layers)
    ordered_layers = layers[order]
    ordered_values = values[order]
    if np.any(np.diff(ordered_layers) <= 0):
        raise ValueError("layer indices must be unique")
    covered_depth = float(ordered_layers[-1] - ordered_layers[0])
    if covered_depth <= EPS:
        raise ValueError("curve must span positive decoder depth")
    return float(np.trapz(ordered_values, ordered_layers) / covered_depth)


def finite_null_p_value(
    observed: float,
    null_statistics: np.ndarray,
) -> float:
    """Return the conservative finite-ensemble upper-tail p-value.

    Each element of ``null_statistics`` must be one independently constructed
    null pipeline. The add-one correction prevents a zero p-value and makes
    the attainable resolution explicit (for example, 1/101 for 100 nulls).
    """
    null_statistics = np.asarray(null_statistics, dtype=np.float64)
    if null_statistics.ndim != 1 or len(null_statistics) < 1:
        raise ValueError("null_statistics must be a nonempty vector")
    if not np.isfinite(observed) or not np.all(np.isfinite(null_statistics)):
        raise ValueError("observed and null statistics must be finite")
    exceedances = int(np.count_nonzero(null_statistics >= observed))
    return float((exceedances + 1) / (len(null_statistics) + 1))


def nullspace_pullback_consensus(
    human_gradient: np.ndarray,
    ai_gradient: np.ndarray,
    within_class_basis: np.ndarray,
) -> tuple[np.ndarray, dict[str, float]]:
    """Project source-specific pullbacks into nullspace and form a consensus."""
    human_gradient = np.asarray(human_gradient, dtype=np.float64)
    ai_gradient = np.asarray(ai_gradient, dtype=np.float64)
    if human_gradient.shape != ai_gradient.shape:
        raise ValueError("Human and AI gradients must have equal shape")
    if human_gradient.ndim != 1:
        raise ValueError("Gradients must be one-dimensional")
    human_null = project_out_basis(
        human_gradient[None, :],
        within_class_basis,
    )[0]
    ai_null = project_out_basis(
        ai_gradient[None, :],
        within_class_basis,
    )[0]
    human_norm = float(np.linalg.norm(human_null))
    ai_norm = float(np.linalg.norm(ai_null))
    if human_norm <= EPS or ai_norm <= EPS:
        raise ValueError("Nullspace projection removed the pullback gradient")
    consensus, diagnostics = cross_source_consensus(
        human_null,
        ai_null,
    )
    return unit_vector(consensus), {
        **diagnostics,
        "human_raw_norm": float(np.linalg.norm(human_gradient)),
        "ai_raw_norm": float(np.linalg.norm(ai_gradient)),
        "human_null_norm": human_norm,
        "ai_null_norm": ai_norm,
        "human_null_retained_fraction": (
            human_norm / max(float(np.linalg.norm(human_gradient)), EPS)
        ),
        "ai_null_retained_fraction": (
            ai_norm / max(float(np.linalg.norm(ai_gradient)), EPS)
        ),
    }
