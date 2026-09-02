"""Diagnostics for the propagation of a fixed residual-stream intervention.

The functions in this module operate on hidden states obtained from the same
teacher-forced token sequence under negative, zero, and positive steering.
Keeping the tokens fixed separates network propagation from sampling-trajectory
divergence during free generation.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


EPS = 1e-12
CONDITIONS = (-1.0, 0.0, 1.0)


def normalized_position_slices(
    length: int,
    bins: int = 4,
) -> list[tuple[str, slice]]:
    """Return an all-token slice followed by approximately equal quantiles."""
    if length < 1:
        raise ValueError("length must be positive")
    if bins < 1:
        raise ValueError("bins must be positive")
    boundaries = np.linspace(0, length, bins + 1, dtype=int)
    result = [("all", slice(0, length))]
    for index, (start, end) in enumerate(
        zip(boundaries[:-1], boundaries[1:]),
        start=1,
    ):
        if end > start:
            result.append((f"q{index}", slice(int(start), int(end))))
    return result


def orthogonal_random_directions(
    forbidden: Mapping[str, np.ndarray],
    *,
    count: int,
    seed: int,
) -> np.ndarray:
    """Sample deterministic orthonormal controls outside a named subspace."""
    if count < 1:
        raise ValueError("count must be positive")
    if not forbidden:
        raise ValueError("forbidden must contain at least one direction")
    vectors = []
    hidden_size = None
    for name, vector in forbidden.items():
        vector = np.asarray(vector, dtype=np.float64)
        if vector.ndim != 1:
            raise ValueError(f"Direction {name!r} must be one-dimensional")
        if hidden_size is None:
            hidden_size = len(vector)
        if len(vector) != hidden_size:
            raise ValueError("Forbidden directions must have equal size")
        norm = float(np.linalg.norm(vector))
        if norm <= EPS:
            raise ValueError(f"Direction {name!r} has zero norm")
        vectors.append(vector / norm)
    forbidden_matrix = np.column_stack(vectors)
    left, singular_values, _ = np.linalg.svd(
        forbidden_matrix,
        full_matrices=False,
    )
    tolerance = (
        max(forbidden_matrix.shape)
        * np.finfo(np.float64).eps
        * float(singular_values[0])
    )
    rank = int(np.count_nonzero(singular_values > tolerance))
    basis = left[:, :rank]
    rng = np.random.default_rng(seed)
    controls = []
    for _ in range(count):
        candidate = rng.normal(size=hidden_size)
        candidate -= basis @ (basis.T @ candidate)
        if controls:
            control_basis = np.column_stack(controls)
            candidate -= control_basis @ (control_basis.T @ candidate)
        norm = float(np.linalg.norm(candidate))
        if norm <= EPS:
            raise RuntimeError("Failed to sample an orthogonal random control")
        controls.append(candidate / norm)
    return np.stack(controls)


def summarize_counterfactual_states(
    condition_states: Mapping[
        float,
        Mapping[int, np.ndarray],
    ],
    directions: Mapping[str, np.ndarray],
    *,
    target: str,
    intervention_scale: float,
    position_bins: int = 4,
) -> list[dict[str, float | int | str]]:
    """Summarize hidden responses for one fixed token sequence.

    ``condition_states`` must contain the identical sequence under signed
    strengths ``-1``, ``0``, and ``+1``. For each layer and normalized token
    region, the response is

    ``half_delta = (h(+1) - h(-1)) / 2``.

    With a unit target direction and an additive intervention of norm
    ``intervention_scale``, the target projection at the injection layer is
    expected to equal that scale. The ratio is therefore a directly
    interpretable propagation-retention measure.
    """
    missing = sorted(set(CONDITIONS) - set(condition_states))
    if missing:
        raise ValueError(f"Missing counterfactual conditions: {missing}")
    if target not in directions:
        raise KeyError(f"Unknown target direction: {target}")
    if intervention_scale <= 0:
        raise ValueError("intervention_scale must be positive")

    layers = sorted(condition_states[0.0])
    if any(set(condition_states[value]) != set(layers) for value in CONDITIONS):
        raise ValueError("All conditions must contain the same layers")

    prepared_directions = {}
    for name, vector in directions.items():
        vector = np.asarray(vector, dtype=np.float64)
        norm = float(np.linalg.norm(vector))
        if norm <= EPS:
            raise ValueError(f"Direction {name!r} has zero norm")
        prepared_directions[name] = vector / norm
    target_vector = prepared_directions[target]

    rows: list[dict[str, float | int | str]] = []
    for layer in layers:
        states = {
            value: np.asarray(
                condition_states[value][layer],
                dtype=np.float64,
            )
            for value in CONDITIONS
        }
        reference_shape = states[0.0].shape
        if len(reference_shape) != 2:
            raise ValueError(
                f"Layer {layer} states must have shape (tokens, hidden)"
            )
        if any(states[value].shape != reference_shape for value in CONDITIONS):
            raise ValueError(
                f"Layer {layer} condition shapes do not match"
            )
        if any(
            len(vector) != reference_shape[1]
            for vector in prepared_directions.values()
        ):
            raise ValueError(
                f"Layer {layer} hidden size does not match the directions"
            )

        for position_scope, position_slice in normalized_position_slices(
            reference_shape[0],
            bins=position_bins,
        ):
            means = {
                value: states[value][position_slice].mean(axis=0)
                for value in CONDITIONS
            }
            half_delta = 0.5 * (means[1.0] - means[-1.0])
            curvature = 0.5 * (means[1.0] + means[-1.0]) - means[0.0]
            response_norm = float(np.linalg.norm(half_delta))
            target_effect = float(half_delta @ target_vector)
            target_alignment = (
                target_effect / response_norm
                if response_norm > EPS
                else 0.0
            )
            curvature_norm = float(np.linalg.norm(curvature))

            for readout, vector in prepared_directions.items():
                rows.append({
                    "layer": int(layer),
                    "position_scope": position_scope,
                    "n_tokens": int(
                        states[0.0][position_slice].shape[0]
                    ),
                    "target": target,
                    "readout": readout,
                    "negative_projection": float(means[-1.0] @ vector),
                    "baseline_projection": float(means[0.0] @ vector),
                    "positive_projection": float(means[1.0] @ vector),
                    "projection_effect": float(half_delta @ vector),
                    "projection_retention": float(
                        (half_delta @ vector) / intervention_scale
                    ),
                    "response_norm": response_norm,
                    "response_norm_retention": (
                        response_norm / intervention_scale
                    ),
                    "target_effect": target_effect,
                    "target_retention": target_effect / intervention_scale,
                    "target_alignment": target_alignment,
                    "curvature_norm": curvature_norm,
                    "curvature_ratio": curvature_norm / intervention_scale,
                })
    return rows
