import numpy as np
import torch

from src.analysis.hidden_geometry import (
    bootstrap_direction_stability,
    centered_gram,
    coherent_displacement_fraction,
    cross_validated_pair_alignment,
    effective_rank,
    group_direction_geometry,
    leading_residual_scores,
    linear_cka_from_grams,
    normalize_rows,
    participation_ratio,
)


def test_linear_cka_is_one_for_scaled_identical_geometry():
    features = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]
    )
    first = centered_gram(features)
    second = centered_gram(7.0 * features)
    assert np.isclose(linear_cka_from_grams(first, second), 1.0)


def test_normalize_rows_and_leading_scores():
    values = torch.tensor([[3.0, 4.0], [0.0, 2.0], [0.0, 0.0]])
    normalized = normalize_rows(values)
    assert torch.allclose(normalized[0], torch.tensor([0.6, 0.8]))
    assert torch.allclose(normalized[1], torch.tensor([0.0, 1.0]))
    assert torch.allclose(normalized[2], torch.zeros(2))

    scores = leading_residual_scores(
        torch.tensor([[-2.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    )
    assert scores.shape == (3,)
    assert np.isclose(float(scores.mean()), 0.0, atol=1e-6)
    assert np.isclose(float(scores.abs().max()), 2.0, atol=1e-5)


def test_effective_dimension_matches_equal_spectrum():
    eigenvalues = torch.ones(4)
    assert np.isclose(participation_ratio(eigenvalues), 4.0)
    assert np.isclose(effective_rank(eigenvalues), 4.0)


def test_cross_validated_pair_alignment_does_not_use_heldout_pair():
    displacements = torch.tensor(
        [[1.0, 0.0], [1.1, 0.0], [0.9, 0.0], [1.2, 0.0], [0.8, 0.0]]
    )
    result = cross_validated_pair_alignment(
        displacements,
        n_folds=5,
        seed=7,
    )
    assert np.all(result.scores > 0)
    assert np.allclose(result.cosines, 1.0)
    assert result.metrics["oof_positive_rate"] == 1.0


def test_bootstrap_direction_is_stable_for_collinear_pairs():
    displacements = torch.tensor([[1.0, 0.0]] * 20)
    result = bootstrap_direction_stability(displacements, n_boot=30, seed=9)
    assert np.isclose(result["bootstrap_direction_cosine_mean"], 1.0)
    assert np.isclose(result["bootstrap_direction_cosine_lo"], 1.0)


def test_group_geometry_detects_group_specific_axis():
    first = torch.tensor([[2.0, 1.0], [2.0, 1.2], [2.0, 0.8]])
    second = torch.tensor([[2.0, -1.0], [2.0, -1.2], [2.0, -0.8]])
    result = group_direction_geometry(
        torch.cat([first, second]),
        ["a"] * 3 + ["b"] * 3,
        n_perm=20,
        seed=3,
    )
    assert result["between_fraction"] > 0.9
    assert set(result["group_sizes"]) == {"a", "b"}


def test_coherent_fraction_is_bounded():
    displacements = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    assert np.isclose(coherent_displacement_fraction(displacements), 1.0)
