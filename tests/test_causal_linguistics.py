import numpy as np

from src.analysis.causal_linguistics import (
    axis_ensemble_calibration,
    paired_mean_inference,
    set_overlap_metrics,
    source_signature_projection,
)


def test_signature_projection_orients_toward_observational_shift():
    observational = np.array(
        [[1.0, 2.0], [1.2, 1.8], [0.8, 2.2]],
    )
    aligned = np.array([[0.5, 1.0], [1.0, 2.0]])
    opposed = -aligned
    aligned_scores, aligned_cosine = source_signature_projection(
        observational,
        aligned,
    )
    opposed_scores, opposed_cosine = source_signature_projection(
        observational,
        opposed,
    )
    assert np.all(aligned_scores > 0)
    assert np.all(opposed_scores < 0)
    assert aligned_cosine > 0.99
    assert opposed_cosine < -0.99


def test_paired_inference_is_deterministic():
    first = paired_mean_inference(
        np.array([1.0, 2.0, 3.0]),
        bootstrap=100,
        permutations=100,
        seed=7,
    )
    second = paired_mean_inference(
        np.array([1.0, 2.0, 3.0]),
        bootstrap=100,
        permutations=100,
        seed=7,
    )
    assert first == second
    assert first["mean"] == 2.0


def test_set_overlap_metrics_handles_opportunities_and_empty_sets():
    recall, precision, opportunity = set_overlap_metrics(
        {"alpha", "beta"},
        {"beta", "gamma"},
    )
    assert recall == 0.5
    assert precision == 0.5
    assert opportunity
    recall, precision, opportunity = set_overlap_metrics(set(), set())
    assert recall == 1.0
    assert precision == 1.0
    assert not opportunity


def test_axis_ensemble_calibration_is_pole_invariant():
    learned = np.array([2.0, 4.0])
    random = np.array([[1.0, 1.0], [-2.0, -2.0], [0.0, 1.0]])

    first = axis_ensemble_calibration(learned, random)
    second = axis_ensemble_calibration(-learned, -random)

    assert first["learned_signed_mean"] == 3.0
    assert first["learned_abs_rank"] == 1.0
    assert np.isclose(first["finite_ensemble_tail_fraction"], 0.25)
    assert first["learned_abs_axis_mean"] == second["learned_abs_axis_mean"]
    assert first["random_abs_axis_mean"] == second["random_abs_axis_mean"]
