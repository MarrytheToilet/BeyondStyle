import numpy as np

from src.analysis.causal_pullback import (
    finite_null_p_value,
    layer_output_dir,
    nullspace_pullback_consensus,
    normalized_curve_area,
    stratified_permutation,
    weighted_gradient_mean,
)


def test_pullback_outputs_use_the_requested_input_layer(tmp_path):
    assert layer_output_dir(tmp_path, 7) == tmp_path / "layer07"
    assert layer_output_dir(tmp_path, 12) == tmp_path / "layer12"


def test_weighted_gradient_mean_respects_batch_sizes():
    gradients = np.array([[1.0, 0.0], [3.0, 2.0]])
    weights = np.array([1.0, 3.0])

    result = weighted_gradient_mean(gradients, weights)

    assert np.allclose(result, np.array([2.5, 1.5]))


def test_nullspace_pullback_removes_forbidden_axis():
    human = np.array([2.0, 3.0, 0.0])
    ai = np.array([4.0, 2.0, 0.0])
    basis = np.array([[1.0, 0.0, 0.0]])

    direction, diagnostics = nullspace_pullback_consensus(
        human,
        ai,
        basis,
    )

    assert np.isclose(direction[0], 0.0)
    assert direction[1] > 0
    assert diagnostics["human_ai_pattern_cosine"] > 0.99
    assert diagnostics["human_null_retained_fraction"] < 1.0


def test_nullspace_pullback_rejects_opposite_source_gradients_without_flip():
    human = np.array([0.0, 1.0, 0.0])
    ai = np.array([0.0, -1.0, 0.0])

    try:
        nullspace_pullback_consensus(
            human,
            ai,
            np.empty((0, 3)),
        )
    except ValueError as error:
        assert "degenerate" in str(error)
    else:
        raise AssertionError("opposite gradients must not be sign-flipped")


def test_stratified_permutation_preserves_environment_membership():
    strata = np.array(["a", "a", "a", "b", "b", "b"])

    first = stratified_permutation(strata, seed=7)
    second = stratified_permutation(strata, seed=7)

    assert np.array_equal(first, second)
    assert np.array_equal(strata[first], strata)
    assert sorted(first.tolist()) == list(range(len(strata)))


def test_normalized_curve_area_respects_irregular_layer_spacing():
    layers = np.array([8, 12, 20])
    values = np.array([0.0, 1.0, 1.0])

    area = normalized_curve_area(layers, values)

    assert np.isclose(area, 10.0 / 12.0)
    assert np.isclose(
        normalized_curve_area(layers[::-1], values[::-1]),
        area,
    )


def test_finite_null_p_value_uses_complete_null_statistics():
    nulls = np.array([0.1, 0.3, 0.5, 0.7])

    assert np.isclose(finite_null_p_value(0.6, nulls), 2.0 / 5.0)
    assert np.isclose(finite_null_p_value(1.0, nulls), 1.0 / 5.0)
