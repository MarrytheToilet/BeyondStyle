import numpy as np

from src.analysis.hidden_language import (
    fit_ridge_reconstruction,
    linguistic_mean_direction,
    project_out_basis,
    reconstruction_metrics,
    residualize_preserving_mean,
    scale_features_preserving_mean,
    select_ridge_alpha,
)


def test_project_out_basis_removes_component():
    basis = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    values = np.array([[2.0, 3.0, 4.0]], dtype=np.float32)
    projected = project_out_basis(values, basis)
    assert np.allclose(projected, [[0.0, 3.0, 4.0]])


def test_residualization_preserves_train_mean():
    nuisance = np.arange(10, dtype=np.float32)[:, None]
    values = 4.0 + 2.0 * (nuisance - nuisance.mean())
    train, _ = residualize_preserving_mean(
        values,
        values,
        nuisance,
        nuisance,
    )
    assert np.isclose(train.mean(), values.mean())
    assert train.std() < 1e-4


def test_feature_scaling_does_not_remove_mean_shift():
    train = np.array([[1.0, 2.0], [3.0, 6.0], [5.0, 10.0]])
    scaled_train, scaled_test, scales = scale_features_preserving_mean(
        train,
        train,
    )
    assert np.allclose(scaled_train.mean(axis=0), train.mean(axis=0) / scales)
    assert np.allclose(scaled_train, scaled_test)


def test_ridge_reconstruction_recovers_linguistic_map():
    features = np.array(
        [[-1.0], [0.0], [1.0], [2.0]],
        dtype=np.float32,
    )
    coefficient = np.array([[2.0, -1.0]], dtype=np.float32)
    hidden = features @ coefficient
    fitted = fit_ridge_reconstruction(hidden, features, alpha=1e-8)
    linguistic = linguistic_mean_direction(features, fitted)
    assert np.allclose(fitted, coefficient, atol=1e-5)
    assert np.allclose(
        linguistic,
        features.mean(axis=0) @ coefficient,
        atol=1e-5,
    )


def test_reconstruction_does_not_attribute_mean_from_zero_mean_features():
    features = np.array([[-2.0], [-1.0], [1.0], [2.0]], dtype=np.float32)
    hidden = np.column_stack(
        [
            np.full(4, 100.0),
            2.0 * features[:, 0],
        ]
    )
    fitted = fit_ridge_reconstruction(hidden, features, alpha=1e-8)
    linguistic = linguistic_mean_direction(features, fitted)
    assert abs(float(linguistic[0])) < 1e-5
    assert np.isclose(linguistic[1], 0.0, atol=1e-5)


def test_reconstruction_metrics_have_interpretable_axis_fraction():
    metrics = reconstruction_metrics(
        np.array([2.0, 0.0]),
        np.array([1.0, 0.0]),
    )
    assert np.isclose(metrics["source_axis_reconstructed_fraction"], 0.5)
    assert np.isclose(metrics["aligned_fraction"], 0.5)
    assert np.isclose(metrics["direction_cosine"], 1.0)
    assert np.isclose(metrics["vector_reconstruction_r2"], 0.75)


def test_alpha_selection_uses_heldout_reconstruction_and_tie_regularization():
    selected, means = select_ridge_alpha(
        {
            0.1: [0.2, 0.4],
            1.0: [0.4, 0.4],
            10.0: [0.4, 0.4],
        }
    )
    assert selected == 10.0
    assert means[0.1] < means[1.0]
