import numpy as np

from scripts.analyze_projection_mapping import discover_layers
from src.analysis.projection_mapping import (
    fit_nullspace_source_direction,
    fit_nuisance_residualizer,
    fit_ridge_projection_map,
    fit_within_class_basis,
    permute_within_environments,
    prepare_mapping_split,
    project_out_basis,
    projection_mapping_metrics,
)


def test_layer_discovery_matches_state_and_direction_files(tmp_path):
    base = tmp_path / "Model" / "tag" / "last_token"
    states = base / "states"
    models = base / "models"
    states.mkdir(parents=True)
    models.mkdir(parents=True)
    for layer in (3, 11):
        (states / f"paired_states_layer{layer:02d}.pt").touch()
        (models / f"mean_diff_nullspace_layer{layer:02d}.pt").touch()
    (states / "paired_states_layer07.pt").touch()
    assert discover_layers(
        tmp_path,
        "Model",
        "tag",
        "last_token",
    ) == [3, 11]


def test_nullspace_direction_removes_top_within_class_axis():
    human = np.array(
        [[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
    )
    ai = human + np.array([0.0, 2.0])
    direction, info = fit_nullspace_source_direction(human, ai, k=1)
    assert abs(direction[0]) < 1e-8
    assert direction[1] > 0.99
    assert info["within_variance_removed"] > 0.99


def test_fitted_within_class_basis_can_project_states():
    human = np.array(
        [[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
    )
    ai = human + np.array([0.0, 1.0])
    basis, _ = fit_within_class_basis(human, ai, k=1)
    projected = project_out_basis(human, basis)
    assert np.allclose(projected[:, 0], 0.0, atol=1e-8)
    assert np.allclose(projected[:, 1], 0.0, atol=1e-8)


def test_nuisance_residualizer_is_train_fitted_and_removes_slope():
    nuisance = np.arange(8, dtype=float)[:, None]
    values = 3.0 + 2.5 * nuisance[:, 0]
    fitted = fit_nuisance_residualizer(values, nuisance)
    residual = fitted.transform(values, nuisance)
    assert abs(residual.mean()) < 1e-8
    assert residual.std() < 2e-6


def test_projection_map_recovers_surface_coordinate_relation():
    nuisance = np.column_stack([np.arange(12), np.ones(12)])
    feature = np.linspace(-2.0, 2.0, 12)[:, None]
    score = 1.5 * feature[:, 0] + 0.2 * nuisance[:, 0]
    prepared = prepare_mapping_split(
        feature[:8],
        feature[8:],
        score[:8],
        score[8:],
        nuisance[:8],
        nuisance[8:],
    )
    coefficient = fit_ridge_projection_map(
        prepared.train_features,
        prepared.train_scores,
        alpha=1e-8,
    )
    prediction = prepared.test_features @ coefficient
    metrics = projection_mapping_metrics(prepared.test_scores, prediction)
    assert metrics["pearson"] > 0.99


def test_projection_metrics_distinguish_aligned_and_reversed_predictions():
    actual = np.arange(10, dtype=float)
    aligned = projection_mapping_metrics(actual, actual)
    reversed_ = projection_mapping_metrics(actual, -actual)
    assert np.isclose(aligned["pearson"], 1.0)
    assert np.isclose(reversed_["pearson"], -1.0)
    assert aligned["sign_agreement"] == 1.0
    assert reversed_["sign_agreement"] == 0.0


def test_permutation_stays_within_environment():
    values = np.arange(12)
    environments = np.repeat(["a", "b", "c"], 4)
    permuted = permute_within_environments(
        values,
        environments,
        np.random.default_rng(7),
    )
    for environment in np.unique(environments):
        mask = environments == environment
        assert set(permuted[mask]) == set(values[mask])
