import numpy as np

from src.analysis.concept_patterns import (
    covariance_activation_pattern,
    cross_source_consensus,
    decompose_source_direction,
    principal_probe_pattern,
    signature_weights,
)
from scripts.build_concept_patterns import (
    _fit_views,
    pooled_projection_scale,
)


def test_signature_weights_follow_standardized_ai_minus_human_shift():
    human = np.array(
        [[0.0, 2.0], [1.0, 3.0], [2.0, 4.0], [3.0, 5.0]],
    )
    ai = human + np.array([2.0, -1.0])

    weights, diagnostics = signature_weights(human, ai)

    assert weights[0] > 0
    assert weights[1] < 0
    assert np.isclose(np.linalg.norm(weights), 1.0)
    assert diagnostics["standardized_shift_norm"] > 0


def test_covariance_pattern_recovers_forward_hidden_axis():
    rng = np.random.default_rng(7)
    scores = rng.normal(size=1000)
    axis = np.array([0.2, -0.4, 0.8, 0.1])
    axis /= np.linalg.norm(axis)
    hidden = scores[:, None] * axis + rng.normal(
        scale=0.05,
        size=(len(scores), len(axis)),
    )

    pattern, diagnostics = covariance_activation_pattern(hidden, scores)

    assert pattern @ axis > 0.99
    assert diagnostics["fitted_score_correlation"] > 0.99


def test_principal_probe_pattern_recovers_predictive_axis():
    rng = np.random.default_rng(9)
    scores = rng.normal(size=400)
    hidden = np.column_stack(
        [
            scores + rng.normal(scale=0.05, size=len(scores)),
            rng.normal(scale=8.0, size=len(scores)),
            rng.normal(scale=4.0, size=len(scores)),
        ]
    )

    pattern, diagnostics = principal_probe_pattern(
        hidden,
        scores,
        components=3,
    )

    assert abs(pattern[0]) > 0.99
    assert diagnostics["fitted_score_correlation"] > 0.99


def test_cross_source_consensus_preserves_semantic_orientation():
    human = np.array([1.0, 0.0, 0.0])
    ai = np.array([0.9, 0.1, 0.0])

    consensus, diagnostics = cross_source_consensus(human, ai)

    assert consensus[0] > 0
    assert diagnostics["human_ai_pattern_cosine"] > 0.9


def test_source_decomposition_is_orthogonal_and_deterministic():
    source = np.array([1.0, 1.0, 1.0, 0.0])
    patterns = {
        "lexical": np.array([1.0, 0.0, 0.0, 0.0]),
        "syntactic": np.array([0.0, 1.0, 0.0, 0.0]),
    }

    first, diagnostics = decompose_source_direction(
        source,
        patterns,
        seed=11,
    )
    second, _ = decompose_source_direction(source, patterns, seed=11)

    assert abs(first["residual"] @ first["lexical"]) < 1e-10
    assert abs(first["residual"] @ first["syntactic"]) < 1e-10
    assert abs(first["random"] @ first["source"]) < 1e-10
    assert abs(first["random"] @ first["lexical"]) < 1e-10
    assert abs(first["random"] @ first["syntactic"]) < 1e-10
    assert np.allclose(first["random"], second["random"])
    assert diagnostics["source_concept_projection_energy"] > 0


def test_pooled_projection_scale_tracks_directional_variance():
    human = np.array([[0.0, 0.0], [2.0, 0.0]])
    ai = np.array([[4.0, 0.0], [6.0, 0.0]])

    active = pooled_projection_scale(human, ai, np.array([1.0, 0.0]))
    inactive = pooled_projection_scale(human, ai, np.array([0.0, 1.0]))

    assert active > inactive
    assert inactive == 1e-6


def test_fit_views_can_preserve_hidden_nuisance_under_target_permutation():
    rng = np.random.default_rng(37)
    n_rows = 120
    hidden_nuisance = rng.normal(size=(n_rows, 1))
    target_nuisance = rng.normal(size=(n_rows, 1))
    hidden = np.column_stack(
        [
            3.0 * hidden_nuisance[:, 0]
            + rng.normal(scale=0.2, size=n_rows),
            rng.normal(size=n_rows),
        ]
    )
    features = (
        4.0 * target_nuisance
        + rng.normal(scale=0.2, size=(n_rows, 1))
    )
    train = np.arange(100)
    test = np.arange(100, n_rows)

    views = _fit_views(
        {"human": hidden, "ai": hidden.copy()},
        {"human": features, "ai": features.copy()},
        {
            "human": hidden_nuisance,
            "ai": hidden_nuisance.copy(),
        },
        feature_nuisance={
            "human": target_nuisance,
            "ai": target_nuisance.copy(),
        },
        train=train,
        test=test,
        basis=np.empty((0, hidden.shape[1])),
    )

    human_view = views["human"]
    assert abs(
        np.corrcoef(
            human_view["train_hidden"][:, 0],
            hidden_nuisance[train, 0],
        )[0, 1]
    ) < 1e-6
    assert abs(
        np.corrcoef(
            human_view["train_features"][:, 0],
            target_nuisance[train, 0],
        )[0, 1]
    ) < 1e-6
