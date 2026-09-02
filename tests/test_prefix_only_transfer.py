import numpy as np
import pandas as pd

from scripts.build_concept_patterns import inventory
from scripts.evaluate_prefix_only_transfer import (
    crossfit_surface_signature_transfer,
    source_auc,
    standardized_gap,
    surface_signature_transfer,
    validate_matched_corpora,
)


def matched_pair(sample_id, ai_text, prompt_family):
    return {
        "sample_id": sample_id,
        "prefix": f"Prefix {sample_id}.",
        "human_continuation": f"Human {sample_id}.",
        "ai_continuation": ai_text,
        "source": "news",
        "domain": "journalistic",
        "generation": {
            "generator_model": "model-a",
            "prompt_family": prompt_family,
        },
    }


def test_matched_control_validation_preserves_order():
    plan = [
        matched_pair("a", "planned a", "ordered"),
        matched_pair("b", "planned b", "ordered"),
    ]
    control = [
        matched_pair("b", "control b", "prefix_only"),
        matched_pair("a", "control a", "prefix_only"),
    ]

    assert validate_matched_corpora(plan, control) == ["a", "b"]


def test_source_metrics_recover_separation():
    human = np.array([[0.0, 0.0], [0.2, 0.0], [-0.1, 0.0]])
    ai = np.array([[1.0, 0.0], [1.2, 0.0], [0.9, 0.0]])
    direction = np.array([1.0, 0.0])

    assert source_auc(human, ai, direction) == 1.0
    assert standardized_gap(human, ai, direction) > 5.0


def test_surface_signature_transfers_on_same_group_shift():
    feature_names, _, columns = inventory(["lexical", "syntactic"])
    n = 8
    plan = {
        "sample_id": [f"s{index}" for index in range(n)],
    }
    control = {
        "sample_id": [f"s{index}" for index in range(n)],
    }
    for feature_index, name in enumerate(feature_names):
        human = np.linspace(0.0, 0.7, n) + feature_index * 0.01
        if feature_index in set(columns["lexical"]):
            shift = 1.0
        elif feature_index in set(columns["syntactic"]):
            shift = 0.5
        else:
            shift = 0.0
        for frame, multiplier in ((plan, 1.0), (control, 0.8)):
            frame[f"human__{name}"] = human
            frame[f"ai__{name}"] = human + shift * multiplier

    result = surface_signature_transfer(
        pd.DataFrame(plan),
        pd.DataFrame(control),
        bootstrap=200,
        permutations=200,
        seed=7,
    )

    assert set(result["concept"]) == {"lexical", "syntactic"}
    assert np.allclose(result["plan_control_shift_cosine"], 1.0)
    assert (result["control_projection_mean"] > 0).all()
    assert (result["control_projection_positive_rate"] == 1.0).all()


def test_crossfit_surface_transfer_never_uses_heldout_samples():
    feature_names, _, columns = inventory(["lexical", "syntactic"])
    n = 12
    generators = np.asarray(["a", "b", "c", "d"] * 3)
    plan = {"sample_id": [f"s{index}" for index in range(n)]}
    control = {"sample_id": [f"s{index}" for index in range(n)]}
    for feature_index, name in enumerate(feature_names):
        human = np.linspace(0.0, 0.5, n) + feature_index * 0.01
        active = (
            feature_index in set(columns["lexical"])
            or feature_index in set(columns["syntactic"])
        )
        shift = 1.0 if active else 0.0
        plan[f"human__{name}"] = human
        plan[f"ai__{name}"] = human + shift
        control[f"human__{name}"] = human
        control[f"ai__{name}"] = human + 0.7 * shift

    summary, folds = crossfit_surface_signature_transfer(
        pd.DataFrame(plan),
        pd.DataFrame(control),
        generators,
        bootstrap=100,
        permutations=100,
        seed=11,
    )

    assert len(folds) == 8
    assert set(folds["heldout_generator"]) == {"a", "b", "c", "d"}
    assert (folds["n_train"] == 9).all()
    assert (folds["n_test"] == 3).all()
    assert np.allclose(summary["plan_control_shift_cosine"], 1.0)
    assert (summary["control_projection_mean"] > 0).all()
    assert (summary["control_projection_positive_rate"] == 1.0).all()
