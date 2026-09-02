import numpy as np
import pandas as pd

from scripts.audit_layerwise_concept_transport import (
    aggregate,
    aggregate_direction_ablations,
)


def test_transport_aggregate_is_sign_invariant():
    rows = []
    for prompt, learned in (("p1", -2.0), ("p2", -4.0)):
        for concept in ("lexical", "syntactic"):
            rows.append({
                "prompt_id": prompt,
                "intervention": concept,
                "intervention_kind": "learned",
                "readout": concept,
                "layer": 8,
                "transport_coefficient": learned,
                "alignment": -0.5,
                "response_norm_coefficient": 4.0,
            })
            for index, random_value in enumerate((-1.0, 2.0)):
                rows.append({
                    "prompt_id": prompt,
                    "intervention": f"random_{index}",
                    "intervention_kind": "random",
                    "readout": concept,
                    "layer": 8,
                    "transport_coefficient": random_value,
                    "alignment": 0.0,
                    "response_norm_coefficient": 2.0,
                })

    learned, contrasts, random_direction = aggregate(
        pd.DataFrame(rows),
        bootstrap=200,
        permutations=200,
        seed=7,
    )

    lexical_learned = learned[learned["concept"].eq("lexical")].iloc[0]
    lexical_contrast = contrasts[
        contrasts["concept"].eq("lexical")
    ].iloc[0]
    assert np.isclose(lexical_learned["signed_mean"], -3.0)
    assert np.isclose(lexical_learned["mean"], 3.0)
    assert np.isclose(lexical_contrast["random_ensemble_mean"], 1.5)
    assert np.isclose(lexical_contrast["mean"], 1.5)
    assert lexical_contrast["random_direction_exceed_rate"] == 0.0
    assert sorted(
        random_direction["transport_coefficient"].unique()
    ) == [1.0, 2.0]


def test_direction_ablations_use_propagation_only_curves():
    rows = []
    values = {
        "lexical": 4.0,
        "syntactic": 4.0,
        "encoding_lexical": 2.0,
        "encoding_syntactic": 2.0,
        "mean_difference": 1.5,
        "source_nullspace": 1.0,
        "random_00": 0.5,
        "random_01": 1.5,
    }
    kinds = {
        "lexical": "pullback",
        "syntactic": "pullback",
        "encoding_lexical": "encoding",
        "encoding_syntactic": "encoding",
        "mean_difference": "raw_mean_difference",
        "source_nullspace": "nullspace_mean_difference",
        "random_00": "random",
        "random_01": "random",
    }
    for prompt in ("p1", "p2"):
        for layer in (8, 12, 17, 22):
            for readout in ("lexical", "syntactic"):
                for intervention, value in values.items():
                    rows.append({
                        "prompt_id": prompt,
                        "intervention": intervention,
                        "intervention_kind": kinds[intervention],
                        "readout": readout,
                        "layer": layer,
                        "transport_coefficient": value,
                    })

    layerwise, global_summary = aggregate_direction_ablations(
        pd.DataFrame(rows),
        injection_layer=8,
        anchors={"lexical": 22, "syntactic": 22},
        bootstrap=200,
        permutations=200,
        seed=7,
    )

    lexical_encoding = global_summary[
        global_summary["concept"].eq("lexical")
        & global_summary["reference_family"].eq("encoding")
    ].iloc[0]
    assert lexical_encoding["propagation_layers"] == "12,17"
    assert np.isclose(lexical_encoding["learned_curve_statistic"], 4.0)
    assert np.isclose(lexical_encoding["reference_curve_statistic"], 2.0)
    assert np.isclose(
        lexical_encoding["learned_minus_reference_mean"],
        2.0,
    )
    random = global_summary[
        global_summary["concept"].eq("lexical")
        & global_summary["reference_family"].eq(
            "projected_random_ensemble"
        )
    ].iloc[0]
    assert random["n_reference_directions"] == 2
    assert np.isclose(random["reference_curve_statistic"], 1.0)
    assert set(layerwise["layer"]) == {12, 17}
