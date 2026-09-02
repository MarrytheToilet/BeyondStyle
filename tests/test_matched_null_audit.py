from argparse import Namespace

import numpy as np
import pandas as pd

from scripts.audit_pullback_matched_nulls import aggregate_matched_nulls


def test_matched_null_global_curve_uses_pipeline_as_null_unit(tmp_path):
    learned_rows = []
    null_rows = []
    for concept in ("lexical", "syntactic"):
        for prompt in ("p1", "p2"):
            for layer in (8, 12, 17, 22):
                learned_rows.append({
                    "prompt_id": prompt,
                    "intervention": concept,
                    "readout": concept,
                    "layer": layer,
                    "transport_coefficient": 3.0,
                })
                for null_axis, value in enumerate((0.5, 1.0, 2.0)):
                    null_rows.append({
                        "prompt_id": prompt,
                        "concept": concept,
                        "null_axis": null_axis,
                        "layer": layer,
                        "transport_coefficient": value,
                    })
    learned_path = tmp_path / "per_prompt_transport.csv"
    pd.DataFrame(learned_rows).to_csv(learned_path, index=False)
    args = Namespace(
        learned_transport=learned_path,
        bootstrap=200,
        permutations=200,
        seed=7,
    )

    layerwise, null_axes, global_summary = aggregate_matched_nulls(
        args,
        pd.DataFrame(null_rows),
        injection_layer=8,
        anchors={"lexical": 22, "syntactic": 22},
    )

    lexical = global_summary[
        global_summary["concept"].eq("lexical")
    ].iloc[0]
    assert lexical["propagation_layers"] == "12,17"
    assert lexical["n_null_pipelines"] == 3
    assert np.isclose(lexical["learned_curve_statistic"], 3.0)
    assert np.isclose(lexical["null_max_curve_statistic"], 2.0)
    assert np.isclose(lexical["finite_null_p_value"], 0.25)
    assert len(layerwise) == 8
    assert len(null_axes) == 24
