import numpy as np
import pandas as pd

from src.analysis.decodability_control_gap import (
    behavior_observations,
    encoding_observations,
    transport_observations,
)


def test_encoding_uses_generator_as_the_observation_unit():
    rows = []
    for environment, values in {"g1": (0.2, 0.4), "g2": (0.6, 0.8)}.items():
        for source, value in zip(("human", "ai"), values):
            rows.append({
                "model": "m",
                "representation": "mean_pool",
                "layer": 7,
                "group": "lexical",
                "test_source": source,
                "holdout_type": "generator",
                "heldout_environment": environment,
                "heldout_score_correlation": value,
            })
            rows.append({
                "model": "m",
                "representation": "mean_pool",
                "layer": 7,
                "group": "syntactic",
                "test_source": source,
                "holdout_type": "generator",
                "heldout_environment": environment,
                "heldout_score_correlation": value / 2,
            })

    result = encoding_observations(pd.DataFrame(rows), {"m": 7})

    lexical = result[result["concept"].eq("lexical")]
    assert np.allclose(lexical["value"], [0.3, 0.7])
    assert lexical["observation_id"].tolist() == ["g1", "g2"]


def test_transport_subtracts_prompt_matched_random_ensemble():
    rows = []
    for prompt, learned, random_values in [
        ("p1", -1.0, (-0.1, 0.3)),
        ("p2", 2.0, (-0.2, 0.4)),
    ]:
        for concept in ("lexical", "syntactic"):
            rows.append({
                "prompt_id": prompt,
                "intervention": concept,
                "intervention_kind": "learned",
                "readout": concept,
                "layer": 32,
                "transport_coefficient": learned,
            })
            for index, value in enumerate(random_values):
                rows.append({
                    "prompt_id": prompt,
                    "intervention": f"random_{index}",
                    "intervention_kind": "random",
                    "readout": concept,
                    "layer": 32,
                    "transport_coefficient": value,
                })

    result = transport_observations(
        pd.DataFrame(rows),
        model="m",
        downstream_layers={"lexical": 32, "syntactic": 32},
    )

    lexical = result[result["concept"].eq("lexical")]
    assert lexical["value"].tolist() == [0.8, 1.7]


def test_behavior_subtracts_random_target_selectivity():
    rows = []
    values = {
        ("lexical", "lexical"): 2.0,
        ("lexical", "syntactic"): 0.0,
        ("lexical", "discourse"): 0.0,
        ("syntactic", "lexical"): 0.0,
        ("syntactic", "syntactic"): 3.0,
        ("syntactic", "discourse"): 0.0,
        ("random", "lexical"): 1.0,
        ("random", "syntactic"): 0.5,
        ("random", "discourse"): 0.0,
    }
    for (component, group), value in values.items():
        rows.append({
            "component": component,
            "group": group,
            "prompt_id": "p1",
            "prompt_index": 34,
            "signature_projection": value,
        })

    result = behavior_observations(pd.DataFrame(rows), model="m")

    lexical = result[result["concept"].eq("lexical")]["value"].item()
    syntactic = result[result["concept"].eq("syntactic")]["value"].item()
    assert lexical == 1.25
    assert syntactic == 3.0
