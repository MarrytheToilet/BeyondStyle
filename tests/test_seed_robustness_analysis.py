import json

import numpy as np
import pandas as pd

from scripts.analyze_seed_robustness import (
    load_generation_records,
    prompt_cluster_summary,
)
from src.linguistics import EnglishLinguisticFeatureExtractor


def test_feature_extractor_uses_current_no_argument_interface():
    extractor = EnglishLinguisticFeatureExtractor()
    assert "function_word_rate" in extractor.feature_names


def test_multi_seed_loader_preserves_prompt_seed_observations(tmp_path):
    rows = []
    for seed_index in range(3):
        rows.append(
            {
                "prompt_id": "p1",
                "prompt_index": 1,
                "sample_seed_index": seed_index,
                "outputs": {
                    "-1.0": {"text": f"negative {seed_index}"},
                    "1.0": {"text": f"positive {seed_index}"},
                },
            }
        )
    for component in ("lexical", "syntactic"):
        (tmp_path / f"generation_component_{component}.json").write_text(
            json.dumps(rows),
            encoding="utf-8",
        )

    records = load_generation_records(
        tmp_path,
        ["lexical", "syntactic"],
    )

    assert len(records) == 12
    assert {row["sample_seed_index"] for row in records} == {0, 1, 2}


def test_prompt_cluster_summary_averages_seeds_before_inference():
    frame = pd.DataFrame(
        {
            "prompt_id": ["p1", "p1", "p2", "p2"],
            "sample_seed_index": [0, 1, 0, 1],
            "value": [1.0, 3.0, 5.0, 7.0],
        }
    )

    result = prompt_cluster_summary(
        frame,
        value_column="value",
        bootstrap=100,
        permutations=100,
        seed=7,
    )

    assert result["n_prompts"] == 2
    assert result["n_seeds"] == 2
    assert result["n_prompt_seed_observations"] == 4
    assert np.isclose(result["mean"], 4.0)
    assert np.isclose(result["seed_mean_min"], 3.0)
    assert np.isclose(result["seed_mean_max"], 5.0)
