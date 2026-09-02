from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.analyze_token_projections import (
    balanced_subset,
    parse_report_layers,
    write_projection_shard,
)
from src.extraction.hidden_states import TokenExtractionResult


class TinyTokenizer:
    def decode(self, token_ids, **_kwargs):
        return f" t{token_ids[0]}"


class TinyExtractor:
    model_name = "toy"
    tokenizer = TinyTokenizer()

    def extract_continuation_tokens_batch(
        self,
        prefixes,
        continuations,
        layers,
        separators,
        to_cpu,
    ):
        assert not to_cpu
        results = []
        for continuation in continuations:
            count = len(continuation)
            states = {
                layer: torch.arange(
                    count * 2,
                    dtype=torch.float32,
                ).reshape(count, 2)
                for layer in layers
            }
            results.append(
                TokenExtractionResult(
                    hidden_states=states,
                    token_ids=torch.arange(1, count + 1),
                    token_count=count + 2,
                    prefix_token_count=2,
                    continuation_token_count=count,
                    separator=" ",
                )
            )
        return results


def records():
    return [
        {
            "sample_id": f"s{index}",
            "pair_id": f"p{index}",
            "prefix": "P",
            "human_continuation": "HH",
            "ai_continuation": "AA",
            "language": "en",
            "domain": f"d{index % 2}",
            "generation": {"generator_model": f"g{index % 2}"},
        }
        for index in range(6)
    ]


def test_balanced_subset_and_report_layer_parsing_are_deterministic():
    first = balanced_subset(records(), 4)
    second = balanced_subset(records(), 4)
    assert [row["sample_id"] for row in first] == [
        row["sample_id"] for row in second
    ]
    assert len({row["generation"]["generator_model"] for row in first}) == 2
    assert parse_report_layers(
        ["last_token:3", "mean_pool:7"],
        model_id="toy",
        layers=[3, 7],
        axis_families=["last_token", "mean_pool"],
    ) == {"last_token": 3, "mean_pool": 7}


def test_projection_shard_contains_scalar_rows_only(tmp_path: Path):
    direction = np.asarray([0.0, 1.0], dtype=np.float32)
    directions = {
        axis: {3: {"g0": direction}}
        for axis in ("last_token", "mean_pool")
    }
    output = tmp_path / "part.parquet"
    count = write_projection_shard(
        extractor=TinyExtractor(),
        model_id="toy",
        records=[records()[0]],
        layers=[3],
        axis_families=["last_token", "mean_pool"],
        directions=directions,
        direction_scheme="generator_holdout",
        output=output,
    )
    frame = pd.read_parquet(output)

    assert count == 8  # two sources × two tokens × two axis families
    assert len(frame) == count
    assert "hidden_states" not in frame
    assert set(frame["projection"]) == {1.0, 3.0}
    assert set(frame["source"]) == {"human", "ai"}
