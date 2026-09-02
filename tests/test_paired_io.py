import json

import pytest

from src.config import extract_dataset_tag
from src.data.paired_io import (
    generator_model,
    load_paired_samples,
    plan_serialization_share,
)


def canonical_record(sample_id: str = "en_test_00001") -> dict:
    return {
        "sample_id": sample_id,
        "language": "en",
        "source": "wikipedia",
        "prefix": "The committee met on Tuesday.",
        "human_continuation": "Members approved the proposal.",
        "ai_continuation": "The proposal received the committee's approval.",
        "generation": {"generator_model": "qwen3.6-flash"},
        "lengths": {"ai_continuation_units": 10},
        "validation": {
            "deterministic_quality": {
                "longest_plan_shared_run": 3,
            }
        },
    }


def test_load_canonical_jsonl(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        json.dumps(canonical_record(), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    samples = load_paired_samples(path)

    assert len(samples) == 1
    assert generator_model(samples[0]) == "qwen3.6-flash"
    assert plan_serialization_share(samples[0]) == pytest.approx(0.3)


def test_duplicate_sample_ids_are_rejected(tmp_path):
    path = tmp_path / "pairs.jsonl"
    record = canonical_record()
    path.write_text(
        "\n".join(json.dumps(record) for _ in range(2)) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate sample_id"):
        load_paired_samples(path)


def test_noncanonical_json_is_rejected(tmp_path):
    path = tmp_path / "pairs.json"
    path.write_text(json.dumps([canonical_record()]), encoding="utf-8")

    with pytest.raises(ValueError, match="must be JSONL"):
        load_paired_samples(path)


def test_canonical_dataset_tag_uses_parent_directory():
    assert (
        extract_dataset_tag("data/paired/en/proposition_aligned/pairs.jsonl")
        == "proposition_aligned"
    )
