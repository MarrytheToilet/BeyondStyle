import json

from scripts.eval_lm_benchmark import (
    _choice_target_index,
    extract_choice_margin_diagnostics,
    run_exists,
)


def test_choice_margin_diagnostics_extracts_continuous_margin():
    results = {
        "samples": {
            "toy_task": [
                {
                    "doc_id": 7,
                    "doc_hash": "abc",
                    "target": 1,
                    "filtered_resps": [
                        [(-3.0, False)],
                        [(-1.5, True)],
                        [(-2.0, False)],
                    ],
                    "acc": 1.0,
                }
            ]
        }
    }

    diagnostics = extract_choice_margin_diagnostics(results)

    row = diagnostics["rows"]["toy_task"][0]
    assert row["target_vs_best_distractor_margin"] == 0.5
    assert diagnostics["summary"]["toy_task"]["n"] == 1
    assert diagnostics["summary"]["toy_task"]["positive_margin_rate"] == 1.0


def test_run_exists_rejects_incompatible_evaluation_design(tmp_path):
    record = {
        "tag": "baseline",
        "tasks": ["piqa", "copa"],
        "limit": 100,
        "num_fewshot": None,
    }
    (tmp_path / "baseline.json").write_text(
        json.dumps(record),
        encoding="utf-8",
    )

    assert run_exists(
        tmp_path,
        "baseline",
        tasks=["piqa", "copa"],
        limit=100,
        num_fewshot=None,
    )
    assert not run_exists(
        tmp_path,
        "baseline",
        tasks=["piqa", "copa"],
        limit=500,
        num_fewshot=None,
    )
    assert not run_exists(
        tmp_path,
        "baseline",
        tasks=["piqa"],
        limit=100,
        num_fewshot=None,
    )


def test_choice_target_index_handles_label_and_one_based_answer():
    assert _choice_target_index({"doc": {"label": 1}}, 2) == 1
    assert _choice_target_index({"doc": {"answer": "2"}}, 2) == 1


def test_choice_target_index_matches_unique_continuation():
    sample = {
        "target": " water flowed from the spout.",
        "arguments": [
            ["context", " the toilet filled with water."],
            ["context", "  water flowed from the spout."],
        ],
    }
    assert _choice_target_index(sample, 2) == 1
