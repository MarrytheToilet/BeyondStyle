#!/usr/bin/env python3
"""Measure a frozen natural-language prefix bank under multiple tokenizers."""

import argparse
import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BANK_NAME = "balanced_natural"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=("en", "zh"), required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Local model/tokenizer paths. They measure but never select rows.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    return parser.parse_args()


def distribution(values: list[int]) -> dict:
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "min": ordered[0],
        "median": statistics.median(ordered),
        "mean": statistics.fmean(ordered),
        "p95": ordered[p95_index],
        "max": ordered[-1],
    }


def token_lengths(tokenizer, texts: list[str], batch_size: int) -> list[int]:
    lengths = []
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(
            texts[start : start + batch_size],
            add_special_tokens=False,
            truncation=False,
            padding=False,
        )
        lengths.extend(len(ids) for ids in encoded["input_ids"])
    return lengths


def main():
    args = parse_args()
    bank_dir = (
        PROJECT_ROOT
        / "data"
        / "prefixes"
        / args.language
        / BANK_NAME
    )
    records = [
        json.loads(line)
        for line in (bank_dir / "samples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    prefixes = [record["prefix"] for record in records]
    continuations = [record["human_continuation"] for record in records]
    separator = " " if args.language == "en" else ""
    full_windows = [
        prefix + separator + continuation
        for prefix, continuation in zip(prefixes, continuations)
    ]

    model_reports = {}
    for model_path in args.models:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=Path(model_path).is_dir(),
        )
        model_id = Path(model_path).name
        prefix_lengths = token_lengths(
            tokenizer, prefixes, args.batch_size
        )
        continuation_lengths = token_lengths(
            tokenizer, continuations, args.batch_size
        )
        full_lengths = token_lengths(
            tokenizer, full_windows, args.batch_size
        )
        model_reports[model_id] = {
            "tokenizer_path": model_path,
            "tokenizer_class": type(tokenizer).__name__,
            "vocab_size": tokenizer.vocab_size,
            "prefix_tokens": distribution(prefix_lengths),
            "continuation_tokens": distribution(continuation_lengths),
            "full_window_tokens": distribution(full_lengths),
        }
        print(
            f"{args.language} {model_id}: "
            f"prefix median={model_reports[model_id]['prefix_tokens']['median']}, "
            f"continuation median="
            f"{model_reports[model_id]['continuation_tokens']['median']}",
            flush=True,
        )

    report = {
        "schema_version": 1,
        "language": args.language,
        "bank_id": BANK_NAME,
        "n_samples": len(records),
        "selection_role": "post_hoc_measurement_only",
        "changes_bank_membership_or_boundaries": False,
        "models": model_reports,
    }
    output = bank_dir / "tokenizer_audit.json"
    temporary = output.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
