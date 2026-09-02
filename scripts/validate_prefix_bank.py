#!/usr/bin/env python3
"""Validate a natural-language prefix bank against its pinned raw subsets."""

import argparse
import hashlib
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_prefix_bank import (
    adapt_row,
    load_registry,
    normalized_window_hash,
    source_subset_file,
)
from src.data.natural_window import NaturalLanguageWindowSplitter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BANK_NAME = "balanced_natural"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=("en", "zh"), required=True)
    return parser.parse_args()


def summary(values: list[int]) -> dict:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_raw_text(
    source_name: str,
    source_records: list[dict],
    spec: dict,
    registry: dict,
    splitter: NaturalLanguageWindowSplitter,
) -> int:
    records_by_key = defaultdict(list)
    for record in source_records:
        document = record["source_document"]
        key = (
            document["raw_source_file"],
            document["raw_source_row_index"],
        )
        records_by_key[key].append(record)

    found_keys = set()
    with source_subset_file(registry, spec).open(encoding="utf-8") as handle:
        for raw_line_index, line in enumerate(handle):
            raw_record = json.loads(line)
            provenance = raw_record["provenance"]
            key = (
                provenance["source_file"],
                provenance["source_row_index"],
            )
            if key not in records_by_key:
                continue
            found_keys.add(key)
            adapted = adapt_row(
                source_name,
                raw_record["native_row"],
                raw_line_index,
                spec,
            )
            text = adapted["text"].strip()
            spans = splitter.sentence_spans(text)
            for record in records_by_key[key]:
                document = record["source_document"]
                start = document["window_start_sentence"]
                end = document["window_end_sentence"]
                prefix_sentences = record["prefix_sentences"]
                continuation_sentences = record["continuation_sentences"]
                if end != start + prefix_sentences + continuation_sentences:
                    raise RuntimeError(
                        f"{record['sample_id']}: inconsistent sentence indices"
                    )
                if end > len(spans):
                    raise RuntimeError(
                        f"{record['sample_id']}: sentence window exceeds document"
                    )
                prefix_end = start + prefix_sentences
                expected_prefix = text[
                    spans[start].start : spans[prefix_end - 1].end
                ].strip()
                expected_continuation = text[
                    spans[prefix_end].start : spans[end - 1].end
                ].strip()
                if record["prefix"] != expected_prefix:
                    raise RuntimeError(
                        f"{record['sample_id']}: prefix is not the recorded "
                        "verbatim source span"
                    )
                if record["human_continuation"] != expected_continuation:
                    raise RuntimeError(
                        f"{record['sample_id']}: continuation is not the "
                        "recorded verbatim source span"
                    )

    missing = set(records_by_key) - found_keys
    if missing:
        raise RuntimeError(
            f"{source_name}: {len(missing)} source rows are absent from raw"
        )
    return len(found_keys)


def main():
    args = parse_args()
    bank_dir = (
        PROJECT_ROOT
        / "data"
        / "prefixes"
        / args.language
        / BANK_NAME
    )
    samples_path = bank_dir / "samples.jsonl"
    manifest_path = bank_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in samples_path.read_text(encoding="utf-8").splitlines()
        if line
    ]

    if manifest["language"] != args.language:
        raise RuntimeError("Manifest language does not match requested language")
    if manifest.get("tokenizer_used_for_selection") is not False:
        raise RuntimeError("Bank is not explicitly tokenizer-independent")
    if manifest["n_samples"] != len(records):
        raise RuntimeError("Manifest row count does not match samples.jsonl")
    if manifest["output_sha256"] != sha256(samples_path):
        raise RuntimeError("samples.jsonl SHA-256 does not match manifest")

    splitter = NaturalLanguageWindowSplitter(
        language=args.language,
        target_prefix_units=manifest["prefix_units"],
        target_continuation_units=manifest["continuation_units"],
        tolerance=manifest["tolerance"],
    )
    ids = set()
    hashes = set()
    prefixes = set()
    intervals = defaultdict(list)
    by_source = defaultdict(list)
    for record in records:
        sample_id = record["sample_id"]
        if record.get("language") != args.language:
            raise RuntimeError(f"{sample_id}: language boundary violation")
        if sample_id in ids:
            raise RuntimeError(f"Duplicate sample_id: {sample_id}")
        ids.add(sample_id)
        expected_hash = normalized_window_hash(
            record["prefix"],
            record["human_continuation"],
        )
        if record["window_sha256"] != expected_hash:
            raise RuntimeError(f"{sample_id}: window hash mismatch")
        if expected_hash in hashes:
            raise RuntimeError(f"{sample_id}: duplicate normalized window")
        hashes.add(expected_hash)
        prefixes.add(record["prefix"])

        if splitter.count_units(record["prefix"]) != record["prefix_units"]:
            raise RuntimeError(f"{sample_id}: prefix unit count mismatch")
        if (
            splitter.count_units(record["human_continuation"])
            != record["continuation_units"]
        ):
            raise RuntimeError(f"{sample_id}: continuation unit count mismatch")
        if not splitter.min_prefix <= record["prefix_units"] <= splitter.max_prefix:
            raise RuntimeError(f"{sample_id}: prefix outside length bounds")
        if not (
            splitter.min_continuation
            <= record["continuation_units"]
            <= splitter.max_continuation
        ):
            raise RuntimeError(f"{sample_id}: continuation outside length bounds")
        quality_issues = sorted(set(
            splitter.prose_quality_issues(record["prefix"])
            + splitter.prose_quality_issues(
                record["human_continuation"]
            )
        ))
        if quality_issues:
            raise RuntimeError(
                f"{sample_id}: failed prose gate: {quality_issues}"
            )

        document = record["source_document"]
        document_key = (
            record["source"],
            document["raw_source_file"],
            document["raw_source_row_index"],
        )
        interval = (
            document["window_start_sentence"],
            document["window_end_sentence"],
        )
        for prior_start, prior_end in intervals[document_key]:
            if interval[0] < prior_end and interval[1] > prior_start:
                raise RuntimeError(
                    f"{sample_id}: overlaps another window from the same document"
                )
        intervals[document_key].append(interval)
        by_source[record["source"]].append(record)

    registry = load_registry(
        PROJECT_ROOT / "data" / "corpora" / f"{args.language}.yaml"
    )
    raw_documents = {}
    for source_name, source_records in by_source.items():
        expected = manifest["sources"][source_name]["accepted"]
        if len(source_records) != expected:
            raise RuntimeError(
                f"{source_name}: expected {expected} samples, "
                f"found {len(source_records)}"
            )
        raw_documents[source_name] = validate_raw_text(
            source_name,
            source_records,
            registry["sources"][source_name],
            registry,
            splitter,
        )

    report = {
        "schema_version": 1,
        "status": "pass",
        "language": args.language,
        "bank_id": manifest["bank_id"],
        "n_samples": len(records),
        "n_unique_sample_ids": len(ids),
        "n_unique_windows": len(hashes),
        "n_unique_prefixes": len(prefixes),
        "source_counts": dict(sorted(Counter(
            record["source"] for record in records
        ).items())),
        "raw_documents_used": dict(sorted(raw_documents.items())),
        "prefix_units": summary(
            [record["prefix_units"] for record in records]
        ),
        "continuation_units": summary(
            [record["continuation_units"] for record in records]
        ),
        "prefix_characters": summary(
            [record["prefix_characters"] for record in records]
        ),
        "continuation_characters": summary(
            [record["continuation_characters"] for record in records]
        ),
        "prefix_sentences": summary(
            [record["prefix_sentences"] for record in records]
        ),
        "continuation_sentences": summary(
            [record["continuation_sentences"] for record in records]
        ),
        "checks": {
            "tokenizer_independent_selection": True,
            "manifest_hash": True,
            "unique_ids": True,
            "unique_normalized_windows": True,
            "natural_unit_bounds": True,
            "model_independent_prose_gate": True,
            "verbatim_raw_spans": True,
            "non_overlapping_windows_per_document": True,
            "language_boundary": all(
                record["language"] == args.language for record in records
            ),
        },
    }
    report_path = bank_dir / "quality_report.json"
    temporary = report_path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
