#!/usr/bin/env python3
"""Merge independently finalized generation workers into one audited result."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--method-key", default="component_lexical")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def comparable_config(config: dict[str, Any]) -> dict[str, Any]:
    comparable = dict(config)
    comparable.pop("shard_index", None)
    return comparable


def main() -> None:
    args = parse_args()
    result_name = f"generation_{args.method_key}.json"
    manifest_name = f"generation_{args.method_key}_manifest.json"
    rows: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []

    for input_dir in args.input_dirs:
        result_path = input_dir / result_name
        manifest_path = input_dir / manifest_name
        if not result_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(
                f"Missing finalized result or manifest in {input_dir}"
            )
        part_rows = json.loads(result_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("n_records") != len(part_rows):
            raise ValueError(f"Manifest record count mismatch in {input_dir}")
        rows.extend(part_rows)
        manifests.append(manifest)
        sources.append({
            "directory": str(input_dir.resolve()),
            "result_sha256": sha256_file(result_path),
            "manifest_sha256": sha256_file(manifest_path),
            "n_records": len(part_rows),
        })

    reference = comparable_config(manifests[0]["config"])
    if any(comparable_config(item["config"]) != reference for item in manifests[1:]):
        raise ValueError("Generation shard configurations differ")
    num_shards = int(reference.get("num_shards", 1))
    shard_indices = sorted(
        int(item["config"].get("shard_index", 0)) for item in manifests
    )
    if shard_indices != list(range(num_shards)):
        raise ValueError(
            f"Expected shard indices 0..{num_shards - 1}, got {shard_indices}"
        )

    keys = [
        (str(row["prompt_id"]), int(row.get("sample_seed_index", 0)))
        for row in rows
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate prompt×seed records across generation shards")
    expected = int(reference["n_prompts"]) * int(reference["num_seeds"])
    if len(rows) != expected:
        raise ValueError(f"Expected {expected} merged records, got {len(rows)}")
    rows.sort(
        key=lambda row: (
            int(row.get("sample_seed_index", 0)),
            int(row["prompt_index"]),
        )
    )

    result_path = args.out_dir / result_name
    atomic_json(result_path, rows)
    merged_config = dict(reference)
    merged_config.pop("num_shards", None)
    merged_config["execution_num_shards"] = num_shards
    elapsed_parts = [
        float(item.get("elapsed_generation_seconds", 0.0))
        for item in manifests
    ]
    generated_tokens = sum(
        int(item.get("generated_tokens", 0)) for item in manifests
    )
    manifest = {
        "schema_version": 1,
        "config": merged_config,
        "selected_layer": manifests[0].get("selected_layer"),
        "selection_rule": manifests[0].get("selection_rule"),
        "selection_table": manifests[0].get("selection_table"),
        "calibration": manifests[0].get("calibration"),
        "model_load": manifests[0].get("model_load"),
        "n_records": len(rows),
        "n_prompts": len({str(row["prompt_id"]) for row in rows}),
        "n_seeds": len({int(row.get("sample_seed_index", 0)) for row in rows}),
        "generated_tokens": generated_tokens,
        "elapsed_generation_seconds_sum": sum(elapsed_parts),
        "elapsed_generation_seconds_parallel_proxy": max(elapsed_parts),
        "result_path": str(result_path.resolve()),
        "result_sha256": sha256_file(result_path),
        "merged_shards": sources,
    }
    atomic_json(args.out_dir / manifest_name, manifest)
    print(
        f"Merged {len(args.input_dirs)} shards into {result_path} "
        f"({len(rows)} records)"
    )


if __name__ == "__main__":
    main()
