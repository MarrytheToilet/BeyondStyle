"""Canonical I/O helpers for paired human/AI continuation datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def _records_from_path(path: Path) -> list[dict[str, Any]]:
    if path.suffix != ".jsonl":
        raise ValueError(f"{path}: canonical paired data must be JSONL")
    with path.open(encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not isinstance(records, list) or not all(
        isinstance(record, dict) for record in records
    ):
        raise ValueError(f"{path}: expected a JSONL record stream or samples list")
    return records


def ai_continuation(record: dict[str, Any]) -> str:
    """Return the canonical AI continuation."""
    value = record.get("ai_continuation")
    return value if isinstance(value, str) and value.strip() else ""


def generator_model(record: dict[str, Any]) -> str:
    """Return generator identity from the canonical schema."""
    generation = record.get("generation")
    if isinstance(generation, dict):
        value = generation.get("generator_model")
        if isinstance(value, str) and value.strip():
            return value
    return "unknown"


def plan_serialization_share(record: dict[str, Any]) -> float:
    """Fraction of AI natural-language units in the longest plan-shared run."""
    validation = record.get("validation")
    lengths = record.get("lengths")
    if not isinstance(validation, dict) or not isinstance(lengths, dict):
        return 0.0
    quality = validation.get("deterministic_quality")
    if not isinstance(quality, dict):
        return 0.0
    shared = quality.get("longest_plan_shared_run")
    units = lengths.get("ai_continuation_units")
    if not isinstance(shared, (int, float)) or not isinstance(
        units, (int, float)
    ):
        return 0.0
    return float(shared / units) if units > 0 else 0.0


def load_paired_samples(
    path: str | Path,
    *,
    require_ai: bool = True,
) -> list[dict[str, Any]]:
    """Load and validate canonical proposition-aligned JSONL samples."""
    dataset_path = Path(path)
    records = _records_from_path(dataset_path)
    normalized: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for position, source in enumerate(records, start=1):
        record = dict(source)
        sample_id = record.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id.strip():
            raise ValueError(
                f"{dataset_path}:{position}: missing non-empty sample_id"
            )
        if sample_id in seen_ids:
            raise ValueError(
                f"{dataset_path}:{position}: duplicate sample_id {sample_id!r}"
            )
        seen_ids.add(sample_id)
        for field in ("prefix", "human_continuation"):
            value = record.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"{dataset_path}:{position}: missing non-empty {field}"
                )
        ai_text = ai_continuation(record)
        if require_ai and not ai_text:
            raise ValueError(
                f"{dataset_path}:{position}: missing AI continuation"
            )
        if ai_text:
            record["ai_continuation"] = ai_text
        normalized.append(record)
    return normalized
