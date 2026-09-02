#!/usr/bin/env python3
"""Select verbatim, document-disjoint prompts from the frozen prefix bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
from typing import Any


ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["en", "zh"], required=True)
    parser.add_argument("--num-prompts", type=int, default=200)
    parser.add_argument("--seed", type=int, default=8236)
    parser.add_argument("--bank", type=Path)
    parser.add_argument("--paired", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{path}:{line_number}: invalid JSON"
                    ) from error
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def document_id(row: dict) -> str:
    source = row["source_document"]
    fields = (
        source.get("dataset_id"),
        source.get("config"),
        source.get("split"),
        source.get("row_id"),
    )
    return "\x1f".join("" if value is None else str(value) for value in fields)


def allocate_balanced(
    total: int,
    groups: list[str],
    capacities: dict[str, int],
) -> dict[str, int]:
    """Allocate near-equal quotas while respecting finite source capacity."""
    base, remainder = divmod(total, len(groups))
    quotas = {
        group: base + (index < remainder)
        for index, group in enumerate(groups)
    }
    shortfall = 0
    for group in groups:
        if quotas[group] > capacities[group]:
            shortfall += quotas[group] - capacities[group]
            quotas[group] = capacities[group]
    while shortfall:
        progressed = False
        for group in groups:
            if quotas[group] < capacities[group]:
                quotas[group] += 1
                shortfall -= 1
                progressed = True
                if not shortfall:
                    break
        if not progressed:
            raise RuntimeError(
                f"only {sum(capacities.values())} eligible documents are "
                f"available for {total} requested prompts"
            )
    return quotas


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def main() -> None:
    args = parse_args()
    if args.num_prompts < 1:
        raise ValueError("--num-prompts must be positive")
    bank_path = args.bank or (
        ROOT
        / "data"
        / "prefixes"
        / args.language
        / "balanced_natural"
        / "samples.jsonl"
    )
    paired_path = args.paired or (
        ROOT
        / "data"
        / "paired"
        / args.language
        / "proposition_aligned"
        / "pairs.jsonl"
    )
    output_path = args.output or (
        ROOT / "data" / "prompts" / args.language / "formal.json"
    )

    bank = read_jsonl(bank_path)
    paired = read_jsonl(paired_path)
    training_documents = {document_id(row) for row in paired}
    eligible = [
        row
        for row in bank
        if document_id(row) not in training_documents
    ]

    # Keep at most one candidate window from each source document before
    # sampling, so prompts cannot silently overrepresent a long document.
    candidates_by_source: dict[str, dict[str, dict]] = {}
    for row in eligible:
        candidates_by_source.setdefault(row["source"], {}).setdefault(
            document_id(row),
            row,
        )
    sources = sorted(candidates_by_source)
    capacities = {
        source: len(candidates_by_source[source])
        for source in sources
    }
    quotas = allocate_balanced(args.num_prompts, sources, capacities)
    rng = random.Random(args.seed)
    selected = []
    for source in sources:
        candidates = list(candidates_by_source[source].values())
        candidates.sort(key=lambda row: row["sample_id"])
        rng.shuffle(candidates)
        if len(candidates) < quotas[source]:
            raise RuntimeError(
                f"{source} has {len(candidates)} eligible documents but "
                f"needs {quotas[source]}"
            )
        selected.extend(candidates[:quotas[source]])
    rng.shuffle(selected)

    prompts = []
    for index, row in enumerate(selected):
        doc_hash = hashlib.sha256(
            document_id(row).encode("utf-8")
        ).hexdigest()
        prompts.append({
            "id": f"{args.language}_formal_{index:04d}",
            "text": row["prefix"],
            "language": args.language,
            "source": row["source"],
            "domain": row["domain"],
            "sample_id": row["sample_id"],
            "source_document_sha256": doc_hash,
            "window_sha256": row["window_sha256"],
            "prefix_units": row["prefix_units"],
            "length_unit": row["length_unit"],
            "selection": "verbatim_from_frozen_prefix_bank",
        })

    ids = [row["id"] for row in prompts]
    texts = [row["text"] for row in prompts]
    document_hashes = [row["source_document_sha256"] for row in prompts]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate prompt IDs")
    if len(set(texts)) != len(texts):
        raise RuntimeError("duplicate prompt text")
    if len(set(document_hashes)) != len(document_hashes):
        raise RuntimeError("multiple prompts selected from one document")

    atomic_json(output_path, prompts)
    manifest_path = output_path.with_suffix(".manifest.json")
    manifest = {
        "schema_version": 1,
        "language": args.language,
        "num_prompts": len(prompts),
        "seed": args.seed,
        "selection": (
            "balanced across prefix-bank sources; one verbatim prefix per "
            "source document; every paired-training source document excluded"
        ),
        "document_disjoint_from_paired_training": True,
        "bank_path": portable_path(bank_path),
        "bank_sha256": file_sha256(bank_path),
        "paired_path": portable_path(paired_path),
        "paired_sha256": file_sha256(paired_path),
        "source_quotas": quotas,
        "eligible_documents_by_source": {
            source: len(rows)
            for source, rows in candidates_by_source.items()
        },
        "output_path": portable_path(output_path),
        "output_sha256": file_sha256(output_path),
    }
    atomic_json(manifest_path, manifest)
    print(f"Wrote {len(prompts)} prompts to {output_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
