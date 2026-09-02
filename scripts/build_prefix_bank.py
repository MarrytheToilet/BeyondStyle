#!/usr/bin/env python3
"""Build a balanced prefix bank from already-downloaded HF row subsets."""

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Tuple
from urllib.parse import unquote, urlparse

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.natural_window import NaturalLanguageWindowSplitter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
LANGUAGES = ("en", "zh")
BANK_NAME = "balanced_natural"
DEFAULT_LENGTHS = {
    "en": {"prefix": 75, "continuation": 125},
    "zh": {"prefix": 120, "continuation": 220},
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--language", choices=LANGUAGES, required=True)
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Default: data/corpora/<language>.yaml",
    )
    parser.add_argument("--sources", nargs="+", default=None)
    parser.add_argument("--per-source", type=int, default=1000)
    parser.add_argument(
        "--prefix-units",
        type=int,
        default=None,
        help="Default: 75 English words or 120 Chinese text units.",
    )
    parser.add_argument(
        "--continuation-units",
        type=int,
        default=None,
        help="Default: 125 English words or 220 Chinese text units.",
    )
    parser.add_argument("--tolerance", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing bank with the same output path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Default: data/prefixes/<language>/balanced_natural/samples.jsonl"
        ),
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_registry(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def project_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def recorded_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def repo_slug(repo_id: str) -> str:
    return repo_id.replace("/", "__")


def source_raw_dir(registry: dict, spec: dict) -> Path:
    return (
        project_path(Path(registry["download_root"]))
        / repo_slug(spec["dataset_id"])
        / spec["snapshot_commit"]
    )


def source_subset_file(registry: dict, spec: dict) -> Path:
    path = source_raw_dir(registry, spec) / f"rows_{spec['sample_rows']}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            f"No local raw subset for {spec['dataset_id']} at {path}. "
            "Run scripts/download_hf_raw_datasets.py first."
        )
    return path


def wiki_lingua_row(item: dict, row_index: int) -> dict:
    article = item.get("article") or {}
    sections = article.get("document") or []
    text = "\n".join(
        section.strip()
        for section in sections
        if isinstance(section, str) and section.strip()
    )
    url = item.get("url", "")
    title = unquote(urlparse(url).path.rsplit("/", 1)[-1]).replace("-", " ")
    return {
        "text": text,
        "row_id": url or row_index,
        "title": title,
        "url": url,
        "native_metadata": {
            "section_names": article.get("section_name") or [],
            "section_count": len(sections),
        },
    }


def pg19_row(item: dict, row_index: int) -> dict:
    return {
        "text": item.get("text", ""),
        "row_id": item.get("url") or row_index,
        "title": item.get("short_book_title", ""),
        "url": item.get("url", ""),
        "native_metadata": {
            "publication_date": item.get("publication_date"),
        },
    }


def generic_row(item: dict, row_index: int, text_field: str) -> dict:
    return {
        "text": item.get(text_field, ""),
        "row_id": item.get("id", row_index),
        "title": item.get("title", ""),
        "url": item.get("url", ""),
        "native_metadata": {},
    }


def adapt_row(source_name: str, item: dict, row_index: int, spec: dict) -> dict:
    if source_name == "wiki_lingua":
        return wiki_lingua_row(item, row_index)
    if source_name == "pg19":
        return pg19_row(item, row_index)
    text_adapter = spec.get("text_adapter", "text")
    if "." in text_adapter:
        raise ValueError(
            f"No adapter implemented for nested field {text_adapter!r} "
            f"in source {source_name}"
        )
    return generic_row(item, row_index, text_adapter)


def source_rows(
    source_name: str,
    spec: dict,
    registry: dict,
    seed: int,
) -> Iterator[dict]:
    path = source_subset_file(registry, spec)
    prioritized_offsets = []
    with path.open("rb") as handle:
        row_index = 0
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            raw_record = json.loads(line)
            provenance = raw_record["provenance"]
            identity = (
                f"{seed}|{source_name}|{provenance['source_file']}|"
                f"{provenance['source_row_index']}"
            )
            priority = hashlib.sha256(identity.encode("utf-8")).digest()
            prioritized_offsets.append((priority, offset, row_index))
            row_index += 1

    prioritized_offsets.sort()
    with path.open("rb") as handle:
        for _, offset, row_index in prioritized_offsets:
            handle.seek(offset)
            raw_record = json.loads(handle.readline())
            row = adapt_row(
                source_name,
                raw_record["native_row"],
                row_index,
                spec,
            )
            row["raw_provenance"] = raw_record["provenance"]
            yield row


def normalized_window_hash(prefix: str, continuation: str) -> str:
    text = re.sub(r"\W+", "", prefix + continuation).lower()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def candidate_suffixes(
    text: str,
    splitter: NaturalLanguageWindowSplitter,
    rng: random.Random,
    source_name: str,
    language: str,
    max_candidate_starts: int,
) -> Iterator[Tuple[str, int]]:
    spans = splitter.sentence_spans(text)
    if len(spans) < 4:
        return

    # Full books often contain title pages and tables of contents. Selecting an
    # internal sentence window uses the existing text without rewriting it.
    first_allowed = max(0, int(len(spans) * 0.05))
    if source_name == "pg19":
        first_allowed = max(first_allowed, min(20, len(spans) // 4))
    last_allowed = len(spans) - 3
    if last_allowed < first_allowed:
        return
    candidate_count = min(
        max_candidate_starts,
        last_allowed - first_allowed + 1,
    )
    starts = rng.sample(
        range(first_allowed, last_allowed + 1),
        candidate_count,
    )
    del language
    for start in starts:
        # Bound the amount of a full book inspected for each candidate while
        # preserving the exact source substring and sentence boundaries.
        end = min(start + 64, len(spans))
        yield text[spans[start].start : spans[end - 1].end], start


def extract_pairs(
    source_name: str,
    row: dict,
    splitter: NaturalLanguageWindowSplitter,
    rng: random.Random,
    language: str,
    max_pairs: int,
    max_candidate_starts: int,
) -> Tuple[list, Counter]:
    diagnostics = Counter()
    raw_text = row.get("text") or ""
    if not isinstance(raw_text, str):
        diagnostics["non_string_text"] += 1
        return [], diagnostics
    text = raw_text.strip()
    if not text:
        diagnostics["empty"] += 1
        return [], diagnostics

    accepted = []
    intervals = []
    for suffix, start_sentence in candidate_suffixes(
        text,
        splitter,
        rng,
        source_name,
        language,
        max_candidate_starts,
    ):
        pair = splitter.split_text(suffix)
        if pair is None:
            diagnostics["invalid_candidate_window"] += 1
            continue
        quality_issues = sorted(set(
            splitter.prose_quality_issues(pair.prefix)
            + splitter.prose_quality_issues(pair.continuation)
        ))
        if quality_issues:
            for issue in quality_issues:
                diagnostics[f"non_prose_{issue}"] += 1
            continue
        sentence_span = (
            pair.metadata["prefix_sentences"]
            + pair.metadata["continuation_sentences"]
        )
        end_sentence = start_sentence + sentence_span
        overlaps = any(
            start_sentence < prior_end and end_sentence > prior_start
            for prior_start, prior_end in intervals
        )
        if overlaps:
            diagnostics["overlapping_candidate_window"] += 1
            continue
        intervals.append((start_sentence, end_sentence))
        accepted.append((pair, start_sentence, end_sentence))
        if len(accepted) >= max_pairs:
            break

    if not accepted:
        diagnostics["document_without_valid_window"] += 1
    return accepted, diagnostics


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    registry_path = project_path(
        args.registry or Path("data/corpora") / f"{args.language}.yaml"
    )
    registry = load_registry(registry_path)
    if registry.get("language") != args.language:
        raise SystemExit(
            f"Registry language {registry.get('language')!r} does not match "
            f"--language {args.language!r}"
        )
    source_specs = registry["sources"]
    selected = args.sources or [
        name
        for name, spec in source_specs.items()
        if spec.get("enabled") and spec.get("tier") == "core"
    ]
    unknown = sorted(set(selected) - set(source_specs))
    if unknown:
        raise SystemExit(f"Unknown sources: {', '.join(unknown)}")

    prefix_units = (
        args.prefix_units
        if args.prefix_units is not None
        else DEFAULT_LENGTHS[args.language]["prefix"]
    )
    continuation_units = (
        args.continuation_units
        if args.continuation_units is not None
        else DEFAULT_LENGTHS[args.language]["continuation"]
    )
    splitter = NaturalLanguageWindowSplitter(
        language=args.language,
        target_prefix_units=prefix_units,
        target_continuation_units=continuation_units,
        tolerance=args.tolerance,
    )

    records = []
    seen_hashes = set()
    reports = {}
    for source_offset, source_name in enumerate(selected):
        spec = source_specs[source_name]
        if not spec.get("enabled"):
            raise SystemExit(
                f"Source {source_name} is disabled in {args.registry}; "
                "enable it only after completing its audit."
            )

        rng = random.Random(args.seed + source_offset)
        rejection_counts = Counter()
        accepted = 0
        scanned = 0
        documents_with_windows = 0
        windows_per_document = Counter()
        for row in source_rows(
            source_name,
            spec,
            registry,
            args.seed + source_offset,
        ):
            if accepted >= args.per_source:
                break
            scanned += 1
            extracted, diagnostics = extract_pairs(
                source_name,
                row,
                splitter,
                rng,
                args.language,
                max_pairs=int(spec.get("max_windows_per_document", 1)),
                max_candidate_starts=int(
                    spec.get("max_candidate_starts", 40)
                ),
            )
            rejection_counts.update(diagnostics)
            if not extracted:
                continue
            accepted_from_document = 0
            for pair, start_sentence, end_sentence in extracted:
                if accepted >= args.per_source:
                    break
                window_hash = normalized_window_hash(
                    pair.prefix, pair.continuation
                )
                if window_hash in seen_hashes:
                    rejection_counts["duplicate_window"] += 1
                    continue
                seen_hashes.add(window_hash)

                sample_id = (
                    f"{args.language}_{source_name}_{accepted:05d}"
                )
                records.append(
                    {
                        "schema_version": 1,
                        "sample_id": sample_id,
                        "source": source_name,
                        "domain": spec["discourse_domain"],
                        "language": args.language,
                        "prefix": pair.prefix,
                        "human_continuation": pair.continuation,
                        "prefix_units": pair.prefix_units,
                        "continuation_units": pair.continuation_units,
                        "length_unit": pair.length_unit,
                        "prefix_characters": len(pair.prefix),
                        "continuation_characters": len(pair.continuation),
                        "prefix_sentences": pair.metadata[
                            "prefix_sentences"
                        ],
                        "continuation_sentences": pair.metadata[
                            "continuation_sentences"
                        ],
                        "source_document": {
                            "dataset_id": spec["dataset_id"],
                            "source_revision": spec.get("source_revision"),
                            "snapshot_commit": spec.get("snapshot_commit"),
                            "config": spec.get("config"),
                            "split": spec.get("prefix_split"),
                            "row_id": str(row["row_id"]),
                            "title": row.get("title", ""),
                            "url": row.get("url", ""),
                            "window_start_sentence": start_sentence,
                            "window_end_sentence": end_sentence,
                            "raw_source_file": row["raw_provenance"][
                                "source_file"
                            ],
                            "raw_source_row_index": row["raw_provenance"][
                                "source_row_index"
                            ],
                            "native_metadata": row.get(
                                "native_metadata", {}
                            ),
                        },
                        "window_sha256": window_hash,
                    }
                )
                accepted += 1
                accepted_from_document += 1
            if accepted_from_document:
                documents_with_windows += 1
                windows_per_document[accepted_from_document] += 1

        reports[source_name] = {
            "requested": args.per_source,
            "accepted": accepted,
            "scanned": scanned,
            "documents_with_windows": documents_with_windows,
            "windows_per_document": {
                str(key): value
                for key, value in sorted(windows_per_document.items())
            },
            "max_windows_per_document": int(
                spec.get("max_windows_per_document", 1)
            ),
            "rejections": dict(sorted(rejection_counts.items())),
            "dataset_id": spec["dataset_id"],
            "source_revision": spec.get("source_revision"),
            "snapshot_commit": spec.get("snapshot_commit"),
            "raw_subset": str(
                source_subset_file(registry, spec).relative_to(PROJECT_ROOT)
            ),
        }
        print(
            f"{source_name}: accepted={accepted}/{args.per_source}, "
            f"scanned={scanned}",
            flush=True,
        )
        if accepted < args.per_source:
            raise SystemExit(
                f"Source {source_name} produced only {accepted}/"
                f"{args.per_source} valid windows"
            )

    random.Random(args.seed).shuffle(records)
    output = project_path(
        args.output
        or Path("data/prefixes")
        / args.language
        / BANK_NAME
        / "samples.jsonl"
    )
    manifest_path = output.parent / "manifest.json"
    if not args.force and (output.exists() or manifest_path.exists()):
        raise SystemExit(
            f"Bank already exists at {output.parent}; pass --force to replace it"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".jsonl.incomplete")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary.replace(output)

    manifest = {
        "schema_version": 1,
        "created_at": utc_now(),
        "language": args.language,
        "bank_id": BANK_NAME,
        "registry": recorded_path(registry_path),
        "seed": args.seed,
        "length_basis": "model_independent_natural_language_units",
        "length_unit": splitter.length_unit,
        "length_unit_definition": (
            "Unicode word-like spans"
            if args.language == "en"
            else "each CJK character plus each contiguous Latin/digit span"
        ),
        "tokenizer_used_for_selection": False,
        "boundary_rule": "complete_sentence_boundaries",
        "prose_filter": "model_independent_prose_gate",
        "prefix_units": prefix_units,
        "continuation_units": continuation_units,
        "tolerance": args.tolerance,
        "sources": reports,
        "n_samples": len(records),
        "output": recorded_path(output),
        "output_sha256": file_sha256(output),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(records)} windows to {output}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
