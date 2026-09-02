#!/usr/bin/env python3
"""Generate the bounded prefix-only construct-control corpus.

The control preserves every human continuation, natural-language prefix,
sample identity, source, domain, and generator assignment from the canonical
proposition-aligned English corpus. The generator receives only the prefix and
a scalar length guide: no human continuation, proposition plan, entities, or
semantic endpoint are exposed.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.synthesize_paired_data import (  # noqa: E402
    ApiPool,
    atomic_json,
    atomic_jsonl,
    load_jsonl,
    sha256_file,
)
from src.data.proposition_alignment import continuation_quality  # noqa: E402


LOGGER = logging.getLogger("synthesize_prefix_only_control")
PROMPT_FAMILY = "prefix_only_natural_continuation"
SCHEMA = "prefix_only_construct_control_pair"
EXPLICIT_REFUSAL_RE = re.compile(
    r"(?:\bas an ai\b|"
    r"\bi (?:cannot|can't|am unable to) "
    r"(?:assist|comply|complete|continue|fulfil|fulfill|generate|provide|"
    r"write)\b|"
    r"\bsorry[,，]?\s+i (?:cannot|can't|am unable))",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-pairs",
        type=Path,
        default=Path("data/paired/en/proposition_aligned/pairs.jsonl"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/paired/en/prefix_only_control"),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--generation-retries", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def prompt_messages(
    record: dict[str, Any],
    *,
    retry_feedback: str = "",
) -> list[dict[str, str]]:
    target = int(record["lengths"]["human_continuation_units"])
    minimum = int(target * 0.60)
    maximum = int(target * 1.50)
    system = (
        "Continue the document naturally and coherently from the supplied "
        "prefix. Infer what should come next from the prefix alone. Return "
        "only the continuation as ordinary prose: no title, list, analysis, "
        "explanation, or repeated prefix. Do not mention these instructions."
    )
    user = (
        f"LENGTH GUIDE: about {target} words; acceptable range "
        f"{minimum}-{maximum}.\n\n"
        "PREFIX — write only the continuation after its final character:\n"
        f"{record['prefix']}"
    )
    if retry_feedback:
        user += f"\n\nFor this retry: {retry_feedback}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def retry_feedback(quality: dict[str, Any]) -> str:
    issues = set(quality["issues"])
    messages = []
    if "continuation_length_out_of_bounds" in issues:
        if quality["units"] > quality["maximum_units"]:
            messages.append(
                "The prior unrecorded draft was too long; write a "
                "substantially shorter fresh continuation."
            )
        else:
            messages.append(
                "The prior unrecorded draft was too short; write a longer "
                "fresh continuation."
            )
    if "repeats_prefix" in issues:
        messages.append("Begin after the prefix; do not repeat it.")
    if {
        "list_or_bullets",
        "markdown_heading",
        "meta_preamble",
        "refusal_or_capability_disclaimer",
    } & issues:
        messages.append(
            "Return plain continuation prose only, without formatting or "
            "meta-commentary."
        )
    if "copies_human_continuation" in issues:
        messages.append("Write a fresh continuation in independent wording.")
    return " ".join(messages) or "Write a fresh natural continuation."


def prefix_only_quality(
    continuation: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    """Use the common prose gate without misclassifying narrative modality."""
    quality = continuation_quality(
        continuation,
        language="en",
        human_continuation=record["human_continuation"],
        prefix=record["prefix"],
        content_plan_text="",
        target_units=int(record["lengths"]["human_continuation_units"]),
    )
    issues = list(quality["issues"])
    if (
        "refusal_or_capability_disclaimer" in issues
        and not EXPLICIT_REFUSAL_RE.search(continuation)
    ):
        issues.remove("refusal_or_capability_disclaimer")
    return {
        **quality,
        "issues": issues,
        "accepted": not issues,
        "refusal_gate": "explicit_generation_capability_meta_language",
    }


def config_fingerprint(
    source_pairs: Path,
    *,
    temperature: float,
    generation_retries: int,
    limit: int | None,
) -> str:
    payload = {
        "schema": SCHEMA,
        "prompt_family": PROMPT_FAMILY,
        "source_pairs_sha256": sha256_file(source_pairs),
        "temperature": temperature,
        "generation_retries": generation_retries,
        "limit": limit,
        "length_ratio": [0.60, 1.50],
        "prompt_contract": (
            "prefix and scalar word-length guide only; no human "
            "continuation, proposition plan, entities, or endpoint"
        ),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def make_control_pair(
    source: dict[str, Any],
    continuation: str,
    quality: dict[str, Any],
    *,
    generation_attempt: int,
    temperature: float,
) -> dict[str, Any]:
    result = {
        key: source[key]
        for key in (
            "sample_id",
            "language",
            "domain",
            "source",
            "source_document",
            "prefix",
            "human_continuation",
        )
    }
    result.update({
        "schema": SCHEMA,
        "pair_id": f"en_prefix_only_{source['sample_id']}",
        "matched_proposition_pair_id": source.get("pair_id"),
        "ai_continuation": continuation,
        "lengths": {
            **source["lengths"],
            "ai_continuation_units": quality["units"],
            "ai_to_human_ratio": quality["length_ratio"],
        },
        "content_control": {
            "method": "prefix_only",
            "generator_received_prefix": True,
            "generator_received_scalar_length_guide": True,
            "generator_received_human_continuation": False,
            "generator_received_proposition_plan": False,
            "generator_received_required_entities": False,
            "generator_received_discourse_endpoint": False,
        },
        "generation": {
            "generator_model": source["generation"]["generator_model"],
            "prompt_family": PROMPT_FAMILY,
            "temperature": temperature,
            "generation_attempt": generation_attempt,
            "transport_normalization": "surrounding_whitespace_only",
            "text_repaired": False,
        },
        "validation": {
            "deterministic_quality": quality,
            "semantic_judge_used": False,
            "semantic_judge_reason": (
                "A semantic plan or human reference cannot be shown to any "
                "generation-stage model in the prefix-only control."
            ),
        },
        "provenance": {
            "matched_source_pair_id": source.get("pair_id"),
            "matched_source_pair_sample_id": source["sample_id"],
            "prefix_window_sha256": source.get("provenance", {}).get(
                "prefix_window_sha256"
            ),
        },
    })
    return result


def run_record(
    api: ApiPool,
    record: dict[str, Any],
    *,
    retries: int,
    temperature: float,
    fingerprint: str,
    work_path: Path,
) -> dict[str, Any]:
    existing = None
    if work_path.is_file():
        existing = json.loads(work_path.read_text(encoding="utf-8"))
        if existing.get("config_fingerprint") != fingerprint:
            raise RuntimeError(
                f"{work_path} belongs to another control configuration"
            )
        if existing.get("status") == "accepted":
            return existing
        for previous in existing.get("attempts", []):
            raw = previous.get("raw_response")
            if not isinstance(raw, str) or not raw.strip():
                continue
            quality = prefix_only_quality(raw.strip(), record)
            previous["deterministic_quality"] = quality
            if quality["accepted"]:
                recovered = {
                    **existing,
                    "status": "accepted",
                    "attempts": existing["attempts"],
                    "pair": make_control_pair(
                        record,
                        raw.strip(),
                        quality,
                        generation_attempt=int(previous["attempt"]),
                        temperature=temperature,
                    ),
                    "recovered_after_quality_gate_correction": True,
                    "updated_at": utc_now(),
                }
                atomic_json(work_path, recovered)
                return recovered

    model = str(record["generation"]["generator_model"])
    attempts = []
    feedback = ""
    for attempt in range(1, retries + 1):
        try:
            max_tokens = min(
                900,
                max(
                    350,
                    int(record["lengths"]["human_continuation_units"]) * 4,
                ),
            )
            raw = api.chat(
                model,
                prompt_messages(record, retry_feedback=feedback),
                max_new_tokens=max_tokens,
                temperature=temperature,
            )
            continuation = raw.strip()
            quality = prefix_only_quality(continuation, record)
            attempts.append({
                "attempt": attempt,
                "raw_response": raw,
                "deterministic_quality": quality,
            })
            if quality["accepted"]:
                result = {
                    "config_fingerprint": fingerprint,
                    "status": "accepted",
                    "sample_id": record["sample_id"],
                    "attempts": attempts,
                    "pair": make_control_pair(
                        record,
                        continuation,
                        quality,
                        generation_attempt=attempt,
                        temperature=temperature,
                    ),
                    "updated_at": utc_now(),
                }
                atomic_json(work_path, result)
                return result
            feedback = retry_feedback(quality)
        except Exception as exc:
            attempts.append({
                "attempt": attempt,
                "error_type": type(exc).__name__,
                "error": str(exc),
            })
    result = {
        "config_fingerprint": fingerprint,
        "status": "failed",
        "sample_id": record["sample_id"],
        "attempts": attempts,
        "updated_at": utc_now(),
    }
    atomic_json(work_path, result)
    return result


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    if args.workers < 1 or args.generation_retries < 1:
        raise ValueError("workers and generation retries must be positive")
    rows = load_jsonl(args.source_pairs)
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("No proposition-aligned source pairs found")
    if any(row.get("language") != "en" for row in rows):
        raise ValueError("The bounded construct control is English-only")
    sample_ids = [str(row["sample_id"]) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Source sample IDs must be unique")
    models = sorted({
        str(row["generation"]["generator_model"]) for row in rows
    })
    api = ApiPool(models, timeout=args.timeout)
    fingerprint = config_fingerprint(
        args.source_pairs,
        temperature=args.temperature,
        generation_retries=args.generation_retries,
        limit=args.limit,
    )
    work_dir = args.out_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_record,
                api,
                row,
                retries=args.generation_retries,
                temperature=args.temperature,
                fingerprint=fingerprint,
                work_path=work_dir / f"{row['sample_id']}.json",
            ): row
            for row in rows
        }
        for completed, future in enumerate(
            as_completed(futures),
            start=1,
        ):
            result = future.result()
            results.append(result)
            LOGGER.info(
                "%d/%d %s %s",
                completed,
                len(rows),
                result["sample_id"],
                result["status"],
            )
    by_id = {result["sample_id"]: result for result in results}
    ordered = [by_id[sample_id] for sample_id in sample_ids]
    failures = [row for row in ordered if row["status"] != "accepted"]
    atomic_jsonl(args.out_dir / "failures.jsonl", failures)
    if failures:
        raise RuntimeError(
            f"{len(failures)} prefix-only continuations exhausted retries; "
            f"inspect {args.out_dir / 'failures.jsonl'}"
        )

    pairs = [row["pair"] for row in ordered]
    pairs_path = args.out_dir / "pairs.jsonl"
    atomic_jsonl(pairs_path, pairs)
    source_generator = {
        row["sample_id"]: row["generation"]["generator_model"]
        for row in rows
    }
    if any(
        pair["generation"]["generator_model"]
        != source_generator[pair["sample_id"]]
        for pair in pairs
    ):
        raise RuntimeError("Generator assignment changed in construct control")
    attempt_counts = Counter(
        int(pair["generation"]["generation_attempt"]) for pair in pairs
    )
    manifest = {
        "schema": "prefix_only_construct_control_manifest",
        "status": "complete",
        "created_at": utc_now(),
        "n_pairs": len(pairs),
        "config_fingerprint": fingerprint,
        "source_pairs": str(args.source_pairs),
        "source_pairs_sha256": sha256_file(args.source_pairs),
        "pairs_sha256": sha256_file(pairs_path),
        "same_sample_ids_as_source": (
            [pair["sample_id"] for pair in pairs] == sample_ids
        ),
        "same_prefixes_as_source": all(
            pair["prefix"] == source["prefix"]
            for pair, source in zip(pairs, rows)
        ),
        "same_human_continuations_as_source": all(
            pair["human_continuation"] == source["human_continuation"]
            for pair, source in zip(pairs, rows)
        ),
        "same_generator_assignments_as_source": True,
        "generator_models": models,
        "generator_model_counts": dict(sorted(Counter(
            pair["generation"]["generator_model"] for pair in pairs
        ).items())),
        "source_counts": dict(sorted(Counter(
            pair["source"] for pair in pairs
        ).items())),
        "generation_attempt_counts": dict(sorted(attempt_counts.items())),
        "prompt_information": {
            "prefix": True,
            "scalar_length_guide": True,
            "human_continuation": False,
            "proposition_plan": False,
            "required_entities": False,
            "discourse_endpoint": False,
        },
        "acceptance": {
            "nominal_natural_unit_length_ratio": [0.60, 1.50],
            "integer_length_bounds": (
                "inclusive floor(target_units * ratio); exact bounds are "
                "stored per pair"
            ),
            "deterministic_prose_gate": True,
            "refusal_gate": (
                "explicit generation-capability meta-language only; "
                "ordinary first-person narrative modality is retained"
            ),
            "long_verbatim_copy_gate": True,
            "semantic_judge": False,
            "text_repair": False,
        },
        "outputs": {
            "pairs": str(pairs_path),
            "failures": str(args.out_dir / "failures.jsonl"),
            "work_directory": str(work_dir),
        },
    }
    atomic_json(args.out_dir / "manifest.json", manifest)
    LOGGER.info("prefix-only control complete: %s", pairs_path)


if __name__ == "__main__":
    main()
