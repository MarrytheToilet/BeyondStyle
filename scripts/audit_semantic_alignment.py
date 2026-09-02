#!/usr/bin/env python3
"""Audit semantic alignment in plan-conditioned and prefix-only pairs.

The encoder is used only as an independent corpus diagnostic. It never
defines a linguistic feature, hidden readout, intervention axis, or example
selection decision.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import sentence_transformers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-pairs",
        type=Path,
        default=Path("data/paired/en/proposition_aligned/pairs.jsonl"),
    )
    parser.add_argument(
        "--prefix-only-pairs",
        type=Path,
        default=Path("data/paired/en/prefix_only_control/pairs.jsonl"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("../../models/bge-base-en-v1.5"),
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--null-permutations", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "outputs/en/05_linguistic/prefix_only_control/"
            "semantic_alignment"
        ),
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def bootstrap_interval(
    values: np.ndarray,
    *,
    draws: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def within_source_derangement(
    sources: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    permutation = np.arange(len(sources))
    for source in sorted(np.unique(sources)):
        indices = np.flatnonzero(sources == source)
        for _ in range(10_000):
            shuffled = rng.permutation(indices)
            if not np.any(shuffled == indices):
                permutation[indices] = shuffled
                break
        else:
            raise RuntimeError(f"Could not derange source stratum {source}")
    return permutation


def validate_pairs(
    plan: list[dict[str, Any]],
    prefix_only: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    control_by_id = {
        str(row["sample_id"]): row for row in prefix_only
    }
    if len(control_by_id) != len(prefix_only):
        raise ValueError("Prefix-only sample IDs are not unique")
    if {str(row["sample_id"]) for row in plan} != set(control_by_id):
        raise ValueError("Plan and prefix-only sample IDs do not match")
    ordered = []
    for row in plan:
        control = control_by_id[str(row["sample_id"])]
        for field in (
            "prefix",
            "human_continuation",
            "source",
            "domain",
        ):
            if row[field] != control[field]:
                raise ValueError(
                    f"{row['sample_id']}: mismatched {field}"
                )
        if (
            row["generation"]["generator_model"]
            != control["generation"]["generator_model"]
        ):
            raise ValueError(
                f"{row['sample_id']}: mismatched generator assignment"
            )
        ordered.append(control)
    return ordered


def main() -> None:
    args = parse_args()
    if args.bootstrap < 1 or args.null_permutations < 1:
        raise ValueError("Bootstrap and null counts must be positive")
    started = datetime.now(timezone.utc)
    plan = load_jsonl(args.plan_pairs)
    prefix_only = validate_pairs(
        plan,
        load_jsonl(args.prefix_only_pairs),
    )
    if not plan:
        raise ValueError("No matched pairs found")

    human_texts = [row["human_continuation"] for row in plan]
    plan_texts = [row["ai_continuation"] for row in plan]
    prefix_texts = [row["ai_continuation"] for row in prefix_only]
    encoder = SentenceTransformer(
        str(args.model),
        device=args.device,
    )
    embeddings = encoder.encode(
        human_texts + plan_texts + prefix_texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    human, plan_ai, prefix_ai = np.split(embeddings, 3)
    plan_cosine = np.sum(human * plan_ai, axis=1)
    prefix_cosine = np.sum(human * prefix_ai, axis=1)
    difference = plan_cosine - prefix_cosine
    sources = np.asarray([row["source"] for row in plan])
    rng = np.random.default_rng(args.seed)

    null_means = []
    for _ in range(args.null_permutations):
        permutation = within_source_derangement(sources, rng)
        null_means.append(float(
            np.sum(human[permutation] * plan_ai, axis=1).mean()
        ))
    null_means_array = np.asarray(null_means)
    finite_p = float(
        (
            1
            + np.sum(null_means_array >= float(plan_cosine.mean()))
        )
        / (args.null_permutations + 1)
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    per_pair = pd.DataFrame({
        "sample_id": [row["sample_id"] for row in plan],
        "source": sources,
        "domain": [row["domain"] for row in plan],
        "generator_model": [
            row["generation"]["generator_model"] for row in plan
        ],
        "plan_conditioned_cosine": plan_cosine,
        "prefix_only_cosine": prefix_cosine,
        "plan_minus_prefix_only": difference,
    })
    per_pair.to_csv(
        args.out_dir / "per_pair_semantic_alignment.csv",
        index=False,
    )
    pd.DataFrame({
        "null_replication": np.arange(1, len(null_means_array) + 1),
        "within_source_deranged_mean_cosine": null_means_array,
    }).to_csv(args.out_dir / "within_source_null_means.csv", index=False)

    summary_rows = []
    for scope, mask in [
        ("all", np.ones(len(plan), dtype=bool)),
        *[
            (str(source), sources == source)
            for source in sorted(np.unique(sources))
        ],
    ]:
        local_plan = plan_cosine[mask]
        local_prefix = prefix_cosine[mask]
        local_difference = difference[mask]
        plan_ci = bootstrap_interval(
            local_plan,
            draws=args.bootstrap,
            rng=rng,
        )
        prefix_ci = bootstrap_interval(
            local_prefix,
            draws=args.bootstrap,
            rng=rng,
        )
        difference_ci = bootstrap_interval(
            local_difference,
            draws=args.bootstrap,
            rng=rng,
        )
        summary_rows.append({
            "scope": scope,
            "n_pairs": int(mask.sum()),
            "plan_conditioned_mean_cosine": float(local_plan.mean()),
            "plan_conditioned_ci_low": plan_ci[0],
            "plan_conditioned_ci_high": plan_ci[1],
            "prefix_only_mean_cosine": float(local_prefix.mean()),
            "prefix_only_ci_low": prefix_ci[0],
            "prefix_only_ci_high": prefix_ci[1],
            "plan_minus_prefix_only_mean": float(
                local_difference.mean()
            ),
            "difference_ci_low": difference_ci[0],
            "difference_ci_high": difference_ci[1],
            "plan_greater_pair_rate": float(
                np.mean(local_difference > 0)
            ),
            "within_source_deranged_mean": (
                float(null_means_array.mean())
                if scope == "all"
                else np.nan
            ),
            "finite_null_p": finite_p if scope == "all" else np.nan,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / "semantic_alignment_summary.csv", index=False)

    overall = summary.iloc[0]
    (args.out_dir / "analysis_summary.md").write_text(
        "\n".join([
            "# Semantic alignment audit",
            "",
            "The frozen encoder is an independent corpus diagnostic and "
            "does not define any linguistic or hidden-state axis.",
            "",
            "| Condition | Mean cosine [95% CI] |",
            "|---|---:|",
            (
                f"| Proposition-conditioned | "
                f"{overall.plan_conditioned_mean_cosine:.3f} "
                f"[{overall.plan_conditioned_ci_low:.3f}, "
                f"{overall.plan_conditioned_ci_high:.3f}] |"
            ),
            (
                f"| Prefix-only | {overall.prefix_only_mean_cosine:.3f} "
                f"[{overall.prefix_only_ci_low:.3f}, "
                f"{overall.prefix_only_ci_high:.3f}] |"
            ),
            (
                f"| Same-source deranged | "
                f"{overall.within_source_deranged_mean:.3f} |"
            ),
            "",
            (
                "Plan minus prefix-only: "
                f"{overall.plan_minus_prefix_only_mean:.3f} "
                f"[{overall.difference_ci_low:.3f}, "
                f"{overall.difference_ci_high:.3f}]; "
                f"{overall.plan_greater_pair_rate:.1%} of pairs are "
                "positive."
            ),
            (
                "The observed plan-conditioned mean exceeds all "
                f"{args.null_permutations} same-source derangements "
                f"(finite-null p={finite_p:.4g})."
            ),
        ]) + "\n",
        encoding="utf-8",
    )

    model_files = {}
    for name in (
        "config.json",
        "modules.json",
        "sentence_bert_config.json",
    ):
        path = args.model / name
        if path.exists():
            model_files[str(path)] = sha256_file(path)
    finished = datetime.now(timezone.utc)
    atomic_json(
        args.out_dir / "manifest.json",
        {
            "schema": "semantic_alignment_audit_v1",
            "status": "complete",
            "n_pairs": len(plan),
            "encoder_role": "independent_corpus_diagnostic_only",
            "model_path": str(args.model),
            "model_metadata_sha256": model_files,
            "normalized_embeddings": True,
            "same_source_derangement": True,
            "input_sha256": {
                str(args.plan_pairs): sha256_file(args.plan_pairs),
                str(args.prefix_only_pairs): sha256_file(
                    args.prefix_only_pairs
                ),
            },
            "bootstrap": args.bootstrap,
            "null_permutations": args.null_permutations,
            "seed": args.seed,
            "python_version": platform.python_version(),
            "sentence_transformers_version": (
                sentence_transformers.__version__
            ),
            "started_at_utc": started.isoformat(),
            "finished_at_utc": finished.isoformat(),
            "duration_seconds": (finished - started).total_seconds(),
        },
    )


if __name__ == "__main__":
    main()
