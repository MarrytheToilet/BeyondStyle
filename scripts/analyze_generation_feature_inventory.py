#!/usr/bin/env python3
"""Audit every feature in a steered surface group using prompt-level inference.

This analysis consumes the paired feature differences produced by
``analyze_seed_robustness.py``.  It keeps prompts, rather than decode seeds, as
the inferential unit and applies Benjamini--Hochberg correction across the
predeclared features in the requested group.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.causal_linguistics import paired_mean_inference  # noqa: E402
from src.linguistics import FEATURE_DEFINITIONS  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--differences", type=Path, required=True)
    parser.add_argument("--observational-features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--component", default="lexical")
    parser.add_argument("--group", default="lexical")
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260903)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    """Return monotone Benjamini--Hochberg adjusted p-values."""
    order = np.argsort(p_values)
    adjusted = np.ones(len(p_values), dtype=float)
    running = 1.0
    for rank, index in reversed(list(enumerate(order, start=1))):
        running = min(running, float(p_values[index]) * len(p_values) / rank)
        adjusted[index] = running
    return adjusted


def main() -> None:
    args = parse_args()
    started = datetime.now(timezone.utc)
    differences = pd.read_csv(args.differences)
    observational = pd.read_csv(args.observational_features)
    feature_names = [
        item.name for item in FEATURE_DEFINITIONS if item.group == args.group
    ]
    if not feature_names:
        raise ValueError(f"Unknown or empty feature group: {args.group}")
    required = {
        "component",
        "prompt_id",
        "prompt_index",
        "sample_seed_index",
        *feature_names,
    }
    missing = required - set(differences)
    if missing:
        raise ValueError(f"Missing difference columns: {sorted(missing)}")
    missing_observational = set(feature_names) - set(observational)
    if missing_observational:
        raise ValueError(
            "Missing observational columns: "
            f"{sorted(missing_observational)}"
        )

    selected = differences[differences["component"] == args.component].copy()
    if selected.empty:
        raise ValueError(f"No rows for component: {args.component}")

    prompt_effect_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for feature_number, feature in enumerate(feature_names):
        pivot = selected.pivot(
            index=["prompt_id", "prompt_index"],
            columns="sample_seed_index",
            values=feature,
        ).sort_index(axis=1)
        if pivot.isna().any().any():
            raise ValueError(f"Incomplete decode-seed coverage for {feature}")
        prompt_means = pivot.mean(axis=1)
        source_values = observational[feature].to_numpy(dtype=float)
        source_mean = float(source_values.mean())
        source_sd = float(source_values.std(ddof=1))
        source_sign = float(np.sign(source_mean))
        inference = paired_mean_inference(
            prompt_means.to_numpy(dtype=float),
            bootstrap=args.bootstrap,
            permutations=args.permutations,
            seed=args.seed + feature_number,
        )
        seed_means = pivot.mean(axis=0)
        summary_rows.append(
            {
                "component": args.component,
                "group": args.group,
                "feature": feature,
                "observational_mean": source_mean,
                "observational_paired_d": (
                    source_mean / source_sd if source_sd > 1e-12 else np.nan
                ),
                "expected_machineward_sign": source_sign,
                "mean": inference["mean"],
                "ci_low": inference["ci_low"],
                "ci_high": inference["ci_high"],
                "aligned_mean": source_sign * inference["mean"],
                "aligned_ci_low": min(
                    source_sign * inference["ci_low"],
                    source_sign * inference["ci_high"],
                ),
                "aligned_ci_high": max(
                    source_sign * inference["ci_low"],
                    source_sign * inference["ci_high"],
                ),
                "p_value": inference["sign_flip_p"],
                "n_prompts": int(len(prompt_means)),
                "n_seeds": int(pivot.shape[1]),
                "seed_means": json.dumps(
                    {
                        str(int(index)): float(value)
                        for index, value in seed_means.items()
                    },
                    sort_keys=True,
                ),
                "all_seed_means_machineward": bool(
                    (source_sign * seed_means.to_numpy(dtype=float) > 0).all()
                ),
            }
        )
        for (prompt_id, prompt_index), value in prompt_means.items():
            prompt_effect_rows.append(
                {
                    "component": args.component,
                    "group": args.group,
                    "feature": feature,
                    "prompt_id": prompt_id,
                    "prompt_index": int(prompt_index),
                    "prompt_mean": float(value),
                    "aligned_prompt_mean": float(source_sign * value),
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary["bh_q_value"] = bh_adjust(
        summary["p_value"].to_numpy(dtype=float)
    )
    summary["resolved_after_bh"] = (
        (summary["bh_q_value"] < 0.05)
        & (summary["aligned_ci_low"] > 0)
    )
    summary = summary.sort_values(
        ["resolved_after_bh", "bh_q_value", "feature"],
        ascending=[False, True, True],
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / "feature_inventory_summary.csv", index=False)
    pd.DataFrame(prompt_effect_rows).to_csv(
        args.out_dir / "prompt_feature_effects.csv", index=False
    )
    lines = [
        f"# {args.group.title()} generation feature inventory",
        "",
        "Intervals and tests use prompt means after averaging paired decode "
        "seeds. BH correction covers the complete predeclared feature group.",
        "",
        "| Feature | Source paired d | Raw change [95% CI] | "
        "Machineward-aligned change | BH q | All seeds aligned |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for row in summary.itertuples():
        lines.append(
            f"| {row.feature} | {row.observational_paired_d:.3f} | "
            f"{row.mean:.4f} [{row.ci_low:.4f}, {row.ci_high:.4f}] | "
            f"{row.aligned_mean:.4f} | {row.bh_q_value:.4f} | "
            f"{'yes' if row.all_seed_means_machineward else 'no'} |"
        )
    (args.out_dir / "analysis_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema": "generation_feature_inventory_v1",
        "status": "complete",
        "started_at_utc": started.isoformat(),
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "differences": str(args.differences.resolve()),
        "differences_sha256": sha256_file(args.differences),
        "observational_features": str(args.observational_features.resolve()),
        "observational_features_sha256": sha256_file(
            args.observational_features
        ),
        "component": args.component,
        "group": args.group,
        "n_features": len(feature_names),
        "inference_unit": "prompt mean across paired decode seeds",
        "multiplicity": "Benjamini-Hochberg across the complete group",
        "bootstrap": args.bootstrap,
        "permutations": args.permutations,
        "seed": args.seed,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
