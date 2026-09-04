#!/usr/bin/env python3
"""Compare canonical and simpler directions on the same generation estimands."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.causal_linguistics import paired_mean_inference  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260902)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_responses(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "prompt_id",
        "prompt_index",
        "sample_seed_index",
        "component",
        "response_group",
        "value",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    key = [
        "prompt_id",
        "prompt_index",
        "sample_seed_index",
        "component",
        "response_group",
    ]
    if frame.duplicated(key).any():
        raise ValueError(f"{path} contains duplicate response keys")
    return frame


def lexical_estimands(frame: pd.DataFrame, component: str) -> pd.DataFrame:
    subset = frame[frame["component"] == component]
    pivot = subset.pivot(
        index=["prompt_id", "prompt_index", "sample_seed_index"],
        columns="response_group",
        values="value",
    )
    required = {"lexical", "syntactic", "discourse"}
    if set(pivot.columns) != required or pivot.isna().any().any():
        raise ValueError(
            f"{component} must have complete lexical/syntactic/discourse responses"
        )
    result = pd.DataFrame(index=pivot.index)
    result["target_projection"] = pivot["lexical"]
    result["component_selectivity"] = pivot["lexical"] - 0.5 * (
        pivot["syntactic"] + pivot["discourse"]
    )
    return result.reset_index()


def main() -> None:
    args = parse_args()
    canonical_path = args.canonical_dir / "prompt_seed_group_responses.csv"
    baseline_path = args.baseline_dir / "prompt_seed_group_responses.csv"
    canonical = load_responses(canonical_path)
    baseline = load_responses(baseline_path)
    specifications = [
        ("canonical_pullback", lexical_estimands(canonical, "lexical")),
        ("encoding_pattern", lexical_estimands(baseline, "lexical")),
        ("source_contrast", lexical_estimands(baseline, "source")),
    ]
    observation_keys = ["prompt_id", "prompt_index", "sample_seed_index"]
    key_sets = {
        label: set(map(tuple, frame[observation_keys].to_numpy()))
        for label, frame in specifications
    }
    if len({frozenset(keys) for keys in key_sets.values()}) != 1:
        raise ValueError("Directions do not share the same prompt×seed observations")

    prompt_frames: dict[str, pd.DataFrame] = {}
    summary_rows = []
    prompt_rows = []
    for direction_number, (direction, frame) in enumerate(specifications):
        prompt = (
            frame.groupby(["prompt_id", "prompt_index"], as_index=False)[
                ["target_projection", "component_selectivity"]
            ]
            .mean()
            .sort_values("prompt_index")
        )
        prompt_frames[direction] = prompt.set_index("prompt_id")
        for estimand_number, estimand in enumerate(
            ["target_projection", "component_selectivity"]
        ):
            values = prompt[estimand].to_numpy(dtype=float)
            inference = paired_mean_inference(
                values,
                bootstrap=args.bootstrap,
                permutations=args.permutations,
                seed=args.seed + 100 * direction_number + estimand_number,
            )
            seed_means = frame.groupby("sample_seed_index")[estimand].mean()
            summary_rows.append(
                {
                    "direction": direction,
                    "estimand": estimand,
                    "n_prompts": int(len(prompt)),
                    "n_seeds": int(frame["sample_seed_index"].nunique()),
                    "seed_means": json.dumps(
                        {
                            str(int(index)): float(value)
                            for index, value in seed_means.items()
                        },
                        sort_keys=True,
                    ),
                    **inference,
                }
            )
        local = prompt.copy()
        local.insert(0, "direction", direction)
        prompt_rows.extend(local.to_dict(orient="records"))

    contrast_rows = []
    canonical_prompt = prompt_frames["canonical_pullback"]
    for comparator_number, comparator in enumerate(
        ["encoding_pattern", "source_contrast"]
    ):
        comparison = prompt_frames[comparator]
        if list(canonical_prompt.index) != list(comparison.index):
            comparison = comparison.reindex(canonical_prompt.index)
        if comparison.isna().any().any():
            raise ValueError(f"Prompt mismatch for comparator {comparator}")
        for estimand_number, estimand in enumerate(
            ["target_projection", "component_selectivity"]
        ):
            values = (
                canonical_prompt[estimand] - comparison[estimand]
            ).to_numpy(dtype=float)
            contrast_rows.append(
                {
                    "reference": "canonical_pullback",
                    "comparator": comparator,
                    "estimand": estimand,
                    "n_prompts": int(len(values)),
                    **paired_mean_inference(
                        values,
                        bootstrap=args.bootstrap,
                        permutations=args.permutations,
                        seed=(
                            args.seed
                            + 1000
                            + 100 * comparator_number
                            + estimand_number
                        ),
                    ),
                }
            )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(prompt_rows).to_csv(
        args.out_dir / "prompt_direction_estimands.csv", index=False
    )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.out_dir / "direction_summary.csv", index=False)
    contrasts = pd.DataFrame(contrast_rows)
    contrasts.to_csv(args.out_dir / "canonical_contrasts.csv", index=False)
    manifest = {
        "started_at": datetime.now(timezone.utc).isoformat(),
        "canonical_input": {
            "path": str(canonical_path.resolve()),
            "sha256": sha256_file(canonical_path),
        },
        "baseline_input": {
            "path": str(baseline_path.resolve()),
            "sha256": sha256_file(baseline_path),
        },
        "analysis_unit": "prompt mean after paired-seed averaging",
        "bootstrap": args.bootstrap,
        "permutations": args.permutations,
        "seed": args.seed,
        "python": platform.python_version(),
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print("\nCanonical minus comparator:\n")
    print(contrasts.to_string(index=False))


if __name__ == "__main__":
    main()
