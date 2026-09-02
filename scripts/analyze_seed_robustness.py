#!/usr/bin/env python3
"""Analyze multi-seed ± steering without treating seeds as new prompts.

The primary interval resamples prompt-level means after averaging paired
decode-seed effects within prompt. Seed-specific means and within-prompt
sampling dispersion are retained as separate diagnostics.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import sys
from typing import Any

import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.causal_linguistics import (  # noqa: E402
    paired_mean_inference,
    source_signature_projection,
)
from src.linguistics import (  # noqa: E402
    FEATURE_DEFINITIONS,
    EnglishLinguisticFeatureExtractor,
)


GROUPS = ("lexical", "syntactic", "discourse")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--direction-family",
        default="canonical_pullback",
        help=(
            "Reader-facing family label recorded in all summaries, e.g. "
            "canonical_pullback or encoding_and_source."
        ),
    )
    parser.add_argument(
        "--observational-features",
        type=Path,
        default=Path(
            "outputs/en/05_linguistic/features/"
            "paired_feature_differences.csv"
        ),
    )
    parser.add_argument(
        "--components",
        nargs="+",
        default=["lexical", "syntactic"],
    )
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--n-process",
        type=int,
        default=1,
        help="spaCy worker processes used for feature extraction.",
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument(
        "--endpoint-strength",
        type=float,
        default=1.0,
        help=(
            "Positive magnitude of the paired symmetric steering endpoints "
            "to compare (default: 1.0)."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_generation_records(
    input_dir: Path,
    components: list[str],
    endpoint_strength: float = 1.0,
) -> list[dict[str, Any]]:
    negative_key = str(float(-endpoint_strength))
    positive_key = str(float(endpoint_strength))
    expected_outputs = {negative_key, positive_key}
    records: list[dict[str, Any]] = []
    observation_sets: dict[str, set[tuple[str, int]]] = {}
    for component in components:
        path = input_dir / f"generation_component_{component}.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        observed: set[tuple[str, int]] = set()
        for row in rows:
            outputs = row["outputs"]
            if set(outputs) != expected_outputs:
                raise ValueError(
                    f"{path} prompt {row['prompt_id']} lacks exact "
                    f"±{endpoint_strength:g} outputs"
                )
            prompt_id = str(row["prompt_id"])
            seed_index = int(row.get("sample_seed_index", 0))
            key = (prompt_id, seed_index)
            if key in observed:
                raise ValueError(f"Duplicate prompt×seed observation: {key}")
            observed.add(key)
            for strength in (negative_key, positive_key):
                records.append(
                    {
                        "component": component,
                        "prompt_id": prompt_id,
                        "prompt_index": int(row["prompt_index"]),
                        "sample_seed_index": seed_index,
                        "strength": float(strength),
                        "text": str(outputs[strength]["text"]),
                    }
                )
        observation_sets[component] = observed
    if len({frozenset(values) for values in observation_sets.values()}) != 1:
        raise ValueError(
            f"Component prompt×seed sets differ: {observation_sets}"
        )
    return records


def prompt_cluster_summary(
    frame: pd.DataFrame,
    *,
    value_column: str,
    bootstrap: int,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    pivot = frame.pivot(
        index="prompt_id",
        columns="sample_seed_index",
        values=value_column,
    ).sort_index(axis=1)
    if pivot.isna().any().any():
        raise ValueError("Every prompt must contain every paired decode seed")
    prompt_means = pivot.mean(axis=1).to_numpy(dtype=float)
    inference = paired_mean_inference(
        prompt_means,
        bootstrap=bootstrap,
        permutations=permutations,
        seed=seed,
    )
    seed_means = pivot.mean(axis=0)
    if pivot.shape[1] == 1:
        within_prompt_sd = pd.Series(0.0, index=pivot.index)
        seed_mean_sd = 0.0
    else:
        within_prompt_sd = pivot.std(axis=1, ddof=1)
        seed_mean_sd = float(seed_means.std(ddof=1))
    return {
        "n_prompts": int(len(pivot)),
        "n_seeds": int(pivot.shape[1]),
        "n_prompt_seed_observations": int(pivot.size),
        **inference,
        "seed_mean_min": float(seed_means.min()),
        "seed_mean_max": float(seed_means.max()),
        "seed_mean_sd": seed_mean_sd,
        "mean_within_prompt_seed_sd": float(within_prompt_sd.mean()),
        "seed_means": json.dumps(
            {
                str(int(index)): float(value)
                for index, value in seed_means.items()
            },
            sort_keys=True,
        ),
        "seed_means_all_positive": bool((seed_means > 0).all()),
        "seed_means_all_negative": bool((seed_means < 0).all()),
    }


def main() -> None:
    args = parse_args()
    if args.endpoint_strength <= 0:
        raise ValueError("--endpoint-strength must be positive")
    started = datetime.now(timezone.utc)
    records = load_generation_records(
        args.input_dir,
        args.components,
        args.endpoint_strength,
    )
    nlp = spacy.load(args.spacy_model)
    extractor = EnglishLinguisticFeatureExtractor()
    documents = nlp.pipe(
        [row["text"] for row in records],
        batch_size=args.batch_size,
        n_process=args.n_process,
    )
    feature_rows = []
    for row, document in tqdm(
        zip(records, documents),
        total=len(records),
        desc="seed-robustness features",
    ):
        feature_rows.append(
            {
                **{key: value for key, value in row.items() if key != "text"},
                **extractor.extract(document, row["text"]),
            }
        )
    features = pd.DataFrame(feature_rows)
    feature_names = [item.name for item in FEATURE_DEFINITIONS]
    difference_rows = []
    for key, subset in features.groupby(
        ["component", "prompt_id", "sample_seed_index"],
        sort=False,
    ):
        component, prompt_id, seed_index = key
        negative = subset[
            subset["strength"] == -args.endpoint_strength
        ]
        positive = subset[
            subset["strength"] == args.endpoint_strength
        ]
        if len(negative) != 1 or len(positive) != 1:
            raise ValueError(f"Non-paired steering observation: {key}")
        negative_row = negative.iloc[0]
        positive_row = positive.iloc[0]
        difference_rows.append(
            {
                "component": component,
                "prompt_id": prompt_id,
                "prompt_index": int(positive_row["prompt_index"]),
                "sample_seed_index": int(seed_index),
                **{
                    name: float(positive_row[name] - negative_row[name])
                    for name in feature_names
                },
            }
        )
    differences = pd.DataFrame(difference_rows)
    observational = pd.read_csv(args.observational_features)
    names_by_group = {
        group: [
            item.name for item in FEATURE_DEFINITIONS if item.group == group
        ]
        for group in GROUPS
    }

    projection_rows = []
    for component in args.components:
        component_rows = differences[
            differences["component"] == component
        ]
        for group in GROUPS:
            names = names_by_group[group]
            projections, _ = source_signature_projection(
                observational[names].to_numpy(),
                component_rows[names].to_numpy(),
            )
            for (_, row), value in zip(
                component_rows.iterrows(),
                projections,
            ):
                projection_rows.append(
                    {
                        "component": component,
                        "prompt_id": row["prompt_id"],
                        "prompt_index": int(row["prompt_index"]),
                        "sample_seed_index": int(
                            row["sample_seed_index"]
                        ),
                        "group": group,
                        "signature_projection": float(value),
                    }
                )
    projections = pd.DataFrame(projection_rows)

    response_estimand_rows = []
    response_summary_rows = []
    for component_number, component in enumerate(args.components):
        component_rows = projections[
            projections["component"] == component
        ]
        for group_number, group in enumerate(GROUPS):
            current = component_rows[
                component_rows["group"] == group
            ][
                [
                    "prompt_id",
                    "prompt_index",
                    "sample_seed_index",
                    "signature_projection",
                ]
            ].copy()
            current = current.rename(
                columns={"signature_projection": "value"}
            )
            current["model"] = args.model_id
            current["direction_family"] = args.direction_family
            current["component"] = component
            current["response_group"] = group
            response_estimand_rows.append(current)
            response_summary_rows.append(
                {
                    "model": args.model_id,
                    "direction_family": args.direction_family,
                    "component": component,
                    "response_group": group,
                    **prompt_cluster_summary(
                        current,
                        value_column="value",
                        bootstrap=args.bootstrap,
                        permutations=args.permutations,
                        seed=(
                            args.seed
                            + 1000
                            + 100 * component_number
                            + group_number
                        ),
                    ),
                }
            )
    response_estimands = pd.concat(
        response_estimand_rows,
        ignore_index=True,
    )
    response_summary = pd.DataFrame(response_summary_rows)

    estimand_rows = []
    summary_rows = []
    pairwise_rows = []
    pairwise_summary_rows = []
    for component_number, component in enumerate(args.components):
        if component not in GROUPS:
            continue
        view = projections[
            projections["component"] == component
        ].pivot(
            index=["prompt_id", "prompt_index", "sample_seed_index"],
            columns="group",
            values="signature_projection",
        ).reset_index()
        off_target = [group for group in GROUPS if group != component]
        estimands = {
            "target_projection": view[component],
            "component_selectivity": (
                view[component] - view[off_target].mean(axis=1)
            ),
        }
        for estimand_number, (estimand, values) in enumerate(
            estimands.items()
        ):
            current = view[
                ["prompt_id", "prompt_index", "sample_seed_index"]
            ].copy()
            current["model"] = args.model_id
            current["direction_family"] = args.direction_family
            current["component"] = component
            current["estimand"] = estimand
            current["value"] = values.to_numpy(dtype=float)
            estimand_rows.append(current)
            summary_rows.append(
                {
                    "model": args.model_id,
                    "direction_family": args.direction_family,
                    "component": component,
                    "estimand": estimand,
                    **prompt_cluster_summary(
                        current,
                        value_column="value",
                        bootstrap=args.bootstrap,
                        permutations=args.permutations,
                        seed=(
                            args.seed
                            + 100 * component_number
                            + estimand_number
                        ),
                    ),
                }
            )
        for off_target_number, comparison_group in enumerate(off_target):
            current = view[
                ["prompt_id", "prompt_index", "sample_seed_index"]
            ].copy()
            current["model"] = args.model_id
            current["direction_family"] = args.direction_family
            current["component"] = component
            current["comparison_group"] = comparison_group
            current["value"] = (
                view[component] - view[comparison_group]
            ).to_numpy(dtype=float)
            pairwise_rows.append(current)
            pairwise_summary_rows.append(
                {
                    "model": args.model_id,
                    "direction_family": args.direction_family,
                    "component": component,
                    "comparison_group": comparison_group,
                    **prompt_cluster_summary(
                        current,
                        value_column="value",
                        bootstrap=args.bootstrap,
                        permutations=args.permutations,
                        seed=(
                            args.seed
                            + 500
                            + 100 * component_number
                            + off_target_number
                        ),
                    ),
                }
            )
    estimand_frame = pd.concat(estimand_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    pairwise_frame = pd.concat(pairwise_rows, ignore_index=True)
    pairwise_summary = pd.DataFrame(pairwise_summary_rows)

    raw_feature_rows = []
    raw_feature_summary_rows = []
    for component_number, component in enumerate(args.components):
        if component not in GROUPS:
            continue
        names = names_by_group[component]
        observational_values = observational[names].to_numpy(dtype=float)
        scales = observational_values.std(axis=0, ddof=1)
        standardized_shift = np.divide(
            observational_values.mean(axis=0),
            scales,
            out=np.zeros_like(scales),
            where=scales > 1e-12,
        )
        top_indices = np.argsort(np.abs(standardized_shift))[::-1][:3]
        component_rows = differences[
            differences["component"] == component
        ]
        for feature_rank, feature_index in enumerate(top_indices, start=1):
            feature = names[int(feature_index)]
            current = component_rows[
                [
                    "prompt_id",
                    "prompt_index",
                    "sample_seed_index",
                    feature,
                ]
            ].copy()
            current = current.rename(columns={feature: "value"})
            current["model"] = args.model_id
            current["direction_family"] = args.direction_family
            current["component"] = component
            current["feature"] = feature
            current["feature_rank"] = feature_rank
            raw_feature_rows.append(current)
            raw_feature_summary_rows.append(
                {
                    "model": args.model_id,
                    "direction_family": args.direction_family,
                    "component": component,
                    "feature": feature,
                    "feature_rank": feature_rank,
                    "observational_paired_d": float(
                        standardized_shift[int(feature_index)]
                    ),
                    **prompt_cluster_summary(
                        current,
                        value_column="value",
                        bootstrap=args.bootstrap,
                        permutations=args.permutations,
                        seed=(
                            args.seed
                            + 800
                            + 100 * component_number
                            + feature_rank
                        ),
                    ),
                }
            )
    raw_feature_frame = pd.concat(raw_feature_rows, ignore_index=True)
    raw_feature_summary = pd.DataFrame(raw_feature_summary_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.out_dir / "text_features.csv", index=False)
    differences.to_csv(
        args.out_dir / "paired_component_differences.csv",
        index=False,
    )
    projections.to_csv(
        args.out_dir / "prompt_seed_signature_projections.csv",
        index=False,
    )
    response_estimands.to_csv(
        args.out_dir / "prompt_seed_group_responses.csv",
        index=False,
    )
    response_summary.to_csv(
        args.out_dir / "group_response_summary.csv",
        index=False,
    )
    estimand_frame.to_csv(
        args.out_dir / "prompt_seed_estimands.csv",
        index=False,
    )
    summary.to_csv(args.out_dir / "seed_robustness_summary.csv", index=False)
    pairwise_frame.to_csv(
        args.out_dir / "prompt_seed_pairwise_selectivity.csv",
        index=False,
    )
    pairwise_summary.to_csv(
        args.out_dir / "pairwise_selectivity_summary.csv",
        index=False,
    )
    raw_feature_frame.to_csv(
        args.out_dir / "prompt_seed_raw_feature_effects.csv",
        index=False,
    )
    raw_feature_summary.to_csv(
        args.out_dir / "raw_feature_effect_summary.csv",
        index=False,
    )

    lines = [
        "# Decode-seed robustness",
        "",
        "Intervals resample prompt-level effects after averaging the paired "
        "decode seeds within each prompt.",
        "",
        "| Component | Estimand | Prompt-mean [95% CI] | Seed means | "
        "Within-prompt seed SD |",
        "|---|---|---:|---:|---:|",
    ]
    for row in summary.itertuples():
        seed_values = json.loads(row.seed_means)
        seed_text = ", ".join(
            f"{key}: {value:.3f}"
            for key, value in seed_values.items()
        )
        lines.append(
            f"| {row.component} | {row.estimand} | "
            f"{row.mean:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}] | "
            f"{seed_text} | {row.mean_within_prompt_seed_sd:.3f} |"
        )
    lines.extend([
        "",
        "## Full surface-response matrix",
        "",
        "| Direction | Response group | Prompt-mean [95% CI] | "
        "Seed means |",
        "|---|---|---:|---:|",
    ])
    for row in response_summary.itertuples():
        seed_values = json.loads(row.seed_means)
        seed_text = ", ".join(
            f"{key}: {value:.3f}"
            for key, value in seed_values.items()
        )
        lines.append(
            f"| {row.component} | {row.response_group} | "
            f"{row.mean:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}] | "
            f"{seed_text} |"
        )
    lines.extend([
        "",
        "## Pairwise selectivity",
        "",
        "| Direction | Target minus comparison | "
        "Prompt-mean [95% CI] |",
        "|---|---|---:|",
    ])
    for row in pairwise_summary.itertuples():
        lines.append(
            f"| {row.component} | {row.comparison_group} | "
            f"{row.mean:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}] |"
        )
    lines.extend([
        "",
        "## Raw effects for the three largest source features",
        "",
        f"| Direction | Feature | Raw +{args.endpoint_strength:g} minus "
        f"-{args.endpoint_strength:g} [95% CI] |",
        "|---|---|---:|",
    ])
    for row in raw_feature_summary.itertuples():
        lines.append(
            f"| {row.component} | {row.feature} | "
            f"{row.mean:.4f} [{row.ci_low:.4f}, {row.ci_high:.4f}] |"
        )
    (args.out_dir / "analysis_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    finished = datetime.now(timezone.utc)
    atomic_json(
        args.out_dir / "manifest.json",
        {
            "schema": "paired_decode_seed_robustness_v1",
            "status": "complete",
            "model_id": args.model_id,
            "direction_family": args.direction_family,
            "input_dir": str(args.input_dir),
            "observational_features": str(args.observational_features),
            "observational_features_sha256": sha256_file(
                args.observational_features
            ),
            "components": args.components,
            "endpoint_strength": args.endpoint_strength,
            "inference_unit": "prompt mean across paired decode seeds",
            "bootstrap": args.bootstrap,
            "permutations": args.permutations,
            "seed": args.seed,
            "python_version": platform.python_version(),
            "spacy_version": spacy.__version__,
            "started_at_utc": started.isoformat(),
            "finished_at_utc": finished.isoformat(),
            "duration_seconds": (finished - started).total_seconds(),
        },
    )
    print(f"Wrote {args.out_dir / 'analysis_summary.md'}")


if __name__ == "__main__":
    main()
