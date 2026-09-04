#!/usr/bin/env python3
"""Enumerate two- and three-feature deletions of the lexical signature."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
FEATURE_ROOT = ROOT / "outputs/en/05_linguistic/features"
PAIRED_PATH = FEATURE_ROOT / "paired_feature_differences.csv"
PREFIX_PATH = ROOT / "outputs/en/05_linguistic/prefix_only_control/features/paired_feature_differences.csv"
DICTIONARY_PATH = FEATURE_ROOT / "feature_dictionary.json"
OUT_DIR = FEATURE_ROOT / "subset_sensitivity"
EPS = 1e-12
TARGETED = {
    "function_plus_mattr": {"function_word_rate", "mattr_50"},
    "two_largest_effects": {"function_word_rate", "pronoun_rate"},
    "three_largest_effects": {
        "function_word_rate", "pronoun_rate", "content_zipf_mean"
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= EPS:
        raise ValueError("Cannot normalize a zero direction")
    return vector / norm


def fit_axis(frame: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, np.ndarray]:
    raw = frame[features].to_numpy(dtype=np.float64)
    scales = raw.std(axis=0, ddof=1)
    if np.any(scales <= EPS):
        raise ValueError("Degenerate feature in subset fit")
    return unit((raw / scales).mean(axis=0)), scales


def score(
    frame: pd.DataFrame,
    features: list[str],
    axis: np.ndarray,
    scales: np.ndarray,
) -> np.ndarray:
    return (frame[features].to_numpy(dtype=np.float64) / scales) @ axis


def residualize_preserving_mean(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_nuisance: np.ndarray,
    test_nuisance: np.ndarray,
    ridge: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    nuisance_mean = train_nuisance.mean(axis=0, keepdims=True)
    train_design = train_nuisance - nuisance_mean
    test_design = test_nuisance - nuisance_mean
    gram = train_design.T @ train_design
    penalty = ridge * max(float(np.trace(gram)) / max(len(gram), 1), 1.0)
    coefficients = np.linalg.solve(
        gram + penalty * np.eye(gram.shape[0]),
        train_design.T @ train_features,
    )
    return (
        train_features - train_design @ coefficients,
        test_features - test_design @ coefficients,
    )


def minimum_holdout(
    frame: pd.DataFrame,
    features: list[str],
    nuisance_names: list[str],
    column: str,
) -> tuple[float, float]:
    raw = frame[features].to_numpy(dtype=np.float64)
    nuisance = frame[nuisance_names].to_numpy(dtype=np.float64)
    labels = frame[column].astype(str).to_numpy()
    axes: list[np.ndarray] = []
    projections: list[float] = []
    for environment in sorted(np.unique(labels)):
        train_mask = labels != environment
        test_mask = ~train_mask
        train, test = residualize_preserving_mean(
            raw[train_mask], raw[test_mask], nuisance[train_mask], nuisance[test_mask]
        )
        scales = train.std(axis=0, ddof=1)
        train = train / scales
        test = test / scales
        axis = unit(train.mean(axis=0))
        axes.append(axis)
        projections.append(float((test @ axis).mean()))
    cosines = [
        float(axes[left] @ axes[right])
        for left in range(len(axes))
        for right in range(left + 1, len(axes))
    ]
    return min(cosines), min(projections)


def main() -> None:
    started = datetime.now(timezone.utc)
    definitions = json.loads(DICTIONARY_PATH.read_text(encoding="utf-8"))
    lexical = [item["name"] for item in definitions if item["group"] == "lexical"]
    nuisance = [item["name"] for item in definitions if item["group"] == "nuisance"]
    paired = pd.read_csv(PAIRED_PATH)
    prefix = pd.read_csv(PREFIX_PATH)
    full_axis, full_scales = fit_axis(paired, lexical)
    full_paired_score = score(paired, lexical, full_axis, full_scales)
    full_prefix_score = score(prefix, lexical, full_axis, full_scales)

    rows: list[dict[str, object]] = []
    for deletion_size in (2, 3):
        for omitted in combinations(lexical, deletion_size):
            retained = [name for name in lexical if name not in omitted]
            axis, scales = fit_axis(paired, retained)
            padded_axis = np.zeros(len(lexical), dtype=np.float64)
            for reduced_index, feature in enumerate(retained):
                padded_axis[lexical.index(feature)] = axis[reduced_index]
            paired_score = score(paired, retained, axis, scales)
            prefix_score = score(prefix, retained, axis, scales)
            generator_cosine, generator_projection = minimum_holdout(
                paired, retained, nuisance, "generator"
            )
            domain_cosine, domain_projection = minimum_holdout(
                paired, retained, nuisance, "domain"
            )
            label = next(
                (name for name, target in TARGETED.items() if set(omitted) == target),
                "",
            )
            rows.append({
                "deletion_size": deletion_size,
                "omitted_features": ";".join(omitted),
                "targeted_case": label,
                "axis_cosine_with_full": float(padded_axis @ full_axis),
                "paired_score_correlation": float(np.corrcoef(full_paired_score, paired_score)[0, 1]),
                "prefix_score_correlation": float(np.corrcoef(full_prefix_score, prefix_score)[0, 1]),
                "paired_projection_mean": float(paired_score.mean()),
                "prefix_projection_mean": float(prefix_score.mean()),
                "minimum_generator_axis_cosine": generator_cosine,
                "minimum_generator_test_projection": generator_projection,
                "minimum_domain_axis_cosine": domain_cosine,
                "minimum_domain_test_projection": domain_projection,
            })

    detail = pd.DataFrame(rows).sort_values(["deletion_size", "omitted_features"])
    metrics = [
        "axis_cosine_with_full", "paired_score_correlation",
        "prefix_score_correlation", "paired_projection_mean",
        "prefix_projection_mean", "minimum_generator_axis_cosine",
        "minimum_generator_test_projection", "minimum_domain_axis_cosine",
        "minimum_domain_test_projection",
    ]
    summary_rows: list[dict[str, object]] = []
    for deletion_size, subset in detail.groupby("deletion_size"):
        record: dict[str, object] = {
            "deletion_size": int(deletion_size), "n_refits": int(len(subset))
        }
        for metric in metrics:
            index = subset[metric].idxmin()
            record[f"minimum_{metric}"] = float(subset.loc[index, metric])
            record[f"worst_omission_for_{metric}"] = subset.loc[index, "omitted_features"]
        summary_rows.append(record)
    summary = pd.DataFrame(summary_rows)
    targeted = detail[detail["targeted_case"] != ""].copy()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = OUT_DIR / "all_two_three_feature_deletions.csv"
    summary_path = OUT_DIR / "worst_case_summary.csv"
    targeted_path = OUT_DIR / "targeted_deletions.csv"
    report_path = OUT_DIR / "analysis_summary.md"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    targeted.to_csv(targeted_path, index=False)

    lines = [
        "# Lexical multi-feature deletion sensitivity", "",
        "All 45 two-feature and 120 three-feature deletions are enumerated.",
        "Generator and domain holdouts repeat train-only nuisance residualization.", "",
        "| Deleted | Refits | Min axis cosine | Min paired-score r | Min prefix-score r | Min generator cosine | Min domain cosine |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.deletion_size} | {row.n_refits} | {row.minimum_axis_cosine_with_full:.3f} | "
            f"{row.minimum_paired_score_correlation:.3f} | {row.minimum_prefix_score_correlation:.3f} | "
            f"{row.minimum_minimum_generator_axis_cosine:.3f} | {row.minimum_minimum_domain_axis_cosine:.3f} |"
        )
    lines.extend(["", "## Targeted deletions", "",
        "| Case | Omitted features | Axis cosine | Paired-score r | Prefix-score r | Generator cosine | Domain cosine |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for row in targeted.itertuples(index=False):
        lines.append(
            f"| {row.targeted_case} | {row.omitted_features} | {row.axis_cosine_with_full:.3f} | "
            f"{row.paired_score_correlation:.3f} | {row.prefix_score_correlation:.3f} | "
            f"{row.minimum_generator_axis_cosine:.3f} | {row.minimum_domain_axis_cosine:.3f} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    finished = datetime.now(timezone.utc)
    manifest = {
        "schema": "surface_signature_multi_feature_deletion_v1",
        "status": "complete", "scope": "lexical",
        "deletion_sizes": [2, 3], "n_refits": int(len(detail)),
        "holdout_preprocessing": "train-only nuisance residualization",
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in (PAIRED_PATH, PREFIX_PATH, DICTIONARY_PATH)},
        "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in (detail_path, summary_path, targeted_path, report_path)},
        "started_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
