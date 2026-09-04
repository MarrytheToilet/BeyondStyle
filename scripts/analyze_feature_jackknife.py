#!/usr/bin/env python3
"""Leave-one-feature-out sensitivity audit for the surface signatures."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
FEATURE_ROOT = ROOT / "outputs/en/05_linguistic/features"
PAIRED_PATH = FEATURE_ROOT / "paired_feature_differences.csv"
PREFIX_PATH = ROOT / "outputs/en/05_linguistic/prefix_only_control/features/paired_feature_differences.csv"
DICTIONARY_PATH = FEATURE_ROOT / "feature_dictionary.json"
OUT_DIR = FEATURE_ROOT / "jackknife_sensitivity"
GROUPS = ("lexical", "syntactic", "discourse")
EPS = 1e-12


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
        raise ValueError("Degenerate feature in jackknife fit")
    return unit((raw / scales).mean(axis=0)), scales


def scores(frame: pd.DataFrame, features: list[str], axis: np.ndarray, scales: np.ndarray) -> np.ndarray:
    return (frame[features].to_numpy(dtype=np.float64) / scales) @ axis


def minimum_holdout(frame: pd.DataFrame, features: list[str], column: str) -> tuple[float, float]:
    axes = []
    projections = []
    for environment in sorted(frame[column].astype(str).unique()):
        train = frame[frame[column].astype(str) != environment]
        test = frame[frame[column].astype(str) == environment]
        axis, scales = fit_axis(train, features)
        axes.append(axis)
        projections.append(float(scores(test, features, axis, scales).mean()))
    cosines = [
        float(axes[left] @ axes[right])
        for left in range(len(axes))
        for right in range(left + 1, len(axes))
    ]
    return min(cosines), min(projections)


def safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) <= EPS or np.std(right) <= EPS:
        raise ValueError("Degenerate score vector")
    return float(np.corrcoef(left, right)[0, 1])


def main() -> None:
    started = datetime.now(timezone.utc)
    definitions = json.loads(DICTIONARY_PATH.read_text())
    paired = pd.read_csv(PAIRED_PATH)
    prefix = pd.read_csv(PREFIX_PATH)
    features_by_group = {
        group: [item["name"] for item in definitions if item["group"] == group]
        for group in GROUPS
    }
    rows = []
    for group, features in features_by_group.items():
        full_axis, full_scales = fit_axis(paired, features)
        full_paired_scores = scores(paired, features, full_axis, full_scales)
        full_prefix_scores = scores(prefix, features, full_axis, full_scales)
        for omitted in features:
            retained = [feature for feature in features if feature != omitted]
            axis, scales = fit_axis(paired, retained)
            padded = np.zeros(len(features), dtype=np.float64)
            retained_indices = [features.index(feature) for feature in retained]
            padded[retained_indices] = axis
            generator_cosine, generator_projection = minimum_holdout(
                paired, retained, "generator"
            )
            domain_cosine, domain_projection = minimum_holdout(paired, retained, "domain")
            paired_scores = scores(paired, retained, axis, scales)
            prefix_scores = scores(prefix, retained, axis, scales)
            rows.append(
                {
                    "group": group,
                    "omitted_feature": omitted,
                    "n_features_retained": len(retained),
                    "axis_cosine_with_full": float(padded @ full_axis),
                    "paired_score_correlation": safe_correlation(full_paired_scores, paired_scores),
                    "prefix_score_correlation": safe_correlation(full_prefix_scores, prefix_scores),
                    "paired_projection_mean": float(paired_scores.mean()),
                    "prefix_projection_mean": float(prefix_scores.mean()),
                    "minimum_generator_axis_cosine": generator_cosine,
                    "minimum_generator_test_projection": generator_projection,
                    "minimum_domain_axis_cosine": domain_cosine,
                    "minimum_domain_test_projection": domain_projection,
                }
            )

    detail = pd.DataFrame(rows)
    summary_rows = []
    metrics = [
        "axis_cosine_with_full",
        "paired_score_correlation",
        "prefix_score_correlation",
        "paired_projection_mean",
        "prefix_projection_mean",
        "minimum_generator_axis_cosine",
        "minimum_generator_test_projection",
        "minimum_domain_axis_cosine",
        "minimum_domain_test_projection",
    ]
    for group, group_rows in detail.groupby("group", sort=False):
        record = {"group": group, "n_jackknife_refits": len(group_rows)}
        for metric in metrics:
            index = group_rows[metric].idxmin()
            record[f"minimum_{metric}"] = float(group_rows.loc[index, metric])
            record[f"worst_feature_for_{metric}"] = group_rows.loc[index, "omitted_feature"]
        summary_rows.append(record)
    summary = pd.DataFrame(summary_rows)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    detail_path = OUT_DIR / "leave_one_feature_out.csv"
    summary_path = OUT_DIR / "summary.csv"
    report_path = OUT_DIR / "analysis_summary.md"
    detail.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    lines = [
        "# Surface-signature leave-one-feature-out audit",
        "",
        "Each row refits the standardized machine-minus-human direction after removing one target feature.",
        "",
        "| Group | Refits | Min axis cosine | Min paired-score r | Min prefix-score r | Min generator projection | Min domain projection |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.group} | {row.n_jackknife_refits} | "
            f"{row.minimum_axis_cosine_with_full:.3f} | "
            f"{row.minimum_paired_score_correlation:.3f} | "
            f"{row.minimum_prefix_score_correlation:.3f} | "
            f"{row.minimum_minimum_generator_test_projection:.3f} | "
            f"{row.minimum_minimum_domain_test_projection:.3f} |"
        )
    report_path.write_text("\n".join(lines) + "\n")

    finished = datetime.now(timezone.utc)
    manifest = {
        "schema": "surface_signature_feature_jackknife_v1",
        "status": "complete",
        "inputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [PAIRED_PATH, PREFIX_PATH, DICTIONARY_PATH]
        },
        "outputs": {
            str(path.relative_to(ROOT)): sha256(path)
            for path in [detail_path, summary_path, report_path]
        },
        "started_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(report_path.read_text())


if __name__ == "__main__":
    main()
