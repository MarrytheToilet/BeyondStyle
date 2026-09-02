#!/usr/bin/env python3
"""Audit redundancy in the frozen linguistic feature groups.

The confirmatory surface directions are fitted to paired machine-minus-human
differences.  This script therefore measures dependence at that same grain,
rather than correlating pooled human and machine texts.  It reports complete
within-group correlation matrices plus compact spectral summaries.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
FEATURE_ROOT = ROOT / "outputs/en/05_linguistic/features"
PAIRED_PATH = FEATURE_ROOT / "paired_feature_differences.csv"
DICTIONARY_PATH = FEATURE_ROOT / "feature_dictionary.json"
OUT_DIR = FEATURE_ROOT / "redundancy_audit"
GROUPS = ("lexical", "syntactic", "discourse")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def effective_rank(eigenvalues: np.ndarray) -> float:
    positive = np.clip(eigenvalues, 0.0, None)
    probabilities = positive / positive.sum()
    probabilities = probabilities[probabilities > 0]
    return float(np.exp(-(probabilities * np.log(probabilities)).sum()))


def participation_ratio(eigenvalues: np.ndarray) -> float:
    positive = np.clip(eigenvalues, 0.0, None)
    return float(positive.sum() ** 2 / np.square(positive).sum())


def main() -> None:
    definitions = json.loads(DICTIONARY_PATH.read_text())
    features_by_group = {
        group: [
            item["name"]
            for item in definitions
            if item["group"] == group
        ]
        for group in GROUPS
    }
    paired = pd.read_csv(PAIRED_PATH)
    summaries: list[dict[str, object]] = []
    long_rows: list[dict[str, object]] = []
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for group, features in features_by_group.items():
        values = paired[features].astype(float)
        correlation = values.corr(method="pearson")
        correlation.to_csv(OUT_DIR / f"{group}_paired_difference_correlation.csv")

        eigenvalues = np.linalg.eigvalsh(correlation.to_numpy())
        eigenvalues = np.sort(np.clip(eigenvalues, 0.0, None))[::-1]
        upper = np.triu_indices(len(features), k=1)
        absolute = np.abs(correlation.to_numpy()[upper])
        maximum_index = int(np.argmax(absolute))
        row_index = int(upper[0][maximum_index])
        column_index = int(upper[1][maximum_index])

        summaries.append({
            "group": group,
            "n_features": len(features),
            "n_pairs": len(values),
            "pc1_variance_fraction": eigenvalues[0] / eigenvalues.sum(),
            "effective_rank_entropy": effective_rank(eigenvalues),
            "effective_rank_participation": participation_ratio(eigenvalues),
            "max_abs_correlation": absolute[maximum_index],
            "max_abs_correlation_feature_1": features[row_index],
            "max_abs_correlation_feature_2": features[column_index],
            "feature_pairs_abs_corr_ge_0_70": int((absolute >= 0.70).sum()),
            "feature_pairs_abs_corr_ge_0_80": int((absolute >= 0.80).sum()),
            "total_feature_pairs": int(len(absolute)),
        })

        for row_feature in features:
            for column_feature in features:
                long_rows.append({
                    "group": group,
                    "feature_1": row_feature,
                    "feature_2": column_feature,
                    "correlation": correlation.loc[row_feature, column_feature],
                })

    summary = pd.DataFrame(summaries)
    long_frame = pd.DataFrame(long_rows)
    summary_path = OUT_DIR / "feature_redundancy_summary.csv"
    long_path = OUT_DIR / "feature_correlation_long.csv"
    summary.to_csv(summary_path, index=False)
    long_frame.to_csv(long_path, index=False)

    manifest = {
        "schema": "paired_feature_redundancy_audit",
        "grain": "500 paired machine-minus-human feature differences",
        "correlation": "Pearson within each frozen feature group",
        "spectral_metrics": {
            "pc1_variance_fraction": (
                "largest correlation-matrix eigenvalue / number of features"
            ),
            "effective_rank_entropy": (
                "exp(Shannon entropy of normalized nonnegative eigenvalues)"
            ),
            "effective_rank_participation": (
                "squared eigenvalue sum / squared-eigenvalue sum"
            ),
        },
        "inputs": {
            str(PAIRED_PATH.relative_to(ROOT)): sha256(PAIRED_PATH),
            str(DICTIONARY_PATH.relative_to(ROOT)): sha256(DICTIONARY_PATH),
        },
        "outputs": {
            str(summary_path.relative_to(ROOT)): sha256(summary_path),
            str(long_path.relative_to(ROOT)): sha256(long_path),
            **{
                str(
                    (OUT_DIR / f"{group}_paired_difference_correlation.csv")
                    .relative_to(ROOT)
                ): sha256(
                    OUT_DIR / f"{group}_paired_difference_correlation.csv"
                )
                for group in GROUPS
            },
        },
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
