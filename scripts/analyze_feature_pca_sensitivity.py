#!/usr/bin/env python3
"""Audit whether surface-signature results depend on correlated coordinates.

An orthogonal PCA rotation that retains every component is algebraically
identical to the original standardized score.  The informative sensitivity
test therefore truncates the centered paired-difference correlation geometry
at fixed explained-variance thresholds, reconstructs the machine-minus-human
mean direction in the retained subspace, and reruns generator/domain holdouts
and prefix-only transfer without refitting on the control corpus.
"""

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
PREFIX_PATH = (
    ROOT
    / "outputs/en/05_linguistic/prefix_only_control/features"
    / "paired_feature_differences.csv"
)
DICTIONARY_PATH = FEATURE_ROOT / "feature_dictionary.json"
OUT_DIR = FEATURE_ROOT / "pca_sensitivity"
GROUPS = ("lexical", "syntactic", "discourse")
THRESHOLDS = (0.80, 0.90, 0.95)
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


def fit_axis(
    frame: pd.DataFrame,
    features: list[str],
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    raw = frame[features].to_numpy(dtype=np.float64)
    scales = raw.std(axis=0, ddof=1)
    if np.any(scales <= EPS):
        bad = [name for name, scale in zip(features, scales) if scale <= EPS]
        raise ValueError(f"Degenerate features: {bad}")
    standardized = raw / scales
    mean = standardized.mean(axis=0)
    centered = standardized - mean
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    eigenvectors = eigenvectors[:, order]
    cumulative = np.cumsum(eigenvalues) / eigenvalues.sum()
    retained = int(np.searchsorted(cumulative, threshold) + 1)
    basis = eigenvectors[:, :retained]
    reconstructed_mean = basis @ (basis.T @ mean)
    return (
        unit(reconstructed_mean),
        scales,
        retained,
        float(cumulative[retained - 1]),
    )


def projection_summary(
    frame: pd.DataFrame,
    features: list[str],
    scales: np.ndarray,
    axis: np.ndarray,
    *,
    seed: int,
    bootstrap: int = 10_000,
) -> dict[str, float]:
    values = frame[features].to_numpy(dtype=np.float64) / scales
    projections = values @ axis
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(projections), size=(bootstrap, len(projections)))
    means = projections[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    effect = values.mean(axis=0)
    return {
        "projection_mean": float(projections.mean()),
        "projection_ci_low": float(low),
        "projection_ci_high": float(high),
        "positive_pair_rate": float(np.mean(projections > 0)),
        "effect_axis_cosine": float(
            effect @ axis / max(float(np.linalg.norm(effect)), EPS)
        ),
    }


def minimum_holdout(
    frame: pd.DataFrame,
    features: list[str],
    *,
    column: str,
    threshold: float,
) -> tuple[float, float]:
    axes: list[np.ndarray] = []
    projections: list[float] = []
    environments = sorted(frame[column].astype(str).unique())
    for environment in environments:
        train = frame[frame[column].astype(str) != environment]
        test = frame[frame[column].astype(str) == environment]
        axis, scales, _, _ = fit_axis(train, features, threshold)
        axes.append(axis)
        values = test[features].to_numpy(dtype=np.float64) / scales
        projections.append(float((values @ axis).mean()))
    pairwise_cosines = [
        float(axes[i] @ axes[j])
        for i in range(len(axes))
        for j in range(i + 1, len(axes))
    ]
    return min(pairwise_cosines), min(projections)


def main() -> None:
    started = datetime.now(timezone.utc)
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
    prefix = pd.read_csv(PREFIX_PATH)
    rows: list[dict[str, object]] = []

    for group_number, (group, features) in enumerate(
        features_by_group.items()
    ):
        full_raw = paired[features].to_numpy(dtype=np.float64)
        full_scales = full_raw.std(axis=0, ddof=1)
        original_axis = unit((full_raw / full_scales).mean(axis=0))
        for threshold_number, threshold in enumerate(THRESHOLDS):
            axis, scales, retained, explained = fit_axis(
                paired,
                features,
                threshold,
            )
            prefix_summary = projection_summary(
                prefix,
                features,
                scales,
                axis,
                seed=20260729 + 100 * group_number + threshold_number,
            )
            generator_cosine, generator_projection = minimum_holdout(
                paired,
                features,
                column="generator",
                threshold=threshold,
            )
            domain_cosine, domain_projection = minimum_holdout(
                paired,
                features,
                column="domain",
                threshold=threshold,
            )
            rows.append(
                {
                    "group": group,
                    "variance_threshold": threshold,
                    "n_features": len(features),
                    "n_components_retained": retained,
                    "variance_explained": explained,
                    "cosine_with_original_axis": float(axis @ original_axis),
                    "minimum_generator_axis_cosine": generator_cosine,
                    "minimum_generator_test_projection": generator_projection,
                    "minimum_domain_axis_cosine": domain_cosine,
                    "minimum_domain_test_projection": domain_projection,
                    **{
                        f"prefix_{key}": value
                        for key, value in prefix_summary.items()
                    },
                }
            )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "pca_sensitivity_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    finished = datetime.now(timezone.utc)
    manifest = {
        "schema": "surface_signature_pca_sensitivity_v1",
        "status": "complete",
        "note": (
            "A full orthogonal PCA rotation is score-invariant. This audit "
            "tests truncated centered correlation geometry at fixed variance "
            "thresholds."
        ),
        "thresholds": list(THRESHOLDS),
        "inputs": {
            str(PAIRED_PATH.relative_to(ROOT)): sha256(PAIRED_PATH),
            str(PREFIX_PATH.relative_to(ROOT)): sha256(PREFIX_PATH),
            str(DICTIONARY_PATH.relative_to(ROOT)): sha256(DICTIONARY_PATH),
        },
        "output": {
            str(summary_path.relative_to(ROOT)): sha256(summary_path),
        },
        "started_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
