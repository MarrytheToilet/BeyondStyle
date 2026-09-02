#!/usr/bin/env python
"""Cross-source linguistic concept readout from the nullspace subspace.

Unlike source-axis projection mapping, this analysis asks whether linguistic
features are linearly accessible anywhere in the hidden subspace after the
top within-class components have been removed.  Human-only readouts are tested
on AI text and AI-only readouts on human text, with outer generator/domain
holdouts and train-only nuisance preprocessing.
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
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
import sklearn
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_projection_mapping import (  # noqa: E402
    DEFAULT_MODEL_SPECS,
    GROUPS,
    ORIENTATIONS,
    feature_inventory,
    load_states,
    orientation_name,
    parse_model_spec,
    source_arrays,
    state_path,
)
from src.analysis.projection_mapping import (  # noqa: E402
    fit_nuisance_residualizer,
    fit_within_class_basis,
    permute_within_environments,
    project_out_basis,
    safe_pearson,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--states-root",
        type=Path,
        default=Path("outputs/en/03_steering_vectors"),
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path(
            "outputs/en/05_linguistic/features/"
            "paired_feature_differences.csv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "outputs/en/05_linguistic/projection_mapping/concept_readout"
        ),
    )
    parser.add_argument(
        "--model-specs",
        nargs="+",
        default=list(DEFAULT_MODEL_SPECS),
    )
    parser.add_argument("--dataset-tag", default="proposition_aligned")
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=["last_token", "mean_pool"],
        default=["last_token", "mean_pool"],
    )
    parser.add_argument(
        "--holdout-types",
        nargs="+",
        choices=["generator", "domain"],
        default=["generator", "domain"],
    )
    parser.add_argument("--null-k", type=int, default=10)
    parser.add_argument("--pca-components", type=int, default=64)
    parser.add_argument(
        "--ridge-alphas",
        nargs="+",
        type=float,
        default=[0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
    )
    parser.add_argument("--permutations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    """Linear centered-kernel alignment for two row-aligned matrices."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left = left - left.mean(axis=0, keepdims=True)
    right = right - right.mean(axis=0, keepdims=True)
    cross = float(np.sum((left.T @ right) ** 2))
    left_norm = float(np.sum((left.T @ left) ** 2))
    right_norm = float(np.sum((right.T @ right) ** 2))
    return cross / max(np.sqrt(left_norm * right_norm), 1e-12)


def select_alpha(
    latent: np.ndarray,
    targets: np.ndarray,
    environments: np.ndarray,
    alphas: list[float],
) -> tuple[float, dict[float, float]]:
    """Select readout regularization by leave-one-environment-out MSE."""
    losses = {float(alpha): [] for alpha in alphas}
    splitter = GroupKFold(
        n_splits=min(3, len(np.unique(environments)))
    )
    for train, test in splitter.split(latent, groups=environments):
        for alpha in losses:
            prediction = Ridge(
                alpha=alpha,
                fit_intercept=False,
            ).fit(latent[train], targets[train]).predict(latent[test])
            losses[alpha].append(
                float(np.mean((targets[test] - prediction) ** 2))
            )
    means = {
        alpha: float(np.mean(values))
        for alpha, values in losses.items()
    }
    best = min(means.values())
    selected = max(
        alpha for alpha, value in means.items() if value == best
    )
    return selected, means


def readout_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
) -> tuple[dict[str, float], np.ndarray]:
    """Aggregate per-feature transfer correlations and geometry."""
    correlations = np.asarray(
        [
            safe_pearson(actual[:, column], predicted[:, column])
            for column in range(actual.shape[1])
        ],
        dtype=np.float64,
    )
    return {
        "median_feature_correlation": float(np.median(correlations)),
        "mean_feature_correlation": float(np.mean(correlations)),
        "positive_feature_fraction": float(
            np.mean(correlations > 0)
        ),
        "linear_cka": linear_cka(actual, predicted),
    }, correlations


def validate_one_state(
    *,
    args: argparse.Namespace,
    frame: pd.DataFrame,
    model: str,
    representation: str,
    layer: int,
    human: np.ndarray,
    ai: np.ndarray,
    feature_arrays: dict[str, np.ndarray],
    nuisance_arrays: dict[str, np.ndarray],
    feature_names: list[str],
    group_columns: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    result_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    basis_rows: list[dict[str, Any]] = []
    hidden = {"human": human, "ai": ai}
    rng = np.random.default_rng(args.seed)

    for holdout_type in args.holdout_types:
        environments = frame[holdout_type].astype(str).to_numpy()
        for heldout in sorted(np.unique(environments)):
            train = np.flatnonzero(environments != heldout)
            test = np.flatnonzero(environments == heldout)
            basis, basis_info = fit_within_class_basis(
                human[train],
                ai[train],
                k=args.null_k,
            )
            null_hidden = {
                source: project_out_basis(values, basis)
                for source, values in hidden.items()
            }
            basis_rows.append(
                {
                    "model": model,
                    "representation": representation,
                    "layer": layer,
                    "holdout_type": holdout_type,
                    "heldout_environment": heldout,
                    "n_train": len(train),
                    "n_test": len(test),
                    **basis_info,
                }
            )

            for train_source in ("human", "ai"):
                hidden_residualizer = fit_nuisance_residualizer(
                    null_hidden[train_source][train],
                    nuisance_arrays[train_source][train],
                )
                train_hidden = hidden_residualizer.transform(
                    null_hidden[train_source][train],
                    nuisance_arrays[train_source][train],
                )
                components = min(
                    args.pca_components,
                    len(train) - 1,
                    train_hidden.shape[1],
                )
                pca = PCA(
                    n_components=components,
                    svd_solver="randomized",
                    random_state=args.seed,
                )
                train_latent = pca.fit_transform(train_hidden)
                latent_scale = train_latent.std(axis=0, ddof=1)
                latent_scale[latent_scale <= 1e-12] = 1.0
                train_latent = train_latent / latent_scale

                target_residualizer = fit_nuisance_residualizer(
                    feature_arrays[train_source][train],
                    nuisance_arrays[train_source][train],
                )
                train_targets = target_residualizer.transform(
                    feature_arrays[train_source][train],
                    nuisance_arrays[train_source][train],
                )
                target_scale = train_targets.std(axis=0, ddof=1)
                target_scale[target_scale <= 1e-12] = 1.0
                train_targets = train_targets / target_scale
                train_environments = environments[train]

                for test_source in ("human", "ai"):
                    test_hidden = hidden_residualizer.transform(
                        null_hidden[test_source][test],
                        nuisance_arrays[test_source][test],
                    )
                    test_latent = pca.transform(test_hidden) / latent_scale
                    test_targets = target_residualizer.transform(
                        feature_arrays[test_source][test],
                        nuisance_arrays[test_source][test],
                    ) / target_scale

                    for target, columns in group_columns.items():
                        selected, inner_losses = select_alpha(
                            train_latent,
                            train_targets[:, columns],
                            train_environments,
                            args.ridge_alphas,
                        )
                        readout = Ridge(
                            alpha=selected,
                            fit_intercept=False,
                        ).fit(
                            train_latent,
                            train_targets[:, columns],
                        )
                        prediction = readout.predict(test_latent)
                        metrics, correlations = readout_metrics(
                            test_targets[:, columns],
                            prediction,
                        )
                        null_statistics = []
                        for _ in range(args.permutations):
                            permuted = permute_within_environments(
                                train_targets[:, columns],
                                train_environments,
                                rng,
                            )
                            null_prediction = Ridge(
                                alpha=selected,
                                fit_intercept=False,
                            ).fit(
                                train_latent,
                                permuted,
                            ).predict(test_latent)
                            null_metrics, _ = readout_metrics(
                                test_targets[:, columns],
                                null_prediction,
                            )
                            null_statistics.append(
                                abs(
                                    null_metrics[
                                        "median_feature_correlation"
                                    ]
                                )
                            )
                        null_values = np.asarray(null_statistics)
                        observed = abs(
                            metrics["median_feature_correlation"]
                        )
                        result_rows.append(
                            {
                                "model": model,
                                "representation": representation,
                                "layer": layer,
                                "holdout_type": holdout_type,
                                "heldout_environment": heldout,
                                "orientation": orientation_name(
                                    train_source,
                                    test_source,
                                ),
                                "train_source": train_source,
                                "test_source": test_source,
                                "target": target,
                                "n_features": len(columns),
                                "n_train": len(train),
                                "n_test": len(test),
                                "pca_components": components,
                                "pca_variance": float(
                                    pca.explained_variance_ratio_.sum()
                                ),
                                "selected_alpha": selected,
                                "inner_mse": inner_losses[selected],
                                **metrics,
                                "permutation_abs_mean": float(
                                    null_values.mean()
                                ),
                                "permutation_abs_q95": float(
                                    np.quantile(null_values, 0.95)
                                ),
                                "permutation_p": float(
                                    (
                                        1
                                        + np.sum(null_values >= observed)
                                    )
                                    / (len(null_values) + 1)
                                ),
                            }
                        )
                        for local_index, column in enumerate(columns):
                            feature_rows.append(
                                {
                                    "model": model,
                                    "representation": representation,
                                    "layer": layer,
                                    "holdout_type": holdout_type,
                                    "heldout_environment": heldout,
                                    "orientation": orientation_name(
                                        train_source,
                                        test_source,
                                    ),
                                    "train_source": train_source,
                                    "test_source": test_source,
                                    "target": target,
                                    "feature": feature_names[column],
                                    "correlation": float(
                                        correlations[local_index]
                                    ),
                                }
                            )
    return (
        pd.DataFrame(result_rows),
        pd.DataFrame(feature_rows),
        pd.DataFrame(basis_rows),
    )


def write_summary(
    output: Path,
    results: pd.DataFrame,
    feature_results: pd.DataFrame,
) -> None:
    cross = results[
        results["train_source"] != results["test_source"]
    ]
    aggregate = (
        cross.groupby(
            [
                "model",
                "representation",
                "holdout_type",
                "orientation",
                "target",
            ],
            as_index=False,
        )
        .agg(
            median_r=("median_feature_correlation", "mean"),
            cka=("linear_cka", "mean"),
            positive_fraction=("positive_feature_fraction", "mean"),
            permutation_q95=("permutation_abs_q95", "mean"),
            permutation_p=("permutation_p", "mean"),
        )
    )
    aggregate.to_csv(
        output / "cross_source_readout_summary.csv",
        index=False,
    )
    # "all" contains every feature once; group-specific rows duplicate them.
    cross_features = feature_results[
        (feature_results["train_source"] != feature_results["test_source"])
        & (feature_results["target"] == "all")
    ]
    feature_summary = (
        cross_features.groupby(
            ["model", "representation", "feature"],
            as_index=False,
        )
        .agg(
            median_correlation=("correlation", "median"),
            mean_correlation=("correlation", "mean"),
            positive_fold_fraction=(
                "correlation",
                lambda values: float(np.mean(values > 0)),
            ),
        )
        .sort_values(
            ["model", "representation", "median_correlation"],
            ascending=[True, True, False],
        )
    )
    feature_summary.to_csv(
        output / "cross_source_feature_summary.csv",
        index=False,
    )

    lines = [
        "# Cross-source nullspace concept readout",
        "",
        "Readouts are trained within one source after train-only nuisance "
        "residualization and evaluated on held-out environments in the "
        "same or opposite source.",
        "",
        "## Group readout",
        "",
        "| Model | Pool | Holdout | Orientation | Target | Median r | "
        "CKA | Permutation |",
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for row in aggregate.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.representation} | "
            f"{row.holdout_type} | {row.orientation} | {row.target} | "
            f"{row.median_r:.3f} | {row.cka:.3f} | "
            f"{row.permutation_q95:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Strongest feature readouts",
            "",
            "| Model | Pool | Feature | Median r | Positive folds |",
            "|---|---|---|---:|---:|",
        ]
    )
    for (model, representation), subset in feature_summary.groupby(
        ["model", "representation"],
        sort=False,
    ):
        for row in subset.head(12).itertuples(index=False):
            lines.append(
                f"| {model} | {representation} | {row.feature} | "
                f"{row.median_correlation:.3f} | "
                f"{row.positive_fold_fraction:.2f} |"
            )
    (output / "analysis_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    started = datetime.now(timezone.utc)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.features)
    feature_names, nuisance_names, groups = feature_inventory()
    features = source_arrays(frame, feature_names)
    nuisance = source_arrays(frame, nuisance_names)
    positions = {
        name: index for index, name in enumerate(feature_names)
    }
    group_columns = {
        group: np.asarray(
            [positions[name] for name in names],
            dtype=int,
        )
        for group, names in groups.items()
    }
    ordered_ids = frame["sample_id"].astype(str).tolist()
    inputs = [args.features]
    result_parts = []
    feature_parts = []
    basis_parts = []

    for model, selected_layers in map(
        parse_model_spec,
        args.model_specs,
    ):
        for representation in args.representations:
            layer = selected_layers[representation]
            path = state_path(
                args.states_root,
                model,
                args.dataset_tag,
                representation,
                layer,
            )
            human, ai, _ = load_states(path, ordered_ids)
            print(
                f"[concept-readout] {model} {representation} L{layer}",
                flush=True,
            )
            results, feature_results, basis_results = validate_one_state(
                args=args,
                frame=frame,
                model=model,
                representation=representation,
                layer=layer,
                human=human,
                ai=ai,
                feature_arrays=features,
                nuisance_arrays=nuisance,
                feature_names=feature_names,
                group_columns=group_columns,
            )
            result_parts.append(results)
            feature_parts.append(feature_results)
            basis_parts.append(basis_results)
            inputs.append(path)

    results = pd.concat(result_parts, ignore_index=True)
    feature_results = pd.concat(feature_parts, ignore_index=True)
    basis_results = pd.concat(basis_parts, ignore_index=True)
    results.to_csv(
        args.out_dir / "readout_validation.csv",
        index=False,
    )
    feature_results.to_csv(
        args.out_dir / "feature_readout.csv",
        index=False,
    )
    basis_results.to_csv(
        args.out_dir / "basis_validation.csv",
        index=False,
    )
    write_summary(args.out_dir, results, feature_results)

    finished = datetime.now(timezone.utc)
    manifest = {
        "schema": "cross_source_nullspace_concept_readout",
        "language": "en",
        "dataset_tag": args.dataset_tag,
        "model_specs": args.model_specs,
        "representations": args.representations,
        "holdout_types": args.holdout_types,
        "null_k": args.null_k,
        "pca_components": args.pca_components,
        "ridge_alphas": args.ridge_alphas,
        "permutations": args.permutations,
        "seed": args.seed,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "torch_version": torch.__version__,
        "input_sha256": {
            str(path): sha256_file(path) for path in sorted(set(inputs))
        },
        "started_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "command": " ".join(sys.argv),
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[concept-readout] wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
