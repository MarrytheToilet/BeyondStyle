#!/usr/bin/env python
"""Projection-first hidden/surface mapping with environment holdouts.

This runner restores the useful core of the earlier projection analysis while
using the current proposition-aligned data and nullspace-only paper protocol.
It deliberately separates:

1. full-sample layer trajectories (exploratory cartography), and
2. train-fitted nullspace directions with held-out mapping validation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import sklearn
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.projection_mapping import (  # noqa: E402
    fit_nullspace_source_direction,
    fit_ridge_projection_map,
    permute_within_environments,
    prepare_mapping_split,
    project_source_coordinate,
    projection_mapping_metrics,
    safe_pearson,
    select_ridge_alpha,
    unit_vector,
)
from src.linguistics import FEATURE_DEFINITIONS  # noqa: E402


DEFAULT_MODEL_SPECS = (
    "Llama-3.1-8B-Instruct:11:11",
    "Qwen3-4B:12:8",
)
REPRESENTATIONS = ("last_token", "mean_pool")
SOURCES = ("human", "ai")
ORIENTATIONS = (
    ("human", "human"),
    ("ai", "ai"),
    ("human", "ai"),
    ("ai", "human"),
)
GROUPS = ("lexical", "syntactic", "discourse")


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
        default=Path("outputs/en/05_linguistic/projection_mapping"),
    )
    parser.add_argument(
        "--model-specs",
        nargs="+",
        default=list(DEFAULT_MODEL_SPECS),
        help="MODEL_ID:LAST_TOKEN_LAYER:MEAN_POOL_LAYER entries.",
    )
    parser.add_argument("--dataset-tag", default="proposition_aligned")
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=list(REPRESENTATIONS),
        default=list(REPRESENTATIONS),
    )
    parser.add_argument(
        "--holdout-types",
        nargs="+",
        choices=["generator", "domain"],
        default=["generator", "domain"],
    )
    parser.add_argument("--null-k", type=int, default=10)
    parser.add_argument(
        "--ridge-alphas",
        nargs="+",
        type=float,
        default=[0.1, 1.0, 10.0, 100.0, 1000.0],
    )
    parser.add_argument("--permutations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--skip-trajectory",
        action="store_true",
        help="Skip full-sample all-layer projection cartography.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_model_spec(spec: str) -> tuple[str, dict[str, int]]:
    parts = spec.rsplit(":", 2)
    if len(parts) != 3 or not parts[0]:
        raise ValueError(
            "Expected MODEL_ID:LAST_TOKEN_LAYER:MEAN_POOL_LAYER, "
            f"got {spec!r}"
        )
    return parts[0], {
        "last_token": int(parts[1]),
        "mean_pool": int(parts[2]),
    }


def state_path(
    root: Path,
    model: str,
    dataset_tag: str,
    representation: str,
    layer: int,
) -> Path:
    return (
        root
        / model
        / dataset_tag
        / representation
        / "states"
        / f"paired_states_layer{layer:02d}.pt"
    )


def direction_path(
    root: Path,
    model: str,
    dataset_tag: str,
    representation: str,
    layer: int,
) -> Path:
    return (
        root
        / model
        / dataset_tag
        / representation
        / "models"
        / f"mean_diff_nullspace_layer{layer:02d}.pt"
    )


def discover_layers(
    root: Path,
    model: str,
    dataset_tag: str,
    representation: str,
) -> list[int]:
    directory = (
        root / model / dataset_tag / representation / "states"
    )
    layers = []
    for path in directory.glob("paired_states_layer*.pt"):
        match = re.search(r"layer(\d+)$", path.stem)
        if match and direction_path(
            root,
            model,
            dataset_tag,
            representation,
            int(match.group(1)),
        ).exists():
            layers.append(int(match.group(1)))
    return sorted(layers)


def load_states(
    path: Path,
    ordered_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    identifiers = payload.get("metadata", {}).get("sample_ids")
    if not identifiers or set(identifiers) != set(ordered_ids):
        raise ValueError(f"State IDs do not match feature IDs in {path}")
    positions = {
        identifier: position
        for position, identifier in enumerate(identifiers)
    }
    order = [positions[identifier] for identifier in ordered_ids]
    return (
        payload["human"][order].float().numpy(),
        payload["ai"][order].float().numpy(),
        payload.get("metadata", {}),
    )


def load_direction(path: Path) -> np.ndarray:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        value = payload["state_dict"]["v_src"]
    elif isinstance(payload, dict) and "v_src" in payload:
        value = payload["v_src"]
    else:
        value = payload
    return unit_vector(value.float().numpy())


def feature_inventory() -> tuple[
    list[str],
    list[str],
    dict[str, list[str]],
]:
    features = [
        item.name
        for item in FEATURE_DEFINITIONS
        if item.group in GROUPS
    ]
    nuisance = [
        item.name
        for item in FEATURE_DEFINITIONS
        if item.group == "nuisance"
    ]
    groups = {
        group: [
            item.name
            for item in FEATURE_DEFINITIONS
            if item.group == group
        ]
        for group in GROUPS
    }
    groups["all"] = list(features)
    return features, nuisance, groups


def source_arrays(
    frame: pd.DataFrame,
    names: list[str],
) -> dict[str, np.ndarray]:
    arrays = {
        source: frame[
            [f"{source}__{name}" for name in names]
        ].to_numpy(dtype=np.float64)
        for source in SOURCES
    }
    for source, values in arrays.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite {source} values in feature table")
    return arrays


def orientation_name(train_source: str, test_source: str) -> str:
    return f"{train_source}_to_{test_source}"


def full_layer_cartography(
    *,
    args: argparse.Namespace,
    frame: pd.DataFrame,
    specs: list[tuple[str, dict[str, int]]],
) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
    score_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    inputs: list[Path] = []
    ordered_ids = frame["sample_id"].astype(str).tolist()
    for model, selected_layers in specs:
        for representation in args.representations:
            layers = discover_layers(
                args.states_root,
                model,
                args.dataset_tag,
                representation,
            )
            for layer in layers:
                states_file = state_path(
                    args.states_root,
                    model,
                    args.dataset_tag,
                    representation,
                    layer,
                )
                direction_file = direction_path(
                    args.states_root,
                    model,
                    args.dataset_tag,
                    representation,
                    layer,
                )
                human, ai, _ = load_states(states_file, ordered_ids)
                direction = load_direction(direction_file)
                human_scores = project_source_coordinate(human, direction)
                ai_scores = project_source_coordinate(ai, direction)
                labels = np.concatenate(
                    [np.zeros(len(human_scores)), np.ones(len(ai_scores))]
                )
                scores = np.concatenate([human_scores, ai_scores])
                pooled = np.sqrt(
                    (
                        human_scores.var(ddof=1)
                        + ai_scores.var(ddof=1)
                    )
                    / 2.0
                )
                summary_rows.append(
                    {
                        "model": model,
                        "representation": representation,
                        "layer": layer,
                        "selected_layer": (
                            layer == selected_layers[representation]
                        ),
                        "human_mean": float(human_scores.mean()),
                        "human_std": float(human_scores.std(ddof=1)),
                        "ai_mean": float(ai_scores.mean()),
                        "ai_std": float(ai_scores.std(ddof=1)),
                        "mean_separation": float(
                            ai_scores.mean() - human_scores.mean()
                        ),
                        "cohens_d": float(
                            (ai_scores.mean() - human_scores.mean())
                            / max(pooled, 1e-12)
                        ),
                        "auc": float(roc_auc_score(labels, scores)),
                    }
                )
                for source, values in (
                    ("human", human_scores),
                    ("ai", ai_scores),
                ):
                    for index, value in enumerate(values):
                        score_rows.append(
                            {
                                "sample_id": ordered_ids[index],
                                "pair_id": frame.iloc[index]["pair_id"],
                                "domain": frame.iloc[index]["domain"],
                                "generator": frame.iloc[index]["generator"],
                                "source": source,
                                "model": model,
                                "representation": representation,
                                "layer": layer,
                                "selected_layer": (
                                    layer
                                    == selected_layers[representation]
                                ),
                                "projection": float(value),
                            }
                        )
                inputs.extend([states_file, direction_file])
    return (
        pd.DataFrame(score_rows),
        pd.DataFrame(summary_rows),
        inputs,
    )


def validate_mapping(
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
    group_columns: dict[str, np.ndarray],
    feature_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validation_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.seed)
    hidden = {"human": human, "ai": ai}

    for holdout_type in args.holdout_types:
        environments = frame[holdout_type].astype(str).to_numpy()
        for heldout in sorted(np.unique(environments)):
            train = np.flatnonzero(environments != heldout)
            test = np.flatnonzero(environments == heldout)
            direction, direction_info = fit_nullspace_source_direction(
                human[train],
                ai[train],
                k=args.null_k,
            )
            scores = {
                source: project_source_coordinate(values, direction)
                for source, values in hidden.items()
            }
            test_labels = np.concatenate(
                [np.zeros(len(test)), np.ones(len(test))]
            )
            test_scores = np.concatenate(
                [scores["human"][test], scores["ai"][test]]
            )
            direction_rows.append(
                {
                    "model": model,
                    "representation": representation,
                    "layer": layer,
                    "holdout_type": holdout_type,
                    "heldout_environment": heldout,
                    "n_train": len(train),
                    "n_test": len(test),
                    "heldout_source_auc": float(
                        roc_auc_score(test_labels, test_scores)
                    ),
                    **direction_info,
                }
            )

            for train_source, test_source in ORIENTATIONS:
                all_prepared = prepare_mapping_split(
                    feature_arrays[train_source][train],
                    feature_arrays[test_source][test],
                    scores[train_source][train],
                    scores[test_source][test],
                    nuisance_arrays[train_source][train],
                    nuisance_arrays[test_source][test],
                )
                train_environment = environments[train]

                # A single-feature map learns its sign on the training source;
                # transfer correlation tests whether that sign generalizes.
                for feature_index, feature in enumerate(feature_names):
                    train_x = all_prepared.train_features[:, feature_index]
                    test_x = all_prepared.test_features[:, feature_index]
                    coefficient = float(
                        train_x @ all_prepared.train_scores
                        / max(float(train_x @ train_x) + 1.0, 1e-12)
                    )
                    train_prediction = train_x * coefficient
                    test_prediction = test_x * coefficient
                    item = next(
                        definition
                        for definition in FEATURE_DEFINITIONS
                        if definition.name == feature
                    )
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
                            "feature": feature,
                            "group": item.group,
                            "coefficient": coefficient,
                            "train_pearson": safe_pearson(
                                all_prepared.train_scores,
                                train_prediction,
                            ),
                            "test_pearson": safe_pearson(
                                all_prepared.test_scores,
                                test_prediction,
                            ),
                            "raw_test_correlation": safe_pearson(
                                all_prepared.test_scores,
                                test_x,
                            ),
                            "sign_transfers": bool(
                                safe_pearson(
                                    all_prepared.train_scores,
                                    train_x,
                                )
                                * safe_pearson(
                                    all_prepared.test_scores,
                                    test_x,
                                )
                                > 0
                            ),
                        }
                    )

                for target, columns in group_columns.items():
                    selected_alpha, inner_losses = select_ridge_alpha(
                        feature_arrays[train_source][train][:, columns],
                        scores[train_source][train],
                        nuisance_arrays[train_source][train],
                        train_environment,
                        args.ridge_alphas,
                    )
                    prepared = prepare_mapping_split(
                        feature_arrays[train_source][train][:, columns],
                        feature_arrays[test_source][test][:, columns],
                        scores[train_source][train],
                        scores[test_source][test],
                        nuisance_arrays[train_source][train],
                        nuisance_arrays[test_source][test],
                    )
                    coefficients = fit_ridge_projection_map(
                        prepared.train_features,
                        prepared.train_scores,
                        alpha=selected_alpha,
                    )
                    prediction = prepared.test_features @ coefficients
                    metrics = projection_mapping_metrics(
                        prepared.test_scores,
                        prediction,
                    )

                    permutation_correlations = []
                    for _ in range(args.permutations):
                        permuted = permute_within_environments(
                            prepared.train_scores,
                            train_environment,
                            rng,
                        )
                        null_coefficients = fit_ridge_projection_map(
                            prepared.train_features,
                            permuted,
                            alpha=selected_alpha,
                        )
                        null_prediction = (
                            prepared.test_features @ null_coefficients
                        )
                        permutation_correlations.append(
                            abs(
                                safe_pearson(
                                    prepared.test_scores,
                                    null_prediction,
                                )
                            )
                        )
                    observed = abs(metrics["pearson"])
                    null_values = np.asarray(permutation_correlations)
                    validation_rows.append(
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
                            "selected_alpha": selected_alpha,
                            "inner_mse": inner_losses[selected_alpha],
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
    return (
        pd.DataFrame(validation_rows),
        pd.DataFrame(feature_rows),
        pd.DataFrame(direction_rows),
    )


def write_summary(
    output: Path,
    validation: pd.DataFrame,
    feature_mapping: pd.DataFrame,
    direction_validation: pd.DataFrame,
) -> None:
    cross = validation[
        validation["train_source"] != validation["test_source"]
    ].copy()
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
            pearson=("pearson", "mean"),
            spearman=("spearman", "mean"),
            geometry=("correlation_geometry", "mean"),
            sign_agreement=("sign_agreement", "mean"),
            permutation_q95=("permutation_abs_q95", "mean"),
            permutation_p=("permutation_p", "mean"),
        )
    )
    aggregate.to_csv(
        output / "cross_source_mapping_summary.csv",
        index=False,
    )

    cross_features = feature_mapping[
        feature_mapping["train_source"]
        != feature_mapping["test_source"]
    ].copy()
    feature_aggregate = (
        cross_features.groupby(
            ["model", "representation", "group", "feature"],
            as_index=False,
        )
        .agg(
            median_transfer_correlation=("test_pearson", "median"),
            mean_transfer_correlation=("test_pearson", "mean"),
            positive_fold_fraction=(
                "test_pearson",
                lambda values: float(np.mean(values > 0)),
            ),
            sign_transfer_fraction=("sign_transfers", "mean"),
        )
        .sort_values(
            ["model", "representation", "median_transfer_correlation"],
            ascending=[True, True, False],
        )
    )
    feature_aggregate.to_csv(
        output / "cross_source_feature_summary.csv",
        index=False,
    )

    lines = [
        "# Nullspace projection-first mapping",
        "",
        "The source coordinate is fitted on training pairs only. Surface "
        "maps are learned within one source and evaluated on held-out "
        "environments in the same or opposite source.",
        "",
        "## Cross-source group mapping",
        "",
        "| Model | Pool | Holdout | Orientation | Target | Pearson | "
        "Permutation |",
        "|---|---|---|---|---|---:|---:|",
    ]
    for row in aggregate.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.representation} | "
            f"{row.holdout_type} | {row.orientation} | {row.target} | "
            f"{row.pearson:.3f} | {row.permutation_q95:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Strongest cross-source feature mappings",
            "",
            "| Model | Pool | Group | Feature | Median r | Positive folds |",
            "|---|---|---|---|---:|---:|",
        ]
    )
    for (model, representation), subset in feature_aggregate.groupby(
        ["model", "representation"],
        sort=False,
    ):
        for row in subset.head(12).itertuples(index=False):
            lines.append(
                f"| {model} | {representation} | {row.group} | "
                f"{row.feature} | {row.median_transfer_correlation:.3f} | "
                f"{row.positive_fold_fraction:.2f} |"
            )
    lines.extend(
        [
            "",
            "## Direction validation",
            "",
            "| Model | Pool | Holdout | Mean held-out source AUC |",
            "|---|---|---|---:|",
        ]
    )
    auc_summary = (
        direction_validation.groupby(
            ["model", "representation", "holdout_type"],
            as_index=False,
        )["heldout_source_auc"]
        .mean()
    )
    for row in auc_summary.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.representation} | "
            f"{row.holdout_type} | {row.heldout_source_auc:.3f} |"
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
    if frame["sample_id"].duplicated().any():
        raise ValueError("Feature sample_id values must be unique")
    specs = [parse_model_spec(spec) for spec in args.model_specs]
    feature_names, nuisance_names, groups = feature_inventory()
    features = source_arrays(frame, feature_names)
    nuisance = source_arrays(frame, nuisance_names)
    feature_positions = {
        name: position
        for position, name in enumerate(feature_names)
    }
    group_columns = {
        group: np.asarray(
            [feature_positions[name] for name in names],
            dtype=int,
        )
        for group, names in groups.items()
    }

    input_paths: list[Path] = [args.features]
    if args.skip_trajectory:
        projection_scores = pd.DataFrame()
        layer_summary = pd.DataFrame()
    else:
        (
            projection_scores,
            layer_summary,
            trajectory_inputs,
        ) = full_layer_cartography(
            args=args,
            frame=frame,
            specs=specs,
        )
        projection_scores.to_csv(
            args.out_dir / "projection_scores.csv",
            index=False,
        )
        layer_summary.to_csv(
            args.out_dir / "layer_projection_summary.csv",
            index=False,
        )
        input_paths.extend(trajectory_inputs)

    validation_parts = []
    feature_parts = []
    direction_parts = []
    ordered_ids = frame["sample_id"].astype(str).tolist()
    for model, selected_layers in specs:
        for representation in args.representations:
            layer = selected_layers[representation]
            path = state_path(
                args.states_root,
                model,
                args.dataset_tag,
                representation,
                layer,
            )
            if not path.exists():
                raise FileNotFoundError(path)
            human, ai, _ = load_states(path, ordered_ids)
            print(
                f"[projection] {model} {representation} L{layer}: "
                f"{len(human)} pairs",
                flush=True,
            )
            (
                validation,
                feature_mapping,
                direction_validation,
            ) = validate_mapping(
                args=args,
                frame=frame,
                model=model,
                representation=representation,
                layer=layer,
                human=human,
                ai=ai,
                feature_arrays=features,
                nuisance_arrays=nuisance,
                group_columns=group_columns,
                feature_names=feature_names,
            )
            validation_parts.append(validation)
            feature_parts.append(feature_mapping)
            direction_parts.append(direction_validation)
            input_paths.append(path)

    validation = pd.concat(validation_parts, ignore_index=True)
    feature_mapping = pd.concat(feature_parts, ignore_index=True)
    direction_validation = pd.concat(
        direction_parts,
        ignore_index=True,
    )
    validation.to_csv(
        args.out_dir / "mapping_validation.csv",
        index=False,
    )
    feature_mapping.to_csv(
        args.out_dir / "feature_mapping.csv",
        index=False,
    )
    direction_validation.to_csv(
        args.out_dir / "direction_validation.csv",
        index=False,
    )
    write_summary(
        args.out_dir,
        validation,
        feature_mapping,
        direction_validation,
    )

    finished = datetime.now(timezone.utc)
    unique_inputs = sorted(set(input_paths))
    manifest = {
        "schema": "nullspace_projection_first_mapping",
        "language": "en",
        "dataset_tag": args.dataset_tag,
        "model_specs": args.model_specs,
        "representations": args.representations,
        "holdout_types": args.holdout_types,
        "null_k": args.null_k,
        "ridge_alphas": args.ridge_alphas,
        "permutations": args.permutations,
        "seed": args.seed,
        "exploratory_layer_trajectory": not args.skip_trajectory,
        "formal_mapping_uses_train_fitted_direction": True,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "torch_version": torch.__version__,
        "input_sha256": {
            str(path): sha256_file(path)
            for path in unique_inputs
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
    print(f"[projection] wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
