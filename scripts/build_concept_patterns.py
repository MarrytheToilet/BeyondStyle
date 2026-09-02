#!/usr/bin/env python
"""Build cross-source linguistic activation patterns for causal steering.

The intervention directions are forward covariance patterns, not probe
weights.  Lexical and syntactic feature groups are oriented by their
standardized paired AI−Human surface shifts.  Human-side and AI-side patterns
must agree without post-hoc sign flipping before they are averaged.
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
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_projection_mapping import (  # noqa: E402
    load_direction,
    load_states,
    source_arrays,
)
from src.analysis.concept_patterns import (  # noqa: E402
    covariance_activation_pattern,
    cross_source_consensus,
    decompose_source_direction,
    pearson,
    principal_probe_pattern,
    signature_weights,
)
from src.analysis.projection_mapping import (  # noqa: E402
    fit_nuisance_residualizer,
    fit_within_class_basis,
    project_out_basis,
)
from src.linguistics import FEATURE_DEFINITIONS  # noqa: E402
from src.steering import MeanDiffSteering  # noqa: E402


DEFAULT_MODEL_SPECS = (
    "Llama-3.1-8B-Instruct:7",
    "Qwen3-4B:8",
)
SOURCES = ("human", "ai")
SUPPORTED_GROUPS = ("lexical", "syntactic", "discourse")


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
        default=Path("outputs/en/05_linguistic/concept_patterns"),
    )
    parser.add_argument(
        "--model-specs",
        nargs="+",
        default=list(DEFAULT_MODEL_SPECS),
        help="MODEL_ID:MEAN_POOL_LAYER entries.",
    )
    parser.add_argument("--dataset-tag", default="proposition_aligned")
    parser.add_argument(
        "--representation",
        choices=["last_token", "mean_pool"],
        default="mean_pool",
    )
    parser.add_argument(
        "--groups",
        nargs="+",
        choices=list(SUPPORTED_GROUPS),
        default=["lexical", "syntactic"],
    )
    parser.add_argument("--null-k", type=int, default=10)
    parser.add_argument(
        "--pattern-estimator",
        choices=["covariance", "pcr_probe"],
        default="covariance",
    )
    parser.add_argument("--probe-components", type=int, default=64)
    parser.add_argument(
        "--holdout",
        choices=["generator", "domain"],
        default="generator",
    )
    parser.add_argument(
        "--minimum-pattern-cosine",
        type=float,
        default=0.0,
        help="Minimum Human/AI pattern cosine for a deployable group.",
    )
    parser.add_argument(
        "--intervention-norm-ratio",
        type=float,
        default=0.05,
        help=(
            "Set the common ±1 intervention L2 norm to this fraction of "
            "the median hidden-state norm."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260725)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_model_spec(spec: str) -> tuple[str, int]:
    model, separator, layer = spec.rpartition(":")
    if not separator or not model:
        raise ValueError(f"Expected MODEL_ID:LAYER, got {spec!r}")
    return model, int(layer)


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


def inventory(
    groups: list[str],
) -> tuple[list[str], list[str], dict[str, np.ndarray]]:
    feature_names = [
        item.name
        for item in FEATURE_DEFINITIONS
        if item.group in SUPPORTED_GROUPS
    ]
    nuisance_names = [
        item.name
        for item in FEATURE_DEFINITIONS
        if item.group == "nuisance"
    ]
    positions = {
        name: index for index, name in enumerate(feature_names)
    }
    columns = {
        group: np.asarray(
            [
                positions[item.name]
                for item in FEATURE_DEFINITIONS
                if item.group == group
            ],
            dtype=int,
        )
        for group in groups
    }
    return feature_names, nuisance_names, columns


def _fit_views(
    hidden: dict[str, np.ndarray],
    features: dict[str, np.ndarray],
    nuisance: dict[str, np.ndarray],
    *,
    feature_nuisance: dict[str, np.ndarray] | None = None,
    train: np.ndarray,
    test: np.ndarray,
    basis: np.ndarray,
) -> dict[str, dict[str, Any]]:
    if feature_nuisance is None:
        feature_nuisance = nuisance
    views: dict[str, dict[str, Any]] = {}
    for source in SOURCES:
        null_hidden = project_out_basis(hidden[source], basis)
        hidden_residualizer = fit_nuisance_residualizer(
            null_hidden[train],
            nuisance[source][train],
        )
        feature_residualizer = fit_nuisance_residualizer(
            features[source][train],
            feature_nuisance[source][train],
        )
        train_hidden = hidden_residualizer.transform(
            null_hidden[train],
            nuisance[source][train],
        )
        test_hidden = hidden_residualizer.transform(
            null_hidden[test],
            nuisance[source][test],
        )
        train_features = feature_residualizer.transform(
            features[source][train],
            feature_nuisance[source][train],
        )
        test_features = feature_residualizer.transform(
            features[source][test],
            feature_nuisance[source][test],
        )
        feature_scale = train_features.std(axis=0, ddof=1)
        feature_scale[feature_scale <= 1e-12] = 1.0
        views[source] = {
            "train_hidden": train_hidden,
            "test_hidden": test_hidden,
            "train_features": train_features / feature_scale,
            "test_features": test_features / feature_scale,
        }
    return views


def fit_patterns(
    hidden: dict[str, np.ndarray],
    features: dict[str, np.ndarray],
    nuisance: dict[str, np.ndarray],
    columns: dict[str, np.ndarray],
    *,
    feature_nuisance: dict[str, np.ndarray] | None = None,
    train: np.ndarray,
    test: np.ndarray,
    null_k: int,
    pattern_estimator: str,
    probe_components: int,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, dict[str, float]],
    list[dict[str, Any]],
]:
    basis, basis_info = fit_within_class_basis(
        hidden["human"][train],
        hidden["ai"][train],
        k=null_k,
    )
    views = _fit_views(
        hidden,
        features,
        nuisance,
        feature_nuisance=feature_nuisance,
        train=train,
        test=test,
        basis=basis,
    )
    patterns: dict[str, np.ndarray] = {}
    diagnostics: dict[str, dict[str, float]] = {}
    validation_rows: list[dict[str, Any]] = []

    for group, group_columns in columns.items():
        weights, weight_info = signature_weights(
            features["human"][train][:, group_columns],
            features["ai"][train][:, group_columns],
        )
        source_patterns = {}
        source_pattern_info = {}
        for source in SOURCES:
            train_scores = (
                views[source]["train_features"][:, group_columns]
                @ weights
            )
            if pattern_estimator == "covariance":
                pattern_function = covariance_activation_pattern
                pattern_kwargs = {}
            else:
                pattern_function = principal_probe_pattern
                pattern_kwargs = {"components": probe_components}
            source_patterns[source], source_pattern_info[source] = (
                pattern_function(
                    views[source]["train_hidden"],
                    train_scores,
                    **pattern_kwargs,
                )
            )
        consensus, consensus_info = cross_source_consensus(
            source_patterns["human"],
            source_patterns["ai"],
        )
        patterns[group] = consensus
        diagnostics[group] = {
            **basis_info,
            **weight_info,
            **consensus_info,
            "human_fitted_score_correlation": source_pattern_info[
                "human"
            ]["fitted_score_correlation"],
            "ai_fitted_score_correlation": source_pattern_info[
                "ai"
            ]["fitted_score_correlation"],
        }
        for source in SOURCES:
            test_scores = (
                views[source]["test_features"][:, group_columns]
                @ weights
            )
            validation_rows.append({
                "group": group,
                "test_source": source,
                "n_train": len(train),
                "n_test": len(test),
                "heldout_score_correlation": pearson(
                    views[source]["test_hidden"] @ consensus,
                    test_scores,
                ),
                "human_ai_pattern_cosine": consensus_info[
                    "human_ai_pattern_cosine"
                ],
                "signature_shift_norm": weight_info[
                    "standardized_shift_norm"
                ],
            })
    return patterns, diagnostics, validation_rows


def save_checkpoint(
    path: Path,
    vector: np.ndarray,
    *,
    layer: int,
    projection_std: float,
    metadata: dict[str, Any],
) -> None:
    vector = np.asarray(vector, dtype=np.float32)
    vector /= max(float(np.linalg.norm(vector)), 1e-12)
    steering = MeanDiffSteering(hidden_size=len(vector), layer=layer)
    with torch.no_grad():
        steering.v_src.copy_(torch.from_numpy(vector))
        steering.is_fitted.fill_(True)
        steering.projection_std.fill_(max(float(projection_std), 1e-6))
    path.parent.mkdir(parents=True, exist_ok=True)
    steering.save(path, metadata=metadata)


def pooled_projection_scale(
    human: np.ndarray,
    ai: np.ndarray,
    vector: np.ndarray,
) -> float:
    """Pooled within-source projection scale for one unit direction."""
    vector = np.asarray(vector, dtype=np.float64)
    vector /= max(float(np.linalg.norm(vector)), 1e-12)
    human_scores = np.asarray(human, dtype=np.float64) @ vector
    ai_scores = np.asarray(ai, dtype=np.float64) @ vector
    denominator = max(len(human_scores) + len(ai_scores) - 2, 1)
    variance = (
        ((human_scores - human_scores.mean()) ** 2).sum()
        + ((ai_scores - ai_scores.mean()) ** 2).sum()
    ) / denominator
    return max(float(np.sqrt(variance)), 1e-6)


def main() -> None:
    args = parse_args()
    if len(args.groups) != len(set(args.groups)):
        raise ValueError("--groups must not contain duplicates")
    if not 0 < args.intervention_norm_ratio < 1:
        raise ValueError("--intervention-norm-ratio must lie in (0, 1)")
    started = datetime.now(timezone.utc)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(args.features)
    feature_names, nuisance_names, columns = inventory(args.groups)
    features = source_arrays(frame, feature_names)
    nuisance = source_arrays(frame, nuisance_names)
    ordered_ids = frame["sample_id"].astype(str).tolist()
    environments = frame[args.holdout].astype(str).to_numpy()
    hidden_indices = np.arange(len(frame))
    validation_parts = []
    decomposition_index = {}
    input_files = {str(args.features): sha256_file(args.features)}

    for model, layer in map(parse_model_spec, args.model_specs):
        states = state_path(
            args.states_root,
            model,
            args.dataset_tag,
            args.representation,
            layer,
        )
        source_checkpoint = direction_path(
            args.states_root,
            model,
            args.dataset_tag,
            args.representation,
            layer,
        )
        human, ai, state_metadata = load_states(states, ordered_ids)
        hidden = {"human": human, "ai": ai}
        fold_patterns: dict[tuple[str, str], np.ndarray] = {}
        model_rows = []
        for heldout in sorted(np.unique(environments)):
            train = hidden_indices[environments != heldout]
            test = hidden_indices[environments == heldout]
            patterns, diagnostics, rows = fit_patterns(
                hidden,
                features,
                nuisance,
                columns,
                train=train,
                test=test,
                null_k=args.null_k,
                pattern_estimator=args.pattern_estimator,
                probe_components=args.probe_components,
            )
            for group, pattern in patterns.items():
                fold_patterns[(str(heldout), group)] = pattern
            for row in rows:
                row.update({
                    "model": model,
                    "representation": args.representation,
                    "layer": layer,
                    "holdout_type": args.holdout,
                    "heldout_environment": heldout,
                    "train_pattern_cross_source_cosine": diagnostics[
                        row["group"]
                    ]["human_ai_pattern_cosine"],
                })
                model_rows.append(row)

        final_patterns, final_diagnostics, _ = fit_patterns(
            hidden,
            features,
            nuisance,
            columns,
            train=hidden_indices,
            test=hidden_indices,
            null_k=args.null_k,
            pattern_estimator=args.pattern_estimator,
            probe_components=args.probe_components,
        )
        for group, info in final_diagnostics.items():
            if (
                info["human_ai_pattern_cosine"]
                < args.minimum_pattern_cosine
            ):
                raise RuntimeError(
                    f"{model} {group} Human/AI pattern cosine "
                    f"{info['human_ai_pattern_cosine']:.3f} is below "
                    f"{args.minimum_pattern_cosine:.3f}"
                )
        for row in model_rows:
            row["fold_final_pattern_cosine"] = float(
                fold_patterns[
                    (str(row["heldout_environment"]), row["group"])
                ]
                @ final_patterns[row["group"]]
            )
        validation_parts.append(pd.DataFrame(model_rows))

        source = load_direction(source_checkpoint)
        directions, decomposition = decompose_source_direction(
            source,
            final_patterns,
            seed=args.seed,
        )
        pooled_hidden = np.concatenate([human, ai], axis=0)
        median_activation_norm = float(
            np.median(np.linalg.norm(pooled_hidden, axis=1))
        )
        shared_intervention_scale = (
            args.intervention_norm_ratio * median_activation_norm
        )
        natural_projection_stds = {
            component: pooled_projection_scale(
                human,
                ai,
                vector,
            )
            for component, vector in directions.items()
        }
        model_dir = (
            args.out_dir
            / model
            / args.representation
            / f"layer{layer:02d}"
        )
        metadata = {
            "method": (
                "cross_source_covariance_activation_pattern"
                if args.pattern_estimator == "covariance"
                else "cross_source_principal_components_probe"
            ),
            "model_id": model,
            "layer": layer,
            "representation": args.representation,
            "null_k": args.null_k,
            "pattern_estimator": args.pattern_estimator,
            "probe_components": args.probe_components,
            "groups": args.groups,
            "holdout_type": args.holdout,
            "feature_file": str(args.features),
            "feature_file_sha256": sha256_file(args.features),
            "source_checkpoint": str(source_checkpoint),
            "source_checkpoint_sha256": sha256_file(source_checkpoint),
            "position_mode": "decode_only",
            "prompt_prefill_untouched": True,
            "strength_calibration": (
                "fixed fraction of median mean-pool hidden-state norm"
            ),
            "intervention_norm_ratio": args.intervention_norm_ratio,
            "median_activation_norm": median_activation_norm,
            "shared_intervention_scale": shared_intervention_scale,
        }
        for component, vector in directions.items():
            save_checkpoint(
                model_dir / "directions" / f"{component}.pt",
                vector,
                layer=layer,
                projection_std=shared_intervention_scale,
                metadata={
                    **metadata,
                    "component": component,
                    "natural_component_projection_std": (
                        natural_projection_stds[component]
                    ),
                },
            )
        payload = {
            "schema": "cross_source_concept_pattern_decomposition",
            "model": model,
            "representation": args.representation,
            "layer": layer,
            "state_metadata": state_metadata,
            "groups": final_diagnostics,
            "source_decomposition": decomposition,
            "strength_calibration": (
                "fixed_fraction_of_median_activation_norm"
            ),
            "intervention_norm_ratio": args.intervention_norm_ratio,
            "median_activation_norm": median_activation_norm,
            "shared_intervention_scale": shared_intervention_scale,
            "natural_component_projection_stds": (
                natural_projection_stds
            ),
            "selection_rule": (
                "predeclared mean-pool layer; lexical and syntactic "
                "patterns require nonnegative Human/AI cosine"
            ),
        }
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "decomposition.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        decomposition_index[model] = payload
        input_files[str(states)] = sha256_file(states)
        input_files[str(source_checkpoint)] = sha256_file(
            source_checkpoint
        )

    validation = pd.concat(validation_parts, ignore_index=True)
    validation.to_csv(args.out_dir / "pattern_validation.csv", index=False)
    summary = (
        validation.groupby(
            ["model", "representation", "layer", "group", "test_source"],
            as_index=False,
        )
        .agg(
            mean_heldout_correlation=(
                "heldout_score_correlation",
                "mean",
            ),
            minimum_heldout_correlation=(
                "heldout_score_correlation",
                "min",
            ),
            mean_cross_source_pattern_cosine=(
                "train_pattern_cross_source_cosine",
                "mean",
            ),
            minimum_fold_final_pattern_cosine=(
                "fold_final_pattern_cosine",
                "min",
            ),
        )
    )
    summary.to_csv(args.out_dir / "pattern_summary.csv", index=False)

    lines = [
        "# Cross-source linguistic activation patterns",
        "",
        f"Estimator: {args.pattern_estimator}. Positive orientation follows "
        "the standardized "
        "paired AI−Human surface signature.",
        "",
        "| Model | Pool | Layer | Group | Test source | Mean held-out r | "
        "Minimum r | H/AI pattern cosine | Fold stability |",
        "|---|---|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.representation} | {row.layer} | "
            f"{row.group} | {row.test_source} | "
            f"{row.mean_heldout_correlation:.3f} | "
            f"{row.minimum_heldout_correlation:.3f} | "
            f"{row.mean_cross_source_pattern_cosine:.3f} | "
            f"{row.minimum_fold_final_pattern_cosine:.3f} |"
        )
    (args.out_dir / "analysis_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    finished = datetime.now(timezone.utc)
    manifest = {
        "schema": "cross_source_linguistic_activation_patterns",
        "status": "complete",
        "language": "en",
        "dataset_tag": args.dataset_tag,
        "representation": args.representation,
        "model_specs": args.model_specs,
        "groups": args.groups,
        "null_k": args.null_k,
        "pattern_estimator": args.pattern_estimator,
        "probe_components": args.probe_components,
        "holdout": args.holdout,
        "minimum_pattern_cosine": args.minimum_pattern_cosine,
        "intervention_norm_ratio": args.intervention_norm_ratio,
        "seed": args.seed,
        "input_sha256": input_files,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "torch_version": torch.__version__,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "command": " ".join(sys.argv),
        "decompositions": decomposition_index,
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote concept patterns to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
