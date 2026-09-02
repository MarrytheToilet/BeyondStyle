#!/usr/bin/env python
"""Cross-fit linguistic reconstruction of nullspace hidden source shifts."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.hidden_language import (  # noqa: E402
    fit_ridge_reconstruction,
    fit_within_class_nullspace_basis,
    linguistic_mean_direction,
    orthonormal_union,
    project_out_basis,
    reconstruction_metrics,
    residualize_preserving_mean,
    scale_features_preserving_mean,
    select_ridge_alpha,
)
from src.config import get_linguistic_dir  # noqa: E402
from src.linguistics import FEATURE_DEFINITIONS  # noqa: E402
from src.steering import MeanDiffSteering  # noqa: E402


DEFAULT_MODELS = (
    "Llama-3.1-8B-Instruct:11",
    "Qwen3-4B:12",
)
GROUPS = ("lexical", "syntactic", "discourse")
TARGETS = (*GROUPS, "all")


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
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--model-specs",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="MODEL_ID:LAYER entries.",
    )
    parser.add_argument("--dataset-tag", default="proposition_aligned")
    parser.add_argument("--representation", default="last_token")
    parser.add_argument("--null-k", type=int, default=10)
    parser.add_argument(
        "--ridge-alphas",
        nargs="+",
        type=float,
        default=[0.1, 1.0, 10.0, 100.0, 1000.0],
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=50,
        help="Train-pair permutations for aggregate held-out inference.",
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


def load_state(
    path: Path,
    ordered_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    state_ids = payload.get("metadata", {}).get("sample_ids")
    if not state_ids or set(state_ids) != set(ordered_ids):
        raise ValueError(f"State IDs do not match feature IDs in {path}")
    positions = {sample_id: index for index, sample_id in enumerate(state_ids)}
    order = [positions[sample_id] for sample_id in ordered_ids]
    return (
        payload["human"][order].float().numpy(),
        payload["ai"][order].float().numpy(),
        payload.get("metadata", {}),
    )


def group_feature_names() -> dict[str, list[str]]:
    return {
        group: [
            item.name for item in FEATURE_DEFINITIONS if item.group == group
        ]
        for group in GROUPS
    }


def nuisance_feature_names() -> list[str]:
    return [
        item.name for item in FEATURE_DEFINITIONS if item.group == "nuisance"
    ]


def prepare_split(
    human: np.ndarray,
    ai: np.ndarray,
    features: np.ndarray,
    nuisance: np.ndarray,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    null_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit every preprocessing operation on train and apply it to test."""
    basis = fit_within_class_nullspace_basis(
        human[train_indices],
        ai[train_indices],
        k=null_k,
    )
    train_hidden = project_out_basis(
        ai[train_indices] - human[train_indices],
        basis,
    )
    test_hidden = project_out_basis(
        ai[test_indices] - human[test_indices],
        basis,
    )
    train_hidden, test_hidden = residualize_preserving_mean(
        train_hidden,
        test_hidden,
        nuisance[train_indices],
        nuisance[test_indices],
    )
    train_features, test_features = residualize_preserving_mean(
        features[train_indices],
        features[test_indices],
        nuisance[train_indices],
        nuisance[test_indices],
    )
    train_features, test_features, _ = scale_features_preserving_mean(
        train_features,
        test_features,
    )
    return train_hidden, test_hidden, train_features, test_features


def nested_alpha_selection(
    human: np.ndarray,
    ai: np.ndarray,
    features: np.ndarray,
    nuisance: np.ndarray,
    generators: np.ndarray,
    outer_train: np.ndarray,
    feature_slices: dict[str, np.ndarray],
    *,
    null_k: int,
    alphas: list[float],
) -> tuple[dict[str, float], dict[str, dict[float, float]]]:
    """Select ridge strength without using the outer held-out generator."""
    scores = {
        target: {alpha: [] for alpha in alphas}
        for target in TARGETS
    }
    inner_labels = generators[outer_train]
    for heldout in sorted(np.unique(inner_labels)):
        train = outer_train[inner_labels != heldout]
        test = outer_train[inner_labels == heldout]
        h_train, h_test, f_train, f_test = prepare_split(
            human,
            ai,
            features,
            nuisance,
            train,
            test,
            null_k,
        )
        total = h_test.mean(axis=0)
        source_axis = h_train.mean(axis=0)
        for target in TARGETS:
            columns = feature_slices[target]
            for alpha in alphas:
                coefficients = fit_ridge_reconstruction(
                    h_train,
                    f_train[:, columns],
                    alpha,
                )
                linguistic = linguistic_mean_direction(
                    f_test[:, columns],
                    coefficients,
                )
                metrics = reconstruction_metrics(
                    total,
                    linguistic,
                    source_axis=source_axis,
                )
                scores[target][alpha].append(
                    metrics["vector_reconstruction_r2"]
                )
    selected: dict[str, float] = {}
    means: dict[str, dict[float, float]] = {}
    for target in TARGETS:
        selected[target], means[target] = select_ridge_alpha(scores[target])
    return selected, means


def _safe_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.std(ddof=1) <= 1e-12 or right.std(ddof=1) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def generator_holdout_validation(
    human: np.ndarray,
    ai: np.ndarray,
    features: np.ndarray,
    nuisance: np.ndarray,
    generators: np.ndarray,
    feature_slices: dict[str, np.ndarray],
    *,
    null_k: int,
    alphas: list[float],
    permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, float], dict[str, np.ndarray]]:
    """Nested leave-one-generator-out reconstruction evaluation."""
    rows: list[dict[str, Any]] = []
    selected_by_target = {target: [] for target in TARGETS}
    null_by_target = {
        target: np.empty(
            (len(np.unique(generators)), permutations),
            dtype=np.float64,
        )
        for target in TARGETS
    }
    out_of_fold_predictions = {
        target: np.empty_like(ai - human, dtype=np.float32)
        for target in (*TARGETS, "source_only")
    }
    rng = np.random.default_rng(seed)
    for fold, heldout in enumerate(sorted(np.unique(generators))):
        test = np.flatnonzero(generators == heldout)
        train = np.flatnonzero(generators != heldout)
        selected, inner_means = nested_alpha_selection(
            human,
            ai,
            features,
            nuisance,
            generators,
            train,
            feature_slices,
            null_k=null_k,
            alphas=alphas,
        )
        h_train, h_test, f_train, f_test = prepare_split(
            human,
            ai,
            features,
            nuisance,
            train,
            test,
            null_k,
        )
        source_axis = h_train.mean(axis=0)
        source_axis /= max(float(np.linalg.norm(source_axis)), 1e-12)
        total = h_test.mean(axis=0)
        axis_total = float(total @ source_axis)
        actual_pair_scores = h_test @ source_axis
        source_only_prediction = np.broadcast_to(
            h_train.mean(axis=0),
            h_test.shape,
        )
        source_only_sse = float(
            np.sum((h_test - source_only_prediction) ** 2)
        )
        for target in TARGETS:
            columns = feature_slices[target]
            alpha = selected[target]
            selected_by_target[target].append(alpha)
            coefficients = fit_ridge_reconstruction(
                h_train,
                f_train[:, columns],
                alpha,
            )
            linguistic = linguistic_mean_direction(
                f_test[:, columns],
                coefficients,
            )
            metrics = reconstruction_metrics(
                total,
                linguistic,
                source_axis=source_axis,
            )
            predicted_pair_scores = (
                f_test[:, columns] @ coefficients
            ) @ source_axis
            pair_predictions = f_test[:, columns] @ coefficients
            out_of_fold_predictions[target][test] = pair_predictions
            pair_sse = float(np.sum((h_test - pair_predictions) ** 2))
            rows.append(
                {
                    "heldout_generator": heldout,
                    "target": target,
                    "n_train": len(train),
                    "n_test": len(test),
                    "selected_alpha": alpha,
                    "inner_vector_r2": inner_means[target][alpha],
                    "pair_score_correlation": _safe_correlation(
                        actual_pair_scores,
                        predicted_pair_scores,
                    ),
                    "pair_incremental_r2_over_source_only": (
                        1.0 - pair_sse / max(source_only_sse, 1e-12)
                    ),
                    **metrics,
                }
            )

            # Random sign flips remove a consistent linguistic source shift.
            # Row permutations would preserve that mean and are therefore not
            # a valid null for source-signature reconstruction.
            train_axis_scores = h_train @ source_axis
            for permutation in range(permutations):
                signs = rng.choice(
                    np.array([-1.0, 1.0]),
                    size=(len(f_train), 1),
                )
                scalar_coefficients = fit_ridge_reconstruction(
                    train_axis_scores[:, None],
                    signs * f_train[:, columns],
                    alpha,
                )
                null_axis = float(
                    f_test[:, columns].mean(axis=0)
                    @ scalar_coefficients[:, 0]
                )
                null_by_target[target][fold, permutation] = (
                    null_axis / axis_total
                    if abs(axis_total) > 1e-12
                    else 0.0
                )
        source_only = h_train.mean(axis=0)
        out_of_fold_predictions["source_only"][test] = (
            source_only_prediction
        )
        rows.append(
            {
                "heldout_generator": heldout,
                "target": "source_only",
                "n_train": len(train),
                "n_test": len(test),
                "selected_alpha": np.nan,
                "inner_vector_r2": np.nan,
                "pair_score_correlation": 0.0,
                "pair_incremental_r2_over_source_only": 0.0,
                "aggregate_permutation_p": np.nan,
                **reconstruction_metrics(
                    total,
                    source_only,
                    source_axis=source_axis,
                ),
            }
        )

    table = pd.DataFrame(rows)
    for target in TARGETS:
        observed = float(
            table.loc[
                table["target"] == target,
                "source_axis_reconstructed_fraction",
            ].mean()
        )
        null_means = null_by_target[target].mean(axis=0)
        p_value = (1 + np.sum(null_means >= observed)) / (
            permutations + 1
        )
        table.loc[
            table["target"] == target,
            "aggregate_permutation_p",
        ] = float(p_value)

    final_alphas: dict[str, float] = {}
    for target, values in selected_by_target.items():
        counts = Counter(values)
        maximum = max(counts.values())
        final_alphas[target] = max(
            alpha for alpha, count in counts.items() if count == maximum
        )
    out_of_fold_directions = {
        target: predictions.mean(axis=0)
        for target, predictions in out_of_fold_predictions.items()
    }
    return table, final_alphas, out_of_fold_directions


def save_component_checkpoint(
    path: Path,
    vector: np.ndarray,
    human: np.ndarray,
    ai: np.ndarray,
    *,
    layer: int,
    metadata: dict[str, Any],
) -> None:
    vector = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError(f"Cannot save zero component direction: {path}")
    vector = vector / norm
    steering = MeanDiffSteering(
        hidden_size=vector.shape[0],
        layer=layer,
    )
    with torch.no_grad():
        steering.v_src.copy_(torch.from_numpy(vector))
        steering.is_fitted.fill_(True)
        human_scores = human @ vector
        ai_scores = ai @ vector
        pooled_variance = (
            ((human_scores - human_scores.mean()) ** 2).sum()
            + ((ai_scores - ai_scores.mean()) ** 2).sum()
        ) / max(len(human_scores) + len(ai_scores) - 2, 1)
        steering.projection_std.fill_(
            max(float(np.sqrt(pooled_variance)), 1e-6)
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    steering.save(path, metadata=metadata)


def construct_cross_fitted_components(
    human: np.ndarray,
    ai: np.ndarray,
    features: np.ndarray,
    nuisance: np.ndarray,
    feature_slices: dict[str, np.ndarray],
    selected_alphas: dict[str, float],
    out_of_fold_directions: dict[str, np.ndarray],
    checkpoint_source: np.ndarray,
    *,
    null_k: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    indices = np.arange(len(human))
    hidden, _, _, _ = prepare_split(
        human,
        ai,
        features,
        nuisance,
        indices,
        indices,
        null_k,
    )
    total = hidden.mean(axis=0)
    source = total / max(float(np.linalg.norm(total)), 1e-12)
    checkpoint = np.asarray(checkpoint_source, dtype=np.float32)
    checkpoint /= max(float(np.linalg.norm(checkpoint)), 1e-12)

    target_results: dict[str, Any] = {}
    for target in TARGETS:
        linguistic = out_of_fold_directions[target]
        target_results[target] = {
            "selected_alpha": selected_alphas[target],
            "n_features": int(len(feature_slices[target])),
            "linguistic_direction_norm": float(np.linalg.norm(linguistic)),
            **reconstruction_metrics(
                total,
                linguistic,
                source_axis=source,
            ),
        }
    source_only = out_of_fold_directions["source_only"]
    source_only_result = {
        "linguistic_direction_norm": float(np.linalg.norm(source_only)),
        **reconstruction_metrics(
            total,
            source_only,
            source_axis=source,
        ),
    }

    explained = out_of_fold_directions["all"]
    residual = total - explained
    forbidden = orthonormal_union([source, explained])
    rng = np.random.default_rng(seed)
    random = rng.normal(size=source.shape).astype(np.float32)
    random = random - forbidden @ (forbidden.T @ random)
    random /= max(float(np.linalg.norm(random)), 1e-12)
    components = {
        "source": source.astype(np.float32),
        "explained": explained.astype(np.float32),
        "residual": residual.astype(np.float32),
        "random": random.astype(np.float32),
    }
    result = {
        "method": "cross_fitted_linguistic_reconstruction",
        "source_checkpoint_cosine": float(source @ checkpoint),
        "total_direction_norm": float(np.linalg.norm(total)),
        "targets": target_results,
        "source_only_baseline": source_only_result,
        "component_norms_before_normalization": {
            key: float(np.linalg.norm(value))
            for key, value in components.items()
        },
        "random_source_cosine": float(random @ source),
        "random_explained_cosine": float(
            random
            @ (
                explained
                / max(float(np.linalg.norm(explained)), 1e-12)
            )
        ),
    }
    return result, components


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir or (get_linguistic_dir("en") / "hidden_language")
    out_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    paired = pd.read_csv(args.features)
    if paired["sample_id"].duplicated().any():
        raise ValueError("Feature sample IDs are not unique")
    ordered_ids = paired["sample_id"].astype(str).tolist()
    generators = paired["generator"].astype(str).to_numpy()
    names_by_group = group_feature_names()
    ordered_names = [
        name for group in GROUPS for name in names_by_group[group]
    ]
    feature_slices: dict[str, np.ndarray] = {}
    offset = 0
    for group in GROUPS:
        width = len(names_by_group[group])
        feature_slices[group] = np.arange(offset, offset + width)
        offset += width
    feature_slices["all"] = np.arange(offset)
    features = paired[ordered_names].to_numpy(dtype=np.float32)
    nuisance = paired[nuisance_feature_names()].to_numpy(dtype=np.float32)

    validation_tables: list[pd.DataFrame] = []
    decompositions: dict[str, Any] = {}
    input_files = {str(args.features): sha256_file(args.features)}
    for spec in args.model_specs:
        model, layer = parse_model_spec(spec)
        base = (
            args.states_root
            / model
            / args.dataset_tag
            / args.representation
        )
        state_path = base / "states" / f"paired_states_layer{layer:02d}.pt"
        source_path = (
            base
            / "models"
            / f"mean_diff_nullspace_layer{layer:02d}.pt"
        )
        human, ai, _ = load_state(state_path, ordered_ids)
        source_payload = torch.load(
            source_path,
            map_location="cpu",
            weights_only=False,
        )
        checkpoint_source = (
            source_payload["state_dict"]["v_src"].float().numpy()
        )
        validation, selected_alphas, out_of_fold_directions = (
            generator_holdout_validation(
                human,
                ai,
                features,
                nuisance,
                generators,
                feature_slices,
                null_k=args.null_k,
                alphas=args.ridge_alphas,
                permutations=args.permutations,
                seed=args.seed,
            )
        )
        validation.insert(0, "model", model)
        validation.insert(1, "layer", layer)
        validation_tables.append(validation)

        decomposition, components = construct_cross_fitted_components(
            human,
            ai,
            features,
            nuisance,
            feature_slices,
            selected_alphas,
            out_of_fold_directions,
            checkpoint_source,
            null_k=args.null_k,
            seed=args.seed,
        )
        model_dir = out_dir / model / args.representation / f"layer{layer:02d}"
        metadata = {
            "method": "cross_fitted_linguistic_reconstruction",
            "model_id": model,
            "layer": layer,
            "representation": args.representation,
            "null_k": args.null_k,
            "feature_file": str(args.features),
            "feature_file_sha256": sha256_file(args.features),
            "source_checkpoint": str(source_path),
            "source_checkpoint_sha256": sha256_file(source_path),
            "selected_alphas": selected_alphas,
            "position_mode": "decode_only",
        }
        for component, vector in components.items():
            save_component_checkpoint(
                model_dir / "directions" / f"{component}.pt",
                vector,
                human,
                ai,
                layer=layer,
                metadata={**metadata, "component": component},
            )
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "decomposition.json").write_text(
            json.dumps(decomposition, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        decompositions[model] = {"layer": layer, **decomposition}
        input_files[str(state_path)] = sha256_file(state_path)
        input_files[str(source_path)] = sha256_file(source_path)

    validation = pd.concat(validation_tables, ignore_index=True)
    validation.to_csv(
        out_dir / "generator_holdout_reconstruction.csv",
        index=False,
    )
    (out_dir / "decomposition_summary.json").write_text(
        json.dumps(decompositions, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    summary = [
        "# Hidden–language source-axis reconstruction",
        "",
        "Ridge strength is selected by nested leave-one-generator-out "
        "validation. Reported outer folds are never used for tuning. "
        "Fractions are signed held-out reconstructions of the hidden source "
        "axis from linguistic contrasts, not in-sample geometric projections.",
        "",
        "| Model | Layer | Target | Alpha | Axis-reconstructed fraction | "
        "Direction cosine | Pair-score correlation | Permutation p |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    for (model, layer, target), subset in validation.groupby(
        ["model", "layer", "target"],
        sort=False,
    ):
        summary.append(
            f"| {model} | {layer} | {target} | "
            f"{'—' if target == 'source_only' else f'{Counter(subset.selected_alpha).most_common(1)[0][0]:g}'} | "
            f"{subset['source_axis_reconstructed_fraction'].mean():.3f} | "
            f"{subset['direction_cosine'].mean():.3f} | "
            f"{subset['pair_score_correlation'].mean():.3f} | "
            f"{'—' if target == 'source_only' else f'{subset.aggregate_permutation_p.iloc[0]:.4g}'} |"
        )
    summary.extend(
        [
            "",
            "## Cross-fitted component construction",
            "",
            "| Model | Source checkpoint cosine | All-feature axis fraction | "
            "All-feature cosine | Vector reconstruction R² | "
            "Pair incremental R² vs source-only |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for model, result in decompositions.items():
        metrics = result["targets"]["all"]
        model_rows = validation[
            (validation["model"] == model)
            & (validation["target"] == "all")
        ]
        summary.append(
            f"| {model} | {result['source_checkpoint_cosine']:.6f} | "
            f"{metrics['source_axis_reconstructed_fraction']:.3f} | "
            f"{metrics['direction_cosine']:.3f} | "
            f"{metrics['vector_reconstruction_r2']:.3f} | "
            f"{model_rows['pair_incremental_r2_over_source_only'].mean():.3f} |"
        )
    summary.extend(
        [
            "",
            "The source-only baseline transfers the training-generator mean "
            "hidden shift without using linguistic features. A linguistic "
            "reconstruction is not considered explanatory unless it improves "
            "held-out pair prediction beyond this baseline.",
        ]
    )
    (out_dir / "analysis_summary.md").write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    finished = datetime.now(timezone.utc)
    manifest = {
        "schema": "hidden_language_source_axis_reconstruction",
        "language": "en",
        "model_specs": args.model_specs,
        "dataset_tag": args.dataset_tag,
        "representation": args.representation,
        "null_k": args.null_k,
        "ridge_alphas": args.ridge_alphas,
        "permutations": args.permutations,
        "seed": args.seed,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "torch_version": torch.__version__,
        "input_sha256": input_files,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "command": " ".join(sys.argv),
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote hidden–language reconstruction to {out_dir}")


if __name__ == "__main__":
    main()
