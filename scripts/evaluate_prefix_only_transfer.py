#!/usr/bin/env python3
"""Evaluate transfer from plan-conditioned to prefix-only continuations.

All axes are fitted on the canonical proposition-aligned corpus. The matched
prefix-only corpus is used only for testing. Generator-held-out folds prevent
the tested generator from contributing to the fitted source or concept axis.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.analyze_projection_mapping import (  # noqa: E402
    load_states,
    source_arrays,
)
from scripts.audit_hidden_propagation import atomic_json, sha256_file  # noqa: E402
from scripts.build_concept_patterns import (  # noqa: E402
    fit_patterns,
    inventory,
    state_path,
)
from src.analysis.causal_linguistics import paired_mean_inference  # noqa: E402
from src.analysis.concept_patterns import signature_weights  # noqa: E402
from src.analysis.projection_mapping import (  # noqa: E402
    fit_nullspace_source_direction,
)
from src.runtime import stable_seed  # noqa: E402


CONCEPTS = ("lexical", "syntactic")
DEFAULT_MODEL_SPECS = (
    "Llama-3.1-8B-Instruct:7:32",
    "Qwen3-4B:8:36",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-pairs",
        type=Path,
        default=Path("data/paired/en/proposition_aligned/pairs.jsonl"),
    )
    parser.add_argument(
        "--control-pairs",
        type=Path,
        default=Path("data/paired/en/prefix_only_control/pairs.jsonl"),
    )
    parser.add_argument(
        "--plan-features",
        type=Path,
        default=Path(
            "outputs/en/05_linguistic/features/"
            "paired_feature_differences.csv"
        ),
    )
    parser.add_argument(
        "--control-features",
        type=Path,
        default=Path(
            "outputs/en/05_linguistic/prefix_only_control/features/"
            "paired_feature_differences.csv"
        ),
    )
    parser.add_argument(
        "--states-root",
        type=Path,
        default=Path("outputs/en/03_steering_vectors"),
    )
    parser.add_argument("--plan-dataset-tag", default="proposition_aligned")
    parser.add_argument("--control-dataset-tag", default="prefix_only_control")
    parser.add_argument(
        "--model-specs",
        nargs="+",
        default=list(DEFAULT_MODEL_SPECS),
        help="MODEL_ID:INJECTION_LAYER:ANCHOR_LAYER entries.",
    )
    parser.add_argument("--null-k", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--surface-only",
        action="store_true",
        help=(
            "Write the generator-cross-fitted surface transfer results "
            "without requiring hidden-state files."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "outputs/en/05_linguistic/prefix_only_control/transfer"
        ),
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def parse_model_spec(spec: str) -> tuple[str, tuple[int, int]]:
    parts = spec.rsplit(":", 2)
    if len(parts) != 3 or not parts[0]:
        raise ValueError(f"Expected MODEL:INJECTION:ANCHOR, got {spec!r}")
    return parts[0], (int(parts[1]), int(parts[2]))


def discover_state_layers(
    root: Path,
    model: str,
    dataset_tag: str,
) -> list[int]:
    directory = root / model / dataset_tag / "mean_pool" / "states"
    layers = []
    for path in directory.glob("paired_states_layer*.pt"):
        match = re.search(r"layer(\d+)$", path.stem)
        if match:
            layers.append(int(match.group(1)))
    if not layers:
        raise FileNotFoundError(f"No mean-pool states under {directory}")
    return sorted(set(layers))


def source_auc(
    human: np.ndarray,
    ai: np.ndarray,
    direction: np.ndarray,
) -> float:
    scores = np.concatenate([human @ direction, ai @ direction])
    labels = np.concatenate([np.zeros(len(human)), np.ones(len(ai))])
    return float(roc_auc_score(labels, scores))


def standardized_gap(
    human: np.ndarray,
    ai: np.ndarray,
    direction: np.ndarray,
) -> float:
    human_scores = human @ direction
    ai_scores = ai @ direction
    pooled = np.sqrt(
        (human_scores.var(ddof=1) + ai_scores.var(ddof=1)) / 2.0
    )
    return float(
        (ai_scores.mean() - human_scores.mean()) / max(pooled, 1e-12)
    )


def validate_matched_corpora(
    plan_pairs: list[dict[str, Any]],
    control_pairs: list[dict[str, Any]],
) -> list[str]:
    plan = {str(row["sample_id"]): row for row in plan_pairs}
    control = {str(row["sample_id"]): row for row in control_pairs}
    if set(plan) != set(control):
        raise ValueError("Plan and prefix-only sample IDs do not match")
    ordered_ids = [str(row["sample_id"]) for row in plan_pairs]
    for sample_id in ordered_ids:
        left = plan[sample_id]
        right = control[sample_id]
        for field in ("prefix", "human_continuation", "source", "domain"):
            if left[field] != right[field]:
                raise ValueError(f"{sample_id}: mismatched {field}")
        if (
            left["generation"]["generator_model"]
            != right["generation"]["generator_model"]
        ):
            raise ValueError(f"{sample_id}: mismatched generator")
    return ordered_ids


def surface_signature_transfer(
    plan_frame: pd.DataFrame,
    control_frame: pd.DataFrame,
    *,
    bootstrap: int,
    permutations: int,
    seed: int,
) -> pd.DataFrame:
    feature_names, _, columns = inventory(list(CONCEPTS))
    plan = source_arrays(plan_frame, feature_names)
    control = source_arrays(control_frame, feature_names)
    rows = []
    for concept in CONCEPTS:
        selected = columns[concept]
        weights, info = signature_weights(
            plan["human"][:, selected],
            plan["ai"][:, selected],
        )
        pooled_scale = np.sqrt(
            (
                plan["human"][:, selected].var(axis=0, ddof=1)
                + plan["ai"][:, selected].var(axis=0, ddof=1)
            )
            / 2.0
        )
        pooled_scale[pooled_scale <= 1e-12] = 1.0
        plan_differences = (
            plan["ai"][:, selected] - plan["human"][:, selected]
        ) / pooled_scale
        control_differences = (
            control["ai"][:, selected] - control["human"][:, selected]
        ) / pooled_scale
        plan_shift = plan_differences.mean(axis=0)
        control_shift = control_differences.mean(axis=0)
        plan_norm = float(np.linalg.norm(plan_shift))
        control_norm = float(np.linalg.norm(control_shift))
        cosine = float(
            plan_shift @ control_shift
            / max(plan_norm * control_norm, 1e-12)
        )
        plan_scores = plan_differences @ weights
        control_scores = control_differences @ weights
        inference = paired_mean_inference(
            control_scores,
            bootstrap=bootstrap,
            permutations=permutations,
            seed=stable_seed(seed, "surface_transfer", concept),
        )
        rows.append({
            "concept": concept,
            "n_pairs": len(control_scores),
            "n_features": len(selected),
            "plan_standardized_shift_norm": plan_norm,
            "control_standardized_shift_norm": control_norm,
            "plan_control_shift_cosine": cosine,
            "plan_projection_mean": float(plan_scores.mean()),
            "control_projection_positive_rate": float(
                np.mean(control_scores > 0)
            ),
            "training_signature_shift_norm": info[
                "standardized_shift_norm"
            ],
            **{
                f"control_projection_{key}": value
                for key, value in inference.items()
            },
        })
    return pd.DataFrame(rows)


def crossfit_surface_signature_transfer(
    plan_frame: pd.DataFrame,
    control_frame: pd.DataFrame,
    generators: np.ndarray,
    *,
    bootstrap: int,
    permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate surface transfer without reusing held-out human samples.

    Each surface axis and its feature scales are fitted on three generators
    from the proposition-aligned corpus. It is then applied to the fourth
    generator's proposition-aligned and prefix-only pairs. Pooling the
    out-of-fold scores gives one prediction per sample from an axis that
    never saw that sample or its shared human continuation.
    """
    feature_names, _, columns = inventory(list(CONCEPTS))
    plan = source_arrays(plan_frame, feature_names)
    control = source_arrays(control_frame, feature_names)
    generators = np.asarray(generators)
    n_pairs = len(plan_frame)
    if len(control_frame) != n_pairs or len(generators) != n_pairs:
        raise ValueError("Surface frames and generators must be row aligned")
    unique_generators = sorted(np.unique(generators))
    if len(unique_generators) < 2:
        raise ValueError("Cross-fitted transfer needs at least two generators")

    summary_rows = []
    fold_rows = []
    for concept in CONCEPTS:
        selected = columns[concept]
        plan_oof_scores = np.full(n_pairs, np.nan, dtype=np.float64)
        control_oof_scores = np.full(n_pairs, np.nan, dtype=np.float64)
        plan_oof_differences = np.full(
            (n_pairs, len(selected)),
            np.nan,
            dtype=np.float64,
        )
        control_oof_differences = np.full_like(
            plan_oof_differences,
            np.nan,
        )
        training_norms = []
        for heldout in unique_generators:
            train = generators != heldout
            test = generators == heldout
            weights, info = signature_weights(
                plan["human"][train][:, selected],
                plan["ai"][train][:, selected],
            )
            pooled_scale = np.sqrt(
                (
                    plan["human"][train][:, selected].var(axis=0, ddof=1)
                    + plan["ai"][train][:, selected].var(axis=0, ddof=1)
                )
                / 2.0
            )
            pooled_scale[pooled_scale <= 1e-12] = 1.0
            plan_differences = (
                plan["ai"][test][:, selected]
                - plan["human"][test][:, selected]
            ) / pooled_scale
            control_differences = (
                control["ai"][test][:, selected]
                - control["human"][test][:, selected]
            ) / pooled_scale
            plan_scores = plan_differences @ weights
            control_scores = control_differences @ weights
            plan_oof_scores[test] = plan_scores
            control_oof_scores[test] = control_scores
            plan_oof_differences[test] = plan_differences
            control_oof_differences[test] = control_differences
            training_norms.append(info["standardized_shift_norm"])

            plan_shift = plan_differences.mean(axis=0)
            control_shift = control_differences.mean(axis=0)
            denominator = float(
                np.linalg.norm(plan_shift)
                * np.linalg.norm(control_shift)
            )
            fold_rows.append({
                "concept": concept,
                "heldout_generator": heldout,
                "n_train": int(train.sum()),
                "n_test": int(test.sum()),
                "plan_control_shift_cosine": float(
                    plan_shift @ control_shift
                    / max(denominator, 1e-12)
                ),
                "plan_projection_mean": float(plan_scores.mean()),
                "control_projection_mean": float(control_scores.mean()),
                "control_projection_positive_rate": float(
                    np.mean(control_scores > 0)
                ),
                "training_signature_shift_norm": info[
                    "standardized_shift_norm"
                ],
            })

        if (
            np.isnan(plan_oof_scores).any()
            or np.isnan(control_oof_scores).any()
            or np.isnan(plan_oof_differences).any()
            or np.isnan(control_oof_differences).any()
        ):
            raise RuntimeError("Cross-fitting left samples without predictions")
        plan_shift = plan_oof_differences.mean(axis=0)
        control_shift = control_oof_differences.mean(axis=0)
        denominator = float(
            np.linalg.norm(plan_shift) * np.linalg.norm(control_shift)
        )
        inference = paired_mean_inference(
            control_oof_scores,
            bootstrap=bootstrap,
            permutations=permutations,
            seed=stable_seed(seed, "surface_transfer_crossfit", concept),
        )
        summary_rows.append({
            "concept": concept,
            "n_pairs": n_pairs,
            "n_features": len(selected),
            "n_generator_folds": len(unique_generators),
            "plan_control_shift_cosine": float(
                plan_shift @ control_shift / max(denominator, 1e-12)
            ),
            "plan_projection_mean": float(plan_oof_scores.mean()),
            "control_projection_positive_rate": float(
                np.mean(control_oof_scores > 0)
            ),
            "training_signature_shift_norm_mean": float(
                np.mean(training_norms)
            ),
            **{
                f"control_projection_{key}": value
                for key, value in inference.items()
            },
        })
    return pd.DataFrame(summary_rows), pd.DataFrame(fold_rows)


def source_axis_transfer(
    *,
    states_root: Path,
    model: str,
    plan_tag: str,
    control_tag: str,
    layers: list[int],
    ordered_ids: list[str],
    generators: np.ndarray,
    null_k: int,
) -> pd.DataFrame:
    rows = []
    for layer in layers:
        plan_human, plan_ai, _ = load_states(
            state_path(
                states_root,
                model,
                plan_tag,
                "mean_pool",
                layer,
            ),
            ordered_ids,
        )
        control_human, control_ai, _ = load_states(
            state_path(
                states_root,
                model,
                control_tag,
                "mean_pool",
                layer,
            ),
            ordered_ids,
        )
        for heldout in sorted(np.unique(generators)):
            train = generators != heldout
            test = generators == heldout
            direction, diagnostics = fit_nullspace_source_direction(
                plan_human[train],
                plan_ai[train],
                k=null_k,
            )
            rows.append({
                "model": model,
                "representation": "mean_pool",
                "layer": layer,
                "heldout_generator": heldout,
                "n_train": int(train.sum()),
                "n_test": int(test.sum()),
                "plan_auc": source_auc(
                    plan_human[test],
                    plan_ai[test],
                    direction,
                ),
                "control_auc": source_auc(
                    control_human[test],
                    control_ai[test],
                    direction,
                ),
                "plan_standardized_gap": standardized_gap(
                    plan_human[test],
                    plan_ai[test],
                    direction,
                ),
                "control_standardized_gap": standardized_gap(
                    control_human[test],
                    control_ai[test],
                    direction,
                ),
                "direction_cosine_vs_raw": diagnostics["cosine_vs_raw"],
            })
    return pd.DataFrame(rows)


def concept_mapping_transfer(
    *,
    states_root: Path,
    model: str,
    plan_tag: str,
    control_tag: str,
    layers: list[int],
    ordered_ids: list[str],
    generators: np.ndarray,
    plan_frame: pd.DataFrame,
    control_frame: pd.DataFrame,
    null_k: int,
) -> pd.DataFrame:
    feature_names, nuisance_names, columns = inventory(list(CONCEPTS))
    plan_features = source_arrays(plan_frame, feature_names)
    control_features = source_arrays(control_frame, feature_names)
    plan_nuisance = source_arrays(plan_frame, nuisance_names)
    control_nuisance = source_arrays(control_frame, nuisance_names)
    n_pairs = len(ordered_ids)
    rows = []

    for layer in layers:
        plan_human, plan_ai, _ = load_states(
            state_path(
                states_root,
                model,
                plan_tag,
                "mean_pool",
                layer,
            ),
            ordered_ids,
        )
        control_human, control_ai, _ = load_states(
            state_path(
                states_root,
                model,
                control_tag,
                "mean_pool",
                layer,
            ),
            ordered_ids,
        )
        for heldout in sorted(np.unique(generators)):
            train = np.flatnonzero(generators != heldout)
            plan_test = np.flatnonzero(generators == heldout)
            plan_patterns, plan_diagnostics, plan_validation = fit_patterns(
                {"human": plan_human, "ai": plan_ai},
                plan_features,
                plan_nuisance,
                columns,
                train=train,
                test=plan_test,
                null_k=null_k,
                pattern_estimator="covariance",
                probe_components=64,
            )
            combined_hidden = {
                "human": np.concatenate(
                    [plan_human, control_human],
                    axis=0,
                ),
                "ai": np.concatenate([plan_ai, control_ai], axis=0),
            }
            combined_features = {
                source: np.concatenate(
                    [plan_features[source], control_features[source]],
                    axis=0,
                )
                for source in ("human", "ai")
            }
            combined_nuisance = {
                source: np.concatenate(
                    [plan_nuisance[source], control_nuisance[source]],
                    axis=0,
                )
                for source in ("human", "ai")
            }
            control_test = n_pairs + plan_test
            (
                control_patterns,
                _control_diagnostics,
                control_validation,
            ) = fit_patterns(
                combined_hidden,
                combined_features,
                combined_nuisance,
                columns,
                train=train,
                test=control_test,
                null_k=null_k,
                pattern_estimator="covariance",
                probe_components=64,
            )
            plan_by_key = {
                (row["group"], row["test_source"]): row
                for row in plan_validation
            }
            control_by_key = {
                (row["group"], row["test_source"]): row
                for row in control_validation
            }
            for concept in CONCEPTS:
                pattern_cosine = float(
                    plan_patterns[concept] @ control_patterns[concept]
                )
                if pattern_cosine < 1.0 - 1e-8:
                    raise RuntimeError(
                        "Training-only concept pattern changed when the "
                        "prefix-only test set was appended"
                    )
                rows.append({
                    "model": model,
                    "representation": "mean_pool",
                    "layer": layer,
                    "heldout_generator": heldout,
                    "concept": concept,
                    "n_train": len(train),
                    "n_test": len(plan_test),
                    "plan_ai_score_correlation": plan_by_key[
                        (concept, "ai")
                    ]["heldout_score_correlation"],
                    "control_ai_score_correlation": control_by_key[
                        (concept, "ai")
                    ]["heldout_score_correlation"],
                    "train_human_ai_pattern_cosine": plan_diagnostics[
                        concept
                    ]["human_ai_pattern_cosine"],
                    "test_append_pattern_cosine": pattern_cosine,
                })
    return pd.DataFrame(rows)


def write_summary(
    path: Path,
    surface: pd.DataFrame,
    source: pd.DataFrame,
    mapping: pd.DataFrame,
) -> None:
    lines = [
        "# Prefix-only construct-control transfer",
        "",
        "Axes are fitted only on proposition-aligned continuations. Every "
        "prefix-only result uses the same prefix, human reference, and "
        "generator assignment but exposes no proposition plan to the "
        "generator.",
        "",
        "## Surface signature",
        "",
        "| Concept | Shift cosine | Frozen-axis mean [95% CI] | "
        "Positive pairs |",
        "|---|---:|---:|---:|",
    ]
    for row in surface.itertuples(index=False):
        lines.append(
            f"| {row.concept} | {row.plan_control_shift_cosine:.3f} | "
            f"{row.control_projection_mean:.3f} "
            f"[{row.control_projection_ci_low:.3f}, "
            f"{row.control_projection_ci_high:.3f}] | "
            f"{row.control_projection_positive_rate:.3f} |"
        )
    lines.extend([
        "",
        "## Generator-held-out hidden transfer",
        "",
        "| Model | Layers | Plan AUC range | Prefix-only AUC range |",
        "|---|---:|---:|---:|",
    ])
    for model, part in source.groupby("model", sort=False):
        layer_summary = (
            part.groupby("layer")["control_auc"].mean().sort_index()
        )
        lines.append(
            f"| {model} | {len(layer_summary)} | "
            f"{part['plan_auc'].min():.3f}–{part['plan_auc'].max():.3f} | "
            f"{part['control_auc'].min():.3f}–"
            f"{part['control_auc'].max():.3f} |"
        )
    lines.extend([
        "",
        "## Generator-held-out surface-to-hidden mapping",
        "",
        "| Model | Concept | Plan AI r | Prefix-only AI r |",
        "|---|---|---:|---:|",
    ])
    grouped = mapping.groupby(["model", "concept"], sort=False)
    for (model, concept), part in grouped:
        lines.append(
            f"| {model} | {concept} | "
            f"{part['plan_ai_score_correlation'].mean():.3f} | "
            f"{part['control_ai_score_correlation'].mean():.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_surface_summary(path: Path, surface: pd.DataFrame) -> None:
    lines = [
        "# Prefix-only surface construct-control transfer",
        "",
        "Each lexical or syntactic axis is fitted on three generators from "
        "the proposition-aligned corpus and evaluated on the held-out "
        "generator. Thus no tested sample or shared human continuation "
        "contributes to its fitted axis.",
        "",
        "| Concept | Shift cosine | Frozen-axis mean [95% CI] | "
        "Positive pairs | Sign-flip p |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in surface.itertuples(index=False):
        lines.append(
            f"| {row.concept} | {row.plan_control_shift_cosine:.3f} | "
            f"{row.control_projection_mean:.3f} "
            f"[{row.control_projection_ci_low:.3f}, "
            f"{row.control_projection_ci_high:.3f}] | "
            f"{row.control_projection_positive_rate:.3f} | "
            f"{row.control_projection_sign_flip_p:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = datetime.now(timezone.utc)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plan_pairs = load_jsonl(args.plan_pairs)
    control_pairs = load_jsonl(args.control_pairs)
    ordered_ids = validate_matched_corpora(plan_pairs, control_pairs)
    generators = np.asarray([
        row["generation"]["generator_model"] for row in plan_pairs
    ])
    plan_frame = (
        pd.read_csv(args.plan_features)
        .set_index("sample_id")
        .loc[ordered_ids]
        .reset_index()
    )
    control_frame = (
        pd.read_csv(args.control_features)
        .set_index("sample_id")
        .loc[ordered_ids]
        .reset_index()
    )
    surface = surface_signature_transfer(
        plan_frame,
        control_frame,
        bootstrap=args.bootstrap,
        permutations=args.permutations,
        seed=args.seed,
    )
    surface_crossfit, surface_crossfit_folds = (
        crossfit_surface_signature_transfer(
            plan_frame,
            control_frame,
            generators,
            bootstrap=args.bootstrap,
            permutations=args.permutations,
            seed=args.seed,
        )
    )
    surface.to_csv(
        args.out_dir / "surface_signature_descriptive.csv",
        index=False,
    )
    surface_crossfit.to_csv(
        args.out_dir / "surface_signature_transfer.csv",
        index=False,
    )
    surface_crossfit_folds.to_csv(
        args.out_dir / "surface_signature_transfer_folds.csv",
        index=False,
    )
    input_paths = {
        str(path): sha256_file(path)
        for path in (
            args.plan_pairs,
            args.control_pairs,
            args.plan_features,
            args.control_features,
        )
    }
    if args.surface_only:
        write_surface_summary(
            args.out_dir / "analysis_summary.md",
            surface_crossfit,
        )
        finished = datetime.now(timezone.utc)
        atomic_json(
            args.out_dir / "manifest.json",
            {
                "schema": "prefix_only_surface_transfer_v1",
                "status": "complete",
                "n_pairs": len(ordered_ids),
                "n_generator_folds": int(len(np.unique(generators))),
                "fit_corpus": args.plan_dataset_tag,
                "test_corpus": args.control_dataset_tag,
                "sample_level_cross_fit": True,
                "input_sha256": input_paths,
                "bootstrap": args.bootstrap,
                "permutations": args.permutations,
                "seed": args.seed,
                "started_at_utc": started.isoformat(),
                "finished_at_utc": finished.isoformat(),
                "duration_seconds": (
                    finished - started
                ).total_seconds(),
                "command": " ".join(sys.argv),
            },
        )
        return
    source_parts = []
    mapping_parts = []
    for model, (injection, anchor) in map(
        parse_model_spec,
        args.model_specs,
    ):
        plan_layers = discover_state_layers(
            args.states_root,
            model,
            args.plan_dataset_tag,
        )
        control_layers = discover_state_layers(
            args.states_root,
            model,
            args.control_dataset_tag,
        )
        if plan_layers != control_layers:
            raise ValueError(
                f"{model}: plan/control extracted layers do not match"
            )
        if injection not in plan_layers or anchor not in plan_layers:
            raise ValueError(f"{model}: requested concept layers are absent")
        source_parts.append(source_axis_transfer(
            states_root=args.states_root,
            model=model,
            plan_tag=args.plan_dataset_tag,
            control_tag=args.control_dataset_tag,
            layers=plan_layers,
            ordered_ids=ordered_ids,
            generators=generators,
            null_k=args.null_k,
        ))
        mapping_parts.append(concept_mapping_transfer(
            states_root=args.states_root,
            model=model,
            plan_tag=args.plan_dataset_tag,
            control_tag=args.control_dataset_tag,
            layers=sorted({injection, anchor}),
            ordered_ids=ordered_ids,
            generators=generators,
            plan_frame=plan_frame,
            control_frame=control_frame,
            null_k=args.null_k,
        ))
        for dataset_tag in (
            args.plan_dataset_tag,
            args.control_dataset_tag,
        ):
            for layer in plan_layers:
                path = state_path(
                    args.states_root,
                    model,
                    dataset_tag,
                    "mean_pool",
                    layer,
                )
                input_paths[str(path)] = sha256_file(path)
    source = pd.concat(source_parts, ignore_index=True)
    mapping = pd.concat(mapping_parts, ignore_index=True)
    source.to_csv(args.out_dir / "source_axis_transfer.csv", index=False)
    (
        source.groupby(["model", "layer"], as_index=False)
        .agg(
            n_generators=("heldout_generator", "nunique"),
            plan_auc_mean=("plan_auc", "mean"),
            plan_auc_min=("plan_auc", "min"),
            control_auc_mean=("control_auc", "mean"),
            control_auc_min=("control_auc", "min"),
            plan_gap_mean=("plan_standardized_gap", "mean"),
            control_gap_mean=("control_standardized_gap", "mean"),
        )
        .to_csv(args.out_dir / "source_axis_layer_summary.csv", index=False)
    )
    mapping.to_csv(
        args.out_dir / "concept_mapping_transfer.csv",
        index=False,
    )
    write_summary(
        args.out_dir / "analysis_summary.md",
        surface_crossfit,
        source,
        mapping,
    )
    finished = datetime.now(timezone.utc)
    atomic_json(
        args.out_dir / "manifest.json",
        {
            "schema": "prefix_only_construct_transfer_v2",
            "status": "complete",
            "n_pairs": len(ordered_ids),
            "models": [
                parse_model_spec(spec)[0] for spec in args.model_specs
            ],
            "representation": "mean_pool",
            "holdout": "generator",
            "surface_sample_level_cross_fit": True,
            "fit_corpus": args.plan_dataset_tag,
            "test_corpus": args.control_dataset_tag,
            "control_contract": (
                "same prefix, human continuation, sample, source, domain, "
                "and generator assignment; generator receives prefix and "
                "scalar length only"
            ),
            "input_sha256": input_paths,
            "bootstrap": args.bootstrap,
            "permutations": args.permutations,
            "seed": args.seed,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "sklearn_version": sklearn.__version__,
            "torch_version": torch.__version__,
            "started_at_utc": started.isoformat(),
            "finished_at_utc": finished.isoformat(),
            "duration_seconds": (finished - started).total_seconds(),
            "command": " ".join(sys.argv),
        },
    )


if __name__ == "__main__":
    main()
