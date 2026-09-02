#!/usr/bin/env python3
"""Audit Jacobian pullbacks against pipeline-matched permutation nulls.

Ordinary random residual-stream directions do not control for the fact that a
learned intervention is explicitly constructed as ``J.T @ readout``.  This
script builds a stronger finite null ensemble:

1. permute the surface-score correspondence within corpus×generator strata;
2. refit the same cross-source layerwise activation patterns;
3. pull each null readout back through the same frozen-model Jacobian;
4. inject each null pullback on the same fixed token sequences; and
5. compare learned transport with the matched null ensemble.

The resulting comparison asks whether the observed trace is exceptional
relative to the complete readout→pullback→fixed-trajectory procedure.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.analyze_projection_mapping import load_states, source_arrays
from scripts.audit_hidden_propagation import (
    atomic_json,
    balanced_prompt_selection,
    capture_symmetric_direction_batch_with_fallback,
    filter_valid_anchor_records,
    fixed_trajectory_provenance,
    json_fingerprint,
    load_anchor_records,
    sha256_file,
)
from scripts.audit_layerwise_concept_transport import (
    load_checkpoint_vector,
    make_steering,
)
from scripts.build_causal_pullback_directions import (
    balanced_selection,
    batch_pullback_gradients,
    load_jsonl,
)
from scripts.build_concept_patterns import (
    fit_patterns,
    inventory,
    state_path,
)
from src.analysis.causal_linguistics import paired_mean_inference
from src.analysis.causal_pullback import (
    finite_null_p_value,
    nullspace_pullback_consensus,
    normalized_curve_area,
    stratified_permutation,
    weighted_gradient_mean,
)
from src.analysis.projection_mapping import fit_within_class_basis
from src.extraction.hidden_states import HiddenStatesExtractor
from src.runtime import model_source_fingerprint, stable_seed


CONCEPTS = ("lexical", "syntactic")


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument(
        "--pairs",
        type=Path,
        default=Path("data/paired/en/proposition_aligned/pairs.jsonl"),
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
        "--states-root",
        type=Path,
        default=Path("outputs/en/03_steering_vectors"),
    )
    parser.add_argument("--dataset-tag", default="proposition_aligned")
    parser.add_argument(
        "--prompts",
        type=Path,
        default=Path("data/prompts/en/formal.json"),
    )
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--canonical-root", type=Path, required=True)
    parser.add_argument("--learned-transport", type=Path, required=True)
    parser.add_argument("--max-pairs", type=int, default=32)
    parser.add_argument("--pullback-batch-size", type=int, default=2)
    parser.add_argument("--readout-chunk-size", type=int, default=4)
    parser.add_argument("--max-prompts", type=int, default=12)
    parser.add_argument("--null-controls", type=int, default=8)
    parser.add_argument(
        "--trajectory-direction-batch-size",
        type=int,
        default=4,
    )
    parser.add_argument("--null-k", type=int, default=10)
    parser.add_argument("--anchor-component", default="random")
    parser.add_argument("--anchor-strength", type=float, default=-1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="bf16",
    )
    parser.add_argument(
        "--attention-backend",
        choices=["auto", "sdpa", "flash_attention_2", "eager"],
        default="sdpa",
    )
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def discover_layers(args: argparse.Namespace) -> list[int]:
    root = (
        args.states_root
        / args.model_id
        / args.dataset_tag
        / "mean_pool"
        / "states"
    )
    layers = sorted(
        int(path.stem.removeprefix("paired_states_layer"))
        for path in root.glob("paired_states_layer[0-9][0-9].pt")
    )
    if not layers:
        raise FileNotFoundError(f"No mean-pool states under {root}")
    return layers


def canonical_configuration(
    root: Path,
) -> tuple[int, float, dict[str, int], dict[str, str]]:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("orientation_uses_generation_outcomes") is not False:
        raise ValueError(
            f"{manifest_path} is not a canonical, outcome-independent freeze"
        )
    selected = manifest["selected_components"]
    anchors = {
        concept: int(
            selected[concept]["fit_metadata"]["downstream_layer"]
        )
        for concept in CONCEPTS
    }
    layer_dirs = sorted(root.glob("layer[0-9][0-9]"))
    if len(layer_dirs) != 1:
        raise ValueError(f"Expected one injection layer under {root}")
    injection_layer = int(layer_dirs[0].name.removeprefix("layer"))
    _, _, scale = load_checkpoint_vector(
        layer_dirs[0] / "directions" / "lexical.pt"
    )
    hashes = {
        str(manifest_path): sha256_file(manifest_path),
        **{
            str(layer_dirs[0] / "directions" / f"{concept}.pt"):
            sha256_file(
                layer_dirs[0] / "directions" / f"{concept}.pt"
            )
            for concept in CONCEPTS
        },
    }
    return injection_layer, scale, anchors, hashes


def build_permuted_readouts(
    args: argparse.Namespace,
    layers: list[int],
    frame: pd.DataFrame,
) -> tuple[
    dict[int, dict[str, np.ndarray]],
    list[dict[str, Any]],
    dict[str, str],
]:
    feature_names, nuisance_names, columns = inventory(list(CONCEPTS))
    features = source_arrays(frame, feature_names)
    nuisance = source_arrays(frame, nuisance_names)
    ordered_ids = frame["sample_id"].astype(str).tolist()
    environments = (
        frame["source"].astype(str)
        + "|"
        + frame["generator"].astype(str)
    ).to_numpy()
    indices = np.arange(len(frame))
    permutations = [
        stratified_permutation(
            environments,
            seed=stable_seed(args.seed, "surface-null", control),
        )
        for control in range(args.null_controls)
    ]
    permutation_diagnostics = []
    for control, permutation in enumerate(permutations):
        permutation_diagnostics.append({
            "null_axis": control,
            "permutation_sha256": hashlib.sha256(
                np.asarray(permutation, dtype=np.int64).tobytes()
            ).hexdigest(),
            "fixed_point_fraction": float(
                np.mean(permutation == np.arange(len(permutation)))
            ),
        })
    if len({
        row["permutation_sha256"]
        for row in permutation_diagnostics
    }) != args.null_controls:
        raise RuntimeError("Matched-null permutation ensemble has duplicates")
    permutation_info = {
        row["null_axis"]: row for row in permutation_diagnostics
    }
    readouts: dict[int, dict[str, np.ndarray]] = {}
    diagnostics: list[dict[str, Any]] = []
    input_hashes = {str(args.features): sha256_file(args.features)}

    for layer in layers:
        states = state_path(
            args.states_root,
            args.model_id,
            args.dataset_tag,
            "mean_pool",
            layer,
        )
        human, ai, _ = load_states(states, ordered_ids)
        hidden = {"human": human, "ai": ai}
        layer_readouts = {}
        for control, permutation in enumerate(permutations):
            permuted_features = {
                source: values[permutation]
                for source, values in features.items()
            }
            permuted_nuisance = {
                source: values[permutation]
                for source, values in nuisance.items()
            }
            patterns, pattern_info, _ = fit_patterns(
                hidden,
                permuted_features,
                nuisance,
                columns,
                feature_nuisance=permuted_nuisance,
                train=indices,
                test=indices,
                null_k=args.null_k,
                pattern_estimator="covariance",
                probe_components=64,
            )
            for concept in CONCEPTS:
                name = f"{concept}__null_{control:02d}"
                layer_readouts[name] = patterns[concept]
                diagnostics.append(
                    {
                        "layer": layer,
                        "concept": concept,
                        "null_axis": control,
                        "permutation_sha256": permutation_info[control][
                            "permutation_sha256"
                        ],
                        "fixed_point_fraction": permutation_info[control][
                            "fixed_point_fraction"
                        ],
                        "human_ai_pattern_cosine": pattern_info[concept][
                            "human_ai_pattern_cosine"
                        ],
                        "standardized_shift_norm": pattern_info[concept][
                            "standardized_shift_norm"
                        ],
                    }
                )
        readouts[layer] = layer_readouts
        input_hashes[str(states)] = sha256_file(states)
        print(f"fitted matched-null readouts at L{layer}", flush=True)
    return readouts, diagnostics, input_hashes


def pullback_null_readouts(
    args: argparse.Namespace,
    extractor: HiddenStatesExtractor,
    readouts: dict[int, dict[str, np.ndarray]],
    *,
    injection_layer: int,
    anchors: dict[str, int],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    records = load_jsonl(args.pairs)
    selected = balanced_selection(records, args.max_pairs)
    input_states = state_path(
        args.states_root,
        args.model_id,
        args.dataset_tag,
        "mean_pool",
        injection_layer,
    )
    state_payload = torch.load(
        input_states,
        map_location="cpu",
        weights_only=False,
    )
    within_basis, _ = fit_within_class_basis(
        state_payload["human"].float().numpy(),
        state_payload["ai"].float().numpy(),
        k=args.null_k,
    )
    readouts_by_anchor: dict[int, dict[str, torch.Tensor]] = {}
    for concept, anchor in anchors.items():
        for control in range(args.null_controls):
            name = f"{concept}__null_{control:02d}"
            readouts_by_anchor.setdefault(anchor, {})[name] = (
                torch.from_numpy(readouts[anchor][name]).float()
            )

    source_gradients: dict[str, dict[str, list[np.ndarray]]] = {
        "human": {},
        "ai": {},
    }
    source_weights: dict[str, dict[str, list[int]]] = {
        "human": {},
        "ai": {},
    }
    for source in ("human", "ai"):
        for start in range(0, len(selected), args.pullback_batch_size):
            batch = selected[start:start + args.pullback_batch_size]
            for anchor, anchor_readouts in readouts_by_anchor.items():
                names = list(anchor_readouts)
                for offset in range(0, len(names), args.readout_chunk_size):
                    chunk_names = names[
                        offset:offset + args.readout_chunk_size
                    ]
                    gradients, _, n_sequences = batch_pullback_gradients(
                        extractor,
                        [str(record["prefix"]) for record in batch],
                        [
                            str(record[f"{source}_continuation"])
                            for record in batch
                        ],
                        input_layer=injection_layer,
                        downstream_layer=anchor,
                        readouts={
                            name: anchor_readouts[name]
                            for name in chunk_names
                        },
                    )
                    for name, gradient in gradients.items():
                        source_gradients[source].setdefault(name, []).append(
                            gradient
                        )
                        source_weights[source].setdefault(name, []).append(
                            n_sequences
                        )
            print(
                f"{source} pullback batch "
                f"{start // args.pullback_batch_size + 1}",
                flush=True,
            )

    axes = {}
    diagnostics = []
    names = sorted(source_gradients["human"])
    for name in names:
        means = {}
        for source in ("human", "ai"):
            means[source] = weighted_gradient_mean(
                np.stack(source_gradients[source][name]),
                np.asarray(source_weights[source][name]),
            )
        axis, info = nullspace_pullback_consensus(
            means["human"],
            means["ai"],
            within_basis,
        )
        axes[name] = axis
        concept, null_text = name.split("__null_")
        diagnostics.append(
            {
                "concept": concept,
                "null_axis": int(null_text),
                **info,
            }
        )
    return axes, diagnostics


def audit_fixed_trajectories(
    args: argparse.Namespace,
    extractor: HiddenStatesExtractor,
    axes: dict[str, np.ndarray],
    readouts: dict[int, dict[str, np.ndarray]],
    *,
    layers: list[int],
    injection_layer: int,
    intervention_scale: float,
) -> tuple[pd.DataFrame, str, dict[str, Any]]:
    prompt_payload = json.loads(args.prompts.read_text(encoding="utf-8"))
    metadata = {
        str(row.get("id") or row.get("prompt_id")): row
        for row in prompt_payload
    }
    anchor_records, anchor_path = load_anchor_records(
        args.generation_root,
        args.anchor_component,
    )
    anchor_records, excluded_anchor_prompt_ids = filter_valid_anchor_records(
        anchor_records,
        args.anchor_strength,
    )
    if not anchor_records:
        raise ValueError(
            "No non-empty continuation is available for the requested "
            "anchor component and strength"
        )
    selected = balanced_prompt_selection(
        anchor_records,
        metadata,
        args.max_prompts,
    )
    trajectory_provenance = fixed_trajectory_provenance(
        selected,
        extractor.tokenizer,
        anchor_strength=args.anchor_strength,
    )
    array_hashes = {}
    for name, vector in sorted(axes.items()):
        array_hashes[f"axis:{name}"] = hashlib.sha256(
            np.asarray(vector, dtype=np.float32).tobytes()
        ).hexdigest()
    for layer, layer_readouts in sorted(readouts.items()):
        for name, vector in sorted(layer_readouts.items()):
            array_hashes[f"readout:L{layer}:{name}"] = (
                hashlib.sha256(
                    np.asarray(vector, dtype=np.float32).tobytes()
                ).hexdigest()
            )
    work_config = {
        "schema": "matched_null_fixed_trajectory_work_v1",
        "model_id": args.model_id,
        "trajectory_token_set_sha256": trajectory_provenance[
            "trajectory_set_sha256"
        ],
        "array_sha256": array_hashes,
        "layers": layers,
        "injection_layer": injection_layer,
        "intervention_scale": intervention_scale,
        "anchor_component": args.anchor_component,
        "anchor_strength": args.anchor_strength,
        "anchor_candidate_count": len(anchor_records),
        "excluded_empty_anchor_prompt_ids": excluded_anchor_prompt_ids,
        "symmetric_condition_batching": True,
        "trajectory_direction_batch_size": (
            args.trajectory_direction_batch_size
        ),
    }
    work_fingerprint = json_fingerprint(work_config)
    work_dir = args.out_dir / ".work_matched_null_trajectories"
    work_manifest = work_dir / "work_manifest.json"
    if work_manifest.exists():
        existing = json.loads(work_manifest.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != work_fingerprint:
            raise RuntimeError(
                "Existing matched-null trajectory shards use another "
                f"configuration: {work_dir}"
            )
    else:
        atomic_json(
            work_manifest,
            {
                "fingerprint": work_fingerprint,
                "config": work_config,
            },
        )
    interventions = {
        name: make_steering(
            vector,
            layer=injection_layer,
            scale=intervention_scale,
        )
        for name, vector in axes.items()
    }
    rows = []
    anchor_key = str(float(args.anchor_strength))
    for prompt_number, record in enumerate(selected, start=1):
        prompt_id = str(record["prompt_id"])
        shard_path = work_dir / f"{prompt_id}.json"
        if shard_path.exists():
            shard = json.loads(shard_path.read_text(encoding="utf-8"))
            if shard.get("fingerprint") != work_fingerprint:
                raise RuntimeError(f"Stale prompt shard: {shard_path}")
            rows.extend(shard["rows"])
            print(
                "matched-null trajectory "
                f"{prompt_number}/{len(selected)} reused",
                flush=True,
            )
            continue
        prefix = str(record["prompt"])
        continuation = str(record["outputs"][anchor_key]["text"]).strip()
        prompt_rows = []
        intervention_items = list(interventions.items())
        for start in range(
            0,
            len(intervention_items),
            args.trajectory_direction_batch_size,
        ):
            conditions = capture_symmetric_direction_batch_with_fallback(
                extractor,
                prefix,
                continuation,
                layers,
                steerings=intervention_items[
                    start:start + args.trajectory_direction_batch_size
                ],
            )
            for name, (negative, positive) in conditions.items():
                concept, null_text = name.split("__null_")
                for layer in layers:
                    response = 0.5 * (
                        positive[layer].mean(axis=0)
                        - negative[layer].mean(axis=0)
                    )
                    effect = float(response @ readouts[layer][name])
                    prompt_rows.append(
                        {
                            "prompt_id": prompt_id,
                            "prompt_index": int(record["prompt_index"]),
                            "concept": concept,
                            "null_axis": int(null_text),
                            "layer": layer,
                            "projection_effect": effect,
                            "transport_coefficient": (
                                effect / intervention_scale
                            ),
                        }
                    )
        atomic_json(
            shard_path,
            {
                "fingerprint": work_fingerprint,
                "prompt_id": prompt_id,
                "rows": prompt_rows,
            },
        )
        rows.extend(prompt_rows)
        print(
            f"matched-null trajectory {prompt_number}/{len(selected)}",
            flush=True,
        )
    return pd.DataFrame(rows), str(anchor_path), trajectory_provenance


def aggregate_matched_nulls(
    args: argparse.Namespace,
    null_frame: pd.DataFrame,
    *,
    injection_layer: int,
    anchors: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    learned_frame = pd.read_csv(args.learned_transport)
    summaries = []
    axis_rows = []
    for concept in CONCEPTS:
        learned_concept = learned_frame[
            (learned_frame["intervention"] == concept)
            & (learned_frame["readout"] == concept)
        ]
        null_concept = null_frame[null_frame["concept"] == concept]
        for layer in sorted(null_concept["layer"].unique()):
            learned = (
                learned_concept[learned_concept["layer"] == layer]
                .set_index("prompt_id")["transport_coefficient"]
                .abs()
            )
            null = null_concept[
                null_concept["layer"] == layer
            ].pivot(
                index="prompt_id",
                columns="null_axis",
                values="transport_coefficient",
            ).abs()
            shared = learned.index.intersection(null.index)
            learned = learned.loc[shared]
            null = null.loc[shared]
            null_prompt_mean = null.mean(axis=1)
            contrast = learned.to_numpy() - null_prompt_mean.to_numpy()
            inference = paired_mean_inference(
                contrast,
                bootstrap=args.bootstrap,
                permutations=args.permutations,
                seed=stable_seed(
                    args.seed,
                    "matched-null-contrast",
                    concept,
                    int(layer),
                ),
            )
            axis_means = null.mean(axis=0)
            learned_mean = float(learned.mean())
            for axis, value in axis_means.items():
                axis_rows.append(
                    {
                        "concept": concept,
                        "layer": int(layer),
                        "null_axis": int(axis),
                        "mean_transport_capacity": float(value),
                    }
                )
            summaries.append(
                {
                    "concept": concept,
                    "layer": int(layer),
                    "n_prompts": len(shared),
                    "n_null_axes": len(axis_means),
                    "learned_mean_transport_capacity": learned_mean,
                    "matched_null_mean_transport_capacity": float(
                        null_prompt_mean.mean()
                    ),
                    "matched_null_axis_min": float(axis_means.min()),
                    "matched_null_axis_max": float(axis_means.max()),
                    "matched_null_axis_exceed_rate": float(
                        np.mean(axis_means >= learned_mean)
                    ),
                    **{
                        f"learned_minus_null_{key}": value
                        for key, value in inference.items()
                    },
                }
            )
    global_rows = []
    for concept in CONCEPTS:
        anchor = int(anchors[concept])
        eligible_layers = sorted(
            int(layer)
            for layer in null_frame.loc[
                null_frame["concept"].eq(concept),
                "layer",
            ].unique()
            if int(layer) not in {int(injection_layer), anchor}
            and int(layer) > int(injection_layer)
        )
        if len(eligible_layers) < 2:
            raise ValueError(
                f"{concept} has fewer than two propagation-only layers"
            )
        learned_curve = (
            learned_frame[
                (learned_frame["intervention"] == concept)
                & (learned_frame["readout"] == concept)
                & (learned_frame["layer"].isin(eligible_layers))
            ]
            .assign(
                transport_capacity=lambda part: part[
                    "transport_coefficient"
                ].abs()
            )
            .groupby("layer")["transport_capacity"]
            .mean()
            .reindex(eligible_layers)
        )
        if learned_curve.isna().any():
            raise ValueError(f"Incomplete learned curve for {concept}")
        observed = normalized_curve_area(
            np.asarray(eligible_layers),
            learned_curve.to_numpy(),
        )
        null_curves = (
            null_frame[
                null_frame["concept"].eq(concept)
                & null_frame["layer"].isin(eligible_layers)
            ]
            .assign(
                transport_capacity=lambda part: part[
                    "transport_coefficient"
                ].abs()
            )
            .groupby(["null_axis", "layer"])["transport_capacity"]
            .mean()
            .unstack("layer")
            .reindex(columns=eligible_layers)
        )
        if null_curves.isna().any().any():
            raise ValueError(f"Incomplete matched-null curves for {concept}")
        null_statistics = np.asarray([
            normalized_curve_area(
                np.asarray(eligible_layers),
                row.to_numpy(),
            )
            for _, row in null_curves.iterrows()
        ])
        global_rows.append({
            "concept": concept,
            "statistic": "depth_normalized_absolute_transport_auc",
            "injection_layer_excluded": int(injection_layer),
            "readout_anchor_excluded": anchor,
            "propagation_layers": ",".join(
                str(layer) for layer in eligible_layers
            ),
            "n_curve_layers": len(eligible_layers),
            "n_prompts": int(
                null_frame[
                    null_frame["concept"].eq(concept)
                ]["prompt_id"].nunique()
            ),
            "n_null_pipelines": len(null_statistics),
            "learned_curve_statistic": observed,
            "null_mean_curve_statistic": float(null_statistics.mean()),
            "null_min_curve_statistic": float(null_statistics.min()),
            "null_max_curve_statistic": float(null_statistics.max()),
            "null_pipeline_exceed_rate": float(
                np.mean(null_statistics >= observed)
            ),
            "finite_null_p_value": finite_null_p_value(
                observed,
                null_statistics,
            ),
        })
    return (
        pd.DataFrame(summaries),
        pd.DataFrame(axis_rows),
        pd.DataFrame(global_rows),
    )


def write_summary(
    path: Path,
    summary: pd.DataFrame,
    global_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Pipeline-matched pullback null audit",
        "",
        "Each null target permutes surface-score correspondence inside "
        "corpus×generator strata, refits layerwise cross-source readouts, "
        "applies the same Jacobian pullback, and is evaluated on the same "
        "fixed token sequences. Reported capacities are sign-invariant.",
        "",
        "| Concept | Layer | Learned | Matched null | Difference | "
        "95% CI | p | Null exceed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.concept} | {row.layer} | "
            f"{row.learned_mean_transport_capacity:.3f} | "
            f"{row.matched_null_mean_transport_capacity:.3f} | "
            f"{row.learned_minus_null_mean:.3f} | "
            f"[{row.learned_minus_null_ci_low:.3f}, "
            f"{row.learned_minus_null_ci_high:.3f}] | "
            f"{row.learned_minus_null_sign_flip_p:.4f} | "
            f"{row.matched_null_axis_exceed_rate:.3f} |"
        )
    lines.extend([
        "",
        "## Global propagation-curve test",
        "",
        "The injection layer and the downstream readout anchor are excluded. "
        "Each finite-null observation is one complete permuted-score "
        "readout→pullback→fixed-trajectory pipeline; prompts are not treated "
        "as independent null draws.",
        "",
        "| Concept | Layers | Learned AUC | Null mean [range] | "
        "Null pipelines | finite-null p |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for row in global_summary.itertuples(index=False):
        lines.append(
            f"| {row.concept} | {row.propagation_layers} | "
            f"{row.learned_curve_statistic:.3f} | "
            f"{row.null_mean_curve_statistic:.3f} "
            f"[{row.null_min_curve_statistic:.3f}, "
            f"{row.null_max_curve_statistic:.3f}] | "
            f"{row.n_null_pipelines} | "
            f"{row.finite_null_p_value:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.null_controls < 2:
        raise ValueError("--null-controls must be at least two")
    if args.pullback_batch_size < 1 or args.readout_chunk_size < 1:
        raise ValueError("batch and chunk sizes must be positive")
    if args.trajectory_direction_batch_size < 1:
        raise ValueError(
            "--trajectory-direction-batch-size must be positive"
        )
    started = datetime.now(timezone.utc)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    layers = discover_layers(args)
    (
        injection_layer,
        intervention_scale,
        anchors,
        canonical_hashes,
    ) = canonical_configuration(args.canonical_root)
    if injection_layer not in layers:
        raise ValueError("State layers omit the canonical injection layer")
    if any(layer not in layers for layer in anchors.values()):
        raise ValueError("State layers omit a canonical downstream anchor")

    state_hashes = {
        str(
            state_path(
                args.states_root,
                args.model_id,
                args.dataset_tag,
                "mean_pool",
                layer,
            )
        ): sha256_file(
            state_path(
                args.states_root,
                args.model_id,
                args.dataset_tag,
                "mean_pool",
                layer,
            )
        )
        for layer in layers
    }
    pipeline_fingerprint = json_fingerprint({
        "schema": "pipeline_matched_pullback_null_fit_v2",
        "model_id": args.model_id,
        "pairs_sha256": sha256_file(args.pairs),
        "features_sha256": sha256_file(args.features),
        "states_sha256": state_hashes,
        "canonical_sha256": canonical_hashes,
        "layers": layers,
        "injection_layer": injection_layer,
        "downstream_anchors": anchors,
        "null_controls": args.null_controls,
        "null_k": args.null_k,
        "max_pairs": args.max_pairs,
        "pullback_batch_size": args.pullback_batch_size,
        "readout_chunk_size": args.readout_chunk_size,
        "seed": args.seed,
    })
    readout_path = args.out_dir / "matched_null_readouts.pt"
    pullback_path = args.out_dir / "matched_null_pullbacks.pt"
    cached_readouts = (
        torch.load(readout_path, map_location="cpu", weights_only=False)
        if readout_path.is_file()
        else {}
    )
    cached_pullbacks = (
        torch.load(pullback_path, map_location="cpu", weights_only=False)
        if pullback_path.is_file()
        else {}
    )
    stages_reused = (
        cached_readouts.get("pipeline_fingerprint")
        == pipeline_fingerprint
        and cached_pullbacks.get("pipeline_fingerprint")
        == pipeline_fingerprint
    )

    model_path = Path(args.model).expanduser()
    model_name = (
        str(model_path.resolve()) if model_path.is_dir() else args.model
    )
    extractor = HiddenStatesExtractor(
        model_name,
        device=args.device,
        dtype=args.torch_dtype,
        attention_backend=args.attention_backend,
    )
    extractor.model.requires_grad_(False)
    if stages_reused:
        readouts = {
            int(layer): {
                name: vector.float().numpy()
                for name, vector in values.items()
            }
            for layer, values in cached_readouts["readouts"].items()
        }
        axes = {
            name: vector.float().numpy()
            for name, vector in cached_pullbacks["axes"].items()
        }
        print(
            "reused matched-null readouts and pullback axes",
            flush=True,
        )
    else:
        frame = pd.read_csv(args.features)
        (
            readouts,
            readout_diagnostics,
            built_state_hashes,
        ) = build_permuted_readouts(
            args,
            layers,
            frame,
        )
        if any(
            built_state_hashes.get(path) != digest
            for path, digest in state_hashes.items()
        ):
            raise RuntimeError("State files changed during null fitting")
        atomic_torch_save(
            readout_path,
            {
                "pipeline_fingerprint": pipeline_fingerprint,
                "layers": layers,
                "readouts": {
                    layer: {
                        name: torch.from_numpy(vector).float()
                        for name, vector in values.items()
                    }
                    for layer, values in readouts.items()
                },
            },
        )
        pd.DataFrame(readout_diagnostics).to_csv(
            args.out_dir / "matched_null_readout_diagnostics.csv",
            index=False,
        )
        axes, pullback_diagnostics = pullback_null_readouts(
            args,
            extractor,
            readouts,
            injection_layer=injection_layer,
            anchors=anchors,
        )
        atomic_torch_save(
            pullback_path,
            {
                "pipeline_fingerprint": pipeline_fingerprint,
                "injection_layer": injection_layer,
                "intervention_scale": intervention_scale,
                "axes": {
                    name: torch.from_numpy(vector).float()
                    for name, vector in axes.items()
                },
            },
        )
        pd.DataFrame(pullback_diagnostics).to_csv(
            args.out_dir / "matched_null_pullback_diagnostics.csv",
            index=False,
        )

    null_frame, anchor_path, trajectory_provenance = audit_fixed_trajectories(
        args,
        extractor,
        axes,
        readouts,
        layers=[layer for layer in layers if layer >= injection_layer],
        injection_layer=injection_layer,
        intervention_scale=intervention_scale,
    )
    learned_manifest_path = args.learned_transport.parent / "manifest.json"
    if not learned_manifest_path.is_file():
        raise FileNotFoundError(
            f"Missing learned-transport manifest: {learned_manifest_path}"
        )
    learned_manifest = json.loads(
        learned_manifest_path.read_text(encoding="utf-8")
    )
    learned_trajectory_hash = learned_manifest.get(
        "trajectory_token_set_sha256"
    )
    current_trajectory_hash = trajectory_provenance[
        "trajectory_set_sha256"
    ]
    if learned_trajectory_hash != current_trajectory_hash:
        raise RuntimeError(
            "Learned and matched-null audits do not use identical fixed "
            "prompt/continuation token trajectories: "
            f"{learned_trajectory_hash!r} != {current_trajectory_hash!r}"
        )
    trajectory_path = args.out_dir / "fixed_trajectory_provenance.json"
    atomic_json(trajectory_path, trajectory_provenance)
    null_frame.to_csv(
        args.out_dir / "per_prompt_matched_null_transport.csv",
        index=False,
    )
    summary, axis_summary, global_summary = aggregate_matched_nulls(
        args,
        null_frame,
        injection_layer=injection_layer,
        anchors=anchors,
    )
    summary.to_csv(
        args.out_dir / "matched_null_transport_summary.csv",
        index=False,
    )
    axis_summary.to_csv(
        args.out_dir / "matched_null_axis_statistics.csv",
        index=False,
    )
    global_summary.to_csv(
        args.out_dir / "global_curve_statistics.csv",
        index=False,
    )
    write_summary(
        args.out_dir / "analysis_summary.md",
        summary,
        global_summary,
    )
    readout_diagnostic_frame = pd.read_csv(
        args.out_dir / "matched_null_readout_diagnostics.csv"
    )
    distinct_permutations = int(
        readout_diagnostic_frame["permutation_sha256"].nunique()
    )
    if distinct_permutations != args.null_controls:
        raise RuntimeError(
            "Saved readout diagnostics do not contain one distinct "
            "permutation per null pipeline"
        )

    finished = datetime.now(timezone.utc)
    atomic_json(
        args.out_dir / "manifest.json",
        {
            "schema": "pipeline_matched_pullback_null_audit_v2",
            "status": "complete",
            "model": model_source_fingerprint(model_name),
            "model_id": args.model_id,
            "pairs": str(args.pairs),
            "pairs_sha256": sha256_file(args.pairs),
            "features": str(args.features),
            "states_sha256": state_hashes,
            "canonical_sha256": canonical_hashes,
            "prompts": str(args.prompts),
            "prompts_sha256": sha256_file(args.prompts),
            "anchor_generation": anchor_path,
            "anchor_generation_sha256": sha256_file(Path(anchor_path)),
            "fixed_trajectory_provenance": str(trajectory_path),
            "fixed_trajectory_provenance_sha256": sha256_file(
                trajectory_path
            ),
            "trajectory_token_set_sha256": current_trajectory_hash,
            "learned_trajectory_token_set_sha256": learned_trajectory_hash,
            "trajectory_identity_verified": True,
            "learned_transport": str(args.learned_transport),
            "learned_transport_sha256": sha256_file(args.learned_transport),
            "injection_layer": injection_layer,
            "downstream_anchors": anchors,
            "layers": layers,
            "intervention_scale": intervention_scale,
            "null_controls": args.null_controls,
            "distinct_null_permutations": distinct_permutations,
            "null_permutation_fixed_point_fraction": {
                "min": float(
                    readout_diagnostic_frame[
                        "fixed_point_fraction"
                    ].min()
                ),
                "mean": float(
                    readout_diagnostic_frame[
                        "fixed_point_fraction"
                    ].mean()
                ),
                "max": float(
                    readout_diagnostic_frame[
                        "fixed_point_fraction"
                    ].max()
                ),
            },
            "symmetric_condition_batching": True,
            "trajectory_direction_batch_size": (
                args.trajectory_direction_batch_size
            ),
            "null_fit_pipeline_fingerprint": pipeline_fingerprint,
            "readout_and_pullback_stages_reused": stages_reused,
            "null_definition": (
                "surface-score correspondence permuted within "
                "corpus-by-generator strata, followed by the identical "
                "readout-fit, Jacobian-pullback, and fixed-trajectory audit"
            ),
            "global_curve_statistic": (
                "depth-normalized trapezoidal area of absolute transport "
                "capacity over post-injection propagation layers, excluding "
                "the injection layer and downstream readout anchor"
            ),
            "global_curve_null_unit": (
                "one complete permuted-score readout-to-pullback-to-fixed-"
                "trajectory pipeline"
            ),
            "max_pairs": args.max_pairs,
            "max_prompts": args.max_prompts,
            "seed": args.seed,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "torch_version": torch.__version__,
            "started_at_utc": started.isoformat(),
            "finished_at_utc": finished.isoformat(),
            "duration_seconds": (finished - started).total_seconds(),
            "command": " ".join(sys.argv),
        },
    )
    print(f"Wrote matched-null audit to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
