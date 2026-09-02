#!/usr/bin/env python3
"""Trace an injected concept into independently validated layerwise readouts.

An L8 lexical or syntactic perturbation can rotate while moving through the
network. Projecting every downstream response onto the original L8 vector
would call that rotation "loss". This audit instead evaluates the response
against the lexical/syntactic activation pattern fitted independently at each
downstream layer, with an ensemble of orthogonal random interventions.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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
from src.analysis.causal_pullback import normalized_curve_area
from src.analysis.causal_linguistics import paired_mean_inference
from src.analysis.hidden_propagation import orthogonal_random_directions
from src.extraction.hidden_states import HiddenStatesExtractor
from src.runtime import model_source_fingerprint, stable_seed
from src.steering import MeanDiffSteering


CONCEPTS = ("lexical", "syntactic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--injection-directions-dir", type=Path, required=True)
    parser.add_argument("--layerwise-root", type=Path, required=True)
    parser.add_argument(
        "--mean-difference-checkpoint",
        type=Path,
        help=(
            "Raw human-minus-machine mean-difference direction at the "
            "injection layer, used only as a direction ablation."
        ),
    )
    parser.add_argument("--anchor-component", default="random")
    parser.add_argument("--anchor-strength", type=float, default=-1.0)
    parser.add_argument("--max-prompts", type=int, default=12)
    parser.add_argument("--random-controls", type=int, default=8)
    parser.add_argument(
        "--trajectory-direction-batch-size",
        type=int,
        default=4,
    )
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
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help=(
            "Recompute sign-invariant summaries from an existing "
            "per_prompt_transport.csv without loading the model."
        ),
    )
    return parser.parse_args()


def load_checkpoint_vector(path: Path) -> tuple[np.ndarray, int, float]:
    steering = MeanDiffSteering.load(path)
    vector = steering.get_direction().float().cpu().numpy()
    vector /= max(float(np.linalg.norm(vector)), 1e-12)
    return (
        vector,
        int(steering.layer),
        float(steering.projection_std.item()),
    )


def canonical_anchor(path: Path) -> int:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = dict(payload.get("metadata", {}))
    selected = metadata.get("selected_checkpoint") or {}
    fit_metadata = selected.get("fit_metadata") or {}
    anchor = fit_metadata.get("downstream_layer")
    if anchor is None:
        raise ValueError(
            f"{path} does not record a frozen downstream readout anchor"
        )
    return int(anchor)


def make_steering(
    vector: np.ndarray,
    *,
    layer: int,
    scale: float,
) -> MeanDiffSteering:
    steering = MeanDiffSteering(hidden_size=len(vector), layer=layer)
    with torch.no_grad():
        steering.v_src.copy_(torch.from_numpy(vector).float())
        steering.is_fitted.fill_(True)
        steering.projection_std.fill_(scale)
    return steering


def load_layerwise_readouts(
    root: Path,
) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, str]]:
    readouts = {}
    hashes = {}
    for layer_dir in sorted(root.glob("layer[0-9][0-9]")):
        layer = int(layer_dir.name.removeprefix("layer"))
        layer_readouts = {}
        for concept in CONCEPTS:
            checkpoint = layer_dir / "directions" / f"{concept}.pt"
            vector, checkpoint_layer, _ = load_checkpoint_vector(checkpoint)
            if checkpoint_layer != layer:
                raise ValueError(f"Layer mismatch in {checkpoint}")
            layer_readouts[concept] = vector
            hashes[str(checkpoint)] = sha256_file(checkpoint)
        readouts[layer] = layer_readouts
    if not readouts:
        raise FileNotFoundError(f"No layerwise readouts under {root}")
    return readouts, hashes


def aggregate(
    frame: pd.DataFrame,
    *,
    bootstrap: int,
    permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    learned_rows = []
    contrast_rows = []
    random_rows = frame[
        frame["intervention_kind"] == "random"
    ].copy()
    random_rows["transport_capacity"] = random_rows[
        "transport_coefficient"
    ].abs()
    random_prompt = (
        random_rows.groupby(
            ["prompt_id", "layer", "readout"],
            as_index=False,
        )["transport_capacity"]
        .mean()
        .rename(columns={
            "transport_capacity": "random_prompt_mean",
        })
    )
    random_direction = (
        random_rows.groupby(
            ["intervention", "layer", "readout"],
            as_index=False,
        )["transport_capacity"]
        .mean()
        .rename(columns={
            "transport_capacity": "transport_coefficient",
        })
    )

    for concept in CONCEPTS:
        target = frame[
            (frame["intervention"] == concept)
            & (frame["readout"] == concept)
        ]
        for layer, part in target.groupby("layer", sort=True):
            learned_capacity = part["transport_coefficient"].abs()
            inference = paired_mean_inference(
                learned_capacity.to_numpy(),
                bootstrap=bootstrap,
                permutations=permutations,
                seed=stable_seed(seed, "learned", concept, int(layer)),
            )
            learned_rows.append({
                "concept": concept,
                "layer": int(layer),
                "n_prompts": int(part["prompt_id"].nunique()),
                "signed_mean": float(
                    part["transport_coefficient"].mean()
                ),
                **inference,
                "alignment_mean": float(part["alignment"].mean()),
                "response_norm_coefficient_mean": float(
                    part["response_norm_coefficient"].mean()
                ),
            })

            paired = part[
                ["prompt_id", "transport_coefficient"]
            ].merge(
                random_prompt[
                    (random_prompt["layer"] == layer)
                    & (random_prompt["readout"] == concept)
                ][["prompt_id", "random_prompt_mean"]],
                on="prompt_id",
                validate="one_to_one",
            )
            paired["learned_capacity"] = paired[
                "transport_coefficient"
            ].abs()
            values = (
                paired["learned_capacity"]
                - paired["random_prompt_mean"]
            ).to_numpy()
            contrast = paired_mean_inference(
                values,
                bootstrap=bootstrap,
                permutations=permutations,
                seed=stable_seed(seed, "contrast", concept, int(layer)),
            )
            direction_values = random_direction[
                (random_direction["layer"] == layer)
                & (random_direction["readout"] == concept)
            ]["transport_coefficient"].to_numpy()
            contrast_rows.append({
                "concept": concept,
                "layer": int(layer),
                "reference": "orthogonal_random_ensemble",
                "n_prompts": len(paired),
                "n_random_directions": len(direction_values),
                "learned_mean": float(
                    paired["learned_capacity"].mean()
                ),
                "random_ensemble_mean": float(
                    paired["random_prompt_mean"].mean()
                ),
                "random_direction_min": float(direction_values.min()),
                "random_direction_max": float(direction_values.max()),
                "random_direction_exceed_rate": float(
                    np.mean(
                        direction_values
                        >= paired["learned_capacity"].mean()
                    )
                ),
                **contrast,
            })
    return (
        pd.DataFrame(learned_rows),
        pd.DataFrame(contrast_rows),
        random_direction,
    )


def aggregate_direction_ablations(
    frame: pd.DataFrame,
    *,
    injection_layer: int,
    anchors: dict[str, int],
    bootstrap: int,
    permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare pullbacks with encoding, source, and random directions.

    The global unit for deterministic direction comparisons is a prompt-level
    area under the propagation-only curve. The injection site and the frozen
    downstream readout anchor are excluded before integration.
    """
    layer_rows = []
    global_rows = []
    baselines = (
        ("encoding", lambda concept: f"encoding_{concept}"),
        ("raw_mean_difference", lambda _concept: "mean_difference"),
        ("nullspace_mean_difference", lambda _concept: "source_nullspace"),
        ("projected_random_ensemble", lambda _concept: None),
    )
    capacity = frame.copy()
    capacity["transport_capacity"] = capacity[
        "transport_coefficient"
    ].abs()

    for concept in CONCEPTS:
        anchor = int(anchors[concept])
        propagation_layers = sorted(
            int(layer)
            for layer in capacity["layer"].unique()
            if int(layer) > int(injection_layer)
            and int(layer) != anchor
        )
        if len(propagation_layers) < 2:
            raise ValueError(
                f"{concept} has fewer than two propagation-only layers"
            )
        learned = capacity[
            capacity["intervention"].eq(concept)
            & capacity["readout"].eq(concept)
            & capacity["layer"].isin(propagation_layers)
        ]
        learned_prompt = learned.pivot(
            index="prompt_id",
            columns="layer",
            values="transport_capacity",
        ).reindex(columns=propagation_layers)
        if learned_prompt.isna().any().any():
            raise ValueError(f"Incomplete pullback curves for {concept}")
        learned_prompt_auc = pd.Series(
            [
                normalized_curve_area(
                    np.asarray(propagation_layers),
                    row.to_numpy(),
                )
                for _, row in learned_prompt.iterrows()
            ],
            index=learned_prompt.index,
            name="learned_auc",
        )

        for family, intervention_for in baselines:
            intervention = intervention_for(concept)
            if family == "projected_random_ensemble":
                baseline = (
                    capacity[
                        capacity["intervention_kind"].eq("random")
                        & capacity["readout"].eq(concept)
                        & capacity["layer"].isin(propagation_layers)
                    ]
                    .groupby(["prompt_id", "layer"], as_index=False)[
                        "transport_capacity"
                    ]
                    .mean()
                )
                n_directions = int(
                    capacity[
                        capacity["intervention_kind"].eq("random")
                    ]["intervention"].nunique()
                )
                display_intervention = "random_ensemble_mean"
            else:
                baseline = capacity[
                    capacity["intervention"].eq(intervention)
                    & capacity["readout"].eq(concept)
                    & capacity["layer"].isin(propagation_layers)
                ]
                n_directions = 1
                display_intervention = str(intervention)
            if baseline.empty:
                raise ValueError(
                    f"Missing {family} direction ablation for {concept}"
                )
            baseline_prompt = baseline.pivot(
                index="prompt_id",
                columns="layer",
                values="transport_capacity",
            ).reindex(columns=propagation_layers)
            shared = learned_prompt_auc.index.intersection(
                baseline_prompt.index
            )
            baseline_prompt = baseline_prompt.loc[shared]
            if baseline_prompt.isna().any().any():
                raise ValueError(
                    f"Incomplete {family} curves for {concept}"
                )
            baseline_prompt_auc = pd.Series(
                [
                    normalized_curve_area(
                        np.asarray(propagation_layers),
                        row.to_numpy(),
                    )
                    for _, row in baseline_prompt.iterrows()
                ],
                index=shared,
                name="baseline_auc",
            )
            differences = (
                learned_prompt_auc.loc[shared] - baseline_prompt_auc
            ).to_numpy()
            inference = paired_mean_inference(
                differences,
                bootstrap=bootstrap,
                permutations=permutations,
                seed=stable_seed(
                    seed,
                    "direction_ablation",
                    concept,
                    family,
                ),
            )
            global_rows.append({
                "concept": concept,
                "reference_family": family,
                "reference_intervention": display_intervention,
                "propagation_layers": ",".join(
                    str(layer) for layer in propagation_layers
                ),
                "injection_layer_excluded": int(injection_layer),
                "readout_anchor_excluded": anchor,
                "n_prompts": len(shared),
                "n_reference_directions": n_directions,
                "learned_curve_statistic": float(
                    learned_prompt_auc.loc[shared].mean()
                ),
                "reference_curve_statistic": float(
                    baseline_prompt_auc.mean()
                ),
                **{
                    f"learned_minus_reference_{key}": value
                    for key, value in inference.items()
                },
            })

            learned_layer = (
                learned.groupby("layer")["transport_capacity"].mean()
            )
            reference_layer = (
                baseline.groupby("layer")["transport_capacity"].mean()
            )
            for layer in propagation_layers:
                layer_rows.append({
                    "concept": concept,
                    "reference_family": family,
                    "layer": layer,
                    "learned_mean_transport_capacity": float(
                        learned_layer.loc[layer]
                    ),
                    "reference_mean_transport_capacity": float(
                        reference_layer.loc[layer]
                    ),
                    "learned_minus_reference": float(
                        learned_layer.loc[layer] - reference_layer.loc[layer]
                    ),
                    "n_prompts": len(shared),
                    "n_reference_directions": n_directions,
                })
    return pd.DataFrame(layer_rows), pd.DataFrame(global_rows)


def write_summary(
    path: Path,
    learned: pd.DataFrame,
    contrasts: pd.DataFrame,
    *,
    injection_layer: int,
    intervention_scale: float,
) -> None:
    merged = learned.merge(
        contrasts,
        on=["concept", "layer", "n_prompts"],
        suffixes=("_learned", "_contrast"),
        validate="one_to_one",
    )
    lines = [
        "# Layerwise concept-transport audit",
        "",
        "Each downstream projection uses a lexical or syntactic activation "
        "pattern fitted independently at that layer. The transport "
        "coefficient divides the symmetric hidden response by the original "
        f"L{injection_layer} intervention norm ({intervention_scale:.4f}). "
        "The table reports absolute transport capacity because the pole of "
        "a symmetric intervention axis is arbitrary after development-set "
        "orientation. "
        "Random controls are orthogonal to the source and both concept "
        "directions at the injection layer.",
        "",
        "| Concept | Layer | Learned transport | Random ensemble | "
        "Advantage | 95% CI | Sign-flip p | Random exceed rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in merged.itertuples(index=False):
        lines.append(
            f"| {row.concept} | {row.layer} | "
            f"{row.mean_learned:.3f} | {row.random_ensemble_mean:.3f} | "
            f"{row.mean_contrast:.3f} | "
            f"[{row.ci_low_contrast:.3f}, {row.ci_high_contrast:.3f}] | "
            f"{row.sign_flip_p_contrast:.4f} | "
            f"{row.random_direction_exceed_rate:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_ablation_summary(
    path: Path,
    global_summary: pd.DataFrame,
) -> None:
    lines = [
        "# Direction ablations",
        "",
        "Each statistic is the depth-normalized area under the absolute "
        "transport-capacity curve after excluding the injection layer and "
        "the frozen downstream readout anchor.",
        "",
        "| Concept | Reference | Pullback AUC | Reference AUC | "
        "Difference | 95% CI | p |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in global_summary.itertuples(index=False):
        lines.append(
            f"| {row.concept} | {row.reference_family} | "
            f"{row.learned_curve_statistic:.3f} | "
            f"{row.reference_curve_statistic:.3f} | "
            f"{row.learned_minus_reference_mean:.3f} | "
            f"[{row.learned_minus_reference_ci_low:.3f}, "
            f"{row.learned_minus_reference_ci_high:.3f}] | "
            f"{row.learned_minus_reference_sign_flip_p:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.trajectory_direction_batch_size < 1:
        raise ValueError(
            "--trajectory-direction-batch-size must be positive"
        )
    started = datetime.now(timezone.utc)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        frame_path = args.out_dir / "per_prompt_transport.csv"
        manifest_path = args.out_dir / "manifest.json"
        if not frame_path.is_file() or not manifest_path.is_file():
            raise FileNotFoundError(
                "Aggregate-only mode requires per_prompt_transport.csv "
                "and manifest.json in --out-dir"
            )
        frame = pd.read_csv(frame_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        learned, contrasts, random_direction = aggregate(
            frame,
            bootstrap=args.bootstrap,
            permutations=args.permutations,
            seed=args.seed,
        )
        learned.to_csv(
            args.out_dir / "learned_transport.csv",
            index=False,
        )
        contrasts.to_csv(
            args.out_dir / "random_ensemble_contrasts.csv",
            index=False,
        )
        random_direction.to_csv(
            args.out_dir / "random_direction_transport.csv",
            index=False,
        )
        write_summary(
            args.out_dir / "analysis_summary.md",
            learned,
            contrasts,
            injection_layer=int(manifest["injection_layer"]),
            intervention_scale=float(manifest["intervention_scale"]),
        )
        if "downstream_anchors" in manifest:
            ablation_layers, ablation_global = (
                aggregate_direction_ablations(
                    frame,
                    injection_layer=int(manifest["injection_layer"]),
                    anchors={
                        concept: int(layer)
                        for concept, layer in manifest[
                            "downstream_anchors"
                        ].items()
                    },
                    bootstrap=args.bootstrap,
                    permutations=args.permutations,
                    seed=args.seed,
                )
            )
            ablation_layers.to_csv(
                args.out_dir / "direction_ablation_layerwise.csv",
                index=False,
            )
            ablation_global.to_csv(
                args.out_dir / "direction_ablation_global.csv",
                index=False,
            )
            write_ablation_summary(
                args.out_dir / "direction_ablation_summary.md",
                ablation_global,
            )
        manifest["transport_summary"] = (
            "absolute learned transport minus prompt-matched mean absolute "
            "transport of the orthogonal random ensemble"
        )
        manifest["aggregate_command"] = " ".join(sys.argv)
        manifest["aggregate_updated_at_utc"] = datetime.now(
            timezone.utc
        ).isoformat()
        atomic_json(manifest_path, manifest)
        print(
            f"Recomputed transport summaries in {args.out_dir}",
            flush=True,
        )
        return

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

    injection_vectors = {}
    injection_hashes = {}
    downstream_anchors = {}
    injection_layer = None
    intervention_scale = None
    for name in ("source", *CONCEPTS):
        path = args.injection_directions_dir / f"{name}.pt"
        vector, layer, scale = load_checkpoint_vector(path)
        if injection_layer is None:
            injection_layer = layer
            intervention_scale = scale
        if layer != injection_layer:
            raise ValueError("Injection directions do not share a layer")
        if not np.isclose(scale, intervention_scale, rtol=1e-6):
            raise ValueError("Injection directions do not share a scale")
        injection_vectors[name] = vector
        injection_hashes[str(path)] = sha256_file(path)
        if name in CONCEPTS:
            downstream_anchors[name] = canonical_anchor(path)
    assert injection_layer is not None
    assert intervention_scale is not None
    if args.mean_difference_checkpoint is None:
        raise ValueError(
            "--mean-difference-checkpoint is required outside "
            "--aggregate-only mode"
        )
    (
        mean_difference_vector,
        mean_difference_layer,
        _mean_difference_natural_scale,
    ) = load_checkpoint_vector(args.mean_difference_checkpoint)
    if mean_difference_layer != injection_layer:
        raise ValueError(
            "Raw mean-difference baseline does not share the injection layer"
        )

    layerwise, layerwise_hashes = load_layerwise_readouts(
        args.layerwise_root,
    )
    layers = sorted(
        layer
        for layer in layerwise
        if layer >= injection_layer
    )
    if injection_layer not in layers:
        raise ValueError("Layerwise readouts omit the injection layer")
    if any(anchor not in layers for anchor in downstream_anchors.values()):
        raise ValueError("Frozen downstream anchor is absent from readouts")
    encoding_vectors = {
        concept: layerwise[injection_layer][concept]
        for concept in CONCEPTS
    }
    random_vectors = orthogonal_random_directions(
        {
            **injection_vectors,
            "raw_mean_difference": mean_difference_vector,
            **{
                f"encoding_{concept}": vector
                for concept, vector in encoding_vectors.items()
            },
        },
        count=args.random_controls,
        seed=args.seed,
    )
    interventions = {
        concept: make_steering(
            injection_vectors[concept],
            layer=injection_layer,
            scale=intervention_scale,
        )
        for concept in CONCEPTS
    }
    interventions.update({
        f"encoding_{concept}": make_steering(
            vector,
            layer=injection_layer,
            scale=intervention_scale,
        )
        for concept, vector in encoding_vectors.items()
    })
    interventions["mean_difference"] = make_steering(
        mean_difference_vector,
        layer=injection_layer,
        scale=intervention_scale,
    )
    interventions["source_nullspace"] = make_steering(
        injection_vectors["source"],
        layer=injection_layer,
        scale=intervention_scale,
    )
    interventions.update({
        f"random_{index:02d}": make_steering(
            vector,
            layer=injection_layer,
            scale=intervention_scale,
        )
        for index, vector in enumerate(random_vectors)
    })
    intervention_kinds = {
        **{concept: "pullback" for concept in CONCEPTS},
        **{
            f"encoding_{concept}": "encoding"
            for concept in CONCEPTS
        },
        "mean_difference": "raw_mean_difference",
        "source_nullspace": "nullspace_mean_difference",
        **{
            f"random_{index:02d}": "random"
            for index in range(len(random_vectors))
        },
    }

    model_path = Path(args.model).expanduser()
    model_name = (
        str(model_path.resolve())
        if model_path.is_dir()
        else args.model
    )
    extractor = HiddenStatesExtractor(
        model_name,
        device=args.device,
        dtype=args.torch_dtype,
        attention_backend=args.attention_backend,
    )
    trajectory_provenance = fixed_trajectory_provenance(
        selected,
        extractor.tokenizer,
        anchor_strength=args.anchor_strength,
    )
    trajectory_path = args.out_dir / "fixed_trajectory_provenance.json"
    atomic_json(trajectory_path, trajectory_provenance)
    config = {
        "schema": "layerwise_concept_transport_v2",
        "model": model_source_fingerprint(model_name),
        "prompts": str(args.prompts),
        "prompts_sha256": sha256_file(args.prompts),
        "anchor_generation": str(anchor_path),
        "anchor_generation_sha256": sha256_file(anchor_path),
        "anchor_component": args.anchor_component,
        "anchor_strength": args.anchor_strength,
        "anchor_candidate_count": len(anchor_records),
        "excluded_empty_anchor_prompt_ids": excluded_anchor_prompt_ids,
        "injection_layer": injection_layer,
        "intervention_scale": intervention_scale,
        "injection_direction_sha256": injection_hashes,
        "mean_difference_checkpoint": str(
            args.mean_difference_checkpoint
        ),
        "mean_difference_checkpoint_sha256": sha256_file(
            args.mean_difference_checkpoint
        ),
        "downstream_anchors": downstream_anchors,
        "direction_ablation_families": {
            "pullback": list(CONCEPTS),
            "encoding": [
                f"encoding_{concept}" for concept in CONCEPTS
            ],
            "raw_mean_difference": ["mean_difference"],
            "nullspace_mean_difference": ["source_nullspace"],
            "projected_random": [
                f"random_{index:02d}"
                for index in range(len(random_vectors))
            ],
        },
        "layerwise_direction_sha256": layerwise_hashes,
        "layers": layers,
        "random_controls": args.random_controls,
        "symmetric_condition_batching": True,
        "trajectory_direction_batch_size": (
            args.trajectory_direction_batch_size
        ),
        "transport_summary": (
            "absolute learned transport minus prompt-matched mean absolute "
            "transport of the orthogonal random ensemble"
        ),
        "selected_prompt_ids": [
            str(row["prompt_id"]) for row in selected
        ],
        "fixed_trajectory_provenance": str(trajectory_path),
        "fixed_trajectory_provenance_sha256": sha256_file(trajectory_path),
        "trajectory_token_set_sha256": trajectory_provenance[
            "trajectory_set_sha256"
        ],
        "device": args.device,
        "torch_dtype": args.torch_dtype,
        "attention_backend": args.attention_backend,
        "bootstrap": args.bootstrap,
        "permutations": args.permutations,
        "seed": args.seed,
    }
    fingerprint = json_fingerprint(config)
    work_dir = args.out_dir / ".work_layerwise_transport"
    work_manifest = work_dir / "work_manifest.json"
    if work_manifest.exists():
        existing = json.loads(work_manifest.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != fingerprint:
            raise RuntimeError(
                f"Existing work directory has another configuration: {work_dir}"
            )
    else:
        atomic_json(
            work_manifest,
            {"fingerprint": fingerprint, "config": config},
        )

    anchor_key = str(float(args.anchor_strength))
    all_rows = []
    for prompt_number, record in enumerate(selected, start=1):
        prompt_id = str(record["prompt_id"])
        shard_path = work_dir / f"{prompt_id}.json"
        if shard_path.exists():
            shard = json.loads(shard_path.read_text(encoding="utf-8"))
            if shard.get("fingerprint") != fingerprint:
                raise RuntimeError(f"Stale prompt shard: {shard_path}")
            all_rows.extend(shard["rows"])
            print(
                f"[{prompt_number}/{len(selected)}] reused {prompt_id}",
                flush=True,
            )
            continue

        prefix = str(record["prompt"])
        continuation = str(
            record["outputs"][anchor_key]["text"]
        ).strip()
        prompt_rows = []
        prompt_metadata = metadata.get(prompt_id, {})
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
            for intervention, (negative, positive) in conditions.items():
                kind = intervention_kinds[intervention]
                for layer in layers:
                    response = 0.5 * (
                        positive[layer].mean(axis=0)
                        - negative[layer].mean(axis=0)
                    )
                    response_norm = float(np.linalg.norm(response))
                    for readout in CONCEPTS:
                        effect = float(
                            response @ layerwise[layer][readout]
                        )
                        prompt_rows.append({
                            "prompt_id": prompt_id,
                            "prompt_index": int(record["prompt_index"]),
                            "source": str(
                                prompt_metadata.get("source", "unknown")
                            ),
                            "domain": str(
                                prompt_metadata.get("domain", "unknown")
                            ),
                            "intervention": intervention,
                            "intervention_kind": kind,
                            "readout": readout,
                            "layer": layer,
                            "projection_effect": effect,
                            "transport_coefficient": (
                                effect / intervention_scale
                            ),
                            "response_norm": response_norm,
                            "response_norm_coefficient": (
                                response_norm / intervention_scale
                            ),
                            "alignment": (
                                effect / response_norm
                                if response_norm > 1e-12
                                else 0.0
                            ),
                        })
        atomic_json(
            shard_path,
            {
                "fingerprint": fingerprint,
                "prompt_id": prompt_id,
                "rows": prompt_rows,
            },
        )
        all_rows.extend(prompt_rows)
        print(
            f"[{prompt_number}/{len(selected)}] completed {prompt_id}",
            flush=True,
        )

    frame = pd.DataFrame(all_rows)
    frame.to_csv(args.out_dir / "per_prompt_transport.csv", index=False)
    learned, contrasts, random_direction = aggregate(
        frame,
        bootstrap=args.bootstrap,
        permutations=args.permutations,
        seed=args.seed,
    )
    learned.to_csv(args.out_dir / "learned_transport.csv", index=False)
    contrasts.to_csv(
        args.out_dir / "random_ensemble_contrasts.csv",
        index=False,
    )
    random_direction.to_csv(
        args.out_dir / "random_direction_transport.csv",
        index=False,
    )
    ablation_layers, ablation_global = aggregate_direction_ablations(
        frame,
        injection_layer=injection_layer,
        anchors=downstream_anchors,
        bootstrap=args.bootstrap,
        permutations=args.permutations,
        seed=args.seed,
    )
    ablation_layers.to_csv(
        args.out_dir / "direction_ablation_layerwise.csv",
        index=False,
    )
    ablation_global.to_csv(
        args.out_dir / "direction_ablation_global.csv",
        index=False,
    )
    write_summary(
        args.out_dir / "analysis_summary.md",
        learned,
        contrasts,
        injection_layer=injection_layer,
        intervention_scale=intervention_scale,
    )
    write_ablation_summary(
        args.out_dir / "direction_ablation_summary.md",
        ablation_global,
    )

    finished = datetime.now(timezone.utc)
    atomic_json(
        args.out_dir / "manifest.json",
        {
            **config,
            "fingerprint": fingerprint,
            "status": "complete",
            "n_prompts": len(selected),
            "n_rows": len(frame),
            "global_curve_statistic": (
                "depth-normalized trapezoidal area of absolute transport "
                "capacity over post-injection propagation layers, excluding "
                "the injection layer and frozen downstream readout anchor"
            ),
            "model_load": extractor.model_load_info,
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
    print(
        f"Wrote layerwise concept transport to {args.out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
