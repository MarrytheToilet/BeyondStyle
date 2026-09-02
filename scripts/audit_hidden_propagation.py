#!/usr/bin/env python3
"""Teacher-forced audit of how steering perturbations propagate across layers.

The audit holds every token fixed and runs the same prompt-continuation
sequence under -1, 0, and +1 interventions. This removes sampling divergence
and directly tests whether the injected residual-stream displacement survives
through later decoder layers.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
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

from src.analysis.causal_linguistics import paired_mean_inference
from src.analysis.hidden_propagation import summarize_counterfactual_states
from src.extraction.hidden_states import (
    HiddenStatesExtractor,
    continuation_separator,
)
from src.runtime import model_source_fingerprint, stable_seed
from src.steering import MeanDiffSteering, SteeringConfig
from src.steering.hooks import get_layer_module, steering_applied


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--directions-dir", type=Path, required=True)
    parser.add_argument(
        "--anchor-component",
        default="random",
        help="Existing generation component used only to supply fixed text.",
    )
    parser.add_argument("--anchor-strength", type=float, default=-1.0)
    parser.add_argument(
        "--components",
        nargs="+",
        default=["lexical", "syntactic", "random"],
    )
    parser.add_argument("--max-prompts", type=int, default=12)
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        help="One-based decoder layers; defaults to a propagation trajectory.",
    )
    parser.add_argument("--position-bins", type=int, default=4)
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
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def fixed_trajectory_provenance(
    records: list[dict],
    tokenizer: Any,
    *,
    anchor_strength: float,
) -> dict[str, Any]:
    """Hash the exact prompt/continuation token trajectories used in an audit."""
    anchor_key = str(float(anchor_strength))
    per_prompt = {}
    for record in sorted(records, key=lambda row: str(row["prompt_id"])):
        prompt_id = str(record["prompt_id"])
        prefix = str(record["prompt"])
        continuation = str(
            record["outputs"][anchor_key]["text"]
        ).strip()
        separator = continuation_separator(prefix, continuation)
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)
        continuation_ids = tokenizer.encode(
            separator + continuation,
            add_special_tokens=False,
        )
        payload = {
            "prompt_id": prompt_id,
            "prompt_index": int(record["prompt_index"]),
            "prefix_token_ids": [int(token) for token in prefix_ids],
            "continuation_token_ids": [
                int(token) for token in continuation_ids
            ],
        }
        per_prompt[prompt_id] = {
            "prompt_index": int(record["prompt_index"]),
            "prefix_tokens": len(prefix_ids),
            "continuation_tokens": len(continuation_ids),
            "prefix_token_sha256": json_fingerprint({
                "token_ids": payload["prefix_token_ids"],
            }),
            "continuation_token_sha256": json_fingerprint({
                "token_ids": payload["continuation_token_ids"],
            }),
            "trajectory_token_sha256": json_fingerprint(payload),
        }
    return {
        "schema": "fixed_trajectory_provenance_v1",
        "n_prompts": len(per_prompt),
        "trajectory_set_sha256": json_fingerprint(per_prompt),
        "per_prompt": per_prompt,
    }


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_directions(
    path: Path,
    components: list[str],
) -> tuple[dict[str, MeanDiffSteering], int, float, dict[str, str]]:
    steerings = {}
    layer = None
    scale = None
    hashes = {}
    for component in components:
        checkpoint = path / f"{component}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        steering = MeanDiffSteering.load(checkpoint)
        component_layer = int(steering.layer)
        component_scale = float(steering.projection_std.item())
        if layer is None:
            layer = component_layer
            scale = component_scale
        if component_layer != layer:
            raise ValueError("All component directions must use one layer")
        if not np.isclose(component_scale, scale, rtol=1e-6, atol=1e-8):
            raise ValueError(
                "All component directions must use the same intervention scale"
            )
        steerings[component] = steering
        hashes[str(checkpoint)] = sha256_file(checkpoint)
    if layer is None or scale is None:
        raise ValueError("--components must not be empty")
    return steerings, layer, scale, hashes


def load_anchor_records(
    generation_root: Path,
    component: str,
) -> tuple[list[dict], Path]:
    path = generation_root / f"generation_component_{component}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a list")
    return payload, path


def filter_valid_anchor_records(
    records: list[dict],
    anchor_strength: float,
) -> tuple[list[dict], list[str]]:
    """Filter rows that cannot define a continuation-token trajectory."""
    anchor_key = str(float(anchor_strength))
    valid = []
    excluded = []
    for record in records:
        prompt_id = str(record.get("prompt_id", "unknown"))
        output = record.get("outputs", {}).get(anchor_key)
        text = output.get("text") if isinstance(output, dict) else None
        if text is None or not str(text).strip():
            excluded.append(prompt_id)
            continue
        valid.append(record)
    return valid, excluded


def balanced_prompt_selection(
    records: list[dict],
    metadata: dict[str, dict],
    maximum: int,
) -> list[dict]:
    """Deterministically round-robin prompts across source corpora."""
    if maximum < 1:
        raise ValueError("--max-prompts must be positive")
    buckets: dict[str, list[dict]] = {}
    for record in sorted(records, key=lambda row: int(row["prompt_index"])):
        prompt_id = str(record["prompt_id"])
        source = str(metadata.get(prompt_id, {}).get("source", "unknown"))
        buckets.setdefault(source, []).append(record)
    selected = []
    offsets = {source: 0 for source in buckets}
    while len(selected) < min(maximum, len(records)):
        advanced = False
        for source in sorted(buckets):
            offset = offsets[source]
            if offset >= len(buckets[source]):
                continue
            selected.append(buckets[source][offset])
            offsets[source] += 1
            advanced = True
            if len(selected) == min(maximum, len(records)):
                break
        if not advanced:
            break
    return sorted(selected, key=lambda row: int(row["prompt_index"]))


def default_layers(
    injection_layer: int,
    num_layers: int,
) -> list[int]:
    candidates = [
        injection_layer - 1,
        injection_layer,
        injection_layer + 1,
        12,
        16,
        20,
        24,
        28,
        32,
        num_layers,
    ]
    return sorted({
        layer
        for layer in candidates
        if 1 <= layer <= num_layers
    })


@torch.inference_mode()
def capture_condition(
    extractor: HiddenStatesExtractor,
    prefix: str,
    continuation: str,
    layers: list[int],
    *,
    steering: MeanDiffSteering | None,
    signed_strength: float,
) -> dict[int, np.ndarray]:
    """Capture continuation-token states under one fixed intervention."""
    inputs, encoded = extractor._prepare_continuation_batch(
        [prefix],
        [continuation],
        None,
    )
    start = int(encoded[0]["continuation_start"])
    end = int(encoded[0]["continuation_end"])
    captures: dict[int, torch.Tensor] = {}
    handles = []

    if signed_strength == 0:
        context = nullcontext()
    else:
        if steering is None:
            raise ValueError("Non-zero intervention requires a steering object")
        module = get_layer_module(extractor.model, steering.layer)
        try:
            steering_device = next(module.parameters()).device
        except StopIteration:
            steering_device = extractor.device
        steering = steering.to(steering_device)
        config = SteeringConfig(
            layer=steering.layer,
            strength=abs(float(signed_strength)),
            direction="human" if signed_strength < 0 else "ai",
            strength_unit="projection_std",
        )
        context = steering_applied(
            extractor.model,
            steering,
            config,
            list(range(start, end)),
        )

    def make_hook(layer: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captures[layer] = hidden

        return hook

    with context:
        # Register captures after the intervention hook so the injection-layer
        # capture observes the actual modified residual stream.
        for layer in layers:
            handles.append(
                get_layer_module(extractor.model, layer)
                .register_forward_hook(make_hook(layer))
            )
        try:
            extractor.forward_model(**inputs, use_cache=False)
        finally:
            for handle in handles:
                handle.remove()

    missing = sorted(set(layers) - set(captures))
    if missing:
        raise RuntimeError(f"Failed to capture layers: {missing}")
    return {
        layer: (
            captures[layer][0, start:end]
            .float()
            .cpu()
            .numpy()
        )
        for layer in layers
    }


@torch.inference_mode()
def capture_symmetric_direction_batch(
    extractor: HiddenStatesExtractor,
    prefix: str,
    continuation: str,
    layers: list[int],
    *,
    steerings: list[tuple[str, MeanDiffSteering]],
) -> dict[
    str,
    tuple[dict[int, np.ndarray], dict[int, np.ndarray]],
]:
    """Capture several matched −1/+1 directions in one forward pass."""
    if not steerings:
        raise ValueError("steerings must contain at least one direction")
    names = [name for name, _ in steerings]
    if len(names) != len(set(names)):
        raise ValueError("steering names must be unique")
    batch_size = 2 * len(steerings)
    inputs, encoded = extractor._prepare_continuation_batch(
        [prefix] * batch_size,
        [continuation] * batch_size,
        None,
    )
    starts = [int(row["continuation_start"]) for row in encoded]
    ends = [int(row["continuation_end"]) for row in encoded]
    if len(set(starts)) != 1 or len(set(ends)) != 1:
        raise RuntimeError("Duplicated fixed trajectories tokenize differently")
    start, end = starts[0], ends[0]
    captures: dict[int, torch.Tensor] = {}
    handles = []
    steering_layers = {steering.layer for _, steering in steerings}
    if len(steering_layers) != 1:
        raise ValueError("Batched steering directions must share one layer")
    steering_layer = next(iter(steering_layers))
    module = get_layer_module(extractor.model, steering_layer)
    try:
        steering_device = next(module.parameters()).device
    except StopIteration:
        steering_device = extractor.device
    prepared = [
        (name, steering.to(steering_device))
        for name, steering in steerings
    ]
    direction = torch.stack([
        steering.get_direction()
        for _, steering in prepared
    ]).to(
        device=steering_device,
        dtype=torch.float32,
    )
    scale = torch.stack([
        steering.projection_std
        for _, steering in prepared
    ]).to(
        device=steering_device,
        dtype=torch.float32,
    )

    def intervention_hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        signs = torch.tensor(
            [-1.0, 1.0] * len(steerings),
            device=hidden.device,
            dtype=hidden.dtype,
        ).view(batch_size, 1, 1)
        mask = torch.zeros(
            hidden.shape[:2],
            device=hidden.device,
            dtype=hidden.dtype,
        )
        mask[:, start:end] = 1.0
        delta = (
            signs
            * mask[:, :, None]
            * scale.to(dtype=hidden.dtype).repeat_interleave(2)
            .view(batch_size, 1, 1)
            * direction.to(device=hidden.device, dtype=hidden.dtype)
            .repeat_interleave(2, dim=0)
            .view(batch_size, 1, -1)
        )
        modified = hidden + delta
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified

    def make_hook(layer: int):
        def hook(_module, _inputs, output):
            captures[layer] = (
                output[0] if isinstance(output, tuple) else output
            )

        return hook

    handles.append(module.register_forward_hook(intervention_hook))
    for layer in layers:
        handles.append(
            get_layer_module(extractor.model, layer)
            .register_forward_hook(make_hook(layer))
        )
    try:
        extractor.forward_model(**inputs, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()

    missing = sorted(set(layers) - set(captures))
    if missing:
        raise RuntimeError(f"Failed to capture layers: {missing}")
    results = {}
    for index, (name, _steering) in enumerate(prepared):
        negative_row = 2 * index
        positive_row = negative_row + 1
        results[name] = (
            {
                layer: captures[layer][
                    negative_row, start:end
                ].float().cpu().numpy()
                for layer in layers
            },
            {
                layer: captures[layer][
                    positive_row, start:end
                ].float().cpu().numpy()
                for layer in layers
            },
        )
    return results


def capture_symmetric_conditions(
    extractor: HiddenStatesExtractor,
    prefix: str,
    continuation: str,
    layers: list[int],
    *,
    steering: MeanDiffSteering,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Capture one matched −1/+1 intervention through the batched path."""
    return capture_symmetric_direction_batch(
        extractor,
        prefix,
        continuation,
        layers,
        steerings=[("condition", steering)],
    )["condition"]


def capture_symmetric_direction_batch_with_fallback(
    extractor: HiddenStatesExtractor,
    prefix: str,
    continuation: str,
    layers: list[int],
    *,
    steerings: list[tuple[str, MeanDiffSteering]],
) -> dict[
    str,
    tuple[dict[int, np.ndarray], dict[int, np.ndarray]],
]:
    """Split a direction batch deterministically if CUDA memory is exhausted."""
    try:
        return capture_symmetric_direction_batch(
            extractor,
            prefix,
            continuation,
            layers,
            steerings=steerings,
        )
    except torch.cuda.OutOfMemoryError:
        if len(steerings) == 1:
            raise
        torch.cuda.empty_cache()
        midpoint = len(steerings) // 2
        print(
            "direction batch OOM; retrying as "
            f"{midpoint}+{len(steerings) - midpoint}",
            flush=True,
        )
        first = capture_symmetric_direction_batch_with_fallback(
            extractor,
            prefix,
            continuation,
            layers,
            steerings=steerings[:midpoint],
        )
        second = capture_symmetric_direction_batch_with_fallback(
            extractor,
            prefix,
            continuation,
            layers,
            steerings=steerings[midpoint:],
        )
        return {**first, **second}


def summarize_runs(
    frame: pd.DataFrame,
    *,
    intervention_layer: int,
    bootstrap: int,
    permutations: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target_rows = frame[
        (frame["target"] == frame["readout"])
        & (frame["position_scope"] == "all")
    ].copy()
    summaries = []
    for (component, layer), part in target_rows.groupby(
        ["component", "layer"],
        sort=True,
    ):
        inference = paired_mean_inference(
            part["projection_retention"].to_numpy(),
            bootstrap=bootstrap,
            permutations=permutations,
            seed=stable_seed(seed, "retention", component, int(layer)),
        )
        summaries.append({
            "component": component,
            "layer": int(layer),
            "relative_layer": int(layer) - intervention_layer,
            "n_prompts": int(part["prompt_id"].nunique()),
            "projection_effect_mean": float(
                part["projection_effect"].mean()
            ),
            "projection_retention_mean": inference["mean"],
            "projection_retention_median": inference["median"],
            "projection_retention_ci_low": inference["ci_low"],
            "projection_retention_ci_high": inference["ci_high"],
            "projection_retention_paired_d": inference["paired_d"],
            "projection_retention_sign_flip_p": inference["sign_flip_p"],
            "response_norm_retention_mean": float(
                part["response_norm_retention"].mean()
            ),
            "target_alignment_mean": float(
                part["target_alignment"].mean()
            ),
            "curvature_ratio_mean": float(
                part["curvature_ratio"].mean()
            ),
        })

    cross = (
        frame[frame["position_scope"] == "all"]
        .groupby(["component", "layer", "readout"], as_index=False)
        .agg(
            n_prompts=("prompt_id", "nunique"),
            projection_effect_mean=("projection_effect", "mean"),
            projection_retention_mean=("projection_retention", "mean"),
        )
    )
    positions = (
        frame[frame["target"] == frame["readout"]]
        .groupby(
            ["component", "layer", "position_scope"],
            as_index=False,
        )
        .agg(
            n_prompts=("prompt_id", "nunique"),
            mean_tokens=("n_tokens", "mean"),
            projection_retention_mean=("projection_retention", "mean"),
            target_alignment_mean=("target_alignment", "mean"),
            curvature_ratio_mean=("curvature_ratio", "mean"),
        )
    )
    contrast_rows = []
    if "random" in set(target_rows["component"]):
        pivot = target_rows.pivot(
            index=["prompt_id", "layer"],
            columns="component",
            values="projection_retention",
        )
        for component in sorted(
            set(target_rows["component"]) - {"random"}
        ):
            if component not in pivot.columns:
                continue
            paired = pivot[[component, "random"]].dropna()
            for layer, part in paired.groupby(level="layer", sort=True):
                values = (
                    part[component] - part["random"]
                ).to_numpy()
                inference = paired_mean_inference(
                    values,
                    bootstrap=bootstrap,
                    permutations=permutations,
                    seed=stable_seed(
                        seed,
                        "random_contrast",
                        component,
                        int(layer),
                    ),
                )
                contrast_rows.append({
                    "component": component,
                    "reference": "random",
                    "layer": int(layer),
                    "relative_layer": int(layer) - intervention_layer,
                    "n_prompts": len(values),
                    **inference,
                })
    return (
        pd.DataFrame(summaries),
        cross,
        positions,
        pd.DataFrame(contrast_rows),
    )


def write_summary(
    path: Path,
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    *,
    intervention_layer: int,
    final_layer: int,
    intervention_scale: float,
) -> None:
    focus = summary[
        summary["layer"].isin({
            intervention_layer - 1,
            intervention_layer,
            intervention_layer + 1,
            final_layer,
        })
    ]
    lines = [
        "# Fixed-token hidden propagation audit",
        "",
        "The same prompt-continuation tokens are evaluated under −1, 0, and "
        "+1 interventions. Projection retention is the symmetric hidden "
        "response `(h(+1)−h(−1))/2` along the target direction, divided by "
        f"the injected norm ({intervention_scale:.4f}).",
        "",
        "| Component | Layer | Relative layer | Retention | 95% CI | "
        "Target alignment | Response norm / injected norm | Curvature |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in focus.itertuples(index=False):
        lines.append(
            f"| {row.component} | {row.layer} | {row.relative_layer:+d} | "
            f"{row.projection_retention_mean:.3f} | "
            f"[{row.projection_retention_ci_low:.3f}, "
            f"{row.projection_retention_ci_high:.3f}] | "
            f"{row.target_alignment_mean:.3f} | "
            f"{row.response_norm_retention_mean:.3f} | "
            f"{row.curvature_ratio_mean:.3f} |"
        )
    final_contrasts = contrasts[contrasts["layer"] == final_layer]
    if not final_contrasts.empty:
        lines.extend([
            "",
            "## Learned directions versus the matched random direction",
            "",
            "| Component | Layer | Retention advantage | 95% CI | "
            "Sign-flip p |",
            "|---|---:|---:|---:|---:|",
        ])
        for row in final_contrasts.itertuples(index=False):
            lines.append(
                f"| {row.component} | {row.layer} | {row.mean:.3f} | "
                f"[{row.ci_low:.3f}, {row.ci_high:.3f}] | "
                f"{row.sign_flip_p:.4f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    started = datetime.now(timezone.utc)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    prompts_payload = json.loads(args.prompts.read_text(encoding="utf-8"))
    prompt_metadata = {
        str(row.get("id") or row.get("prompt_id")): row
        for row in prompts_payload
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
        prompt_metadata,
        args.max_prompts,
    )
    steerings, injection_layer, intervention_scale, direction_hashes = (
        load_directions(args.directions_dir, args.components)
    )

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
    layers = (
        sorted(set(args.layers))
        if args.layers
        else default_layers(injection_layer, extractor.num_layers)
    )
    if injection_layer not in layers:
        raise ValueError("--layers must include the intervention layer")
    if any(layer < 1 or layer > extractor.num_layers for layer in layers):
        raise ValueError("--layers contains an invalid decoder layer")

    direction_vectors = {
        component: steering.get_direction().float().cpu().numpy()
        for component, steering in steerings.items()
    }
    config = {
        "schema": "fixed_token_hidden_propagation_v1",
        "model": model_source_fingerprint(model_name),
        "prompts": str(args.prompts),
        "prompts_sha256": sha256_file(args.prompts),
        "anchor_generation": str(anchor_path),
        "anchor_generation_sha256": sha256_file(anchor_path),
        "anchor_component": args.anchor_component,
        "anchor_strength": args.anchor_strength,
        "anchor_candidate_count": len(anchor_records),
        "excluded_empty_anchor_prompt_ids": excluded_anchor_prompt_ids,
        "components": args.components,
        "direction_sha256": direction_hashes,
        "injection_layer": injection_layer,
        "intervention_scale": intervention_scale,
        "layers": layers,
        "position_bins": args.position_bins,
        "selected_prompt_ids": [
            str(row["prompt_id"]) for row in selected
        ],
        "device": args.device,
        "torch_dtype": args.torch_dtype,
        "attention_backend": args.attention_backend,
        "bootstrap": args.bootstrap,
        "permutations": args.permutations,
        "seed": args.seed,
    }
    fingerprint = json_fingerprint(config)
    work_dir = args.out_dir / ".work_hidden_propagation"
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

    all_rows = []
    anchor_key = str(float(args.anchor_strength))
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

        continuation = str(
            record["outputs"][anchor_key]["text"]
        ).strip()
        if not continuation:
            raise ValueError(f"Empty anchor continuation for {prompt_id}")
        prefix = str(record["prompt"])
        baseline = capture_condition(
            extractor,
            prefix,
            continuation,
            layers,
            steering=None,
            signed_strength=0.0,
        )
        prompt_rows = []
        metadata = prompt_metadata.get(prompt_id, {})
        for component in args.components:
            condition_states = {
                0.0: baseline,
                -1.0: capture_condition(
                    extractor,
                    prefix,
                    continuation,
                    layers,
                    steering=steerings[component],
                    signed_strength=-1.0,
                ),
                1.0: capture_condition(
                    extractor,
                    prefix,
                    continuation,
                    layers,
                    steering=steerings[component],
                    signed_strength=1.0,
                ),
            }
            rows = summarize_counterfactual_states(
                condition_states,
                direction_vectors,
                target=component,
                intervention_scale=intervention_scale,
                position_bins=args.position_bins,
            )
            for row in rows:
                row.update({
                    "component": component,
                    "prompt_id": prompt_id,
                    "prompt_index": int(record["prompt_index"]),
                    "source": str(metadata.get("source", "unknown")),
                    "domain": str(metadata.get("domain", "unknown")),
                    "anchor_component": args.anchor_component,
                    "anchor_strength": float(args.anchor_strength),
                    "continuation_tokens": int(
                        condition_states[0.0][injection_layer].shape[0]
                    ),
                })
            prompt_rows.extend(rows)
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
    frame.to_csv(args.out_dir / "per_prompt_propagation.csv", index=False)
    summary, cross, positions, contrasts = summarize_runs(
        frame,
        intervention_layer=injection_layer,
        bootstrap=args.bootstrap,
        permutations=args.permutations,
        seed=args.seed,
    )
    summary.to_csv(args.out_dir / "propagation_summary.csv", index=False)
    cross.to_csv(args.out_dir / "cross_readout_summary.csv", index=False)
    positions.to_csv(args.out_dir / "position_summary.csv", index=False)
    contrasts.to_csv(
        args.out_dir / "random_control_contrasts.csv",
        index=False,
    )
    write_summary(
        args.out_dir / "analysis_summary.md",
        summary,
        contrasts,
        intervention_layer=injection_layer,
        final_layer=max(layers),
        intervention_scale=intervention_scale,
    )

    finished = datetime.now(timezone.utc)
    manifest = {
        **config,
        "fingerprint": fingerprint,
        "status": "complete",
        "n_prompts": len(selected),
        "n_rows": len(frame),
        "model_load": extractor.model_load_info,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "torch_version": torch.__version__,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "command": " ".join(sys.argv),
    }
    atomic_json(args.out_dir / "manifest.json", manifest)
    print(f"Wrote hidden propagation audit to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
