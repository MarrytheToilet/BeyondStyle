#!/usr/bin/env python3
"""Build nullspace-constrained transport-aware concept directions.

For each automatic concept readout at a downstream layer, this script
backpropagates the readout through a frozen language model to a shared additive
intervention at an earlier layer. Human and AI continuations contribute
separate pullback gradients; their nullspace-projected consensus is saved as
the causal intervention direction.
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
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.audit_hidden_propagation import atomic_json, sha256_file
from src.analysis.causal_pullback import (
    layer_output_dir,
    nullspace_pullback_consensus,
    weighted_gradient_mean,
)
from src.analysis.hidden_propagation import orthogonal_random_directions
from src.analysis.projection_mapping import fit_within_class_basis
from src.extraction.hidden_states import HiddenStatesExtractor
from src.runtime import model_source_fingerprint
from src.steering import MeanDiffSteering
from src.steering.hooks import get_layer_module


CONCEPTS = ("lexical", "syntactic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--input-directions-dir", type=Path, required=True)
    parser.add_argument("--downstream-directions-dir", type=Path, required=True)
    parser.add_argument("--input-layer", type=int, required=True)
    parser.add_argument("--downstream-layer", type=int, required=True)
    parser.add_argument("--max-pairs", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--null-k", type=int, default=10)
    parser.add_argument(
        "--minimum-source-cosine",
        type=float,
        default=0.0,
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
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def json_fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not all(
                str(record.get(field, "")).strip()
                for field in (
                    "sample_id",
                    "prefix",
                    "human_continuation",
                    "ai_continuation",
                )
            ):
                raise ValueError(
                    f"{path}:{line_number} lacks a required text field"
                )
            records.append(record)
    if not records:
        raise ValueError(f"{path} contains no records")
    return records


def generator_name(record: dict) -> str:
    generation = record.get("generation", {})
    return str(
        generation.get("generator_model")
        or record.get("generator")
        or "unknown"
    )


def balanced_selection(
    records: list[dict],
    maximum: int,
) -> list[dict]:
    """Round-robin over source-corpus × generator environments."""
    if maximum < 1:
        raise ValueError("--max-pairs must be positive")
    buckets: dict[str, list[dict]] = {}
    for record in records:
        environment = (
            f"{record.get('source', 'unknown')}|{generator_name(record)}"
        )
        buckets.setdefault(environment, []).append(record)
    selected = []
    offsets = {environment: 0 for environment in buckets}
    while len(selected) < min(maximum, len(records)):
        advanced = False
        for environment in sorted(buckets):
            offset = offsets[environment]
            if offset >= len(buckets[environment]):
                continue
            selected.append(buckets[environment][offset])
            offsets[environment] += 1
            advanced = True
            if len(selected) == min(maximum, len(records)):
                break
        if not advanced:
            break
    return selected


def load_direction(path: Path) -> tuple[np.ndarray, int, float]:
    steering = MeanDiffSteering.load(path)
    vector = steering.get_direction().float().cpu().numpy()
    vector /= max(float(np.linalg.norm(vector)), 1e-12)
    return (
        vector,
        int(steering.layer),
        float(steering.projection_std.item()),
    )


def save_direction(
    path: Path,
    vector: np.ndarray,
    *,
    layer: int,
    scale: float,
    metadata: dict[str, Any],
) -> None:
    vector = np.asarray(vector, dtype=np.float32)
    vector /= max(float(np.linalg.norm(vector)), 1e-12)
    steering = MeanDiffSteering(hidden_size=len(vector), layer=layer)
    with torch.no_grad():
        steering.v_src.copy_(torch.from_numpy(vector))
        steering.is_fitted.fill_(True)
        steering.projection_std.fill_(scale)
    path.parent.mkdir(parents=True, exist_ok=True)
    steering.save(path, metadata=metadata)


def atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def batch_pullback_gradients(
    extractor: HiddenStatesExtractor,
    prefixes: list[str],
    continuations: list[str],
    *,
    input_layer: int,
    downstream_layer: int,
    readouts: dict[str, torch.Tensor],
) -> tuple[dict[str, np.ndarray], dict[str, float], int]:
    """Differentiate downstream concept scores through a shared input delta."""
    inputs, encoded = extractor._prepare_continuation_batch(
        prefixes,
        continuations,
        None,
    )
    batch_size, sequence_length = inputs["input_ids"].shape
    mask = torch.zeros(
        (batch_size, sequence_length),
        dtype=torch.float32,
        device=inputs["input_ids"].device,
    )
    for row_index, row in enumerate(encoded):
        mask[
            row_index,
            int(row["continuation_start"]):int(row["continuation_end"]),
        ] = 1.0

    delta = torch.zeros(
        extractor.hidden_size,
        dtype=torch.float32,
        device=inputs["input_ids"].device,
        requires_grad=True,
    )
    capture: dict[str, torch.Tensor] = {}

    def intervention_hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        modified = hidden + (
            mask[:, :, None].to(hidden.dtype)
            * delta.to(hidden.dtype)[None, None, :]
        )
        if isinstance(output, tuple):
            return (modified,) + output[1:]
        return modified

    def downstream_hook(_module, _inputs, output):
        capture["hidden"] = output[0] if isinstance(output, tuple) else output

    intervention_handle = get_layer_module(
        extractor.model,
        input_layer,
    ).register_forward_hook(intervention_hook)
    downstream_handle = get_layer_module(
        extractor.model,
        downstream_layer,
    ).register_forward_hook(downstream_hook)
    try:
        with torch.enable_grad():
            extractor.forward_model(**inputs, use_cache=False)
            hidden = capture["hidden"].float()
            token_counts = mask.sum(dim=1).clamp_min(1.0)
            objectives = {}
            for concept, readout in readouts.items():
                scores = hidden @ readout.to(
                    device=hidden.device,
                    dtype=hidden.dtype,
                )
                per_sequence = (
                    (scores * mask).sum(dim=1) / token_counts
                )
                objectives[concept] = per_sequence.mean()
            gradients = {}
            names = list(readouts)
            for index, concept in enumerate(names):
                gradient = torch.autograd.grad(
                    objectives[concept],
                    delta,
                    retain_graph=index < len(names) - 1,
                    create_graph=False,
                )[0]
                gradients[concept] = (
                    gradient.detach().float().cpu().numpy()
                )
            objective_values = {
                concept: float(value.detach().cpu())
                for concept, value in objectives.items()
            }
    finally:
        intervention_handle.remove()
        downstream_handle.remove()
    return gradients, objective_values, len(prefixes)


def gradient_stability(
    batch_gradients: np.ndarray,
    consensus: np.ndarray,
) -> tuple[float, float]:
    norms = np.linalg.norm(batch_gradients, axis=1)
    active = norms > 1e-12
    if not np.any(active):
        return 0.0, 0.0
    cosines = (
        batch_gradients[active] @ consensus
    ) / norms[active]
    return float(cosines.mean()), float(cosines.min())


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.downstream_layer <= args.input_layer:
        raise ValueError(
            "--downstream-layer must be later than --input-layer"
        )
    started = datetime.now(timezone.utc)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(args.pairs)
    selected = balanced_selection(records, args.max_pairs)

    input_vectors = {}
    input_hashes = {}
    intervention_scale = None
    for name in ("source", *CONCEPTS):
        path = args.input_directions_dir / f"{name}.pt"
        vector, layer, scale = load_direction(path)
        if layer != args.input_layer:
            raise ValueError(f"{path} is not an L{args.input_layer} direction")
        if intervention_scale is None:
            intervention_scale = scale
        if not np.isclose(scale, intervention_scale, rtol=1e-6):
            raise ValueError("Input directions do not share one scale")
        input_vectors[name] = vector
        input_hashes[str(path)] = sha256_file(path)
    assert intervention_scale is not None

    downstream_vectors = {}
    downstream_hashes = {}
    for concept in CONCEPTS:
        path = args.downstream_directions_dir / f"{concept}.pt"
        vector, layer, _ = load_direction(path)
        if layer != args.downstream_layer:
            raise ValueError(
                f"{path} is not an L{args.downstream_layer} direction"
            )
        downstream_vectors[concept] = vector
        downstream_hashes[str(path)] = sha256_file(path)

    states_payload = torch.load(
        args.states,
        map_location="cpu",
        weights_only=False,
    )
    human_states = states_payload["human"].float().numpy()
    ai_states = states_payload["ai"].float().numpy()
    within_basis, basis_info = fit_within_class_basis(
        human_states,
        ai_states,
        k=args.null_k,
    )

    model_path = Path(args.model).expanduser()
    model_name = (
        str(model_path.resolve())
        if model_path.is_dir()
        else args.model
    )
    config = {
        "schema": "nullspace_jacobian_pullback_v1",
        "model": model_source_fingerprint(model_name),
        "pairs": str(args.pairs),
        "pairs_sha256": sha256_file(args.pairs),
        "states": str(args.states),
        "states_sha256": sha256_file(args.states),
        "input_direction_sha256": input_hashes,
        "downstream_direction_sha256": downstream_hashes,
        "input_layer": args.input_layer,
        "downstream_layer": args.downstream_layer,
        "intervention_scale": intervention_scale,
        "max_pairs": args.max_pairs,
        "batch_size": args.batch_size,
        "null_k": args.null_k,
        "minimum_source_cosine": args.minimum_source_cosine,
        "selected_sample_ids": [
            str(record["sample_id"]) for record in selected
        ],
        "selected_environments": [
            {
                "sample_id": str(record["sample_id"]),
                "source": str(record.get("source", "unknown")),
                "generator": generator_name(record),
            }
            for record in selected
        ],
        "device": args.device,
        "torch_dtype": args.torch_dtype,
        "attention_backend": args.attention_backend,
        "seed": args.seed,
    }
    fingerprint = json_fingerprint(config)
    work_dir = args.out_dir / ".work_pullback"
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

    extractor = HiddenStatesExtractor(
        model_name,
        device=args.device,
        dtype=args.torch_dtype,
        attention_backend=args.attention_backend,
    )
    extractor.model.requires_grad_(False)
    readout_tensors = {
        concept: torch.from_numpy(vector).float()
        for concept, vector in downstream_vectors.items()
    }

    source_batches: dict[str, list[dict[str, Any]]] = {
        "human": [],
        "ai": [],
    }
    for source in source_batches:
        for start in range(0, len(selected), args.batch_size):
            batch = selected[start:start + args.batch_size]
            shard_path = (
                work_dir / f"{source}_batch_{start:05d}.pt"
            )
            if shard_path.exists():
                shard = torch.load(
                    shard_path,
                    map_location="cpu",
                    weights_only=False,
                )
                if shard.get("fingerprint") != fingerprint:
                    raise RuntimeError(f"Stale gradient shard: {shard_path}")
            else:
                continuations = [
                    str(record[f"{source}_continuation"])
                    for record in batch
                ]
                gradients, objectives, n_sequences = (
                    batch_pullback_gradients(
                        extractor,
                        [str(record["prefix"]) for record in batch],
                        continuations,
                        input_layer=args.input_layer,
                        downstream_layer=args.downstream_layer,
                        readouts=readout_tensors,
                    )
                )
                shard = {
                    "fingerprint": fingerprint,
                    "source": source,
                    "start": start,
                    "sample_ids": [
                        str(record["sample_id"]) for record in batch
                    ],
                    "n_sequences": n_sequences,
                    "gradients": {
                        concept: torch.from_numpy(gradient)
                        for concept, gradient in gradients.items()
                    },
                    "objectives": objectives,
                }
                atomic_torch_save(shard_path, shard)
            source_batches[source].append(shard)
            print(
                f"{source} batch {start // args.batch_size + 1}/"
                f"{(len(selected) + args.batch_size - 1) // args.batch_size}",
                flush=True,
            )

    directions = {}
    diagnostics = {}
    for concept in CONCEPTS:
        source_means = {}
        source_matrices = {}
        for source, batches in source_batches.items():
            matrix = np.stack([
                batch["gradients"][concept].float().numpy()
                for batch in batches
            ])
            weights = np.array([
                batch["n_sequences"] for batch in batches
            ])
            source_matrices[source] = matrix
            source_means[source] = weighted_gradient_mean(matrix, weights)
        direction, info = nullspace_pullback_consensus(
            source_means["human"],
            source_means["ai"],
            within_basis,
        )
        if info["human_ai_pattern_cosine"] < args.minimum_source_cosine:
            raise RuntimeError(
                f"{concept} Human/AI pullback cosine "
                f"{info['human_ai_pattern_cosine']:.3f} is below "
                f"{args.minimum_source_cosine:.3f}"
            )
        human_stability = gradient_stability(
            source_matrices["human"],
            direction,
        )
        ai_stability = gradient_stability(
            source_matrices["ai"],
            direction,
        )
        info.update({
            "cosine_vs_encoding_direction": float(
                direction @ input_vectors[concept]
            ),
            "human_batch_consensus_cosine_mean": human_stability[0],
            "human_batch_consensus_cosine_min": human_stability[1],
            "ai_batch_consensus_cosine_mean": ai_stability[0],
            "ai_batch_consensus_cosine_min": ai_stability[1],
        })
        directions[concept] = direction
        diagnostics[concept] = info

    random = orthogonal_random_directions(
        {
            "source": input_vectors["source"],
            **directions,
        },
        count=1,
        seed=args.seed,
    )[0]
    directions_to_save = {
        "source": input_vectors["source"],
        **directions,
        "random": random,
    }
    shared_metadata = {
        "method": "nullspace_jacobian_pullback",
        "model_id": model_source_fingerprint(model_name)["model_id"],
        "layer": args.input_layer,
        "downstream_layer": args.downstream_layer,
        "representation": "mean_pool",
        "null_k": args.null_k,
        "intervention_norm_ratio": 0.05,
        "shared_intervention_scale": intervention_scale,
        "n_training_pairs": len(selected),
        "human_ai_gradients_fitted_independently": True,
        "prompt_prefill_untouched": True,
        "position_mode": "decode_only",
    }
    output_layer_dir = layer_output_dir(args.out_dir, args.input_layer)
    direction_dir = output_layer_dir / "directions"
    for component, vector in directions_to_save.items():
        save_direction(
            direction_dir / f"{component}.pt",
            vector,
            layer=args.input_layer,
            scale=intervention_scale,
            metadata={
                **shared_metadata,
                "component": component,
                "diagnostics": diagnostics.get(component, {}),
            },
        )

    diagnostics["pullback_direction_cosine"] = float(
        directions["lexical"] @ directions["syntactic"]
    )
    decomposition = {
        "schema": "nullspace_jacobian_pullback",
        "model": model_source_fingerprint(model_name)["model_id"],
        "representation": "mean_pool",
        "layer": args.input_layer,
        "downstream_layer": args.downstream_layer,
        "groups": diagnostics,
        "nullspace": basis_info,
        "intervention_scale": intervention_scale,
        "selection_rule": (
            "Human and AI pullback gradients are fitted independently, "
            "projected out of the top within-class PCs, and averaged without "
            "post-hoc sign flipping."
        ),
    }
    atomic_json(
        output_layer_dir / "decomposition.json",
        decomposition,
    )
    lines = [
        "# Nullspace Jacobian-pullback directions",
        "",
        "| Concept | Human/AI cosine | Cosine vs encoding | "
        "Human batch stability | AI batch stability |",
        "|---|---:|---:|---:|---:|",
    ]
    for concept in CONCEPTS:
        info = diagnostics[concept]
        lines.append(
            f"| {concept} | {info['human_ai_pattern_cosine']:.3f} | "
            f"{info['cosine_vs_encoding_direction']:.3f} | "
            f"{info['human_batch_consensus_cosine_mean']:.3f} | "
            f"{info['ai_batch_consensus_cosine_mean']:.3f} |"
        )
    (args.out_dir / "analysis_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    finished = datetime.now(timezone.utc)
    atomic_json(
        args.out_dir / "manifest.json",
        {
            **config,
            "fingerprint": fingerprint,
            "status": "complete",
            "diagnostics": diagnostics,
            "nullspace": basis_info,
            "model_load": extractor.model_load_info,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "started_at_utc": started.isoformat(),
            "finished_at_utc": finished.isoformat(),
            "duration_seconds": (finished - started).total_seconds(),
            "command": " ".join(sys.argv),
        },
    )
    print(f"Wrote causal pullback directions to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
