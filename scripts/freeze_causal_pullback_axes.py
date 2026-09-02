#!/usr/bin/env python3
"""Freeze canonical causal pullback axes for confirmatory generation.

The pole of each axis is fixed by the pullback construction itself:
positive intervention follows the gradient of the machine-oriented
downstream linguistic readout.  Free-generation outcomes never select or
flip an axis.  The script also freezes an orthonormal random-axis ensemble
for behavior-level null calibration.
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

from src.analysis.hidden_propagation import orthogonal_random_directions
from src.steering import MeanDiffSteering


CONCEPTS = ("lexical", "syntactic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    for concept in CONCEPTS:
        parser.add_argument(
            f"--{concept}-checkpoint",
            type=Path,
            required=True,
        )
    parser.add_argument(
        "--sealed-prompt-start",
        type=int,
        default=34,
        help="First prompt index reserved for confirmatory generation.",
    )
    parser.add_argument("--random-controls", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def load_checkpoint(
    path: Path,
) -> tuple[np.ndarray, int, float, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    steering = MeanDiffSteering.load(path)
    vector = steering.get_direction().float().numpy()
    vector /= max(float(np.linalg.norm(vector)), 1e-12)
    return (
        vector,
        int(steering.layer),
        float(steering.projection_std.item()),
        dict(payload.get("metadata", {})),
    )


def save_checkpoint(
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


def main() -> None:
    args = parse_args()
    if args.sealed_prompt_start < 0:
        raise ValueError("--sealed-prompt-start must be nonnegative")
    if args.random_controls < 2:
        raise ValueError("--random-controls must be at least two")
    started = datetime.now(timezone.utc)
    source, input_layer, source_scale, source_metadata = load_checkpoint(
        args.source_checkpoint
    )

    selected: dict[str, np.ndarray] = {}
    checkpoint_provenance: dict[str, dict[str, Any]] = {}
    scale: float | None = None
    for concept in CONCEPTS:
        checkpoint = getattr(args, f"{concept}_checkpoint")
        vector, layer, component_scale, metadata = load_checkpoint(checkpoint)
        if layer != input_layer:
            raise ValueError(
                f"{checkpoint} is L{layer}; expected L{input_layer}"
            )
        if scale is None:
            scale = component_scale
        elif not np.isclose(component_scale, scale, rtol=1e-6):
            raise ValueError("Selected checkpoints do not share one scale")
        selected[concept] = vector
        checkpoint_provenance[concept] = {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "fit_metadata": metadata,
            "orientation": (
                "canonical positive Jacobian pullback of the "
                "machine-oriented downstream readout"
            ),
        }
    assert scale is not None

    random_vectors = orthogonal_random_directions(
        {"source": source, **selected},
        count=args.random_controls,
        seed=args.seed,
    )
    directions = {
        "source": source,
        **selected,
        "random": random_vectors[0],
        **{
            f"random_{index:02d}": vector
            for index, vector in enumerate(random_vectors[1:], start=1)
        },
    }

    common_metadata = {
        "method": "canonical_nullspace_jacobian_pullback_axis",
        "layer": input_layer,
        "representation": "mean_pool",
        "shared_intervention_scale": scale,
        "source_checkpoint_original_scale": source_scale,
        "orientation_rule": (
            "positive Jacobian pullback of the machine-oriented "
            "downstream linguistic readout"
        ),
        "orientation_uses_generation_outcomes": False,
        "sealed_prompt_indices_start": args.sealed_prompt_start,
        "prompt_prefill_untouched": True,
        "position_mode": "decode_only",
    }
    direction_dir = args.out_dir / f"layer{input_layer:02d}" / "directions"
    for component, vector in directions.items():
        save_checkpoint(
            direction_dir / f"{component}.pt",
            vector,
            layer=input_layer,
            scale=scale,
            metadata={
                **common_metadata,
                "component": component,
                "source_metadata": (
                    source_metadata if component == "source" else None
                ),
                "selected_checkpoint": checkpoint_provenance.get(component),
            },
        )

    lexical_syntactic_cosine = float(
        selected["lexical"] @ selected["syntactic"]
    )
    random_basis_error = float(
        np.max(
            np.abs(
                random_vectors @ random_vectors.T
                - np.eye(args.random_controls)
            )
        )
    )
    decomposition = {
        "schema": "canonical_causal_pullback_axes",
        "representation": "mean_pool",
        "layer": input_layer,
        "intervention_scale": scale,
        "selected_components": checkpoint_provenance,
        "orientation_rule": common_metadata["orientation_rule"],
        "orientation_uses_generation_outcomes": False,
        "random_controls": args.random_controls,
        "geometry": {
            "lexical_syntactic_cosine": lexical_syntactic_cosine,
            "random_source_max_abs_cosine": float(
                np.max(np.abs(random_vectors @ source))
            ),
            "random_concept_max_abs_cosine": float(
                max(
                    np.max(np.abs(random_vectors @ selected[name]))
                    for name in CONCEPTS
                )
            ),
            "random_orthonormality_max_error": random_basis_error,
        },
        "selection_rule": (
            "Concept checkpoints and downstream anchors are supplied before "
            "confirmatory generation; axis poles are never selected from "
            "surface-generation outcomes."
        ),
    }
    atomic_json(
        args.out_dir / f"layer{input_layer:02d}" / "decomposition.json",
        decomposition,
    )
    lines = [
        "# Canonical causal pullback axes",
        "",
        "| Concept | Anchor layer | Orientation |",
        "|---|---:|---|",
    ]
    for concept in CONCEPTS:
        metadata = checkpoint_provenance[concept]["fit_metadata"]
        lines.append(
            f"| {concept} | {metadata.get('downstream_layer', 'NA')} | "
            "positive downstream machine-oriented readout gradient |"
        )
    lines.extend(
        [
            "",
            f"Frozen orthonormal random controls: {args.random_controls}.",
        ]
    )
    (args.out_dir / "analysis_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    finished = datetime.now(timezone.utc)
    atomic_json(
        args.out_dir / "manifest.json",
        {
            "schema": "canonical_causal_pullback_axes_v2",
            "source_checkpoint": str(args.source_checkpoint),
            "source_checkpoint_sha256": sha256_file(
                args.source_checkpoint
            ),
            "selected_components": checkpoint_provenance,
            "orientation_rule": common_metadata["orientation_rule"],
            "orientation_uses_generation_outcomes": False,
            "sealed_prompt_start": args.sealed_prompt_start,
            "random_controls": args.random_controls,
            "seed": args.seed,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "started_at_utc": started.isoformat(),
            "finished_at_utc": finished.isoformat(),
            "duration_seconds": (finished - started).total_seconds(),
            "command": " ".join(sys.argv),
            "status": "complete",
        },
    )
    print(f"Wrote frozen causal axes to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
