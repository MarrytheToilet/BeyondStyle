#!/usr/bin/env python3
"""Audit pullback-axis sensitivity using frozen per-pair gradients.

This audit changes only two estimator choices that do not require another
model forward pass:

* the number of within-class principal directions projected out; and
* the number of balanced, deterministically ordered fit pairs averaged.

Each alternative is compared with the frozen ``k=10, n=32`` axis.  The
result is an estimator-stability diagnostic, not a replacement for the
fixed-trajectory transport audit.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
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

from src.analysis.causal_pullback import nullspace_pullback_consensus
from src.analysis.projection_mapping import fit_within_class_basis


CONCEPTS = ("lexical", "syntactic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-ids",
        nargs="+",
        default=["Llama-3.1-8B-Instruct", "Qwen3-4B"],
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/en/05_linguistic/causal_pullback"),
    )
    parser.add_argument("--null-k-values", nargs="+", type=int,
                        default=[0, 5, 10, 20, 50])
    parser.add_argument("--fit-pair-values", nargs="+", type=int,
                        default=[8, 16, 24, 32])
    parser.add_argument("--canonical-null-k", type=int, default=10)
    parser.add_argument("--canonical-fit-pairs", type=int, default=32)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "outputs/en/05_linguistic/causal_pullback/"
            "design_sensitivity"
        ),
    )
    return parser.parse_args()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_gradients(
    work_dir: Path,
    source: str,
    concept: str,
    n_pairs: int,
) -> np.ndarray:
    rows = []
    for index in range(n_pairs):
        path = work_dir / f"{source}_batch_{index:05d}.pt"
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if int(payload["n_sequences"]) != 1:
            raise ValueError(
                "Sensitivity audit requires one saved sequence per shard: "
                f"{path}"
            )
        rows.append(payload["gradients"][concept].float().numpy())
    return np.stack(rows)


def estimate_axis(
    human_gradients: np.ndarray,
    ai_gradients: np.ndarray,
    basis: np.ndarray,
    n_pairs: int,
) -> tuple[np.ndarray, dict[str, float]]:
    return nullspace_pullback_consensus(
        human_gradients[:n_pairs].mean(axis=0),
        ai_gradients[:n_pairs].mean(axis=0),
        basis,
    )


def main() -> None:
    args = parse_args()
    started = datetime.now(timezone.utc)
    rows: list[dict[str, Any]] = []
    for model_id in args.model_ids:
        model_root = args.root / model_id / "mean_pool"
        work_dir = model_root / ".work_pullback"
        work_manifest = json.loads(
            (work_dir / "work_manifest.json").read_text(encoding="utf-8")
        )
        config = work_manifest["config"]
        if int(config["batch_size"]) != 1:
            raise ValueError(f"{model_id} did not save per-pair gradients")
        if int(config["max_pairs"]) < args.canonical_fit_pairs:
            raise ValueError(f"{model_id} lacks the canonical fit pairs")

        states = torch.load(
            config["states"],
            map_location="cpu",
            weights_only=False,
        )
        human_states = states["human"].float().numpy()
        ai_states = states["ai"].float().numpy()
        bases = {
            k: fit_within_class_basis(human_states, ai_states, k=k)[0]
            for k in sorted(
                set(args.null_k_values + [args.canonical_null_k])
            )
        }
        gradients = {
            source: {
                concept: load_gradients(
                    work_dir,
                    source,
                    concept,
                    args.canonical_fit_pairs,
                )
                for concept in CONCEPTS
            }
            for source in ("human", "ai")
        }
        canonical = {}
        for concept in CONCEPTS:
            canonical[concept], _ = estimate_axis(
                gradients["human"][concept],
                gradients["ai"][concept],
                bases[args.canonical_null_k],
                args.canonical_fit_pairs,
            )

        for null_k in args.null_k_values:
            for concept in CONCEPTS:
                axis, diagnostics = estimate_axis(
                    gradients["human"][concept],
                    gradients["ai"][concept],
                    bases[null_k],
                    args.canonical_fit_pairs,
                )
                rows.append({
                    "model": model_id,
                    "concept": concept,
                    "sensitivity": "null_k",
                    "value": null_k,
                    "fit_pairs": args.canonical_fit_pairs,
                    "null_k": null_k,
                    "cosine_vs_canonical": float(
                        axis @ canonical[concept]
                    ),
                    "human_ai_pullback_cosine": float(
                        diagnostics["human_ai_pattern_cosine"]
                    ),
                })

        for n_pairs in args.fit_pair_values:
            if n_pairs > args.canonical_fit_pairs:
                raise ValueError(
                    f"fit-pair value {n_pairs} exceeds saved gradients"
                )
            for concept in CONCEPTS:
                axis, diagnostics = estimate_axis(
                    gradients["human"][concept],
                    gradients["ai"][concept],
                    bases[args.canonical_null_k],
                    n_pairs,
                )
                rows.append({
                    "model": model_id,
                    "concept": concept,
                    "sensitivity": "fit_pairs",
                    "value": n_pairs,
                    "fit_pairs": n_pairs,
                    "null_k": args.canonical_null_k,
                    "cosine_vs_canonical": float(
                        axis @ canonical[concept]
                    ),
                    "human_ai_pullback_cosine": float(
                        diagnostics["human_ai_pattern_cosine"]
                    ),
                })

    frame = pd.DataFrame(rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dir / "direction_sensitivity.csv", index=False)
    finished = datetime.now(timezone.utc)
    atomic_json(
        args.out_dir / "manifest.json",
        {
            "schema": "pullback_design_sensitivity_v1",
            "status": "complete",
            "models": args.model_ids,
            "canonical_null_k": args.canonical_null_k,
            "canonical_fit_pairs": args.canonical_fit_pairs,
            "null_k_values": args.null_k_values,
            "fit_pair_values": args.fit_pair_values,
            "interpretation": (
                "Estimator-direction stability only; no new trajectory "
                "or causal-response claim."
            ),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "started_at_utc": started.isoformat(),
            "finished_at_utc": finished.isoformat(),
            "duration_seconds": (finished - started).total_seconds(),
        },
    )
    print(f"Wrote {args.out_dir / 'direction_sensitivity.csv'}")


if __name__ == "__main__":
    main()
