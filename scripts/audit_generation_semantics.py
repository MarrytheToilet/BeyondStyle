#!/usr/bin/env python3
"""Audit semantic preservation for paired multi-seed steering outputs.

The inferential unit is the prompt. Cosine similarity is first computed for
each paired negative/positive decode using the same sampling seed, then
averaged over seeds within prompt before bootstrap inference.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.causal_linguistics import paired_mean_inference  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--model-path", type=Path, required=True,
        help="Local sentence-transformer directory (download the checkpoint first).",
    )
    parser.add_argument("--components", nargs="+", default=["lexical"])
    parser.add_argument("--endpoint-strength", type=float, default=2.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--permutations", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260902)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def local_encoder_path(path: Path) -> Path:
    """Fail clearly when a Hub identifier is passed to this offline audit."""
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise ValueError(
            f"--model-path must be an existing local directory, got {path}. "
            "Download the encoder with `hf download REPO_ID --local-dir DIR` "
            "and pass DIR; this audit does not download models."
        )
    return resolved


def load_pairs(
    input_dir: Path,
    components: list[str],
    endpoint_strength: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    if endpoint_strength <= 0:
        raise ValueError("--endpoint-strength must be positive")
    negative_key = str(float(-endpoint_strength))
    positive_key = str(float(endpoint_strength))
    expected_outputs = {negative_key, positive_key}
    rows: list[dict[str, Any]] = []
    inputs: list[dict[str, Any]] = []
    component_keys: dict[str, set[tuple[str, int]]] = {}
    for component in components:
        path = input_dir / f"generation_component_{component}.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        inputs.append(
            {
                "component": component,
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "n_rows": len(payload),
            }
        )
        keys: set[tuple[str, int]] = set()
        for row in payload:
            outputs = row["outputs"]
            if set(outputs) != expected_outputs:
                raise ValueError(
                    f"{path} prompt {row['prompt_id']} does not contain exact "
                    f"endpoints {sorted(expected_outputs)}"
                )
            prompt_id = str(row["prompt_id"])
            seed_index = int(row.get("sample_seed_index", 0))
            key = (prompt_id, seed_index)
            if key in keys:
                raise ValueError(
                    f"Duplicate prompt×seed observation for {component}: {key}"
                )
            keys.add(key)
            rows.append(
                {
                    "component": component,
                    "prompt_id": prompt_id,
                    "prompt_index": int(row["prompt_index"]),
                    "sample_seed_index": seed_index,
                    "negative_text": str(outputs[negative_key]["text"]),
                    "positive_text": str(outputs[positive_key]["text"]),
                }
            )
        component_keys[component] = keys
    if len({frozenset(keys) for keys in component_keys.values()}) != 1:
        raise ValueError("Components do not share the same prompt×seed set")
    return pd.DataFrame(rows), inputs


def main() -> None:
    args = parse_args()
    model_path = local_encoder_path(args.model_path)
    started = datetime.now(timezone.utc)
    pairs, inputs = load_pairs(
        args.input_dir,
        args.components,
        args.endpoint_strength,
    )

    from sentence_transformers import SentenceTransformer
    import sentence_transformers

    encoder = SentenceTransformer(
        str(model_path),
        device=args.device,
        local_files_only=True,
    )
    texts = pairs["negative_text"].tolist() + pairs["positive_text"].tolist()
    embeddings = encoder.encode(
        texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    n_pairs = len(pairs)
    pairs["semantic_similarity"] = np.einsum(
        "ij,ij->i", embeddings[:n_pairs], embeddings[n_pairs:]
    )
    pairs["exact_text_identical"] = (
        pairs["negative_text"] == pairs["positive_text"]
    )

    prompt_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for component_number, (component, subset) in enumerate(
        pairs.groupby("component", sort=False)
    ):
        counts = subset.groupby("prompt_id")["sample_seed_index"].nunique()
        if counts.nunique() != 1:
            raise ValueError(f"Incomplete seed coverage for {component}")
        prompt_means = (
            subset.groupby(["prompt_id", "prompt_index"], as_index=False)
            .agg(
                semantic_similarity=("semantic_similarity", "mean"),
                within_prompt_seed_sd=("semantic_similarity", "std"),
                exact_text_identity_rate=("exact_text_identical", "mean"),
                n_seeds=("sample_seed_index", "nunique"),
            )
            .sort_values("prompt_index")
        )
        prompt_means.insert(0, "component", component)
        prompt_rows.extend(prompt_means.to_dict(orient="records"))
        inference = paired_mean_inference(
            prompt_means["semantic_similarity"].to_numpy(dtype=float),
            bootstrap=args.bootstrap,
            permutations=args.permutations,
            seed=args.seed + component_number,
        )
        summaries.append(
            {
                "component": component,
                "n_prompts": int(len(prompt_means)),
                "n_seeds": int(counts.iloc[0]),
                "n_prompt_seed_pairs": int(len(subset)),
                "minimum_prompt_mean": float(
                    prompt_means["semantic_similarity"].min()
                ),
                "mean_within_prompt_seed_sd": float(
                    prompt_means["within_prompt_seed_sd"].fillna(0).mean()
                ),
                "exact_text_identity_rate": float(
                    subset["exact_text_identical"].mean()
                ),
                **inference,
            }
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pair_columns = [
        "component",
        "prompt_id",
        "prompt_index",
        "sample_seed_index",
        "semantic_similarity",
        "exact_text_identical",
    ]
    pairs[pair_columns].to_csv(
        args.out_dir / "semantic_similarity_by_prompt_seed.csv", index=False
    )
    pd.DataFrame(prompt_rows).to_csv(
        args.out_dir / "semantic_similarity_by_prompt.csv", index=False
    )
    summary = pd.DataFrame(summaries)
    summary.to_csv(args.out_dir / "semantic_similarity_summary.csv", index=False)
    manifest = {
        "started_at": started.isoformat(),
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(args.input_dir.resolve()),
        "inputs": inputs,
        "model_path": str(model_path),
        "components": args.components,
        "endpoint_strength": args.endpoint_strength,
        "device": args.device,
        "batch_size": args.batch_size,
        "bootstrap": args.bootstrap,
        "permutations": args.permutations,
        "seed": args.seed,
        "python": platform.python_version(),
        "sentence_transformers": sentence_transformers.__version__,
        "analysis_unit": (
            "prompt mean after pairing endpoints within decode seed and "
            "averaging over seeds"
        ),
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
