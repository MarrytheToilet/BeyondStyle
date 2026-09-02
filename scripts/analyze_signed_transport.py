#!/usr/bin/env python3
"""Summarize the signed fixed-token response of the canonical pullbacks.

The matched-pipeline comparison must use absolute transport because a
permuted target has no semantic pole.  The learned axes do have a canonical
machine-oriented pole, however, so this audit separately asks whether +w
moves each layer-specific learned readout in that declared direction.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta


ROOT = Path(__file__).resolve().parent.parent
PULLBACK_ROOT = ROOT / "outputs" / "en" / "05_linguistic" / "causal_pullback"
OUT = PULLBACK_ROOT / "signed_transport_summary.csv"
MANIFEST = PULLBACK_ROOT / "signed_transport_summary.manifest.json"

MODEL_SPECS = {
    "Llama-3.1-8B-Instruct": {"injection": 7, "anchor": 32},
    "Qwen3-4B": {"injection": 8, "anchor": 36},
}
CONCEPTS = ("lexical", "syntactic")
BOOTSTRAP_REPETITIONS = 10_000
SEED = 20260729


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_binomial_interval(
    positive: int,
    total: int,
    *,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Return the two-sided Clopper--Pearson interval."""
    low = (
        0.0
        if positive == 0
        else float(beta.ppf(alpha / 2, positive, total - positive + 1))
    )
    high = (
        1.0
        if positive == total
        else float(
            beta.ppf(
                1 - alpha / 2,
                positive + 1,
                total - positive,
            )
        )
    )
    return low, high


def main() -> None:
    rng = np.random.default_rng(SEED)
    records: list[dict[str, object]] = []
    sources: list[Path] = []

    for model, spec in MODEL_SPECS.items():
        path = (
            PULLBACK_ROOT
            / model
            / "mean_pool"
            / "confirmatory_hidden_audit"
            / "layerwise_transport"
            / "per_prompt_transport.csv"
        )
        sources.append(path)
        frame = pd.read_csv(path)
        learned = frame[
            (frame["intervention_kind"] == "pullback")
            & (frame["intervention"] == frame["readout"])
            & frame["intervention"].isin(CONCEPTS)
        ].copy()

        for concept in CONCEPTS:
            concept_frame = learned[learned["intervention"] == concept]
            for layer, layer_frame in concept_frame.groupby(
                "layer",
                sort=True,
            ):
                values = layer_frame["transport_coefficient"].to_numpy(
                    dtype=np.float64
                )
                if len(values) != 100:
                    raise ValueError(
                        f"{model} {concept} L{layer}: expected 100 "
                        f"fixed sequences, found {len(values)}"
                    )
                bootstrap_indices = rng.integers(
                    0,
                    len(values),
                    size=(BOOTSTRAP_REPETITIONS, len(values)),
                )
                bootstrap_means = values[bootstrap_indices].mean(axis=1)
                mean_low, mean_high = np.quantile(
                    bootstrap_means,
                    [0.025, 0.975],
                )
                positive = int(np.sum(values > 0))
                positive_low, positive_high = exact_binomial_interval(
                    positive,
                    len(values),
                )
                layer = int(layer)
                if layer == int(spec["injection"]):
                    role = "injection"
                elif layer == int(spec["anchor"]):
                    role = "anchor"
                else:
                    role = "propagation"
                records.append(
                    {
                        "model": model,
                        "concept": concept,
                        "layer": layer,
                        "role": role,
                        "n_sequences": len(values),
                        "signed_transport_mean": float(values.mean()),
                        "signed_transport_ci_low": float(mean_low),
                        "signed_transport_ci_high": float(mean_high),
                        "positive_count": positive,
                        "positive_fraction": positive / len(values),
                        "positive_fraction_ci_low": positive_low,
                        "positive_fraction_ci_high": positive_high,
                    }
                )

    summary = pd.DataFrame.from_records(records)
    summary.to_csv(OUT, index=False)
    propagation = summary[summary["role"] == "propagation"]
    manifest = {
        "analysis": "canonical learned-axis signed fixed-token transport",
        "created_by": str(Path(__file__).relative_to(ROOT)),
        "bootstrap": {
            "unit": "fixed sequence",
            "repetitions": BOOTSTRAP_REPETITIONS,
            "seed": SEED,
            "interval": "percentile 95%",
        },
        "positive_fraction_interval": "two-sided 95% Clopper-Pearson",
        "filters": {
            "intervention_kind": "pullback",
            "intervention_equals_readout": True,
        },
        "interpretation": (
            "The canonical +w pole is machine-oriented. This signed audit "
            "does not replace the absolute learned-versus-matched-null "
            "test, because permuted null directions have arbitrary poles."
        ),
        "propagation_totals": [
            {
                "model": model,
                "concept": concept,
                "positive": int(part["positive_count"].sum()),
                "total": int(part["n_sequences"].sum()),
            }
            for (model, concept), part in propagation.groupby(
                ["model", "concept"],
                sort=False,
            )
        ],
        "sources": [
            {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
            }
            for path in sources
        ],
        "output": {
            "path": str(OUT.relative_to(ROOT)),
            "sha256": sha256_file(OUT),
        },
    }
    MANIFEST.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUT.relative_to(ROOT)} ({len(summary)} layer rows)")
    print(f"Wrote {MANIFEST.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
