#!/usr/bin/env python3
"""Combine greedy, temperature-0.7, and temperature-1 audits."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = (
    ROOT
    / "outputs/en/05_linguistic/causal_pullback/decoding_regime_summary"
)
MODELS = ("Llama-3.1-8B-Instruct", "Qwen3-4B")
REGIMES = {
    "greedy": "decode_t0_robustness",
    "temperature 0.7": "confirmatory_seed_robustness",
    "temperature 1.0": "decode_t1_robustness",
}


def main() -> None:
    rows: list[pd.DataFrame] = []
    for model in MODELS:
        base = (
            ROOT
            / "outputs/en/05_linguistic/causal_pullback"
            / model
            / "mean_pool"
        )
        for regime, directory in REGIMES.items():
            path = base / directory / "seed_robustness_summary.csv"
            if not path.exists():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path)
            frame.insert(1, "decoding_regime", regime)
            rows.append(frame)
    combined = pd.concat(rows, ignore_index=True)
    OUT.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT / "decoding_regime_summary.csv", index=False)
    manifest = {
        "schema": "decoding_regime_summary_v1",
        "models": list(MODELS),
        "regimes": list(REGIMES),
        "source_rows": int(len(combined)),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUT / 'decoding_regime_summary.csv'}")


if __name__ == "__main__":
    main()
