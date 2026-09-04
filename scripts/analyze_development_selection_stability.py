#!/usr/bin/env python3
"""Replay development-only configuration selection under prompt resampling."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs/en/05_linguistic/causal_pullback/calibration_stability"
BOOTSTRAP = 10_000
SEED = 20260903


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_estimands(path: Path) -> dict[str, np.ndarray]:
    frame = pd.read_csv(path)
    frame = frame[
        (frame["component"] == "lexical")
        & frame["estimand"].isin(["target_projection", "component_selectivity"])
    ]
    pivot = frame.pivot(
        index=["prompt_index", "sample_seed_index"],
        columns="estimand", values="value",
    ).reset_index()
    if set(pivot["prompt_index"]) != set(range(20)):
        raise ValueError(f"Expected prompt indices 0--19 in {path}")
    ordered = pivot.sort_values(["prompt_index", "sample_seed_index"])
    n_seeds = int(ordered["sample_seed_index"].nunique())
    return {
        "target": ordered["target_projection"].to_numpy().reshape(20, n_seeds),
        "selectivity": ordered["component_selectivity"].to_numpy().reshape(20, n_seeds),
    }


def parse_cell(path: Path) -> tuple[int, float]:
    return (
        int(path.parents[1].name.removeprefix("layer")),
        float(path.parent.name.removeprefix("strength").replace("p", ".")),
    )


def load_model(model: str) -> tuple[dict, dict, tuple, list[Path]]:
    base = ROOT / "outputs/en/05_linguistic/causal_pullback"
    inputs: list[Path] = []
    if model == "Llama":
        cells = {
            (7, 1.0): base / "Llama-3.1-8B-Instruct/mean_pool/t0_dev_analysis/prompt_seed_estimands.csv",
            (7, 2.0): base / "Llama-3.1-8B-Instruct/mean_pool/strength2_dev_analysis/prompt_seed_estimands.csv",
        }
        quality = {cell: True for cell in cells}
        selected = (7, 2.0)
    elif model == "Qwen":
        root = base / "Qwen3-4B/mean_pool/injection_layer_sweep/dev"
        grid_path = root.parent / "development_grid.csv"
        grid = pd.read_csv(grid_path)
        quality = {(int(row.layer), float(row.strength)): bool(row.quality_eligible) for row in grid.itertuples()}
        cells = {parse_cell(path): path for path in sorted(root.glob("layer*/strength*/prompt_seed_estimands.csv"))}
        selected = (22, 3.0)
        inputs.append(grid_path)
    elif model == "Gemma 4":
        root = base / "gemma-4-E2B-it/mean_pool/injection_layer_sweep/lengthmatched_dev"
        grid_path = root.parent / "development_lengthmatched_grid.csv"
        grid = pd.read_csv(grid_path)
        quality = {(int(row.layer), float(row.strength)): bool(row.quality_eligible) for row in grid.itertuples()}
        cells = {parse_cell(path): path for path in sorted(root.glob("layer*/strength*/prompt_seed_estimands.csv"))}
        selected = (3, 0.25)
        inputs.append(grid_path)
    else:
        raise ValueError(model)
    if set(cells) != set(quality):
        raise ValueError(f"{model}: development grid and estimand files disagree")
    inputs.extend(cells.values())
    return {cell: load_estimands(path) for cell, path in cells.items()}, quality, selected, inputs


def statistics(frame: dict[str, np.ndarray], prompt_indices: np.ndarray) -> dict[str, float]:
    sampled_target = frame["target"][prompt_indices]
    sampled_selectivity = frame["selectivity"][prompt_indices]
    target = float(sampled_target.mean())
    selectivity = float(sampled_selectivity.mean())
    return {
        "target": target,
        "selectivity": selectivity,
        "target_seed_min": float(sampled_target.mean(axis=0).min()),
        "selectivity_seed_min": float(sampled_selectivity.mean(axis=0).min()),
        "score": min(target, selectivity),
    }


def select(
    frames: dict[tuple[int, float], dict[str, np.ndarray]],
    quality: dict[tuple[int, float], bool],
    prompts: np.ndarray,
) -> tuple[tuple[int, float] | None, dict]:
    stats = {cell: statistics(frame, prompts) for cell, frame in frames.items()}
    eligible = [cell for cell, values in stats.items() if quality[cell]
        and values["target"] > 0 and values["selectivity"] > 0
        and values["target_seed_min"] > 0 and values["selectivity_seed_min"] > 0]
    if not eligible:
        return None, stats
    chosen = sorted(eligible, key=lambda cell: (-stats[cell]["score"], cell[1], cell[0]))[0]
    return chosen, stats


def label(cell: tuple[int, float] | None) -> str:
    return "no_eligible_cell" if cell is None else f"L{cell[0]}/{cell[1]:g}"


def main() -> None:
    started = datetime.now(timezone.utc)
    rng = np.random.default_rng(SEED)
    summaries, frequencies, loo_rows = [], [], []
    input_paths: list[Path] = []
    for model in ("Llama", "Qwen", "Gemma 4"):
        frames, quality, frozen, inputs = load_model(model)
        input_paths.extend(inputs)
        full_choice, full_stats = select(frames, quality, np.arange(20))
        if full_choice != frozen:
            raise RuntimeError(f"{model}: replay chose {full_choice}, expected {frozen}")
        counts: Counter[str] = Counter()
        for _ in range(BOOTSTRAP):
            choice, _ = select(frames, quality, rng.integers(0, 20, size=20))
            counts[label(choice)] += 1
        loo_matches = []
        for heldout in range(20):
            choice, _ = select(frames, quality, np.delete(np.arange(20), heldout))
            match = choice == frozen
            loo_matches.append(match)
            loo_rows.append({
                "model": model, "heldout_prompt_index": heldout,
                "selected_cell": label(choice), "matches_frozen_selection": match,
            })
        frozen_label = label(frozen)
        eligible_n = BOOTSTRAP - counts["no_eligible_cell"]
        summaries.append({
            "model": model, "n_prompts": 20, "n_cells": len(frames),
            "quality_eligible_cells": int(sum(quality.values())),
            "frozen_selection": frozen_label,
            "full_target_mean": full_stats[frozen]["target"],
            "full_selectivity_mean": full_stats[frozen]["selectivity"],
            "bootstrap_replicates": BOOTSTRAP,
            "bootstrap_any_eligible_rate": eligible_n / BOOTSTRAP,
            "bootstrap_frozen_selection_rate_all": counts[frozen_label] / BOOTSTRAP,
            "bootstrap_frozen_selection_rate_given_eligible": counts[frozen_label] / eligible_n if eligible_n else np.nan,
            "leave_one_out_frozen_selection_rate": float(np.mean(loo_matches)),
        })
        for selected_cell, count in counts.most_common():
            frequencies.append({
                "model": model, "selected_cell": selected_cell,
                "count": count, "rate": count / BOOTSTRAP,
            })

    summary = pd.DataFrame(summaries)
    frequency = pd.DataFrame(frequencies)
    leave_one_out = pd.DataFrame(loo_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "selection_stability_summary.csv"
    frequency_path = OUT_DIR / "bootstrap_selection_frequencies.csv"
    loo_path = OUT_DIR / "leave_one_prompt_out_selections.csv"
    report_path = OUT_DIR / "analysis_summary.md"
    summary.to_csv(summary_path, index=False)
    frequency.to_csv(frequency_path, index=False)
    leave_one_out.to_csv(loo_path, index=False)
    lines = [
        "# Development-selection stability", "",
        "The frozen selection rule is replayed on 10,000 prompt bootstraps and 20 leave-one-prompt-out folds.",
        "Response-quality gates stay fixed; effect and seedwise sign gates are recomputed per resample.", "",
        "| Model | Cells | Frozen | Any eligible | Frozen/all | Frozen/eligible | Frozen/LOO |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.model} | {row.n_cells} | {row.frozen_selection} | {row.bootstrap_any_eligible_rate:.3f} | "
            f"{row.bootstrap_frozen_selection_rate_all:.3f} | {row.bootstrap_frozen_selection_rate_given_eligible:.3f} | "
            f"{row.leave_one_out_frozen_selection_rate:.3f} |"
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    finished = datetime.now(timezone.utc)
    outputs = (summary_path, frequency_path, loo_path, report_path)
    manifest = {
        "schema": "development_selection_stability_v1", "status": "complete",
        "bootstrap_replicates": BOOTSTRAP, "bootstrap_seed": SEED,
        "resampling_unit": "development prompt",
        "selection_rule": "positive target/selectivity means and seed means; frozen quality gates; maximize weaker mean",
        "inputs": {str(path.relative_to(ROOT)): sha256(path) for path in sorted(set(input_paths))},
        "outputs": {str(path.relative_to(ROOT)): sha256(path) for path in outputs},
        "started_at_utc": started.isoformat(), "finished_at_utc": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(report_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
