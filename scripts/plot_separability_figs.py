#!/usr/bin/env python3
"""Generate §4.2 separability figures from `insample_metrics.json` + `random_holdout.json`.

Outputs go next to the JSON in `outputs/{lang}/02_separability/<model>/<dataset>/figures/`.
Color scheme: light pink (AI) + light blue (Human) + accent purple, via `palette.py`.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt

from src.visualization.palette import (
    setup_style, save_fig, H, A, H_D, A_D, PALETTE, BP_SEQ,
)


def _load(p: Path):
    with open(p) as f:
        return json.load(f)


METHOD_COLORS = {
    "mean_diff":            H_D,
    "mean_diff_nullspace":  PALETTE["accent"],
    "mean_diff_fisher":     A_D,
}
METHOD_LABELS = {
    "mean_diff":           "Raw mean-diff",
    "mean_diff_nullspace": "Nullspace-projected",
    "mean_diff_fisher":    "Fisher-whitened",
}


def plot_layer_auc(layer_metrics: dict, fig_dir: Path) -> Path:
    """Per-layer AUC for each direction-estimation method."""
    methods = list(layer_metrics["methods"].keys())
    layers = sorted({int(l) for m in methods for l in layer_metrics["methods"][m]})
    fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.6))
    for m in methods:
        ys = [layer_metrics["methods"][m].get(str(l), {}).get("auc", np.nan) for l in layers]
        ax.plot(layers, ys, marker="o", lw=2, ms=6,
                color=METHOD_COLORS.get(m, "#888"),
                label=METHOD_LABELS.get(m, m))
    ax.set_xlabel("Layer")
    ax.set_ylabel("ROC-AUC (in-sample)")
    ax.set_ylim(0.92, 1.005)
    ax.set_title("In-sample per-layer separability of $\\hat v_{src}$")
    ax.legend(loc="lower right", frameon=False)
    return save_fig(fig, fig_dir, "insample_auc.png")


def plot_layer_geometry(layer_metrics: dict, fig_dir: Path) -> Path:
    """Hull-norm + Fisher ratio across layers (geometry view, mean_diff method)."""
    m = "mean_diff" if "mean_diff" in layer_metrics["methods"] else next(iter(layer_metrics["methods"]))
    rec = layer_metrics["methods"][m]
    layers = sorted(int(l) for l in rec)
    hull = [rec[str(l)].get("hull_centroid_dist_norm", np.nan) for l in layers]
    fisher = [rec[str(l)].get("fisher_ratio", np.nan) for l in layers]

    fig, ax1 = plt.subplots(1, 1, figsize=(6.5, 3.6))
    ax1.plot(layers, hull, marker="o", color=H_D, lw=2, label="Hull-norm distance")
    ax1.set_xlabel("Layer")
    ax1.set_ylabel("Normalised hull–centroid dist", color=H_D)
    ax1.tick_params(axis="y", labelcolor=H_D)

    ax2 = ax1.twinx()
    ax2.plot(layers, fisher, marker="s", color=A_D, lw=2, label="Fisher ratio")
    ax2.set_ylabel("Fisher ratio", color=A_D)
    ax2.tick_params(axis="y", labelcolor=A_D)
    ax2.grid(False)

    ax1.set_title("Layer geometry of human vs AI clouds")
    return save_fig(fig, fig_dir, "layer_geometry.png")


def plot_heldout_bands(heldout: dict, fig_dir: Path) -> Path:
    """Mean ± std held-out AUC bands per method across layers."""
    layers = sorted(int(l) for l in heldout["results"])
    methods = list(next(iter(heldout["results"].values())).keys())
    name_map = {"raw": "mean_diff", "nullspace": "mean_diff_nullspace", "fisher": "mean_diff_fisher"}

    fig, ax = plt.subplots(1, 1, figsize=(6.5, 3.6))
    for m in methods:
        means = np.array([heldout["results"][str(l)][m]["heldout_auc_mean"] for l in layers])
        stds  = np.array([heldout["results"][str(l)][m]["heldout_auc_std"]  for l in layers])
        c = METHOD_COLORS.get(name_map.get(m, m), "#888")
        label = METHOD_LABELS.get(name_map.get(m, m), m)
        ax.plot(layers, means, marker="o", lw=2, color=c, label=label)
        ax.fill_between(layers, means - stds, means + stds, color=c, alpha=0.18, lw=0)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Random pair held-out ROC-AUC")
    ax.set_ylim(0.94, 1.005)
    n = heldout.get("n_total_pairs", "?")
    runs = heldout.get("n_runs", "?")
    ax.set_title(f"Random pair held-out separability ({runs}× random 80/20 splits, N={n})")
    ax.legend(loc="lower right", frameon=False)
    return save_fig(fig, fig_dir, "random_holdout_auc.png")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=Path,
                   default=Path("outputs/en/02_separability/Llama-3.1-8B-Instruct/mixed_500"))
    args = p.parse_args()

    setup_style()
    fig_dir = args.in_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    lm = _load(args.in_dir / "insample_metrics.json")
    print("→", plot_layer_auc(lm, fig_dir))
    print("→", plot_layer_geometry(lm, fig_dir))

    ho_path = args.in_dir / "random_holdout.json"
    if ho_path.exists():
        ho = _load(ho_path)
        print("→", plot_heldout_bands(ho, fig_dir))


if __name__ == "__main__":
    main()
