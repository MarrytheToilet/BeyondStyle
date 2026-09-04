#!/usr/bin/env python3
"""Figure: fixed-token transport against 100 complete refitted null pipelines.

Panels A and B: learned depth profile drawn as a bold curve inside the ribbon
its own refitted nulls occupy (outer band = full range, inner = interquartile).
Panel C: ridgeline densities of the null curve statistic for the four
prespecified tests, with the learned statistic as a vertical marker.  The
per-null statistic is recomputed with the audit's depth-normalized trapezoid
and reproduces the recorded learned/mean/min/max values exactly.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_figstyle import (  # noqa: E402
    FULL_WIDTH, GROUP_COLOR, GROUP_LABEL, GROUP_MARKER, GROUP_TINT, INK, MUTED,
    RULE, SURFACE, clean, panel, save, use_paper_style,
)

ROOT = Path(__file__).resolve().parents[1]
PULLBACK = ROOT / "outputs/en/05_linguistic/causal_pullback"
OUT = ROOT / "ACL27/figures/transport_null"
CONCEPTS = ("lexical", "syntactic")
MODELS = (
    {"key": "Llama-3.1-8B-Instruct", "label": "Llama 3.1 8B", "short": "Llama", "depth": 32},
    {"key": "Qwen3-4B", "label": "Qwen3 4B", "short": "Qwen", "depth": 36},
)


def audit_root(model: str) -> Path:
    base = PULLBACK / model / "mean_pool"
    confirmatory = base / "confirmatory_hidden_audit"
    return confirmatory if (confirmatory / "matched_nulls").exists() else base


def curve_statistic(values: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.trapz(values, x, axis=-1) / (x[-1] - x[0])


def load(model: str, depth: int) -> dict:
    root = audit_root(model)
    globals_ = pd.read_csv(root / "matched_nulls/global_curve_statistics.csv")
    nulls = pd.read_csv(root / "matched_nulls/matched_null_axis_statistics.csv")
    learned = pd.read_csv(root / "layerwise_transport/learned_transport.csv")
    layers = [int(v) for v in str(globals_.loc[0, "propagation_layers"]).split(",")]
    x = np.asarray(layers, dtype=float) / depth
    out = {"x": x, "concepts": {}}
    for concept in CONCEPTS:
        summary = globals_[globals_["concept"].eq(concept)].iloc[0]
        curve = learned[learned["concept"].eq(concept)].set_index("layer")["mean"].reindex(layers).to_numpy(float)
        matrix = (nulls[nulls["concept"].eq(concept)]
                  .pivot(index="null_axis", columns="layer", values="mean_transport_capacity")[layers]
                  .to_numpy(float))
        out["concepts"][concept] = {
            "curve": curve, "null_matrix": matrix,
            "learned_statistic": float(summary["learned_curve_statistic"]),
            "null_statistics": curve_statistic(matrix, x),
            "p_value": float(summary["finite_null_p_value"]),
        }
    return out


def panel_profile(axis: plt.Axes, data: dict, letter: str, label: str) -> None:
    x = data["x"]
    for concept in CONCEPTS:
        block = data["concepts"][concept]
        color = GROUP_COLOR[concept]
        m = block["null_matrix"]
        axis.fill_between(x, m.min(axis=0), m.max(axis=0), color=color, alpha=0.11, linewidth=0, zorder=1)
        axis.fill_between(x, np.percentile(m, 25, axis=0), np.percentile(m, 75, axis=0),
                          color=color, alpha=0.18, linewidth=0, zorder=2)
        axis.plot(x, block["curve"], color=color, linewidth=1.7, zorder=4)
        axis.scatter(x, block["curve"], s=14, marker=GROUP_MARKER[concept], color=color,
                     edgecolor=SURFACE, linewidth=0.55, zorder=5)
    if letter == "A":
        ends = {c: data["concepts"][c]["curve"][-1] for c in CONCEPTS}
        if abs(ends["lexical"] - ends["syntactic"]) < 0.07:
            mid = (ends["lexical"] + ends["syntactic"]) / 2
            ends["lexical"], ends["syntactic"] = mid + 0.04, mid - 0.04
        for concept in CONCEPTS:
            axis.text(x[-1] + 0.018, ends[concept], GROUP_LABEL[concept], ha="left",
                      va="center", fontsize=6.0, color=GROUP_COLOR[concept], fontweight="bold")
        axis.set_xlim(x[0] - 0.04, x[-1] + 0.22)
    else:
        axis.set_xlim(x[0] - 0.04, x[-1] + 0.06)
    axis.set_ylim(0.0, 0.70)
    axis.set_yticks([0.0, 0.2, 0.4, 0.6])
    axis.set_xticks([0.4, 0.6, 0.8])
    axis.set_xlabel("Relative depth")
    clean(axis, grid="y")
    panel(axis, letter, label, y=1.03)


def panel_ridges(axis: plt.Axes, bundle: dict[str, dict]) -> None:
    rows = [(s["short"], c, bundle[s["key"]]["concepts"][c]) for s in MODELS for c in CONCEPTS]
    grid = np.linspace(0.08, 0.51, 400)
    for i, (short, concept, block) in enumerate(rows):
        base = float(len(rows) - 1 - i)
        color = GROUP_COLOR[concept]
        nulls = block["null_statistics"]
        density = gaussian_kde(nulls)(grid)
        density = density / density.max() * 0.82
        axis.fill_between(grid, base, base + density, color=GROUP_TINT[concept], zorder=2)
        axis.plot(grid, base + density, color=color, linewidth=0.9, zorder=3)
        axis.plot([grid[0], grid[-1]], [base, base], color=RULE, linewidth=0.5, zorder=1)
        t = block["learned_statistic"]
        axis.plot([t, t], [base - 0.05, base + 0.88], color=color, linewidth=2.0, zorder=5,
                  solid_capstyle="round")
        cleared = t > nulls.max()
        axis.text(grid[-1] - 0.004, base + 0.45, f"$p = {block['p_value']:.4f}$",
                  ha="right", va="center", fontsize=6.0, color=INK if cleared else MUTED, zorder=6)
        axis.text(grid[0] - 0.005, base + 0.30, f"{short}\n{GROUP_LABEL[concept].lower()}",
                  ha="right", va="center", fontsize=5.6, color=color, fontweight="bold",
                  linespacing=1.15)
    axis.set_xlim(grid[0], grid[-1])
    axis.set_ylim(-0.15, len(rows) + 0.05)
    axis.set_yticks([])
    axis.set_xticks([0.1, 0.2, 0.3, 0.4])
    axis.set_xlabel("Depth-normalized transport area $T$")
    clean(axis, grid=None)
    axis.spines["left"].set_visible(False)
    panel(axis, "C", "Learned statistic against its null density", y=1.03)


def draw() -> None:
    use_paper_style()
    bundle = {s["key"]: load(s["key"], s["depth"]) for s in MODELS}
    figure = plt.figure(figsize=(FULL_WIDTH, 1.32))
    outer = figure.add_gridspec(1, 2, width_ratios=[1.58, 1.22], wspace=0.30,
                                left=0.060, right=0.995, top=0.86, bottom=0.185)
    profiles = outer[0, 0].subgridspec(1, 2, wspace=0.10)
    first = figure.add_subplot(profiles[0, 0])
    panel_profile(first, bundle[MODELS[0]["key"]], "A", MODELS[0]["label"])
    first.set_ylabel("Mean transport capacity")
    second = figure.add_subplot(profiles[0, 1])
    panel_profile(second, bundle[MODELS[1]["key"]], "B", MODELS[1]["label"])
    second.set_yticklabels([])
    panel_ridges(figure.add_subplot(outer[0, 1]), bundle)
    save(figure, OUT)


if __name__ == "__main__":
    draw()
