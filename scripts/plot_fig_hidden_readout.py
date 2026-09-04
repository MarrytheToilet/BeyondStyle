#!/usr/bin/env python3
"""Appendix figure: generator-held-out readout across depth in three observers.

One panel per observer.  Lines are the mean held-out correlation between the
readout and the surface score over four generator folds and both source sides;
faint points are those eight observations.  Dotted rules mark the intervention
layer and the readout anchor.  Values are read from the frozen validation file.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_figstyle import (  # noqa: E402
    FAINT, FULL_WIDTH, GROUP_COLOR, GROUP_MARKER, INK_SOFT, MUTED, SURFACE,
    axis_legend, clean, group_handle, panel, save, use_paper_style, zero_rule,
)

ROOT = Path(__file__).resolve().parents[1]
VALIDATION = ROOT / "outputs/en/05_linguistic/concept_patterns_layerwise/pattern_validation.csv"
OUT = ROOT / "ACL27/figures/hidden_readout"
GROUPS = ("lexical", "syntactic", "discourse")
MODELS = (
    {"key": "Llama-3.1-8B-Instruct", "label": "Llama 3.1 8B Instruct", "intervention": 7, "anchor": 32, "depth": 32},
    {"key": "Qwen3-4B", "label": "Qwen3 4B", "intervention": 8, "anchor": 36, "depth": 36},
    {"key": "gemma-4-E2B-it", "label": "Gemma 4 E2B-IT", "intervention": 8, "anchor": 35, "depth": 35},
)


def draw() -> None:
    use_paper_style()
    frame = pd.read_csv(VALIDATION)
    rng = np.random.default_rng(3)
    figure = plt.figure(figsize=(FULL_WIDTH, 1.55))
    grid = figure.add_gridspec(1, 3, wspace=0.14, left=0.075, right=0.995, top=0.84, bottom=0.23)
    for index, spec in enumerate(MODELS):
        axis = figure.add_subplot(grid[0, index])
        block = frame[frame["model"].eq(spec["key"])]
        layers = np.array(sorted(block["layer"].unique()))
        for group in GROUPS:
            color = GROUP_COLOR[group]
            sub = block[block["group"].eq(group)]
            means = np.array([sub[sub["layer"].eq(L)]["heldout_score_correlation"].mean() for L in layers])
            pts = sub[["layer", "heldout_score_correlation"]].to_numpy(float)
            axis.scatter(pts[:, 0] + rng.uniform(-0.35, 0.35, len(pts)), pts[:, 1], s=4.5,
                         color=color, alpha=0.28, linewidth=0, zorder=2)
            axis.plot(layers, means, color=color, linewidth=1.4, zorder=4)
            axis.scatter(layers, means, s=13, marker=GROUP_MARKER[group], color=color,
                         edgecolor=SURFACE, linewidth=0.5, zorder=5)
        for L, name in ((spec["intervention"], r"$\ell$"), (spec["anchor"], r"$k$")):
            axis.axvline(L, color=FAINT, linewidth=0.7, linestyle=(0, (2, 2)), zorder=1)
            axis.text(L, 0.71, name, ha="center", va="bottom", fontsize=6.4, color=MUTED)
        zero_rule(axis, vertical=False)
        axis.set_xlim(layers[0] - 1.5, spec["depth"] + 1.5)
        axis.set_xticks(layers)
        axis.set_xticklabels([str(L) for L in layers], fontsize=5.8)
        axis.set_ylim(-0.32, 0.72)
        axis.set_yticks([-0.2, 0.0, 0.2, 0.4, 0.6])
        axis.set_xlabel("Layer")
        clean(axis, grid="y")
        if index == 0:
            axis.set_ylabel("Held-out correlation")
        else:
            axis.set_yticklabels([])
        if index == len(MODELS) - 1:
            axis_legend(axis, [group_handle(g) for g in GROUPS], loc="lower right", fontsize=5.8)
        panel(axis, "ABC"[index], spec["label"], y=1.05)
    save(figure, OUT)


if __name__ == "__main__":
    draw()
