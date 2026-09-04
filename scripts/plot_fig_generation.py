#!/usr/bin/env python3
"""Figure: selective lexical control of free generation in three model families.

Panel A is a selectivity plane: for each model the target response (y) is
plotted against each off-target group's response (x) with 95% prompt-clustered
bootstrap intervals; points above the diagonal are selective.  Panel B shows
the empirical cumulative distribution of the seed-averaged prompt-level target
effect for each model.  Panel C shows the three decode-seed means of target and
selectivity for each model with the pooled estimate.  All values are read from
the frozen held-out summaries.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_figstyle import (  # noqa: E402
    FAINT, FULL_WIDTH, GRID, GROUP_COLOR, GROUP_LABEL, GROUP_MARKER, INK,
    INK_SOFT, LEX, LEX_MID, LEX_TINT, MUTED, RULE, SURFACE, axis_legend, clean,
    panel, save, use_paper_style, zero_rule,
)

ROOT = Path(__file__).resolve().parents[1]
PULLBACK = ROOT / "outputs/en/05_linguistic/causal_pullback"
OUT = ROOT / "ACL27/figures/generation_control"


def result_root(model: str, legacy: str) -> Path:
    """Prefer the public reproduction layout, with legacy artifact fallback."""
    current = PULLBACK / model / "mean_pool/selected_generation/heldout_analysis"
    return current if current.exists() else PULLBACK / model / "mean_pool" / legacy


MODELS = (
    {"label": "Llama 3.1 8B Instruct", "short": "Llama", "color": "#1E2732",
     "root": result_root("Llama-3.1-8B-Instruct", "strength2_seed_robustness_dual5090")},
    {"label": "Qwen3 4B", "short": "Qwen", "color": "#2E7EB8",
     "root": result_root("Qwen3-4B", "injection_layer_sweep/heldout_selected")},
    {"label": "Gemma 4 E2B-IT", "short": "Gemma", "color": "#7FB4DC",
     "root": result_root("gemma-4-E2B-it", "injection_layer_sweep/heldout_selected_lengthmatched")},
)
OFF_TARGET = ("syntactic", "discourse")
MARK = dict(edgecolor=SURFACE, linewidth=0.55, zorder=6)


def load(spec: dict) -> dict:
    root = spec["root"]
    robust = pd.read_csv(root / "seed_robustness_summary.csv")
    robust = robust[robust["component"].eq("lexical")].set_index("estimand")
    groups = pd.read_csv(root / "group_response_summary.csv")
    groups = groups[groups["component"].eq("lexical")].set_index("response_group")
    est = pd.read_csv(root / "prompt_seed_estimands.csv")
    est = est[est["estimand"].eq("target_projection")]

    def triple(row) -> tuple[float, float, float]:
        return float(row["mean"]), float(row["ci_low"]), float(row["ci_high"])

    def seeds(row) -> list[float]:
        return [float(v) for v in json.loads(row["seed_means"]).values()]

    return {
        "target": triple(robust.loc["target_projection"]),
        "selectivity": triple(robust.loc["component_selectivity"]),
        "off": {g: triple(groups.loc[g]) for g in OFF_TARGET},
        "seeds": {"target": seeds(robust.loc["target_projection"]),
                  "selectivity": seeds(robust.loc["component_selectivity"])},
        "prompt_means": np.sort(est.groupby("prompt_id")["value"].mean().to_numpy()),
    }


def selectivity_plane(axis: plt.Axes, bundle: list[dict]) -> None:
    lo, hi = -0.75, 1.65
    axis.fill_between([lo, hi], [lo, hi], [lo, lo], color="#F1F3F5", zorder=0)
    axis.plot([lo, hi], [lo, hi], color=RULE, linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)
    axis.set_xlim(lo, 0.85)
    axis.set_ylim(-0.15, 1.95)
    axis.text(0.82, -0.10, "not selective", ha="right", va="bottom", fontsize=5.2, color=MUTED, zorder=2)
    for spec, data in zip(MODELS, bundle):
        ty, tlo, thi = data["target"]
        xs = []
        for group in OFF_TARGET:
            x, xlo, xhi = data["off"][group]
            xs.append(x)
            color = GROUP_COLOR[group]
            axis.plot([xlo, xhi], [ty, ty], color=color, linewidth=0.9, zorder=4)
            axis.plot([x, x], [tlo, thi], color=color, linewidth=0.9, zorder=4)
            axis.scatter([x], [ty], s=17, marker=GROUP_MARKER[group], color=color, **MARK)
        axis.plot(xs, [ty, ty], color=FAINT, linewidth=0.6, zorder=3)
        axis.text(max(xs) + 0.24, ty, spec["short"], ha="left", va="center", fontsize=6.0,
                  color=INK, fontweight="bold", zorder=7)
    zero_rule(axis)
    zero_rule(axis, vertical=False)
    axis.set_xticks([-0.5, 0.0, 0.5])
    axis.set_yticks([0.0, 0.5, 1.0, 1.5])
    axis.set_xlabel("Off-target response")
    axis.set_ylabel("Target response")
    clean(axis, grid=None)
    handles = [Line2D([0], [0], marker=GROUP_MARKER[g], color=GROUP_COLOR[g], markerfacecolor=GROUP_COLOR[g],
                      markeredgecolor=SURFACE, markersize=3.8, linewidth=0.9, label=f"vs. {GROUP_LABEL[g].lower()}")
               for g in OFF_TARGET]
    axis_legend(axis, handles, loc="upper left", fontsize=5.6, labelspacing=0.15)
    panel(axis, "A", "Selectivity plane", y=1.04)


def prompt_ecdf(axis: plt.Axes, bundle: list[dict]) -> None:
    for spec, data in zip(MODELS, bundle):
        values = data["prompt_means"]
        y = np.arange(1, values.size + 1) / values.size
        axis.step(values, y, where="post", color=spec["color"], linewidth=1.3, zorder=3)
    axis.axhline(0.5, color=GRID, linewidth=0.6, zorder=1)
    zero_rule(axis)
    axis.set_xlim(-4.0, 6.0)
    axis.set_ylim(0, 1.0)
    axis.set_xticks([-4, -2, 0, 2, 4, 6])
    axis.set_yticks([0, 0.5, 1.0])
    axis.set_yticklabels(["0", ".5", "1"])
    axis.set_xlabel("Prompt-level target effect")
    axis.set_ylabel("Share of prompts")
    clean(axis, grid=None)
    handles = [Line2D([0], [0], color=s["color"], linewidth=1.3, label=s["short"]) for s in MODELS]
    axis_legend(axis, handles, loc="upper left", fontsize=5.6, labelspacing=0.15)
    panel(axis, "B", "166 held-out prompts", y=1.04)


def seed_ladder(axis: plt.Axes, bundle: list[dict]) -> None:
    styles = {"target": (LEX, "o"), "selectivity": (INK, "D")}
    for i, (spec, data) in enumerate(zip(MODELS, bundle)):
        for offset, key in ((-0.17, "target"), (0.17, "selectivity")):
            color, marker = styles[key]
            x = i + offset
            seeds = data["seeds"][key]
            axis.plot([x, x], [min(seeds), max(seeds)], color=color, linewidth=0.8, alpha=0.7, zorder=3)
            axis.scatter([x] * len(seeds), seeds, s=9, color=SURFACE, edgecolor=color, linewidth=0.8, zorder=4)
            axis.scatter([x], [data[key][0]], s=20, marker=marker, color=color, **MARK)
    zero_rule(axis, vertical=False)
    axis.set_xlim(-0.6, len(MODELS) - 0.4)
    axis.set_ylim(-0.15, 2.05)
    axis.set_xticks(range(len(MODELS)))
    axis.set_xticklabels(["Llama 3.1\n8B Instruct", "Qwen3\n4B", "Gemma 4\nE2B-IT"], fontsize=5.7)
    axis.set_yticks([0.0, 0.5, 1.0, 1.5])
    axis.set_ylabel("Effect")
    clean(axis, grid="y")
    handles = [
        Line2D([0], [0], marker="o", color=LEX, markerfacecolor=LEX, markeredgecolor=SURFACE, markersize=4.2,
               linewidth=0, label="target, pooled"),
        Line2D([0], [0], marker="D", color=INK, markerfacecolor=INK, markeredgecolor=SURFACE, markersize=3.8,
               linewidth=0, label="selectivity, pooled"),
        Line2D([0], [0], marker="o", color=MUTED, markerfacecolor=SURFACE, markeredgecolor=MUTED, markersize=3.2,
               linewidth=0, label="single seed"),
    ]
    axis_legend(axis, handles, loc="upper left", fontsize=5.4, labelspacing=0.12, ncol=1)
    panel(axis, "C", "Three decode seeds", y=1.04)


def draw() -> None:
    use_paper_style()
    bundle = [load(spec) for spec in MODELS]
    figure = plt.figure(figsize=(FULL_WIDTH, 1.52))
    grid = figure.add_gridspec(1, 3, width_ratios=[1.0, 1.05, 1.0], wspace=0.36,
                               left=0.065, right=0.995, top=0.87, bottom=0.21)
    selectivity_plane(figure.add_subplot(grid[0, 0]), bundle)
    prompt_ecdf(figure.add_subplot(grid[0, 1]), bundle)
    seed_ladder(figure.add_subplot(grid[0, 2]), bundle)
    save(figure, OUT)


if __name__ == "__main__":
    draw()
