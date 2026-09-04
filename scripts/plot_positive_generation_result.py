#!/usr/bin/env python3
"""Render the held-out selective lexical-control figures from frozen CSVs."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parents[1]
LLAMA_DATA = ROOT / "outputs/en/05_linguistic/causal_pullback/Llama-3.1-8B-Instruct/mean_pool/strength2_seed_robustness_dual5090"
OUT = ROOT / "ACL27/figures"

MODEL_SPECS = (
    {
        "label": "Llama",
        "analysis": LLAMA_DATA,
        "semantics": LLAMA_DATA / "guardrails_minilm/semantics",
    },
    {
        "label": "Qwen",
        "analysis": ROOT / "outputs/en/05_linguistic/causal_pullback/Qwen3-4B/mean_pool/injection_layer_sweep/heldout_selected",
        "semantics": ROOT / "outputs/en/05_linguistic/causal_pullback/Qwen3-4B/mean_pool/injection_layer_sweep/heldout_selected/guardrails/semantics",
    },
    {
        "label": "Gemma 4",
        "analysis": ROOT / "outputs/en/05_linguistic/causal_pullback/gemma-4-E2B-it/mean_pool/injection_layer_sweep/heldout_selected_lengthmatched",
        "semantics": ROOT / "outputs/en/05_linguistic/causal_pullback/gemma-4-E2B-it/mean_pool/injection_layer_sweep/heldout_selected_lengthmatched/guardrails/semantics",
    },
)

INK = "#243143"
MUTED = "#667386"
GRID = "#DCE7EF"
BLUE = "#55A8E2"
BLUE_DARK = "#287CB8"
BLUE_LIGHT = "#D8EFFB"
PINK = "#F28EB1"
PINK_DARK = "#C85D87"
PINK_LIGHT = "#FCE1EB"
WHITE = "#FFFFFF"


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.0,
        "axes.titlesize": 8.7,
        "axes.labelsize": 8.1,
        "xtick.labelsize": 7.4,
        "ytick.labelsize": 7.4,
        "axes.edgecolor": "#AAB7C4",
        "axes.linewidth": 0.65,
        "axes.facecolor": WHITE,
        "figure.facecolor": WHITE,
        "savefig.facecolor": WHITE,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def clean_axis(axis: plt.Axes) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(width=0.6, length=2.5, color="#8EA0B2")
    axis.grid(axis="x", color=GRID, linewidth=0.55, zorder=0)


def panel_title(axis: plt.Axes, letter: str, title: str, y: float = 1.13) -> None:
    axis.text(0.0, y, letter, transform=axis.transAxes, ha="left", va="bottom", fontsize=10.4, fontweight="bold", color=INK)
    axis.text(0.14, y, title, transform=axis.transAxes, ha="left", va="bottom", fontsize=8.8, color=INK)


def load_llama_summary() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed = pd.read_csv(LLAMA_DATA / "seed_robustness_summary.csv")
    group = pd.read_csv(LLAMA_DATA / "group_response_summary.csv")
    pairwise = pd.read_csv(LLAMA_DATA / "pairwise_selectivity_summary.csv")
    prompt_estimands = pd.read_csv(LLAMA_DATA / "prompt_seed_estimands.csv")
    return seed, group, pairwise, prompt_estimands


def load_crossmodel_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    effects: list[dict[str, object]] = []
    semantics: list[dict[str, object]] = []
    for spec in MODEL_SPECS:
        seed = pd.read_csv(spec["analysis"] / "seed_robustness_summary.csv")
        lexical = seed[seed["component"].eq("lexical")]
        for estimand in ("target_projection", "component_selectivity"):
            row = lexical[lexical["estimand"].eq(estimand)].iloc[0]
            effects.append({
                "model": spec["label"],
                "estimand": estimand,
                "mean": float(row["mean"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
            })
        pairwise = pd.read_csv(spec["analysis"] / "pairwise_selectivity_summary.csv")
        pairwise = pairwise[pairwise["component"].eq("lexical")].set_index("comparison_group")
        for group, estimand in (("syntactic", "target_minus_syntax"), ("discourse", "target_minus_discourse")):
            row = pairwise.loc[group]
            effects.append({
                "model": spec["label"],
                "estimand": estimand,
                "mean": float(row["mean"]),
                "ci_low": float(row["ci_low"]),
                "ci_high": float(row["ci_high"]),
            })
        semantic = pd.read_csv(
            spec["semantics"] / "semantic_similarity_summary.csv"
        )
        row = semantic[semantic["component"].eq("lexical")].iloc[0]
        semantics.append({
            "model": spec["label"],
            "mean": float(row["mean"]),
            "ci_low": float(row["ci_low"]),
            "ci_high": float(row["ci_high"]),
        })
    return pd.DataFrame(effects), pd.DataFrame(semantics)


def prompt_means(prompt_estimands: pd.DataFrame) -> pd.DataFrame:
    return prompt_estimands.groupby(["prompt_id", "prompt_index", "estimand"], as_index=False)["value"].mean()


def plot_main(effects: pd.DataFrame, semantics: pd.DataFrame) -> None:
    figure = plt.figure(figsize=(3.45, 2.08))
    grid = figure.add_gridspec(1, 2, width_ratios=[1.22, 0.78], wspace=0.45)
    response = figure.add_subplot(grid[0, 0])
    semantic = figure.add_subplot(grid[0, 1])
    models = [spec["label"] for spec in MODEL_SPECS]
    y = np.arange(len(models))[::-1]
    style = {
        "target_projection": (BLUE, BLUE_DARK, "o", 0.24, True),
        "component_selectivity": (PINK, PINK_DARK, "D", 0.08, True),
        "target_minus_syntax": (WHITE, BLUE_DARK, "o", -0.08, False),
        "target_minus_discourse": (WHITE, PINK_DARK, "D", -0.24, False),
    }

    response.axvline(0, color=INK, linewidth=0.75, zorder=1)
    for estimand, (color, dark, marker, offset, primary) in style.items():
        rows = effects[effects["estimand"].eq(estimand)].set_index("model")
        for ypos, model in zip(y, models):
            row = rows.loc[model]
            response.plot(
                [row["ci_low"], row["ci_high"]],
                [ypos + offset, ypos + offset],
                color=dark,
                linewidth=1.05 if primary else 0.85,
                linestyle="-" if primary else (0, (2.0, 1.2)),
                solid_capstyle="round",
                zorder=3,
            )
            response.scatter(
                row["mean"],
                ypos + offset,
                s=23 if primary else 18,
                marker=marker,
                color=color,
                edgecolor=WHITE if primary else dark,
                linewidth=0.55 if primary else 0.85,
                zorder=4,
            )
    response.set_yticks(y, models)
    response.set_xlim(-0.12, 1.66)
    response.set_xticks([0.0, 0.5, 1.0, 1.5])
    response.set_ylim(-0.56, 2.56)
    response.set_xlabel("Paired surface-score effect")
    response.tick_params(axis="y", length=0, pad=2)
    panel_title(response, "A", "Lexical realization", y=1.43)
    clean_axis(response)
    response.legend(
        handles=[
            Line2D([0], [0], marker="o", color=BLUE_DARK, markerfacecolor=BLUE, markeredgecolor=WHITE, label="Target", markersize=4.5, linewidth=1.0),
            Line2D([0], [0], marker="D", color=PINK_DARK, markerfacecolor=PINK, markeredgecolor=WHITE, label="Selectivity", markersize=4.2, linewidth=1.0),
            Line2D([0], [0], marker="o", color=BLUE_DARK, markerfacecolor=WHITE, markeredgecolor=BLUE_DARK, label="Target - syntax", markersize=4.0, linewidth=0.8, linestyle=(0, (2.0, 1.2))),
            Line2D([0], [0], marker="D", color=PINK_DARK, markerfacecolor=WHITE, markeredgecolor=PINK_DARK, label="Target - discourse", markersize=3.8, linewidth=0.8, linestyle=(0, (2.0, 1.2))),
        ],
        loc="lower center",
        bbox_to_anchor=(0.57, 0.995),
        ncol=2,
        frameon=False,
        fontsize=5.8,
        handlelength=1.35,
        handletextpad=0.28,
        columnspacing=0.55,
        labelspacing=0.25,
        borderpad=0.0,
    )

    semantic_rows = semantics.set_index("model")
    for ypos, model in zip(y, models):
        row = semantic_rows.loc[model]
        semantic.plot(
            [row["ci_low"], row["ci_high"]],
            [ypos, ypos],
            color=BLUE_DARK,
            linewidth=1.05,
            solid_capstyle="round",
            zorder=3,
        )
        semantic.scatter(
            row["mean"],
            ypos,
            s=23,
            marker="o",
            facecolor=WHITE,
            edgecolor=BLUE_DARK,
            linewidth=0.9,
            zorder=4,
        )
    semantic.set_yticks(y, [""] * len(models))
    semantic.set_xlim(0.59, 0.79)
    semantic.set_xticks([0.60, 0.70, 0.80])
    semantic.set_ylim(-0.56, 2.56)
    semantic.set_xlabel("MiniLM cosine (focused)")
    semantic.tick_params(axis="y", length=0)
    panel_title(semantic, "B", "Semantic guardrail", y=1.43)
    clean_axis(semantic)

    figure.subplots_adjust(left=0.16, right=0.985, top=0.67, bottom=0.21)
    figure.savefig(OUT / "representation_realization_audit.pdf", bbox_inches="tight", pad_inches=0.025)
    figure.savefig(OUT / "representation_realization_audit.png", dpi=260, bbox_inches="tight", pad_inches=0.025)
    plt.close(figure)


def smooth_density(values: np.ndarray, lower: float, upper: float) -> tuple[np.ndarray, np.ndarray]:
    grid = np.linspace(lower, upper, 240)
    density = gaussian_kde(values, bw_method=0.36)(grid)
    return grid, density


def plot_appendix(seed: pd.DataFrame, group: pd.DataFrame, pairwise: pd.DataFrame, prompt_estimands: pd.DataFrame) -> None:
    means = prompt_means(prompt_estimands)
    target = means[means.estimand == "target_projection"].set_index("prompt_id")["value"]
    selective = means[means.estimand == "component_selectivity"].set_index("prompt_id")["value"]
    joined = pd.concat([target.rename("target"), selective.rename("selectivity")], axis=1).dropna()
    figure, axes = plt.subplots(1, 3, figsize=(7.25, 2.42), gridspec_kw={"wspace": 0.34})

    scatter = axes[0]
    scatter.axhline(0, color=GRID, linewidth=0.7)
    scatter.axvline(0, color=GRID, linewidth=0.7)
    scatter.scatter(joined.target, joined.selectivity, s=9, color=BLUE, alpha=0.34, edgecolors="none", rasterized=True)
    scatter.scatter(joined.target.mean(), joined.selectivity.mean(), s=43, marker="D", color=PINK, edgecolor=WHITE, linewidth=0.7, zorder=4)
    scatter.set_xlabel("Target projection")
    scatter.set_ylabel("Component selectivity")
    scatter.set_xlim(-4.8, 4.8)
    scatter.set_ylim(-5.0, 5.0)
    scatter.set_box_aspect(1)
    panel_title(scatter, "A", "Prompt geometry")
    scatter.grid(color=GRID, linewidth=0.45, alpha=0.75)
    scatter.spines["top"].set_visible(False)
    scatter.spines["right"].set_visible(False)

    density = axes[1]
    rng = np.random.default_rng(20260902)
    for values, color, light, label, offset in [(joined.target.to_numpy(), BLUE, BLUE_LIGHT, "Target", 0.0), (joined.selectivity.to_numpy(), PINK, PINK_LIGHT, "Selectivity", 1.0)]:
        grid, curve = smooth_density(values, -5.2, 5.2)
        curve = curve / curve.max() * 0.34
        density.fill_between(grid, offset, offset + curve, color=light, linewidth=0)
        density.plot(grid, offset + curve, color=color, linewidth=1.1)
        jitter = rng.uniform(-0.045, 0.045, len(values))
        order = np.argsort(values)
        density.scatter(values[order], offset - 0.10 + jitter, s=5, color=color, alpha=0.20, edgecolors="none", rasterized=True)
        density.scatter(values.mean(), offset, s=34, marker="D", color=color, edgecolor=WHITE, linewidth=0.65, zorder=4)
    density.axvline(0, color=INK, linewidth=0.7)
    density.set_yticks([0, 1], ["Target", "Selectivity"])
    density.set_xlim(-5.2, 5.2)
    density.set_ylim(-0.24, 1.47)
    density.set_xlabel("Prompt-mean paired effect")
    density.set_box_aspect(1)
    density.tick_params(axis="y", length=0)
    panel_title(density, "B", "Smoothed distributions")
    clean_axis(density)

    forest = axes[2]
    target_row = seed[seed.estimand == "target_projection"].iloc[0]
    selectivity_row = seed[seed.estimand == "component_selectivity"].iloc[0]
    syntax_row = pairwise[pairwise.comparison_group == "syntactic"].iloc[0]
    discourse_row = pairwise[pairwise.comparison_group == "discourse"].iloc[0]
    rows = [("Target", target_row, BLUE, "o"), ("Selectivity", selectivity_row, PINK, "D"), ("Target − syntax", syntax_row, BLUE_DARK, "o"), ("Target − discourse", discourse_row, PINK_DARK, "D")]
    y = np.arange(4)[::-1]
    forest.axvline(0, color=INK, linewidth=0.8)
    for ypos, (_, row, color, marker) in zip(y, rows):
        forest.plot([row.ci_low, row.ci_high], [ypos, ypos], color=color, linewidth=1.25, solid_capstyle="round")
        forest.scatter(row["mean"], ypos, s=28, marker=marker, color=color, edgecolor=WHITE, linewidth=0.6, zorder=3)
    forest.set_yticks(y, [row[0] for row in rows])
    forest.set_xlim(-0.17, 0.55)
    forest.set_xticks([0.0, 0.2, 0.4])
    forest.set_xlabel("Mean [95% CI]")
    forest.set_box_aspect(1)
    forest.tick_params(axis="y", length=0, pad=2)
    panel_title(forest, "C", "Selective contrasts")
    clean_axis(forest)

    figure.subplots_adjust(left=0.075, right=0.988, top=0.80, bottom=0.19)
    figure.savefig(OUT / "appendix_generation_robustness.pdf", bbox_inches="tight", pad_inches=0.025)
    figure.savefig(OUT.parent / "build" / "previews" / "appendix_generation_robustness.png", dpi=260, bbox_inches="tight", pad_inches=0.025)
    plt.close(figure)


def main() -> None:
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    effects, semantics = load_crossmodel_summary()
    # The main-text figure is now scripts/plot_fig_generation.py.
    seed, group, pairwise, prompt_estimands = load_llama_summary()
    plot_appendix(seed, group, pairwise, prompt_estimands)


if __name__ == "__main__":
    main()
