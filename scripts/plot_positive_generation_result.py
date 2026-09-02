#!/usr/bin/env python3
"""Render the held-out selective lexical-control figures from frozen CSVs."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs/en/05_linguistic/causal_pullback/Llama-3.1-8B-Instruct/mean_pool/strength2_seed_robustness_dual5090"
OUT = ROOT / "ACL27/figures"

INK = "#243143"
MUTED = "#667386"
GRID = "#DCE7EF"
BLUE = "#55A8E2"
BLUE_DARK = "#287CB8"
BLUE_LIGHT = "#D8EFFB"
PINK = "#F28EB1"
PINK_DARK = "#C85D87"
PINK_LIGHT = "#FCE1EB"
GRAY = "#A8B3BF"
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


def panel_title(axis: plt.Axes, letter: str, title: str) -> None:
    axis.text(0.0, 1.13, letter, transform=axis.transAxes, ha="left", va="bottom", fontsize=10.4, fontweight="bold", color=INK)
    axis.text(0.09, 1.13, title, transform=axis.transAxes, ha="left", va="bottom", fontsize=8.8, color=INK)


def load_summary() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed = pd.read_csv(DATA / "seed_robustness_summary.csv")
    group = pd.read_csv(DATA / "group_response_summary.csv")
    pairwise = pd.read_csv(DATA / "pairwise_selectivity_summary.csv")
    prompt_estimands = pd.read_csv(DATA / "prompt_seed_estimands.csv")
    return seed, group, pairwise, prompt_estimands


def prompt_means(prompt_estimands: pd.DataFrame) -> pd.DataFrame:
    return prompt_estimands.groupby(["prompt_id", "prompt_index", "estimand"], as_index=False)["value"].mean()


def plot_main(seed: pd.DataFrame, group: pd.DataFrame, pairwise: pd.DataFrame) -> None:
    figure = plt.figure(figsize=(5.05, 2.46))
    grid = figure.add_gridspec(1, 2, width_ratios=[1.16, 0.84], wspace=0.30)
    forest = figure.add_subplot(grid[0, 0])
    seeds = figure.add_subplot(grid[0, 1])

    lexical = group[group.response_group == "lexical"].iloc[0]
    syntactic = group[group.response_group == "syntactic"].iloc[0]
    discourse = group[group.response_group == "discourse"].iloc[0]
    selectivity = seed[seed.estimand == "component_selectivity"].iloc[0]
    rows = [
        ("Target: lexical", lexical, BLUE, "o"),
        ("Off-target: syntax", syntactic, GRAY, "s"),
        ("Off-target: discourse", discourse, GRAY, "s"),
        ("Component selectivity", selectivity, PINK, "D"),
    ]
    y = np.arange(len(rows))[::-1]
    forest.axvspan(0.0, 0.54, color="#F3FAFE", zorder=0)
    forest.axvline(0, color=INK, linewidth=0.8, zorder=1)
    for ypos, (_, row, color, marker) in zip(y, rows):
        forest.plot([row.ci_low, row.ci_high], [ypos, ypos], color=color, linewidth=1.35, solid_capstyle="round", zorder=3)
        forest.scatter(row["mean"], ypos, s=31, marker=marker, color=color, edgecolor=WHITE, linewidth=0.65, zorder=4)
        forest.text(0.53, ypos, f"{row['mean']:.3f}", ha="right", va="center", fontsize=7.2, color=INK)
    forest.set_yticks(y, [row[0] for row in rows])
    forest.set_xlim(-0.23, 0.55)
    forest.set_xticks([-0.2, 0.0, 0.2, 0.4])
    forest.set_xlabel("Paired +10% minus −10% response")
    forest.tick_params(axis="y", length=0, pad=3)
    panel_title(forest, "A", "Held-out component effects")
    clean_axis(forest)

    target_means = json.loads(seed[seed.estimand == "target_projection"].iloc[0].seed_means)
    selective_means = json.loads(selectivity.seed_means)
    seed_ids = np.array([0, 1, 2])
    target_values = np.array([target_means[str(index)] for index in seed_ids])
    selective_values = np.array([selective_means[str(index)] for index in seed_ids])
    for index in seed_ids:
        seeds.plot([0, 1], [target_values[index], selective_values[index]], color="#B8C6D2", linewidth=0.75, zorder=1)
    seeds.scatter(np.zeros(3), target_values, s=37, color=BLUE, edgecolor=WHITE, linewidth=0.7, zorder=3)
    seeds.scatter(np.ones(3), selective_values, s=37, marker="D", color=PINK, edgecolor=WHITE, linewidth=0.7, zorder=3)
    seeds.errorbar([0], [seed[seed.estimand == "target_projection"].iloc[0]["mean"]], yerr=[[seed[seed.estimand == "target_projection"].iloc[0]["mean"] - seed[seed.estimand == "target_projection"].iloc[0].ci_low], [seed[seed.estimand == "target_projection"].iloc[0].ci_high - seed[seed.estimand == "target_projection"].iloc[0]["mean"]]], fmt="o", color=BLUE_DARK, markersize=6.1, linewidth=1.2, capsize=2.2, zorder=4)
    seeds.errorbar([1], [selectivity["mean"]], yerr=[[selectivity["mean"] - selectivity.ci_low], [selectivity.ci_high - selectivity["mean"]]], fmt="D", color=PINK_DARK, markersize=5.5, linewidth=1.2, capsize=2.2, zorder=4)
    seeds.axhline(0, color=INK, linewidth=0.8, zorder=1)
    seeds.set_xlim(-0.42, 1.42)
    seeds.set_ylim(-0.08, 0.57)
    seeds.set_xticks([0, 1], ["Target", "Selectivity"])
    seeds.set_ylabel("Prompt-mean effect")
    seeds.grid(axis="y", color=GRID, linewidth=0.55, zorder=0)
    seeds.spines["top"].set_visible(False)
    seeds.spines["right"].set_visible(False)
    seeds.tick_params(width=0.6, length=2.5, color="#8EA0B2")
    panel_title(seeds, "B", "Three-seed replication")
    seeds.legend(handles=[Line2D([0], [0], marker="o", color="none", markerfacecolor="#BCC7D1", markeredgecolor=WHITE, label="seed mean", markersize=5.0), Line2D([0], [0], marker="o", color=INK, markerfacecolor=WHITE, label="mean + 95% CI", markersize=5.0)], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=6.0, handletextpad=0.22, columnspacing=0.45, borderpad=0.0)

    figure.subplots_adjust(left=0.205, right=0.985, top=0.83, bottom=0.205)
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
    figure.savefig(OUT / "appendix_generation_robustness.png", dpi=260, bbox_inches="tight", pad_inches=0.025)
    plt.close(figure)


def main() -> None:
    setup_style()
    OUT.mkdir(parents=True, exist_ok=True)
    seed, group, pairwise, prompt_estimands = load_summary()
    plot_main(seed, group, pairwise)
    plot_appendix(seed, group, pairwise, prompt_estimands)


if __name__ == "__main__":
    main()
