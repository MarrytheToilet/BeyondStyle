#!/usr/bin/env python3
"""Figure: the surface signature as an effect wheel and two stability radars.

Panel A is a polar "effect wheel": one radial bar per measurement, arranged in
three arcs (lexical, syntactic, discourse).  Bars grow outward for machineward
positive shifts and inward for negative ones from a zero ring; saturated bars
pass BH-FDR, tinted bars do not.  The hub lists each block's Mahalanobis shift.
Panels B and C are radars over the nine held-out conditions (four generators,
four domains, and the untouched prefix-only interface): B the cosine between
the full direction and the held-out refit, C the share of held-out pairs that
project machineward.  All values come from the frozen feature statistics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_figstyle import (  # noqa: E402
    FAINT, FULL_WIDTH, GRID, GROUP_COLOR, GROUP_LABEL, GROUP_MARKER, GROUP_TINT,
    INK, INK_SOFT, MUTED, RULE, SURFACE, group_handle, save, use_paper_style,
)

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "outputs/en/05_linguistic/features"
TRANSFER = ROOT / "outputs/en/05_linguistic/prefix_only_control/transfer"
OUT = ROOT / "ACL27/figures/surface_signature"
GROUPS = ("lexical", "syntactic", "discourse")

LABELS = {
    "function_word_rate": "function words", "pronoun_rate": "pronouns",
    "contraction_rate": "contractions", "mattr_50": "MATTR-50",
    "content_zipf_mean": "content Zipf", "repeated_trigram_rate": "rep. trigrams",
    "abstract_lexicon_rate": "abstract", "evaluative_lexicon_rate": "evaluative",
    "emphasis_lexicon_rate": "emphasis", "institutional_lexicon_rate": "institutional",
    "dependency_depth_mean": "dep. depth", "dependency_length_mean": "dep. length",
    "clause_density": "clauses", "coordination_density": "coordination",
    "subordination_density": "subordination", "passive_clause_rate": "passives",
    "nominalization_rate": "nominalization", "sentence_length_cv": "sent.-length CV",
    "fragment_rate": "fragments", "apposition_density": "apposition",
    "parenthetical_density": "parentheticals", "adjacent_pos_parallelism": "POS parallel.",
    "causal_connective_rate": "causal conn.", "contrast_connective_rate": "contrast conn.",
    "concession_connective_rate": "concessives", "purpose_connective_rate": "purpose conn.",
    "continuity_connective_rate": "continuity", "elaboration_connective_rate": "elaboration",
    "explicit_relation_rate": "explicit rel.", "transition_opening_rate": "transitions",
    "enumeration_marker_rate": "enumeration", "summary_marker_rate": "summary",
    "entity_persistence": "entity persist.", "subject_continuity": "subj. continuity",
    "adjacent_content_overlap": "adj. overlap",
}
# Radar axes, clockwise from the top: prefix-only, four generators, four domains.
CONDITIONS = [
    ("interface", "prefix_only", "prefix-only"),
    ("generator", "MiniMax-M2.5", "MiniMax"), ("generator", "deepseek-v4-flash", "DeepSeek"),
    ("generator", "glm-5.2", "GLM"), ("generator", "qwen3.6-flash", "Qwen3.6"),
    ("domain", "literary_narrative", "literary"), ("domain", "journalistic", "news"),
    ("domain", "instructional", "instruct."), ("domain", "encyclopedic", "encycl."),
]
D_MIN, D_MAX = -1.35, 0.80
R_IN, R_OUT = 0.42, 1.00


def radius(d: float) -> float:
    return R_IN + (d - D_MIN) / (D_MAX - D_MIN) * (R_OUT - R_IN)


def effect_wheel(axis: plt.Axes, frame: pd.DataFrame, groups: dict) -> None:
    gap = 4.0
    slots = len(frame) + gap * len(GROUPS)
    step = 2 * np.pi / slots
    axis.set_theta_zero_location("N")
    axis.set_theta_direction(-1)
    r_zero = radius(0.0)
    theta_fine = np.linspace(0, 2 * np.pi, 400)
    for level in (-1.0, -0.5, 0.5):
        axis.plot(theta_fine, np.full_like(theta_fine, radius(level)), color=GRID, linewidth=0.5, zorder=1)
    axis.plot(theta_fine, np.full_like(theta_fine, r_zero), color=INK, linewidth=0.7, zorder=3)

    position = gap / 2
    for group in GROUPS:
        block = frame[frame["group"].eq(group)].sort_values("paired_d", ascending=False)
        for _, row in block.iterrows():
            theta = position * step
            d = float(row["paired_d"])
            resolved = float(row["fdr_q"]) < 0.05
            color = GROUP_COLOR[group] if resolved else GROUP_TINT[group]
            lo, hi = sorted((r_zero, radius(d)))
            axis.bar(theta, hi - lo, width=step * 0.78, bottom=lo, color=color,
                     edgecolor="none", zorder=4)
            deg = np.degrees(theta)
            flip = 90 < deg < 270
            axis.text(theta, R_OUT + 0.05, LABELS[row["feature"]], rotation=(-deg + 90) + (180 if flip else 0),
                      rotation_mode="anchor", ha="right" if flip else "left", va="center",
                      fontsize=4.7, color=INK_SOFT if resolved else MUTED, zorder=6)
            position += 1
        position += gap

    # Scale labels sit in the gap at the top of the wheel.
    for level, text in ((-1.0, "−1"), (0.0, "0"), (0.5, ".5")):
        axis.text(0.0, radius(level), text, ha="center", va="center", fontsize=4.5, color=MUTED,
                  bbox=dict(boxstyle="round,pad=0.10", facecolor=SURFACE, edgecolor="none"), zorder=7)
    # Hub: the Mahalanobis shift of each block, in block colour.
    axis.annotate("‖Δ‖", xy=(0.5, 0.5), xycoords="axes fraction", xytext=(0, 8.4), textcoords="offset points",
                  ha="center", va="center", fontsize=4.8, color=MUTED)
    for k, group in enumerate(GROUPS):
        shift = groups[group]["mahalanobis_mean_shift"]
        axis.annotate(f"{shift:.2f}", xy=(0.5, 0.5), xycoords="axes fraction", xytext=(0, 2.6 - 5.6 * k),
                      textcoords="offset points", ha="center", va="center", fontsize=4.8,
                      color=GROUP_COLOR[group], fontweight="bold")
    axis.set_rlim(0, 1.46)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.spines["polar"].set_visible(False)
    axis.grid(False)


def radar(axis: plt.Axes, values: dict[str, list[float]], *, rings: tuple[float, ...],
          rmax: float, tick_labels: bool) -> None:
    n = len(CONDITIONS)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    axis.set_theta_zero_location("N")
    axis.set_theta_direction(-1)
    for t in theta:
        axis.plot([t, t], [0, rmax], color=GRID, linewidth=0.5, zorder=1)
    fine = np.linspace(0, 2 * np.pi, 300)
    for level in rings:
        axis.plot(fine, np.full_like(fine, level), color=GRID if level < rmax else RULE,
                  linewidth=0.5, zorder=1)
    for group in GROUPS:
        vals = np.array(values[group], dtype=float)
        color = GROUP_COLOR[group]
        mask = ~np.isnan(vals)
        if mask.all():
            closed_t = np.append(theta, theta[0])
            closed_v = np.append(vals, vals[0])
            axis.fill(closed_t, closed_v, color=GROUP_TINT[group], alpha=0.45, zorder=2)
            axis.plot(closed_t, closed_v, color=color, linewidth=1.1, zorder=3)
        else:
            axis.plot(theta[mask], vals[mask], color=color, linewidth=1.1, zorder=3)
        axis.scatter(theta[mask], vals[mask], s=11, marker=GROUP_MARKER[group], color=color,
                     edgecolor=SURFACE, linewidth=0.5, zorder=4)
    for t, (_, _, label) in zip(theta, CONDITIONS):
        deg = np.degrees(t)
        ha = "center" if deg in (0.0, 180.0) else ("left" if deg < 180 else "right")
        axis.text(t, rmax * 1.12, label, ha=ha, va="center", fontsize=5.0, color=INK_SOFT, zorder=6)
    if tick_labels:
        for level in rings[:-1]:
            axis.text(np.pi, level, f"{level:g}", ha="center", va="center", fontsize=4.6, color=MUTED,
                      bbox=dict(boxstyle="round,pad=0.12", facecolor=SURFACE, edgecolor="none"), zorder=7)
    axis.set_rlim(0, rmax * 1.36)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.spines["polar"].set_visible(False)
    axis.grid(False)


def load_radar_values(holdout: pd.DataFrame, transfer: pd.DataFrame, column: str, control_column: str) -> dict:
    values = {}
    for group in GROUPS:
        row_values = []
        for etype, env, _ in CONDITIONS:
            if etype == "interface":
                hit = transfer[transfer["concept"].eq(group)]
                row_values.append(float(hit[control_column].iloc[0]) if len(hit) else np.nan)
            else:
                row = holdout[holdout["group"].eq(group) & holdout["environment_type"].eq(etype)
                              & holdout["heldout_environment"].eq(env)]
                row_values.append(float(row[column].iloc[0]))
        values[group] = row_values
    return values


def draw() -> None:
    use_paper_style()
    frame = pd.read_csv(FEATURES / "feature_statistics.csv")
    groups = json.loads((FEATURES / "group_statistics.json").read_text())
    holdout = pd.read_csv(FEATURES / "environment_holdout_statistics.csv")
    transfer = pd.read_csv(TRANSFER / "surface_signature_transfer.csv")

    figure = plt.figure(figsize=(FULL_WIDTH, 1.96))
    grid = figure.add_gridspec(1, 3, width_ratios=[1.36, 1.0, 1.0], wspace=0.26,
                               left=0.005, right=0.995, top=0.90, bottom=0.07)
    wheel = figure.add_subplot(grid[0, 0], projection="polar")
    effect_wheel(wheel, frame, groups)
    cos = figure.add_subplot(grid[0, 1], projection="polar")
    radar(cos, load_radar_values(holdout, transfer, "heldout_direction_cosine", "plan_control_shift_cosine"),
          rings=(0.5, 1.0), rmax=1.0, tick_labels=True)
    rate = figure.add_subplot(grid[0, 2], projection="polar")
    radar(rate, load_radar_values(holdout, transfer, "test_positive_pair_rate", "control_projection_positive_rate"),
          rings=(0.5, 1.0), rmax=1.0, tick_labels=True)

    titles = (("A", "Paired effect $d$ of 35 measurements"), ("B", "Held-out direction cosine"),
              ("C", "Held-out machineward pair rate"))
    for axis, (letter, title) in zip((wheel, cos, rate), titles):
        box = axis.get_position()
        figure.text(box.x0 + 0.01, 0.955, letter, fontsize=8.6, fontweight="bold", color=INK, va="center")
        figure.text(box.x0 + 0.01 + 0.022, 0.955, title, fontsize=7.4, color=INK_SOFT, va="center")
    figure.legend(handles=[group_handle(g) for g in GROUPS], loc="lower center", bbox_to_anchor=(0.69, -0.01),
                  ncol=3, frameon=False, fontsize=6.0, handlelength=1.0, columnspacing=0.8, handletextpad=0.35)
    save(figure, OUT)


if __name__ == "__main__":
    draw()
