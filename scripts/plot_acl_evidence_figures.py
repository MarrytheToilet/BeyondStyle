#!/usr/bin/env python3
"""Build the appendix surface-transfer and pullback-robustness figures.

Generation robustness is drawn by plot_positive_generation_result.py. This
script never overwrites that figure with the retired generation layout.
"""

from __future__ import annotations

import argparse
import json
import os
from functools import lru_cache
from pathlib import Path
import re

os.environ.setdefault("MPLCONFIGDIR", "/tmp/beyondstyle-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Polygon
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
FIGURE_DIR = ROOT / "ACL27" / "figures"
PREVIEW_DIR = ROOT / "ACL27" / "build" / "previews"
FEATURE_ROOT = ROOT / "outputs" / "en" / "05_linguistic" / "features"
PREFIX_ROOT = (
    ROOT / "outputs" / "en" / "05_linguistic" / "prefix_only_control"
)
PULLBACK_ROOT = (
    ROOT / "outputs" / "en" / "05_linguistic" / "causal_pullback"
)
PAIRS = (
    ROOT
    / "data"
    / "paired"
    / "en"
    / "proposition_aligned"
    / "pairs.jsonl"
)

# GAP-inspired, print-safe scientific palette. The saturated roots carry the
# observations; pale variants are reserved for uncertainty and regions.
INK = "#353640"
MUTED = "#626571"
GRID = "#E5EBF1"
BG = "#FFFFFF"
NAVY = "#4F8CC9"
NAVY_DARK = "#356FA8"
NAVY_LIGHT = "#E8F1FA"
TEAL = "#E8669A"
TEAL_DARK = "#BE4777"
TEAL_LIGHT = "#FBEAF1"
OCHRE = "#A8ABB8"
OCHRE_DARK = "#737685"
OCHRE_LIGHT = "#F0F1F4"
GRAY = "#B8C1CA"
GRAY_DARK = "#707986"
GRAY_LIGHT = "#EEF2F5"
PALE_BAND = "#F5F8FB"

# One visual-weight system across the paper. Scatter sizes are areas in pt².
MARK_RAW = 14.0
MARK_DETAIL = 24.0
MARK_SUMMARY = 52.0
MARK_HIGHLIGHT = 64.0
LINE_HAIR = 0.65
LINE_STANDARD = 1.05
LINE_SUMMARY = 1.45

MODELS = ("Llama-3.1-8B-Instruct", "Qwen3-4B")
MODEL_LABEL = {
    "Llama-3.1-8B-Instruct": "Llama",
    "Qwen3-4B": "Qwen",
    "gemma-4-E2B-it": "Gemma",
}
MODEL_MARKER = {
    "Llama-3.1-8B-Instruct": "o",
    "Qwen3-4B": "s",
    "gemma-4-E2B-it": "^",
}
GENERATOR_MARKER = {
    "MiniMax-M2.5": "o",
    "deepseek-v4-flash": "s",
    "glm-5.2": "^",
    "qwen3.6-flash": "D",
}
CONCEPTS = ("lexical", "syntactic")
GROUP_STYLE = {
    "lexical": {
        "label": "Lexical",
        "color": NAVY,
        "dark": NAVY_DARK,
        "light": NAVY_LIGHT,
        "marker": "o",
    },
    "syntactic": {
        "label": "Syntactic",
        "color": TEAL,
        "dark": TEAL_DARK,
        "light": TEAL_LIGHT,
        "marker": "D",
    },
    "discourse": {
        "label": "Discourse",
        "color": OCHRE,
        "dark": OCHRE_DARK,
        "light": OCHRE_LIGHT,
        "marker": "s",
    },
}
ENVIRONMENT_LABEL = {
    "MiniMax-M2.5": "MiniMax",
    "deepseek-v4-flash": "DeepSeek",
    "glm-5.2": "GLM",
    "qwen3.6-flash": "Qwen",
    "encyclopedic": "Encycl.",
    "instructional": "Instr.",
    "journalistic": "News",
    "literary_narrative": "Narrative",
}
FEATURE_LABEL = {
    "function_word_rate": "function words",
    "pronoun_rate": "pronouns",
    "content_zipf_mean": "content-word frequency",
    "mattr_50": "lexical diversity",
    "dependency_length_mean": "dependency length",
    "sentence_length_cv": "sentence variation",
    "dependency_depth_mean": "dependency depth",
    "coordination_density": "coordination",
    "nominalization_rate": "nominalization",
    "transition_opening_rate": "transition openings",
}
CUES = (
    ("however", "however"),
    ("despite", "despite"),
    ("crucial", "crucial"),
    ("stated that", "stated that"),
    ("noted that", "noted that"),
)


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [
            "Arial",
            "Helvetica",
            "DejaVu Sans",
            "Liberation Sans",
        ],
        "font.size": 12.8,
        "axes.facecolor": BG,
        "figure.facecolor": BG,
        "axes.edgecolor": "#50525C",
        "axes.labelcolor": INK,
        "axes.labelsize": 12.0,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
        "legend.fontsize": 10.0,
        "legend.frameon": False,
        "grid.color": GRID,
        "grid.linewidth": 0.65,
        "grid.alpha": 0.90,
        "lines.solid_capstyle": "round",
        "lines.dash_capstyle": "round",
        "lines.linewidth": LINE_STANDARD,
        "lines.markersize": 4.0,
        "patch.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def save_figure(figure: plt.Figure, stem: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        FIGURE_DIR / f"{stem}.pdf",
        bbox_inches="tight",
        facecolor=BG,
    )
    figure.savefig(
        PREVIEW_DIR / f"{stem}.png",
        dpi=260,
        bbox_inches="tight",
        facecolor=BG,
    )
    plt.close(figure)


def panel_title(
    axis: plt.Axes,
    letter: str,
    heading: str,
    note: str = "",
    *,
    y: float = 1.08,
    note_gap: float = 0.085,
    letter_size: float = 12.4,
    heading_size: float = 12.0,
    heading_offset: float = 21.0,
) -> None:
    axis.text(
        0.0,
        y,
        letter,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=letter_size,
        fontweight="bold",
        color=INK,
    )
    axis.annotate(
        heading,
        xy=(0.0, y),
        xycoords=axis.transAxes,
        xytext=(heading_offset, 0),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=heading_size,
        fontweight="normal",
        color=INK,
    )
    # Statistical decoding belongs in the caption, not as a second tier of
    # tiny prose beneath every panel title.


def finish_axis(axis: plt.Axes, grid_axis: str = "both") -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    if grid_axis != "none":
        axis.grid(axis=grid_axis, zorder=0)
        axis.set_axisbelow(True)


def covariance_ellipse(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    edge: str,
    face: str = "none",
    level: float,
    linewidth: float = 1.25,
    alpha: float = 0.9,
    zorder: int = 2,
) -> None:
    """Draw a covariance ellipse without adding a density encoding."""
    points = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 3:
        return
    covariance = np.cov(points, rowvar=False)
    values, vectors = np.linalg.eigh(covariance)
    order = values.argsort()[::-1]
    values = np.maximum(values[order], 0.0)
    vectors = vectors[:, order]
    scale = {0.5: np.sqrt(1.386294), 0.9: np.sqrt(4.605170)}[level]
    width, height = 2.0 * scale * np.sqrt(values)
    angle = np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0]))
    axis.add_patch(
        Ellipse(
            xy=points.mean(axis=0),
            width=width,
            height=height,
            angle=angle,
            facecolor=face,
            edgecolor=edge,
            linewidth=linewidth,
            alpha=alpha,
            zorder=zorder,
        )
    )


def direction_ablation_frame() -> pd.DataFrame:
    frames = []
    for model in MODELS:
        frame = pd.read_csv(
            PULLBACK_ROOT / model / "mean_pool"
            / "confirmatory_hidden_audit" / "layerwise_transport"
            / "direction_ablation_global.csv"
        )
        frame["model"] = model
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def draw_dominance_plane(axis: plt.Axes) -> None:
    data = direction_ablation_frame()
    families = (
        ("encoding", "Encode"),
        ("raw_mean_difference", "Raw"),
        ("nullspace_mean_difference", "Null"),
        ("projected_random_ensemble", "Random"),
    )
    model_offset = {MODELS[0]: -0.13, MODELS[1]: 0.13}
    for family_index, (family, _) in enumerate(families):
        for concept in CONCEPTS:
            style = GROUP_STYLE[concept]
            part = data[
                data["reference_family"].eq(family)
                & data["concept"].eq(concept)
            ]
            points = []
            for model in MODELS:
                row = part[part["model"].eq(model)].iloc[0]
                margin = (
                    float(row["learned_curve_statistic"])
                    - float(row["reference_curve_statistic"])
                )
                points.append((family_index + model_offset[model], margin))
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                color=style["light"],
                linewidth=1.25,
                zorder=1,
            )
            axis.scatter(
                [point[0] for point in points],
                [point[1] for point in points],
                s=42,
                marker=style["marker"],
                facecolor=style["color"],
                edgecolor=style["dark"],
                linewidth=0.85,
                zorder=3,
            )
    axis.axhline(0, color=INK, linewidth=0.9)
    axis.text(
        0.04, 0.96, "left Llama · right Qwen", transform=axis.transAxes,
        ha="left", va="top", fontsize=8.6,
        color=MUTED,
    )
    axis.set_xlim(-0.45, 3.45)
    axis.set_ylim(-0.015, 0.39)
    axis.set_xticks(range(len(families)), [label for _, label in families])
    axis.tick_params(axis="x", labelsize=8.6)
    axis.set_ylabel("Canonical − comparator AUC")
    panel_title(
        axis,
        "A",
        "Canonical margin",
        y=1.08,
        letter_size=10.6,
        heading_size=10.2,
        heading_offset=18.0,
    )
    axis.set_box_aspect(1)
    finish_axis(axis, "y")


@lru_cache(maxsize=1)
def crossfit_surface_scores() -> pd.DataFrame:
    """Reconstruct the frozen out-of-fold surface scores per paired item."""
    plan = pd.read_csv(FEATURE_ROOT / "paired_feature_differences.csv")
    control = pd.read_csv(
        PREFIX_ROOT / "features" / "paired_feature_differences.csv"
    )
    if not np.array_equal(plan["sample_id"], control["sample_id"]):
        raise ValueError("Plan and prefix-only feature rows are not aligned")
    dictionary = json.loads(
        (FEATURE_ROOT / "feature_dictionary.json").read_text(encoding="utf-8")
    )
    group_features = {
        group: [item["name"] for item in dictionary if item["group"] == group]
        for group in CONCEPTS
    }
    generators = plan["generator"].to_numpy()
    records = []
    for concept, features in group_features.items():
        plan_human = plan[[f"human__{name}" for name in features]].to_numpy(float)
        plan_machine = plan[[f"ai__{name}" for name in features]].to_numpy(float)
        control_human = control[
            [f"human__{name}" for name in features]
        ].to_numpy(float)
        control_machine = control[
            [f"ai__{name}" for name in features]
        ].to_numpy(float)
        for heldout in sorted(np.unique(generators)):
            train = generators != heldout
            test = generators == heldout
            scale = np.sqrt(
                (
                    plan_human[train].var(axis=0, ddof=1)
                    + plan_machine[train].var(axis=0, ddof=1)
                )
                / 2.0
            )
            scale[scale <= 1e-12] = 1.0
            shift = (
                plan_machine[train] - plan_human[train]
            ).mean(axis=0) / scale
            weights = shift / np.linalg.norm(shift)
            plan_scores = (
                (plan_machine[test] - plan_human[test]) / scale
            ) @ weights
            control_scores = (
                (control_machine[test] - control_human[test]) / scale
            ) @ weights
            test_indices = np.flatnonzero(test)
            for index, plan_score, control_score in zip(
                test_indices, plan_scores, control_scores
            ):
                records.append({
                    "sample_id": str(plan.iloc[index]["sample_id"]),
                    "generator": str(heldout),
                    "concept": concept,
                    "plan_score": float(plan_score),
                    "control_score": float(control_score),
                })
    return pd.DataFrame(records)


def draw_generator_transfer_map(
    axis: plt.Axes,
    concept: str,
    letter: str,
    *,
    show_y: bool,
) -> None:
    scores = crossfit_surface_scores()
    view = scores[scores["concept"].eq(concept)]
    generators = [
        "MiniMax-M2.5", "deepseek-v4-flash", "glm-5.2", "qwen3.6-flash",
    ]
    style = GROUP_STYLE[concept]
    for generator in generators:
        part = view[view["generator"].eq(generator)]
        marker = GENERATOR_MARKER[generator]
        axis.scatter(
            part["plan_score"],
            part["control_score"],
            s=MARK_RAW,
            marker="o",
            facecolor=style["color"],
            edgecolor="none",
            alpha=0.20,
            zorder=2,
        )
        plan_mean = float(part["plan_score"].mean())
        prefix_mean = float(part["control_score"].mean())
        axis.annotate(
            "",
            xy=(plan_mean, prefix_mean),
            xytext=(plan_mean, plan_mean),
            arrowprops={
                "arrowstyle": "-|>",
                "color": style["dark"],
                "linewidth": 0.8,
                "alpha": 0.72,
                "mutation_scale": 8,
            },
            zorder=3,
        )
        axis.scatter(
            plan_mean,
            prefix_mean,
            s=MARK_HIGHLIGHT,
            marker=marker,
            facecolor=BG,
            edgecolor=style["dark"],
            linewidth=0.95,
            zorder=4,
        )
    covariance_ellipse(
        axis,
        view["plan_score"].to_numpy(float),
        view["control_score"].to_numpy(float),
        edge=style["dark"],
        face=style["light"],
        level=0.5,
        linewidth=LINE_STANDARD,
        alpha=0.62,
        zorder=1,
    )
    values = np.r_[scores["plan_score"], scores["control_score"]]
    low, high = np.quantile(values, [0.004, 0.996])
    padding = 0.05 * (high - low)
    low, high = low - padding, high + padding
    axis.axline((0, 0), slope=1, color=GRAY, linewidth=1.0, linestyle="--")
    axis.axhline(0, color=INK, linewidth=0.8)
    axis.axvline(0, color=INK, linewidth=0.8)
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_xlabel("Plan projection")
    axis.tick_params(axis="y", labelleft=show_y)
    if show_y:
        axis.set_ylabel("Prefix-only projection")
    panel_title(
        axis,
        letter,
        f"{style['label']} pair transfer",
        y=1.10,
    )
    finish_axis(axis)


def draw_readout_gap_distribution(axis: plt.Axes) -> None:
    data = pd.read_csv(
        PREFIX_ROOT / "transfer" / "concept_mapping_transfer.csv"
    )
    row_positions = {
        ("Input", "lexical"): 3.0,
        ("Input", "syntactic"): 2.0,
        ("Anchor", "lexical"): 1.0,
        ("Anchor", "syntactic"): 0.0,
    }
    model_offsets = {
        "Llama-3.1-8B-Instruct": 0.13,
        "Qwen3-4B": 0.0,
        "gemma-4-E2B-it": -0.13,
    }
    rng = np.random.default_rng(20260901)
    for stage in ("Input", "Anchor"):
        stage_rows = []
        for model, model_data in data.groupby("model", sort=True):
            layer = (
                model_data["layer"].min()
                if stage == "Input"
                else model_data["layer"].max()
            )
            part = model_data[model_data["layer"].eq(layer)].copy()
            part["stage"] = stage
            stage_rows.append(part)
        stage_frame = pd.concat(stage_rows, ignore_index=True)
        for concept in CONCEPTS:
            style = GROUP_STYLE[concept]
            part = stage_frame[stage_frame["concept"].eq(concept)].copy()
            values = (
                part["control_ai_score_correlation"]
                - part["plan_ai_score_correlation"]
            ).to_numpy(float)
            position = row_positions[(stage, concept)]
            violin = axis.violinplot(
                [values],
                positions=[position],
                widths=0.58,
                showmeans=False,
                showmedians=False,
                showextrema=False,
                bw_method=0.25,
                vert=False,
            )
            for body in violin["bodies"]:
                body.set_facecolor(style["light"])
                body.set_edgecolor(style["dark"])
                body.set_linewidth(0.8)
                body.set_alpha(0.82)
            for model, model_part in part.groupby("model", sort=True):
                model_values = (
                    model_part["control_ai_score_correlation"]
                    - model_part["plan_ai_score_correlation"]
                ).to_numpy(float)
                y = np.full(len(model_values), position + model_offsets[model])
                y += rng.normal(0.0, 0.018, len(model_values))
                axis.scatter(
                    model_values,
                    y,
                    s=MARK_DETAIL,
                    marker=MODEL_MARKER[model],
                    facecolor=BG,
                    edgecolor=style["dark"],
                    linewidth=0.75,
                    zorder=3,
                )
            bootstrap = np.asarray([
                rng.choice(values, size=len(values), replace=True).mean()
                for _ in range(2000)
            ])
            low, high = np.quantile(bootstrap, [0.025, 0.975])
            mean = float(values.mean())
            axis.plot(
                [low, high],
                [position, position],
                color=style["dark"],
                linewidth=LINE_SUMMARY,
                zorder=4,
            )
            axis.scatter(
                mean,
                position,
                s=MARK_SUMMARY,
                marker="D",
                facecolor=style["color"],
                edgecolor=style["dark"],
                linewidth=0.8,
                zorder=5,
            )
    axis.axvline(0, color=INK, linewidth=0.9)
    axis.set_xlim(-0.13, 0.33)
    axis.set_ylim(-0.48, 3.48)
    axis.set_yticks(
        [3, 2, 1, 0],
        ["Input · Lex", "Input · Syn", "Anchor · Lex", "Anchor · Syn"],
    )
    axis.tick_params(axis="y", length=0, pad=3, labelsize=8.8)
    axis.set_xlabel("Prefix − plan readout r", fontsize=10.0)
    panel_title(axis, "C", "Readout displacement", y=1.10)
    finish_axis(axis, "x")


def draw_depth_delta_combined(axis: plt.Axes) -> None:
    data = pd.read_csv(
        PREFIX_ROOT / "transfer" / "source_axis_transfer.csv"
    )
    model_style = {
        "Llama-3.1-8B-Instruct": {
            "label": "Llama",
            "color": NAVY,
            "dark": NAVY_DARK,
            "light": NAVY_LIGHT,
            "marker": "o",
            "label_offset": -8,
        },
        "Qwen3-4B": {
            "label": "Qwen",
            "color": TEAL,
            "dark": TEAL_DARK,
            "light": TEAL_LIGHT,
            "marker": "s",
            "label_offset": 8,
        },
    }
    for model, style in model_style.items():
        view = data[data["model"].eq(model)].copy()
        view["delta"] = view["control_auc"] - view["plan_auc"]
        maximum = float(view["layer"].max())
        summary = view.groupby("layer")["delta"].agg(
            mean="mean", low="min", high="max"
        ).reset_index()
        x = summary["layer"].to_numpy(float) / maximum
        axis.fill_between(
            x,
            summary["low"].to_numpy(float),
            summary["high"].to_numpy(float),
            color=style["light"],
            alpha=0.55,
            linewidth=0,
            zorder=1,
        )
        axis.plot(
            x,
            summary["mean"].to_numpy(float),
            color=style["dark"],
            linewidth=LINE_SUMMARY,
            marker=style["marker"],
            markersize=3.2,
            markerfacecolor=BG,
            markeredgecolor=style["dark"],
            markeredgewidth=0.75,
            label=style["label"],
            zorder=3,
        )
        axis.annotate(
            f"{style['label']} {summary['mean'].iloc[-1]:+.03f}",
            (x[-1], float(summary["mean"].iloc[-1])),
            xytext=(4, style["label_offset"]),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8.2,
            color=style["dark"],
        )
    axis.axhline(0, color=INK, linewidth=0.9)
    axis.set_xlim(0.0, 1.09)
    axis.set_ylim(-0.075, 0.095)
    axis.set_xlabel("Relative layer depth")
    axis.set_ylabel("Prefix − plan AUC", fontsize=10.0, labelpad=2)
    panel_title(axis, "D", "Depthwise transfer", y=1.10)
    finish_axis(axis)


def build_surface_transfer() -> None:
    figure = plt.figure(figsize=(13.2, 3.42))
    outer = figure.add_gridspec(
        1,
        2,
        width_ratios=[2.10, 2.02],
        wspace=0.26,
    )
    transfer_group = outer[0].subgridspec(
        1, 2, width_ratios=[1.0, 1.0], wspace=0.24,
    )
    diagnostic_group = outer[1].subgridspec(
        1, 2, width_ratios=[0.94, 1.02], wspace=0.28,
    )
    draw_generator_transfer_map(
        figure.add_subplot(transfer_group[0]), "lexical", "A", show_y=True
    )
    draw_generator_transfer_map(
        figure.add_subplot(transfer_group[1]), "syntactic", "B", show_y=False
    )
    draw_readout_gap_distribution(figure.add_subplot(diagnostic_group[0]))
    draw_depth_delta_combined(figure.add_subplot(diagnostic_group[1]))
    figure.legend(
        handles=[
            Line2D(
                [0], [0], marker=GENERATOR_MARKER[g], linestyle="none",
                color=GRAY_DARK, markerfacecolor=BG,
                label=ENVIRONMENT_LABEL[g],
            )
            for g in (
                "MiniMax-M2.5", "deepseek-v4-flash", "glm-5.2",
                "qwen3.6-flash",
            )
        ],
        loc="upper center",
        bbox_to_anchor=(0.30, 0.995),
        ncol=4,
        fontsize=9.2,
        columnspacing=0.75,
        handletextpad=0.25,
    )
    figure.legend(
        handles=[
            Line2D(
                [0], [0], marker=MODEL_MARKER[m], linestyle="none",
                color=GRAY_DARK, markerfacecolor=BG,
                label=MODEL_LABEL[m],
            )
            for m in (
                "Llama-3.1-8B-Instruct", "Qwen3-4B", "gemma-4-E2B-it",
            )
        ],
        loc="upper center",
        bbox_to_anchor=(0.67, 0.995),
        ncol=3,
        fontsize=9.2,
        columnspacing=0.75,
        handletextpad=0.25,
    )
    figure.subplots_adjust(left=0.06, right=0.992, top=0.78, bottom=0.18)
    save_figure(figure, "appendix_surface_transfer")


def draw_sensitivity_plane(axis: plt.Axes, model: str, letter: str) -> None:
    data = pd.read_csv(
        PULLBACK_ROOT / "design_sensitivity" / "direction_sensitivity.csv"
    )
    view = data[data["model"].eq(model)]
    for concept in CONCEPTS:
        style = GROUP_STYLE[concept]
        for family, marker in (("null_k", "o"), ("fit_pairs", "s")):
            part = view[
                view["concept"].eq(concept)
                & view["sensitivity"].eq(family)
            ].sort_values("value")
            axis.plot(
                part["human_ai_pullback_cosine"],
                part["cosine_vs_canonical"],
                color=style["dark"],
                linewidth=0.8,
                linestyle="-" if family == "null_k" else "--",
                alpha=0.48,
                zorder=1,
            )
            axis.scatter(
                part["human_ai_pullback_cosine"],
                part["cosine_vs_canonical"],
                s=38,
                marker=marker,
                facecolor=style["color"] if family == "null_k" else BG,
                edgecolor=style["dark"],
                linewidth=0.9,
                zorder=3,
            )
            canonical_value = 10 if family == "null_k" else 32
            canonical = part[part["value"].eq(canonical_value)]
            if not canonical.empty:
                axis.scatter(
                    canonical["human_ai_pullback_cosine"],
                    canonical["cosine_vs_canonical"],
                    s=58,
                    marker=marker,
                    facecolor="none",
                    edgecolor=INK,
                    linewidth=0.7,
                    zorder=4,
                )
    axis.axhspan(0.95, 1.008, color=PALE_BAND, zorder=0)
    axis.axhline(0.95, color=GRAY, linewidth=0.85, linestyle="--")
    axis.text(
        0.5, 1.01, "● rank   □ fit   black rim canonical",
        transform=axis.transAxes, ha="center", va="bottom",
        fontsize=7.4, color=MUTED,
    )
    axis.set_xlim(0.79, 0.955)
    axis.set_ylim(0.89, 1.008)
    axis.set_xlabel("Human/machine cosine", fontsize=9.6)
    axis.set_ylabel("Cosine with canonical", fontsize=9.6)
    panel_title(
        axis,
        letter,
        f"{MODEL_LABEL[model]} sensitivity",
        y=1.08,
        letter_size=10.6,
        heading_size=10.2,
        heading_offset=18.0,
    )
    axis.set_box_aspect(1)
    finish_axis(axis)


def build_hidden_robustness() -> None:
    figure, axes = plt.subplots(
        1,
        3,
        figsize=(9.6, 3.55),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0], "wspace": 0.38},
    )
    draw_dominance_plane(axes[0])
    draw_sensitivity_plane(axes[1], "Llama-3.1-8B-Instruct", "B")
    draw_sensitivity_plane(axes[2], "Qwen3-4B", "C")
    figure.legend(
        handles=[
            Line2D(
                [0], [0], marker="o", linestyle="none",
                color=NAVY_DARK, markerfacecolor=NAVY,
                label="Lexical",
            ),
            Line2D(
                [0], [0], marker="o", linestyle="none",
                color=TEAL_DARK, markerfacecolor=TEAL,
                label="Syntactic",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        fontsize=9.6,
        columnspacing=1.25,
        handletextpad=0.35,
    )
    figure.subplots_adjust(left=0.075, right=0.985, top=0.76, bottom=0.18)
    save_figure(figure, "appendix_hidden_robustness")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--figures", nargs="+", choices=("surface", "hidden"),
        default=["surface", "hidden"],
    )
    args = parser.parse_args()
    setup_style()
    if "surface" in args.figures:
        build_surface_transfer()
    if "hidden" in args.figures:
        build_hidden_robustness()


if __name__ == "__main__":
    main()
