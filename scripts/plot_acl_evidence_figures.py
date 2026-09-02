#!/usr/bin/env python3
"""Build relationship-first evidence figures for the ACL manuscript.

The panels read only frozen outputs. They deliberately avoid workflow
diagrams, heatmaps, and rows of bar-like summaries. Each view instead joins
two or more estimands so that it reveals a relationship: prevalence versus
effect, effect versus support, alignment versus transfer, learned behavior
versus a joint null envelope, or variation across decoding seeds.
"""

from __future__ import annotations

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


def cue_statistics() -> pd.DataFrame:
    with PAIRS.open(encoding="utf-8") as handle:
        pairs = [json.loads(line) for line in handle]
    records = []
    for label, form in CUES:
        pattern = re.compile(
            r"(?<![A-Za-z])" + re.escape(form) + r"(?![A-Za-z])",
            re.IGNORECASE,
        )
        human = np.array([
            bool(pattern.search(row["human_continuation"]))
            for row in pairs
        ])
        machine = np.array([
            bool(pattern.search(row["ai_continuation"]))
            for row in pairs
        ])
        records.append({
            "form": label,
            "human_count": int(human.sum()),
            "machine_count": int(machine.sum()),
            "human_prevalence": 100.0 * float(human.mean()),
            "machine_prevalence": 100.0 * float(machine.mean()),
        })
    return pd.DataFrame(records)


def draw_cue_plane(axis: plt.Axes) -> None:
    data = cue_statistics().sort_values("machine_prevalence")
    axis.axline((0, 0), slope=1, color=GRAY, linewidth=1.0, linestyle="--")
    for _, row in data.iterrows():
        x = float(row["human_prevalence"])
        y = float(row["machine_prevalence"])
        axis.plot([x, x], [x, y], color=NAVY_LIGHT, linewidth=2.3, zorder=1)
        axis.scatter(
            x, x, s=28, facecolor=BG, edgecolor=NAVY,
            linewidth=1.1, zorder=2,
        )
        axis.scatter(
            x, y, s=58, facecolor=NAVY, edgecolor=NAVY_DARK,
            linewidth=1.0, zorder=3,
        )
        dx = 5 if x < 1.0 else 6
        dy = 1 if str(row["form"]) not in {"crucial", "despite"} else -2
        axis.annotate(
            f"“{row['form']}”  {int(row['machine_count'])}/500",
            (x, y), xytext=(dx, dy), textcoords="offset points",
            ha="left", va="center", fontsize=8.6, color=NAVY_DARK,
        )
    axis.text(
        14.8, 14.1, "equal prevalence", rotation=45,
        ha="right", va="bottom", fontsize=8.3, color=MUTED,
    )
    axis.set_xlim(-0.6, 18.3)
    axis.set_ylim(-0.6, 18.3)
    axis.set_xticks([0, 5, 10, 15])
    axis.set_yticks([0, 5, 10, 15])
    axis.set_xlabel("Human documents (%)")
    axis.set_ylabel("Machine documents (%)")
    panel_title(
        axis, "A", "Visible lexical cues",
        "vertical displacement is machine enrichment", y=1.12,
    )
    finish_axis(axis)


def feature_landscape_frame() -> pd.DataFrame:
    statistics = pd.read_csv(FEATURE_ROOT / "feature_statistics.csv")
    differences = pd.read_csv(FEATURE_ROOT / "paired_feature_differences.csv")
    statistics = statistics[statistics["group"].isin(GROUP_STYLE)].copy()
    supports = []
    agreements = []
    for _, row in statistics.iterrows():
        values = differences[str(row["feature"])].to_numpy(float)
        active = np.abs(values) > 1e-12
        supports.append(float(active.mean()))
        oriented = values[active] * np.sign(float(row["paired_d"]))
        agreements.append(float((oriented > 0).mean()))
    statistics["active_pair_rate"] = supports
    statistics["directional_agreement"] = agreements
    return statistics


def draw_feature_landscape(axis: plt.Axes, letter: str = "A") -> None:
    data = feature_landscape_frame()
    for group in ("discourse", "syntactic", "lexical"):
        style = GROUP_STYLE[group]
        view = data[data["group"].eq(group)]
        sizes = 18.0 + 52.0 * np.sqrt(
            view["active_pair_rate"].to_numpy()
        )
        axis.scatter(
            view["paired_d"], view["directional_agreement"], s=sizes,
            marker=style["marker"], facecolor=style["color"],
            edgecolor=style["dark"], linewidth=0.75, alpha=0.88,
            zorder=3, label=style["label"],
        )
    label_offsets = {
        "function_word_rate": (7, -1),
        "mattr_50": (-7, 7),
        "dependency_length_mean": (7, 7),
        "transition_opening_rate": (7, -10),
    }
    for feature, offset in label_offsets.items():
        row = data[data["feature"].eq(feature)].iloc[0]
        axis.annotate(
            FEATURE_LABEL[feature],
            (float(row["paired_d"]), float(row["directional_agreement"])),
            xytext=offset, textcoords="offset points",
            ha="right" if offset[0] < 0 else "left",
            va="bottom" if offset[1] >= 0 else "top", fontsize=7.9,
            color=GROUP_STYLE[str(row["group"])]["dark"],
            bbox={
                "facecolor": BG,
                "edgecolor": "none",
                "alpha": 0.78,
                "pad": 0.35,
            },
        )
    axis.axvline(0, color=INK, linewidth=0.95, zorder=1)
    axis.axhline(0.5, color=GRAY, linewidth=0.9, linestyle="--", zorder=1)
    axis.set_xlim(-1.34, 0.77)
    axis.set_ylim(0.49, 1.015)
    axis.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    axis.set_xlabel("Paired standardized shift")
    axis.set_ylabel("Directional agreement")
    axis.legend(
        loc="lower left", ncol=1, handletextpad=0.35,
        borderaxespad=0.2, fontsize=8.7, markerscale=0.74,
        labelspacing=0.20,
    )
    panel_title(
        axis, letter, "Feature landscape",
        "35 frozen features; labels mark the dominant shifts", y=1.12,
    )
    finish_axis(axis)


def draw_stability_orbit(axis: plt.Axes) -> None:
    holdout = pd.read_csv(FEATURE_ROOT / "environment_holdout_statistics.csv")
    generators = [
        "MiniMax-M2.5", "deepseek-v4-flash", "glm-5.2", "qwen3.6-flash",
    ]
    domains = [
        "encyclopedic", "instructional", "journalistic",
        "literary_narrative",
    ]
    angles = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    path_angles = np.r_[angles[:8], angles[0]]
    for group in ("discourse", "syntactic", "lexical"):
        style = GROUP_STYLE[group]
        values = []
        for env_type, envs in (("generator", generators), ("domain", domains)):
            part = holdout[
                holdout["environment_type"].eq(env_type)
                & holdout["group"].eq(group)
            ].set_index("heldout_environment")
            values.extend([
                float(part.loc[environment, "heldout_direction_cosine"])
                for environment in envs
            ])
        values_array = np.asarray(values)
        axis.plot(
            path_angles, np.r_[values_array, values_array[0]],
            color=style["dark"], linewidth=1.8,
            marker=style["marker"], markersize=4.2,
            markerfacecolor=style["color"], markeredgewidth=0.6,
            label=style["label"], zorder=3,
        )
    transfer = pd.read_csv(
        PREFIX_ROOT / "transfer" / "surface_signature_transfer.csv"
    ).set_index("concept")
    for group in CONCEPTS:
        style = GROUP_STYLE[group]
        axis.scatter(
            angles[8], float(transfer.loc[group, "plan_control_shift_cosine"]),
            s=62, marker="*", facecolor=style["color"],
            edgecolor=style["dark"], linewidth=0.8, zorder=5,
        )
    axis.set_theta_offset(np.pi / 2)
    axis.set_theta_direction(-1)
    axis.set_ylim(0.45, 1.02)
    axis.set_yticks([0.50, 0.75, 0.95], [".50", ".75", ".95"])
    axis.set_rlabel_position(197)
    axis.set_xticks(
        angles,
        [
            "MiniMax", "DeepSeek", "GLM", "Qwen", "Encycl.",
            "Instr.", "News", "Narrative", "Prefix★",
        ],
        fontsize=7.9,
    )
    axis.grid(color=GRID, linewidth=0.65)
    axis.spines["polar"].set_color("#B5BFC4")
    axis.spines["polar"].set_linewidth(1.0)
    axis.legend(
        loc="lower center", bbox_to_anchor=(0.5, -0.23), ncol=3,
        columnspacing=0.7, handlelength=1.2, handletextpad=0.3,
        fontsize=7.9,
    )
    panel_title(
        axis, "C", "Stability orbit",
        "four generators · four domains · prefix-only transfer",
        y=1.19, note_gap=0.075,
    )


def draw_interface_phase(axis: plt.Axes) -> None:
    data = pd.read_csv(
        PREFIX_ROOT / "transfer" / "concept_mapping_transfer.csv"
    )
    data = data[data["model"].isin(MODELS)].copy()
    for model in MODELS:
        model_data = data[data["model"].eq(model)]
        layers = sorted(model_data["layer"].unique())
        for concept in CONCEPTS:
            style = GROUP_STYLE[concept]
            for layer in layers:
                part = model_data[
                    model_data["concept"].eq(concept)
                    & model_data["layer"].eq(layer)
                ]
                late = layer == max(layers)
                axis.scatter(
                    part["plan_ai_score_correlation"],
                    part["control_ai_score_correlation"],
                    s=54 if late else 34, marker=MODEL_MARKER[model],
                    facecolor=style["color"] if late else BG,
                    edgecolor=style["dark"], linewidth=1.05,
                    alpha=0.88, zorder=3,
                )
    axis.axline((0, 0), slope=1, color=GRAY, linestyle="--", linewidth=1.0)
    proportion = float(
        (data["control_ai_score_correlation"]
         > data["plan_ai_score_correlation"]).mean()
    )
    axis.text(
        0.97, 0.05, f"{100 * proportion:.0f}% above diagonal",
        transform=axis.transAxes, ha="right", va="bottom",
        fontsize=8.4, color=MUTED,
    )
    axis.set_xlim(0.24, 0.71)
    axis.set_ylim(0.24, 0.71)
    axis.set_xlabel("Plan-conditioned readout r")
    axis.set_ylabel("Prefix-only readout r")
    legend = [
        Line2D([0], [0], marker="o", linestyle="none", color=INK,
               markerfacecolor=BG, label="Llama"),
        Line2D([0], [0], marker="s", linestyle="none", color=INK,
               markerfacecolor=BG, label="Qwen"),
        Line2D([0], [0], marker="o", linestyle="none", color=GRAY_DARK,
               markerfacecolor=BG, label="input"),
        Line2D([0], [0], marker="o", linestyle="none", color=GRAY_DARK,
               markerfacecolor=GRAY_DARK, label="anchor"),
    ]
    axis.legend(
        handles=legend, loc="upper left", ncol=2, fontsize=7.9,
        handletextpad=0.25, columnspacing=0.7, borderaxespad=0.2,
    )
    panel_title(
        axis, "D", "Interface-transfer field",
        "each point is one held-out generator at one layer", y=1.12,
    )
    finish_axis(axis)


def build_main_signature() -> None:
    figure = plt.figure(figsize=(14.4, 5.25))
    grid = figure.add_gridspec(
        1, 4, width_ratios=[0.92, 1.34, 1.08, 1.12], wspace=0.53,
    )
    draw_cue_plane(figure.add_subplot(grid[0]))
    draw_feature_landscape(figure.add_subplot(grid[1]))
    draw_stability_orbit(figure.add_subplot(grid[2], projection="polar"))
    draw_interface_phase(figure.add_subplot(grid[3]))
    figure.subplots_adjust(left=0.055, right=0.99, top=0.80, bottom=0.19)
    save_figure(figure, "machine_signature_atlas")


def draw_holdout_relationship(axis: plt.Axes) -> None:
    data = pd.read_csv(FEATURE_ROOT / "environment_holdout_statistics.csv")
    data = data[data["environment_type"].isin(["generator", "domain"])].copy()
    for env_type, marker in (("generator", "o"), ("domain", "^")):
        for group in ("discourse", "syntactic", "lexical"):
            style = GROUP_STYLE[group]
            view = data[
                data["environment_type"].eq(env_type)
                & data["group"].eq(group)
            ]
            axis.scatter(
                view["heldout_direction_cosine"], view["test_projection_d"],
                s=58, marker=marker,
                facecolor=style["color"] if env_type == "generator" else BG,
                edgecolor=style["dark"], linewidth=1.05,
                alpha=0.9, zorder=3,
            )
    annotation_specs = {
        ("generator", "MiniMax-M2.5", "discourse"): (5, -8),
        ("generator", "deepseek-v4-flash", "discourse"): (-5, 5),
        ("domain", "instructional", "discourse"): (5, 4),
        ("domain", "encyclopedic", "syntactic"): (-5, 5),
        ("domain", "literary_narrative", "syntactic"): (5, -8),
    }
    for key, offset in annotation_specs.items():
        env_type, environment, group = key
        row = data[
            data["environment_type"].eq(env_type)
            & data["heldout_environment"].eq(environment)
            & data["group"].eq(group)
        ].iloc[0]
        label = ENVIRONMENT_LABEL[str(row["heldout_environment"])]
        axis.annotate(
            label,
            (float(row["heldout_direction_cosine"]),
             float(row["test_projection_d"])),
            xytext=offset, textcoords="offset points",
            ha="right" if offset[0] < 0 else "left",
            va="top" if offset[1] < 0 else "bottom", fontsize=7.5,
            color=GROUP_STYLE[str(row["group"])]["dark"],
        )
    axis.axvline(0.90, color=GRAY, linestyle="--", linewidth=0.9)
    axis.axhline(0.0, color=INK, linewidth=0.9)
    axis.set_xlim(-0.03, 1.03)
    axis.set_ylim(-0.15, 2.55)
    axis.set_xlabel("Held-out/full direction cosine")
    axis.set_ylabel("Held-out standardized projection")
    legend = [
        Line2D([0], [0], marker="o", linestyle="none", color=INK,
               markerfacecolor=INK, label="generator"),
        Line2D([0], [0], marker="^", linestyle="none", color=INK,
               markerfacecolor=BG, label="domain"),
    ] + [
        Line2D(
            [0], [0], marker=GROUP_STYLE[group]["marker"],
            linestyle="none", color=GROUP_STYLE[group]["dark"],
            markerfacecolor=GROUP_STYLE[group]["color"],
            label=GROUP_STYLE[group]["label"],
        )
        for group in ("lexical", "syntactic", "discourse")
    ]
    axis.legend(
        handles=legend, loc="upper left", ncol=2, fontsize=8.1,
        handletextpad=0.3, columnspacing=0.7, borderaxespad=0.2,
    )
    panel_title(
        axis, "A", "Alignment is not held-out effect",
        "24 folds jointly locate direction agreement and test separation",
        y=1.10,
    )
    finish_axis(axis)


def draw_surface_fold_map(axis: plt.Axes) -> None:
    data = pd.read_csv(
        PREFIX_ROOT / "transfer" / "surface_signature_transfer_folds.csv"
    )
    generator_markers = {
        "MiniMax-M2.5": "o",
        "deepseek-v4-flash": "s",
        "glm-5.2": "^",
        "qwen3.6-flash": "D",
    }
    for _, row in data.iterrows():
        concept = str(row["concept"])
        style = GROUP_STYLE[concept]
        size = 38 + 30 * float(row["control_projection_mean"])
        axis.scatter(
            float(row["plan_control_shift_cosine"]),
            float(row["control_projection_positive_rate"]),
            s=size,
            marker=generator_markers[str(row["heldout_generator"])],
            facecolor=style["color"],
            edgecolor=style["dark"], linewidth=1.1,
            alpha=0.9, zorder=3,
        )
    axis.axvline(0.80, color=GRAY, linestyle="--", linewidth=0.9)
    axis.axhline(0.75, color=GRAY, linestyle="--", linewidth=0.9)
    axis.set_xlim(0.64, 0.95)
    axis.set_ylim(0.70, 1.00)
    axis.set_xlabel("Plan/prefix signature cosine")
    axis.set_ylabel("Prefix pairs in machine direction")
    axis.text(
        0.98, 0.04, "marker area ∝ prefix projection",
        transform=axis.transAxes, ha="right", va="bottom",
        fontsize=7.8, color=MUTED,
    )
    handles = [
        Line2D(
            [0], [0], marker=marker, linestyle="none", color=GRAY_DARK,
            markerfacecolor=BG, label=ENVIRONMENT_LABEL[generator],
        )
        for generator, marker in generator_markers.items()
    ] + [
        Line2D(
            [0], [0], marker="o", linestyle="none", color=style["dark"],
            markerfacecolor=style["color"], label=style["label"],
        )
        for style in (GROUP_STYLE["lexical"], GROUP_STYLE["syntactic"])
    ]
    axis.legend(
        handles=handles,
        loc="upper left",
        ncol=2,
        fontsize=7.4,
        handletextpad=0.25,
        columnspacing=0.55,
        borderaxespad=0.2,
    )
    panel_title(
        axis, "B", "Plan-removal fold map",
        "each glyph combines orientation, prevalence, and magnitude",
        y=1.10,
    )
    finish_axis(axis)


def draw_depth_arc(axis: plt.Axes, model: str, letter: str) -> None:
    data = pd.read_csv(
        PREFIX_ROOT / "transfer" / "source_axis_transfer.csv"
    )
    view = data[data["model"].eq(model)].copy()
    layers = sorted(view["layer"].unique())
    theta = np.linspace(-0.82 * np.pi, 0.82 * np.pi, len(layers))
    summary = view.groupby("layer").agg(
        plan_mean=("plan_auc", "mean"),
        plan_min=("plan_auc", "min"),
        plan_max=("plan_auc", "max"),
        control_mean=("control_auc", "mean"),
        control_min=("control_auc", "min"),
        control_max=("control_auc", "max"),
    ).reindex(layers)
    axis.fill_between(
        theta, summary["plan_min"], summary["plan_max"],
        color=NAVY_LIGHT, alpha=0.34, linewidth=0,
    )
    axis.fill_between(
        theta, summary["control_min"], summary["control_max"],
        color=TEAL_LIGHT, alpha=0.34, linewidth=0,
    )
    axis.plot(
        theta, summary["plan_mean"], color=NAVY_DARK,
        linewidth=1.8, linestyle="--", marker="o", markersize=3.5,
        label="plan",
    )
    axis.plot(
        theta, summary["control_mean"], color=TEAL_DARK,
        linewidth=2.0, marker="D", markersize=3.4, label="prefix",
    )
    axis.set_ylim(0.75, 1.01)
    tick_indices = sorted(set([0, len(layers) // 2, len(layers) - 1]))
    axis.set_xticks(
        theta[tick_indices], [f"L{layers[index]}" for index in tick_indices],
        fontsize=8.0,
    )
    axis.set_yticks([0.8, 0.9, 1.0], [".8", ".9", "1.0"])
    axis.set_rlabel_position(180)
    axis.grid(color=GRID, linewidth=0.65)
    axis.spines["polar"].set_color("#B5BFC4")
    axis.spines["polar"].set_linewidth(1.0)
    axis.text(
        0.5, -0.08, MODEL_LABEL[model], transform=axis.transAxes,
        ha="center", va="top", fontsize=9.4, color=INK,
    )
    if letter:
        panel_title(
            axis, letter, "Depth arcs after plan removal",
            "line: four-generator mean · band: full generator range",
            y=1.29, note_gap=0.07,
        )
        axis.legend(
            loc="lower center", bbox_to_anchor=(1.08, -0.27), ncol=2,
            fontsize=8.0, handlelength=1.4, columnspacing=0.8,
        )


def build_surface_transfer() -> None:
    figure = plt.figure(figsize=(14.4, 5.05))
    grid = figure.add_gridspec(
        1, 4, width_ratios=[1.35, 1.05, 0.88, 0.88], wspace=0.48,
    )
    draw_holdout_relationship(figure.add_subplot(grid[0]))
    draw_surface_fold_map(figure.add_subplot(grid[1]))
    draw_depth_arc(
        figure.add_subplot(grid[2], projection="polar"),
        "Llama-3.1-8B-Instruct", "C",
    )
    draw_depth_arc(
        figure.add_subplot(grid[3], projection="polar"), "Qwen3-4B", "",
    )
    figure.subplots_adjust(left=0.065, right=0.99, top=0.80, bottom=0.18)
    save_figure(figure, "appendix_surface_transfer")


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


def draw_robustness_glyph(
    axis: plt.Axes,
    data: pd.DataFrame,
    model: str,
    concept: str,
    letter: str,
) -> None:
    style = GROUP_STYLE[concept]
    part = data[
        data["model"].eq(model) & data["concept"].eq(concept)
    ]
    null = part[part["sensitivity"].eq("null_k")].sort_values("value")
    fit = part[part["sensitivity"].eq("fit_pairs")].sort_values("value")
    labels = [f"k{int(value)}" for value in null["value"]] + [
        f"n{int(value)}" for value in fit["value"]
    ]
    values = np.r_[
        null["cosine_vs_canonical"].to_numpy(float),
        fit["cosine_vs_canonical"].to_numpy(float),
    ]
    angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False)
    split = len(null)
    for index, (angle, value) in enumerate(zip(angles, values)):
        is_null = index < split
        axis.plot(
            [angle, angle], [0.88, value], color=style["light"],
            linewidth=2.2, zorder=1,
        )
        axis.scatter(
            angle, value, s=40, marker="o" if is_null else "D",
            facecolor=style["color"] if is_null else BG,
            edgecolor=style["dark"], linewidth=1.0, zorder=3,
        )
    axis.set_theta_offset(np.pi / 2)
    axis.set_theta_direction(-1)
    axis.set_ylim(0.88, 1.01)
    axis.set_xticks(angles, labels, fontsize=7.5)
    axis.set_yticks([0.90, 0.95, 1.0], [".90", ".95", "1"])
    axis.set_rlabel_position(205)
    axis.grid(color=GRID, linewidth=0.65)
    axis.spines["polar"].set_color("#B5BFC4")
    axis.spines["polar"].set_linewidth(1.0)
    minimum = float(values.min())
    axis.text(
        0.5, 0.5, f"min\n{minimum:.3f}", transform=axis.transAxes,
        ha="center", va="center", fontsize=8.5, color=style["dark"],
    )
    panel_title(
        axis, letter, f"{MODEL_LABEL[model]} · {style['label']}",
        "● nuisance rank  ·  ◇ fit pairs", y=1.17, note_gap=0.07,
    )


def build_hidden_robustness() -> None:
    sensitivity = pd.read_csv(
        PULLBACK_ROOT / "design_sensitivity" / "direction_sensitivity.csv"
    )
    figure = plt.figure(figsize=(14.4, 5.45))
    outer = figure.add_gridspec(
        1, 2, width_ratios=[1.05, 2.0], wspace=0.30,
    )
    draw_dominance_plane(figure.add_subplot(outer[0]))
    glyph_grid = outer[1].subgridspec(2, 2, hspace=0.62, wspace=0.25)
    letters = iter(("B", "C", "D", "E"))
    for row, model in enumerate(MODELS):
        for col, concept in enumerate(CONCEPTS):
            draw_robustness_glyph(
                figure.add_subplot(glyph_grid[row, col], projection="polar"),
                sensitivity, model, concept, next(letters),
            )
    figure.subplots_adjust(left=0.07, right=0.985, top=0.84, bottom=0.16)
    save_figure(figure, "appendix_hidden_robustness")


def confirmatory_root(model: str) -> Path:
    return PULLBACK_ROOT / model / "mean_pool"


def learned_behavior_point(model: str, concept: str) -> tuple[float, float]:
    response = pd.read_csv(
        confirmatory_root(model) / "confirmatory_analysis"
        / "causal_group_statistics.csv"
    )
    target = response[
        response["component"].eq(concept)
        & response["group"].eq(concept)
    ].iloc[0]
    other = response[
        response["component"].eq(concept)
        & ~response["group"].eq(concept)
        & response["group"].isin(["lexical", "syntactic", "discourse"])
    ]["signature_projection_mean"].mean()
    target_value = float(target["signature_projection_mean"])
    return abs(target_value), abs(target_value - float(other))


def random_behavior_points(model: str, concept: str) -> pd.DataFrame:
    frame = pd.read_csv(
        confirmatory_root(model) / "confirmatory_analysis"
        / "behavior_null_axis_statistics.csv"
    )
    view = frame[frame["concept"].eq(concept)].pivot(
        index="axis", columns="estimand", values="abs_axis_mean",
    )
    return view.loc[:, ["target_projection", "component_selectivity"]]


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Return a 2-D monotonic-chain hull without a SciPy dependency."""
    unique = sorted({(float(x), float(y)) for x, y in points})
    if len(unique) <= 2:
        return np.asarray(unique)

    def cross(
        origin: tuple[float, float],
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return (
            (first[0] - origin[0]) * (second[1] - origin[1])
            - (first[1] - origin[1]) * (second[0] - origin[0])
        )

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1])


def draw_null_envelope(
    axis: plt.Axes,
    model: str,
    concept: str,
    letter: str,
    *,
    show_x: bool,
    show_y: bool,
) -> None:
    style = GROUP_STYLE[concept]
    random = random_behavior_points(model, concept)
    points = random.to_numpy(float)
    hull = convex_hull(points)
    if len(hull) >= 3:
        axis.add_patch(
            Polygon(
                hull, closed=True, facecolor=GRAY_LIGHT,
                edgecolor=GRAY, linewidth=1.1, alpha=0.70, zorder=1,
            )
        )
    axis.scatter(
        points[:, 0], points[:, 1], s=28, facecolor=BG,
        edgecolor=GRAY_DARK, linewidth=0.85, zorder=2,
    )
    learned = learned_behavior_point(model, concept)
    axis.scatter(
        learned[0], learned[1], s=84, marker="D",
        facecolor=style["color"], edgecolor=style["dark"],
        linewidth=1.2, zorder=4,
    )
    axis.annotate(
        "learned", learned, xytext=(5, 4), textcoords="offset points",
        fontsize=7.6, color=style["dark"],
    )
    axis.set_xlim(-0.012, 0.46)
    axis.set_ylim(-0.012, 0.46)
    axis.set_aspect("equal", adjustable="box")
    axis.tick_params(labelbottom=show_x, labelleft=show_y)
    if show_x:
        axis.set_xlabel("|target movement|", fontsize=9.2)
    if show_y:
        axis.set_ylabel("|component selectivity|", fontsize=9.2)
    panel_title(
        axis, letter, f"{MODEL_LABEL[model]} · {style['label']}",
        "gray hull: eight orthonormal axes", y=1.14, note_gap=0.075,
    )
    finish_axis(axis)


def parse_seed_means(value: str) -> dict[int, float]:
    return {int(key): float(item) for key, item in json.loads(value).items()}


def seed_braid_values(
    model: str,
    concept: str,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    root = confirmatory_root(model) / "confirmatory_seed_robustness"
    summaries = pd.read_csv(root / "seed_robustness_summary.csv")
    pairwise = pd.read_csv(root / "pairwise_selectivity_summary.csv")
    target_row = summaries[
        summaries["component"].eq(concept)
        & summaries["estimand"].eq("target_projection")
    ].iloc[0]
    select_row = summaries[
        summaries["component"].eq(concept)
        & summaries["estimand"].eq("component_selectivity")
    ].iloc[0]
    other_groups = [
        group for group in ("lexical", "syntactic", "discourse")
        if group != concept
    ]
    pair_rows = [
        pairwise[
            pairwise["component"].eq(concept)
            & pairwise["comparison_group"].eq(group)
        ].iloc[0]
        for group in other_groups
    ]
    seed_maps = [
        parse_seed_means(str(target_row["seed_means"])),
        parse_seed_means(str(select_row["seed_means"])),
        *[parse_seed_means(str(row["seed_means"])) for row in pair_rows],
    ]
    seeds = sorted(set.intersection(*(set(values) for values in seed_maps)))
    matrix = np.asarray([
        [values[seed] for values in seed_maps]
        for seed in seeds
    ])
    aggregate = np.asarray([
        float(target_row["mean"]),
        float(select_row["mean"]),
        *[float(row["mean"]) for row in pair_rows],
    ])
    labels = [
        "target", "avg.\nselect.",
        f"vs {other_groups[0][:4]}", f"vs {other_groups[1][:4]}",
    ]
    return labels, matrix, aggregate


def draw_seed_braid(
    axis: plt.Axes,
    model: str,
    concept: str,
    letter: str,
    *,
    show_x: bool,
    show_y: bool,
) -> None:
    style = GROUP_STYLE[concept]
    labels, matrix, aggregate = seed_braid_values(model, concept)
    x = np.arange(len(labels))
    seed_markers = ("o", "s", "^")
    seed_alpha = (0.42, 0.58, 0.74)
    for index, values in enumerate(matrix):
        axis.plot(
            x, values, color=style["color"], linewidth=1.25,
            alpha=seed_alpha[index], marker=seed_markers[index],
            markersize=4.0, markerfacecolor=BG,
            markeredgecolor=style["dark"], markeredgewidth=0.8, zorder=2,
        )
    axis.plot(
        x, aggregate, color=style["dark"], linewidth=2.4,
        marker="D", markersize=5.2, markerfacecolor=style["color"],
        markeredgecolor=style["dark"], markeredgewidth=0.8, zorder=4,
    )
    axis.axhspan(-0.025, 0.025, color=PALE_BAND, zorder=0)
    axis.axhline(0, color=INK, linewidth=0.9, zorder=1)
    axis.set_xlim(-0.22, 3.22)
    axis.set_ylim(-0.34, 0.73)
    axis.set_xticks(x, labels if show_x else [""] * len(labels))
    axis.tick_params(labelleft=show_y)
    if show_y:
        axis.set_ylabel("Signed effect", fontsize=9.2)
    panel_title(
        axis, letter, f"{MODEL_LABEL[model]} · {style['label']}",
        "thin paths: seeds · thick path: clustered mean",
        y=1.14, note_gap=0.075,
    )
    finish_axis(axis, "y")


def build_generation_robustness() -> None:
    figure = plt.figure(figsize=(14.4, 6.45))
    outer = figure.add_gridspec(2, 4, hspace=0.62, wspace=0.34)
    cases = [
        ("Llama-3.1-8B-Instruct", "lexical"),
        ("Llama-3.1-8B-Instruct", "syntactic"),
        ("Qwen3-4B", "lexical"),
        ("Qwen3-4B", "syntactic"),
    ]
    top_letters = ("A", "B", "C", "D")
    bottom_letters = ("E", "F", "G", "H")
    for column, (model, concept) in enumerate(cases):
        draw_null_envelope(
            figure.add_subplot(outer[0, column]), model, concept,
            top_letters[column], show_x=True, show_y=column == 0,
        )
        draw_seed_braid(
            figure.add_subplot(outer[1, column]), model, concept,
            bottom_letters[column], show_x=True, show_y=column == 0,
        )
    figure.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.13)
    save_figure(figure, "appendix_generation_robustness")


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


def draw_cue_constellation(axis: plt.Axes) -> None:
    data = cue_statistics()
    ratio = np.log2(
        (data["machine_count"].to_numpy(float) + 0.5)
        / (data["human_count"].to_numpy(float) + 0.5)
    )
    prevalence = data["machine_prevalence"].to_numpy(float)
    sizes = 8.5 + 4.2 * np.sqrt(data["machine_count"].to_numpy(float) / 83.0)
    display_position = {
        "however": (1.36, 16.7),
        "despite": (2.24, 7.4),
        "crucial": (2.72, 4.7),
        "noted that": (4.55, 2.45),
        "stated that": (5.08, 3.85),
    }
    for index, row in data.iterrows():
        label_x, label_y = display_position[str(row["form"])]
        axis.plot(
            [ratio[index], label_x],
            [prevalence[index], label_y],
            color=NAVY_LIGHT,
            linewidth=0.9,
            zorder=1,
        )
        axis.scatter(
            ratio[index],
            prevalence[index],
            s=12,
            facecolor=BG,
            edgecolor=NAVY_DARK,
            linewidth=0.7,
            zorder=2,
        )
        axis.text(
            label_x,
            label_y,
            str(row["form"]),
            ha="center",
            va="center",
            fontsize=float(sizes[index]),
            color=NAVY_DARK,
            bbox={
                "boxstyle": "round,pad=0.20",
                "facecolor": NAVY_LIGHT,
                "edgecolor": "none",
                "alpha": 0.72,
            },
            zorder=3,
        )
        axis.annotate(
            f"{int(row['human_count'])}→{int(row['machine_count'])}",
            (label_x, label_y),
            xytext=(0, -13),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=7.5,
            color=MUTED,
        )
    axis.axvline(0, color=GRAY, linewidth=0.9, linestyle="--")
    axis.set_xlim(0.8, 5.45)
    axis.set_ylim(0.7, 18.4)
    axis.set_xlabel("Smoothed log₂ document-rate ratio")
    axis.set_ylabel("Machine documents (%)")
    panel_title(
        axis,
        "A",
        "Lexical cue constellation",
        "the words themselves are data marks; labels show human→machine counts",
        y=1.10,
    )
    finish_axis(axis)


def draw_feature_strips(axis: plt.Axes) -> None:
    data = feature_landscape_frame()
    group_y = {"lexical": 2.0, "syntactic": 1.0, "discourse": 0.0}
    for group, center in group_y.items():
        style = GROUP_STYLE[group]
        view = data[data["group"].eq(group)].sort_values("paired_d")
        jitter = np.resize(
            np.array([0.0, 0.08, -0.08, 0.15, -0.15]), len(view)
        )
        size = 24 + 64 * np.sqrt(view["active_pair_rate"].to_numpy(float))
        axis.axhline(center, color=style["light"], linewidth=9.0, alpha=0.42)
        axis.scatter(
            view["paired_d"],
            center + jitter,
            s=size,
            marker=style["marker"],
            facecolor=BG,
            edgecolor=style["dark"],
            linewidth=1.15,
            zorder=3,
        )
    callouts = {
        "function_word_rate": (-1.12, 2.28),
        "mattr_50": (0.53, 2.28),
        "dependency_length_mean": (-0.60, 1.28),
        "nominalization_rate": (0.28, 1.28),
        "subject_continuity": (-0.22, 0.28),
        "transition_opening_rate": (0.25, 0.28),
    }
    short_labels = {
        "function_word_rate": "function words",
        "mattr_50": "lexical diversity",
        "dependency_length_mean": "dependency length",
        "nominalization_rate": "nominalization",
        "subject_continuity": "subject continuity",
        "transition_opening_rate": "transition openings",
    }
    for feature, target in callouts.items():
        row = data[data["feature"].eq(feature)].iloc[0]
        center = group_y[str(row["group"])]
        axis.annotate(
            short_labels[feature],
            (float(row["paired_d"]), center),
            xytext=target,
            textcoords="data",
            ha="center",
            va="bottom",
            fontsize=7.9,
            color=GROUP_STYLE[str(row["group"])]["dark"],
            arrowprops={
                "arrowstyle": "-",
                "color": GROUP_STYLE[str(row["group"])]["light"],
                "linewidth": 0.9,
            },
        )
    axis.axvline(0, color=INK, linewidth=0.95)
    axis.set_xlim(-1.38, 0.79)
    axis.set_ylim(-0.35, 2.42)
    axis.set_yticks([2, 1, 0], ["Lexical", "Syntactic", "Discourse"])
    axis.tick_params(axis="y", length=0, pad=7)
    axis.set_xlabel("Paired standardized shift   human ← 0 → machine")
    panel_title(
        axis,
        "B",
        "Full feature fingerprint",
        "all 35 frozen measurements, not only the six largest effects",
        y=1.10,
    )
    finish_axis(axis, "x")


def style_violin(
    axis: plt.Axes,
    values: np.ndarray,
    position: float,
    *,
    face: str,
    edge: str,
    alpha: float,
    width: float = 0.58,
) -> None:
    result = axis.violinplot(
        [values],
        positions=[position],
        vert=False,
        widths=width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method=0.22,
    )
    for body in result["bodies"]:
        body.set_facecolor(face)
        body.set_edgecolor(edge)
        body.set_linewidth(0.8)
        body.set_alpha(alpha)
    q10, median, q90 = np.quantile(values, [0.1, 0.5, 0.9])
    axis.plot(
        [q10, q90], [position, position],
        color=edge, linewidth=LINE_STANDARD,
    )
    axis.scatter(
        median,
        position,
        s=18,
        facecolor=edge,
        edgecolor=BG,
        linewidth=0.45,
        zorder=4,
    )


def draw_pair_transfer_overview(axis: plt.Axes) -> None:
    scores = crossfit_surface_scores()
    for concept in CONCEPTS:
        style = GROUP_STYLE[concept]
        view = scores[scores["concept"].eq(concept)]
        axis.scatter(
            view["plan_score"],
            view["control_score"],
            s=MARK_RAW,
            marker="o",
            facecolor=style["color"],
            edgecolor="none",
            alpha=0.24,
            zorder=2,
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
        axis.scatter(
            view["plan_score"].mean(),
            view["control_score"].mean(),
            s=MARK_SUMMARY,
            marker="o",
            facecolor=style["color"],
            edgecolor=INK,
            linewidth=0.9,
            zorder=4,
        )
    values = np.r_[scores["plan_score"], scores["control_score"]]
    low, high = np.quantile(values, [0.005, 0.995])
    padding = 0.05 * (high - low)
    low, high = low - padding, high + padding
    axis.axline((0, 0), slope=1, color=GRAY, linewidth=1.0, linestyle="--")
    axis.axhline(0, color=INK, linewidth=0.8)
    axis.axvline(0, color=INK, linewidth=0.8)
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_xlabel("Plan projection")
    axis.set_ylabel("Prefix-only projection")
    axis.legend(
        handles=[
            Line2D(
                [0], [0], marker="o",
                linestyle="none", color=GROUP_STYLE[c]["dark"],
                markerfacecolor=GROUP_STYLE[c]["color"],
                label=GROUP_STYLE[c]["label"],
            )
            for c in CONCEPTS
        ],
        loc="lower right",
        fontsize=8.8,
        handletextpad=0.3,
    )
    panel_title(
        axis,
        "C",
        "Pairwise transfer",
        y=1.10,
    )
    finish_axis(axis)


def draw_holdout_plane_clean(axis: plt.Axes) -> None:
    data = pd.read_csv(FEATURE_ROOT / "environment_holdout_statistics.csv")
    data = data[data["environment_type"].isin(["generator", "domain"])].copy()
    for group in ("discourse", "syntactic", "lexical"):
        style = GROUP_STYLE[group]
        for env_type in ("generator", "domain"):
            view = data[
                data["group"].eq(group)
                & data["environment_type"].eq(env_type)
            ]
            axis.scatter(
                view["heldout_direction_cosine"],
                view["test_projection_d"],
                s=42,
                marker="o",
                facecolor=style["color"] if env_type == "generator" else BG,
                edgecolor=style["dark"],
                linewidth=0.8,
                alpha=0.90,
                zorder=3,
            )
    axis.axvspan(0.9, 1.03, color=PALE_BAND, zorder=0)
    axis.axvline(0.9, color=GRAY, linewidth=0.85, linestyle="--")
    axis.axhline(0, color=INK, linewidth=0.9)
    annotations = [
        ("domain", "instructional", "discourse", (7, 5), "discourse\nweak"),
        ("domain", "encyclopedic", "syntactic", (-7, 6), "syntax bends\nby domain"),
    ]
    for env_type, environment, group, offset, label in annotations:
        row = data[
            data["environment_type"].eq(env_type)
            & data["heldout_environment"].eq(environment)
            & data["group"].eq(group)
        ].iloc[0]
        axis.annotate(
            label,
            (float(row["heldout_direction_cosine"]),
             float(row["test_projection_d"])),
            xytext=offset,
            textcoords="offset points",
            ha="right" if offset[0] < 0 else "left",
            va="bottom",
            fontsize=8.4,
            color=GROUP_STYLE[group]["dark"],
        )
    axis.set_xlim(-0.03, 1.03)
    axis.set_ylim(-0.12, 2.56)
    axis.set_xlabel("Held-out/full direction cosine")
    axis.set_ylabel("Held-out standardized projection")
    handles = [
        Line2D([0], [0], marker="o", linestyle="none", color=INK,
               markerfacecolor=INK, label="generator"),
        Line2D([0], [0], marker="o", linestyle="none", color=INK,
               markerfacecolor=BG, label="domain"),
    ] + [
        Line2D(
            [0], [0], marker="o",
            linestyle="none", color=GROUP_STYLE[group]["dark"],
            markerfacecolor=GROUP_STYLE[group]["color"],
            label=GROUP_STYLE[group]["label"],
        )
        for group in ("lexical", "syntactic", "discourse")
    ]
    axis.legend(
        handles=handles,
        loc="upper left",
        ncol=2,
        fontsize=8.6,
        handletextpad=0.3,
        columnspacing=0.65,
        borderaxespad=0.2,
    )
    panel_title(
        axis,
        "D",
        "Environment stability",
        "direction agreement alone does not guarantee held-out separation",
        y=1.10,
    )
    finish_axis(axis)


def draw_feature_effect_distributions(axis: plt.Axes) -> None:
    data = feature_landscape_frame()
    centers = {"lexical": 2.0, "syntactic": 1.0, "discourse": 0.0}
    for group, center in centers.items():
        style = GROUP_STYLE[group]
        view = data[data["group"].eq(group)].sort_values("paired_d").copy()
        values = view["paired_d"].to_numpy(float)
        style_violin(
            axis,
            values,
            center,
            face=style["light"],
            edge=style["dark"],
            alpha=0.90,
            width=0.76,
        )
        offsets = np.linspace(-0.22, 0.22, len(view))
        for offset, (_, row) in zip(offsets, view.iterrows()):
            effect = float(row["paired_d"])
            mean = float(row["paired_difference_mean"])
            scale = abs(mean / effect) if abs(effect) > 1e-12 else np.nan
            if np.isfinite(scale) and scale > 1e-12:
                low = float(row["paired_difference_ci_low"]) / scale
                high = float(row["paired_difference_ci_high"]) / scale
                axis.plot(
                    [low, high],
                    [center + offset, center + offset],
                    color=style["dark"],
                    linewidth=LINE_HAIR,
                    alpha=0.56,
                    zorder=2,
                )
            axis.plot(
                [effect, effect],
                [center + offset - 0.025, center + offset + 0.025],
                color=style["dark"],
                linewidth=0.9,
                zorder=3,
            )
        axis.text(
            0.73,
            center + 0.29,
            rf"$\mu={values.mean():+.2f}$",
            ha="right",
            va="center",
            fontsize=8.5,
            color=style["dark"],
        )
    axis.axvline(0, color=INK, linewidth=0.95)
    axis.set_xlim(-1.38, 0.79)
    axis.set_ylim(-0.48, 2.48)
    axis.set_yticks([2, 1, 0], ["Lexical", "Syntactic", "Discourse"])
    axis.tick_params(axis="y", length=0, pad=5)
    axis.set_xlabel("Paired standardized shift")
    panel_title(axis, "B", "Effect distributions", y=1.12)
    finish_axis(axis, "x")


def build_main_signature_v3() -> None:
    figure = plt.figure(figsize=(13.2, 3.45))
    grid = figure.add_gridspec(
        1,
        4,
        width_ratios=[1.08, 0.92, 1.0, 1.08],
        wspace=0.30,
    )
    draw_feature_landscape(figure.add_subplot(grid[0]), letter="A")
    draw_feature_effect_distributions(figure.add_subplot(grid[1]))
    draw_pair_transfer_overview(figure.add_subplot(grid[2]))
    draw_holdout_plane_clean(figure.add_subplot(grid[3]))
    figure.subplots_adjust(left=0.065, right=0.992, top=0.84, bottom=0.20)
    save_figure(figure, "machine_signature_atlas")


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


def draw_readout_displacement(axis: plt.Axes) -> None:
    data = pd.read_csv(PREFIX_ROOT / "transfer" / "concept_mapping_transfer.csv")
    for model in sorted(data["model"].unique()):
        model_data = data[data["model"].eq(model)]
        input_layer = int(model_data["layer"].min())
        anchor_layer = int(model_data["layer"].max())
        for concept in CONCEPTS:
            style = GROUP_STYLE[concept]
            starts = []
            ends = []
            for generator in sorted(model_data["heldout_generator"].unique()):
                pair = model_data[
                    model_data["concept"].eq(concept)
                    & model_data["heldout_generator"].eq(generator)
                ]
                start = pair[pair["layer"].eq(input_layer)].iloc[0]
                end = pair[pair["layer"].eq(anchor_layer)].iloc[0]
                start_xy = (
                    float(start["plan_ai_score_correlation"]),
                    float(start["control_ai_score_correlation"]),
                )
                end_xy = (
                    float(end["plan_ai_score_correlation"]),
                    float(end["control_ai_score_correlation"]),
                )
                starts.append(start_xy)
                ends.append(end_xy)
                axis.annotate(
                    "",
                    xy=end_xy,
                    xytext=start_xy,
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": style["color"],
                        "linewidth": 0.75,
                        "alpha": 0.32,
                        "mutation_scale": 7,
                    },
                    zorder=1,
                )
            start_mean = np.mean(np.asarray(starts), axis=0)
            end_mean = np.mean(np.asarray(ends), axis=0)
            axis.annotate(
                "",
                xy=end_mean,
                xytext=start_mean,
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": style["dark"],
                    "linewidth": 1.8,
                    "mutation_scale": 10,
                },
                zorder=3,
            )
            axis.scatter(
                *end_mean,
                s=58,
                marker=MODEL_MARKER[model],
                facecolor=style["color"],
                edgecolor=style["dark"],
                linewidth=1.0,
                zorder=4,
            )
    axis.axline((0, 0), slope=1, color=GRAY, linewidth=0.9, linestyle="--")
    axis.set_xlim(0.04, 0.72)
    axis.set_ylim(0.04, 0.72)
    axis.set_xlabel("Plan readout r")
    axis.set_ylabel("Prefix-only readout r")
    panel_title(axis, "C", "Input-to-anchor displacement", y=1.10)
    handles = [
        Line2D(
            [0], [0], marker=MODEL_MARKER[m], linestyle="none",
            color=GRAY_DARK, markerfacecolor=BG, label=MODEL_LABEL[m],
        )
        for m in ("Llama-3.1-8B-Instruct", "Qwen3-4B", "gemma-4-E2B-it")
    ] + [
        Line2D(
            [0], [0], color=GROUP_STYLE[c]["dark"], linewidth=2,
            label=GROUP_STYLE[c]["label"],
        )
        for c in CONCEPTS
    ]
    axis.legend(
        handles=handles,
        loc="lower right",
        ncol=2,
        fontsize=7.8,
        columnspacing=0.7,
        handletextpad=0.3,
    )
    finish_axis(axis)


def draw_readout_phase_small(axis: plt.Axes, stage: str, letter: str) -> None:
    data = pd.read_csv(
        PREFIX_ROOT / "transfer" / "concept_mapping_transfer.csv"
    )
    for model in sorted(data["model"].unique()):
        model_data = data[data["model"].eq(model)]
        layer = (
            model_data["layer"].min()
            if stage == "input"
            else model_data["layer"].max()
        )
        for concept in CONCEPTS:
            style = GROUP_STYLE[concept]
            view = model_data[
                model_data["layer"].eq(layer)
                & model_data["concept"].eq(concept)
            ]
            axis.scatter(
                view["plan_ai_score_correlation"],
                view["control_ai_score_correlation"],
                s=42,
                marker=MODEL_MARKER[model],
                facecolor=style["color"],
                edgecolor=style["dark"],
                linewidth=0.9,
                alpha=0.82,
                zorder=3,
            )
    axis.axline((0, 0), slope=1, color=GRAY, linewidth=0.9, linestyle="--")
    axis.set_xlim(0.04, 0.72)
    axis.set_ylim(0.04, 0.72)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Plan r", fontsize=9.6)
    axis.set_ylabel("Prefix r", fontsize=9.6)
    panel_title(
        axis,
        letter,
        "Input layers" if stage == "input" else "Readout anchors",
        "one point per generator",
        y=1.14,
        note_gap=0.075,
    )
    if stage == "anchor":
        handles = [
            Line2D(
                [0], [0], marker=MODEL_MARKER[model], linestyle="none",
                color=GRAY_DARK, markerfacecolor=BG,
                label=MODEL_LABEL[model],
            )
            for model in (
                "Llama-3.1-8B-Instruct", "Qwen3-4B", "gemma-4-E2B-it"
            )
        ]
        axis.legend(
            handles=handles,
            loc="lower right",
            fontsize=8.4,
            handletextpad=0.3,
            borderaxespad=0.2,
        )
    finish_axis(axis)


def draw_depth_delta(axis: plt.Axes, model: str, letter: str) -> None:
    data = pd.read_csv(
        PREFIX_ROOT / "transfer" / "source_axis_transfer.csv"
    )
    view = data[data["model"].eq(model)].copy()
    max_layer = float(view["layer"].max())
    view["delta"] = view["control_auc"] - view["plan_auc"]
    summary = view.groupby("layer")["delta"].agg(
        mean="mean",
        low="min",
        high="max",
    ).reset_index()
    x = summary["layer"].to_numpy(float) / max_layer
    axis.fill_between(
        x,
        summary["low"].to_numpy(float),
        summary["high"].to_numpy(float),
        color=NAVY_LIGHT,
        alpha=0.82,
        linewidth=0,
        zorder=1,
    )
    axis.plot(
        x,
        summary["mean"].to_numpy(float),
        color=NAVY_DARK,
        linewidth=1.9,
        zorder=4,
    )
    axis.axhline(0, color=INK, linewidth=0.9)
    axis.set_xlim(0.0, 1.02)
    axis.set_ylim(-0.075, 0.095)
    axis.set_xlabel("Relative layer depth", fontsize=9.0)
    axis.set_ylabel("Prefix − plan AUC", fontsize=9.0)
    panel_title(
        axis,
        letter,
        MODEL_LABEL[model],
        y=1.14,
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


def build_surface_transfer_v3() -> None:
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


def build_hidden_robustness_v3() -> None:
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


def draw_seed_phase(
    axis: plt.Axes,
    model: str,
    concept: str,
    letter: str,
    *,
    show_y: bool,
) -> None:
    root = confirmatory_root(model) / "confirmatory_seed_robustness"
    summary = pd.read_csv(root / "seed_robustness_summary.csv")
    target = summary[
        summary["component"].eq(concept)
        & summary["estimand"].eq("target_projection")
    ].iloc[0]
    selectivity = summary[
        summary["component"].eq(concept)
        & summary["estimand"].eq("component_selectivity")
    ].iloc[0]
    target_seeds = parse_seed_means(str(target["seed_means"]))
    select_seeds = parse_seed_means(str(selectivity["seed_means"]))
    seeds = sorted(set(target_seeds) & set(select_seeds))
    style = GROUP_STYLE[concept]
    for seed in seeds:
        x = target_seeds[seed]
        y = select_seeds[seed]
        axis.scatter(
            x,
            y,
            s=42,
            facecolor=BG,
            edgecolor=style["dark"],
            linewidth=1.0,
            zorder=3,
        )
        axis.annotate(
            str(seed + 1),
            (x, y),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7.3,
            color=style["dark"],
        )
    x_mean = float(target["mean"])
    y_mean = float(selectivity["mean"])
    axis.plot(
        [float(target["ci_low"]), float(target["ci_high"])],
        [y_mean, y_mean],
        color=style["light"],
        linewidth=4.0,
        zorder=1,
    )
    axis.plot(
        [x_mean, x_mean],
        [float(selectivity["ci_low"]), float(selectivity["ci_high"])],
        color=style["light"],
        linewidth=4.0,
        zorder=1,
    )
    axis.scatter(
        x_mean,
        y_mean,
        s=72,
        marker="D",
        facecolor=style["color"],
        edgecolor=style["dark"],
        linewidth=1.1,
        zorder=4,
    )
    axis.axhline(0, color=INK, linewidth=0.85)
    axis.axvline(0, color=INK, linewidth=0.85)
    axis.set_xlim(-0.31, 0.55)
    axis.set_ylim(-0.31, 0.55)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Target projection", fontsize=9.0)
    axis.tick_params(labelleft=show_y)
    if show_y:
        axis.set_ylabel("Component selectivity", fontsize=9.0)
    panel_title(
        axis,
        letter,
        f"{MODEL_LABEL[model]} · {GROUP_STYLE[concept]['label']}",
        "numbered points: seeds · cross: clustered 95% intervals",
        y=1.14,
        note_gap=0.075,
    )
    finish_axis(axis)


def draw_model_null_envelopes(
    axis: plt.Axes,
    model: str,
    letter: str,
    *,
    show_y: bool,
) -> None:
    for concept in CONCEPTS:
        style = GROUP_STYLE[concept]
        points = random_behavior_points(model, concept).to_numpy(float)
        hull = convex_hull(points)
        if len(hull) >= 3:
            axis.add_patch(
                Polygon(
                    hull,
                    closed=True,
                    facecolor=style["light"],
                    edgecolor=style["dark"],
                    linewidth=1.0,
                    alpha=0.34,
                    zorder=1,
                )
            )
        axis.scatter(
            points[:, 0],
            points[:, 1],
            s=28,
            facecolor=BG,
            edgecolor=style["dark"],
            linewidth=0.8,
            alpha=0.78,
            zorder=2,
        )
        learned = learned_behavior_point(model, concept)
        axis.scatter(
            learned[0],
            learned[1],
            s=82,
            marker="D",
            facecolor=style["color"],
            edgecolor=style["dark"],
            linewidth=1.1,
            zorder=4,
        )
        axis.annotate(
            "Lex" if concept == "lexical" else "Syn",
            learned,
            xytext=(6, 3),
            textcoords="offset points",
            fontsize=8.2,
            color=style["dark"],
        )
    axis.set_xlim(-0.012, 0.46)
    axis.set_ylim(-0.012, 0.46)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("|target movement|", fontsize=11.5)
    axis.tick_params(labelleft=show_y)
    if show_y:
        axis.set_ylabel("|component selectivity|", fontsize=11.5)
    panel_title(
        axis,
        letter,
        f"{MODEL_LABEL[model]} joint null envelopes",
        "open points: eight controls · diamonds: learned directions",
        y=1.11,
    )
    finish_axis(axis)


def draw_model_seed_phases(
    axis: plt.Axes,
    model: str,
    letter: str,
    *,
    show_y: bool,
) -> None:
    summary = pd.read_csv(
        confirmatory_root(model)
        / "confirmatory_seed_robustness"
        / "seed_robustness_summary.csv"
    )
    for concept in CONCEPTS:
        style = GROUP_STYLE[concept]
        target = summary[
            summary["component"].eq(concept)
            & summary["estimand"].eq("target_projection")
        ].iloc[0]
        selectivity = summary[
            summary["component"].eq(concept)
            & summary["estimand"].eq("component_selectivity")
        ].iloc[0]
        target_seeds = parse_seed_means(str(target["seed_means"]))
        select_seeds = parse_seed_means(str(selectivity["seed_means"]))
        for seed in sorted(set(target_seeds) & set(select_seeds)):
            x = target_seeds[seed]
            y = select_seeds[seed]
            axis.scatter(
                x,
                y,
                s=40,
                facecolor=BG,
                edgecolor=style["dark"],
                linewidth=1.0,
                zorder=3,
            )
            axis.annotate(
                f"{'L' if concept == 'lexical' else 'S'}{seed + 1}",
                (x, y),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7.3,
                color=style["dark"],
            )
        x_mean = float(target["mean"])
        y_mean = float(selectivity["mean"])
        axis.plot(
            [float(target["ci_low"]), float(target["ci_high"])],
            [y_mean, y_mean],
            color=style["light"],
            linewidth=4.0,
            zorder=1,
        )
        axis.plot(
            [x_mean, x_mean],
            [float(selectivity["ci_low"]), float(selectivity["ci_high"])],
            color=style["light"],
            linewidth=4.0,
            zorder=1,
        )
        axis.scatter(
            x_mean,
            y_mean,
            s=74,
            marker="D",
            facecolor=style["color"],
            edgecolor=style["dark"],
            linewidth=1.1,
            zorder=4,
        )
    axis.axhline(0, color=INK, linewidth=0.85)
    axis.axvline(0, color=INK, linewidth=0.85)
    axis.set_xlim(-0.31, 0.55)
    axis.set_ylim(-0.31, 0.55)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("Target projection", fontsize=11.5)
    axis.tick_params(labelleft=show_y)
    if show_y:
        axis.set_ylabel("Component selectivity", fontsize=11.5)
    panel_title(
        axis,
        letter,
        f"{MODEL_LABEL[model]} seed coordinates",
        "L/S + number: lexical/syntactic seed · cross: clustered 95% CI",
        y=1.11,
    )
    finish_axis(axis)


@lru_cache(maxsize=None)
def prompt_seed_coordinates(model: str, concept: str) -> pd.DataFrame:
    frame = pd.read_csv(
        confirmatory_root(model)
        / "confirmatory_seed_robustness"
        / "prompt_seed_estimands.csv"
    )
    view = frame[frame["component"].eq(concept)].pivot_table(
        index=["prompt_id", "prompt_index", "sample_seed_index"],
        columns="estimand",
        values="value",
    ).reset_index()
    return view


def signed_behavior_controls(model: str, concept: str) -> pd.DataFrame:
    frame = pd.read_csv(
        confirmatory_root(model)
        / "confirmatory_analysis"
        / "behavior_null_axis_statistics.csv"
    )
    return frame[frame["concept"].eq(concept)].pivot(
        index="axis", columns="estimand", values="signed_mean"
    ).loc[:, ["target_projection", "component_selectivity"]]


def draw_prompt_marginal(
    axis: plt.Axes,
    values: np.ndarray,
    *,
    style: dict[str, str],
    orientation: str,
    limit: float,
) -> None:
    """Add a quiet Gaussian KDE without introducing another legend."""
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    centers = np.linspace(-limit, limit, 180)
    spread = float(np.std(values, ddof=1))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    robust_spread = float((q75 - q25) / 1.349)
    scale = min(spread, robust_spread) if robust_spread > 0 else spread
    bandwidth = 0.9 * scale * max(len(values), 2) ** (-0.2)
    bandwidth = float(np.clip(bandwidth, 0.20, 0.55))
    standardized = (centers[:, None] - values[None, :]) / bandwidth
    counts = np.exp(-0.5 * standardized ** 2).mean(axis=1)
    counts /= bandwidth * np.sqrt(2.0 * np.pi)
    if orientation == "x":
        axis.fill_between(
            centers, 0, counts,
            color=style["light"], alpha=0.92, linewidth=0,
        )
        axis.plot(centers, counts, color=style["dark"], linewidth=0.9)
        axis.set_xlim(-limit, limit)
        axis.set_ylim(0, max(counts.max() * 1.08, 1e-6))
    else:
        axis.fill_betweenx(
            centers, 0, counts,
            color=style["light"], alpha=0.92, linewidth=0,
        )
        axis.plot(counts, centers, color=style["dark"], linewidth=0.9)
        axis.set_ylim(-limit, limit)
        axis.set_xlim(0, max(counts.max() * 1.08, 1e-6))
    axis.set_axis_off()


def draw_prompt_constellation(
    axis: plt.Axes,
    model: str,
    concept: str,
    letter: str,
    *,
    show_y: bool,
    show_x: bool,
    marginal_x: plt.Axes | None = None,
    marginal_y: plt.Axes | None = None,
    show_title: bool = True,
) -> None:
    style = GROUP_STYLE[concept]
    seeds = prompt_seed_coordinates(model, concept)
    means = (
        seeds.groupby("prompt_id", as_index=False)[
            ["target_projection", "component_selectivity"]
        ]
        .mean()
    )
    limit = 4.2
    visible_mask = (
        means["target_projection"].abs().le(limit)
        & means["component_selectivity"].abs().le(limit)
    )
    visible = means[visible_mask]
    if marginal_x is not None:
        draw_prompt_marginal(
            marginal_x,
            visible["target_projection"].to_numpy(float),
            style=style,
            orientation="x",
            limit=limit,
        )
    if marginal_y is not None:
        draw_prompt_marginal(
            marginal_y,
            visible["component_selectivity"].to_numpy(float),
            style=style,
            orientation="y",
            limit=limit,
        )
    axis.fill_between(
        [0.0, limit],
        [0.0, 0.0],
        [limit, limit],
        color=style["light"],
        alpha=0.62,
        linewidth=0,
        zorder=0,
    )
    axis.scatter(
        visible["target_projection"],
        visible["component_selectivity"],
        s=MARK_DETAIL,
        marker=style["marker"],
        facecolor=style["color"],
        edgecolor=BG,
        linewidth=0.16,
        alpha=0.27,
        zorder=2,
    )
    covariance_ellipse(
        axis,
        visible["target_projection"].to_numpy(float),
        visible["component_selectivity"].to_numpy(float),
        edge=style["dark"],
        level=0.5,
        linewidth=0.9,
        alpha=0.72,
        zorder=3,
    )
    controls = signed_behavior_controls(model, concept)
    axis.scatter(
        controls["target_projection"],
        controls["component_selectivity"],
        s=MARK_DETAIL,
        marker="o",
        facecolor=BG,
        edgecolor=GRAY_DARK,
        linewidth=0.75,
        alpha=0.90,
        zorder=3,
    )
    summary = pd.read_csv(
        confirmatory_root(model)
        / "confirmatory_seed_robustness"
        / "seed_robustness_summary.csv"
    )
    target = summary[
        summary["component"].eq(concept)
        & summary["estimand"].eq("target_projection")
    ].iloc[0]
    selectivity = summary[
        summary["component"].eq(concept)
        & summary["estimand"].eq("component_selectivity")
    ].iloc[0]
    x_mean = float(target["mean"])
    y_mean = float(selectivity["mean"])
    axis.plot(
        [float(target["ci_low"]), float(target["ci_high"])],
        [y_mean, y_mean],
        color=style["dark"],
        linewidth=LINE_SUMMARY,
        zorder=4,
    )
    axis.plot(
        [x_mean, x_mean],
        [float(selectivity["ci_low"]), float(selectivity["ci_high"])],
        color=style["dark"],
        linewidth=LINE_SUMMARY,
        zorder=4,
    )
    axis.scatter(
        x_mean,
        y_mean,
        s=MARK_SUMMARY,
        marker="D",
        facecolor=style["color"],
        edgecolor=style["dark"],
        linewidth=0.8,
        zorder=5,
    )
    axis.axhline(0, color=INK, linewidth=0.9)
    axis.axvline(0, color=INK, linewidth=0.9)
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_xticks([-4, -2, 0, 2, 4])
    axis.set_yticks([-4, -2, 0, 2, 4])
    axis.tick_params(axis="x", labelbottom=show_x)
    if show_x:
        axis.set_xlabel("Target projection")
    axis.tick_params(labelleft=show_y)
    if show_y:
        axis.set_ylabel("Component selectivity")
    axis.set_aspect("equal", adjustable="box")
    if show_title:
        panel_title(
            marginal_x if marginal_x is not None else axis,
            letter,
            f"{MODEL_LABEL[model]} · {style['label']}",
            y=1.03 if marginal_x is not None else 1.08,
        )
    finish_axis(axis)


def build_generation_robustness_v3() -> None:
    figure = plt.figure(figsize=(6.25, 6.05))
    grid = figure.add_gridspec(2, 2, wspace=0.07, hspace=0.09)
    panels = (
        ("Llama-3.1-8B-Instruct", "lexical", "A"),
        ("Qwen3-4B", "lexical", "B"),
        ("Llama-3.1-8B-Instruct", "syntactic", "C"),
        ("Qwen3-4B", "syntactic", "D"),
    )
    for index, (model, concept, letter) in enumerate(panels):
        row, column = divmod(index, 2)
        joint = grid[row, column].subgridspec(
            2,
            2,
            height_ratios=[0.12, 1.0],
            width_ratios=[1.0, 0.12],
            hspace=0.01,
            wspace=0.01,
        )
        main_axis = figure.add_subplot(joint[1, 0])
        top_axis = figure.add_subplot(joint[0, 0], sharex=main_axis)
        right_axis = figure.add_subplot(joint[1, 1], sharey=main_axis)
        draw_prompt_constellation(
            main_axis,
            model,
            concept,
            letter,
            show_y=column == 0,
            show_x=row == 1,
            marginal_x=top_axis,
            marginal_y=right_axis,
            show_title=False,
        )
        main_axis.set_xlabel("")
        main_axis.set_ylabel("")
        main_axis.text(
            0.02,
            0.98,
            letter,
            transform=main_axis.transAxes,
            ha="left",
            va="top",
            fontsize=11.2,
            fontweight="bold",
            color=INK,
            zorder=8,
        )
    figure.text(
        0.335, 0.915, "Llama", ha="center", va="center",
        fontsize=12.2, color=INK,
    )
    figure.text(
        0.755, 0.915, "Qwen", ha="center", va="center",
        fontsize=12.2, color=INK,
    )
    figure.text(
        0.018, 0.642, "Lexical", ha="center", va="center",
        rotation=90, fontsize=11.8, color=NAVY_DARK,
    )
    figure.text(
        0.018, 0.296, "Syntactic", ha="center", va="center",
        rotation=90, fontsize=11.8, color=TEAL_DARK,
    )
    figure.text(
        0.073, 0.475, "Component selectivity", ha="center", va="center",
        rotation=90, fontsize=12.2, color=INK,
    )
    figure.text(
        0.545, 0.012, "Target projection", ha="center", va="center",
        fontsize=11.2, color=INK,
    )
    figure.legend(
        handles=[
            Line2D(
                [0], [0], marker="o", linestyle="none",
                color=GRAY_DARK, markerfacecolor=BG,
                label="behavior null",
            ),
            Line2D(
                [0], [0], marker="D", linestyle="none",
                color=INK, markerfacecolor=GRAY_LIGHT,
                label="clustered mean ± 95% CI",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=2,
        fontsize=11.8,
        columnspacing=1.25,
        handletextpad=0.40,
    )
    figure.subplots_adjust(left=0.115, right=0.985, top=0.875, bottom=0.095)
    save_figure(figure, "appendix_generation_robustness")


def main() -> None:
    setup_style()
    build_main_signature_v3()
    build_surface_transfer_v3()
    build_hidden_robustness_v3()
    build_generation_robustness_v3()


if __name__ == "__main__":
    main()
