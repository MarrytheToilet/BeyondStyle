#!/usr/bin/env python3
"""Build the three result-only figures used by the ACL paper.

The script reads only frozen, already-produced observations.  It does not
rerun models or select new layers, prompts, directions, or concepts.
Method flow diagrams are intentionally excluded: every active panel encodes
an observed estimate, interval, holdout result, or null comparison.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/beyondstyle-matplotlib")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.decodability_control_gap import (  # noqa: E402
    encoding_observations,
    summarize_gap,
)
from src.analysis.causal_pullback import normalized_curve_area  # noqa: E402


INK = "#353640"
MUTED = "#626571"
GRID = "#E5EBF1"
BG = "#FFFFFF"
WHITE = "#FFFFFF"
# GAP-inspired roots: saturated observations, pale uncertainty and regions.
BLUE = "#4F8CC9"
BLUE_DARK = "#356FA8"
BLUE_LIGHT = "#BBD4EC"
BLUE_OPEN = "#E8F1FA"
PINK = "#E8669A"
PINK_DARK = "#BE4777"
PINK_LIGHT = "#F3B8CF"
PINK_OPEN = "#FBEAF1"
NEUTRAL_OPEN = "#EEF2F5"
NEUTRAL = "#B8C1CA"

MODEL_SPECS = {
    "Llama-3.1-8B-Instruct": {
        "label": "Llama 3.1 8B",
        "input": 7,
        "max_layer": 32,
        "anchors": {"lexical": 32, "syntactic": 32},
        "manifest": (
            "outputs/en/05_linguistic/causal_pullback/"
            "Llama-3.1-8B-Instruct/mean_pool/frozen_axes/"
            "layer07/decomposition.json"
        ),
    },
    "Qwen3-4B": {
        "label": "Qwen3 4B",
        "input": 8,
        "max_layer": 36,
        "anchors": {"lexical": 36, "syntactic": 36},
        "manifest": (
            "outputs/en/05_linguistic/causal_pullback/"
            "Qwen3-4B/mean_pool/frozen_axes/layer08/decomposition.json"
        ),
    },
}
CONCEPTS = ("lexical", "syntactic")
ROWS = tuple(
    (model, concept)
    for model in MODEL_SPECS
    for concept in CONCEPTS
)
CONCEPT_STYLE = {
    "lexical": {
        "label": "Lexical",
        "color": BLUE,
        "dark": BLUE_DARK,
        "light": BLUE_LIGHT,
        "open": BLUE_OPEN,
    },
    "syntactic": {
        "label": "Syntactic",
        "color": PINK,
        "dark": PINK_DARK,
        "light": PINK_LIGHT,
        "open": PINK_OPEN,
    },
}
GENERATOR_LABELS = {
    "MiniMax-M2.5": "MiniMax",
    "deepseek-v4-flash": "DeepSeek",
    "glm-5.2": "GLM",
    "qwen3.6-flash": "Qwen",
}
DOMAIN_LABELS = {
    "encyclopedic": "Encycl.",
    "instructional": "Instr.",
    "journalistic": "News",
    "literary_narrative": "Narrative",
}
FEATURE_LABELS = {
    "function_word_rate": "Function words",
    "pronoun_rate": "Pronouns",
    "content_zipf_mean": "Content-word frequency",
    "mattr_50": "Lexical diversity (MATTR)",
    "institutional_lexicon_rate": "Institutional vocabulary",
    "evaluative_lexicon_rate": "Evaluative vocabulary",
    "dependency_length_mean": "Dependency length",
    "sentence_length_cv": "Sentence-length variation",
    "dependency_depth_mean": "Dependency depth",
    "coordination_density": "Coordination",
    "nominalization_rate": "Nominalization",
    "subordination_density": "Subordination",
    "transition_opening_rate": "Transition openings",
    "subject_continuity": "Subject continuity",
    "contrast_connective_rate": "Contrast connectives",
    "enumeration_marker_rate": "Enumeration markers",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("ACL27/figures"),
    )
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260727)
    parser.add_argument(
        "--figures",
        nargs="+",
        choices=["signature", "surface", "hidden", "audit", "chain"],
        default=["signature", "surface", "hidden", "audit"],
    )
    return parser.parse_args()


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
        "lines.linewidth": 1.35,
        "lines.markersize": 4.0,
        "patch.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def save_figure(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = ROOT / "ACL27" / "build" / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_dir / f"{stem}.pdf",
        bbox_inches="tight",
        facecolor=BG,
    )
    figure.savefig(
        preview_dir / f"{stem}.png",
        dpi=240,
        bbox_inches="tight",
        facecolor=BG,
    )
    plt.close(figure)


def panel_label(
    axis: plt.Axes,
    label: str,
    text: str,
    *,
    x: float = 0.0,
    y: float = 1.035,
    letter_size: float = 12.4,
    text_size: float = 12.0,
    text_offset: float = 20.0,
) -> None:
    axis.text(
        x,
        y,
        label,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=letter_size,
        fontweight="bold",
        color=INK,
    )
    axis.annotate(
        text,
        xy=(x, y),
        xycoords=axis.transAxes,
        xytext=(text_offset, 0),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=text_size,
        fontweight="normal",
        color=INK,
    )


def add_box(
    axis: plt.Axes,
    xy: tuple[float, float],
    size: tuple[float, float],
    text: str,
    *,
    face: str,
    edge: str,
    fontsize: float = 15.0,
    linewidth: float = 1.7,
) -> None:
    x, y = xy
    width, height = size
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.025",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        transform=axis.transAxes,
        clip_on=False,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height / 2,
        text,
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        linespacing=1.25,
    )


def add_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = MUTED,
    linewidth: float = 1.8,
    linestyle: str = "-",
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=15,
        linewidth=linewidth,
        linestyle=linestyle,
        color=color,
        transform=axis.transAxes,
        clip_on=False,
        shrinkA=0,
        shrinkB=0,
    )
    axis.add_patch(arrow)


def standardized_interval(row: pd.Series) -> tuple[float, float]:
    effect = float(row["paired_d"])
    mean = float(row["paired_difference_mean"])
    if abs(effect) < 1e-12 or abs(mean) < 1e-12:
        return effect, effect
    sd = abs(mean / effect)
    return (
        float(row["paired_difference_ci_low"]) / sd,
        float(row["paired_difference_ci_high"]) / sd,
    )


def load_construct_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    feature_stats = pd.read_csv(
        ROOT / "outputs/en/05_linguistic/features/feature_statistics.csv"
    )
    holdout = pd.read_csv(
        ROOT
        / "outputs/en/05_linguistic/features/"
        "environment_holdout_statistics.csv"
    )
    holdout = holdout[
        holdout["environment_type"].isin(["generator", "domain"])
    ].copy()
    return feature_stats, holdout


def draw_construct_pipeline(axis: plt.Axes) -> None:
    axis.set_axis_off()
    panel_label(
        axis,
        "A",
        "Identification chain · measurement is not control",
        x=0.006,
        y=0.94,
    )
    add_box(
        axis,
        (0.010, 0.25),
        (0.135, 0.48),
        "Two corpora\nPLAN  .925\nPREFIX  .727",
        face=BLUE_OPEN,
        edge=BLUE_DARK,
        fontsize=10.9,
    )
    add_box(
        axis,
        (0.175, 0.25),
        (0.135, 0.48),
        "Frozen features\nLex 10 · Syn 12\nDisc 13",
        face=BLUE_OPEN,
        edge=BLUE_DARK,
        fontsize=10.9,
    )
    add_box(
        axis,
        (0.340, 0.25),
        (0.135, 0.48),
        "Stability gate\nGenerator · domain\nPlan-free transfer",
        face=NEUTRAL_OPEN,
        edge=NEUTRAL,
        fontsize=10.9,
    )
    add_box(
        axis,
        (0.505, 0.25),
        (0.130, 0.48),
        "Hidden readout\nHeld-out generator\nTwo source sides",
        face=PINK_OPEN,
        edge=PINK_DARK,
        fontsize=10.9,
    )
    add_box(
        axis,
        (0.665, 0.25),
        (0.130, 0.48),
        "Canonical pullback\nNullspace + Jacobian\nPole fixed upstream",
        face=PINK_OPEN,
        edge=PINK_DARK,
        fontsize=10.9,
    )
    add_box(
        axis,
        (0.835, 0.54),
        (0.155, 0.25),
        "FIXED TOKENS\nsigned response\n+ matched-pipeline null",
        face=WHITE,
        edge=BLUE_DARK,
        fontsize=10.5,
    )
    add_box(
        axis,
        (0.835, 0.08),
        (0.155, 0.25),
        "FREE GENERATION\nfull response matrix\n+ seeds + guardrails",
        face=WHITE,
        edge=PINK_DARK,
        fontsize=10.5,
    )
    for left, right, color in (
        (0.145, 0.175, BLUE_DARK),
        (0.310, 0.340, BLUE_DARK),
        (0.475, 0.505, PINK_DARK),
        (0.635, 0.665, PINK_DARK),
    ):
        add_arrow(axis, (left, 0.49), (right, 0.49), color=color)
    add_arrow(axis, (0.795, 0.49), (0.835, 0.665), color=BLUE_DARK)
    add_arrow(axis, (0.795, 0.49), (0.835, 0.205), color=PINK_DARK)
    axis.text(
        0.902,
        0.43,
        "separate estimands",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=9.8,
        color=MUTED,
    )


def draw_feature_forest(
    axis: plt.Axes,
    feature_stats: pd.DataFrame,
) -> None:
    selected_names = list(FEATURE_LABELS)
    data = feature_stats[
        feature_stats["feature"].isin(selected_names)
    ].copy()
    data["order"] = data["feature"].map(
        {name: idx for idx, name in enumerate(selected_names)}
    )
    data = data.sort_values("order", ascending=False).reset_index(drop=True)

    for y, row in data.iterrows():
        effect = float(row["paired_d"])
        low, high = standardized_interval(row)
        color = PINK_DARK if effect > 0 else BLUE_DARK
        fill = PINK if effect > 0 else BLUE
        axis.plot(
            [0, effect],
            [y, y],
            color=fill,
            linewidth=4.0,
            alpha=0.5,
            zorder=1,
        )
        axis.plot(
            [low, high],
            [y, y],
            color=color,
            linewidth=1.7,
            zorder=3,
        )
        axis.scatter(
            [effect],
            [y],
            s=68,
            facecolor=fill,
            edgecolor=color,
            linewidth=1.5,
            zorder=4,
        )
        axis.text(
            effect + (0.045 if effect >= 0 else -0.045),
            y,
            f"{effect:+.2f}",
            ha="left" if effect >= 0 else "right",
            va="center",
            fontsize=12.0,
            color=color,
        )

    group_ranges: dict[str, list[int]] = {}
    for y, row in data.iterrows():
        group_ranges.setdefault(str(row["group"]), []).append(y)
    for group, ys in group_ranges.items():
        y0, y1 = min(ys) - 0.5, max(ys) + 0.5
        axis.add_patch(
            Rectangle(
                (-1.43, y0),
                2.30,
                y1 - y0,
                facecolor=(
                    BLUE_OPEN if group == "lexical"
                    else PINK_OPEN if group == "syntactic"
                    else NEUTRAL_OPEN
                ),
                edgecolor="none",
                alpha=0.45,
                zorder=-2,
            )
        )
        axis.text(
            0.79,
            (y0 + y1) / 2,
            group.upper(),
            ha="right",
            va="center",
            fontsize=10.5,
            color=MUTED,
        )

    axis.axvline(0, color=INK, linewidth=1.0, zorder=0)
    axis.set_xlim(-1.43, 0.86)
    axis.set_ylim(-0.7, len(data) - 0.3)
    axis.set_yticks(
        np.arange(len(data)),
        [FEATURE_LABELS[name] for name in data["feature"]],
    )
    axis.tick_params(axis="y", length=0, pad=8)
    axis.set_xlabel("Paired standardized shift  ·  Human  ←  0  →  Machine")
    axis.grid(axis="x")
    axis.grid(axis="y", visible=False)
    panel_label(axis, "B", "What changes when content is conditioned")
    axis.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color=BLUE,
                markerfacecolor=BLUE,
                markeredgecolor=BLUE_DARK,
                linewidth=3,
                label="Human-oriented",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color=PINK,
                markerfacecolor=PINK,
                markeredgecolor=PINK_DARK,
                linewidth=3,
                label="Machine-oriented",
            ),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, -0.21),
        ncol=2,
        frameon=False,
        columnspacing=1.7,
        handlelength=2.0,
    )


def draw_stability_heatmaps(
    generator_axis: plt.Axes,
    domain_axis: plt.Axes,
    holdout: pd.DataFrame,
) -> None:
    groups = ["lexical", "syntactic", "discourse"]
    specs = [
        (
            generator_axis,
            "generator",
            list(GENERATOR_LABELS),
            GENERATOR_LABELS,
            LinearSegmentedColormap.from_list(
                "generator_blue",
                [WHITE, BLUE_OPEN, BLUE, BLUE_DARK],
            ),
            PINK_DARK,
            "C",
            "Direction stability across generators",
        ),
        (
            domain_axis,
            "domain",
            list(DOMAIN_LABELS),
            DOMAIN_LABELS,
            LinearSegmentedColormap.from_list(
                "domain_pink",
                [WHITE, PINK_OPEN, PINK, PINK_DARK],
            ),
            BLUE_DARK,
            "D",
            "Direction stability across domains",
        ),
    ]
    for (
        axis,
        env_type,
        envs,
        labels,
        cmap,
        outline_color,
        letter,
        header,
    ) in specs:
        matrix = np.full((len(groups), len(envs)), np.nan)
        subset = holdout[holdout["environment_type"] == env_type]
        for row, group in enumerate(groups):
            for col, environment in enumerate(envs):
                hit = subset[
                    (subset["group"] == group)
                    & (subset["heldout_environment"] == environment)
                ]
                if len(hit) == 1:
                    matrix[row, col] = float(
                        hit.iloc[0]["heldout_direction_cosine"]
                    )
        axis.imshow(
            matrix,
            vmin=0,
            vmax=1,
            aspect="auto",
            cmap=cmap,
            interpolation="nearest",
        )
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = matrix[row, col]
                text_color = INK if value < 0.70 else WHITE
                axis.text(
                    col,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=12.0,
                    color=text_color,
                )
        for row in (0, 1):
            axis.add_patch(
                Rectangle(
                    (-0.48, row - 0.48),
                    len(envs) - 0.04,
                    0.96,
                    fill=False,
                    edgecolor=outline_color,
                    linewidth=1.8,
                )
            )
        axis.set_xticks(
            np.arange(len(envs)),
            [labels[env] for env in envs],
            rotation=28,
            ha="right",
        )
        axis.set_yticks(
            np.arange(len(groups)),
            [name.capitalize() for name in groups],
        )
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        panel_label(axis, letter, header)
        minima = np.nanmin(matrix, axis=1)
        axis.text(
            1.0,
            -0.32,
            "row minima  " + " · ".join(f"{v:.2f}" for v in minima),
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontsize=11.5,
            color=MUTED,
        )


def build_construct_atlas(output_dir: Path) -> None:
    feature_stats, holdout = load_construct_data()
    figure = plt.figure(figsize=(14.4, 9.1))
    grid = figure.add_gridspec(
        2,
        2,
        height_ratios=[1.55, 4.8],
        width_ratios=[1.15, 1.0],
        hspace=0.27,
        wspace=0.24,
    )
    pipeline_axis = figure.add_subplot(grid[0, :])
    forest_axis = figure.add_subplot(grid[1, 0])
    right = grid[1, 1].subgridspec(2, 1, hspace=0.57)
    generator_axis = figure.add_subplot(right[0, 0])
    domain_axis = figure.add_subplot(right[1, 0])

    draw_construct_pipeline(pipeline_axis)
    draw_feature_forest(forest_axis, feature_stats)
    draw_stability_heatmaps(generator_axis, domain_axis, holdout)
    figure.subplots_adjust(
        left=0.13,
        right=0.985,
        top=0.985,
        bottom=0.13,
    )
    save_figure(figure, output_dir, "machine_signature_atlas")


def load_geometry() -> dict[tuple[str, str], dict[str, float]]:
    values: dict[tuple[str, str], dict[str, float]] = {}
    for model, spec in MODEL_SPECS.items():
        manifest = json.loads((ROOT / str(spec["manifest"])).read_text())
        for concept in CONCEPTS:
            diagnostics = (
                manifest["selected_components"][concept]["fit_metadata"][
                    "diagnostics"
                ]
            )
            values[(model, concept)] = {
                "human_ai": float(
                    diagnostics["human_ai_pattern_cosine"]
                ),
                "encoding": float(
                    diagnostics["cosine_vs_encoding_direction"]
                ),
            }
    return values


def hidden_audit_root(model: str) -> Path:
    base = (
        ROOT
        / "outputs/en/05_linguistic/causal_pullback"
        / model
        / "mean_pool"
    )
    confirmatory = base / "confirmatory_hidden_audit"
    if (
        confirmatory
        / "matched_nulls/matched_null_transport_summary.csv"
    ).exists():
        return confirmatory
    return base


def matched_null_count() -> int:
    path = (
        hidden_audit_root(next(iter(MODEL_SPECS)))
        / "matched_nulls/matched_null_axis_statistics.csv"
    )
    frame = pd.read_csv(path)
    for column in ("null_replicate", "null_id", "replicate"):
        if column in frame:
            return int(frame[column].nunique())
    first_layer = frame["layer"].iloc[0]
    first_concept = frame["concept"].iloc[0]
    return int(
        len(
            frame[
                (frame["layer"] == first_layer)
                & (frame["concept"] == first_concept)
            ]
        )
    )


def transport_path(model: str) -> Path:
    return hidden_audit_root(model) / "layerwise_transport/per_prompt_transport.csv"


def transport_frame(model: str, concept: str) -> pd.DataFrame:
    frame = pd.read_csv(transport_path(model))
    return frame[
        (frame["readout"] == concept)
        & (
            (frame["intervention"] == concept)
            | frame["intervention"].str.startswith("random_")
        )
    ].copy()


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    seed: int,
    repetitions: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(
        0,
        len(values),
        size=(repetitions, len(values)),
    )
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(
        np.quantile(means, 0.975)
    )


def draw_trace_flow(axis: plt.Axes) -> None:
    axis.set_axis_off()
    panel_label(
        axis,
        "A",
        "Learned and null targets traverse the same readout→pullback pipeline",
    )
    boxes = [
        (0.015, 0.30, 0.16, "Surface target\n$s_g(x)$", BLUE_OPEN, BLUE_DARK),
        (0.225, 0.30, 0.16, "Cross-source readout\n$v_g^\\top h$", BLUE_OPEN, BLUE_DARK),
        (0.435, 0.30, 0.16, "Canonical pullback\n$w_g=P J^\\top v_g$", PINK_OPEN, PINK_DARK),
        (0.645, 0.30, 0.16, "Fixed-token trace\n$|v_g^{j\\top}\\Delta h^j|$", PINK_OPEN, PINK_DARK),
        (0.855, 0.30, 0.13, "Learned−null\nadvantage", WHITE, INK),
    ]
    for x, y, width, text, face, edge in boxes:
        add_box(
            axis,
            (x, y),
            (width, 0.43),
            text,
            face=face,
            edge=edge,
            fontsize=14.0,
        )
    for left, right, color in [
        (0.175, 0.225, BLUE_DARK),
        (0.385, 0.435, BLUE_DARK),
        (0.595, 0.645, PINK_DARK),
        (0.805, 0.855, INK),
    ]:
        add_arrow(axis, (left, 0.515), (right, 0.515), color=color)
    axis.text(
        0.305,
        0.09,
        "Matched null: permute target correspondence within "
        "corpus×generator strata, then refit every downstream step",
        transform=axis.transAxes,
        ha="left",
        va="center",
        color=MUTED,
        fontsize=12.5,
    )
    add_arrow(
        axis,
        (0.26, 0.17),
        (0.43, 0.30),
        color=NEUTRAL,
        linestyle="--",
        linewidth=1.5,
    )


def draw_transport_panel(
    axis: plt.Axes,
    model: str,
    concept: str,
    geometry: dict[tuple[str, str], dict[str, float]],
    *,
    seed: int,
    repetitions: int,
    letter: str,
) -> None:
    spec = MODEL_SPECS[model]
    style = CONCEPT_STYLE[concept]
    null_style = CONCEPT_STYLE[
        "syntactic" if concept == "lexical" else "lexical"
    ]
    root = hidden_audit_root(model)
    learned_table = pd.read_csv(
        root / "layerwise_transport/learned_transport.csv"
    )
    learned_table = learned_table[
        learned_table["concept"] == concept
    ].sort_values("layer")
    matched = pd.read_csv(
        root / "matched_nulls/matched_null_transport_summary.csv"
    )
    matched = matched[matched["concept"] == concept].sort_values("layer")
    null_axes = pd.read_csv(
        root / "matched_nulls/matched_null_axis_statistics.csv"
    )
    null_axes = null_axes[null_axes["concept"] == concept]
    layers = matched["layer"].astype(int).tolist()
    if learned_table["layer"].astype(int).tolist() != layers:
        raise ValueError(f"Layer mismatch for {model} {concept}")

    x = np.asarray(layers, dtype=float) / float(spec["max_layer"])
    learned_array = learned_table["mean"].to_numpy(float)
    low_array = learned_table["ci_low"].to_numpy(float)
    high_array = learned_table["ci_high"].to_numpy(float)
    null_mean = matched[
        "matched_null_mean_transport_capacity"
    ].to_numpy(float)
    null_quantiles = (
        null_axes.groupby("layer")["mean_transport_capacity"]
        .quantile([0.05, 0.95])
        .unstack()
        .reindex(layers)
    )
    null_low = null_quantiles[0.05].to_numpy(float)
    null_high = null_quantiles[0.95].to_numpy(float)

    axis.fill_between(
        x,
        null_low,
        null_high,
        color=null_style["open"],
        edgecolor=null_style["light"],
        linewidth=1.0,
        alpha=0.95,
        label=f"Central 90% of {matched_null_count()} matched nulls",
        zorder=0,
    )
    for x_value, layer in zip(x, layers):
        null_values = np.sort(null_axes[
            null_axes["layer"] == layer
        ]["mean_transport_capacity"].to_numpy(float))
        jitter = np.linspace(-0.009, 0.009, len(null_values))
        axis.scatter(
            x_value + jitter,
            null_values,
            s=11,
            facecolor=WHITE,
            edgecolor=null_style["dark"],
            linewidth=0.55,
            alpha=0.55,
            zorder=1,
        )
    axis.plot(
        x,
        null_mean,
        color=null_style["dark"],
        linewidth=1.7,
        linestyle="--",
        zorder=1,
    )
    axis.fill_between(
        x,
        low_array,
        high_array,
        color=style["light"],
        alpha=0.75,
        linewidth=0,
        zorder=2,
    )
    axis.plot(
        x,
        learned_array,
        color=style["dark"],
        linewidth=3.2,
        marker="o",
        markersize=7.5,
        markerfacecolor=style["color"],
        markeredgecolor=style["dark"],
        markeredgewidth=1.2,
        zorder=3,
    )
    anchor = int(spec["anchors"][concept])
    anchor_x = anchor / float(spec["max_layer"])
    anchor_row = layers.index(anchor)
    axis.scatter(
        [anchor_x],
        [learned_array[anchor_row]],
        s=180,
        facecolor=WHITE,
        edgecolor=INK,
        linewidth=2.0,
        zorder=4,
    )
    axis.scatter(
        [anchor_x],
        [learned_array[anchor_row]],
        s=62,
        facecolor=style["color"],
        edgecolor=style["dark"],
        linewidth=1.2,
        zorder=5,
    )
    axis.axvline(
        anchor_x,
        color=INK,
        linewidth=1.0,
        linestyle=":",
        zorder=0,
    )
    axis.text(
        anchor_x,
        axis.get_ylim()[1] if axis.get_ylim()[1] else 1.0,
        "",
    )
    axis.set_xlim(max(0, x.min() - 0.03), 1.025)
    range_values = np.concatenate([
        learned_array,
        low_array,
        high_array,
        null_low,
        null_high,
    ])
    padding = max(0.08, np.ptp(range_values) * 0.18)
    axis.set_ylim(0, max(0.0, float(range_values.max())) + padding)
    axis.set_xticks(
        x,
        [f"L{layer}" for layer in layers],
        rotation=0,
    )
    axis.set_ylabel("Transport capacity")
    axis.grid(axis="y")
    axis.grid(axis="x", visible=False)
    panel_label(
        axis,
        letter,
        f"{spec['label']} · {style['label']}",
    )
    geom = geometry[(model, concept)]
    injection = int(spec["input"])
    global_frame = pd.read_csv(
        root / "matched_nulls/global_curve_statistics.csv"
    )
    global_row = global_frame[
        global_frame["concept"] == concept
    ].iloc[0]
    axis.text(
        0.025,
        0.965,
        (
            f"H↔M pullback  {geom['human_ai']:.2f}\n"
            f"pullback↔encode  {geom['encoding']:.2f}\n"
            f"curve AUC  {global_row['learned_curve_statistic']:.3f} "
            f"/ null {global_row['null_mean_curve_statistic']:.3f}\n"
            f"finite-null p  {global_row['finite_null_p_value']:.4f}"
        ),
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=11.8,
        color=INK,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": WHITE,
            "edgecolor": style["light"],
            "linewidth": 1.2,
            "alpha": 0.96,
        },
    )
    at_right_edge = anchor == max(layers)
    early_anchor = anchor <= layers[len(layers) // 2]
    axis.annotate(
        "readout anchor",
        xy=(anchor_x, learned_array[anchor_row]),
        xytext=(
            (-4, 22) if at_right_edge
            else (26, 18) if early_anchor
            else (0, 22)
        ),
        textcoords="offset points",
        ha=(
            "right" if at_right_edge
            else "left" if early_anchor
            else "center"
        ),
        va="bottom",
        fontsize=10.5,
        color=INK,
        arrowprops={
            "arrowstyle": "-",
            "color": INK,
            "linewidth": 0.9,
        },
    )
    injection_x = injection / float(spec["max_layer"])
    injection_row = layers.index(injection)
    axis.scatter(
        [injection_x],
        [learned_array[injection_row]],
        s=100,
        marker="D",
        facecolor=WHITE,
        edgecolor=INK,
        linewidth=1.4,
        zorder=6,
    )


def build_hidden_trace(
    output_dir: Path,
    *,
    seed: int,
    repetitions: int,
) -> None:
    geometry = load_geometry()
    figure = plt.figure(figsize=(14.4, 8.4))
    grid = figure.add_gridspec(
        3,
        2,
        height_ratios=[1.2, 2.15, 2.15],
        hspace=0.52,
        wspace=0.25,
    )
    flow_axis = figure.add_subplot(grid[0, :])
    draw_trace_flow(flow_axis)
    letters = ("B", "C", "D", "E")
    for index, ((model, concept), letter) in enumerate(zip(ROWS, letters)):
        axis = figure.add_subplot(grid[1 + index // 2, index % 2])
        draw_transport_panel(
            axis,
            model,
            concept,
            geometry,
            seed=seed + index * 1009,
            repetitions=repetitions,
            letter=letter,
        )
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=INK,
                marker="D",
                markerfacecolor=WHITE,
                markeredgecolor=INK,
                markeredgewidth=1.4,
                linewidth=0,
                markersize=8,
                label="Direct injection",
            ),
            Line2D(
                [0],
                [0],
                color=INK,
                marker="o",
                markerfacecolor=WHITE,
                markeredgecolor=INK,
                markeredgewidth=1.8,
                linewidth=0,
                markersize=10,
                label="Readout anchor",
            ),
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor=PINK_OPEN,
                edgecolor=BLUE_DARK,
                label=(
                    f"Central 90% of {matched_null_count()} matched-null "
                    "pipelines"
                ),
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.002),
        ncol=3,
        frameon=False,
        columnspacing=2.0,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.98,
        bottom=0.10,
    )
    save_figure(figure, output_dir, "hidden_causal_trace")


def violin_body(
    axis: plt.Axes,
    values: np.ndarray,
    *,
    color: str,
    edge: str,
    position: float = 0.0,
    width: float = 0.78,
    bandwidth: float = 0.25,
) -> None:
    violin = axis.violinplot(
        [np.asarray(values, dtype=float)],
        positions=[position],
        vert=False,
        widths=width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method=bandwidth,
    )
    for body in violin["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(edge)
        body.set_linewidth(1.1)
        body.set_alpha(0.52)


def row_summary(
    summary: pd.DataFrame,
    model: str,
    concept: str,
    stage: str,
) -> pd.Series:
    hit = summary[
        (summary["model"] == model)
        & (summary["concept"] == concept)
        & (summary["stage"] == stage)
    ]
    if len(hit) != 1:
        raise ValueError((model, concept, stage, len(hit)))
    return hit.iloc[0]

def confirmatory_analysis_root(model: str) -> Path:
    return (
        ROOT
        / "outputs/en/05_linguistic/causal_pullback"
        / model
        / "mean_pool/confirmatory_analysis"
    )


def seed_robustness_root(model: str) -> Path:
    return (
        ROOT
        / "outputs/en/05_linguistic/causal_pullback"
        / model
        / "mean_pool/confirmatory_seed_robustness"
    )


def matched_null_summary(model: str) -> pd.DataFrame:
    path = (
        hidden_audit_root(model)
        / "matched_nulls/matched_null_transport_summary.csv"
    )
    return pd.read_csv(path)


def draw_compact_readout(
    axis: plt.Axes,
    model: str,
    observations: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    y_positions = {"lexical": 1.0, "syntactic": 0.0}
    offsets = np.linspace(-0.13, 0.13, 4)
    for concept in CONCEPTS:
        style = CONCEPT_STYLE[concept]
        y = y_positions[concept]
        rows = observations[
            (observations["model"] == model)
            & (observations["concept"] == concept)
            & (observations["stage"] == "encoding")
        ]
        stats = row_summary(summary, model, concept, "encoding")
        for offset, (_, row) in zip(offsets, rows.iterrows()):
            axis.scatter(
                row["value"],
                y + offset,
                s=42,
                facecolor=style["light"],
                edgecolor=style["dark"],
                linewidth=1.1,
                zorder=3,
            )
        axis.plot(
            [stats["ci_low"], stats["ci_high"]],
            [y + 0.25, y + 0.25],
            color=style["dark"],
            linewidth=2.8,
            zorder=4,
        )
        axis.scatter(
            stats["mean"],
            y + 0.25,
            s=68,
            facecolor=WHITE,
            edgecolor=style["dark"],
            linewidth=1.8,
            zorder=5,
        )
        axis.text(
            0.99,
            y + 0.25,
            f"{stats['mean']:.2f}",
            transform=axis.get_yaxis_transform(),
            ha="right",
            va="center",
            color=style["dark"],
            fontsize=11.5,
        )
    axis.set_xlim(0.30, 0.58)
    axis.set_ylim(-0.42, 1.48)
    axis.set_yticks([0, 1])
    axis.set_yticklabels(["Syntactic", "Lexical"])
    axis.axvline(0, color=INK, linestyle="--", linewidth=0.9)
    axis.grid(axis="x", alpha=0.65)
    axis.grid(axis="y", visible=False)


def draw_matched_layer_fingerprint(
    axis: plt.Axes,
    model: str,
    *,
    x_limits: tuple[float, float],
) -> None:
    frame = matched_null_summary(model)
    y_positions = {"lexical": 1.0, "syntactic": 0.0}
    for concept in CONCEPTS:
        style = CONCEPT_STYLE[concept]
        anchor = int(MODEL_SPECS[model]["anchors"][concept])
        injection = int(MODEL_SPECS[model]["input"])
        rows = frame[
            (frame["concept"] == concept)
            & (~frame["layer"].isin([injection, anchor]))
        ].sort_values("layer")
        y = y_positions[concept]
        offsets = np.linspace(-0.19, 0.19, len(rows))
        significant = rows["learned_minus_null_ci_low"] > 0
        for offset, (_, row), supported in zip(
            offsets,
            rows.iterrows(),
            significant,
        ):
            axis.plot(
                [
                    row["learned_minus_null_ci_low"],
                    row["learned_minus_null_ci_high"],
                ],
                [y + offset, y + offset],
                color=style["dark"],
                linewidth=1.15,
                alpha=0.72,
                zorder=2,
            )
            axis.scatter(
                row["learned_minus_null_mean"],
                y + offset,
                s=58 if supported else 45,
                facecolor=style["color"] if supported else WHITE,
                edgecolor=style["dark"],
                linewidth=1.25,
                zorder=3,
            )
        axis.text(
            0.99,
            y + 0.28,
            (
                f"CI {int(significant.sum())}/{len(rows)} · "
                f"all-{matched_null_count()} "
                f"{int((rows['matched_null_axis_exceed_rate'] == 0).sum())}"
                f"/{len(rows)}"
            ),
            transform=axis.get_yaxis_transform(),
            ha="right",
            va="center",
            color=style["dark"],
            fontsize=11.5,
        )
    axis.axvline(0, color=INK, linestyle="--", linewidth=1.0)
    axis.set_xlim(*x_limits)
    axis.set_ylim(-0.42, 1.48)
    axis.set_yticks([0, 1])
    axis.set_yticklabels([])
    axis.grid(axis="x", alpha=0.65)
    axis.grid(axis="y", visible=False)


def draw_response_matrix(
    axis: plt.Axes,
    model: str,
) -> None:
    root = confirmatory_analysis_root(model)
    frame = pd.read_csv(root / "causal_group_statistics.csv")
    matrix = np.empty((2, 3), dtype=float)
    supported = np.zeros((2, 3), dtype=bool)
    groups = ("lexical", "syntactic", "discourse")
    for row_number, component in enumerate(CONCEPTS):
        for column_number, group in enumerate(groups):
            row = frame[
                (frame["component"] == component)
                & (frame["group"] == group)
            ].iloc[0]
            matrix[row_number, column_number] = float(
                row["signature_projection_mean"]
            )
            supported[row_number, column_number] = (
                row["signature_projection_ci_low"] > 0
                or row["signature_projection_ci_high"] < 0
            )
    # One fixed scale makes the two model-row matrices comparable.
    limit = 0.40
    color_map = LinearSegmentedColormap.from_list(
        "response_matrix",
        [BLUE_DARK, BLUE_LIGHT, WHITE, PINK_LIGHT, PINK_DARK],
        N=256,
    )
    axis.imshow(
        matrix,
        aspect="auto",
        cmap=color_map,
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    for row_number in range(2):
        for column_number in range(3):
            value = matrix[row_number, column_number]
            text_color = (
                WHITE if abs(value) > 0.62 * limit else INK
            )
            axis.text(
                column_number,
                row_number,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=12.0,
                color=text_color,
            )
            if supported[row_number, column_number]:
                axis.add_patch(
                    Rectangle(
                        (
                            column_number - 0.47,
                            row_number - 0.46,
                        ),
                        0.94,
                        0.92,
                        fill=False,
                        edgecolor=(
                            PINK_DARK if value < 0 else BLUE_DARK
                        ),
                        linewidth=1.8,
                    )
                )
    axis.set_xticks(range(3))
    axis.set_xticklabels(["Lexical", "Syntactic", "Discourse"])
    axis.set_yticks(range(2))
    axis.set_yticklabels(["Lexical axis", "Syntactic axis"])
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)


def selectivity_prompt_values(
    root: Path,
    concept: str,
) -> np.ndarray:
    projections = pd.read_csv(
        root / "prompt_signature_projections.csv"
    )
    view = projections[
        projections["component"] == concept
    ].pivot(
        index="prompt_id",
        columns="group",
        values="signature_projection",
    )
    off_target = [
        group
        for group in ("lexical", "syntactic", "discourse")
        if group != concept
    ]
    return (
        view[concept] - view[off_target].mean(axis=1)
    ).to_numpy(dtype=float)


def seed_robustness_values(
    model: str,
    concept: str,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    root = seed_robustness_root(model)
    values = pd.read_csv(root / "prompt_seed_estimands.csv")
    view = values[
        (values["component"] == concept)
        & (values["estimand"] == "component_selectivity")
    ]
    pivot = view.pivot(
        index="prompt_id",
        columns="sample_seed_index",
        values="value",
    ).sort_index(axis=1)
    summary = pd.read_csv(root / "seed_robustness_summary.csv")
    row = summary[
        (summary["component"] == concept)
        & (summary["estimand"] == "component_selectivity")
    ].iloc[0]
    return (
        pivot.mean(axis=1).to_numpy(dtype=float),
        pivot.mean(axis=0).to_numpy(dtype=float),
        row,
    )


def draw_seed_robustness(
    axis: plt.Axes,
    model: str,
    *,
    limit: float,
) -> None:
    y_positions = {"lexical": 1.0, "syntactic": 0.0}
    for concept in CONCEPTS:
        style = CONCEPT_STYLE[concept]
        y = y_positions[concept]
        prompt_means, seed_means, stats = seed_robustness_values(
            model,
            concept,
        )
        violin_body(
            axis,
            np.clip(prompt_means, -limit, limit),
            color=style["color"],
            edge=style["dark"],
            position=y,
            width=0.52,
            bandwidth=0.24,
        )
        axis.plot(
            [stats["ci_low"], stats["ci_high"]],
            [y + 0.28, y + 0.28],
            color=style["dark"],
            linewidth=2.7,
            zorder=4,
        )
        axis.scatter(
            stats["mean"],
            y + 0.28,
            marker="D",
            s=68,
            facecolor=style["color"],
            edgecolor=style["dark"],
            linewidth=1.8,
            zorder=5,
        )
        axis.scatter(
            seed_means,
            y - 0.29 + np.array([-0.045, 0.0, 0.045]),
            s=43,
            facecolor=WHITE,
            edgecolor=style["dark"],
            linewidth=1.25,
            zorder=5,
        )
        axis.text(
            0.99,
            y + 0.32,
            (
                f"{stats['mean']:+.2f} "
                f"[{stats['ci_low']:+.2f}, {stats['ci_high']:+.2f}]"
            ),
            transform=axis.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=10.8,
            color=style["dark"],
        )
    axis.axvline(0, color=INK, linestyle="--", linewidth=0.95)
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-0.48, 1.55)
    axis.set_yticks([0, 1])
    axis.set_yticklabels(["Syntactic", "Lexical"])
    axis.grid(axis="x", alpha=0.65)
    axis.grid(axis="y", visible=False)


def build_response_audit(
    output_dir: Path,
    *,
    seed: int,
    repetitions: int,
) -> None:
    validation = pd.read_csv(
        ROOT
        / "outputs/en/05_linguistic/concept_patterns/"
        "pattern_validation.csv"
    )
    observations = encoding_observations(
        validation,
        {
            model: int(spec["input"])
            for model, spec in MODEL_SPECS.items()
        },
    )
    summary = summarize_gap(
        observations,
        bootstrap=repetitions,
        permutations=20000,
        seed=20260726,
    )
    matched_intervals = []
    for model in MODEL_SPECS:
        frame = matched_null_summary(model)
        for concept in CONCEPTS:
            anchor = int(MODEL_SPECS[model]["anchors"][concept])
            injection = int(MODEL_SPECS[model]["input"])
            matched_intervals.append(
                frame[
                    (frame["concept"] == concept)
                    & (~frame["layer"].isin([injection, anchor]))
                ][
                    [
                        "learned_minus_null_ci_low",
                        "learned_minus_null_ci_high",
                    ]
                ].to_numpy(dtype=float)
            )
    matched_intervals_array = np.concatenate(matched_intervals, axis=0)
    matched_low = min(
        0.0,
        float(matched_intervals_array[:, 0].min()),
    )
    matched_high = max(
        0.0,
        float(matched_intervals_array[:, 1].max()),
    )
    matched_span = max(0.1, matched_high - matched_low)
    matched_x_limits = (
        matched_low - 0.07 * matched_span,
        matched_high + 0.07 * matched_span,
    )
    seed_prompt_means = np.concatenate([
        seed_robustness_values(model, concept)[0]
        for model in MODEL_SPECS
        for concept in CONCEPTS
    ])
    seed_means = np.concatenate([
        seed_robustness_values(model, concept)[1]
        for model in MODEL_SPECS
        for concept in CONCEPTS
    ])
    selectivity_limit = max(
        0.5,
        float(np.quantile(np.abs(seed_prompt_means), 0.985)),
        1.15 * float(np.max(np.abs(seed_means))),
    )
    figure = plt.figure(figsize=(15.8, 8.0))
    grid = figure.add_gridspec(
        2,
        4,
        width_ratios=[0.95, 1.18, 1.36, 1.46],
        hspace=0.42,
        wspace=0.28,
    )
    headers = [
        (
            "A  Cross-generator\nreadout",
            "four held-out generators",
        ),
        (
            "B  Matched-null\ntrace",
            "unsigned |τ| · learned − fitted null",
        ),
        (
            "C  Generation\nresponse matrix",
            "signed surface effect · +1 minus −1",
        ),
        (
            "D  Decode-seed\nrobustness",
            "signed selectivity · prompt means + 3 seeds",
        ),
    ]
    all_axes: list[list[plt.Axes]] = []
    for row_number, model in enumerate(MODEL_SPECS):
        axes = [
            figure.add_subplot(grid[row_number, column])
            for column in range(4)
        ]
        all_axes.append(axes)
        draw_compact_readout(
            axes[0],
            model,
            observations,
            summary,
        )
        draw_matched_layer_fingerprint(
            axes[1],
            model,
            x_limits=matched_x_limits,
        )
        draw_response_matrix(axes[2], model)
        draw_seed_robustness(
            axes[3],
            model,
            limit=selectivity_limit,
        )
        axes[0].text(
            -0.52,
            0.50,
            MODEL_SPECS[model]["label"],
            transform=axes[0].transAxes,
            ha="center",
            va="center",
            rotation=90,
            fontsize=14.0,
            color=INK,
        )
        if row_number == 0:
            for column, axis in enumerate(axes):
                title, subtitle = headers[column]
                axis.text(
                    0,
                    1.27,
                    title,
                    transform=axis.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=13.2,
                    color=INK,
                    linespacing=1.04,
                )
                axis.text(
                    0,
                    1.11,
                    subtitle,
                    transform=axis.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=10.2,
                    color=MUTED,
                )
        if row_number == 0:
            for axis in (axes[0], axes[1], axes[3]):
                axis.tick_params(axis="x", labelbottom=False)
        axes[0].set_xlabel(
            "Held-out\nPearson r" if row_number == 1 else ""
        )
        axes[1].set_xlabel(
            "Unsigned fixed-token\n|τ| advantage"
            if row_number == 1
            else ""
        )
        axes[3].set_xlabel(
            "Component selectivity" if row_number == 1 else ""
        )
    figure.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=BLUE,
                markeredgecolor=BLUE_DARK,
                markersize=8,
                label="Lexical",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=PINK,
                markeredgecolor=PINK_DARK,
                markersize=8,
                label="Syntactic",
            ),
            Line2D(
                [0],
                [0],
                marker="D",
                color="none",
                markerfacecolor=PINK,
                markeredgecolor=PINK_DARK,
                markersize=7,
                label="Prompt-mean effect ± CI",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="none",
                markerfacecolor=WHITE,
                markeredgecolor=BLUE_DARK,
                markersize=7,
                label="Decode-seed mean",
            ),
            Line2D(
                [0],
                [0],
                color=MUTED,
                linestyle="--",
                linewidth=1.0,
                label="Zero reference",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.55, 0.008),
        ncol=5,
        frameon=False,
        columnspacing=1.45,
    )
    figure.subplots_adjust(
        left=0.095,
        right=0.985,
        top=0.88,
        bottom=0.22,
    )
    save_figure(figure, output_dir, "representation_realization_audit")


def draw_chain_compact(axis: plt.Axes) -> None:
    """A shallow overview strip; the body panels carry the evidence."""
    axis.set_axis_off()
    panel_label(axis, "A", "Identification chain", x=0.002, y=0.88)
    nodes = [
        ("Two interfaces", "plan + prefix", BLUE_OPEN, BLUE_DARK),
        ("Frozen measures", "10 lex · 12 syn · 13 disc", BLUE_OPEN, BLUE_DARK),
        ("Stability gate", "generator · domain", WHITE, NEUTRAL),
        ("Hidden readout", "held-out generator", PINK_OPEN, PINK_DARK),
        ("Pullback", "nullspace · Jacobian", PINK_OPEN, PINK_DARK),
    ]
    left = 0.105
    width = 0.125
    gap = 0.020
    for index, (title, subtitle, face, edge) in enumerate(nodes):
        x = left + index * (width + gap)
        add_box(
            axis,
            (x, 0.20),
            (width, 0.52),
            f"{index + 1}  {title}\n{subtitle}",
            face=face,
            edge=edge,
            fontsize=10.2,
        )
        if index:
            add_arrow(
                axis,
                (x - gap + 0.003, 0.46),
                (x - 0.004, 0.46),
                color=BLUE_DARK if index < 3 else PINK_DARK,
                linewidth=1.4,
            )
    branch_x = left + len(nodes) * (width + gap)
    add_box(
        axis,
        (branch_x, 0.46),
        (0.125, 0.30),
        "Fixed tokens\ntrace + fitted null",
        face=BLUE_OPEN,
        edge=BLUE_DARK,
        fontsize=9.6,
    )
    add_box(
        axis,
        (branch_x, 0.06),
        (0.125, 0.30),
        "Free generation\nmatrix + seeds",
        face=PINK_OPEN,
        edge=PINK_DARK,
        fontsize=9.6,
    )
    add_arrow(
        axis,
        (branch_x - gap + 0.003, 0.46),
        (branch_x - 0.004, 0.61),
        color=BLUE_DARK,
        linewidth=1.3,
    )
    add_arrow(
        axis,
        (branch_x - gap + 0.003, 0.46),
        (branch_x - 0.004, 0.21),
        color=PINK_DARK,
        linewidth=1.3,
    )


def draw_feature_forest_dense(
    axis: plt.Axes,
    feature_stats: pd.DataFrame,
) -> None:
    selected = [
        "function_word_rate",
        "pronoun_rate",
        "content_zipf_mean",
        "mattr_50",
        "dependency_length_mean",
        "sentence_length_cv",
        "dependency_depth_mean",
        "coordination_density",
        "nominalization_rate",
        "transition_opening_rate",
        "subject_continuity",
        "contrast_connective_rate",
    ]
    data = (
        feature_stats[feature_stats["feature"].isin(selected)]
        .assign(order=lambda frame: frame["feature"].map(
            {feature: index for index, feature in enumerate(selected)}
        ))
        .sort_values("order", ascending=False)
        .reset_index(drop=True)
    )
    group_face = {
        "lexical": BLUE_OPEN,
        "syntactic": PINK_OPEN,
        "discourse": NEUTRAL_OPEN,
    }
    for group, group_rows in data.groupby("group"):
        indices = group_rows.index.to_numpy()
        axis.axhspan(
            indices.min() - 0.48,
            indices.max() + 0.48,
            color=group_face[str(group)],
            alpha=0.62,
            zorder=-3,
        )
        axis.text(
            0.985,
            float(indices.mean()),
            str(group).upper(),
            transform=axis.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=9.7,
            color=MUTED,
        )
    for y, row in data.iterrows():
        effect = float(row["paired_d"])
        low, high = standardized_interval(row)
        positive = effect >= 0
        fill = PINK if positive else BLUE
        dark = PINK_DARK if positive else BLUE_DARK
        axis.plot([0, effect], [y, y], color=fill, linewidth=4.3, alpha=0.52)
        axis.plot([low, high], [y, y], color=dark, linewidth=1.6, zorder=3)
        axis.scatter(
            effect,
            y,
            s=62,
            facecolor=fill,
            edgecolor=dark,
            linewidth=1.4,
            zorder=4,
        )
        axis.text(
            effect + (0.04 if positive else -0.04),
            y,
            f"{effect:+.2f}",
            ha="left" if positive else "right",
            va="center",
            fontsize=10.8,
            color=dark,
        )
    axis.axvline(0, color=INK, linewidth=1.0)
    axis.set_xlim(-1.42, 0.83)
    axis.set_ylim(-0.58, len(data) - 0.42)
    axis.set_yticks(
        np.arange(len(data)),
        [FEATURE_LABELS[name] for name in data["feature"]],
    )
    axis.tick_params(axis="y", length=0, pad=7)
    axis.set_xlabel("Paired standardized shift   Human  ←  0  →  Machine")
    axis.grid(axis="x")
    axis.grid(axis="y", visible=False)
    panel_label(axis, "A", "Surface-feature fingerprint", y=1.015)


def draw_stability_dense(
    axis: plt.Axes,
    holdout: pd.DataFrame,
    *,
    environment_type: str,
    letter: str,
) -> None:
    if environment_type == "generator":
        environments = list(GENERATOR_LABELS)
        labels = GENERATOR_LABELS
        cmap = LinearSegmentedColormap.from_list(
            "dense_generator", [WHITE, BLUE_OPEN, BLUE, BLUE_DARK]
        )
        subtitle = "Generator holdout"
    else:
        environments = list(DOMAIN_LABELS)
        labels = DOMAIN_LABELS
        cmap = LinearSegmentedColormap.from_list(
            "dense_domain", [WHITE, PINK_OPEN, PINK, PINK_DARK]
        )
        subtitle = "Domain holdout"
    groups = ("lexical", "syntactic", "discourse")
    matrix = np.full((3, 4), np.nan)
    view = holdout[holdout["environment_type"] == environment_type]
    for row, group in enumerate(groups):
        for column, environment in enumerate(environments):
            hit = view[
                (view["group"] == group)
                & (view["heldout_environment"] == environment)
            ]
            matrix[row, column] = float(hit.iloc[0]["heldout_direction_cosine"])
    axis.imshow(matrix, vmin=0, vmax=1, cmap=cmap, aspect="auto")
    for row in range(3):
        for column in range(4):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=10.6,
                color=WHITE if value > 0.72 else INK,
            )
    outline = PINK_DARK if environment_type == "generator" else BLUE_DARK
    for row in (0, 1):
        axis.add_patch(
            Rectangle(
                (-0.48, row - 0.47),
                3.96,
                0.94,
                fill=False,
                edgecolor=outline,
                linewidth=1.5,
            )
        )
    axis.set_xticks(
        range(4),
        [labels[environment] for environment in environments],
        rotation=22,
        ha="right",
    )
    axis.set_yticks(range(3), ["Lexical", "Syntactic", "Discourse"])
    axis.tick_params(length=0, labelsize=10.5)
    for spine in axis.spines.values():
        spine.set_visible(False)
    panel_label(axis, letter, subtitle, y=1.06)
    minima = np.nanmin(matrix, axis=1)
    axis.text(
        1.0,
        -0.30,
        "min  " + "  ·  ".join(f"{value:.2f}" for value in minima),
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9.8,
        color=MUTED,
    )


def draw_interface_transfer(axis: plt.Axes) -> None:
    transfer = pd.read_csv(
        ROOT
        / "outputs/en/05_linguistic/prefix_only_control/transfer/"
        "surface_signature_transfer.csv"
    ).set_index("concept")
    axis.set_xlim(0.65, 1.0)
    axis.set_ylim(-0.75, 2.35)
    rows = [
        ("Lexical", "lexical", 1.7),
        ("Syntactic", "syntactic", 0.9),
    ]
    for label, concept, y in rows:
        style = CONCEPT_STYLE[concept]
        cosine = float(transfer.loc[concept, "plan_control_shift_cosine"])
        positive = float(
            transfer.loc[concept, "control_projection_positive_rate"]
        )
        axis.plot([positive, cosine], [y, y], color=style["light"], linewidth=5)
        axis.scatter(
            positive,
            y,
            s=62,
            marker="s",
            facecolor=WHITE,
            edgecolor=style["dark"],
            linewidth=1.5,
            zorder=3,
        )
        axis.scatter(
            cosine,
            y,
            s=68,
            marker="o",
            facecolor=style["color"],
            edgecolor=style["dark"],
            linewidth=1.5,
            zorder=3,
        )
        axis.text(
            0.657,
            y,
            label,
            ha="left",
            va="center",
            fontsize=10.8,
            color=INK,
        )
    axis.plot([0.727, 0.925], [0.05, 0.05], color=NEUTRAL, linewidth=4.5)
    axis.scatter(
        0.727, 0.05, s=60, marker="s", facecolor=WHITE,
        edgecolor=INK, linewidth=1.4, zorder=3
    )
    axis.scatter(
        0.925, 0.05, s=66, marker="o", facecolor=NEUTRAL_OPEN,
        edgecolor=INK, linewidth=1.4, zorder=3
    )
    axis.text(
        0.657, 0.05, "Semantic", ha="left", va="center",
        fontsize=10.8, color=INK
    )
    axis.text(
        0.727, -0.28, ".727", ha="center", va="top",
        fontsize=9.7, color=MUTED
    )
    axis.text(
        0.925, -0.28, ".925", ha="center", va="top",
        fontsize=9.7, color=MUTED
    )
    axis.set_xticks([0.7, 0.8, 0.9, 1.0])
    axis.set_yticks([])
    axis.grid(axis="x")
    axis.grid(axis="y", visible=False)
    panel_label(axis, "D", "Plan-free interface transfer", y=1.06)
    axis.legend(
        handles=[
            Line2D(
                [0], [0], marker="o", color="none", markerfacecolor=BLUE,
                markeredgecolor=BLUE_DARK, label="Direction cosine"
            ),
            Line2D(
                [0], [0], marker="s", color="none", markerfacecolor=WHITE,
                markeredgecolor=BLUE_DARK, label="Positive-pair rate"
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.62),
        ncol=2,
        frameon=False,
        fontsize=10.0,
        columnspacing=1.0,
        handletextpad=0.35,
    )


def build_construct_atlas_dense(output_dir: Path) -> None:
    feature_stats, holdout = load_construct_data()
    figure = plt.figure(figsize=(14.4, 6.45))
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=[1.48, 1.0],
        wspace=0.25,
    )
    forest_axis = figure.add_subplot(grid[0, 0])
    right = grid[0, 1].subgridspec(
        3, 1, height_ratios=[1.0, 1.0, 0.94], hspace=0.78
    )
    draw_feature_forest_dense(forest_axis, feature_stats)
    draw_stability_dense(
        figure.add_subplot(right[0]), holdout,
        environment_type="generator", letter="B"
    )
    draw_stability_dense(
        figure.add_subplot(right[1]), holdout,
        environment_type="domain", letter="C"
    )
    draw_interface_transfer(figure.add_subplot(right[2]))
    figure.subplots_adjust(
        left=0.125, right=0.985, top=0.945, bottom=0.145
    )
    save_figure(figure, output_dir, "machine_signature_atlas")


def trace_rows() -> list[dict[str, Any]]:
    geometry = load_geometry()
    output: list[dict[str, Any]] = []
    for model, concept in ROWS:
        spec = MODEL_SPECS[model]
        summary = matched_null_summary(model)
        layers = summary[
            (summary["concept"] == concept)
            & (~summary["layer"].isin([
                int(spec["input"]), int(spec["anchors"][concept])
            ]))
        ].sort_values("layer")
        global_row = pd.read_csv(
            hidden_audit_root(model)
            / "matched_nulls/global_curve_statistics.csv"
        )
        global_row = global_row[global_row["concept"] == concept].iloc[0]
        output.append({
            "model": model,
            "concept": concept,
            "layers": layers,
            "global": global_row,
            "geometry": geometry[(model, concept)],
        })
    return output


def draw_trace_matrix(axis: plt.Axes, rows: list[dict[str, Any]]) -> None:
    matrix = np.stack([
        row["layers"]["learned_minus_null_mean"].to_numpy(float)
        for row in rows
    ])
    max_abs = max(0.12, float(np.abs(matrix).max()))
    cmap = LinearSegmentedColormap.from_list(
        "trace_difference", [BLUE_DARK, BLUE_OPEN, WHITE, PINK_OPEN, PINK_DARK]
    )
    axis.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        vmin=-max_abs,
        vmax=max_abs,
        interpolation="nearest",
    )
    for row_index, row in enumerate(rows):
        for column, (_, value_row) in enumerate(row["layers"].iterrows()):
            value = float(value_row["learned_minus_null_mean"])
            supported = float(value_row["learned_minus_null_ci_low"]) > 0
            axis.text(
                column,
                row_index - 0.09,
                f"L{int(value_row['layer'])}",
                ha="center",
                va="center",
                fontsize=9.4,
                color=MUTED if abs(value) < 0.68 * max_abs else WHITE,
            )
            axis.text(
                column,
                row_index + 0.16,
                f"{value:+.2f}",
                ha="center",
                va="center",
                fontsize=10.6,
                color=INK if abs(value) < 0.68 * max_abs else WHITE,
            )
            if supported:
                axis.add_patch(
                    Rectangle(
                        (column - 0.47, row_index - 0.45),
                        0.94, 0.90, fill=False,
                        edgecolor=CONCEPT_STYLE[row["concept"]]["dark"],
                        linewidth=1.5,
                    )
                )
    axis.set_xticks(range(5), ["1st", "2nd", "3rd", "4th", "5th"])
    axis.set_yticks(
        range(4),
        [
            f"{MODEL_SPECS[row['model']]['label']} · "
            f"{CONCEPT_STYLE[row['concept']]['label']}"
            for row in rows
        ],
    )
    axis.tick_params(length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)
    panel_label(axis, "A", "Layerwise capacity beyond null")
    axis.text(
        1.0,
        -0.24,
        "outlined cell: prompt-bootstrap interval > 0",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=10.2,
        color=MUTED,
    )


def draw_auc_dumbbell(axis: plt.Axes, rows: list[dict[str, Any]]) -> None:
    y_values = np.arange(4)[::-1]
    for y, row in zip(y_values, rows):
        global_row = row["global"]
        style = CONCEPT_STYLE[row["concept"]]
        null_min = float(global_row["null_min_curve_statistic"])
        null_max = float(global_row["null_max_curve_statistic"])
        null_mean = float(global_row["null_mean_curve_statistic"])
        learned = float(global_row["learned_curve_statistic"])
        axis.plot(
            [null_min, null_max], [y, y],
            color=NEUTRAL, linewidth=7, alpha=0.48, zorder=1
        )
        axis.plot(
            [null_mean, learned], [y, y],
            color=style["light"], linewidth=3.2, zorder=2
        )
        axis.scatter(
            null_mean, y, s=64, facecolor=WHITE, edgecolor=MUTED,
            linewidth=1.4, zorder=3
        )
        axis.scatter(
            learned, y, s=74, facecolor=style["color"],
            edgecolor=style["dark"], linewidth=1.5, zorder=4
        )
        axis.text(
            0.405,
            y,
            f"p={float(global_row['finite_null_p_value']):.3f}",
            ha="right",
            va="center",
            fontsize=10.2,
            color=style["dark"],
        )
    axis.set_xlim(0.10, 0.415)
    axis.set_ylim(-0.6, 3.6)
    axis.set_yticks(
        y_values,
        [
            f"{MODEL_SPECS[row['model']]['label'].split()[0]} · "
            f"{CONCEPT_STYLE[row['concept']]['label'][:3]}"
            for row in rows
        ],
    )
    axis.set_xlabel("Depth-normalized curve area")
    axis.grid(axis="x")
    axis.grid(axis="y", visible=False)
    panel_label(axis, "B", "Matched-null curve evidence")
    axis.legend(
        handles=[
            Line2D(
                [0], [0], marker="o", color=NEUTRAL,
                markerfacecolor=WHITE, markeredgecolor=MUTED,
                linewidth=5, label="Null range + mean"
            ),
            Line2D(
                [0], [0], marker="o", color="none",
                markerfacecolor=PINK, markeredgecolor=PINK_DARK,
                label="Learned"
            ),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, -0.35),
        frameon=False,
        ncol=2,
        fontsize=10.2,
        columnspacing=1.0,
    )


def draw_geometry_dumbbell(axis: plt.Axes, rows: list[dict[str, Any]]) -> None:
    y_values = np.arange(4)[::-1]
    for y, row in zip(y_values, rows):
        geometry = row["geometry"]
        style = CONCEPT_STYLE[row["concept"]]
        encoding = float(geometry["encoding"])
        agreement = float(geometry["human_ai"])
        axis.plot(
            [encoding, agreement], [y, y],
            color=style["light"], linewidth=5.0, zorder=1
        )
        axis.scatter(
            encoding, y, s=70, marker="s", facecolor=WHITE,
            edgecolor=style["dark"], linewidth=1.5, zorder=3
        )
        axis.scatter(
            agreement, y, s=76, marker="o", facecolor=style["color"],
            edgecolor=style["dark"], linewidth=1.5, zorder=3
        )
        axis.text(
            encoding, y - 0.27, f"{encoding:.2f}",
            ha="center", va="top", fontsize=9.4, color=MUTED
        )
        axis.text(
            agreement, y - 0.27, f"{agreement:.2f}",
            ha="center", va="top", fontsize=9.4, color=MUTED
        )
    axis.set_xlim(0, 1.0)
    axis.set_ylim(-0.6, 3.6)
    axis.set_yticks(y_values, [""] * 4)
    axis.set_xlabel("Cosine similarity")
    axis.grid(axis="x")
    axis.grid(axis="y", visible=False)
    panel_label(axis, "C", "Geometric separation")
    axis.legend(
        handles=[
            Line2D(
                [0], [0], marker="s", color="none", markerfacecolor=WHITE,
                markeredgecolor=BLUE_DARK, label="Pullback / encoding"
            ),
            Line2D(
                [0], [0], marker="o", color="none", markerfacecolor=BLUE,
                markeredgecolor=BLUE_DARK, label="Human / machine pullbacks"
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.35),
        frameon=False,
        ncol=1,
        fontsize=10.2,
    )


def build_hidden_trace_dense(output_dir: Path) -> None:
    rows = trace_rows()
    figure = plt.figure(figsize=(14.4, 4.25))
    grid = figure.add_gridspec(
        1,
        3,
        width_ratios=[1.72, 1.0, 1.0],
        wspace=0.34,
    )
    draw_trace_matrix(figure.add_subplot(grid[0]), rows)
    draw_auc_dumbbell(figure.add_subplot(grid[1]), rows)
    draw_geometry_dumbbell(figure.add_subplot(grid[2]), rows)
    figure.subplots_adjust(
        left=0.13, right=0.985, top=0.90, bottom=0.25
    )
    save_figure(figure, output_dir, "hidden_causal_trace")


def seed_estimand_values(
    model: str,
    concept: str,
    estimand: str,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    root = seed_robustness_root(model)
    values = pd.read_csv(root / "prompt_seed_estimands.csv")
    view = values[
        (values["component"] == concept)
        & (values["estimand"] == estimand)
    ]
    pivot = view.pivot(
        index="prompt_id",
        columns="sample_seed_index",
        values="value",
    ).sort_index(axis=1)
    summary = pd.read_csv(root / "seed_robustness_summary.csv")
    row = summary[
        (summary["component"] == concept)
        & (summary["estimand"] == estimand)
    ].iloc[0]
    return (
        pivot.mean(axis=1).to_numpy(float),
        pivot.mean(axis=0).to_numpy(float),
        row,
    )


def draw_seed_estimands_dense(
    axis: plt.Axes,
    model: str,
    *,
    limit: float,
) -> None:
    specs = [
        ("lexical", "target_projection", 3, "Lex · target"),
        ("lexical", "component_selectivity", 2, "Lex · selective"),
        ("syntactic", "target_projection", 1, "Syn · target"),
        ("syntactic", "component_selectivity", 0, "Syn · selective"),
    ]
    for concept, estimand, y, _ in specs:
        style = CONCEPT_STYLE[concept]
        prompts, seeds, stats = seed_estimand_values(model, concept, estimand)
        is_target = estimand == "target_projection"
        violin_body(
            axis,
            np.clip(prompts, -limit, limit),
            color=style["color"] if is_target else style["light"],
            edge=style["dark"],
            position=y,
            width=0.52,
            bandwidth=0.24,
        )
        axis.plot(
            [float(stats["ci_low"]), float(stats["ci_high"])],
            [y + 0.23, y + 0.23],
            color=style["dark"],
            linewidth=2.2,
            zorder=4,
        )
        axis.scatter(
            float(stats["mean"]),
            y + 0.23,
            s=58,
            marker="D" if is_target else "o",
            facecolor=style["color"] if is_target else WHITE,
            edgecolor=style["dark"],
            linewidth=1.5,
            zorder=5,
        )
        axis.scatter(
            seeds,
            y - 0.24 + np.array([-0.035, 0.0, 0.035]),
            s=34,
            facecolor=WHITE,
            edgecolor=style["dark"],
            linewidth=1.1,
            zorder=5,
        )
        axis.text(
            limit * 0.98,
            y + 0.23,
            f"{float(stats['mean']):+.2f}",
            ha="right",
            va="center",
            fontsize=9.8,
            color=style["dark"],
        )
    axis.axvline(0, color=INK, linestyle="--", linewidth=0.9)
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-0.55, 3.55)
    axis.set_yticks([3, 2, 1, 0], [spec[3] for spec in specs])
    axis.grid(axis="x")
    axis.grid(axis="y", visible=False)


def build_response_audit_dense(
    output_dir: Path,
    *,
    repetitions: int,
) -> None:
    validation = pd.read_csv(
        ROOT
        / "outputs/en/05_linguistic/concept_patterns/"
        "pattern_validation.csv"
    )
    observations = encoding_observations(
        validation,
        {
            model: int(spec["input"])
            for model, spec in MODEL_SPECS.items()
        },
    )
    summary = summarize_gap(
        observations,
        bootstrap=repetitions,
        permutations=20000,
        seed=20260726,
    )
    all_prompt_values = []
    all_seed_values = []
    for model in MODEL_SPECS:
        for concept in CONCEPTS:
            for estimand in ("target_projection", "component_selectivity"):
                prompts, seeds, _ = seed_estimand_values(
                    model, concept, estimand
                )
                all_prompt_values.append(prompts)
                all_seed_values.append(seeds)
    prompt_values = np.concatenate(all_prompt_values)
    seed_values = np.concatenate(all_seed_values)
    limit = max(
        0.75,
        float(np.quantile(np.abs(prompt_values), 0.985)),
        1.18 * float(np.max(np.abs(seed_values))),
    )
    figure = plt.figure(figsize=(14.4, 6.3))
    grid = figure.add_gridspec(
        2,
        3,
        width_ratios=[1.0, 1.08, 1.62],
        hspace=0.30,
        wspace=0.34,
    )
    for row_number, model in enumerate(MODEL_SPECS):
        readout_axis = figure.add_subplot(grid[row_number, 0])
        response_axis = figure.add_subplot(grid[row_number, 1])
        seed_axis = figure.add_subplot(grid[row_number, 2])
        draw_compact_readout(readout_axis, model, observations, summary)
        draw_response_matrix(response_axis, model)
        draw_seed_estimands_dense(seed_axis, model, limit=limit)
        readout_axis.text(
            -0.52,
            0.5,
            MODEL_SPECS[model]["label"],
            transform=readout_axis.transAxes,
            ha="center",
            va="center",
            rotation=90,
            fontsize=13.0,
            color=INK,
        )
        if row_number == 0:
            panel_label(
                readout_axis, "A", "Cross-generator readout",
                y=1.22
            )
            readout_axis.text(
                0, 1.08, "four held-out generators + mean interval",
                transform=readout_axis.transAxes,
                ha="left", va="bottom", fontsize=10.2, color=MUTED
            )
            panel_label(
                response_axis, "B", "Signed response matrix",
                y=1.22
            )
            response_axis.text(
                0, 1.08, "+1 minus −1 generated surface score",
                transform=response_axis.transAxes,
                ha="left", va="bottom", fontsize=10.2, color=MUTED
            )
            panel_label(
                seed_axis, "C", "Three-seed generation audit",
                y=1.22
            )
            seed_axis.text(
                0, 1.08, "prompt distributions · clustered CI · seed means",
                transform=seed_axis.transAxes,
                ha="left", va="bottom", fontsize=10.2, color=MUTED
            )
            readout_axis.tick_params(axis="x", labelbottom=False)
            seed_axis.tick_params(axis="x", labelbottom=False)
        else:
            readout_axis.set_xlabel("Held-out-generator Pearson r")
            seed_axis.set_xlabel("Signed generated-surface effect")
    figure.legend(
        handles=[
            Line2D(
                [0], [0], marker="D", color="none",
                markerfacecolor=BLUE, markeredgecolor=BLUE_DARK,
                label="Target mean ± CI"
            ),
            Line2D(
                [0], [0], marker="o", color="none",
                markerfacecolor=WHITE, markeredgecolor=PINK_DARK,
                label="Selectivity mean ± CI"
            ),
            Line2D(
                [0], [0], marker="o", color="none",
                markerfacecolor=WHITE, markeredgecolor=BLUE_DARK,
                label="Decode-seed mean"
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.62, 0.008),
        ncol=3,
        frameon=False,
        fontsize=11.0,
        columnspacing=1.4,
    )
    figure.subplots_adjust(
        left=0.10, right=0.985, top=0.88, bottom=0.19
    )
    save_figure(figure, output_dir, "representation_realization_audit")


def result_panel_title(
    axis: plt.Axes,
    letter: str,
    title: str,
    subtitle: str = "",
    *,
    y: float = 1.12,
) -> None:
    """Place a short title and optional decoding key outside the data region."""
    axis.text(
        0.0,
        y,
        letter,
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=13.0,
        fontweight="bold",
        color=INK,
    )
    axis.annotate(
        title,
        xy=(0.0, y),
        xycoords=axis.transAxes,
        xytext=(21, 0),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=12.6,
        fontweight="normal",
        color=INK,
    )
    # Decoding keys and statistical definitions live in the caption.  A
    # single title tier reads more formally at ACL two-column scale.


def clean_result_axis(
    axis: plt.Axes,
    *,
    grid_axis: str = "x",
) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.8, alpha=0.75)
    if grid_axis in {"x", "y"}:
        other = "y" if grid_axis == "x" else "x"
        axis.grid(axis=other, visible=False)


def result_covariance_ellipse(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    edge: str,
    level: float,
    linewidth: float,
    alpha: float,
) -> None:
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
            facecolor="none",
            edgecolor=edge,
            linewidth=linewidth,
            alpha=alpha,
            zorder=2,
        )
    )


def draw_signature_facet(
    axis: plt.Axes,
    feature_stats: pd.DataFrame,
    *,
    group: str,
    features: list[str],
    show_x: bool,
) -> None:
    style = {
        "lexical": CONCEPT_STYLE["lexical"],
        "syntactic": CONCEPT_STYLE["syntactic"],
        "discourse": {
            "color": NEUTRAL,
            "dark": MUTED,
            "light": NEUTRAL_OPEN,
        },
    }[group]
    ordered = (
        feature_stats[feature_stats["feature"].isin(features)]
        .assign(
            order=lambda frame: frame["feature"].map(
                {feature: index for index, feature in enumerate(features)}
            )
        )
        .sort_values("order")
    )
    y_values = np.arange(len(ordered))[::-1]
    for y, (_, row) in zip(y_values, ordered.iterrows()):
        effect = float(row["paired_d"])
        low, high = standardized_interval(row)
        axis.axhspan(
            y - 0.40,
            y + 0.40,
            color=style.get("open", NEUTRAL_OPEN),
            alpha=0.58,
            linewidth=0,
            zorder=0,
        )
        axis.plot(
            [low, high],
            [y, y],
            color=style["dark"],
            linewidth=1.65,
            zorder=2,
        )
        axis.plot(
            [low, low, np.nan, high, high],
            [y - 0.10, y + 0.10, np.nan, y - 0.10, y + 0.10],
            color=style["dark"],
            linewidth=1.0,
            zorder=2,
        )
        axis.scatter(
            effect,
            y,
            s=50,
            facecolor=style["color"],
            edgecolor=style["dark"],
            linewidth=1.15,
            zorder=3,
        )
        axis.text(
            effect + (0.045 if effect >= 0 else -0.045),
            y,
            f"{effect:+.2f}",
            ha="left" if effect >= 0 else "right",
            va="center",
            fontsize=9.3,
            color=style["dark"],
        )
    axis.axvline(0, color=INK, linewidth=1.0, zorder=0)
    axis.set_xlim(-1.40, 0.82)
    axis.set_ylim(-0.55, len(ordered) - 0.45)
    axis.set_yticks(
        y_values,
        [FEATURE_LABELS[feature] for feature in ordered["feature"]],
    )
    axis.tick_params(axis="y", length=0, pad=5, labelsize=10.1)
    axis.tick_params(axis="x", labelbottom=show_x)
    if show_x:
        axis.set_xlabel("Paired standardized shift   human  |  machine")
    clean_result_axis(axis)


def holdout_group_style(group: str) -> dict[str, str]:
    if group == "lexical":
        return CONCEPT_STYLE["lexical"]
    if group == "syntactic":
        return CONCEPT_STYLE["syntactic"]
    return {
        "color": NEUTRAL,
        "dark": MUTED,
        "light": NEUTRAL_OPEN,
    }


def draw_holdout_ranges(
    axis: plt.Axes,
    holdout: pd.DataFrame,
    *,
    environment_type: str,
    letter: str,
    title: str,
) -> None:
    groups = ("lexical", "syntactic", "discourse")
    y_values = np.array([2.0, 1.0, 0.0])
    view = holdout[holdout["environment_type"] == environment_type]
    tick_labels = []
    for y, group in zip(y_values, groups):
        values = (
            view[view["group"] == group]
            .sort_values("heldout_environment")[
                "heldout_direction_cosine"
            ]
            .to_numpy(float)
        )
        style = holdout_group_style(group)
        axis.axhspan(
            y - 0.35,
            y + 0.35,
            color=style.get("open", NEUTRAL_OPEN),
            alpha=0.52,
            linewidth=0,
            zorder=0,
        )
        offsets = np.linspace(-0.14, 0.14, len(values))
        axis.scatter(
            values,
            np.full(len(values), y) + offsets,
            s=38,
            facecolor=WHITE,
            edgecolor=style["dark"],
            linewidth=1.10,
            zorder=3,
        )
        axis.scatter(
            np.median(values),
            y,
            marker="D",
            s=52,
            facecolor=style["color"],
            edgecolor=style["dark"],
            linewidth=1.15,
            zorder=4,
        )
        tick_labels.append(
            f"{group.capitalize()}  ·  min {values.min():.2f}"
        )
    axis.set_xlim(0.0, 1.03)
    axis.set_ylim(-0.50, 2.50)
    axis.set_yticks(y_values, tick_labels)
    axis.set_xticks([0.0, 0.25, 0.50, 0.75, 1.0])
    axis.tick_params(axis="y", length=0, pad=5, labelsize=9.8)
    result_panel_title(
        axis,
        letter,
        title,
        "four environments · direction-cosine range",
        y=1.17,
    )
    clean_result_axis(axis)


def draw_interface_rail_new(axis: plt.Axes) -> None:
    transfer = pd.read_csv(
        ROOT
        / "outputs/en/05_linguistic/prefix_only_control/transfer/"
        "surface_signature_transfer.csv"
    ).set_index("concept")
    rows = [
        (
            "Lexical",
            "lexical",
            float(transfer.loc["lexical", "plan_control_shift_cosine"]),
            float(
                transfer.loc[
                    "lexical", "control_projection_positive_rate"
                ]
            ),
            "dir",
            "pos",
        ),
        (
            "Syntactic",
            "syntactic",
            float(transfer.loc["syntactic", "plan_control_shift_cosine"]),
            float(
                transfer.loc[
                    "syntactic", "control_projection_positive_rate"
                ]
            ),
            "dir",
            "pos",
        ),
        ("Semantic", "discourse", 0.925, 0.727, "plan", "prefix"),
    ]
    for y, (label, concept, upper, lower, upper_name, lower_name) in zip(
        [2.0, 1.0, 0.0],
        rows,
    ):
        style = holdout_group_style(concept)
        axis.plot(
            [lower, upper],
            [y - 0.10, y + 0.10],
            color=NEUTRAL,
            linewidth=1.1,
            zorder=1,
        )
        axis.scatter(
            upper,
            y + 0.11,
            s=54,
            marker="o",
            facecolor=style["color"],
            edgecolor=style["dark"],
            linewidth=1.35,
            zorder=3,
        )
        axis.scatter(
            lower,
            y - 0.11,
            s=48,
            marker="s",
            facecolor=WHITE,
            edgecolor=style["dark"],
            linewidth=1.35,
            zorder=3,
        )
        axis.text(
            upper + 0.008,
            y + 0.11,
            f"{upper_name} {upper:.2f}",
            ha="left",
            va="center",
            fontsize=8.8,
            color=style["dark"],
        )
        axis.text(
            lower - 0.008,
            y - 0.11,
            f"{lower_name} {lower:.2f}",
            ha="right",
            va="center",
            fontsize=8.8,
            color=style["dark"],
        )
    axis.set_xlim(0.66, 1.02)
    axis.set_ylim(-0.48, 2.48)
    axis.set_yticks([2, 1, 0], ["Lexical", "Syntactic", "Semantic"])
    axis.set_xticks([0.7, 0.8, 0.9, 1.0])
    axis.tick_params(axis="y", length=0, pad=5, labelsize=9.8)
    axis.set_xlabel("Agreement / rate")
    result_panel_title(
        axis,
        "D",
        "Plan-free transfer",
        "paired rails expose interface sensitivity",
        y=1.17,
    )
    clean_result_axis(axis)


def build_construct_atlas_result_only(output_dir: Path) -> None:
    feature_stats, holdout = load_construct_data()
    figure = plt.figure(figsize=(13.2, 5.35))
    outer = figure.add_gridspec(
        1,
        2,
        width_ratios=[1.42, 1.0],
        wspace=0.28,
    )
    left = outer[0, 0].subgridspec(3, 1, hspace=0.08)
    right = outer[0, 1].subgridspec(3, 1, hspace=0.54)
    feature_sets = [
        (
            "lexical",
            [
                "function_word_rate",
                "pronoun_rate",
                "content_zipf_mean",
                "mattr_50",
            ],
        ),
        (
            "syntactic",
            [
                "dependency_length_mean",
                "sentence_length_cv",
                "dependency_depth_mean",
                "coordination_density",
            ],
        ),
        (
            "discourse",
            [
                "transition_opening_rate",
                "subject_continuity",
                "contrast_connective_rate",
                "enumeration_marker_rate",
            ],
        ),
    ]
    feature_axes = [
        figure.add_subplot(left[index])
        for index in range(3)
    ]
    for index, (axis, (group, features)) in enumerate(
        zip(feature_axes, feature_sets)
    ):
        draw_signature_facet(
            axis,
            feature_stats,
            group=group,
            features=features,
            show_x=index == 2,
        )
    result_panel_title(
        feature_axes[0],
        "A",
        "Surface signature",
        "twelve frozen measurements · paired 95% intervals",
        y=1.17,
    )
    draw_holdout_ranges(
        figure.add_subplot(right[0]),
        holdout,
        environment_type="generator",
        letter="B",
        title="Generator stability",
    )
    draw_holdout_ranges(
        figure.add_subplot(right[1]),
        holdout,
        environment_type="domain",
        letter="C",
        title="Domain stability",
    )
    draw_interface_rail_new(figure.add_subplot(right[2]))
    figure.subplots_adjust(
        left=0.13,
        right=0.982,
        top=0.91,
        bottom=0.15,
    )
    save_figure(figure, output_dir, "machine_signature_atlas")


def build_surface_result_summary(output_dir: Path) -> None:
    """Compact single-column summary of surface effect and transfer evidence."""
    feature_stats, holdout = load_construct_data()
    transfer = pd.read_csv(
        ROOT
        / "outputs/en/05_linguistic/prefix_only_control/transfer/"
        "surface_signature_transfer.csv"
    ).set_index("concept")
    with (
        ROOT / "outputs/en/05_linguistic/features/group_statistics.json"
    ).open(encoding="utf-8") as handle:
        group_stats = json.load(handle)

    selected = [
        ("function_word_rate", "Function words"),
        ("pronoun_rate", "Pronouns"),
        ("content_zipf_mean", "Word frequency"),
        ("mattr_50", "Lex. diversity"),
        ("dependency_length_mean", "Dependency length"),
        ("sentence_length_cv", "Sentence variation"),
    ]
    indexed = feature_stats.set_index("feature")
    effect_rows = [
        {
            "label": label,
            "group": str(indexed.loc[name, "group"]),
            "value": float(indexed.loc[name, "paired_d"]),
        }
        for name, label in selected
    ]

    figure = plt.figure(figsize=(3.42, 1.78))
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=[1.08, 1.0],
        wspace=0.60,
    )
    effect_axis = figure.add_subplot(grid[0, 0])
    stability_axis = figure.add_subplot(grid[0, 1])

    y_positions = np.arange(len(effect_rows))[::-1]
    for y, row in zip(y_positions, effect_rows):
        style = CONCEPT_STYLE[row["group"]]
        marker = "o" if row["group"] == "lexical" else "s"
        effect_axis.plot(
            [0.0, row["value"]],
            [y, y],
            color=style["light"],
            linewidth=1.25,
            zorder=1,
        )
        effect_axis.scatter(
            row["value"],
            y,
            s=14,
            marker=marker,
            facecolor=style["color"],
            edgecolor=style["dark"],
            linewidth=0.55,
            zorder=2,
        )
        effect_axis.annotate(
            f"{row['value']:+.3f}",
            (row["value"], y),
            xytext=(3, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=5.0,
            color=style["dark"],
        )
    effect_axis.axvline(0.0, color=NEUTRAL, linewidth=0.55, zorder=0)
    effect_axis.set_xlim(-1.57, 0.88)
    effect_axis.set_ylim(-0.55, len(effect_rows) - 0.45)
    effect_axis.set_yticks(y_positions, [row["label"] for row in effect_rows])
    effect_axis.set_xticks([-1.0, 0.0, 0.5])
    effect_axis.set_xlabel("Paired standardized shift", fontsize=5.5, labelpad=1.5)
    effect_axis.tick_params(axis="x", labelsize=5.0, length=2, pad=1)
    effect_axis.tick_params(axis="y", labelsize=5.2, length=0, pad=2)
    effect_axis.grid(axis="x", linewidth=0.45)
    effect_axis.text(
        0.0, 1.045, "A", transform=effect_axis.transAxes,
        ha="left", va="bottom", fontsize=7.3, fontweight="bold", color=INK,
    )
    effect_axis.annotate(
        "Largest paired effects", xy=(0.0, 1.045), xycoords="axes fraction",
        xytext=(10, 0), textcoords="offset points", ha="left", va="bottom",
        fontsize=6.2, color=INK,
    )
    for spine in ("top", "right", "left"):
        effect_axis.spines[spine].set_visible(False)

    concepts = ["lexical", "syntactic", "discourse"]
    tests = ["Gen.\nmin", "Domain\nmin", "Prefix\ncos.", "Prefix\npositive"]
    values = {
        "lexical": [
            float(
                holdout[
                    (holdout["environment_type"] == "generator")
                    & (holdout["group"] == "lexical")
                ]["heldout_direction_cosine"].min()
            ),
            float(
                holdout[
                    (holdout["environment_type"] == "domain")
                    & (holdout["group"] == "lexical")
                ]["heldout_direction_cosine"].min()
            ),
            float(transfer.loc["lexical", "plan_control_shift_cosine"]),
            float(transfer.loc["lexical", "control_projection_positive_rate"]),
        ],
        "syntactic": [
            float(
                holdout[
                    (holdout["environment_type"] == "generator")
                    & (holdout["group"] == "syntactic")
                ]["heldout_direction_cosine"].min()
            ),
            float(
                holdout[
                    (holdout["environment_type"] == "domain")
                    & (holdout["group"] == "syntactic")
                ]["heldout_direction_cosine"].min()
            ),
            float(transfer.loc["syntactic", "plan_control_shift_cosine"]),
            float(transfer.loc["syntactic", "control_projection_positive_rate"]),
        ],
        "discourse": [
            float(
                holdout[
                    (holdout["environment_type"] == "generator")
                    & (holdout["group"] == "discourse")
                ]["heldout_direction_cosine"].min()
            ),
            float(
                holdout[
                    (holdout["environment_type"] == "domain")
                    & (holdout["group"] == "discourse")
                ]["heldout_direction_cosine"].min()
            ),
            np.nan,
            np.nan,
        ],
    }
    matrix_style = {
        "lexical": CONCEPT_STYLE["lexical"],
        "syntactic": CONCEPT_STYLE["syntactic"],
        "discourse": {
            "color": NEUTRAL,
            "dark": MUTED,
            "open": NEUTRAL_OPEN,
        },
    }
    for y, concept in zip([2, 1, 0], concepts):
        style = matrix_style[concept]
        for x, value in enumerate(values[concept]):
            if np.isnan(value):
                stability_axis.text(
                    x, y, "–", ha="center", va="center",
                    fontsize=5.7, color=NEUTRAL,
                )
                continue
            stability_axis.add_patch(
                Rectangle(
                    (x - 0.42, y - 0.34),
                    0.84,
                    0.68,
                    facecolor=style["color"],
                    edgecolor=style["dark"],
                    linewidth=0.50,
                    alpha=0.12 + 0.58 * value,
                )
            )
            value_label = f"{100.0 * value:.1f}%" if x == 3 else f"{value:.3f}"
            stability_axis.text(
                x, y, value_label, ha="center", va="center",
                fontsize=4.5, color=INK,
            )
    shift_labels = [
        f"Lexical  {group_stats['lexical']['mahalanobis_mean_shift']:.3f}",
        f"Syntactic  {group_stats['syntactic']['mahalanobis_mean_shift']:.3f}",
        f"Discourse  {group_stats['discourse']['mahalanobis_mean_shift']:.3f}",
    ]
    stability_axis.set_xlim(-0.52, 3.52)
    stability_axis.set_ylim(-0.52, 2.52)
    stability_axis.set_xticks(range(4), tests)
    stability_axis.set_yticks([2, 1, 0], shift_labels)
    stability_axis.tick_params(axis="x", labelsize=4.8, length=0, pad=2)
    stability_axis.tick_params(axis="y", labelsize=5.0, length=0, pad=2)
    stability_axis.text(
        0.0, 1.045, "B", transform=stability_axis.transAxes,
        ha="left", va="bottom", fontsize=7.3, fontweight="bold", color=INK,
    )
    stability_axis.annotate(
        "Transfer stability", xy=(0.0, 1.045),
        xycoords="axes fraction", xytext=(10, 0), textcoords="offset points",
        ha="left", va="bottom", fontsize=6.2, color=INK,
    )
    for spine in stability_axis.spines.values():
        spine.set_visible(False)

    figure.subplots_adjust(left=0.255, right=0.995, top=0.85, bottom=0.22)
    save_figure(figure, output_dir, "surface_result_summary")


def draw_layer_trajectories(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
    *,
    show_title: bool = True,
) -> None:
    line_style = {
        "Llama-3.1-8B-Instruct": "-",
        "Qwen3-4B": "--",
    }
    marker = {
        "Llama-3.1-8B-Instruct": "o",
        "Qwen3-4B": "s",
    }
    for row in rows:
        style = CONCEPT_STYLE[row["concept"]]
        values = row["layers"]
        maximum = float(MODEL_SPECS[row["model"]]["max_layer"])
        x = values["layer"].to_numpy(float) / maximum
        mean = values["learned_minus_null_mean"].to_numpy(float)
        low = values["learned_minus_null_ci_low"].to_numpy(float)
        high = values["learned_minus_null_ci_high"].to_numpy(float)
        axis.fill_between(
            x,
            low,
            high,
            color=style["light"],
            alpha=0.42,
            linewidth=0,
            zorder=1,
        )
        axis.plot(
            x,
            mean,
            color=style["dark"],
            linestyle=line_style[row["model"]],
            linewidth=1.65,
            marker=marker[row["model"]],
            markersize=4.5,
            markerfacecolor=style["color"],
            markeredgecolor=style["dark"],
            zorder=3,
        )
    axis.axhline(0, color=INK, linewidth=1.0, zorder=0)
    axis.set_xlim(0.30, 1.02)
    axis.set_ylim(-0.085, 0.270)
    axis.set_xticks(
        [0.4, 0.6, 0.8, 1.0],
        ["40%", "60%", "80%", "100%"],
    )
    axis.set_xlabel("Relative decoder depth")
    axis.set_ylabel("Learned − matched-null capacity")
    if show_title:
        result_panel_title(axis, "A", "Layerwise excess capacity")
    clean_result_axis(axis, grid_axis="both")


def null_curve_values(row: dict[str, Any]) -> np.ndarray:
    model = row["model"]
    concept = row["concept"]
    layers = row["layers"]["layer"].to_numpy(int)
    frame = pd.read_csv(
        hidden_audit_root(model)
        / "matched_nulls/matched_null_axis_statistics.csv"
    )
    curves = (
        frame[
            frame["concept"].eq(concept)
            & frame["layer"].isin(layers)
        ]
        .pivot(
            index="null_axis",
            columns="layer",
            values="mean_transport_capacity",
        )
        .reindex(columns=layers)
    )
    if curves.isna().any().any():
        raise ValueError(f"Incomplete null curves for {model} {concept}")
    return np.asarray([
        normalized_curve_area(layers, values)
        for values in curves.to_numpy(float)
    ])


def draw_null_distributions(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
    *,
    show_title: bool = True,
    embedded: bool = False,
) -> None:
    y_values = np.arange(len(rows))[::-1]
    rng = np.random.default_rng(20260730)
    for y, row in zip(y_values, rows):
        style = CONCEPT_STYLE[row["concept"]]
        null_values = null_curve_values(row)
        q10, median, q90 = np.quantile(null_values, [0.10, 0.50, 0.90])
        axis.plot(
            [q10, q90],
            [y, y],
            color=NEUTRAL,
            linewidth=2.0,
            zorder=1,
        )
        jitter = np.clip(rng.normal(0.0, 0.105, len(null_values)), -0.22, 0.22)
        axis.scatter(
            null_values,
            y + jitter,
            s=16,
            facecolor=NEUTRAL,
            edgecolor="none",
            alpha=0.50,
            zorder=2,
        )
        axis.scatter(
            median,
            y,
            s=26,
            marker="o",
            facecolor=WHITE,
            edgecolor=NEUTRAL,
            linewidth=1.1,
            zorder=3,
        )
        learned = float(row["global"]["learned_curve_statistic"])
        axis.scatter(
            learned,
            y,
            marker="D",
            s=74,
            facecolor=style["color"],
            edgecolor=style["dark"],
            linewidth=1.5,
            zorder=4,
        )
        axis.text(
            0.99,
            y + 0.24,
            f"p={float(row['global']['finite_null_p_value']):.3f}",
            transform=axis.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=8.4,
            color=style["dark"],
        )
    axis.set_xlim(0.10, 0.42)
    axis.set_ylim(-0.55, 3.55)
    axis.set_yticks(
        y_values,
        (
            ["L·Lex", "L·Syn", "Q·Lex", "Q·Syn"]
            if embedded
            else [
                (
                    "Llama" if row["model"] == "Llama-3.1-8B-Instruct"
                    else "Qwen"
                )
                + " · "
                + ("Lex" if row["concept"] == "lexical" else "Syn")
                for row in rows
            ]
        ),
    )
    axis.set_xlabel(
        "Curve area" if embedded else "Depth-normalized curve area",
        fontsize=8.2 if embedded else None,
    )
    axis.tick_params(
        axis="y", length=0, pad=3 if embedded else 5,
        labelsize=7.7 if embedded else 10.1,
    )
    if embedded:
        axis.tick_params(axis="x", labelsize=7.5, pad=2)
    if show_title:
        result_panel_title(axis, "B", "Complete-pipeline null")
    clean_result_axis(axis)


def prompt_excess_curve_areas(model: str, concept: str) -> pd.Series:
    """Prompt-level learned-minus-matched-null curve area."""
    spec = MODEL_SPECS[model]
    excluded = {int(spec["input"]), int(spec["anchors"][concept])}
    learned = pd.read_csv(
        hidden_audit_root(model)
        / "layerwise_transport/per_prompt_transport.csv"
    )
    learned = learned[
        learned["intervention"].eq(concept)
        & learned["readout"].eq(concept)
        & ~learned["layer"].isin(excluded)
    ].copy()
    learned["capacity"] = learned["transport_coefficient"].abs()
    learned_wide = learned.pivot(
        index="prompt_id", columns="layer", values="capacity"
    )

    null = pd.read_csv(
        hidden_audit_root(model)
        / "matched_nulls/per_prompt_matched_null_transport.csv"
    )
    null = null[
        null["concept"].eq(concept)
        & ~null["layer"].isin(excluded)
    ].copy()
    null["capacity"] = null["transport_coefficient"].abs()
    null_wide = (
        null.groupby(["prompt_id", "layer"])["capacity"]
        .mean()
        .unstack("layer")
    )
    layers = sorted(set(learned_wide.columns) & set(null_wide.columns))
    prompt_ids = learned_wide.index.intersection(null_wide.index)
    difference = (
        learned_wide.loc[prompt_ids, layers]
        - null_wide.loc[prompt_ids, layers]
    )
    return pd.Series(
        {
            prompt_id: normalized_curve_area(
                np.asarray(layers, float),
                difference.loc[prompt_id].to_numpy(float),
            )
            for prompt_id in prompt_ids
        },
        name=model,
    )


def draw_result_marginal(
    axis: plt.Axes,
    values: np.ndarray,
    *,
    style: dict[str, str],
    orientation: str,
    limits: tuple[float, float],
) -> None:
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    grid = np.linspace(limits[0], limits[1], 180)
    spread = float(np.std(values, ddof=1))
    q25, q75 = np.quantile(values, [0.25, 0.75])
    robust_spread = float((q75 - q25) / 1.349)
    scale = min(spread, robust_spread) if robust_spread > 0 else spread
    bandwidth = max(0.9 * scale * max(len(values), 2) ** (-0.2), 0.006)
    z = (grid[:, None] - values[None, :]) / bandwidth
    density = np.exp(-0.5 * z ** 2).mean(axis=1)
    density /= bandwidth * np.sqrt(2.0 * np.pi)
    if orientation == "x":
        axis.fill_between(grid, 0, density, color=style["open"], alpha=0.78)
        axis.plot(grid, density, color=style["dark"], linewidth=0.8)
        axis.set_xlim(*limits)
    else:
        axis.fill_betweenx(grid, 0, density, color=style["open"], alpha=0.78)
        axis.plot(density, grid, color=style["dark"], linewidth=0.8)
        axis.set_ylim(*limits)
    axis.set_axis_off()


def draw_cross_model_prompt_agreement(
    axis: plt.Axes,
    *,
    marginal_x: plt.Axes | None = None,
    marginal_y: plt.Axes | None = None,
    show_title: bool = True,
) -> None:
    combined = []
    concept_frames: dict[str, pd.DataFrame] = {}
    for concept in CONCEPTS:
        style = CONCEPT_STYLE[concept]
        frame = pd.concat(
            [
                prompt_excess_curve_areas("Llama-3.1-8B-Instruct", concept),
                prompt_excess_curve_areas("Qwen3-4B", concept),
            ],
            axis=1,
        ).dropna()
        frame.columns = ["llama", "qwen"]
        combined.append(frame)
        concept_frames[concept] = frame
        axis.scatter(
            frame["llama"],
            frame["qwen"],
            s=25,
            marker="o" if concept == "lexical" else "D",
            facecolor=style["color"],
            edgecolor=style["dark"],
            linewidth=0.35,
            alpha=0.46,
            zorder=3,
        )
        result_covariance_ellipse(
            axis,
            frame["llama"].to_numpy(float),
            frame["qwen"].to_numpy(float),
            edge=style["dark"],
            level=0.5,
            linewidth=1.45,
            alpha=0.92,
        )
        coefficients = np.polyfit(frame["llama"], frame["qwen"], 1)
        x_fit = np.quantile(frame["llama"], [0.03, 0.97])
        axis.plot(
            x_fit,
            np.polyval(coefficients, x_fit),
            color=style["dark"],
            linewidth=1.35,
            zorder=2,
        )
        axis.scatter(
            frame["llama"].mean(),
            frame["qwen"].mean(),
            s=74,
            marker="D",
            facecolor=style["color"],
            edgecolor=style["dark"],
            linewidth=1.2,
            zorder=5,
        )
    all_points = pd.concat(combined)
    x_low, x_high = np.quantile(all_points["llama"], [0.0, 1.0])
    y_low, y_high = np.quantile(all_points["qwen"], [0.0, 1.0])
    x_pad = 0.08 * (x_high - x_low)
    y_pad = 0.10 * (y_high - y_low)
    axis.axhline(0, color=INK, linewidth=0.9)
    axis.axvline(0, color=INK, linewidth=0.9)
    axis.set_xlim(x_low - x_pad, x_high + x_pad)
    axis.set_ylim(y_low - y_pad, y_high + y_pad)
    x_limits = axis.get_xlim()
    y_limits = axis.get_ylim()
    if marginal_x is not None:
        for concept, frame in concept_frames.items():
            draw_result_marginal(
                marginal_x,
                frame["llama"].to_numpy(float),
                style=CONCEPT_STYLE[concept],
                orientation="x",
                limits=x_limits,
            )
    if marginal_y is not None:
        for concept, frame in concept_frames.items():
            draw_result_marginal(
                marginal_y,
                frame["qwen"].to_numpy(float),
                style=CONCEPT_STYLE[concept],
                orientation="y",
                limits=y_limits,
            )
    axis.set_xlabel("Llama excess curve area")
    axis.set_ylabel("Qwen excess curve area")
    if show_title:
        result_panel_title(axis, "C", "Cross-model prompt agreement")
    clean_result_axis(axis, grid_axis="both")


def draw_geometry_pairs_new(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
) -> None:
    y_values = np.arange(len(rows))[::-1]
    for y, row in zip(y_values, rows):
        style = CONCEPT_STYLE[row["concept"]]
        encoding = float(row["geometry"]["encoding"])
        agreement = float(row["geometry"]["human_ai"])
        axis.plot(
            [encoding, agreement],
            [y, y],
            color=style["light"],
            linewidth=5.0,
            zorder=1,
        )
        axis.scatter(
            encoding,
            y,
            marker="s",
            s=62,
            facecolor=WHITE,
            edgecolor=style["dark"],
            linewidth=1.4,
            zorder=3,
        )
        axis.scatter(
            agreement,
            y,
            marker="o",
            s=68,
            facecolor=style["color"],
            edgecolor=style["dark"],
            linewidth=1.4,
            zorder=3,
        )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(-0.55, 3.55)
    axis.set_yticks(y_values, [""] * len(rows))
    axis.set_xlabel("Cosine similarity")
    axis.tick_params(axis="y", length=0)
    result_panel_title(
        axis,
        "C",
        "Geometry separates",
        "□ encoding direction · ● human/machine pullbacks",
    )
    clean_result_axis(axis)


def draw_single_layer_trajectory(
    axis: plt.Axes,
    row: dict[str, Any],
    *,
    limits: tuple[float, float],
    show_x: bool,
    show_y: bool,
) -> None:
    """Draw one model×concept trace without spending legend channels."""
    style = CONCEPT_STYLE[row["concept"]]
    layers = row["layers"]
    maximum = float(MODEL_SPECS[row["model"]]["max_layer"])
    x = layers["layer"].to_numpy(float) / maximum
    mean = layers["learned_minus_null_mean"].to_numpy(float)
    low = layers["learned_minus_null_ci_low"].to_numpy(float)
    high = layers["learned_minus_null_ci_high"].to_numpy(float)
    axis.fill_between(
        x, low, high, color=style["open"], alpha=0.92,
        linewidth=0, zorder=1,
    )
    axis.plot(
        x, mean, color=style["dark"], linewidth=1.25, zorder=2,
    )
    axis.scatter(
        x, mean, s=19,
        marker="o" if row["concept"] == "lexical" else "D",
        facecolor=style["color"], edgecolor=style["dark"],
        linewidth=0.55, zorder=3,
    )
    axis.axhline(0, color=INK, linewidth=0.75, zorder=0)
    axis.text(
        0.05, 0.07,
        f"p={float(row['global']['finite_null_p_value']):.3f}",
        transform=axis.transAxes, ha="left", va="bottom",
        fontsize=7.4, color=style["dark"],
    )
    axis.set_xlim(0.28, 1.02)
    axis.set_ylim(*limits)
    axis.set_xticks([0.4, 0.6, 0.8, 1.0])
    axis.tick_params(
        axis="x", labelbottom=show_x, labelsize=7.5, pad=1.5,
    )
    axis.tick_params(
        axis="y", labelleft=show_y, labelsize=7.5, pad=1.5,
    )
    if show_x:
        axis.set_xlabel("Relative depth", fontsize=8.1, labelpad=2)
    if show_y:
        axis.set_ylabel("Excess capacity", fontsize=8.1, labelpad=2)
    clean_result_axis(axis, grid_axis="both")
    axis.set_box_aspect(1)


def draw_single_null_axis(
    axis: plt.Axes,
    row: dict[str, Any],
    *,
    show_title: bool,
) -> None:
    """Draw one compact vertical curve-area null axis beside its trace."""
    style = CONCEPT_STYLE[row["concept"]]
    null_values = null_curve_values(row)
    q10, median, q90 = np.quantile(null_values, [0.10, 0.50, 0.90])
    learned = float(row["global"]["learned_curve_statistic"])
    seed = (
        20260901
        + (0 if row["model"].startswith("Llama") else 101)
        + (0 if row["concept"] == "lexical" else 17)
    )
    rng = np.random.default_rng(seed)
    jitter = np.clip(
        rng.normal(0.0, 0.052, len(null_values)), -0.12, 0.12,
    )
    axis.plot(
        [0.0, 0.0], [q10, q90], color=NEUTRAL,
        linewidth=1.15, zorder=1,
    )
    axis.scatter(
        jitter, null_values, s=6.0, facecolor=NEUTRAL,
        edgecolor="none", alpha=0.48, zorder=2,
    )
    axis.scatter(
        0.0, median, s=11, marker="o", facecolor=WHITE,
        edgecolor=NEUTRAL, linewidth=0.65, zorder=3,
    )
    axis.scatter(
        0.0, learned, s=28, marker="D", facecolor=style["color"],
        edgecolor=style["dark"], linewidth=0.75, zorder=4,
    )
    axis.set_xlim(-0.18, 0.18)
    axis.set_ylim(0.10, 0.42)
    axis.set_xticks([])
    axis.set_yticks([0.10, 0.40], [".10", ".40"])
    axis.yaxis.tick_right()
    axis.tick_params(
        axis="y", labelsize=5.8, length=2.0, width=0.55, pad=1.0,
        colors=MUTED,
    )
    axis.spines["left"].set_visible(False)
    axis.spines["bottom"].set_visible(False)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_color(NEUTRAL)
    axis.spines["right"].set_linewidth(0.65)
    axis.grid(axis="y", color=GRID, linewidth=0.55, alpha=0.75)
    axis.grid(axis="x", visible=False)
    if show_title:
        axis.set_title("area", fontsize=6.5, color=MUTED, pad=3)


def build_hidden_trace_result_only(output_dir: Path) -> None:
    rows = trace_rows()
    all_intervals = np.concatenate([
        row["layers"][[
            "learned_minus_null_ci_low", "learned_minus_null_ci_high"
        ]].to_numpy(float).ravel()
        for row in rows
    ])
    lower = min(-0.02, float(np.nanmin(all_intervals)))
    upper = max(0.02, float(np.nanmax(all_intervals)))
    padding = 0.08 * (upper - lower)
    common_limits = (lower - padding, upper + padding)

    # Keep the two result panels side by side.  The compact source aspect ratio
    # lets the complete figure remain legible when included at one-column width.
    figure = plt.figure(figsize=(5.40, 2.70))
    outer = figure.add_gridspec(
        1,
        2,
        width_ratios=[1.52, 1.0],
        wspace=0.22,
    )
    evidence = outer[0].subgridspec(
        2, 2, wspace=0.12, hspace=0.08,
    )
    lookup = {
        (row["model"], row["concept"]): row for row in rows
    }
    evidence_axes: list[list[plt.Axes]] = []
    for row_index, concept in enumerate(CONCEPTS):
        axis_row = []
        for column, model in enumerate(MODEL_SPECS):
            cell = evidence[row_index, column].subgridspec(
                1, 2, width_ratios=[1.0, 0.31], wspace=0.04,
            )
            axis = figure.add_subplot(cell[0])
            draw_single_layer_trajectory(
                axis,
                lookup[(model, concept)],
                limits=common_limits,
                show_x=row_index == 1,
                show_y=column == 0,
            )
            if row_index == 0:
                axis.set_title(
                    "Llama" if model.startswith("Llama") else "Qwen",
                    fontsize=8.9, color=INK, pad=3,
                )
            null_axis = figure.add_subplot(cell[1])
            draw_single_null_axis(
                null_axis,
                lookup[(model, concept)],
                show_title=row_index == 0,
            )
            axis_row.append(axis)
        evidence_axes.append(axis_row)
    # One shared response label is enough for the 2×2 matrix and prevents the
    # two row labels from colliding at compact, one-column scale.
    evidence_axes[0][0].set_ylabel("")
    evidence_axes[1][0].set_ylabel("")
    figure.text(
        0.012, 0.47, "Excess capacity", rotation=90,
        ha="center", va="center", fontsize=8.1, color=INK,
    )
    agreement_axis = figure.add_subplot(outer[0, 1])
    draw_cross_model_prompt_agreement(
        agreement_axis,
        show_title=False,
    )
    agreement_axis.set_xlabel("Llama excess curve area", fontsize=9.3)
    agreement_axis.set_ylabel("Qwen excess curve area", fontsize=9.3)
    agreement_axis.tick_params(labelsize=8.1, pad=2)
    agreement_axis.set_box_aspect(1)
    figure.text(
        0.045, 0.905, "A", ha="left", va="center",
        fontsize=11.2, fontweight="bold", color=INK,
    )
    figure.text(
        0.080, 0.905, "Depthwise transport",
        ha="left", va="center",
        fontsize=10.2, color=INK,
    )
    figure.text(
        0.630, 0.905, "B", ha="left", va="center",
        fontsize=11.2, fontweight="bold", color=INK,
    )
    figure.text(
        0.665, 0.905, "Cross-model prompt agreement",
        ha="left", va="center", fontsize=10.2, color=INK,
    )
    figure.legend(
        handles=[
            Line2D(
                [0], [0], marker="o", linestyle="none",
                color=BLUE_DARK, markerfacecolor=BLUE, label="Lexical",
            ),
            Line2D(
                [0], [0], marker="D", linestyle="none",
                color=PINK_DARK, markerfacecolor=PINK, label="Syntactic",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.47, 0.995),
        ncol=2,
        fontsize=8.6,
        columnspacing=0.9,
        handletextpad=0.35,
    )
    figure.subplots_adjust(
        left=0.075,
        right=0.985,
        top=0.785,
        bottom=0.15,
    )
    save_figure(figure, output_dir, "hidden_causal_trace")


def draw_readout_summary_all(
    axis: plt.Axes,
    observations: pd.DataFrame,
    summary: pd.DataFrame,
    *,
    show_title: bool = True,
) -> None:
    y_values = np.arange(len(ROWS))[::-1]
    offsets = np.linspace(-0.14, 0.14, 4)
    for y, (model, concept) in zip(y_values, ROWS):
        style = CONCEPT_STYLE[concept]
        values = observations[
            (observations["model"] == model)
            & (observations["concept"] == concept)
            & (observations["stage"] == "encoding")
        ]["value"].to_numpy(float)
        stats = row_summary(summary, model, concept, "encoding")
        axis.scatter(
            values,
            y + offsets,
            s=35,
            facecolor=WHITE,
            edgecolor=style["dark"],
            linewidth=1.1,
            zorder=2,
        )
        axis.plot(
            [float(stats["ci_low"]), float(stats["ci_high"])],
            [y, y],
            color=style["dark"],
            linewidth=1.7,
            zorder=1,
        )
        for bound in (float(stats["ci_low"]), float(stats["ci_high"])):
            axis.plot(
                [bound, bound],
                [y - 0.10, y + 0.10],
                color=style["dark"],
                linewidth=0.9,
                zorder=1,
            )
        axis.scatter(
            float(stats["mean"]),
            y,
            s=70,
            facecolor=style["color"],
            edgecolor=style["dark"],
            linewidth=1.4,
            zorder=3,
        )
        axis.text(
            0.99,
            y,
            f"{float(stats['mean']):.2f}",
            transform=axis.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=10.2,
            color=style["dark"],
        )
    axis.set_xlim(0.30, 0.59)
    axis.set_ylim(-0.55, 3.55)
    axis.set_yticks(
        y_values,
        ["Llama · Lex", "Llama · Syn", "Qwen · Lex", "Qwen · Syn"],
    )
    axis.tick_params(axis="y", length=0, pad=5, labelsize=10.2)
    axis.set_xlabel("Held-out-generator Pearson r")
    if show_title:
        result_panel_title(axis, "A", "Cross-generator readout")
    clean_result_axis(axis)


def response_profile_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = ("lexical", "syntactic", "discourse")
    for model, concept in ROWS:
        frame = pd.read_csv(
            confirmatory_analysis_root(model)
            / "causal_group_statistics.csv"
        )
        view = (
            frame[frame["component"].eq(concept)]
            .set_index("group")
            .reindex(groups)
        )
        rows.append({
            "model": model,
            "concept": concept,
            "groups": groups,
            "mean": view["signature_projection_mean"].to_numpy(float),
            "low": view["signature_projection_ci_low"].to_numpy(float),
            "high": view["signature_projection_ci_high"].to_numpy(float),
        })
    return rows


def draw_model_readout_compact(
    axis: plt.Axes,
    observations: pd.DataFrame,
    summary: pd.DataFrame,
    model: str,
    *,
    show_x: bool,
) -> None:
    y_positions = {"lexical": 1.0, "syntactic": 0.0}
    offsets = np.linspace(-0.12, 0.12, 4)
    for concept in CONCEPTS:
        style = CONCEPT_STYLE[concept]
        y = y_positions[concept]
        values = observations[
            (observations["model"] == model)
            & (observations["concept"] == concept)
            & (observations["stage"] == "encoding")
        ]["value"].to_numpy(float)
        stats = row_summary(summary, model, concept, "encoding")
        axis.scatter(
            values, y + offsets, s=21, facecolor=WHITE,
            edgecolor=style["dark"], linewidth=0.75, zorder=2,
        )
        axis.plot(
            [float(stats["ci_low"]), float(stats["ci_high"])],
            [y, y], color=style["dark"], linewidth=1.45, zorder=1,
        )
        axis.scatter(
            float(stats["mean"]), y, s=43,
            facecolor=style["color"], edgecolor=style["dark"],
            linewidth=0.9, zorder=3,
        )
    axis.set_xlim(0.30, 0.59)
    axis.set_ylim(-0.46, 1.46)
    axis.set_yticks([1, 0], ["Lexical", "Syntactic"])
    axis.tick_params(axis="y", length=0, labelsize=7.8, pad=2)
    axis.tick_params(
        axis="x", labelbottom=show_x, labelsize=7.5, pad=1.5,
    )
    axis.set_ylabel(
        "Llama" if model.startswith("Llama") else "Qwen",
        fontsize=8.5, labelpad=3,
    )
    if show_x:
        axis.set_xlabel("Held-out r", fontsize=8.1, labelpad=2)
    clean_result_axis(axis)
    axis.set_box_aspect(1)


def draw_model_response_heatmap(
    axis: plt.Axes,
    model: str,
    *,
    cmap: LinearSegmentedColormap,
    limit: float,
    show_x: bool,
) -> Any:
    groups = ("lexical", "syntactic", "discourse")
    rows = [
        row for row in response_profile_rows() if row["model"] == model
    ]
    matrix = np.vstack([row["mean"] for row in rows])
    image = axis.imshow(
        matrix, cmap=cmap, vmin=-limit, vmax=limit,
        aspect="auto", interpolation="nearest", zorder=0,
    )
    for row_index, row in enumerate(rows):
        style = CONCEPT_STYLE[row["concept"]]
        for column, (group, mean) in enumerate(zip(groups, row["mean"])):
            if group == row["concept"]:
                axis.add_patch(
                    Rectangle(
                        (column - 0.46, row_index - 0.46), 0.92, 0.92,
                        fill=False, edgecolor=style["dark"],
                        linewidth=1.25, zorder=2,
                    )
                )
            text_color = INK if abs(float(mean)) < 0.72 * limit else WHITE
            axis.text(
                column, row_index, f"{float(mean):+.2f}",
                ha="center", va="center", fontsize=6.7,
                color=text_color, zorder=3,
            )
    axis.set_xticks(
        np.arange(3), ["Lex", "Syn", "Disc"] if show_x else ["", "", ""],
    )
    axis.set_yticks([])
    axis.tick_params(axis="x", length=0, labelsize=7.7, pad=2)
    axis.set_xlim(-0.5, 2.5)
    axis.set_ylim(1.5, -0.5)
    axis.grid(False)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.set_box_aspect(1)
    return image


def draw_response_profiles_new(
    axis: plt.Axes,
    *,
    show_title: bool = True,
    show_y_labels: bool = True,
) -> None:
    groups = ("lexical", "syntactic", "discourse")
    rows = response_profile_rows()
    matrix = np.vstack([row["mean"] for row in rows])
    limit = float(np.nanmax(np.abs(matrix)))
    cmap = LinearSegmentedColormap.from_list(
        "fresh_signed",
        [PINK, PINK_OPEN, WHITE, BLUE_OPEN, BLUE],
    )
    axis.imshow(
        matrix,
        cmap=cmap,
        vmin=-limit,
        vmax=limit,
        aspect="auto",
        interpolation="nearest",
        zorder=0,
    )
    for row_index, row in enumerate(rows):
        style = CONCEPT_STYLE[row["concept"]]
        for column, (group, mean) in enumerate(zip(groups, row["mean"])):
            is_target = group == row["concept"]
            if is_target:
                axis.add_patch(
                    Rectangle(
                        (column - 0.46, row_index - 0.46),
                        0.92,
                        0.92,
                        fill=False,
                        edgecolor=style["dark"],
                        linewidth=1.8,
                        zorder=2,
                    )
                )
            text_color = INK if abs(float(mean)) < 0.72 * limit else WHITE
            axis.text(
                column,
                row_index,
                f"{float(mean):+.2f}",
                ha="center",
                va="center",
                fontsize=10.0,
                color=text_color,
                zorder=3,
            )
    row_labels = [
        (
            "Llama" if row["model"] == "Llama-3.1-8B-Instruct" else "Qwen"
        )
        + " · "
        + ("Lex" if row["concept"] == "lexical" else "Syn")
        for row in rows
    ]
    axis.set_xticks(np.arange(len(groups)), ["Lexical", "Syntactic", "Discourse"])
    axis.set_yticks(
        np.arange(len(rows)),
        row_labels if show_y_labels else [""] * len(row_labels),
    )
    axis.tick_params(axis="both", length=0, labelsize=10.0)
    axis.set_xlim(-0.5, 2.5)
    axis.set_ylim(3.5, -0.5)
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.grid(False)
    axis.text(
        0.5,
        1.015,
        "response: pink −  ·  blue +  ·  outline = target",
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.1,
        color=MUTED,
    )
    if show_title:
        result_panel_title(axis, "B", "Signed response matrix")


def seed_summary_rows() -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for model in MODEL_SPECS:
        for concept in CONCEPTS:
            for estimand in ("target_projection", "component_selectivity"):
                _, seeds, stats = seed_estimand_values(
                    model,
                    concept,
                    estimand,
                )
                output.append({
                    "model": model,
                    "concept": concept,
                    "estimand": estimand,
                    "seeds": seeds,
                    "stats": stats,
                })
    return output


def draw_seed_summary_new(
    axis: plt.Axes,
    rows: list[dict[str, Any]],
    *,
    show_title: bool = True,
) -> None:
    lookup = {
        (row["model"], row["concept"], row["estimand"]): row
        for row in rows
    }
    y_values = np.arange(len(ROWS))[::-1]
    offsets = {"target_projection": 0.13, "component_selectivity": -0.13}
    for y, (model, concept) in zip(y_values, ROWS):
        style = CONCEPT_STYLE[concept]
        for estimand in ("target_projection", "component_selectivity"):
            row = lookup[(model, concept, estimand)]
            stats = row["stats"]
            is_target = estimand == "target_projection"
            row_y = y + offsets[estimand]
            axis.plot(
                [float(stats["ci_low"]), float(stats["ci_high"])],
                [row_y, row_y],
                color=style["dark"],
                linewidth=1.55,
                zorder=1,
            )
            seeds = np.asarray(row["seeds"], float)
            seed_y = row_y + np.linspace(-0.035, 0.035, len(seeds))
            axis.scatter(
                seeds,
                seed_y,
                s=22,
                facecolor=WHITE,
                edgecolor=style["dark"],
                linewidth=0.9,
                zorder=2,
            )
            axis.scatter(
                float(stats["mean"]),
                row_y,
                marker="D" if is_target else "o",
                s=62 if is_target else 54,
                facecolor=style["color"] if is_target else WHITE,
                edgecolor=style["dark"],
                linewidth=1.25,
                zorder=3,
            )
    labels = ["Lex" if concept == "lexical" else "Syn" for _, concept in ROWS]
    axis.axvline(0, color=INK, linewidth=1.0, zorder=0)
    axis.axhline(1.5, color=GRID, linewidth=1.1, zorder=0)
    axis.set_xlim(-0.62, 0.62)
    axis.set_ylim(-0.50, 3.50)
    axis.set_yticks(y_values, labels)
    axis.tick_params(axis="y", length=0, pad=5, labelsize=9.4)
    axis.set_xlabel("Signed replicated effect", fontsize=9.4, labelpad=3)
    if show_title:
        result_panel_title(axis, "C", "Three-seed replication")
    clean_result_axis(axis)


def build_response_audit_result_only(
    output_dir: Path,
    *,
    repetitions: int,
) -> None:
    validation = pd.read_csv(
        ROOT
        / "outputs/en/05_linguistic/concept_patterns/"
        "pattern_validation.csv"
    )
    observations = encoding_observations(
        validation,
        {
            model: int(spec["input"])
            for model, spec in MODEL_SPECS.items()
        },
    )
    summary = summarize_gap(
        observations,
        bootstrap=repetitions,
        permutations=20000,
        seed=20260726,
    )
    profiles = response_profile_rows()
    response_limit = float(np.nanmax(np.abs(np.vstack([
        row["mean"] for row in profiles
    ]))))
    response_cmap = LinearSegmentedColormap.from_list(
        "fresh_signed_compact",
        [PINK, PINK_OPEN, WHITE, BLUE_OPEN, BLUE],
    )

    # Preserve the horizontal A/B composition while keeping the full figure
    # compact enough for a one-column include in the manuscript.
    figure = plt.figure(figsize=(5.00, 2.75))
    outer = figure.add_gridspec(
        1, 2, width_ratios=[1.0, 1.0], wspace=0.18,
    )
    representation = outer[0].subgridspec(
        2, 2, width_ratios=[1.0, 1.0], wspace=0.08, hspace=0.06,
    )
    readout_axes: list[plt.Axes] = []
    heatmap_axes: list[plt.Axes] = []
    heatmap_image = None
    for row_index, model in enumerate(MODEL_SPECS):
        readout_axis = figure.add_subplot(representation[row_index, 0])
        response_axis = figure.add_subplot(representation[row_index, 1])
        draw_model_readout_compact(
            readout_axis, observations, summary, model,
            show_x=row_index == 1,
        )
        heatmap_image = draw_model_response_heatmap(
            response_axis, model, cmap=response_cmap,
            limit=response_limit, show_x=row_index == 1,
        )
        readout_axes.append(readout_axis)
        heatmap_axes.append(response_axis)
    readout_axes[0].set_title(
        "Held-out readout", fontsize=8.2, color=INK, pad=2,
    )
    colorbar_axis = heatmap_axes[0].inset_axes([0.06, 1.12, 0.88, 0.055])
    colorbar = figure.colorbar(
        heatmap_image, cax=colorbar_axis, orientation="horizontal",
    )
    colorbar.set_ticks([-response_limit, 0, response_limit])
    colorbar.set_ticklabels([
        f"−{response_limit:.2f}", "0", f"+{response_limit:.2f}",
    ])
    colorbar.ax.tick_params(labelsize=6.8, length=2, pad=1)
    colorbar.set_label("Signed response", fontsize=7.2, labelpad=4)
    colorbar.ax.xaxis.set_label_position("top")
    seed_axis = figure.add_subplot(outer[0, 1])
    draw_seed_summary_new(
        seed_axis,
        seed_summary_rows(),
        show_title=False,
    )
    seed_axis.set_box_aspect(1)
    seed_axis.legend(
        handles=[
            Line2D(
                [0], [0], marker="D", linestyle="none",
                color=MUTED, markerfacecolor=NEUTRAL,
                label="target projection",
            ),
            Line2D(
                [0], [0], marker="o", linestyle="none",
                color=MUTED, markerfacecolor=WHITE,
                label="component selectivity",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.025),
        ncol=2,
        fontsize=7.2,
        columnspacing=0.55,
        handletextpad=0.3,
    )
    figure.text(
        0.050, 0.910, "A", ha="left", va="center",
        fontsize=11.2, fontweight="bold", color=INK,
    )
    figure.text(
        0.085, 0.910, "Readout + response",
        ha="left", va="center", fontsize=10.2, color=INK,
    )
    figure.text(
        0.550, 0.910, "B", ha="left", va="center",
        fontsize=11.2, fontweight="bold", color=INK,
    )
    figure.text(
        0.585, 0.910, "Three-seed replication",
        ha="left", va="center", fontsize=10.2, color=INK,
    )
    figure.subplots_adjust(
        left=0.095,
        right=0.985,
        top=0.785,
        bottom=0.13,
    )
    save_figure(figure, output_dir, "representation_realization_audit")


def build_evidence_chain_summary(output_dir: Path) -> None:
    """Plot the frozen evidence chain as two vertical grouped-bar panels."""
    metric_labels = ("Prop.", "Prefix", "H/M", "Enc.", "Learn.", "Null")
    metric_positions = np.array([0.0, 0.98, 2.25, 3.23, 4.50, 5.48])
    panel_specs = (
        {
            "letter": "A",
            "title": "Llama 3.1 8B",
            "lexical": np.array([0.451, 0.470, 0.916, 0.265, 0.368, 0.201]),
            "syntactic": np.array([0.395, 0.421, 0.930, 0.197, 0.255, 0.202]),
        },
        {
            "letter": "B",
            "title": "Qwen3 4B",
            "lexical": np.array([0.442, 0.544, 0.926, 0.245, 0.315, 0.214]),
            "syntactic": np.array([0.450, 0.451, 0.942, 0.193, 0.253, 0.225]),
        },
    )

    figure, axes = plt.subplots(1, 2, figsize=(5.25, 2.82), sharey=True)
    bar_width = 0.31
    for panel_index, (axis, panel) in enumerate(zip(axes, panel_specs)):
        lexical_bars = axis.bar(
            metric_positions - bar_width / 2,
            panel["lexical"],
            width=bar_width,
            color=BLUE_LIGHT,
            edgecolor=BLUE_DARK,
            linewidth=0.60,
            zorder=3,
        )
        syntactic_bars = axis.bar(
            metric_positions + bar_width / 2,
            panel["syntactic"],
            width=bar_width,
            color=PINK_LIGHT,
            edgecolor=PINK_DARK,
            linewidth=0.60,
            zorder=3,
        )

        for bars in (lexical_bars, syntactic_bars):
            for bar in bars:
                axis.annotate(
                    f"{bar.get_height():.3f}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    rotation=90,
                    fontsize=8.2,
                    fontweight="normal",
                    color=INK,
                )

        axis.set_xlim(-0.50, 5.98)
        axis.set_ylim(0.0, 1.08)
        axis.set_xticks(metric_positions)
        axis.set_xticklabels(metric_labels, fontsize=8.4, fontweight="normal")
        axis.set_yticks((0.0, 0.25, 0.50, 0.75, 1.00))
        axis.tick_params(axis="x", length=0, pad=3)
        axis.tick_params(axis="y", labelsize=8.9, pad=2)
        axis.grid(axis="y", linewidth=0.60)
        axis.grid(axis="x", visible=False)
        axis.axvline(1.615, color=GRID, linewidth=0.75, zorder=1)
        axis.axvline(3.865, color=GRID, linewidth=0.75, zorder=1)
        panel_label(
            axis,
            panel["letter"],
            panel["title"],
            x=0.0,
            y=1.055,
            letter_size=11.2,
            text_size=10.2,
            text_offset=18.0,
        )

        axis.text(
            0.49,
            -0.145,
            "Readout",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8.6,
            fontweight="normal",
            color=MUTED,
        )
        axis.text(
            2.74,
            -0.145,
            "Geometry",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8.6,
            fontweight="normal",
            color=MUTED,
        )
        axis.text(
            4.99,
            -0.145,
            "Transport",
            transform=axis.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=8.6,
            fontweight="normal",
            color=MUTED,
        )
        if panel_index == 0:
            axis.set_ylabel("Value", fontsize=9.5, labelpad=2)

    legend_handles = (
        Rectangle(
            (0, 0), 1, 1,
            facecolor=BLUE_LIGHT,
            edgecolor=BLUE_DARK,
            linewidth=0.60,
        ),
        Rectangle(
            (0, 0), 1, 1,
            facecolor=PINK_LIGHT,
            edgecolor=PINK_DARK,
            linewidth=0.60,
        ),
    )
    figure.legend(
        legend_handles,
        ("Lexical", "Syntactic"),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.015),
        ncol=2,
        handlelength=1.35,
        handleheight=0.78,
        handletextpad=0.42,
        columnspacing=1.25,
        borderaxespad=0.0,
        fontsize=9.8,
    )
    figure.subplots_adjust(
        left=0.095,
        right=0.990,
        top=0.825,
        bottom=0.205,
        wspace=0.115,
    )
    save_figure(figure, output_dir, "evidence_chain_summary")


def main() -> None:
    args = parse_args()
    setup_style()
    if "signature" in args.figures:
        build_construct_atlas_result_only(args.out_dir)
    if "surface" in args.figures:
        build_surface_result_summary(args.out_dir)
    if "hidden" in args.figures:
        build_hidden_trace_result_only(args.out_dir)
    if "audit" in args.figures:
        build_response_audit_result_only(
            args.out_dir,
            repetitions=args.bootstrap,
        )
    if "chain" in args.figures:
        build_evidence_chain_summary(args.out_dir)


if __name__ == "__main__":
    main()
