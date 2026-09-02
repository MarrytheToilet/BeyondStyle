"""Compact paper atlas for token-level nullspace projection evidence."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from .palette import CJK_FONT
from .separability_style import (
    BG,
    BLUE,
    BLUE_DARK,
    BLUE_LIGHT,
    GRID,
    INK,
    MUTED,
    PINK,
    PINK_DARK,
    PINK_LIGHT,
    setup_separability_style,
)


AXIS_LABELS = {
    "last_token": "Last-token-fit axis",
    "mean_pool": "Mean-pool-fit axis",
}


def _clean_token(value: str, maximum: int = 18) -> str:
    text = " ".join(str(value).strip().split())
    text = text if text else "␠"
    return text if len(text) <= maximum else text[: maximum - 1] + "…"


def _contains_cjk(text: str) -> bool:
    return any(
        "\u3400" <= character <= "\u4dbf"
        or "\u4e00" <= character <= "\u9fff"
        or "\uf900" <= character <= "\ufaff"
        for character in text
    )


def _layer_panel(
    axis,
    frame: pd.DataFrame,
    *,
    selected_layer: int,
    show_legend: bool,
) -> None:
    styles = {
        "last_continuation_token": {
            "color": BLUE_DARK,
            "fill": BLUE,
            "marker": "o",
            "linestyle": "-",
            "label": "Final-token score",
        },
        "continuation_mean": {
            "color": PINK_DARK,
            "fill": PINK,
            "marker": "D",
            "linestyle": "--",
            "label": "Mean-token score",
        },
    }
    for representation, style in styles.items():
        part = frame[
            frame["token_representation"] == representation
        ].sort_values("layer")
        if part.empty:
            continue
        axis.plot(
            part["layer"],
            part["auc"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.4,
            marker=style["marker"],
            markersize=5.8,
            markerfacecolor=style["fill"],
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=style["label"],
            zorder=3,
        )
        chosen = part[part["layer"] == selected_layer]
        if not chosen.empty:
            axis.scatter(
                [selected_layer],
                [chosen.iloc[0]["auc"]],
                s=115,
                facecolor=style["fill"],
                edgecolor=INK,
                linewidth=1.5,
                marker=style["marker"],
                zorder=5,
            )
    axis.axvline(
        selected_layer,
        color=GRID,
        linewidth=6,
        alpha=0.55,
        zorder=0,
    )
    axis.axhline(0.5, color=MUTED, linewidth=1.0, linestyle=":", zorder=1)
    axis.set_ylim(0.45, 1.03)
    axis.set_xlabel("Decoder layer")
    axis.set_ylabel("Held-out source AUC")
    if show_legend:
        axis.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=1,
            frameon=False,
            fontsize=10.8,
            handlelength=2.5,
        )


def _position_panel(
    axis,
    frame: pd.DataFrame,
    *,
    show_legend: bool,
) -> None:
    x = frame["position_midpoint"].to_numpy(dtype=float)
    for prefix, color, dark, marker, linestyle, label in (
        ("human", BLUE, BLUE_DARK, "o", "-", "Human"),
        ("ai", PINK, PINK_DARK, "D", "--", "AI"),
    ):
        mean = frame[f"{prefix}_mean"].to_numpy(dtype=float)
        low = frame[f"{prefix}_ci_low"].to_numpy(dtype=float)
        high = frame[f"{prefix}_ci_high"].to_numpy(dtype=float)
        axis.fill_between(x, low, high, color=color, alpha=0.28, linewidth=0)
        axis.plot(
            x,
            mean,
            color=dark,
            linewidth=2.5,
            linestyle=linestyle,
            marker=marker,
            markersize=5.8,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label=label,
            zorder=3,
        )
    axis.vlines(
        x,
        frame["human_mean"],
        frame["ai_mean"],
        color=PINK_LIGHT,
        linewidth=1.1,
        alpha=0.78,
        zorder=1,
    )
    axis.axhline(0, color=MUTED, linewidth=1.0, linestyle=":", zorder=1)
    axis.set_xlim(0, 1)
    axis.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    axis.set_xticklabels(["Start", "25%", "50%", "75%", "End"])
    axis.set_xlabel("Normalized continuation position")
    axis.set_ylabel("Pair-balanced coordinate (z)")
    if show_legend:
        axis.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=2,
            frameon=False,
            fontsize=11.0,
            columnspacing=1.2,
        )


def _context_panel(axis, frame: pd.DataFrame) -> None:
    if len(frame) < 4:
        axis.text(
            0.5,
            0.5,
            "Insufficient exact\nsame-token matches",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color=MUTED,
            fontsize=12,
        )
        axis.set_axis_off()
        return
    x = frame["source_balanced_affinity_mean"].to_numpy(dtype=float)
    y = frame["context_shift_mean"].to_numpy(dtype=float)
    sizes = 20 + 13 * np.sqrt(frame["n_paired_samples"].to_numpy(dtype=float))
    colors = np.where(x >= 0, PINK, BLUE)
    edges = np.where(x >= 0, PINK_DARK, BLUE_DARK)
    axis.scatter(
        x,
        y,
        s=sizes,
        c=colors,
        edgecolors=edges,
        linewidths=0.7,
        alpha=0.48,
        zorder=3,
    )
    axis.axhline(0, color=MUTED, linewidth=1.0, linestyle=":", zorder=1)
    axis.axvline(0, color=MUTED, linewidth=1.0, linestyle=":", zorder=1)
    axis.set_xlabel("Source-balanced token affinity")
    axis.set_ylabel("Same-token shift  (AI − Human)")

    top = axis.inset_axes([0.0, 0.82, 1.0, 0.18], sharex=axis)
    right = axis.inset_axes([0.82, 0.0, 0.18, 1.0], sharey=axis)
    x_bins = np.histogram_bin_edges(x, bins="auto")
    y_bins = np.histogram_bin_edges(y, bins="auto")
    top.hist(x[x < 0], bins=x_bins, color=BLUE, alpha=0.68)
    top.hist(x[x >= 0], bins=x_bins, color=PINK, alpha=0.68)
    right.hist(
        y[y < 0],
        bins=y_bins,
        color=BLUE,
        alpha=0.68,
        orientation="horizontal",
    )
    right.hist(
        y[y >= 0],
        bins=y_bins,
        color=PINK,
        alpha=0.68,
        orientation="horizontal",
    )
    for marginal in (top, right):
        marginal.set_facecolor("none")
        marginal.grid(False)
        marginal.set_axis_off()


def _vocabulary_panel(
    axis,
    frame: pd.DataFrame,
    *,
    per_side: int = 6,
) -> None:
    if frame.empty:
        axis.text(
            0.5,
            0.5,
            "No source-balanced tokens\nwith nonzero interval",
            transform=axis.transAxes,
            ha="center",
            va="center",
            color=MUTED,
            fontsize=12,
        )
        axis.set_axis_off()
        return
    negative = (
        frame[frame["field_side"] == "Human-oriented"]
        .nsmallest(per_side, "source_balanced_affinity_mean")
        .sort_values("source_balanced_affinity_mean")
    )
    positive = (
        frame[frame["field_side"] == "AI-oriented"]
        .nlargest(per_side, "source_balanced_affinity_mean")
        .sort_values("source_balanced_affinity_mean")
    )
    selected = pd.concat([negative, positive], ignore_index=True)
    if selected.empty:
        axis.set_axis_off()
        return
    y = np.arange(len(selected))
    values = selected["source_balanced_affinity_mean"].to_numpy(dtype=float)
    colors = np.where(values < 0, BLUE, PINK)
    dark = np.where(values < 0, BLUE_DARK, PINK_DARK)
    axis.hlines(y, 0, values, color=colors, linewidth=4.2, alpha=0.58)
    axis.scatter(
        values,
        y,
        s=63,
        c=colors,
        edgecolors=dark,
        linewidths=1.1,
        zorder=3,
    )
    axis.axvline(0, color=MUTED, linewidth=1.0, linestyle=":", zorder=1)
    axis.set_yticks(y)
    token_labels = [
        _clean_token(value) for value in selected["token_text"]
    ]
    label_style = {"fontfamily": CJK_FONT} if (
        CJK_FONT and any(_contains_cjk(label) for label in token_labels)
    ) else {}
    axis.set_yticklabels(
        token_labels,
        fontsize=10.6,
        **label_style,
    )
    axis.set_xlabel("Source-balanced token affinity")
    axis.tick_params(axis="y", length=0)
    axis.grid(axis="y", visible=False)


def render_token_projection_atlas(
    *,
    layer_summary: pd.DataFrame,
    position_trajectory: pd.DataFrame,
    same_token_context: pd.DataFrame,
    vocabulary_field: pd.DataFrame,
    report_layers: dict[str, int],
    output: Path,
) -> Path:
    """Render one two-row, four-evidence atlas and return its PNG path."""
    setup_separability_style()
    axes_order = [
        axis for axis in ("last_token", "mean_pool") if axis in report_layers
    ]
    if not axes_order:
        raise ValueError("No report axes to render")
    figure, axes = plt.subplots(
        len(axes_order),
        4,
        figsize=(19.2, 5.0 * len(axes_order)),
        gridspec_kw={
            "width_ratios": [1.05, 1.25, 1.45, 1.28],
            "wspace": 0.42,
            "hspace": 0.42,
        },
        squeeze=False,
    )
    for row_index, axis_family in enumerate(axes_order):
        selected_layer = report_layers[axis_family]
        layer_part = layer_summary[
            layer_summary["axis_family"].astype(str) == axis_family
        ]
        position_part = position_trajectory[
            position_trajectory["axis_family"].astype(str) == axis_family
        ].sort_values("position_midpoint")
        context_part = same_token_context[
            same_token_context["axis_family"].astype(str) == axis_family
        ]
        vocabulary_part = vocabulary_field[
            vocabulary_field["axis_family"].astype(str) == axis_family
        ]
        _layer_panel(
            axes[row_index, 0],
            layer_part,
            selected_layer=selected_layer,
            show_legend=row_index == 0,
        )
        _position_panel(
            axes[row_index, 1],
            position_part,
            show_legend=row_index == 0,
        )
        _context_panel(axes[row_index, 2], context_part)
        _vocabulary_panel(axes[row_index, 3], vocabulary_part)
        axes[row_index, 0].text(
            -0.26,
            0.5,
            f"{AXIS_LABELS.get(axis_family, axis_family)}\nL{selected_layer}",
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=13.0,
            fontweight="bold",
            color=(
                BLUE_DARK if axis_family == "last_token" else PINK_DARK
            ),
        )
        for column in range(4):
            axes[row_index, column].text(
                0.01,
                0.99,
                f"{chr(97 + row_index * 4 + column)}",
                transform=axes[row_index, column].transAxes,
                ha="left",
                va="top",
                fontsize=12.0,
                fontweight="bold",
                color=INK,
                bbox={
                    "boxstyle": "round,pad=0.16",
                    "facecolor": BG,
                    "edgecolor": "none",
                    "alpha": 0.8,
                },
                zorder=10,
            )
    figure.subplots_adjust(left=0.075, right=0.985, top=0.91, bottom=0.08)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=220, facecolor=BG, bbox_inches="tight")
    pdf = output.with_suffix(".pdf")
    figure.savefig(pdf, facecolor=BG, bbox_inches="tight")
    plt.close(figure)
    return output
