"""Canonical styling helpers for the per-folder separability figures."""

from __future__ import annotations

import math
from collections.abc import Iterable

import matplotlib.pyplot as plt

from src.visualization.palette import CJK_FONT


# Keep these colours identical to the canonical hidden-geometry figures.
INK = "#27313A"
MUTED = "#6F7B86"
GRID = "#E4E9ED"
BLUE = "#75D7F0"
BLUE_DARK = "#28A9D0"
BLUE_LIGHT = "#BDEFFC"
PINK = "#FFA6C8"
PINK_DARK = "#EB6FA1"
PINK_LIGHT = "#FFD1E2"
BG = "#FBFCFD"
DPI = 200

METHOD_STYLE = {
    "nullspace": {
        "label": "Nullspace",
        "color": BLUE_DARK,
        "fill": BLUE_LIGHT,
        "marker": "o",
        "linewidth": 2.8,
        "markersize": 7.2,
    },
    "raw": {
        "label": "Raw control",
        "color": PINK_DARK,
        "fill": PINK_LIGHT,
        "marker": "D",
        "linewidth": 2.2,
        "markersize": 6.2,
    },
}


def setup_separability_style() -> None:
    """Apply the compact, title-free visual language used by the paper figures."""
    # Match the hidden-geometry figures for Latin text, with CJK as fallback.
    sans = ["DejaVu Sans"]
    if CJK_FONT:
        sans.append(CJK_FONT)
    sans += ["Arial", "Liberation Sans"]
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": sans,
        "font.size": 11.5,
        "axes.facecolor": BG,
        "figure.facecolor": BG,
        "axes.edgecolor": "#B7C0C8",
        "axes.labelcolor": INK,
        "axes.labelsize": 13.0,
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 11.5,
        "ytick.labelsize": 11.5,
        "legend.fontsize": 12.0,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "grid.alpha": 0.78,
        "lines.solid_capstyle": "round",
        "lines.dash_capstyle": "round",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def focused_auc_limits(
    values: Iterable[float],
    *,
    absolute_floor: float = 0.40,
    pad: float = 0.025,
) -> tuple[float, float]:
    """Return an honest but space-efficient AUC range on 0.05 increments."""
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return 0.50, 1.01
    lower = math.floor((min(finite) - pad) / 0.05) * 0.05
    lower = max(absolute_floor, min(lower, 0.85))
    # The small headroom keeps markers at AUC=1.0 from being clipped.
    return lower, 1.035
