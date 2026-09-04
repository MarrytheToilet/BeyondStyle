#!/usr/bin/env python3
"""Shared design system for the ACL manuscript figures.

One module owns the palette, the typography, the panel furniture, and the
page geometry so that every figure in ``ACL27/figures`` reads as one system.

Palette provenance
------------------
The categorical palette carries exactly two slots, because the paper's
categorical contrast is exactly two-valued: the lexical axis against the
syntactic axis.  Those two hues were validated with the data-visualisation
palette checker (lightness band, chroma floor, CVD separation, normal-vision
floor, and contrast against the paper surface all pass; worst adjacent CVD
separation is dE 9.2 under protanopia and 28.0 under tritanopia).

The discourse group is deliberately *not* a third categorical slot.  It is an
off-target reference series, so it wears a neutral slate that sits outside the
categorical lightness band on purpose and always carries a distinct open marker
plus a direct label.  Refitted-null material is background rather than a
series, so it wears neutral fills.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch

# --- page geometry -----------------------------------------------------------
# ACL two-column: \textwidth = 455.244pt, \columnwidth = 219.086pt (TeX points).
FULL_WIDTH = 6.300
COL_WIDTH = 3.031

# --- ink ---------------------------------------------------------------------
INK = "#1E2732"
INK_SOFT = "#3C4754"
MUTED = "#67717E"
FAINT = "#98A2AE"
GRID = "#E6ECF2"
RULE = "#C2CDD8"
SURFACE = "#FFFFFF"

# --- categorical slots (validated pair) --------------------------------------
LEX = "#2E7EB8"
LEX_MID = "#7FB4DC"
LEX_TINT = "#DCEBF7"
LEX_WASH = "#F1F8FD"

SYN = "#C9527F"
SYN_MID = "#E49BB6"
SYN_TINT = "#FAE0E9"
SYN_WASH = "#FEF4F7"

# --- reference / off-target / background -------------------------------------
DISC = "#44505E"
DISC_TINT = "#DDE3E9"
NULL_LINE = "#C8D2DC"
NULL_BAND = "#E9EEF3"

GROUP_COLOR = {"lexical": LEX, "syntactic": SYN, "discourse": DISC}
GROUP_TINT = {"lexical": LEX_TINT, "syntactic": SYN_TINT, "discourse": DISC_TINT}
GROUP_WASH = {"lexical": LEX_WASH, "syntactic": SYN_WASH, "discourse": "#F4F6F8"}
GROUP_MARKER = {"lexical": "o", "syntactic": "D", "discourse": "s"}
GROUP_LABEL = {"lexical": "Lexical", "syntactic": "Syntactic", "discourse": "Discourse"}


def use_paper_style() -> None:
    """Install the manuscript-wide Matplotlib defaults."""
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.titlesize": 7.6,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.6,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.6,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": RULE,
            "axes.linewidth": 0.6,
            "axes.facecolor": SURFACE,
            "figure.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.012,
            "lines.solid_capstyle": "round",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "pdf.compression": 9,
        }
    )


def clean(axis: plt.Axes, *, grid: str | None = "y") -> plt.Axes:
    """Strip the top/right spines and lay a recessive hairline grid."""
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.tick_params(width=0.55, length=2.2, color=FAINT, pad=1.8)
    if grid in ("x", "both"):
        axis.grid(axis="x", color=GRID, linewidth=0.5, zorder=0)
    if grid in ("y", "both"):
        axis.grid(axis="y", color=GRID, linewidth=0.5, zorder=0)
    axis.set_axisbelow(True)
    return axis


def panel(
    axis: plt.Axes,
    letter: str,
    title: str,
    *,
    y: float = 1.02,
    x: float = 0.0,
    dx: float | None = None,
) -> None:
    """Bold panel letter plus a sentence-case title, both on the ink tokens.

    ``dx`` is the gap between letter and title in axes units.  Narrow panels
    need a larger value because the same physical gap is a larger fraction of
    a narrow axis; leaving it unset derives the gap from the axis width.
    """
    axis.text(
        x, y, letter, transform=axis.transAxes, ha="left", va="bottom",
        fontsize=8.6, fontweight="bold", color=INK,
    )
    if dx is None:
        width_in = axis.get_figure().get_size_inches()[0] * axis.get_position().width
        dx = 0.115 / max(width_in, 0.35)
    axis.text(
        x + dx, y, title, transform=axis.transAxes, ha="left", va="bottom",
        fontsize=7.4, color=INK_SOFT,
    )


def zero_rule(axis: plt.Axes, *, vertical: bool = True, value: float = 0.0) -> None:
    """A single solid hairline for the null reference."""
    if vertical:
        axis.axvline(value, color=INK, linewidth=0.7, zorder=2)
    else:
        axis.axhline(value, color=INK, linewidth=0.7, zorder=2)


def axis_legend(
    axis: plt.Axes,
    handles: list,
    *,
    loc: str = "lower right",
    ncol: int = 1,
    **kwargs,
) -> None:
    defaults = dict(
        frameon=False,
        handlelength=1.25,
        handletextpad=0.34,
        columnspacing=0.75,
        labelspacing=0.28,
        borderpad=0.0,
        fontsize=6.4,
    )
    defaults.update(kwargs)
    axis.legend(handles=handles, loc=loc, ncol=ncol, **defaults)


def group_handle(group: str, *, label: str | None = None, filled: bool = True) -> Line2D:
    color = GROUP_COLOR[group]
    return Line2D(
        [0], [0],
        marker=GROUP_MARKER[group],
        color=color,
        markerfacecolor=color if filled else SURFACE,
        markeredgecolor=SURFACE if filled else color,
        markeredgewidth=0.6,
        markersize=3.6,
        linewidth=1.1,
        label=label or GROUP_LABEL[group],
    )


def chip(
    axis: plt.Axes,
    x: float,
    y: float,
    text: str,
    *,
    face: str,
    edge: str | None = None,
    color: str = INK,
    fontsize: float = 6.0,
    pad: float = 0.28,
    weight: str = "normal",
    ha: str = "center",
    va: str = "center",
    zorder: float = 6,
) -> None:
    """A rounded label chip, used by the schematic figure."""
    axis.text(
        x, y, text, ha=ha, va=va, fontsize=fontsize, color=color, zorder=zorder,
        fontweight=weight,
        bbox=dict(
            boxstyle=f"round,pad={pad},rounding_size=0.30",
            facecolor=face,
            edgecolor=edge or "none",
            linewidth=0.55,
        ),
    )


def card(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    face: str = SURFACE,
    edge: str = RULE,
    linewidth: float = 0.6,
    radius: float = 0.02,
    zorder: float = 1,
) -> FancyBboxPatch:
    """A rounded container used to group schematic content."""
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        facecolor=face, edgecolor=edge, linewidth=linewidth, zorder=zorder,
    )
    axis.add_patch(box)
    return box


def blank(axis: plt.Axes) -> plt.Axes:
    """A coordinate canvas with no visible axes, for schematic drawing."""
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_xticks([])
    axis.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        axis.spines[side].set_visible(False)
    return axis


def save(figure: plt.Figure, path, *, dpi: int = 400) -> None:
    """Write the vector figure, plus a raster preview under build/previews."""
    path.parent.mkdir(parents=True, exist_ok=True)
    preview_dir = path.parent.parent / "build" / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(path.with_suffix(".pdf"))
    figure.savefig(preview_dir / (path.stem + ".png"), dpi=dpi)
    plt.close(figure)
    print(f"wrote {path.with_suffix('.pdf')}")
