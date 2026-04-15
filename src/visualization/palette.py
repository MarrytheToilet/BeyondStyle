"""Shared palette, style, and save utility."""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path

PALETTE = {
    "human":       "#A8D8EA",   # soft sky blue
    "ai":          "#F7C5CC",   # soft blush pink
    "human_dark":  "#4A90B8",   # emphasis blue
    "ai_dark":     "#D45B7A",   # emphasis pink
    "bg":          "#FAFBFD",
    "grid":        "#E6E9ED",
    "accent":      "#6C5CE7",   # purple accent
}

# Short aliases used throughout
H, A = PALETTE["human"], PALETTE["ai"]
H_D, A_D = PALETTE["human_dark"], PALETTE["ai_dark"]
BG = PALETTE["bg"]

# Custom colormaps: blue ↔ pink
BP_CMAP = LinearSegmentedColormap.from_list("bp", [H_D, "#F0F4F8", A_D])
BP_SEQ  = LinearSegmentedColormap.from_list("bp_seq", [H_D, "#C8B8DB", A_D])

DPI = 200


def setup_style():
    """Apply BeyondStyle matplotlib rcParams."""
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "axes.facecolor":    BG,
        "figure.facecolor":  BG,
        "axes.grid":         True,
        "grid.alpha":        0.30,
        "grid.color":        PALETTE["grid"],
        "grid.linewidth":    0.5,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.edgecolor":    "#CCCCCC",
        "axes.labelsize":    10,
        "axes.titlesize":    12,
        "axes.titleweight":  "bold",
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
    })


def save_fig(fig, path: Path, name: str) -> Path:
    """Save figure to *path/name* and return the full path."""
    out = Path(path) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=DPI, bbox_inches="tight", facecolor=BG)
    print(f"  ✓ {out}")
    return out
