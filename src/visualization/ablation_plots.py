"""Part C — Denoising comparison plots.

Compares 3 validated methods: Raw MeanDiff, NullSpace, Fisher LDA.
All use the project's light-blue / light-pink palette.

Naming: fig09, fig10 (continues from Part B: fig05–fig08).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .palette import H, A, H_D, A_D, BG, BP_SEQ, save_fig, setup_style, L

# ── method styling (blue–pink spectrum) ──────────────────

METHOD_STYLE = {
    "raw":       {"color": H_D,    "label": "Raw MeanDiff", "marker": "o"},
    "nullspace": {"color": H,      "label": "NullSpace",    "marker": "^"},
    "fisher":    {"color": A_D,    "label": "Fisher LDA",   "marker": "D"},
}

_METHOD_ORDER = ["raw", "nullspace", "fisher"]


def _denoise_of(method_key: str) -> str:
    for dn in ("fisher", "nullspace"):
        if method_key.endswith(f"_{dn}"):
            return dn
    return "raw"


def _style(dn: str):
    return METHOD_STYLE.get(dn, METHOD_STYLE["raw"])


def _unpack(metrics):
    return metrics.get("methods", metrics)


# ─────────────────────────────────────────────────────────
# fig09 · Denoising Method Comparison
#
# Three panels:
#   Left   — AUC bar chart  (all methods, best layer)
#   Centre — Accuracy bar chart (shows Fisher at 100%)
#   Right  — Cosine-vs-raw  (shows direction novelty)
# ─────────────────────────────────────────────────────────
def plot_denoise_comparison(metrics, layers, method_keys=None, fig_dir=None, language: str = "en"):
    """Three-panel summary comparing denoising methods at their best layer."""
    setup_style()
    m = _unpack(metrics)
    if method_keys is None:
        method_keys = sorted(m.keys())

    # Collect stats per method
    records = []
    for mk in method_keys:
        dn = _denoise_of(mk)
        if dn not in METHOD_STYLE:
            continue
        best_ly = max(layers, key=lambda ly: m[mk].get(str(ly), {}).get("auc", 0))
        info = m[mk].get(str(best_ly), {})
        di = info.get("denoise_info", {})
        cos = di.get("cosine_vs_raw", di.get("cosine_shift", 1.0 if dn == "raw" else 0))
        records.append({
            "dn": dn, "mk": mk, "layer": best_ly,
            "auc": info.get("auc", 0),
            "acc": info.get("accuracy", 0),
            "cos": abs(cos),
        })

    # Sort by canonical order
    order_map = {k: i for i, k in enumerate(_METHOD_ORDER)}
    records.sort(key=lambda r: order_map.get(r["dn"], 99))

    n = len(records)
    xs = np.arange(n)
    colors = [_style(r["dn"])["color"] for r in records]
    labels = [_style(r["dn"])["label"] for r in records]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    bar_kw = dict(edgecolor="white", linewidth=1.2, width=0.55)

    # ── AUC ──
    ax = axes[0]
    vals = [r["auc"] for r in records]
    bars = ax.bar(xs, vals, color=colors, **bar_kw)
    ax.set_ylim(0.95, 1.005)
    ax.set_ylabel(L(language, "auc"))
    ax.set_title(L(language, "class_auc"))
    for bar, v, r in zip(bars, vals, records):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.001,
                f"{v:.4f}\n(L{r['layer']})", ha="center", fontsize=8.5, fontweight="bold")

    # ── Accuracy ──
    ax = axes[1]
    vals = [r["acc"] for r in records]
    bars = ax.bar(xs, vals, color=colors, **bar_kw)
    ax.set_ylim(0.97, 1.005)
    ax.set_ylabel(L(language, "accuracy"))
    ax.set_title(L(language, "best_layer_acc"))
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.001,
                f"{v:.1%}", ha="center", fontsize=9, fontweight="bold")

    # ── Cosine-vs-raw ──
    ax = axes[2]
    vals = [r["cos"] for r in records]
    bars = ax.bar(xs, vals, color=colors, **bar_kw)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel(L(language, "cos_v_raw"))
    ax.set_title(L(language, "dir_sim_raw"))
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02,
                f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    ax.axhline(1.0, color="#DDD", ls="--", lw=0.8, zorder=0)

    for ax in axes:
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, fontsize=9)

    fig.suptitle(L(language, "fig09_suptitle"),
                 fontsize=12, fontweight="bold", y=1.03)
    plt.tight_layout()
    if fig_dir:
        save_fig(fig, fig_dir, "fig09_denoise_comparison.png")
    return fig


# ─────────────────────────────────────────────────────────
# fig10 · Projection Violins (Human vs AI per method)
#
# One panel per method; light-blue violins for Human,
# light-pink violins for AI.
# ─────────────────────────────────────────────────────────
def plot_projection_violins(states_loader, method_keys=None, fig_dir=None, language: str = "en"):
    """Violin plot of H vs A projections onto each method's direction.

    Args:
        states_loader: dict {method_key: (v_direction, h_proj_array, a_proj_array)}
        method_keys: ordered list of keys to plot.
        fig_dir: output directory.
    """
    setup_style()
    data = states_loader if isinstance(states_loader, dict) else {}

    if method_keys is None:
        method_keys = sorted(data.keys())

    # Filter to only the 3 kept methods, in canonical order
    filtered = []
    for mk in method_keys:
        dn = _denoise_of(mk)
        if dn in METHOD_STYLE and mk in data:
            filtered.append(mk)
    order_map = {k: i for i, k in enumerate(_METHOD_ORDER)}
    filtered.sort(key=lambda mk: order_map.get(_denoise_of(mk), 99))

    n = len(filtered)
    if n == 0:
        return

    fig, axes = plt.subplots(1, n, figsize=(3.8 * n, 5), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, mk in zip(axes, filtered):
        dn = _denoise_of(mk)
        s = _style(dn)
        _, hp, ap = data[mk]

        parts_h = ax.violinplot(hp, positions=[0], showextrema=False, widths=0.7)
        parts_a = ax.violinplot(ap, positions=[1], showextrema=False, widths=0.7)

        for pc in parts_h["bodies"]:
            pc.set_facecolor(H)
            pc.set_edgecolor(H_D)
            pc.set_alpha(0.85)
        for pc in parts_a["bodies"]:
            pc.set_facecolor(A)
            pc.set_edgecolor(A_D)
            pc.set_alpha(0.85)

        # Quartile lines + median dot
        for pos, vals, dark in [(0, hp, H_D), (1, ap, A_D)]:
            q1, med, q3 = np.percentile(vals, [25, 50, 75])
            ax.vlines(pos, q1, q3, color=dark, linewidth=2.5, zorder=4)
            ax.scatter([pos], [med], color=dark, s=40, zorder=5, edgecolors="white", linewidths=0.6)

        ax.set_xticks([0, 1])
        ax.set_xticklabels([L(language, "human"), L(language, "ai")], fontsize=10)
        ax.set_title(s["label"], fontsize=11, fontweight="bold", color=s["color"])

    fig.suptitle(L(language, "fig10_title"),
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    if fig_dir:
        save_fig(fig, fig_dir, "fig10_projection_violins.png")
    return fig
