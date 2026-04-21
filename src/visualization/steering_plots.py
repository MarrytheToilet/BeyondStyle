"""Part B — Steering results plots."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import roc_curve

from .palette import H, A, H_D, A_D, BP_SEQ, save_fig, L


# ─────────────────────────────────────────────────────────
# B1 · Layer-wise metrics (Hull Centroid Dist / Fisher / AUC)
# ─────────────────────────────────────────────────────────
def plot_layer_metrics(metrics, layers, method="mean_diff", fig_dir=None, language: str = "en"):
    """Bar charts: normalized hull centroid distance (primary), Fisher ratio, AUC."""
    def _get(key, default=0):
        return [metrics["methods"][method][str(ly)].get(key, default) for ly in layers]

    hcd_norm = _get("hull_centroid_dist_norm")
    fr_vals = _get("fisher_ratio")
    auc_vals = _get("auc")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    method_label = "MeanDiff"
    fig.suptitle(L(language, "fig05_suptitle", method=method_label),
                 fontsize=14, fontweight="bold", y=1.02)
    w = 1.3

    ax = axes[0]
    ax.bar(layers, hcd_norm, width=w, color=H, edgecolor=H_D, lw=0.5, alpha=0.85)
    best_i = int(np.argmax(hcd_norm))
    ax.bar(layers[best_i], hcd_norm[best_i], width=w, color=A, edgecolor=A_D,
           lw=0.5, alpha=0.9)
    ax.set_title(L(language, "norm_hull"), pad=6, fontweight="bold")
    ax.set_xlabel(L(language, "layer"))
    ax.set_ylabel(L(language, "centroid_per_diam"))
    ax.annotate(f"{L(language, 'best')}: L{layers[best_i]}\n{hcd_norm[best_i]:.3f}",
                xy=(layers[best_i], hcd_norm[best_i]),
                xytext=(layers[best_i] + 4, hcd_norm[best_i] * 0.85),
                fontsize=8, fontweight="bold", color=A_D,
                arrowprops=dict(arrowstyle="->", color=A_D, lw=1))

    ax = axes[1]
    ax.bar(layers, fr_vals, width=w, color=H, edgecolor=H_D, lw=0.5, alpha=0.85)
    best_i = int(np.argmax(fr_vals))
    ax.bar(layers[best_i], fr_vals[best_i], width=w, color=A, edgecolor=A_D,
           lw=0.5, alpha=0.9)
    ax.set_title(L(language, "fisher_ratio"), pad=6)
    ax.set_xlabel(L(language, "layer"))
    ax.set_ylabel(L(language, "j_higher"))

    ax = axes[2]
    ax.bar(layers, auc_vals, width=w, color=H, edgecolor=H_D, lw=0.5, alpha=0.85)
    best_i = int(np.argmax(auc_vals))
    ax.bar(layers[best_i], auc_vals[best_i], width=w, color=A, edgecolor=A_D,
           lw=0.5, alpha=0.9)
    ax.set_ylim(0.90, 1.005)
    ax.set_title(L(language, "auc_ref"), pad=6)
    ax.set_xlabel(L(language, "layer"))
    ax.set_ylabel(L(language, "auc"))

    plt.tight_layout()
    if fig_dir:
        save_fig(fig, fig_dir, "fig05_layer_metrics.png")
    return fig


# ─────────────────────────────────────────────────────────
# B2 · ROC curves (blue → pink gradient)
# ─────────────────────────────────────────────────────────
def plot_roc_curves(states, md_models, metrics, layers, fig_dir=None, language: str = "en"):
    """ROC curves for all layers, coloured on a blue→pink gradient."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    norm_ly = plt.Normalize(vmin=min(layers), vmax=max(layers))

    for ly in layers:
        ps = states[ly]
        v = md_models[ly].get_direction().cpu().float()
        hp = (ps.human.float() @ v).numpy()
        ap = (ps.ai.float() @ v).numpy()
        scores = np.concatenate([hp, ap])
        lbls = np.concatenate([np.zeros(len(hp)), np.ones(len(ap))])
        fpr, tpr, _ = roc_curve(lbls, scores)
        auc_val = metrics["methods"]["mean_diff"][str(ly)]["auc"]
        ax.plot(fpr, tpr, color=BP_SEQ(norm_ly(ly)), lw=1.6, alpha=0.85,
                label=f"L{ly} ({auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "--", color="#bbb", lw=0.8, alpha=0.5)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_title(L(language, "fig06_title"), pad=8)
    ax.set_xlabel(L(language, "fpr"))
    ax.set_ylabel(L(language, "tpr"))
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.85, edgecolor="#ddd",
              title=L(language, "layer_auc"), title_fontsize=8)

    sm = plt.cm.ScalarMappable(cmap=BP_SEQ, norm=norm_ly)
    cbar = plt.colorbar(sm, ax=ax, shrink=0.65, pad=0.02)
    cbar.set_label(L(language, "layer"), fontsize=9)

    plt.tight_layout()
    if fig_dir:
        save_fig(fig, fig_dir, "fig06_roc_curves.png")
    return fig


# ─────────────────────────────────────────────────────────
# B3 · Separation landscape (violin)
# ─────────────────────────────────────────────────────────
def plot_separation_landscape(states, md_models, layers, fig_dir=None, language: str = "en"):
    """Violin plots of z-scored projections at each layer."""
    fig, ax = plt.subplots(figsize=(12, 5))
    positions = np.arange(len(layers))

    for i, ly in enumerate(layers):
        ps = states[ly]
        v = md_models[ly].get_direction().cpu().float()
        hp = (ps.human.float() @ v).numpy()
        ap = (ps.ai.float() @ v).numpy()

        all_vals = np.concatenate([hp, ap])
        mu, sigma = all_vals.mean(), all_vals.std() + 1e-10
        hp_z = (hp - mu) / sigma
        ap_z = (ap - mu) / sigma

        vp_h = ax.violinplot(hp_z, positions=[i - 0.18], widths=0.3,
                              showmedians=True, showextrema=False)
        vp_a = ax.violinplot(ap_z, positions=[i + 0.18], widths=0.3,
                              showmedians=True, showextrema=False)

        for pc in vp_h["bodies"]:
            pc.set_facecolor(H)
            pc.set_edgecolor(H_D)
            pc.set_alpha(0.6)
        vp_h["cmedians"].set_color(H_D)
        for pc in vp_a["bodies"]:
            pc.set_facecolor(A)
            pc.set_edgecolor(A_D)
            pc.set_alpha(0.6)
        vp_a["cmedians"].set_color(A_D)

    ax.set_xticks(positions)
    ax.set_xticklabels([f"L{ly}" for ly in layers])
    ax.set_title(L(language, "fig08_title"), pad=8)
    ax.set_ylabel(L(language, "norm_proj"))
    ax.set_xlabel(L(language, "layer"))
    ax.legend(handles=[Patch(fc=H, ec=H_D, alpha=0.6, label=L(language, "human")),
                        Patch(fc=A, ec=A_D, alpha=0.6, label=L(language, "ai"))],
              fontsize=9, framealpha=0.8, edgecolor="#ddd")

    plt.tight_layout()
    if fig_dir:
        save_fig(fig, fig_dir, "fig08_separation_landscape.png")
    return fig
