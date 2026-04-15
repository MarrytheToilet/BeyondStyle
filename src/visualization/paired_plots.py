"""Part A — Paired data analysis plots."""

import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.decomposition import PCA

from .palette import H, A, H_D, A_D, BG, BP_CMAP, PALETTE, save_fig


# ─────────────────────────────────────────────────────────
# A1 · Projection distributions (top-k layers)
# ─────────────────────────────────────────────────────────
def plot_projection_distributions(states, md_models, metrics, layers,
                                  top_k=4, fig_dir=None):
    """Histogram of human/AI projections onto v_src for the best *top_k* layers."""
    ranked = sorted(layers,
                    key=lambda ly: metrics["methods"]["mean_diff"][str(ly)]["auc"],
                    reverse=True)[:top_k]

    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    fig.suptitle("Projection onto Steering Direction  $v_{src}$",
                 fontsize=14, fontweight="bold", y=1.0)

    for idx, ly in enumerate(ranked):
        ax = axes.flat[idx]
        ps = states[ly]
        v = md_models[ly].get_direction().cpu().float()
        hp = (ps.human.float() @ v).numpy()
        ap = (ps.ai.float() @ v).numpy()

        lo, hi = min(hp.min(), ap.min()), max(hp.max(), ap.max())
        bins = np.linspace(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo), 45)

        ax.hist(hp, bins=bins, alpha=0.55, color=H, edgecolor=H_D, lw=0.4, label="Human")
        ax.hist(ap, bins=bins, alpha=0.55, color=A, edgecolor=A_D, lw=0.4, label="AI")
        ax.axvline(hp.mean(), color=H_D, ls="--", lw=1.4, alpha=0.8)
        ax.axvline(ap.mean(), color=A_D, ls="--", lw=1.4, alpha=0.8)

        auc = metrics["methods"]["mean_diff"][str(ly)]["auc"]
        ax.set_title(f"Layer {ly}   AUC = {auc:.3f}", pad=6)
        ax.set_xlabel("Projection")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8, loc="upper right", framealpha=0.75, edgecolor="#ddd")

    plt.tight_layout()
    if fig_dir:
        save_fig(fig, fig_dir, "fig01_projection_distributions.png")
    return fig


# ─────────────────────────────────────────────────────────
# A2 · PCA embedding + v_src alignment
# ─────────────────────────────────────────────────────────
def plot_pca_embedding(states, md_models, best_layer, fig_dir=None):
    """PCA scatter (left) + v_src alignment with principal components (right)."""
    ps = states[best_layer]
    combined = torch.cat([ps.human, ps.ai], dim=0).float().numpy()
    labels = np.array([0] * len(ps.human) + [1] * len(ps.ai))
    v_raw = md_models[best_layer].get_direction().cpu().float().numpy()

    pca2 = PCA(n_components=2).fit(combined)
    proj = pca2.transform(combined)
    var = pca2.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                              gridspec_kw={"width_ratios": [1.3, 1]})

    # ── scatter ──
    ax = axes[0]
    ax.scatter(proj[labels == 0, 0], proj[labels == 0, 1],
               c=H, edgecolor=H_D, s=28, alpha=0.55, lw=0.3, label="Human", zorder=2)
    ax.scatter(proj[labels == 1, 0], proj[labels == 1, 1],
               c=A, edgecolor=A_D, s=28, alpha=0.55, lw=0.3, label="AI", zorder=2)

    v_pca = pca2.transform(v_raw.reshape(1, -1))[0]
    origin = proj.mean(axis=0)
    scale = np.linalg.norm(proj.std(axis=0)) * 2.2
    v_norm = v_pca / (np.linalg.norm(v_pca) + 1e-10) * scale
    ax.annotate("", xy=(origin[0] + v_norm[0], origin[1] + v_norm[1]),
                xytext=(origin[0], origin[1]),
                arrowprops=dict(arrowstyle="->", color="#333", lw=2.2))
    ax.text(origin[0] + v_norm[0] * 1.12, origin[1] + v_norm[1] * 1.12,
            "$v_{src}$", fontsize=12, fontweight="bold", ha="center")

    ax.set_title(f"PCA — Layer {best_layer}   (Var explained: {var.sum():.1%})", pad=8)
    ax.set_xlabel(f"PC1 ({var[0]:.1%})")
    ax.set_ylabel(f"PC2 ({var[1]:.1%})")
    ax.legend(fontsize=9, markerscale=1.5, framealpha=0.8, edgecolor="#ddd")

    # ── alignment bars ──
    ax = axes[1]
    n_pc = 10
    pca_full = PCA(n_components=n_pc).fit(combined)
    cos_vals = [
        np.dot(v_raw, pca_full.components_[i])
        / (np.linalg.norm(v_raw) * np.linalg.norm(pca_full.components_[i]))
        for i in range(n_pc)
    ]
    bar_colors = [PALETTE["accent"] if abs(c) > 0.15 else "#C8CCD0" for c in cos_vals]
    ax.bar(range(n_pc), cos_vals, color=bar_colors, edgecolor="white", lw=0.5, alpha=0.85)
    ax.set_xticks(range(n_pc))
    ax.set_xticklabels([f"PC{i + 1}" for i in range(n_pc)])
    ax.axhline(0, color="#bbb", lw=0.6)
    ax.set_title("$v_{src}$  Alignment with PCA Components", pad=8)
    ax.set_ylabel("Cosine Similarity")
    ax.text(0.95, 0.92, f"cos($v_{{src}}$, PC1) = {cos_vals[0]:.3f}",
            transform=ax.transAxes, ha="right", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ddd", alpha=0.9))

    plt.tight_layout()
    if fig_dir:
        save_fig(fig, fig_dir, "fig02_pca_embedding.png")
    return fig


# ─────────────────────────────────────────────────────────
# A3 · Direction similarity heatmap
# ─────────────────────────────────────────────────────────
def plot_direction_similarity(md_models, layers, fig_dir=None):
    """Cosine similarity matrix of v_src across layers (blue–white–pink)."""
    n = len(layers)
    dirs = {ly: md_models[ly].get_direction().cpu().float().numpy() for ly in layers}
    cos_mat = np.zeros((n, n))
    for i, li in enumerate(layers):
        for j, lj in enumerate(layers):
            cos_mat[i, j] = (
                np.dot(dirs[li], dirs[lj])
                / (np.linalg.norm(dirs[li]) * np.linalg.norm(dirs[lj]))
            )

    # Use vmin=0 since all cosine sims are non-negative → full blue-to-pink range
    from matplotlib.colors import LinearSegmentedColormap
    hm_cmap = LinearSegmentedColormap.from_list(
        "bp_hm", [H_D, "#E8EDF2", "#F5F0F2", A_D])

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cos_mat, cmap=hm_cmap, vmin=0, vmax=1, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"L{ly}" for ly in layers])
    ax.set_yticklabels([f"L{ly}" for ly in layers])
    for i in range(n):
        for j in range(n):
            c = cos_mat[i, j]
            txt_col = "white" if c > 0.75 or c < 0.25 else "#333"
            ax.text(j, i, f"{c:.2f}", ha="center", va="center",
                    fontsize=7, color=txt_col)
    cbar = plt.colorbar(im, ax=ax, shrink=0.78, pad=0.02)
    cbar.set_label("Cosine Similarity", fontsize=10)
    ax.set_title("$v_{src}$  Cosine Similarity Across Layers", pad=10)

    plt.tight_layout()
    if fig_dir:
        save_fig(fig, fig_dir, "fig03_direction_similarity.png")
    return fig


# ─────────────────────────────────────────────────────────
# A4 · Steering trajectory: AI moves toward / away from human anchor
# ─────────────────────────────────────────────────────────
def plot_embedding_shift(states, md_models, best_layer,
                         alphas=None, fig_dir=None):
    """Show how steering pushes AI toward (−α) or away from (+α) human anchors.

    Left: PCA scatter — human anchors fixed, AI centroid traces a trajectory
          across α values.
    Right: Mean L2 distance between each (AI_steered, Human) pair vs α.
    """
    if alphas is None:
        alphas = [-5, -3, -2, -1, 0, 1, 2, 3, 5]

    ps = states[best_layer]
    h_vecs = ps.human.float()          # (N, D)
    a_vecs = ps.ai.float()             # (N, D)
    v = md_models[best_layer].get_direction().cpu().float()  # (D,)

    combined = torch.cat([h_vecs, a_vecs], dim=0).numpy()
    pca2 = PCA(n_components=2).fit(combined)
    var = pca2.explained_variance_ratio_

    h_proj = pca2.transform(h_vecs.numpy())   # (N, 2)
    a_proj = pca2.transform(a_vecs.numpy())   # (N, 2)

    # steered AI at each α
    a_steered_projs = {}
    distances = []
    for alpha in alphas:
        a_st = (a_vecs + alpha * v).numpy()
        a_st_proj = pca2.transform(a_st)
        a_steered_projs[alpha] = a_st_proj
        # per-sample L2 distance to paired human
        d = np.linalg.norm(a_st_proj - h_proj, axis=1).mean()
        distances.append(d)

    # ── color gradient for α: blue (neg) → gray (0) → pink (pos) ──
    from matplotlib.colors import LinearSegmentedColormap
    alpha_cmap = LinearSegmentedColormap.from_list(
        "alpha_bp", [H_D, "#B0B8C0", A_D])
    a_min, a_max = min(alphas), max(alphas)
    a_norm = plt.Normalize(vmin=a_min, vmax=a_max)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # ── LEFT: PCA trajectory ──
    ax = axes[0]
    # human anchor cloud (fixed)
    ax.scatter(h_proj[:, 0], h_proj[:, 1], c=H, edgecolor=H_D,
               s=24, alpha=0.35, lw=0.2, zorder=1)
    h_c = h_proj.mean(0)
    ax.scatter(*h_c, c=H_D, s=140, marker="D", edgecolor="white",
               lw=1.5, zorder=10, label="Human (anchor)")

    # AI centroids at each α, connected by trajectory line
    centroids = np.array([a_steered_projs[a].mean(0) for a in alphas])
    ax.plot(centroids[:, 0], centroids[:, 1], color="#999",
            ls="--", lw=1.2, alpha=0.6, zorder=3)
    for i, alpha in enumerate(alphas):
        c = alpha_cmap(a_norm(alpha))
        marker = "o" if alpha != 0 else "s"
        sz = 100 if alpha == 0 else 60
        zord = 8 if alpha == 0 else 5
        ax.scatter(centroids[i, 0], centroids[i, 1], c=[c],
                   edgecolor="white", s=sz, marker=marker,
                   lw=1.0, zorder=zord)
        # label a few key α values
        if alpha in (min(alphas), 0, max(alphas)):
            label = f"α={alpha:+.0f}" if alpha != 0 else "α=0\n(baseline)"
            ax.annotate(label,
                        xy=(centroids[i, 0], centroids[i, 1]),
                        xytext=(8, 8), textcoords="offset points",
                        fontsize=8, color="#555",
                        arrowprops=dict(arrowstyle="-", color="#bbb", lw=0.6))

    # arrow: toward human at negative α
    ax.annotate("", xy=centroids[0], xytext=centroids[len(alphas) // 2],
                arrowprops=dict(arrowstyle="-|>", color=H_D, lw=1.8,
                                mutation_scale=14, alpha=0.7))
    # arrow: away from human at positive α
    ax.annotate("", xy=centroids[-1], xytext=centroids[len(alphas) // 2],
                arrowprops=dict(arrowstyle="-|>", color=A_D, lw=1.8,
                                mutation_scale=14, alpha=0.7))

    ax.set_title(f"AI Centroid Trajectory — Layer {best_layer}", pad=8)
    ax.set_xlabel(f"PC1 ({var[0]:.1%})")
    ax.set_ylabel(f"PC2 ({var[1]:.1%})")
    ax.legend(fontsize=8, framealpha=0.8, edgecolor="#ddd", loc="best")

    sm = plt.cm.ScalarMappable(cmap=alpha_cmap, norm=a_norm)
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Steering strength α", fontsize=9)

    # ── RIGHT: distance curve ──
    ax = axes[1]
    colors_line = [alpha_cmap(a_norm(a)) for a in alphas]
    ax.plot(alphas, distances, color="#999", lw=1.5, zorder=1)
    for i, alpha in enumerate(alphas):
        ax.scatter(alpha, distances[i], c=[colors_line[i]],
                   edgecolor="white", s=80, lw=1.0, zorder=3)

    # shade regions
    zero_idx = alphas.index(0)
    d0 = distances[zero_idx]
    ax.axhline(d0, color="#bbb", ls=":", lw=0.8, alpha=0.6)
    ax.fill_between([a_min, 0], ax.get_ylim()[0], ax.get_ylim()[1],
                    alpha=0.06, color=H_D)
    ax.fill_between([0, a_max], ax.get_ylim()[0], ax.get_ylim()[1],
                    alpha=0.06, color=A_D)
    ax.text(a_min / 2, ax.get_ylim()[1] * 0.97, "← toward Human",
            ha="center", fontsize=8, color=H_D, fontstyle="italic")
    ax.text(a_max / 2, ax.get_ylim()[1] * 0.97, "away from Human →",
            ha="center", fontsize=8, color=A_D, fontstyle="italic")

    ax.set_title("Mean Distance to Human Anchor vs α", pad=8)
    ax.set_xlabel("Steering strength α")
    ax.set_ylabel("Mean L2 distance (PCA space)")

    fig.suptitle("Steering Trajectory: AI Continuation vs Human Anchor",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    if fig_dir:
        save_fig(fig, fig_dir, "fig04_embedding_shift.png")
    return fig
