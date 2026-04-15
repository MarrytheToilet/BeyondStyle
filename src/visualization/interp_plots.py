"""
Visualization for steering vector interpretability analysis.

Figures:
  fig_interp01 — Logit Lens: top tokens promoted/suppressed by v_src
  fig_interp02 — Token projection heatmap: token × source → mean projection
  fig_interp03 — POS-tag projection: mean v_src projection per POS tag
  fig_interp04 — Dependency-role projection: mean v_src projection per dep label
  fig_interp05 — Syntactic feature correlations: bar chart of Pearson r
  fig_interp06 — POS distribution comparison: Human vs AI
  fig_interp07 — Cross-method logit lens comparison
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from typing import Dict, List, Optional

from .palette import setup_style, save_fig, PALETTE, H, A, H_D, A_D, BG


def plot_logit_lens(results: dict, fig_dir: Path, top_k: int = 30):
    """fig_interp01: Horizontal bar chart of top tokens from logit lens."""
    setup_style()

    for key, data in results.get("logit_lens", {}).items():
        fig, (ax_ai, ax_hu) = plt.subplots(1, 2, figsize=(14, max(8, top_k * 0.28)))
        fig.suptitle(f"Logit Lens — {key}", fontsize=14, fontweight="bold", y=1.01)

        # AI-ward (positive logits)
        tokens_ai = [d["token"] for d in data["top_positive_ai"][:top_k]]
        logits_ai = [d["logit"] for d in data["top_positive_ai"][:top_k]]
        y_pos = range(len(tokens_ai))
        ax_ai.barh(y_pos, logits_ai, color=A_D, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax_ai.set_yticks(y_pos)
        ax_ai.set_yticklabels(tokens_ai, fontsize=8, fontfamily="monospace")
        ax_ai.invert_yaxis()
        ax_ai.set_xlabel("Logit (v_src → AI direction)")
        ax_ai.set_title("Tokens promoted by v_src (→ AI)", fontsize=11)

        # Human-ward (negative logits)
        tokens_hu = [d["token"] for d in data["top_negative_human"][:top_k]]
        logits_hu = [d["logit"] for d in data["top_negative_human"][:top_k]]
        y_pos = range(len(tokens_hu))
        ax_hu.barh(y_pos, logits_hu, color=H_D, alpha=0.85, edgecolor="white", linewidth=0.5)
        ax_hu.set_yticks(y_pos)
        ax_hu.set_yticklabels(tokens_hu, fontsize=8, fontfamily="monospace")
        ax_hu.invert_yaxis()
        ax_hu.set_xlabel("Logit (v_src → Human direction)")
        ax_hu.set_title("Tokens promoted by −v_src (→ Human)", fontsize=11)

        fig.tight_layout()
        save_fig(fig, fig_dir, f"fig_interp01_logit_lens_{key}.png")
        plt.close(fig)


def plot_token_projection(results: dict, fig_dir: Path, top_k: int = 25):
    """fig_interp02: Top tokens by mean projection (AI vs Human side)."""
    setup_style()

    for key, data in results.get("token_projection", {}).items():
        ai_tokens = data.get("top_ai_tokens", [])[:top_k]
        human_tokens = data.get("top_human_tokens", [])[:top_k]

        fig, (ax_ai, ax_hu) = plt.subplots(1, 2, figsize=(14, max(7, top_k * 0.28)))
        fig.suptitle(f"Token-Level Projection — {key}", fontsize=14, fontweight="bold", y=1.01)

        # AI-ward tokens
        if ai_tokens:
            tokens = [d["token"] for d in ai_tokens]
            projs = [d["mean_proj"] for d in ai_tokens]
            counts = [d["count"] for d in ai_tokens]
            y = range(len(tokens))
            bars = ax_ai.barh(y, projs, color=A_D, alpha=0.85, edgecolor="white", linewidth=0.5)
            ax_ai.set_yticks(y)
            ax_ai.set_yticklabels([f"{t}  (n={c})" for t, c in zip(tokens, counts)],
                                   fontsize=8, fontfamily="monospace")
            ax_ai.invert_yaxis()
            ax_ai.set_xlabel("Mean projection onto v_src")
            ax_ai.set_title("Highest projection (AI-like context)", fontsize=11)

        # Human-ward tokens
        if human_tokens:
            tokens = [d["token"] for d in human_tokens]
            projs = [d["mean_proj"] for d in human_tokens]
            counts = [d["count"] for d in human_tokens]
            y = range(len(tokens))
            bars = ax_hu.barh(y, projs, color=H_D, alpha=0.85, edgecolor="white", linewidth=0.5)
            ax_hu.set_yticks(y)
            ax_hu.set_yticklabels([f"{t}  (n={c})" for t, c in zip(tokens, counts)],
                                   fontsize=8, fontfamily="monospace")
            ax_hu.invert_yaxis()
            ax_hu.set_xlabel("Mean projection onto v_src")
            ax_hu.set_title("Lowest projection (Human-like context)", fontsize=11)

        fig.tight_layout()
        save_fig(fig, fig_dir, f"fig_interp02_token_projection_{key}.png")
        plt.close(fig)


def plot_pos_projection(results: dict, fig_dir: Path):
    """fig_interp03: Mean projection per POS tag (sorted by mean)."""
    setup_style()

    for key, data in results.get("syntactic", {}).items():
        pos = data.get("pos_stats", {})
        if not pos:
            continue

        tags = sorted(pos.keys(), key=lambda t: pos[t]["mean"], reverse=True)
        means = [pos[t]["mean"] for t in tags]
        stds = [pos[t]["std"] for t in tags]
        counts = [pos[t]["count"] for t in tags]

        fig, ax = plt.subplots(figsize=(12, max(5, len(tags) * 0.35)))
        colors = [A_D if m > 0 else H_D for m in means]
        y = range(len(tags))
        ax.barh(y, means, xerr=stds, color=colors, alpha=0.8,
                edgecolor="white", linewidth=0.5, capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{t}  (n={c})" for t, c in zip(tags, counts)],
                           fontsize=9)
        ax.invert_yaxis()
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Mean projection onto v_src")
        ax.set_title(f"POS Tag Projection — {key}", fontsize=13, fontweight="bold")

        fig.tight_layout()
        save_fig(fig, fig_dir, f"fig_interp03_pos_projection_{key}.png")
        plt.close(fig)


def plot_dep_projection(results: dict, fig_dir: Path, top_k: int = 25):
    """fig_interp04: Mean projection per dependency label."""
    setup_style()

    for key, data in results.get("syntactic", {}).items():
        dep = data.get("dep_stats", {})
        if not dep:
            continue

        # Sort by absolute mean, take top-k
        tags = sorted(dep.keys(), key=lambda t: abs(dep[t]["mean"]), reverse=True)[:top_k]
        tags = sorted(tags, key=lambda t: dep[t]["mean"], reverse=True)
        means = [dep[t]["mean"] for t in tags]
        stds = [dep[t]["std"] for t in tags]
        counts = [dep[t]["count"] for t in tags]

        fig, ax = plt.subplots(figsize=(12, max(5, len(tags) * 0.35)))
        colors = [A_D if m > 0 else H_D for m in means]
        y = range(len(tags))
        ax.barh(y, means, xerr=stds, color=colors, alpha=0.8,
                edgecolor="white", linewidth=0.5, capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{t}  (n={c})" for t, c in zip(tags, counts)],
                           fontsize=9)
        ax.invert_yaxis()
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Mean projection onto v_src")
        ax.set_title(f"Dependency Label Projection — {key}", fontsize=13, fontweight="bold")

        fig.tight_layout()
        save_fig(fig, fig_dir, f"fig_interp04_dep_projection_{key}.png")
        plt.close(fig)


def plot_syntactic_correlations(results: dict, fig_dir: Path):
    """fig_interp05: Sentence-level feature correlations with mean projection."""
    setup_style()

    # Collect all keys for comparison
    all_keys = list(results.get("syntactic", {}).keys())
    if not all_keys:
        return

    # Get union of all feature names
    feat_names = set()
    for key in all_keys:
        corr = results["syntactic"][key].get("sentence_correlations", {})
        feat_names.update(corr.keys())
    feat_names = sorted(feat_names)

    if not feat_names:
        return

    fig, ax = plt.subplots(figsize=(12, max(5, len(feat_names) * 0.4)))

    bar_width = 0.8 / max(len(all_keys), 1)
    colors = [A_D, PALETTE["accent"], H_D, "#E67E22", "#27AE60", "#2C3E50"]

    for i, key in enumerate(all_keys):
        corr = results["syntactic"][key].get("sentence_correlations", {})
        vals = [corr.get(fn, 0) for fn in feat_names]
        y = np.arange(len(feat_names)) + i * bar_width
        ax.barh(y, vals, height=bar_width, label=key,
                color=colors[i % len(colors)], alpha=0.8, edgecolor="white", linewidth=0.3)

    ax.set_yticks(np.arange(len(feat_names)) + bar_width * (len(all_keys) - 1) / 2)
    ax.set_yticklabels(feat_names, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Pearson r (feature vs mean projection)")
    ax.set_title("Syntactic Feature Correlations with v_src Projection",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")

    # Add R² annotations
    for key in all_keys:
        r2 = results["syntactic"][key].get("syntactic_r_squared", 0)
        ax.text(0.98, 0.02 + all_keys.index(key) * 0.04,
                f"{key}: R²={r2:.3f}",
                transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
                color=colors[all_keys.index(key) % len(colors)])

    fig.tight_layout()
    save_fig(fig, fig_dir, "fig_interp05_syntactic_correlations.png")
    plt.close(fig)


def plot_pos_distribution(results: dict, fig_dir: Path, top_k: int = 15):
    """fig_interp06: Side-by-side POS distributions for Human vs AI."""
    setup_style()

    for key, data in results.get("syntactic", {}).items():
        pos_dist = data.get("pos_distribution", {})
        if not pos_dist or "human" not in pos_dist or "ai" not in pos_dist:
            continue

        # Merge POS tags and sort by total frequency
        all_pos = set(list(pos_dist["human"].keys()) + list(pos_dist["ai"].keys()))
        tag_totals = {t: pos_dist["human"].get(t, 0) + pos_dist["ai"].get(t, 0)
                      for t in all_pos}
        tags = sorted(tag_totals.keys(), key=lambda t: tag_totals[t], reverse=True)[:top_k]

        h_vals = [pos_dist["human"].get(t, 0) for t in tags]
        a_vals = [pos_dist["ai"].get(t, 0) for t in tags]

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(tags))
        w = 0.35
        ax.bar(x - w/2, h_vals, w, label="Human", color=H_D, alpha=0.85, edgecolor="white")
        ax.bar(x + w/2, a_vals, w, label="AI", color=A_D, alpha=0.85, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(tags, fontsize=10)
        ax.set_ylabel("Proportion")
        ax.set_title(f"POS Distribution: Human vs AI — {key}", fontsize=13, fontweight="bold")
        ax.legend()

        fig.tight_layout()
        save_fig(fig, fig_dir, f"fig_interp06_pos_distribution_{key}.png")
        plt.close(fig)


def plot_logit_lens_comparison(results: dict, fig_dir: Path, top_k: int = 15):
    """fig_interp07: Cross-method logit lens comparison (shared top tokens)."""
    setup_style()

    logit_data = results.get("logit_lens", {})
    if len(logit_data) < 2:
        return

    # Group by layer
    from collections import defaultdict
    by_layer = defaultdict(dict)
    for key, data in logit_data.items():
        layer = data["layer"]
        by_layer[layer][key] = data

    for layer, methods in by_layer.items():
        if len(methods) < 2:
            continue

        # Collect union of top tokens across methods
        all_tokens_pos = set()
        all_tokens_neg = set()
        for data in methods.values():
            all_tokens_pos.update(d["token"] for d in data["top_positive_ai"][:top_k])
            all_tokens_neg.update(d["token"] for d in data["top_negative_human"][:top_k])

        # Build logit lookup per method
        method_logits = {}
        for key, data in methods.items():
            lookup = {}
            for d in data["top_positive_ai"] + data["top_negative_human"]:
                lookup[d["token"]] = d["logit"]
            method_logits[key] = lookup

        # Pick top tokens by average absolute logit across methods
        def avg_logit(tok):
            return np.mean([abs(ml.get(tok, 0)) for ml in method_logits.values()])

        top_pos = sorted(all_tokens_pos, key=avg_logit, reverse=True)[:top_k]
        top_neg = sorted(all_tokens_neg, key=avg_logit, reverse=True)[:top_k]

        fig, axes = plt.subplots(1, 2, figsize=(16, max(7, top_k * 0.35)))
        fig.suptitle(f"Logit Lens Comparison — Layer {layer}", fontsize=14, fontweight="bold")
        colors = [A_D, PALETTE["accent"], H_D]

        for ax, tokens, title in [(axes[0], top_pos, "AI-ward tokens"),
                                   (axes[1], top_neg, "Human-ward tokens")]:
            bar_w = 0.8 / len(methods)
            for i, (mkey, ml) in enumerate(methods.items()):
                vals = [ml.get(t, 0) for t in tokens]
                y = np.arange(len(tokens)) + i * bar_w
                ax.barh(y, vals, height=bar_w, label=mkey.split("_L")[0],
                        color=colors[i % len(colors)], alpha=0.8, edgecolor="white", linewidth=0.3)
            ax.set_yticks(np.arange(len(tokens)) + bar_w * (len(methods) - 1) / 2)
            ax.set_yticklabels(tokens, fontsize=8, fontfamily="monospace")
            ax.invert_yaxis()
            ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
            ax.set_title(title, fontsize=11)
            ax.legend(fontsize=8)

        fig.tight_layout()
        save_fig(fig, fig_dir, f"fig_interp07_logit_comparison_L{layer}.png")
        plt.close(fig)


def plot_all_interpretability(results: dict, fig_dir: Path):
    """Generate all interpretability figures."""
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Saving figures to {fig_dir}/")
    plot_logit_lens(results, fig_dir)
    plot_token_projection(results, fig_dir)
    plot_pos_projection(results, fig_dir)
    plot_dep_projection(results, fig_dir)
    plot_syntactic_correlations(results, fig_dir)
    plot_pos_distribution(results, fig_dir)
    plot_logit_lens_comparison(results, fig_dir)
    if "sae" in results:
        plot_sae_decomposition(results, fig_dir)
        plot_sae_feature_overlap(results, fig_dir)
    print(f"  ✓ All interpretability figures generated")


# ═══════════════════════════════════════════════════════════
#  SAE figures
# ═══════════════════════════════════════════════════════════

def plot_sae_decomposition(results: dict, fig_dir: Path, top_k: int = 30):
    """fig_interp08: Top SAE features by activation for each (layer, method)."""
    setup_style()

    sae_data = results.get("sae", {}).get("decompositions", {})
    if not sae_data:
        return

    for key, data in sae_data.items():
        features = data.get("top_features", [])[:top_k]
        if not features:
            continue

        fig, (ax_act, ax_cos) = plt.subplots(1, 2, figsize=(14, max(6, top_k * 0.28)))
        fig.suptitle(f"SAE Decomposition — {key}", fontsize=14, fontweight="bold", y=1.01)

        indices = [f"#{f['index']}" for f in features]
        activations = [f["activation"] for f in features]
        cosines = [f["cosine_dec"] for f in features]

        y = range(len(indices))

        # Activation bar chart
        ax_act.barh(y, activations, color=PALETTE["accent"], alpha=0.85,
                     edgecolor="white", linewidth=0.5)
        ax_act.set_yticks(y)
        ax_act.set_yticklabels(indices, fontsize=8, fontfamily="monospace")
        ax_act.invert_yaxis()
        ax_act.set_xlabel("Feature Activation")
        ax_act.set_title("Top features by activation", fontsize=11)

        # Cosine bar chart
        colors_cos = [A_D if c > 0 else H_D for c in cosines]
        ax_cos.barh(y, cosines, color=colors_cos, alpha=0.85,
                     edgecolor="white", linewidth=0.5)
        ax_cos.set_yticks(y)
        ax_cos.set_yticklabels(indices, fontsize=8, fontfamily="monospace")
        ax_cos.invert_yaxis()
        ax_cos.axvline(0, color="gray", linewidth=0.8, linestyle="--")
        ax_cos.set_xlabel("cos(decoder atom, v_src)")
        ax_cos.set_title("Decoder alignment with v_src", fontsize=11)

        # Add summary text
        recon = data.get("reconstruction_cosine", 0)
        l0 = data.get("l0", 0)
        d_sae = data.get("d_sae", 0)
        fig.text(0.5, -0.02,
                 f"recon_cos={recon:.4f}  |  L0={l0}  |  d_sae={d_sae}",
                 ha="center", fontsize=10, style="italic")

        fig.tight_layout()
        save_fig(fig, fig_dir, f"fig_interp08_sae_decomp_{key}.png")
        plt.close(fig)


def plot_sae_feature_overlap(results: dict, fig_dir: Path):
    """fig_interp09: Heatmap of feature overlap across methods."""
    setup_style()

    overlap = results.get("sae", {}).get("feature_overlap", {})
    if not overlap:
        return

    # Group by layer
    from collections import defaultdict
    by_layer = defaultdict(dict)
    for pair_key, info in overlap.items():
        parts = pair_key.split(" vs ")
        if len(parts) != 2:
            continue
        # Extract layer from key like "mean_diff_L16 vs mean_diff_fisher_L16"
        l1 = parts[0].split("_L")[-1]
        l2 = parts[1].split("_L")[-1]
        if l1 == l2:
            m1 = parts[0].rsplit("_L", 1)[0]
            m2 = parts[1].rsplit("_L", 1)[0]
            by_layer[int(l1)][(m1, m2)] = info["jaccard"]

    for layer, pairs in by_layer.items():
        # Build matrix
        methods = sorted(set(m for pair in pairs for m in pair))
        n = len(methods)
        if n < 2:
            continue

        matrix = np.eye(n)
        for (m1, m2), j in pairs.items():
            i1, i2 = methods.index(m1), methods.index(m2)
            matrix[i1, i2] = j
            matrix[i2, i1] = j

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_xticklabels([m.replace("mean_diff_", "").replace("mean_diff", "raw")
                            for m in methods], fontsize=9, rotation=45, ha="right")
        ax.set_yticks(range(n))
        ax.set_yticklabels([m.replace("mean_diff_", "").replace("mean_diff", "raw")
                            for m in methods], fontsize=9)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=10)
        plt.colorbar(im, ax=ax, label="Jaccard similarity")
        ax.set_title(f"SAE Feature Overlap (top-30) — Layer {layer}",
                     fontsize=13, fontweight="bold")

        fig.tight_layout()
        save_fig(fig, fig_dir, f"fig_interp09_sae_overlap_L{layer}.png")
        plt.close(fig)
