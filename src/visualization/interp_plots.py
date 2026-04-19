"""
Interpretability visualizations — consolidated, information-dense figures.

Palette: light-blue (#B8D4E8) = Human, light-pink (#F5C6D0) = AI.
White background, minimal gridlines, large legible fonts.

Produces 6 focused figures (flat, no subdirectories):
  fig01 — Logit Lens + Contrastive Token Projection (combined)
  fig02 — Construction Grammar Patterns (log-ratio dot plot)
  fig03 — Contrastive N-grams (2-gram + 3-gram side-by-side)
  fig04 — Sentence Openings + Structural Feature Dumbbell
  fig05 — Information Density Profile
  fig06 — Lexical Bundle Repertoire
"""

from __future__ import annotations
import math
import warnings
from pathlib import Path
from typing import Dict, Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Palette ──────────────────────────────────────────────────────────
C_HUMAN   = "#B8D4E8"   # light blue
C_AI      = "#F5C6D0"   # light pink
C_HUMAN_D = "#6AAFE0"   # deeper blue for emphasis / edges
C_AI_D    = "#E8899A"   # deeper pink for emphasis / edges
C_GREY    = "#999999"
C_BG      = "#FFFFFF"
C_GRID    = "#ECECEC"
C_TEXT    = "#2C2C2C"


def _style(ax, title="", xlabel="", ylabel="", grid_axis="y"):
    """Apply consistent academic style."""
    ax.set_facecolor(C_BG)
    ax.figure.set_facecolor(C_BG)
    if title:
        ax.set_title(title, fontsize=13, fontweight="bold", color=C_TEXT, pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10.5, color=C_TEXT)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10.5, color=C_TEXT)
    ax.tick_params(colors=C_TEXT, labelsize=9.5)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["bottom", "left"]:
        ax.spines[spine].set_color(C_GRID)
        ax.spines[spine].set_linewidth(0.8)
    if grid_axis:
        ax.grid(axis=grid_axis, color=C_GRID, linewidth=0.5, alpha=0.7)
        ax.set_axisbelow(True)


def _save(fig, path: Path):
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=C_BG)
    plt.close(fig)
    print(f"    ✓ {path.name}")


def _parse_key(key: str):
    """'mean_diff_fisher_L16' → ('mean_diff_fisher', 16)"""
    parts = key.rsplit("_L", 1)
    return parts[0], int(parts[1]) if len(parts) > 1 else 0


def _pick_best_layer(results: dict, section: str = "logit_lens",
                     preferred: int = 16) -> int:
    """Pick best available layer, preferring `preferred`."""
    keys = list(results.get(section, {}).keys())
    layers = set()
    for k in keys:
        _, l = _parse_key(k)
        layers.add(l)
    return preferred if preferred in layers else (min(layers) if layers else preferred)


# =====================================================================
#  Fig 01 — Logit Lens + Contrastive Token Projection (combined)
# =====================================================================

def _plot_fig01(results: dict, fig_dir: Path, layer: int = 16,
                method: str = "mean_diff", n_tok: int = 15):
    """
    Left: Logit Lens (AI-promoted ↔ Human-promoted tokens).
    Right: Source-exclusive token projections.
    """
    key = f"{method}_L{layer}"

    # Logit lens data
    ll = results.get("logit_lens", {}).get(key, {})
    pos_tok = ll.get("top_positive_ai", [])[:n_tok]
    neg_tok = ll.get("top_negative_human", [])[:n_tok]

    # Contrastive token data
    tp = results.get("token_projection", {}).get(key, {})
    ai_exc = tp.get("ai_exclusive_tokens", [])[:n_tok]
    hu_exc = tp.get("human_exclusive_tokens", [])[:n_tok]

    if not pos_tok and not ai_exc:
        return

    fig, (ax_ll, ax_tp) = plt.subplots(1, 2, figsize=(16, max(6, n_tok * 0.38)))

    # ── Left: Logit Lens ──
    if pos_tok or neg_tok:
        all_toks = [(t["token"], t["logit"], "human") for t in neg_tok[::-1]] + \
                   [(t["token"], t["logit"], "ai") for t in pos_tok]
        labels_ll = [e[0] for e in all_toks]
        vals_ll = [e[1] for e in all_toks]
        colors_ll = [C_AI if e[2] == "ai" else C_HUMAN for e in all_toks]
        edges_ll = [C_AI_D if e[2] == "ai" else C_HUMAN_D for e in all_toks]

        ax_ll.barh(range(len(labels_ll)), vals_ll, color=colors_ll,
                   edgecolor=edges_ll, linewidth=0.6, zorder=3, height=0.7)
        ax_ll.set_yticks(range(len(labels_ll)))
        ax_ll.set_yticklabels([f'"{t}"' for t in labels_ll], fontsize=9,
                              fontfamily="monospace")
        ax_ll.axvline(0, color=C_TEXT, linewidth=0.7, alpha=0.5)
        _style(ax_ll, title=f"Logit Lens — L{layer}",
               xlabel="← Human-promoted     AI-promoted →", grid_axis="x")

    # ── Right: Contrastive Projections ──
    if ai_exc or hu_exc:
        entries = [(t["token"], t["mean"], "human") for t in hu_exc] + \
                  [(t["token"], t["mean"], "ai") for t in ai_exc]
        entries.sort(key=lambda e: e[1])
        labels_tp = [e[0] for e in entries]
        vals_tp = [e[1] for e in entries]
        colors_tp = [C_AI if e[2] == "ai" else C_HUMAN for e in entries]
        edges_tp = [C_AI_D if e[2] == "ai" else C_HUMAN_D for e in entries]

        ax_tp.barh(range(len(labels_tp)), vals_tp, color=colors_tp,
                   edgecolor=edges_tp, linewidth=0.6, zorder=3, height=0.7)
        ax_tp.set_yticks(range(len(labels_tp)))
        ax_tp.set_yticklabels([f'"{t}"' for t in labels_tp], fontsize=9,
                              fontfamily="monospace")
        ax_tp.axvline(0, color=C_TEXT, linewidth=0.7, alpha=0.5)

        ai_m = tp.get("global_ai_mean", 0)
        hu_m = tp.get("global_human_mean", 0)
        ax_tp.text(0.97, 0.03, f"Mean proj: AI={ai_m:+.3f}  H={hu_m:+.3f}",
                   transform=ax_tp.transAxes, ha="right", va="bottom",
                   fontsize=8, color=C_GREY,
                   bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GRID))
        _style(ax_tp, title=f"Source-Exclusive Token Projection — L{layer}",
               xlabel="← Human-exclusive     AI-exclusive →", grid_axis="x")

    fig.suptitle("Steering Vector Vocabulary Decomposition",
                 fontsize=15, fontweight="bold", color=C_TEXT, y=1.01)
    fig.tight_layout()
    _save(fig, fig_dir / "fig01_logit_lens_and_tokens.png")


# =====================================================================
#  Fig 02 — Construction Grammar Patterns
# =====================================================================

def _plot_fig02(deep: dict, fig_dir: Path):
    """Log-ratio dot plot of construction grammar patterns with examples."""
    constr = deep.get("construction_patterns", {})
    if not constr:
        return

    items = []
    for name, data in constr.items():
        total = data["human_count"] + data["ai_count"]
        if total < 3:
            continue
        ratio = data["ratio"]
        items.append((name, ratio, data["human_per_text"],
                      data["ai_per_text"], data.get("description", ""), total))

    items.sort(key=lambda x: x[1])

    fig, ax = plt.subplots(figsize=(13, max(7, len(items) * 0.38)))
    labels = [it[0].replace("_", " ") for it in items]
    ratios = [it[1] for it in items]
    y = np.arange(len(labels))

    colors = [C_AI_D if r > 1.5 else C_HUMAN_D if r < 0.67 else C_GREY
              for r in ratios]
    log_ratios = [math.log2(max(r, 0.01)) for r in ratios]

    bars = ax.barh(y, log_ratios, color=colors, alpha=0.75, height=0.65,
                   edgecolor=[c if c != C_GREY else "#BBBBBB" for c in colors],
                   linewidth=0.5)
    ax.axvline(0, color=C_TEXT, linewidth=0.8, linestyle="-", alpha=0.4)

    for i, (name, ratio, h_pt, a_pt, desc, total) in enumerate(items):
        side = "left" if log_ratios[i] >= 0 else "right"
        offset = 5 if log_ratios[i] >= 0 else -5
        ha = "left" if log_ratios[i] >= 0 else "right"
        ax.annotate(f"×{ratio:.1f}  H:{h_pt:.2f}  A:{a_pt:.2f}",
                    xy=(log_ratios[i], i), xytext=(offset, 0),
                    textcoords="offset points", fontsize=7.5, va="center",
                    ha=ha, color=C_TEXT)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5)
    _style(ax, title="Construction Grammar: AI / Human Frequency Ratio",
           xlabel="log₂(ratio)    ← Human-preferred          AI-preferred →",
           grid_axis="x")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C_AI_D, alpha=0.75, label="AI-preferred (>1.5×)"),
        Patch(facecolor=C_GREY, alpha=0.75, label="Neutral"),
        Patch(facecolor=C_HUMAN_D, alpha=0.75, label="Human-preferred (<0.67×)"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right",
              framealpha=0.9, edgecolor=C_GRID)

    fig.tight_layout()
    _save(fig, fig_dir / "fig02_construction_patterns.png")


# =====================================================================
#  Fig 03 — Contrastive N-grams (2-gram + 3-gram combined)
# =====================================================================

def _plot_fig03(deep: dict, fig_dir: Path, top_k: int = 15):
    """Side-by-side: AI-distinctive vs Human-distinctive n-grams."""
    ngrams = deep.get("contrastive_ngrams", {})
    if not ngrams:
        return

    panels = []
    for n_label in ["2-gram", "3-gram"]:
        data = ngrams.get(n_label, {})
        if data:
            panels.append((n_label, data))
    if not panels:
        return

    n_panels = len(panels)
    fig, axes = plt.subplots(n_panels, 2, figsize=(16, 5.5 * n_panels))
    if n_panels == 1:
        axes = [axes]

    for row, (n_label, data) in enumerate(panels):
        ai_top = data.get("ai_distinctive", [])[:top_k]
        human_top = data.get("human_distinctive", [])[:top_k]

        # AI panel
        ax_ai = axes[row][0]
        if ai_top:
            labels_a = [d["ngram"] for d in reversed(ai_top)]
            vals_a = [d["g2"] for d in reversed(ai_top)]
            y_a = np.arange(len(labels_a))
            ax_ai.barh(y_a, vals_a, color=C_AI, edgecolor=C_AI_D,
                       linewidth=0.5, height=0.7)
            ax_ai.set_yticks(y_a)
            ax_ai.set_yticklabels(labels_a, fontsize=9)
            for i, (lbl, val) in enumerate(zip(labels_a, vals_a)):
                freq_info = ai_top[len(ai_top)-1-i]
                ax_ai.text(val + 0.3, i, f'{freq_info["ai_per_1k"]:.1f}/1k',
                           va="center", fontsize=7.5, color=C_GREY)
        _style(ax_ai, title=f"AI-Distinctive {n_label}s",
               xlabel="G² (log-likelihood ratio)", grid_axis="x")

        # Human panel
        ax_h = axes[row][1]
        if human_top:
            labels_h = [d["ngram"] for d in reversed(human_top)]
            vals_h = [abs(d["g2"]) for d in reversed(human_top)]
            y_h = np.arange(len(labels_h))
            ax_h.barh(y_h, vals_h, color=C_HUMAN, edgecolor=C_HUMAN_D,
                       linewidth=0.5, height=0.7)
            ax_h.set_yticks(y_h)
            ax_h.set_yticklabels(labels_h, fontsize=9)
            for i, (lbl, val) in enumerate(zip(labels_h, vals_h)):
                freq_info = human_top[len(human_top)-1-i]
                ax_h.text(val + 0.3, i, f'{freq_info["human_per_1k"]:.1f}/1k',
                          va="center", fontsize=7.5, color=C_GREY)
        _style(ax_h, title=f"Human-Distinctive {n_label}s",
               xlabel="G² (log-likelihood ratio)", grid_axis="x")

    fig.suptitle("Contrastive Multi-Word Patterns",
                 fontsize=15, fontweight="bold", color=C_TEXT, y=1.01)
    fig.tight_layout()
    _save(fig, fig_dir / "fig03_contrastive_ngrams.png")


# =====================================================================
#  Fig 04 — Sentence Openings + Structural Dumbbell
# =====================================================================

def _plot_fig04(results: dict, deep: dict, fig_dir: Path,
                layer: int = 16, method: str = "mean_diff"):
    """Left: sentence opening distribution. Right: structural feature dumbbell."""
    fig, (ax_open, ax_struct) = plt.subplots(1, 2, figsize=(17, 7))

    # ── Left: Sentence Openings ──
    templates = deep.get("syntactic_templates", {})
    openings = templates.get("sentence_openings", {})
    if openings:
        sorted_ops = sorted(openings.items(),
                            key=lambda x: abs(x[1]["delta_pct"]), reverse=True)
        op_labels = [op.replace("_", " ") for op, _ in sorted_ops]
        h_pct = [d["human_pct"] for _, d in sorted_ops]
        a_pct = [d["ai_pct"] for _, d in sorted_ops]

        y = np.arange(len(op_labels))
        h = 0.35
        ax_open.barh(y + h/2, h_pct, h, color=C_HUMAN, edgecolor=C_HUMAN_D,
                     linewidth=0.5, label="Human")
        ax_open.barh(y - h/2, a_pct, h, color=C_AI, edgecolor=C_AI_D,
                     linewidth=0.5, label="AI")

        # Delta annotations
        for i, (op, d) in enumerate(sorted_ops):
            delta = d["delta_pct"]
            sign = "+" if delta > 0 else ""
            ax_open.text(max(h_pct[i], a_pct[i]) + 0.5, i,
                         f"Δ={sign}{delta:.1f}%", va="center", fontsize=7.5,
                         color=C_AI_D if delta > 0 else C_HUMAN_D,
                         fontweight="bold")

        ax_open.set_yticks(y)
        ax_open.set_yticklabels(op_labels, fontsize=9.5)
        ax_open.legend(fontsize=9, loc="lower right", framealpha=0.9,
                       edgecolor=C_GRID)
        _style(ax_open, title="Sentence Opening Patterns",
               xlabel="% of sentences", grid_axis="x")

    # ── Right: Structural Feature Dumbbell ──
    structural = results.get("structural", {})
    key = f"{method}_L{layer}"
    sc = structural.get(key, {}).get("structural_comparison", {})
    if sc:
        skip = {"unique_token_ratio"}
        feats = [(k, v) for k, v in sc.items() if k not in skip and "ratio" not in k]
        if not feats:
            feats = list(sc.items())
        feats.sort(key=lambda x: abs(x[1].get("delta", 0)), reverse=True)
        feats = feats[:10]
        feats.reverse()

        names = [f[0].replace("_", " ") for f in feats]
        h_vals = [f[1]["human_mean"] for f in feats]
        a_vals = [f[1]["ai_mean"] for f in feats]
        y = np.arange(len(names))

        for i in range(len(names)):
            ax_struct.plot([h_vals[i], a_vals[i]], [i, i],
                          color=C_GRID, linewidth=2.5, zorder=1)
        ax_struct.scatter(h_vals, y, color=C_HUMAN, edgecolor=C_HUMAN_D,
                         s=70, linewidth=0.8, zorder=3, label="Human")
        ax_struct.scatter(a_vals, y, color=C_AI, edgecolor=C_AI_D,
                         s=70, linewidth=0.8, zorder=3, label="AI")

        ax_struct.set_yticks(y)
        ax_struct.set_yticklabels(names, fontsize=9.5)
        ax_struct.legend(fontsize=9, framealpha=0.9, edgecolor=C_GRID)
        _style(ax_struct, title="Structural Feature Comparison",
               xlabel="Feature value", grid_axis="x")

    fig.suptitle("Syntactic Structure Analysis",
                 fontsize=15, fontweight="bold", color=C_TEXT, y=1.01)
    fig.tight_layout()
    _save(fig, fig_dir / "fig04_sentence_structure.png")


# =====================================================================
#  Fig 05 — Information Density Profile
# =====================================================================

def _plot_fig05(deep: dict, fig_dir: Path):
    """Paired bar chart of information-theoretic and lexical diversity metrics."""
    density = deep.get("information_density", {})
    comparison = density.get("comparison", {})
    if not comparison:
        return

    # Curated metrics grouped for readability
    groups = [
        ("Lexical\nDiversity", ["ttr", "hapax_ratio", "mean_ttr_window50"]),
        ("Sentence\nVariation", ["sent_length_mean", "sent_length_cv"]),
        ("Information\nContent", ["mean_surprisal", "std_surprisal", "surprisal_skew"]),
        ("Rare Word\nBehavior", ["high_info_word_ratio", "rare_word_burstiness"]),
    ]

    # Flatten available keys
    all_keys = []
    group_boundaries = []
    group_labels = []
    for gname, keys in groups:
        avail = [k for k in keys if k in comparison]
        if avail:
            group_boundaries.append(len(all_keys))
            group_labels.append(gname)
            all_keys.extend(avail)

    if not all_keys:
        return

    h_vals = [comparison[k]["human"] for k in all_keys]
    a_vals = [comparison[k]["ai"] for k in all_keys]
    tick_labels = [k.replace("_", "\n") for k in all_keys]

    fig, ax = plt.subplots(figsize=(15, 6))
    x = np.arange(len(all_keys))
    w = 0.35

    ax.bar(x - w/2, h_vals, w, color=C_HUMAN, edgecolor=C_HUMAN_D,
           linewidth=0.5, label="Human", zorder=3)
    ax.bar(x + w/2, a_vals, w, color=C_AI, edgecolor=C_AI_D,
           linewidth=0.5, label="AI", zorder=3)

    # Delta annotations
    for i, k in enumerate(all_keys):
        delta = comparison[k].get("delta", 0)
        if delta is None:
            continue
        higher = max(h_vals[i], a_vals[i])
        sign = "+" if delta > 0 else ""
        color = C_AI_D if delta > 0 else C_HUMAN_D
        ax.text(i, higher + abs(higher) * 0.04 + 0.01, f"{sign}{delta:.3f}",
                ha="center", va="bottom", fontsize=7.5, color=color,
                fontweight="bold")

    # Group separators
    for gb in group_boundaries[1:]:
        ax.axvline(gb - 0.5, color=C_GRID, linewidth=1.2, linestyle="--", alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=8.5, rotation=25, ha="right")
    ax.legend(fontsize=10, framealpha=0.9, edgecolor=C_GRID)
    _style(ax, title="Information Density Profile: Human vs AI",
           ylabel="Value", grid_axis="y")
    fig.tight_layout()
    _save(fig, fig_dir / "fig05_info_density.png")


# =====================================================================
#  Fig 06 — Lexical Bundles
# =====================================================================

def _plot_fig06(deep: dict, fig_dir: Path, top_k: int = 12):
    """Source-exclusive formulaic sequences across n-word sizes."""
    bundles = deep.get("lexical_bundles", {})
    if not bundles:
        return

    n_labels = sorted(bundles.keys())
    n_with_data = [n for n in n_labels
                   if bundles[n].get("ai_exclusive") or bundles[n].get("human_exclusive")]
    if not n_with_data:
        return

    fig, axes = plt.subplots(1, len(n_with_data),
                             figsize=(6.5 * len(n_with_data), 7))
    if len(n_with_data) == 1:
        axes = [axes]

    for ax, n_label in zip(axes, n_with_data):
        data = bundles[n_label]
        ai_excl = data.get("ai_exclusive", [])[:top_k]
        h_excl = data.get("human_exclusive", [])[:top_k]

        all_items = (
            [(b["bundle"], b["freq"], "ai") for b in ai_excl] +
            [(b["bundle"], b["freq"], "human") for b in h_excl]
        )
        all_items.sort(key=lambda x: x[1], reverse=True)
        all_items = all_items[:top_k * 2]

        if not all_items:
            continue

        labels = [it[0] for it in reversed(all_items)]
        freqs = [it[1] for it in reversed(all_items)]
        colors = [C_AI if it[2] == "ai" else C_HUMAN for it in reversed(all_items)]
        edges = [C_AI_D if it[2] == "ai" else C_HUMAN_D for it in reversed(all_items)]

        y_pos = np.arange(len(labels))
        ax.barh(y_pos, freqs, color=colors, edgecolor=edges,
                linewidth=0.5, height=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8.5)

        ai_types = data.get("ai_bundle_types", 0)
        h_types = data.get("human_bundle_types", 0)
        ax.text(0.97, 0.03,
                f"Bundle types: AI={ai_types}  H={h_types}",
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, color=C_GREY,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C_GRID))
        _style(ax, title=f"Exclusive {n_label} Bundles",
               xlabel="Frequency", grid_axis="x")

    # Shared legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=C_AI, edgecolor=C_AI_D, label="AI-exclusive"),
        Patch(facecolor=C_HUMAN, edgecolor=C_HUMAN_D, label="Human-exclusive"),
    ]
    axes[-1].legend(handles=legend_elements, fontsize=9, loc="lower right",
                    framealpha=0.9, edgecolor=C_GRID)

    fig.suptitle("Formulaic Sequence Repertoire",
                 fontsize=15, fontweight="bold", color=C_TEXT, y=1.01)
    fig.tight_layout()
    _save(fig, fig_dir / "fig06_lexical_bundles.png")


# =====================================================================
#  Dispatcher
# =====================================================================

def plot_all_interpretability(results: dict, fig_dir: Path):
    """Generate all interpretability figures — consolidated, flat directory."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    warnings.filterwarnings("ignore", category=UserWarning)

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "figure.facecolor": C_BG,
        "axes.facecolor": C_BG,
        "savefig.facecolor": C_BG,
    })

    # Determine best layer for legacy sections
    layer = _pick_best_layer(results, "logit_lens", preferred=16)

    # Fig 01 — Logit Lens + Contrastive Tokens
    _plot_fig01(results, fig_dir, layer=layer)

    deep = results.get("deep", {})
    if not deep:
        print("    ⚠ No deep analysis results — only fig01 generated")
        return

    # Fig 02 — Construction Grammar
    _plot_fig02(deep, fig_dir)

    # Fig 03 — Contrastive N-grams
    _plot_fig03(deep, fig_dir)

    # Fig 04 — Sentence Structure (openings + dumbbell)
    _plot_fig04(results, deep, fig_dir, layer=layer)

    # Fig 05 — Information Density
    _plot_fig05(deep, fig_dir)

    # Fig 06 — Lexical Bundles
    _plot_fig06(deep, fig_dir)

    print("  ✓ All interpretability figures generated (6 figures)")
