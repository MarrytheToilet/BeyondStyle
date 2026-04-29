#!/usr/bin/env python3
"""§3.1 figure from `content_confound_results.json`.

A single 3-panel summary figure goes to
`outputs/{lang}/01_data_corpus/figures/data_corpus_summary.png`.
Soft-pastel palette: light blue = Human, soft pink = AI, lavender = neutral.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from src.visualization.palette import setup_style, save_fig

# Soft pastels (override the darker variants for this figure family)
SOFT = {
    "human":   "#A8D8EA",   # light blue (human)
    "ai":      "#F7C5CC",   # soft pink (ai = chat)
    "free":    "#C8B8DB",   # lavender (ai = free)
    "neutral": "#D9D9D9",   # light grey
    "edge_h":  "#7FB7D6",
    "edge_a":  "#E8A6B0",
    "edge_f":  "#A691C2",
    "ink":     "#3F3F4A",
}


def _load(p): return json.load(open(p))


def panel_cosine(ax, d):
    rows = d.get("per_sample_cosines", [])
    h = np.array([r["cos_sketch_human"] for r in rows if r.get("cos_sketch_human") is not None])
    c = np.array([r["cos_sketch_chat"]  for r in rows if r.get("cos_sketch_chat")  is not None])
    f = np.array([r["cos_sketch_free"]  for r in rows if r.get("cos_sketch_free")  is not None])
    bins = np.linspace(min(h.min(), c.min(), f.min()),
                       max(h.max(), c.max(), f.max()), 28)
    ax.hist(h, bins=bins, color=SOFT["human"], alpha=0.85,
            label=f"Human  μ={h.mean():.3f}", edgecolor=SOFT["edge_h"], lw=0.6)
    ax.hist(c, bins=bins, color=SOFT["ai"], alpha=0.7,
            label=f"AI-chat  μ={c.mean():.3f}", edgecolor=SOFT["edge_a"], lw=0.6)
    ax.hist(f, bins=bins, color=SOFT["free"], alpha=0.55,
            label=f"AI-free  μ={f.mean():.3f}", edgecolor=SOFT["edge_f"], lw=0.6)
    for v, c_ in [(h.mean(), SOFT["edge_h"]), (c.mean(), SOFT["edge_a"]), (f.mean(), SOFT["edge_f"])]:
        ax.axvline(v, color=c_, ls="--", lw=1)
    ax.set_xlabel("cos(sketch, continuation)")
    ax.set_ylabel("Pairs")
    ax.set_title("(a) Sketch-adherence distribution by source", fontsize=12, loc="left")
    ax.legend(loc="upper left", frameon=False, fontsize=9)


def panel_confound_auc(ax, d):
    p1 = d["p1_sketch_only_classifier"]
    p5 = d["p5_chat_free_cross_transfer"]
    items = [
        ("Sketch-text\n(chat)",   p1["chat_vs_human"]["sketch_text_only_auc"]["mean"], "control"),
        ("Sketch-emb\n(chat)",    p1["chat_vs_human"]["sketch_emb_only_auc"]["mean"],  "control"),
        ("Sketch-text\n(free)",   p1["free_vs_human"]["sketch_text_only_auc"]["mean"], "control"),
        ("Sketch-emb\n(free)",    p1["free_vs_human"]["sketch_emb_only_auc"]["mean"],  "control"),
        ("Full-text\nTF-IDF (chat)", p1["chat_vs_human"]["full_text_tfidf_auc_upper_bound"]["mean"], "ub"),
        ("Full-text\nTF-IDF (free)", p1["free_vs_human"]["full_text_tfidf_auc_upper_bound"]["mean"], "ub"),
        ("$v_{chat}\\!\\to\\!$\nfree vs human", p5["auc_v_chat_on_free_vs_human"], "transfer"),
        ("$v_{free}\\!\\to\\!$\nchat vs human", p5["auc_v_free_on_chat_vs_human"], "transfer"),
    ]
    color_for = {"control": SOFT["human"],
                 "ub":      SOFT["neutral"],
                 "transfer": SOFT["ai"]}
    edge_for  = {"control": SOFT["edge_h"],
                 "ub":      "#B8B8B8",
                 "transfer": SOFT["edge_a"]}
    labels, vals, kinds = zip(*items)
    xs = np.arange(len(labels))
    bars = ax.bar(xs, vals,
                  color=[color_for[k] for k in kinds],
                  edgecolor=[edge_for[k] for k in kinds],
                  lw=0.8, width=0.78)
    ax.axhline(0.5, color="#999", ls="--", lw=1, zorder=0)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=8.5, rotation=15, ha="right")
    ax.set_ylim(0.4, 1.04)
    ax.set_ylabel("AUC")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.014, f"{v:.2f}",
                ha="center", va="bottom", fontsize=8.5, color=SOFT["ink"])
    ax.set_title("(b) Sketch-confound controls and cross-transfer", fontsize=12, loc="left")
    # legend proxies
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=color_for["control"],  edgecolor=edge_for["control"],  label="sketch-only (control)"),
               Patch(facecolor=color_for["ub"],       edgecolor=edge_for["ub"],       label="full-text (upper bound)"),
               Patch(facecolor=color_for["transfer"], edgecolor=edge_for["transfer"], label="cross-transfer $v$")]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=8.5,
              ncol=3, bbox_to_anchor=(0.0, -0.46))


def panel_regression(ax_left, ax_right, d):
    p3 = d["p3_regression_controls"]
    coefs = p3["control_coefs"]

    ax_left.bar(["controls only", "+ is_ai"],
                [p3["controls_only_R2"], p3["with_is_ai_R2"]],
                color=[SOFT["human"], SOFT["ai"]],
                edgecolor=[SOFT["edge_h"], SOFT["edge_a"]], lw=0.9, width=0.55)
    ax_left.set_ylabel("R$^2$ on source projection $z$")
    ax_left.set_title(f"(c) $\\Delta$R$^2$ from is_ai = {p3['delta_R2_from_is_ai']:.3f}",
                      fontsize=12, loc="left")

    keys = [k for k in coefs if k != "const"]
    vals = [coefs[k] for k in keys]
    ax_right.barh(keys, vals,
                  color=SOFT["free"], edgecolor=SOFT["edge_f"], lw=0.9)
    ax_right.axvline(0, color="#999", lw=0.8)
    ax_right.set_xlabel("Coefficient on $z$")
    ax_right.set_title(f"(d) is_ai effect $\\beta$={p3['is_ai_coef']:.3f}  "
                       f"(p={p3['is_ai_p_clustered']:.0e})",
                       fontsize=12, loc="left")


def make_figure(d, out_path: Path):
    fig = plt.figure(figsize=(13.5, 8.4))
    gs = GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.40,
                  height_ratios=[1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1:])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1:])

    panel_cosine(ax_a, d)
    panel_confound_auc(ax_b, d)
    panel_regression(ax_c, ax_d, d)

    fig.suptitle("Construct validity of the paired corpus  (§3.1)",
                 fontsize=14, fontweight="bold", y=0.995)
    return save_fig(fig, out_path.parent, out_path.name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", type=Path, default=Path("outputs/en/01_data_corpus"))
    args = p.parse_args()

    setup_style()
    fig_dir = args.in_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    d = _load(args.in_dir / "content_confound_results.json")
    print("→", make_figure(d, fig_dir / "data_corpus_summary.png"))


if __name__ == "__main__":
    main()
