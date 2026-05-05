"""
Plot held-out separability figures from the JSONs produced by
scripts/eval_heldout_separability.py.

Outputs (per language):
    <fig_dir>/random_holdout_auc_<lang>.png    — line plot, held-out AUC ± std per layer per estimator,
                                          with in-sample AUC as a thin dashed reference.

Usage:
    python scripts/plot_heldout_separability.py \
        --in outputs/en/02_separability/Llama-3.1-8B-Instruct/mixed_500/random_holdout.json \
        --out outputs/en/02_separability/Llama-3.1-8B-Instruct/mixed_500/figures \
        --lang en
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
from visualization.palette import H_D, A_D, save_fig  # noqa: E402


METHOD_STYLE = {
    "raw":       {"color": H_D,        "marker": "o", "label": "Raw mean-diff"},
    "nullspace": {"color": A_D,        "marker": "s", "label": "Null-space ($k{=}10$)"},
    "fisher":    {"color": "#6C5CE7",  "marker": "^", "label": "Fisher LDA"},
}


def plot_one(in_path: str, out_dir: str, lang: str):
    with open(in_path) as f:
        data = json.load(f)
    res = data["results"]
    layers = sorted(int(k) for k in res.keys())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    fig.suptitle(
        f"Random pair held-out source-direction separability "
        f"({data['n_runs']}× {int((1-data['test_frac'])*100)}/{int(data['test_frac']*100)} "
        f"pair splits, $N_{{\\mathrm{{train}}}}{{=}}{res[str(layers[0])]['raw']['n_train']}$, "
        f"$N_{{\\mathrm{{test}}}}{{=}}{res[str(layers[0])]['raw']['n_test']}$, lang={lang})",
        fontsize=15, fontweight="bold", y=1.02,
    )

    # --- Panel 1: held-out AUC ± std, with in-sample dashed
    ax = axes[0]
    for method, style in METHOD_STYLE.items():
        means = np.array([res[str(ly)][method]["heldout_auc_mean"] for ly in layers])
        stds  = np.array([res[str(ly)][method]["heldout_auc_std"]  for ly in layers])
        insample = np.array([res[str(ly)][method]["insample_auc"] for ly in layers])
        ax.fill_between(layers, means - stds, means + stds, color=style["color"], alpha=0.18)
        ax.plot(layers, means, marker=style["marker"], color=style["color"], lw=2.0,
                markersize=7, label=f"{style['label']} (held-out)")
        ax.plot(layers, insample, color=style["color"], lw=1.0, ls="--", alpha=0.6,
                label=f"{style['label']} (in-sample)")
    ax.set_xlabel("Layer", fontsize=13)
    ax.set_ylabel("ROC-AUC", fontsize=13)
    ax.set_title("Random pair held-out vs in-sample AUC", fontsize=14, fontweight="bold", pad=8)
    ax.set_ylim(0.86, 1.005)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9, loc="lower right", ncol=1, framealpha=0.9)

    # --- Panel 2: held-out − in-sample gap (overfit indicator)
    ax = axes[1]
    for method, style in METHOD_STYLE.items():
        means = np.array([res[str(ly)][method]["heldout_auc_mean"] for ly in layers])
        insample = np.array([res[str(ly)][method]["insample_auc"] for ly in layers])
        gap = insample - means
        stds = np.array([res[str(ly)][method]["heldout_auc_std"] for ly in layers])
        ax.errorbar(layers, gap, yerr=stds, marker=style["marker"], color=style["color"],
                    lw=2.0, markersize=7, label=style["label"], capsize=4)
    ax.axhline(0.0, color="#888", lw=1.0, ls=":")
    ax.set_xlabel("Layer", fontsize=13)
    ax.set_ylabel("In-sample AUC $-$ Held-out AUC", fontsize=13)
    ax.set_title("Generalisation gap per layer", fontsize=14, fontweight="bold", pad=8)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="upper left", framealpha=0.9)

    plt.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    save_fig(fig, out_dir, f"random_holdout_auc_{lang}.png")
    plt.close(fig)
    print(f"Saved -> {out_dir}/random_holdout_auc_{lang}.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--lang", default="en")
    args = p.parse_args()
    plot_one(args.in_path, args.out, args.lang)
