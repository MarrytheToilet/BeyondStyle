"""
Random pair-held-out separability protocol for the source direction.

For each layer's cached paired hidden-state file, for each estimator
{raw, nullspace}, run R random 80/20 splits over pairs:
    1. Estimate v on the train pairs only.
    2. Project the held-out test pairs and compute ROC-AUC and accuracy.
Also report the in-sample AUC fitted on all pairs for comparison.

Usage:
    python scripts/eval_heldout_separability.py \\
        --states_glob "outputs/en/03_steering_vectors/Llama-3.1-8B-Instruct/proposition_aligned/last_token/states/paired_states_layer*.pt" \\
        --out outputs/en/02_separability/Llama-3.1-8B-Instruct/proposition_aligned/last_token/random_holdout.json
"""

import argparse
import glob
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.steering.denoise import compute_denoised_direction


def fit_v(method: str, h_tr: torch.Tensor, a_tr: torch.Tensor, null_k: int) -> torch.Tensor:
    if method == "raw":
        v = (a_tr - h_tr).mean(dim=0)
        return v / v.norm()
    v, _ = compute_denoised_direction(h_tr, a_tr, method=method, null_k=null_k)
    return v


def fit_threshold(
    v: torch.Tensor,
    human: torch.Tensor,
    ai: torch.Tensor,
) -> float:
    """Fit the accuracy threshold using training labels only."""
    return float(((human @ v).mean() + (ai @ v).mean()).item() / 2)


def eval_v(
    v: torch.Tensor,
    h_te: torch.Tensor,
    a_te: torch.Tensor,
    threshold: float,
):
    sc = torch.cat([h_te @ v, a_te @ v]).numpy()
    y = np.concatenate([np.zeros(len(h_te)), np.ones(len(a_te))])
    auc = float(roc_auc_score(y, sc))
    pred = (sc > threshold).astype(int)
    acc = float((pred == y).mean())
    return auc, acc


def run(states_glob: str, out_path: str, n_runs: int, test_frac: float, null_k: int, seed: int):
    files = sorted(glob.glob(states_glob))
    if not files:
        raise FileNotFoundError(f"No state files match: {states_glob}")
    results = {}
    state_metadata = None
    for fp in files:
        m = re.search(r"layer(\d+)", fp)
        if not m:
            continue
        layer = int(m.group(1))
        d = torch.load(fp, map_location="cpu", weights_only=False)
        H, A = d["human"].float(), d["ai"].float()
        metadata = d.get("metadata") or {}
        if state_metadata is None:
            state_metadata = metadata
        elif (
            metadata.get("sample_ids") != state_metadata.get("sample_ids")
            or metadata.get("dataset_sha256")
            != state_metadata.get("dataset_sha256")
            or metadata.get("pooling") != state_metadata.get("pooling")
        ):
            raise ValueError(f"State metadata mismatch at {fp}")
        N = len(H)
        rng = np.random.default_rng(seed)
        perms = [rng.permutation(N) for _ in range(n_runs)]
        n_test = int(N * test_frac)

        layer_res = {}
        for method in ["raw", "nullspace"]:
            aucs, accs = [], []
            for perm in perms:
                te, tr = perm[:n_test], perm[n_test:]
                v = fit_v(method, H[tr], A[tr], null_k)
                threshold = fit_threshold(v, H[tr], A[tr])
                auc, acc = eval_v(v, H[te], A[te], threshold)
                aucs.append(auc)
                accs.append(acc)
            v_full = fit_v(method, H, A, null_k)
            full_threshold = fit_threshold(v_full, H, A)
            in_auc, in_acc = eval_v(
                v_full,
                H,
                A,
                full_threshold,
            )
            layer_res[method] = {
                "heldout_auc_mean": float(np.mean(aucs)),
                "heldout_auc_std": float(np.std(aucs)),
                "heldout_acc_mean": float(np.mean(accs)),
                "heldout_acc_std": float(np.std(accs)),
                "insample_auc": in_auc,
                "insample_acc": in_acc,
                "n_test": n_test,
                "n_train": N - n_test,
                "n_runs": n_runs,
            }
        results[layer] = layer_res
        line = f"L{layer:02d}  " + "  ".join(
            f"{m}: heldout AUC={layer_res[m]['heldout_auc_mean']:.3f}"
            f"±{layer_res[m]['heldout_auc_std']:.3f} (in={layer_res[m]['insample_auc']:.3f})"
            for m in ["raw", "nullspace"]
        )
        print(line)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "protocol": (
            f"{n_runs}x random {int((1-test_frac)*100)}/{int(test_frac*100)} splits, "
            "fit v and the accuracy threshold on train pairs only, evaluate "
            "AUC + frozen-threshold accuracy on held-out test pairs"
        ),
        "n_total_pairs": N,
        "test_frac": test_frac,
        "n_runs": n_runs,
        "null_k": null_k,
        "seed": seed,
        "representation": (state_metadata or {}).get("pooling"),
        "dataset_path": (state_metadata or {}).get("dataset_path"),
        "dataset_sha256": (state_metadata or {}).get("dataset_sha256"),
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved -> {out_path}")

    md_path = str(Path(out_path).with_suffix("")) + "_summary.md"
    write_md_summary(payload, md_path)
    print(f"Saved -> {md_path}")

    fig_path = str(Path(out_path).with_suffix("")) + "_per_layer.png"
    plot_per_layer(payload, fig_path)
    print(f"Saved -> {fig_path}")


def plot_per_layer(payload: dict, fig_path: str) -> None:
    """Title-free per-layer held-out AUC figure (mean ± split s.d.)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator

    from src.visualization.separability_style import (
        BG,
        DPI,
        INK,
        METHOD_STYLE,
        MUTED,
        PINK_DARK,
        focused_auc_limits,
        setup_separability_style,
    )
    setup_separability_style()

    # Nullspace is the primary result; raw is retained as an explicit control.
    methods = ["nullspace", "raw"]
    layers = sorted(int(l) for l in payload["results"])
    series = {}
    scale_values = []

    fig, ax = plt.subplots(figsize=(8.8, 4.9), facecolor=BG)
    for m in methods:
        means, stds, ins = [], [], []
        for ly in layers:
            r = payload["results"][str(ly)] if str(ly) in payload["results"] else payload["results"][ly]
            means.append(r[m]["heldout_auc_mean"])
            stds.append(r[m]["heldout_auc_std"])
            ins.append(r[m]["insample_auc"])
        means = np.asarray(means); stds = np.asarray(stds); ins = np.asarray(ins)
        series[m] = (means, stds, ins)
        scale_values.extend(means - stds)
        scale_values.extend(means + stds)
        scale_values.extend(ins)
        style = METHOD_STYLE[m]
        ax.fill_between(layers, means - stds, means + stds,
                        color=style["fill"], alpha=0.58, linewidth=0, zorder=1)
        ax.plot(
            layers,
            ins,
            linestyle=(0, (3.2, 2.6)),
            color=style["color"],
            linewidth=1.55,
            alpha=0.58,
            zorder=2,
        )
        ax.plot(
            layers,
            means,
            color=style["color"],
            marker=style["marker"],
            linewidth=style["linewidth"],
            markersize=style["markersize"],
            markerfacecolor=BG,
            markeredgecolor=style["color"],
            markeredgewidth=1.8,
            zorder=4 if m == "nullspace" else 3,
        )

    # Cross-colour emphasis makes the peak primary layer visible without black.
    null_means = series["nullspace"][0]
    best_i = int(np.argmax(null_means))
    best_x, best_y = layers[best_i], float(null_means[best_i])
    ax.scatter(
        [best_x],
        [best_y],
        s=150,
        facecolors="none",
        edgecolors=PINK_DARK,
        linewidths=2.1,
        zorder=6,
    )
    align = "left" if best_i < len(layers) / 2 else "right"
    x_offset = 9 if align == "left" else -9
    y_offset = -27 if best_y > 0.955 else 16
    ax.annotate(
        f"peak L{best_x}  ·  {best_y:.3f}",
        xy=(best_x, best_y),
        xytext=(x_offset, y_offset),
        textcoords="offset points",
        ha=align,
        va="center",
        fontsize=10.5,
        color=INK,
        bbox={"boxstyle": "round,pad=0.24", "fc": BG, "ec": PINK_DARK, "lw": 1.0},
        zorder=7,
    )

    ax.set_xlabel("Decoder layer")
    ax.set_ylabel("Held-out ROC–AUC (focused scale)")
    ax.set_xticks(layers)
    ax.set_ylim(*focused_auc_limits(scale_values, absolute_floor=0.50))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.tick_params(axis="both", width=0.8, length=4)

    legend_handles = [
        Line2D(
            [0], [0],
            color=METHOD_STYLE[m]["color"],
            marker=METHOD_STYLE[m]["marker"],
            markerfacecolor=BG,
            markeredgewidth=1.7,
            linewidth=METHOD_STYLE[m]["linewidth"],
            markersize=METHOD_STYLE[m]["markersize"],
            label=METHOD_STYLE[m]["label"],
        )
        for m in methods
    ]
    legend_handles.append(
        Line2D(
            [0], [0],
            color=MUTED,
            linewidth=1.6,
            linestyle=(0, (3.2, 2.6)),
            label="In-sample reference",
        )
    )
    ax.legend(
        handles=legend_handles,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.015),
        frameon=False,
        handlelength=2.8,
        columnspacing=1.8,
        borderaxespad=0,
    )

    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.16, top=0.87)
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=DPI, bbox_inches="tight", facecolor=BG)
    fig.savefig(fig_path.replace(".png", ".pdf"), bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def write_md_summary(payload: dict, md_path: str) -> None:
    """Render a paper-ready Held-out separability table (mean ± std)."""
    methods = ["raw", "nullspace"]
    layers = sorted(int(l) for l in payload["results"])
    lines = [
        "# Random pair held-out separability of the source direction",
        "",
        f"- Protocol: **{payload['protocol']}**",
        f"- N pairs: **{payload['n_total_pairs']}**  (train={payload['n_total_pairs']-int(payload['n_total_pairs']*payload['test_frac'])}, test={int(payload['n_total_pairs']*payload['test_frac'])})",
        f"- null_k = {payload['null_k']}, seed = {payload.get('seed', 'n/a')}",
        "",
        "Mean ± std over the random splits.  In-sample AUC (fit on all pairs) is shown for reference. "
        "See ``generator_holdout_*`` for the leave-one-generator-out version.",
        "",
        "| Layer | "
        + " | ".join(
            f"{m.title()} held-out AUC | {m.title()} held-out Acc"
            for m in methods
        )
        + " |",
        "|---:|" + "---:|" * (2 * len(methods)),
    ]
    for ly in layers:
        row = [f"{ly}"]
        layer_res = payload["results"][str(ly)] if str(ly) in payload["results"] else payload["results"][ly]
        for m in methods:
            r = layer_res[m]
            row.append(f"{r['heldout_auc_mean']:.3f} ± {r['heldout_auc_std']:.3f}")
            row.append(f"{r['heldout_acc_mean']:.3f} ± {r['heldout_acc_std']:.3f}")
        lines.append("| " + " | ".join(row) + " |")
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--states_glob", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n_runs", type=int, default=5)
    p.add_argument("--test_frac", type=float, default=0.2)
    p.add_argument("--null_k", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args.states_glob, args.out, args.n_runs, args.test_frac, args.null_k, args.seed)
