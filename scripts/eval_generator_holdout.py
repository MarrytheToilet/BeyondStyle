"""
Generator-held-out separability of the source direction.

For each generator g present in the paired dataset:
    * Train pairs = all samples whose AI continuation was produced by
      generators ≠ g.
    * Test pairs  = all samples whose AI continuation was produced by g.
    * For each estimator {raw, nullspace}, fit v on the train split
      and report ROC-AUC + accuracy on the held-out generator.

Per-sample states are aligned with the dataset by index (the steering
pipeline saves them in the same order as ``data["samples"]``).

Usage:
    python scripts/eval_generator_holdout.py \
        --dataset data/paired/en/proposition_aligned/pairs.jsonl \
        --states_glob "outputs/en/03_steering_vectors/Llama-3.1-8B-Instruct/proposition_aligned/last_token/states/paired_states_layer*.pt" \
        --out outputs/en/02_separability/Llama-3.1-8B-Instruct/proposition_aligned/last_token/generator_holdout.json
"""

import argparse
import glob
import json
from pathlib import Path
import re
import sys
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.steering.denoise import compute_denoised_direction
from src.data.paired_io import generator_model, load_paired_samples


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


def bootstrap_eval(v: torch.Tensor, h_te: torch.Tensor, a_te: torch.Tensor,
                   threshold: float, n_boot: int,
                   rng: np.random.Generator):
    """Resample the test pairs with replacement to get an AUC/Acc CI."""
    n = len(h_te)
    sc_h = (h_te @ v).numpy()
    sc_a = (a_te @ v).numpy()
    aucs, accs = [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sh, sa = sc_h[idx], sc_a[idx]
        sc = np.concatenate([sh, sa])
        y = np.concatenate([np.zeros(n), np.ones(n)])
        try:
            aucs.append(roc_auc_score(y, sc))
        except ValueError:
            continue
        accs.append(
            float(((sc > threshold).astype(int) == y).mean())
        )
    return aucs, accs


def short_name(model_source: str) -> str:
    """Strip provider prefix and simplify long ids for display."""
    s = model_source.split("/")[-1] if "/" in model_source else model_source
    s = s.replace("-Instruct", "").replace("Instruct-", "")
    return s


def display_name(model_source: str) -> str:
    """Paper-friendly generator name while preserving unknown model ids."""
    s = short_name(model_source)
    known = {
        "minimax-m2.5": "MiniMax-M2.5",
        "deepseek-v4-flash": "DeepSeek-V4-Flash",
        "glm-5.2": "GLM-5.2",
        "qwen3.6-flash": "Qwen3.6-Flash",
    }
    return known.get(s.lower(), s)


def run(dataset_path: str, states_glob: str, out_path: str, null_k: int,
        n_boot: int = 200, seed: int = 0) -> None:
    samples = load_paired_samples(dataset_path)
    gens = [generator_model(sample) for sample in samples]
    unique_gens = sorted(set(gens))
    if len(unique_gens) < 2:
        raise ValueError(f"Need ≥2 distinct generators in dataset; got {unique_gens}")

    idx_by_gen = defaultdict(list)
    for i, g in enumerate(gens):
        idx_by_gen[g].append(i)

    files = sorted(glob.glob(states_glob))
    if not files:
        raise FileNotFoundError(f"No state files match: {states_glob}")

    rng = np.random.default_rng(seed)
    per_layer = {}
    expected_sample_ids = [sample["sample_id"] for sample in samples]
    state_metadata = None
    for fp in files:
        m = re.search(r"layer(\d+)", fp)
        if not m:
            continue
        layer = int(m.group(1))
        d = torch.load(fp, map_location="cpu", weights_only=False)
        H, A = d["human"].float(), d["ai"].float()
        metadata = d.get("metadata") or {}
        if metadata.get("sample_ids") != expected_sample_ids:
            raise ValueError(
                f"{fp}: state sample IDs do not match the paired dataset"
            )
        if state_metadata is None:
            state_metadata = metadata
        elif (
            metadata.get("dataset_sha256")
            != state_metadata.get("dataset_sha256")
            or metadata.get("pooling") != state_metadata.get("pooling")
        ):
            raise ValueError(f"State metadata mismatch at {fp}")
        if H.shape[0] != len(samples):
            print(f"  ⚠ L{layer}: states ({H.shape[0]}) ≠ samples ({len(samples)}); skipping")
            continue

        per_gen = {}
        for held in unique_gens:
            te = sorted(idx_by_gen[held])
            held_indices = set(idx_by_gen[held])
            tr = [
                i for i in range(len(samples))
                if i not in held_indices
            ]
            te_t = torch.tensor(te, dtype=torch.long)
            tr_t = torch.tensor(tr, dtype=torch.long)

            method_res = {}
            for method in ["raw", "nullspace"]:
                v = fit_v(method, H[tr_t], A[tr_t], null_k)
                threshold = fit_threshold(v, H[tr_t], A[tr_t])
                auc, acc = eval_v(
                    v,
                    H[te_t],
                    A[te_t],
                    threshold,
                )
                b_aucs, b_accs = bootstrap_eval(
                    v,
                    H[te_t],
                    A[te_t],
                    threshold,
                    n_boot,
                    rng,
                )
                lo, hi = np.percentile(b_aucs, [2.5, 97.5]) if b_aucs else (auc, auc)
                acc_lo, acc_hi = np.percentile(b_accs, [2.5, 97.5]) if b_accs else (acc, acc)
                method_res[method] = {
                    "auc": auc, "acc": acc,
                    "auc_boot_mean": float(np.mean(b_aucs)) if b_aucs else auc,
                    "auc_boot_std":  float(np.std(b_aucs))  if b_aucs else 0.0,
                    "auc_ci_lo": float(lo), "auc_ci_hi": float(hi),
                    "acc_boot_mean": float(np.mean(b_accs)) if b_accs else acc,
                    "acc_boot_std":  float(np.std(b_accs))  if b_accs else 0.0,
                    "acc_ci_lo": float(acc_lo), "acc_ci_hi": float(acc_hi),
                    "n_train": len(tr), "n_test": len(te),
                }
            per_gen[held] = method_res

        per_layer[layer] = per_gen
        line = f"L{layer:02d}  " + " | ".join(
            f"{short_name(g)}: " + "/".join(
                f"{m}={per_gen[g][m]['auc']:.3f}"
                f"[{per_gen[g][m]['auc_ci_lo']:.3f},{per_gen[g][m]['auc_ci_hi']:.3f}]"
                for m in ["raw", "nullspace"]
            )
            for g in unique_gens
        )
        print(line)

    payload = {
        "protocol": (
            "Leave-one-generator-out: for each generator g, train v on "
            "samples with model_source ≠ g, including the accuracy threshold; "
            "evaluate AUC/Acc on samples with model_source = g. Confidence "
            "intervals are 95% "
            f"percentile bootstraps over the test pairs (B={n_boot})."
        ),
        "n_total_pairs": len(samples),
        "generators": {g: len(idx_by_gen[g]) for g in unique_gens},
        "null_k": null_k,
        "n_boot": n_boot,
        "seed": seed,
        "representation": (state_metadata or {}).get("pooling"),
        "dataset_path": (state_metadata or {}).get("dataset_path"),
        "dataset_sha256": (state_metadata or {}).get("dataset_sha256"),
        "results": per_layer,
    }
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved -> {out_path}")

    md_path = str(Path(out_path).with_suffix("")) + "_summary.md"
    write_md_summary(payload, md_path)
    print(f"Saved -> {md_path}")

    fig_path = str(Path(out_path).with_suffix("")) + "_per_generator.png"
    plot_per_generator(payload, fig_path)
    print(f"Saved -> {fig_path}")


def write_md_summary(payload: dict, md_path: str) -> None:
    methods = ["raw", "nullspace"]
    layers = sorted(int(l) for l in payload["results"])
    gens = list(payload["generators"].keys())

    lines = [
        "# Generator-held-out separability of the source direction",
        "",
        f"- Protocol: **{payload['protocol']}**",
        f"- N total pairs: **{payload['n_total_pairs']}**",
        f"- Generators (n samples): "
        + ", ".join(f"`{short_name(g)}` ({n})" for g, n in payload["generators"].items()),
        f"- null_k = {payload['null_k']}",
        "",
        "Per-layer ROC-AUC on the held-out generator. Each cell is `raw / nullspace`.",
        "",
        "| Layer | " + " | ".join(short_name(g) for g in gens) + " |",
        "|---:|" + "---:|" * len(gens),
    ]
    for ly in layers:
        per_gen = payload["results"][str(ly)] if str(ly) in payload["results"] else payload["results"][ly]
        row = [f"{ly}"]
        for g in gens:
            r = per_gen[g]
            row.append(" / ".join(f"{r[m]['auc']:.3f}" for m in methods))
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Mean across layers (per held-out generator)",
        "",
        "| Generator | "
        + " | ".join(f"{m.title()} AUC | {m.title()} Acc" for m in methods)
        + " |",
        "|:---|" + "---:|" * (2 * len(methods)),
    ]
    for g in gens:
        row = [f"`{short_name(g)}` (n={payload['generators'][g]})"]
        for m in methods:
            aucs = [payload["results"][str(ly) if str(ly) in payload["results"] else ly][g][m]["auc"]
                    for ly in layers]
            accs = [payload["results"][str(ly) if str(ly) in payload["results"] else ly][g][m]["acc"]
                    for ly in layers]
            row.append(f"{np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
            row.append(f"{np.mean(accs):.3f} ± {np.std(accs):.3f}")
        lines.append("| " + " | ".join(row) + " |")

    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def plot_per_generator(payload: dict, fig_path: str) -> None:
    """Compact title-free small multiples for leave-one-generator-out AUC."""
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
        PINK_DARK,
        focused_auc_limits,
        setup_separability_style,
    )
    setup_separability_style()

    methods = ["nullspace", "raw"]

    layers = sorted(int(l) for l in payload["results"])
    gens = list(payload["generators"].keys())
    n = len(gens)
    ncol = 2 if n > 2 else n
    nrow = (n + ncol - 1) // ncol
    all_values = []
    for ly in layers:
        per_gen = payload["results"][str(ly) if str(ly) in payload["results"] else ly]
        for g in gens:
            for m in methods:
                r = per_gen[g][m]
                all_values.extend([
                    r.get("auc_ci_lo", r["auc"]),
                    r["auc"],
                    r.get("auc_ci_hi", r["auc"]),
                ])
    y_limits = focused_auc_limits(all_values, absolute_floor=0.40)

    fig, axes = plt.subplots(nrow, ncol, figsize=(5.25 * ncol, 3.25 * nrow),
                             facecolor=BG, sharey=True, squeeze=False)

    for i, g in enumerate(gens):
        ax = axes[i // ncol][i % ncol]
        null_ys = None
        for m in methods:
            ys, los, his = [], [], []
            for ly in layers:
                r = payload["results"][str(ly) if str(ly) in payload["results"] else ly][g][m]
                ys.append(r["auc"])
                los.append(r.get("auc_ci_lo", r["auc"]))
                his.append(r.get("auc_ci_hi", r["auc"]))
            ys = np.asarray(ys); los = np.asarray(los); his = np.asarray(his)
            if m == "nullspace":
                null_ys = ys
            style = METHOD_STYLE[m]
            ax.fill_between(
                layers,
                los,
                his,
                color=style["fill"],
                alpha=0.56,
                linewidth=0,
                zorder=1,
            )
            ax.plot(
                layers,
                ys,
                color=style["color"],
                marker=style["marker"],
                linewidth=style["linewidth"],
                markersize=style["markersize"] - 0.7,
                markerfacecolor=BG,
                markeredgecolor=style["color"],
                markeredgewidth=1.6,
                zorder=4 if m == "nullspace" else 3,
            )

        ax.text(
            0.025,
            1.035,
            f"({chr(97 + i)})  {display_name(g)}   ·   n = {payload['generators'][g]}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=11.4,
            fontweight="semibold",
            color=INK,
            clip_on=False,
            zorder=8,
        )

        best_i = int(np.argmax(null_ys))
        best_x, best_y = layers[best_i], float(null_ys[best_i])
        ax.scatter(
            [best_x],
            [best_y],
            s=118,
            facecolors="none",
            edgecolors=PINK_DARK,
            linewidths=1.9,
            zorder=7,
        )
        ha = "left" if best_i < len(layers) / 2 else "right"
        ax.annotate(
            f"L{best_x}",
            xy=(best_x, best_y),
            xytext=(7 if ha == "left" else -7, -18 if best_y > 0.965 else 12),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=9.8,
            fontweight="semibold",
            color=PINK_DARK,
            zorder=8,
        )

        ax.set_xticks(layers)
        ax.set_ylim(*y_limits)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(True, axis="y")
        ax.grid(False, axis="x")
        ax.tick_params(axis="both", width=0.8, length=3.6)

    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    legend_handles = [
        Line2D(
            [0], [0],
            color=METHOD_STYLE[m]["color"],
            marker=METHOD_STYLE[m]["marker"],
            markerfacecolor=BG,
            markeredgewidth=1.6,
            linewidth=METHOD_STYLE[m]["linewidth"],
            markersize=METHOD_STYLE[m]["markersize"],
            label=f"{METHOD_STYLE[m]['label']} (95% bootstrap CI)",
        )
        for m in methods
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
        handlelength=2.8,
        columnspacing=2.4,
    )
    fig.supxlabel("Decoder layer", fontsize=13.0, color=INK, y=0.018)
    fig.supylabel(
        "Held-out ROC–AUC (focused scale)",
        fontsize=13.0,
        color=INK,
        x=0.018,
    )
    fig.subplots_adjust(
        left=0.095,
        right=0.985,
        bottom=0.105,
        top=0.90,
        wspace=0.13,
        hspace=0.32,
    )
    Path(fig_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, dpi=DPI, bbox_inches="tight", facecolor=BG)
    fig.savefig(fig_path.replace(".png", ".pdf"), bbox_inches="tight", facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--states_glob", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--null_k", type=int, default=10)
    p.add_argument("--n_boot", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    run(args.dataset, args.states_glob, args.out, args.null_k,
        n_boot=args.n_boot, seed=args.seed)
