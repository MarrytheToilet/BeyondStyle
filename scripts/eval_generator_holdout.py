"""
Generator-held-out separability of the source direction.

For each generator g present in the paired dataset:
    * Train pairs = all samples whose AI continuation was produced by
      generators ≠ g.
    * Test pairs  = all samples whose AI continuation was produced by g.
    * For each estimator {raw, nullspace, fisher}, fit v on the train split
      and report ROC-AUC + accuracy on the held-out generator.

Per-sample states are aligned with the dataset by index (the steering
pipeline saves them in the same order as ``data["samples"]``).

Usage:
    python scripts/eval_generator_holdout.py \
        --dataset data/paired/en/paired_mixed_500.json \
        --states_glob "outputs/en/03_steering_vectors/Llama-3.1-8B-Instruct/mixed_500/states/paired_states_layer*.pt" \
        --out outputs/en/02_separability/Llama-3.1-8B-Instruct/mixed_500/generator_holdout.json
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

ROOT = os.path.join(os.path.dirname(__file__), os.pardir)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
from steering.denoise import compute_denoised_direction


def fit_v(method: str, h_tr: torch.Tensor, a_tr: torch.Tensor, null_k: int) -> torch.Tensor:
    if method == "raw":
        v = (a_tr - h_tr).mean(dim=0)
        return v / v.norm()
    v, _ = compute_denoised_direction(h_tr, a_tr, method=method, null_k=null_k)
    return v


def eval_v(v: torch.Tensor, h_te: torch.Tensor, a_te: torch.Tensor):
    sc = torch.cat([h_te @ v, a_te @ v]).numpy()
    y = np.concatenate([np.zeros(len(h_te)), np.ones(len(a_te))])
    auc = float(roc_auc_score(y, sc))
    thr = (sc[y == 0].mean() + sc[y == 1].mean()) / 2
    pred = (sc > thr).astype(int)
    acc = float((pred == y).mean())
    return auc, acc


def bootstrap_eval(v: torch.Tensor, h_te: torch.Tensor, a_te: torch.Tensor,
                   n_boot: int, rng: np.random.Generator):
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
        thr = (sh.mean() + sa.mean()) / 2
        accs.append(float(((sc > thr).astype(int) == y).mean()))
    return aucs, accs


def short_name(model_source: str) -> str:
    """Strip provider prefix and simplify long ids for display."""
    s = model_source.split("/")[-1] if "/" in model_source else model_source
    s = s.replace("-Instruct", "").replace("Instruct-", "")
    return s


def run(dataset_path: str, states_glob: str, out_path: str, null_k: int,
        n_boot: int = 200, seed: int = 0) -> None:
    with open(dataset_path) as f:
        samples = json.load(f)["samples"]
    gens = [s.get("metadata", {}).get("model_source", "?") for s in samples]
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
    for fp in files:
        m = re.search(r"layer(\d+)", fp)
        if not m:
            continue
        layer = int(m.group(1))
        d = torch.load(fp, map_location="cpu", weights_only=False)
        H, A = d["human"].float(), d["ai"].float()
        if H.shape[0] != len(samples):
            print(f"  ⚠ L{layer}: states ({H.shape[0]}) ≠ samples ({len(samples)}); skipping")
            continue

        per_gen = {}
        for held in unique_gens:
            te = sorted(idx_by_gen[held])
            tr = sorted(i for i in range(len(samples)) if i not in idx_by_gen[held])
            te_t = torch.tensor(te, dtype=torch.long)
            tr_t = torch.tensor(tr, dtype=torch.long)

            method_res = {}
            for method in ["raw", "nullspace", "fisher"]:
                v = fit_v(method, H[tr_t], A[tr_t], null_k)
                auc, acc = eval_v(v, H[te_t], A[te_t])
                b_aucs, b_accs = bootstrap_eval(v, H[te_t], A[te_t], n_boot, rng)
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
                for m in ["raw", "nullspace", "fisher"]
            )
            for g in unique_gens
        )
        print(line)

    payload = {
        "protocol": (
            "Leave-one-generator-out: for each generator g, train v on "
            "samples with model_source ≠ g, evaluate AUC/Acc on samples "
            "with model_source = g.  Confidence intervals are 95% "
            f"percentile bootstraps over the test pairs (B={n_boot})."
        ),
        "n_total_pairs": len(samples),
        "generators": {g: len(idx_by_gen[g]) for g in unique_gens},
        "null_k": null_k,
        "n_boot": n_boot,
        "seed": seed,
        "results": per_layer,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved -> {out_path}")

    md_path = os.path.splitext(out_path)[0] + "_summary.md"
    write_md_summary(payload, md_path)
    print(f"Saved -> {md_path}")

    fig_path = os.path.splitext(out_path)[0] + "_per_generator.png"
    plot_per_generator(payload, fig_path)
    print(f"Saved -> {fig_path}")


def write_md_summary(payload: dict, md_path: str) -> None:
    methods = ["raw", "nullspace", "fisher"]
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
        "Per-layer ROC-AUC on the held-out generator. Each cell is `raw / nullspace / fisher`.",
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
    """One subplot per held-out generator; lines per method across layers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, ROOT)
    from src.visualization.palette import setup_style, H_D, A_D, PALETTE, BG, DPI
    setup_style()

    methods = ["raw", "nullspace", "fisher"]
    colors = {"raw": A_D, "nullspace": H_D, "fisher": PALETTE["accent"]}
    fills = {"raw": PALETTE["ai"], "nullspace": PALETTE["human"], "fisher": "#D7D2F4"}

    layers = sorted(int(l) for l in payload["results"])
    gens = list(payload["generators"].keys())
    n = len(gens)
    ncol = min(n, 4)
    nrow = (n + ncol - 1) // ncol

    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 3.0 * nrow),
                             facecolor=BG, sharey=True, squeeze=False)

    for i, g in enumerate(gens):
        ax = axes[i // ncol][i % ncol]
        for m in methods:
            ys, los, his = [], [], []
            for ly in layers:
                r = payload["results"][str(ly) if str(ly) in payload["results"] else ly][g][m]
                ys.append(r["auc"])
                los.append(r.get("auc_ci_lo", r["auc"]))
                his.append(r.get("auc_ci_hi", r["auc"]))
            ys = np.asarray(ys); los = np.asarray(los); his = np.asarray(his)
            ax.fill_between(layers, los, his, color=fills[m], alpha=0.45, linewidth=0)
            ax.plot(layers, ys, "-o", color=colors[m], linewidth=1.8,
                    markersize=4.5, label=m)
        ax.set_title(f"held-out: {short_name(g)}\n(n={payload['generators'][g]})",
                     fontsize=10)
        ax.set_xticks(layers)
        ax.set_xlabel("Layer")
        ax.set_ylim(0.85, 1.005)
        ax.grid(True, axis="y", alpha=0.3)
        if i % ncol == 0:
            ax.set_ylabel("ROC-AUC")

    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Generator-held-out separability  (train: other generators · test: held-out generator)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    os.makedirs(os.path.dirname(fig_path) or ".", exist_ok=True)
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
