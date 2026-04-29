"""
Held-out separability protocol for the source direction.

For each layer's cached paired hidden-state file, for each estimator
{raw, nullspace, fisher}, run R random 80/20 splits over pairs:
    1. Estimate v on the train pairs only.
    2. Project held-out test pairs and compute ROC-AUC and accuracy.
Also report the in-sample AUC fitted on all pairs for comparison.

Usage:
    python scripts/eval_heldout_separability.py \
        --states_glob "outputs/en/03_steering_vectors/Llama-3.1-8B-Instruct/mixed_500/states/paired_states_layer*.pt" \
        --out outputs/en/02_separability/Llama-3.1-8B-Instruct/mixed_500/heldout_auc.json
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), os.pardir, "src"))
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


def run(states_glob: str, out_path: str, n_runs: int, test_frac: float, null_k: int, seed: int):
    files = sorted(glob.glob(states_glob))
    if not files:
        raise FileNotFoundError(f"No state files match: {states_glob}")
    results = {}
    for fp in files:
        m = re.search(r"layer(\d+)", fp)
        if not m:
            continue
        layer = int(m.group(1))
        d = torch.load(fp, map_location="cpu", weights_only=False)
        H, A = d["human"].float(), d["ai"].float()
        N = len(H)
        rng = np.random.default_rng(seed)
        perms = [rng.permutation(N) for _ in range(n_runs)]
        n_test = int(N * test_frac)

        layer_res = {}
        for method in ["raw", "nullspace", "fisher"]:
            aucs, accs = [], []
            for perm in perms:
                te, tr = perm[:n_test], perm[n_test:]
                v = fit_v(method, H[tr], A[tr], null_k)
                auc, acc = eval_v(v, H[te], A[te])
                aucs.append(auc)
                accs.append(acc)
            v_full = fit_v(method, H, A, null_k)
            in_auc, in_acc = eval_v(v_full, H, A)
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
            for m in ["raw", "nullspace", "fisher"]
        )
        print(line)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    payload = {
        "protocol": (
            f"{n_runs}x random {int((1-test_frac)*100)}/{int(test_frac*100)} splits, "
            "fit v on train pairs, evaluate AUC + accuracy on held-out test pairs"
        ),
        "n_total_pairs": N,
        "test_frac": test_frac,
        "n_runs": n_runs,
        "null_k": null_k,
        "seed": seed,
        "results": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved -> {out_path}")

    md_path = os.path.splitext(out_path)[0] + "_summary.md"
    write_md_summary(payload, md_path)
    print(f"Saved -> {md_path}")


def write_md_summary(payload: dict, md_path: str) -> None:
    """Render a paper-ready Held-out separability table (mean ± std)."""
    methods = ["raw", "nullspace", "fisher"]
    layers = sorted(int(l) for l in payload["results"])
    lines = [
        "# Held-out separability of the source direction",
        "",
        f"- Protocol: **{payload['protocol']}**",
        f"- N pairs: **{payload['n_total_pairs']}**  (train={payload['n_total_pairs']-int(payload['n_total_pairs']*payload['test_frac'])}, test={int(payload['n_total_pairs']*payload['test_frac'])})",
        f"- null_k = {payload['null_k']}, seed = {payload.get('seed', 'n/a')}",
        "",
        "Mean ± std over the random splits.  In-sample AUC (fit on all pairs) is shown for reference.",
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
