#!/usr/bin/env python3
"""Cross-domain transfer evaluation.

For each (denoise method, layer):
  - load source-domain steering direction
  - project target-domain held-out hidden states
  - report AUC, accuracy, projection gap, Cohen's d

Default: CNN/DM (mixed_500)  →  OpenWebText (openwebtext_mixed_300) and reverse.
Outputs:
  outputs/en/02_separability/{model_id}/cross_domain/transfer_summary.{json,md}
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extraction import PairedHiddenStates
from src.steering import MeanDiffSteering


def project_and_score(v: torch.Tensor, ps: PairedHiddenStates) -> dict:
    h = ps.human.float()
    a = ps.ai.float()
    v = v.float().to(h.device)
    hp = (h @ v).cpu().numpy()
    ap = (a @ v).cpu().numpy()
    scores = np.concatenate([hp, ap])
    labels = np.concatenate([np.zeros(len(hp)), np.ones(len(ap))])
    auc = float(roc_auc_score(labels, scores))
    threshold = (hp.mean() + ap.mean()) / 2
    preds = (scores > threshold).astype(int)
    acc = float((preds == labels).mean())
    gap = float(ap.mean() - hp.mean())
    pooled_std = float(np.sqrt((hp.var(ddof=1) + ap.var(ddof=1)) / 2))
    cohens_d = gap / pooled_std if pooled_std > 0 else 0.0
    return dict(
        auc=auc, accuracy=acc, projection_gap=gap, cohens_d=cohens_d,
        n_human=len(hp), n_ai=len(ap),
        human_proj_mean=float(hp.mean()),
        ai_proj_mean=float(ap.mean()),
    )


def evaluate_transfer(src_models_dir: Path, tgt_states_dir: Path,
                      layers: list[int], denoise_methods: list[str]) -> dict:
    rows = []
    for denoise in denoise_methods:
        method_key = "mean_diff" if denoise == "raw" else f"mean_diff_{denoise}"
        for ly in layers:
            mp = src_models_dir / f"{method_key}_layer{ly:02d}.pt"
            sp = tgt_states_dir / f"paired_states_layer{ly:02d}.pt"
            if not mp.exists():
                print(f"  [skip] missing model {mp}")
                continue
            if not sp.exists():
                print(f"  [skip] missing target states {sp}")
                continue
            steer = MeanDiffSteering.load(mp)
            v = steer.get_direction()
            ps = PairedHiddenStates.load(sp)
            metrics = project_and_score(v, ps)
            metrics.update(layer=ly, denoise=denoise, method_key=method_key)
            rows.append(metrics)
            print(f"  {denoise:10s} L{ly:02d}  AUC={metrics['auc']:.3f}  "
                  f"acc={metrics['accuracy']:.1%}  gap={metrics['projection_gap']:+.3f}  "
                  f"d={metrics['cohens_d']:+.2f}")
    return rows


def write_summary(out_path: Path, sections: dict[str, dict]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump(sections, f, indent=2)

    md = ["# Cross-Domain Transfer Summary", ""]
    for name, sec in sections.items():
        md.append(f"## {name}")
        md.append("")
        md.append(f"- source direction: `{sec['source']}`")
        md.append(f"- target states:    `{sec['target']}`")
        md.append("")
        md.append("| denoise | layer | AUC | acc | proj gap | Cohen's d |")
        md.append("|---|---|---|---|---|---|")
        for r in sec["rows"]:
            md.append(
                f"| {r['denoise']} | {r['layer']} | {r['auc']:.3f} | "
                f"{r['accuracy']:.1%} | {r['projection_gap']:+.3f} | {r['cohens_d']:+.2f} |"
            )
        md.append("")
    with open(out_path.with_suffix(".md"), "w") as f:
        f.write("\n".join(md))
    print(f"\nSaved → {out_path.with_suffix('.json')}")
    print(f"Saved → {out_path.with_suffix('.md')}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default="Llama-3.1-8B-Instruct")
    p.add_argument("--language", default="en")
    p.add_argument("--src-tag", default="mixed_500", help="Source dataset tag (CNN/DM)")
    p.add_argument("--tgt-tag", default="openwebtext_mixed_300", help="Target dataset tag (OWT)")
    p.add_argument("--layers", type=int, nargs="+", default=[7, 11, 15, 20, 24, 28])
    p.add_argument("--denoise", nargs="+", default=["raw", "nullspace"])
    p.add_argument("--reverse", action="store_true",
                   help="Also evaluate the reverse direction (target→source).")
    p.add_argument("--out", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    base = Path("outputs") / args.language / "03_steering_vectors" / args.model_id

    src_models = base / args.src_tag / "models"
    src_states = base / args.src_tag / "states"
    tgt_models = base / args.tgt_tag / "models"
    tgt_states = base / args.tgt_tag / "states"

    sections = {}

    print(f"\n=== {args.src_tag} → {args.tgt_tag} ===")
    rows = evaluate_transfer(src_models, tgt_states, args.layers, args.denoise)
    sections[f"{args.src_tag} → {args.tgt_tag}"] = {
        "source": str(src_models),
        "target": str(tgt_states),
        "rows": rows,
    }

    if args.reverse:
        print(f"\n=== {args.tgt_tag} → {args.src_tag} ===")
        rows = evaluate_transfer(tgt_models, src_states, args.layers, args.denoise)
        sections[f"{args.tgt_tag} → {args.src_tag}"] = {
            "source": str(tgt_models),
            "target": str(src_states),
            "rows": rows,
        }

    out_dir = (Path("outputs") / args.language / "02_separability"
               / args.model_id / "cross_domain")
    out_path = Path(args.out) if args.out else (out_dir / "transfer_summary")
    write_summary(out_path, sections)


if __name__ == "__main__":
    main()
