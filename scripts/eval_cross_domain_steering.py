#!/usr/bin/env python3
"""Cross-domain steering: apply a direction trained on one dataset to prompts
sampled from a different dataset.

Default: CNN/DM-trained layer-15 NULLSPACE direction, evaluated on OpenWebText
prefixes.

Outputs JSON in the same format as ``steering_pipeline``'s
``generation_{method}.json``, written under
  outputs/{lang}/03_steering_vectors/{model}/{tgt_tag}/generation/from_{src_tag}/
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import get_model_path, get_model_id
from src.steering import MeanDiffSteering
from src.generation import SteeredGenerator, SteeredGenerationConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=None,
                   help="Local model path (default: LOCAL_MODEL_PATH)")
    p.add_argument("--language", default="en")
    p.add_argument("--src-tag", default="mixed_500",
                   help="Tag of dataset whose direction we use (CNN/DM)")
    p.add_argument("--tgt-dataset",
                   default="data/paired/en/paired_openwebtext_mixed_300.json",
                   help="Paired JSON whose prefixes we use as test prompts")
    p.add_argument("--tgt-tag", default="openwebtext_mixed_300")
    p.add_argument("--method-key", default="mean_diff_nullspace",
                   help="Steering model key (e.g. mean_diff, mean_diff_nullspace)")
    p.add_argument("--layers", type=int, nargs="+", default=[11, 15, 20])
    p.add_argument("--alphas", type=float, nargs="+", default=[-2.0, 0.0, 2.0])
    p.add_argument("--n-prompts", type=int, default=100)
    p.add_argument("--max-new-tokens", type=int, default=192)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    return p.parse_args()


def main():
    args = parse_args()

    model_path = args.model or get_model_path()
    model_id = get_model_id(model_path)
    base = Path("outputs") / args.language / "03_steering_vectors" / model_id
    src_models_dir = base / args.src_tag / "models"
    out_dir = base / args.tgt_tag / "generation" / f"from_{args.src_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.tgt_dataset) as f:
        data = json.load(f)
    samples = data["samples"]
    rng = random.Random(args.seed)
    rng.shuffle(samples)
    prompts = [s["prefix"] for s in samples[:args.n_prompts]]
    print(f"Loaded {len(prompts)} prompts from {args.tgt_dataset}")

    print(f"Loading model: {model_path}")
    generator = SteeredGenerator(model_path, device=args.device)

    method_results = []
    cfg = SteeredGenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    for ly in args.layers:
        mp = src_models_dir / f"{args.method_key}_layer{ly:02d}.pt"
        if not mp.exists():
            print(f"  [skip] missing {mp}")
            continue
        steer = MeanDiffSteering.load(mp)
        print(f"\n── L{ly} ── direction: {mp.name}")

        for i, prompt in enumerate(prompts):
            outputs = generator.generate_comparison(
                prompt=prompt, steering=steer,
                strengths=args.alphas, config=cfg,
            )
            entry = {
                "layer": ly,
                "method": args.method_key,
                "prompt": prompt,
                "src_tag": args.src_tag,
                "tgt_tag": args.tgt_tag,
                "sample_id": samples[i].get("sample_id"),
                "alpha_direction": "negative=toward_human, positive=toward_ai",
                "outputs": {},
            }
            for a in sorted(outputs.keys()):
                label = "toward_human" if a < 0 else ("toward_ai" if a > 0 else "baseline")
                entry["outputs"][str(a)] = {"direction": label, "text": outputs[a]}
            method_results.append(entry)
            if (i + 1) % 10 == 0:
                print(f"  L{ly}: {i+1}/{len(prompts)} prompts done")

    out_path = out_dir / f"generation_{args.method_key}.json"
    with open(out_path, "w") as f:
        json.dump(method_results, f, indent=2, ensure_ascii=False)
    print(f"\nSaved → {out_path}  ({len(method_results)} entries)")


if __name__ == "__main__":
    main()
