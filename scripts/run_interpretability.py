#!/usr/bin/env python3
"""Interpretability analysis of steering vectors.

Analyzes v_src from lexical and syntactic perspectives:
  - Logit Lens: which tokens does v_src promote/suppress?
  - Token projection: which words carry the strongest source signal?
  - Syntactic analysis: POS/dep correlations, structural R²

Usage:
    # Default: best layer (L16), all 3 methods
    python scripts/run_interpretability.py

    # Specific layers and methods
    python scripts/run_interpretability.py --layers 10 14 16 20 --methods mean_diff mean_diff_fisher

    # Quick test (fewer samples)
    python scripts/run_interpretability.py --max_samples 20 --layers 16
"""

import argparse
import json
import sys
import torch
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    get_model_path, get_model_id, get_steering_dir,
    get_dataset_path, extract_dataset_tag, get_output_base,
)


def parse_args():
    p = argparse.ArgumentParser(description="Steering vector interpretability")
    p.add_argument("--model", default=None, help="Model path (default: from .env)")
    p.add_argument("--dataset", default=None, help="Dataset path")
    p.add_argument("--layers", type=int, nargs="+", default=[16],
                   help="Layers to analyze (default: 16)")
    p.add_argument("--methods", type=str, nargs="+",
                   default=["mean_diff", "mean_diff_nullspace", "mean_diff_fisher"],
                   help="Steering method keys to analyze")
    p.add_argument("--top_k", type=int, default=50, help="Top-k features to report")
    p.add_argument("--max_samples", type=int, default=100,
                   help="Max samples for token/syntax analysis")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default=None,
                   help="Output JSON path (default: outputs/interp/{model_id}/...)")
    p.add_argument("--keep_raw", action="store_true",
                   help="Keep raw per-token projection data (large)")
    p.add_argument("--skip_plots", action="store_true",
                   help="Skip figure generation")
    # SAE options
    p.add_argument("--skip_sae", action="store_true",
                   help="Skip SAE training and decomposition")
    p.add_argument("--sae_expansion", type=int, default=4,
                   help="SAE dictionary size = d_model × expansion (default: 4)")
    p.add_argument("--sae_l1", type=float, default=5e-3,
                   help="SAE L1 sparsity coefficient (default: 5e-3)")
    p.add_argument("--sae_epochs", type=int, default=200,
                   help="SAE training epochs (default: 200)")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve paths
    model_path = args.model or get_model_path()
    model_p = Path(model_path).expanduser().resolve()
    if model_p.is_dir():
        model_path = str(model_p)
    model_id = get_model_id(model_path)

    dataset_path = args.dataset or get_dataset_path()
    dataset_tag = extract_dataset_tag(dataset_path)
    steer_dir = get_steering_dir(model_id, dataset_tag=dataset_tag or None)
    models_dir = steer_dir / "models"

    # Output path
    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = get_output_base() / "interp" / model_id / (dataset_tag or "default")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "interpretability_results.json"

    print("=" * 60)
    print("Steering Vector Interpretability Analysis")
    print("=" * 60)
    print(f"  Model    : {model_path}")
    print(f"  Model ID : {model_id}")
    print(f"  Dataset  : {dataset_path}")
    print(f"  Steer dir: {steer_dir}")
    print(f"  Layers   : {args.layers}")
    print(f"  Methods  : {args.methods}")
    print(f"  Samples  : {args.max_samples}")
    print(f"  Output   : {out_path}")

    # Check which models actually exist
    available = []
    for method in args.methods:
        for layer in args.layers:
            p = models_dir / f"{method}_layer{layer:02d}.pt"
            if p.exists():
                available.append(f"{method}_L{layer}")
            else:
                print(f"  ⚠ Not found: {p.name}")
    print(f"  Available: {len(available)} combinations")

    if not available:
        print("\n❌ No steering models found. Run steering_pipeline.py first.")
        sys.exit(1)

    # Load model
    print(f"\n{'='*60}")
    print("Loading model...")
    print("=" * 60)
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer

    extra_kw = {}
    if os.path.isdir(model_path):
        extra_kw["local_files_only"] = True

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, **extra_kw)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map=args.device, trust_remote_code=True, **extra_kw)
    model.eval()
    print(f"  Model loaded: {model.config.num_hidden_layers} layers, "
          f"hidden_size={model.config.hidden_size}")

    # Load dataset samples
    print(f"\nLoading dataset...")
    with open(dataset_path) as f:
        samples = json.load(f)["samples"]
    print(f"  {len(samples)} samples loaded")

    # ── Part A: Lexical + Syntactic analysis (requires model on GPU) ──
    print(f"\n{'='*60}")
    print("Part A: Lexical + Syntactic Analysis")
    print("=" * 60)

    from src.steering.interpretability import run_interpretability

    results = run_interpretability(
        model=model,
        tokenizer=tokenizer,
        models_dir=models_dir,
        samples=samples,
        layers=args.layers,
        methods=args.methods,
        device=args.device,
        top_k=args.top_k,
        max_samples=args.max_samples,
        keep_raw=args.keep_raw,
    )

    # Free GPU for SAE training
    del model
    torch.cuda.empty_cache()

    # ── Part B: SAE decomposition (train custom SAE per layer) ──
    if not args.skip_sae:
        print(f"\n{'='*60}")
        print("Part B: SAE Training + Decomposition")
        print("=" * 60)

        from src.steering.sae import train_and_decompose, SAETrainConfig, compute_feature_overlap

        sae_config = SAETrainConfig(
            d_sae_multiplier=args.sae_expansion,
            l1_coeff=args.sae_l1,
            epochs=args.sae_epochs,
        )
        sae_out_dir = out_path.parent / "sae_models"

        decompositions, sae_train_metrics = train_and_decompose(
            states_dir=steer_dir / "states",
            models_dir=models_dir,
            layers=args.layers,
            methods=args.methods,
            train_config=sae_config,
            device=args.device,
            top_k=args.top_k,
            sae_out_dir=sae_out_dir,
        )

        # Feature overlap analysis
        overlap = compute_feature_overlap(decompositions, top_k=30)

        results["sae"] = {
            "train_metrics": {str(k): {kk: vv for kk, vv in v.items() if kk != "history"}
                              for k, v in sae_train_metrics.items()},
            "decompositions": {k: v.to_dict(include_activations=True)
                               for k, v in decompositions.items()},
            "feature_overlap": overlap,
        }

        print(f"\n  Feature overlap (top-30):")
        for pair, info in overlap.items():
            print(f"    {pair}: Jaccard={info['jaccard']:.3f} "
                  f"(∩={info['intersection']}, ∪={info['union']})")

    # Save results
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  ✓ Results saved to {out_path}")

    # Generate plots
    if not args.skip_plots:
        print(f"\n{'='*60}")
        print("Generating Figures")
        print("=" * 60)
        fig_dir = out_path.parent / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        from src.visualization.interp_plots import plot_all_interpretability
        plot_all_interpretability(results, fig_dir)

    print(f"\n{'='*60}")
    print(f"✅ Interpretability analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
