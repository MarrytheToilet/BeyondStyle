#!/usr/bin/env python3
"""Interpretability analysis of steering vectors (v2).

Analyzes v_src from three perspectives:
  - Logit Lens: which *meaningful* tokens does v_src promote/suppress?
  - Contrastive Token Projection: same token in H vs A context → Δ
  - Structural Fingerprint: discourse markers, hedging, passive voice,
    within-source regression to isolate genuine structural effects
  - SAE Decomposition: sparse dictionary features (with tuned sparsity)

Usage:
    python scripts/run_interpretability.py
    python scripts/run_interpretability.py --layers 10 14 16 20 --methods mean_diff mean_diff_fisher
    python scripts/run_interpretability.py --max_samples 20 --layers 16 --skip_sae
"""

import argparse
import json
import sys
import torch
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    get_model_path, get_model_id, get_steering_dir,
    get_dataset_path, extract_dataset_tag, get_output_base,
    get_interp_dir, resolve_language,
)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def parse_args():
    p = argparse.ArgumentParser(description="Steering vector interpretability (v2)")
    p.add_argument("--model", default=None, help="Model path (default: from .env)")
    p.add_argument("--dataset", default=None, help="Dataset path")
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="Layers to analyze (default: auto-detect from saved models)")
    p.add_argument("--n_layers", type=int, default=4,
                   help="When --layers is omitted, pick N evenly-spaced layers (default: 4)")
    p.add_argument("--methods", type=str, nargs="+",
                   default=["mean_diff", "mean_diff_nullspace", "mean_diff_fisher"],
                   help="Steering method keys to analyze")
    p.add_argument("--top_k", type=int, default=50, help="Top-k features to report")
    p.add_argument("--max_samples", type=int, default=100,
                   help="Max samples for token/syntax analysis")
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default=None)
    p.add_argument("--keep_raw", action="store_true")
    p.add_argument("--skip_plots", action="store_true")
    p.add_argument("--skip_sae", action="store_true")
    p.add_argument("--skip_deep", action="store_true",
                   help="Skip deep structural analysis (n-grams, constructions, etc.)")
    p.add_argument("--skip_vector", action="store_true",
                   help="Skip vector-based deep interpretability (attribution, MLP neurons, etc.)")
    p.add_argument("--vector_n_samples", type=int, default=50,
                   help="Number of samples for vector attribution (default: 50)")
    # SAE options — tuned for sparsity
    p.add_argument("--sae_expansion", type=int, default=1,
                   help="SAE dictionary size = d_model × expansion (default: 1)")
    p.add_argument("--sae_l1", type=float, default=5e-2,
                   help="SAE L1 sparsity coefficient (default: 5e-2)")
    p.add_argument("--sae_epochs", type=int, default=500,
                   help="SAE training epochs (default: 500)")
    p.add_argument("--language", type=str, default=None,
                   help="Language code (en/zh). Auto-detected from dataset path if omitted.")
    return p.parse_args()


def main():
    args = parse_args()

    model_path = args.model or get_model_path()
    model_p = Path(model_path).expanduser().resolve()
    if model_p.is_dir():
        model_path = str(model_p)
    model_id = get_model_id(model_path)

    dataset_path = args.dataset or get_dataset_path()
    dataset_tag = extract_dataset_tag(dataset_path)
    language = resolve_language(args.language, dataset_path)
    steer_dir = get_steering_dir(model_id, dataset_tag=dataset_tag or None, language=language)
    models_dir = steer_dir / "models"

    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = get_interp_dir(model_id, dataset_tag=dataset_tag or "default", language=language)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "interpretability_results.json"

    # ── Resolve layers ──
    # Auto-detect from saved steering models when --layers is not given
    if args.layers is None:
        # Discover all layers that have at least one trained model
        all_saved = set()
        if models_dir.is_dir():
            for f in models_dir.glob("*_layer*.pt"):
                try:
                    all_saved.add(int(f.stem.split("layer")[-1]))
                except ValueError:
                    pass
        if all_saved:
            from src.config import auto_select_layers
            # Pick n_layers evenly from available layers
            sorted_layers = sorted(all_saved)
            if len(sorted_layers) <= args.n_layers:
                layers = sorted_layers
            else:
                # Pick n evenly spaced from saved set
                step = (len(sorted_layers) - 1) / (args.n_layers - 1)
                indices = [round(i * step) for i in range(args.n_layers)]
                layers = [sorted_layers[i] for i in sorted(set(indices))]
            print(f"  Auto-selected {len(layers)} layers from {len(all_saved)} available: {layers}")
        else:
            # No saved models — fall back to model-based auto-select
            from src.config import get_sweep_layers
            layers = get_sweep_layers(model_path, n_select=args.n_layers)
            print(f"  Auto-selected {len(layers)} layers (model-based): {layers}")
    else:
        layers = args.layers

    print("=" * 60)
    print("Steering Vector Interpretability Analysis (v2)")
    print("=" * 60)
    print(f"  Model    : {model_path}")
    print(f"  Model ID : {model_id}")
    print(f"  Dataset  : {dataset_path}")
    print(f"  Steer dir: {steer_dir}")
    print(f"  Layers   : {layers}")
    print(f"  Methods  : {args.methods}")
    print(f"  Samples  : {args.max_samples}")
    print(f"  Output   : {out_path}")

    available = []
    for method in args.methods:
        for layer in layers:
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

    # Load dataset
    print(f"\nLoading dataset...")
    with open(dataset_path) as f:
        samples = json.load(f)["samples"]
    print(f"  {len(samples)} samples loaded")

    # ── Part A: Lexical + Structural analysis ──
    print(f"\n{'='*60}")
    print("Part A: Lexical + Structural Fingerprint Analysis")
    print("=" * 60)

    from src.steering.interpretability import run_interpretability

    results = run_interpretability(
        model=model, tokenizer=tokenizer,
        models_dir=models_dir, samples=samples,
        layers=layers, methods=args.methods,
        device=args.device, top_k=args.top_k,
        max_samples=args.max_samples, keep_raw=args.keep_raw,
    )

    # ── Part C: Deep structural analysis (CPU-only, no model needed) ──
    if not args.skip_deep:
        print(f"\n{'='*60}")
        print("Part C: Deep Structural Analysis")
        print("=" * 60)

        from src.steering.deep_interp import run_deep_interpretability

        deep_results = run_deep_interpretability(
            samples=samples,
            max_samples=args.max_samples,
            top_k=args.top_k,
            language=language,
        )
        results["deep"] = deep_results

    # ── Part D: Vector-based deep interpretability (needs GPU + model) ──
    if not args.skip_vector:
        print(f"\n{'='*60}")
        print("Part D: Vector-Based Deep Interpretability")
        print("=" * 60)

        from src.steering.vector_interp import run_vector_interpretability

        # Extract texts for attribution analyses
        texts_human = [s["human_continuation"] for s in samples if "human_continuation" in s]
        texts_ai = [s.get("ai_continuation_guided") or s.get("ai_continuation_free", "") for s in samples]
        texts_ai = [t for t in texts_ai if t]

        # Determine best target layer (mid-to-upper range)
        sorted_layers_for_vec = sorted(layers)
        target_layer = sorted_layers_for_vec[len(sorted_layers_for_vec) * 3 // 4] if sorted_layers_for_vec else 16

        vec_results = run_vector_interpretability(
            model=model,
            tokenizer=tokenizer,
            states_dir=steer_dir / "states",
            models_dir=steer_dir / "models",
            texts_human=texts_human,
            texts_ai=texts_ai,
            target_layer=target_layer,
            method="mean_diff",
            max_samples=args.vector_n_samples,
        )
        results["vector_interp"] = vec_results

        # Save vector results separately too
        vec_out = out_path.parent / "vector_interp_results.json"
        with open(vec_out, "w") as f:
            json.dump(vec_results, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
        print(f"  ✓ Vector interp results saved to {vec_out}")

    # Free GPU for SAE
    del model
    torch.cuda.empty_cache()

    # ── Part B: SAE decomposition ──
    if not args.skip_sae:
        print(f"\n{'='*60}")
        print("Part B: SAE Training + Decomposition")
        print(f"  expansion={args.sae_expansion}, L1={args.sae_l1}, epochs={args.sae_epochs}")
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
        json.dump(results, f, indent=2, ensure_ascii=False, cls=_NumpyEncoder)
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

        # Vector interp plots (fig07–fig11)
        if "vector_interp" in results:
            from src.visualization.interp_plots import plot_vector_interpretability
            plot_vector_interpretability(results["vector_interp"], fig_dir)

    print(f"\n{'='*60}")
    print(f"✅ Interpretability analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
