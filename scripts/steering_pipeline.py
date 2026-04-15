#!/usr/bin/env python3
"""
End-to-End Steering Pipeline

Supports two modes:
  1. Single-layer (default):
       python scripts/steering_pipeline.py --layer 16

  2. Multi-layer analysis:
       python scripts/steering_pipeline.py --multi

Model is read from BEYONDSTYLE_MODEL_PATH in .env (overridable via --model).
Outputs are organized under outputs/steering/{model_id}/:
  states/             — cached hidden-state tensors per layer
  models/             — trained steering models per layer
  layer_metrics.json  — per-layer metrics for all methods
  generation/         — generation outputs per method

Alpha convention:
  α < 0  →  steer toward Human
  α > 0  →  steer toward AI
  α = 0  →  baseline (no steering)
"""

import argparse
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (get_model_path, get_model_id, get_steering_dir,
                       get_dataset_dir, get_dataset_path, get_test_prompts_path,
                       extract_dataset_tag)
from src.extraction import PairedExtractor, PairedHiddenStates
from src.steering import MeanDiffSteering, compute_layer_geometry, DENOISE_METHODS
from src.generation import SteeredGenerator, SteeredGenerationConfig

# ── defaults ──────────────────────────────────────────────
SWEEP_LAYERS = [4, 8, 10, 12, 14, 16, 20, 24, 28, 32]
ALPHAS = [-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0]

TEST_PROMPTS = [
    "The investigation into the incident has raised serious questions about",
    "Officials confirmed that the policy changes would take effect immediately, affecting",
    "Witnesses described the scene as chaotic, with emergency responders arriving",
    "The economic impact of the decision is expected to ripple through",
    "Community leaders have called for an independent review of the",
]


def load_test_prompts(path: str) -> list:
    """Load test prompts from a JSON file. Expected format: list of strings."""
    with open(path) as f:
        prompts = json.load(f)
    if isinstance(prompts, list):
        return [str(p) for p in prompts]
    raise ValueError(f"Expected a JSON list of strings in {path}")


# ── CLI ───────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Steering pipeline")
    p.add_argument("--model", default=None,
                   help="Local model path (default: from BEYONDSTYLE_MODEL_PATH in .env)")
    p.add_argument("--dataset", default=None,
                   help="Dataset path (default: from BEYONDSTYLE_DATASET in .env, or auto-detect)")
    p.add_argument("--output_dir", default=None,
                   help="Output dir (default: outputs/steering/{model_id}/{dataset_tag})")
    p.add_argument("--device", default="cuda")

    # mode selection
    p.add_argument("--multi", action="store_true",
                   help="Run multi-layer analysis (default layers: 4-32)")
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="Override layers (e.g. --layers 8 12 16 24)")
    p.add_argument("--layer", type=int, default=16,
                   help="Single-layer mode: which layer")

    # steering
    p.add_argument("--method", choices=["mean_diff"],
                   default="mean_diff",
                   help="Steering method to train")
    p.add_argument("--denoise", type=str, nargs="+",
                   default=["raw"],
                   choices=list(DENOISE_METHODS),
                   help="Denoising methods for v_src (default: raw). "
                        "Use --denoise raw pca nullspace combined for ablation.")
    p.add_argument("--null_k", type=int, default=10,
                   help="Number of content PCs to remove for nullspace methods (default: 10)")

    # generation
    p.add_argument("--alphas", type=float, nargs="+", default=None,
                   help="Override α list (e.g. --alphas -2 -1 0 1 2)")
    p.add_argument("--gen_layers", type=int, nargs="+", default=None,
                   help="Layers for generation demo (default: top-3 by hull centroid distance)")
    p.add_argument("--test_prompts_file", type=str, default=None,
                   help="JSON file with test prompts (default: from .env or data/prompts/default.json)")
    p.add_argument("--skip_generation", action="store_true")
    p.add_argument("--skip_extraction", action="store_true")

    return p.parse_args()


# ── Step 1: extraction ────────────────────────────────────
def extract_states(model_name, dataset_path, layers, out_dir, device,
                   skip_extraction=False):
    """Extract paired hidden states for requested layers. One forward pass per sample."""
    states_dir = out_dir / "states"
    states_dir.mkdir(parents=True, exist_ok=True)

    # Check which layers are already cached
    cached, needed = {}, []
    for ly in layers:
        p = states_dir / f"paired_states_layer{ly:02d}.pt"
        if p.exists() and skip_extraction:
            cached[ly] = PairedHiddenStates.load(p)
        else:
            needed.append(ly)

    if cached:
        print(f"  Loaded cached states for layers: {sorted(cached.keys())}")

    if needed:
        print(f"  Extracting layers {needed} from {dataset_path}")
        with open(dataset_path) as f:
            samples = json.load(f)["samples"]
        print(f"  {len(samples)} samples, {len(needed)} layers")

        extractor = PairedExtractor(model_name, device=device)
        new_states = extractor.extract_multiple_layers(
            samples=samples, layers=needed, pooling="last", show_progress=True,
        )
        for ly, ps in new_states.items():
            ps.save(states_dir / f"paired_states_layer{ly:02d}.pt")
            cached[ly] = ps
        print(f"  Saved states to {states_dir}")

    return cached


# ── Step 2: train steering ────────────────────────────────
def train_all_layers(all_states, methods, out_dir, device,
                     denoise_methods=None, null_k=10, **kwargs):
    """Train steering at every layer for each method × denoise variant."""
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    denoise_methods = denoise_methods or ["raw"]

    all_layer_metrics = {}  # {method_key: {layer: metrics}}
    all_steering = {}       # {method_key: {layer: model}}

    for method in methods:
        for denoise in denoise_methods:
            # method_key distinguishes raw from denoised variants
            method_key = method if denoise == "raw" else f"{method}_{denoise}"
            print(f"\n  ── {method_key} ──")
            layer_metrics = {}
            steering_models = {}

            for ly in sorted(all_states):
                ps = all_states[ly]
                hidden_size = ps.human.shape[-1]

                steer = MeanDiffSteering(hidden_size=hidden_size, layer=ly)
                fit_metrics = steer.fit(
                    ps.human, ps.ai, method="paired",
                    denoise=denoise, null_k=null_k,
                )

                save_path = models_dir / f"{method_key}_layer{ly:02d}.pt"
                steer.save(save_path, metadata={
                    "method": method, "denoise": denoise,
                    "null_k": null_k, "layer": ly,
                })
                steering_models[ly] = steer

                # Compute metrics
                lm = {"layer": ly, "method": method, "denoise": denoise}
                if fit_metrics.get("denoise_info"):
                    lm["denoise_info"] = fit_metrics["denoise_info"]
                try:
                    from sklearn.metrics import roc_auc_score
                    h = ps.human.float(); a = ps.ai.float()

                    v = steer.get_direction().cpu().float()
                    hp = (h @ v).numpy(); ap = (a @ v).numpy()

                    scores = np.concatenate([hp, ap])
                    labels = np.concatenate([np.zeros(len(hp)), np.ones(len(ap))])
                    lm["auc"] = float(roc_auc_score(labels, scores))
                    threshold = (hp.mean() + ap.mean()) / 2
                    preds = (scores > threshold).astype(int)
                    lm["accuracy"] = float((preds == labels).mean())
                    lm["separation"] = float(ap.mean() - hp.mean())
                    lm["human_proj_mean"] = float(hp.mean())
                    lm["ai_proj_mean"] = float(ap.mean())

                    # Overlap
                    ai_min, ai_max = ap.min(), ap.max()
                    h_in_ai = ((hp >= ai_min) & (hp <= ai_max)).mean()
                    human_min, human_max = hp.min(), hp.max()
                    a_in_h = ((ap >= human_min) & (ap <= human_max)).mean()
                    lm["overlap"] = float((h_in_ai + a_in_h) / 2)

                    # Geometry metrics
                    geo = compute_layer_geometry(
                        h.numpy(), a.numpy(), pca_components=2,
                    )
                    lm["hull_centroid_dist"] = geo["hull_centroid_dist"]
                    lm["hull_centroid_dist_norm"] = geo["hull_centroid_dist_norm"]
                    lm["fisher_ratio"] = geo["fisher_ratio"]
                    lm["pca_variance_explained"] = geo["pca_variance_explained"]
                    lm["h_hull_diameter"] = geo["h_hull_diameter"]
                    lm["a_hull_diameter"] = geo["a_hull_diameter"]
                except Exception as e:
                    lm.update({"auc": 0, "accuracy": 0, "separation": 0,
                               "overlap": 1, "hull_centroid_dist": 0,
                               "hull_centroid_dist_norm": 0,
                               "fisher_ratio": 0})

                layer_metrics[ly] = lm
                print(f"    Layer {ly:2d}:  AUC={lm['auc']:.3f}  "
                      f"acc={lm['accuracy']:.1%}  "
                      f"hull_norm={lm.get('hull_centroid_dist_norm',0):.3f}  "
                      f"fisher={lm.get('fisher_ratio',0):.1f}")

            all_layer_metrics[method_key] = layer_metrics
            all_steering[method_key] = steering_models

    # Save unified summary
    summary = {
        "n_samples": int(all_states[sorted(all_states)[0]].n_samples),
        "methods": {m: {str(ly): v for ly, v in lm.items()}
                    for m, lm in all_layer_metrics.items()},
    }
    summary_path = out_dir / "layer_metrics.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved → {summary_path}")

    return all_steering, all_layer_metrics


# ── Step 3: generation demo ──────────────────────────────
def run_generation(model_name, all_steering, all_layer_metrics,
                   alphas, gen_layers, out_dir, device, test_prompts=None):
    """Generate at selected layers × multiple α values."""
    gen_dir = out_dir / "generation"
    gen_dir.mkdir(parents=True, exist_ok=True)

    prompts = test_prompts or TEST_PROMPTS

    # auto-pick top layers by normalized hull centroid distance (primary geometric metric)
    ref_method = "mean_diff" if "mean_diff" in all_layer_metrics else list(all_layer_metrics.keys())[0]
    ref_metrics = all_layer_metrics[ref_method]
    if gen_layers is None:
        ranked = sorted(
            ref_metrics,
            key=lambda ly: ref_metrics[ly].get("hull_centroid_dist_norm", 0),
            reverse=True,
        )
        gen_layers = ranked[:3]

    print(f"  Generation layers: {gen_layers}")
    print(f"  α values: {alphas}")
    print(f"  Methods : {list(all_steering.keys())}")
    print(f"  NOTE: α < 0 → toward Human,  α > 0 → toward AI")

    generator = SteeredGenerator(model_name, device=device)

    for method_name, steering_models in all_steering.items():
        print(f"\n  ── Method: {method_name} ──")
        method_results = []

        for ly in gen_layers:
            steer = steering_models[ly]
            for prompt in prompts:
                outputs = generator.generate_comparison(
                    prompt=prompt,
                    steering=steer,
                    strengths=alphas,
                    config=SteeredGenerationConfig(max_new_tokens=256, temperature=0.7),
                )

                entry = {
                    "layer": ly,
                    "method": method_name,
                    "prompt": prompt,
                    "alpha_direction": "negative=toward_human, positive=toward_ai",
                    "outputs": {},
                }
                for a in sorted(outputs.keys()):
                    if a < 0:
                        label = "toward_human"
                    elif a > 0:
                        label = "toward_ai"
                    else:
                        label = "baseline"
                    entry["outputs"][str(a)] = {
                        "direction": label,
                        "text": outputs[a],
                    }
                method_results.append(entry)

                # Print brief preview
                print(f"\n  [L{ly}] {prompt}")
                for a in [min(alphas), 0.0, max(alphas)]:
                    label = "→Human" if a < 0 else ("baseline" if a == 0 else "→AI")
                    text = outputs.get(a, "")[:90]
                    print(f"    α={a:+.1f} ({label:>8}): {text}...")

        out_path = gen_dir / f"generation_{method_name}.json"
        with open(out_path, "w") as f:
            json.dump(method_results, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved → {out_path}")

    print(f"\n  All generation results in {gen_dir}/")


# ── main ──────────────────────────────────────────────────
def main():
    args = parse_args()

    # Resolve model path: CLI > .env
    model = args.model or get_model_path()
    # Ensure absolute path for local directories
    model_p = Path(model).expanduser().resolve()
    if model_p.is_dir():
        model = str(model_p)
    model_id = get_model_id(model)

    # Resolve dataset path: CLI > env > auto-detect
    dataset = args.dataset or get_dataset_path()

    # Resolve output dir: CLI > auto (outputs/steering/{model_id}/{dataset_tag})
    dataset_tag = extract_dataset_tag(dataset)
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = get_steering_dir(model_id, dataset_tag=dataset_tag or None)
    out_dir.mkdir(parents=True, exist_ok=True)

    alphas = args.alphas or ALPHAS

    if args.multi:
        layers = args.layers or SWEEP_LAYERS
    else:
        layers = [args.layer]

    methods = [args.method]
    denoise_methods = args.denoise

    print("=" * 60)
    print("Steering Pipeline")
    print("=" * 60)
    print(f"  Model   : {model}")
    print(f"  Model ID: {model_id}")
    print(f"  Dataset : {dataset}")
    print(f"  Methods : {methods}")
    print(f"  Denoise : {denoise_methods}")
    print(f"  Layers  : {layers}")
    print(f"  Output  : {out_dir}")
    print(f"  Device  : {args.device}")

    # Resolve test prompts: CLI > env > default file > built-in
    test_prompts = None
    prompts_file = args.test_prompts_file or get_test_prompts_path()
    try:
        test_prompts = load_test_prompts(prompts_file)
        print(f"  Prompts : {len(test_prompts)} from {prompts_file}")
    except FileNotFoundError:
        print(f"  Prompts : {len(TEST_PROMPTS)} built-in (file not found: {prompts_file})")

    # Step 1
    print(f"\n{'='*60}")
    print("Step 1: Extract Paired Hidden States")
    print("=" * 60)
    all_states = extract_states(
        model, dataset, layers, out_dir, args.device,
        skip_extraction=args.skip_extraction,
    )

    # Step 2
    print(f"\n{'='*60}")
    print("Step 2: Train Steering")
    print("=" * 60)
    all_steering, all_layer_metrics = train_all_layers(
        all_states, methods, out_dir, args.device,
        denoise_methods=denoise_methods, null_k=args.null_k,
    )

    # Step 3
    if not args.skip_generation:
        print(f"\n{'='*60}")
        print("Step 3: Generation Demo")
        print("=" * 60)
        run_generation(
            model, all_steering, all_layer_metrics,
            alphas, args.gen_layers, out_dir, args.device,
            test_prompts=test_prompts,
        )

    print(f"\n{'='*60}")
    print(f"✅ Pipeline complete!  [{model_id}]")
    print("=" * 60)
    print(f"\nOutputs in {out_dir}/:")
    print(f"  states/              — hidden-state tensors per layer")
    print(f"  models/              — steering models per layer")
    print(f"  layer_metrics.json   — per-layer metrics for all methods")
    if not args.skip_generation:
        print(f"  generation/          — generation outputs per method")
        print(f"    α < 0 = toward human,  α > 0 = toward AI")


if __name__ == "__main__":
    main()
