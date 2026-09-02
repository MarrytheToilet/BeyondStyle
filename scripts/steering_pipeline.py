#!/usr/bin/env python3
"""
End-to-End Steering Pipeline

Supports two modes:
  1. Single-layer (default):
       python scripts/steering_pipeline.py --layer 16

  2. Multi-layer analysis:
       python scripts/steering_pipeline.py --multi

Model is read from LOCAL_MODEL_PATH in .env (overridable via --model).
Outputs are organized under outputs/steering/{model_id}/:
  states/             — cached hidden-state tensors per layer
  models/             — trained steering models per layer
  insample_metrics.json  — fit-on-all per-layer metrics for all methods
  generation/         — generation outputs per method

Strength convention:
  β = -1  →  source-opposed intervention
  β = +1  →  source-aligned intervention
"""

import argparse
import gc
import hashlib
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import (get_model_path, get_model_id, get_steering_dir,
                       get_dataset_dir, get_dataset_path, get_test_prompts_path,
                       extract_dataset_tag, get_sweep_layers, resolve_language)
from src.extraction import PairedExtractor, PairedHiddenStates
from src.steering import MeanDiffSteering, compute_layer_geometry, DENOISE_METHODS
from src.generation import SteeredGenerator, SteeredGenerationConfig
from src.runtime import model_source_fingerprint
from src.layer_indexing import LAYER_CONVENTION
from src.data.paired_io import load_paired_samples
from src.runtime import stable_seed

# ── defaults ──────────────────────────────────────────────
# SWEEP_LAYERS is now auto-derived via get_sweep_layers(model_path)
ALPHAS = [-1.0, 1.0]

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
                   help="Local model path (default: from LOCAL_MODEL_PATH in .env)")
    p.add_argument("--dataset", default=None,
                   help="Dataset path (default: from PAIRED_DATASET in .env, or auto-detect)")
    p.add_argument("--output_dir", default=None,
                   help="Output dir (default: outputs/steering/{model_id}/{dataset_tag})")
    p.add_argument("--device", default="cuda")

    # mode selection
    p.add_argument("--multi", action="store_true",
                   help="Run multi-layer analysis (default layers: 4-32)")
    p.add_argument("--layers", type=int, nargs="+", default=None,
                   help="Override one-based decoder layers "
                        "(e.g. --layers 8 12 16 24)")
    p.add_argument("--layer", type=int, default=16,
                   help="Single-layer mode: one-based decoder layer")

    # steering
    p.add_argument("--method", choices=["mean_diff"],
                   default="mean_diff",
                   help="Steering method to train")
    p.add_argument("--denoise", type=str, nargs="+",
                   default=["nullspace"],
                   choices=list(DENOISE_METHODS),
                   help="Direction estimators. Nullspace is the primary default; "
                        "raw is available only as an explicit control.")
    p.add_argument("--null_k", type=int, default=10,
                   help="Number of content PCs to remove for nullspace methods (default: 10)")

    # generation
    p.add_argument("--alphas", type=float, nargs="+", default=None,
                   help="Override symmetric β list (default: -1 +1)")
    p.add_argument("--gen_layers", type=int, nargs="+", default=None,
                   help="Layers for generation demo (default: top-3 by hull centroid distance)")
    p.add_argument("--test_prompts_file", type=str, default=None,
                   help="JSON file with test prompts (default: from .env or data/prompts/default.json)")
    p.add_argument("--batch_size", type=int, default=4,
                   help="Generation batch size; halves automatically on CUDA OOM.")
    p.add_argument("--extraction_batch_size", type=int, default=4,
                   help="Pair batch size for hidden-state extraction; each "
                        "forward contains human+AI sequences and halves on OOM.")
    p.add_argument("--seed", type=int, default=8236,
                   help="Base seed for paired sampling across steering strengths.")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument(
        "--steering_positions",
        choices=["decode_only", "last", "all"],
        default="decode_only",
        help="decode_only leaves prompt prefill untouched (formal default).",
    )
    p.add_argument(
        "--strength_unit",
        choices=["projection_std", "absolute"],
        default="projection_std",
        help="Cross-model standardized β or diagnostic absolute vector units.",
    )
    p.add_argument(
        "--torch_dtype",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
    )
    p.add_argument(
        "--attention_backend",
        choices=["auto", "sdpa", "flash_attention_2", "eager"],
        default="auto",
    )
    p.add_argument(
        "--no_tf32",
        action="store_true",
        help="Disable TF32 acceleration for float32 CUDA operations.",
    )
    p.add_argument("--skip_generation", action="store_true")
    p.add_argument("--skip_extraction", action="store_true")
    p.add_argument("--poolings", type=str, nargs="+",
                   default=["last", "full_mean"],
                   choices=["last", "full_mean"],
                   help="Representations to extract: final token and/or "
                        "continuation-token mean.")
    p.add_argument("--language", type=str, default=None,
                   help="Language code (en/zh). Auto-detected from dataset path if omitted.")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Optional sample cap for smoke tests only.")

    return p.parse_args()


# ── Step 1: extraction ────────────────────────────────────
def extract_states(model_name, dataset_path, layers, out_dir, device,
                   skip_extraction=False, pooling: str = "last",
                   max_samples=None):
    """Extract paired hidden states for requested layers. One forward pass per sample."""
    states_dir = out_dir / "states"
    states_dir.mkdir(parents=True, exist_ok=True)

    cached, needed = {}, []
    for ly in layers:
        p = states_dir / f"paired_states_layer{ly:02d}.pt"
        if p.exists() and skip_extraction:
            cached[ly] = PairedHiddenStates.load(p)
        else:
            needed.append(ly)

    if cached:
        print(f"  Loaded cached states for layers: {sorted(cached.keys())} (pooling={pooling})")

    if needed:
        print(f"  Extracting layers {needed} from {dataset_path} (pooling={pooling})")
        samples = load_paired_samples(dataset_path)
        if max_samples is not None:
            if max_samples < 1:
                raise ValueError("--max_samples must be positive")
            samples = samples[:max_samples]
        print(f"  {len(samples)} samples, {len(needed)} layers")

        extractor = PairedExtractor(model_name, device=device)
        new_states = extractor.extract_multiple_layers(
            samples=samples, layers=needed, pooling=pooling, show_progress=True,
        )
        for ly, ps in new_states.items():
            ps.save(states_dir / f"paired_states_layer{ly:02d}.pt")
            cached[ly] = ps
        print(f"  Saved states to {states_dir}")

    return cached


def representation_name(pooling: str) -> str:
    return "last_token" if pooling == "last" else "mean_pool"


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_states_for_poolings(
    model_name,
    dataset_path,
    layers,
    out_dir,
    device,
    poolings,
    *,
    skip_extraction=False,
    max_samples=None,
    extraction_batch_size=4,
    torch_dtype="auto",
    attention_backend="auto",
    allow_tf32=True,
):
    """Extract all requested representations with batched paired forwards."""
    dataset_hash = file_sha256(dataset_path)
    expected_model_identity = model_source_fingerprint(model_name)
    states_by_pooling = {pooling: {} for pooling in poolings}
    missing_by_pooling = {}
    for pooling in poolings:
        states_dir = out_dir / representation_name(pooling) / "states"
        states_dir.mkdir(parents=True, exist_ok=True)
        missing = []
        for layer in layers:
            path = states_dir / f"paired_states_layer{layer:02d}.pt"
            if path.exists() and skip_extraction:
                cached = PairedHiddenStates.load(path)
                metadata = cached.metadata or {}
                cache_matches = (
                    metadata.get("dataset_sha256") == dataset_hash
                    and metadata.get("pooling") == pooling
                    and metadata.get("sample_limit") == max_samples
                    and (
                        metadata.get("model_identity")
                        == expected_model_identity
                        or (
                            metadata.get("model_identity") is None
                            and metadata.get("model") == model_name
                        )
                    )
                )
                if cache_matches:
                    states_by_pooling[pooling][layer] = cached
                else:
                    print(f"  Ignoring stale state cache: {path}")
                    missing.append(layer)
            else:
                missing.append(layer)
        if missing:
            missing_by_pooling[pooling] = missing

    if not missing_by_pooling:
        print("  Loaded all requested representations from cache")
        return states_by_pooling

    samples = load_paired_samples(dataset_path)
    if max_samples is not None:
        if max_samples < 1:
            raise ValueError("--max_samples must be positive")
        samples = samples[:max_samples]
    extract_poolings = list(missing_by_pooling)
    extract_layers = sorted({
        layer
        for missing in missing_by_pooling.values()
        for layer in missing
    })
    print(
        f"  Extracting {len(samples)} pairs at layers {extract_layers}; "
        f"representations={extract_poolings}"
    )
    extractor = PairedExtractor(
        model_name,
        device=device,
        dtype=torch_dtype,
        attention_backend=attention_backend,
        allow_tf32=allow_tf32,
    )
    extracted = extractor.extract_multiple_layers_for_poolings(
        samples=samples,
        layers=extract_layers,
        poolings=extract_poolings,
        show_progress=True,
        batch_size=extraction_batch_size,
    )
    for pooling, per_layer in extracted.items():
        states_dir = out_dir / representation_name(pooling) / "states"
        for layer, paired_states in per_layer.items():
            paired_states.metadata.update({
                "dataset_path": str(dataset_path),
                "dataset_sha256": dataset_hash,
                "sample_limit": max_samples,
            })
            paired_states.save(
                states_dir / f"paired_states_layer{layer:02d}.pt"
            )
            states_by_pooling[pooling][layer] = paired_states
    return states_by_pooling


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
                    "layer_convention": LAYER_CONVENTION,
                    "pooling": (ps.metadata or {}).get("pooling"),
                    "dataset_path": (
                        (ps.metadata or {}).get("dataset_path")
                    ),
                    "dataset_sha256": (
                        (ps.metadata or {}).get("dataset_sha256")
                    ),
                })
                steering_models[ly] = steer

                # Compute metrics
                lm = {"layer": ly, "method": method, "denoise": denoise}
                lm["projection_std"] = fit_metrics["projection_std"]
                lm["standardized_separation"] = fit_metrics[
                    "standardized_separation"
                ]
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
                    lm["pca_variance_explained"] = geo["pca_variance_explained"]
                    lm["h_hull_diameter"] = geo["h_hull_diameter"]
                    lm["a_hull_diameter"] = geo["a_hull_diameter"]
                except Exception as e:
                    lm.update({"auc": 0, "accuracy": 0, "separation": 0,
                               "overlap": 1, "hull_centroid_dist": 0,
                               "hull_centroid_dist_norm": 0,
                               })

                layer_metrics[ly] = lm
                print(f"    Layer {ly:2d}:  AUC={lm['auc']:.3f}  "
                      f"acc={lm['accuracy']:.1%}  "
                      f"hull_norm={lm.get('hull_centroid_dist_norm',0):.3f}")

            all_layer_metrics[method_key] = layer_metrics
            all_steering[method_key] = steering_models

    # Save unified summary
    summary = {
        "n_samples": int(all_states[sorted(all_states)[0]].n_samples),
        "representation": (
            all_states[sorted(all_states)[0]].metadata or {}
        ).get("pooling"),
        "dataset_path": (
            all_states[sorted(all_states)[0]].metadata or {}
        ).get("dataset_path"),
        "dataset_sha256": (
            all_states[sorted(all_states)[0]].metadata or {}
        ).get("dataset_sha256"),
        "methods": {m: {str(ly): v for ly, v in lm.items()}
                    for m, lm in all_layer_metrics.items()},
    }
    summary_path = out_dir / "insample_metrics.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved → {summary_path}")
    md_path = summary_path.with_name("insample_metrics_summary.md")
    write_layer_metrics_md(summary, md_path)
    print(f"  Saved → {md_path}")

    return all_steering, all_layer_metrics


def write_layer_metrics_md(summary: dict, md_path: Path) -> None:
    """Write a paper-ready in-sample separability table, one block per method.

    All metrics are fitted *and* evaluated on the full set of paired states;
    they characterise the linear separability of the source direction
    without a held-out split.  See ``random_holdout_*`` and
    ``generator_holdout_*`` for the train/test versions."""
    methods = list(summary["methods"].keys())
    layers = sorted({int(l) for m in methods for l in summary["methods"][m]})
    lines = [
        "# In-sample per-layer separability metrics",
        "",
        f"- N pairs: **{summary['n_samples']}**",
        f"- Methods: {', '.join(methods)}",
        f"- Layers: {layers}",
        "",
        "Fit + evaluation share the full corpus (no held-out split).  "
        "Columns: ROC-AUC, best-threshold accuracy, projected mean separation, "
        "normalised hull-centroid distance.",
        "",
    ]
    for m in methods:
        lines += [
            f"## `{m}`",
            "",
            "| Layer | AUC | Acc | Sep | Hull-norm |",
            "|---:|---:|---:|---:|---:|",
        ]
        for ly in layers:
            r = summary["methods"][m].get(str(ly), {})
            if not r:
                lines.append(f"| {ly} | — | — | — | — |")
                continue
            lines.append(
                f"| {ly} "
                f"| {r.get('auc', float('nan')):.3f} "
                f"| {r.get('accuracy', float('nan')):.3f} "
                f"| {r.get('separation', float('nan')):.3f} "
                f"| {r.get('hull_centroid_dist_norm', float('nan')):.3f} |"
            )
        lines.append("")
    md_path.write_text("\n".join(lines).rstrip() + "\n")


# ── Step 3: generation demo ──────────────────────────────
def run_generation(
    model_name,
    all_steering,
    all_layer_metrics,
    alphas,
    gen_layers,
    out_dir,
    device,
    test_prompts=None,
    *,
    generator=None,
    batch_size=4,
    base_seed=8236,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.95,
    steering_positions="decode_only",
    strength_unit="projection_std",
    torch_dtype="auto",
    attention_backend="auto",
    allow_tf32=True,
):
    """Generate batched, paired-RNG comparisons with CUDA OOM backoff."""
    gen_dir = out_dir / "generation"
    gen_dir.mkdir(parents=True, exist_ok=True)

    prompts = test_prompts or TEST_PROMPTS
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if 0.0 not in alphas:
        raise ValueError("alphas must include 0.0 for the paired baseline")

    # Nullspace is the primary selection signal; raw is only a control.
    ref_method = (
        "mean_diff_nullspace"
        if "mean_diff_nullspace" in all_layer_metrics
        else list(all_layer_metrics.keys())[0]
    )
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
    print(f"  Batch   : {batch_size}")
    print(f"  Seed    : {base_seed} (paired across strengths)")
    print(f"  Hook    : {steering_positions}")
    print(f"  Unit    : {strength_unit}")
    print(f"  NOTE: β < 0 → toward Human,  β > 0 → toward AI")

    generator = generator or SteeredGenerator(
        model_name,
        device=device,
        torch_dtype=torch_dtype,
        attention_backend=attention_backend,
        allow_tf32=allow_tf32,
    )

    for method_name, steering_models in all_steering.items():
        print(f"\n  ── Method: {method_name} ──")
        method_results = []
        method_batch_records = []

        for ly in gen_layers:
            steer = steering_models[ly]
            offset = 0
            active_batch_size = min(batch_size, len(prompts))
            while offset < len(prompts):
                prompt_batch = prompts[offset: offset + active_batch_size]
                batch_seed = stable_seed(
                    base_seed,
                    model_name,
                    out_dir.name,
                    method_name,
                    ly,
                    offset,
                )
                config = SteeredGenerationConfig(
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    apply_to_positions=steering_positions,
                    strength_unit=strength_unit,
                    seed=batch_seed,
                    paired_sampling=True,
                    use_cache=True,
                )
                try:
                    outputs = generator.generate_batch_comparison(
                    prompts=prompt_batch,
                    steering=steer,
                    strengths=alphas,
                    config=config,
                )
                except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
                    is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or (
                        "out of memory" in str(error).lower()
                    )
                    if not is_oom or active_batch_size == 1:
                        raise
                    active_batch_size = max(1, active_batch_size // 2)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    print(
                        f"  CUDA OOM at L{ly}; retrying with "
                        f"batch_size={active_batch_size}",
                        flush=True,
                    )
                    continue

                comparison_stats = {
                    str(alpha): stats
                    for alpha, stats in generator.last_comparison_stats.items()
                }
                batch_id = f"L{ly:02d}_O{offset:05d}"
                method_batch_records.append({
                    "batch_id": batch_id,
                    "layer": ly,
                    "offset": offset,
                    "size": len(prompt_batch),
                    "seed": batch_seed,
                    "stats_by_strength": comparison_stats,
                })
                for prompt_index, prompt in enumerate(prompt_batch):
                    entry = {
                        "layer": ly,
                        "method": method_name,
                        "prompt": prompt,
                        "alpha_direction": (
                            "negative=toward_human, positive=toward_ai"
                        ),
                        "generation_metadata": {
                            "batch_id": batch_id,
                            "batch_seed": batch_seed,
                            "batch_offset": offset,
                            "position_mode": steering_positions,
                            "prompt_prefill_untouched": (
                                steering_positions == "decode_only"
                            ),
                            "strength_unit": strength_unit,
                            "paired_sampling": True,
                        },
                        "outputs": {},
                    }
                    for alpha in sorted(outputs):
                        label = (
                            "toward_human"
                            if alpha < 0
                            else "toward_ai"
                            if alpha > 0
                            else "baseline"
                        )
                        entry["outputs"][str(alpha)] = {
                            "direction": label,
                            "text": outputs[alpha][prompt_index],
                        }
                    method_results.append(entry)

                preview_prompt = prompt_batch[0]
                print(f"\n  [L{ly}] {preview_prompt}")
                for alpha in [min(alphas), 0.0, max(alphas)]:
                    label = (
                        "→Human"
                        if alpha < 0
                        else "baseline"
                        if alpha == 0
                        else "→AI"
                    )
                    text = outputs.get(alpha, [""])[0][:90]
                    print(
                        f"    β={alpha:+.1f} ({label:>8}): {text}..."
                    )
                offset += len(prompt_batch)

        out_path = gen_dir / f"generation_{method_name}.json"
        with open(out_path, "w") as f:
            json.dump(method_results, f, indent=2, ensure_ascii=False)
        print(f"\n  Saved → {out_path}")
        manifest_path = gen_dir / f"generation_{method_name}_manifest.json"
        manifest = {
            "schema_version": 1,
            "model": model_name,
            "method": method_name,
            "layers": gen_layers,
            "strengths": sorted(float(alpha) for alpha in alphas),
            "strength_unit": strength_unit,
            "position_mode": steering_positions,
            "prompt_prefill_untouched": steering_positions == "decode_only",
            "paired_sampling": True,
            "base_seed": base_seed,
            "generation": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "requested_batch_size": batch_size,
            },
            "model_load": generator.model_load_info,
            "batches": method_batch_records,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"  Saved → {manifest_path}")

    print(f"\n  All generation results in {gen_dir}/")
    return generator


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

    # Resolve dataset path: CLI > env > canonical language path.
    dataset = args.dataset or get_dataset_path(args.language or "en")

    # Resolve output dir: CLI > auto (outputs/{lang}/steering/{model_id}/{dataset_tag})
    dataset_tag = extract_dataset_tag(dataset)
    language = resolve_language(args.language, dataset)
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = get_steering_dir(model_id, dataset_tag=dataset_tag or None, language=language)
    out_dir.mkdir(parents=True, exist_ok=True)

    alphas = args.alphas or ALPHAS

    if args.multi:
        layers = args.layers or get_sweep_layers(model)
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
    print(f"  Pooling : {args.poolings}")
    print(f"  Samples : {args.max_samples or 'all'}")
    print(f"  Device  : {args.device}")
    print(f"  Extract batch: {args.extraction_batch_size}")
    if not args.skip_generation:
        print(f"  Gen batch: {args.batch_size}")
        print(f"  Gen seed : {args.seed}")
        print(f"  Hook mode: {args.steering_positions}")
        print(f"  Strength : {args.strength_unit}")
        print(f"  Dtype    : {args.torch_dtype}")
        print(f"  Attention: {args.attention_backend}")

    # Resolve test prompts: CLI > env > default file > built-in
    test_prompts = None
    prompts_file = args.test_prompts_file or get_test_prompts_path(language)
    try:
        test_prompts = load_test_prompts(prompts_file)
        print(f"  Prompts : {len(test_prompts)} from {prompts_file}")
    except FileNotFoundError:
        print(f"  Prompts : {len(TEST_PROMPTS)} built-in (file not found: {prompts_file})")

    all_states_by_pooling = extract_states_for_poolings(
        model,
        dataset,
        layers,
        out_dir,
        args.device,
        args.poolings,
        skip_extraction=args.skip_extraction,
        max_samples=args.max_samples,
        extraction_batch_size=args.extraction_batch_size,
        torch_dtype=args.torch_dtype,
        attention_backend=args.attention_backend,
        allow_tf32=not args.no_tf32,
    )

    shared_generator = None
    if not args.skip_generation:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        shared_generator = SteeredGenerator(
            model,
            device=args.device,
            torch_dtype=args.torch_dtype,
            attention_backend=args.attention_backend,
            allow_tf32=not args.no_tf32,
        )

    for pooling in args.poolings:
        representation = representation_name(pooling)
        representation_dir = out_dir / representation
        representation_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n{'='*60}")
        print(f"Representation: {representation}")
        print("=" * 60)
        all_states = all_states_by_pooling[pooling]
        all_steering, all_layer_metrics = train_all_layers(
            all_states,
            methods,
            representation_dir,
            args.device,
            denoise_methods=denoise_methods,
            null_k=args.null_k,
        )
        if not args.skip_generation:
            shared_generator = run_generation(
                model,
                all_steering,
                all_layer_metrics,
                alphas,
                args.gen_layers,
                representation_dir,
                args.device,
                test_prompts=test_prompts,
                generator=shared_generator,
                batch_size=args.batch_size,
                base_seed=args.seed,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                steering_positions=args.steering_positions,
                strength_unit=args.strength_unit,
                torch_dtype=args.torch_dtype,
                attention_backend=args.attention_backend,
                allow_tf32=not args.no_tf32,
            )

    print(f"\n{'='*60}")
    print(f"✅ Pipeline complete!  [{model_id}]")
    print("=" * 60)
    print(f"\nOutputs in {out_dir}/{{last_token,mean_pool}}/:")
    print("  states/              — hidden-state tensors per layer")
    print("  models/              — primary nullspace direction (raw only if requested)")
    print("  insample_metrics.json — fit-on-all per-layer metrics")
    if not args.skip_generation:
        print(f"  generation/          — generation outputs per method")
        print(f"    β < 0 = toward human,  β > 0 = toward AI")


if __name__ == "__main__":
    main()
