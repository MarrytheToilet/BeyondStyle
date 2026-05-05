#!/usr/bin/env python3
"""Re-extract token-level projection for ONE (layer, method) and merge it
into an existing ``interpretability_results.json``.  Re-renders fig01 with
a larger top-k.  Other cached entries are preserved.

Usage:
    python scripts/refresh_token_projection.py \
        --interp-dir outputs/en/06_interpretability/Llama-3.1-8B-Instruct/mixed_500 \
        --layer 15 --method mean_diff --top_k 200 --display 80
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import (
    get_model_path,
    get_model_id,
    get_dataset_path,
    extract_dataset_tag,
    get_steering_dir,
    resolve_language,
)
from src.steering.interpretability import token_level_projection


def _NumpyEncoder(o):  # pragma: no cover
    import numpy as np
    if isinstance(o, np.generic):
        return o.item()
    raise TypeError(type(o))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interp-dir", type=Path, required=True,
                    help="Directory holding interpretability_results.json")
    ap.add_argument("--language", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--method", default="mean_diff",
                    choices=["mean_diff", "mean_diff_nullspace", "mean_diff_fisher"])
    ap.add_argument("--top_k", type=int, default=200,
                    help="How many entries to retain in the cached JSON")
    ap.add_argument("--display", type=int, default=100,
                    help="How many rows to render per panel in fig01.md")
    ap.add_argument("--max_samples", type=int, default=100)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    cache_path = args.interp_dir / "interpretability_results.json"
    if not cache_path.exists():
        ap.error(f"{cache_path} not found")
    cache = json.loads(cache_path.read_text())

    model_path = args.model or get_model_path()
    model_id = get_model_id(model_path)
    dataset_path = args.dataset or get_dataset_path()
    dataset_tag = extract_dataset_tag(dataset_path)
    language = resolve_language(args.language, dataset_path)
    steer_dir = get_steering_dir(model_id, dataset_tag=dataset_tag or None,
                                 language=language)

    weight_path = steer_dir / "models" / f"{args.method}_layer{args.layer:02d}.pt"
    if not weight_path.exists():
        ap.error(f"Steering vector not found: {weight_path}")

    print(f"  Loading samples from {dataset_path}")
    with open(dataset_path) as f:
        samples = json.load(f)["samples"]
    samples = samples[: args.max_samples]
    print(f"  {len(samples)} samples")

    print(f"  Loading {weight_path.name}")
    state = torch.load(weight_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        v_src = state["state_dict"]["v_src"]
    elif isinstance(state, dict) and "v_src" in state:
        v_src = state["v_src"]
    else:
        v_src = state

    print(f"  Loading model {model_path}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_path)
    mdl = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map=args.device,
    )
    mdl.eval()

    print(f"  Re-extracting token projection at L{args.layer} ({args.method})")
    with torch.no_grad():
        tp = token_level_projection(
            v_src=v_src, model=mdl, tokenizer=tok,
            samples=samples, layer=args.layer,
            method=args.method, max_samples=args.max_samples,
        )
    block = tp.to_dict(top_k=args.top_k)
    print(f"    AI-exclusive   : {len(block['ai_exclusive_tokens'])}"
          f"  Human-exclusive: {len(block['human_exclusive_tokens'])}"
          f"  Contrastive    : {len(block['contrastive_tokens'])}")

    key = f"{args.method}_L{args.layer}"
    cache.setdefault("token_projection", {})[key] = block
    cache_path.write_text(
        json.dumps(cache, indent=2, ensure_ascii=False, default=_NumpyEncoder)
    )
    print(f"  ✓ Updated {cache_path}")

    print("  Rendering fig01")
    from src.visualization.interp_plots import _plot_fig01, _setup_fonts
    _setup_fonts(language)
    fig_dir = args.interp_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _plot_fig01(cache, fig_dir, layer=args.layer, method=args.method,
                n_tok=args.display, language=language)
    print("  ✓ fig01 refreshed")


if __name__ == "__main__":
    main()
