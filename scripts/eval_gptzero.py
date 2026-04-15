#!/usr/bin/env python3
"""Evaluate steered text with AI detection tools (GPTZero, etc.).

Generates text at various steering strengths, then submits to AI detection
APIs to measure how human/AI the text appears to external detectors.

Usage:
    # Generate steered samples and evaluate (when API key is available)
    python scripts/eval_gptzero.py --alpha -2.0 -1.0 0.0 1.0 2.0

    # Generate samples only (no API call)
    python scripts/eval_gptzero.py --alpha -1.0 0.0 1.0 --generate-only

    # Evaluate existing samples file
    python scripts/eval_gptzero.py --input outputs/eval/gptzero_samples.json

Environment:
    GPTZERO_API_KEY  — GPTZero API key (optional, skip detection if missing)
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import get_model_path, get_model_id, get_steering_dir, get_dataset_path, extract_dataset_tag


# ── default prompts for generation ───────────────────────
DEFAULT_EVAL_PROMPTS = [
    "The economic implications of artificial intelligence are becoming increasingly clear as",
    "Scientists have long debated the origins of consciousness, with recent research suggesting that",
    "In the small coastal town, life moved at its own unhurried pace. Every morning,",
    "The investigation revealed several critical failures in the safety protocols that had been",
    "Education systems around the world face unprecedented challenges in adapting to the rapid",
    "Walking through the old neighborhood brought back memories she thought she had forgotten.",
    "The latest findings from the climate research team paint a sobering picture of",
    "Local authorities announced sweeping reforms to the public transportation system, promising that",
    "The relationship between diet and mental health has gained significant attention in recent",
    "As the sun set behind the mountains, the village prepared for its annual festival,",
]


def generate_steered_samples(
    model_path: str,
    steering_path: str,
    alphas: List[float],
    prompts: List[str],
    max_new_tokens: int = 200,
    temperature: float = 0.7,
) -> Dict:
    """Generate text at each alpha value for each prompt."""
    from src.generation import SteeredGenerator, SteeredGenerationConfig

    gen = SteeredGenerator(model_path, device="cuda", torch_dtype="float16")
    samples = []

    for i, prompt in enumerate(prompts):
        print(f"  Prompt {i+1}/{len(prompts)}: {prompt[:60]}...")
        entry = {"prompt": prompt, "generations": {}}

        for alpha in alphas:
            config = SteeredGenerationConfig(
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                steering_strength=abs(alpha),
                steering_direction="human" if alpha < 0 else "ai",
            )

            from src.steering import MeanDiffSteering
            steering = MeanDiffSteering.load(steering_path) if alpha != 0 else None

            text = gen.generate(prompt, steering=steering, config=config)
            entry["generations"][str(alpha)] = text
            print(f"    α={alpha:+.1f}: {text[:80]}...")

        samples.append(entry)

    return {
        "model": model_path,
        "steering": steering_path,
        "alphas": alphas,
        "timestamp": datetime.now().isoformat(),
        "samples": samples,
    }


def detect_gptzero(text: str, api_key: str) -> Optional[Dict]:
    """Call GPTZero API to detect AI-generated text.

    Returns:
        Dict with detection scores, or None on failure.
    """
    import requests

    url = "https://api.gptzero.me/v2/predict/text"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
    }
    payload = {"document": text}

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return {
            "completely_generated_prob": data.get("documents", [{}])[0].get("completely_generated_prob"),
            "average_generated_prob": data.get("documents", [{}])[0].get("average_generated_prob"),
            "class_probabilities": data.get("documents", [{}])[0].get("class_probabilities"),
        }
    except Exception as e:
        print(f"    ⚠ GPTZero API error: {e}")
        return None


def run_detection(samples_data: Dict, api_key: str, rate_limit: float = 1.0) -> Dict:
    """Run GPTZero detection on all generated samples."""
    results = []
    for entry in samples_data["samples"]:
        prompt = entry["prompt"]
        detection = {"prompt": prompt, "results": {}}

        for alpha_str, text in entry["generations"].items():
            full_text = prompt + " " + text
            print(f"  Detecting α={alpha_str}: {text[:50]}...")
            score = detect_gptzero(full_text, api_key)
            detection["results"][alpha_str] = score
            time.sleep(rate_limit)

        results.append(detection)

    return {
        **{k: v for k, v in samples_data.items() if k != "samples"},
        "detection_results": results,
        "detector": "gptzero",
        "detection_timestamp": datetime.now().isoformat(),
    }


def print_detection_summary(results: Dict):
    """Print summary of detection results."""
    if "detection_results" not in results:
        print("No detection results found.")
        return

    print("\n" + "=" * 70)
    print("  GPTZero AI Detection Summary")
    print("=" * 70)

    # Aggregate by alpha
    alpha_scores = {}
    for entry in results["detection_results"]:
        for alpha_str, score in entry["results"].items():
            if score is None:
                continue
            prob = score.get("completely_generated_prob")
            if prob is not None:
                alpha_scores.setdefault(alpha_str, []).append(prob)

    print(f"\n  {'Alpha':>8}  {'Mean AI Prob':>12}  {'Std':>8}  {'N':>4}")
    print("  " + "-" * 40)
    for alpha_str in sorted(alpha_scores.keys(), key=float):
        vals = alpha_scores[alpha_str]
        import numpy as np
        mean = np.mean(vals)
        std = np.std(vals)
        print(f"  {float(alpha_str):>+8.1f}  {mean:>12.4f}  {std:>8.4f}  {len(vals):>4}")

    print("\n  ↑ Lower AI Prob = more human-like to GPTZero")


def main():
    parser = argparse.ArgumentParser(description="AI detection evaluation with GPTZero")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--alpha", type=float, nargs="+", default=[-2.0, -1.0, 0.0, 1.0, 2.0])
    parser.add_argument("--denoise", type=str, default="fisher",
                        choices=["raw", "nullspace", "fisher"])
    parser.add_argument("--layer", type=int, default=14)
    parser.add_argument("--prompts", type=str, default=None,
                        help="JSON file with prompts list (default: built-in)")
    parser.add_argument("--max-tokens", type=int, default=200)
    parser.add_argument("--generate-only", action="store_true",
                        help="Only generate samples, skip detection")
    parser.add_argument("--input", type=str, default=None,
                        help="Load existing samples file instead of generating")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    out_dir = ROOT / "outputs" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.input:
        # Load existing samples
        with open(args.input) as f:
            samples_data = json.load(f)
        print(f"Loaded {len(samples_data['samples'])} samples from {args.input}")
    else:
        # Generate new samples
        model_path = args.model or get_model_path()
        model_id = get_model_id(model_path)
        dataset_tag = extract_dataset_tag(get_dataset_path())
        steer_dir = get_steering_dir(model_id, dataset_tag=dataset_tag)

        if args.denoise == "raw":
            steering_path = steer_dir / "models" / f"mean_diff_layer{args.layer:02d}.pt"
        else:
            steering_path = steer_dir / "models" / f"mean_diff_{args.denoise}_layer{args.layer:02d}.pt"

        prompts = DEFAULT_EVAL_PROMPTS
        if args.prompts:
            with open(args.prompts) as f:
                prompts = json.load(f)

        print(f"Generating steered samples...")
        print(f"  Model:    {model_id}")
        print(f"  Steering: {steering_path.name}")
        print(f"  Alphas:   {args.alpha}")

        samples_data = generate_steered_samples(
            model_path, str(steering_path), args.alpha, prompts,
            max_new_tokens=args.max_tokens,
        )

        # Save samples
        samples_path = args.output or str(out_dir / "gptzero_samples.json")
        with open(samples_path, "w") as f:
            json.dump(samples_data, f, indent=2, ensure_ascii=False)
        print(f"✅ Samples saved to {samples_path}")

    if args.generate_only:
        print("Skipping detection (--generate-only).")
        return

    # Run detection
    api_key = os.environ.get("GPTZERO_API_KEY")
    if not api_key:
        print("\n⚠ GPTZERO_API_KEY not set. Skipping AI detection.")
        print("  Set it in .env to enable: GPTZERO_API_KEY=your_key")
        return

    print(f"\nRunning GPTZero detection...")
    results = run_detection(samples_data, api_key)

    out_path = args.output or str(out_dir / "gptzero_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"✅ Detection results saved to {out_path}")

    print_detection_summary(results)


if __name__ == "__main__":
    main()
