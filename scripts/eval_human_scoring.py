#!/usr/bin/env python3
"""Human AI-feel scoring evaluation pipeline.

Generates steered text samples and creates an evaluation form for human raters
to score how "AI-like" each text sounds. Outputs scoring sheets and aggregates
results after scoring.

Workflow:
    1. Generate: Create steered samples → eval_human_samples.json
    2. Export:   Convert to scoring sheet → eval_human_sheet.json (shuffled, blinded)
    3. Score:    Humans fill in scores (1-5 scale)
    4. Analyze:  Aggregate scores → eval_human_results.json

Usage:
    # Step 1+2: Generate samples and create scoring sheet
    python scripts/eval_human_scoring.py generate --alpha -2.0 -1.0 0.0 1.0 2.0

    # Step 3: Print the scoring sheet for manual evaluation
    python scripts/eval_human_scoring.py show

    # Step 3 (alternative): Interactive scoring in terminal
    python scripts/eval_human_scoring.py score

    # Step 4: Analyze completed scores
    python scripts/eval_human_scoring.py analyze
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import get_model_path, get_model_id, get_steering_dir, get_dataset_path, extract_dataset_tag, resolve_language

EVAL_DIR = ROOT / "outputs" / "eval"
SAMPLES_FILE = EVAL_DIR / "human_scoring_samples.json"
SHEET_FILE = EVAL_DIR / "human_scoring_sheet.json"
SCORES_FILE = EVAL_DIR / "human_scoring_results.json"

# ── Scoring rubric ───────────────────────────────────────
RUBRIC = """
╔══════════════════════════════════════════════════════════╗
║              AI-Feel Scoring Rubric (1-5)               ║
╠══════════════════════════════════════════════════════════╣
║  1 = Definitely Human                                   ║
║      Natural flow, personal voice, unexpected details,  ║
║      informal transitions, subjective opinions           ║
║                                                         ║
║  2 = Likely Human                                       ║
║      Mostly natural but occasionally generic phrasing   ║
║                                                         ║
║  3 = Uncertain / Could Be Either                        ║
║      Mix of natural and formulaic elements              ║
║                                                         ║
║  4 = Likely AI                                          ║
║      Structured, comprehensive, hedging language,       ║
║      balanced viewpoints, smooth but predictable        ║
║                                                         ║
║  5 = Definitely AI                                      ║
║      Highly structured, list-heavy, disclaimers,        ║
║      "it's important to note", overly balanced          ║
╚══════════════════════════════════════════════════════════╝
"""

EVAL_PROMPTS = [
    "The economic implications of artificial intelligence are becoming increasingly clear as",
    "Scientists have long debated the origins of consciousness, with recent research suggesting that",
    "In the small coastal town, life moved at its own unhurried pace. Every morning,",
    "The investigation revealed several critical failures in the safety protocols that had been",
    "Education systems around the world face unprecedented challenges in adapting to the rapid",
    "Walking through the old neighborhood brought back memories she thought she had forgotten.",
    "The latest findings from the climate research team paint a sobering picture of",
    "Local authorities announced sweeping reforms to the public transportation system, promising that",
]


def cmd_generate(args):
    """Generate steered samples and create blinded scoring sheet."""
    from src.generation import SteeredGenerator, SteeredGenerationConfig
    from src.steering import MeanDiffSteering

    model_path = args.model or get_model_path()
    model_id = get_model_id(model_path)
    dataset_tag = extract_dataset_tag(get_dataset_path())
    steer_dir = get_steering_dir(model_id, dataset_tag=dataset_tag)

    if args.denoise == "raw":
        steering_path = steer_dir / "models" / f"mean_diff_layer{args.layer:02d}.pt"
    else:
        steering_path = steer_dir / "models" / f"mean_diff_{args.denoise}_layer{args.layer:02d}.pt"

    prompts = EVAL_PROMPTS
    if args.prompts:
        with open(args.prompts) as f:
            prompts = json.load(f)

    alphas = sorted(set(args.alpha))

    print(f"Generating steered samples for human evaluation")
    print(f"  Model:    {model_id}")
    print(f"  Steering: {steering_path.name}")
    print(f"  Alphas:   {alphas}")
    print(f"  Prompts:  {len(prompts)}")

    gen = SteeredGenerator(model_path, device="cuda", torch_dtype="float16")
    steering = MeanDiffSteering.load(str(steering_path))

    samples = []
    for i, prompt in enumerate(prompts):
        print(f"\n  Prompt {i+1}/{len(prompts)}: {prompt[:60]}...")
        entry = {"prompt_id": i, "prompt": prompt, "generations": {}}

        for alpha in alphas:
            config = SteeredGenerationConfig(
                max_new_tokens=args.max_tokens,
                temperature=0.7,
                steering_strength=abs(alpha),
                steering_direction="human" if alpha < 0 else "ai",
            )
            text = gen.generate(
                prompt,
                steering=steering if alpha != 0 else None,
                config=config,
            )
            entry["generations"][str(alpha)] = text
            label = "baseline" if alpha == 0 else f"α={alpha:+.1f}"
            print(f"    [{label}]: {text[:80]}...")

        samples.append(entry)

    # Save raw samples (with labels — for analysis)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    raw_data = {
        "model": model_id,
        "steering": str(steering_path),
        "denoise": args.denoise,
        "layer": args.layer,
        "alphas": alphas,
        "timestamp": datetime.now().isoformat(),
        "samples": samples,
    }
    with open(SAMPLES_FILE, "w") as f:
        json.dump(raw_data, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Raw samples saved to {SAMPLES_FILE}")

    # Create blinded scoring sheet (shuffled, no alpha labels)
    sheet_items = []
    for entry in samples:
        for alpha_str, text in entry["generations"].items():
            sheet_items.append({
                "item_id": f"p{entry['prompt_id']}_a{alpha_str}",
                "prompt": entry["prompt"],
                "text": text,
                "score": None,
                "notes": "",
                # Hidden metadata (for analysis only)
                "_alpha": float(alpha_str),
                "_prompt_id": entry["prompt_id"],
            })

    random.seed(42)
    random.shuffle(sheet_items)

    # Assign sequential display IDs
    for idx, item in enumerate(sheet_items):
        item["display_id"] = idx + 1

    sheet = {
        "rubric": RUBRIC.strip(),
        "instructions": (
            "Rate each text continuation on the AI-feel scale (1-5). "
            "Read the prompt, then the continuation. Score how AI-generated the text feels. "
            "Do NOT look at _alpha or _prompt_id fields — they are for analysis only."
        ),
        "total_items": len(sheet_items),
        "items": sheet_items,
    }
    with open(SHEET_FILE, "w") as f:
        json.dump(sheet, f, indent=2, ensure_ascii=False)
    print(f"✅ Scoring sheet saved to {SHEET_FILE}")
    print(f"   {len(sheet_items)} items to score (shuffled, blinded)")


def cmd_show(args):
    """Display scoring sheet for manual evaluation."""
    if not SHEET_FILE.exists():
        print(f"ERROR: No scoring sheet found. Run 'generate' first.")
        sys.exit(1)

    with open(SHEET_FILE) as f:
        sheet = json.load(f)

    print(sheet["rubric"])
    print(f"\nInstructions: {sheet['instructions']}")
    print(f"Total items: {sheet['total_items']}")
    print("=" * 70)

    for item in sheet["items"]:
        scored = "✓" if item["score"] is not None else " "
        print(f"\n[{scored}] Item #{item['display_id']}")
        print(f"  Prompt: {item['prompt'][:80]}...")
        print(f"  Text:   {item['text'][:200]}")
        if item['text'] and len(item['text']) > 200:
            print(f"          {item['text'][200:400]}")
        if item["score"] is not None:
            print(f"  Score:  {item['score']}/5  Notes: {item.get('notes', '')}")
        print("-" * 70)


def cmd_score(args):
    """Interactive terminal scoring session."""
    if not SHEET_FILE.exists():
        print(f"ERROR: No scoring sheet found. Run 'generate' first.")
        sys.exit(1)

    with open(SHEET_FILE) as f:
        sheet = json.load(f)

    print(RUBRIC)
    print("Type a score (1-5) for each item, or 's' to skip, 'q' to quit.\n")

    items = sheet["items"]
    # Find first unscored item
    start_idx = 0
    for i, item in enumerate(items):
        if item["score"] is None:
            start_idx = i
            break

    scored_count = sum(1 for it in items if it["score"] is not None)
    print(f"Progress: {scored_count}/{len(items)} scored. Starting from #{items[start_idx]['display_id']}.\n")

    for item in items[start_idx:]:
        print(f"\n{'='*70}")
        print(f"Item #{item['display_id']} / {len(items)}")
        print(f"{'='*70}")
        print(f"PROMPT: {item['prompt']}")
        print(f"\nTEXT:")
        print(f"  {item['text']}")
        print()

        while True:
            try:
                inp = input("Score (1-5), 's' to skip, 'q' to quit: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                inp = "q"

            if inp == "q":
                # Save progress
                with open(SHEET_FILE, "w") as f:
                    json.dump(sheet, f, indent=2, ensure_ascii=False)
                scored_count = sum(1 for it in items if it["score"] is not None)
                print(f"\nSaved progress: {scored_count}/{len(items)} scored.")
                return
            elif inp == "s":
                break
            elif inp in ("1", "2", "3", "4", "5"):
                item["score"] = int(inp)
                notes = input("Notes (optional, press Enter to skip): ").strip()
                if notes:
                    item["notes"] = notes
                break
            else:
                print("Invalid input. Enter 1-5, 's', or 'q'.")

    # Save final
    with open(SHEET_FILE, "w") as f:
        json.dump(sheet, f, indent=2, ensure_ascii=False)
    scored_count = sum(1 for it in items if it["score"] is not None)
    print(f"\n✅ Scoring complete! {scored_count}/{len(items)} scored.")


def cmd_analyze(args):
    """Analyze completed scoring sheet and produce results."""
    import numpy as np

    if not SHEET_FILE.exists():
        print(f"ERROR: No scoring sheet found. Run 'generate' first.")
        sys.exit(1)

    with open(SHEET_FILE) as f:
        sheet = json.load(f)

    items = sheet["items"]
    scored = [it for it in items if it["score"] is not None]
    total = len(items)

    if not scored:
        print("No scores found. Run 'score' first.")
        sys.exit(1)

    print(f"Analyzing {len(scored)}/{total} scored items.")

    # Group by alpha
    alpha_scores = {}
    for item in scored:
        alpha = item["_alpha"]
        alpha_scores.setdefault(alpha, []).append(item["score"])

    print("\n" + "=" * 60)
    print("  Human AI-Feel Scoring Results")
    print("=" * 60)
    print(RUBRIC)
    print(f"\n  {'Alpha':>8}  {'Mean':>6}  {'Std':>6}  {'Med':>4}  {'N':>4}  {'Range':>8}")
    print("  " + "-" * 48)

    results_table = {}
    for alpha in sorted(alpha_scores.keys()):
        scores = alpha_scores[alpha]
        arr = np.array(scores, dtype=float)
        mean = arr.mean()
        std = arr.std()
        med = np.median(arr)
        lo, hi = arr.min(), arr.max()
        n = len(scores)
        print(f"  {alpha:>+8.1f}  {mean:>6.2f}  {std:>6.2f}  {med:>4.1f}  {n:>4}  [{lo:.0f}-{hi:.0f}]")
        results_table[str(alpha)] = {
            "mean": round(float(mean), 3),
            "std": round(float(std), 3),
            "median": float(med),
            "n": n,
            "scores": scores,
        }

    # Correlation: alpha vs score
    all_alphas_list = [it["_alpha"] for it in scored]
    all_scores_list = [it["score"] for it in scored]
    from scipy import stats
    r, p = stats.pearsonr(all_alphas_list, all_scores_list)
    tau, p_tau = stats.kendalltau(all_alphas_list, all_scores_list)

    print(f"\n  Correlation:")
    print(f"    Pearson r  = {r:.3f}  (p = {p:.2e})")
    print(f"    Kendall τ  = {tau:.3f}  (p = {p_tau:.2e})")

    baseline_scores = alpha_scores.get(0.0, [])
    if baseline_scores:
        baseline_mean = np.mean(baseline_scores)
        print(f"\n  Baseline (α=0.0) mean score: {baseline_mean:.2f}")
        for alpha in sorted(alpha_scores.keys()):
            if alpha == 0.0:
                continue
            delta = np.mean(alpha_scores[alpha]) - baseline_mean
            direction = "more AI" if delta > 0 else "more human"
            print(f"    α={alpha:+.1f}: Δ={delta:+.2f} ({direction})")

    # Save results
    final = {
        "timestamp": datetime.now().isoformat(),
        "n_scored": len(scored),
        "n_total": total,
        "alpha_results": results_table,
        "correlation": {
            "pearson_r": round(r, 4),
            "pearson_p": float(p),
            "kendall_tau": round(tau, 4),
            "kendall_p": float(p_tau),
        },
    }

    with open(SCORES_FILE, "w") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Results saved to {SCORES_FILE}")


def main():
    parser = argparse.ArgumentParser(description="Human AI-feel scoring evaluation")
    sub = parser.add_subparsers(dest="command")

    # generate
    gen_p = sub.add_parser("generate", help="Generate steered samples and scoring sheet")
    gen_p.add_argument("--model", type=str, default=None)
    gen_p.add_argument("--alpha", type=float, nargs="+", default=[-2.0, -1.0, 0.0, 1.0, 2.0])
    gen_p.add_argument("--denoise", type=str, default="fisher",
                       choices=["raw", "nullspace", "fisher"])
    gen_p.add_argument("--layer", type=int, default=14)
    gen_p.add_argument("--prompts", type=str, default=None)
    gen_p.add_argument("--max-tokens", type=int, default=200)

    # show
    sub.add_parser("show", help="Display scoring sheet")

    # score
    sub.add_parser("score", help="Interactive terminal scoring")

    # analyze
    sub.add_parser("analyze", help="Analyze completed scores")

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "show":
        cmd_show(args)
    elif args.command == "score":
        cmd_score(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
