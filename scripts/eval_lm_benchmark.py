#!/usr/bin/env python3
"""Evaluate steered LLM on standard benchmarks via lm-eval.

Supports three modes:
  1. Single run:  test one (layer, alpha, denoise) combination
  2. Alpha sweep: compare multiple alphas at a fixed layer
  3. Grid sweep:  test all (layer × alpha × denoise) combinations

The model is loaded ONCE; steering hooks are swapped between runs.
Each run is saved as a separate JSON file with structured naming.
Already-completed runs are automatically skipped (resume-friendly).

Output structure:
  outputs/eval/{model_id}/
  ├── baseline.json                       # α=0.0 (always run first, shared)
  ├── fisher_L14_a+1.0.json              # individual steered runs
  ├── fisher_L14_a-1.0.json
  ├── raw_L16_a+2.0.json
  ├── ...
  ├── raw/                                # full lm-eval output per run
  │   ├── baseline_raw.json
  │   ├── fisher_L14_a+1.0_raw.json
  │   └── ...
  └── summary_latest.json                # aggregated comparison

Usage:
    # Quick sanity check
    python scripts/eval_lm_benchmark.py --alpha 0.0 1.0 --tasks wikitext --limit 10

    # Full single-layer sweep
    python scripts/eval_lm_benchmark.py --alpha 0.0 -1.0 1.0 2.0

    # Multi-layer grid sweep
    python scripts/eval_lm_benchmark.py --mode grid \\
        --layers 10 14 16 20 --alpha 0.0 -1.0 1.0 --denoise fisher raw

    # Resume interrupted grid (skips existing results)
    python scripts/eval_lm_benchmark.py --mode grid --resume

    # Test specific task group
    python scripts/eval_lm_benchmark.py --alpha 0.0 1.0 --task-group reasoning
"""

import argparse
import gc
import json
import sys
import time
import types
from datetime import datetime
from itertools import groupby
from pathlib import Path

# ── lm-eval compatibility patch (transformers >=5.x) ─────
sys.modules.setdefault(
    "lm_eval.models.hf_vlms",
    types.ModuleType("lm_eval.models.hf_vlms"),
)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import lm_eval
from src.config import (
    get_model_path, get_model_id, get_steering_dir,
    get_dataset_path, extract_dataset_tag,
)


# ── Benchmark task registry ─────────────────────────────────
TASK_GROUPS = {
    "reasoning": ["mmlu", "arc_easy", "arc_challenge", "openbookqa"],
    "commonsense": ["hellaswag", "winogrande", "piqa", "boolq"],
    "reading": ["race"],
    "perplexity": ["wikitext"],
}

ALL_TASKS = [
    "mmlu",             # Massive Multitask Language Understanding
    "hellaswag",        # Commonsense NLI
    "arc_easy",         # Science QA (easy)
    "arc_challenge",    # Science QA (hard)
    "openbookqa",       # Open-domain science QA
    "winogrande",       # Coreference resolution
    "piqa",             # Physical intuition QA
    "boolq",            # Boolean QA (yes/no)
    "race",             # Reading comprehension (middle+high school)
    "wikitext",         # Perplexity (WikiText-2)
]

TASK_ALIASES = {
    "mmlu": "mmlu",
    "hellaswag": "hellaswag",
    "arc_easy": "arc_easy",
    "arc_challenge": "arc_challenge",
    "openbookqa": "openbookqa",
    "winogrande": "winogrande",
    "piqa": "piqa",
    "boolq": "boolq",
    "race": "race",
    "wikitext": "wikitext",
    "truthfulqa": "truthfulqa_mc2",
    "gsm8k": "gsm8k",
}

# Primary metric per task (for summary tables)
TASK_PRIMARY_METRIC = {
    "mmlu": "acc",
    "hellaswag": "acc_norm",
    "arc_easy": "acc",
    "arc_challenge": "acc_norm",
    "openbookqa": "acc_norm",
    "winogrande": "acc",
    "piqa": "acc_norm",
    "boolq": "acc",
    "race": "acc",
    "wikitext": "word_perplexity",
}

DEFAULT_LAYERS = [10, 14, 16, 20]
DEFAULT_ALPHAS = [0.0, -0.5, -1.0, -2.0, 0.5, 1.0, 2.0]


# ── Utilities ────────────────────────────────────────────────

def resolve_steering_path(steer_dir: Path, denoise: str, layer: int) -> Path:
    if denoise == "raw":
        return steer_dir / "models" / f"mean_diff_layer{layer:02d}.pt"
    return steer_dir / "models" / f"mean_diff_{denoise}_layer{layer:02d}.pt"


def run_tag(denoise: str, layer: int, alpha: float) -> str:
    """Structured tag for a single run.

    baseline | fisher_L14_a+1.0 | raw_L16_a-2.0
    """
    if alpha == 0.0:
        return "baseline"
    sign = "+" if alpha >= 0 else ""
    return f"{denoise}_L{layer:02d}_a{sign}{alpha}"


def get_eval_dir(model_id: str) -> Path:
    return ROOT / "outputs" / "eval" / model_id


def extract_scores(results: dict) -> dict:
    """Extract key metrics from lm-eval results dict."""
    scores = {}
    for task_name, task_data in results.get("results", {}).items():
        entry = {}
        for key, val in task_data.items():
            if key.endswith(",none") or key in (
                "acc", "acc_norm", "word_perplexity",
                "byte_perplexity", "bits_per_byte",
            ):
                clean = key.replace(",none", "") if key.endswith(",none") else key
                if isinstance(val, (int, float)):
                    entry[clean] = round(val, 6)
        if entry:
            scores[task_name] = entry
    return scores


def save_run(eval_dir: Path, tag: str, meta: dict, scores: dict,
             raw_results: dict):
    """Save structured summary + full raw lm-eval output."""
    eval_dir.mkdir(parents=True, exist_ok=True)

    record = {"tag": tag, **meta, "scores": scores,
              "timestamp": datetime.now().isoformat()}
    (eval_dir / f"{tag}.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False))

    # Raw lm-eval output (for reproducibility)
    raw_dir = eval_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    raw_safe = {}
    for k, v in raw_results.items():
        if k == "samples":
            continue
        try:
            json.dumps(v)
            raw_safe[k] = v
        except (TypeError, ValueError):
            raw_safe[k] = str(v)
    (raw_dir / f"{tag}_raw.json").write_text(
        json.dumps(raw_safe, indent=2, ensure_ascii=False))


def run_exists(eval_dir: Path, tag: str) -> bool:
    return (eval_dir / f"{tag}.json").is_file()


# ── Core evaluation ──────────────────────────────────────────

def run_single_eval(hflm, steering_model, alpha, tasks,
                    num_fewshot=None, limit=None):
    """Run lm-eval with optional steering hooks on an already-loaded HFLM."""
    from src.evaluation.steered_model import apply_steering_hooks

    hook_handle = None
    if steering_model is not None and alpha != 0.0:
        hook_handle = apply_steering_hooks(hflm._model, steering_model, alpha=alpha)

    task_list = [TASK_ALIASES.get(t, t) for t in tasks]
    kw = {}
    if num_fewshot is not None:
        kw["num_fewshot"] = num_fewshot
    if limit is not None:
        kw["limit"] = limit

    results = lm_eval.simple_evaluate(model=hflm, tasks=task_list, **kw)

    if hook_handle is not None:
        hook_handle.remove()

    return results


# ── Summary ──────────────────────────────────────────────────

def build_summary(eval_dir: Path, tasks: list) -> dict:
    """Aggregate all run results into a comparison summary."""
    summary = {
        "generated_at": datetime.now().isoformat(),
        "tasks": tasks,
        "runs": {},
        "comparison": {},
    }

    for f in sorted(eval_dir.glob("*.json")):
        if f.name.startswith("summary"):
            continue
        try:
            data = json.loads(f.read_text())
            tag = data.get("tag", f.stem)
            summary["runs"][tag] = data
        except Exception:
            continue

    for task in tasks:
        primary = TASK_PRIMARY_METRIC.get(task, "acc")
        table = {}
        for tag, data in summary["runs"].items():
            val = data.get("scores", {}).get(task, {}).get(primary)
            if val is not None:
                table[tag] = val
        if table:
            summary["comparison"][task] = {"metric": primary, "values": table}

    return summary


def print_comparison(summary: dict, tasks: list):
    """Print formatted comparison table."""
    runs = summary.get("runs", {})
    if not runs:
        return

    tags = sorted(runs.keys(), key=lambda t: (t != "baseline", t))
    col_w = max(max((len(t) for t in tags), default=8), 12)

    print(f"\n{'='*90}")
    print("  BENCHMARK COMPARISON")
    print(f"{'='*90}")

    # Header
    hdr = f"  {'Task':<20} {'Metric':<15}"
    for t in tags:
        hdr += f"  {t:>{col_w}}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))

    baseline_scores = summary["runs"].get("baseline", {}).get("scores", {})

    for task in tasks:
        comp = summary.get("comparison", {}).get(task)
        if not comp:
            continue
        metric = comp["metric"]
        values = comp["values"]
        bv = values.get("baseline")

        row = f"  {task:<20} {metric:<15}"
        for t in tags:
            v = values.get(t)
            if v is None:
                row += f"  {'—':>{col_w}}"
            elif t == "baseline":
                row += f"  {v:>{col_w}.4f}"
            else:
                delta = v - bv if bv is not None else 0
                sign = "+" if delta >= 0 else ""
                cell = f"{v:.4f}({sign}{delta:.3f})"
                row += f"  {cell:>{col_w}}"
        print(row)

    print(f"{'='*90}")


# ── Main ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate steered LLM on standard benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --alpha 0.0 1.0 --tasks wikitext --limit 10
  %(prog)s --alpha 0.0 -1.0 1.0 2.0
  %(prog)s --mode grid --layers 10 14 16 20 --denoise fisher raw
  %(prog)s --mode grid --resume
  %(prog)s --task-group reasoning --alpha 0.0 1.0
""",
    )

    parser.add_argument("--mode", choices=["single", "grid"], default="single",
                        help="single: one layer. grid: sweep layers × alphas × denoise")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--denoise", type=str, nargs="+", default=["fisher"],
                        choices=["raw", "nullspace", "fisher"])
    parser.add_argument("--layer", type=int, default=14,
                        help="Layer for single mode")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Layers for grid mode")
    parser.add_argument("--alpha", type=float, nargs="+", default=None,
                        help="Steering strengths")

    parser.add_argument("--tasks", type=str, nargs="+", default=None)
    parser.add_argument("--task-group", type=str, default=None,
                        choices=list(TASK_GROUPS.keys()) + ["all"])

    parser.add_argument("--num-fewshot", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed runs")
    parser.add_argument("--no-summary", action="store_true")

    args = parser.parse_args()

    # ── Resolve configuration ──
    model_path = args.model or get_model_path()
    model_id = get_model_id(model_path)
    dataset_tag = extract_dataset_tag(get_dataset_path())
    steer_dir = get_steering_dir(model_id, dataset_tag=dataset_tag)
    eval_dir = get_eval_dir(model_id)
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Tasks
    if args.task_group:
        tasks = ALL_TASKS if args.task_group == "all" else TASK_GROUPS[args.task_group]
    elif args.tasks:
        tasks = args.tasks
    else:
        tasks = ALL_TASKS

    # Alphas
    alphas = args.alpha if args.alpha else DEFAULT_ALPHAS

    # Layers & denoise
    if args.mode == "grid":
        layers = args.layers or DEFAULT_LAYERS
        denoise_methods = args.denoise
    else:
        layers = [args.layer]
        denoise_methods = args.denoise[:1]

    # ── Build run plan ──
    need_baseline = 0.0 in alphas
    nonzero_alphas = sorted(set(a for a in alphas if a != 0.0))

    run_plan = []
    for denoise in denoise_methods:
        for layer in layers:
            sp = resolve_steering_path(steer_dir, denoise, layer)
            if not sp.exists():
                print(f"⚠️  Missing: {sp.name}, skipping")
                continue
            for alpha in nonzero_alphas:
                run_plan.append({
                    "denoise": denoise, "layer": layer, "alpha": alpha,
                    "tag": run_tag(denoise, layer, alpha),
                    "steering_path": str(sp),
                })

    total = (1 if need_baseline else 0) + len(run_plan)
    skip = 0
    if args.resume:
        if need_baseline and run_exists(eval_dir, "baseline"):
            skip += 1
        skip += sum(1 for r in run_plan if run_exists(eval_dir, r["tag"]))

    # ── Print plan ──
    print(f"\n{'='*70}")
    print(f"  BeyondStyle lm-eval Benchmark")
    print(f"{'='*70}")
    print(f"  Model:       {model_id}")
    print(f"  Tasks ({len(tasks)}):  {tasks}")
    print(f"  Denoise:     {denoise_methods}")
    print(f"  Layers:      {layers}")
    print(f"  Alphas:      {sorted(set(alphas))}")
    print(f"  Total runs:  {total}" +
          (f" ({skip} cached, {total-skip} to run)" if args.resume else ""))
    print(f"  Output:      {eval_dir}")
    print(f"  Limit:       {args.limit or 'full'}")
    print(f"{'='*70}\n")

    if total - skip == 0 and args.resume:
        print("✅ All runs cached. Remove --resume to re-run.")
        if not args.no_summary:
            summary = build_summary(eval_dir, tasks)
            print_comparison(summary, tasks)
        return

    # ── Load model ONCE ──
    from src.evaluation.steered_model import load_hflm, _load_steering_model

    t0 = time.time()
    print("Loading model (one-time)...", flush=True)
    hflm = load_hflm(
        pretrained=model_path,
        device=args.device,
        batch_size=args.batch_size,
        dtype="float16",
    )
    print(f"Model loaded in {time.time()-t0:.1f}s\n", flush=True)

    done = 0

    # ── Baseline ──
    if need_baseline:
        tag = "baseline"
        if args.resume and run_exists(eval_dir, tag):
            print(f"  ⏭️  [{done+1}/{total}] {tag} (cached)")
        else:
            print(f"\n  {'─'*56}")
            print(f"  [{done+1}/{total}] BASELINE (α=0.0)")
            print(f"  {'─'*56}", flush=True)

            t1 = time.time()
            results = run_single_eval(hflm, None, 0.0, tasks,
                                      num_fewshot=args.num_fewshot,
                                      limit=args.limit)
            elapsed = time.time() - t1
            scores = extract_scores(results)

            meta = {"model": model_id, "alpha": 0.0, "denoise": None,
                    "layer": None, "tasks": tasks, "limit": args.limit,
                    "elapsed_seconds": round(elapsed, 1)}
            save_run(eval_dir, tag, meta, scores, results)

            print(f"\n  ✅ Baseline ({elapsed:.0f}s):")
            for t, m in scores.items():
                print(f"     {t}: {', '.join(f'{k}={v:.4f}' for k,v in m.items())}")
        done += 1

    # ── Steered runs (grouped by denoise+layer to minimize vector loads) ──
    def gk(r):
        return (r["denoise"], r["layer"], r["steering_path"])

    for (denoise, layer, sp), grp in groupby(sorted(run_plan, key=gk), key=gk):
        grp = list(grp)
        sm = _load_steering_model(sp, device=hflm.device)
        print(f"\n📦 Steering: {denoise} L{layer} "
              f"(dim={sm.get_direction().shape[0]})", flush=True)

        for ri in grp:
            alpha, tag = ri["alpha"], ri["tag"]
            done += 1

            if args.resume and run_exists(eval_dir, tag):
                print(f"  ⏭️  [{done}/{total}] {tag} (cached)")
                continue

            print(f"\n  {'─'*56}")
            print(f"  [{done}/{total}] {tag}  (α={alpha:+.1f})")
            print(f"  {'─'*56}", flush=True)

            t1 = time.time()
            results = run_single_eval(hflm, sm, alpha, tasks,
                                      num_fewshot=args.num_fewshot,
                                      limit=args.limit)
            elapsed = time.time() - t1
            scores = extract_scores(results)

            meta = {"model": model_id, "alpha": alpha, "denoise": denoise,
                    "layer": layer, "steering_path": sp, "tasks": tasks,
                    "limit": args.limit, "elapsed_seconds": round(elapsed, 1)}
            save_run(eval_dir, tag, meta, scores, results)

            print(f"\n  ✅ {tag} ({elapsed:.0f}s):")
            for t, m in scores.items():
                print(f"     {t}: {', '.join(f'{k}={v:.4f}' for k,v in m.items())}")

        del sm
        gc.collect()

    # ── Summary ──
    if not args.no_summary:
        summary = build_summary(eval_dir, tasks)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        (eval_dir / f"summary_{ts}.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False))
        (eval_dir / "summary_latest.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False))

        print(f"\n📊 Summary → {eval_dir / f'summary_{ts}.json'}")
        print_comparison(summary, tasks)

    print(f"\n✅ Done. {total} runs, results in {eval_dir}/")


if __name__ == "__main__":
    main()
