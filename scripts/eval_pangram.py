#!/usr/bin/env python3
"""Pangram AI-detection evaluation for pre-generated steered text.

Mirrors ``scripts/eval_winston.py`` but uses the Pangram SDK.

Usage::

    python scripts/eval_pangram.py \
        --generation-file outputs/en/03_steering_vectors/Llama-3.1-8B-Instruct/mixed_500/generation/cross_openwebtext_mixed_300/generation_mean_diff_nullspace.json \
        --layers 15 --alphas -2 0 2 -n 100

Reads a JSON list of entries::

    [{"layer": 15, "method": "...", "prompt": "...",
      "outputs": {"-2.0": {"text": "..."}, "0.0": {...}, ...}}, ...]

Sends only the **continuation** text (no prefix) to Pangram and writes
results next to the generation file::

    pangram_results.json    — raw per-(entry, alpha) results
    pangram_summary.json    — aggregated per-(layer, alpha) metrics
    pangram_summary.md      — human-readable summary

Environment:
    PANGRAM_API_KEY  — required
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kw: x  # noqa: E731

from pangram import Pangram


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--generation-file", "-g", required=True,
                   help="Path to generation JSON")
    p.add_argument("--layers", type=int, nargs="+", default=None)
    p.add_argument("--alphas", type=float, nargs="+", default=None)
    p.add_argument("-n", "--num-samples", type=int, default=None,
                   help="Cap samples per layer (default: all)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--out-dir", default=None,
                   help="Output dir (default: alongside the generation file)")
    p.add_argument("--tag", default="pangram",
                   help="Output filename prefix (default: pangram)")
    return p.parse_args()


def load_entries(path: Path, layers, alphas, num_samples):
    with open(path) as f:
        entries = json.load(f)
    if not isinstance(entries, list):
        raise ValueError(f"Expected list in {path}")
    if layers is not None:
        layers_set = {int(x) for x in layers}
        entries = [e for e in entries if int(e.get("layer", -1)) in layers_set]
    if alphas is not None:
        want = {f"{a:.1f}" for a in alphas} | {str(a) for a in alphas} | {str(int(a)) for a in alphas if float(a).is_integer()}
        filt = []
        for e in entries:
            kept = {k: v for k, v in e.get("outputs", {}).items() if k in want}
            if kept:
                filt.append({**e, "outputs": kept})
        entries = filt
    if num_samples and num_samples > 0:
        by_layer: Dict[int, List[Dict]] = {}
        for e in entries:
            by_layer.setdefault(int(e.get("layer", -1)), []).append(e)
        entries = []
        for ly in sorted(by_layer):
            entries.extend(by_layer[ly][:num_samples])
    return entries


def detect_pangram(client: Pangram, text: str, retries: int = 3) -> Optional[Dict]:
    """Returns a normalized dict; None on permanent failure."""
    last_err = None
    for attempt in range(retries + 1):
        try:
            r = client.predict(text)
            return {
                "fraction_ai": r.get("fraction_ai"),
                "fraction_ai_assisted": r.get("fraction_ai_assisted"),
                "fraction_human": r.get("fraction_human"),
                "num_ai_segments": r.get("num_ai_segments"),
                "windows": r.get("windows"),
                "length": len(text),
            }
        except Exception as e:
            last_err = str(e)
            if attempt < retries:
                time.sleep(1.5 ** attempt)
    print(f"    ⚠ Pangram error after {retries+1} attempts: {last_err}")
    return None


def run_detection(entries, api_key, workers=4, retries=3):
    clients = [Pangram(api_key=api_key) for _ in range(max(1, workers))]
    results = [
        {"layer": e.get("layer"), "method": e.get("method"),
         "prompt": e["prompt"], "results": {}}
        for e in entries
    ]
    tasks = []
    for i, e in enumerate(entries):
        for a, gen in e["outputs"].items():
            text = gen["text"] if isinstance(gen, dict) else str(gen)
            tasks.append((i, a, text))

    fail = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futs = {}
        for k, (i, a, t) in enumerate(tasks):
            cli = clients[k % len(clients)]
            futs[pool.submit(detect_pangram, cli, t, retries)] = (i, a)
        for fut in tqdm(as_completed(futs), total=len(futs),
                        desc="Pangram", unit="req"):
            i, a = futs[fut]
            sc = fut.result()
            results[i]["results"][a] = sc
            if sc is None:
                fail += 1

    stats = {"total": len(tasks), "ok": len(tasks) - fail, "failed": fail}
    if fail:
        print(f"  ⚠ {fail}/{len(tasks)} requests failed")
    return results, stats


def _safe_stats(vals):
    if not vals:
        return {"n": 0, "mean": None, "std": None, "median": None,
                "min": None, "max": None}
    return {
        "n": len(vals), "mean": mean(vals),
        "std": pstdev(vals) if len(vals) > 1 else 0.0,
        "median": median(vals), "min": min(vals), "max": max(vals),
    }


def aggregate(results):
    buckets: Dict[Tuple, Dict[str, List]] = {}
    for r in results:
        ly = r.get("layer")
        for a, sc in r["results"].items():
            key = (ly, a)
            b = buckets.setdefault(key, {
                "fai": [], "faia": [], "fhum": [], "nseg": [], "len": [],
                "n_total": 0, "n_ok": 0, "n_fail": 0,
            })
            b["n_total"] += 1
            if sc is None:
                b["n_fail"] += 1
                continue
            b["n_ok"] += 1
            for k_src, k_dst in [
                ("fraction_ai", "fai"),
                ("fraction_ai_assisted", "faia"),
                ("fraction_human", "fhum"),
                ("num_ai_segments", "nseg"),
                ("length", "len"),
            ]:
                v = sc.get(k_src)
                if v is not None:
                    b[k_dst].append(float(v))

    summary = {}
    for (ly, a), b in buckets.items():
        fai = b["fai"]
        summary[f"L{ly}_a{a}"] = {
            "layer": ly,
            "alpha": float(a),
            "counts": {"total": b["n_total"], "ok": b["n_ok"],
                       "failed": b["n_fail"]},
            "fraction_ai": _safe_stats(fai),
            "fraction_ai_assisted": _safe_stats(b["faia"]),
            "fraction_human": _safe_stats(b["fhum"]),
            "num_ai_segments": _safe_stats(b["nseg"]),
            "length_chars": _safe_stats(b["len"]),
            "fraction_called_ai_at_0.5": (
                sum(1 for x in fai if x >= 0.5) / len(fai)) if fai else None,
            "fraction_called_ai_at_0.9": (
                sum(1 for x in fai if x >= 0.9) / len(fai)) if fai else None,
        }
    return summary


def _fmt(v, prec=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{prec}f}"
    return str(v)


def render_md(agg, stats, meta):
    L = ["# Pangram eval", ""]
    L.append(f"- Source: `{meta['source_file']}`")
    L.append(f"- Layers: {meta['layers']}  Alphas: {meta['alphas']}  "
             f"n≤{meta['num_samples']}")
    L.append(f"- Requests: total={stats['total']}, ok={stats['ok']}, "
             f"failed={stats['failed']}")
    L.append("")
    headers = ["layer", "α", "n", "fAI mean", "fAI std", "fAI median",
               "fAI assisted", "fHuman", "num_AI_seg",
               "%≥0.5", "%≥0.9", "len"]
    L.append("| " + " | ".join(headers) + " |")
    L.append("|" + "|".join(["---"] * len(headers)) + "|")
    rows = sorted(agg.values(), key=lambda r: (r["layer"], r["alpha"]))
    for r in rows:
        f = r["fraction_ai"]; fa = r["fraction_ai_assisted"]
        fh = r["fraction_human"]; ns = r["num_ai_segments"]
        ln = r["length_chars"]
        L.append("| " + " | ".join([
            str(r["layer"]), f"{r['alpha']:+.1f}", str(f["n"]),
            _fmt(f["mean"]), _fmt(f["std"]), _fmt(f["median"]),
            _fmt(fa["mean"]), _fmt(fh["mean"]), _fmt(ns["mean"], 2),
            _fmt(r["fraction_called_ai_at_0.5"], 3),
            _fmt(r["fraction_called_ai_at_0.9"], 3),
            _fmt(ln["mean"], 1),
        ]) + " |")
    L.append("")
    L.append("> `fraction_ai` ∈ [0,1] = Pangram-estimated AI fraction "
             "(higher → more AI-like).  α<0 → toward human, α>0 → toward AI.")
    return "\n".join(L) + "\n"


def main():
    args = parse_args()
    api_key = os.environ.get("PANGRAM_API_KEY")
    if not api_key:
        # Pangram() will read it itself, but we want to fail fast
        try:
            from dotenv import load_dotenv
            load_dotenv()
            api_key = os.environ.get("PANGRAM_API_KEY")
        except ImportError:
            pass
    if not api_key:
        sys.exit("PANGRAM_API_KEY env var required")

    src = Path(args.generation_file)
    if not src.exists():
        sys.exit(f"Generation file not found: {src}")

    out_dir = Path(args.out_dir) if args.out_dir else src.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading entries from {src}")
    entries = load_entries(src, args.layers, args.alphas, args.num_samples)
    print(f"  {len(entries)} entries after filtering")

    results, stats = run_detection(
        entries, api_key, workers=args.workers, retries=args.retries,
    )
    agg = aggregate(results)

    raw_path = out_dir / f"{args.tag}_results.json"
    sum_path = out_dir / f"{args.tag}_summary.json"
    md_path = out_dir / f"{args.tag}_summary.md"

    with open(raw_path, "w") as f:
        json.dump({"results": results, "stats": stats}, f, indent=2)
    with open(sum_path, "w") as f:
        json.dump(agg, f, indent=2)

    meta = {
        "source_file": str(src),
        "layers": args.layers,
        "alphas": args.alphas,
        "num_samples": args.num_samples,
    }
    md_path.write_text(render_md(agg, stats, meta))
    print(f"\nSaved → {raw_path}")
    print(f"Saved → {sum_path}")
    print(f"Saved → {md_path}")


if __name__ == "__main__":
    main()
