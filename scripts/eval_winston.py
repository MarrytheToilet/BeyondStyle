#!/usr/bin/env python3
"""Winston AI detection evaluation for pre-generated steered text.

Reads generation files produced by ``scripts/steering_pipeline.py``::

    outputs/<language>/steering/<model>/<dataset>/generation/generation_<method>.json

Each entry::

    {"layer": 15, "method": "...", "prompt": "...",
     "outputs": {"-2.0": {"text": "..."}, "0.0": {...}, ...}}

Only the **continuation** (``outputs[α]["text"]``) is sent to Winston —
the shared prompt prefix is excluded so it cannot bias the comparison
across α values.

Defaults: ``--method raw nullspace --layer 15 --alpha -2 0 2 -n 100``.
For each method we save one results JSON plus a Markdown summary, and
a combined comparison summary across methods.

Environment:
    WINSTON_API_KEY  — Winston API bearer token (required).
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Dict, List, Optional, Tuple

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


WINSTON_URL = "https://api.gowinston.ai/v2/ai-content-detection"
# Winston rejects texts shorter than this many characters with VALIDATION_FAILED.
MIN_TEXT_CHARS = 300

METHOD_FILES = {
    "raw":       "generation_mean_diff.json",
    "fisher":    "generation_mean_diff_fisher.json",
    "nullspace": "generation_mean_diff_nullspace.json",
}


# ─────────────────────────────────────────────────────────
# Loading & filtering
# ─────────────────────────────────────────────────────────
def resolve_generation_file(
    base_dir: Path, language: str, model: str, dataset: str, method: str,
) -> Path:
    path = (base_dir / language / "steering" / model / dataset /
            "generation" / METHOD_FILES[method])
    if not path.exists():
        raise FileNotFoundError(f"Generation file not found: {path}")
    return path


def load_entries(
    path: Path,
    layers: Optional[List[int]] = None,
    alphas: Optional[List[float]] = None,
    num_samples: Optional[int] = None,
) -> List[Dict]:
    with open(path) as f:
        entries = json.load(f)
    if not isinstance(entries, list):
        raise ValueError(f"Expected a list of entries in {path}")

    if layers is not None:
        layers_set = {int(x) for x in layers}
        entries = [e for e in entries if int(e.get("layer", -1)) in layers_set]

    if alphas is not None:
        want = {f"{a:.1f}" for a in alphas} | {str(a) for a in alphas}
        filtered = []
        for e in entries:
            kept = {k: v for k, v in e.get("outputs", {}).items() if k in want}
            if kept:
                filtered.append({**e, "outputs": kept})
        entries = filtered

    if num_samples is not None and num_samples > 0:
        by_layer: Dict[int, List[Dict]] = {}
        for e in entries:
            by_layer.setdefault(int(e.get("layer", -1)), []).append(e)
        entries = []
        for layer_val in sorted(by_layer):
            entries.extend(by_layer[layer_val][:num_samples])

    return entries


# ─────────────────────────────────────────────────────────
# Winston detection
# ─────────────────────────────────────────────────────────
def detect_winston(
    text: str,
    api_key: str,
    session: Optional[requests.Session] = None,
    sentences: bool = True,
    language: str = "auto",
    timeout: float = 60.0,
    retries: int = 3,
    backoff: float = 1.5,
) -> Optional[Dict]:
    """Score one text. Returns a normalized dict, ``None`` on permanent failure.

    Convention: ``ai_likelihood`` ∈ [0, 1], **higher = more AI-like**.
    Winston's raw ``score`` is on [0, 100] where higher = more *human*-like.
    """
    if len(text) < MIN_TEXT_CHARS:
        return {"ai_likelihood": None, "skipped": "too_short", "length": len(text)}

    sess = session or requests.Session()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"text": text, "sentences": sentences, "language": language}

    last_err: Optional[str] = None
    for attempt in range(retries + 1):
        try:
            r = sess.post(WINSTON_URL, json=payload, headers=headers, timeout=timeout)
            if r.status_code == 200:
                data = r.json()
                score = data.get("score")
                if score is None:
                    ai_likelihood = None
                    human_norm = None
                else:
                    human_norm = float(score) / 100.0
                    ai_likelihood = 1.0 - human_norm

                sent_list = []
                for s in (data.get("sentences") or []):
                    sent_list.append({
                        "length": s.get("length"),
                        "score": s.get("score"),
                        "text": s.get("text"),
                    })
                return {
                    "ai_likelihood": ai_likelihood,
                    "human_score_norm": human_norm,
                    "score_raw": score,
                    "length": data.get("length"),
                    "readability_score": data.get("readability_score"),
                    "attack_detected": data.get("attack_detected"),
                    "language": data.get("language"),
                    "credits_used": data.get("credits_used"),
                    "credits_remaining": data.get("credits_remaining"),
                    "sentences": sent_list,
                }
            if 400 <= r.status_code < 500 and r.status_code not in (408, 429):
                try:
                    err = r.json()
                except Exception:
                    err = {"error": r.text[:200]}
                return {"ai_likelihood": None, "error": err, "status": r.status_code}
            last_err = f"HTTP {r.status_code}: {r.text[:200]}"
        except Exception as e:
            last_err = str(e)
        if attempt < retries:
            time.sleep(backoff ** attempt)
    print(f"    ⚠ Winston API error after {retries+1} attempts: {last_err}")
    return None


def run_detection(
    entries: List[Dict],
    api_key: str,
    workers: int = 4,
    retries: int = 3,
    desc: str = "Winston",
) -> Tuple[List[Dict], Dict[str, int]]:
    from concurrent.futures import ThreadPoolExecutor, as_completed
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x  # noqa: E731

    results: List[Dict] = [
        {"layer": e.get("layer"), "method": e.get("method"),
         "prompt": e["prompt"], "results": {}}
        for e in entries
    ]

    # Build flat task list. **Only the continuation text is sent**, no prompt.
    tasks = []
    for i, entry in enumerate(entries):
        for alpha_str, gen in entry["outputs"].items():
            text = gen["text"] if isinstance(gen, dict) else str(gen)
            tasks.append((i, alpha_str, text))

    sessions = [requests.Session() for _ in range(max(1, workers))]
    fail = skipped = 0

    def _job(text, sess):
        return detect_winston(text, api_key, session=sess, retries=retries)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {}
        for k, (i, a, txt) in enumerate(tasks):
            sess = sessions[k % len(sessions)]
            futures[pool.submit(_job, txt, sess)] = (i, a)
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc=desc, unit="req"):
            i, a = futures[fut]
            score = fut.result()
            results[i]["results"][a] = score
            if score is None:
                fail += 1
            elif score.get("ai_likelihood") is None:
                skipped += 1

    stats = {"total": len(tasks), "failed": fail, "skipped": skipped,
             "ok": len(tasks) - fail - skipped}
    if fail or skipped:
        print(f"  ⚠ {fail} failed, {skipped} skipped/invalid out of {len(tasks)} requests.")
    return results, stats


# ─────────────────────────────────────────────────────────
# Aggregation / metrics
# ─────────────────────────────────────────────────────────
def _safe_stats(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "median": None,
                "min": None, "max": None}
    return {
        "n": len(values),
        "mean": mean(values),
        "std": pstdev(values) if len(values) > 1 else 0.0,
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def aggregate(results: List[Dict]) -> Dict:
    """Aggregate per-(layer, alpha) metrics from raw API results."""
    buckets: Dict[Tuple, Dict[str, List]] = {}
    for entry in results:
        layer = entry.get("layer")
        for alpha_str, score in entry["results"].items():
            key = (layer, alpha_str)
            b = buckets.setdefault(key, {
                "ai": [], "human": [], "readability": [],
                "length": [], "sent_ai_frac": [], "sent_score_means": [],
                "attack_zws": 0, "attack_homo": 0,
                "n_total": 0, "n_ok": 0, "n_skip": 0, "n_fail": 0,
            })
            b["n_total"] += 1
            if score is None:
                b["n_fail"] += 1
                continue
            ai = score.get("ai_likelihood")
            if ai is None:
                b["n_skip"] += 1
                continue
            b["n_ok"] += 1
            b["ai"].append(float(ai))
            if score.get("human_score_norm") is not None:
                b["human"].append(float(score["human_score_norm"]))
            if score.get("readability_score") is not None:
                b["readability"].append(float(score["readability_score"]))
            if score.get("length") is not None:
                b["length"].append(int(score["length"]))
            atk = score.get("attack_detected") or {}
            if atk.get("zero_width_space"):
                b["attack_zws"] += 1
            if atk.get("homoglyph_attack"):
                b["attack_homo"] += 1

            sents = score.get("sentences") or []
            if sents:
                # Winston sentence-level `score` is also human-likeness 0-100.
                sent_human = [float(s["score"]) / 100.0 for s in sents
                              if s.get("score") is not None]
                if sent_human:
                    sent_ai = [1 - h for h in sent_human]
                    b["sent_ai_frac"].append(
                        sum(1 for a in sent_ai if a >= 0.5) / len(sent_ai)
                    )
                    b["sent_score_means"].append(sum(sent_ai) / len(sent_ai))

    summary: Dict[str, Dict] = {}
    for (layer, alpha), b in buckets.items():
        key = f"L{layer}_a{alpha}"
        ai = b["ai"]
        summary[key] = {
            "layer": layer,
            "alpha": float(alpha),
            "counts": {"total": b["n_total"], "ok": b["n_ok"],
                       "skipped": b["n_skip"], "failed": b["n_fail"]},
            "ai_likelihood": _safe_stats(ai),
            "fraction_called_ai_at_0.5": (
                sum(1 for x in ai if x >= 0.5) / len(ai) if ai else None),
            "fraction_called_ai_at_0.9": (
                sum(1 for x in ai if x >= 0.9) / len(ai) if ai else None),
            "human_score_norm": _safe_stats(b["human"]),
            "readability_score": _safe_stats(b["readability"]),
            "length_chars": _safe_stats(b["length"]),
            "sentence_ai_fraction": _safe_stats(b["sent_ai_frac"]),
            "sentence_mean_ai_score": _safe_stats(b["sent_score_means"]),
            "attacks": {"zero_width_space": b["attack_zws"],
                        "homoglyph_attack": b["attack_homo"]},
        }
    return summary


def _fmt(v, prec=4) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{prec}f}"
    return str(v)


def render_markdown_summary(method: str, agg: Dict, stats: Dict,
                            meta: Dict) -> str:
    """Build a Markdown summary for one method."""
    lines = []
    lines.append(f"# Winston eval — {method}")
    lines.append("")
    lines.append(f"- Source: `{meta['source_file']}`")
    lines.append(f"- Model: `{meta['model']}`  Dataset: `{meta['dataset']}`  "
                 f"Lang: `{meta['language']}`")
    lines.append(f"- Layers: {meta['layers']}  Alphas: {meta['alphas']}  "
                 f"Samples per (layer, α): up to {meta['num_samples']}")
    lines.append(f"- Requests: total={stats['total']}, ok={stats['ok']}, "
                 f"skipped={stats['skipped']}, failed={stats['failed']}")
    lines.append(f"- Generated: {meta['timestamp']}")
    lines.append("")
    lines.append("## Per (layer, α) metrics")
    lines.append("")
    headers = ["layer", "α", "n", "AI mean", "AI std", "AI median",
               "%≥0.5", "%≥0.9", "readab.", "len(chars)",
               "sent %AI", "sent meanAI"]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    rows = sorted(agg.values(), key=lambda r: (r["layer"] is None, r["layer"],
                                               r["alpha"]))
    for r in rows:
        ai = r["ai_likelihood"]
        rd = r["readability_score"]
        ln = r["length_chars"]
        sf = r["sentence_ai_fraction"]
        sm = r["sentence_mean_ai_score"]
        lines.append("| " + " | ".join([
            str(r["layer"]),
            f"{r['alpha']:+.1f}",
            str(ai["n"]),
            _fmt(ai["mean"]), _fmt(ai["std"]), _fmt(ai["median"]),
            _fmt(r["fraction_called_ai_at_0.5"], 3),
            _fmt(r["fraction_called_ai_at_0.9"], 3),
            _fmt(rd["mean"], 2),
            _fmt(ln["mean"], 1),
            _fmt(sf["mean"], 3),
            _fmt(sm["mean"], 4),
        ]) + " |")
    lines.append("")
    lines.append("> AI likelihood ∈ [0,1], higher = more AI-like to Winston. "
                 "`%≥t` is the fraction of samples whose AI likelihood passes "
                 "threshold *t*. Sentence metrics are computed per sample then "
                 "averaged.")
    return "\n".join(lines) + "\n"


def render_combined_summary(by_method: Dict[str, Dict], meta: Dict) -> str:
    """Compare methods side-by-side per (layer, α)."""
    lines = ["# Winston eval — combined summary", ""]
    lines.append(f"- Methods: {list(by_method)}")
    lines.append(f"- Layers: {meta['layers']}  Alphas: {meta['alphas']}  "
                 f"n≤{meta['num_samples']}")
    lines.append(f"- Generated: {meta['timestamp']}")
    lines.append("")
    lines.append("## Mean AI likelihood per method")
    lines.append("")
    # gather all (layer, alpha) keys
    keys = set()
    for agg in by_method.values():
        keys.update((v["layer"], v["alpha"]) for v in agg.values())
    keys = sorted(keys, key=lambda k: (k[0] is None, k[0], k[1]))

    methods = list(by_method)
    headers = ["layer", "α"] + [f"{m} mean" for m in methods] \
              + [f"{m} %≥0.5" for m in methods]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for layer, alpha in keys:
        row = [str(layer), f"{alpha:+.1f}"]
        for m in methods:
            agg = by_method[m]
            entry = next((v for v in agg.values()
                          if v["layer"] == layer and v["alpha"] == alpha), None)
            row.append(_fmt(entry["ai_likelihood"]["mean"]) if entry else "—")
        for m in methods:
            agg = by_method[m]
            entry = next((v for v in agg.values()
                          if v["layer"] == layer and v["alpha"] == alpha), None)
            row.append(_fmt(entry["fraction_called_ai_at_0.5"], 3)
                       if entry else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    return "\n".join(lines) + "\n"


def print_table(method: str, agg: Dict) -> None:
    print(f"\n  ── {method} ──")
    print(f"  {'Layer':>6} {'Alpha':>7} {'N':>4} {'AI mean':>9} "
          f"{'AI std':>8} {'%≥.5':>6} {'%≥.9':>6} {'readab':>7} {'len':>6}")
    rows = sorted(agg.values(), key=lambda r: (r["layer"] is None, r["layer"],
                                               r["alpha"]))
    for r in rows:
        ai = r["ai_likelihood"]
        print(f"  {str(r['layer']):>6} {r['alpha']:>+7.1f} {ai['n']:>4} "
              f"{_fmt(ai['mean']):>9} {_fmt(ai['std']):>8} "
              f"{_fmt(r['fraction_called_ai_at_0.5'],3):>6} "
              f"{_fmt(r['fraction_called_ai_at_0.9'],3):>6} "
              f"{_fmt(r['readability_score']['mean'],2):>7} "
              f"{_fmt(r['length_chars']['mean'],1):>6}")


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Winston AI-detection scoring over pre-generated steered text.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--base-dir", type=Path, default=ROOT / "outputs")
    p.add_argument("--language", "-L", default="en")
    p.add_argument("--model", "-m", default="Llama-3.1-8B-Instruct")
    p.add_argument("--dataset", "-d", default="mixed_500")
    p.add_argument("--method", nargs="+",
                   choices=list(METHOD_FILES),
                   default=["raw", "nullspace"],
                   help="One or more steering / denoising methods to evaluate.")
    p.add_argument("--layer", type=int, nargs="+", default=[15])
    p.add_argument("--alpha", type=float, nargs="+", default=[-2.0, 0.0, 2.0])
    p.add_argument("-n", "--num-samples", type=int, default=100)
    p.add_argument("--workers", "-w", type=int, default=4)
    p.add_argument("--retries", type=int, default=3)
    p.add_argument("--max-calls", type=int, default=None,
                   help="Hard cap on number of API calls per method (debug budget).")
    p.add_argument("--input", type=Path, default=None,
                   help="Override: single generation JSON file. When set, only one "
                        "'--method' tag is used (for output naming).")
    return p.parse_args()


def cap_entries(entries: List[Dict], max_calls: Optional[int]) -> List[Dict]:
    if not max_calls or max_calls <= 0:
        return entries
    used = 0
    capped: List[Dict] = []
    for e in entries:
        if used >= max_calls:
            break
        outs = {}
        for a, g in e["outputs"].items():
            if used >= max_calls:
                break
            outs[a] = g
            used += 1
        if outs:
            capped.append({**e, "outputs": outs})
    return capped


def main() -> None:
    args = parse_args()

    api_key = os.environ.get("WINSTON_API_KEY")
    if not api_key:
        raise SystemExit("WINSTON_API_KEY is not set. Add it to .env or export it.")

    out_dir = args.base_dir / args.language / "05_perception" / "winston"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")

    by_method_agg: Dict[str, Dict] = {}
    layer_tag = "-".join(str(l) for l in args.layer)

    methods = args.method if not args.input else [args.method[0]]
    for method in methods:
        gen_path = args.input or resolve_generation_file(
            args.base_dir, args.language, args.model, args.dataset, method,
        )
        entries = load_entries(
            gen_path, layers=args.layer, alphas=args.alpha,
            num_samples=args.num_samples,
        )
        if not entries:
            print(f"⚠ No entries matched filters in {gen_path}; skipping {method}")
            continue
        entries = cap_entries(entries, args.max_calls)

        uniq_layers = sorted({int(e["layer"]) for e in entries})
        uniq_alphas = sorted({a for e in entries for a in e["outputs"]}, key=float)
        total_calls = sum(len(e["outputs"]) for e in entries)

        print(f"\n=== Winston eval :: method={method} ===")
        print(f"  Source : {gen_path}")
        print(f"  Entries: {len(entries)}  layers={uniq_layers}  alphas={uniq_alphas}")
        print(f"  Calls  : {total_calls}  workers={args.workers}  retries={args.retries}")

        results, stats = run_detection(
            entries, api_key, workers=args.workers,
            retries=args.retries, desc=f"Winston[{method}]",
        )
        agg = aggregate(results)
        by_method_agg[method] = agg

        meta = {
            "source_file": str(gen_path),
            "language": args.language,
            "model": args.model,
            "dataset": args.dataset,
            "method": method,
            "layers": uniq_layers,
            "alphas": uniq_alphas,
            "num_samples": args.num_samples,
            "timestamp": timestamp,
        }

        # Raw results JSON
        results_path = out_dir / f"winston_{method}_L{layer_tag}.json"
        with open(results_path, "w") as f:
            json.dump({
                "detector": "winston",
                "input_text_policy": "continuation_only",
                **meta,
                "stats": stats,
                "summary": agg,
                "results": results,
            }, f, indent=2, ensure_ascii=False)
        print(f"  ✅ Results → {results_path}")

        # Per-method Markdown summary
        md_path = out_dir / f"summary_{method}_L{layer_tag}.md"
        md_path.write_text(render_markdown_summary(method, agg, stats, meta),
                           encoding="utf-8")
        print(f"  📝 Summary → {md_path}")

        print_table(method, agg)

    # Combined summary across methods
    if len(by_method_agg) > 1:
        combined_meta = {
            "layers": args.layer,
            "alphas": args.alpha,
            "num_samples": args.num_samples,
            "timestamp": timestamp,
        }
        combined_path = out_dir / f"summary_combined_L{layer_tag}.md"
        combined_path.write_text(
            render_combined_summary(by_method_agg, combined_meta),
            encoding="utf-8",
        )
        print(f"\n📊 Combined summary → {combined_path}")


if __name__ == "__main__":
    main()
