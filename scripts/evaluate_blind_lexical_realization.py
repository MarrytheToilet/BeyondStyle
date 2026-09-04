#!/usr/bin/env python3
"""Blind pairwise evaluation of held-out lexical-steering generations.

The evaluator never sees model names, intervention signs, or surface scores.
One predeclared seed is selected per prompt so prompts, rather than repeated
decodes, remain the unit of evaluation.  The script is resumable: every API
response is appended to JSONL before aggregate statistics are written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "Llama": ROOT / "outputs/en/03_steering_vectors/Llama-3.1-8B-Instruct/proposition_aligned/mean_pool/causal_pullback_confirmatory_generation/generation_component_lexical.json",
    "Qwen": ROOT / "outputs/en/03_steering_vectors/Qwen3-4B/proposition_aligned/mean_pool/causal_pullback_qwen_layer22_strength3p0_heldout_generation/generation_component_lexical.json",
    "Gemma 4": ROOT / "outputs/en/03_steering_vectors/gemma-4-E2B-it/proposition_aligned/mean_pool/causal_pullback_gemma_lengthmatched_layer03_strength0p25_heldout_generation/generation_component_lexical.json",
}

SYSTEM = """You are evaluating two anonymized continuations of the same prefix. Judge only the text shown. The order is randomized; do not infer how either text was produced.

The target lexical tendency is a distributed pattern: greater vocabulary diversity, less reliance on very frequent content words, and less function-word-heavy phrasing. It is not simply length, syntax, discourse organization, factual detail, or formality. Decide which continuation more strongly exhibits that lexical tendency. Also assess whether the pair preserves broadly equivalent content and whether either continuation has quality problems.

Return exactly one JSON object with these keys:
lexical_preference: "A", "B", or "tie"
lexical_salience: integer 1-5 (1 imperceptible, 5 obvious)
content_equivalence: integer 1-5 (1 different, 5 essentially equivalent)
coherence_A: integer 1-5
coherence_B: integer 1-5
fluency_A: integer 1-5
fluency_B: integer 1-5
mechanical_artifact_A: true or false
mechanical_artifact_B: true or false
confidence: integer 1-5
rationale: at most 35 words

Do not add markdown or any text outside the JSON object."""


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluators", default="qwen3.8-flash,deepseek-v4-pro")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="Prompts per target model; 0 means all.")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/en/05_linguistic/external_evaluation")
    return parser.parse_args()


def selected_rows(path: Path, limit: int) -> list[dict[str, Any]]:
    rows = json.loads(path.read_text())
    chosen = [r for r in rows if int(r["sample_seed_index"]) == (int(r["prompt_index"]) - 34) % 3]
    chosen.sort(key=lambda r: int(r["prompt_index"]))
    if limit:
        chosen = chosen[:limit]
    return chosen


def output_pair(row: dict[str, Any]) -> tuple[str, str]:
    outputs = row["outputs"]
    negative = next(v["text"] for v in outputs.values() if v["direction"] == "opposite_component")
    positive = next(v["text"] for v in outputs.values() if v["direction"] == "along_component")
    return negative, positive


def make_tasks(evaluators: list[str], limit: int) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for target_model, path in INPUTS.items():
        for row in selected_rows(path, limit):
            negative, positive = output_pair(row)
            prompt_index = int(row["prompt_index"])
            seed_index = int(row["sample_seed_index"])
            swap = bool(stable_int(f"{target_model}|{prompt_index}|{seed_index}") % 2)
            text_a, text_b = (positive, negative) if swap else (negative, positive)
            plus_label = "A" if swap else "B"
            user = f"PREFIX:\n{row['prompt']}\n\nCONTINUATION A:\n{text_a}\n\nCONTINUATION B:\n{text_b}"
            for evaluator in evaluators:
                tasks.append({
                    "task_id": f"{target_model}|{prompt_index}|{seed_index}|{evaluator}",
                    "target_model": target_model,
                    "prompt_id": row["prompt_id"],
                    "prompt_index": prompt_index,
                    "seed_index": seed_index,
                    "evaluator": evaluator,
                    "plus_label": plus_label,
                    "prompt_hash": hashlib.sha256(row["prompt"].encode()).hexdigest(),
                    "minus_hash": hashlib.sha256(negative.encode()).hexdigest(),
                    "plus_hash": hashlib.sha256(positive.encode()).hexdigest(),
                    "user": user,
                })
    return tasks


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"No JSON object in response: {text[:160]}")
    obj = json.loads(text[start : end + 1])
    required = {
        "lexical_preference", "lexical_salience", "content_equivalence",
        "coherence_A", "coherence_B", "fluency_A", "fluency_B",
        "mechanical_artifact_A", "mechanical_artifact_B", "confidence", "rationale",
    }
    if set(obj) != required:
        raise ValueError(f"Unexpected keys: {sorted(obj)}")
    if obj["lexical_preference"] not in {"A", "B", "tie"}:
        raise ValueError("Invalid lexical_preference")
    for key in ("lexical_salience", "content_equivalence", "coherence_A", "coherence_B", "fluency_A", "fluency_B", "confidence"):
        obj[key] = int(obj[key])
        if not 1 <= obj[key] <= 5:
            raise ValueError(f"{key} outside 1-5")
    obj["mechanical_artifact_A"] = bool(obj["mechanical_artifact_A"])
    obj["mechanical_artifact_B"] = bool(obj["mechanical_artifact_B"])
    return obj


def evaluate(client: OpenAI, task: dict[str, Any]) -> dict[str, Any]:
    last_error = ""
    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=task["evaluator"],
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": task["user"]}],
                temperature=0,
                max_tokens=350,
            )
            raw = response.choices[0].message.content or ""
            judgment = extract_json(raw)
            record = {k: v for k, v in task.items() if k != "user"}
            record["judgment"] = judgment
            record["raw_response"] = raw
            return record
        except Exception as exc:  # API and schema failures are retried together.
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(2 ** attempt)
    raise RuntimeError(f"{task['task_id']}: {last_error}")


def bootstrap_ci(values: list[float], repeats: int, seed: int) -> tuple[float, float, float]:
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=float)
    for start in range(0, repeats, 500):
        stop = min(start + 500, repeats)
        idx = rng.integers(0, len(x), size=(stop - start, len(x)))
        means[start:stop] = x[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(x.mean()), float(lo), float(hi)


def decode(record: dict[str, Any]) -> dict[str, Any]:
    j = record["judgment"]
    plus = record["plus_label"]
    minus = "B" if plus == "A" else "A"
    preference = j["lexical_preference"]
    signed = 0 if preference == "tie" else (1 if preference == plus else -1)
    return {
        **{k: record[k] for k in ("target_model", "prompt_index", "evaluator")},
        "signed_preference": signed,
        "plus_preferred": int(preference == plus),
        "minus_preferred": int(preference == minus),
        "tie": int(preference == "tie"),
        "salience": j["lexical_salience"],
        "content_equivalence": j["content_equivalence"],
        "coherence_delta": j[f"coherence_{plus}"] - j[f"coherence_{minus}"],
        "fluency_delta": j[f"fluency_{plus}"] - j[f"fluency_{minus}"],
        "plus_artifact": int(j[f"mechanical_artifact_{plus}"]),
        "minus_artifact": int(j[f"mechanical_artifact_{minus}"]),
        "confidence": j["confidence"],
    }


def summarize(records: list[dict[str, Any]], repeats: int) -> dict[str, Any]:
    decoded = [decode(r) for r in records]
    summary: dict[str, Any] = {"n_records": len(records), "groups": {}}
    groups = [(m, e) for m in INPUTS for e in sorted({r["evaluator"] for r in records if r["target_model"] == m})]
    groups += [(m, "pooled") for m in INPUTS]
    for model, evaluator in groups:
        rows = [r for r in decoded if r["target_model"] == model and (evaluator == "pooled" or r["evaluator"] == evaluator)]
        if not rows:
            continue
        # Pool evaluator votes within each prompt before prompt-level inference.
        prompt_values: dict[int, list[float]] = {}
        for row in rows:
            prompt_values.setdefault(row["prompt_index"], []).append(row["signed_preference"])
        prompt_means = [float(np.mean(v)) for v in prompt_values.values()]
        mean, lo, hi = bootstrap_ci(prompt_means, repeats, stable_int(f"{model}|{evaluator}") % (2**32))
        summary["groups"][f"{model}|{evaluator}"] = {
            "n_prompts": len(prompt_values),
            "signed_preference_mean": mean,
            "signed_preference_ci": [lo, hi],
            "plus_preference_rate": float(np.mean([r["plus_preferred"] for r in rows])),
            "minus_preference_rate": float(np.mean([r["minus_preferred"] for r in rows])),
            "tie_rate": float(np.mean([r["tie"] for r in rows])),
            "mean_salience": float(np.mean([r["salience"] for r in rows])),
            "mean_content_equivalence": float(np.mean([r["content_equivalence"] for r in rows])),
            "mean_coherence_delta": float(np.mean([r["coherence_delta"] for r in rows])),
            "mean_fluency_delta": float(np.mean([r["fluency_delta"] for r in rows])),
            "plus_artifact_rate": float(np.mean([r["plus_artifact"] for r in rows])),
            "minus_artifact_rate": float(np.mean([r["minus_artifact"] for r in rows])),
            "mean_confidence": float(np.mean([r["confidence"] for r in rows])),
        }
    return summary


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    lines = [
        "# Blind external-model evaluation", "",
        "The evaluators saw randomized A/B continuations without model names, intervention signs, or surface scores. One predeclared seed was used per prompt.", "",
        "| Target | Evaluator | n | Signed preference [95% CI] | + preferred | Tie | Content eq. | Coherence delta | Fluency delta | Artifact + / - |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, row in summary["groups"].items():
        model, evaluator = key.split("|", 1)
        ci = row["signed_preference_ci"]
        lines.append(
            f"| {model} | {evaluator} | {row['n_prompts']} | {row['signed_preference_mean']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | "
            f"{row['plus_preference_rate']:.3f} | {row['tie_rate']:.3f} | {row['mean_content_equivalence']:.2f} | "
            f"{row['mean_coherence_delta']:.3f} | {row['mean_fluency_delta']:.3f} | "
            f"{row['plus_artifact_rate']:.3f} / {row['minus_artifact_rate']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    evaluators = [x.strip() for x in args.evaluators.split(",") if x.strip()]
    api_key = os.environ.get("API_KEY")
    base_url = os.environ.get("API_BASE_URL")
    if not api_key or not base_url:
        raise SystemExit("Set API_KEY/API_BASE_URL (or OpenAI equivalents).")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_dir / "blind_judgments.jsonl"
    tasks = make_tasks(evaluators, args.limit)
    completed: dict[str, dict[str, Any]] = {}
    if raw_path.exists():
        for line in raw_path.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                completed[record["task_id"]] = record
    pending = [t for t in tasks if t["task_id"] not in completed]
    print(f"tasks={len(tasks)} completed={len(completed)} pending={len(pending)}", flush=True)
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=120, max_retries=0)
    lock = threading.Lock()
    random.Random(8236).shuffle(pending)
    with raw_path.open("a") as handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate, client, task): task for task in pending}
        for number, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            with lock:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                completed[record["task_id"]] = record
            if number % 20 == 0 or number == len(pending):
                print(f"finished {number}/{len(pending)} new tasks", flush=True)
    records = [completed[t["task_id"]] for t in tasks]
    summary = summarize(records, args.bootstrap)
    summary.update({"evaluators": evaluators, "input_files": {k: str(v) for k, v in INPUTS.items()}, "seed_rule": "(prompt_index - 34) mod 3", "randomization_seed": "sha256(target_model|prompt_index|seed_index)"})
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_markdown(summary, args.out_dir / "summary.md")
    print((args.out_dir / "summary.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
