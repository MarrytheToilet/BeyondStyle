#!/usr/bin/env python3
"""Evaluate steered LLM on standard benchmarks via lm-eval.

Supports three modes:
  1. Single run:  test one (layer, alpha, denoise) combination
  2. Symmetric contrast: compare β=-1 and β=+1 at a fixed layer
  3. Diagnostic grid: test explicitly requested combinations

The model is loaded ONCE; steering hooks are swapped between runs.
Each run is saved as a separate JSON file with structured naming.
Already-completed runs are automatically skipped (resume-friendly).

Output structure:
  outputs/{language}/04_lmeval/{model_id}/proposition_aligned/{representation}/
  ├── baseline.json                       # α=0.0 (always run first, shared)
  ├── nullspace_L14_a+1.0.json           # individual steered runs
  ├── nullspace_L14_a-1.0.json
  ├── raw/                                # full lm-eval output per run
  │   ├── baseline_raw.json
  │   ├── nullspace_L14_a+1.0_raw.json
  │   └── ...
  └── summary_latest.json                # aggregated comparison

Usage:
    # Quick sanity check
    python scripts/eval_lm_benchmark.py --alpha 0.0 1.0 --tasks wikitext --limit 10

    # Formal source-direction guardrail
    python scripts/eval_lm_benchmark.py --alpha 0.0 -1.0 1.0

    # Multi-layer grid sweep
    python scripts/eval_lm_benchmark.py --mode grid \\
        --layers 10 14 16 20 --alpha 0.0 -1.0 1.0 --denoise nullspace raw

    # Resume interrupted grid (skips existing results)
    python scripts/eval_lm_benchmark.py --mode grid --resume

    # Test specific task group
    python scripts/eval_lm_benchmark.py --alpha 0.0 1.0 --task-group reasoning
"""

import argparse
import gc
import json
import math
import re
import statistics
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
    get_model_path, get_model_id, get_output_base, get_steering_dir,
    get_eval_dir as _get_eval_dir,
)
from src.runtime import (
    configure_torch_runtime,
    resolve_attention_backend,
    resolve_torch_dtype,
)
from src.steering import load_calibrated_mean_diff


# ── Benchmark task registry ─────────────────────────────────
SCIENCE_PHYSICAL_WORLD_TASKS = [
    "piqa",
    "openbookqa",
    "sciq",
    "arc_challenge",
]

HUMAN_SOCIAL_WORLD_TASKS = [
    "social_iqa",
    "copa",
    "winogrande",
]

LANGUAGE_READING_CONTROL_TASKS = [
    "wic",
    "rte",
    "boolq",
]

FAST_TASKS = (
    SCIENCE_PHYSICAL_WORLD_TASKS
    + HUMAN_SOCIAL_WORLD_TASKS
    + LANGUAGE_READING_CONTROL_TASKS
)

TASK_GROUPS = {
    "fast": FAST_TASKS,
    "science_physical_world": SCIENCE_PHYSICAL_WORLD_TASKS,
    "human_social_world": HUMAN_SOCIAL_WORLD_TASKS,
    "language_reading_control": LANGUAGE_READING_CONTROL_TASKS,
    "reasoning": ["mmlu", "arc_easy", "arc_challenge", "openbookqa"],
    "commonsense": ["hellaswag", "winogrande", "piqa", "boolq"],
    "reading": ["race"],
    "perplexity": ["wikitext"],
}

TASK_CATEGORIES = {
    **{
        task: "science_physical_world"
        for task in SCIENCE_PHYSICAL_WORLD_TASKS
    },
    **{
        task: "human_social_world"
        for task in HUMAN_SOCIAL_WORLD_TASKS
    },
    **{
        task: "language_reading_control"
        for task in LANGUAGE_READING_CONTROL_TASKS
    },
}

ALL_TASKS = [
    "mmlu",             # Massive Multitask Language Understanding
    "hellaswag",        # Commonsense NLI
    "arc_easy",         # Science QA (easy)
    "arc_challenge",    # Science QA (hard)
    "openbookqa",       # Open-domain science QA
    "winogrande",       # Coreference resolution
    "piqa",             # Physical intuition QA
    "social_iqa",       # Social commonsense QA
    "copa",             # Causal reasoning
    "wic",              # Word sense in context
    "rte",              # Textual entailment
    "sciq",             # Science QA
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
    "social_iqa": "social_iqa",
    "copa": "copa",
    "wic": "wic",
    "rte": "rte",
    "sciq": "sciq",
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
    "social_iqa": "acc",
    "copa": "acc",
    "wic": "acc",
    "rte": "acc",
    "sciq": "acc_norm",
    "boolq": "acc",
    "race": "acc",
    "wikitext": "word_perplexity",
}

DEFAULT_LAYERS = [10, 14, 16, 20]
DEFAULT_ALPHAS = [0.0, -1.0, 1.0]


# ── Utilities ────────────────────────────────────────────────

def resolve_steering_path(steer_dir: Path, denoise: str, layer: int) -> Path:
    if denoise == "raw":
        return steer_dir / "models" / f"mean_diff_layer{layer:02d}.pt"
    return steer_dir / "models" / f"mean_diff_{denoise}_layer{layer:02d}.pt"


def run_tag(denoise: str, layer: int, alpha: float) -> str:
    """Structured tag for a single run.

    baseline | nullspace_L14_a+1.0 | raw_L16_a-2.0
    """
    if alpha == 0.0:
        return "baseline"
    sign = "+" if alpha >= 0 else ""
    return f"{denoise}_L{layer:02d}_a{sign}{alpha}"


def get_eval_dir(
    model_id: str,
    language: str = "en",
    dataset_tag: str = "proposition_aligned",
    representation: str = "last_token",
) -> Path:
    return (
        _get_eval_dir(model_id, language=language)
        / dataset_tag
        / representation
    )


def select_robust_layer(path: Path, method: str) -> int:
    """Freeze the layer using worst unseen-generator AUC."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = []
    for layer_text, generator_results in payload["results"].items():
        aucs = [
            float(method_results[method]["auc"])
            for method_results in generator_results.values()
        ]
        candidates.append((
            min(aucs),
            sum(aucs) / len(aucs),
            -int(layer_text),
        ))
    if not candidates:
        raise RuntimeError(f"No layer results in {path}")
    return -max(candidates)[2]


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
             raw_results: dict, sample_diagnostics: dict | None = None):
    """Save structured summary + full raw lm-eval output."""
    eval_dir.mkdir(parents=True, exist_ok=True)

    record = {"tag": tag, **meta, "scores": scores,
              "timestamp": datetime.now().isoformat()}
    if sample_diagnostics is not None:
        record["sample_diagnostics"] = sample_diagnostics.get("summary", {})
        sample_dir = eval_dir / "samples"
        sample_dir.mkdir(parents=True, exist_ok=True)
        sample_path = sample_dir / f"{tag}_choice_margins.json"
        sample_path.write_text(
            json.dumps(
                sample_diagnostics.get("rows", {}),
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        record["sample_diagnostics_path"] = str(sample_path)
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


def run_exists(
    eval_dir: Path,
    tag: str,
    *,
    strength_unit: str | None = None,
    position_mode: str | None = None,
    tasks: list[str] | None = None,
    limit: int | None = None,
    num_fewshot: int | None = None,
) -> bool:
    path = eval_dir / f"{tag}.json"
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if tasks is not None and record.get("tasks") != tasks:
        return False
    if record.get("limit") != limit:
        return False
    if record.get("num_fewshot") != num_fewshot:
        return False
    if tag == "baseline":
        return True
    return record.get("strength_unit") == strength_unit and (
        record.get("position_mode") == position_mode
    )


def _first_numeric(value):
    """Find the first finite log-likelihood in an lm-eval response."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (list, tuple)):
        for item in value:
            number = _first_numeric(item)
            if number is not None:
                return number
    return None


def _choice_target_index(sample: dict, choice_count: int) -> int | None:
    """Recover the gold choice index across lm-eval task conventions."""
    doc = sample.get("doc")
    if isinstance(doc, dict):
        label = doc.get("label")
        if (
            isinstance(label, int)
            and not isinstance(label, bool)
            and 0 <= label < choice_count
        ):
            return label
        answer = doc.get("answer")
        if isinstance(answer, str) and answer.strip().isdigit():
            one_based = int(answer.strip())
            if 1 <= one_based <= choice_count:
                return one_based - 1
        if (
            isinstance(answer, int)
            and not isinstance(answer, bool)
            and 1 <= answer <= choice_count
        ):
            return answer - 1

    target = sample.get("target")
    if (
        isinstance(target, int)
        and not isinstance(target, bool)
        and 0 <= target < choice_count
    ):
        return target

    # Some multiple-choice tasks expose the gold continuation rather than its
    # index. Match it against the request continuation when it is unique.
    if isinstance(target, str):
        normalized_target = " ".join(target.split()).casefold()
        matches = []
        for index, arguments in enumerate(sample.get("arguments", [])):
            if not isinstance(arguments, (list, tuple)) or not arguments:
                continue
            continuation = arguments[-1]
            if not isinstance(continuation, str):
                continue
            normalized_continuation = " ".join(
                continuation.split()
            ).casefold()
            if normalized_continuation == normalized_target:
                matches.append(index)
        if len(matches) == 1:
            return matches[0]
    return None


def extract_choice_margin_diagnostics(results: dict) -> dict:
    """Keep compact paired sensitivity data without duplicating task text.

    Accuracy only changes when the target and a distractor exchange rank.
    The target-minus-best-distractor log-likelihood margin is continuous and
    therefore exposes smaller paired responses under steering.
    """
    compact_rows = {}
    summary = {}
    for task, samples in (results.get("samples") or {}).items():
        task_rows = []
        margins = []
        for sample in samples:
            response_scores = [
                _first_numeric(response)
                for response in sample.get("filtered_resps", [])
            ]
            if not response_scores or any(
                score is None for score in response_scores
            ):
                continue
            target = _choice_target_index(sample, len(response_scores))
            if target is None:
                continue
            target_score = float(response_scores[target])
            distractors = [
                float(score)
                for index, score in enumerate(response_scores)
                if index != target
            ]
            if not distractors:
                continue
            margin = target_score - max(distractors)
            margins.append(margin)
            task_rows.append({
                "doc_id": sample.get("doc_id"),
                "doc_hash": sample.get("doc_hash"),
                "target_index": target,
                "choice_loglikelihoods": response_scores,
                "target_vs_best_distractor_margin": margin,
                "acc": sample.get("acc"),
                "acc_norm": sample.get("acc_norm"),
            })
        compact_rows[task] = task_rows
        if margins:
            summary[task] = {
                "n": len(margins),
                "mean_target_margin": statistics.fmean(margins),
                "median_target_margin": statistics.median(margins),
                "positive_margin_rate": sum(
                    margin > 0 for margin in margins
                ) / len(margins),
            }
    return {"summary": summary, "rows": compact_rows}


# ── Core evaluation ──────────────────────────────────────────

def run_single_eval(
    hflm,
    steering_model,
    alpha,
    tasks,
    *,
    task_manager=None,
    strength_unit="projection_std",
    position_mode="all",
    num_fewshot=None,
    limit=None,
):
    """Run lm-eval with optional steering hooks on an already-loaded HFLM."""
    from src.evaluation.steered_model import apply_steering_hooks

    hook_handle = None
    if steering_model is not None and alpha != 0.0:
        hook_handle = apply_steering_hooks(
            hflm._model,
            steering_model,
            alpha=alpha,
            positions=position_mode,
            strength_unit=strength_unit,
        )

    task_list = [TASK_ALIASES.get(t, t) for t in tasks]
    kw = {}
    if num_fewshot is not None:
        kw["num_fewshot"] = num_fewshot
    if limit is not None:
        kw["limit"] = limit

    try:
        return lm_eval.simple_evaluate(
            model=hflm,
            tasks=task_list,
            task_manager=task_manager,
            confirm_run_unsafe_code=True,
            **kw,
        )
    finally:
        if hook_handle is not None:
            hook_handle.remove()


# ── Summary ──────────────────────────────────────────────────

def _stderr_key(metric: str) -> str:
    return f"{metric}_stderr"


def _parse_run_tag(tag: str):
    """Parse 'nullspace_L15_a+2.0' / 'baseline' / 'raw_L11_a-1.0' into
    (denoise, layer, alpha). Returns ('baseline', None, 0.0) for baseline."""
    if tag == "baseline":
        return ("baseline", None, 0.0)
    parts = tag.split("_")
    denoise = parts[0]
    layer = None
    alpha = None
    for p in parts[1:]:
        if p.startswith("L") and p[1:].isdigit():
            layer = int(p[1:])
        elif p.startswith("a"):
            try:
                alpha = float(p[1:])
            except ValueError:
                pass
    return (denoise, layer, alpha)


def _fmt_score(scores_for_task: dict, metric: str, decimals: int = 3) -> str:
    if not scores_for_task or metric not in scores_for_task:
        return "—"
    val = scores_for_task[metric]
    se = scores_for_task.get(_stderr_key(metric))
    if se is None:
        return f"{val:.{decimals}f}"
    return f"{val:.{decimals}f} ± {se:.{decimals}f}"


def write_markdown_summary(summary: dict, out_path: Path) -> Path:
    """Write a markdown report:
       1. Baseline row (one per task, with stderr).
       2. One table per (denoise method) × all layers × all α, columns = tasks.
       Each cell = "value ± stderr" (perplexity for wikitext, no stderr available).
    """
    runs = summary.get("runs", {})
    tasks = summary.get("tasks", [])
    if not runs or not tasks:
        return out_path

    metrics = {t: TASK_PRIMARY_METRIC.get(t, "acc") for t in tasks}

    # Group by denoise
    by_denoise: dict = {}
    baseline_scores = None
    for tag, data in runs.items():
        denoise, layer, alpha = _parse_run_tag(tag)
        if denoise == "baseline":
            baseline_scores = data.get("scores", {})
            continue
        by_denoise.setdefault(denoise, []).append((layer, alpha, tag, data))

    lines = []
    lines.append(f"# lm-eval steering summary")
    lines.append("")
    lines.append(f"- Generated at: `{summary.get('generated_at', '')}`")
    lines.append(f"- Tasks: {', '.join(tasks)}")
    categories = summary.get("task_categories", {})
    for category in (
        "science_physical_world",
        "human_social_world",
        "language_reading_control",
    ):
        members = [task for task in tasks if categories.get(task) == category]
        if members:
            lines.append(
                f"- `{category}`: {', '.join(members)}"
            )
    lines.append(f"- Cell format: `value ± stderr` (perplexity for `wikitext`, no stderr).")
    lines.append("")

    # Baseline row
    lines.append("## Baseline (no steering, $\\alpha{=}0$)")
    lines.append("")
    header = "| Task | Metric | Score |"
    sep = "| --- | --- | --- |"
    lines.append(header)
    lines.append(sep)
    if baseline_scores is not None:
        for t in tasks:
            m = metrics[t]
            lines.append(f"| {t} | {m} | {_fmt_score(baseline_scores.get(t, {}), m)} |")
    lines.append("")

    # Per-method tables
    for denoise in sorted(by_denoise.keys()):
        rows = sorted(
            by_denoise[denoise],
            key=lambda r: (r[0] if r[0] is not None else -1, r[1] if r[1] is not None else 0.0),
        )
        lines.append(f"## Denoise = `{denoise}`")
        lines.append("")
        header = "| Layer | α | " + " | ".join(f"{t}<br>({metrics[t]})" for t in tasks) + " |"
        sep = "| --- | --- | " + " | ".join(["---"] * len(tasks)) + " |"
        lines.append(header)
        lines.append(sep)
        for layer, alpha, tag, data in rows:
            scores = data.get("scores", {})
            cells = " | ".join(_fmt_score(scores.get(t, {}), metrics[t]) for t in tasks)
            alpha_str = f"{alpha:+.1f}" if alpha is not None else "—"
            lines.append(f"| {layer} | {alpha_str} | {cells} |")
        lines.append("")

    out_path.write_text("\n".join(lines))
    return out_path


def _wilcoxon_p(deltas: list) -> float:
    """Two-sided Wilcoxon signed-rank p-value testing H0: median(delta)=0.

    Returns NaN when fewer than 2 non-zero deltas (test is undefined). Uses
    SciPy when available; otherwise falls back to a small-sample exact
    permutation test over sign assignments (n ≤ 12).
    """
    import math
    nz = [float(d) for d in deltas if not math.isclose(d, 0.0)]
    n = len(nz)
    if n < 2:
        return float("nan")
    try:
        from scipy.stats import wilcoxon  # type: ignore
        # zero_method='wilcox' is the default; alternative='two-sided' is default
        res = wilcoxon(nz, zero_method="wilcox", alternative="two-sided",
                       mode="exact" if n <= 25 else "approx")
        return float(res.pvalue)
    except Exception:
        # Exact sign-flip permutation on absolute ranks (small n)
        from itertools import product
        ranks = list(range(1, n + 1))
        signs = [1 if v > 0 else -1 for v in sorted(nz, key=abs)]
        # observed signed-rank sum
        obs = sum(s * r for s, r in zip(signs, ranks))
        count = 0
        total = 0
        for combo in product([-1, 1], repeat=n):
            total += 1
            stat = sum(s * r for s, r in zip(combo, ranks))
            if abs(stat) >= abs(obs):
                count += 1
        return count / total


def _sig_marker(p: float) -> str:
    """Return significance marker for a p-value."""
    import math
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""




def write_steering_grid_summary(summary: dict, out_path: Path) -> Path:
    """Write a compact (denoise × layer × α) grid table.

    For every steered run we compute the per-task Δ relative to baseline on the
    nine accuracy-based tasks (excluding ``wikitext``), then report:

      * #↑/#↓        — number of accuracy tasks improved / declined
      * mean Δacc    — mean change in percentage points
      * SD Δacc      — cross-task standard deviation
      * best/worst   — most positive / most negative single-task Δ
      * Δppl wikitext — perplexity drop (positive = lower perplexity)
      * p (Wilcoxon) — two-sided signed-rank test on the 9 paired Δ values
      * Random vec. ΔAcc — mean Δacc of a same-norm random direction at the
        same (layer, α), if available (tag prefix ``random``).
    """
    runs = summary.get("runs", {})
    tasks = summary.get("tasks", [])
    if not runs or not tasks:
        return out_path

    metrics = {t: TASK_PRIMARY_METRIC.get(t, "acc") for t in tasks}
    acc_tasks = [t for t in tasks if t != "wikitext"]
    has_wt = "wikitext" in tasks

    baseline = runs.get("baseline", {}).get("scores", {})
    if not baseline:
        return out_path

    def _val(scores: dict, task: str):
        m = metrics[task]
        v = scores.get(task, {}).get(m)
        return float(v) if v is not None else None

    def _deltas(scores: dict) -> list:
        out = []
        for t in acc_tasks:
            v = _val(scores, t)
            b = _val(baseline, t)
            if v is None or b is None:
                continue
            out.append((v - b) * 100.0)  # to percentage points
        return out

    def _ppl_delta(scores: dict):
        if not has_wt:
            return None
        v = _val(scores, "wikitext")
        b = _val(baseline, "wikitext")
        if v is None or b is None:
            return None
        return b - v  # positive = perplexity dropped

    # Group runs
    grouped: dict = {}        # (denoise, layer, alpha) -> scores
    random_grouped: dict = {} # (layer, alpha) -> [scores...]
    for tag, data in runs.items():
        if tag == "baseline":
            continue
        denoise, layer, alpha = _parse_run_tag(tag)
        if denoise == "baseline":
            continue
        scores = data.get("scores", {})
        if denoise.lower().startswith("random"):
            random_grouped.setdefault((layer, alpha), []).append(scores)
        else:
            grouped[(denoise, layer, alpha)] = scores

    if not grouped:
        return out_path

    # Pooled-across-layers Wilcoxon: for each (denoise, alpha) collect per-task
    # Δ across every layer to get a single, well-powered significance estimate
    # (n = #tasks × #layers ≈ 27). The same p is shown on each of the layer
    # rows that share the (estimator, α) — it summarises the directional effect.
    pooled_p: dict = {}
    pooled_n: dict = {}
    pooled_bucket: dict = {}
    for (denoise, _layer, alpha), scores in grouped.items():
        d = _deltas(scores)
        if d:
            pooled_bucket.setdefault((denoise, alpha), []).extend(d)
    for key, vec in pooled_bucket.items():
        pooled_p[key] = _wilcoxon_p(vec)
        pooled_n[key] = len(vec)

    # Compute per-cell stats
    rows = []
    for (denoise, layer, alpha), scores in grouped.items():
        deltas = _deltas(scores)
        if not deltas:
            continue
        n_up = sum(1 for d in deltas if d > 0)
        n_dn = sum(1 for d in deltas if d < 0)
        mean_d = sum(deltas) / len(deltas)
        if len(deltas) > 1:
            mu = mean_d
            sd_d = (sum((d - mu) ** 2 for d in deltas) / (len(deltas) - 1)) ** 0.5
        else:
            sd_d = 0.0
        best = max(deltas)
        worst = min(deltas)
        p = pooled_p.get((denoise, alpha))
        n_pool = pooled_n.get((denoise, alpha), 0)
        ppl_d = _ppl_delta(scores)

        rand_mean = None
        rand_pool = random_grouped.get((layer, alpha))
        if rand_pool:
            rand_deltas = []
            for sc in rand_pool:
                rd = _deltas(sc)
                if rd:
                    rand_deltas.extend(rd)
            if rand_deltas:
                rand_mean = sum(rand_deltas) / len(rand_deltas)

        rows.append({
            "denoise": denoise, "layer": layer, "alpha": alpha,
            "n_up": n_up, "n_dn": n_dn,
            "mean": mean_d, "sd": sd_d,
            "best": best, "worst": worst,
            "p": p, "n_pool": n_pool, "ppl_delta": ppl_d,
            "random_mean": rand_mean,
        })

    rows.sort(key=lambda r: (r["denoise"],
                             r["layer"] if r["layer"] is not None else -1,
                             r["alpha"] if r["alpha"] is not None else 0.0))

    def _fmt_pct(v, sign=True):
        if v is None:
            return "—"
        return (f"{v:+.2f}" if sign else f"{v:.2f}") + "%"

    def _fmt_p(p):
        import math
        if p is None or (isinstance(p, float) and math.isnan(p)):
            return "—"
        if p < 1e-3:
            return f"<0.001{_sig_marker(p)}"
        return f"{p:.3f}{_sig_marker(p)}"

    lines = [
        "## Steering grid summary (Δ vs. baseline, accuracy tasks only)",
        "",
        f"- Generated at: `{summary.get('generated_at', '')}`",
        f"- Δ values are in **percentage points**, computed across "
        f"{len(acc_tasks)} accuracy tasks: {', '.join(acc_tasks)}.",
        f"- p-value: two-sided Wilcoxon signed-rank test on Δacc pooled "
        f"across all layers for each (estimator, α) cell "
        f"(n = {len(acc_tasks)} tasks × #layers); H0: median Δ = 0. "
        f"`*` p<0.05, `**` p<0.01, `***` p<0.001. The same p is shown on "
        f"every layer row that shares the (estimator, α).",
        "- ΔPPL (wikitext) = baseline_ppl − steered_ppl; positive ⇒ perplexity dropped.",
        "- Random vec. = mean Δacc of same-norm random direction at the same "
        "(layer, α), averaged across runs with tag prefix `random_`. `—` if absent.",
        "",
        "| Estimator | Layer | α | #↑/#↓ | Mean Δacc | SD Δacc | Best | Worst | p (pooled) | Random vec. ΔAcc | ΔPPL (wikitext) |",
        "|:---|---:|---:|:---:|---:|---:|---:|---:|:---:|---:|---:|",
    ]
    for r in rows:
        ppl_str = "—" if r["ppl_delta"] is None else f"{r['ppl_delta']:+.3f}"
        rand_str = _fmt_pct(r["random_mean"]) if r["random_mean"] is not None else "—"
        lines.append(
            f"| {r['denoise']} | {r['layer']} | {r['alpha']:+.1f} "
            f"| {r['n_up']}/{r['n_dn']} "
            f"| {_fmt_pct(r['mean'])} | {_fmt_pct(r['sd'], sign=False)} "
            f"| {_fmt_pct(r['best'])} | {_fmt_pct(r['worst'])} "
            f"| {_fmt_p(r['p'])} "
            f"| {rand_str} "
            f"| {ppl_str} |"
        )
    lines.append("")

    # Append to existing markdown if present (replacing any old grid section).
    existing = out_path.read_text() if out_path.exists() else ""
    marker = "## Steering grid summary"
    if existing and marker in existing:
        existing = existing.split(marker)[0].rstrip()
    payload = (existing.rstrip() + "\n\n" + "\n".join(lines)
               if existing else "\n".join(lines))
    out_path.write_text(payload)
    return out_path



def build_summary(
    eval_dir: Path,
    tasks: list,
    *,
    strength_unit: str | None = None,
    position_mode: str | None = None,
) -> dict:
    """Aggregate all run results into a comparison summary."""
    summary = {
        "generated_at": datetime.now().isoformat(),
        "tasks": tasks,
        "task_categories": {
            task: TASK_CATEGORIES.get(task, "other")
            for task in tasks
        },
        "runs": {},
        "comparison": {},
    }

    for f in sorted(eval_dir.glob("*.json")):
        if f.name.startswith("summary"):
            continue
        try:
            data = json.loads(f.read_text())
            tag = data.get("tag", f.stem)
            if tag != "baseline" and strength_unit is not None:
                if (
                    data.get("strength_unit") != strength_unit
                    or data.get("position_mode") != position_mode
                ):
                    continue
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
  %(prog)s --alpha 0.0 -1.0 1.0
  %(prog)s --mode grid --layers 10 14 16 20 --denoise nullspace raw
  %(prog)s --mode grid --resume
  %(prog)s --task-group reasoning --alpha 0.0 1.0
""",
    )

    parser.add_argument("--mode", choices=["single", "grid"], default="single",
                        help="single: one layer. grid: sweep layers × alphas × denoise")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--language", choices=["en", "zh"], default="en")
    parser.add_argument("--dataset-tag", default="proposition_aligned")
    parser.add_argument(
        "--experiment-tag",
        default=None,
        help=(
            "Optional output subdirectory for a distinct sample cap or task set."
        ),
    )
    parser.add_argument(
        "--representation",
        choices=["last_token", "mean_pool"],
        default="last_token",
    )
    parser.add_argument(
        "--batch-size",
        default="auto",
        help="Integer, 'auto', or 'auto:N' for lm-eval OOM-aware sizing.",
    )
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
    )
    parser.add_argument(
        "--attention-backend",
        choices=["auto", "sdpa", "flash_attention_2", "eager"],
        default="auto",
    )
    parser.add_argument("--no-tf32", action="store_true")

    parser.add_argument("--denoise", type=str, nargs="+", default=["nullspace"],
                        choices=["raw", "nullspace"])
    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Explicit layer override; otherwise use frozen robust selection.",
    )
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="Layers for grid mode")
    parser.add_argument("--alpha", type=float, nargs="+", default=None,
                        help="Signed steering strengths (β under the formal unit)")
    parser.add_argument(
        "--strength-unit",
        choices=["projection_std", "absolute"],
        default="projection_std",
    )

    parser.add_argument("--tasks", type=str, nargs="+", default=None)
    parser.add_argument("--task-group", type=str, default=None,
        choices=list(TASK_GROUPS.keys()) + ["all"])

    parser.add_argument("--num-fewshot", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed runs")
    parser.add_argument("--no-summary", action="store_true")

    args = parser.parse_args()
    if args.experiment_tag and not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]*",
        args.experiment_tag,
    ):
        raise ValueError(
            "--experiment-tag must contain only letters, digits, '.', '_', "
            "or '-' and must start with a letter or digit"
        )
    if args.batch_size.isdigit():
        args.batch_size = int(args.batch_size)
    elif not (
        args.batch_size == "auto"
        or (
            args.batch_size.startswith("auto:")
            and args.batch_size.split(":", 1)[1].isdigit()
        )
    ):
        raise ValueError("--batch-size must be an integer, auto, or auto:N")

    # ── Resolve configuration ──
    model_path = args.model or get_model_path()
    model_id = get_model_id(model_path)
    steer_dir = (
        get_steering_dir(
            model_id,
            dataset_tag=args.dataset_tag,
            language=args.language,
        )
        / args.representation
    )
    eval_dir = get_eval_dir(
        model_id,
        language=args.language,
        dataset_tag=args.dataset_tag,
        representation=args.representation,
    )
    if args.experiment_tag:
        eval_dir = eval_dir / args.experiment_tag
    eval_dir.mkdir(parents=True, exist_ok=True)
    robust_layer = None
    needs_robust_layer = (
        args.layer is None
        and not (args.mode == "grid" and args.layers)
    )
    if needs_robust_layer:
        selection_path = (
            get_output_base()
            / args.language
            / "02_separability"
            / model_id
            / args.dataset_tag
            / args.representation
            / "generator_holdout.json"
        )
        robust_layer = select_robust_layer(
            selection_path,
            args.denoise[0],
        )

    # Tasks
    if args.task_group:
        tasks = ALL_TASKS if args.task_group == "all" else TASK_GROUPS[args.task_group]
    elif args.tasks:
        tasks = args.tasks
    else:
        tasks = FAST_TASKS

    # Alphas
    alphas = args.alpha if args.alpha else DEFAULT_ALPHAS

    # Layers & denoise
    if args.mode == "grid":
        layers = args.layers or [args.layer or robust_layer]
        denoise_methods = args.denoise
    else:
        layers = [args.layer or robust_layer]
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
        if need_baseline and run_exists(
            eval_dir,
            "baseline",
            tasks=tasks,
            limit=args.limit,
            num_fewshot=args.num_fewshot,
        ):
            skip += 1
        skip += sum(
            1
            for run in run_plan
            if run_exists(
                eval_dir,
                run["tag"],
                strength_unit=args.strength_unit,
                position_mode="all",
                tasks=tasks,
                limit=args.limit,
                num_fewshot=args.num_fewshot,
            )
        )

    # ── Print plan ──
    print(f"\n{'='*70}")
    print(f"  lm-eval Benchmark")
    print(f"{'='*70}")
    print(f"  Model:       {model_id}")
    print(f"  Language:    {args.language}")
    print(f"  Dataset:     {args.dataset_tag}")
    print(f"  Repr:        {args.representation}")
    print(f"  Tasks ({len(tasks)}):  {tasks}")
    print(f"  Denoise:     {denoise_methods}")
    print(f"  Layers:      {layers}")
    print(f"  Alphas:      {sorted(set(alphas))}")
    print(f"  Unit:        {args.strength_unit}")
    print("  Positions:   all (sequence-likelihood intervention)")
    print(f"  Total runs:  {total}" +
          (f" ({skip} cached, {total-skip} to run)" if args.resume else ""))
    print(f"  Output:      {eval_dir}")
    print(f"  Limit:       {args.limit or 'full'}")
    print(f"{'='*70}\n")

    from lm_eval.tasks import TaskManager

    task_manager = TaskManager(include_path=ROOT / "lm_eval_tasks")

    if total - skip == 0 and args.resume:
        print("✅ All runs cached. Remove --resume to re-run.")
        if not args.no_summary:
            summary = build_summary(
                eval_dir,
                tasks,
                strength_unit=args.strength_unit,
                position_mode="all",
            )
            print_comparison(summary, tasks)
        return

    # ── Load model ONCE ──
    from src.evaluation.steered_model import load_hflm, _load_steering_model

    t0 = time.time()
    print("Loading model (one-time)...", flush=True)
    configure_torch_runtime(allow_tf32=not args.no_tf32)
    resolved_dtype = resolve_torch_dtype(args.torch_dtype, args.device)
    resolved_attention = resolve_attention_backend(args.attention_backend)
    hflm = load_hflm(
        pretrained=model_path,
        device=args.device,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        dtype=resolved_dtype,
        attn_implementation=resolved_attention,
        trust_remote_code=True,
    )
    print(f"Model loaded in {time.time()-t0:.1f}s\n", flush=True)

    done = 0

    # ── Baseline ──
    if need_baseline:
        tag = "baseline"
        if args.resume and run_exists(
            eval_dir,
            tag,
            tasks=tasks,
            limit=args.limit,
            num_fewshot=args.num_fewshot,
        ):
            print(f"  ⏭️  [{done+1}/{total}] {tag} (cached)")
        else:
            print(f"\n  {'─'*56}")
            print(f"  [{done+1}/{total}] BASELINE (α=0.0)")
            print(f"  {'─'*56}", flush=True)

            t1 = time.time()
            results = run_single_eval(hflm, None, 0.0, tasks,
                                      task_manager=task_manager,
                                      strength_unit=args.strength_unit,
                                      num_fewshot=args.num_fewshot,
                                      limit=args.limit)
            elapsed = time.time() - t1
            scores = extract_scores(results)
            sample_diagnostics = extract_choice_margin_diagnostics(results)

            meta = {"model": model_id, "alpha": 0.0, "denoise": None,
                    "layer": None, "tasks": tasks, "limit": args.limit,
                    "num_fewshot": args.num_fewshot,
                    "experiment_tag": args.experiment_tag,
                    "strength_unit": args.strength_unit,
                    "position_mode": "none",
                    "resolved_dtype": str(resolved_dtype),
                    "resolved_attention_backend": resolved_attention,
                    "elapsed_seconds": round(elapsed, 1)}
            save_run(
                eval_dir,
                tag,
                meta,
                scores,
                results,
                sample_diagnostics,
            )

            print(f"\n  ✅ Baseline ({elapsed:.0f}s):")
            for t, m in scores.items():
                print(f"     {t}: {', '.join(f'{k}={v:.4f}' for k,v in m.items())}")
        done += 1

    # ── Steered runs (grouped by denoise+layer to minimize vector loads) ──
    def gk(r):
        return (r["denoise"], r["layer"], r["steering_path"])

    for (denoise, layer, sp), grp in groupby(sorted(run_plan, key=gk), key=gk):
        grp = list(grp)
        states_path = (
            steer_dir
            / "states"
            / f"paired_states_layer{layer:02d}.pt"
        )
        sm, calibration = load_calibrated_mean_diff(
            Path(sp),
            states_path,
        )
        sm = sm.to(hflm.device)
        print(f"\n📦 Steering: {denoise} L{layer} "
              f"(dim={sm.get_direction().shape[0]})", flush=True)

        for ri in grp:
            alpha, tag = ri["alpha"], ri["tag"]
            done += 1

            if args.resume and run_exists(
                eval_dir,
                tag,
                strength_unit=args.strength_unit,
                position_mode="all",
                tasks=tasks,
                limit=args.limit,
                num_fewshot=args.num_fewshot,
            ):
                print(f"  ⏭️  [{done}/{total}] {tag} (cached)")
                continue

            print(f"\n  {'─'*56}")
            print(f"  [{done}/{total}] {tag}  (α={alpha:+.1f})")
            print(f"  {'─'*56}", flush=True)

            t1 = time.time()
            results = run_single_eval(hflm, sm, alpha, tasks,
                                      task_manager=task_manager,
                                      strength_unit=args.strength_unit,
                                      position_mode="all",
                                      num_fewshot=args.num_fewshot,
                                      limit=args.limit)
            elapsed = time.time() - t1
            scores = extract_scores(results)
            sample_diagnostics = extract_choice_margin_diagnostics(results)

            meta = {"model": model_id, "alpha": alpha, "denoise": denoise,
                    "layer": layer, "steering_path": sp, "tasks": tasks,
                    "num_fewshot": args.num_fewshot,
                    "experiment_tag": args.experiment_tag,
                    "strength_unit": args.strength_unit,
                    "projection_std": calibration["projection_std"],
                    "calibration_source": calibration["source"],
                    "position_mode": "all",
                    "resolved_dtype": str(resolved_dtype),
                    "resolved_attention_backend": resolved_attention,
                    "limit": args.limit, "elapsed_seconds": round(elapsed, 1)}
            save_run(
                eval_dir,
                tag,
                meta,
                scores,
                results,
                sample_diagnostics,
            )

            print(f"\n  ✅ {tag} ({elapsed:.0f}s):")
            for t, m in scores.items():
                print(f"     {t}: {', '.join(f'{k}={v:.4f}' for k,v in m.items())}")

        del sm
        gc.collect()

    # ── Summary ──
    if not args.no_summary:
        summary = build_summary(
            eval_dir,
            tasks,
            strength_unit=args.strength_unit,
            position_mode="all",
        )

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        (eval_dir / f"summary_{ts}.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False))
        (eval_dir / "summary_latest.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False))

        md_path = write_markdown_summary(summary, eval_dir / f"summary_{ts}.md")
        write_markdown_summary(summary, eval_dir / "summary_latest.md")

        write_steering_grid_summary(summary, eval_dir / f"summary_{ts}.md")
        write_steering_grid_summary(summary, eval_dir / "summary_latest.md")

        print(f"\n📊 Summary → {eval_dir / f'summary_{ts}.json'}")
        print(f"📝 Markdown → {md_path}")
        print_comparison(summary, tasks)

    print(f"\n✅ Done. {total} runs, results in {eval_dir}/")


if __name__ == "__main__":
    main()
