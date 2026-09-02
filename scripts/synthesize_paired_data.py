#!/usr/bin/env python3
"""Build the canonical proposition-aligned human/AI continuation dataset.

For every accepted pair:

1. A planner sees the prefix and human continuation and extracts a wording-
   independent proposition plan.
2. A different generator sees only the prefix and the proposition plan.
3. A third model rejects clear losses of core content, major unsupported
   claims, or contradictions.

Generated prose that fails a deterministic gate or the independent judge is
rejected and regenerated.  Text is never repaired.  Per-slot work files make
the process resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import statistics
import sys
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import (
    PROJECT_ROOT,
    get_api_base_url,
    get_api_key,
    get_api_models,
)
from src.data.proposition_alignment import (
    continuation_quality,
    extract_json_object,
    validate_content_plan,
    validate_relaxed_judgment,
)
from src.generation.backends import APIGenerator


LOGGER = logging.getLogger("synthesize_paired_data")
SCHEMA_NAME = "proposition_aligned_pair"
PROTOCOL_NAME = "proposition_aligned"
PREFIX_BANK_NAME = "balanced_natural"
VALIDATOR_NAME = "simple_core_content"
PROMPT_FAMILIES = ("ordered_continuation",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--language",
        choices=("en", "zh", "both"),
        default="both",
    )
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated override; default: API_MODELS/API_MODEL.",
    )
    parser.add_argument("--seed", type=int, default=8236)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--planner-retries", type=int, default=3)
    parser.add_argument("--generation-retries", type=int, default=3)
    parser.add_argument("--judge-retries", type=int, default=2)
    parser.add_argument("--candidate-replacements", type=int, default=5)
    parser.add_argument(
        "--smoke-models",
        action="store_true",
        help="Call every configured model once and exit.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_integer(*parts: object) -> int:
    payload = "\x1f".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )
    temporary.replace(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def balanced_sequence(
    values: list[str] | tuple[str, ...],
    count: int,
    *,
    seed: int,
    label: str,
) -> list[str]:
    if not values:
        raise ValueError(f"cannot balance an empty sequence for {label}")
    base, remainder = divmod(count, len(values))
    sequence: list[str] = []
    for index, value in enumerate(values):
        sequence.extend([value] * (base + (index < remainder)))
    random.Random(stable_integer(seed, label)).shuffle(sequence)
    return sequence


def balanced_cross_sequence(
    left_values: list[str] | tuple[str, ...],
    right_values: list[str] | tuple[str, ...],
    count: int,
    *,
    seed: int,
    label: str,
) -> list[tuple[str, str]]:
    """Balance both margins and every left×right contingency cell."""
    if not left_values or not right_values:
        raise ValueError(f"cannot cross-balance empty values for {label}")
    left_targets = Counter(
        balanced_sequence(
            left_values,
            count,
            seed=seed,
            label=f"{label}:left_margin",
        )
    )
    right_targets = Counter(
        balanced_sequence(
            right_values,
            count,
            seed=seed,
            label=f"{label}:right_margin",
        )
    )
    base = count // (len(left_values) * len(right_values))
    pairs = [
        (left, right)
        for left in left_values
        for right in right_values
        for _ in range(base)
    ]
    left_residual = [
        left
        for left in left_values
        for _ in range(
            left_targets[left] - base * len(right_values)
        )
    ]
    right_residual = [
        right
        for right in right_values
        for _ in range(
            right_targets[right] - base * len(left_values)
        )
    ]
    if len(left_residual) != len(right_residual):
        raise RuntimeError(f"{label}: inconsistent residual margins")

    rng = random.Random(stable_integer(seed, label, "residual"))
    residual_pairs: list[tuple[str, str]] | None = None
    for _ in range(10_000):
        candidate_left = list(left_residual)
        candidate_right = list(right_residual)
        rng.shuffle(candidate_left)
        rng.shuffle(candidate_right)
        candidate = list(zip(candidate_left, candidate_right))
        if len(candidate) == len(set(candidate)):
            residual_pairs = candidate
            break
    if residual_pairs is None:
        raise RuntimeError(
            f"{label}: could not construct unit-spread contingency table"
        )
    pairs.extend(residual_pairs)
    if len(pairs) != count:
        raise RuntimeError(f"{label}: cross-balanced sequence has wrong size")
    rng.shuffle(pairs)
    return pairs


def language_name(language: str) -> str:
    return "English" if language == "en" else "Chinese"


def planner_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    language = record["language"]
    if language == "en":
        system = """
You are a content-abstraction annotator. Extract a compact semantic plan from
a reference continuation. Record content only: no style, tone, quality,
rhetorical-device, authorship, or wording instructions.

Return exactly one JSON object with this schema:
{
  "language": "en",
  "propositions": ["3 to 10 atomic factual propositions"],
  "required_entities": ["entities or numbers needed to preserve reference"],
  "discourse_endpoint": "the last content point reached by the reference"
}

Make each proposition independently checkable. Cover the reference's central
claims, events, relations, quantities, and conclusion. Do not quote the
reference and do not copy long sentence fragments. Each required_entities item
must be a short name, date, number, or noun phrase rather than a description.
Do not add facts. JSON only.
""".strip()
        user = (
            "SHARED PREFIX:\n"
            f"{record['prefix']}\n\n"
            "REFERENCE HUMAN CONTINUATION:\n"
            f"{record['human_continuation']}"
        )
    else:
        system = """
你是内容抽象标注员。请从参考续文中抽取简洁的语义计划，只记录内容；不要记录文风、
语气、质量、修辞手段、作者身份或措辞要求。

只返回一个符合以下结构的 JSON 对象：
{
  "language": "zh",
  "propositions": ["3至10条可独立核验的原子事实命题"],
  "required_entities": ["保留参考内容所必需的实体、时间或数字"],
  "discourse_endpoint": "参考续文最终推进到的内容节点"
}

命题需要覆盖参考续文的核心主张、事件、关系、数量与结论。不要引用参考原文或照抄
长句。required_entities 的每一项只能是简短名称、日期、数字或名词短语，不能写成
描述句。不要添加事实。只能输出 JSON。
""".strip()
        user = (
            "共享前缀：\n"
            f"{record['prefix']}\n\n"
            "参考人类续文：\n"
            f"{record['human_continuation']}"
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def simple_generator_messages(
    record: dict[str, Any],
    plan: dict[str, Any],
    *,
    retry_feedback: str = "",
) -> list[dict[str, str]]:
    """One concise, model-neutral continuation prompt."""
    language = record["language"]
    target = record["continuation_units"]
    minimum = int(target * 0.60)
    maximum = int(target * 1.50)
    if language == "en":
        system = (
            "Continue the document naturally from the supplied prefix. Use "
            "the ordered content plan to preserve its core facts, entities, "
            "and chronology, but write them in your own words as natural "
            "prose. Avoid inventing factual details or adding commentary. "
            "Return only the continuation, without a title, list, explanation, "
            "or repeated prefix."
        )
        user = (
            f"ORDERED CONTENT PLAN:\n{numbered_plan(plan)}\n\n"
            f"LENGTH GUIDE: about {target} words; acceptable range "
            f"{minimum}-{maximum}.\n\n"
            "PREFIX — write only the continuation after its final character:\n"
            f"{record['prefix']}"
        )
        if retry_feedback:
            user += f"\n\nFor this retry: {retry_feedback}"
    else:
        system = (
            "请从给定前缀后自然续写。依据有序内容计划保留核心事实、实体和时间顺序；"
            "请用自己的语言组织成自然正文，避免编造计划外的事实细节或加入评论。"
            "只输出续文本身，不要标题、列表、解释或重复前缀。"
        )
        user = (
            f"有序内容计划：\n{numbered_plan(plan)}\n\n"
            f"长度参考：目标约{target}个中文文本单位，可接受范围{minimum}至"
            f"{maximum}。\n\n"
            "前缀——只写最后一个字符之后的续文：\n"
            f"{record['prefix']}"
        )
        if retry_feedback:
            user += f"\n\n本次重试请注意：{retry_feedback}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def relaxed_judge_messages(
    record: dict[str, Any],
    plan: dict[str, Any],
    continuation: str,
) -> list[dict[str, str]]:
    """Ask one judge to reject only clear, material content failures."""
    if record["language"] == "en":
        system = """
Judge only major semantic alignment, not style or wording. Minor omissions,
ordinary connective language, and harmless elaboration are acceptable. Fail
only when the candidate loses core content, adds a clear major factual claim
not supported by the prefix or plan, or creates a clear contradiction.

Return exactly one JSON object:
{
  "core_content_coverage": "complete, mostly_complete, or poor",
  "missing_core_facts": [],
  "major_unsupported_claims": [],
  "major_contradictions": [],
  "verdict": "pass or fail",
  "brief_reason": "one short explanation"
}
Use pass for complete or mostly_complete coverage when there is no major
unsupported claim or contradiction. JSON only.
""".strip()
        user = (
            f"PREFIX:\n{record['prefix']}\n\n"
            f"CONTENT PLAN:\n{numbered_plan(plan)}\n\n"
            f"CANDIDATE:\n{continuation}"
        )
    else:
        system = """
只判断重大的语义对齐问题，不评价文风或措辞。允许少量非核心遗漏、普通连接语和无害
展开。只有候选续文丢失核心内容、加入前缀和计划均不支持的明显重大事实，或产生明显
矛盾时才判定失败。

只返回一个 JSON 对象：
{
  "core_content_coverage": "complete、mostly_complete 或 poor",
  "missing_core_facts": [],
  "major_unsupported_claims": [],
  "major_contradictions": [],
  "verdict": "pass 或 fail",
  "brief_reason": "一句简短理由"
}
当覆盖为 complete 或 mostly_complete，且没有重大新增事实或矛盾时，判定为 pass。
只能输出 JSON。
""".strip()
        user = (
            f"前缀：\n{record['prefix']}\n\n"
            f"内容计划：\n{numbered_plan(plan)}\n\n"
            f"候选续文：\n{continuation}"
        )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def numbered_plan(plan: dict[str, Any]) -> str:
    lines = [
        f"{index}. {proposition}"
        for index, proposition in enumerate(plan["propositions"], start=1)
    ]
    lines.append(
        "Required entities: "
        + json.dumps(plan["required_entities"], ensure_ascii=False)
    )
    lines.append("Endpoint: " + plan["discourse_endpoint"])
    return "\n".join(lines)


def retry_feedback_from_quality(
    quality: dict[str, Any],
    language: str,
) -> str:
    issues = set(quality["issues"])
    messages = []
    if "continuation_length_out_of_bounds" in issues:
        if quality["units"] > quality["maximum_units"]:
            messages.append(
                (
                    "The prior unrecorded draft was too long. Write a "
                    "substantially shorter fresh draft within the stated range."
                )
                if language == "en"
                else "上一份未收录草稿过长。请重新写一份明显更短、严格处于规定范围内的正文。"
            )
        else:
            messages.append(
                (
                    "The prior unrecorded draft was too short. Write a longer "
                    "fresh draft within the stated range."
                )
                if language == "en"
                else "上一份未收录草稿过短。请重新写一份更长、严格处于规定范围内的正文。"
            )
    if "extreme_plan_serialization" in issues:
        messages.append(
            (
                "The prior unrecorded draft copied the semantic brief. "
                "Re-express every fact with new syntax and wording."
            )
            if language == "en"
            else "上一份未收录草稿直接照搬了语义简报。请用新的句法和措辞重新表达全部事实。"
        )
    if not messages:
        messages.append(
            (
                "The prior unrecorded draft failed a formal prose constraint. "
                "Write a fresh draft and follow every instruction exactly."
            )
            if language == "en"
            else "上一份未收录草稿未通过形式约束。请重新写作并严格遵守全部要求。"
        )
    return " ".join(messages)


class ApiPool:
    """One reusable client per configured model."""

    TRANSPORT_ATTEMPTS = 8
    RETRYABLE_ERRORS = {
        "APIConnectionError",
        "APITimeoutError",
        "RateLimitError",
        "InternalServerError",
    }

    def __init__(self, models: list[str], timeout: int):
        key = get_api_key()
        base_url = get_api_base_url()
        self.generators = {
            model: APIGenerator(
                api_key=key,
                model_name=model,
                base_url=base_url,
                timeout=timeout,
            )
            for model in models
        }
        self._locks = {model: threading.Lock() for model in models}

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int,
        temperature: float,
    ) -> str:
        # The SDK client is generally thread-safe.  The per-model lock also
        # avoids concurrent requests to endpoints with model-level limits.
        with self._locks[model]:
            for attempt in range(1, self.TRANSPORT_ATTEMPTS + 1):
                try:
                    return self.generators[model].chat(
                        messages,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                    )
                except Exception as exc:
                    retryable = (
                        type(exc).__name__ in self.RETRYABLE_ERRORS
                        or type(exc).__name__.endswith("ConnectionError")
                        or type(exc).__name__.endswith("TimeoutError")
                    )
                    if not retryable or attempt == self.TRANSPORT_ATTEMPTS:
                        raise
                    delay = min(60, 2 ** (attempt - 1))
                    LOGGER.warning(
                        "%s transport failure (%s), retrying in %ds "
                        "[%d/%d]",
                        model,
                        type(exc).__name__,
                        delay,
                        attempt,
                        self.TRANSPORT_ATTEMPTS,
                    )
                    time.sleep(delay)
        raise RuntimeError("unreachable API retry state")


def smoke_models(api: ApiPool, models: list[str]) -> None:
    LOGGER.info("Running one JSON smoke request against each model")
    for model in models:
        response = api.chat(
            model,
            [
                {
                    "role": "system",
                    "content": (
                        'Return only this JSON object: '
                        '{"status":"ok","language":"en"}'
                    ),
                },
                {"role": "user", "content": "Run the connectivity check."},
            ],
            max_new_tokens=512,
            temperature=0,
        )
        value = extract_json_object(response)
        if value.get("status") != "ok":
            raise RuntimeError(f"{model}: unexpected smoke response")
        LOGGER.info("Smoke passed: %s", model)


def role_models(
    generator_model: str,
    models: list[str],
) -> tuple[str, str]:
    index = models.index(generator_model)
    ordered_others = [
        models[(index + offset) % len(models)]
        for offset in range(1, len(models))
    ]
    # The initial smoke check showed that DeepSeek-V4-Flash repeatedly expands short
    # continuations beyond the plan cardinality and copies long reference
    # fragments.  It remains in the generator and judge pools, but is skipped
    # for the abstraction role.
    planner_candidates = [
        model
        for model in ordered_others
        if "deepseek" not in model.casefold()
    ]
    if not planner_candidates:
        planner_candidates = ordered_others
    planner_model = planner_candidates[0]
    judge_models = [
        model for model in ordered_others if model != planner_model
    ]
    if not judge_models:
        raise ValueError(
            "Planner, generator, and judge must use distinct models"
        )
    return planner_model, judge_models[0]


def load_bank(language: str) -> tuple[Path, dict, list[dict]]:
    bank_dir = (
        PROJECT_ROOT / "data" / "prefixes" / language / PREFIX_BANK_NAME
    )
    samples_path = bank_dir / "samples.jsonl"
    manifest_path = bank_dir / "manifest.json"
    if not samples_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"missing prefix bank: {bank_dir}")
    return (
        bank_dir,
        json.loads(manifest_path.read_text(encoding="utf-8")),
        load_jsonl(samples_path),
    )


def build_slots(
    records: list[dict[str, Any]],
    *,
    language: str,
    count: int,
    models: list[str],
    seed: int,
    replacements: int,
) -> list[dict[str, Any]]:
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_source[record["source"]].append(record)
    sources = sorted(by_source)
    source_model_sequence = balanced_cross_sequence(
        sources,
        models,
        count,
        seed=seed,
        label=f"{language}:source_by_model",
    )
    prompt_sequence = balanced_sequence(
        PROMPT_FAMILIES,
        count,
        seed=seed,
        label=f"{language}:prompts",
    )

    source_sequence = [item[0] for item in source_model_sequence]
    model_sequence = [item[1] for item in source_model_sequence]
    quotas = Counter(source_sequence)
    occurrences: Counter[str] = Counter()
    pools: dict[str, list[dict[str, Any]]] = {}
    for source, source_records in by_source.items():
        pools[source] = sorted(
            source_records,
            key=lambda record: stable_integer(
                seed,
                language,
                source,
                record["sample_id"],
            ),
        )
        required = quotas[source] * replacements
        if len(pools[source]) < required:
            raise RuntimeError(
                f"{source}: need {required} candidates for replacement "
                f"queues, found {len(pools[source])}"
            )

    slots = []
    for slot_index in range(count):
        source = source_sequence[slot_index]
        source_occurrence = occurrences[source]
        occurrences[source] += 1
        quota = quotas[source]
        candidates = pools[source][source_occurrence::quota][:replacements]
        generator_model = model_sequence[slot_index]
        planner_model, judge_model = role_models(generator_model, models)
        slots.append(
            {
                "slot_index": slot_index,
                "slot_id": f"{language}_slot_{slot_index:05d}",
                "source": source,
                "candidate_sample_ids": [
                    candidate["sample_id"] for candidate in candidates
                ],
                "generator_model": generator_model,
                "planner_model": planner_model,
                "judge_model": judge_model,
                "prompt_family": prompt_sequence[slot_index],
            }
        )
    return slots


def run_planner(
    api: ApiPool,
    record: dict[str, Any],
    *,
    model: str,
    retries: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    attempts = []
    for attempt_index in range(1, retries + 1):
        try:
            raw = api.chat(
                model,
                planner_messages(record),
                max_new_tokens=1200,
                temperature=0.2,
            )
            parsed = extract_json_object(raw)
            plan, issues = validate_content_plan(
                parsed,
                language=record["language"],
                human_continuation=record["human_continuation"],
            )
            attempts.append(
                {
                    "attempt": attempt_index,
                    "raw_response": raw,
                    "issues": issues,
                }
            )
            if plan is not None:
                return plan, attempts
        except Exception as exc:
            attempts.append(
                {
                    "attempt": attempt_index,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return None, attempts


def run_judge(
    api: ApiPool,
    record: dict[str, Any],
    plan: dict[str, Any],
    continuation: str,
    *,
    model: str,
    retries: int,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    attempts = []
    for attempt_index in range(1, retries + 1):
        try:
            raw = api.chat(
                model,
                relaxed_judge_messages(record, plan, continuation),
                max_new_tokens=700,
                temperature=0,
            )
            parsed = extract_json_object(raw)
            judgment, issues = validate_relaxed_judgment(parsed)
            attempts.append(
                {
                    "attempt": attempt_index,
                    "raw_response": raw,
                    "issues": issues,
                }
            )
            # A well-formed fail is substantive, not a parse failure.  Do not
            # ask the same judge to relabel it.
            if judgment is not None:
                return judgment, attempts
            if not any(
                issue.startswith("invalid_") for issue in issues
            ):
                return None, attempts
        except Exception as exc:
            attempts.append(
                {
                    "attempt": attempt_index,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
    return None, attempts


def make_pair(
    record: dict[str, Any],
    slot: dict[str, Any],
    plan: dict[str, Any],
    continuation: str,
    quality: dict[str, Any],
    judgment: dict[str, Any],
    *,
    fact_order: list[int],
    generation_attempt: int,
    candidate_index: int,
    planner_attempt_count: int,
    judge_attempt_count: int,
) -> dict[str, Any]:
    language = record["language"]
    return {
        "schema": SCHEMA_NAME,
        "pair_id": (
            f"{language}_{PROTOCOL_NAME}_{slot['slot_index']:05d}"
        ),
        "slot_id": slot["slot_id"],
        "sample_id": record["sample_id"],
        "language": language,
        "domain": record["domain"],
        "source": record["source"],
        "source_document": record["source_document"],
        "prefix": record["prefix"],
        "human_continuation": record["human_continuation"],
        "ai_continuation": continuation,
        "lengths": {
            "length_unit": record["length_unit"],
            "prefix_units": record["prefix_units"],
            "human_continuation_units": record["continuation_units"],
            "ai_continuation_units": quality["units"],
            "ai_to_human_ratio": quality["length_ratio"],
        },
        "content_control": {
            "method": "proposition_plan",
            "planner_model": slot["planner_model"],
            "plan": plan,
            "generation_fact_order": [
                index + 1 for index in fact_order
            ],
            "generation_facts_presented_in_reference_order": True,
            "planner_attempt_count": planner_attempt_count,
            "generator_never_received_human_continuation": True,
        },
        "generation": {
            "generator_model": slot["generator_model"],
            "prompt_family": slot["prompt_family"],
            "temperature": 0.7,
            "generation_attempt": generation_attempt,
            "transport_normalization": "surrounding_whitespace_only",
            "text_repaired": False,
        },
        "validation": {
            "judge_model": slot["judge_model"],
            "judge_attempt_count": judge_attempt_count,
            "judgment": judgment,
            "deterministic_quality": quality,
        },
        "provenance": {
            "prefix_bank": PREFIX_BANK_NAME,
            "prefix_window_sha256": record["window_sha256"],
            "candidate_replacement_index": candidate_index,
        },
    }


def synthesize_candidate(
    api: ApiPool,
    record: dict[str, Any],
    slot: dict[str, Any],
    args: argparse.Namespace,
    *,
    candidate_index: int,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    audit: dict[str, Any] = {
        "sample_id": record["sample_id"],
        "candidate_replacement_index": candidate_index,
        "planner_attempts": [],
        "generation_attempts": [],
    }
    plan, planner_attempts = run_planner(
        api,
        record,
        model=slot["planner_model"],
        retries=args.planner_retries,
    )
    audit["planner_attempts"] = planner_attempts
    if plan is None:
        audit["final_reason"] = "planner_exhausted"
        return None, audit

    plan_text = " ".join(plan["propositions"])
    fact_order = list(range(len(plan["propositions"])))
    audit["generation_fact_order"] = [
        index + 1 for index in fact_order
    ]
    retry_feedback = ""
    for generation_attempt in range(1, args.generation_retries + 1):
        generation_audit: dict[str, Any] = {
            "attempt": generation_attempt,
            "judge_attempts": [],
        }
        try:
            max_tokens = (
                min(900, max(350, record["continuation_units"] * 4))
                if record["language"] == "en"
                else min(900, max(450, record["continuation_units"] * 3))
            )
            raw = api.chat(
                slot["generator_model"],
                simple_generator_messages(
                    record,
                    plan,
                    retry_feedback=retry_feedback,
                ),
                max_new_tokens=max_tokens,
                temperature=0.7,
            )
            # This only removes transport whitespace; internal generated text
            # remains byte-for-byte unchanged and is never repaired.
            continuation = raw.strip()
            quality = continuation_quality(
                continuation,
                language=record["language"],
                human_continuation=record["human_continuation"],
                prefix=record["prefix"],
                content_plan_text=plan_text,
                target_units=record["continuation_units"],
            )
            generation_audit["raw_response"] = raw
            generation_audit["deterministic_quality"] = quality
            if not quality["accepted"]:
                generation_audit["final_reason"] = "deterministic_gate_failed"
                audit["generation_attempts"].append(generation_audit)
                retry_feedback = retry_feedback_from_quality(
                    quality,
                    record["language"],
                )
                continue

            judgment, judge_attempts = run_judge(
                api,
                record,
                plan,
                continuation,
                model=slot["judge_model"],
                retries=args.judge_retries,
            )
            generation_audit["judge_attempts"] = judge_attempts
            if judgment is None:
                generation_audit["final_reason"] = "judge_failed"
                audit["generation_attempts"].append(generation_audit)
                retry_feedback = (
                    "The prior unrecorded draft failed semantic audit. Cover "
                    "every planned fact, but add no evaluation, motivation, "
                    "cause, consequence, or certainty not present in the plan."
                    if record["language"] == "en"
                    else
                    "上一份未收录草稿未通过语义审核。请覆盖全部计划事实，但不要增加计划中"
                    "没有的评价、动机、原因、后果或确定性变化。"
                )
                continue

            generation_audit["final_reason"] = "accepted"
            audit["generation_attempts"].append(generation_audit)
            pair = make_pair(
                record,
                slot,
                plan,
                continuation,
                quality,
                judgment,
                fact_order=fact_order,
                generation_attempt=generation_attempt,
                candidate_index=candidate_index,
                planner_attempt_count=len(planner_attempts),
                judge_attempt_count=len(judge_attempts),
            )
            audit["final_reason"] = "accepted"
            return pair, audit
        except Exception as exc:
            generation_audit.update(
                {
                    "final_reason": "api_or_processing_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            audit["generation_attempts"].append(generation_audit)

    audit["final_reason"] = "generation_exhausted"
    return None, audit


def config_fingerprint(
    *,
    language: str,
    bank_hash: str,
    models: list[str],
    args: argparse.Namespace,
) -> str:
    config = {
        "schema": SCHEMA_NAME,
        "validator": VALIDATOR_NAME,
        "protocol": PROTOCOL_NAME,
        "language": language,
        "prefix_bank": PREFIX_BANK_NAME,
        "bank_hash": bank_hash,
        "num_samples": args.num_samples,
        "models": models,
        "seed": args.seed,
        "planner_retries": args.planner_retries,
        "generation_retries": args.generation_retries,
        "judge_retries": args.judge_retries,
        "candidate_replacements": args.candidate_replacements,
    }
    return hashlib.sha256(
        json.dumps(config, sort_keys=True).encode("utf-8")
    ).hexdigest()


def run_slot(
    api: ApiPool,
    slot: dict[str, Any],
    records_by_id: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    *,
    fingerprint: str,
    work_path: Path,
) -> dict[str, Any]:
    if work_path.is_file():
        existing = json.loads(work_path.read_text(encoding="utf-8"))
        if existing.get("config_fingerprint") != fingerprint:
            raise RuntimeError(
                f"{work_path} belongs to another configuration; remove the "
                "incomplete canonical output before restarting"
            )
        if existing.get("status") == "accepted":
            return existing

    result: dict[str, Any] = {
        "config_fingerprint": fingerprint,
        "slot": slot,
        "status": "failed",
        "candidate_audits": [],
        "updated_at": utc_now(),
    }
    for candidate_index, sample_id in enumerate(
        slot["candidate_sample_ids"],
        start=1,
    ):
        pair, audit = synthesize_candidate(
            api,
            records_by_id[sample_id],
            slot,
            args,
            candidate_index=candidate_index,
        )
        result["candidate_audits"].append(audit)
        result["updated_at"] = utc_now()
        if pair is not None:
            result["status"] = "accepted"
            result["pair"] = pair
            atomic_json(work_path, result)
            return result
        atomic_json(work_path, result)
    return result


def numeric_summary(values: list[float | int]) -> dict[str, float]:
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.fmean(values),
        "max": max(values),
    }


def build_quality_report(
    pairs: list[dict[str, Any]],
    slot_results: list[dict[str, Any]],
) -> dict[str, Any]:
    ratios = [pair["lengths"]["ai_to_human_ratio"] for pair in pairs]
    ai_units = [pair["lengths"]["ai_continuation_units"] for pair in pairs]
    generation_attempts = [
        pair["generation"]["generation_attempt"] for pair in pairs
    ]
    replacement_indices = [
        pair["provenance"]["candidate_replacement_index"] for pair in pairs
    ]
    plan_shared_runs = [
        pair["validation"]["deterministic_quality"][
            "longest_plan_shared_run"
        ]
        for pair in pairs
    ]
    source_model = Counter(
        (pair["source"], pair["generation"]["generator_model"])
        for pair in pairs
    )
    sources = sorted({pair["source"] for pair in pairs})
    generator_models = sorted({
        pair["generation"]["generator_model"] for pair in pairs
    })
    all_source_model_counts = [
        source_model[(source, model)]
        for source in sources
        for model in generator_models
    ]
    rejected_candidates = sum(
        sum(
            audit.get("final_reason") != "accepted"
            for audit in result["candidate_audits"]
        )
        for result in slot_results
    )
    rejected_generations = sum(
        sum(
            attempt.get("final_reason") != "accepted"
            for audit in result["candidate_audits"]
            for attempt in audit.get("generation_attempts", [])
        )
        for result in slot_results
    )
    return {
        "status": "pass",
        "n_pairs": len(pairs),
        "all_deterministic_gates_passed": all(
            pair["validation"]["deterministic_quality"]["accepted"]
            for pair in pairs
        ),
        "all_independent_judges_passed": all(
            pair["validation"]["judgment"]["verdict"] == "pass"
            for pair in pairs
        ),
        "all_core_content_acceptable": all(
            pair["validation"]["judgment"]["core_content_coverage"]
            in ("complete", "mostly_complete")
            for pair in pairs
        ),
        "plan_serialization_diagnostic_count": sum(
            pair["validation"]["deterministic_quality"][
                "plan_serialization_diagnostic"
            ]
            for pair in pairs
        ),
        "extreme_plan_serialization_count": sum(
            (
                pair["validation"]["deterministic_quality"].get(
                    "plan_serialization_share",
                    pair["validation"]["deterministic_quality"].get(
                        "longest_plan_shared_run",
                        0,
                    )
                    / pair["lengths"]["ai_continuation_units"],
                )
                >= 0.80
            )
            for pair in pairs
        ),
        "source_counts": dict(sorted(Counter(
            pair["source"] for pair in pairs
        ).items())),
        "generator_model_counts": dict(sorted(Counter(
            pair["generation"]["generator_model"] for pair in pairs
        ).items())),
        "planner_model_counts": dict(sorted(Counter(
            pair["content_control"]["planner_model"] for pair in pairs
        ).items())),
        "judge_model_counts": dict(sorted(Counter(
            pair["validation"]["judge_model"] for pair in pairs
        ).items())),
        "prompt_family_counts": dict(sorted(Counter(
            pair["generation"]["prompt_family"] for pair in pairs
        ).items())),
        "source_by_generator_model": {
            f"{source}::{model}": count
            for (source, model), count in sorted(source_model.items())
        },
        "source_by_generator_cell_spread": (
            max(all_source_model_counts) - min(all_source_model_counts)
        ),
        "ai_units": numeric_summary(ai_units),
        "ai_to_human_length_ratio": numeric_summary(ratios),
        "longest_plan_shared_run": numeric_summary(plan_shared_runs),
        "generation_attempt": numeric_summary(generation_attempts),
        "candidate_replacement_index": numeric_summary(replacement_indices),
        "rejected_candidate_count": rejected_candidates,
        "rejected_generation_count": rejected_generations,
        "text_repair_count": 0,
    }


def build_language(
    language: str,
    api: ApiPool,
    models: list[str],
    args: argparse.Namespace,
) -> None:
    bank_dir, bank_manifest, records = load_bank(language)
    bank_path = bank_dir / "samples.jsonl"
    bank_hash = sha256_file(bank_path)
    if bank_manifest.get("output_sha256") != bank_hash:
        raise RuntimeError(f"{bank_path}: prefix-bank hash mismatch")
    if len(models) < 3:
        raise ValueError(
            "At least three API models are required so planner, generator, "
            "and judge are all different."
        )

    output_dir = (
        PROJECT_ROOT
        / "data"
        / "paired"
        / language
        / PROTOCOL_NAME
    )
    work_dir = output_dir / "work"
    pairs_path = output_dir / "pairs.jsonl"
    manifest_path = output_dir / "manifest.json"
    quality_path = output_dir / "quality_report.json"
    failures_path = output_dir / "failures.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    fingerprint = config_fingerprint(
        language=language,
        bank_hash=bank_hash,
        models=models,
        args=args,
    )
    if pairs_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("config_fingerprint") != fingerprint:
            raise RuntimeError(
                f"{output_dir} already contains another configuration; "
                "remove it explicitly before rebuilding the canonical dataset"
            )
        existing_pairs = load_jsonl(pairs_path)
        if len(existing_pairs) == args.num_samples:
            LOGGER.info(
                "%s already complete with %d pairs; nothing to regenerate",
                language,
                len(existing_pairs),
            )
            return

    slots = build_slots(
        records,
        language=language,
        count=args.num_samples,
        models=models,
        seed=args.seed,
        replacements=args.candidate_replacements,
    )
    records_by_id = {record["sample_id"]: record for record in records}
    LOGGER.info(
        "Starting %s dataset: %d slots, %d workers, %d source corpora",
        language,
        len(slots),
        args.workers,
        len(set(slot["source"] for slot in slots)),
    )

    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                run_slot,
                api,
                slot,
                records_by_id,
                args,
                fingerprint=fingerprint,
                work_path=work_dir / f"{slot['slot_id']}.json",
            ): slot
            for slot in slots
        }
        completed = 0
        for future in as_completed(futures):
            slot = futures[future]
            result = future.result()
            results.append(result)
            completed += 1
            LOGGER.info(
                "%s %d/%d: %s -> %s",
                language,
                completed,
                len(slots),
                slot["slot_id"],
                result["status"],
            )

    results.sort(key=lambda item: item["slot"]["slot_index"])
    failed = [result for result in results if result["status"] != "accepted"]
    failures = [
        {
            "slot": result["slot"],
            "status": result["status"],
            "candidate_audits": result["candidate_audits"],
        }
        for result in results
        if any(
            audit.get("final_reason") != "accepted"
            or any(
                attempt.get("final_reason") != "accepted"
                for attempt in audit.get("generation_attempts", [])
            )
            for audit in result["candidate_audits"]
        )
    ]
    atomic_jsonl(failures_path, failures)
    if failed:
        raise RuntimeError(
            f"{language}: {len(failed)} slots exhausted all candidate "
            f"replacements; inspect {failures_path}"
        )

    pairs = [result["pair"] for result in results]
    if len({pair["sample_id"] for pair in pairs}) != len(pairs):
        raise RuntimeError(f"{language}: duplicate sample IDs in accepted pairs")
    atomic_jsonl(pairs_path, pairs)
    quality = build_quality_report(pairs, results)
    atomic_json(quality_path, quality)
    manifest = {
        "schema": "proposition_aligned_manifest",
        "protocol": PROTOCOL_NAME,
        "validator": VALIDATOR_NAME,
        "created_at": utc_now(),
        "language": language,
        "n_pairs": len(pairs),
        "config_fingerprint": fingerprint,
        "source_prefix_bank": {
            "name": PREFIX_BANK_NAME,
            "samples_path": str(bank_path.relative_to(PROJECT_ROOT)),
            "samples_sha256": bank_hash,
            "tokenizer_used_for_selection": False,
        },
        "models": models,
        "role_assignment": {
            "all_three_roles_distinct_per_pair": True,
            "planner_policy": (
                "nearest subsequent non-DeepSeek model; DeepSeek-V4-Flash "
                "failed the abstraction-format check and remains available "
                "as generator and judge"
            ),
            "judge_policy": (
                "one model distinct from the generator and planner performs "
                "a relaxed major-content audit"
            ),
        },
        "sampling": {
            "seed": args.seed,
            "source_balancing": True,
            "generator_model_balancing": True,
            "source_by_generator_joint_balancing": True,
            "maximum_source_by_generator_cell_spread": 1,
            "prompt_family_balancing": True,
            "generation_facts_preserve_reference_order": True,
            "candidate_replacements_per_slot": args.candidate_replacements,
        },
        "acceptance": {
            "natural_unit_length_ratio": [0.60, 1.50],
            "deterministic_prose_gate": True,
            "long_verbatim_copy_gate": True,
            "content_plan_serialization_diagnostic": {
                "en_maximum_shared_run_words": 19,
                "zh_maximum_shared_run_text_units": 49,
                "rejects_candidate": False,
            },
            "extreme_plan_serialization_gate": {
                "maximum_longest_run_share": 0.80,
                "rejects_candidate": True,
            },
            "independent_semantic_judges": 1,
            "complete_or_mostly_complete_core_content_required": True,
            "minor_omissions_allowed": True,
            "harmless_elaboration_allowed": True,
            "major_unsupported_claims_allowed": False,
            "major_contradictions_allowed": False,
            "text_repair": False,
            "transport_normalization": "surrounding_whitespace_only",
        },
        "outputs": {
            "pairs": {
                "path": str(pairs_path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(pairs_path),
            },
            "quality_report": {
                "path": str(quality_path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(quality_path),
            },
            "failures": {
                "path": str(failures_path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(failures_path),
            },
            "work_directory": str(work_dir.relative_to(PROJECT_ROOT)),
        },
        "quality_status": quality["status"],
    }
    atomic_json(manifest_path, manifest)
    LOGGER.info(
        "%s complete: %s (%s)",
        language,
        pairs_path,
        manifest["outputs"]["pairs"]["sha256"],
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    if args.num_samples <= 0:
        raise ValueError("--num-samples must be positive")
    if args.workers <= 0:
        raise ValueError("--workers must be positive")
    models = (
        [item.strip() for item in args.models.split(",") if item.strip()]
        if args.models
        else get_api_models()
    )
    if not models:
        raise ValueError("No models configured in API_MODELS or API_MODEL")
    if len(models) != len(set(models)):
        raise ValueError("Configured API model names must be unique")
    api = ApiPool(models, timeout=args.timeout)
    if args.smoke_models:
        smoke_models(api, models)
        return
    languages = ("en", "zh") if args.language == "both" else (args.language,)
    for language in languages:
        build_language(language, api, models, args)


if __name__ == "__main__":
    main()
