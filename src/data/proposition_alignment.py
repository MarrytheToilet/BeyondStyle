"""Validation helpers for proposition-aligned continuation synthesis.

The helpers in this module deliberately avoid model tokenizers.  They validate
plans and generated continuations using the same natural-language units as the
prefix bank, and they reject unsuitable text instead of rewriting it.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import Any

from src.data.natural_window import NaturalLanguageWindowSplitter


PLAN_KEYS = {
    "language",
    "propositions",
    "required_entities",
    "discourse_endpoint",
}
REFUSAL_RE = re.compile(
    r"(?:\b(?:i cannot|i can't|i am unable|as an ai|sorry[,，]?\s+i)\b|"
    r"抱歉|无法(?:完成|继续|提供)|作为(?:一个)?人工智能)",
    re.IGNORECASE,
)
META_RE = re.compile(
    r"(?:^(?:continuation|answer|response|续写|回答|正文)\s*[:：]|"
    r"\b(?:here is|here's) (?:the|a) continuation\b|"
    r"以下是(?:续写|正文|回答))",
    re.IGNORECASE,
)
BULLET_RE = re.compile(
    r"(?m)^\s*(?:[-*•]|\d+[.)、]|[一二三四五六七八九十]+、)\s+"
)
MARKDOWN_HEADING_RE = re.compile(r"(?m)^\s{0,3}#{1,6}\s+")


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse one JSON object, allowing only surrounding transport chatter."""
    stripped = text.strip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("response does not contain a JSON object")
        try:
            value = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON object: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("JSON response must be an object")
    return value


def _normalized_words(text: str) -> list[str]:
    return re.findall(r"\b[\w'’\-]+\b", text.casefold(), re.UNICODE)


def _normalized_zh_chars(text: str) -> list[str]:
    return re.findall(
        r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFFA-Za-z0-9]",
        text.casefold(),
    )


def longest_shared_run(
    left: str,
    right: str,
    language: str,
) -> int:
    """Return the longest contiguous shared run in natural-language units."""
    if language == "en":
        left_units = _normalized_words(left)
        right_units = _normalized_words(right)
    elif language == "zh":
        left_units = _normalized_zh_chars(left)
        right_units = _normalized_zh_chars(right)
    else:
        raise ValueError(f"unsupported language: {language}")
    if not left_units or not right_units:
        return 0
    match = SequenceMatcher(
        None,
        left_units,
        right_units,
        autojunk=False,
    ).find_longest_match()
    return match.size


def validate_content_plan(
    value: dict[str, Any],
    *,
    language: str,
    human_continuation: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate and canonicalize a planner JSON response."""
    issues: list[str] = []
    missing = PLAN_KEYS - value.keys()
    if missing:
        issues.append(f"missing_plan_keys:{','.join(sorted(missing))}")

    plan_language = value.get("language")
    if plan_language != language:
        issues.append("plan_language_mismatch")

    propositions = value.get("propositions")
    if not isinstance(propositions, list):
        issues.append("propositions_not_list")
        propositions = []
    elif not 3 <= len(propositions) <= 10:
        issues.append("proposition_count_outside_3_10")

    clean_propositions: list[str] = []
    for index, item in enumerate(propositions):
        if not isinstance(item, str) or not item.strip():
            issues.append(f"invalid_proposition:{index + 1}")
            continue
        proposition = item.strip()
        splitter = NaturalLanguageWindowSplitter(
            language=language,
            target_prefix_units=1,
            target_continuation_units=1,
            tolerance=0,
        )
        units = splitter.count_units(proposition)
        maximum = 45 if language == "en" else 80
        if units > maximum:
            issues.append(f"proposition_too_long:{index + 1}")
        clean_propositions.append(proposition)

    entities = value.get("required_entities")
    if not isinstance(entities, list):
        issues.append("required_entities_not_list")
        entities = []
    clean_entities: list[str] = []
    for index, item in enumerate(entities):
        if not isinstance(item, str) or not item.strip():
            issues.append(f"invalid_required_entity:{index + 1}")
            continue
        entity = item.strip()
        entity_splitter = NaturalLanguageWindowSplitter(
            language=language,
            target_prefix_units=1,
            target_continuation_units=1,
            tolerance=0,
        )
        maximum_entity_units = 12 if language == "en" else 30
        if entity_splitter.count_units(entity) > maximum_entity_units:
            issues.append(f"required_entity_too_long:{index + 1}")
        clean_entities.append(entity)
    if len(clean_entities) > 24:
        issues.append("too_many_required_entities")

    endpoint = value.get("discourse_endpoint")
    if not isinstance(endpoint, str) or not endpoint.strip():
        issues.append("invalid_discourse_endpoint")
        clean_endpoint = ""
    else:
        clean_endpoint = endpoint.strip()

    # Check each semantic field independently. Concatenating list items creates
    # artificial cross-boundary runs and falsely labels entity inventories as
    # copied prose.
    copy_limit = 18 if language == "en" else 36
    for index, proposition in enumerate(clean_propositions, start=1):
        if longest_shared_run(
            proposition,
            human_continuation,
            language,
        ) >= copy_limit:
            issues.append(f"proposition_copies_human_wording:{index}")
    if longest_shared_run(
        clean_endpoint,
        human_continuation,
        language,
    ) >= copy_limit:
        issues.append("endpoint_copies_human_wording")

    canonical = {
        "language": language,
        "propositions": clean_propositions,
        "required_entities": clean_entities,
        "discourse_endpoint": clean_endpoint,
    }
    return (canonical if not issues else None), sorted(set(issues))


def continuation_quality(
    text: str,
    *,
    language: str,
    human_continuation: str,
    prefix: str,
    content_plan_text: str = "",
    target_units: int,
    minimum_ratio: float = 0.60,
    maximum_ratio: float = 1.50,
) -> dict[str, Any]:
    """Evaluate a generated continuation without modifying its content."""
    splitter = NaturalLanguageWindowSplitter(
        language=language,
        target_prefix_units=1,
        target_continuation_units=1,
        tolerance=0,
    )
    issues: list[str] = []
    units = splitter.count_units(text)
    lower = max(1, int(target_units * minimum_ratio))
    upper = max(lower, int(target_units * maximum_ratio))
    if not lower <= units <= upper:
        issues.append("continuation_length_out_of_bounds")
    issues.extend(splitter.prose_quality_issues(text))
    if REFUSAL_RE.search(text):
        issues.append("refusal_or_capability_disclaimer")
    if META_RE.search(text):
        issues.append("meta_preamble")
    if BULLET_RE.search(text):
        issues.append("list_or_bullets")
    if MARKDOWN_HEADING_RE.search(text):
        issues.append("markdown_heading")

    copy_limit = 30 if language == "en" else 80
    human_shared_run = longest_shared_run(
        text,
        human_continuation,
        language,
    )
    if human_shared_run >= copy_limit:
        issues.append("copies_human_continuation")

    prefix_shared_run = longest_shared_run(text, prefix, language)
    prefix_copy_limit = 25 if language == "en" else 60
    if prefix_shared_run >= prefix_copy_limit:
        issues.append("repeats_prefix")

    plan_shared_run = (
        longest_shared_run(text, content_plan_text, language)
        if content_plan_text
        else 0
    )
    # Same-content realizations inevitably share entities, quantities, and
    # short relation phrases.  These thresholds target sentence-scale
    # serialization rather than ordinary proposition overlap.
    plan_copy_limit = 20 if language == "en" else 50
    plan_serialization_share = (
        plan_shared_run / units if units > 0 else 0.0
    )
    if plan_serialization_share >= 0.80:
        issues.append("extreme_plan_serialization")

    return {
        "accepted": not issues,
        "issues": sorted(set(issues)),
        "units": units,
        "target_units": target_units,
        "minimum_units": lower,
        "maximum_units": upper,
        "length_ratio": units / target_units if target_units else None,
        "longest_human_shared_run": human_shared_run,
        "longest_prefix_shared_run": prefix_shared_run,
        "longest_plan_shared_run": plan_shared_run,
        "plan_serialization_share": plan_serialization_share,
        "plan_copy_limit": plan_copy_limit,
        "plan_serialization_diagnostic": (
            plan_shared_run >= plan_copy_limit
        ),
        "transport_whitespace_stripped": True,
    }


def validate_relaxed_judgment(
    value: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate a coarse judge that only rejects major content failures."""
    issues: list[str] = []
    coverage = value.get("core_content_coverage")
    if coverage not in ("complete", "mostly_complete", "poor"):
        issues.append("invalid_core_content_coverage")

    def validate_string_list(field: str) -> list[str]:
        raw_items = value.get(field)
        if not isinstance(raw_items, list):
            issues.append(f"invalid_{field}")
            return []
        cleaned: list[str] = []
        for position, item in enumerate(raw_items, start=1):
            if not isinstance(item, str) or not item.strip():
                issues.append(f"invalid_{field}_item:{position}")
                continue
            cleaned.append(item.strip())
        return cleaned

    missing = validate_string_list("missing_core_facts")
    unsupported = validate_string_list("major_unsupported_claims")
    contradictions = validate_string_list("major_contradictions")

    verdict = value.get("verdict")
    if verdict not in ("pass", "fail"):
        issues.append("invalid_judge_verdict")
    elif verdict != "pass":
        issues.append("judge_verdict_fail")
    reason = value.get("brief_reason")
    if not isinstance(reason, str) or not reason.strip():
        issues.append("invalid_brief_reason")
        reason = ""

    if coverage == "poor":
        issues.append("poor_core_content_coverage")
    if unsupported:
        issues.append("major_unsupported_claims_present")
    if contradictions:
        issues.append("major_contradictions_present")

    canonical = {
        "core_content_coverage": coverage,
        "missing_core_facts": missing,
        "major_unsupported_claims": unsupported,
        "major_contradictions": contradictions,
        "verdict": verdict,
        "brief_reason": reason.strip(),
    }
    return (canonical if not issues else None), sorted(set(issues))
