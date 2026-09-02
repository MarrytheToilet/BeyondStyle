"""Model-independent natural-language prefix/continuation windowing."""

import re
import statistics
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


EN_UNIT_RE = re.compile(r"\b\w+(?:['’\-]\w+)*\b", re.UNICODE)
ZH_UNIT_RE = re.compile(
    r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]|[A-Za-z0-9]+"
)
EN_SENTENCE_END_RE = re.compile(r"""[.!?]+(?:["”’])?(?=\s+|$)""")
ZH_SENTENCE_END_RE = re.compile(r"""[。！？!?]+(?:["”’」』】》])?""")
URL_RE = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
REFERENCE_HEADING_RE = re.compile(
    r"(?:^|\n|\s)(?:references|bibliography|works cited|"
    r"参考文献|參考文獻|参考资料|參考資料|外部链接|外部連結)(?:\s|$|[:：])",
    re.IGNORECASE,
)
CODE_KEYWORD_RE = re.compile(
    r"\b(?:import|class|public\s+static|function|def|return)\b"
)
DOT_LEADER_RE = re.compile(r"(?:\.\s*){4,}")
LIST_HEADING_RE = re.compile(
    r"(?:演员阵容|演員陣容|收录歌曲|收錄歌曲|曲目列表|角色列表|"
    r"获奖名单|獲獎名單|cast list|track listing)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SentenceSpan:
    start: int
    end: int


@dataclass
class NaturalLanguageWindow:
    prefix: str
    continuation: str
    prefix_units: int
    continuation_units: int
    length_unit: str
    metadata: dict = field(default_factory=dict)


class NaturalLanguageWindowSplitter:
    """Select sentence-boundary windows without consulting a tokenizer."""

    def __init__(
        self,
        language: str,
        target_prefix_units: int,
        target_continuation_units: int,
        tolerance: float = 0.20,
    ):
        if language not in ("en", "zh"):
            raise ValueError(f"Unsupported language: {language}")
        if target_prefix_units <= 0 or target_continuation_units <= 0:
            raise ValueError("Natural-language length targets must be positive")
        if not 0 <= tolerance < 1:
            raise ValueError("tolerance must be in [0, 1)")

        self.language = language
        self.target_prefix_units = target_prefix_units
        self.target_continuation_units = target_continuation_units
        self.tolerance = tolerance
        self.min_prefix = int(target_prefix_units * (1 - tolerance))
        self.max_prefix = int(target_prefix_units * (1 + tolerance))
        self.min_continuation = int(
            target_continuation_units * (1 - tolerance)
        )
        self.max_continuation = int(
            target_continuation_units * (1 + tolerance)
        )
        self.length_unit = "word" if language == "en" else "zh_text_unit"

    def count_units(self, text: str) -> int:
        pattern = EN_UNIT_RE if self.language == "en" else ZH_UNIT_RE
        return len(pattern.findall(text))

    def prose_quality_issues(self, text: str) -> List[str]:
        """Return model-independent reasons why a span is not ordinary prose."""
        issues = []
        units = self.count_units(text)
        if not units:
            return ["no_natural_language_units"]

        characters_per_unit = len(text) / units
        maximum_density = 13.0 if self.language == "en" else 2.8
        if characters_per_unit > maximum_density:
            issues.append("abnormal_character_density")
        if URL_RE.search(text):
            issues.append("url_or_embedded_media")
        if '{"' in text or text.count('":') >= 2:
            issues.append("json_like")
        if text.count("|") >= 3:
            issues.append("table_like")
        if DOT_LEADER_RE.search(text):
            issues.append("dot_leader_list")
        if len(re.findall(r" {2,}", text)) >= 6:
            issues.append("aligned_list")
        if LIST_HEADING_RE.search(text):
            issues.append("list_heading")
        nonempty_lines = [
            line.strip() for line in text.splitlines() if line.strip()
        ]
        if (
            len(nonempty_lines) >= 5
            and statistics.median(map(len, nonempty_lines)) < 50
        ):
            issues.append("short_line_list")
        if text.count("$") >= 2 or text.count("=") >= 4:
            issues.append("equation_block")
        if (
            CODE_KEYWORD_RE.search(text)
            and (text.count("{") + text.count("}") + text.count(";")) >= 3
        ):
            issues.append("code_like")
        if REFERENCE_HEADING_RE.search(text):
            issues.append("reference_list")

        if self.language == "zh":
            cjk_units = len(re.findall(
                r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]",
                text,
            ))
            if cjk_units / units < 0.60:
                issues.append("low_chinese_unit_share")
        return sorted(set(issues))

    def sentence_spans(self, text: str) -> List[SentenceSpan]:
        boundary_re = (
            EN_SENTENCE_END_RE
            if self.language == "en"
            else ZH_SENTENCE_END_RE
        )
        spans = []
        start = 0
        for match in boundary_re.finditer(text):
            end = match.end()
            span = self._trimmed_span(text, start, end)
            if span is not None and self.count_units(
                text[span.start : span.end]
            ):
                spans.append(span)
            start = end
        if start < len(text):
            span = self._trimmed_span(text, start, len(text))
            if span is not None and self.count_units(
                text[span.start : span.end]
            ):
                spans.append(span)
        return spans

    @staticmethod
    def _trimmed_span(
        text: str,
        start: int,
        end: int,
    ) -> Optional[SentenceSpan]:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        return SentenceSpan(start, end) if start < end else None

    def _choose_end(
        self,
        text: str,
        spans: List[SentenceSpan],
        start_index: int,
        target: int,
        minimum: int,
        maximum: int,
    ) -> Optional[Tuple[int, int]]:
        candidates = []
        units = 0
        for index in range(start_index, len(spans)):
            span = spans[index]
            units += self.count_units(text[span.start : span.end])
            if minimum <= units <= maximum:
                candidates.append((abs(units - target), index + 1, units))
            if units > maximum:
                break
        if not candidates:
            return None
        _, end_index, selected_units = min(candidates)
        return end_index, selected_units

    def split_text(self, text: str) -> Optional[NaturalLanguageWindow]:
        spans = self.sentence_spans(text)
        if len(spans) < 2:
            return None

        prefix_choice = self._choose_end(
            text,
            spans,
            start_index=0,
            target=self.target_prefix_units,
            minimum=self.min_prefix,
            maximum=self.max_prefix,
        )
        if prefix_choice is None:
            return None
        prefix_end, prefix_units = prefix_choice

        continuation_choice = self._choose_end(
            text,
            spans,
            start_index=prefix_end,
            target=self.target_continuation_units,
            minimum=self.min_continuation,
            maximum=self.max_continuation,
        )
        if continuation_choice is None:
            return None
        continuation_end, continuation_units = continuation_choice

        prefix = text[spans[0].start : spans[prefix_end - 1].end].strip()
        continuation = text[
            spans[prefix_end].start : spans[continuation_end - 1].end
        ].strip()
        return NaturalLanguageWindow(
            prefix=prefix,
            continuation=continuation,
            prefix_units=prefix_units,
            continuation_units=continuation_units,
            length_unit=self.length_unit,
            metadata={
                "prefix_sentences": prefix_end,
                "continuation_sentences": continuation_end - prefix_end,
            },
        )
