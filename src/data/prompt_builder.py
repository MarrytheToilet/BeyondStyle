"""
Two-step prompt construction for content-aligned AI continuations.

Supports two modes:

**Light mode** (default):
  Step 1: Analyse human continuation → extract topic + key entities only.
  Step 2: Build a lightly constrained prompt (topic + entities, no structure).

**Full mode** (legacy):
  Step 1: Full analysis → topic, entities, content_direction, scope.
  Step 2: Detailed constrained prompt.

The light mode strikes a balance: AI writes about the *same topic* as the
human continuation (so differences aren't just about content divergence) but
is otherwise free to use its own style, structure, and phrasing.
"""

import json
import logging
import re
from dataclasses import dataclass, field, asdict
from typing import Optional, List

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------

@dataclass
class ContentConstraint:
    """Structured content constraint extracted from a human continuation."""
    topic: str
    key_entities: List[str] = field(default_factory=list)
    content_direction: str = ""
    scope: str = ""
    raw_analysis: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ContentConstraint":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# ------------------------------------------------------------------
# Prompt templates — LIGHT mode (topic + entities only)
# ------------------------------------------------------------------

LIGHT_ANALYSIS_SYSTEM_PROMPT = """\
You are a concise text analyst.  Given a **prefix** and the **human-written continuation**, output a JSON object with exactly two keys:

{
  "topic": "<one short phrase describing the central topic>",
  "key_entities": ["<entity1>", "<entity2>", "<entity3>"]
}

Rules:
- topic: one phrase, max 10 words.
- key_entities: 2–4 most important proper nouns or concepts.
- Output **only** the JSON object, no markdown fences, no explanation."""

LIGHT_ANALYSIS_USER_TEMPLATE = """\
Prefix:
{prefix}

Human continuation:
{human_continuation}"""

LIGHT_CONSTRAINED_SYSTEM_PROMPT = """\
Continue the given text naturally. Stay on topic: {topic}. \
You may reference: {entities}. Write approximately {target_words} words. \
Do not add titles, bullet points, or meta-commentary. /no_think"""

LIGHT_CONSTRAINED_USER_TEMPLATE = """{prefix}"""


# ------------------------------------------------------------------
# Prompt templates — Chinese (LIGHT mode)
# ------------------------------------------------------------------

LIGHT_ANALYSIS_SYSTEM_PROMPT_ZH = """\
你是一位简洁的文本分析助手。给定一个**前文**和**人类续写**，输出一个恰好包含两个键的JSON对象：

{
  "topic": "<中心主题，最多10个词>",
  "key_entities": ["<实体1>", "<实体2>", "<实体3>"]
}

规则：
- topic: 一个短语，最多10个词。
- key_entities: 2–4个最重要的专有名词或概念。
- 仅输出JSON对象，无markdown围栏，无解释。"""

LIGHT_ANALYSIS_USER_TEMPLATE_ZH = """\
前文：
{prefix}

人类续写：
{human_continuation}"""

LIGHT_CONSTRAINED_SYSTEM_PROMPT_ZH = """\
自然地续写给定的文本。围绕主题：{topic}。\
可以提及：{entities}。大约写{target_words}个字。\
不要添加标题、列表或元评论。 /no_think"""

LIGHT_CONSTRAINED_USER_TEMPLATE_ZH = """{prefix}"""


# ------------------------------------------------------------------
# Prompt templates — FULL mode (legacy, detailed constraints)
# ------------------------------------------------------------------

# Step 1: ask the LLM to analyse the human continuation
ANALYSIS_SYSTEM_PROMPT = """\
You are a concise text analyst.  Given a **prefix** and the **human-written continuation** that follows it, output a JSON object with exactly four keys:

{
  "topic": "<one-sentence summary of the continuation's central topic>",
  "key_entities": ["<entity1>", "<entity2>", ...],
  "content_direction": "<how the continuation develops: what argument / narrative / explanation it advances>",
  "scope": "<what the continuation covers and where it stops>"
}

Rules:
- Be factual and concise; each value should be 1–2 sentences max.
- key_entities: up to 6 most important proper nouns, concepts, or technical terms.
- Output **only** the JSON object, no markdown fences, no explanation."""

ANALYSIS_USER_TEMPLATE = """\
Prefix:
{prefix}

Human continuation:
{human_continuation}"""

# Step 2: constrained continuation prompt
CONSTRAINED_SYSTEM_PROMPT = """\
You are a helpful writing assistant. Continue the given text naturally.
Do not add a title, bullet points, or meta-commentary. Do not explain what you are doing. Just write the next part of the text directly.

Content guidance:
- Topic: {topic}
- Key entities: {entities}
- Target length: approximately {target_words} words"""

CONSTRAINED_USER_TEMPLATE = """\
Continue the following text naturally (approximately {target_words} words):

{prefix}"""


# ------------------------------------------------------------------
# PromptBuilder
# ------------------------------------------------------------------

class PromptBuilder:
    """
    Two-step prompt builder.

    Step 1 — ``analyse_continuation``: call LLM to extract content
    constraints from the human continuation.

    Step 2 — ``build_continuation_prompt``: produce (system, user) pair
    that guides the model to generate content-aligned continuation.

    Set ``light=True`` for minimal constraints (topic + entities only).
    """

    def __init__(self, generator, temperature: float = 0.3, light: bool = False,
                 language: str = "en"):
        """
        Args:
            generator: A backend generator that exposes ``generate(prompt, …)``.
            temperature: Temperature for the *analysis* call (low = deterministic).
            light: If True, use light constraint mode (topic + entities only).
            language: Language code ("en" or "zh").
        """
        self.generator = generator
        self.temperature = temperature
        self.light = light
        self.language = language

    # ----- Step 1 -----

    def analyse_continuation(
        self,
        prefix: str,
        human_continuation: str,
    ) -> ContentConstraint:
        """Extract content constraints from a human continuation."""

        if self.light:
            if self.language in ("zh", "chinese"):
                sys_prompt = LIGHT_ANALYSIS_SYSTEM_PROMPT_ZH
                user_tpl = LIGHT_ANALYSIS_USER_TEMPLATE_ZH
            else:
                sys_prompt = LIGHT_ANALYSIS_SYSTEM_PROMPT
                user_tpl = LIGHT_ANALYSIS_USER_TEMPLATE
        else:
            sys_prompt = ANALYSIS_SYSTEM_PROMPT
            user_tpl = ANALYSIS_USER_TEMPLATE

        user_msg = user_tpl.format(
            prefix=prefix,
            human_continuation=human_continuation,
        )

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_msg},
        ]

        if hasattr(self.generator, "chat"):
            raw = self.generator.chat(
                messages=messages,
                max_new_tokens=300,
                temperature=self.temperature,
            )
        else:
            full_prompt = (
                f"[System]\n{ANALYSIS_SYSTEM_PROMPT}\n\n"
                f"[User]\n{user_msg}"
            )
            raw = self.generator.generate(
                prompt=full_prompt,
                max_new_tokens=300,
                temperature=self.temperature,
            )

        return self._parse_analysis(raw)

    def _parse_analysis(self, raw: str) -> ContentConstraint:
        """Parse the LLM JSON response into a ContentConstraint."""
        # Strip markdown fences if present
        cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback: try to extract a JSON-like block
            m = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if m:
                try:
                    data = json.loads(m.group())
                except json.JSONDecodeError:
                    logger.warning("Could not parse analysis JSON; using raw text")
                    return ContentConstraint(
                        topic=cleaned[:200],
                        raw_analysis=raw,
                    )
            else:
                logger.warning("No JSON found in analysis response")
                return ContentConstraint(topic=cleaned[:200], raw_analysis=raw)

        return ContentConstraint(
            topic=data.get("topic", ""),
            key_entities=data.get("key_entities", []),
            content_direction=data.get("content_direction", ""),
            scope=data.get("scope", ""),
            raw_analysis=raw,
        )

    # ----- Step 2 -----

    def build_continuation_prompt(
        self,
        prefix: str,
        constraint: ContentConstraint,
        target_words: int = 150,
    ) -> tuple:
        """
        Return ``(system_prompt, user_prompt)`` incorporating content
        constraints for the actual continuation generation call.
        """
        entities_str = ", ".join(constraint.key_entities) if constraint.key_entities else "N/A"

        if self.light:
            if self.language in ("zh", "chinese"):
                system = LIGHT_CONSTRAINED_SYSTEM_PROMPT_ZH.format(
                    topic=constraint.topic or "与前文相同",
                    entities=entities_str,
                    target_words=target_words,
                )
                user = LIGHT_CONSTRAINED_USER_TEMPLATE_ZH.format(prefix=prefix)
            else:
                system = LIGHT_CONSTRAINED_SYSTEM_PROMPT.format(
                    topic=constraint.topic or "same as prefix",
                    entities=entities_str,
                    target_words=target_words,
                )
                user = LIGHT_CONSTRAINED_USER_TEMPLATE.format(prefix=prefix)
        else:
            system = CONSTRAINED_SYSTEM_PROMPT.format(
                topic=constraint.topic or "same as prefix",
                entities=entities_str,
                target_words=target_words,
            )
            user = CONSTRAINED_USER_TEMPLATE.format(prefix=prefix, target_words=target_words)

        return system, user
