"""Prompt templates for continuation generation."""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class PromptTemplate:
    """A prompt template for a specific condition."""
    name: str
    language: str
    instruction: str
    
    def format(self, prefix: str) -> str:
        """Format the prompt with a prefix."""
        return f"{self.instruction}\n\n{prefix}"


# English prompt templates
EN_TEMPLATES = [
    PromptTemplate(
        name="standard",
        language="en",
        instruction="Continue the text.",
    ),
    PromptTemplate(
        name="formal",
        language="en",
        instruction="Continue the text in a more formal style.",
    ),
    PromptTemplate(
        name="informal",
        language="en",
        instruction="Continue the text in a more natural and informal style.",
    ),
    PromptTemplate(
        name="beginner",
        language="en",
        instruction="Continue the text so that a beginner can easily understand it.",
    ),
    PromptTemplate(
        name="detailed",
        language="en",
        instruction="Continue the text with more explanatory detail.",
    ),
]

# Chinese prompt templates
ZH_TEMPLATES = [
    PromptTemplate(
        name="standard",
        language="zh",
        instruction="请继续写下去。",
    ),
    PromptTemplate(
        name="formal",
        language="zh",
        instruction="请继续写下去，并使用更正式的表达方式。",
    ),
    PromptTemplate(
        name="informal",
        language="zh",
        instruction="请继续写下去，并使用更自然随意的表达方式。",
    ),
    PromptTemplate(
        name="beginner",
        language="zh",
        instruction="请继续写下去，并让初学者也容易理解。",
    ),
    PromptTemplate(
        name="detailed",
        language="zh",
        instruction="请继续写下去，并补充更多解释细节。",
    ),
]


def get_prompt_templates(language: str) -> List[PromptTemplate]:
    """Get prompt templates for a language."""
    if language == "en":
        return EN_TEMPLATES
    elif language == "zh":
        return ZH_TEMPLATES
    else:
        raise ValueError(f"Unsupported language: {language}")


def get_all_conditions() -> Dict[str, List[str]]:
    """Get all condition names by language."""
    return {
        "en": [t.name for t in EN_TEMPLATES],
        "zh": [t.name for t in ZH_TEMPLATES],
    }
