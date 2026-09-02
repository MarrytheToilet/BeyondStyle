"""Predefined lexical, syntactic, discourse, and nuisance measurements.

The inventory is fixed independently of the human/machine source labels. It
favors transparent measurements that can be audited from a parsed document
over learned style classifiers.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import math
import re
from typing import Any, Iterable

from wordfreq import zipf_frequency


CONTENT_POS = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
FUNCTION_POS = {"PRON", "DET", "ADP", "AUX", "CCONJ", "SCONJ", "PART"}
CLAUSE_DEPS = {"ROOT", "acl", "advcl", "ccomp", "xcomp", "relcl"}
SUBORDINATE_DEPS = {"acl", "advcl", "ccomp", "xcomp", "relcl", "mark"}
SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj", "csubjpass"}

EVALUATIVE_WORDS = {
    "appropriate", "beneficial", "clear", "critical", "crucial", "effective",
    "essential", "important", "ideal", "notable", "significant", "valuable",
}
EMPHASIS_WORDS = {
    "absolutely", "certainly", "clearly", "especially", "fundamentally",
    "indeed", "particularly", "remarkably", "simply", "truly", "undoubtedly",
}
INSTITUTIONAL_WORDS = {
    "approach", "context", "development", "framework", "implementation",
    "initiative", "institution", "mechanism", "objective", "organization",
    "policy", "process", "strategy", "structure", "system",
}
ABSTRACT_WORDS = {
    "aspect", "concept", "condition", "consideration", "dimension", "factor",
    "impact", "implication", "issue", "perspective", "principle", "purpose",
    "relationship", "role", "significance", "value",
}
NOMINALIZATION_SUFFIXES = (
    "tion", "sion", "ment", "ness", "ity", "ance", "ence", "ism", "ship",
)

CONNECTIVES: dict[str, tuple[str, ...]] = {
    "causal": (
        "as a result", "because", "consequently", "hence", "therefore", "thus",
    ),
    "contrast": (
        "by contrast", "conversely", "however", "in contrast", "instead",
        "nevertheless", "on the other hand", "otherwise", "yet",
    ),
    "concession": (
        "although", "despite", "even though", "in spite of", "nonetheless",
        "though", "while",
    ),
    "purpose": (
        "for the purpose of", "in order to", "so as to", "so that", "to ensure",
    ),
    "continuity": (
        "additionally", "also", "furthermore", "in addition", "meanwhile",
        "moreover", "subsequently", "then",
    ),
    "elaboration": (
        "for example", "for instance", "in particular", "namely", "specifically",
    ),
}
ENUMERATION_MARKERS = (
    "first", "firstly", "second", "secondly", "third", "thirdly", "finally",
    "next",
)
SUMMARY_MARKERS = (
    "in conclusion", "in short", "in summary", "overall", "to conclude",
    "to summarize",
)


@dataclass(frozen=True)
class FeatureDefinition:
    name: str
    group: str
    description: str
    unit: str


def _definition(
    name: str,
    group: str,
    description: str,
    unit: str = "proportion",
) -> FeatureDefinition:
    return FeatureDefinition(name, group, description, unit)


FEATURE_DEFINITIONS: tuple[FeatureDefinition, ...] = (
    _definition("function_word_rate", "lexical", "Function POS tokens / word tokens."),
    _definition("pronoun_rate", "lexical", "Pronouns / word tokens."),
    _definition("contraction_rate", "lexical", "Orthographic contractions / word tokens."),
    _definition("mattr_50", "lexical", "Moving-average type-token ratio, window 50."),
    _definition("content_zipf_mean", "lexical", "Mean English Zipf frequency of content lemmas.", "Zipf"),
    _definition("repeated_trigram_rate", "lexical", "Repeated lemma-trigram instances / lemma trigrams."),
    _definition("abstract_lexicon_rate", "lexical", "Fixed abstract lexicon matches / word tokens."),
    _definition("evaluative_lexicon_rate", "lexical", "Fixed evaluative lexicon matches / word tokens."),
    _definition("emphasis_lexicon_rate", "lexical", "Fixed emphasis lexicon matches / word tokens."),
    _definition("institutional_lexicon_rate", "lexical", "Fixed institutional lexicon matches / word tokens."),
    _definition("dependency_depth_mean", "syntactic", "Mean token-to-root dependency depth.", "edges"),
    _definition("dependency_length_mean", "syntactic", "Mean absolute head-dependent distance.", "tokens"),
    _definition("clause_density", "syntactic", "Verbal clause heads / sentences.", "per sentence"),
    _definition("coordination_density", "syntactic", "Coordinating dependencies / sentences.", "per sentence"),
    _definition("subordination_density", "syntactic", "Subordinate dependencies / sentences.", "per sentence"),
    _definition("passive_clause_rate", "syntactic", "Passive verbal heads / verbal heads."),
    _definition("nominalization_rate", "syntactic", "Suffix-marked nominalizations / nouns."),
    _definition("sentence_length_cv", "syntactic", "Sentence-length coefficient of variation.", "ratio"),
    _definition("fragment_rate", "syntactic", "Sentences without a verbal predicate / sentences."),
    _definition("apposition_density", "syntactic", "Appositive dependencies / sentences.", "per sentence"),
    _definition("parenthetical_density", "syntactic", "Parenthetical spans / sentences.", "per sentence"),
    _definition("adjacent_pos_parallelism", "syntactic", "Adjacent-sentence POS sequence similarity."),
    _definition("causal_connective_rate", "discourse", "Causal connective matches / sentences.", "per sentence"),
    _definition("contrast_connective_rate", "discourse", "Contrast connective matches / sentences.", "per sentence"),
    _definition("concession_connective_rate", "discourse", "Concessive connective matches / sentences.", "per sentence"),
    _definition("purpose_connective_rate", "discourse", "Purpose connective matches / sentences.", "per sentence"),
    _definition("continuity_connective_rate", "discourse", "Continuity connective matches / sentences.", "per sentence"),
    _definition("elaboration_connective_rate", "discourse", "Elaboration connective matches / sentences.", "per sentence"),
    _definition("explicit_relation_rate", "discourse", "All fixed connective matches / sentences.", "per sentence"),
    _definition("transition_opening_rate", "discourse", "Sentences beginning with a fixed connective / sentences."),
    _definition("enumeration_marker_rate", "discourse", "Explicit enumeration openings / sentences."),
    _definition("summary_marker_rate", "discourse", "Explicit summary marker matches / sentences.", "per sentence"),
    _definition("entity_persistence", "discourse", "Adjacent-sentence noun set Jaccard."),
    _definition("subject_continuity", "discourse", "Adjacent-sentence subject-lemma set overlap."),
    _definition("adjacent_content_overlap", "discourse", "Adjacent-sentence content-lemma set Jaccard."),
    _definition("word_count", "nuisance", "Non-punctuation token count.", "tokens"),
    _definition("sentence_count", "nuisance", "Parser sentence count.", "sentences"),
    _definition("paragraph_count", "nuisance", "Non-empty paragraph count.", "paragraphs"),
    _definition("mean_sentence_length", "nuisance", "Mean word tokens per sentence.", "tokens"),
    _definition("named_entity_rate", "nuisance", "Named-entity token count / word tokens."),
    _definition("number_rate", "nuisance", "Number-like tokens / word tokens."),
    _definition("character_count", "nuisance", "Non-whitespace character count.", "characters"),
)


def feature_dictionary() -> list[dict[str, str]]:
    return [asdict(definition) for definition in FEATURE_DEFINITIONS]


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else 0.0


def _mattr(items: list[str], window: int = 50) -> float:
    if not items:
        return 0.0
    if len(items) <= window:
        return len(set(items)) / len(items)
    values = [
        len(set(items[start : start + window])) / window
        for start in range(len(items) - window + 1)
    ]
    return float(sum(values) / len(values))


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    return float(sum(values) / len(values)) if values else 0.0


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _dependency_depth(token: Any) -> int:
    depth = 0
    current = token
    seen: set[int] = set()
    while current.head.i != current.i and current.i not in seen:
        seen.add(current.i)
        current = current.head
        depth += 1
        if depth > 100:
            break
    return depth


def _phrase_pattern(phrase: str) -> re.Pattern[str]:
    escaped = r"\s+".join(re.escape(part) for part in phrase.split())
    return re.compile(rf"(?<!\w){escaped}(?!\w)", flags=re.IGNORECASE)


def _starts_with_marker(text: str, marker: str) -> bool:
    return bool(
        re.match(
            rf"^{re.escape(marker)}(?:\b|$)",
            text,
            flags=re.IGNORECASE,
        )
    )


CONNECTIVE_PATTERNS = {
    group: tuple(_phrase_pattern(phrase) for phrase in phrases)
    for group, phrases in CONNECTIVES.items()
}
ALL_OPENING_MARKERS = tuple(
    sorted(
        {
            *[phrase for phrases in CONNECTIVES.values() for phrase in phrases],
            *ENUMERATION_MARKERS,
            *SUMMARY_MARKERS,
        },
        key=len,
        reverse=True,
    )
)


class EnglishLinguisticFeatureExtractor:
    """Extract the frozen English feature inventory from a spaCy document."""

    language = "en"

    def __init__(self) -> None:
        self.feature_names = tuple(item.name for item in FEATURE_DEFINITIONS)

    def extract(self, doc: Any, raw_text: str | None = None) -> dict[str, float]:
        if not doc.has_annotation("DEP") or not doc.has_annotation("SENT_START"):
            raise ValueError("A spaCy pipeline with dependency parsing is required")

        text = raw_text if raw_text is not None else doc.text
        words = [token for token in doc if not token.is_space and not token.is_punct]
        alpha_words = [token for token in words if token.is_alpha]
        lemmas = [
            (token.lemma_ or token.lower_).lower()
            for token in alpha_words
            if (token.lemma_ or token.lower_).strip()
        ]
        content_tokens = [token for token in alpha_words if token.pos_ in CONTENT_POS]
        content_lemmas = [
            (token.lemma_ or token.lower_).lower() for token in content_tokens
        ]
        sentences = [sentence for sentence in doc.sents if sentence.text.strip()]
        sentence_count = len(sentences)
        word_count = len(words)

        function_count = sum(token.pos_ in FUNCTION_POS for token in alpha_words)
        pronoun_count = sum(token.pos_ == "PRON" for token in alpha_words)
        contraction_count = len(
            re.findall(
                r"\b(?:\w+n't|\w+'(?:re|ve|ll|d|m|s))\b",
                text,
                flags=re.IGNORECASE,
            )
        )
        trigrams = list(zip(lemmas, lemmas[1:], lemmas[2:]))
        repeated_trigrams = len(trigrams) - len(set(trigrams))
        content_zipf = [
            zipf_frequency(lemma, "en")
            for token, lemma in zip(content_tokens, content_lemmas)
            if token.pos_ != "PROPN" and len(lemma) > 1
        ]
        lower_words = [token.lower_ for token in alpha_words]

        dependency_tokens = [
            token for token in doc if not token.is_space and not token.is_punct
        ]
        dependency_depths = [_dependency_depth(token) for token in dependency_tokens]
        dependency_lengths = [
            abs(token.i - token.head.i)
            for token in dependency_tokens
            if token.head.i != token.i
        ]
        verbal_heads = [token for token in doc if token.pos_ in {"VERB", "AUX"}]
        clause_heads = [
            token for token in verbal_heads if token.dep_ in CLAUSE_DEPS
        ]
        passive_heads = [
            token
            for token in verbal_heads
            if "Pass" in token.morph.get("Voice")
            or any(child.dep_ in {"auxpass", "nsubjpass"} for child in token.children)
        ]
        noun_tokens = [token for token in alpha_words if token.pos_ == "NOUN"]
        nominalizations = [
            token
            for token in noun_tokens
            if len(token.lower_) > 5
            and token.lower_.endswith(NOMINALIZATION_SUFFIXES)
        ]

        sentence_lengths = [
            sum(not token.is_space and not token.is_punct for token in sentence)
            for sentence in sentences
        ]
        sentence_mean = _mean(sentence_lengths)
        sentence_sd = (
            math.sqrt(
                sum((length - sentence_mean) ** 2 for length in sentence_lengths)
                / max(len(sentence_lengths) - 1, 1)
            )
            if len(sentence_lengths) > 1
            else 0.0
        )
        fragments = [
            sentence
            for sentence in sentences
            if not any(token.pos_ in {"VERB", "AUX"} for token in sentence)
        ]
        pos_sequences = [
            [token.pos_ for token in sentence if not token.is_space and not token.is_punct]
            for sentence in sentences
        ]
        pos_parallelism = [
            SequenceMatcher(None, left, right, autojunk=False).ratio()
            for left, right in zip(pos_sequences, pos_sequences[1:])
            if left and right
        ]

        normalized_text = re.sub(r"\s+", " ", text.lower()).strip()
        connective_counts = {
            group: sum(len(pattern.findall(normalized_text)) for pattern in patterns)
            for group, patterns in CONNECTIVE_PATTERNS.items()
        }
        sentence_openings = [
            re.sub(r"^[\W_]+", "", sentence.text.lower().strip())
            for sentence in sentences
        ]
        transition_openings = sum(
            any(_starts_with_marker(opening, marker) for marker in ALL_OPENING_MARKERS)
            for opening in sentence_openings
        )
        enumeration_openings = sum(
            any(_starts_with_marker(opening, marker) for marker in ENUMERATION_MARKERS)
            or bool(re.match(r"^\(?[0-9]+[.)]\s", opening))
            for opening in sentence_openings
        )
        summary_count = sum(
            len(_phrase_pattern(marker).findall(normalized_text))
            for marker in SUMMARY_MARKERS
        )

        entity_sets: list[set[str]] = []
        subject_sets: list[set[str]] = []
        content_sets: list[set[str]] = []
        for sentence in sentences:
            entity_sets.append(
                {
                    (token.lemma_ or token.lower_).lower()
                    for token in sentence
                    if token.pos_ in {"NOUN", "PROPN"} and token.is_alpha
                }
            )
            subject_sets.append(
                {
                    (token.lemma_ or token.lower_).lower()
                    for token in sentence
                    if token.dep_ in SUBJECT_DEPS and token.is_alpha
                }
            )
            content_sets.append(
                {
                    (token.lemma_ or token.lower_).lower()
                    for token in sentence
                    if token.pos_ in CONTENT_POS and token.is_alpha
                }
            )
        adjacent_entities = [
            _jaccard(left, right) for left, right in zip(entity_sets, entity_sets[1:])
        ]
        adjacent_subjects = [
            float(bool(left & right))
            for left, right in zip(subject_sets, subject_sets[1:])
            if left or right
        ]
        adjacent_content = [
            _jaccard(left, right) for left, right in zip(content_sets, content_sets[1:])
        ]

        parenthetical_count = min(text.count("("), text.count(")")) + len(
            re.findall(r"\s[—–]\s[^—–\n]+?\s[—–]\s", text)
        )
        paragraphs = [
            paragraph.strip()
            for paragraph in re.split(r"\n\s*\n", text)
            if paragraph.strip()
        ]
        entity_token_count = sum(len(entity) for entity in doc.ents)

        values = {
            "function_word_rate": _safe_ratio(function_count, len(alpha_words)),
            "pronoun_rate": _safe_ratio(pronoun_count, len(alpha_words)),
            "contraction_rate": _safe_ratio(contraction_count, len(alpha_words)),
            "mattr_50": _mattr(lemmas, window=50),
            "content_zipf_mean": _mean(content_zipf),
            "repeated_trigram_rate": _safe_ratio(repeated_trigrams, len(trigrams)),
            "abstract_lexicon_rate": _safe_ratio(
                sum(word in ABSTRACT_WORDS for word in lower_words), len(alpha_words)
            ),
            "evaluative_lexicon_rate": _safe_ratio(
                sum(word in EVALUATIVE_WORDS for word in lower_words), len(alpha_words)
            ),
            "emphasis_lexicon_rate": _safe_ratio(
                sum(word in EMPHASIS_WORDS for word in lower_words), len(alpha_words)
            ),
            "institutional_lexicon_rate": _safe_ratio(
                sum(word in INSTITUTIONAL_WORDS for word in lower_words), len(alpha_words)
            ),
            "dependency_depth_mean": _mean(dependency_depths),
            "dependency_length_mean": _mean(dependency_lengths),
            "clause_density": _safe_ratio(len(clause_heads), sentence_count),
            "coordination_density": _safe_ratio(
                sum(token.dep_ == "conj" for token in doc), sentence_count
            ),
            "subordination_density": _safe_ratio(
                sum(token.dep_ in SUBORDINATE_DEPS for token in doc), sentence_count
            ),
            "passive_clause_rate": _safe_ratio(len(passive_heads), len(verbal_heads)),
            "nominalization_rate": _safe_ratio(len(nominalizations), len(noun_tokens)),
            "sentence_length_cv": _safe_ratio(sentence_sd, sentence_mean),
            "fragment_rate": _safe_ratio(len(fragments), sentence_count),
            "apposition_density": _safe_ratio(
                sum(token.dep_ == "appos" for token in doc), sentence_count
            ),
            "parenthetical_density": _safe_ratio(parenthetical_count, sentence_count),
            "adjacent_pos_parallelism": _mean(pos_parallelism),
            **{
                f"{group}_connective_rate": _safe_ratio(count, sentence_count)
                for group, count in connective_counts.items()
            },
            "explicit_relation_rate": _safe_ratio(
                sum(connective_counts.values()), sentence_count
            ),
            "transition_opening_rate": _safe_ratio(
                transition_openings, sentence_count
            ),
            "enumeration_marker_rate": _safe_ratio(
                enumeration_openings, sentence_count
            ),
            "summary_marker_rate": _safe_ratio(summary_count, sentence_count),
            "entity_persistence": _mean(adjacent_entities),
            "subject_continuity": _mean(adjacent_subjects),
            "adjacent_content_overlap": _mean(adjacent_content),
            "word_count": float(word_count),
            "sentence_count": float(sentence_count),
            "paragraph_count": float(len(paragraphs)),
            "mean_sentence_length": sentence_mean,
            "named_entity_rate": _safe_ratio(entity_token_count, word_count),
            "number_rate": _safe_ratio(sum(token.like_num for token in words), word_count),
            "character_count": float(sum(not char.isspace() for char in text)),
        }
        missing = set(self.feature_names) - set(values)
        extra = set(values) - set(self.feature_names)
        if missing or extra:
            raise RuntimeError(
                f"Feature inventory mismatch: missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )
        if not all(math.isfinite(value) for value in values.values()):
            raise ValueError("Non-finite linguistic feature value")
        return values
