import math

import spacy

from src.linguistics import (
    FEATURE_DEFINITIONS,
    EnglishLinguisticFeatureExtractor,
    feature_dictionary,
)


def test_feature_inventory_is_unique_and_grouped():
    definitions = feature_dictionary()
    names = [item["name"] for item in definitions]
    assert len(names) == len(set(names))
    assert {item["group"] for item in definitions} == {
        "lexical",
        "syntactic",
        "discourse",
        "nuisance",
    }


def test_english_features_are_complete_and_finite():
    nlp = spacy.load("en_core_web_sm")
    text = (
        "First, the committee reviewed the proposal. "
        "However, it rejected the plan because costs increased. "
        "As a result, the organization adopted a simpler framework."
    )
    values = EnglishLinguisticFeatureExtractor().extract(nlp(text), text)
    assert set(values) == {item.name for item in FEATURE_DEFINITIONS}
    assert all(math.isfinite(value) for value in values.values())
    assert values["contrast_connective_rate"] > 0
    assert values["causal_connective_rate"] > 0
    assert values["transition_opening_rate"] > 0
    assert values["sentence_count"] == 3


def test_empty_like_text_does_not_produce_nan():
    nlp = spacy.load("en_core_web_sm")
    text = "..."
    values = EnglishLinguisticFeatureExtractor().extract(nlp(text), text)
    assert all(math.isfinite(value) for value in values.values())
