import numpy as np

from scripts.audit_semantic_alignment import (
    validate_pairs,
    within_source_derangement,
)


def pair(sample_id, source, generator, machine):
    return {
        "sample_id": sample_id,
        "prefix": f"Prefix {sample_id}",
        "human_continuation": f"Human {sample_id}",
        "ai_continuation": machine,
        "source": source,
        "domain": f"{source}-domain",
        "generation": {"generator_model": generator},
    }


def test_validate_pairs_restores_plan_order():
    plan = [
        pair("a", "news", "g1", "planned a"),
        pair("b", "books", "g2", "planned b"),
    ]
    control = [
        pair("b", "books", "g2", "prefix b"),
        pair("a", "news", "g1", "prefix a"),
    ]

    ordered = validate_pairs(plan, control)

    assert [row["sample_id"] for row in ordered] == ["a", "b"]


def test_within_source_derangement_has_no_fixed_points_or_crossings():
    sources = np.asarray(["a"] * 5 + ["b"] * 5)
    permutation = within_source_derangement(
        sources,
        np.random.default_rng(7),
    )

    assert not np.any(permutation == np.arange(len(sources)))
    assert np.array_equal(sources[permutation], sources)
