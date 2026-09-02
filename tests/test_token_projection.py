import numpy as np
import pandas as pd

from src.analysis.token_projection import (
    empirical_vocabulary_field,
    layer_projection_summary,
    normalize_token_text,
    pair_balanced_standardize,
    paired_token_context_contrasts,
    position_projection_trajectory,
    token_class,
)


def synthetic_records():
    rows = []
    for sample in range(12):
        for source in ("human", "ai"):
            source_shift = (
                0.0
                if source == "human"
                else 2.0 + 0.02 * (sample % 3)
            )
            for token_index, (token_id, token_text) in enumerate(
                [
                    (10, " shared"),
                    (11, " anchor"),
                    (12, " peak"),
                    (20 + sample, " unique"),
                ]
            ):
                rows.append(
                    {
                        "model": "toy",
                        "direction_scheme": "generator_holdout",
                        "axis_family": "mean_pool",
                        "layer": 4,
                        "sample_id": f"s{sample}",
                        "source": source,
                        "generator": f"g{sample % 3}",
                        "domain": f"d{sample % 2}",
                        "token_id": token_id,
                        "token_text": token_text,
                        "token_index": token_index,
                        "token_count": 4,
                        "relative_position": token_index / 3,
                        "projection": (
                            source_shift
                            + token_index
                            + (
                                [0.0, 0.1, 0.8, -0.2][token_index]
                                if source == "ai"
                                else 0.0
                            )
                            + (sample - 5.5) * 0.01
                        ),
                    }
                )
    return pd.DataFrame(rows)


def test_token_normalization_and_classification():
    assert normalize_token_text("  Café\u200b ") == "café"
    assert token_class("  Café ") == "word"
    assert token_class("42") == "number"
    assert token_class("...") == "punctuation"


def test_pair_balanced_scale_is_based_on_documents_not_token_rows():
    frame = synthetic_records()
    standardized, scale = pair_balanced_standardize(frame)

    document_means = (
        frame.groupby(["sample_id", "source"])["projection"].mean()
    )
    assert np.isclose(
        scale.iloc[0]["projection_center"],
        document_means.mean(),
    )
    assert np.isfinite(standardized["projection_z"]).all()


def test_layer_and_position_summaries_recover_positive_ai_shift():
    standardized, _ = pair_balanced_standardize(synthetic_records())
    layer = layer_projection_summary(standardized)
    trajectory = position_projection_trajectory(
        standardized,
        bins=3,
        bootstrap_repetitions=20,
    )

    assert (layer["auc"] == 1.0).all()
    assert (layer["paired_delta_mean"] > 0).all()
    assert (trajectory["paired_delta_mean"] > 0).all()
    assert (trajectory["paired_delta_ci_low"] > 0).all()


def test_same_token_contrast_requires_occurrence_on_both_pair_sides():
    standardized, _ = pair_balanced_standardize(synthetic_records())
    contrasts = paired_token_context_contrasts(
        standardized,
        min_paired_samples=10,
        bootstrap_repetitions=30,
    )

    assert set(contrasts["token_id"]) == {10, 11, 12}
    by_token = contrasts.set_index("token_id")
    assert by_token.loc[10, "context_shift_ci_high"] < 0
    assert by_token.loc[12, "context_shift_ci_low"] > 0
    assert (contrasts["generator_sign_agreement"] == 1.0).all()


def test_empirical_field_has_both_coordinate_sides():
    standardized, _ = pair_balanced_standardize(synthetic_records())
    contrasts = paired_token_context_contrasts(
        standardized,
        min_paired_samples=10,
        bootstrap_repetitions=30,
    )
    field = empirical_vocabulary_field(
        contrasts,
        top_k=1,
        wordlike_only=True,
    )

    assert set(field["field_side"]) == {"AI-oriented", "Human-oriented"}
    assert set(field["token_id"]) == {10, 12}


def test_empty_same_token_result_preserves_output_schema():
    standardized, _ = pair_balanced_standardize(synthetic_records())
    contrasts = paired_token_context_contrasts(
        standardized,
        min_paired_samples=100,
        bootstrap_repetitions=5,
    )
    field = empirical_vocabulary_field(contrasts)

    assert contrasts.empty
    assert "context_shift_q" in contrasts
    assert field.empty
    assert {"field_side", "field_rank"} <= set(field)
