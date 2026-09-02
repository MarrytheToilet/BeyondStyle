"""Auditable token-level summaries of nullspace source coordinates.

The functions in this module operate on scalar token projections, never on
token frequencies alone.  Two design choices are central:

1. coordinates are standardized from pair-balanced document means so long
   continuations do not determine the scale; and
2. token-type contrasts retain only pairs in which the exact tokenizer token
   occurs in both the human and AI continuation.

The latter estimates a same-token contextual shift.  It is intentionally
different from a logit lens: no intermediate residual state is decoded as if
it were already in the final vocabulary space.
"""

from __future__ import annotations

import hashlib
import math
import unicodedata
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import rankdata, ttest_1samp


EPS = 1e-12
CORE_KEYS = ["model", "direction_scheme", "axis_family", "layer"]
TOKEN_CONTRAST_COLUMNS = [
    *CORE_KEYS,
    "token_id",
    "token_text",
    "coordinate_definition",
    "token_normalized",
    "token_class",
    "n_paired_samples",
    "context_shift_mean",
    "context_shift_ci_low",
    "context_shift_ci_high",
    "context_shift_effect_dz",
    "context_shift_positive_fraction",
    "context_shift_p",
    "context_shift_q",
    "generator_sign_agreement",
    "n_generator_environments",
    "domain_sign_agreement",
    "n_domain_environments",
    "source_balanced_affinity_mean",
    "source_balanced_affinity_ci_low",
    "source_balanced_affinity_ci_high",
    "source_balanced_affinity_p",
    "source_balanced_affinity_q",
]


def normalize_token_text(text: str) -> str:
    """Return a display-safe, model-local normalized token surface."""
    value = unicodedata.normalize("NFKC", str(text))
    value = "".join(
        character
        for character in value
        if unicodedata.category(character) not in {"Cc", "Cf"}
    )
    return " ".join(value.casefold().strip().split())


def token_class(text: str) -> str:
    """Coarse token class for filtering empirical vocabulary fields."""
    value = normalize_token_text(text)
    if not value:
        return "whitespace"
    has_letter = any(character.isalpha() for character in value)
    has_digit = any(character.isdigit() for character in value)
    if has_letter and has_digit:
        return "alphanumeric"
    if has_letter:
        return "word"
    if has_digit:
        return "number"
    if all(unicodedata.category(character).startswith("P") for character in value):
        return "punctuation"
    return "symbol"


def _validate_columns(frame: pd.DataFrame, names: Iterable[str]) -> None:
    missing = sorted(set(names) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing token-projection columns: {missing}")


def pair_balanced_standardize(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Add global and within-document standardized token coordinates.

    ``projection_z`` retains the document-level source offset and is used for
    source geometry and position trajectories. ``projection_within_document_z``
    subtracts each continuation's own token mean before applying the same
    pair-balanced scale; it is the appropriate coordinate for exact-token
    contextual comparisons because a global AI-document shift can no longer
    make every token appear locally AI-oriented.
    """
    _validate_columns(
        frame,
        [
            *CORE_KEYS,
            "sample_id",
            "source",
            "projection",
        ],
    )
    document = (
        frame.groupby(
            [*CORE_KEYS, "sample_id", "source"],
            observed=True,
            sort=False,
        )["projection"]
        .mean()
        .reset_index(name="document_projection")
    )
    scale = (
        document.groupby(CORE_KEYS, observed=True, sort=False)[
            "document_projection"
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(
            columns={
                "mean": "projection_center",
                "std": "projection_scale",
                "count": "n_documents",
            }
        )
    )
    scale["projection_scale"] = (
        scale["projection_scale"].fillna(0.0).clip(lower=EPS)
    )
    standardized = frame.merge(
        scale[CORE_KEYS + ["projection_center", "projection_scale"]],
        on=CORE_KEYS,
        how="left",
        validate="many_to_one",
    )
    standardized = standardized.merge(
        document[
            [
                *CORE_KEYS,
                "sample_id",
                "source",
                "document_projection",
            ]
        ],
        on=[*CORE_KEYS, "sample_id", "source"],
        how="left",
        validate="many_to_one",
    )
    standardized["projection_z"] = (
        standardized["projection"] - standardized["projection_center"]
    ) / standardized["projection_scale"]
    standardized["projection_within_document_z"] = (
        standardized["projection"]
        - standardized["document_projection"]
    ) / standardized["projection_scale"]
    return standardized, scale


def _binary_auc(human: np.ndarray, ai: np.ndarray) -> float:
    values = np.concatenate([human, ai])
    ranks = rankdata(values)
    n_human = len(human)
    n_ai = len(ai)
    if not n_human or not n_ai:
        return float("nan")
    ai_rank_sum = float(ranks[n_human:].sum())
    return (
        ai_rank_sum - n_ai * (n_ai + 1) / 2.0
    ) / (n_human * n_ai)


def _cohen_d(human: np.ndarray, ai: np.ndarray) -> float:
    if len(human) < 2 or len(ai) < 2:
        return float("nan")
    denominator = len(human) + len(ai) - 2
    pooled = math.sqrt(
        (
            (len(human) - 1) * float(np.var(human, ddof=1))
            + (len(ai) - 1) * float(np.var(ai, ddof=1))
        )
        / max(denominator, 1)
    )
    return float((np.mean(ai) - np.mean(human)) / max(pooled, EPS))


def layer_projection_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize mean-token and final-token source geometry by layer."""
    _validate_columns(
        frame,
        [
            *CORE_KEYS,
            "sample_id",
            "source",
            "token_index",
            "token_count",
            "projection_z",
        ],
    )
    rows: list[dict] = []
    for keys, part in frame.groupby(CORE_KEYS, observed=True, sort=True):
        base = dict(zip(CORE_KEYS, keys))
        for representation, selected in (
            ("continuation_mean", part),
            (
                "last_continuation_token",
                part[part["token_index"] == part["token_count"] - 1],
            ),
        ):
            sample_scores = (
                selected.groupby(
                    ["sample_id", "source"],
                    observed=True,
                )["projection_z"]
                .mean()
                .unstack("source")
                .dropna(subset=["human", "ai"])
            )
            human = sample_scores["human"].to_numpy(dtype=float)
            ai = sample_scores["ai"].to_numpy(dtype=float)
            delta = ai - human
            rows.append(
                {
                    **base,
                    "token_representation": representation,
                    "n_pairs": len(sample_scores),
                    "auc": _binary_auc(human, ai),
                    "cohen_d": _cohen_d(human, ai),
                    "paired_delta_mean": float(np.mean(delta)),
                    "paired_delta_sd": (
                        float(np.std(delta, ddof=1))
                        if len(delta) > 1
                        else float("nan")
                    ),
                    "positive_pair_fraction": float(np.mean(delta > 0)),
                }
            )
    return pd.DataFrame(rows)


def _stable_seed(seed: int, values: tuple) -> int:
    digest = hashlib.sha256(
        (str(seed) + "|" + "|".join(map(str, values))).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def _bootstrap_means(
    values: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return float("nan"), float("nan")
    if len(values) == 1 or repetitions <= 0:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(seed)
    lower = []
    remaining = int(repetitions)
    # Bounded chunks prevent a very common token from allocating a large
    # repetitions × pairs matrix.
    while remaining:
        size = min(remaining, 256)
        indices = rng.integers(0, len(values), size=(size, len(values)))
        lower.append(values[indices].mean(axis=1))
        remaining -= size
    draws = np.concatenate(lower)
    return tuple(np.quantile(draws, [0.025, 0.975]).astype(float))


def position_projection_trajectory(
    frame: pd.DataFrame,
    *,
    bins: int = 10,
    bootstrap_repetitions: int = 500,
    seed: int = 20260725,
) -> pd.DataFrame:
    """Estimate a pair-balanced Human/AI coordinate trajectory."""
    if bins < 2:
        raise ValueError("bins must be at least two")
    _validate_columns(
        frame,
        [
            *CORE_KEYS,
            "sample_id",
            "source",
            "relative_position",
            "projection_z",
        ],
    )
    work = frame.copy()
    work["position_bin"] = np.minimum(
        (work["relative_position"].clip(0.0, 1.0) * bins).astype(int),
        bins - 1,
    )
    sample_bin = (
        work.groupby(
            [*CORE_KEYS, "sample_id", "source", "position_bin"],
            observed=True,
            sort=False,
        )["projection_z"]
        .mean()
        .reset_index()
    )
    rows: list[dict] = []
    for keys, part in sample_bin.groupby(
        [*CORE_KEYS, "position_bin"],
        observed=True,
        sort=True,
    ):
        base = dict(zip([*CORE_KEYS, "position_bin"], keys))
        paired = (
            part.pivot(index="sample_id", columns="source", values="projection_z")
            .dropna(subset=["human", "ai"])
        )
        human = paired["human"].to_numpy(dtype=float)
        ai = paired["ai"].to_numpy(dtype=float)
        delta = ai - human
        group_seed = _stable_seed(seed, keys)
        human_ci = _bootstrap_means(
            human,
            repetitions=bootstrap_repetitions,
            seed=group_seed,
        )
        ai_ci = _bootstrap_means(
            ai,
            repetitions=bootstrap_repetitions,
            seed=group_seed + 1,
        )
        delta_ci = _bootstrap_means(
            delta,
            repetitions=bootstrap_repetitions,
            seed=group_seed + 2,
        )
        rows.append(
            {
                **base,
                "position_start": base["position_bin"] / bins,
                "position_end": (base["position_bin"] + 1) / bins,
                "position_midpoint": (base["position_bin"] + 0.5) / bins,
                "n_pairs": len(paired),
                "human_mean": float(np.mean(human)),
                "human_ci_low": human_ci[0],
                "human_ci_high": human_ci[1],
                "ai_mean": float(np.mean(ai)),
                "ai_ci_low": ai_ci[0],
                "ai_ci_high": ai_ci[1],
                "paired_delta_mean": float(np.mean(delta)),
                "paired_delta_ci_low": delta_ci[0],
                "paired_delta_ci_high": delta_ci[1],
                "positive_pair_fraction": float(np.mean(delta > 0)),
            }
        )
    return pd.DataFrame(rows)


def _sign_agreement_by_environment(
    values: pd.DataFrame,
    column: str,
    reference: float,
    *,
    minimum: int = 3,
) -> tuple[float, int]:
    means = [
        float(part["context_shift"].mean())
        for _, part in values.groupby(column, observed=True)
        if len(part) >= minimum
    ]
    if not means or abs(reference) <= EPS:
        return float("nan"), len(means)
    return float(np.mean(np.sign(means) == np.sign(reference))), len(means)


def _benjamini_hochberg(values: pd.Series) -> pd.Series:
    array = values.to_numpy(dtype=np.float64)
    order = np.argsort(array)
    ranked = array[order]
    adjusted = ranked * len(array) / np.arange(1, len(array) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return pd.Series(result, index=values.index)


def _one_sample_p(values: np.ndarray) -> float:
    """Stable two-sided one-sample p-value for constant vectors."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2:
        return 1.0
    if float(np.std(values, ddof=1)) <= EPS:
        return 0.0 if abs(float(np.mean(values))) > EPS else 1.0
    return float(ttest_1samp(values, 0.0).pvalue)


def paired_token_context_contrasts(
    frame: pd.DataFrame,
    *,
    min_paired_samples: int = 10,
    bootstrap_repetitions: int = 500,
    seed: int = 20260725,
    coordinate_column: str = "projection_within_document_z",
) -> pd.DataFrame:
    """Measure same-token contextual shifts on exact within-pair matches.

    Occurrences are first averaged inside each sample/source. A token then
    contributes only for paired continuations in which it occurs on both
    sides. This prevents source-specific token frequency from masquerading as
    a contextual hidden-state difference.
    """
    _validate_columns(
        frame,
        [
            *CORE_KEYS,
            "sample_id",
            "source",
            "generator",
            "domain",
            "token_id",
            "token_text",
            coordinate_column,
        ],
    )
    if min_paired_samples < 2:
        raise ValueError("min_paired_samples must be at least two")
    occurrence = (
        frame.groupby(
            [
                *CORE_KEYS,
                "sample_id",
                "source",
                "generator",
                "domain",
                "token_id",
                "token_text",
            ],
            observed=True,
            sort=False,
        )[coordinate_column]
        .mean()
        .reset_index()
    )
    identity = [
        *CORE_KEYS,
        "sample_id",
        "generator",
        "domain",
        "token_id",
        "token_text",
    ]
    paired = (
        occurrence.pivot_table(
            index=identity,
            columns="source",
            values=coordinate_column,
            aggfunc="first",
            observed=True,
        )
        .dropna(subset=["human", "ai"])
        .reset_index()
    )
    paired["context_shift"] = paired["ai"] - paired["human"]
    paired["source_balanced_affinity"] = (
        paired["ai"] + paired["human"]
    ) / 2.0

    rows: list[dict] = []
    token_keys = [*CORE_KEYS, "token_id", "token_text"]
    for keys, part in paired.groupby(token_keys, observed=True, sort=True):
        if len(part) < min_paired_samples:
            continue
        base = dict(zip(token_keys, keys))
        shift = part["context_shift"].to_numpy(dtype=float)
        affinity = part["source_balanced_affinity"].to_numpy(dtype=float)
        group_seed = _stable_seed(seed, keys)
        shift_ci = _bootstrap_means(
            shift,
            repetitions=bootstrap_repetitions,
            seed=group_seed,
        )
        affinity_ci = _bootstrap_means(
            affinity,
            repetitions=bootstrap_repetitions,
            seed=group_seed + 1,
        )
        shift_mean = float(np.mean(shift))
        generator_agreement, n_generators = _sign_agreement_by_environment(
            part,
            "generator",
            shift_mean,
        )
        domain_agreement, n_domains = _sign_agreement_by_environment(
            part,
            "domain",
            shift_mean,
        )
        display = str(base["token_text"])
        rows.append(
            {
                **base,
                "coordinate_definition": coordinate_column,
                "token_normalized": normalize_token_text(display),
                "token_class": token_class(display),
                "n_paired_samples": len(part),
                "context_shift_mean": shift_mean,
                "context_shift_ci_low": shift_ci[0],
                "context_shift_ci_high": shift_ci[1],
                "context_shift_effect_dz": shift_mean
                / max(float(np.std(shift, ddof=1)), EPS),
                "context_shift_positive_fraction": float(np.mean(shift > 0)),
                "context_shift_p": _one_sample_p(shift),
                "generator_sign_agreement": generator_agreement,
                "n_generator_environments": n_generators,
                "domain_sign_agreement": domain_agreement,
                "n_domain_environments": n_domains,
                "source_balanced_affinity_mean": float(np.mean(affinity)),
                "source_balanced_affinity_ci_low": affinity_ci[0],
                "source_balanced_affinity_ci_high": affinity_ci[1],
                "source_balanced_affinity_p": _one_sample_p(affinity),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return pd.DataFrame(columns=TOKEN_CONTRAST_COLUMNS)
    result["context_shift_q"] = (
        result.groupby(CORE_KEYS, observed=True)["context_shift_p"]
        .transform(_benjamini_hochberg)
    )
    result["source_balanced_affinity_q"] = (
        result.groupby(CORE_KEYS, observed=True)[
            "source_balanced_affinity_p"
        ].transform(_benjamini_hochberg)
    )
    return result.sort_values(
        [*CORE_KEYS, "context_shift_mean"],
        ascending=[True, True, True, True, False],
    ).reset_index(drop=True)


def empirical_vocabulary_field(
    contrasts: pd.DataFrame,
    *,
    top_k: int = 20,
    wordlike_only: bool = True,
) -> pd.DataFrame:
    """Select a compact, source-balanced empirical token field.

    Ranking uses the midpoint of matched Human/AI contexts, not the source
    difference. Confidence intervals must exclude zero. Consequently, the
    output describes which exact tokenizer tokens occupy the positive or
    negative side of the hidden coordinate while balancing source.
    """
    if contrasts.empty:
        return contrasts.assign(
            field_side=pd.Series(dtype="object"),
            field_rank=pd.Series(dtype="int64"),
        )
    work = contrasts.copy()
    if wordlike_only:
        work = work[work["token_class"].isin(["word", "alphanumeric"])]
    work = work[
        (
            (work["source_balanced_affinity_ci_low"] > 0)
            | (work["source_balanced_affinity_ci_high"] < 0)
        )
        & work["token_normalized"].ne("")
    ]
    selections = []
    for _, part in work.groupby(CORE_KEYS, observed=True, sort=True):
        positive = part[
            part["source_balanced_affinity_mean"] > 0
        ].nlargest(top_k, "source_balanced_affinity_mean").copy()
        positive["field_side"] = "AI-oriented"
        negative = part[
            part["source_balanced_affinity_mean"] < 0
        ].nsmallest(top_k, "source_balanced_affinity_mean").copy()
        negative["field_side"] = "Human-oriented"
        for selected in (positive, negative):
            selected["field_rank"] = np.arange(1, len(selected) + 1)
            selections.append(selected)
    return (
        pd.concat(selections, ignore_index=True)
        if selections
        else work.iloc[0:0].copy()
    )
