"""Unified measurements for the decodability--control gap.

The three stages intentionally use different observable units:

``D``  generator-held-out correlation between a hidden readout and its
       linguistic surface score;
``T``  fixed-token hidden transport along the target readout, relative to an
       orthogonal random-direction ensemble;
``B``  free-generation target selectivity, relative to the same frozen random
       direction.

The stages are not rescaled to look comparable.  Their signs and uncertainty,
not their raw magnitudes, establish the encoding--transport--behavior chain.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.analysis.causal_linguistics import paired_mean_inference
from src.runtime import stable_seed


CONCEPTS = ("lexical", "syntactic")
SURFACE_GROUPS = ("lexical", "syntactic", "discourse")


def _require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def encoding_observations(
    validation: pd.DataFrame,
    input_layers: Mapping[str, int],
) -> pd.DataFrame:
    """Collapse Human/AI test sides within each held-out generator.

    Averaging the two source sides avoids treating them as independent
    replications.  The held-out generator is the inferential unit.
    """
    _require_columns(
        validation,
        {
            "model",
            "representation",
            "layer",
            "group",
            "test_source",
            "holdout_type",
            "heldout_environment",
            "heldout_score_correlation",
        },
    )
    parts = []
    for model, layer in input_layers.items():
        selected = validation[
            validation["model"].eq(model)
            & validation["representation"].eq("mean_pool")
            & validation["layer"].eq(int(layer))
            & validation["holdout_type"].eq("generator")
            & validation["group"].isin(CONCEPTS)
        ].copy()
        if selected.empty:
            raise ValueError(f"No encoding rows for {model} at L{layer}")
        source_counts = selected.groupby(
            ["group", "heldout_environment"]
        )["test_source"].nunique()
        if not source_counts.eq(2).all():
            raise ValueError(
                f"{model} encoding rows do not contain both source sides"
            )
        collapsed = (
            selected.groupby(
                ["group", "heldout_environment"],
                as_index=False,
            )["heldout_score_correlation"]
            .mean()
            .rename(columns={
                "group": "concept",
                "heldout_environment": "observation_id",
                "heldout_score_correlation": "value",
            })
        )
        collapsed.insert(0, "model", model)
        collapsed.insert(2, "stage", "encoding")
        collapsed["input_layer"] = int(layer)
        parts.append(collapsed)
    return pd.concat(parts, ignore_index=True)


def transport_observations(
    transport: pd.DataFrame,
    *,
    model: str,
    downstream_layers: Mapping[str, int],
) -> pd.DataFrame:
    """Return sign-invariant target transport minus random-axis transport."""
    _require_columns(
        transport,
        {
            "prompt_id",
            "intervention",
            "intervention_kind",
            "readout",
            "layer",
            "transport_coefficient",
        },
    )
    rows = []
    for concept in CONCEPTS:
        downstream_layer = int(downstream_layers[concept])
        selected = transport[
            transport["layer"].eq(int(downstream_layer))
            & transport["readout"].eq(concept)
        ]
        learned = selected[
            selected["intervention"].eq(concept)
            & selected["intervention_kind"].eq("learned")
        ][["prompt_id", "transport_coefficient"]].rename(
            columns={"transport_coefficient": "learned"}
        )
        random = selected[
            selected["intervention_kind"].eq("random")
        ][["prompt_id", "transport_coefficient"]].copy()
        random["transport_coefficient"] = random[
            "transport_coefficient"
        ].abs()
        random = (
            random.groupby("prompt_id", as_index=False)[
                "transport_coefficient"
            ]
            .mean()
            .rename(columns={"transport_coefficient": "random_abs"})
        )
        merged = learned.merge(
            random,
            on="prompt_id",
            validate="one_to_one",
        )
        if merged.empty:
            raise ValueError(
                f"No transport observations for {model} {concept}"
            )
        for row in merged.itertuples(index=False):
            rows.append({
                "model": model,
                "concept": concept,
                "stage": "transport",
                "observation_id": str(row.prompt_id),
                "value": float(abs(row.learned) - row.random_abs),
                "downstream_layer": int(downstream_layer),
            })
    return pd.DataFrame(rows)


def _surface_selectivity(
    frame: pd.DataFrame,
    *,
    component: str,
    concept: str,
) -> pd.DataFrame:
    selected = frame[frame["component"].eq(component)]
    pivot = selected.pivot(
        index=["prompt_id", "prompt_index"],
        columns="group",
        values="signature_projection",
    )
    missing = set(SURFACE_GROUPS) - set(pivot.columns)
    if missing:
        raise ValueError(
            f"{component} lacks surface groups: {sorted(missing)}"
        )
    off_target = [
        group for group in SURFACE_GROUPS if group != concept
    ]
    values = pivot[concept] - pivot[off_target].mean(axis=1)
    return values.rename(component).reset_index()


def behavior_observations(
    projections: pd.DataFrame,
    *,
    model: str,
) -> pd.DataFrame:
    """Return concept selectivity minus prompt-matched random selectivity."""
    _require_columns(
        projections,
        {
            "component",
            "group",
            "prompt_id",
            "prompt_index",
            "signature_projection",
        },
    )
    rows = []
    for concept in CONCEPTS:
        learned = _surface_selectivity(
            projections,
            component=concept,
            concept=concept,
        )
        random = _surface_selectivity(
            projections,
            component="random",
            concept=concept,
        )
        merged = learned.merge(
            random,
            on=["prompt_id", "prompt_index"],
            validate="one_to_one",
        )
        if merged.empty:
            raise ValueError(
                f"No behavioral observations for {model} {concept}"
            )
        for row in merged.itertuples(index=False):
            rows.append({
                "model": model,
                "concept": concept,
                "stage": "behavior",
                "observation_id": str(row.prompt_id),
                "prompt_index": int(row.prompt_index),
                "value": float(
                    getattr(row, concept) - getattr(row, "random")
                ),
            })
    return pd.DataFrame(rows)


def summarize_gap(
    observations: pd.DataFrame,
    *,
    bootstrap: int,
    permutations: int,
    seed: int,
) -> pd.DataFrame:
    """Compute uncertainty for every model, concept, and stage."""
    _require_columns(
        observations,
        {"model", "concept", "stage", "observation_id", "value"},
    )
    rows = []
    for keys, part in observations.groupby(
        ["model", "concept", "stage"],
        sort=True,
    ):
        model, concept, stage = keys
        values = part["value"].to_numpy(dtype=float)
        inference = paired_mean_inference(
            values,
            bootstrap=bootstrap,
            permutations=permutations,
            seed=stable_seed(seed, model, concept, stage),
        )
        rows.append({
            "model": model,
            "concept": concept,
            "stage": stage,
            "n_observations": len(values),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            **inference,
        })
    return pd.DataFrame(rows)
