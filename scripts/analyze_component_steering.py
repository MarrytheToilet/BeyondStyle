#!/usr/bin/env python
"""Analyze symmetric component steering against the language signature."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import platform
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import spacy
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.causal_linguistics import (  # noqa: E402
    axis_ensemble_calibration,
    paired_mean_inference,
    set_overlap_metrics,
    source_signature_projection,
)
from src.linguistics import (  # noqa: E402
    FEATURE_DEFINITIONS,
    EnglishLinguisticFeatureExtractor,
)


DEFAULT_COMPONENTS = ("source", "explained", "residual", "random")
KNOWN_COMPONENTS = (
    "source",
    "explained",
    "lexical",
    "syntactic",
    "discourse",
    "residual",
    "random",
    "random_01",
    "random_02",
    "random_03",
    "random_04",
    "random_05",
    "random_06",
    "random_07",
)
GROUPS = ("lexical", "syntactic", "discourse")
GUARDRAILS = (
    ("prompt_content_recall", "content", None),
    ("prompt_content_precision", "content", None),
    ("prompt_entity_recall", "content", "prompt_has_entities"),
    ("prompt_number_recall", "content", "prompt_has_numbers"),
    ("word_count", "quality", None),
    ("repeated_trigram_rate", "quality", None),
    ("fragment_rate", "quality", None),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument(
        "--observational-features",
        type=Path,
        default=Path(
            "outputs/en/05_linguistic/features/"
            "paired_feature_differences.csv"
        ),
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument(
        "--components",
        nargs="+",
        choices=KNOWN_COMPONENTS,
        default=list(DEFAULT_COMPONENTS),
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--semantic-model",
        type=Path,
        help=(
            "Optional local SentenceTransformer checkpoint for paired-output "
            "semantic-similarity guardrails."
        ),
    )
    parser.add_argument("--semantic-device", default="cpu")
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--endpoint-strength",
        type=float,
        default=1.0,
        help=(
            "Positive magnitude of the paired symmetric steering endpoints "
            "to compare (default: 1.0)."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_records(
    input_dir: Path,
    components: list[str],
    endpoint_strength: float,
) -> list[dict[str, Any]]:
    negative_key = str(float(-endpoint_strength))
    positive_key = str(float(endpoint_strength))
    expected_outputs = {negative_key, positive_key}
    records: list[dict[str, Any]] = []
    prompt_sets: dict[str, set[str]] = {}
    for component in components:
        path = input_dir / f"generation_component_{component}.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        prompt_sets[component] = {str(row["prompt_id"]) for row in rows}
        for row in rows:
            outputs = row["outputs"]
            if set(outputs) != expected_outputs:
                raise ValueError(
                    f"{path} prompt {row['prompt_id']} lacks exact "
                    f"±{endpoint_strength:g} outputs"
                )
            for strength in (negative_key, positive_key):
                records.append(
                    {
                        "component": component,
                        "prompt_id": str(row["prompt_id"]),
                        "prompt_index": int(row["prompt_index"]),
                        "strength": float(strength),
                        "prompt": row["prompt"],
                        "text": outputs[strength]["text"],
                    }
                )
    if len({frozenset(values) for values in prompt_sets.values()}) != 1:
        raise ValueError(f"Component prompt sets differ: {prompt_sets}")
    return records


def audit_hooks(
    input_dir: Path,
    components: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for component in components:
        work = input_dir / f".work_component_{component}"
        batches = sorted(work.glob("batch_*.json"))
        conditions = []
        for path in batches:
            payload = json.loads(path.read_text(encoding="utf-8"))
            for strength, stats in payload["stats_by_strength"].items():
                steering = stats["steering"]
                hook = steering["hook_stats"]
                conditions.append(
                    {
                        "batch": path.name,
                        "strength": float(strength),
                        "position_mode": steering["position_mode"],
                        "prefill_calls_skipped": hook["prefill_calls_skipped"],
                        "steered_calls": hook["steered_calls"],
                        "steered_token_positions": hook[
                            "steered_token_positions"
                        ],
                    }
                )
        passed = bool(conditions) and all(
            row["position_mode"] == "decode_only"
            and row["prefill_calls_skipped"] >= 1
            and row["steered_calls"] > 0
            and row["steered_token_positions"] > 0
            for row in conditions
        )
        result[component] = {
            "passed": passed,
            "n_conditions": len(conditions),
            "conditions": conditions,
        }
        if not passed:
            raise RuntimeError(f"Decode-only hook audit failed for {component}")
    return result


def _content_terms(doc: Any) -> set[str]:
    return {
        (token.lemma_ or token.lower_).lower()
        for token in doc
        if token.is_alpha
        and not token.is_stop
        and token.pos_ in {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
    }


def _entity_terms(doc: Any) -> set[str]:
    return {
        " ".join(entity.text.lower().split())
        for entity in doc.ents
        if entity.text.strip()
    }


def _number_terms(doc: Any) -> set[str]:
    return {
        token.lower_
        for token in doc
        if token.like_num and token.text.strip()
    }


def prompt_overlap_guardrails(prompt_doc: Any, output_doc: Any) -> dict[str, Any]:
    content_recall, content_precision, _ = set_overlap_metrics(
        _content_terms(prompt_doc),
        _content_terms(output_doc),
    )
    entity_recall, _, has_entities = set_overlap_metrics(
        _entity_terms(prompt_doc),
        _entity_terms(output_doc),
    )
    number_recall, _, has_numbers = set_overlap_metrics(
        _number_terms(prompt_doc),
        _number_terms(output_doc),
    )
    return {
        "prompt_content_recall": content_recall,
        "prompt_content_precision": content_precision,
        "prompt_entity_recall": entity_recall,
        "prompt_number_recall": number_recall,
        "prompt_has_entities": has_entities,
        "prompt_has_numbers": has_numbers,
    }


def contrast_pairs_for_components(
    components: list[str],
) -> list[tuple[str, str]]:
    """Return interpretable planned contrasts available in one run."""
    requested = set(components)
    candidates = [
        ("explained", "source"),
        ("explained", "residual"),
        ("explained", "random"),
        ("lexical", "source"),
        ("syntactic", "source"),
        ("discourse", "source"),
        ("lexical", "residual"),
        ("syntactic", "residual"),
        ("discourse", "residual"),
        ("lexical", "random"),
        ("syntactic", "random"),
        ("discourse", "random"),
        ("source", "random"),
        ("residual", "random"),
    ]
    return [
        pair
        for pair in candidates
        if pair[0] in requested and pair[1] in requested
    ]


def semantic_pair_guardrails(
    text_features: pd.DataFrame,
    *,
    model_path: Path,
    device: str,
    batch_size: int,
    bootstrap: int,
    permutations: int,
    seed: int,
    endpoint_strength: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Compare symmetric outputs semantically against random steering."""
    from sentence_transformers import SentenceTransformer
    import sentence_transformers

    encoder = SentenceTransformer(
        str(model_path.expanduser().resolve()),
        device=device,
        local_files_only=True,
    )
    embeddings = encoder.encode(
        text_features["text"].tolist(),
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    row_embeddings = {
        (
            str(row.component),
            str(row.prompt_id),
            float(row.strength),
        ): embeddings[index]
        for index, row in enumerate(text_features.itertuples(index=False))
    }
    pair_rows = []
    for (component, prompt_id), subset in text_features.groupby(
        ["component", "prompt_id"],
        sort=False,
    ):
        negative = row_embeddings[
            (component, prompt_id, -endpoint_strength)
        ]
        positive = row_embeddings[
            (component, prompt_id, endpoint_strength)
        ]
        pair_rows.append(
            {
                "component": component,
                "prompt_id": prompt_id,
                "semantic_similarity": float(negative @ positive),
                "exact_text_identical": bool(
                    subset.loc[
                        subset["strength"] == -endpoint_strength,
                        "text",
                    ].iloc[0]
                    == subset.loc[
                        subset["strength"] == endpoint_strength,
                        "text",
                    ].iloc[0]
                ),
            }
        )
    pairs = pd.DataFrame(pair_rows)
    summaries = []
    for component_number, (component, subset) in enumerate(
        pairs.groupby("component", sort=False)
    ):
        inference = paired_mean_inference(
            subset["semantic_similarity"].to_numpy(),
            bootstrap=bootstrap,
            permutations=permutations,
            seed=seed + 5000 + component_number,
        )
        summaries.append(
            {
                "component": component,
                "n_prompts": len(subset),
                "minimum_similarity": float(
                    subset["semantic_similarity"].min()
                ),
                "exact_text_identity_rate": float(
                    subset["exact_text_identical"].mean()
                ),
                **{
                    key: value
                    for key, value in inference.items()
                    if key
                    in {
                        "mean",
                        "median",
                        "ci_low",
                        "ci_high",
                    }
                },
            }
        )
    summary = pd.DataFrame(summaries)

    contrasts = []
    if "random" in set(pairs["component"]):
        pivot = pairs.pivot(
            index="prompt_id",
            columns="component",
            values="semantic_similarity",
        )
        for component_number, component in enumerate(pivot.columns):
            if component == "random":
                continue
            values = (pivot[component] - pivot["random"]).to_numpy()
            contrasts.append(
                {
                    "component": component,
                    "reference": "random",
                    "n_prompts": len(values),
                    **paired_mean_inference(
                        values,
                        bootstrap=bootstrap,
                        permutations=permutations,
                        seed=seed + 6000 + component_number,
                    ),
                }
            )
    return (
        pairs,
        summary,
        pd.DataFrame(contrasts),
        sentence_transformers.__version__,
    )


def main() -> None:
    args = parse_args()
    if args.endpoint_strength <= 0:
        raise ValueError("--endpoint-strength must be positive")
    if len(args.components) != len(set(args.components)):
        raise ValueError("--components must not contain duplicates")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    records = load_records(
        args.input_dir,
        args.components,
        args.endpoint_strength,
    )
    hook_audit = audit_hooks(args.input_dir, args.components)
    nlp = spacy.load(args.spacy_model)
    extractor = EnglishLinguisticFeatureExtractor()
    prompts = {
        row["prompt_id"]: row["prompt"]
        for row in records
    }
    prompt_documents = {
        prompt_id: doc
        for prompt_id, doc in zip(
            prompts,
            nlp.pipe(prompts.values(), batch_size=args.batch_size),
        )
    }
    documents = nlp.pipe(
        [row["text"] for row in records],
        batch_size=args.batch_size,
    )
    feature_rows: list[dict[str, Any]] = []
    for row, doc in tqdm(
        zip(records, documents),
        total=len(records),
        desc="component features",
    ):
        feature_rows.append(
            {
                **{key: value for key, value in row.items() if key != "text"},
                "text": row["text"],
                **extractor.extract(doc, row["text"]),
                **prompt_overlap_guardrails(
                    prompt_documents[row["prompt_id"]],
                    doc,
                ),
            }
        )
    text_features = pd.DataFrame(feature_rows)
    feature_names = [item.name for item in FEATURE_DEFINITIONS]
    difference_rows: list[dict[str, Any]] = []
    for (component, prompt_id), subset in text_features.groupby(
        ["component", "prompt_id"],
        sort=False,
    ):
        negative = subset[
            subset["strength"] == -args.endpoint_strength
        ].iloc[0]
        positive = subset[
            subset["strength"] == args.endpoint_strength
        ].iloc[0]
        difference_rows.append(
            {
                "component": component,
                "prompt_id": prompt_id,
                "prompt_index": int(positive["prompt_index"]),
                "exact_text_identical": positive["text"] == negative["text"],
                **{
                    name: float(positive[name] - negative[name])
                    for name in feature_names
                },
            }
        )
    differences = pd.DataFrame(difference_rows)
    observational = pd.read_csv(args.observational_features)

    group_rows: list[dict[str, Any]] = []
    prompt_projections: list[pd.DataFrame] = []
    names_by_group = {
        group: [
            item.name for item in FEATURE_DEFINITIONS if item.group == group
        ]
        for group in GROUPS
    }
    for component_number, component in enumerate(args.components):
        component_rows = differences[differences["component"] == component]
        for group_number, group in enumerate(GROUPS):
            names = names_by_group[group]
            projections, cosine = source_signature_projection(
                observational[names].to_numpy(),
                component_rows[names].to_numpy(),
            )
            inference = paired_mean_inference(
                projections,
                bootstrap=args.bootstrap,
                permutations=args.permutations,
                seed=args.seed + 100 * component_number + group_number,
            )
            group_rows.append(
                {
                    "component": component,
                    "group": group,
                    "n_prompts": len(component_rows),
                    "mean_effect_cosine_to_source_signature": cosine,
                    **{
                        f"signature_projection_{key}": value
                        for key, value in inference.items()
                    },
                    "exact_text_identity_rate": float(
                        component_rows["exact_text_identical"].mean()
                    ),
                }
            )
            prompt_projections.append(
                pd.DataFrame(
                    {
                        "component": component,
                        "group": group,
                        "prompt_id": component_rows["prompt_id"].to_numpy(),
                        "prompt_index": component_rows[
                            "prompt_index"
                        ].to_numpy(),
                        "signature_projection": projections,
                    }
                )
            )
    group_statistics = pd.DataFrame(group_rows)
    projection_table = pd.concat(prompt_projections, ignore_index=True)

    contrasts = []
    contrast_pairs = contrast_pairs_for_components(args.components)
    for group_number, group in enumerate(GROUPS):
        subset = projection_table[projection_table["group"] == group]
        pivot = subset.pivot(
            index="prompt_id",
            columns="component",
            values="signature_projection",
        )
        for contrast_number, (left, right) in enumerate(contrast_pairs):
            values = (pivot[left] - pivot[right]).to_numpy()
            inference = paired_mean_inference(
                values,
                bootstrap=args.bootstrap,
                permutations=args.permutations,
                seed=args.seed + 1000 + 100 * group_number + contrast_number,
            )
            contrasts.append(
                {
                    "group": group,
                    "contrast": f"{left}_minus_{right}",
                    "n_prompts": len(values),
                    **inference,
                }
            )
    contrast_table = pd.DataFrame(contrasts)

    selectivity_rows: list[dict[str, Any]] = []
    for component_number, component in enumerate(args.components):
        if component not in GROUPS:
            continue
        subset = projection_table[
            projection_table["component"] == component
        ].pivot(
            index="prompt_id",
            columns="group",
            values="signature_projection",
        )
        off_target = [
            group for group in GROUPS if group != component
        ]
        selectivity = (
            subset[component] - subset[off_target].mean(axis=1)
        ).to_numpy()
        inference = paired_mean_inference(
            selectivity,
            bootstrap=args.bootstrap,
            permutations=args.permutations,
            seed=args.seed + 2000 + component_number,
        )
        selectivity_rows.append(
            {
                "component": component,
                "target_group": component,
                "off_target_groups": ",".join(off_target),
                "n_prompts": len(selectivity),
                **inference,
            }
        )
    selectivity_statistics = pd.DataFrame(selectivity_rows)

    random_components = [
        component
        for component in args.components
        if component == "random" or component.startswith("random_")
    ]
    behavior_null_rows: list[dict[str, Any]] = []
    behavior_null_axis_rows: list[dict[str, Any]] = []
    if len(random_components) >= 2:
        component_views = {
            component: projection_table[
                projection_table["component"] == component
            ].pivot(
                index="prompt_id",
                columns="group",
                values="signature_projection",
            )
            for component in args.components
        }
        for concept_number, concept in enumerate(
            component for component in GROUPS if component in args.components
        ):
            off_target = [group for group in GROUPS if group != concept]
            estimands = {
                "target_projection": (
                    component_views[concept][concept]
                ),
                "component_selectivity": (
                    component_views[concept][concept]
                    - component_views[concept][off_target].mean(axis=1)
                ),
            }
            for estimand_number, (estimand, learned_series) in enumerate(
                estimands.items()
            ):
                prompt_ids = learned_series.index
                random_axis_values = []
                for random_component in random_components:
                    random_view = component_views[random_component].loc[
                        prompt_ids
                    ]
                    values = (
                        random_view[concept]
                        if estimand == "target_projection"
                        else (
                            random_view[concept]
                            - random_view[off_target].mean(axis=1)
                        )
                    ).to_numpy(dtype=float)
                    random_axis_values.append(values)
                    behavior_null_axis_rows.append(
                        {
                            "concept": concept,
                            "estimand": estimand,
                            "axis": random_component,
                            "n_prompts": len(values),
                            "signed_mean": float(values.mean()),
                            "abs_axis_mean": float(abs(values.mean())),
                        }
                    )
                learned_values = learned_series.to_numpy(dtype=float)
                random_matrix = np.stack(random_axis_values)
                inference = paired_mean_inference(
                    learned_values,
                    bootstrap=args.bootstrap,
                    permutations=args.permutations,
                    seed=(
                        args.seed
                        + 2500
                        + 100 * concept_number
                        + estimand_number
                    ),
                )
                behavior_null_rows.append(
                    {
                        "concept": concept,
                        "estimand": estimand,
                        "n_prompts": len(learned_values),
                        **{
                            f"learned_{key}": value
                            for key, value in inference.items()
                        },
                        **axis_ensemble_calibration(
                            learned_values,
                            random_matrix,
                        ),
                    }
                )
    behavior_null_summary = pd.DataFrame(behavior_null_rows)
    behavior_null_axes = pd.DataFrame(behavior_null_axis_rows)

    guardrail_rows: list[dict[str, Any]] = []
    for component_number, component in enumerate(args.components):
        component_rows = text_features[
            text_features["component"] == component
        ]
        for metric_number, (metric, category, opportunity) in enumerate(
            GUARDRAILS
        ):
            pivot = component_rows.pivot(
                index="prompt_id",
                columns="strength",
                values=metric,
            )
            if opportunity is not None:
                eligible = component_rows.groupby("prompt_id")[
                    opportunity
                ].all()
                pivot = pivot.loc[eligible[eligible].index]
            if pivot.empty:
                continue
            values = (
                pivot[args.endpoint_strength]
                - pivot[-args.endpoint_strength]
            ).to_numpy()
            inference = paired_mean_inference(
                values,
                bootstrap=args.bootstrap,
                permutations=args.permutations,
                seed=(
                    args.seed
                    + 3000
                    + 100 * component_number
                    + metric_number
                ),
            )
            guardrail_rows.append(
                {
                    "component": component,
                    "category": category,
                    "metric": metric,
                    "n_prompts": len(pivot),
                    "negative_mean": float(
                        pivot[-args.endpoint_strength].mean()
                    ),
                    "positive_mean": float(
                        pivot[args.endpoint_strength].mean()
                    ),
                    **inference,
                }
            )
    guardrail_statistics = pd.DataFrame(guardrail_rows)

    semantic_pairs = pd.DataFrame()
    semantic_summary = pd.DataFrame()
    semantic_contrasts = pd.DataFrame()
    sentence_transformers_version = None
    if args.semantic_model is not None:
        (
            semantic_pairs,
            semantic_summary,
            semantic_contrasts,
            sentence_transformers_version,
        ) = semantic_pair_guardrails(
            text_features,
            model_path=args.semantic_model,
            device=args.semantic_device,
            batch_size=args.batch_size,
            bootstrap=args.bootstrap,
            permutations=args.permutations,
            seed=args.seed,
            endpoint_strength=args.endpoint_strength,
        )

    text_features.to_csv(args.out_dir / "text_features.csv", index=False)
    differences.to_csv(
        args.out_dir / "paired_component_differences.csv",
        index=False,
    )
    projection_table.to_csv(
        args.out_dir / "prompt_signature_projections.csv",
        index=False,
    )
    group_statistics.to_csv(
        args.out_dir / "causal_group_statistics.csv",
        index=False,
    )
    contrast_table.to_csv(
        args.out_dir / "component_contrasts.csv",
        index=False,
    )
    selectivity_statistics.to_csv(
        args.out_dir / "target_selectivity_statistics.csv",
        index=False,
    )
    behavior_null_summary.to_csv(
        args.out_dir / "behavior_null_summary.csv",
        index=False,
    )
    behavior_null_axes.to_csv(
        args.out_dir / "behavior_null_axis_statistics.csv",
        index=False,
    )
    guardrail_statistics.to_csv(
        args.out_dir / "guardrail_statistics.csv",
        index=False,
    )
    semantic_pairs.to_csv(
        args.out_dir / "semantic_pair_guardrails.csv",
        index=False,
    )
    semantic_summary.to_csv(
        args.out_dir / "semantic_guardrail_summary.csv",
        index=False,
    )
    semantic_contrasts.to_csv(
        args.out_dir / "semantic_guardrail_contrasts.csv",
        index=False,
    )
    (args.out_dir / "hook_audit.json").write_text(
        json.dumps(hook_audit, indent=2),
        encoding="utf-8",
    )
    lines = [
        "# Symmetric component-steering analysis",
        "",
        f"The primary outcome is the +{args.endpoint_strength:g} minus "
        f"−{args.endpoint_strength:g} language-feature change "
        "projected onto the observational AI−Human signature.",
        "",
        "| Component | Group | Projection | 95% CI | Cosine | Exact-text identity |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in group_statistics.itertuples():
        lines.append(
            f"| {row.component} | {row.group} | "
            f"{row.signature_projection_mean:.3f} | "
            f"[{row.signature_projection_ci_low:.3f}, "
            f"{row.signature_projection_ci_high:.3f}] | "
            f"{row.mean_effect_cosine_to_source_signature:.3f} | "
            f"{row.exact_text_identity_rate:.3f} |"
        )
    if not selectivity_statistics.empty:
        lines.extend(
            [
                "",
                "## Target selectivity",
                "",
                "Selectivity is the target-group projection minus the mean "
                "projection onto the two non-target signatures.",
                "",
                "| Component | Target | Off-target groups | Selectivity | "
                "95% CI |",
                "|---|---|---|---:|---:|",
            ]
        )
        for row in selectivity_statistics.itertuples():
            lines.append(
                f"| {row.component} | {row.target_group} | "
                f"{row.off_target_groups} | {row.mean:.3f} | "
                f"[{row.ci_low:.3f}, {row.ci_high:.3f}] |"
            )
    if not behavior_null_summary.empty:
        lines.extend(
            [
                "",
                "## Frozen random-axis ensemble",
                "",
                "Signed learned effects are reported without subtracting a "
                "random direction. Because a random axis has no semantic "
                "pole, calibration compares absolute axis-level means.",
                "",
                "| Concept | Estimand | Learned | 95% CI | "
                "Random | Rank | Tail fraction |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in behavior_null_summary.itertuples():
            lines.append(
                f"| {row.concept} | {row.estimand} | "
                f"{row.learned_signed_mean:.3f} | "
                f"[{row.learned_ci_low:.3f}, "
                f"{row.learned_ci_high:.3f}] | "
                f"{row.random_abs_axis_mean:.3f} | "
                f"{int(row.learned_abs_rank)}/"
                f"{int(row.n_random_axes) + 1} | "
                f"{row.finite_ensemble_tail_fraction:.3f} |"
            )
    lines.extend(
        [
            "",
            "## Minimal content and quality guardrails",
            "",
            f"Values are +{args.endpoint_strength:g} minus "
            f"−{args.endpoint_strength:g} paired changes. Content overlap is "
            "measured against the unchanged prompt; quality checks reuse "
            "the frozen automatic feature extractor.",
            "",
            "| Component | Category | Metric | n | Δ mean | 95% CI |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for row in guardrail_statistics.itertuples():
        lines.append(
            f"| {row.component} | {row.category} | {row.metric} | "
            f"{row.n_prompts} | {row.mean:.3f} | "
            f"[{row.ci_low:.3f}, {row.ci_high:.3f}] |"
        )
    if not semantic_summary.empty:
        lines.extend(
            [
                "",
                "## Paired-output semantic guardrail",
                "",
                "Cosine similarity is computed directly between the "
                f"−{args.endpoint_strength:g} and "
                f"+{args.endpoint_strength:g} continuations from the same "
                "prompt. The random-direction "
                "condition calibrates ordinary sampling-trajectory divergence.",
                "",
                "| Component | n | Mean similarity | 95% CI | Minimum | "
                "Exact identity |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in semantic_summary.itertuples():
            lines.append(
                f"| {row.component} | {row.n_prompts} | {row.mean:.3f} | "
                f"[{row.ci_low:.3f}, {row.ci_high:.3f}] | "
                f"{row.minimum_similarity:.3f} | "
                f"{row.exact_text_identity_rate:.3f} |"
            )
    (args.out_dir / "analysis_summary.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    input_hashes = {
        str(args.observational_features): sha256_file(
            args.observational_features
        )
    }
    for component in args.components:
        path = args.input_dir / f"generation_component_{component}.json"
        input_hashes[str(path)] = sha256_file(path)
    finished = datetime.now(timezone.utc)
    manifest = {
        "schema": "symmetric_component_linguistic_analysis",
        "input_dir": str(args.input_dir),
        "observational_features": str(args.observational_features),
        "input_sha256": input_hashes,
        "spacy_model": args.spacy_model,
        "spacy_version": spacy.__version__,
        "python_version": platform.python_version(),
        "bootstrap": args.bootstrap,
        "permutations": args.permutations,
        "seed": args.seed,
        "components": args.components,
        "endpoint_strength": args.endpoint_strength,
        "n_prompts": int(differences["prompt_id"].nunique()),
        "guardrails": [name for name, _, _ in GUARDRAILS],
        "semantic_model": (
            str(args.semantic_model.expanduser().resolve())
            if args.semantic_model is not None
            else None
        ),
        "semantic_device": (
            args.semantic_device if args.semantic_model is not None else None
        ),
        "sentence_transformers_version": sentence_transformers_version,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "command": " ".join(sys.argv),
    }
    (args.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote component analysis to {args.out_dir}")


if __name__ == "__main__":
    main()
