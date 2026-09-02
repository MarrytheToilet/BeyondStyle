#!/usr/bin/env python
"""Cross-fitted token-level cartography of nullspace source directions.

Primary mode fits one nullspace direction per held-out generator using the
existing paired document states, then projects every continuation token from
that generator onto a direction that never saw it. The script writes compact
scalar Parquet shards and derives:

* layer-wise final-token and continuation-mean geometry;
* pair-balanced within-continuation position trajectories;
* exact same-token Human/AI contextual contrasts; and
* a source-balanced empirical vocabulary field.

This is not a logit lens. Intermediate hidden states are not decoded with the
final unembedding matrix, and token-field claims remain model-tokenizer local.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
from typing import Any

import numpy as np
import pandas as pd
import pyarrow
import torch
import transformers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.projection_mapping import (  # noqa: E402
    fit_nullspace_source_direction,
    unit_vector,
)
from src.analysis.token_projection import (  # noqa: E402
    empirical_vocabulary_field,
    layer_projection_summary,
    pair_balanced_standardize,
    paired_token_context_contrasts,
    position_projection_trajectory,
)
from src.data.paired_io import (  # noqa: E402
    generator_model,
    load_paired_samples,
)
from src.extraction.hidden_states import HiddenStatesExtractor  # noqa: E402
from src.runtime import model_source_fingerprint  # noqa: E402
from src.visualization.token_projection import (  # noqa: E402
    render_token_projection_atlas,
)


AXIS_FAMILIES = ("last_token", "mean_pool")
DEFAULT_REPORT_LAYERS = {
    "Llama-3.1-8B-Instruct": {"last_token": 11, "mean_pool": 11},
    "Qwen3-4B": {"last_token": 12, "mean_pool": 8},
    "gemma-4-E2B-it": {"last_token": 3, "mean_pool": 12},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        help="Local model directory or Hugging Face model ID.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Canonical output/checkpoint name; defaults to model basename.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/paired/en/proposition_aligned/pairs.jsonl"),
    )
    parser.add_argument(
        "--states-root",
        type=Path,
        default=Path("outputs/en/03_steering_vectors"),
    )
    parser.add_argument("--dataset-tag", default="proposition_aligned")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/en/05_linguistic/token_projection"),
        help="The canonical model ID is appended automatically.",
    )
    parser.add_argument(
        "--axis-families",
        nargs="+",
        choices=list(AXIS_FAMILIES),
        default=list(AXIS_FAMILIES),
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="Defaults to the intersection of available nullspace state layers.",
    )
    parser.add_argument(
        "--report-layers",
        nargs="+",
        default=None,
        metavar="AXIS:LAYER",
        help="Layers for position/token reports; all layers still enter layer summary.",
    )
    parser.add_argument(
        "--direction-scheme",
        choices=["generator_holdout", "checkpoint"],
        default="generator_holdout",
    )
    parser.add_argument("--null-k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument(
        "--attention-backend",
        choices=["auto", "sdpa", "flash_attention_2", "eager"],
        default="auto",
    )
    parser.add_argument("--position-bins", type=int, default=10)
    parser.add_argument("--min-paired-token-samples", type=int, default=10)
    parser.add_argument("--bootstrap-repetitions", type=int, default=500)
    parser.add_argument("--vocabulary-top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument(
        "--analysis-only",
        action="store_true",
        help="Do not load a model; aggregate existing complete shards.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only this model's token-projection output.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_path(
    root: Path,
    model: str,
    dataset_tag: str,
    axis_family: str,
    layer: int,
) -> Path:
    return (
        root
        / model
        / dataset_tag
        / axis_family
        / "states"
        / f"paired_states_layer{layer:02d}.pt"
    )


def checkpoint_path(
    root: Path,
    model: str,
    dataset_tag: str,
    axis_family: str,
    layer: int,
) -> Path:
    return (
        root
        / model
        / dataset_tag
        / axis_family
        / "models"
        / f"mean_diff_nullspace_layer{layer:02d}.pt"
    )


def discover_layers(
    root: Path,
    model: str,
    dataset_tag: str,
    axis_families: list[str],
) -> list[int]:
    layer_sets = []
    for axis_family in axis_families:
        directory = (
            root / model / dataset_tag / axis_family / "states"
        )
        available = {
            int(path.stem.rsplit("layer", 1)[1])
            for path in directory.glob("paired_states_layer*.pt")
            if checkpoint_path(
                root,
                model,
                dataset_tag,
                axis_family,
                int(path.stem.rsplit("layer", 1)[1]),
            ).is_file()
        }
        layer_sets.append(available)
    common = set.intersection(*layer_sets) if layer_sets else set()
    if not common:
        raise FileNotFoundError(
            f"No common nullspace state/checkpoint layers for {model}"
        )
    return sorted(common)


def parse_report_layers(
    values: list[str] | None,
    *,
    model_id: str,
    layers: list[int],
    axis_families: list[str],
) -> dict[str, int]:
    if values is None:
        defaults = DEFAULT_REPORT_LAYERS.get(model_id, {})
        result = {
            axis: defaults[axis]
            for axis in axis_families
            if axis in defaults and defaults[axis] in layers
        }
        if len(result) != len(axis_families):
            result = {axis: layers[len(layers) // 2] for axis in axis_families}
        return result
    result: dict[str, int] = {}
    for value in values:
        try:
            axis, raw_layer = value.rsplit(":", 1)
            layer = int(raw_layer)
        except ValueError as error:
            raise ValueError(
                f"Expected AXIS:LAYER in --report-layers, got {value!r}"
            ) from error
        if axis not in axis_families or layer not in layers:
            raise ValueError(
                f"Report layer {value!r} is outside requested axes/layers"
            )
        result[axis] = layer
    missing = sorted(set(axis_families) - set(result))
    if missing:
        raise ValueError(f"Missing report layers for axes: {missing}")
    return result


def balanced_subset(
    records: list[dict[str, Any]],
    maximum: int | None,
) -> list[dict[str, Any]]:
    """Deterministic generator/domain round-robin diagnostic subset."""
    if maximum is None or maximum >= len(records):
        return records
    if maximum < 1:
        raise ValueError("--max-samples must be positive")
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        key = (generator_model(record), str(record.get("domain", "unknown")))
        groups.setdefault(key, []).append(record)
    selected = []
    positions = {key: 0 for key in groups}
    keys = sorted(groups)
    while len(selected) < maximum:
        changed = False
        for key in keys:
            position = positions[key]
            if position < len(groups[key]):
                selected.append(groups[key][position])
                positions[key] += 1
                changed = True
                if len(selected) == maximum:
                    break
        if not changed:
            break
    return selected


def load_aligned_states(
    path: Path,
    ordered_ids: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    identifiers = payload.get("metadata", {}).get("sample_ids")
    if not identifiers or set(identifiers) != set(ordered_ids):
        raise ValueError(f"State IDs do not match paired data in {path}")
    positions = {
        identifier: position
        for position, identifier in enumerate(identifiers)
    }
    order = [positions[identifier] for identifier in ordered_ids]
    return (
        payload["human"][order].float().numpy(),
        payload["ai"][order].float().numpy(),
    )


def load_checkpoint_direction(path: Path) -> np.ndarray:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if "state_dict" in payload:
        value = payload["state_dict"]["v_src"]
    else:
        value = payload["v_src"]
    return unit_vector(value.float().numpy()).astype(np.float32)


def direction_cache_signature(
    *,
    args: argparse.Namespace,
    model_id: str,
    layers: list[int],
    all_records: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[Path]]:
    inputs = [args.dataset]
    if args.direction_scheme == "generator_holdout":
        inputs.extend(
            state_path(
                args.states_root,
                model_id,
                args.dataset_tag,
                axis,
                layer,
            )
            for axis in args.axis_families
            for layer in layers
        )
    else:
        inputs.extend(
            checkpoint_path(
                args.states_root,
                model_id,
                args.dataset_tag,
                axis,
                layer,
            )
            for axis in args.axis_families
            for layer in layers
        )
    missing = [path for path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing direction inputs: {missing[:3]}")
    return {
        "model_id": model_id,
        "direction_scheme": args.direction_scheme,
        "null_k": args.null_k,
        "layers": layers,
        "axis_families": args.axis_families,
        "sample_ids": [row["sample_id"] for row in all_records],
        "input_sha256": {
            str(path): sha256_file(path) for path in inputs
        },
    }, inputs


def build_directions(
    *,
    args: argparse.Namespace,
    model_id: str,
    layers: list[int],
    all_records: list[dict[str, Any]],
    model_dir: Path,
) -> tuple[dict[str, dict[int, dict[str, np.ndarray]]], pd.DataFrame, Path]:
    signature, _ = direction_cache_signature(
        args=args,
        model_id=model_id,
        layers=layers,
        all_records=all_records,
    )
    cache = model_dir / "crossfit_directions.pt"
    if cache.is_file():
        payload = torch.load(cache, map_location="cpu", weights_only=False)
        if payload.get("signature") == signature:
            directions = {
                axis: {
                    int(layer): {
                        environment: value.numpy()
                        for environment, value in environments.items()
                    }
                    for layer, environments in per_layer.items()
                }
                for axis, per_layer in payload["directions"].items()
            }
            return directions, pd.DataFrame(payload["audit"]), cache

    ordered_ids = [row["sample_id"] for row in all_records]
    environments = np.asarray(
        [generator_model(row) for row in all_records],
        dtype=object,
    )
    directions: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    audit = []
    for axis in args.axis_families:
        directions[axis] = {}
        for layer in layers:
            directions[axis][layer] = {}
            if args.direction_scheme == "checkpoint":
                path = checkpoint_path(
                    args.states_root,
                    model_id,
                    args.dataset_tag,
                    axis,
                    layer,
                )
                vector = load_checkpoint_direction(path)
                directions[axis][layer]["__all__"] = vector
                audit.append(
                    {
                        "axis_family": axis,
                        "layer": layer,
                        "heldout_generator": "__none__",
                        "n_train": len(all_records),
                        "n_test": len(all_records),
                        "null_k": args.null_k,
                        "residual_norm": float("nan"),
                        "cosine_vs_raw": float("nan"),
                    }
                )
                continue

            path = state_path(
                args.states_root,
                model_id,
                args.dataset_tag,
                axis,
                layer,
            )
            human, ai = load_aligned_states(path, ordered_ids)
            for environment in sorted(set(environments)):
                train = np.flatnonzero(environments != environment)
                test = np.flatnonzero(environments == environment)
                vector, info = fit_nullspace_source_direction(
                    human[train],
                    ai[train],
                    k=args.null_k,
                )
                directions[axis][layer][str(environment)] = (
                    vector.astype(np.float32)
                )
                audit.append(
                    {
                        "axis_family": axis,
                        "layer": layer,
                        "heldout_generator": str(environment),
                        "n_train": len(train),
                        "n_test": len(test),
                        **info,
                    }
                )
    serializable = {
        axis: {
            layer: {
                environment: torch.from_numpy(value)
                for environment, value in environments.items()
            }
            for layer, environments in per_layer.items()
        }
        for axis, per_layer in directions.items()
    }
    torch.save(
        {
            "signature": signature,
            "directions": serializable,
            "audit": audit,
        },
        cache,
    )
    return directions, pd.DataFrame(audit), cache


def extraction_signature(
    *,
    args: argparse.Namespace,
    model_id: str,
    layers: list[int],
    report_layers: dict[str, int],
    records: list[dict[str, Any]],
    direction_cache: Path,
) -> dict[str, Any]:
    return {
        "schema": "token_projection_scalar_shards_v1",
        "model": model_source_fingerprint(args.model),
        "model_id": model_id,
        "dataset": str(args.dataset),
        "dataset_sha256": sha256_file(args.dataset),
        "dataset_tag": args.dataset_tag,
        "sample_ids": [row["sample_id"] for row in records],
        "layers": layers,
        "axis_families": args.axis_families,
        "report_layers": report_layers,
        "direction_scheme": args.direction_scheme,
        "direction_cache_sha256": sha256_file(direction_cache),
        "null_k": args.null_k,
        "batch_size": args.batch_size,
        "separator_policy": "stored_language_en_space_zh_empty_else_natural",
        "relative_position": "(zero_based_token_index + 0.5) / token_count",
        "layer_convention": "decoder_block_output_1_based",
    }


def tokenizer_texts(tokenizer, token_ids: list[int]) -> list[str]:
    cache: dict[int, str] = {}
    values = []
    for token_id in token_ids:
        if token_id not in cache:
            cache[token_id] = tokenizer.decode(
                [token_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        values.append(cache[token_id])
    return values


def write_projection_shard(
    *,
    extractor: HiddenStatesExtractor,
    model_id: str,
    records: list[dict[str, Any]],
    layers: list[int],
    axis_families: list[str],
    directions: dict[str, dict[int, dict[str, np.ndarray]]],
    direction_scheme: str,
    output: Path,
) -> int:
    prefixes = [row["prefix"] for row in records]
    separators = [
        (
            " " if row.get("language") == "en"
            else "" if row.get("language") == "zh"
            else None
        )
        for row in records
    ]
    human = [row["human_continuation"] for row in records]
    ai = [row["ai_continuation"] for row in records]
    extracted = extractor.extract_continuation_tokens_batch(
        prefixes=prefixes + prefixes,
        continuations=human + ai,
        layers=layers,
        separators=separators + separators,
        to_cpu=False,
    )
    row_records = records + records
    sources = ["human"] * len(records) + ["ai"] * len(records)
    frames = []
    for result, record, source in zip(extracted, row_records, sources):
        token_ids = result.token_ids.detach().cpu().tolist()
        token_text = tokenizer_texts(extractor.tokenizer, token_ids)
        count = len(token_ids)
        common = {
            "model": model_id,
            "direction_scheme": direction_scheme,
            "sample_id": record["sample_id"],
            "pair_id": record.get("pair_id", record["sample_id"]),
            "source": source,
            "generator": generator_model(record),
            "domain": str(record.get("domain", "unknown")),
            "token_id": np.asarray(token_ids, dtype=np.int32),
            "token_text": token_text,
            "token_index": np.arange(count, dtype=np.int16),
            "token_count": np.full(count, count, dtype=np.int16),
            "relative_position": (
                np.arange(count, dtype=np.float32) + 0.5
            ) / count,
        }
        environment = (
            "__all__"
            if direction_scheme == "checkpoint"
            else generator_model(record)
        )
        for axis in axis_families:
            for layer in layers:
                vector = torch.as_tensor(
                    directions[axis][layer][environment],
                    device=result.hidden_states[layer].device,
                    dtype=result.hidden_states[layer].dtype,
                )
                projection = (
                    result.hidden_states[layer]
                    .matmul(vector)
                    .float()
                    .detach()
                    .cpu()
                    .numpy()
                )
                frames.append(
                    pd.DataFrame(
                        {
                            **common,
                            "axis_family": axis,
                            "layer": np.full(count, layer, dtype=np.int16),
                            "projection": projection,
                        }
                    )
                )
    table = pd.concat(frames, ignore_index=True)
    temporary = output.with_suffix(".parquet.incomplete")
    table.to_parquet(
        temporary,
        index=False,
        compression="zstd",
    )
    temporary.replace(output)
    return len(table)


def run_extraction(
    *,
    args: argparse.Namespace,
    model_id: str,
    records: list[dict[str, Any]],
    layers: list[int],
    directions: dict[str, dict[int, dict[str, np.ndarray]]],
    shards_dir: Path,
) -> dict[str, Any]:
    completed_ids: set[str] = set()
    for shard in shards_dir.glob("part_*.parquet"):
        completed_ids.update(
            pd.read_parquet(shard, columns=["sample_id"])["sample_id"]
            .astype(str)
            .unique()
            .tolist()
        )
    requested_ids = {str(record["sample_id"]) for record in records}
    if requested_ids <= completed_ids:
        return {"model_load": None, "new_rows": 0, "resumed": True}

    extractor = HiddenStatesExtractor(
        model_name=args.model,
        device=args.device,
        dtype=args.torch_dtype,
        attention_backend=args.attention_backend,
    )
    first_axis = next(iter(directions.values()))
    first_layer = next(iter(first_axis.values()))
    first_vector = next(iter(first_layer.values()))
    if extractor.hidden_size != len(first_vector):
        raise ValueError(
            "Model hidden size does not match the nullspace directions"
        )
    current_batch_size = args.batch_size
    offset = 0
    new_rows = 0
    while offset < len(records):
        if str(records[offset]["sample_id"]) in completed_ids:
            offset += 1
            continue
        end = min(offset + current_batch_size, len(records))
        while (
            end > offset + 1
            and any(
                str(record["sample_id"]) in completed_ids
                for record in records[offset:end]
            )
        ):
            end -= 1
        batch_records = records[offset:end]
        output = shards_dir / f"part_{offset:05d}_{end:05d}.parquet"
        try:
            new_rows += write_projection_shard(
                extractor=extractor,
                model_id=model_id,
                records=batch_records,
                layers=layers,
                axis_families=args.axis_families,
                directions=directions,
                direction_scheme=args.direction_scheme,
                output=output,
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or (
                "out of memory" in str(error).lower()
            )
            if not is_oom or current_batch_size == 1:
                raise
            current_batch_size = max(1, current_batch_size // 2)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            continue
        offset = end
        completed_ids.update(
            str(record["sample_id"]) for record in batch_records
        )
        print(f"Projected {offset}/{len(records)} paired continuations")
    return {
        "model_load": extractor.model_load_info,
        "new_rows": new_rows,
        "resumed": False,
        "effective_batch_size": current_batch_size,
    }


def analyze_shards(
    *,
    args: argparse.Namespace,
    model_id: str,
    model_dir: Path,
    report_layers: dict[str, int],
) -> dict[str, Any]:
    shards = sorted((model_dir / "token_records").glob("part_*.parquet"))
    if not shards:
        raise FileNotFoundError("No token projection shards to analyze")
    frame = pd.read_parquet(shards)
    for column in (
        "model",
        "direction_scheme",
        "axis_family",
        "source",
        "generator",
        "domain",
    ):
        frame[column] = frame[column].astype("category")
    standardized, scale = pair_balanced_standardize(frame)
    scale.to_csv(model_dir / "coordinate_standardization.csv", index=False)

    layer = layer_projection_summary(standardized)
    layer.to_csv(model_dir / "layer_projection_summary.csv", index=False)
    report_mask = np.zeros(len(standardized), dtype=bool)
    for axis, selected_layer in report_layers.items():
        report_mask |= (
            (standardized["axis_family"] == axis)
            & (standardized["layer"] == selected_layer)
        ).to_numpy()
    report = standardized.loc[report_mask].copy()
    position = position_projection_trajectory(
        report,
        bins=args.position_bins,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    position.to_csv(model_dir / "position_trajectory.csv", index=False)
    contrasts = paired_token_context_contrasts(
        report,
        min_paired_samples=args.min_paired_token_samples,
        bootstrap_repetitions=args.bootstrap_repetitions,
        seed=args.seed,
    )
    contrasts.to_csv(model_dir / "same_token_context.csv", index=False)
    vocabulary = empirical_vocabulary_field(
        contrasts,
        top_k=args.vocabulary_top_k,
        wordlike_only=True,
    )
    vocabulary.to_csv(model_dir / "empirical_vocabulary_field.csv", index=False)
    figure_path = render_token_projection_atlas(
        layer_summary=layer,
        position_trajectory=position,
        same_token_context=contrasts,
        vocabulary_field=vocabulary,
        report_layers=report_layers,
        output=model_dir / "figures" / "token_projection_atlas.png",
    )

    summary_path = model_dir / "analysis_summary.md"
    selected_layer = layer[
        layer.apply(
            lambda row: report_layers.get(str(row["axis_family"]))
            == int(row["layer"]),
            axis=1,
        )
    ]
    robust_tokens = (
        contrasts[
            (contrasts["context_shift_q"] <= 0.05)
            & (contrasts["generator_sign_agreement"] >= 0.75)
            & (contrasts["domain_sign_agreement"] >= 0.75)
        ]
        if not contrasts.empty
        else contrasts
    )
    lines = [
        f"# Token-level nullspace projection: {model_id}",
        "",
        f"- Direction scheme: `{args.direction_scheme}`.",
        f"- Report layers: `{json.dumps(report_layers, sort_keys=True)}`.",
        f"- Scalar token rows: {len(frame):,}.",
        f"- Exact same-token contrasts: {len(contrasts):,}; "
        f"FDR/stability-qualified: {len(robust_tokens):,}.",
        "- Same-token contrasts use within-continuation-centered coordinates; "
        "the global document source offset is removed first.",
        "- The empirical vocabulary field is source-balanced, locally centered, "
        "and tokenizer-local; it is not a Tuned Lens or direct-logit-lens decoding.",
        "",
        "## Selected-layer source geometry",
        "",
        "| Axis | Layer | Token representation | AUC | Cohen d | Positive pairs |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in selected_layer.itertuples(index=False):
        lines.append(
            f"| {row.axis_family} | {row.layer} | "
            f"{row.token_representation} | {row.auc:.3f} | "
            f"{row.cohen_d:.3f} | {row.positive_pair_fraction:.3f} |"
        )
    if not vocabulary.empty:
        lines.extend(
            [
                "",
                "## Empirical source-balanced token field",
                "",
            ]
        )
        for (axis, side), part in vocabulary.groupby(
            ["axis_family", "field_side"],
            observed=True,
        ):
            tokens = ", ".join(
                repr(value)
                for value in part.sort_values("field_rank")[
                    "token_text"
                ].head(8)
            )
            lines.append(f"- {axis}, {side}: {tokens}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "scalar_token_rows": len(frame),
        "same_token_contrasts": len(contrasts),
        "robust_same_token_contrasts": len(robust_tokens),
        "vocabulary_field_rows": len(vocabulary),
        "shards": [str(path) for path in shards],
        "figures": [
            str(figure_path),
            str(figure_path.with_suffix(".pdf")),
        ],
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    model_id = args.model_id or Path(args.model.rstrip("/")).name
    model_dir = args.out_dir / model_id
    if args.overwrite and model_dir.exists():
        shutil.rmtree(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = model_dir / "token_records"
    shards_dir.mkdir(exist_ok=True)

    all_records = load_paired_samples(args.dataset)
    records = balanced_subset(all_records, args.max_samples)
    layers = args.layers or discover_layers(
        args.states_root,
        model_id,
        args.dataset_tag,
        args.axis_families,
    )
    report_layers = parse_report_layers(
        args.report_layers,
        model_id=model_id,
        layers=layers,
        axis_families=args.axis_families,
    )
    directions, direction_audit, direction_cache = build_directions(
        args=args,
        model_id=model_id,
        layers=layers,
        all_records=all_records,
        model_dir=model_dir,
    )
    direction_audit.to_csv(model_dir / "direction_audit.csv", index=False)
    signature = extraction_signature(
        args=args,
        model_id=model_id,
        layers=layers,
        report_layers=report_layers,
        records=records,
        direction_cache=direction_cache,
    )
    work_manifest = model_dir / "work_manifest.json"
    if work_manifest.is_file():
        previous = json.loads(work_manifest.read_text(encoding="utf-8"))
        if previous.get("extraction_signature") != signature:
            raise RuntimeError(
                "Existing token shards have a different configuration. "
                "Use --overwrite to replace this model output."
            )
    else:
        work_manifest.write_text(
            json.dumps(
                {
                    "status": "extracting",
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "extraction_signature": signature,
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    if args.analysis_only:
        runtime = {
            **(
                previous.get("runtime", {})
                if work_manifest.is_file() and "previous" in locals()
                else {}
            ),
            "analysis_only_refresh": True,
        }
    else:
        runtime = run_extraction(
            args=args,
            model_id=model_id,
            records=records,
            layers=layers,
            directions=directions,
            shards_dir=shards_dir,
        )
    analysis = analyze_shards(
        args=args,
        model_id=model_id,
        model_dir=model_dir,
        report_layers=report_layers,
    )
    manifest = {
        "status": "complete",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "extraction_signature": signature,
        "runtime": runtime,
        "analysis": analysis,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "pandas": pd.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "interpretation_contract": {
            "primary_direction": "within_class_nullspace",
            "current_sample_excluded": (
                args.direction_scheme == "generator_holdout"
            ),
            "same_token_requires_both_sources_in_pair": True,
            "same_token_global_document_offset_removed": True,
            "token_field_is_source_balanced": True,
            "intermediate_unembedding_used": False,
            "tuned_lens_used": False,
            "cross_model_token_ids_compared": False,
        },
    }
    work_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote token projection analysis to {model_dir}")


if __name__ == "__main__":
    main()
