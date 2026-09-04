#!/usr/bin/env python3
"""Canonical, resumable generated-token-only steering experiment runner.

One process owns one target model and can run both continuation
representations without reloading the model. Batch shards and manifests make
long server jobs resumable and auditable.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import get_model_id, get_output_base
from src.generation import SteeredGenerator, SteeredGenerationConfig
from src.runtime import model_source_fingerprint, stable_seed
from src.steering import (
    MeanDiffSteering,
    load_calibrated_mean_diff,
    pooled_projection_std,
)


METHOD_FILE = {
    "nullspace": "mean_diff_nullspace",
    "raw": "mean_diff",
}
COMPONENTS = (
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
COMPONENT_FAMILIES = (
    "hidden_language",
    "concept_patterns",
    "concept_patterns_layerwise",
    "causal_pullback",
)

CHAT_CONTINUATION_INSTRUCTION = (
    "Continue the passage below naturally for 80 to 120 words. Write only "
    "the continuation; "
    "do not explain, summarize, or comment on the passage.\n\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--language", choices=["en", "zh"], required=True)
    parser.add_argument(
        "--representations",
        nargs="+",
        choices=["last_token", "mean_pool"],
        default=["last_token", "mean_pool"],
    )
    parser.add_argument("--dataset-tag", default="proposition_aligned")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--method", choices=["nullspace", "raw"], default="nullspace")
    parser.add_argument(
        "--component",
        choices=COMPONENTS,
        help=(
            "Use a hidden-language component checkpoint instead of the "
            "ordinary source estimator."
        ),
    )
    parser.add_argument(
        "--components",
        nargs="+",
        choices=COMPONENTS,
        help="Run several component checkpoints while loading the model once.",
    )
    parser.add_argument(
        "--component-family",
        choices=COMPONENT_FAMILIES,
        default="hidden_language",
        help=(
            "Checkpoint family under outputs/<lang>/05_linguistic. "
            "Use concept_patterns for covariance-aware linguistic patterns "
            "or causal_pullback for Jacobian-pulled intervention directions."
        ),
    )
    parser.add_argument(
        "--component-root",
        type=Path,
        help=(
            "Optional explicit component checkpoint root containing layerNN/. "
            "Intended for development diagnostics; canonical runs derive the "
            "root from --component-family."
        ),
    )
    parser.add_argument(
        "--strengths",
        type=float,
        nargs="+",
        default=[-1.0, 1.0],
    )
    parser.add_argument("--num-seeds", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=8236)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help=(
            "Split the globally ordered batch plan across this many workers. "
            "Workers retain canonical batch IDs and paired RNG streams."
        ),
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="Zero-based worker index used with --num-shards.",
    )
    parser.add_argument(
        "--prompt-start",
        type=int,
        default=0,
        help="Skip this many prompts before applying --max-prompts.",
    )
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument(
        "--cache-implementation",
        choices=["static", "dynamic"],
        default="dynamic",
        help=(
            "DynamicCache is the validated default for hooked steering. "
            "StaticCache is opt-in because it may trigger costly "
            "torch.compile recompilation and CUDA graph capture."
        ),
    )
    parser.add_argument(
        "--pad-to-multiple-of",
        type=int,
        default=8,
        help="Pad prompt widths to this multiple; use 1 to disable.",
    )
    parser.add_argument(
        "--no-length-bucketing",
        action="store_true",
        help="Keep input prompt order instead of batching similar lengths.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--torch-dtype",
        choices=["auto", "bf16", "fp16", "fp32"],
        default="auto",
    )
    parser.add_argument(
        "--attention-backend",
        choices=["auto", "sdpa", "flash_attention_2", "eager"],
        default="auto",
    )
    parser.add_argument("--no-tf32", action="store_true")
    parser.add_argument(
        "--layer",
        type=int,
        help="Override frozen worst-generator layer selection.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Defaults to OUTPUT_DIR/outputs.",
    )
    parser.add_argument(
        "--output-subdir",
        help=(
            "Override the final generation directory name. This is useful "
            "for an isolated smoke test without changing checkpoint lookup."
        ),
    )
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument(
        "--chat-template",
        action="store_true",
        help=(
            "Wrap each prompt with the model tokenizer's user-turn chat "
            "template before generation. Disabled by default so existing "
            "prefix-continuation experiments remain unchanged."
        ),
    )
    parser.add_argument(
        "--chat-continuation-instruction",
        action="store_true",
        help=(
            "When --chat-template is enabled, prepend an explicit instruction "
            "that requests continuation-only output. This keeps instruction-"
            "tuned chat models on the same continuation task as raw-prefix "
            "models."
        ),
    )
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_fingerprint(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


@contextmanager
def exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Another process already owns this experiment: {path}"
            ) from error
        yield


def load_prompts(path: Path, language: str) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    prompts = []
    for index, item in enumerate(payload):
        if isinstance(item, str):
            text = item.strip()
            provided_id = None
        elif isinstance(item, dict):
            text = str(
                item.get("text")
                or item.get("prefix")
                or item.get("prompt")
                or ""
            ).strip()
            provided_id = item.get("id") or item.get("sample_id")
        else:
            raise TypeError(f"Unsupported prompt row {index}: {type(item)}")
        if not text:
            raise ValueError(f"Empty prompt at row {index}")
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        prompt_id = str(provided_id or f"{language}_{index:04d}_{text_hash}")
        prompts.append({
            "prompt_id": prompt_id,
            "prompt_index": index,
            "text": text,
            "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        })
    if len({row["prompt_id"] for row in prompts}) != len(prompts):
        raise ValueError("Prompt IDs must be unique")
    return prompts


def order_prompts_by_length(
    prompts: list[dict],
    tokenizer,
    *,
    max_prompt_tokens: int,
) -> list[dict]:
    """Place the longest prompt first and reduce within-batch padding.

    StaticCache grows to the largest cache seen. Descending order therefore
    initializes its final shape once, while adjacent prompts have comparable
    prefill lengths. Final records are restored to ``prompt_index`` order.
    """
    encoded = tokenizer(
        [row["text"] for row in prompts],
        add_special_tokens=True,
        truncation=True,
        max_length=max_prompt_tokens,
        padding=False,
    )
    lengths = [len(token_ids) for token_ids in encoded["input_ids"]]
    return [
        row
        for _, _, row in sorted(
            zip(
                lengths,
                [row["prompt_index"] for row in prompts],
                prompts,
            ),
            key=lambda item: (-item[0], item[1]),
        )
    ]


def apply_chat_template_to_prompts(
    prompts: list[dict],
    tokenizer,
    *,
    continuation_instruction: bool = False,
) -> list[dict]:
    """Return prompt records formatted as one user turn for chat models."""
    formatted = []
    for row in prompts:
        user_content = row["text"]
        if continuation_instruction:
            user_content = CHAT_CONTINUATION_INSTRUCTION + user_content
        messages = [{"role": "user", "content": user_content}]
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        if not isinstance(text, str) or not text:
            raise ValueError("Tokenizer returned an empty chat-formatted prompt")
        formatted.append(
            {
                **row,
                "raw_text": row["text"],
                "text": text,
                "formatted_sha256": hashlib.sha256(
                    text.encode("utf-8")
                ).hexdigest(),
            }
        )
    return formatted


def select_robust_layer(
    evaluation_path: Path,
    method: str,
) -> tuple[int, list[dict]]:
    payload = json.loads(evaluation_path.read_text(encoding="utf-8"))
    rows = []
    for layer_text, generator_results in payload["results"].items():
        layer = int(layer_text)
        aucs = [
            float(method_results[method]["auc"])
            for method_results in generator_results.values()
        ]
        rows.append({
            "layer": layer,
            "worst_generator_auc": min(aucs),
            "mean_generator_auc": float(np.mean(aucs)),
        })
    rows.sort(
        key=lambda row: (
            -row["worst_generator_auc"],
            -row["mean_generator_auc"],
            row["layer"],
        )
    )
    return rows[0]["layer"], rows


def resolve_component_assets(
    component_root: Path,
    component: str,
    layer: int | None = None,
) -> tuple[int, Path, Path, list[dict]]:
    """Resolve one canonical component checkpoint and its decomposition."""
    if layer is None:
        layer_dirs = sorted(component_root.glob("layer[0-9][0-9]"))
        if len(layer_dirs) != 1:
            raise ValueError(
                f"Expected exactly one layer directory in {component_root}, "
                f"found {[path.name for path in layer_dirs]}"
            )
        layer_dir = layer_dirs[0]
        layer = int(layer_dir.name.removeprefix("layer"))
    else:
        layer_dir = component_root / f"layer{layer:02d}"
    selection_path = layer_dir / "decomposition.json"
    checkpoint_path = layer_dir / "directions" / f"{component}.pt"
    if not selection_path.exists() or not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing component assets: {selection_path}, {checkpoint_path}"
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    return layer, checkpoint_path, selection_path, [selection]


def run_method_key(args: argparse.Namespace) -> str:
    if args.component:
        return f"component_{args.component}"
    return METHOD_FILE[args.method]


def load_and_calibrate_direction(
    checkpoint_path: Path,
    states_path: Path,
) -> tuple[MeanDiffSteering, dict]:
    steering, calibration = load_calibrated_mean_diff(
        checkpoint_path,
        states_path,
    )
    return steering, {
        **calibration,
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "states_sha256": file_sha256(states_path),
    }


class ResumableRepresentationRun:
    def __init__(
        self,
        *,
        args: argparse.Namespace,
        generator: SteeredGenerator,
        model_id: str,
        representation: str,
        prompts: list[dict],
        steering: MeanDiffSteering,
        selected_layer: int,
        selection_table: list[dict],
        selection_path: Path,
        calibration: dict,
        output_dir: Path,
    ):
        self.args = args
        self.generator = generator
        self.model_id = model_id
        self.representation = representation
        self.prompts = prompts
        self.steering = steering
        self.selected_layer = selected_layer
        self.selection_table = selection_table
        self.selection_path = selection_path
        self.calibration = calibration
        self.method_key = run_method_key(args)
        self.output_dir = output_dir
        self.work_dir = output_dir / f".work_{self.method_key}"
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.config = self._config_payload()
        self.fingerprint = json_fingerprint(self.config)
        self._validate_or_create_work_manifest()

    def _config_payload(self) -> dict:
        return {
            "schema_version": 1,
            "model_id": self.model_id,
            "model_identity": model_source_fingerprint(self.args.model),
            "language": self.args.language,
            "dataset_tag": self.args.dataset_tag,
            "representation": self.representation,
            "method": self.method_key,
            "component": self.args.component,
            "component_family": self.args.component_family,
            "component_root_override": (
                str(self.args.component_root.expanduser().resolve())
                if self.args.component_root is not None
                else None
            ),
            "layer": self.selected_layer,
            "strengths": sorted(float(value) for value in self.args.strengths),
            "strength_unit": "projection_std",
            "position_mode": "decode_only",
            "prompt_prefill_untouched": True,
            "prompts_file": self.args.prompts.name,
            "prompts_sha256": file_sha256(self.args.prompts),
            "prompt_format": (
                "tokenizer_chat_template_continuation_user_turn"
                if self.args.chat_template
                and self.args.chat_continuation_instruction
                else "tokenizer_chat_template_user_turn"
                if self.args.chat_template
                else "raw_prefix"
            ),
            "chat_continuation_instruction": (
                CHAT_CONTINUATION_INSTRUCTION
                if self.args.chat_template
                and self.args.chat_continuation_instruction
                else None
            ),
            "prompt_start": self.args.prompt_start,
            "n_prompts": len(self.prompts),
            "num_seeds": self.args.num_seeds,
            "base_seed": self.args.base_seed,
            "requested_batch_size": self.args.batch_size,
            "num_shards": self.args.num_shards,
            "shard_index": self.args.shard_index,
            "max_new_tokens": self.args.max_new_tokens,
            "temperature": self.args.temperature,
            "top_p": self.args.top_p,
            "max_prompt_tokens": self.args.max_prompt_tokens,
            "cache_implementation": self.args.cache_implementation,
            "pad_to_multiple_of": self.args.pad_to_multiple_of,
            "batch_order": (
                "input"
                if self.args.no_length_bucketing
                else "descending_token_length"
            ),
            "device": self.args.device,
            "torch_dtype": self.args.torch_dtype,
            "attention_backend": self.args.attention_backend,
            "allow_tf32": not self.args.no_tf32,
            "checkpoint_sha256": self.calibration["checkpoint_sha256"],
            "states_sha256": self.calibration["states_sha256"],
            "selection_sha256": file_sha256(self.selection_path),
        }

    def _validate_or_create_work_manifest(self) -> None:
        path = self.work_dir / "work_manifest.json"
        if path.exists():
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("fingerprint") != self.fingerprint:
                raise RuntimeError(
                    f"Existing work directory has a different configuration: "
                    f"{self.work_dir}. Move it aside before changing the run."
                )
            return
        atomic_json(path, {
            "fingerprint": self.fingerprint,
            "config": self.config,
        })

    def _batch_path(self, batch_id: str) -> Path:
        return self.work_dir / f"batch_{batch_id}.json"

    def _split_path(self, batch_id: str) -> Path:
        return self.work_dir / f"batch_{batch_id}.split.json"

    def _run_batch(
        self,
        seed_index: int,
        batch_id: str,
        rows: list[dict],
    ) -> None:
        batch_path = self._batch_path(batch_id)
        if batch_path.exists():
            return
        split_path = self._split_path(batch_id)
        if split_path.exists():
            split = json.loads(split_path.read_text(encoding="utf-8"))
            midpoint = int(split["left_size"])
            self._run_batch(seed_index, split["left_id"], rows[:midpoint])
            self._run_batch(seed_index, split["right_id"], rows[midpoint:])
            return

        batch_seed = stable_seed(
            self.args.base_seed,
            self.model_id,
            self.representation,
            seed_index,
            batch_id,
        )
        config = SteeredGenerationConfig(
            max_new_tokens=self.args.max_new_tokens,
            temperature=self.args.temperature,
            top_p=self.args.top_p,
            steering_strength=1.0,
            steering_direction="ai",
            apply_to_positions="decode_only",
            strength_unit="projection_std",
            seed=batch_seed,
            paired_sampling=True,
            use_cache=True,
            cache_implementation=self.args.cache_implementation,
            max_prompt_tokens=self.args.max_prompt_tokens,
            pad_to_multiple_of=self.args.pad_to_multiple_of,
        )
        try:
            outputs = self.generator.generate_batch_comparison(
                [row["text"] for row in rows],
                self.steering,
                strengths=self.args.strengths,
                config=config,
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as error:
            is_oom = isinstance(error, torch.cuda.OutOfMemoryError) or (
                "out of memory" in str(error).lower()
            )
            if not is_oom or len(rows) == 1:
                raise
            midpoint = len(rows) // 2
            split = {
                "left_id": batch_id + "a",
                "right_id": batch_id + "b",
                "left_size": midpoint,
                "reason": f"{type(error).__name__}: CUDA out of memory",
            }
            atomic_json(split_path, split)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(
                f"  OOM: split {batch_id} ({len(rows)}) into "
                f"{split['left_id']} ({midpoint}) and "
                f"{split['right_id']} ({len(rows) - midpoint})",
                flush=True,
            )
            self._run_batch(seed_index, split["left_id"], rows[:midpoint])
            self._run_batch(seed_index, split["right_id"], rows[midpoint:])
            return

        records = []
        for row_index, row in enumerate(rows):
            record = {
                "layer": self.selected_layer,
                "method": self.method_key,
                "representation": self.representation,
                "prompt_id": row["prompt_id"],
                "prompt_index": row["prompt_index"],
                "prompt": row["text"],
                "sample_seed_index": seed_index,
                "alpha_direction": (
                    "negative=opposite_component, positive=along_component"
                    if self.args.component
                    else "negative=toward_human, positive=toward_ai"
                ),
                "generation_metadata": {
                    "batch_id": batch_id,
                    "batch_seed": batch_seed,
                    "position_mode": "decode_only",
                    "prompt_prefill_untouched": True,
                    "strength_unit": "projection_std",
                    "paired_sampling": True,
                },
                "outputs": {},
            }
            for strength in sorted(outputs):
                record["outputs"][str(strength)] = {
                    "direction": (
                        (
                            "opposite_component"
                            if strength < 0
                            else "along_component"
                            if strength > 0
                            else "baseline"
                        )
                        if self.args.component
                        else (
                            "toward_human"
                            if strength < 0
                            else "toward_ai"
                            if strength > 0
                            else "baseline"
                        )
                    ),
                    "text": outputs[strength][row_index],
                }
            records.append(record)
        atomic_json(batch_path, {
            "fingerprint": self.fingerprint,
            "batch_id": batch_id,
            "seed_index": seed_index,
            "seed": batch_seed,
            "records": records,
            "stats_by_strength": {
                str(strength): stats
                for strength, stats
                in self.generator.last_comparison_stats.items()
            },
        })
        print(
            f"  completed {self.representation} {batch_id}: "
            f"{len(rows)} prompts",
            flush=True,
        )

    def _collect_batch(self, batch_id: str) -> list[dict]:
        batch_path = self._batch_path(batch_id)
        if batch_path.exists():
            return [json.loads(batch_path.read_text(encoding="utf-8"))]
        split = json.loads(
            self._split_path(batch_id).read_text(encoding="utf-8")
        )
        return (
            self._collect_batch(split["left_id"])
            + self._collect_batch(split["right_id"])
        )

    def run(self) -> None:
        ordered_prompts = (
            self.prompts
            if self.args.no_length_bucketing
            else order_prompts_by_length(
                self.prompts,
                self.generator.tokenizer,
                max_prompt_tokens=self.args.max_prompt_tokens,
            )
        )
        planned_ids = []
        expected = 0
        batch_ordinal = 0
        for seed_index in range(self.args.num_seeds):
            for start in range(
                0,
                len(ordered_prompts),
                self.args.batch_size,
            ):
                batch_id = f"S{seed_index:02d}_B{start:05d}"
                rows = ordered_prompts[start: start + self.args.batch_size]
                assigned = (
                    batch_ordinal % self.args.num_shards
                    == self.args.shard_index
                )
                batch_ordinal += 1
                if not assigned:
                    continue
                planned_ids.append(batch_id)
                expected += len(rows)
                self._run_batch(
                    seed_index,
                    batch_id,
                    rows,
                )

        shards = [
            shard
            for batch_id in planned_ids
            for shard in self._collect_batch(batch_id)
        ]
        records = [
            record
            for shard in shards
            for record in shard["records"]
        ]
        records.sort(
            key=lambda row: (
                row["sample_seed_index"],
                row["prompt_index"],
            )
        )
        if len(records) != expected:
            raise RuntimeError(
                f"Expected {expected} completed records, got {len(records)}"
            )

        result_path = (
            self.output_dir / f"generation_{self.method_key}.json"
        )
        atomic_json(result_path, records)
        total_tokens = sum(
            stats["generated_tokens"]
            for shard in shards
            for stats in shard["stats_by_strength"].values()
        )
        total_seconds = sum(
            stats["elapsed_seconds"]
            for shard in shards
            for stats in shard["stats_by_strength"].values()
        )
        manifest = {
            "fingerprint": self.fingerprint,
            "config": self.config,
            "selected_layer": self.selected_layer,
            "selection_rule": (
                (
                    {
                        "concept_patterns": (
                            "generator-held-out covariance activation-pattern "
                            "validation"
                        ),
                        "concept_patterns_layerwise": (
                            "layer-specific covariance activation-pattern "
                            "validation"
                        ),
                        "causal_pullback": (
                            "nullspace-constrained Jacobian pullback from "
                            "independently fitted downstream concept readout"
                        ),
                    }.get(
                        self.args.component_family,
                        "nested generator-held-out language decomposition",
                    )
                )
                if self.args.component
                else (
                    "maximize worst held-out-generator AUC; tie-break by mean "
                    "held-out-generator AUC, then shallower layer"
                )
            ),
            "selection_table": self.selection_table,
            "calibration": self.calibration,
            "model_load": self._resolved_model_load_info(),
            "n_records": len(records),
            "n_batches": len(shards),
            "generated_tokens": total_tokens,
            "elapsed_generation_seconds": total_seconds,
            "aggregate_tokens_per_second": (
                total_tokens / max(total_seconds, 1e-9)
            ),
            "result_path": str(result_path),
            "result_sha256": file_sha256(result_path),
        }
        manifest_path = (
            self.output_dir
            / f"generation_{self.method_key}_manifest.json"
        )
        atomic_json(manifest_path, manifest)
        print(f"  finalized {result_path}", flush=True)
        print(f"  manifest  {manifest_path}", flush=True)

    def _resolved_model_load_info(self) -> dict:
        """Preserve resolved load metadata when a fully complete run resumes."""
        if self.generator.model_load_info:
            return self.generator.model_load_info
        manifest_path = (
            self.output_dir
            / f"generation_{self.method_key}_manifest.json"
        )
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
            return previous.get("model_load", {})
        return {}


def warm_up(
    generator: SteeredGenerator,
    prompt: str,
    steering: MeanDiffSteering,
) -> None:
    config = SteeredGenerationConfig(
        max_new_tokens=2,
        temperature=0.0,
        seed=0,
        apply_to_positions="decode_only",
        strength_unit="projection_std",
    )
    generator.generate_batch_comparison(
        [prompt],
        steering,
        strengths=[0.0, 1.0],
        config=config,
    )


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.num_seeds < 1:
        raise ValueError("batch-size and num-seeds must be positive")
    if args.num_shards < 1:
        raise ValueError("--num-shards must be positive")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards)")
    if args.pad_to_multiple_of < 1:
        raise ValueError("--pad-to-multiple-of must be positive")
    if args.prompt_start < 0:
        raise ValueError("--prompt-start must be nonnegative")
    if args.output_subdir is not None:
        candidate = Path(args.output_subdir)
        if (
            candidate.name != args.output_subdir
            or args.output_subdir in {"", ".", ".."}
        ):
            raise ValueError("--output-subdir must be one directory name")
    if sorted(set(args.strengths)) != sorted(args.strengths):
        raise ValueError("--strengths must not contain duplicates")
    if args.component and args.components:
        raise ValueError("Use either --component or --components, not both")
    requested_components = (
        list(args.components)
        if args.components
        else [args.component]
        if args.component
        else []
    )
    if len(requested_components) != len(set(requested_components)):
        raise ValueError("--components must not contain duplicates")
    if (
        requested_components
        and args.component_family == "hidden_language"
        and args.representations != ["last_token"]
    ):
        raise ValueError(
            "Hidden-language components are currently defined only for "
            "--representations last_token"
        )

    model_path = Path(args.model).expanduser().resolve()
    model = str(model_path) if model_path.is_dir() else args.model
    args.model = model
    model_id = get_model_id(model)
    output_root = (
        args.output_root.expanduser().resolve()
        if args.output_root is not None
        else get_output_base()
    )
    prompts = load_prompts(args.prompts, args.language)
    prompts = prompts[args.prompt_start:]
    if not prompts:
        raise ValueError("--prompt-start leaves no prompts")
    if args.max_prompts is not None:
        if args.max_prompts < 1:
            raise ValueError("--max-prompts must be positive")
        prompts = prompts[:args.max_prompts]

    steering_base = (
        output_root
        / args.language
        / "03_steering_vectors"
        / model_id
        / args.dataset_tag
    )
    evaluation_base = (
        output_root
        / args.language
        / "02_separability"
        / model_id
        / args.dataset_tag
    )
    generator = SteeredGenerator(
        model,
        device=args.device,
        torch_dtype=args.torch_dtype,
        attention_backend=args.attention_backend,
        allow_tf32=not args.no_tf32,
    )
    if args.chat_template:
        prompts = apply_chat_template_to_prompts(
            prompts,
            generator.tokenizer,
            continuation_instruction=args.chat_continuation_instruction,
        )

    warmed_up = False
    component_sequence: list[str | None] = requested_components or [None]
    for representation in args.representations:
        representation_dir = steering_base / representation
        for component in component_sequence:
            args.component = component
            method_key = run_method_key(args)
            if component:
                component_root = (
                    args.component_root.expanduser().resolve()
                    if args.component_root is not None
                    else (
                        output_root
                        / args.language
                        / "05_linguistic"
                        / args.component_family
                        / model_id
                        / representation
                    )
                )
                (
                    selected_layer,
                    checkpoint_path,
                    selection_path,
                    selection_table,
                ) = resolve_component_assets(
                    component_root,
                    component,
                    args.layer,
                )
            else:
                selection_path = (
                    evaluation_base
                    / representation
                    / "generator_holdout.json"
                )
                selected_layer, selection_table = select_robust_layer(
                    selection_path,
                    args.method,
                )
                if args.layer is not None:
                    selected_layer = args.layer
                checkpoint_path = (
                    representation_dir
                    / "models"
                    / f"{method_key}_layer{selected_layer:02d}.pt"
                )
            states_path = (
                representation_dir
                / "states"
                / f"paired_states_layer{selected_layer:02d}.pt"
            )
            steering, calibration = load_and_calibrate_direction(
                checkpoint_path,
                states_path,
            )
            default_output_subdir = (
                (
                    "component_generation"
                    if args.component_family == "hidden_language"
                    else f"{args.component_family}_generation"
                )
                if component
                else "generation"
            )
            output_dir = representation_dir / (
                args.output_subdir or default_output_subdir
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[{model_id} · {representation} · "
                f"{component or args.method}] L{selected_layer} "
                f"projection_std={calibration['projection_std']:.4f}",
                flush=True,
            )
            if not args.no_warmup and not warmed_up:
                warm_up(generator, prompts[0]["text"], steering)
                warmed_up = True

            lock_path = output_dir / f".{method_key}.lock"
            with exclusive_lock(lock_path):
                run = ResumableRepresentationRun(
                    args=args,
                    generator=generator,
                    model_id=model_id,
                    representation=representation,
                    prompts=prompts,
                    steering=steering,
                    selected_layer=selected_layer,
                    selection_table=selection_table,
                    selection_path=selection_path,
                    calibration=calibration,
                    output_dir=output_dir,
                )
                run.run()


if __name__ == "__main__":
    main()
