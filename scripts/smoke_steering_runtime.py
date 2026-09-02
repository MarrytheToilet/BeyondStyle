#!/usr/bin/env python3
"""Run a tiny real-model test of the optimized steering generation path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.generation import SteeredGenerationConfig, SteeredGenerator
from src.steering import MeanDiffSteering
from src.steering.hooks import _get_decoder_layers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--layer", type=int)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument(
        "--cache-implementation",
        choices=["dynamic", "static"],
        default="dynamic",
    )
    parser.add_argument("--pad-to-multiple-of", type=int, default=8)
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
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.pad_to_multiple_of < 1:
        raise ValueError("--pad-to-multiple-of must be positive")
    generator = SteeredGenerator(
        args.model,
        device=args.device,
        torch_dtype=args.torch_dtype,
        attention_backend=args.attention_backend,
    )
    model = generator.model
    layers = _get_decoder_layers(model)
    layer = args.layer or max(1, len(layers) // 3)
    if not 1 <= layer <= len(layers):
        raise ValueError(f"--layer must be in [1, {len(layers)}]")

    model_config = getattr(model.config, "text_config", model.config)
    hidden_size = getattr(model_config, "hidden_size", None)
    if hidden_size is None:
        raise ValueError(
            "Could not infer hidden_size from the model text configuration"
        )
    steering = MeanDiffSteering(hidden_size=hidden_size, layer=layer)
    direction = torch.randn(hidden_size)
    direction /= direction.norm()
    steering.v_src.copy_(direction)
    steering.projection_std.fill_(0.25)
    steering.is_fitted.fill_(True)

    prompt_bank = [
        "The committee published its findings",
        "After reviewing the available evidence",
        "The research team documented the procedure",
        "Several explanations remain compatible with the observations",
    ]
    prompts = [
        prompt_bank[index % len(prompt_bank)]
        for index in range(args.batch_size)
    ]
    config = SteeredGenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=0.7,
        top_p=0.95,
        seed=8236,
        paired_sampling=True,
        use_cache=True,
        cache_implementation=args.cache_implementation,
        pad_to_multiple_of=args.pad_to_multiple_of,
        apply_to_positions="decode_only",
        strength_unit="projection_std",
    )
    outputs = generator.generate_batch_comparison(
        prompts,
        steering,
        strengths=[-0.5, 0.0, 0.5],
        config=config,
    )
    stats = generator.last_comparison_stats
    for strength in (-0.5, 0.5):
        hook_stats = stats[strength]["steering"]["hook_stats"]
        if hook_stats["prefill_calls_skipped"] != 1:
            raise RuntimeError(
                f"prefill was not skipped exactly once at beta={strength}"
            )
        expected_decode_calls = max(args.max_new_tokens - 1, 0)
        if hook_stats["steered_calls"] > expected_decode_calls:
            raise RuntimeError("steering affected more calls than cached decoding")
    if stats[0.0]["steering"]["enabled"]:
        raise RuntimeError("baseline unexpectedly installed a steering hook")

    report = {
        "ok": True,
        "model": args.model,
        "layer": layer,
        "batch_size": len(prompts),
        "cache_implementation": args.cache_implementation,
        "pad_to_multiple_of": args.pad_to_multiple_of,
        "strength_unit": "projection_std",
        "position_mode": "decode_only",
        "prompt_prefill_untouched": True,
        "model_load": generator.model_load_info,
        "stats": {str(key): value for key, value in stats.items()},
        "output_previews": {
            str(strength): [text[:120] for text in texts]
            for strength, texts in outputs.items()
        },
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.json_out:
        atomic_json(args.json_out, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
