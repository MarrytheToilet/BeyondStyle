#!/usr/bin/env python3
"""Verify that a public layer number selects the same state and hook site.

This is an integration smoke test for real Hugging Face checkpoints. It
compares the output captured from decoder block ``L`` with
``outputs.hidden_states[L]`` under BeyondStyle's one-based convention.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.layer_indexing import LAYER_CONVENTION, hidden_state_index
from src.extraction.hidden_states import HiddenStatesExtractor
from src.steering.hooks import get_layer_module


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local HF model directory")
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="bfloat16",
    )
    parser.add_argument("--text", default="Layer alignment smoke test.")
    return parser.parse_args()


def main():
    args = parse_args()
    local_only = os.path.isdir(args.model)
    load_kwargs = {
        "trust_remote_code": True,
        "local_files_only": local_only,
    }
    tokenizer = AutoTokenizer.from_pretrained(args.model, **load_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=getattr(torch, args.dtype),
        **load_kwargs,
    ).to(args.device)
    model.eval()

    # Multimodal wrappers such as Gemma 4 expose the language-backbone
    # dimensions under ``text_config`` rather than on the top-level config.
    # Keep this lookup aligned with the extraction and generation runtimes.
    text_config = getattr(model.config, "text_config", model.config)
    num_layers = text_config.num_hidden_layers
    layers = args.layers or [1, max(1, num_layers // 2), num_layers]
    captures = {}
    handles = []

    def make_hook(layer):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captures[layer] = hidden.detach().float().cpu()

        return hook

    for layer in layers:
        handles.append(
            get_layer_module(model, layer).register_forward_hook(make_hook(layer))
        )

    inputs = tokenizer(args.text, return_tensors="pt")
    inputs = {key: value.to(args.device) for key, value in inputs.items()}
    try:
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True,
                use_cache=False,
            )
    finally:
        for handle in handles:
            handle.remove()

    extractor = HiddenStatesExtractor(
        args.model,
        device=args.device,
        dtype=getattr(torch, args.dtype),
        model=model,
        tokenizer=tokenizer,
    )
    extracted = extractor.extract(
        args.text,
        aggregation="none",
        layers=layers,
    )

    results = []
    all_passed = True
    for layer in layers:
        hf_state = outputs.hidden_states[
            hidden_state_index(layer, num_layers)
        ].detach().float().cpu()
        observed = captures[layer]
        extracted_state = extracted[layer].float().cpu().unsqueeze(0)
        max_abs_diff = float((observed - extracted_state).abs().max())
        hf_max_abs_diff = float((observed - hf_state).abs().max())
        passed = bool(
            torch.allclose(observed, extracted_state, atol=1e-6, rtol=1e-5)
        )
        all_passed = all_passed and passed
        results.append(
            {
                "layer": layer,
                "hidden_state_index": layer,
                "decoder_block_index": layer - 1,
                "max_abs_diff": max_abs_diff,
                "hf_tuple_max_abs_diff": hf_max_abs_diff,
                "hf_tuple_may_include_final_norm": layer == num_layers,
                "passed": passed,
            }
        )

    report = {
        "model": str(Path(args.model).resolve()),
        "model_type": model.config.model_type,
        "num_hidden_layers": num_layers,
        "layer_convention": LAYER_CONVENTION,
        "results": results,
        "all_passed": all_passed,
    }
    print(json.dumps(report, indent=2))
    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
