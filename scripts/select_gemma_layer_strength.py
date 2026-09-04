#!/usr/bin/env python3
"""Freeze a low-dose Gemma injection layer and strength on development data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import statistics

import pandas as pd


WORD_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?")


def generation_quality(layer: int, strength: float) -> dict[str, float]:
    layer_tag = f"{layer:02d}"
    strength_tag = str(strength).replace(".", "p")
    path = Path(
        "outputs/en/03_steering_vectors/gemma-4-E2B-it/"
        "proposition_aligned/mean_pool/"
        f"causal_pullback_gemma_lengthmatched_layer{layer_tag}_"
        f"strength{strength_tag}_dev_generation/"
        "generation_component_lexical.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    endpoint_stats = []
    for endpoint in sorted(payload[0]["outputs"], key=float):
        word_counts = []
        repetition_rates = []
        ascii_letter_rates = []
        for row in payload:
            text = row["outputs"][endpoint]["text"]
            words = WORD_PATTERN.findall(text)
            word_counts.append(len(words))
            fourgrams = [
                tuple(words[index : index + 4])
                for index in range(max(0, len(words) - 3))
            ]
            repetition_rates.append(
                0.0
                if not fourgrams
                else 1.0 - len(set(fourgrams)) / len(fourgrams)
            )
            letters = [character for character in text if character.isalpha()]
            ascii_letter_rates.append(
                1.0
                if not letters
                else sum(character.isascii() for character in letters)
                / len(letters)
            )
        endpoint_stats.append(
            {
                "mean_words": statistics.mean(word_counts),
                "median_words": statistics.median(word_counts),
                "short_rate": sum(value < 20 for value in word_counts)
                / len(word_counts),
                "mean_fourgram_repetition": statistics.mean(repetition_rates),
                "mean_ascii_letter_rate": statistics.mean(ascii_letter_rates),
            }
        )
    means = [item["mean_words"] for item in endpoint_stats]
    return {
        "minimum_endpoint_median_words": min(
            item["median_words"] for item in endpoint_stats
        ),
        "maximum_endpoint_short_rate": max(
            item["short_rate"] for item in endpoint_stats
        ),
        "maximum_endpoint_fourgram_repetition": max(
            item["mean_fourgram_repetition"] for item in endpoint_stats
        ),
        "endpoint_mean_word_count_ratio": min(means) / max(means),
        "minimum_endpoint_ascii_letter_rate": min(
            item["mean_ascii_letter_rate"] for item in endpoint_stats
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "outputs/en/05_linguistic/causal_pullback/gemma-4-E2B-it/"
            "mean_pool/injection_layer_sweep/lengthmatched_dev"
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            "outputs/en/05_linguistic/causal_pullback/gemma-4-E2B-it/"
            "mean_pool/injection_layer_sweep/selected_lengthmatched_config.json"
        ),
    )
    args = parser.parse_args()

    rows: list[dict] = []
    for path in sorted(args.root.glob("layer*/strength*/seed_robustness_summary.csv")):
        layer = int(path.parents[1].name.removeprefix("layer"))
        strength = float(
            path.parent.name.removeprefix("strength").replace("p", ".")
        )
        summary = pd.read_csv(path)
        lexical = summary[summary["component"] == "lexical"].set_index("estimand")
        target = lexical.loc["target_projection"]
        selectivity = lexical.loc["component_selectivity"]
        quality = generation_quality(layer, strength)
        rows.append(
            {
                "layer": layer,
                "strength": strength,
                "target_mean": float(target["mean"]),
                "target_ci_low": float(target["ci_low"]),
                "target_seed_min": float(target["seed_mean_min"]),
                "selectivity_mean": float(selectivity["mean"]),
                "selectivity_ci_low": float(selectivity["ci_low"]),
                "selectivity_seed_min": float(selectivity["seed_mean_min"]),
                **quality,
            }
        )
    if not rows:
        raise FileNotFoundError(f"No completed development cells below {args.root}")

    table = pd.DataFrame(rows)
    table["quality_eligible"] = (
        (table["minimum_endpoint_median_words"] >= 40)
        & (table["maximum_endpoint_short_rate"] <= 0.20)
        & (table["maximum_endpoint_fourgram_repetition"] <= 0.20)
        & (table["endpoint_mean_word_count_ratio"] >= 0.65)
        & (table["minimum_endpoint_ascii_letter_rate"] >= 0.98)
    )
    table["eligible"] = (
        (table["target_mean"] > 0)
        & (table["selectivity_mean"] > 0)
        & (table["target_seed_min"] > 0)
        & (table["selectivity_seed_min"] > 0)
        & table["quality_eligible"]
    )
    table["selection_score"] = table[
        ["target_mean", "selectivity_mean"]
    ].min(axis=1)
    candidates = table[table["eligible"]].copy()
    gate_passed = not candidates.empty
    if candidates.empty:
        candidates = table.copy()
    chosen = candidates.sort_values(
        ["selection_score", "strength", "layer"],
        ascending=[False, True, True],
    ).iloc[0]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    table_path = args.out.with_name("development_lengthmatched_grid.csv")
    table.sort_values(["layer", "strength"]).to_csv(table_path, index=False)
    payload = {
        "selection_scope": "development prompts 0-19 only",
        "test_scope": "untouched prompts 34-199",
        "candidate_layers": sorted(table["layer"].unique().tolist()),
        "candidate_strengths": sorted(table["strength"].unique().tolist()),
        "eligibility_rule": (
            "target and component-selectivity means are positive, and both "
            "estimands are positive in every paired decoding seed; both "
            "endpoints must also pass the predeclared response-length, "
            "short-output, repetition, and length-ratio guardrails"
        ),
        "quality_rule": {
            "minimum_endpoint_median_words": 40,
            "maximum_endpoint_short_rate": 0.20,
            "maximum_endpoint_fourgram_repetition": 0.20,
            "minimum_endpoint_mean_word_count_ratio": 0.65,
            "minimum_endpoint_ascii_letter_rate": 0.98,
        },
        "ranking_rule": (
            "maximize min(target mean, component-selectivity mean); tie-break "
            "by smaller dose, then shallower layer"
        ),
        "gate_passed": bool(gate_passed),
        "selected_layer": int(chosen["layer"]),
        "selected_strength": float(chosen["strength"]),
        "selected_statistics": {
            key: float(chosen[key])
            for key in (
                "target_mean",
                "target_ci_low",
                "target_seed_min",
                "selectivity_mean",
                "selectivity_ci_low",
                "selectivity_seed_min",
                "selection_score",
                "minimum_endpoint_median_words",
                "maximum_endpoint_short_rate",
                "maximum_endpoint_fourgram_repetition",
                "endpoint_mean_word_count_ratio",
                "minimum_endpoint_ascii_letter_rate",
            )
        },
        "development_grid": str(table_path),
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
