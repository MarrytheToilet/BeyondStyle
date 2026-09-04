#!/usr/bin/env python3
"""Audit domain and prompt heterogeneity in the held-out generation result."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, kruskal, spearmanr


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ANALYSIS = (
    ROOT
    / "outputs/en/05_linguistic/causal_pullback/Llama-3.1-8B-Instruct"
    / "mean_pool/strength2_seed_robustness_dual5090/prompt_seed_estimands.csv"
)
DEFAULT_GENERATIONS = (
    ROOT
    / "outputs/en/03_steering_vectors/Llama-3.1-8B-Instruct"
    / "proposition_aligned/mean_pool"
    / "causal_pullback_strength2_seed_robustness_dual5090"
    / "generation_component_lexical.json"
)
DEFAULT_PROMPTS = ROOT / "data/prompts/en/formal.json"
DEFAULT_OUT = DEFAULT_ANALYSIS.parent / "prompt_heterogeneity"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--estimands", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--generations", type=Path, default=DEFAULT_GENERATIONS)
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--examples-per-pole", type=int, default=5)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap_mean(values: np.ndarray, *, draws: int, rng: np.random.Generator) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, [0.025, 0.975]))


def load_prompt_metadata(path: Path) -> pd.DataFrame:
    prompts = json.loads(path.read_text())
    if not isinstance(prompts, list):
        raise ValueError("Prompt file must contain a JSON list")
    rows = []
    for index, prompt in enumerate(prompts):
        rows.append(
            {
                "prompt_id": prompt["id"],
                "prompt_index_from_file": index,
                "domain": prompt["domain"],
                "source": prompt["source"],
                "prompt_words": len(prompt["text"].split()),
                "prompt": prompt["text"],
            }
        )
    return pd.DataFrame(rows)


def validate_and_aggregate(estimands: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    required = {
        "prompt_id",
        "prompt_index",
        "sample_seed_index",
        "component",
        "estimand",
        "value",
    }
    missing = required - set(estimands.columns)
    if missing:
        raise ValueError(f"Missing estimand columns: {sorted(missing)}")
    if estimands[list(required)].isna().any().any():
        raise ValueError("Required estimand fields contain missing values")
    if set(estimands["component"]) != {"lexical"}:
        raise ValueError("Expected only the lexical intervention component")
    key = ["prompt_id", "sample_seed_index", "estimand"]
    if estimands.duplicated(key).any():
        raise ValueError("Duplicate prompt-seed-estimand rows")
    expected_estimands = {"target_projection", "component_selectivity"}
    if set(estimands["estimand"]) != expected_estimands:
        raise ValueError(f"Unexpected estimands: {sorted(set(estimands['estimand']))}")
    seed_counts = estimands.groupby(["prompt_id", "estimand"])["sample_seed_index"].nunique()
    if not (seed_counts == 3).all():
        raise ValueError("Every prompt-estimand pair must contain exactly three seeds")

    prompt_rows = (
        estimands.groupby(["prompt_id", "prompt_index", "estimand"], as_index=False)
        .agg(value=("value", "mean"), seed_sd=("value", "std"))
        .merge(metadata, on="prompt_id", how="left", validate="many_to_one")
    )
    if prompt_rows["domain"].isna().any():
        raise ValueError("Some estimand prompt IDs are absent from the prompt bank")
    if not (prompt_rows["prompt_index"] == prompt_rows["prompt_index_from_file"]).all():
        raise ValueError("Prompt indices disagree with the frozen prompt bank")
    if prompt_rows["prompt_id"].nunique() != 166:
        raise ValueError("Expected 166 held-out prompts")
    return prompt_rows


def summarize(prompt_rows: pd.DataFrame, *, draws: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    rows = []
    diagnostics = []
    for estimand, estimand_rows in prompt_rows.groupby("estimand", sort=True):
        groups = [group["value"].to_numpy() for _, group in estimand_rows.groupby("domain")]
        kw = kruskal(*groups)
        rho = spearmanr(estimand_rows["prompt_words"], estimand_rows["value"])
        diagnostics.append(
            {
                "estimand": estimand,
                "domain_kruskal_h": float(kw.statistic),
                "domain_kruskal_p": float(kw.pvalue),
                "prompt_length_spearman_rho": float(rho.statistic),
                "prompt_length_spearman_p": float(rho.pvalue),
            }
        )
        for domain, group in [("all", estimand_rows), *estimand_rows.groupby("domain", sort=True)]:
            values = group["value"].to_numpy(dtype=float)
            low, high = bootstrap_mean(values, draws=draws, rng=rng)
            reversals = int(np.sum(values < 0))
            reversal_ci = binomtest(reversals, len(values)).proportion_ci(confidence_level=0.95)
            rows.append(
                {
                    "estimand": estimand,
                    "domain": domain,
                    "n_prompts": len(values),
                    "mean": float(values.mean()),
                    "ci_low": low,
                    "ci_high": high,
                    "median": float(np.median(values)),
                    "sign_reversal_count": reversals,
                    "sign_reversal_rate": float(reversals / len(values)),
                    "sign_reversal_ci_low": float(reversal_ci.low),
                    "sign_reversal_ci_high": float(reversal_ci.high),
                    "mean_seed_sd": float(group["seed_sd"].mean()),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(diagnostics)


def select_examples(
    prompt_rows: pd.DataFrame,
    estimands: pd.DataFrame,
    generations_path: Path,
    *,
    per_pole: int,
) -> pd.DataFrame:
    targets = prompt_rows[prompt_rows["estimand"] == "target_projection"].copy()
    selected = pd.concat(
        [
            targets.nsmallest(per_pole, "value").assign(pole="sign_reversal"),
            targets.nlargest(per_pole, "value").assign(pole="strong_shift"),
        ],
        ignore_index=True,
    )
    target_seed = estimands[estimands["estimand"] == "target_projection"].copy()
    target_seed = target_seed.merge(
        selected[["prompt_id", "value"]].rename(columns={"value": "prompt_mean"}),
        on="prompt_id",
        validate="many_to_one",
    )
    target_seed["distance"] = (target_seed["value"] - target_seed["prompt_mean"]).abs()
    representatives = (
        target_seed.sort_values(["prompt_id", "distance", "sample_seed_index"])
        .drop_duplicates("prompt_id")
        [["prompt_id", "sample_seed_index"]]
    )
    generation_rows = json.loads(generations_path.read_text())
    lookup = {
        (row["prompt_id"], int(row["sample_seed_index"])): row
        for row in generation_rows
    }
    records = []
    for row in selected.merge(representatives, on="prompt_id").itertuples(index=False):
        generated = lookup[(row.prompt_id, int(row.sample_seed_index))]
        strengths = sorted(float(value) for value in generated["outputs"])
        negative, positive = strengths[0], strengths[-1]
        records.append(
            {
                "pole": row.pole,
                "prompt_id": row.prompt_id,
                "prompt_index": row.prompt_index,
                "domain": row.domain,
                "target_projection": row.value,
                "representative_seed": int(row.sample_seed_index),
                "prompt": row.prompt,
                "negative_text": generated["outputs"][str(negative)]["text"],
                "positive_text": generated["outputs"][str(positive)]["text"],
            }
        )
    return pd.DataFrame(records)


def write_report(summary: pd.DataFrame, diagnostics: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# Held-out prompt heterogeneity",
        "",
        "The analysis unit is the prompt mean over three paired decode seeds.",
        "Negative values are described as sign reversals, not as failed generations.",
        "",
        "## Domain summary",
        "",
        "| Estimand | Domain | n | Mean [95% CI] | Sign reversals |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.estimand} | {row.domain} | {row.n_prompts} | "
            f"{row.mean:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}] | "
            f"{row.sign_reversal_count}/{row.n_prompts} ({row.sign_reversal_rate:.1%}) |"
        )
    lines.extend(["", "## Diagnostics", ""])
    for row in diagnostics.itertuples(index=False):
        lines.append(
            f"- {row.estimand}: domain Kruskal--Wallis "
            f"H={row.domain_kruskal_h:.3f}, p={row.domain_kruskal_p:.4f}; "
            f"prompt-length Spearman rho={row.prompt_length_spearman_rho:.3f}, "
            f"p={row.prompt_length_spearman_p:.4f}."
        )
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    started = datetime.now(timezone.utc)
    estimands = pd.read_csv(args.estimands)
    metadata = load_prompt_metadata(args.prompts)
    prompt_rows = validate_and_aggregate(estimands, metadata)
    summary, diagnostics = summarize(prompt_rows, draws=args.bootstrap, seed=args.seed)
    examples = select_examples(
        prompt_rows,
        estimands,
        args.generations,
        per_pole=args.examples_per_pole,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = args.out_dir / "prompt_estimands_with_metadata.csv"
    summary_path = args.out_dir / "domain_summary.csv"
    diagnostics_path = args.out_dir / "diagnostics.csv"
    examples_path = args.out_dir / "qualitative_examples.csv"
    report_path = args.out_dir / "analysis_summary.md"
    prompt_rows.to_csv(prompt_path, index=False)
    summary.to_csv(summary_path, index=False)
    diagnostics.to_csv(diagnostics_path, index=False)
    examples.to_csv(examples_path, index=False)
    write_report(summary, diagnostics, report_path)

    finished = datetime.now(timezone.utc)
    manifest = {
        "schema": "prompt_heterogeneity_v1",
        "status": "complete",
        "analysis_unit": "prompt mean over three paired decode seeds",
        "inputs": {
            str(args.estimands): sha256(args.estimands),
            str(args.generations): sha256(args.generations),
            str(args.prompts): sha256(args.prompts),
        },
        "outputs": {
            str(path): sha256(path)
            for path in [prompt_path, summary_path, diagnostics_path, examples_path, report_path]
        },
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(report_path.read_text())


if __name__ == "__main__":
    main()
