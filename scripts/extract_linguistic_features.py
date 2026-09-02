#!/usr/bin/env python
"""Extract frozen English features and run paired source comparisons."""

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
import scipy
from sklearn.covariance import LedoitWolf
import sklearn
import spacy
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import get_dataset_path, get_linguistic_dir  # noqa: E402
from src.linguistics import (  # noqa: E402
    FEATURE_DEFINITIONS,
    EnglishLinguisticFeatureExtractor,
    feature_dictionary,
)


DEFAULT_SEED = 20260725


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["en"], default="en")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--spacy-model", default="en_core_web_sm")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-process", type=int, default=1)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--permutations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_pairs(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if limit is not None:
        rows = rows[:limit]
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("sample_id values are not unique")
    return rows


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = np.empty_like(ranked, dtype=np.float64)
    running = 1.0
    for index in range(len(ranked) - 1, -1, -1):
        rank = index + 1
        running = min(running, ranked[index] * len(ranked) / rank)
        adjusted[index] = running
    restored = np.empty_like(adjusted)
    restored[order] = adjusted
    return np.clip(restored, 0.0, 1.0)


def feature_statistics(
    differences: pd.DataFrame,
    bootstrap: int,
    permutations: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for definition in FEATURE_DEFINITIONS:
        feature = definition.name
        values = differences[feature].to_numpy(dtype=np.float64)
        observed = float(values.mean())
        sample_sd = float(values.std(ddof=1))
        if bootstrap:
            indices = rng.integers(0, len(values), size=(bootstrap, len(values)))
            bootstrap_means = values[indices].mean(axis=1)
            ci_low, ci_high = np.quantile(bootstrap_means, [0.025, 0.975])
        else:
            ci_low = ci_high = observed
        if permutations:
            signs = rng.choice(
                np.array([-1.0, 1.0]),
                size=(permutations, len(values)),
            )
            null_means = (signs * values).mean(axis=1)
            p_value = (1 + np.sum(np.abs(null_means) >= abs(observed))) / (
                permutations + 1
            )
        else:
            p_value = 1.0
        rows.append(
            {
                "feature": feature,
                "group": definition.group,
                "unit": definition.unit,
                "n_pairs": len(values),
                "human_mean": float(differences[f"human__{feature}"].mean()),
                "ai_mean": float(differences[f"ai__{feature}"].mean()),
                "paired_difference_mean": observed,
                "paired_difference_median": float(np.median(values)),
                "paired_difference_ci_low": float(ci_low),
                "paired_difference_ci_high": float(ci_high),
                "paired_d": observed / sample_sd if sample_sd > 1e-12 else 0.0,
                "positive_pair_rate": float(np.mean(values > 0)),
                "sign_flip_p": float(p_value),
            }
        )
    table = pd.DataFrame(rows)
    primary = table["group"] != "nuisance"
    table["fdr_q"] = 1.0
    table.loc[primary, "fdr_q"] = benjamini_hochberg(
        table.loc[primary, "sign_flip_p"].to_numpy()
    )
    return table


def group_statistics(
    differences: pd.DataFrame,
    permutations: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    result: dict[str, Any] = {}
    for group in ("lexical", "syntactic", "discourse"):
        names = [
            item.name for item in FEATURE_DEFINITIONS if item.group == group
        ]
        matrix = differences[names].to_numpy(dtype=np.float64)
        scales = matrix.std(axis=0, ddof=1, keepdims=True)
        active = scales.ravel() > 1e-12
        matrix = matrix[:, active]
        active_names = [name for name, keep in zip(names, active) if keep]
        standardized = matrix / scales[:, active]
        covariance = LedoitWolf().fit(standardized)
        mean_vector = standardized.mean(axis=0)
        statistic = float(mean_vector @ covariance.precision_ @ mean_vector)
        if permutations:
            signs = rng.choice(
                np.array([-1.0, 1.0]),
                size=(permutations, len(standardized)),
            )
            null_means = (
                signs[:, :, None] * standardized[None, :, :]
            ).mean(axis=1)
            null_statistics = np.einsum(
                "bi,ij,bj->b",
                null_means,
                covariance.precision_,
                null_means,
            )
            p_value = (1 + np.sum(null_statistics >= statistic)) / (
                permutations + 1
            )
        else:
            p_value = 1.0
        result[group] = {
            "n_pairs": int(len(standardized)),
            "n_features": len(active_names),
            "active_features": active_names,
            "mahalanobis_mean_shift": float(np.sqrt(max(statistic, 0.0))),
            "sign_flip_p": float(p_value),
        }
    return result


def _residualize_preserving_mean(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_nuisance: np.ndarray,
    test_nuisance: np.ndarray,
    ridge: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove nuisance slopes fitted on train without removing source means."""
    nuisance_mean = train_nuisance.mean(axis=0, keepdims=True)
    train_design = train_nuisance - nuisance_mean
    test_design = test_nuisance - nuisance_mean
    gram = train_design.T @ train_design
    penalty = ridge * max(float(np.trace(gram)) / max(len(gram), 1), 1.0)
    coefficients = np.linalg.solve(
        gram + penalty * np.eye(gram.shape[0]),
        train_design.T @ train_features,
    )
    return (
        train_features - train_design @ coefficients,
        test_features - test_design @ coefficients,
    )


def environment_holdout_statistics(
    differences: pd.DataFrame,
    bootstrap: int,
    seed: int,
) -> pd.DataFrame:
    """Fit each linguistic direction without one environment and test it."""
    rng = np.random.default_rng(seed)
    nuisance_names = [
        item.name for item in FEATURE_DEFINITIONS if item.group == "nuisance"
    ]
    nuisance = differences[nuisance_names].to_numpy(dtype=np.float64)
    rows: list[dict[str, Any]] = []
    for environment_type in ("generator", "domain", "source"):
        labels = differences[environment_type].astype(str).to_numpy()
        for heldout in sorted(np.unique(labels)):
            train_mask = labels != heldout
            test_mask = ~train_mask
            for group in ("lexical", "syntactic", "discourse"):
                names = [
                    item.name
                    for item in FEATURE_DEFINITIONS
                    if item.group == group
                ]
                matrix = differences[names].to_numpy(dtype=np.float64)
                train, test = _residualize_preserving_mean(
                    matrix[train_mask],
                    matrix[test_mask],
                    nuisance[train_mask],
                    nuisance[test_mask],
                )
                scales = train.std(axis=0, ddof=1)
                active = scales > 1e-12
                train = train[:, active] / scales[active]
                test = test[:, active] / scales[active]
                train_mean = train.mean(axis=0)
                norm = np.linalg.norm(train_mean)
                if norm <= 1e-12:
                    direction = np.zeros_like(train_mean)
                else:
                    direction = train_mean / norm
                train_scores = train @ direction
                test_scores = test @ direction
                test_mean = float(test_scores.mean())
                test_sd = float(test_scores.std(ddof=1))
                heldout_mean = test.mean(axis=0)
                heldout_norm = np.linalg.norm(heldout_mean)
                cosine = (
                    float(direction @ heldout_mean / heldout_norm)
                    if heldout_norm > 1e-12
                    else 0.0
                )
                if bootstrap:
                    indices = rng.integers(
                        0,
                        len(test_scores),
                        size=(bootstrap, len(test_scores)),
                    )
                    boot_means = test_scores[indices].mean(axis=1)
                    ci_low, ci_high = np.quantile(
                        boot_means,
                        [0.025, 0.975],
                    )
                else:
                    ci_low = ci_high = test_mean
                rows.append(
                    {
                        "environment_type": environment_type,
                        "heldout_environment": heldout,
                        "group": group,
                        "n_train": int(train_mask.sum()),
                        "n_test": int(test_mask.sum()),
                        "n_active_features": int(active.sum()),
                        "train_projection_mean": float(train_scores.mean()),
                        "test_projection_mean": test_mean,
                        "test_projection_ci_low": float(ci_low),
                        "test_projection_ci_high": float(ci_high),
                        "test_projection_d": (
                            test_mean / test_sd if test_sd > 1e-12 else 0.0
                        ),
                        "test_positive_pair_rate": float(
                            np.mean(test_scores > 0)
                        ),
                        "heldout_direction_cosine": cosine,
                        "nuisance_residualized": True,
                    }
                )
    return pd.DataFrame(rows)


def write_summary(
    path: Path,
    feature_table: pd.DataFrame,
    group_table: dict[str, Any],
    holdout_table: pd.DataFrame,
    prompt_families: list[str],
) -> None:
    lines = [
        "# Automatic linguistic feature analysis",
        "",
        "- Parser success rate: 1.000",
        f"- Matched pairs: {int(feature_table['n_pairs'].max())}",
        "- Difference convention: AI − Human",
        "- Inference: paired bootstrap intervals and paired sign-flip tests",
        "",
        "## Group-level omnibus shifts",
        "",
        "| Group | Features | Mahalanobis shift | Sign-flip p |",
        "|---|---:|---:|---:|",
    ]
    for group, values in group_table.items():
        lines.append(
            f"| {group} | {values['n_features']} | "
            f"{values['mahalanobis_mean_shift']:.3f} | "
            f"{values['sign_flip_p']:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Largest standardized paired differences",
            "",
            "| Feature | Group | AI − Human | Paired d | 95% CI | FDR q |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    primary = feature_table[feature_table["group"] != "nuisance"].copy()
    primary["magnitude"] = primary["paired_d"].abs()
    for row in primary.sort_values("magnitude", ascending=False).head(12).itertuples():
        lines.append(
            f"| {row.feature} | {row.group} | "
            f"{row.paired_difference_mean:.4f} | {row.paired_d:.3f} | "
            f"[{row.paired_difference_ci_low:.4f}, "
            f"{row.paired_difference_ci_high:.4f}] | {row.fdr_q:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Leave-one-environment-out stability",
            "",
            "Directions are fitted after train-fold nuisance residualization. "
            "A positive held-out projection means that the AI−Human shift "
            "agrees with the training environments.",
            "",
            "| Environment | Group | Minimum cosine | Minimum positive-pair rate | All mean shifts positive |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for (environment, group), subset in holdout_table.groupby(
        ["environment_type", "group"],
        sort=False,
    ):
        lines.append(
            f"| {environment} | {group} | "
            f"{subset['heldout_direction_cosine'].min():.3f} | "
            f"{subset['test_positive_pair_rate'].min():.3f} | "
            f"{bool((subset['test_projection_mean'] > 0).all())} |"
        )
    prompt_text = ", ".join(
        f"`{family}`" for family in prompt_families
    )
    lines.extend(
        [
            "",
            "The current paired corpus contains "
            f"{len(prompt_families)} prompt family/families "
            f"({prompt_text}), so within-corpus prompt-family holdout is not "
            "reported.",
            "",
            "These are prespecified automatic measurements, not human judgments. "
            "Feature-level significance is descriptive; the three group-level "
            "tests are the primary Phase-A endpoints.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    dataset = args.dataset or Path(get_dataset_path(args.language))
    out_dir = args.out_dir or (get_linguistic_dir(args.language) / "features")
    out_dir.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)

    rows = load_pairs(dataset, args.limit)
    if not rows:
        raise ValueError("No paired rows found")
    nlp = spacy.load(args.spacy_model)
    extractor = EnglishLinguisticFeatureExtractor()

    parse_inputs: list[tuple[str, dict[str, Any]]] = []
    for row in rows:
        common = {
            "sample_id": row["sample_id"],
            "pair_id": row.get("pair_id", row["sample_id"]),
            "source": row.get("source", ""),
            "domain": row.get("domain", ""),
            "generator": row.get("generation", {}).get("generator_model", ""),
            "prompt_family": row.get("generation", {}).get("prompt_family", ""),
        }
        parse_inputs.append(
            (row["human_continuation"], {**common, "source_label": "human"})
        )
        parse_inputs.append(
            (row["ai_continuation"], {**common, "source_label": "ai"})
        )

    records: list[dict[str, Any]] = []
    texts = [item[0] for item in parse_inputs]
    metadata = [item[1] for item in parse_inputs]
    documents = nlp.pipe(
        texts,
        batch_size=args.batch_size,
        n_process=args.n_process,
    )
    for text, meta, doc in tqdm(
        zip(texts, metadata, documents),
        total=len(texts),
        desc="linguistic features",
    ):
        records.append({**meta, **extractor.extract(doc, text)})

    text_table = pd.DataFrame(records)
    feature_names = [item.name for item in FEATURE_DEFINITIONS]
    if not np.isfinite(text_table[feature_names].to_numpy()).all():
        raise ValueError("Feature table contains non-finite values")
    if len(text_table) != 2 * len(rows):
        raise RuntimeError("Parser did not return one document per text")

    ordered_ids = [row["sample_id"] for row in rows]
    human = (
        text_table[text_table["source_label"] == "human"]
        .set_index("sample_id")
        .loc[ordered_ids]
    )
    ai = (
        text_table[text_table["source_label"] == "ai"]
        .set_index("sample_id")
        .loc[ordered_ids]
    )
    paired = pd.DataFrame(
        {
            "sample_id": ordered_ids,
            "pair_id": [row.get("pair_id", row["sample_id"]) for row in rows],
            "source": [row.get("source", "") for row in rows],
            "domain": [row.get("domain", "") for row in rows],
            "generator": [
                row.get("generation", {}).get("generator_model", "")
                for row in rows
            ],
            "prompt_family": [
                row.get("generation", {}).get("prompt_family", "")
                for row in rows
            ],
        }
    )
    feature_columns: dict[str, np.ndarray] = {}
    for name in feature_names:
        human_values = human[name].to_numpy(dtype=np.float64)
        ai_values = ai[name].to_numpy(dtype=np.float64)
        feature_columns[f"human__{name}"] = human_values
        feature_columns[f"ai__{name}"] = ai_values
        feature_columns[name] = ai_values - human_values
    paired = pd.concat([paired, pd.DataFrame(feature_columns)], axis=1)

    statistics = feature_statistics(
        paired,
        bootstrap=args.bootstrap,
        permutations=args.permutations,
        seed=args.seed,
    )
    groups = group_statistics(
        paired,
        permutations=args.permutations,
        seed=args.seed + 1,
    )
    holdouts = environment_holdout_statistics(
        paired,
        bootstrap=args.bootstrap,
        seed=args.seed + 2,
    )
    dictionary = feature_dictionary()

    text_table.to_csv(out_dir / "text_features.csv", index=False)
    paired.to_csv(out_dir / "paired_feature_differences.csv", index=False)
    statistics.to_csv(out_dir / "feature_statistics.csv", index=False)
    holdouts.to_csv(out_dir / "environment_holdout_statistics.csv", index=False)
    (out_dir / "group_statistics.json").write_text(
        json.dumps(groups, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (out_dir / "feature_dictionary.json").write_text(
        json.dumps(dictionary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_summary(
        out_dir / "analysis_summary.md",
        statistics,
        groups,
        holdouts,
        sorted(paired["prompt_family"].astype(str).unique()),
    )

    finished = datetime.now(timezone.utc)
    manifest = {
        "schema": "automatic_linguistic_features",
        "language": args.language,
        "dataset": str(dataset),
        "dataset_sha256": sha256_file(dataset),
        "feature_dictionary_sha256": sha256_json(dictionary),
        "spacy_model": args.spacy_model,
        "spacy_version": spacy.__version__,
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "scipy_version": scipy.__version__,
        "sklearn_version": sklearn.__version__,
        "seed": args.seed,
        "bootstrap": args.bootstrap,
        "permutations": args.permutations,
        "n_pairs": len(rows),
        "n_texts": len(text_table),
        "parser_success_rate": 1.0,
        "started_at_utc": started.isoformat(),
        "finished_at_utc": finished.isoformat(),
        "duration_seconds": (finished - started).total_seconds(),
        "command": " ".join(sys.argv),
        "outputs": [
            "text_features.csv",
            "paired_feature_differences.csv",
            "feature_statistics.csv",
            "environment_holdout_statistics.csv",
            "group_statistics.json",
            "feature_dictionary.json",
            "analysis_summary.md",
        ],
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote {len(text_table)} texts and {len(paired)} pairs to {out_dir}")


if __name__ == "__main__":
    main()
