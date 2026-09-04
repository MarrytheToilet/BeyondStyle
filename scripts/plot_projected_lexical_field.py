#!/usr/bin/env python3
"""Plot recognizable AI-associated expressions in hidden coordinates.

The plot is not a frequency ranking. For each expression, every machine-text
occurrence is projected onto the leave-one-generator-out source direction and
centered by its continuation's mean coordinate. Repeated occurrences are
averaged within a document, so each document contributes once. Multi-token
expressions use the mean coordinate of their constituent tokens.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/beyondstyle-matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.analysis.token_projection import (  # noqa: E402
    normalize_token_text,
    pair_balanced_standardize,
)


TOKEN_ROOT = ROOT / "outputs" / "en" / "05_linguistic" / "token_projection"
SUMMARY_PATH = TOKEN_ROOT / "ai_cue_projection_summary.csv"
FIGURE_DIR = ROOT / "ACL27" / "figures"
PREVIEW_DIR = ROOT / "ACL27" / "build" / "previews"

MODELS = (
    "Llama-3.1-8B-Instruct",
    "Qwen3-4B",
    "gemma-4-E2B-it",
)
MODEL_LABEL = {
    "Llama-3.1-8B-Instruct": "Llama",
    "Qwen3-4B": "Qwen",
    "gemma-4-E2B-it": "Gemma 4",
}
MODEL_MARKER = {
    "Llama-3.1-8B-Instruct": "o",
    "Qwen3-4B": "s",
    "gemma-4-E2B-it": "^",
}
REPORT_LAYER = {
    "Llama-3.1-8B-Instruct": 11,
    "Qwen3-4B": 8,
    "gemma-4-E2B-it": 12,
}

# Chosen for linguistic recognizability, then evaluated by projection. The
# values below never enter selection or ordering.
EXPRESSIONS = (
    "crucial",
    "essential",
    "ensuring",
    "significant",
    "ultimately",
    "furthermore",
    "additionally",
    "stated that",
)
Y_POSITION = {expression: float(index) for index, expression in enumerate(EXPRESSIONS)}
MODEL_OFFSET = {
    "Llama-3.1-8B-Instruct": -0.19,
    "Qwen3-4B": 0.00,
    "gemma-4-E2B-it": 0.19,
}

# Kept in step with scripts/paper_figstyle.py so all six figures read as one
# system; the two hues are the validated categorical pair.
INK = "#1E2732"
MUTED = "#67717E"
GRID = "#E6ECF2"
BLUE = "#2E7EB8"
BLUE_DARK = "#2E7EB8"
BLUE_LIGHT = "#F1F8FD"
PINK_DARK = "#C9527F"
PINK_LIGHT = "#FEF4F7"

N_BOOTSTRAP = 10_000
N_SIGN_FLIPS = 20_000


def stable_seed(*parts: str) -> int:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def load_model_tokens(model: str) -> pd.DataFrame:
    columns = [
        "model",
        "direction_scheme",
        "sample_id",
        "source",
        "generator",
        "domain",
        "token_text",
        "token_index",
        "axis_family",
        "layer",
        "projection",
    ]
    frames = []
    for path in sorted((TOKEN_ROOT / model / "token_records").glob("part_*.parquet")):
        frame = pd.read_parquet(path, columns=columns)
        frame = frame[
            frame["axis_family"].eq("mean_pool")
            & frame["layer"].eq(REPORT_LAYER[model])
        ]
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No token records found for {model}")
    tokens = pd.concat(frames, ignore_index=True)
    tokens, _ = pair_balanced_standardize(tokens)
    tokens = tokens[tokens["source"].eq("ai")].copy()
    tokens["token_normalized"] = tokens["token_text"].map(normalize_token_text)
    return tokens


def expression_documents(tokens: pd.DataFrame, expression: str) -> pd.DataFrame:
    target = expression.split()
    records: list[dict[str, object]] = []
    document_keys = ["sample_id", "generator", "domain"]
    for key, group in tokens.groupby(document_keys, observed=True, sort=False):
        group = group.sort_values("token_index", kind="stable")
        normalized = group["token_normalized"].tolist()
        coordinates = group["projection_within_document_z"].to_numpy(dtype=float)
        occurrences = []
        width = len(target)
        for start in range(0, len(normalized) - width + 1):
            if normalized[start : start + width] == target:
                occurrences.append(float(coordinates[start : start + width].mean()))
        if occurrences:
            records.append({
                "sample_id": key[0],
                "generator": key[1],
                "domain": key[2],
                "document_projection": float(np.mean(occurrences)),
                "n_occurrences": len(occurrences),
            })
    return pd.DataFrame.from_records(records)


def bootstrap_ci(values: np.ndarray, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(N_BOOTSTRAP, dtype=float)
    for start in range(0, N_BOOTSTRAP, 1000):
        stop = min(start + 1000, N_BOOTSTRAP)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def sign_flip_p(values: np.ndarray, seed: int) -> float:
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    exceedances = 0
    for start in range(0, N_SIGN_FLIPS, 1000):
        batch = min(1000, N_SIGN_FLIPS - start)
        signs = rng.integers(0, 2, size=(batch, len(values)), dtype=np.int8) * 2 - 1
        null_means = (signs * values).mean(axis=1)
        exceedances += int(np.count_nonzero(null_means >= observed))
    return (exceedances + 1.0) / (N_SIGN_FLIPS + 1.0)


def bh_adjust(p_values: pd.Series) -> pd.Series:
    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0.0, 1.0)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return pd.Series(result, index=p_values.index)


def environment_summary(documents: pd.DataFrame, column: str) -> tuple[int, int]:
    means = documents.groupby(column, observed=True)["document_projection"].mean()
    return int((means > 0).sum()), int(len(means))


def build_summary() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in MODELS:
        tokens = load_model_tokens(model)
        for expression in EXPRESSIONS:
            documents = expression_documents(tokens, expression)
            if documents.empty:
                raise ValueError(f"No occurrences of {expression!r} for {model}")
            values = documents["document_projection"].to_numpy(dtype=float)
            ci_low, ci_high = bootstrap_ci(
                values, stable_seed(model, expression, "bootstrap")
            )
            generator_positive, n_generators = environment_summary(documents, "generator")
            domain_positive, n_domains = environment_summary(documents, "domain")
            rows.append({
                "model": model,
                "expression": expression,
                "n_documents": len(documents),
                "n_occurrences": int(documents["n_occurrences"].sum()),
                "mean_projection": float(values.mean()),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "p_machineward": sign_flip_p(
                    values, stable_seed(model, expression, "sign-flip")
                ),
                "positive_generators": generator_positive,
                "n_generators": n_generators,
                "positive_domains": domain_positive,
                "n_domains": n_domains,
                "coordinate_definition": (
                    "machine-source expression span projected onto the "
                    "leave-one-generator-out source direction, centered by "
                    "the continuation mean, and averaged once per document"
                ),
            })
    summary = pd.DataFrame(rows)
    summary["q_machineward"] = (
        summary.groupby("model", sort=False, group_keys=False)["p_machineward"]
        .apply(bh_adjust)
    )
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)
    return summary


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 7.2,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.edgecolor": "#C2CDD8",
        "axes.labelcolor": INK,
        "axes.labelsize": 7.2,
        "axes.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": INK,
        "xtick.labelsize": 6.6,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 6.4,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def main() -> None:
    setup_style()
    data = build_summary()
    figure, axis = plt.subplots(figsize=(3.031, 2.00))

    axis.axvspan(-0.65, 0.0, color=PINK_LIGHT, alpha=0.65, zorder=0)
    axis.axvspan(0.0, 4.45, color=BLUE_LIGHT, alpha=0.65, zorder=0)
    for y in Y_POSITION.values():
        axis.axhline(y, color=GRID, linewidth=0.50, zorder=0)
    axis.axvline(0.0, color=INK, linewidth=0.7, zorder=1)

    for model in MODELS:
        view = data[data["model"].eq(model)].set_index("expression")
        for expression in EXPRESSIONS:
            row = view.loc[expression]
            estimate = float(row["mean_projection"])
            low = float(row["ci_low"])
            high = float(row["ci_high"])
            significant = float(row["q_machineward"]) < 0.05
            y = Y_POSITION[expression] + MODEL_OFFSET[model]
            axis.errorbar(
                estimate,
                y,
                xerr=[[estimate - low], [high - estimate]],
                fmt=MODEL_MARKER[model],
                markersize=3.6,
                markerfacecolor=BLUE if significant else "white",
                markeredgecolor=BLUE_DARK,
                markeredgewidth=0.7,
                ecolor=BLUE_DARK,
                elinewidth=0.62,
                capsize=1.25,
                capthick=0.62,
                alpha=0.94,
                zorder=3,
            )

    axis.set_xlim(-0.65, 4.45)
    axis.set_ylim(-0.55, len(EXPRESSIONS) - 0.45)
    axis.set_xticks([0, 1, 2, 3, 4])
    axis.set_yticks([Y_POSITION[expression] for expression in EXPRESSIONS])
    axis.set_yticklabels(EXPRESSIONS)
    axis.invert_yaxis()
    axis.set_xlabel("Machineward projection (within-document z)")
    axis.tick_params(axis="y", length=0, pad=3)
    axis.tick_params(axis="x", length=2.4, width=0.62)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_visible(False)

    axis.text(
        0.01, 1.015, "local humanward",
        transform=axis.transAxes, ha="left", va="bottom",
        fontsize=6.4, color=PINK_DARK,
    )
    axis.text(
        0.99, 1.015, "local machineward",
        transform=axis.transAxes, ha="right", va="bottom",
        fontsize=6.4, color=BLUE_DARK,
    )
    handles = [
        Line2D(
            [0], [0], marker=MODEL_MARKER[model], linestyle="none",
            markerfacecolor="white", markeredgecolor=INK,
            markeredgewidth=0.7, markersize=3.8,
            label=MODEL_LABEL[model],
        )
        for model in MODELS
    ]
    axis.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.065),
        ncol=3,
        columnspacing=0.75,
        handletextpad=0.23,
        borderaxespad=0,
    )

    figure.subplots_adjust(left=0.255, right=0.985, top=0.82, bottom=0.20)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        FIGURE_DIR / "projected_lexical_field.pdf",
        bbox_inches="tight",
        pad_inches=0.02,
    )
    figure.savefig(
        PREVIEW_DIR / "projected_lexical_field.png",
        dpi=260,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(figure)


if __name__ == "__main__":
    main()
