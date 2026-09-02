#!/usr/bin/env python
"""Mine paired hidden states for cross-layer and cross-model geometry.

The analysis treats ``AI - Human`` as the primary observation.  It writes
machine-readable tables plus paper-ready figures under one canonical output
directory; repeated runs replace the same artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, PathPatch, Rectangle, Wedge
from matplotlib.path import Path as MplPath
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linear_sum_assignment
from scipy.stats import gaussian_kde, spearmanr
import seaborn as sns
import sklearn
from sklearn.decomposition import PCA
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.hidden_geometry import (  # noqa: E402
    bootstrap_direction_stability,
    centered_gram,
    coherent_displacement_fraction,
    cross_validated_pair_alignment,
    group_direction_geometry,
    leading_residual_scores,
    linear_cka_from_grams,
    normalize_rows,
    row_cosines,
    spectral_summary,
    unit_vector,
)


MODEL_ORDER = [
    "Llama-3.1-8B-Instruct",
    "Qwen3-4B",
    "gemma-4-E2B-it",
]
REPRESENTATION_ORDER = ["last_token", "mean_pool"]
REPRESENTATION_LABEL = {
    "last_token": "Last continuation token",
    "mean_pool": "Continuation mean",
}
MODEL_LABEL = {
    "Llama-3.1-8B-Instruct": "Llama 3.1 8B",
    "Qwen3-4B": "Qwen3 4B",
    "Qwen3.5-4B": "Qwen3.5 4B",
    "gemma-4-E2B-it": "Gemma 4 E2B",
}
MODEL_SHORT_LABEL = {
    "Llama-3.1-8B-Instruct": "Llama",
    "Qwen3-4B": "Qwen3",
    "Qwen3.5-4B": "Qwen3.5",
    "gemma-4-E2B-it": "Gemma",
}
PRIMARY_METHOD = "nullspace"
CONTROL_METHODS = ["raw"]
METHOD_ORDER = [PRIMARY_METHOD, *CONTROL_METHODS]

INK = "#27313A"
MUTED = "#6F7B86"
GRID = "#E4E9ED"
BLUE = "#75D7F0"
BLUE_DARK = "#28A9D0"
BLUE_LIGHT = "#BDEFFC"
BLUE_OPEN = "#E9F9FE"
PINK = "#FFA6C8"
PINK_DARK = "#EB6FA1"
PINK_LIGHT = "#FFD1E2"
PINK_OPEN = "#FFF0F6"
MATCH_OUTLINE = "#AFA4C3"
LIGHT_BG = "#FBFCFD"
REPRESENTATION_COLOR = {"last_token": BLUE_DARK, "mean_pool": PINK_DARK}
METHOD_STYLE = {"raw": "--", "nullspace": "-"}
SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "beyondstyle_blue",
    ["#F8FBFD", BLUE_OPEN, BLUE, BLUE_DARK],
)
PINK_CMAP = LinearSegmentedColormap.from_list(
    "beyondstyle_pink",
    ["#FFFFFF", PINK_OPEN, PINK, PINK_DARK],
)
BLUE_CMAP = LinearSegmentedColormap.from_list(
    "beyondstyle_bright_blue",
    ["#FFFFFF", BLUE_OPEN, BLUE, BLUE_DARK],
)
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "beyondstyle_diverging",
    [BLUE_DARK, "#F8F9FA", PINK_DARK],
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--states-root",
        type=Path,
        default=Path("outputs/en/03_steering_vectors"),
    )
    parser.add_argument(
        "--eval-root",
        type=Path,
        default=Path("outputs/en/02_separability"),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/paired/en/proposition_aligned/pairs.jsonl"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/en/02_separability/hidden_geometry"),
    )
    parser.add_argument("--dataset-tag", default="proposition_aligned")
    parser.add_argument(
        "--models",
        nargs=3,
        default=MODEL_ORDER,
        metavar=("MODEL_A", "MODEL_B", "MODEL_C"),
        help="Exactly three model directory identifiers in display order.",
    )
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--permutations", type=int, default=200)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )
    parser.add_argument(
        "--refresh-alignment-atlas",
        action="store_true",
        help=(
            "Redraw only cross_model_alignment_atlas from saved CSVs; "
            "do not recompute hidden-state statistics."
        ),
    )
    return parser.parse_args()


def configure_style() -> None:
    sns.set_theme(style="whitegrid")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "DejaVu Sans",
                "Arial",
                "Liberation Sans",
            ],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#B8C0C8",
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.labelsize": 9.8,
            "axes.titlesize": 9.8,
            "axes.titleweight": "semibold",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8.7,
            "ytick.labelsize": 8.7,
            "grid.color": GRID,
            "grid.alpha": 0.55,
            "grid.linewidth": 0.6,
            "legend.fontsize": 9.2,
            "legend.frameon": False,
            "font.size": 9.1,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
        }
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_dataset(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    sample_ids = [row["sample_id"] for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("Dataset sample_id values must be unique")
    return rows


def state_inventory(
    states_root: Path,
    dataset_tag: str,
) -> dict[tuple[str, str], list[Path]]:
    inventory: dict[tuple[str, str], list[Path]] = defaultdict(list)
    for path in states_root.glob(
        f"*/{dataset_tag}/*/states/paired_states_layer*.pt"
    ):
        relative = path.relative_to(states_root)
        model, _, representation = relative.parts[:3]
        inventory[(model, representation)].append(path)
    for key in inventory:
        inventory[key].sort(key=layer_from_path)
    return dict(inventory)


def layer_from_path(path: Path) -> int:
    match = re.search(r"layer(\d+)", path.stem)
    if not match:
        raise ValueError(f"Could not parse layer from {path}")
    return int(match.group(1))


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_aligned_state(
    path: Path,
    canonical_ids: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    metadata = payload.get("metadata", {})
    state_ids = metadata.get("sample_ids")
    if not state_ids:
        raise ValueError(f"Missing sample_ids in {path}")
    if len(state_ids) != len(set(state_ids)):
        raise ValueError(f"Duplicate sample_ids in {path}")
    if set(state_ids) != set(canonical_ids):
        missing = sorted(set(canonical_ids) - set(state_ids))[:5]
        extra = sorted(set(state_ids) - set(canonical_ids))[:5]
        raise ValueError(
            f"State/data sample mismatch in {path}: "
            f"missing={missing}, extra={extra}"
        )
    positions = {sample_id: index for index, sample_id in enumerate(state_ids)}
    order = torch.as_tensor(
        [positions[sample_id] for sample_id in canonical_ids],
        dtype=torch.long,
    )
    human = payload["human"].index_select(0, order).float().to(device)
    ai = payload["ai"].index_select(0, order).float().to(device)
    return human, ai, metadata


def load_saved_direction(
    state_path: Path,
    method: str,
    device: torch.device,
) -> torch.Tensor:
    layer = layer_from_path(state_path)
    model_dir = state_path.parent.parent / "models"
    suffix = "" if method == "raw" else "_nullspace"
    direction_path = model_dir / f"mean_diff{suffix}_layer{layer:02d}.pt"
    payload = torch.load(direction_path, map_location="cpu", weights_only=False)
    direction = payload["state_dict"]["v_src"].float().to(device)
    return unit_vector(direction)


def read_formal_evaluations(
    eval_root: Path,
    inventory: dict[tuple[str, str], list[Path]],
    dataset_tag: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    random_rows: list[dict[str, Any]] = []
    generator_rows: list[dict[str, Any]] = []
    for model, representation in inventory:
        base = eval_root / model / dataset_tag / representation
        random_payload = json.loads(
            (base / "random_holdout.json").read_text(encoding="utf-8")
        )
        generator_payload = json.loads(
            (base / "generator_holdout.json").read_text(encoding="utf-8")
        )
        for layer_text, methods in random_payload["results"].items():
            for method, values in methods.items():
                random_rows.append(
                    {
                        "model": model,
                        "representation": representation,
                        "layer": int(layer_text),
                        "method": method,
                        **values,
                    }
                )
        for layer_text, generators in generator_payload["results"].items():
            for generator, methods in generators.items():
                for method, values in methods.items():
                    generator_rows.append(
                        {
                            "model": model,
                            "representation": representation,
                            "layer": int(layer_text),
                            "generator": generator,
                            "method": method,
                            **values,
                        }
                    )
    return pd.DataFrame(random_rows), pd.DataFrame(generator_rows)


def safe_spearman(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    if np.std(first) <= 1e-12 or np.std(second) <= 1e-12:
        return 0.0, 1.0
    statistic = spearmanr(first, second)
    return float(statistic.statistic), float(statistic.pvalue)


def numeric_summary(values: Any) -> dict[str, float | int] | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    if not array.size:
        return None
    return {
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def nuisance_features(rows: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    prefix = np.asarray(
        [row["lengths"]["prefix_units"] for row in rows],
        dtype=float,
    )
    human = np.asarray(
        [row["lengths"]["human_continuation_units"] for row in rows],
        dtype=float,
    )
    ai = np.asarray(
        [row["lengths"]["ai_continuation_units"] for row in rows],
        dtype=float,
    )
    proposition_count = np.asarray(
        [
            len(
                row.get("content_control", {})
                .get("plan", {})
                .get("propositions", [])
            )
            for row in rows
        ],
        dtype=float,
    )
    mean_continuation = (human + ai) / 2.0
    return {
        "prefix_units": prefix,
        "mean_continuation_units": mean_continuation,
        "prefix_to_continuation_ratio": prefix
        / np.maximum(mean_continuation, 1.0),
        "ai_human_length_gap": (ai - human) / np.maximum(human, 1.0),
        "proposition_count": proposition_count,
    }


def benjamini_hochberg(p_values: pd.Series) -> np.ndarray:
    values = p_values.to_numpy(dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0, 1)
    return result


def save_figure(fig: plt.Figure, output_stem: Path) -> None:
    fig.savefig(output_stem.with_suffix(".png"), dpi=240)
    fig.savefig(output_stem.with_suffix(".pdf"))
    plt.close(fig)


def layer_values(
    frame: pd.DataFrame,
    model: str,
    representation: str,
) -> list[int]:
    subset = frame[
        (frame["model"] == model)
        & (frame["representation"] == representation)
    ]
    return sorted(subset["layer"].unique().tolist())


def plot_evidence_atlas(
    layer_metrics: pd.DataFrame,
    random_eval: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(
        len(MODEL_ORDER),
        4,
        figsize=(16, 10.5),
        sharex="col",
    )
    metric_specs = [
        ("oof_positive_rate", "Held-out pair agreement", (0.45, 1.01)),
        (
            "angular_coherent_displacement_fraction",
            "Shared angular displacement",
            (0, None),
        ),
        (
            "angular_effective_rank",
            "Angular residual effective rank",
            (0, None),
        ),
    ]

    for row_index, model in enumerate(MODEL_ORDER):
        auc_axis = axes[row_index, 0]
        for representation in REPRESENTATION_ORDER:
            for method in METHOD_ORDER:
                subset = random_eval[
                    (random_eval["model"] == model)
                    & (random_eval["representation"] == representation)
                    & (random_eval["method"] == method)
                ].sort_values("relative_depth")
                label = (
                    f"{REPRESENTATION_LABEL[representation]} · {method}"
                    if row_index == 0
                    else None
                )
                auc_axis.plot(
                    subset["relative_depth"],
                    subset["heldout_auc_mean"],
                    color=REPRESENTATION_COLOR[representation],
                    linestyle=METHOD_STYLE[method],
                    marker="o" if method == "nullspace" else None,
                    markersize=3,
                    linewidth=1.8,
                    label=label,
                )
        auc_axis.axhline(0.5, color=MUTED, linewidth=0.8, linestyle=":")
        auc_axis.set_ylim(0.45, 1.01)
        auc_axis.set_ylabel(MODEL_LABEL[model], fontweight="semibold")
        if row_index == 0:
            auc_axis.set_title("Random-held-out source AUC")

        for column_offset, (metric, title, limits) in enumerate(
            metric_specs,
            start=1,
        ):
            axis = axes[row_index, column_offset]
            for representation in REPRESENTATION_ORDER:
                subset = layer_metrics[
                    (layer_metrics["model"] == model)
                    & (layer_metrics["representation"] == representation)
                ].sort_values("relative_depth")
                axis.plot(
                    subset["relative_depth"],
                    subset[metric],
                    color=REPRESENTATION_COLOR[representation],
                    marker="o",
                    markersize=3,
                    linewidth=1.8,
                    label=REPRESENTATION_LABEL[representation]
                    if row_index == 0
                    else None,
                )
            lower, upper = limits
            axis.set_ylim(
                bottom=lower,
                top=upper,
            )
            if metric == "oof_positive_rate":
                axis.axhline(
                    0.5,
                    color=MUTED,
                    linewidth=0.8,
                    linestyle=":",
                )
            if row_index == 0:
                axis.set_title(title)

    for axis in axes[-1]:
        axis.set_xlabel("Relative decoder depth")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.005),
    )
    fig.suptitle(
        "Paired hidden-state evidence atlas",
        y=1.045,
        fontsize=15,
        fontweight="semibold",
        color=INK,
    )
    fig.text(
        0.5,
        1.015,
        "English proposition-aligned continuations; 500 matched pairs per model",
        ha="center",
        color=MUTED,
    )
    fig.tight_layout()
    save_figure(fig, output_dir / "hidden_state_evidence_atlas")


def heatmap_panel(
    axis: plt.Axes,
    matrix: np.ndarray,
    x_labels: list[int],
    y_labels: list[int],
    title: str,
    cmap: Any = SEQUENTIAL_CMAP,
    vmin: float = 0,
    vmax: float = 1,
    annotate: bool = False,
) -> Any:
    image = axis.imshow(
        matrix,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    axis.set_xticks(np.arange(len(x_labels)), x_labels, rotation=45)
    axis.set_yticks(np.arange(len(y_labels)), y_labels)
    axis.set_title(title)
    axis.grid(False)
    if annotate:
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                axis.text(
                    column,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=6.5,
                    color="white" if value > (vmin + vmax) / 2 else INK,
                )
    return image


def plot_within_model_cka(
    cka_frame: pd.DataFrame,
    output_dir: Path,
    geometry: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2))
    image = None
    for row_index, representation in enumerate(REPRESENTATION_ORDER):
        for column_index, model in enumerate(MODEL_ORDER):
            subset = cka_frame[
                (cka_frame["scope"] == "within_model")
                & (cka_frame["geometry"] == geometry)
                & (cka_frame["model_a"] == model)
                & (cka_frame["representation_a"] == representation)
            ]
            layers = sorted(subset["layer_a"].unique())
            matrix = (
                subset.pivot(index="layer_a", columns="layer_b", values="cka")
                .reindex(index=layers, columns=layers)
                .to_numpy()
            )
            image = heatmap_panel(
                axes[row_index, column_index],
                matrix,
                layers,
                layers,
                f"{MODEL_LABEL[model]} · {REPRESENTATION_LABEL[representation]}",
                annotate=True,
            )
            axes[row_index, column_index].set_xlabel("Layer")
            axes[row_index, column_index].set_ylabel("Layer")
    if image is not None:
        fig.colorbar(
            image,
            ax=axes,
            fraction=0.018,
            pad=0.02,
            label="Linear CKA",
        )
    fig.suptitle(
        f"Cross-layer alignment of paired-displacement geometry ({geometry})",
        y=1.01,
        fontsize=15,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.965,
        (
            "Unit-normalized pair displacements; 1 indicates identical relational structure"
            if geometry == "angular"
            else "Raw pair displacements; 1 indicates identical relational structure"
        ),
        ha="center",
        color=MUTED,
    )
    fig.subplots_adjust(top=0.91, right=0.91, wspace=0.30, hspace=0.34)
    save_figure(
        fig,
        output_dir / f"within_model_displacement_cka_{geometry}",
    )


def plot_cross_model_cka(
    cka_frame: pd.DataFrame,
    output_dir: Path,
    geometry: str,
) -> None:
    pairs = [
        (MODEL_ORDER[0], MODEL_ORDER[1]),
        (MODEL_ORDER[0], MODEL_ORDER[2]),
        (MODEL_ORDER[1], MODEL_ORDER[2]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2))
    image = None
    for row_index, representation in enumerate(REPRESENTATION_ORDER):
        for column_index, (first_model, second_model) in enumerate(pairs):
            subset = cka_frame[
                (cka_frame["scope"] == "cross_model")
                & (cka_frame["geometry"] == geometry)
                & (cka_frame["model_a"] == first_model)
                & (cka_frame["model_b"] == second_model)
                & (cka_frame["representation_a"] == representation)
            ]
            first_layers = sorted(subset["layer_a"].unique())
            second_layers = sorted(subset["layer_b"].unique())
            matrix = (
                subset.pivot(index="layer_a", columns="layer_b", values="cka")
                .reindex(index=first_layers, columns=second_layers)
                .to_numpy()
            )
            image = heatmap_panel(
                axes[row_index, column_index],
                matrix,
                second_layers,
                first_layers,
                f"{MODEL_LABEL[first_model]} × {MODEL_LABEL[second_model]}\n"
                f"{REPRESENTATION_LABEL[representation]}",
            )
            axes[row_index, column_index].set_xlabel(
                f"{MODEL_LABEL[second_model]} layer"
            )
            axes[row_index, column_index].set_ylabel(
                f"{MODEL_LABEL[first_model]} layer"
            )
    if image is not None:
        fig.colorbar(
            image,
            ax=axes,
            fraction=0.018,
            pad=0.02,
            label="Linear CKA",
        )
    fig.suptitle(
        f"Cross-model alignment of paired-displacement geometry ({geometry})",
        y=1.01,
        fontsize=15,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.965,
        (
            "Unit-normalized displacements for the same 500 pairs; valid across unequal hidden sizes"
            if geometry == "angular"
            else "Raw displacements for the same 500 pairs; valid across unequal hidden sizes"
        ),
        ha="center",
        color=MUTED,
    )
    fig.subplots_adjust(top=0.89, right=0.91, wspace=0.35, hspace=0.40)
    save_figure(
        fig,
        output_dir / f"cross_model_displacement_cka_{geometry}",
    )


def plot_generator_holdout(
    generator_eval: pd.DataFrame,
    output_dir: Path,
) -> None:
    generators = sorted(generator_eval["generator"].unique())
    generator_labels = [_short_generator_label(name) for name in generators]
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(14.8, 7.5),
        sharey=True,
    )
    image = None
    for row_index, representation in enumerate(REPRESENTATION_ORDER):
        for column_index, model in enumerate(MODEL_ORDER):
            subset = generator_eval[
                (generator_eval["model"] == model)
                & (generator_eval["representation"] == representation)
                & (generator_eval["method"] == "nullspace")
            ]
            layers = sorted(subset["layer"].unique())
            matrix = (
                subset.pivot(index="generator", columns="layer", values="auc")
                .reindex(index=generators, columns=layers)
                .to_numpy()
            )
            image = heatmap_panel(
                axes[row_index, column_index],
                matrix,
                layers,
                list(range(len(generators))),
                f"{MODEL_LABEL[model]} · {REPRESENTATION_LABEL[representation]}",
                vmin=0.5,
                vmax=1.0,
                annotate=True,
            )
            axes[row_index, column_index].set_yticks(
                np.arange(len(generators)),
                generator_labels,
            )
            axes[row_index, column_index].set_xlabel("Layer")
            if column_index == 0:
                axes[row_index, column_index].set_ylabel(
                    "Held-out generator"
                )
            else:
                axes[row_index, column_index].tick_params(
                    axis="y",
                    labelleft=False,
                )
    if image is not None:
        color_axis = fig.add_axes([0.925, 0.20, 0.015, 0.62])
        fig.colorbar(
            image,
            cax=color_axis,
            label="Leave-one-generator-out AUC",
        )
    fig.suptitle(
        "Source separability for unseen generators",
        y=0.985,
        fontsize=15,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.94,
        "Nullspace direction fitted without the evaluated generator; n=125 held-out pairs per cell",
        ha="center",
        color=MUTED,
    )
    fig.subplots_adjust(
        top=0.84,
        right=0.89,
        bottom=0.09,
        left=0.10,
        wspace=0.12,
        hspace=0.38,
    )
    save_figure(fig, output_dir / "generator_holdout_atlas")


def plot_representation_agreement(
    representation_frame: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), sharey=True)
    measures = [
        ("raw_direction_cosine", "Raw direction", BLUE_DARK, "-"),
        (
            "nullspace_direction_cosine",
            "Nullspace direction",
            BLUE_DARK,
            "--",
        ),
        ("raw_displacement_cka", "Raw displacement CKA", PINK_DARK, ":"),
        (
            "angular_displacement_cka",
            "Angular displacement CKA",
            PINK_DARK,
            "-",
        ),
        (
            "oof_score_spearman",
            "Pair-score Spearman",
            PINK_DARK,
            "--",
        ),
    ]
    for axis, model in zip(axes, MODEL_ORDER):
        subset = representation_frame[
            representation_frame["model"] == model
        ].sort_values("relative_depth")
        for column, label, color, linestyle in measures:
            axis.plot(
                subset["relative_depth"],
                subset[column],
                marker="o",
                markersize=3.5,
                linewidth=1.8,
                linestyle=linestyle,
                color=color,
                label=label,
            )
        axis.set_title(MODEL_LABEL[model])
        axis.set_xlabel("Relative decoder depth")
        axis.set_ylim(-0.05, 1.02)
        axis.axhline(0, color=MUTED, linewidth=0.7)
    axes[0].set_ylabel("Agreement")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 1.04),
    )
    fig.suptitle(
        "Agreement between last-token and continuation-mean representations",
        y=1.14,
        fontsize=15,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        1.075,
        "Direction, relational geometry, and held-out pair ranking are distinct comparisons",
        ha="center",
        color=MUTED,
    )
    fig.tight_layout()
    save_figure(fig, output_dir / "representation_agreement")


def plot_spectral_fingerprints(
    spectra: dict[tuple[str, str, int, str], np.ndarray],
    layer_lookup: dict[tuple[str, str], list[int]],
    output_dir: Path,
    geometry: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.8), sharey=True)
    colors = [BLUE_DARK, MUTED, PINK_DARK]
    labels = ["Early", "Middle", "Late"]
    for row_index, representation in enumerate(REPRESENTATION_ORDER):
        for column_index, model in enumerate(MODEL_ORDER):
            axis = axes[row_index, column_index]
            layers = layer_lookup[(model, representation)]
            selected_indices = [0, len(layers) // 2, len(layers) - 1]
            for label, color, index in zip(labels, colors, selected_indices):
                layer = layers[index]
                eigenvalues = spectra[
                    (model, representation, layer, geometry)
                ]
                cumulative = np.cumsum(eigenvalues) / max(
                    float(eigenvalues.sum()),
                    1e-12,
                )
                maximum = min(80, len(cumulative))
                axis.plot(
                    np.arange(1, maximum + 1),
                    cumulative[:maximum],
                    color=color,
                    linewidth=1.8,
                    label=f"{label} · L{layer}",
                )
            axis.set_title(
                f"{MODEL_LABEL[model]} · {REPRESENTATION_LABEL[representation]}"
            )
            axis.set_xlabel("Residual component rank")
            axis.set_xlim(1, 80)
            axis.set_ylim(0, 1.01)
            if column_index == 0:
                axis.set_ylabel("Cumulative variance")
            axis.legend(fontsize=7.5)
    fig.suptitle(
        f"Spectral fingerprints of pair-specific displacement ({geometry})",
        y=1.01,
        fontsize=15,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.965,
        (
            "Unit-normalized displacements; spectrum follows removal of the shared mean"
            if geometry == "angular"
            else "Raw displacements; spectrum follows removal of the shared mean"
        ),
        ha="center",
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(
        fig,
        output_dir / f"displacement_spectral_fingerprints_{geometry}",
    )


def plot_nuisance_correlations(
    nuisance_frame: pd.DataFrame,
    output_dir: Path,
    target: str,
    title: str,
    subtitle: str,
    output_name: str,
    signed: bool,
) -> None:
    feature_order = [
        "prefix_units",
        "mean_continuation_units",
        "prefix_to_continuation_ratio",
        "ai_human_length_gap",
        "proposition_count",
    ]
    feature_labels = [
        "Prefix length",
        "Continuation length",
        "Prefix / continuation",
        "AI−human length gap",
        "Proposition count",
    ]
    nuisance_frame = nuisance_frame[nuisance_frame["target"] == target]
    maximum = max(0.2, float(nuisance_frame["rho"].abs().max()))
    maximum = min(1.0, np.ceil(maximum * 10) / 10)
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(14.8, 7.5),
        sharey=True,
    )
    image = None
    for row_index, representation in enumerate(REPRESENTATION_ORDER):
        for column_index, model in enumerate(MODEL_ORDER):
            subset = nuisance_frame[
                (nuisance_frame["model"] == model)
                & (nuisance_frame["representation"] == representation)
            ]
            layers = sorted(subset["layer"].unique())
            matrix = (
                subset.pivot(index="feature", columns="layer", values="rho")
                .reindex(index=feature_order, columns=layers)
                .to_numpy()
            )
            image = heatmap_panel(
                axes[row_index, column_index],
                matrix,
                layers,
                list(range(len(feature_order))),
                f"{MODEL_LABEL[model]} · {REPRESENTATION_LABEL[representation]}",
                cmap=DIVERGING_CMAP if signed else SEQUENTIAL_CMAP,
                vmin=-maximum if signed else 0,
                vmax=maximum,
                annotate=True,
            )
            axes[row_index, column_index].set_yticks(
                np.arange(len(feature_order)),
                feature_labels,
            )
            axes[row_index, column_index].set_xlabel("Layer")
            if column_index == 0:
                axes[row_index, column_index].set_ylabel(
                    "Potential confound"
                )
            else:
                axes[row_index, column_index].tick_params(
                    axis="y",
                    labelleft=False,
                )
    if image is not None:
        color_axis = fig.add_axes([0.925, 0.20, 0.015, 0.62])
        fig.colorbar(
            image,
            cax=color_axis,
            label=(
                "Spearman ρ"
                if signed
                else "Absolute Spearman ρ (PC sign is arbitrary)"
            ),
        )
    fig.suptitle(
        title,
        y=0.985,
        fontsize=15,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.94,
        subtitle,
        ha="center",
        color=MUTED,
    )
    fig.subplots_adjust(
        top=0.84,
        right=0.89,
        bottom=0.09,
        left=0.13,
        wspace=0.12,
        hspace=0.38,
    )
    save_figure(fig, output_dir / output_name)


def plot_magnitude_residual_diagnostics(
    layer_metrics: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex="col")
    for column_index, model in enumerate(MODEL_ORDER):
        spectrum_axis = axes[0, column_index]
        norm_axis = axes[1, column_index]
        for representation in REPRESENTATION_ORDER:
            subset = layer_metrics[
                (layer_metrics["model"] == model)
                & (layer_metrics["representation"] == representation)
            ].sort_values("relative_depth")
            color = REPRESENTATION_COLOR[representation]
            spectrum_axis.plot(
                subset["relative_depth"],
                subset["variance_top_1"],
                color=color,
                linestyle=":",
                linewidth=1.8,
                label=f"{REPRESENTATION_LABEL[representation]} · raw",
            )
            spectrum_axis.plot(
                subset["relative_depth"],
                subset["angular_variance_top_1"],
                color=color,
                linestyle="-",
                marker="o",
                markersize=3,
                linewidth=1.8,
                label=f"{REPRESENTATION_LABEL[representation]} · angular",
            )
            norm_axis.plot(
                subset["relative_depth"],
                subset["p99_to_median_displacement_norm"],
                color=color,
                marker="o",
                markersize=3,
                linewidth=1.8,
                label=REPRESENTATION_LABEL[representation],
            )
        spectrum_axis.set_title(MODEL_LABEL[model])
        spectrum_axis.set_ylim(0, 1.01)
        spectrum_axis.set_ylabel(
            "Residual PC1 variance"
            if column_index == 0
            else None
        )
        norm_axis.set_ylabel(
            "P99 / median pair norm"
            if column_index == 0
            else None
        )
        norm_axis.set_xlabel("Relative decoder depth")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        "Displacement magnitude and residual concentration",
        y=1.09,
        fontsize=15,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        1.045,
        "Angular spectra remove pair-norm differences; norm ratios expose outlier sensitivity",
        ha="center",
        color=MUTED,
    )
    fig.tight_layout()
    save_figure(fig, output_dir / "magnitude_residual_diagnostics")


def plot_pair_alignment_ridgelines(
    pair_frame: pd.DataFrame,
    output_dir: Path,
) -> None:
    quantile_low = float(pair_frame["oof_cosine"].quantile(0.002))
    quantile_high = float(pair_frame["oof_cosine"].quantile(0.998))
    x_low = min(-0.05, quantile_low)
    x_high = max(0.35, quantile_high)
    x_grid = np.linspace(x_low, x_high, 320)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2), sharex=True)

    for row_index, representation in enumerate(REPRESENTATION_ORDER):
        for column_index, model in enumerate(MODEL_ORDER):
            axis = axes[row_index, column_index]
            subset = pair_frame[
                (pair_frame["model"] == model)
                & (pair_frame["representation"] == representation)
            ]
            layers = sorted(subset["layer"].unique())
            color = REPRESENTATION_COLOR[representation]
            for layer_index, layer in enumerate(layers):
                values = subset[subset["layer"] == layer][
                    "oof_cosine"
                ].to_numpy()
                if np.std(values) <= 1e-8:
                    density = np.zeros_like(x_grid)
                else:
                    density = gaussian_kde(values)(x_grid)
                if density.max() > 0:
                    density = density / density.max() * 0.78
                baseline = float(layer_index)
                axis.fill_between(
                    x_grid,
                    baseline,
                    baseline + density,
                    color=color,
                    alpha=0.45,
                    linewidth=0,
                )
                axis.plot(
                    x_grid,
                    baseline + density,
                    color=color,
                    linewidth=1.1,
                )
                axis.hlines(
                    baseline,
                    x_low,
                    x_high,
                    color=GRID,
                    linewidth=0.5,
                )
            axis.axvline(0, color=INK, linestyle=":", linewidth=0.9)
            axis.set_yticks(
                np.arange(len(layers)) + 0.1,
                [f"L{layer}" for layer in layers],
            )
            axis.set_title(
                f"{MODEL_LABEL[model]} · "
                f"{REPRESENTATION_LABEL[representation]}"
            )
            axis.set_xlabel("Held-out cosine with source direction")
            if column_index == 0:
                axis.set_ylabel("Decoder layer")
            axis.grid(False)
            axis.spines["left"].set_visible(False)
    fig.suptitle(
        "Distributional evolution of paired source alignment",
        y=1.01,
        fontsize=15,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.965,
        "Each ridge contains 500 pairs; its direction was fitted without the scored pair",
        ha="center",
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, output_dir / "pair_alignment_ridgelines")


def _alluvial_category(values: np.ndarray) -> np.ndarray:
    return np.digitize(values, bins=np.asarray([0.0, 0.10, 0.25]))


def _alluvial_ribbon(
    axis: plt.Axes,
    x_left: float,
    x_right: float,
    left_bottom: float,
    left_top: float,
    right_bottom: float,
    right_top: float,
    color: str,
    alpha: float,
) -> None:
    control_left = x_left + (x_right - x_left) * 0.42
    control_right = x_right - (x_right - x_left) * 0.42
    vertices = [
        (x_left, left_bottom),
        (control_left, left_bottom),
        (control_right, right_bottom),
        (x_right, right_bottom),
        (x_right, right_top),
        (control_right, right_top),
        (control_left, left_top),
        (x_left, left_top),
        (x_left, left_bottom),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    axis.add_patch(
        PathPatch(
            MplPath(vertices, codes),
            facecolor=color,
            edgecolor="none",
            alpha=alpha,
        )
    )


def plot_pair_alignment_alluvial(
    pair_frame: pd.DataFrame,
    output_dir: Path,
) -> None:
    category_labels = [
        "Contrary (<0)",
        "Weak (0–.10)",
        "Moderate (.10–.25)",
        "Strong (≥.25)",
    ]
    category_colors = [BLUE_DARK, BLUE, PINK, PINK_DARK]
    category_count = len(category_labels)
    gap = 0.018
    node_width = 0.085
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.4))

    for row_index, representation in enumerate(REPRESENTATION_ORDER):
        for column_index, model in enumerate(MODEL_ORDER):
            axis = axes[row_index, column_index]
            subset = pair_frame[
                (pair_frame["model"] == model)
                & (pair_frame["representation"] == representation)
            ]
            pivot = subset.pivot(
                index="sample_id",
                columns="layer",
                values="oof_cosine",
            )
            layers = sorted(pivot.columns.tolist())
            categories = _alluvial_category(pivot[layers].to_numpy())
            n_pairs = categories.shape[0]
            usable_height = 1.0 - gap * (category_count - 1)
            bases = np.zeros((len(layers), category_count), dtype=float)
            heights = np.zeros_like(bases)

            for layer_index in range(len(layers)):
                counts = np.bincount(
                    categories[:, layer_index],
                    minlength=category_count,
                )
                heights[layer_index] = counts / n_pairs * usable_height
                cursor = 0.0
                for category in range(category_count):
                    bases[layer_index, category] = cursor
                    cursor += heights[layer_index, category]
                    if category < category_count - 1:
                        cursor += gap

            for layer_index in range(len(layers) - 1):
                transitions = np.zeros(
                    (category_count, category_count),
                    dtype=int,
                )
                for source_category, target_category in zip(
                    categories[:, layer_index],
                    categories[:, layer_index + 1],
                ):
                    transitions[source_category, target_category] += 1
                source_offsets = bases[layer_index].copy()
                target_offsets = bases[layer_index + 1].copy()
                x_left = layer_index + node_width / 2
                x_right = layer_index + 1 - node_width / 2
                for source_category in range(category_count):
                    for target_category in range(category_count):
                        count = transitions[
                            source_category,
                            target_category,
                        ]
                        if not count:
                            continue
                        height = count / n_pairs * usable_height
                        left_bottom = source_offsets[source_category]
                        right_bottom = target_offsets[target_category]
                        _alluvial_ribbon(
                            axis,
                            x_left,
                            x_right,
                            left_bottom,
                            left_bottom + height,
                            right_bottom,
                            right_bottom + height,
                            category_colors[source_category],
                            alpha=0.24,
                        )
                        source_offsets[source_category] += height
                        target_offsets[target_category] += height

            for layer_index, layer in enumerate(layers):
                for category in range(category_count):
                    axis.add_patch(
                        Rectangle(
                            (
                                layer_index - node_width / 2,
                                bases[layer_index, category],
                            ),
                            node_width,
                            heights[layer_index, category],
                            facecolor=category_colors[category],
                            edgecolor="white",
                            linewidth=0.45,
                            zorder=5,
                        )
                    )
            axis.set_xlim(-0.18, len(layers) - 0.82)
            axis.set_ylim(0, 1)
            axis.set_xticks(np.arange(len(layers)), [f"L{layer}" for layer in layers])
            axis.set_yticks([])
            axis.set_title(
                f"{MODEL_LABEL[model]} · "
                f"{REPRESENTATION_LABEL[representation]}"
            )
            axis.set_xlabel("Decoder layer")
            axis.grid(False)
            for spine in axis.spines.values():
                spine.set_visible(False)

    legend_handles = [
        Patch(facecolor=color, edgecolor="none", label=label)
        for color, label in zip(category_colors, category_labels)
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.99),
        frameon=False,
    )
    fig.suptitle(
        "Layer-to-layer flow of paired source alignment",
        y=1.045,
        fontsize=15,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        1.008,
        "Ribbon width is the fraction of the same 500 pairs moving between fixed cosine bands",
        ha="center",
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_figure(fig, output_dir / "pair_alignment_alluvial")


def plot_layer_geometry_phase_portrait(
    layer_metrics: pd.DataFrame,
    random_eval: pd.DataFrame,
    output_dir: Path,
) -> None:
    nullspace = random_eval[random_eval["method"] == "nullspace"]
    merged = layer_metrics.merge(
        nullspace[
            [
                "model",
                "representation",
                "layer",
                "heldout_auc_mean",
            ]
        ],
        on=["model", "representation", "layer"],
        how="inner",
    )
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.6), sharex=True)
    for axis, model in zip(axes, MODEL_ORDER):
        for representation in REPRESENTATION_ORDER:
            subset = merged[
                (merged["model"] == model)
                & (merged["representation"] == representation)
            ].sort_values("layer")
            color = REPRESENTATION_COLOR[representation]
            x_values = subset["heldout_auc_mean"].to_numpy()
            y_values = subset["angular_effective_rank"].to_numpy()
            sizes = (
                45
                + 1050
                * subset["angular_coherent_displacement_fraction"].to_numpy()
            )
            for index in range(len(subset) - 1):
                axis.annotate(
                    "",
                    xy=(x_values[index + 1], y_values[index + 1]),
                    xytext=(x_values[index], y_values[index]),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": color,
                        "alpha": 0.45,
                        "linewidth": 1.1,
                        "mutation_scale": 9,
                    },
                    zorder=1,
                )
            axis.scatter(
                x_values,
                y_values,
                s=sizes,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                alpha=0.82,
                label=REPRESENTATION_LABEL[representation],
                zorder=3,
            )
            for _, point in subset.iterrows():
                axis.text(
                    point["heldout_auc_mean"],
                    point["angular_effective_rank"],
                    f"L{int(point['layer'])}",
                    fontsize=6.5,
                    ha="center",
                    va="center",
                    color=INK,
                    zorder=4,
                )
        axis.set_title(MODEL_LABEL[model])
        axis.set_xlabel("Random-held-out nullspace AUC")
        axis.grid(True, alpha=0.35)
    axes[0].set_ylabel("Angular residual effective rank")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.865),
    )
    fig.suptitle(
        "Layer geometry phase portrait",
        y=0.975,
        fontsize=15,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.915,
        "Arrows follow decoder depth; marker area represents shared angular displacement",
        ha="center",
        color=MUTED,
    )
    fig.subplots_adjust(
        left=0.065,
        right=0.985,
        bottom=0.12,
        top=0.76,
        wspace=0.17,
    )
    save_figure(fig, output_dir / "layer_geometry_phase_portrait")


def _classical_mds(distance_matrix: np.ndarray) -> np.ndarray:
    count = distance_matrix.shape[0]
    centering = np.eye(count) - np.ones((count, count)) / count
    gram = -0.5 * centering @ (distance_matrix**2) @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1][:2]
    positive = np.clip(eigenvalues[order], 0, None)
    coordinates = eigenvectors[:, order] * np.sqrt(positive)
    return coordinates


def _short_generator_label(name: str) -> str:
    replacements = {
        "MiniMax-M2.5": "MiniMax",
        "deepseek-v4-flash": "DeepSeek",
        "glm-5.2": "GLM",
        "qwen3.6-flash": "Qwen",
    }
    return replacements.get(name, name)


def plot_generator_direction_constellations(
    group_details: dict[str, Any],
    generator_eval: pd.DataFrame,
    output_dir: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.2))
    for row_index, representation in enumerate(REPRESENTATION_ORDER):
        for column_index, model in enumerate(MODEL_ORDER):
            axis = axes[row_index, column_index]
            subset = generator_eval[
                (generator_eval["model"] == model)
                & (generator_eval["representation"] == representation)
                & (generator_eval["method"] == "nullspace")
            ]
            layer_score = subset.groupby("layer")["auc"].min()
            selected_layer = int(layer_score.idxmax())
            detail = group_details[
                f"{model}/{representation}/layer{selected_layer:02d}"
            ]["angular_generator"]
            generators = sorted(detail["group_to_global_cosines"])
            names = ["Global"] + generators
            cosine = np.eye(len(names), dtype=float)
            for generator_index, generator in enumerate(generators, start=1):
                value = detail["group_to_global_cosines"][generator]
                cosine[0, generator_index] = value
                cosine[generator_index, 0] = value
            pairwise = detail["pairwise_group_cosines"]
            for first_index, first in enumerate(generators, start=1):
                for second_index, second in enumerate(
                    generators[first_index:],
                    start=first_index + 1,
                ):
                    key = f"{first}|||{second}"
                    reverse = f"{second}|||{first}"
                    value = pairwise.get(key, pairwise.get(reverse))
                    cosine[first_index, second_index] = value
                    cosine[second_index, first_index] = value
            distance = np.sqrt(np.clip(2 - 2 * cosine, 0, None))
            coordinates = _classical_mds(distance)
            coordinates -= coordinates[0]
            if len(coordinates) > 1:
                angle = np.arctan2(coordinates[1, 1], coordinates[1, 0])
                rotation = np.asarray(
                    [
                        [np.cos(-angle), -np.sin(-angle)],
                        [np.sin(-angle), np.cos(-angle)],
                    ]
                )
                coordinates = coordinates @ rotation.T
            for first_index in range(len(names)):
                for second_index in range(first_index + 1, len(names)):
                    value = cosine[first_index, second_index]
                    axis.plot(
                        coordinates[
                            [first_index, second_index],
                            0,
                        ],
                        coordinates[
                            [first_index, second_index],
                            1,
                        ],
                        color=PINK if first_index == 0 else BLUE,
                        linewidth=0.5 + 3.2 * max(value, 0),
                        alpha=0.12 + 0.48 * max(value, 0),
                        zorder=1,
                    )
            axis.scatter(
                coordinates[1:, 0],
                coordinates[1:, 1],
                s=170,
                color=BLUE,
                edgecolor=BLUE_DARK,
                linewidth=0.8,
                zorder=3,
            )
            axis.scatter(
                coordinates[0, 0],
                coordinates[0, 1],
                s=290,
                color=PINK,
                edgecolor=PINK_DARK,
                linewidth=1.0,
                zorder=4,
            )
            extent = max(
                float(np.ptp(coordinates[:, 0])),
                float(np.ptp(coordinates[:, 1])),
                1e-3,
            )
            high_label_boundary = (
                float(coordinates[:, 1].min())
                + 0.78 * float(np.ptp(coordinates[:, 1]))
            )
            for name_index, name in enumerate(names):
                label_position = coordinates[name_index].copy()
                horizontal_alignment = "center"
                vertical_alignment = "center"
                if name_index > 0:
                    direction = coordinates[name_index] - coordinates[0]
                    direction /= max(float(np.linalg.norm(direction)), 1e-8)
                    label_position += direction * extent * 0.055
                    horizontal_alignment = (
                        "left" if direction[0] >= 0 else "right"
                    )
                    vertical_alignment = (
                        "bottom" if direction[1] >= 0 else "top"
                    )
                    if coordinates[name_index, 1] >= high_label_boundary:
                        label_position = coordinates[name_index].copy()
                        label_position[1] -= extent * 0.055
                        horizontal_alignment = "center"
                        vertical_alignment = "top"
                axis.text(
                    label_position[0],
                    label_position[1],
                    (
                        "Global"
                        if name == "Global"
                        else _short_generator_label(name)
                    ),
                    fontsize=7.2,
                    ha=horizontal_alignment,
                    va=vertical_alignment,
                    color=INK,
                    zorder=5,
                )
            axis.set_title(
                f"{MODEL_LABEL[model]} · "
                f"{REPRESENTATION_LABEL[representation]}\n"
                f"robust layer L{selected_layer}"
            )
            axis.set_aspect("equal", adjustable="datalim")
            axis.set_xticks([])
            axis.set_yticks([])
            axis.grid(False)
            for spine in axis.spines.values():
                spine.set_visible(False)
    fig.suptitle(
        "Generator-direction constellations",
        y=1.01,
        fontsize=15,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.97,
        "Classical MDS of angular direction distance; thicker links indicate larger cosine",
        ha="center",
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, output_dir / "generator_direction_constellations")


def plot_cross_model_layer_alignment_ribbons(
    cka_frame: pd.DataFrame,
    output_dir: Path,
) -> None:
    model_pairs = [
        (MODEL_ORDER[0], MODEL_ORDER[1]),
        (MODEL_ORDER[0], MODEL_ORDER[2]),
        (MODEL_ORDER[1], MODEL_ORDER[2]),
    ]
    angular = cka_frame[
        (cka_frame["scope"] == "cross_model")
        & (cka_frame["geometry"] == "angular")
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.0))
    for row_index, representation in enumerate(REPRESENTATION_ORDER):
        for column_index, (first_model, second_model) in enumerate(model_pairs):
            axis = axes[row_index, column_index]
            subset = angular[
                (angular["representation_a"] == representation)
                & (angular["model_a"] == first_model)
                & (angular["model_b"] == second_model)
            ]
            first_layers = sorted(subset["layer_a"].unique())
            second_layers = sorted(subset["layer_b"].unique())
            matrix = (
                subset.pivot(index="layer_a", columns="layer_b", values="cka")
                .reindex(index=first_layers, columns=second_layers)
                .to_numpy()
            )
            first_indices, second_indices = linear_sum_assignment(-matrix)
            first_depth = np.asarray(first_layers) / max(first_layers)
            second_depth = np.asarray(second_layers) / max(second_layers)
            for first_index, second_index in zip(
                first_indices,
                second_indices,
            ):
                value = matrix[first_index, second_index]
                y_left = first_depth[first_index]
                y_right = second_depth[second_index]
                vertices = [
                    (0.07, y_left),
                    (0.38, y_left),
                    (0.62, y_right),
                    (0.93, y_right),
                ]
                path = MplPath(
                    vertices,
                    [
                        MplPath.MOVETO,
                        MplPath.CURVE4,
                        MplPath.CURVE4,
                        MplPath.CURVE4,
                    ],
                )
                axis.add_patch(
                    PathPatch(
                        path,
                        fill=False,
                        color=PINK,
                        linewidth=1.0 + 8.0 * value,
                        alpha=0.12 + 0.48 * value,
                        capstyle="round",
                    )
                )
                axis.add_patch(
                    PathPatch(
                        path,
                        fill=False,
                        color=BLUE_DARK,
                        linewidth=0.55,
                        alpha=0.65,
                    )
                )
            axis.scatter(
                np.full(len(first_layers), 0.05),
                first_depth,
                s=52,
                color=BLUE,
                edgecolor=BLUE_DARK,
                linewidth=0.6,
                zorder=4,
            )
            axis.scatter(
                np.full(len(second_layers), 0.95),
                second_depth,
                s=52,
                color=PINK,
                edgecolor=PINK_DARK,
                linewidth=0.6,
                zorder=4,
            )
            for layer, depth in zip(first_layers, first_depth):
                axis.text(
                    0.0,
                    depth,
                    f"L{layer}",
                    ha="right",
                    va="center",
                    fontsize=7,
                )
            for layer, depth in zip(second_layers, second_depth):
                axis.text(
                    1.0,
                    depth,
                    f"L{layer}",
                    ha="left",
                    va="center",
                    fontsize=7,
                )
            axis.text(
                0.05,
                1.075,
                MODEL_LABEL[first_model],
                ha="center",
                fontsize=8,
                fontweight="semibold",
            )
            axis.text(
                0.95,
                1.075,
                MODEL_LABEL[second_model],
                ha="center",
                fontsize=8,
                fontweight="semibold",
            )
            axis.set_title(REPRESENTATION_LABEL[representation])
            axis.set_xlim(-0.12, 1.12)
            axis.set_ylim(0.02, 1.12)
            axis.set_xticks([])
            axis.set_yticks([])
            axis.grid(False)
            for spine in axis.spines.values():
                spine.set_visible(False)
    fig.suptitle(
        "Optimal cross-model layer correspondences",
        y=1.015,
        fontsize=15,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.975,
        "Hungarian matching maximizes total angular CKA; ribbon width encodes matched similarity",
        ha="center",
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_figure(fig, output_dir / "cross_model_layer_alignment_ribbons")


def robust_layer_lookup(
    generator_eval: pd.DataFrame,
) -> dict[tuple[str, str], int]:
    """Select the layer maximizing worst-generator nullspace AUC."""
    selected: dict[tuple[str, str], int] = {}
    nullspace = generator_eval[generator_eval["method"] == "nullspace"]
    for model in MODEL_ORDER:
        for representation in REPRESENTATION_ORDER:
            subset = nullspace[
                (nullspace["model"] == model)
                & (nullspace["representation"] == representation)
            ]
            selected[(model, representation)] = int(
                subset.groupby("layer")["auc"].min().idxmax()
            )
    return selected


def _standardize_plane(values: np.ndarray) -> np.ndarray:
    scale = np.std(values, axis=0, ddof=1)
    scale[scale < 1e-8] = 1.0
    return values / scale


def _density_contours(
    axis: plt.Axes,
    points: np.ndarray,
    color: str,
) -> None:
    if np.linalg.matrix_rank(np.cov(points.T)) < 2:
        return
    x_min, x_max = np.quantile(points[:, 0], [0.01, 0.99])
    y_min, y_max = np.quantile(points[:, 1], [0.01, 0.99])
    x_grid, y_grid = np.meshgrid(
        np.linspace(x_min, x_max, 70),
        np.linspace(y_min, y_max, 70),
    )
    density = gaussian_kde(points.T)(
        np.vstack([x_grid.ravel(), y_grid.ravel()])
    ).reshape(x_grid.shape)
    positive = density[density > 0]
    if positive.size == 0:
        return
    levels = np.quantile(positive, [0.55, 0.72, 0.86, 0.94])
    levels = np.unique(levels)
    if len(levels) >= 2:
        axis.contour(
            x_grid,
            y_grid,
            density,
            levels=levels,
            colors=[color],
            linewidths=[0.65, 0.85, 1.05, 1.3][-len(levels) :],
            alpha=0.68,
            zorder=2,
        )


def plot_paired_endpoint_flow_fields(
    endpoint_states: dict[
        tuple[str, str],
        tuple[np.ndarray, np.ndarray, np.ndarray, int],
    ],
    layer_metrics: pd.DataFrame,
    random_eval: pd.DataFrame,
    generator_eval: pd.DataFrame,
    output_dir: Path,
    seed: int,
) -> None:
    """Map paired Human→AI state movement in a source/residual plane."""
    fig = plt.figure(figsize=(14.8, 8.15))
    outer = fig.add_gridspec(
        2,
        3,
        left=0.055,
        right=0.985,
        bottom=0.075,
        top=0.905,
        hspace=0.245,
        wspace=0.15,
    )
    rng = np.random.default_rng(seed)
    for row_index, representation in enumerate(REPRESENTATION_ORDER):
        for column_index, model in enumerate(MODEL_ORDER):
            nested = outer[row_index, column_index].subgridspec(
                2,
                1,
                height_ratios=[0.16, 0.84],
                hspace=0.02,
            )
            marginal_axis = fig.add_subplot(nested[0])
            axis = fig.add_subplot(nested[1], sharex=marginal_axis)
            human, ai, source_axis, layer = endpoint_states[
                (model, representation)
            ]
            pooled = np.vstack([human, ai]).astype(np.float64, copy=False)
            center = pooled.mean(axis=0, keepdims=True)
            centered = pooled - center
            source_axis = source_axis.astype(np.float64, copy=False)
            source_axis /= max(np.linalg.norm(source_axis), 1e-12)
            residual = centered - np.outer(
                centered @ source_axis,
                source_axis,
            )
            residual_axis = PCA(
                n_components=1,
                svd_solver="randomized",
                random_state=seed,
            ).fit(residual).components_[0]
            plane = np.column_stack(
                [centered @ source_axis, centered @ residual_axis]
            )
            plane = _standardize_plane(plane)
            human_plane = plane[: len(human)]
            ai_plane = plane[len(human) :]
            source_low, source_high = np.quantile(
                plane[:, 0],
                [0.005, 0.995],
            )
            source_padding = max(
                0.08 * (source_high - source_low),
                0.10,
            )
            source_limits = (
                source_low - source_padding,
                source_high + source_padding,
            )
            source_grid = np.linspace(*source_limits, 220)
            human_density = _kde_1d(
                human_plane[:, 0],
                source_grid,
            )
            ai_density = _kde_1d(
                ai_plane[:, 0],
                source_grid,
            )
            marginal_axis.fill_between(
                source_grid,
                0,
                human_density,
                color=BLUE,
                alpha=0.48,
                linewidth=0,
            )
            marginal_axis.fill_between(
                source_grid,
                0,
                ai_density,
                color=PINK,
                alpha=0.48,
                linewidth=0,
            )
            marginal_axis.plot(
                source_grid,
                human_density,
                color=BLUE_DARK,
                linewidth=1.05,
            )
            marginal_axis.plot(
                source_grid,
                ai_density,
                color=PINK_DARK,
                linewidth=1.05,
            )
            marginal_axis.axvline(
                np.median(human_plane[:, 0]),
                color=BLUE_DARK,
                linewidth=0.9,
                alpha=0.8,
            )
            marginal_axis.axvline(
                np.median(ai_plane[:, 0]),
                color=PINK_DARK,
                linewidth=0.9,
                alpha=0.8,
            )
            marginal_axis.set_xlim(source_limits)
            marginal_axis.set_ylim(0, 1.08)
            marginal_axis.axis("off")

            _density_contours(axis, human_plane, BLUE_DARK)
            _density_contours(axis, ai_plane, PINK_DARK)
            selected = rng.choice(
                len(human_plane),
                size=min(75, len(human_plane)),
                replace=False,
            )
            delta = ai_plane[selected] - human_plane[selected]
            axis.quiver(
                human_plane[selected, 0],
                human_plane[selected, 1],
                delta[:, 0],
                delta[:, 1],
                angles="xy",
                scale_units="xy",
                scale=1,
                color=MUTED,
                alpha=0.19,
                width=0.0024,
                headwidth=4.2,
                headlength=5.0,
                zorder=1,
            )
            axis.scatter(
                human_plane[:, 0],
                human_plane[:, 1],
                s=9,
                color=BLUE,
                alpha=0.25,
                edgecolor="none",
                zorder=3,
            )
            axis.scatter(
                ai_plane[:, 0],
                ai_plane[:, 1],
                s=9,
                color=PINK,
                alpha=0.25,
                edgecolor="none",
                zorder=3,
            )
            human_centroid = human_plane.mean(axis=0)
            ai_centroid = ai_plane.mean(axis=0)
            axis.annotate(
                "",
                xy=ai_centroid,
                xytext=human_centroid,
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": PINK_DARK,
                    "linewidth": 2.5,
                    "mutation_scale": 14,
                },
                zorder=5,
            )
            axis.scatter(
                *human_centroid,
                marker="D",
                s=52,
                color=BLUE_DARK,
                edgecolor="white",
                linewidth=0.8,
                zorder=6,
            )
            axis.scatter(
                *ai_centroid,
                marker="D",
                s=52,
                color=PINK_DARK,
                edgecolor="white",
                linewidth=0.8,
                zorder=6,
            )
            axis.axvline(0, color=GRID, linewidth=0.7, zorder=0)
            axis.axhline(0, color=GRID, linewidth=0.7, zorder=0)
            axis.set_xlim(source_limits)
            metric_row = layer_metrics[
                (layer_metrics["model"] == model)
                & (layer_metrics["representation"] == representation)
                & (layer_metrics["layer"] == layer)
            ].iloc[0]
            heldout_auc = float(
                random_eval[
                    (random_eval["model"] == model)
                    & (
                        random_eval["representation"]
                        == representation
                    )
                    & (random_eval["layer"] == layer)
                    & (random_eval["method"] == "nullspace")
                ]["heldout_auc_mean"].iloc[0]
            )
            worst_generator_auc = float(
                generator_eval[
                    (generator_eval["model"] == model)
                    & (
                        generator_eval["representation"]
                        == representation
                    )
                    & (generator_eval["layer"] == layer)
                    & (generator_eval["method"] == "nullspace")
                ]["auc"].min()
            )
            axis.text(
                0.97,
                0.035,
                (
                    f"held-out AUC  {heldout_auc:.3f}\n"
                    f"worst generator  {worst_generator_auc:.3f}\n"
                    f"nullspace agreement  "
                    f"{metric_row['nullspace_pair_positive_rate']:.3f}"
                ),
                transform=axis.transAxes,
                ha="right",
                va="bottom",
                fontsize=8.2,
                color=INK,
                linespacing=1.35,
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": "white",
                    "edgecolor": GRID,
                    "alpha": 0.88,
                },
                zorder=8,
            )
            representation_label = (
                "Last token"
                if representation == "last_token"
                else "Mean pool"
            )
            panel_letter = chr(
                ord("a") + row_index * len(MODEL_ORDER) + column_index
            )
            marginal_axis.text(
                0.01,
                1.08,
                f"({panel_letter})  {MODEL_LABEL[model]} · "
                f"{representation_label} · L{layer}",
                transform=marginal_axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=10.0,
                fontweight="semibold",
                color=INK,
                clip_on=False,
            )
            axis.grid(False)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=BLUE,
            markeredgecolor="none",
            label="Human",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=PINK,
            markeredgecolor="none",
            label="AI",
        ),
        Line2D(
            [0],
            [0],
            color=MUTED,
            linewidth=1.2,
            marker=">",
            label="75 sampled Human→AI vectors",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=9.2,
        columnspacing=1.7,
        handletextpad=0.65,
    )
    fig.supxlabel(
        "Nullspace source coordinate (standardized)",
        x=0.52,
        y=0.012,
        fontsize=10.5,
    )
    fig.supylabel(
        "Orthogonal state coordinate (z)",
        x=0.012,
        y=0.49,
        fontsize=10.5,
    )
    save_figure(fig, output_dir / "paired_endpoint_flow_fields")


def plot_layer_diagnostic_glyph_matrix(
    layer_metrics: pd.DataFrame,
    random_eval: pd.DataFrame,
    generator_eval: pd.DataFrame,
    nuisance_frame: pd.DataFrame,
    robust_layers: dict[tuple[str, str], int],
    output_dir: Path,
) -> None:
    """Compress five layer-selection diagnostics into one glyph matrix."""
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4))
    petal_specs = [
        ("Held-out AUC", PINK, 45, 135),
        ("Worst-generator AUC", BLUE, 135, 225),
        ("Pair agreement", PINK_LIGHT, 225, 315),
        ("Bootstrap stability", BLUE_LIGHT, 315, 405),
    ]
    source_nuisance = nuisance_frame[
        nuisance_frame["target"] == "source_direction_projection"
    ]

    for axis, model in zip(axes, MODEL_ORDER):
        axis.axhspan(0.52, 1.48, color=BLUE_OPEN, alpha=0.42, zorder=0)
        axis.axhspan(-0.48, 0.48, color=PINK_OPEN, alpha=0.48, zorder=0)
        for row_index, representation in enumerate(REPRESENTATION_ORDER):
            y_value = 1 - row_index
            subset = layer_metrics[
                (layer_metrics["model"] == model)
                & (layer_metrics["representation"] == representation)
            ].sort_values("layer")
            layers = subset["layer"].astype(int).tolist()
            for x_value, layer in enumerate(layers):
                metric_row = subset[subset["layer"] == layer].iloc[0]
                heldout_auc = float(
                    random_eval[
                        (random_eval["model"] == model)
                        & (
                            random_eval["representation"]
                            == representation
                        )
                        & (random_eval["layer"] == layer)
                        & (random_eval["method"] == "nullspace")
                    ]["heldout_auc_mean"].iloc[0]
                )
                worst_generator_auc = float(
                    generator_eval[
                        (generator_eval["model"] == model)
                        & (
                            generator_eval["representation"]
                            == representation
                        )
                        & (generator_eval["layer"] == layer)
                        & (generator_eval["method"] == "nullspace")
                    ]["auc"].min()
                )
                nuisance = float(
                    source_nuisance[
                        (source_nuisance["model"] == model)
                        & (
                            source_nuisance["representation"]
                            == representation
                        )
                        & (source_nuisance["layer"] == layer)
                    ]["rho"].abs().max()
                )
                values = [
                    np.clip((heldout_auc - 0.5) / 0.5, 0, 1),
                    np.clip((worst_generator_auc - 0.5) / 0.5, 0, 1),
                    np.clip(
                        (metric_row["oof_positive_rate"] - 0.5) / 0.5,
                        0,
                        1,
                    ),
                    np.clip(
                        metric_row["bootstrap_direction_cosine_lo"],
                        0,
                        1,
                    ),
                ]
                for value, (_, color, theta1, theta2) in zip(
                    values,
                    petal_specs,
                ):
                    radius = 0.13 + 0.27 * value
                    axis.add_patch(
                        Wedge(
                            (x_value, y_value),
                            radius,
                            theta1,
                            theta2,
                            facecolor=color,
                            edgecolor="white",
                            linewidth=0.8,
                            alpha=0.88,
                            zorder=3,
                        )
                    )
                nuisance_radius = 0.025 + 0.075 * np.clip(
                    nuisance / 0.5,
                    0,
                    1,
                )
                axis.add_patch(
                    Circle(
                        (x_value, y_value),
                        nuisance_radius,
                        facecolor="white",
                        edgecolor=INK,
                        linewidth=0.55,
                        zorder=5,
                    )
                )
                if layer == robust_layers[(model, representation)]:
                    axis.add_patch(
                        Circle(
                            (x_value, y_value),
                            0.44,
                            facecolor="none",
                            edgecolor=INK,
                            linewidth=1.35,
                            zorder=6,
                        )
                    )
                    axis.scatter(
                        [x_value],
                        [y_value + 0.48],
                        marker="*",
                        s=45,
                        color=INK,
                        zorder=7,
                    )
            if row_index == 0:
                axis.set_xticks(range(len(layers)), layers)
            axis.text(
                -0.62,
                y_value,
                (
                    "Last token"
                    if representation == "last_token"
                    else "Mean pool"
                ),
                ha="right",
                va="center",
                fontsize=8.2,
                color=INK,
                fontweight="semibold",
            )
        axis.set_xlim(-0.78, 7.55)
        axis.set_ylim(-0.62, 1.62)
        axis.set_aspect("equal")
        axis.set_anchor("N")
        axis.set_title(MODEL_LABEL[model], pad=10)
        axis.set_xlabel("Decoder layer")
        axis.set_yticks([])
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_visible(False)

    legend_handles = [
        Patch(
            facecolor=color,
            edgecolor="white",
            label=label,
        )
        for label, color, _, _ in petal_specs
    ]
    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker="o",
                markersize=5,
                markerfacecolor="white",
                markeredgecolor=INK,
                linestyle="",
                label="Larger center hole = stronger nuisance association",
            ),
            Line2D(
                [0],
                [0],
                marker="*",
                markersize=7,
                color=INK,
                linestyle="",
                label="Robust layer",
            ),
        ]
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.815),
        fontsize=8.3,
    )
    fig.suptitle(
        "Layer-selection diagnostic glyph matrix",
        y=0.985,
        fontsize=15,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.90,
        "Top / left / bottom / right petals encode held-out AUC / worst-generator AUC / pair agreement / bootstrap stability",
        ha="center",
        color=MUTED,
    )
    fig.subplots_adjust(
        left=0.055,
        right=0.99,
        bottom=0.08,
        top=0.62,
        wspace=0.12,
    )
    save_figure(fig, output_dir / "layer_diagnostic_glyph_matrix")


def _auc_cell_color(value: float, root_color: str) -> tuple[float, ...]:
    strength = float(np.clip((value - 0.5) / 0.5, 0, 1))
    white = np.asarray(to_rgb("#FFFFFF"))
    root = np.asarray(to_rgb(root_color))
    rgb = white * (1 - strength) + root * strength
    return (*rgb.tolist(), 1.0)


def plot_layer_selection_robustness_atlas(
    layer_metrics: pd.DataFrame,
    random_eval: pd.DataFrame,
    generator_eval: pd.DataFrame,
    nuisance_frame: pd.DataFrame,
    robust_layers: dict[tuple[str, str], int],
    output_dir: Path,
) -> None:
    """Show layerwise held-out performance and generator-level spread."""
    del layer_metrics, nuisance_frame
    nullspace_random = random_eval[random_eval["method"] == "nullspace"]
    nullspace_generator = generator_eval[
        generator_eval["method"] == "nullspace"
    ]
    generators = sorted(nullspace_generator["generator"].unique())
    generator_markers = {
        generator: marker
        for generator, marker in zip(generators, ("o", "s", "^", "D"))
    }
    all_auc = np.concatenate(
        [
            nullspace_random["heldout_auc_mean"].to_numpy(),
            nullspace_generator["auc"].to_numpy(),
        ]
    )
    y_min = max(0.5, float(all_auc.min()) - 0.035)
    y_max = min(1.01, float(all_auc.max()) + 0.018)
    density_grid = np.linspace(y_min, y_max, 220)

    fig = plt.figure(figsize=(15.4, 5.35))
    outer = fig.add_gridspec(
        1,
        3,
        left=0.055,
        right=0.985,
        bottom=0.14,
        top=0.89,
        wspace=0.20,
    )

    for model_index, model in enumerate(MODEL_ORDER):
        nested = outer[model_index].subgridspec(
            1,
            2,
            width_ratios=[0.84, 0.16],
            wspace=0.03,
        )
        main_axis = fig.add_subplot(nested[0, 0])
        density_axis = fig.add_subplot(nested[0, 1], sharey=main_axis)
        main_axis.axhline(
            0.90,
            color=GRID,
            linewidth=0.8,
            linestyle="--",
            zorder=0,
        )

        for representation in REPRESENTATION_ORDER:
            color = BLUE if representation == "last_token" else PINK
            dark_color = (
                BLUE_DARK
                if representation == "last_token"
                else PINK_DARK
            )
            light_color = (
                BLUE_LIGHT
                if representation == "last_token"
                else PINK_LIGHT
            )
            complement = (
                PINK_DARK
                if representation == "last_token"
                else BLUE_DARK
            )
            marker = "o" if representation == "last_token" else "D"
            representation_offset = (
                -0.010 if representation == "last_token" else 0.010
            )

            heldout = nullspace_random[
                (nullspace_random["model"] == model)
                & (
                    nullspace_random["representation"]
                    == representation
                )
            ].sort_values("layer")
            layers = heldout["layer"].astype(int).to_numpy()
            x_values = heldout["relative_depth"].to_numpy()
            x_plot = x_values + representation_offset
            heldout_values = heldout["heldout_auc_mean"].to_numpy()
            generator_subset = nullspace_generator[
                (nullspace_generator["model"] == model)
                & (
                    nullspace_generator["representation"]
                    == representation
                )
            ]
            generator_matrix = (
                generator_subset.pivot(
                    index="layer",
                    columns="generator",
                    values="auc",
                )
                .reindex(index=layers, columns=generators)
                .to_numpy()
            )
            lower = generator_matrix.min(axis=1)
            upper = generator_matrix.max(axis=1)
            quartile_low = np.quantile(
                generator_matrix,
                0.25,
                axis=1,
            )
            quartile_high = np.quantile(
                generator_matrix,
                0.75,
                axis=1,
            )

            main_axis.fill_between(
                x_plot,
                lower,
                upper,
                color=light_color,
                alpha=0.32,
                linewidth=0,
                zorder=1,
            )
            main_axis.fill_between(
                x_plot,
                quartile_low,
                quartile_high,
                color=color,
                alpha=0.20,
                linewidth=0,
                zorder=2,
            )
            main_axis.plot(
                x_plot,
                heldout_values,
                color=dark_color,
                linewidth=1.6,
                linestyle=(
                    "-"
                    if representation == "last_token"
                    else "--"
                ),
                alpha=0.92,
                zorder=4,
            )

            generator_offsets = np.linspace(
                -0.006,
                0.006,
                len(generators),
            )
            for generator_index, generator in enumerate(generators):
                main_axis.scatter(
                    x_plot + generator_offsets[generator_index],
                    generator_matrix[:, generator_index],
                    marker=generator_markers[generator],
                    s=15,
                    facecolor=color,
                    edgecolor="white",
                    linewidth=0.35,
                    alpha=0.48,
                    zorder=3,
                )

            robust_layer = robust_layers[(model, representation)]
            robust_index = int(np.flatnonzero(layers == robust_layer)[0])
            point_edges = [
                complement if layer == robust_layer else dark_color
                for layer in layers
            ]
            point_widths = [
                1.9 if layer == robust_layer else 0.8
                for layer in layers
            ]
            main_axis.scatter(
                x_plot,
                heldout_values,
                marker=marker,
                s=[
                    76 if layer == robust_layer else 30
                    for layer in layers
                ],
                facecolor="white",
                edgecolor=point_edges,
                linewidth=point_widths,
                zorder=6,
            )
            main_axis.annotate(
                f"L{robust_layer}",
                (
                    x_plot[robust_index],
                    heldout_values[robust_index],
                ),
                xytext=(
                    0,
                    11 if representation == "last_token" else -15,
                ),
                textcoords="offset points",
                ha="center",
                va=(
                    "bottom"
                    if representation == "last_token"
                    else "top"
                ),
                fontsize=8.2,
                fontweight="semibold",
                color=complement,
            )

            pooled_generator_auc = generator_matrix.ravel()
            density = _kde_1d(
                pooled_generator_auc,
                density_grid,
            )
            direction = -1 if representation == "last_token" else 1
            density_axis.fill_betweenx(
                density_grid,
                0,
                direction * density,
                color=color,
                alpha=0.55,
                linewidth=0,
            )
            density_axis.plot(
                direction * density,
                density_grid,
                color=dark_color,
                linewidth=1.1,
            )

        density_axis.axvline(0, color=GRID, linewidth=0.7)
        density_axis.text(
            -0.52,
            y_max - 0.006,
            "LT",
            ha="center",
            va="top",
            fontsize=7.6,
            color=BLUE_DARK,
            fontweight="semibold",
        )
        density_axis.text(
            0.52,
            y_max - 0.006,
            "MP",
            ha="center",
            va="top",
            fontsize=7.6,
            color=PINK_DARK,
            fontweight="semibold",
        )
        density_axis.set_xlim(-1.08, 1.08)
        density_axis.set_ylim(y_min, y_max)
        density_axis.axis("off")

        main_axis.set_xlim(0.04, 1.04)
        main_axis.set_ylim(y_min, y_max)
        main_axis.set_xticks(
            [0.1, 0.3, 0.5, 0.7, 0.9],
            ["10%", "30%", "50%", "70%", "90%"],
        )
        if model_index != 0:
            main_axis.set_yticklabels([])
        main_axis.text(
            0.02,
            0.975,
            f"({chr(ord('a') + model_index)})  {MODEL_LABEL[model]}",
            transform=main_axis.transAxes,
            ha="left",
            va="top",
            fontsize=10.0,
            fontweight="semibold",
            color=INK,
            bbox={
                "boxstyle": "square,pad=0.18",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.76,
            },
            zorder=10,
        )
        main_axis.grid(axis="x", visible=False)
        main_axis.grid(axis="y", color=GRID, linewidth=0.6)
        for spine in ("top", "right"):
            main_axis.spines[spine].set_visible(False)

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=BLUE_DARK,
            marker="o",
            markerfacecolor="white",
            markeredgecolor=BLUE_DARK,
            linewidth=1.6,
            label="Last-token AUC",
        ),
        Line2D(
            [0],
            [0],
            color=PINK_DARK,
            marker="D",
            markerfacecolor="white",
            markeredgecolor=PINK_DARK,
            linewidth=1.6,
            linestyle="--",
            label="Mean-pool AUC",
        ),
    ]
    legend_handles.extend(
        [
            Line2D(
                [0],
                [0],
                marker=generator_markers[generator],
                markersize=5.8,
                markerfacecolor=BLUE,
                markeredgecolor="white",
                linestyle="",
                label=_short_generator_label(generator),
            )
            for generator in generators
        ]
    )
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=7.8,
            markerfacecolor="white",
            markeredgecolor=PINK_DARK,
            markeredgewidth=1.8,
            linestyle="",
            label="Selected layer",
        )
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=7,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=9.0,
        columnspacing=1.05,
        handletextpad=0.40,
    )
    fig.supxlabel(
        "Relative decoder depth",
        x=0.5,
        y=0.018,
        fontsize=10.5,
    )
    fig.supylabel(
        "AUC (focused scale)",
        x=0.012,
        y=0.50,
        fontsize=10.5,
    )
    save_figure(fig, output_dir / "layer_selection_robustness_atlas")


def _kde_1d(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if np.std(values) < 1e-8:
        return np.zeros_like(grid)
    density = gaussian_kde(values)(grid)
    maximum = max(float(density.max()), 1e-12)
    return density / maximum


def plot_pair_concordance_cartography(
    pair_frame: pd.DataFrame,
    robust_layers: dict[tuple[str, str], int],
    output_dir: Path,
) -> None:
    """Compare the two continuation representations at pair level."""
    generator_markers = {
        "qwen3.6-flash": "o",
        "glm-5.2": "s",
        "deepseek-v4-flash": "^",
        "MiniMax-M2.5": "D",
    }
    model_frames: dict[str, pd.DataFrame] = {}
    all_values: list[np.ndarray] = []
    for model in MODEL_ORDER:
        last_layer = robust_layers[(model, "last_token")]
        mean_layer = robust_layers[(model, "mean_pool")]
        last = pair_frame[
            (pair_frame["model"] == model)
            & (pair_frame["representation"] == "last_token")
            & (pair_frame["layer"] == last_layer)
        ][["sample_id", "nullspace_cosine", "generator"]].rename(
            columns={"nullspace_cosine": "last_cosine"}
        )
        mean = pair_frame[
            (pair_frame["model"] == model)
            & (pair_frame["representation"] == "mean_pool")
            & (pair_frame["layer"] == mean_layer)
        ][["sample_id", "nullspace_cosine"]].rename(
            columns={"nullspace_cosine": "mean_cosine"}
        )
        merged = last.merge(mean, on="sample_id", validate="one_to_one")
        model_frames[model] = merged
        all_values.extend(
            [
                merged["last_cosine"].to_numpy(),
                merged["mean_cosine"].to_numpy(),
            ]
        )
    minimum = min(float(values.min()) for values in all_values)
    maximum = max(float(values.max()) for values in all_values)
    padding = (maximum - minimum) * 0.06
    limits = (minimum - padding, maximum + padding)
    density_grid = np.linspace(*limits, 220)

    fig = plt.figure(figsize=(15.1, 5.25))
    outer = fig.add_gridspec(
        1,
        3,
        left=0.055,
        right=0.985,
        bottom=0.14,
        top=0.89,
        wspace=0.18,
    )
    for model_index, model in enumerate(MODEL_ORDER):
        nested = outer[model_index].subgridspec(
            2,
            2,
            height_ratios=[0.18, 0.82],
            width_ratios=[0.82, 0.18],
            hspace=0.03,
            wspace=0.03,
        )
        top_axis = fig.add_subplot(nested[0, 0])
        main_axis = fig.add_subplot(nested[1, 0])
        right_axis = fig.add_subplot(nested[1, 1])
        frame = model_frames[model]
        x_values = frame["last_cosine"].to_numpy()
        y_values = frame["mean_cosine"].to_numpy()
        difference = y_values - x_values
        comparison_grid = np.linspace(*limits, 240)
        main_axis.fill_between(
            comparison_grid,
            limits[0],
            comparison_grid,
            color=BLUE_OPEN,
            alpha=0.62,
            linewidth=0,
            zorder=0,
        )
        main_axis.fill_between(
            comparison_grid,
            comparison_grid,
            limits[1],
            color=PINK_OPEN,
            alpha=0.62,
            linewidth=0,
            zorder=0,
        )
        last_stronger = difference < 0
        mean_stronger = ~last_stronger
        if last_stronger.sum() >= 20:
            _density_contours(
                main_axis,
                np.column_stack(
                    [x_values[last_stronger], y_values[last_stronger]]
                ),
                BLUE_DARK,
            )
        if mean_stronger.sum() >= 20:
            _density_contours(
                main_axis,
                np.column_stack(
                    [x_values[mean_stronger], y_values[mean_stronger]]
                ),
                PINK_DARK,
            )

        for generator, marker in generator_markers.items():
            mask = frame["generator"].to_numpy() == generator
            point_colors = [
                PINK if value >= 0 else BLUE
                for value in difference[mask]
            ]
            main_axis.scatter(
                x_values[mask],
                y_values[mask],
                s=14,
                marker=marker,
                color=point_colors,
                alpha=0.34,
                edgecolor="none",
                zorder=2,
            )
            centroid = np.asarray(
                [x_values[mask].mean(), y_values[mask].mean()]
            )
            main_axis.scatter(
                centroid[0],
                centroid[1],
                s=72,
                marker=marker,
                facecolor="white",
                edgecolor=(
                    BLUE_DARK
                    if centroid[1] >= centroid[0]
                    else PINK_DARK
                ),
                linewidth=1.2,
                zorder=5,
            )

        main_axis.plot(
            limits,
            limits,
            linestyle="--",
            linewidth=1.0,
            color=MUTED,
            zorder=1,
        )
        main_axis.axvline(0, color=GRID, linewidth=0.8, zorder=0)
        main_axis.axhline(0, color=GRID, linewidth=0.8, zorder=0)
        main_axis.set_xlim(limits)
        main_axis.set_ylim(limits)
        main_axis.set_aspect("equal", adjustable="box")
        if model_index != 0:
            main_axis.set_yticklabels([])
        main_axis.grid(False)

        both_positive = np.mean((x_values > 0) & (y_values > 0))
        discordant = np.mean((x_values > 0) != (y_values > 0))
        median_delta = float(np.median(difference))
        rho, _ = spearmanr(x_values, y_values)
        main_axis.text(
            0.03,
            0.97,
            (
                f"ρ = {rho:.2f}\n"
                f"both positive = {both_positive:.1%}\n"
                f"sign discordance = {discordant:.1%}\n"
                f"median Δ(mean−last) = {median_delta:+.3f}"
            ),
            transform=main_axis.transAxes,
            ha="left",
            va="top",
            fontsize=8.2,
            color=INK,
            linespacing=1.35,
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": GRID,
                "alpha": 0.9,
            },
            zorder=8,
        )

        x_density = _kde_1d(x_values, density_grid)
        y_density = _kde_1d(y_values, density_grid)
        top_axis.fill_between(
            density_grid,
            0,
            x_density,
            color=BLUE,
            alpha=0.68,
        )
        top_axis.plot(density_grid, x_density, color=BLUE_DARK, linewidth=1.1)
        top_axis.set_xlim(limits)
        top_axis.set_ylim(0, 1.08)
        top_axis.axis("off")
        top_axis.text(
            0.01,
            1.06,
            f"({chr(ord('a') + model_index)})  {MODEL_LABEL[model]} · "
            f"LT L{robust_layers[(model, 'last_token')]} / "
            f"MP L{robust_layers[(model, 'mean_pool')]}",
            transform=top_axis.transAxes,
            ha="left",
            va="bottom",
            fontsize=10.0,
            fontweight="semibold",
            color=INK,
            clip_on=False,
        )

        right_axis.fill_betweenx(
            density_grid,
            0,
            y_density,
            color=PINK,
            alpha=0.68,
        )
        right_axis.plot(y_density, density_grid, color=PINK_DARK, linewidth=1.1)
        right_axis.set_ylim(limits)
        right_axis.set_xlim(0, 1.08)
        right_axis.axis("off")

    handles = [
        Line2D(
            [0],
            [0],
            marker=marker,
            linestyle="",
            markerfacecolor="white",
            markeredgecolor=MUTED,
            label=_short_generator_label(generator),
        )
        for generator, marker in generator_markers.items()
    ]
    handles.extend(
        [
            Patch(facecolor=BLUE, edgecolor="none", label="Last token stronger"),
            Patch(facecolor=PINK, edgecolor="none", label="Mean pool stronger"),
        ]
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=6,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=9.0,
        columnspacing=1.20,
        handletextpad=0.48,
    )
    fig.supxlabel(
        "Last-token nullspace cosine",
        x=0.5,
        y=0.018,
        fontsize=10.5,
    )
    fig.supylabel(
        "Mean-pooled nullspace cosine",
        x=0.012,
        y=0.48,
        fontsize=10.5,
    )
    save_figure(fig, output_dir / "pair_concordance_cartography")


def _simplex_coordinates(
    shared: np.ndarray,
    structured: np.ndarray,
    diffuse: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    height = np.sqrt(3) / 2
    x_values = 0.5 * shared + diffuse
    y_values = height * shared
    return x_values, y_values


def _draw_simplex(axis: plt.Axes) -> None:
    height = np.sqrt(3) / 2
    vertices = np.asarray([[0, 0], [1, 0], [0.5, height], [0, 0]])
    axis.plot(vertices[:, 0], vertices[:, 1], color=INK, linewidth=1.0)
    for fraction in (0.25, 0.5, 0.75):
        grid_sets = [
            _simplex_coordinates(
                np.asarray([fraction, fraction]),
                np.asarray([0, 1 - fraction]),
                np.asarray([1 - fraction, 0]),
            ),
            _simplex_coordinates(
                np.asarray([0, 1 - fraction]),
                np.asarray([fraction, fraction]),
                np.asarray([1 - fraction, 0]),
            ),
            _simplex_coordinates(
                np.asarray([0, 1 - fraction]),
                np.asarray([1 - fraction, 0]),
                np.asarray([fraction, fraction]),
            ),
        ]
        for x_values, y_values in grid_sets:
            axis.plot(
                x_values,
                y_values,
                color=GRID,
                linewidth=0.55,
                zorder=0,
            )
    axis.text(
        0.5,
        height + 0.045,
        "Shared source energy",
        ha="center",
        va="bottom",
        fontsize=8,
        color=MUTED,
    )
    axis.text(
        -0.015,
        -0.045,
        "Structured residual",
        ha="left",
        va="top",
        fontsize=8,
        color=MUTED,
    )
    axis.text(
        1.015,
        -0.045,
        "Diffuse residual",
        ha="right",
        va="top",
        fontsize=8,
        color=MUTED,
    )
    axis.set_xlim(-0.06, 1.06)
    axis.set_ylim(-0.08, height + 0.09)
    axis.set_aspect("equal")
    axis.axis("off")


def plot_angular_energy_simplex(
    layer_metrics: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Decompose unit-displacement energy into exact three-part shares."""
    fig, axes = plt.subplots(1, 3, figsize=(14.6, 4.9))
    for axis, model in zip(axes, MODEL_ORDER):
        _draw_simplex(axis)
        for representation in REPRESENTATION_ORDER:
            subset = layer_metrics[
                (layer_metrics["model"] == model)
                & (layer_metrics["representation"] == representation)
            ].sort_values("layer")
            shared = subset[
                "angular_coherent_displacement_fraction"
            ].to_numpy()
            residual = 1.0 - shared
            structured = (
                residual * subset["angular_variance_top_1"].to_numpy()
            )
            diffuse = residual - structured
            x_values, y_values = _simplex_coordinates(
                shared,
                structured,
                diffuse,
            )
            color = REPRESENTATION_COLOR[representation]
            for index in range(len(x_values) - 1):
                axis.annotate(
                    "",
                    xy=(x_values[index + 1], y_values[index + 1]),
                    xytext=(x_values[index], y_values[index]),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": color,
                        "alpha": 0.52,
                        "linewidth": 1.2,
                        "mutation_scale": 9,
                    },
                    zorder=2,
                )
            axis.scatter(
                x_values,
                y_values,
                s=54,
                color=color,
                edgecolor="white",
                linewidth=0.7,
                alpha=0.9,
                zorder=3,
            )
            layers = subset["layer"].astype(int).to_numpy()
            for point_index, (layer, x_value, y_value) in enumerate(zip(
                layers,
                x_values,
                y_values,
            )):
                if point_index not in (0, len(layers) - 1):
                    continue
                horizontal_offset = (
                    -12 if representation == "last_token" else 12
                )
                vertical_offset = 9 if point_index == 0 else -9
                axis.annotate(
                    f"L{layer}",
                    xy=(x_value, y_value),
                    xytext=(horizontal_offset, vertical_offset),
                    textcoords="offset points",
                    ha=(
                        "right"
                        if representation == "last_token"
                        else "left"
                    ),
                    va="center",
                    fontsize=6.8,
                    fontweight="semibold",
                    color=INK,
                    zorder=4,
                )
        axis.set_title(MODEL_LABEL[model], pad=8)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color=color,
            label=REPRESENTATION_LABEL[representation],
        )
        for representation, color in REPRESENTATION_COLOR.items()
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.89),
    )
    fig.suptitle(
        "Angular displacement energy simplex",
        y=1.0,
        fontsize=15,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.945,
        "Exact partition: coherent mean direction + residual PC1 + remaining residual; arrows follow depth",
        ha="center",
        color=MUTED,
    )
    fig.subplots_adjust(
        left=0.03,
        right=0.98,
        bottom=0.04,
        top=0.78,
        wspace=0.1,
    )
    save_figure(fig, output_dir / "angular_energy_simplex")


def plot_robust_layer_radial_fingerprints(
    layer_metrics: pd.DataFrame,
    random_eval: pd.DataFrame,
    nuisance_frame: pd.DataFrame,
    robust_layers: dict[tuple[str, str], int],
    output_dir: Path,
) -> None:
    labels = [
        "AUC",
        "Pair\nagreement",
        "Bootstrap\nstability",
        "Generator\nagreement",
        "Residual\nrichness",
        "Nuisance\nsafety",
    ]
    maximum_rank = max(
        float(layer_metrics["angular_effective_rank"].max()),
        1.0,
    )
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(13.4, 8.4),
        subplot_kw={"projection": "polar"},
    )
    for row_index, representation in enumerate(REPRESENTATION_ORDER):
        for column_index, model in enumerate(MODEL_ORDER):
            axis = axes[row_index, column_index]
            layer = robust_layers[(model, representation)]
            metrics = layer_metrics[
                (layer_metrics["model"] == model)
                & (layer_metrics["representation"] == representation)
                & (layer_metrics["layer"] == layer)
            ].iloc[0]
            auc = random_eval[
                (random_eval["model"] == model)
                & (random_eval["representation"] == representation)
                & (random_eval["layer"] == layer)
                & (random_eval["method"] == "nullspace")
            ]["heldout_auc_mean"].iloc[0]
            nuisance_subset = nuisance_frame[
                (nuisance_frame["model"] == model)
                & (nuisance_frame["representation"] == representation)
                & (nuisance_frame["layer"] == layer)
                & (
                    nuisance_frame["target"]
                    == "source_direction_projection"
                )
            ]
            max_nuisance = float(nuisance_subset["rho"].abs().max())
            values = np.asarray(
                [
                    np.clip((auc - 0.5) / 0.5, 0, 1),
                    metrics["oof_positive_rate"],
                    metrics["bootstrap_direction_cosine_lo"],
                    np.clip(
                        (
                            metrics[
                                "angular_generator_direction_cosine_min"
                            ]
                            + 1
                        )
                        / 2,
                        0,
                        1,
                    ),
                    metrics["angular_effective_rank"] / maximum_rank,
                    1 - np.clip(max_nuisance / 0.5, 0, 1),
                ]
            )
            angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False)
            width = 2 * np.pi / len(values) * 0.72
            colors = [
                BLUE,
                PINK,
                BLUE,
                PINK,
                BLUE,
                PINK,
            ]
            axis.bar(
                angles,
                values,
                width=width,
                bottom=0,
                color=colors,
                edgecolor="white",
                linewidth=1.1,
                alpha=0.9,
            )
            axis.scatter(
                angles,
                values,
                s=18,
                color=INK,
                zorder=4,
            )
            axis.set_ylim(0, 1)
            axis.set_xticks(angles, labels)
            axis.tick_params(axis="x", labelsize=7.2, pad=2)
            axis.set_yticks([0.25, 0.5, 0.75, 1.0])
            axis.set_yticklabels([])
            axis.grid(color=GRID, alpha=0.8)
            axis.spines["polar"].set_color(GRID)
            axis.set_title(
                f"{MODEL_LABEL[model]} · "
                f"{REPRESENTATION_LABEL[representation]}\n"
                f"robust layer L{layer}",
                pad=16,
            )
            axis.text(
                0.5,
                0.5,
                f"{auc:.3f}",
                transform=axis.transAxes,
                ha="center",
                va="center",
                fontsize=11,
                fontweight="semibold",
                color=INK,
            )
    fig.suptitle(
        "Robust-layer diagnostic fingerprints",
        y=1.0,
        fontsize=15,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.966,
        "Petal radius is a 0–1 diagnostic score; center text is held-out nullspace AUC",
        ha="center",
        color=MUTED,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=2.1, w_pad=1.2)
    save_figure(fig, output_dir / "robust_layer_radial_fingerprints")


def _circle_point(angle: float, radius: float) -> np.ndarray:
    return np.asarray([np.cos(angle) * radius, np.sin(angle) * radius])


def plot_cross_model_layer_chord_wheels(
    cka_frame: pd.DataFrame,
    layer_lookup: dict[tuple[str, str], list[int]],
    output_dir: Path,
) -> None:
    """Render Hungarian-matched cross-model layers as circular chords."""
    model_pairs = [
        (MODEL_ORDER[0], MODEL_ORDER[1]),
        (MODEL_ORDER[0], MODEL_ORDER[2]),
        (MODEL_ORDER[1], MODEL_ORDER[2]),
    ]
    sector_centers = np.deg2rad([90, 210, 330])
    sector_width = np.deg2rad(72)
    fig, axes = plt.subplots(1, 2, figsize=(13.8, 7.0))
    for axis, representation in zip(axes, REPRESENTATION_ORDER):
        node_positions: dict[tuple[str, int], np.ndarray] = {}
        node_angles: dict[tuple[str, int], float] = {}
        for model_index, model in enumerate(MODEL_ORDER):
            layers = layer_lookup[(model, representation)]
            angles = np.linspace(
                sector_centers[model_index] - sector_width / 2,
                sector_centers[model_index] + sector_width / 2,
                len(layers),
            )
            arc_points = np.asarray(
                [_circle_point(angle, 1.0) for angle in angles]
            )
            axis.plot(
                arc_points[:, 0],
                arc_points[:, 1],
                color=(
                    BLUE_DARK
                    if representation == "last_token"
                    else PINK_DARK
                ),
                linewidth=7,
                solid_capstyle="round",
                alpha=0.82,
                zorder=3,
            )
            for layer, angle, point in zip(layers, angles, arc_points):
                node_positions[(model, layer)] = point
                node_angles[(model, layer)] = angle
                label_point = _circle_point(angle, 1.09)
                axis.text(
                    label_point[0],
                    label_point[1],
                    str(layer),
                    ha="center",
                    va="center",
                    fontsize=6.2,
                    color=MUTED,
                )
            model_label_point = _circle_point(
                sector_centers[model_index],
                1.34,
            )
            axis.text(
                model_label_point[0],
                model_label_point[1],
                MODEL_LABEL[model],
                ha="center",
                va="center",
                fontsize=9.2,
                fontweight="semibold",
                color=INK,
            )

        for first_model, second_model in model_pairs:
            subset = cka_frame[
                (cka_frame["scope"] == "cross_model")
                & (cka_frame["geometry"] == "angular")
                & (cka_frame["representation_a"] == representation)
                & (cka_frame["model_a"] == first_model)
                & (cka_frame["model_b"] == second_model)
            ]
            matrix = subset.pivot(
                index="layer_a",
                columns="layer_b",
                values="cka",
            )
            rows, columns = linear_sum_assignment(
                -matrix.to_numpy()
            )
            for row, column in zip(rows, columns):
                first_layer = int(matrix.index[row])
                second_layer = int(matrix.columns[column])
                value = float(matrix.iloc[row, column])
                start = node_positions[(first_model, first_layer)]
                end = node_positions[(second_model, second_layer)]
                vertices = [
                    tuple(start),
                    (0.0, 0.0),
                    tuple(end),
                ]
                path = MplPath(
                    vertices,
                    [
                        MplPath.MOVETO,
                        MplPath.CURVE3,
                        MplPath.CURVE3,
                    ],
                )
                axis.add_patch(
                    PathPatch(
                        path,
                        facecolor="none",
                        edgecolor=(
                            BLUE
                            if representation == "last_token"
                            else PINK
                        ),
                        linewidth=0.35 + 4.0 * value,
                        alpha=0.08 + 0.55 * value,
                        zorder=1,
                    )
                )
        axis.add_patch(
            plt.Circle(
                (0, 0),
                0.13,
                facecolor=BLUE_OPEN if representation == "last_token" else PINK_OPEN,
                edgecolor="white",
                linewidth=1.0,
                zorder=4,
            )
        )
        axis.text(
            0,
            0,
            "CKA",
            ha="center",
            va="center",
            fontsize=8,
            fontweight="semibold",
            color=INK,
            zorder=5,
        )
        axis.set_title(REPRESENTATION_LABEL[representation], pad=12)
        axis.set_xlim(-1.43, 1.43)
        axis.set_ylim(-1.43, 1.43)
        axis.set_aspect("equal")
        axis.axis("off")
    fig.suptitle(
        "Cross-model layer concordance chord wheels",
        y=0.99,
        fontsize=15,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.945,
        "Hungarian-matched layer pairs only; chord width and opacity encode angular displacement CKA",
        ha="center",
        color=MUTED,
    )
    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        bottom=0.02,
        top=0.88,
        wspace=0.06,
    )
    save_figure(fig, output_dir / "cross_model_layer_chord_wheels")


def plot_cross_model_geometry_wheels(
    cka_frame: pd.DataFrame,
    layer_metrics: pd.DataFrame,
    random_eval: pd.DataFrame,
    layer_lookup: dict[tuple[str, str], list[int]],
    robust_layers: dict[tuple[str, str], int],
    output_dir: Path,
) -> None:
    """Combine cross-model CKA, layer AUC, rank, and robust selection."""
    model_pairs = [
        (MODEL_ORDER[0], MODEL_ORDER[1]),
        (MODEL_ORDER[0], MODEL_ORDER[2]),
        (MODEL_ORDER[1], MODEL_ORDER[2]),
    ]
    sector_centers = np.deg2rad([90, 210, 330])
    sector_width = np.deg2rad(72)
    fig, axes = plt.subplots(1, 2, figsize=(14.4, 7.6))

    for axis, representation in zip(axes, REPRESENTATION_ORDER):
        color = BLUE if representation == "last_token" else PINK
        dark_color = (
            BLUE_DARK if representation == "last_token" else PINK_DARK
        )
        open_color = (
            BLUE_OPEN if representation == "last_token" else PINK_OPEN
        )
        rank_subset = layer_metrics[
            layer_metrics["representation"] == representation
        ]["angular_effective_rank"]
        rank_min = float(rank_subset.min())
        rank_span = max(float(rank_subset.max()) - rank_min, 1e-8)
        node_positions: dict[tuple[str, int], np.ndarray] = {}

        for model_index, model in enumerate(MODEL_ORDER):
            layers = layer_lookup[(model, representation)]
            angles = np.linspace(
                sector_centers[model_index] - sector_width / 2,
                sector_centers[model_index] + sector_width / 2,
                len(layers),
            )
            arc_points = np.asarray(
                [_circle_point(angle, 1.0) for angle in angles]
            )
            axis.plot(
                arc_points[:, 0],
                arc_points[:, 1],
                color=open_color,
                linewidth=13,
                solid_capstyle="round",
                zorder=2,
            )
            axis.plot(
                arc_points[:, 0],
                arc_points[:, 1],
                color=dark_color,
                linewidth=2.2,
                solid_capstyle="round",
                alpha=0.9,
                zorder=3,
            )
            for layer, angle, point in zip(layers, angles, arc_points):
                node_positions[(model, layer)] = point
                metric_row = layer_metrics[
                    (layer_metrics["model"] == model)
                    & (
                        layer_metrics["representation"]
                        == representation
                    )
                    & (layer_metrics["layer"] == layer)
                ].iloc[0]
                auc = float(
                    random_eval[
                        (random_eval["model"] == model)
                        & (
                            random_eval["representation"]
                            == representation
                        )
                        & (random_eval["layer"] == layer)
                        & (random_eval["method"] == "nullspace")
                    ]["heldout_auc_mean"].iloc[0]
                )
                auc_score = np.clip((auc - 0.5) / 0.5, 0, 1)
                rank_score = (
                    float(metric_row["angular_effective_rank"]) - rank_min
                ) / rank_span
                rank_start = _circle_point(angle, 1.08)
                rank_end = _circle_point(
                    angle,
                    1.08 + 0.19 * rank_score,
                )
                axis.plot(
                    [rank_start[0], rank_end[0]],
                    [rank_start[1], rank_end[1]],
                    color=color,
                    linewidth=4.0,
                    alpha=0.82,
                    solid_capstyle="round",
                    zorder=4,
                )
                robust = layer == robust_layers[(model, representation)]
                axis.scatter(
                    point[0],
                    point[1],
                    s=28 + 155 * auc_score**2,
                    color=color,
                    edgecolor=INK if robust else "white",
                    linewidth=1.8 if robust else 0.75,
                    zorder=6,
                )
                if robust:
                    star_point = _circle_point(angle, 1.44)
                    axis.scatter(
                        star_point[0],
                        star_point[1],
                        marker="*",
                        s=65,
                        color=INK,
                        zorder=8,
                    )
                label_point = _circle_point(angle, 1.30)
                axis.text(
                    label_point[0],
                    label_point[1],
                    str(layer),
                    ha="center",
                    va="center",
                    fontsize=6.2,
                    color=MUTED,
                )
            model_label_point = _circle_point(
                sector_centers[model_index],
                1.67,
            )
            axis.text(
                model_label_point[0],
                model_label_point[1],
                MODEL_LABEL[model],
                ha="center",
                va="center",
                fontsize=9.2,
                fontweight="semibold",
                color=INK,
            )

        for first_model, second_model in model_pairs:
            subset = cka_frame[
                (cka_frame["scope"] == "cross_model")
                & (cka_frame["geometry"] == "angular")
                & (cka_frame["representation_a"] == representation)
                & (cka_frame["model_a"] == first_model)
                & (cka_frame["model_b"] == second_model)
            ]
            matrix = subset.pivot(
                index="layer_a",
                columns="layer_b",
                values="cka",
            )
            rows, columns = linear_sum_assignment(-matrix.to_numpy())
            for row, column in zip(rows, columns):
                first_layer = int(matrix.index[row])
                second_layer = int(matrix.columns[column])
                value = float(matrix.iloc[row, column])
                start = node_positions[(first_model, first_layer)]
                end = node_positions[(second_model, second_layer)]
                path = MplPath(
                    [tuple(start), (0.0, 0.0), tuple(end)],
                    [
                        MplPath.MOVETO,
                        MplPath.CURVE3,
                        MplPath.CURVE3,
                    ],
                )
                axis.add_patch(
                    PathPatch(
                        path,
                        facecolor="none",
                        edgecolor=color,
                        linewidth=0.3 + 3.8 * value,
                        alpha=0.07 + 0.48 * value,
                        zorder=1,
                    )
                )

        axis.add_patch(
            Circle(
                (0, 0),
                0.15,
                facecolor=open_color,
                edgecolor="white",
                linewidth=1.0,
                zorder=7,
            )
        )
        axis.text(
            0,
            0,
            "angular\nCKA",
            ha="center",
            va="center",
            fontsize=7.6,
            fontweight="semibold",
            color=INK,
            zorder=8,
        )
        axis.set_title(REPRESENTATION_LABEL[representation], pad=2)
        axis.set_xlim(-1.78, 1.78)
        axis.set_ylim(-1.78, 1.78)
        axis.set_aspect("equal")
        axis.axis("off")

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=5,
            markerfacecolor=BLUE,
            markeredgecolor="white",
            linestyle="",
            label="Node area = held-out AUC",
        ),
        Line2D(
            [0],
            [0],
            color=PINK,
            linewidth=4,
            label="Outer tick length = angular effective rank",
        ),
        Line2D(
            [0],
            [0],
            color=BLUE,
            linewidth=2.5,
            alpha=0.5,
            label="Chord strength = matched-layer CKA",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            markersize=8,
            color=INK,
            linestyle="",
            label="Robust layer",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.865),
        fontsize=8.3,
    )
    fig.suptitle(
        "Cross-model hidden-geometry wheels",
        y=0.992,
        fontsize=15,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.945,
        "All eight sampled layers per model are retained; only optimal one-to-one cross-model matches are drawn",
        ha="center",
        color=MUTED,
    )
    fig.subplots_adjust(
        left=0.025,
        right=0.975,
        bottom=0.02,
        top=0.78,
        wspace=0.02,
    )
    save_figure(fig, output_dir / "cross_model_geometry_wheels")


def plot_cross_model_alignment_atlas(
    cka_frame: pd.DataFrame,
    random_eval: pd.DataFrame,
    layer_lookup: dict[tuple[str, str], list[int]],
    robust_layers: dict[tuple[str, str], int],
    output_dir: Path,
) -> None:
    """Combine complete CKA matrices with a compact matched-layer network."""
    model_pairs = [
        (MODEL_ORDER[0], MODEL_ORDER[1]),
        (MODEL_ORDER[0], MODEL_ORDER[2]),
        (MODEL_ORDER[1], MODEL_ORDER[2]),
    ]
    fig = plt.figure(figsize=(15.6, 7.2))
    grid = fig.add_gridspec(
        2,
        4,
        left=0.055,
        right=0.985,
        bottom=0.075,
        top=0.89,
        width_ratios=[1, 1, 1, 1.28],
        hspace=0.34,
        wspace=0.28,
    )

    for row_index, representation in enumerate(REPRESENTATION_ORDER):
        color = BLUE if representation == "last_token" else PINK
        dark_color = (
            BLUE_DARK if representation == "last_token" else PINK_DARK
        )
        complement = (
            PINK_DARK if representation == "last_token" else BLUE_DARK
        )
        cmap = BLUE_CMAP if representation == "last_token" else PINK_CMAP
        matched_edges: list[
            tuple[str, int, str, int, float]
        ] = []

        for column_index, (first_model, second_model) in enumerate(
            model_pairs
        ):
            axis = fig.add_subplot(grid[row_index, column_index])
            subset = cka_frame[
                (cka_frame["scope"] == "cross_model")
                & (cka_frame["geometry"] == "angular")
                & (cka_frame["representation_a"] == representation)
                & (cka_frame["model_a"] == first_model)
                & (cka_frame["model_b"] == second_model)
            ]
            matrix = subset.pivot(
                index="layer_a",
                columns="layer_b",
                values="cka",
            )
            image = axis.imshow(
                matrix.to_numpy(),
                origin="lower",
                aspect="equal",
                interpolation="nearest",
                cmap=cmap,
                vmin=0,
                vmax=1,
            )
            rows, columns = linear_sum_assignment(-matrix.to_numpy())
            for matrix_row, matrix_column in zip(rows, columns):
                value = float(matrix.iloc[matrix_row, matrix_column])
                first_layer = int(matrix.index[matrix_row])
                second_layer = int(matrix.columns[matrix_column])
                matched_edges.append(
                    (
                        first_model,
                        first_layer,
                        second_model,
                        second_layer,
                        value,
                    )
                )
                axis.add_patch(
                    Rectangle(
                        (matrix_column - 0.46, matrix_row - 0.46),
                        0.92,
                        0.92,
                        facecolor="none",
                        edgecolor=MATCH_OUTLINE,
                        linewidth=1.2,
                    )
                )
                axis.text(
                    matrix_column,
                    matrix_row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7.0,
                    color=(
                        "white"
                        if value > 0.66
                        else MATCH_OUTLINE
                    ),
                    fontweight="semibold",
                )
            axis.set_xticks(
                np.arange(len(matrix.columns)),
                matrix.columns.astype(int),
                rotation=45,
            )
            axis.set_yticks(
                np.arange(len(matrix.index)),
                matrix.index.astype(int),
            )
            axis.set_xlabel(MODEL_LABEL[second_model])
            axis.set_ylabel(MODEL_LABEL[first_model])
            axis.grid(False)
            if row_index == 0:
                axis.text(
                    0.0,
                    1.045,
                    f"{MODEL_LABEL[first_model]} ↔ "
                    f"{MODEL_LABEL[second_model]}",
                    transform=axis.transAxes,
                    ha="left",
                    va="bottom",
                    fontsize=10.0,
                    fontweight="semibold",
                    color=INK,
                    clip_on=False,
                )
            if column_index == 0:
                axis.text(
                    -0.39,
                    0.5,
                    (
                        "Last token"
                        if representation == "last_token"
                        else "Mean pool"
                    ),
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    rotation=90,
                    fontsize=10.2,
                    fontweight="semibold",
                    color=dark_color,
                )

        network_axis = fig.add_subplot(grid[row_index, 3])
        model_x = {
            MODEL_ORDER[0]: 0.0,
            MODEL_ORDER[1]: 1.0,
            MODEL_ORDER[2]: 2.0,
        }
        for model in MODEL_ORDER:
            layers = layer_lookup[(model, representation)]
            maximum_layer = max(layers)
            for layer in layers:
                y_value = layer / maximum_layer
                auc = float(
                    random_eval[
                        (random_eval["model"] == model)
                        & (
                            random_eval["representation"]
                            == representation
                        )
                        & (random_eval["layer"] == layer)
                        & (random_eval["method"] == "nullspace")
                    ]["heldout_auc_mean"].iloc[0]
                )
                auc_score = np.clip((auc - 0.5) / 0.5, 0, 1)
                robust = layer == robust_layers[(model, representation)]
                network_axis.scatter(
                    model_x[model],
                    y_value,
                    s=24 + 115 * auc_score**2,
                    color=color,
                    edgecolor=complement if robust else "white",
                    linewidth=1.8 if robust else 0.7,
                    zorder=5,
                )
                if robust:
                    network_axis.scatter(
                        model_x[model] + 0.075,
                        y_value + 0.025,
                        marker="*",
                        s=42,
                        color=complement,
                        zorder=7,
                    )
                network_axis.text(
                    model_x[model] - 0.07,
                    y_value,
                    str(layer),
                    ha="right",
                    va="center",
                    fontsize=7.1,
                    color=MUTED,
                )
            network_axis.plot(
                [model_x[model], model_x[model]],
                [
                    min(layers) / maximum_layer,
                    1.0,
                ],
                color=dark_color,
                linewidth=1.2,
                alpha=0.55,
                zorder=1,
            )

        for (
            first_model,
            first_layer,
            second_model,
            second_layer,
            value,
        ) in matched_edges:
            first_maximum = max(
                layer_lookup[(first_model, representation)]
            )
            second_maximum = max(
                layer_lookup[(second_model, representation)]
            )
            x_start = model_x[first_model]
            x_end = model_x[second_model]
            y_start = first_layer / first_maximum
            y_end = second_layer / second_maximum
            midpoint_x = (x_start + x_end) / 2
            vertices = [
                (x_start, y_start),
                (midpoint_x, y_start),
                (midpoint_x, y_end),
                (x_end, y_end),
            ]
            path = MplPath(
                vertices,
                [
                    MplPath.MOVETO,
                    MplPath.CURVE4,
                    MplPath.CURVE4,
                    MplPath.CURVE4,
                ],
            )
            network_axis.add_patch(
                PathPatch(
                    path,
                    facecolor="none",
                    edgecolor=color,
                    linewidth=0.25 + 2.5 * value,
                    alpha=0.05 + 0.32 * value,
                    zorder=2,
                )
            )
        network_axis.set_xlim(-0.32, 2.28)
        network_axis.set_ylim(0.0, 1.08)
        network_axis.set_xticks(
            [0, 1, 2],
            [MODEL_SHORT_LABEL[model] for model in MODEL_ORDER],
        )
        network_axis.set_ylabel("Relative decoder depth")
        network_axis.grid(False)
        for spine in ("top", "right"):
            network_axis.spines[spine].set_visible(False)
        if row_index == 0:
            network_axis.text(
                0.0,
                1.045,
                "Matched-layer network",
                transform=network_axis.transAxes,
                ha="left",
                va="bottom",
                fontsize=10.0,
                fontweight="semibold",
                color=INK,
                clip_on=False,
            )

    legend_handles = [
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor="none",
            edgecolor=MATCH_OUTLINE,
            linewidth=1.2,
            label="Outline = optimal layer match",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=6,
            markerfacecolor=BLUE,
            markeredgecolor="white",
            linestyle="",
            label="Node area = held-out AUC",
        ),
        Line2D(
            [0],
            [0],
            marker="*",
            markersize=7,
            color=PINK_DARK,
            linestyle="",
            label="Star/ring = selected layer",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.995),
        fontsize=9.0,
        columnspacing=1.30,
        handletextpad=0.52,
    )
    save_figure(fig, output_dir / "cross_model_alignment_atlas")


def write_summary(
    layer_metrics: pd.DataFrame,
    random_eval: pd.DataFrame,
    generator_eval: pd.DataFrame,
    representation_frame: pd.DataFrame,
    output_dir: Path,
    language: str,
) -> None:
    language_name = "Chinese" if language == "zh" else "English"
    lines = [
        "# Hidden-state geometry analysis",
        "",
        f"This is an exploratory analysis of 500 {language_name} "
        "proposition-aligned human/AI continuation pairs. Layer maxima below "
        "are descriptive and "
        "must not be reused as confirmatory evidence on the same pairs.",
        "",
        "The within-class nullspace estimator is the primary method throughout; "
        "raw mean difference is retained only as a control.",
        "",
        "## Formal held-out separability",
        "",
    ]
    for model in MODEL_ORDER:
        for representation in REPRESENTATION_ORDER:
            subset = random_eval[
                (random_eval["model"] == model)
                & (random_eval["representation"] == representation)
                & (random_eval["method"] == "nullspace")
            ]
            best = subset.loc[subset["heldout_auc_mean"].idxmax()]
            lines.append(
                f"- {MODEL_LABEL[model]}, "
                f"{REPRESENTATION_LABEL[representation]}: maximum "
                f"random-held-out nullspace AUC "
                f"{best['heldout_auc_mean']:.3f} at layer "
                f"{int(best['layer'])}."
            )
    lines.extend(["", "## Pair geometry", ""])
    for model in MODEL_ORDER:
        subset = layer_metrics[layer_metrics["model"] == model]
        lines.append(
            f"- {MODEL_LABEL[model]}: full-sample nullspace pair agreement spans "
            f"{subset['nullspace_pair_positive_rate'].min():.3f}–"
            f"{subset['nullspace_pair_positive_rate'].max():.3f}; raw-direction bootstrap "
            f"stability lower bounds span "
            f"{subset['bootstrap_direction_cosine_lo'].min():.3f}–"
            f"{subset['bootstrap_direction_cosine_lo'].max():.3f}."
        )
    lines.extend(["", "## Generator robustness", ""])
    nullspace = generator_eval[generator_eval["method"] == "nullspace"]
    for model in MODEL_ORDER:
        subset = nullspace[nullspace["model"] == model]
        lines.append(
            f"- {MODEL_LABEL[model]}: leave-one-generator-out nullspace AUC "
            f"range across all layers and generators is "
            f"{subset['auc'].min():.3f}–{subset['auc'].max():.3f}."
        )
    lines.extend(["", "## Representation agreement", ""])
    for model in MODEL_ORDER:
        subset = representation_frame[representation_frame["model"] == model]
        lines.append(
            f"- {MODEL_LABEL[model]}: last-token vs mean-pooled displacement "
            f"CKA ranges {subset['displacement_cka'].min():.3f}–"
            f"{subset['displacement_cka'].max():.3f}; nullspace pair-score "
            f"Spearman correlation ranges "
            f"{subset['nullspace_score_spearman'].min():.3f}–"
            f"{subset['nullspace_score_spearman'].max():.3f}."
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- Separability, direction stability, and cross-model CKA establish "
            "representation geometry; they do not by themselves identify a "
            "discourse construct.",
            "- Length/plan correlations are diagnostics, not proof that all "
            "semantic-content confounding has been removed.",
            "- CKA compares sample-level relational structure across unequal "
            "hidden spaces; it does not provide a neuron-wise alignment.",
            "- Steering-layer selection must combine these results with a "
            "frozen selection rule and generated-token-only causal tests.",
            "",
        ]
    )
    (output_dir / "analysis_summary.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    global MODEL_ORDER
    args = parse_args()
    MODEL_ORDER = list(args.models)
    unknown_models = [
        model
        for model in MODEL_ORDER
        if model not in MODEL_LABEL or model not in MODEL_SHORT_LABEL
    ]
    if unknown_models:
        raise ValueError(
            f"Missing display labels for models: {unknown_models}"
        )
    configure_style()
    language = (
        "zh"
        if "zh" in args.dataset.resolve().parts
        else "en"
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = args.out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    if args.refresh_alignment_atlas:
        cka_path = args.out_dir / "cka_comparisons.csv"
        random_path = args.out_dir / "random_holdout_metrics.csv"
        generator_path = args.out_dir / "generator_holdout_metrics.csv"
        missing_tables = [
            path
            for path in (cka_path, random_path, generator_path)
            if not path.exists()
        ]
        if missing_tables:
            raise FileNotFoundError(
                "Missing saved tables required for plot-only refresh: "
                + ", ".join(str(path) for path in missing_tables)
            )
        cka_frame = pd.read_csv(cka_path)
        random_eval = pd.read_csv(random_path)
        generator_eval = pd.read_csv(generator_path)
        layer_lookup = {
            (model, representation): sorted(
                random_eval[
                    (random_eval["model"] == model)
                    & (random_eval["representation"] == representation)
                ]["layer"].astype(int).unique()
            )
            for model in MODEL_ORDER
            for representation in REPRESENTATION_ORDER
        }
        robust_layers = robust_layer_lookup(generator_eval)
        plot_cross_model_alignment_atlas(
            cka_frame,
            random_eval,
            layer_lookup,
            robust_layers,
            figures_dir,
        )
        print(
            "Refreshed cross-model alignment atlas from saved tables in "
            f"{args.out_dir}",
            flush=True,
        )
        return
    for stale_figure in [
        *figures_dir.glob("*.png"),
        *figures_dir.glob("*.pdf"),
    ]:
        stale_figure.unlink()
    device = select_device(args.device)

    rows = load_dataset(args.dataset)
    canonical_ids = [row["sample_id"] for row in rows]
    generators = [row["generation"]["generator_model"] for row in rows]
    sources = [row["source"] for row in rows]
    nuisance = nuisance_features(rows)
    inventory = state_inventory(args.states_root, args.dataset_tag)
    missing = [
        (model, representation)
        for model in MODEL_ORDER
        for representation in REPRESENTATION_ORDER
        if (model, representation) not in inventory
    ]
    if missing:
        raise FileNotFoundError(f"Missing state inventories: {missing}")

    random_eval, generator_eval = read_formal_evaluations(
        args.eval_root,
        inventory,
        args.dataset_tag,
    )
    robust_layers = robust_layer_lookup(generator_eval)
    maximum_layer = {
        key: max(layer_from_path(path) for path in paths)
        for key, paths in inventory.items()
    }
    random_eval["relative_depth"] = random_eval.apply(
        lambda row: row["layer"]
        / maximum_layer[(row["model"], row["representation"])],
        axis=1,
    )
    generator_eval["relative_depth"] = generator_eval.apply(
        lambda row: row["layer"]
        / maximum_layer[(row["model"], row["representation"])],
        axis=1,
    )

    layer_rows: list[dict[str, Any]] = []
    nuisance_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    group_details: dict[str, Any] = {}
    grams: dict[tuple[str, str, int, str], torch.Tensor] = {}
    directions: dict[tuple[str, str, int, str], torch.Tensor] = {}
    oof_scores: dict[tuple[str, str, int], np.ndarray] = {}
    nullspace_scores: dict[tuple[str, str, int], np.ndarray] = {}
    spectra: dict[tuple[str, str, int, str], np.ndarray] = {}
    layer_lookup: dict[tuple[str, str], list[int]] = {}
    state_metadata: dict[str, Any] = {}
    endpoint_states: dict[
        tuple[str, str],
        tuple[np.ndarray, np.ndarray, np.ndarray, int],
    ] = {}

    total_files = sum(len(paths) for paths in inventory.values())
    completed = 0
    for model in MODEL_ORDER:
        for representation in REPRESENTATION_ORDER:
            paths = inventory[(model, representation)]
            layer_lookup[(model, representation)] = [
                layer_from_path(path) for path in paths
            ]
            for state_path in paths:
                layer = layer_from_path(state_path)
                human, ai, metadata = load_aligned_state(
                    state_path,
                    canonical_ids,
                    device,
                )
                displacements = ai - human
                angular_displacements = normalize_rows(displacements)
                raw_direction = unit_vector(displacements.mean(dim=0))
                saved_raw = load_saved_direction(state_path, "raw", device)
                nullspace_direction = load_saved_direction(
                    state_path,
                    "nullspace",
                    device,
                )
                if layer == robust_layers[(model, representation)]:
                    endpoint_states[(model, representation)] = (
                        human.detach().cpu().numpy().copy(),
                        ai.detach().cpu().numpy().copy(),
                        nullspace_direction.detach().cpu().numpy().copy(),
                        layer,
                    )
                if float((raw_direction @ saved_raw).item()) < 0.999:
                    raise ValueError(
                        f"Saved raw direction does not match states: {state_path}"
                    )

                alignment = cross_validated_pair_alignment(
                    displacements,
                    n_folds=5,
                    seed=args.seed,
                )
                stability = bootstrap_direction_stability(
                    displacements,
                    n_boot=args.bootstrap,
                    seed=args.seed + layer,
                )
                spectrum = spectral_summary(displacements)
                angular_spectrum = spectral_summary(angular_displacements)
                residual_pc1_scores = (
                    leading_residual_scores(displacements)
                    .detach()
                    .cpu()
                    .numpy()
                )
                angular_pc1_scores = (
                    leading_residual_scores(angular_displacements)
                    .detach()
                    .cpu()
                    .numpy()
                )
                generator_geometry = group_direction_geometry(
                    displacements,
                    generators,
                    n_perm=args.permutations,
                    seed=args.seed + layer,
                )
                source_geometry = group_direction_geometry(
                    displacements,
                    sources,
                    n_perm=args.permutations,
                    seed=args.seed + 10_000 + layer,
                )
                angular_generator_geometry = group_direction_geometry(
                    angular_displacements,
                    generators,
                    n_perm=args.permutations,
                    seed=args.seed + 20_000 + layer,
                )
                angular_source_geometry = group_direction_geometry(
                    angular_displacements,
                    sources,
                    n_perm=args.permutations,
                    seed=args.seed + 30_000 + layer,
                )
                null_cosines = (
                    row_cosines(displacements, nullspace_direction)
                    .detach()
                    .cpu()
                    .numpy()
                )
                null_scores = (
                    (displacements @ nullspace_direction)
                    .detach()
                    .cpu()
                    .numpy()
                )
                pair_norms = (
                    displacements.norm(dim=1).detach().cpu().numpy()
                )
                relative_depth = (
                    layer / maximum_layer[(model, representation)]
                )
                pair_rows.extend(
                    {
                        "model": model,
                        "representation": representation,
                        "layer": layer,
                        "relative_depth": relative_depth,
                        "sample_id": canonical_ids[index],
                        "generator": generators[index],
                        "source": sources[index],
                        "oof_projection": float(alignment.scores[index]),
                        "oof_cosine": float(alignment.cosines[index]),
                        "oof_fold": int(alignment.fold_ids[index]),
                        "nullspace_projection": float(null_scores[index]),
                        "nullspace_cosine": float(null_cosines[index]),
                        "pair_displacement_norm": float(pair_norms[index]),
                        "residual_pc1_score": float(
                            residual_pc1_scores[index]
                        ),
                        "angular_residual_pc1_score": float(
                            angular_pc1_scores[index]
                        ),
                    }
                    for index in range(len(canonical_ids))
                )

                key = (model, representation, layer)
                grams[(*key, "raw")] = (
                    centered_gram(displacements).detach().cpu()
                )
                grams[(*key, "angular")] = (
                    centered_gram(angular_displacements).detach().cpu()
                )
                directions[(*key, "raw")] = raw_direction.detach().cpu()
                directions[(*key, "nullspace")] = (
                    nullspace_direction.detach().cpu()
                )
                oof_scores[key] = alignment.scores
                nullspace_scores[key] = null_scores
                spectra[(*key, "raw")] = spectrum.pop("eigenvalues")
                spectra[(*key, "angular")] = angular_spectrum.pop(
                    "eigenvalues"
                )
                metadata_key = f"{model}/{representation}"
                if metadata_key not in state_metadata:
                    state_metadata[metadata_key] = {
                        "model": metadata.get("model"),
                        "hidden_size": metadata.get("hidden_size"),
                        "pooling": metadata.get("pooling"),
                        "dataset_path": metadata.get("dataset_path"),
                        "dataset_sha256": metadata.get("dataset_sha256"),
                        "layers": layer_lookup[(model, representation)],
                        "human_token_count": numeric_summary(
                            metadata.get("human_continuation_tokens")
                        ),
                        "ai_token_count": numeric_summary(
                            metadata.get("ai_continuation_tokens")
                        ),
                    }
                layer_rows.append(
                    {
                        "model": model,
                        "representation": representation,
                        "layer": layer,
                        "relative_depth": layer
                        / maximum_layer[(model, representation)],
                        "n_pairs": len(displacements),
                        "hidden_size": displacements.shape[1],
                        **alignment.metrics,
                        **stability,
                        **spectrum,
                        **{
                            f"angular_{name}": value
                            for name, value in angular_spectrum.items()
                        },
                        "coherent_displacement_fraction": (
                            coherent_displacement_fraction(displacements)
                        ),
                        "angular_coherent_displacement_fraction": (
                            coherent_displacement_fraction(
                                angular_displacements
                            )
                        ),
                        "mean_pair_displacement_norm": float(
                            pair_norms.mean()
                        ),
                        "median_pair_displacement_norm": float(
                            np.median(pair_norms)
                        ),
                        "p99_to_median_displacement_norm": float(
                            np.quantile(pair_norms, 0.99)
                            / max(np.median(pair_norms), 1e-12)
                        ),
                        "nullspace_vs_raw_cosine": float(
                            (nullspace_direction @ raw_direction).item()
                        ),
                        "nullspace_pair_positive_rate": float(
                            np.mean(null_scores > 0)
                        ),
                        "nullspace_pair_cosine_median": float(
                            np.median(null_cosines)
                        ),
                        "generator_between_fraction": generator_geometry[
                            "between_fraction"
                        ],
                        "generator_permutation_p": generator_geometry[
                            "permutation_p"
                        ],
                        "generator_direction_cosine_mean": generator_geometry[
                            "group_to_global_cosine_mean"
                        ],
                        "generator_direction_cosine_min": generator_geometry[
                            "group_to_global_cosine_min"
                        ],
                        "angular_generator_between_fraction": (
                            angular_generator_geometry["between_fraction"]
                        ),
                        "angular_generator_permutation_p": (
                            angular_generator_geometry["permutation_p"]
                        ),
                        "angular_generator_direction_cosine_mean": (
                            angular_generator_geometry[
                                "group_to_global_cosine_mean"
                            ]
                        ),
                        "angular_generator_direction_cosine_min": (
                            angular_generator_geometry[
                                "group_to_global_cosine_min"
                            ]
                        ),
                        "source_between_fraction": source_geometry[
                            "between_fraction"
                        ],
                        "source_permutation_p": source_geometry[
                            "permutation_p"
                        ],
                        "source_direction_cosine_mean": source_geometry[
                            "group_to_global_cosine_mean"
                        ],
                        "source_direction_cosine_min": source_geometry[
                            "group_to_global_cosine_min"
                        ],
                        "angular_source_between_fraction": (
                            angular_source_geometry["between_fraction"]
                        ),
                        "angular_source_permutation_p": (
                            angular_source_geometry["permutation_p"]
                        ),
                    }
                )
                group_details[
                    f"{model}/{representation}/layer{layer:02d}"
                ] = {
                    "generator": generator_geometry,
                    "source": source_geometry,
                    "angular_generator": angular_generator_geometry,
                    "angular_source": angular_source_geometry,
                }
                for feature, values in nuisance.items():
                    rho, p_value = safe_spearman(alignment.scores, values)
                    nuisance_rows.append(
                        {
                            "model": model,
                            "representation": representation,
                            "layer": layer,
                            "relative_depth": layer
                            / maximum_layer[(model, representation)],
                            "target": "source_direction_projection",
                            "feature": feature,
                            "rho": rho,
                            "p_value": p_value,
                        }
                    )
                    pc1_rho, pc1_p = safe_spearman(
                        residual_pc1_scores,
                        values,
                    )
                    nuisance_rows.append(
                        {
                            "model": model,
                            "representation": representation,
                            "layer": layer,
                            "relative_depth": layer
                            / maximum_layer[(model, representation)],
                            "target": "residual_pc1_raw",
                            "feature": feature,
                            "rho": abs(pc1_rho),
                            "p_value": pc1_p,
                        }
                    )
                    angular_pc1_rho, angular_pc1_p = safe_spearman(
                        angular_pc1_scores,
                        values,
                    )
                    nuisance_rows.append(
                        {
                            "model": model,
                            "representation": representation,
                            "layer": layer,
                            "relative_depth": layer
                            / maximum_layer[(model, representation)],
                            "target": "residual_pc1_angular",
                            "feature": feature,
                            "rho": abs(angular_pc1_rho),
                            "p_value": angular_pc1_p,
                        }
                    )

                completed += 1
                print(
                    f"[{completed:02d}/{total_files}] "
                    f"{model} {representation} L{layer:02d}",
                    flush=True,
                )
                del human, ai, displacements, angular_displacements
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    layer_metrics = pd.DataFrame(layer_rows)
    nuisance_frame = pd.DataFrame(nuisance_rows)
    pair_frame = pd.DataFrame(pair_rows)
    nuisance_frame["q_value_global"] = benjamini_hochberg(
        nuisance_frame["p_value"]
    )

    cka_rows: list[dict[str, Any]] = []
    direction_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        for representation in REPRESENTATION_ORDER:
            layers = layer_lookup[(model, representation)]
            for first_layer in layers:
                for second_layer in layers:
                    first_key = (model, representation, first_layer)
                    second_key = (model, representation, second_layer)
                    for geometry in ("raw", "angular"):
                        cka_rows.append(
                            {
                                "scope": "within_model",
                                "geometry": geometry,
                                "model_a": model,
                                "model_b": model,
                                "representation_a": representation,
                                "representation_b": representation,
                                "layer_a": first_layer,
                                "layer_b": second_layer,
                                "cka": linear_cka_from_grams(
                                    grams[(*first_key, geometry)],
                                    grams[(*second_key, geometry)],
                                ),
                            }
                        )
                    for method in METHOD_ORDER:
                        direction_rows.append(
                            {
                                "scope": "within_model",
                                "model_a": model,
                                "model_b": model,
                                "representation_a": representation,
                                "representation_b": representation,
                                "layer_a": first_layer,
                                "layer_b": second_layer,
                                "method": method,
                                "cosine": float(
                                    (
                                        directions[(*first_key, method)]
                                        @ directions[(*second_key, method)]
                                    ).item()
                                ),
                            }
                        )

    model_pairs = [
        (MODEL_ORDER[first], MODEL_ORDER[second])
        for first in range(len(MODEL_ORDER))
        for second in range(first + 1, len(MODEL_ORDER))
    ]
    for representation in REPRESENTATION_ORDER:
        for first_model, second_model in model_pairs:
            for first_layer in layer_lookup[(first_model, representation)]:
                for second_layer in layer_lookup[(second_model, representation)]:
                    first_key = (first_model, representation, first_layer)
                    second_key = (second_model, representation, second_layer)
                    for geometry in ("raw", "angular"):
                        cka_rows.append(
                            {
                                "scope": "cross_model",
                                "geometry": geometry,
                                "model_a": first_model,
                                "model_b": second_model,
                                "representation_a": representation,
                                "representation_b": representation,
                                "layer_a": first_layer,
                                "layer_b": second_layer,
                                "cka": linear_cka_from_grams(
                                    grams[(*first_key, geometry)],
                                    grams[(*second_key, geometry)],
                                ),
                            }
                        )

    representation_rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        shared_layers = sorted(
            set(layer_lookup[(model, "last_token")])
            & set(layer_lookup[(model, "mean_pool")])
        )
        for layer in shared_layers:
            last_key = (model, "last_token", layer)
            mean_key = (model, "mean_pool", layer)
            raw_score_rho, raw_score_p = safe_spearman(
                oof_scores[last_key],
                oof_scores[mean_key],
            )
            nullspace_score_rho, nullspace_score_p = safe_spearman(
                nullspace_scores[last_key],
                nullspace_scores[mean_key],
            )
            raw_displacement_cka = linear_cka_from_grams(
                grams[(*last_key, "raw")],
                grams[(*mean_key, "raw")],
            )
            angular_displacement_cka = linear_cka_from_grams(
                grams[(*last_key, "angular")],
                grams[(*mean_key, "angular")],
            )
            row = {
                "model": model,
                "layer": layer,
                "relative_depth": layer
                / maximum_layer[(model, "last_token")],
                "raw_direction_cosine": float(
                    (
                        directions[(*last_key, "raw")]
                        @ directions[(*mean_key, "raw")]
                    ).item()
                ),
                "nullspace_direction_cosine": float(
                    (
                        directions[(*last_key, "nullspace")]
                        @ directions[(*mean_key, "nullspace")]
                    ).item()
                ),
                "raw_displacement_cka": raw_displacement_cka,
                "angular_displacement_cka": angular_displacement_cka,
                "displacement_cka": angular_displacement_cka,
                "nullspace_score_spearman": nullspace_score_rho,
                "nullspace_score_spearman_p": nullspace_score_p,
                "raw_oof_score_spearman": raw_score_rho,
                "raw_oof_score_spearman_p": raw_score_p,
            }
            representation_rows.append(row)
            for geometry, value in (
                ("raw", raw_displacement_cka),
                ("angular", angular_displacement_cka),
            ):
                cka_rows.append(
                    {
                        "scope": "cross_representation",
                        "geometry": geometry,
                        "model_a": model,
                        "model_b": model,
                        "representation_a": "last_token",
                        "representation_b": "mean_pool",
                        "layer_a": layer,
                        "layer_b": layer,
                        "cka": value,
                    }
                )
            for method in METHOD_ORDER:
                direction_rows.append(
                    {
                        "scope": "cross_representation",
                        "model_a": model,
                        "model_b": model,
                        "representation_a": "last_token",
                        "representation_b": "mean_pool",
                        "layer_a": layer,
                        "layer_b": layer,
                        "method": method,
                        "cosine": row[f"{method}_direction_cosine"],
                    }
                )

    cka_frame = pd.DataFrame(cka_rows)
    direction_frame = pd.DataFrame(direction_rows)
    representation_frame = pd.DataFrame(representation_rows)

    layer_metrics.to_csv(args.out_dir / "layer_geometry_metrics.csv", index=False)
    random_eval.to_csv(args.out_dir / "random_holdout_metrics.csv", index=False)
    generator_eval.to_csv(
        args.out_dir / "generator_holdout_metrics.csv",
        index=False,
    )
    nuisance_frame.to_csv(
        args.out_dir / "nuisance_correlations.csv",
        index=False,
    )
    pair_frame.to_csv(
        args.out_dir / "pair_geometry_scores.csv",
        index=False,
    )
    cka_frame.to_csv(args.out_dir / "cka_comparisons.csv", index=False)
    direction_frame.to_csv(
        args.out_dir / "direction_cosines.csv",
        index=False,
    )
    representation_frame.to_csv(
        args.out_dir / "representation_agreement.csv",
        index=False,
    )
    (args.out_dir / "group_geometry_details.json").write_text(
        json.dumps(group_details, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    plot_paired_endpoint_flow_fields(
        endpoint_states,
        layer_metrics,
        random_eval,
        generator_eval,
        figures_dir,
        args.seed,
    )
    plot_layer_selection_robustness_atlas(
        layer_metrics,
        random_eval,
        generator_eval,
        nuisance_frame,
        robust_layers,
        figures_dir,
    )
    plot_cross_model_alignment_atlas(
        cka_frame,
        random_eval,
        layer_lookup,
        robust_layers,
        figures_dir,
    )
    plot_pair_concordance_cartography(
        pair_frame,
        robust_layers,
        figures_dir,
    )
    write_summary(
        layer_metrics,
        random_eval,
        generator_eval,
        representation_frame,
        args.out_dir,
        language,
    )

    manifest = {
        "analysis": "paired_hidden_state_geometry",
        "dataset": str(args.dataset),
        "dataset_sha256": file_sha256(args.dataset),
        "language": language,
        "n_pairs": len(rows),
        "models": MODEL_ORDER,
        "representations": REPRESENTATION_ORDER,
        "methods": METHOD_ORDER,
        "primary_method": PRIMARY_METHOD,
        "control_methods": CONTROL_METHODS,
        "seed": args.seed,
        "bootstrap_replicates": args.bootstrap,
        "permutation_replicates": args.permutations,
        "device": str(device),
        "layer_convention": "decoder_block_output_1_based",
        "paper_figure_set": [
            "paired_endpoint_flow_fields",
            "layer_selection_robustness_atlas",
            "cross_model_alignment_atlas",
            "pair_concordance_cartography",
        ],
        "definitions": {
            "paired_displacement": "AI hidden state minus matched human hidden state",
            "pair_agreement": "fraction of held-out pair displacements with positive projection on a raw direction fitted on the other folds",
            "nullspace_pair_agreement": "fraction of pair displacements with positive projection on the full-sample within-class nullspace direction; used for exploratory geometry only",
            "coherent_displacement_fraction": "squared norm of the mean pair displacement divided by mean squared pair-displacement norm",
            "angular_geometry": "the same analysis after L2-normalizing every paired displacement, removing pair-norm differences",
            "effective_rank": "entropy effective rank of the centered pair-displacement covariance",
            "cka": "linear centered-kernel alignment of centered pair-displacement sample Gram matrices",
            "group_between_fraction": "one-way ANOVA between-group energy divided by total centered pair-displacement energy",
        },
        "state_metadata": state_metadata,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "matplotlib": matplotlib.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
    }
    (args.out_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Saved hidden-state analysis to {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
