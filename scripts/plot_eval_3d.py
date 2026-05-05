#!/usr/bin/env python3
"""Visualisations for the lm-eval steering grid.

Generates three figure groups under <eval_dir>/figures/:

  1. eval3d_raw/  — 3D bar charts (one per denoise × layer, raw scores)
  2. eval3d_norm/ — 3D bar charts (normalised: z-scored per task)
  3. heatmap/     — α-trend heatmaps per (denoise × layer) showing how each
                    task responds to steering strength (Δ vs baseline,
                    rows clustered by trend signature).

Palette: shared pastel pink (AI) + pastel blue (Human).

Usage:
    python scripts/plot_eval_3d.py \
        --eval-dir outputs/en/04_lmeval/Llama-3.1-8B-Instruct
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.cluster.hierarchy import linkage, leaves_list

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.visualization.palette import (PALETTE, H, A, H_D, A_D, BG, setup_style)

TASK_PRIMARY_METRIC = {
    "mmlu": "acc",
    "hellaswag": "acc_norm",
    "arc_easy": "acc",
    "arc_challenge": "acc_norm",
    "openbookqa": "acc_norm",
    "winogrande": "acc",
    "piqa": "acc_norm",
    "boolq": "acc",
    "race": "acc",
    "wikitext": "word_perplexity",
    "truthfulqa_mc2": "acc",
}
LOWER_IS_BETTER = {"wikitext"}

FILE_RE = re.compile(
    r"^(?P<denoise>raw|nullspace|fisher)_L(?P<layer>\d+)_a(?P<alpha>[+-]?[0-9.]+)\.json$"
)

CMAP_HEAT = LinearSegmentedColormap.from_list("hp_heat", [H_D, H, "#F4F6FA", A, A_D])


# ── data loading ──────────────────────────────────────────
def normalise_metric(task: str, value: float | None) -> float:
    if value is None:
        return np.nan
    if task in LOWER_IS_BETTER:
        return 1.0 / max(value, 1e-6)
    return value


def load_score(path: Path, task: str) -> float | None:
    data = json.loads(path.read_text())
    s = data.get("scores", {}).get(task)
    if not s:
        return None
    metric = TASK_PRIMARY_METRIC.get(task, "acc")
    return s.get(metric)


def discover_grid(eval_dir: Path):
    grid: dict = {}
    for f in eval_dir.glob("*.json"):
        m = FILE_RE.match(f.name)
        if not m:
            continue
        grid.setdefault(m["denoise"], {}).setdefault(int(m["layer"]), {})[float(m["alpha"])] = f
    return grid


# ── 3D bar plot ──────────────────────────────────────────
def _pin_wikitext_last(order: list[int], tasks: list[str]) -> list[int]:
    """Move any wikitext-like task to the end of the ordering."""
    pinned = [i for i in order if "wikitext" not in tasks[i].lower()]
    tail = [i for i in order if "wikitext" in tasks[i].lower()]
    return pinned + tail


def _build_matrix(alpha_files, baseline_path, tasks, normalise, task_order=None):
    alphas_full = sorted(set(list(alpha_files) + [0.0]))
    n_alphas = len(alphas_full)
    scores = np.full((len(tasks), n_alphas), np.nan)
    for j, a in enumerate(alphas_full):
        path = baseline_path if a == 0.0 else alpha_files[a]
        for i, t in enumerate(tasks):
            v = load_score(path, t)
            scores[i, j] = normalise_metric(t, v)
    if normalise:
        mn = np.nanmin(scores, axis=1, keepdims=True)
        mx = np.nanmax(scores, axis=1, keepdims=True)
        rng = np.where((mx - mn) == 0, 1.0, mx - mn)
        scores_plot = (scores - mn) / rng
        z_label = "min–max normalised  [0, 1]"
        z_top = 1.05
    else:
        scores_plot = scores
        z_label = "score (higher = better)"
        z_top = float(np.nanmax(scores_plot)) * 1.10 if np.isfinite(scores_plot).any() else 1.0
    if task_order is None:
        order = np.argsort(np.nanmean(scores_plot, axis=1))
    else:
        order = np.asarray(task_order)
    return alphas_full, scores_plot[order], [tasks[i] for i in order], z_label, z_top


def _draw_3d_on(ax, alphas_full, scores_plot, tasks_sorted, z_label, z_top,
                title: str):
    """Render bars on a 3D Axes with proper back-to-front order to avoid z-clipping."""
    ax.set_facecolor(BG)
    n_alphas = len(alphas_full)
    n_tasks = len(tasks_sorted)
    dx, dy = 0.62, 0.62

    # View angle: camera in -y, +x quadrant. To prevent see-through clipping,
    # iterate from FAR to NEAR (high y → low y, low x → high x).
    elev, azim = 22, -55
    ax.view_init(elev=elev, azim=azim)

    y_order = range(n_tasks - 1, -1, -1)   # back rows (high y) first
    x_order = range(n_alphas)              # left columns (low x) first

    for i in y_order:
        for j in x_order:
            z = scores_plot[i, j]
            if np.isnan(z) or z <= 0:
                # still draw a tiny stub so axis ticks line up
                z = max(z, 1e-6) if not np.isnan(z) else 0
                if z == 0:
                    continue
            a = alphas_full[j]
            if a == 0.0:
                color, edge, bar_alpha = "#E8ECF2", "#B6BCC4", 0.90
            elif a > 0:
                color, edge = A, A_D
                bar_alpha = 0.65 if abs(a) >= 1.5 else 0.45
            else:
                color, edge = H, H_D
                bar_alpha = 0.65 if abs(a) >= 1.5 else 0.45
            ax.bar3d(j - dx / 2, i - dy / 2, 0, dx, dy, z,
                     color=color, edgecolor=edge, linewidth=0.5,
                     alpha=bar_alpha, shade=True)

    ax.set_xticks(range(n_alphas))
    ax.set_xticklabels([f"{a:+g}" if a != 0 else "base" for a in alphas_full],
                       fontsize=8.5)
    ax.set_yticks(range(n_tasks))
    ax.set_yticklabels(tasks_sorted, fontsize=8)
    ax.set_xlabel("α", fontsize=9, labelpad=4)
    ax.set_ylabel("task (low → high)", fontsize=8.5, labelpad=4)
    ax.set_zlabel(z_label, fontsize=8.5, labelpad=2)
    ax.set_zlim(0, z_top)
    ax.set_title(title, fontsize=11, weight="bold", pad=8, color="#3B3B3B")

    ax.grid(True, color=PALETTE["grid"], linewidth=0.4)
    ax.xaxis.pane.set_facecolor(BG)
    ax.yaxis.pane.set_facecolor(BG)
    ax.zaxis.pane.set_facecolor("white")
    ax.tick_params(axis="both", colors="#555555", pad=1)


def plot_3d_row(denoise: str, layer_files: dict[int, dict[float, Path]],
                baseline_path: Path, tasks: list[str], out_path: Path,
                normalise: bool):
    """One row of 3 subplots for a single denoise method, layers low→high.
    Task ordering is shared across all panels (mean across all layers)."""
    layers = sorted(layer_files)

    accum = np.zeros(len(tasks))
    counts = np.zeros(len(tasks))
    for layer in layers:
        af = layer_files[layer]
        alphas_full = sorted(set(list(af) + [0.0]))
        for a in alphas_full:
            path = baseline_path if a == 0.0 else af[a]
            for i, t in enumerate(tasks):
                v = normalise_metric(t, load_score(path, t))
                if not np.isnan(v):
                    accum[i] += v
                    counts[i] += 1
    counts[counts == 0] = 1
    shared_order = np.argsort(accum / counts).tolist()
    shared_order = _pin_wikitext_last(shared_order, tasks)

    fig = plt.figure(figsize=(6.0 * len(layers), 6.5), facecolor="white")
    for k, layer in enumerate(layers):
        alphas_full, scores_plot, tasks_sorted, z_label, z_top = _build_matrix(
            layer_files[layer], baseline_path, tasks, normalise,
            task_order=shared_order)
        ax = fig.add_subplot(1, len(layers), k + 1, projection="3d")
        title = f"Layer {layer}"
        _draw_3d_on(ax, alphas_full, scores_plot, tasks_sorted,
                    z_label, z_top, title)

    suffix = "  ·  min–max normalised" if normalise else ""
    fig.suptitle(f"{denoise.upper()}{suffix}",
                 fontsize=15, weight="bold", color="#3B3B3B", y=0.99)

    legend = [
        Patch(facecolor=H, edgecolor=H_D, label="α<0  toward Human"),
        Patch(facecolor="#E8ECF2", edgecolor="#B6BCC4", label="α=0  baseline"),
        Patch(facecolor=A, edgecolor=A_D, label="α>0  toward AI"),
    ]
    fig.legend(handles=legend, loc="lower center",
               bbox_to_anchor=(0.5, -0.01), ncol=3, fontsize=10,
               framealpha=0.9, edgecolor="#CCCCCC")

    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.05,
                        wspace=0.10)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── heatmap ──────────────────────────────────────────────
def _build_delta(alpha_files, baseline_path, tasks):
    alphas = sorted(a for a in alpha_files if a != 0.0)
    base = np.array([normalise_metric(t, load_score(baseline_path, t)) for t in tasks])
    delta = np.full((len(tasks), len(alphas)), np.nan)
    for j, a in enumerate(alphas):
        for i, t in enumerate(tasks):
            v = normalise_metric(t, load_score(alpha_files[a], t))
            if not np.isnan(v) and not np.isnan(base[i]) and base[i] != 0:
                delta[i, j] = (v - base[i]) / abs(base[i]) * 100.0
    return alphas, delta


def plot_heatmap_row(denoise: str, layer_files: dict[int, dict[float, Path]],
                     baseline_path: Path, tasks: list[str], out_path: Path):
    layers = sorted(layer_files)

    per_layer_delta = {}
    vmax_global = 0.0
    alphas_ref = None
    for layer in layers:
        alphas, delta = _build_delta(layer_files[layer], baseline_path, tasks)
        per_layer_delta[layer] = (alphas, delta)
        alphas_ref = alphas
        if np.isfinite(delta).any():
            vmax_global = max(vmax_global, float(np.nanmax(np.abs(delta))))

    # shared row order: cluster on stacked deltas across all layers
    stacked = np.concatenate([per_layer_delta[l][1] for l in layers], axis=1)
    valid = ~np.isnan(stacked).any(axis=1)
    shared_order = np.arange(len(tasks))
    if valid.sum() >= 2:
        sub = stacked[valid]
        try:
            Z = linkage(sub, method="average", metric="correlation")
            sub_order = leaves_list(Z)
        except Exception:
            mid = sub.shape[1] // 2
            sub_order = np.argsort(np.nanmean(sub[:, mid:], axis=1)
                                   - np.nanmean(sub[:, :mid], axis=1))
        valid_idx = np.where(valid)[0][sub_order]
        invalid_idx = np.where(~valid)[0]
        shared_order = np.concatenate([valid_idx, invalid_idx])
    shared_order = np.array(_pin_wikitext_last(list(shared_order), tasks))

    panels = []
    for layer in layers:
        alphas, delta = per_layer_delta[layer]
        delta = delta[shared_order]
        tasks_sorted = [tasks[i] for i in shared_order]
        panels.append((layer, alphas, delta, tasks_sorted))
    vmax_global = max(vmax_global, 1.0)

    fig, axes = plt.subplots(1, len(layers),
                             figsize=(6.2 * len(layers), 0.62 * len(tasks) + 3.6),
                             facecolor="white")
    if len(layers) == 1:
        axes = [axes]

    im = None
    for k, (ax, (layer, alphas, delta, tasks_sorted)) in enumerate(zip(axes, panels)):
        ax.set_facecolor(BG)
        im = ax.imshow(delta, aspect="auto", cmap=CMAP_HEAT,
                       vmin=-vmax_global, vmax=vmax_global, interpolation="nearest")
        for i in range(delta.shape[0]):
            for j in range(delta.shape[1]):
                v = delta[i, j]
                if np.isnan(v):
                    continue
                txt_color = "#3B3B3B" if abs(v) < vmax_global * 0.6 else "white"
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                        fontsize=15, color=txt_color, weight="medium")
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([f"{a:+g}" for a in alphas], fontsize=17)
        ax.set_yticks(range(len(tasks_sorted)))
        if k == 0:
            ax.set_yticklabels(tasks_sorted, fontsize=16)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("α", fontsize=18, labelpad=6)
        ax.set_title(f"Layer {layer}", fontsize=20, weight="bold", color="#3B3B3B", pad=10)
        for spine in ax.spines.values():
            spine.set_color("#CCCCCC")
        ax.tick_params(colors="#555555", length=0)

    fig.suptitle(f"{denoise.upper()}  ·  α-response heatmap",
                 fontsize=22, weight="bold", color="#3B3B3B", y=1.02)
    fig.subplots_adjust(wspace=0.08)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("% change vs baseline", fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ── main ─────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-dir", type=Path, required=True)
    args = p.parse_args()

    setup_style()
    eval_dir = args.eval_dir
    baseline_path = eval_dir / "baseline.json"
    if not baseline_path.is_file():
        raise SystemExit(f"baseline.json missing in {eval_dir}")

    tasks = json.loads(baseline_path.read_text())["tasks"]
    grid = discover_grid(eval_dir)

    fig_root = eval_dir / "figures"
    raw_dir = fig_root / "eval3d_raw"
    norm_dir = fig_root / "eval3d_norm"
    heat_dir = fig_root / "heatmap"

    n = 0
    for denoise in ["raw", "nullspace", "fisher"]:
        if denoise not in grid:
            continue
        layer_files = grid[denoise]
        plot_3d_row(denoise, layer_files, baseline_path, tasks,
                    raw_dir / f"eval3d_{denoise}.png", normalise=False)
        plot_3d_row(denoise, layer_files, baseline_path, tasks,
                    norm_dir / f"eval3d_{denoise}_norm.png", normalise=True)
        plot_heatmap_row(denoise, layer_files, baseline_path, tasks,
                         heat_dir / f"heatmap_{denoise}.png")
        print(f"  ✓ {denoise}: 3D raw row + 3D norm row + heatmap row")
        n += 1

    print(f"\n✅ {n} method rows × 3 plots written under {fig_root}/")


if __name__ == "__main__":
    main()
