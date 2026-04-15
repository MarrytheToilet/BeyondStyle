#!/usr/bin/env python3
"""Generate all BeyondStyle analysis figures.

Usage:
    python scripts/run_analysis.py                         # auto-detect from .env
    python scripts/run_analysis.py --dataset outputs/dataset/paired_dataset_Qwen3.5-397B-A17B.json
    python scripts/run_analysis.py --fig-dir my_figs       # custom figure output
"""

import argparse
import json
import sys
from pathlib import Path
from scipy import stats
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import get_model_path, get_model_id, get_steering_dir, get_figures_dir, get_dataset_path, extract_dataset_tag
from src.extraction import PairedHiddenStates
from src.steering import MeanDiffSteering
from src.visualization import (
    setup_style,
    plot_projection_distributions,
    plot_pca_embedding,
    plot_direction_similarity,
    plot_embedding_shift,
    plot_layer_metrics,
    plot_roc_curves,
    plot_separation_landscape,
    plot_denoise_comparison,
    plot_projection_violins,
)


def load_data(steer_dir: Path):
    """Load metrics, states, and trained models."""
    with open(steer_dir / "layer_metrics.json") as f:
        metrics = json.load(f)

    # Discover all method keys (e.g. mean_diff, mean_diff_pca, ...)
    method_keys = list(metrics["methods"].keys())
    base_key = "mean_diff" if "mean_diff" in method_keys else method_keys[0]

    layers = sorted(int(k) for k in metrics["methods"][base_key])
    states = {}
    for ly in layers:
        states[ly] = PairedHiddenStates.load(steer_dir / "states" / f"paired_states_layer{ly:02d}.pt")

    # Load all available models
    all_models = {}  # {method_key: {layer: model}}
    for mk in method_keys:
        models = {}
        for ly in layers:
            model_path = steer_dir / "models" / f"{mk}_layer{ly:02d}.pt"
            if model_path.exists():
                models[ly] = MeanDiffSteering.load(model_path)
        if models:
            all_models[mk] = models

    # For backward compatibility, also return the base models
    md_models = all_models.get(base_key, {})

    return metrics, layers, states, md_models, all_models, method_keys


def _best_layer(metrics, layers):
    """Select best layer by normalized hull centroid distance, AUC fallback."""
    md = metrics["methods"]["mean_diff"]
    if md[str(layers[0])].get("hull_centroid_dist_norm") is not None:
        return max(layers, key=lambda ly: md[str(ly)].get("hull_centroid_dist_norm", 0))
    return max(layers, key=lambda ly: md[str(ly)]["auc"])


def print_stats_table(states, md_models, layers):
    """Print statistical significance table."""
    rows = []
    for ly in layers:
        ps = states[ly]
        v = md_models[ly].get_direction().cpu().float()
        hp = (ps.human.float() @ v).numpy()
        ap = (ps.ai.float() @ v).numpy()
        t, p = stats.ttest_ind(hp, ap)
        pooled = np.sqrt((hp.std() ** 2 + ap.std() ** 2) / 2)
        d = (ap.mean() - hp.mean()) / (pooled + 1e-10)
        size = "huge" if abs(d) > 2 else "large" if abs(d) > 0.8 else "medium" if abs(d) > 0.5 else "small"
        rows.append((ly, t, p, d, size))

    print(f"\n{'Layer':>6}  {'t-stat':>8}  {'p-value':>12}  {'Cohen d':>8}  {'Effect':>8}")
    print("-" * 50)
    for ly, t, p, d, size in rows:
        print(f"{ly:>6}  {t:>8.2f}  {p:>12.2e}  {d:>8.2f}  {size:>8}")


def print_summary(metrics, states, layers, model_id=None):
    """Print summary of best results."""
    best_ly = _best_layer(metrics, layers)
    bm = metrics["methods"]["mean_diff"][str(best_ly)]

    model_label = model_id or "unknown"

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Dataset     : {states[layers[0]].n_samples} paired samples")
    print(f"  Model       : {model_label}")
    print(f"  Best layer  : {best_ly}  (by normalized hull centroid distance)")
    print()
    print(f"  {'Norm Hull Dist':20} {bm.get('hull_centroid_dist_norm',0):>10.3f}")
    print(f"  {'Hull Centroid Dist':20} {bm.get('hull_centroid_dist',0):>10.2f}")
    print(f"  {'Fisher Ratio':20} {bm.get('fisher_ratio',0):>10.2f}")
    print(f"  {'AUC':20} {bm['auc']:>10.3f}")
    print(f"  {'Accuracy':20} {bm['accuracy']:>9.1%}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Generate BeyondStyle analysis figures")
    parser.add_argument("--model", type=str, default=None,
                        help="Model path (default: from BEYONDSTYLE_MODEL_PATH in .env)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset path — auto-derives steer-dir and fig-dir from filename "
                             "(default: from BEYONDSTYLE_DATASET in .env, or auto-detect)")
    parser.add_argument("--steer-dir", type=Path, default=None,
                        help="Directory with steering artifacts (overrides --dataset)")
    parser.add_argument("--fig-dir", type=Path, default=None,
                        help="Output directory for figures (overrides --dataset)")
    args = parser.parse_args()

    # Resolve model identity
    model_path = args.model or get_model_path()
    model_id = get_model_id(model_path)

    # Derive dataset tag: CLI > env > auto-detect
    dataset = args.dataset or get_dataset_path()
    dataset_tag = extract_dataset_tag(dataset)

    # Resolve directories: explicit CLI > --dataset auto > bare model_id
    steer_dir = args.steer_dir or get_steering_dir(model_id, dataset_tag=dataset_tag)
    fig_dir = args.fig_dir or get_figures_dir(model_id, dataset_tag=dataset_tag)

    setup_style()

    print(f"Model: {model_id}")
    print("Loading data …")
    metrics, layers, states, md_models, all_models, method_keys = load_data(steer_dir)
    best_ly = _best_layer(metrics, layers)
    print(f"  {len(layers)} layers  |  {states[layers[0]].n_samples} samples  |  best = L{best_ly} (hull centroid dist)")
    print(f"  Methods: {method_keys}")

    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Part A: Paired Data ──
    print("\n── Part A: Paired Data Analysis ──")
    plot_projection_distributions(states, md_models, metrics, layers, fig_dir=fig_dir)
    plot_pca_embedding(states, md_models, best_ly, fig_dir=fig_dir)
    plot_direction_similarity(md_models, layers, fig_dir=fig_dir)
    plot_embedding_shift(states, md_models, best_ly, fig_dir=fig_dir)
    print_stats_table(states, md_models, layers)

    # ── Part B: Steering Results ──
    print("\n── Part B: Steering Results ──")
    plot_layer_metrics(metrics, layers, fig_dir=fig_dir)
    plot_roc_curves(states, md_models, metrics, layers, fig_dir=fig_dir)
    plot_separation_landscape(states, md_models, layers, fig_dir=fig_dir)

    # ── Part C: Denoising Comparison (if multiple methods) ──
    has_ablation = len(method_keys) > 1

    if has_ablation:
        print("\n── Part C: Denoising Comparison ──")
        plot_denoise_comparison(metrics, layers, method_keys=method_keys, fig_dir=fig_dir)

        # Violin plot: projection distributions per method at best layer
        print("  Building projection violin data …")
        ps_best = states[best_ly]
        h = ps_best.human.float()
        a = ps_best.ai.float()
        violin_data = {}
        for mk in method_keys:
            model = all_models.get(mk, {}).get(best_ly)
            if model is None:
                continue
            v = model.get_direction().cpu().float()
            hp = (h @ v).numpy()
            ap = (a @ v).numpy()
            violin_data[mk] = (v, hp, ap)
        if violin_data:
            plot_projection_violins(violin_data, method_keys=method_keys, fig_dir=fig_dir)

    print_summary(metrics, states, layers, model_id=model_id)
    import matplotlib.pyplot as plt
    plt.close("all")
    print(f"\n✅ All figures saved to {fig_dir}")


if __name__ == "__main__":
    main()
