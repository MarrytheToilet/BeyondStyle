"""Visualization module for BeyondStyle analysis."""

from .palette import setup_style, PALETTE, save_fig
from .paired_plots import (
    plot_projection_distributions,
    plot_pca_embedding,
    plot_direction_similarity,
    plot_embedding_shift,
)
from .steering_plots import (
    plot_layer_metrics,
    plot_roc_curves,
    plot_separation_landscape,
)
from .ablation_plots import (
    plot_denoise_comparison,
    plot_projection_violins,
)
from .interp_plots import (
    plot_all_interpretability,
)

__all__ = [
    "setup_style", "PALETTE", "save_fig",
    # Part A - paired analysis
    "plot_projection_distributions",
    "plot_pca_embedding",
    "plot_direction_similarity",
    "plot_embedding_shift",
    # Part B - steering results
    "plot_layer_metrics",
    "plot_roc_curves",
    "plot_separation_landscape",
    # Part C - denoising comparison
    "plot_denoise_comparison",
    "plot_projection_violins",
    # Part D - interpretability
    "plot_all_interpretability",
]
