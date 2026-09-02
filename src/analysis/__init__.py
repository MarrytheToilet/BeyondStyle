"""Quantitative analyses of paired hidden-state geometry."""

from .causal_pullback import (
    finite_null_p_value,
    nullspace_pullback_consensus,
    normalized_curve_area,
    weighted_gradient_mean,
)
from .hidden_geometry import (
    bootstrap_direction_stability,
    centered_gram,
    cross_validated_pair_alignment,
    effective_rank,
    group_direction_geometry,
    leading_residual_scores,
    linear_cka_from_grams,
    normalize_rows,
    participation_ratio,
)
from .hidden_propagation import (
    normalized_position_slices,
    orthogonal_random_directions,
    summarize_counterfactual_states,
)
from .projection_mapping import (
    fit_nullspace_source_direction,
    fit_nuisance_residualizer,
    fit_ridge_projection_map,
    fit_within_class_basis,
    prepare_mapping_split,
    project_out_basis,
    project_source_coordinate,
    projection_mapping_metrics,
)
from .token_projection import (
    empirical_vocabulary_field,
    layer_projection_summary,
    pair_balanced_standardize,
    paired_token_context_contrasts,
    position_projection_trajectory,
)

__all__ = [
    "finite_null_p_value",
    "nullspace_pullback_consensus",
    "normalized_curve_area",
    "weighted_gradient_mean",
    "bootstrap_direction_stability",
    "centered_gram",
    "cross_validated_pair_alignment",
    "effective_rank",
    "group_direction_geometry",
    "leading_residual_scores",
    "linear_cka_from_grams",
    "normalize_rows",
    "participation_ratio",
    "normalized_position_slices",
    "orthogonal_random_directions",
    "summarize_counterfactual_states",
    "fit_nullspace_source_direction",
    "fit_nuisance_residualizer",
    "fit_ridge_projection_map",
    "fit_within_class_basis",
    "prepare_mapping_split",
    "project_out_basis",
    "project_source_coordinate",
    "projection_mapping_metrics",
    "empirical_vocabulary_field",
    "layer_projection_summary",
    "pair_balanced_standardize",
    "paired_token_context_contrasts",
    "position_projection_trajectory",
]
