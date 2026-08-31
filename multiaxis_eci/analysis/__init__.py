"""Post-fit analysis, split by concern.

- stats: ECI transform + SOTA / all-models / timeline / human / forest tables
- rotation: factor identification (canonicalization, per-draw alignment, rotations)
- convergence: identified r-hat per MIRT family
- factors: trace introspection + loading/score tables
- timelines: release-date frames for abilities and difficulties
- forecast: AI-frontier vs human-tier crossover projection
- fitview: FitView + prepare_fit, the single rotation/identity contract

The package namespace re-exports the full public API so call sites keep
`from analysis import ...`.
"""
from multiaxis_eci.analysis.convergence import (
    mirt_identified_ess,
    mirt_identified_rhat,
    mirt_identified_rhat_interaction,
    mirt_identified_rhat_nc,
    mirt_identified_rhat_sparse,
    nc_difficulty_draws,
)
from multiaxis_eci.analysis.factors import (
    factor_scores_df,
    loadings_table,
    mirt_factors_from_trace,
    tau_spectrum_df,
    trace_anchors,
    trace_axis_names,
    trace_loading_prior,
)
from multiaxis_eci.analysis.fitspec import (
    FLAGSHIP,
    FLAGSHIP_MAJORITY_CHAINS,
    FLAGSHIP_THIN,
    FLAGSHIP_TRACE,
    FitSpec,
    open_flagship,
    spec_json,
)
from multiaxis_eci.analysis.fitview import FitView, prepare_fit
from multiaxis_eci.analysis.forecast import ForecastResult, mirt_crossover_df, mirt_frontier_forecast
from multiaxis_eci.analysis.rotation import (
    AlignResult,
    _aligned_reproducibility,
    align_factor_signs,
    align_rotations,
    alignment_report,
    apply_rotation,
    canonicalize_factors,
    crosschain_axis_reproducibility,
    permutation_matched_reproducibility,
    factor_corr_df,
    geomin_rotate,
    promax_rotate,
)
from multiaxis_eci.analysis.stats import (
    ECITransform,
    _release_dates,
    all_models_stats_df,
    capability_draws,
    eci_affine,
    eci_transform,
    flat_C,
    forest_stats_df,
    forest_stats_from_draws,
    human_stats_df,
    post_stats,
    sota_stats_df,
    timeline_stats_df,
)
from multiaxis_eci.analysis.timelines import (
    loadings_forest_df,
    mirt_difficulty_timeline_df,
    mirt_human_axis_stats,
    mirt_informed_mask,
    mirt_model_timeline_df,
    nc_difficulty_timeline_df,
)

__all__ = [
    "ECITransform", "post_stats", "capability_draws", "flat_C",
    "eci_transform", "eci_affine", "_release_dates",
    "sota_stats_df", "all_models_stats_df", "timeline_stats_df",
    "human_stats_df", "forest_stats_df", "forest_stats_from_draws",
    "canonicalize_factors", "align_factor_signs", "AlignResult",
    "align_rotations", "_aligned_reproducibility", "alignment_report",
    "crosschain_axis_reproducibility", "permutation_matched_reproducibility",
    "promax_rotate", "geomin_rotate",
    "apply_rotation", "factor_corr_df",
    "mirt_identified_ess",
    "mirt_identified_rhat", "mirt_identified_rhat_interaction",
    "mirt_identified_rhat_nc", "mirt_identified_rhat_sparse",
    "nc_difficulty_draws",
    "trace_anchors", "trace_axis_names", "trace_loading_prior",
    "mirt_factors_from_trace", "tau_spectrum_df", "loadings_table",
    "factor_scores_df",
    "mirt_informed_mask", "mirt_model_timeline_df",
    "mirt_difficulty_timeline_df", "mirt_human_axis_stats",
    "loadings_forest_df", "nc_difficulty_timeline_df",
    "ForecastResult", "mirt_frontier_forecast", "mirt_crossover_df",
    "FitView", "prepare_fit",
    "FitSpec", "spec_json", "FLAGSHIP", "FLAGSHIP_MAJORITY_CHAINS",
    "FLAGSHIP_THIN", "FLAGSHIP_TRACE", "open_flagship",
]
