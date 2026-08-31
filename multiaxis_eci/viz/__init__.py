"""Plotly figure builders, split by concern.

- core: save_fig, trace grids, forests, the capability/difficulty timeline
- gof: PIT / density / predicted-vs-observed / residual figures
- forecast: frontier-forecast figures
- mirt: per-fit MIRT figures (loadings, factor correlation, Q-matrix, K-vs-1D)
- compare: cross-fit comparison figures (cmp_* family)
- dashboard: figure-set assemblers + the self-contained dashboard HTML

The output directory stays `plots/` (config.PLOTS_DIR); this package holds
the code. The namespace re-exports the full public API so call sites use
`from viz import ...`.
"""
from multiaxis_eci.viz.compare import (
    alignment_methods_fig,
    cmp_convergence_fig,
    cmp_gof_fig,
    cmp_loo_vs_trust_fig,
    cmp_loo_waic_fig,
    cmp_pareto_k_fig,
    cmp_per_benchmark_rmse_fig,
    cmp_pit_ecdf_fig,
    cmp_tau_spectrum_fig,
    delpd_se,
)
from multiaxis_eci.viz.core import (
    all_models_forest_fig,
    capability_timeline_fig,
    forest_fig,
    hyperparams_fig,
    raw_scores_by_date_fig,
    save_fig,
    sota_forest_fig,
    subplot_grid,
    trace_posterior_grid,
)
from multiaxis_eci.viz.dashboard import (
    assemble_dashboard,
    build_axis_figures,
    build_comparison,
    build_fit_figures,
    build_gof_figures,
    signed_display_frames,
    write_dashboard,
)
from multiaxis_eci.viz.forecast import (
    capability_forecast_fig,
    crossover_dotwhisker_fig,
    exceedance_prob_fig,
)
from multiaxis_eci.viz.gof import (
    benchmark_icc_fig,
    benchmark_obs_vs_pred_fig,
    density_overlay_fig,
    pit_ecdf_fig,
    pit_hist_fig,
    pred_vs_obs_fig,
    residuals_per_benchmark_fig,
)
from multiaxis_eci.viz.mirt import (
    axes_frontier_fig,
    axes_scatter_matrix_fig,
    binary_qmatrix_fig,
    factor_corr_fig,
    factor_vs_1d_fig,
    forest_grid_fig,
    loadings_grid_fig,
    loadings_heatmap_fig,
    per_bench_r2_delta_fig,
    ppca_spectrum_fig,
    pred_scatter_fig,
)

__all__ = [
    "save_fig", "trace_posterior_grid", "hyperparams_fig", "forest_fig",
    "all_models_forest_fig", "sota_forest_fig", "capability_timeline_fig", "raw_scores_by_date_fig",
    "capability_forecast_fig", "crossover_dotwhisker_fig", "exceedance_prob_fig",
    "pit_hist_fig", "pit_ecdf_fig", "density_overlay_fig", "pred_vs_obs_fig",
    "benchmark_obs_vs_pred_fig", "benchmark_icc_fig",
    "residuals_per_benchmark_fig",
    "loadings_heatmap_fig", "factor_corr_fig", "binary_qmatrix_fig",
    "forest_grid_fig", "loadings_grid_fig", "subplot_grid",
    "ppca_spectrum_fig", "factor_vs_1d_fig", "pred_scatter_fig",
    "per_bench_r2_delta_fig", "axes_frontier_fig", "axes_scatter_matrix_fig",
    "cmp_per_benchmark_rmse_fig", "cmp_gof_fig", "cmp_convergence_fig",
    "cmp_pit_ecdf_fig",
    "cmp_loo_waic_fig",
    "cmp_pareto_k_fig", "cmp_loo_vs_trust_fig", "cmp_tau_spectrum_fig",
    "alignment_methods_fig",
    "delpd_se",
    "build_gof_figures", "build_fit_figures", "build_axis_figures",
    "signed_display_frames",
    "build_comparison", "assemble_dashboard", "write_dashboard",
]
