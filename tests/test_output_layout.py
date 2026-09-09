"""Locks on the 5_outputs/<data generation>/ layout and the figure file names.

Written for the 2026-09 restructure: results/ and plots/ merged into one tree
whose first level is the data generation, figures beside their tables with
the interactive twins under html/, and explicit file names instead of the
`_C` / `_D` / `_1` shorthands.
"""
from __future__ import annotations

import plotly.graph_objects as go

from multiaxis_eci import analysis, config
from multiaxis_eci.viz import figure_filename, save_fig


def test_output_tree_is_dated_first():
    assert config.RESULTS_DIR.parent == config.OUTPUTS_DIR
    assert config.OUTPUTS_DIR.name == "5_outputs"
    assert config.RESULTS_DIR.name == f"data{config.DATA_TAG}"
    assert config.COMPARISONS_DIR == config.RESULTS_DIR / "comparisons"
    assert config.DIAGNOSTICS_DIR == config.RESULTS_DIR / "diagnostics"
    assert config.LEGACY_RESULTS_DIR == config.OUTPUTS_DIR / "pre_pipeline"


def test_fitspec_folders_sit_under_the_data_generation():
    spec = analysis.FitSpec(K=4, human_merge=True, lineage_prior=True, lineage_bm=True)
    assert spec.results_dir == config.RESULTS_DIR / "mirt_humanmerge_lineageprior_lineagebm"
    # K is in the figures folder, not the results folder: two Ks of one flag set
    # share their tables and traces but never their figures.
    assert spec.figures_dir == spec.results_dir / "figures" / "k4"
    assert spec.trace_path.parent == spec.results_dir
    # A legacy folder name with the data suffix still parses back to the same spec.
    tag = analysis.fitspec._folder_tag(
        config.OUTPUTS_DIR / "old" / "mirt_humanmerge_lineageprior_lineagebm_data20260908" / "t.nc")
    assert tag == "_humanmerge_lineageprior_lineagebm"


def test_figure_filenames_are_explicit():
    assert figure_filename("timeline_2_reasoning") == "timeline_axis2_reasoning"
    assert figure_filename("timeline_1_math_all") == "timeline_axis1_math_all"
    assert figure_filename("loadings_3_agentic") == "loadings_axis3_agentic"
    assert figure_filename("forecast_1_math_when") == "forecast_axis1_math_crossover_dates"
    assert figure_filename("forecast_1_math_prob") == "forecast_axis1_math_exceedance_probability"
    for unchanged in ("gof_pit", "timeline_difficulty", "loadings_heatmap", "hyperparameters"):
        assert figure_filename(unchanged) == unchanged


def test_save_fig_writes_png_beside_html_subfolder(tmp_path, monkeypatch):
    # Keep the test offline: kaleido is not needed to check the layout.
    monkeypatch.setattr(go.Figure, "write_image", lambda self, path, **kw: open(path, "wb").close())
    save_fig(go.Figure(), "benchmark_difficulty", tmp_path / "figures")
    assert (tmp_path / "figures" / "benchmark_difficulty.png").exists()
    assert (tmp_path / "figures" / "html" / "benchmark_difficulty.html").exists()
