"""End-to-end correctness tests for the ECI Bayesian pipeline.

Layout
------
* data loader      — invariants on shape, indexing, censoring
* analysis helpers — anchor pinning, ECI affine, timeline split, human levels
* model + sampler  — tiny NUTS run, verifies the full Beta IRT compiles + samples
* PPC + plots      — every plot builder returns a valid go.Figure
* CLI              — argparse contract for 2_fit.py

The analysis / plot tests run against a synthetic InferenceData built from a
seeded RNG — no NUTS needed, so they finish in a fraction of a second. The
single sampling test uses tiny draws (8 × 1) just to prove the model compiles
and produces an InferenceData with the right variables.
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pymc as pm
import pytensor
import pytest

from multiaxis_eci import analysis
import multiaxis_eci.analysis.rotation as rotation
from multiaxis_eci import viz
from multiaxis_eci.analysis import (
    align_factor_signs, align_rotations, alignment_report, all_models_stats_df,
    apply_rotation, canonicalize_factors,
    eci_transform,
    flat_C, forest_stats_df, human_stats_df,
    mirt_factors_from_trace, mirt_identified_rhat, mirt_identified_rhat_nc, prepare_fit,
    mirt_informed_mask, nc_difficulty_draws, post_stats,
    promax_rotate, sota_stats_df, timeline_stats_df, trace_anchors,
    trace_loading_prior,
)
from multiaxis_eci.config import (
    ANCHOR_HIGH, ANCHOR_LOW, DATA_DIR, ECI_EPS, HUMAN_ORDER,
    PROJECT_ROOT, RELEASE_DATES, SOTA_MODELS, ZERO_DIAG_THRESHOLD,
)
from multiaxis_eci.data import (
    ECIData, clip_scores_to_floors, drop_model_benchmark_cells,
    drop_model_observations, drop_zero_scores, find_model_idx,
    load_benchmark_floors, load_eci_data,
)
from multiaxis_eci.lineage import LINEAGE_MAP, build_lineage_structure
from multiaxis_eci.models.qmatrix import QMATRIX_VARIANTS, axes_as_list
from multiaxis_eci.models.mirt import build_mirt_model
from multiaxis_eci.models.mirt_nc import _validate_qmatrix, build_mirt_nc_model
from multiaxis_eci.persistence import save_pit, save_summary, save_trace
from multiaxis_eci.viz import (
    all_models_forest_fig, capability_timeline_fig, density_overlay_fig,
    forest_fig, forest_grid_fig, hyperparams_fig, loadings_grid_fig,
    pit_ecdf_fig, pit_hist_fig, pred_vs_obs_fig,
    residuals_per_benchmark_fig, sota_forest_fig, subplot_grid,
)
from multiaxis_eci.ppc import (
    _beta_draw, _flatten_over_chains, _thin_sel, boundary_mask, compute_gof,
    pit_values, posterior_predictive_mirt, posterior_predictive_mirt_nc,
)


# ───────────────────────── Fixtures ────────────────────────────────────────
@pytest.fixture(scope="session")
def data() -> ECIData:
    return load_eci_data()


@pytest.fixture(scope="session")
def raw_df(data: ECIData) -> pd.DataFrame:
    """Mirror of what load_eci_data() actually keeps post-filtering.

    Data prep (merge/dedup) is done by the preprocessing notebook and lives in
    the complete `1_data/processed/benchmarks_merged.csv` (every benchmark). With
    the canonical defaults (collapse_effort_variants=False,
    drop_low_obs_models=False) the modeling-stage transforms that change the row
    count are (1) the unconditional retirement drop, (2) the curated exclusion
    filter applied at fit time and (3) the human-baseline merge, so this mirror
    applies all three.
    """
    from multiaxis_eci.data import (PROCESSED_FILE, _load_human_baselines_as_models,
                      load_excluded_benchmarks, load_retired_benchmarks)

    df = pd.read_csv(PROCESSED_FILE)
    # Mirror load_eci_data's two benchmark filters, retirement first.
    df = df[~df["benchmark"].isin(load_retired_benchmarks()
                                  | load_excluded_benchmarks())].reset_index(drop=True)
    humans = _load_human_baselines_as_models()
    humans = humans[humans["benchmark"].isin(df["benchmark"].unique())]
    df = pd.concat([df, humans], ignore_index=True)
    return df.reset_index(drop=True)


@pytest.fixture(scope="session")
def synth_trace(data: ECIData):
    """Synthetic K=1 MIRT InferenceData mimicking a canonical-preset posterior.

    Fast — no NUTS needed for downstream analysis/plot tests. Built so that
    eci_transform and SOTA lookups produce finite, sensible numbers.
    """
    rng = np.random.default_rng(0)
    n_chain, n_draw = 2, 100
    M, B = data.n_models, data.n_benchmarks

    # theta — give SOTA + anchor models specific draws so eci_transform is finite
    theta = rng.normal(0.0, 1.0, size=(n_chain, n_draw, M, 1))
    low_idx = find_model_idx(data.mlookup, ANCHOR_LOW[0])
    high_idx = find_model_idx(data.mlookup, ANCHOR_HIGH[0])
    theta[..., low_idx, 0]  = rng.normal(-1.0, 0.05, size=(n_chain, n_draw))
    theta[..., high_idx, 0] = rng.normal( 1.5, 0.05, size=(n_chain, n_draw))

    # ZeroSumNormal-style: D sums to zero
    D_raw = rng.normal(0.0, 1.0, size=(n_chain, n_draw, B))
    D = D_raw - D_raw.mean(axis=-1, keepdims=True)

    A = np.exp(rng.normal(np.log(0.5), 0.3, size=(n_chain, n_draw, B, 1)))

    sigma_b = np.full((n_chain, n_draw, B), 0.08)
    phi_b = 1.0 / (4.0 * sigma_b ** 2) - 1.0

    tau_CD = np.full((n_chain, n_draw), 3.0)
    tau_A_normal = np.full((n_chain, n_draw), 0.5)
    tau_A = np.broadcast_to(tau_A_normal[..., None], (n_chain, n_draw, 1)).copy()

    idata = az.from_dict(
        posterior={
            "theta": theta, "A": A, "D": D,
            "sigma_b": sigma_b, "phi_b": phi_b,
            "tau_CD": tau_CD, "tau_A": tau_A, "tau_A_normal": tau_A_normal,
        },
        coords={"model": data.mlookup["model"].tolist(),
                "bench": data.blookup["benchmark"].tolist(),
                "latent": ["axis1"]},
        dims={"theta": ["model", "latent"], "A": ["bench", "latent"],
              "D": ["bench"], "sigma_b": ["bench"], "phi_b": ["bench"],
              "tau_A": ["latent"]})
    idata.posterior.attrs["mirt_loading_prior"] = "normal"
    return idata


# ───────────────────────── Data loader ─────────────────────────────────────
class TestData:
    def test_shape_matches_filtered_csv(self, data, raw_df):
        assert data.n_obs == len(raw_df)
        assert data.n_models == raw_df["model_version"].nunique()
        assert data.n_benchmarks == raw_df["benchmark"].nunique()

    def test_excluded_benchmarks_dropped(self, data):
        present = set(data.blookup["benchmark"].values)
        for b in data.excluded_benchmarks:
            assert b not in present, f"excluded benchmark {b} still in dataset"
        assert {"ARC-AGI", "HellaSwag"}.issubset(data.excluded_benchmarks)

    def test_exclusion_total_losses_are_not_silent(self, capsys):
        """A model — or worse, a human tier — whose every observation sits on
        excluded benchmarks leaves the canonical fit entirely. The tier drop
        is automatic and expected under the canonical scope, but it must be
        announced (a printed scope line, like the model drops). On current
        data the Committee of Skilled Generalists (only baseline: the
        excluded PIQA) is the live example."""
        d = load_eci_data()
        assert "human tiers:" in capsys.readouterr().out
        d_all = load_eci_data(include_all_benchmarks=True)
        # The full-scope fit keeps every model the canonical scope loses.
        assert d_all.n_models > d.n_models
        assert d_all.is_human.sum() >= d.is_human.sum()

    def test_eci_data_merge_adds_mmlu_and_bbh(self, data):
        """The pipeline (`1_data/1_pipeline/pipeline.ipynb`) contributes MMLU + BBH
        with the SOTA models attached under their versioned IDs (not the
        eci_data.csv "pretty" names). Contract duplicated here so a future
        pipeline change that drops these benchmarks or those mappings is
        caught immediately."""
        expected_mapped_ids = {
            "claude-3-5-sonnet-20241022",
            "claude-3-5-sonnet-20240620",
            "claude-3-opus-20240229",
            "gemini-1.5-pro-002",
            "gpt-4-0314",
            "gpt-4-0613",
            "gpt-4-turbo-2024-04-09",
            "gpt-4o-2024-05-13",
            "gpt-4o-2024-08-06",
            "gpt-4o-2024-11-20",
        }
        present_b = set(data.blookup["benchmark"].values)
        present_m = set(data.mlookup["model"].values)
        assert "MMLU" in present_b
        assert "BIG-Bench Hard (BBH)" in present_b
        # Mapped SOTA models pick up the new evaluations under their versioned IDs
        for versioned in expected_mapped_ids:
            assert versioned in present_m, versioned
        # The eci_data pretty name "GPT-4 (Mar 2023)" should NOT appear independently
        # — it was mapped to gpt-4-0314
        assert "GPT-4 (Mar 2023)" not in present_m

    def test_processed_file_contract(self):
        """Smoke test: the processed file exists, has the expected 9 columns,
        and carries every benchmark (exclusions are applied at fit time by
        load_eci_data, not during data generation)."""
        from multiaxis_eci.data import PROCESSED_FILE, load_excluded_benchmarks
        assert PROCESSED_FILE.exists(), (
            f"{PROCESSED_FILE} missing — run 1_data/1_pipeline/pipeline.ipynb and "
            f"copy output/benchmarks_merged.csv into 1_data/processed/"
        )
        df = pd.read_csv(PROCESSED_FILE)
        assert set(df.columns) == {
            "model_version", "score", "release_date", "organization",
            "benchmark", "stderr", "source", "category",
        }
        assert (df["category"] != "Tier 2 Excluded").all()
        # The pipeline no longer pre-excludes — the curated "easy-for-humans"
        # benchmarks must be present in the complete processed file. The fit-time
        # drop is covered by test_excluded_benchmarks_dropped.
        present = set(df["benchmark"].unique())
        excluded = load_excluded_benchmarks()
        missing = excluded - present
        assert not missing, (
            f"excluded benchmarks absent from the complete processed file: "
            f"{sorted(missing)} — re-run the pipeline (exclusions now happen at "
            f"fit time, so data generation must keep every benchmark)"
        )

    def test_no_mojibake_in_model_names(self):
        """SEAL model names are JSON-decoded per __next_f chunk (json.loads),
        which preserves literal UTF-8 like the U+2020 dagger footnote marker.
        The old `.decode("unicode_escape")` treated bytes as Latin-1 and mangled
        multi-byte chars into mojibake (dagger -> 'â' + C1 control bytes). Guard
        against that regression: no model name may contain a C1 control char."""
        from multiaxis_eci.data import PROCESSED_FILE
        df = pd.read_csv(PROCESSED_FILE)
        bad = [m for m in df["model_version"].astype(str).unique()
               if any(0x80 <= ord(c) <= 0x9f for c in m)]
        assert not bad, f"mojibake (C1 control chars) in model names: {bad[:10]}"

    def test_drop_zero_scores_removes_zeros(self, data):
        d2 = drop_zero_scores(data)
        assert d2.n_obs == data.n_obs - int(data.zero_score_mask.sum())
        assert not d2.zero_score_mask.any()
        assert (d2.scores > 0).all()
        assert d2.n_models == data.n_models  # models with all-zero rows just become low-obs
        assert d2.n_obs_per_model.sum() == d2.n_obs

    def test_n_eff_from_stderr_and_row_alignment(self, data):
        """n_eff is one effective test length per observation, np.inf where no
        stderr is reported, and every observation-subsetting helper carries it
        along — a shifted n_eff would silently re-weight the wrong cells."""
        assert data.n_eff.shape == data.scores.shape
        finite = np.isfinite(data.n_eff)
        assert finite.any(), "no reported stderr survives into the fit data"
        assert np.all(data.n_eff[finite] >= 2.0)
        # the conversion: n_eff = p(1-p)/se^2 on the cell's own observed score
        p = data.scores[finite]
        se = np.sqrt(p * (1.0 - p) / data.n_eff[finite])
        assert np.all((se > 0) & (se < 0.5))
        # humans report no stderr
        assert np.all(np.isinf(data.n_eff[data.is_human[data.model_idx]]))

        sg = [m for m in data.mlookup["model"] if m == "Skilled Generalist"]
        gpqa = [b for b in data.blookup["benchmark"] if "GPQA" in b]
        for sub in (drop_zero_scores(data),
                    drop_model_observations(data, sg),
                    drop_model_benchmark_cells(data, sg[0] if sg else "none", gpqa),
                    clip_scores_to_floors(data, load_benchmark_floors(data))):
            assert sub.n_eff.shape == sub.scores.shape == (sub.n_obs,)
        # clipping moves scores only; the instrument precision is unchanged
        clipped = clip_scores_to_floors(data, load_benchmark_floors(data))
        np.testing.assert_array_equal(clipped.n_eff, data.n_eff)

    def test_low_obs_flag(self, data):
        from multiaxis_eci.config import LOW_OBS_THRESHOLD
        assert data.is_low_obs.dtype == bool
        assert data.is_low_obs.shape == (data.n_models,)
        # Humans are exempt from the flag regardless of obs count.
        expected = (data.n_obs_per_model < LOW_OBS_THRESHOLD) & ~data.is_human
        assert (data.is_low_obs == expected).all()
        assert not data.is_low_obs[data.is_human].any()
        # Counts add up
        assert data.n_obs_per_model.sum() == data.n_obs

    def test_indices_in_range(self, data):
        assert data.model_idx.min() >= 0
        assert data.model_idx.max() == data.n_models - 1
        assert data.bench_idx.min() >= 0
        assert data.bench_idx.max() == data.n_benchmarks - 1

    def test_lookup_alignment(self, data):
        # mlookup is 1-indexed: row i corresponds to model_idx i+1 (== position i)
        assert (data.mlookup["model_idx"].values == np.arange(1, data.n_models + 1)).all()
        assert (data.blookup["benchmark_idx"].values == np.arange(1, data.n_benchmarks + 1)).all()
        # Sorted alphabetically as the loader claims
        models = data.mlookup["model"].tolist()
        assert models == sorted(models)

    def test_zero_score_mask_matches_scores(self, data):
        assert (data.zero_score_mask == (data.scores == 0.0)).all()
        assert data.zero_diag_threshold == ZERO_DIAG_THRESHOLD

    def test_find_model_idx_round_trip(self, data):
        for name in [ANCHOR_LOW[0], ANCHOR_HIGH[0], SOTA_MODELS[0], SOTA_MODELS[-1]]:
            idx = find_model_idx(data.mlookup, name)
            assert 0 <= idx < data.n_models
            assert data.mlookup["model"].iloc[idx] == name

    def test_find_model_idx_raises_on_unknown(self, data):
        with pytest.raises(KeyError):
            find_model_idx(data.mlookup, "definitely-not-a-model")

    def test_scores_in_unit_interval(self, data):
        assert (data.scores >= 0.0).all()
        assert (data.scores <= 1.0).all()


# ──────────────── Era filter (load_eci_data date bounds) ───────────────────
class TestEraFilter:
    def test_min_release_date_filter(self, data):
        """min_release_date: known-pre-cutoff models dropped, undated + humans
        kept, both ECI anchors survive (they are 2024/2025 releases)."""
        post = load_eci_data(min_release_date="2024-01-01")
        assert post.n_models < data.n_models
        present = set(post.mlookup["model"])
        # known-2023 models are gone (release dates in the processed file)
        assert "gpt-4-0314" not in present
        assert "LLaMA-65B" not in present
        # Humans carry no release date, so the era filter never targets them
        # directly. A tier can still leave INDIRECTLY: once the pre-cutoff
        # models go, a benchmark whose only takers were old loses all AI
        # coverage and drops, and a tier scored only there has nothing left to
        # be compared against (at the 2024 cutoff this costs Committee of
        # Average Humans). Assert the invariant, not the count.
        assert post.is_human.sum() >= data.is_human.sum() - 1
        for tier in ("Average Human", "Skilled Generalist", "Domain Expert"):
            assert tier in present, f"{tier} must survive the era filter"
        assert ANCHOR_LOW[0] in present and ANCHOR_HIGH[0] in present
        # every dated surviving model is >= cutoff
        from multiaxis_eci.data import PROCESSED_FILE
        raw = pd.read_csv(PROCESSED_FILE)
        d = pd.to_datetime(raw["release_date"], errors="coerce")
        per_model = d.groupby(raw["model_version"]).max().dropna()
        dated_survivors = [m for m in present if m in per_model.index]
        assert (per_model.loc[dated_survivors] >= pd.Timestamp("2024-01-01")).all()

    def test_max_release_date_filter(self, data):
        """Holdout split: known-post-cutoff models dropped, undated + humans
        kept, and no dated survivor is on/after the cutoff."""
        cutoff = "2025-07-01"
        past = load_eci_data(max_release_date=cutoff)
        assert past.n_models < data.n_models
        present = set(past.mlookup["model"])
        assert past.is_human.sum() == data.is_human.sum()
        # a pre-cutoff staple survives; the post-cutoff ECI anchor is gone
        assert ANCHOR_LOW[0] in present            # Claude 3.5 Sonnet, 2024-10
        assert ANCHOR_HIGH[0] not in present       # GPT-5, 2025-08
        from multiaxis_eci.data import PROCESSED_FILE
        raw = pd.read_csv(PROCESSED_FILE)
        d = pd.to_datetime(raw["release_date"], errors="coerce")
        per_model = d.groupby(raw["model_version"]).max().dropna()
        dated_survivors = [m for m in present if m in per_model.index]
        assert (per_model.loc[dated_survivors] < pd.Timestamp(cutoff)).all()
        # min + max compose to a window
        window = load_eci_data(min_release_date="2024-01-01",
                               max_release_date=cutoff)
        assert window.n_models <= past.n_models


class TestScopeFlags:
    """--drop-benchmarks and --cyber, the two per-fit scope levers."""

    def test_drop_benchmarks_removes_exactly_those_names(self, data_all):
        assert load_eci_data(include_all_benchmarks=True,
                             drop_benchmarks=None).n_obs == data_all.n_obs

        targets = ["GBAEval", "VPCT"]
        got = load_eci_data(include_all_benchmarks=True, drop_benchmarks=targets)
        kept, before = set(got.blookup["benchmark"]), set(data_all.blookup["benchmark"])
        assert before - kept == set(targets)
        # Human rows on a dropped benchmark must go too (VPCT has one).
        assert got.is_human.sum() <= data_all.is_human.sum()

    def test_drop_benchmarks_warns_on_unknown_name(self, data_all):
        """A typo must be visible, not silently fit the full scope. It is a
        warning rather than an error because a name already absent has to be a
        no-op: the retirement list removes names a caller may still pass."""
        with pytest.warns(UserWarning, match="not in the table"):
            got = load_eci_data(include_all_benchmarks=True,
                                drop_benchmarks=["GBAEvaal"])
        assert got.n_obs == data_all.n_obs

    def test_dropping_a_retired_benchmark_is_a_silent_no_op(self, data_all):
        """The retirement list already removed them, so naming one changes
        nothing and raises no warning."""
        from multiaxis_eci.data import load_retired_benchmarks
        retired = sorted(load_retired_benchmarks())
        assert retired, "retired_benchmarks.txt is empty"
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            got = load_eci_data(include_all_benchmarks=True,
                                drop_benchmarks=retired)
        assert (got.n_obs, got.n_models, got.n_benchmarks) == (
            data_all.n_obs, data_all.n_models, data_all.n_benchmarks)
        for b in retired:
            assert b not in set(data_all.blookup["benchmark"])

    def test_cyber_is_additive_and_never_shadows_a_pipeline_row(self, data_all):
        """Cyber rows land only on models already fit, only on benchmarks the
        pipeline does not supply, and leave every existing observation intact."""
        from multiaxis_eci.data import CURATED_DIR

        assert load_eci_data(include_all_benchmarks=True,
                             fit_cyber=False).n_obs == data_all.n_obs, \
            "cyber must be off by default"

        if not (CURATED_DIR / "cyber_benchmarks.csv").exists():
            pytest.skip("cyber_benchmarks.csv absent (run diagnostics.fetch_cyber_eci)")

        got = load_eci_data(include_all_benchmarks=True, fit_cyber=True)
        assert got.n_obs > data_all.n_obs
        # No new test-takers: a cyber-only model carries no cross-benchmark info.
        assert set(got.mlookup["model"]) == set(data_all.mlookup["model"])
        # Purely additive on the benchmark side.
        assert set(got.blookup["benchmark"]) > set(data_all.blookup["benchmark"])
        # Every (model, benchmark) cell of the base scope survives exactly once.
        # Both index arrays are 0-based positions into their lookup frame.
        def cells(d):
            mv = np.asarray(d.mlookup["model"])[d.model_idx]
            bn = np.asarray(d.blookup["benchmark"])[d.bench_idx]
            return pd.Series(list(zip(mv, bn))).value_counts()
        base_cells, got_cells = cells(data_all), cells(got)
        assert (base_cells == got_cells.reindex(base_cells.index)).all(), \
            "a cyber row shadowed or duplicated a pipeline observation"


# ───────────────────────── Analysis helpers ────────────────────────────────
class TestAnalysis:
    def test_post_stats_ordering(self):
        s = np.linspace(0.0, 1.0, 1000)
        mean, lo, hi = post_stats(s, hdi_prob=0.94)
        assert lo <= mean <= hi
        assert 0.4 < mean < 0.6  # ~0.5 for uniform[0,1]

    def test_flat_C_shape(self, synth_trace, data):
        C_flat = flat_C(synth_trace)
        n_chain = synth_trace.posterior.sizes["chain"]
        n_draw  = synth_trace.posterior.sizes["draw"]
        assert C_flat.shape == (n_chain * n_draw, data.n_models)

    def test_eci_transform_pins_anchors(self, synth_trace, data):
        C_flat = flat_C(synth_trace)
        t = eci_transform(C_flat, data)
        low_idx  = find_model_idx(data.mlookup, ANCHOR_LOW[0])
        high_idx = find_model_idx(data.mlookup, ANCHOR_HIGH[0])

        eci_low  = t.apply(C_flat[:, low_idx])
        eci_high = t.apply(C_flat[:, high_idx])
        np.testing.assert_allclose(eci_low,  ANCHOR_LOW[1],  rtol=0, atol=1e-9)
        np.testing.assert_allclose(eci_high, ANCHOR_HIGH[1], rtol=0, atol=1e-9)

    def test_sota_stats_df_columns_and_dates(self, synth_trace, data, raw_df):
        df = sota_stats_df(synth_trace, data, raw_df)
        expected = {"model", "release_date", "C_mean", "C_hdi_low", "C_hdi_high",
                    "ECI_mean", "ECI_hdi_low", "ECI_hdi_high"}
        assert expected.issubset(df.columns)
        # Models with no surviving rows are skipped (with a warning print) —
        # the resulting df is a subset of SOTA_MODELS in original order.
        assert 0 < len(df) <= len(SOTA_MODELS)
        assert set(df["model"]).issubset(SOTA_MODELS)
        model_dates, _ = analysis._release_dates(raw_df)
        for _, row in df.iterrows():
            expected_date = model_dates.get(row["model"])
            if pd.isna(row["release_date"]):
                assert expected_date is None or pd.isna(expected_date)
            else:
                assert row["release_date"] == expected_date
            assert row["C_hdi_low"]   <= row["C_mean"]   <= row["C_hdi_high"]
            assert row["ECI_hdi_low"] <= row["ECI_mean"] <= row["ECI_hdi_high"]

    def test_forest_stats_df_sorted_ascending(self, synth_trace, data):
        bench_names = data.blookup["benchmark"].tolist()
        df = forest_stats_df(synth_trace, "D", bench_names)
        assert len(df) == data.n_benchmarks
        assert (df["mean"].values[:-1] <= df["mean"].values[1:]).all()

    def test_all_models_stats_df_shape_and_sorted(self, synth_trace, data):
        df = all_models_stats_df(synth_trace, data)
        assert len(df) == data.n_models
        assert set(df.columns) >= {"name", "mean", "hdi_low", "hdi_high",
                                    "n_obs", "is_low_obs"}
        assert (df["mean"].values[:-1] <= df["mean"].values[1:]).all()
        assert (df["hdi_low"] <= df["mean"]).all()
        assert (df["mean"] <= df["hdi_high"]).all()
        # Low-obs flag matches data.is_low_obs after the name → idx mapping
        for _, row in df.iterrows():
            idx = data.mlookup.loc[data.mlookup["model"] == row["name"], "model_idx"].iloc[0] - 1
            assert bool(data.is_low_obs[idx]) == bool(row["is_low_obs"])

    def test_timeline_stats_df_split(self, synth_trace, data, raw_df):
        tl = timeline_stats_df(synth_trace, data, raw_df)
        kinds = set(tl["kind"].unique())
        assert kinds == {"model", "benchmark"}
        assert tl["release_date"].notna().all()
        # Every kind=='benchmark' row name must be a known benchmark
        bench_set = set(data.blookup["benchmark"].values)
        bench_rows = tl[tl["kind"] == "benchmark"]
        assert set(bench_rows["name"]).issubset(bench_set)

    def test_human_stats_df(self, synth_trace, data):
        df = human_stats_df(synth_trace, data)
        # Humans are now fitted in the IRT — should have rows iff is_human is set
        if data.is_human.any():
            assert len(df) == int(data.is_human.sum())
            assert set(df.columns) >= {"name", "mean", "hdi_low", "hdi_high", "n_obs"}
            assert (df["hdi_low"] <= df["mean"]).all()
            assert (df["mean"] <= df["hdi_high"]).all()
            assert (df["mean"].values[:-1] <= df["mean"].values[1:]).all()
        else:
            assert len(df) == 0


# ───────────────────────── Model + sampler ─────────────────────────────────
class TestModel:
    """The canonical K=1 configuration of the MIRT builder."""

    def test_model_uses_single_beta_likelihood(self, data):
        m = build_mirt_model(data, 1, loading_prior="normal")
        assert {v.name for v in m.observed_RVs} == {"obs"}

    def test_phi_sigma_variance_identity(self, data):
        """phi_b = 1/(4σ²)-1 makes Var(Beta(μφ,(1-μ)φ)) = 4σ²μ(1-μ), so σ is the
        score SD at μ=0.5 — the interpretation the whole noise model rests on."""
        from scipy.stats import beta as beta_dist
        m = build_mirt_model(data, 1, loading_prior="normal")
        sig, phi = pm.draw([m["sigma_b"], m["phi_b"]], draws=1, random_seed=0)
        assert np.allclose(phi, 1.0 / (4.0 * sig ** 2) - 1.0)       # the code's φ
        s = 0.08
        ph = 1.0 / (4.0 * s ** 2) - 1.0
        for mu in (0.1, 0.5, 0.9):
            v = beta_dist(mu * ph, (1.0 - mu) * ph).var()
            assert np.isclose(v, 4.0 * s ** 2 * mu * (1.0 - mu), rtol=1e-6)
        assert np.isclose(beta_dist(0.5 * ph, 0.5 * ph).std(), s, rtol=1e-6)

    def test_boundary_clipping(self, data):
        """Exact-0/1 scores are clipped onto [ε,1-ε], interior untouched, and the
        full logp stays finite (un-clipped boundary rows would be Beta logp -inf)."""
        if not ((data.scores == 0.0) | (data.scores == 1.0)).any():
            pytest.skip("no boundary scores in current data")
        m = build_mirt_model(data, 1, loading_prior="normal")
        observed = m.rvs_to_values[m["obs"]].eval()                # what the Beta sees
        assert observed.min() >= ECI_EPS and observed.max() <= 1.0 - ECI_EPS
        # "Interior" = inside the clip bounds: scores in (0, ECI_EPS) exist in the
        # data (GBAEval 0.000172) and are correctly moved up to ECI_EPS.
        interior = (data.scores >= ECI_EPS) & (data.scores <= 1.0 - ECI_EPS)
        assert np.allclose(observed[interior], data.scores[interior])
        assert np.isfinite(m.compile_logp()(m.initial_point()))

    @pytest.mark.slow
    def test_tiny_sample_runs(self, data):
        """Drawing just a few samples to prove NUTS compiles and runs."""
        with build_mirt_model(data, 1, loading_prior="normal"):
            trace = pm.sample(
                draws=5, tune=5, chains=1, cores=1,
                progressbar=False, random_seed=0,
                return_inferencedata=True,
                compute_convergence_checks=False,
            )
        post = trace.posterior
        assert post["theta"].shape[-2] == data.n_models
        assert post["D"].shape[-1] == data.n_benchmarks
        assert np.isfinite(post["theta"].values).all()


# ───────────────────────── PPC + plots ─────────────────────────────────────
class TestPPC:
    def test_posterior_predictive_shape(self, synth_trace, data):
        y_rep = posterior_predictive_mirt(synth_trace, data)
        n_chain = synth_trace.posterior.sizes["chain"]
        n_draw  = synth_trace.posterior.sizes["draw"]
        assert y_rep.shape == (n_chain * n_draw, data.n_obs)
        assert (y_rep >= 0.0).all()
        assert (y_rep <= 1.0).all()

    def test_pit_values_length(self, synth_trace, data):
        y_rep = posterior_predictive_mirt(synth_trace, data)
        excl = boundary_mask(data)
        pit = pit_values(y_rep, data.scores, excl)
        assert pit.shape == ((~excl).sum(),)
        assert (0.0 <= pit).all() and (pit <= 1.0).all()

    def test_boundary_mask_covers_both_edges(self, data):
        from multiaxis_eci.config import ECI_EPS
        m = boundary_mask(data)
        assert m[data.zero_score_mask].all()                 # exact zeros out
        assert m[data.scores >= 1.0 - ECI_EPS].all()         # exact ones out
        interior = (data.scores > 0.0) & (data.scores < 1.0 - ECI_EPS)
        assert not m[interior].any()                          # nothing else

    def test_gof_metrics(self, synth_trace, data):
        y_rep = posterior_predictive_mirt(synth_trace, data)
        mu = posterior_predictive_mirt(synth_trace, data, return_mean=True)
        gof = compute_gof(y_rep, data, mu)
        for k in ["rmse", "mae", "bayesian_r2",
                  "ks_stat", "ks_p", "pit_mean", "pit_var"]:
            assert k in gof.metrics
            assert np.isfinite(gof.metrics[k])
        # Gelman Bayesian R² = Var(fit)/(Var(fit)+Var(resid)) is a ratio in [0,1]
        assert 0.0 <= gof.metrics["bayesian_r2"] <= 1.0
        assert gof.metrics["n_obs"] == data.n_obs
        assert gof.metrics["n_nonzero_score"] + gof.metrics["n_zero_score"] == data.n_obs

    def test_pp_reproduces_sigmoid_link(self, synth_trace, data):
        """The K=1 posterior predictive must reconstruct μ = σ(A·θ − D). E[Beta]=μ,
        so the PP mean tracks the analytic per-draw link mean (MC noise only) — a
        wrong link (dropped A, flipped sign, no sigmoid) breaks the match."""
        post = synth_trace.posterior
        theta = post["theta"].values[..., 0].reshape(-1, data.n_models)
        D = post["D"].values.reshape(-1, data.n_benchmarks)
        A = post["A"].values[..., 0].reshape(-1, data.n_benchmarks)
        eta = A[:, data.bench_idx] * theta[:, data.model_idx] - D[:, data.bench_idx]
        mu_mean = (1.0 / (1.0 + np.exp(-eta))).mean(axis=0)        # exact E[y_rep] per obs
        y_pred_mean = posterior_predictive_mirt(synth_trace, data).mean(axis=0)
        # 0.95, not higher: the fixture's K=1 loadings put most mu mid-range,
        # where 200 draws of Beta-sampling MC noise alone hold the corr near
        # 0.97. A wrong link — dropped A, flipped sign, missing sigmoid —
        # lands FAR below (< 0.9), so the bar keeps its detection power
        # without being brittle to fixture-shape changes.
        assert np.corrcoef(y_pred_mean, mu_mean)[0, 1] > 0.95
        assert np.abs(y_pred_mean - mu_mean).mean() < 0.02

    def test_pp_floor_lifts_mu(self, synth_trace, data):
        """Floor-aware PP: mu = c + (1-c)*sigmoid, so E[y_rep] per obs >= c and
        never below the unfloored mean. A large uniform floor makes this visible."""
        floors = np.full(data.n_benchmarks, 0.4)
        plain = posterior_predictive_mirt(synth_trace, data).mean(axis=0)
        floored = posterior_predictive_mirt(synth_trace, data, floor_c=floors).mean(axis=0)
        assert (floored >= 0.4 - 0.05).all()        # >= floor up to Beta MC noise
        assert (floored >= plain - 1e-9).all()      # flooring never lowers the mean

    def test_pp_known_se_adds_instrument_noise(self, synth_trace, data):
        """known_se-aware PP: a short instrument (small n_eff) widens the
        predictive spread, an infinite one leaves the per-benchmark noise alone."""
        plain = posterior_predictive_mirt(synth_trace, data).std(axis=0)
        inert = posterior_predictive_mirt(
            synth_trace, data, n_eff=np.full(data.n_obs, np.inf)).std(axis=0)
        short = posterior_predictive_mirt(
            synth_trace, data, n_eff=np.full(data.n_obs, 5.0)).std(axis=0)
        # same phi either way, so only Beta MC noise separates plain from inert
        assert np.abs(inert - plain).mean() < 0.01
        # the fixture's across-draw mu spread dominates, so the added instrument
        # noise shows as a modest but near-universal widening (~1.3x on average)
        assert short.mean() > 1.15 * plain.mean()
        assert (short > plain).mean() > 0.9


class TestFloors:
    # Floors/clips are exploration-side (on by default there; --no-floors opts
    # out), and exploration fits the full
    # benchmark set. Exercise them on data_all (include_all_benchmarks=True) so
    # the test mirrors that scope: on the canonical scope the curated-excluded
    # benchmarks (OpenBookQA/VPCT/SimpleBench) drop out and their clip rows read
    # as out-of-scope, tripping the file's drift guard spuriously.
    def test_clip_raises_below_floor_rows(self, data_all):
        floors = load_benchmark_floors(data_all)
        below = data_all.scores < floors[data_all.bench_idx]
        clipped = clip_scores_to_floors(data_all, floors)
        # every score now sits at or above its benchmark floor
        assert (clipped.scores >= floors[clipped.bench_idx] - 1e-12).all()
        # only the previously-below rows moved; the rest are untouched
        assert np.array_equal(clipped.scores[~below], data_all.scores[~below])
        assert (clipped.scores[below] > data_all.scores[below]).all()

    def test_load_floors_shape_and_range(self, data_all):
        floors = load_benchmark_floors(data_all)
        assert floors.shape == (data_all.n_benchmarks,)
        assert ((floors >= 0.0) & (floors < 1.0)).all()


class TestPlots:
    def test_hyperparams(self, synth_trace):
        assert isinstance(hyperparams_fig(synth_trace), go.Figure)

    def test_forest_D(self, synth_trace, data):
        df = forest_stats_df(synth_trace, "D", data.blookup["benchmark"].tolist())
        assert isinstance(forest_fig(df, "D", "test"), go.Figure)

    def test_sota_forest(self, synth_trace, data, raw_df):
        df = sota_stats_df(synth_trace, data, raw_df)
        assert isinstance(sota_forest_fig(df, "ECI", "t", "ECI"), go.Figure)

    def test_all_models_forest(self, synth_trace, data):
        df = all_models_stats_df(synth_trace, data)
        fig = all_models_forest_fig(df, highlight={ANCHOR_LOW[0]})
        assert isinstance(fig, go.Figure)

    def test_axis_grids(self):
        # One panel per axis, drawn from synthetic frames: the forest grid must
        # keep every row of every panel, the loadings grid must cut to top_n by
        # share, and the shared legend must carry each row kind exactly once.
        dfs = [pd.DataFrame({"name": [f"m{i}" for i in range(5)],
                             "kind": ["model"] * 3 + ["frontier", "human"],
                             "mean": np.arange(5.0),
                             "hdi_low": np.arange(5.0) - 1,
                             "hdi_high": np.arange(5.0) + 1})
               for _ in range(4)]
        fg = forest_grid_fig(dfs, [f"axis{k+1}" for k in range(4)])
        assert sum(len(t.x) for t in fg.data) == 20
        assert sum(t.showlegend for t in fg.data) == 3

        rows = [{"axis": f"axis{k+1}", "benchmark": f"b{b}",
                 "loading_median": 0.1 * b, "hdi_low": 0.0, "hdi_high": 0.2 * b,
                 "axis_share": b / 10.0} for k in range(4) for b in range(10)]
        lg = loadings_grid_fig(pd.DataFrame(rows), top_n=3)
        assert [len(t.x) for t in lg.data] == [3] * 4
        assert isinstance(subplot_grid([fg, lg], ["a", "b"]), go.Figure)

    def test_capability_timeline(self, synth_trace, data, raw_df):
        tl = timeline_stats_df(synth_trace, data, raw_df)
        humans = human_stats_df(synth_trace, data)
        fig = capability_timeline_fig(tl, human_stats=humans)
        assert isinstance(fig, go.Figure)

    def test_pit_hist_and_ecdf(self, synth_trace, data):
        y_rep = posterior_predictive_mirt(synth_trace, data)
        gof = compute_gof(y_rep, data)
        assert isinstance(pit_hist_fig(gof.pit), go.Figure)
        assert isinstance(pit_ecdf_fig(gof.pit), go.Figure)

    def test_density_and_pred_vs_obs(self, synth_trace, data):
        y_rep = posterior_predictive_mirt(synth_trace, data)
        gof = compute_gof(y_rep, data)
        assert isinstance(density_overlay_fig(y_rep, data.scores, n_samples=10),
                          go.Figure)
        hover = [f"{i}" for i in range(data.n_obs)]
        assert isinstance(pred_vs_obs_fig(data.scores, gof.y_pred_mean, hover),
                          go.Figure)

    def test_residuals_per_benchmark(self, synth_trace, data):
        y_rep = posterior_predictive_mirt(synth_trace, data)
        gof = compute_gof(y_rep, data)
        nonzero = ~data.zero_score_mask
        df = pd.DataFrame({
            "benchmark": data.blookup["benchmark"].values[data.bench_idx[nonzero]],
            "residual":  data.scores[nonzero] - gof.y_pred_mean[nonzero],
        })
        assert isinstance(residuals_per_benchmark_fig(df), go.Figure)


# ───────────────────────── Persistence ─────────────────────────────────────
class TestPersistence:
    def test_round_trip_pit(self, synth_trace, data, tmp_path):
        y_rep = posterior_predictive_mirt(synth_trace, data)
        gof = compute_gof(y_rep, data)
        out = tmp_path / "pit.csv"
        save_pit(gof.pit, data, out)
        assert out.exists()
        df = pd.read_csv(out)
        assert len(df) == (~boundary_mask(data)).sum()
        assert set(["obs_idx", "model", "benchmark", "score", "pit"]).issubset(df.columns)


# ───────────────────────── CLI ─────────────────────────────────────────────
class TestCLI:
    def test_help_exits_clean(self):
        py = sys.executable
        r = subprocess.run(
            [py, str(PROJECT_ROOT / "2_fit.py"), "--help"],
            cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=60,
        )
        assert r.returncode == 0, r.stderr
        assert "--preset" in r.stdout
        assert "--skip-sampling" in r.stdout
        assert "--drop-zero-scores" in r.stdout
        assert "--loading-prior" in r.stdout

    def test_unknown_arg_fails(self):
        py = sys.executable
        r = subprocess.run(
            [py, str(PROJECT_ROOT / "2_fit.py"), "--bogus-flag"],
            cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=60,
        )
        assert r.returncode != 0
        assert "unrecognized" in r.stderr or "error" in r.stderr.lower()

    def test_invalid_preset_fails(self):
        py = sys.executable
        r = subprocess.run(
            [py, str(PROJECT_ROOT / "2_fit.py"), "--preset", "nonsense"],
            cwd=str(PROJECT_ROOT),
            capture_output=True, text=True, timeout=60,
        )
        assert r.returncode != 0


# ───────────────────────── MIRT (multidimensional) ─────────────────────────
@pytest.fixture
def mirt_synth(data: ECIData):
    """Synthetic K=3 MIRT InferenceData with named dims (model/bench/latent).

    Function-scoped so tests can safely mutate posterior.attrs (the anchor
    metadata) without leaking across tests. Non-negative loadings, as the real
    model produces.
    """
    rng = np.random.default_rng(1)
    nc, nd, K = 2, 60, 3
    M, B = data.n_models, data.n_benchmarks
    A = np.abs(rng.normal(0.0, 0.5, (nc, nd, B, K)))
    theta = rng.normal(0.0, 1.0, (nc, nd, M, K))
    tau = np.abs(rng.normal(0.5, 0.1, (nc, nd, K)))
    D = rng.normal(0.0, 1.0, (nc, nd, B))
    # vary sigma_b per draw (a constant would give a degenerate 0/0 r-hat)
    sigma_b = np.abs(rng.normal(0.08, 0.01, (nc, nd, B)))
    return az.from_dict(
        posterior={"A": A, "theta": theta, "tau_A": tau, "D": D, "sigma_b": sigma_b},
        coords={"model": list(range(M)), "bench": list(range(B)),
                "latent": ["axis1", "axis2", "axis3"]},
        dims={"A": ["bench", "latent"], "theta": ["model", "latent"],
              "tau_A": ["latent"], "D": ["bench"], "sigma_b": ["bench"]},
    )


def _eta(A, th):
    """eta[s,b,m] = sum_k A[s,b,k] theta[s,m,k] — the rotation/permutation
    invariant linear predictor used to check that relabelings preserve the fit."""
    return np.einsum("sbk,smk->sbm", A, th)


def _logp(model):
    """Total logp at the deterministic initial point (transformed-space zeros)."""
    return float(model.compile_logp()(model.initial_point()))


class TestMIRT:
    # ── model construction ──────────────────────────────────────────────
    def test_build_creates_expected_vars(self, data):
        model = build_mirt_model(data, K=3)
        names = ({v.name for v in model.free_RVs}
                 | {d.name for d in model.deterministics})
        for v in ["A", "theta", "tau_A", "D", "sigma_b"]:
            assert v in names, f"missing {v}"

    def test_anchors_zero_off_axis_loadings(self, data):
        """The off-by-one regression test: anchors must pin the INTENDED
        benchmark (by position, not the 1-indexed benchmark_idx column)."""
        bench = data.blookup["benchmark"].tolist()
        anchors = {bench[5]: 0, bench[10]: 1, bench[20]: 2}
        model = build_mirt_model(data, K=3, anchors=anchors)
        A_draws = pm.draw(model["A"], draws=4, random_seed=0)   # (4, B, K), prior
        for bname, ax in anchors.items():
            bi = bench.index(bname)
            off = [k for k in range(3) if k != ax]
            assert np.all(A_draws[:, bi, off] == 0.0), f"{bname} leaked off-axis"

    def test_anchor_unknown_benchmark_raises(self, data):
        with pytest.raises(ValueError):
            build_mirt_model(data, K=3, anchors={"NotARealBenchmark": 0})

    def test_ceiling_noise_vars_and_default_off(self, data):
        """ceiling_noise adds the per-benchmark ceiling pair; the default model
        carries neither (off = the plain 2PL, guarded by the golden logp)."""
        m = build_mirt_model(data, K=1, ceiling_noise=True)
        names = ({v.name for v in m.free_RVs} | {d.name for d in m.deterministics})
        assert {"ceiling_gap", "ceiling_d"} <= names
        d_draws = pm.draw(m["ceiling_d"], draws=4, random_seed=0)
        assert d_draws.shape == (4, data.n_benchmarks)
        assert np.all((d_draws > 0) & (d_draws <= 1))
        m0 = build_mirt_model(data, K=1)
        names0 = ({v.name for v in m0.free_RVs} | {d.name for d in m0.deterministics})
        assert not ({"ceiling_gap", "ceiling_d"} & names0)

    def test_ceiling_noise_gap_is_noise_sized(self, data_all):
        """the estimated ceiling stays a noise-sized gap under 1, never a wall."""
        m = build_mirt_model(data_all, K=1, ceiling_noise=True)
        d = pm.draw(m["ceiling_d"], draws=400, random_seed=0)
        assert d.shape == (400, data_all.n_benchmarks)
        assert np.all(d <= 1.0 + 1e-12)
        # noise-sized: the median gap is ~0.034 (Beta(1,20): P(gap > 0.10) = 0.122)
        gap = 1.0 - d
        assert 0.02 < np.median(gap) < 0.05
        assert (gap > 0.10).mean() < 0.15

    def test_invalid_loading_prior_raises(self, data):
        with pytest.raises(ValueError):
            build_mirt_model(data, K=3, loading_prior="nonsense")

    # ── known_se: instrument precision split off from sigma_b ──────────────
    def test_known_se_all_inf_reduces_exactly(self, data):
        """A cell with no reported stderr has n_eff = inf, whose relative
        variance 1/n_eff is exactly 0, so an all-inf fit IS the current model."""
        import dataclasses
        d = dataclasses.replace(data, n_eff=np.full(data.n_obs, np.inf))
        m0 = build_mirt_model(d, K=3, loading_prior="normal")
        m1 = build_mirt_model(d, K=3, loading_prior="normal", known_se=True)
        np.testing.assert_allclose(_logp(m1), _logp(m0), rtol=1e-12)
        assert ({v.name for v in m0.free_RVs}
                == {v.name for v in m1.free_RVs}), "known_se added a free RV"

    def test_known_se_formula_one_cell(self, data):
        """The logp shift is exactly the Beta re-weighting the split implies.

        At the initial point theta and D_z are zero, so eta = 0 and mu = 0.5 on
        every row; sigma_b is at its own initial value. That makes the whole
        shift computable in scipy: relative variances add, 1/(1+phi_n) =
        1/n_eff + 4*sigma_b^2, and only the cells with a reported stderr move.
        """
        from scipy.stats import beta as beta_dist
        m0 = build_mirt_model(data, K=3, loading_prior="normal")
        m1 = build_mirt_model(data, K=3, loading_prior="normal", known_se=True)
        point = m0.initial_point()
        assert np.all(point["theta_t_zerosum__"] == 0) and np.all(point["D_z"] == 0)
        y = m0.rvs_to_values[m0["obs"]].eval()          # clipped observed scores

        s2 = np.exp(point["sigma_b_log__"])[data.bench_idx] ** 2
        phi_b = 1.0 / (4.0 * s2) - 1.0
        phi_n = 1.0 / (1.0 / data.n_eff + 4.0 * s2) - 1.0
        moved = np.isfinite(data.n_eff)
        assert moved.any() and not moved.all(), "need a mix of measured cells"
        np.testing.assert_allclose(phi_n[~moved], phi_b[~moved], rtol=0)
        assert np.all(phi_n[moved] < phi_b[moved])      # noise added, never removed

        want = float(np.sum(beta_dist.logpdf(y, 0.5 * phi_n, 0.5 * phi_n)
                            - beta_dist.logpdf(y, 0.5 * phi_b, 0.5 * phi_b)))
        np.testing.assert_allclose(_logp(m1) - _logp(m0), want, rtol=1e-9)

    def test_known_se_requires_n_eff(self, data):
        import dataclasses
        with pytest.raises(ValueError, match="n_eff"):
            build_mirt_model(dataclasses.replace(data, n_eff=None), K=3,
                             known_se=True)
        with pytest.raises(ValueError, match="shape"):
            build_mirt_model(dataclasses.replace(data, n_eff=np.full(3, np.inf)),
                             K=3, known_se=True)

    # ── pooled_noise: the sigma_b population is learned, not fixed ──────────
    def test_pooled_noise_hierarchy(self, data):
        """The flag replaces the free per-benchmark sigma_b with a non-centered
        hierarchy, and keeps "sigma_b" as a bench-dimensioned deterministic
        because identified r-hat, PPC and the dashboard all index it by name.
        """
        hyper = {"sigma_b_mu", "sigma_b_tau", "sigma_b_z"}
        on = build_mirt_model(data, K=3, loading_prior="normal", pooled_noise=True)
        assert hyper <= {v.name for v in on.free_RVs}
        assert "sigma_b" not in {v.name for v in on.free_RVs}
        [det] = [d for d in on.deterministics if d.name == "sigma_b"]
        assert on.named_vars_to_dims["sigma_b"] == ("bench",)

        off = build_mirt_model(data, K=3, loading_prior="normal")
        assert "sigma_b" in {v.name for v in off.free_RVs}
        assert not (hyper & {v.name for v in off.free_RVs})

        # z_b = 0 and mu_s = log 0.05 at the initial point, so the hierarchy
        # starts exactly where the fixed prior does. replace_rvs_by_values is
        # what makes the deterministic a function of the point (compiling the
        # raw graph would leave the RVs random).
        [graph] = on.replace_rvs_by_values([det])
        fn = pytensor.function(on.value_vars, graph, on_unused_input="ignore")
        point = on.initial_point()
        sigma_b = fn(*[point[v.name] for v in on.value_vars])
        np.testing.assert_allclose(sigma_b, np.full(data.n_benchmarks, 0.05),
                                   rtol=1e-12)

    def test_align_factor_signs_fixes_flips_and_preserves_eta(self):
        """Random per-draw ± flips of (A, theta) columns are resolved to one
        consistent orientation, and eta = A·theta^T is unchanged draw-by-draw."""
        rng = np.random.default_rng(0)
        S, B, M, K = 8, 12, 9, 3
        A0 = np.broadcast_to(rng.normal(0, 1, (1, B, K)), (S, B, K))
        th = rng.normal(0, 1, (S, M, K))
        flips = rng.choice([-1.0, 1.0], size=(S, K))
        A_flip = A0 * flips[:, None, :]
        th_flip = th * flips[:, None, :]
        A2, th2, ref = align_factor_signs(A_flip, th_flip)
        assert np.allclose(_eta(A2, th2), _eta(A_flip, th_flip))   # invariant
        for k in range(K):
            assert np.all(A2[:, ref[k], k] > 0)                    # convention holds
        assert np.allclose(A2, A2[:1])                             # flips resolved

    def test_trace_loading_prior_detection_and_prepare_fit_gate(self, data, mirt_synth):
        """A signed-tagged trace must NOT be promax-rotated on top by
        prepare_fit (per-draw alignment already owns the identification)."""
        assert trace_loading_prior(mirt_synth) == ""
        mirt_synth.posterior.attrs["mirt_loading_prior"] = "signed"
        assert trace_loading_prior(mirt_synth) == "signed"
        view = prepare_fit(mirt_synth, data)
        assert not view.rotated

    # ── signed-free loading prior + per-draw alignment ────────────────────
    def test_signed_build_creates_expected_vars(self, data):
        """'signed': iid Normal loadings × ONE shared learned scale — signed
        (both signs in the prior), no per-axis tau, no ordering, no mask."""
        model = build_mirt_model(data, K=3, loading_prior="signed")
        free = {v.name for v in model.free_RVs}
        det = {d.name for d in model.deterministics}
        for v in ["A", "theta", "tau_A", "D", "sigma_b"]:
            assert v in (free | det), f"missing {v}"
        assert "tau_A_signed" in free and "A_z" in free
        assert "tau_A" in det                       # broadcast, not per-axis RV
        A, tau = pm.draw([model["A"], model["tau_A"]], draws=20, random_seed=0)
        assert (A < 0).any() and (A > 0).any()      # signed
        assert np.allclose(tau, tau[..., :1])       # one scale shared across axes
        assert float(np.asarray(tau).std()) > 1e-6  # learned, not a constant

    def test_signed_mutually_exclusive_with_anchors_and_blocks(self, data):
        """Anchors break the exact rotation symmetry outright — hard error.
        The ordered human/lineage blocks only SOFTLY break it (they weakly
        anchor the orientation), so that combination is allowed with a loud
        warning: per-draw alignment becomes an approximation."""
        bench0 = data.blookup["benchmark"].tolist()[0]
        with pytest.raises(ValueError):
            build_mirt_model(data, K=3, loading_prior="signed",
                             anchors={bench0: 0})
        with pytest.warns(UserWarning, match="APPROXIMATE"):
            build_mirt_model(data, K=3, loading_prior="signed",
                             human_order=HUMAN_ORDER)

    def test_match_columns_recovers_signed_permutation(self):
        """Both matchers (exact Hungarian and greedy) undo a known signed
        permutation of the columns exactly."""
        rng = np.random.default_rng(0)
        L = rng.normal(size=(12, 3))
        perm0, signs0 = np.array([2, 0, 1]), np.array([1.0, -1.0, 1.0])
        P0 = rotation._signed_perm_matrix(perm0, signs0)
        for matcher in (rotation._match_columns, rotation._greedy_match_columns):
            perm, signs = matcher(L @ P0, L)
            P = rotation._signed_perm_matrix(perm, signs)
            assert np.allclose((L @ P0) @ P, L, atol=1e-12)

    def test_alignment_recovers_rotation(self):
        """The core contract: draws that are one true loading matrix under
        random per-draw rotations come back to that matrix (up to a signed
        permutation) under every method, with eta untouched. WOP gets a looser
        bar — Procrustes converges to A consistent orientation, not necessarily
        the simple-structure one (that difference is why we compare methods)."""
        rng = np.random.default_rng(42)
        S, B, M, K = 40, 20, 30, 3
        L0 = np.zeros((B, K))
        for k in range(K):
            L0[k * 6:(k + 1) * 6, k] = rng.uniform(0.8, 1.5, 6)
        th0 = rng.normal(size=(M, K))
        A_d = np.empty((S, B, K)); th_d = np.empty((S, M, K))
        for s in range(S):
            q, r = np.linalg.qr(rng.normal(size=(K, K)))
            R = q * np.sign(np.diag(r))
            A_d[s] = (L0 + rng.normal(0, 0.03, (B, K))) @ R
            th_d[s] = (th0 + rng.normal(0, 0.03, (M, K))) @ R
        for method, bar in [("varimax", 0.95), ("wop", 0.85),
                            ("matchalign", 0.95), ("promax", 0.95),
                            ("geomin", 0.95)]:
            res = align_rotations(A_d, th_d, method=method)
            assert np.allclose(_eta(res.A, res.theta), _eta(A_d, th_d), atol=1e-6)
            Am = res.A.mean(0)
            perm, signs = rotation._match_columns(Am, L0)
            Am = Am @ rotation._signed_perm_matrix(perm, signs)
            for k in range(K):
                c = abs(np.corrcoef(Am[:, k], L0[:, k])[0, 1])
                assert c > bar, f"{method} axis{k + 1}: |corr|={c:.3f} <= {bar}"

    def test_prepare_fit_signed_gates_alignment(self, mirt_synth, data):
        """A signed-tagged trace: per-draw alignment instead of promax-on-mean
        (rotated=False), and the aligned view predicts identically to the raw
        draws (eta preserved through the whole gate)."""
        mirt_synth.posterior.attrs["mirt_loading_prior"] = "signed"
        view = prepare_fit(mirt_synth, data)
        assert not view.rotated and view.A is not None and view.K == 3
        post = mirt_synth.posterior
        S = post.sizes["chain"] * post.sizes["draw"]
        A_raw = post["A"].values.reshape(S, data.n_benchmarks, 3)
        th_raw = post["theta"].values.reshape(S, data.n_models, 3)
        assert np.allclose(_eta(view.A, view.theta), _eta(A_raw, th_raw), atol=1e-6)

    def test_alignment_report_runs(self, mirt_synth, data):
        """End-to-end report on a small synthetic trace: all four methods, the
        sign-confidence counts, and the method-agreement table come back."""
        mirt_synth.posterior.attrs["mirt_loading_prior"] = "signed"
        rep = alignment_report(mirt_synth, data,
                               methods=("varimax", "matchalign"))
        assert set(rep["methods"]) == {"varimax", "matchalign"}
        for m, entry in rep["methods"].items():
            assert set(entry["sign_counts"]) == {"axis1", "axis2", "axis3"}
            assert np.isfinite(entry["aligned_max_rhat_A"])
            assert "sign_confident" in entry["loadings"].columns
        assert set(rep["agreement"].columns) == \
            {"method_a", "method_b", "axis", "abs_corr"}
        assert len(rep["agreement"]) == 3          # 1 method pair × 3 axes

    def test_plt_triangular_pattern(self, data):
        """PLT identification: founder r has exact zeros ABOVE the diagonal,
        a strictly POSITIVE diagonal loading (sign folded, HalfNormal-shaped),
        and free signed cells below; non-founder rows keep the signed prior."""
        bl = data.blookup["benchmark"].tolist()
        founders = bl[:3]                                   # axis r <- bl[r]
        for prior in ("signed",):
            model = build_mirt_model(data, K=3, loading_prior=prior,
                                     plt_founders=founders)
            A = pm.draw(model["A"], draws=200, random_seed=0)   # (200, B, K)
            for r in range(3):
                assert np.all(A[:, r, r + 1:] == 0.0), (prior, r)   # upper zeros
                assert np.all(A[:, r, r] > 0.0), (prior, r)         # positive diag
            assert (A[:, 2, :2] < 0).any(), prior           # below-diag stays signed
            assert (A[:, 3:, :] < 0).any(), prior           # non-founders signed

    def test_plt_validation(self, data):
        """PLT refuses: non-signed priors, anchors alongside, a founder count
        != K, duplicate founders, and unknown benchmarks."""
        bl = data.blookup["benchmark"].tolist()
        ok = bl[:3]
        with pytest.raises(ValueError, match="signed"):
            build_mirt_model(data, K=3, loading_prior="normal", plt_founders=ok)
        with pytest.raises(ValueError, match="exactly K"):
            build_mirt_model(data, K=3, loading_prior="signed",
                             plt_founders=bl[:2])
        with pytest.raises(ValueError, match="distinct"):
            build_mirt_model(data, K=3, loading_prior="signed",
                             plt_founders=[bl[0], bl[0], bl[1]])
        with pytest.raises(ValueError, match="not in data"):
            build_mirt_model(data, K=3, loading_prior="signed",
                             plt_founders=[bl[0], bl[1], "NotARealBenchmark"])

    @pytest.mark.slow
    def test_signed_tiny_sample_runs(self, data):
        """The signed model compiles and NUTS runs (finite logp everywhere)."""
        with build_mirt_model(data, K=2, loading_prior="signed"):
            trace = pm.sample(
                draws=5, tune=5, chains=1, cores=1,
                progressbar=False, random_seed=0,
                return_inferencedata=True,
                compute_convergence_checks=False,
            )
        post = trace.posterior
        assert post["A"].shape[-2:] == (data.n_benchmarks, 2)
        assert np.isfinite(post["A"].values).all()

    # ── normal loading scale (the "no ARD" for confirmatory fits) ────────
    def test_normal_loading_scale(self, data):
        """'normal' = ONE learned loading scale broadcast to all axes: not
        per-axis (not ARD), but learned (not a constant)."""
        m = build_mirt_model(data, K=3, loading_prior="normal")
        free = {v.name for v in m.free_RVs}
        det = {d.name for d in m.deterministics}
        assert "tau_A_normal" in free          # one learned scalar
        assert "tau_A" in det and "tau_A" not in free   # broadcast, not per-axis RV
        assert "tau_A" not in free             # (no per-axis ARD vector)
        # equal across axes within a draw (shared), but varies across draws (learned)
        tau = pm.draw(m["tau_A"], draws=50, random_seed=0)   # (50, 3)
        assert np.allclose(tau, tau[:, :1])    # all axes share one value
        assert float(tau.std()) > 1e-6         # learned, not a constant

    def test_normal_composes_with_anchors(self, data):
        """'normal' is just a scale choice — it composes with anchors
        (the confirmatory use case)."""
        bench = data.blookup["benchmark"].tolist()
        m = build_mirt_model(data, K=3, loading_prior="normal",
                             anchors={bench[0]: 0, bench[1]: 1})
        assert "tau_A_normal" in {v.name for v in m.free_RVs}

    # ── canonicalization ────────────────────────────────────────────────
    def test_rank_track_orders_by_tau_and_preserves_eta(self):
        rng = np.random.default_rng(0)
        S, B, K = 5, 4, 3
        A = np.abs(rng.normal(size=(S, B, K)))
        th = rng.normal(size=(S, 6, K))
        tau = np.tile([0.2, 0.9, 0.5], (S, 1))            # desc order: 1,2,0
        A2, th2, tau2 = canonicalize_factors(A, th, tau)
        assert np.allclose(tau2, np.tile([0.9, 0.5, 0.2], (S, 1)))
        assert np.allclose(_eta(A2, th2), _eta(A, th))    # relabel preserves fit

    def test_rank_track_sorts_each_draw_independently(self):
        """The sort key VARIES across draws here, so a regression from the
        per-draw argsort to one global permutation (which the constant-key
        fixtures above cannot distinguish) fails this test. This is the
        identification path the flagship (normal-prior, rank-tracked) fit
        rides, so it gets its own lock."""
        rng = np.random.default_rng(7)
        S, B, M, K = 6, 4, 5, 3
        A = np.abs(rng.normal(size=(S, B, K)))
        th = rng.normal(size=(S, M, K))
        tau = np.abs(rng.normal(size=(S, K))) + 0.1        # distinct order per draw
        A2, th2, tau2 = canonicalize_factors(A, th, tau)
        assert np.allclose(_eta(A2, th2), _eta(A, th))
        for s in range(S):                                 # each draw on its own
            order = np.argsort(-tau[s])
            assert np.array_equal(tau2[s], tau[s][order])
            assert np.array_equal(A2[s], A[s][:, order])
            assert np.array_equal(th2[s], th[s][:, order])

    def test_rank_track_false_is_identity(self):
        rng = np.random.default_rng(0)
        A = np.abs(rng.normal(size=(3, 4, 2)))
        th = rng.normal(size=(3, 5, 2))
        tau = np.abs(rng.normal(size=(3, 2)))
        A2, th2, tau2 = canonicalize_factors(A, th, tau, rank_track=False)
        assert np.array_equal(A2, A) and np.array_equal(th2, th) and np.array_equal(tau2, tau)

    def test_constant_tau_falls_back_to_loading_energy(self):
        S, B, K = 4, 5, 3
        th = np.random.default_rng(0).normal(size=(S, 6, K))
        tau = np.full((S, K), 0.5)                         # no tau ordering signal
        A = np.zeros((S, B, K))
        A[:, :, 0] = 0.1; A[:, :, 1] = 1.0; A[:, :, 2] = 0.5   # energy order: 1,2,0
        A2, _, _ = canonicalize_factors(A, th, tau)
        assert np.all(A2[:, :, 0] == 1.0)
        assert np.all(A2[:, :, 1] == 0.5)
        assert np.all(A2[:, :, 2] == 0.1)

    # ── rotation ────────────────────────────────────────────────────────
    def test_promax_preserves_predictions(self):
        rng = np.random.default_rng(2)
        B, K = 12, 3
        L = np.abs(rng.normal(size=(B, K)))
        Tload, Ttheta, Phi = promax_rotate(L)
        th = rng.normal(size=(40, K))
        assert np.allclose((L @ Tload) @ (th @ Ttheta).T, L @ th.T, atol=1e-8)
        assert np.allclose(np.diag(Phi), 1.0, atol=1e-6)   # Phi is a correlation matrix

    def test_apply_rotation_preserves_eta(self):
        rng = np.random.default_rng(4)
        S, B, M, K = 6, 10, 8, 3
        A = np.abs(rng.normal(size=(S, B, K)))
        th = rng.normal(size=(S, M, K))
        Tload, Ttheta, Phi = promax_rotate(A.mean(0))
        A2, th2, _ = apply_rotation(A, th, Phi, Tload, Ttheta)
        assert np.allclose(_eta(A2, th2), _eta(A, th), atol=1e-6)

    # ── informedness filter ─────────────────────────────────────────────
    def test_informed_mask_threshold(self):
        rng = np.random.default_rng(3)
        S, M, K = 300, 3, 2
        th = np.empty((S, M, K))
        th[:, 0, :] = rng.normal(0, 0.1, (S, K))           # tight -> informed
        th[:, 1, :] = rng.normal(0, 1.0, (S, K))           # wide  -> not
        th[:, 2, :] = rng.normal(0, 0.5, (S, K))           # borderline
        mask = mirt_informed_mask(th, sd_cap=0.6)
        assert mask[0].all()
        assert not mask[1].any()

    # ── anchor metadata (Option C) ──────────────────────────────────────
    def test_trace_anchors_detection(self, mirt_synth):
        assert trace_anchors(mirt_synth) == {}
        mirt_synth.posterior.attrs["mirt_anchors"] = json.dumps({"FrontierMath": 0})
        assert trace_anchors(mirt_synth) == {"FrontierMath": 0}

    def test_factors_skip_ranktrack_when_anchored(self, mirt_synth):
        mirt_synth.posterior.attrs["mirt_anchors"] = json.dumps({"FrontierMath": 0})
        _, _, tau_auto = mirt_factors_from_trace(mirt_synth)              # auto: skip
        _, _, tau_explicit = mirt_factors_from_trace(mirt_synth, rank_track=False)
        assert np.array_equal(tau_auto, tau_explicit)

    # ── unified fit view-model (prepare_fit) — the single frame contract ──
    def test_prepare_fit_positive_prior_never_rotates(self, mirt_synth, data):
        """Unanchored K≥2 fit on a non-negative loading prior → the raw
        rank-tracked frame, whatever the trace says. Positivity pins the
        rotation, so loadings and abilities pass through
        mirt_factors_from_trace untouched (the permutation relabel is kept, no
        rotation on top), Phi is the raw ability correlation, and a display
        rotation recorded on the trace changes nothing."""
        mirt_synth.posterior.attrs["mirt_display_rotation"] = "promax"
        view = prepare_fit(mirt_synth, data)
        assert not view.is_nc and not view.anchored and not view.rotated
        assert view.K == 3 and view.A is not None
        S = (mirt_synth.posterior.sizes["chain"] * mirt_synth.posterior.sizes["draw"])
        assert view.A.shape == (S, data.n_benchmarks, 3)
        assert view.theta.shape == (S, data.n_models, 3)
        A_rt, th_rt, _ = mirt_factors_from_trace(mirt_synth)
        assert np.allclose(view.A, A_rt)
        assert np.allclose(view.theta, th_rt)
        assert np.allclose(view.Phi, np.corrcoef(th_rt.mean(0).T))
        assert np.allclose(view.Phi, view.Phi_raw)

    def test_prepare_fit_anchored_no_rotation(self, mirt_synth, data):
        """Anchored fit → axes pinned, NO promax; Phi is the raw ability correlation."""
        bench = data.blookup["benchmark"].tolist()
        mirt_synth.posterior.attrs["mirt_anchors"] = json.dumps({bench[0]: 0})
        view = prepare_fit(mirt_synth, data)
        assert view.anchored and not view.rotated
        assert np.allclose(view.Phi, view.Phi_raw)

    def test_prepare_fit_nc_has_no_loadings(self, mirt_nc_synth, data_all):
        """Non-comp fit → is_nc, no loadings; require_A() fails loudly (not None)."""
        view = prepare_fit(mirt_nc_synth, data_all)
        assert view.is_nc and view.A is None and not view.rotated and view.K == 3
        with pytest.raises(ValueError):
            view.require_A()

    # ── identified convergence ──────────────────────────────────────────
    def test_identified_rhat_runs(self, mirt_synth, data):
        out = mirt_identified_rhat(mirt_synth, data, n_obs_sample=50)
        for key in ["eta_max_rhat", "eta_mean_rhat", "D_max_rhat", "sigma_b_max_rhat"]:
            assert key in out and np.isfinite(out[key])

    # ── lineage prior ───────────────────────────────────────────────────
    def test_lineage_structure_contiguous(self, data):
        """B/C incidence is well-formed: each node has one founder, founder rows
        carry no deltas, every delta column is used, and n_deltas closes."""
        lin = build_lineage_structure(data.mlookup)
        assert lin is not None and lin.n_chains >= 2
        assert lin.n_deltas == lin.n_nodes - lin.n_chains
        assert lin.B.shape == (lin.n_nodes, lin.n_chains)
        assert lin.C.shape == (lin.n_nodes, lin.n_deltas)
        assert np.allclose(lin.B.sum(axis=1), 1.0)              # one founder per node
        assert (lin.C.sum(axis=1) == 0).sum() == lin.n_chains   # founders carry no delta
        assert (lin.C.sum(axis=0) > 0).all()                    # no orphan delta column
        assert len(lin.row_idx) == len(lin.node_idx) == len(lin.offset_group)

    def test_lineage_parent_branches_off_the_spine(self, data, tmp_path):
        """An explicit `parent` moves a node off the spine: it inherits its branch
        point's path, and the spine steps straight over it instead of through it.

        Built as a 4-node chain A -> B -> C, plus X parented on B. Without the
        parent column X would splice in by date between B and C, so C's step
        would start at X. With it, C steps from B and X hangs off B.
        """
        rows = [("mA", "V", "ch", "A", None,  "2024-01-01"),
                ("mB", "V", "ch", "B", None,  "2024-02-01"),
                ("mX", "V", "ch", "X", "B",   "2024-02-15"),
                ("mC", "V", "ch", "C", None,  "2024-03-01")]
        cols = ["raw_string", "vendor", "chain", "node", "parent", "node_date"]
        df = pd.DataFrame(rows, columns=cols).assign(variant="bare", in_chain="yes", n_obs=1)
        ml = pd.DataFrame({"model": [r[0] for r in rows],
                           "model_idx": range(1, len(rows) + 1)})

        p = tmp_path / "branched.csv"
        df.to_csv(p, index=False)
        lin = build_lineage_structure(ml, csv_path=p)
        assert (lin.n_nodes, lin.n_chains, lin.n_deltas) == (4, 1, 3)
        depth = lin.C.sum(axis=1)                       # deltas back to the founder
        pos = {n: i for i, n in enumerate(["A", "B", "X", "C"])}   # date order
        assert depth[pos["A"]] == 0                     # founder
        assert depth[pos["B"]] == 1
        assert depth[pos["X"]] == 2                     # B + its own branch step
        assert depth[pos["C"]] == 2                     # B + its own step, NOT via X
        # C's incoming step spans B -> C (29 d), not X -> C (15 d)
        c_only = np.where((lin.C[pos["C"]] == 1) & (lin.C[pos["B"]] == 0))[0]
        assert len(c_only) == 1
        assert lin.delta_dt[c_only[0]] == pytest.approx(29 / 365.25, rel=1e-9)

        q = tmp_path / "flat.csv"
        df.drop(columns=["parent"]).to_csv(q, index=False)
        flat = build_lineage_structure(ml, csv_path=q)
        assert flat.C.sum(axis=1)[pos["C"]] == 3        # spliced: A -> B -> X -> C

    def test_lineage_parent_rejects_backwards_branch(self, data, tmp_path):
        """A branch must post-date its branch point; the BM step needs dt > 0."""
        rows = [("mA", "V", "ch", "A", None, "2024-01-01"),
                ("mB", "V", "ch", "B", None, "2024-03-01"),
                ("mX", "V", "ch", "X", "B",  "2024-02-01")]
        cols = ["raw_string", "vendor", "chain", "node", "parent", "node_date"]
        df = pd.DataFrame(rows, columns=cols).assign(variant="bare", in_chain="yes", n_obs=1)
        ml = pd.DataFrame({"model": [r[0] for r in rows], "model_idx": [1, 2, 3]})
        p = tmp_path / "backwards.csv"
        df.to_csv(p, index=False)
        with pytest.raises(ValueError, match="is not after its parent"):
            build_lineage_structure(ml, csv_path=p)

    def test_lineage_node_dropout_renumbers(self, data):
        """A node whose only variant is absent from the data must vanish without
        leaving a gap in C — the renumber path. Drop one full mid-chain node."""
        lin = build_lineage_structure(data.mlookup)
        # find a chain with >=3 nodes, drop the rows of its middle node
        m = pd.read_csv(LINEAGE_MAP)
        m = m[m["in_chain"].astype(str).str.lower() == "yes"]
        m = m[m["raw_string"].isin(data.mlookup["model"])]
        c = next(c for c, g in m.groupby("chain") if g["node"].nunique() >= 3)
        nodes = m[m["chain"] == c].sort_values("node_date")["node"].unique()
        drop = m[(m["chain"] == c) & (m["node"] == nodes[1])]["raw_string"]
        reduced = data.mlookup[~data.mlookup["model"].isin(drop)]
        lin2 = build_lineage_structure(reduced)
        assert lin2.n_nodes == lin.n_nodes - 1
        assert lin2.n_deltas == lin2.n_nodes - lin2.n_chains
        assert (lin2.C.sum(axis=0) > 0).all()                   # still gap-free

    def test_lineage_none_leaves_default_theta(self, data):
        """Without a lineage, theta is the original single ZeroSumNormal."""
        m = build_mirt_model(data, K=3)
        free = {v.name for v in m.free_RVs}
        assert "theta_t" in free
        assert not any(n.startswith("lin_") for n in free)

    def test_lineage_block_builds_and_pools_aliases(self, data):
        """The lineage block adds its RVs, theta stays a (model,latent)
        deterministic, and rows sharing a (node,variant) group get identical
        theta (alias pooling)."""
        lin = build_lineage_structure(data.mlookup)
        m = build_mirt_model(data, K=3, lineage=lin)
        free = {v.name for v in m.free_RVs}
        for v in ["lin_drift", "lin_spread", "lin_delta_z",
                  "lin_offset_z", "lin_offset_sd"]:
            assert v in free, f"missing {v}"
        # Founder levels are sliced out of the shared ZeroSumNormal, so they are
        # a deterministic, not a free RV of their own.
        assert "lin_base" in {v.name for v in m.deterministics}
        key = list(zip(lin.node_idx.tolist(), lin.offset_group.tolist()))
        pooled = next(k for k in key if key.count(k) > 1)       # a shared group
        rows = lin.row_idx[[i for i, k in enumerate(key) if k == pooled]]
        th = pm.draw(m["theta"], draws=2, random_seed=0)        # (2, M, K)
        assert np.allclose(th[:, rows, :] - th[:, rows[:1], :], 0.0)

    def test_lineage_dt_matches_release_gaps(self, data):
        """delta_dt is the release gap in years for the step each delta column
        bridges, and delta_chain says which chain that column belongs to.
        Recomputed here straight from the CSV, independent of the builder.

        A node's predecessor is its explicit `parent` when it has one, else the
        previous SPINE node by date — parented nodes hang off the spine and so
        never separate two spine releases.
        """
        lin = build_lineage_structure(data.mlookup)
        m = pd.read_csv(LINEAGE_MAP).drop_duplicates("raw_string")
        m = m[m["in_chain"].astype(str).str.lower() == "yes"]
        m = m[m["raw_string"].isin(set(data.mlookup["model"]))].copy()
        m["date"] = pd.to_datetime(m["node_date"], errors="coerce")
        if "parent" not in m.columns:
            m["parent"] = pd.NA
        assert lin.delta_dt.shape == (lin.n_deltas,)
        assert (lin.delta_dt > 0).all(), "a step has a non-positive time gap"
        for ci, chain in enumerate(lin.chain_names):
            nd = (m[m["chain"] == chain][["node", "date", "parent"]]
                  .drop_duplicates("node").sort_values("date"))
            date_of = dict(zip(nd["node"], nd["date"]))
            spine = nd[nd["parent"].isna()]["node"].tolist()
            want = []
            for node, parent in zip(nd["node"], nd["parent"]):
                pred = parent if pd.notna(parent) else (
                    spine[spine.index(node) - 1] if node in spine and spine.index(node) else None)
                if pred is not None:
                    want.append((date_of[node] - date_of[pred]).days / 365.25)
            got = lin.delta_dt[lin.delta_chain == ci]
            np.testing.assert_allclose(sorted(got), sorted(want), rtol=1e-12)

    def test_lineage_bm_scales_steps_by_time(self, data):
        """lineage_bm=True: the shared drift becomes a rate per year (no
        per-family hierarchy) and a step's prior sd grows as sqrt(dt) — so the
        step across the longest gap is more variable than the step across the
        shortest."""
        lin = build_lineage_structure(data.mlookup)
        m = build_mirt_model(data, K=2, lineage=lin, lineage_bm=True)
        free = {v.name for v in m.free_RVs}
        assert {"lin_drift", "lin_spread"} <= free
        assert not {n for n in free if n.startswith("lin_rate")}
        assert "lin_rate" not in {v.name for v in m.deterministics}
        # Recover psi[node] = theta[row] - tau_o * offset_z[group] from joint
        # draws; one representative row per node suffices (aliases pool).
        th, oz, tau = pm.draw([m["theta"], m["lin_offset_z"], m["lin_offset_sd"]],
                              draws=400, random_seed=0)
        row_of, grp_of = {}, {}
        for r, n, g in zip(lin.row_idx, lin.node_idx, lin.offset_group):
            row_of.setdefault(n, r)
            grp_of.setdefault(n, g)
        psi = np.stack([th[:, row_of[n], :] - tau[:, None] * oz[:, grp_of[n], :]
                        for n in range(lin.n_nodes)], axis=1)     # (draws, nodes, K)
        chain_of = lin.B.argmax(axis=1)
        same_chain = chain_of[1:] == chain_of[:-1]
        # Delta columns are built chain-block by chain-block in the same order
        # as the node pairs above, so delta_dt aligns with steps directly.
        steps = (psi[:, 1:, :] - psi[:, :-1, :])[:, same_chain, :]
        dt = lin.delta_dt
        assert steps.shape[1] == dt.size
        assert (np.diff(lin.delta_chain) >= 0).all()
        sd = steps.std(axis=0).mean(axis=1)                        # per step, over axes
        lo, hi = dt.argmin(), dt.argmax()
        assert dt[hi] / dt[lo] > 5, "fixture lacks a wide enough gap spread"
        assert sd[hi] > sd[lo], f"sd did not grow with dt: {sd[lo]:.3f} vs {sd[hi]:.3f}"

    def test_lineage_bm_requires_lineage(self, data):
        with pytest.raises(ValueError, match="lineage_bm"):
            build_mirt_model(data, K=2, lineage_bm=True)

    def test_structured_bases_share_one_zerosumnormal(self, data):
        """Human roots and chain founders are sliced out of ONE ZeroSumNormal
        that also holds the unstructured models, so neither has a private
        Normal base and the sum-to-zero spans every starting point. Its width
        must account for all three groups."""
        lin = build_lineage_structure(data.mlookup)
        m = build_mirt_model(data, K=3, human_order=HUMAN_ORDER, lineage=lin)
        free = {v.name for v in m.free_RVs}
        assert "theta_zsn_t" in free
        assert not {"theta_rest_t", "theta_t"} & free      # no separate anchor
        assert {"theta_h_base", "lin_base"} <= {v.name for v in m.deterministics}
        n_human = sum(t in set(data.mlookup["model"]) for t in HUMAN_ORDER)
        n_rest = data.n_models - n_human - len(lin.row_idx)
        zsn = m["theta_zsn_t"]
        # rest rows + one root per human tree + one founder per chain
        assert zsn.type.shape[1] > n_rest
        assert zsn.type.shape[1] <= n_rest + n_human + lin.n_chains

    def test_time_prior_shifts_theta_by_beta_times_year(self, data):
        """The time prior adds beta[k]*t[m] to theta and nothing else. Evaluated
        with time_beta = 1 on every axis, theta must equal the no-trend theta
        plus the covariate broadcast across axes — so a covariate applied to the
        wrong rows, transposed, or double-counted fails here. The golden-logp
        locks cannot see this: time_beta is 0 at the initial point."""
        from multiaxis_eci.data import release_time_covariate
        lin = build_lineage_structure(data.mlookup)
        t = release_time_covariate(data.mlookup, lin)
        kw = dict(K=3, human_order=HUMAN_ORDER, lineage=lin)
        off = build_mirt_model(data, **kw)
        on = build_mirt_model(data, **kw, time_t=t)
        assert "time_beta" not in {v.name for v in off.free_RVs}
        assert "time_beta" in {v.name for v in on.free_RVs}

        # ONE point drives both models, so the trend is the only thing that can
        # differ. replace_rvs_by_values makes theta a deterministic function of
        # that point; compiling the raw graph would leave the RVs random.
        def theta_at(m, point):
            [graph] = m.replace_rvs_by_values([m["theta"]])
            fn = pytensor.function(m.value_vars, graph, on_unused_input="ignore")
            return fn(*[point[v.name] for v in m.value_vars])

        point = on.initial_point(random_seed=0)
        th_off = theta_at(off, {k: v for k, v in point.items() if k != "time_beta"})
        point["time_beta"] = np.ones(3)
        th_on = theta_at(on, point)
        np.testing.assert_allclose(th_on, th_off + t[:, None], atol=1e-10)

    def test_time_prior_covariate_is_centered_and_chain_constant(self, data):
        """Centering keeps the trend out of the overall level the ZeroSumNormal
        pins; the founder-date rule keeps it constant within a chain, so lineage
        increments still measure within-chain climb. Undated rows sit at 0."""
        from multiaxis_eci.data import release_time_covariate
        lin = build_lineage_structure(data.mlookup)
        t = release_time_covariate(data.mlookup, lin)
        assert t.shape == (data.n_models,)
        assert abs(t.sum()) < 1e-8                      # dated rows are centered
        chain_of_row = lin.B[lin.node_idx].argmax(1)
        for c in range(lin.n_chains):
            rows = lin.row_idx[chain_of_row == c]
            assert np.ptp(t[rows]) == 0.0, "chained rows must share a founder date"
        # Humans have no release date by nature, so they keep today's prior center.
        names = data.mlookup["model"].tolist()
        for tier in HUMAN_ORDER:
            if tier in names:
                assert t[names.index(tier)] == 0.0

    def test_time_prior_rejects_misaligned_covariate(self, data):
        from multiaxis_eci.data import release_time_covariate
        with pytest.raises(ValueError, match="theta-row order"):
            release_time_covariate(data.mlookup.iloc[::-1])
        with pytest.raises(ValueError, match="one centered year per model row"):
            build_mirt_model(data, K=2, time_t=np.zeros(data.n_models + 1))

    def test_private_bases_keeps_the_zerosumnormal_to_unstructured_rows(self, data):
        """shared_base_zsn=False samples each human root and chain founder as a
        private Normal(0, 1) and narrows the ZeroSumNormal to the unstructured
        rows. Free-parameter count and initial-point logp are identical to the
        default, so the golden-logp locks cannot see the swap and this is the
        only check on the branch."""
        lin = build_lineage_structure(data.mlookup)
        m = build_mirt_model(data, K=3, human_order=HUMAN_ORDER, lineage=lin,
                             shared_base_zsn=False)
        free = {v.name for v in m.free_RVs}
        assert {"theta_h_base", "lin_base"} <= free   # sampled, not sliced
        n_human = sum(t in set(data.mlookup["model"]) for t in HUMAN_ORDER)
        n_rest = data.n_models - n_human - len(lin.row_idx)
        assert m["theta_zsn_t"].type.shape[1] == n_rest

    def test_human_prior_partial_order(self, data):
        """HUMAN_ORDER is a PARTIAL order (tier → parent map): along every
        parent chain child >= parent holds on every axis in every prior draw,
        while branch tiers — Top Performer vs Committee of Domain Experts —
        stay unordered (draws go both ways). (Original hard prior, restored
        2026-07-05 as the hypothesis-matrix reference condition.)"""
        m = build_mirt_model(data, K=2, human_order=HUMAN_ORDER)
        assert "delta_h" in {v.name for v in m.free_RVs}
        # Root levels are sliced out of the shared ZeroSumNormal.
        assert "theta_h_base" in {v.name for v in m.deterministics}
        names = data.mlookup.sort_values("model_idx")["model"].tolist()
        idx = {t: names.index(t) for t in HUMAN_ORDER if t in names}
        assert len(idx) >= 2, "fixture data lost its human tiers"
        th = pm.draw(m["theta"], draws=50, random_seed=0)   # (50, M, K)
        for child in idx:
            parent = HUMAN_ORDER[child]
            while parent is not None and parent not in idx:
                parent = HUMAN_ORDER[parent]                # contract absents
            if parent is not None:
                assert (th[:, idx[child], :] >= th[:, idx[parent], :] - 1e-9).all(), \
                    f"{child} dips below its ancestor {parent}"
        if "Top Performer" in idx and "Committee of Domain Experts" in idx:
            gap = th[:, idx["Top Performer"], :] - th[:, idx["Committee of Domain Experts"], :]
            assert (gap > 0).any() and (gap < 0).any()

    def test_human_prior_legacy_list_is_chain(self, data):
        """A plain list still means one monotone chain, weakest → strongest."""
        names = data.mlookup.sort_values("model_idx")["model"].tolist()
        chain = [t for t in ("Average Human", "Domain Expert", "Top Performer")
                 if t in names]
        if len(chain) < 2:
            pytest.skip("chain tiers not in fixture data")
        m = build_mirt_model(data, K=2, human_order=chain)
        th = pm.draw(m["theta"], draws=20, random_seed=0)
        rows = [names.index(t) for t in chain]
        assert (np.diff(th[:, rows, :], axis=1) >= -1e-9).all()

    # ── frontier forecasting / crossover math ────────────────────────────
    def test_crossover_math_on_synthetic_trend(self):
        """A monotone linear frontier recovers its slope, projects a below-cloud
        human tier to a PAST crossover (passed) and an above-cloud tier to a
        FUTURE date."""
        from multiaxis_eci.analysis import mirt_crossover_df, mirt_frontier_forecast

        rng = np.random.default_rng(0)
        S = 600
        names = [f"m{i}" for i in range(6)] + ["Average Human", "Top Performer"]
        n = len(names)
        is_human = np.array([m in ("Average Human", "Top Performer") for m in names])
        d = ECIData(
            scores=np.zeros(1), zero_score_mask=np.zeros(1, bool),
            model_idx=np.zeros(1, int), bench_idx=np.zeros(1, int),
            mlookup=pd.DataFrame({"model": names, "model_idx": np.arange(1, n + 1)}),
            blookup=pd.DataFrame({"benchmark": ["b"], "benchmark_idx": [1]}),
            n_models=n, n_benchmarks=1, n_obs=1, zero_diag_threshold=0.01,
            n_obs_per_model=np.full(n, 10), is_low_obs=np.zeros(n, bool),
            excluded_benchmarks=set(), is_human=is_human,
            bench_category=None, is_sota=np.zeros(n, bool))
        true = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 1.0, 3.5])   # slope 1/yr
        theta = np.repeat(true[None, :], S, axis=0)[:, :, None] \
            + rng.normal(0, 0.08, (S, n, 1))
        raw = pd.DataFrame({
            "model_version": [f"m{i}" for i in range(6)], "benchmark": ["b"] * 6,
            "release_date": pd.to_datetime(
                ["2023-01-01", "2023-07-01", "2024-01-01",
                 "2024-07-01", "2025-01-01", "2025-07-01"])})

        fc = mirt_frontier_forecast(theta, 0, d, raw, sd_cap=None, drop_low_obs=False)
        assert abs(float(np.median(fc.slope)) - 1.0) < 0.15    # ~1 logit/yr recovered
        cx = mirt_crossover_df(fc, theta, 0, d, axis_name="axis1",
                               today="2026-01-01")
        avg = cx[cx.tier == "Average Human"].iloc[0]
        top = cx[cx.tier == "Top Performer"].iloc[0]
        assert avg.status.startswith("passed")                 # 1.0 << frontier
        assert avg.p_passed_now > 0.99
        assert top.status == "future"                          # 3.5 not reached by 2026-01
        assert pd.Timestamp(top.crossover_date_median) > pd.Timestamp("2026-01-01")

        # Frozen fit set: the OLS runs on exactly the named models, no candidate
        # walk and no SD/low-obs filter, so the same set can be compared across
        # subsets of the posterior. Same slope here — m0/m2/m4 sit on the line.
        fr = mirt_frontier_forecast(theta, 0, d, raw, sd_cap=None,
                                    drop_low_obs=False,
                                    fit_names=["m0", "m2", "m4"])
        assert fr.fit_basis == "frozen"
        assert fr.fit_names == ["m0", "m2", "m4"]
        assert abs(float(np.median(fr.slope)) - 1.0) < 0.15
        with pytest.raises(ValueError):
            mirt_frontier_forecast(theta, 0, d, raw, sd_cap=None,
                                   drop_low_obs=False, fit_names=["nope"])

        # weights="precision": an early record measured WIDE keeps its place in
        # the fit but loses its leverage. Blow up m0's posterior SD (25x the
        # others) and bias its median high: the unweighted OLS lets that single
        # point produce negative-slope draws and drag the median slope well
        # under the truth, the precision-weighted fit stays on it. With equal
        # SDs the two estimators must agree draw for draw (WLS reduces to OLS).
        theta_w = theta.copy()
        theta_w[:, 0, 0] = 1.8 + rng.normal(0, 2.0, S)
        ols = mirt_frontier_forecast(theta_w, 0, d, raw, sd_cap=None,
                                     drop_low_obs=False,
                                     fit_names=[f"m{i}" for i in range(6)])
        wls = mirt_frontier_forecast(theta_w, 0, d, raw, sd_cap=None,
                                     drop_low_obs=False, weights="precision",
                                     fit_names=[f"m{i}" for i in range(6)])
        assert (wls.slope > 0).mean() > 0.99 > (ols.slope > 0).mean()
        assert abs(float(np.median(wls.slope)) - 1.0) \
            < abs(float(np.median(ols.slope)) - 1.0)
        same = mirt_frontier_forecast(theta, 0, d, raw, sd_cap=None,
                                      drop_low_obs=False, weights="precision",
                                      fit_names=["m0", "m2", "m4"])
        np.testing.assert_allclose(same.slope, fr.slope, rtol=0.2)

        # estimator="theilsen": recovers the clean slope like OLS, and a single
        # aberrant POINT VALUE (not just noisy: biased in every draw) moves the
        # pairwise-median slope far less than the mean-based OLS.
        ts = mirt_frontier_forecast(theta, 0, d, raw, sd_cap=None,
                                    drop_low_obs=False, estimator="theilsen",
                                    fit_names=[f"m{i}" for i in range(6)])
        assert abs(float(np.median(ts.slope)) - 1.0) < 0.15
        theta_b = theta.copy()
        theta_b[:, 0, 0] += 3.0                      # m0 biased high, every draw
        ols_b = mirt_frontier_forecast(theta_b, 0, d, raw, sd_cap=None,
                                       drop_low_obs=False,
                                       fit_names=[f"m{i}" for i in range(6)])
        ts_b = mirt_frontier_forecast(theta_b, 0, d, raw, sd_cap=None,
                                      drop_low_obs=False, estimator="theilsen",
                                      fit_names=[f"m{i}" for i in range(6)])
        assert abs(float(np.median(ts_b.slope)) - 1.0) \
            < abs(float(np.median(ols_b.slope)) - 1.0) - 0.3

        # fit_basis="envelope": the per-draw running max. Non-negative recent
        # rate by construction, forward line anchored at the last record, and
        # three crossing regimes: observed step date inside the window
        # (Average Human at 1.0 is first reached by m2, 2024-01), forward
        # projection above it (Top Performer at 3.5, one year past m5's 2.5),
        # and window-start censoring / floored backcast below it.
        env = mirt_frontier_forecast(theta, 0, d, raw, sd_cap=None,
                                     drop_low_obs=False, fit_basis="envelope")
        assert env.kind == "envelope" and (env.slope >= 0).all()
        assert abs(float(np.median(env.slope)) - 1.0) < 0.2
        cxe = mirt_crossover_df(env, theta, 0, d, axis_name="axis1",
                                today="2026-01-01")
        avg_e = cxe[cxe.tier == "Average Human"].iloc[0]
        top_e = cxe[cxe.tier == "Top Performer"].iloc[0]
        assert avg_e.status.startswith("passed")
        assert (pd.Timestamp("2023-09-01") < pd.Timestamp(avg_e.crossover_date_median)
                < pd.Timestamp("2024-06-01"))
        assert (pd.Timestamp("2026-01-01") < pd.Timestamp(top_e.crossover_date_median)
                < pd.Timestamp("2027-06-01"))
        # A tier below the very first record: censored at the window start
        # without a floor, backcast at the early rate (but never past the
        # floor) with one.
        theta_lo = theta.copy()
        theta_lo[:, 6, 0] = -0.5 + rng.normal(0, 0.05, S)   # Average Human low
        env0 = mirt_frontier_forecast(theta_lo, 0, d, raw, sd_cap=None,
                                      drop_low_obs=False, fit_basis="envelope")
        c0 = mirt_crossover_df(env0, theta_lo, 0, d, axis_name="axis1",
                               today="2026-01-01")
        c0 = c0[c0.tier == "Average Human"].iloc[0]
        assert abs((pd.Timestamp(c0.crossover_date_median)
                    - pd.Timestamp("2023-01-01")).days) < 40   # censored at start
        envf = mirt_frontier_forecast(theta_lo, 0, d, raw, sd_cap=None,
                                      drop_low_obs=False, fit_basis="envelope",
                                      backcast_floor="2022-06-01")
        cf = mirt_crossover_df(envf, theta_lo, 0, d, axis_name="axis1",
                               today="2026-01-01")
        cf = cf[cf.tier == "Average Human"].iloc[0]
        assert (pd.Timestamp("2022-05-25") <= pd.Timestamp(cf.crossover_date_median)
                < pd.Timestamp("2023-01-01"))                  # backcast, floored


# ─────────────────── MIRT non-compensatory (conjunctive) ────────────────────
@pytest.fixture(scope="session")
def data_all() -> ECIData:
    """Full-benchmark data (include_all_benchmarks=True) — what the non-comp and
    confirmatory MIRT drivers actually fit on (every category present)."""
    return load_eci_data(include_all_benchmarks=True)


@pytest.fixture
def mirt_nc_synth(data_all: ECIData):
    """Synthetic non-comp K=3 InferenceData: theta/c/sigma_b/phi_b + a multi-
    loaded 0/1 Q in constant_data. Fast — no NUTS for the PP / r-hat / difficulty
    tests. Q has genuine 2-axis conjunctions so it is not the degenerate case."""
    rng = np.random.default_rng(7)
    nch, nd, K = 2, 60, 3
    M, B = data_all.n_models, data_all.n_benchmarks
    theta = rng.normal(0.0, 1.0, (nch, nd, M, K))
    c = rng.normal(1.0, 0.5, (nch, nd, B, K))
    sigma_b = np.abs(rng.normal(0.08, 0.01, (nch, nd, B)))
    phi_b = 1.0 / (4.0 * sigma_b ** 2) - 1.0
    Q = np.zeros((B, K))
    for b in range(B):
        Q[b, b % K] = 1.0
        if b % 4 == 0:                                   # genuine 2-axis conjunctions
            Q[b, (b + 1) % K] = 1.0
    return az.from_dict(
        posterior={"theta": theta, "c": c, "sigma_b": sigma_b, "phi_b": phi_b},
        constant_data={"Q": Q},
        coords={"model": list(range(M)), "bench": list(range(B)),
                "latent": ["axis1", "axis2", "axis3"]},
        dims={"theta": ["model", "latent"], "c": ["bench", "latent"],
              "sigma_b": ["bench"], "phi_b": ["bench"], "Q": ["bench", "latent"]},
    )


def _model_var_names(model):
    return {v.name for v in model.free_RVs} | {d.name for d in model.deterministics}


def _bq(data, K=3, variant="full"):
    """Lazy import of the diagnostics Q-matrix builder (keeps its module-level
    directory creation out of test-collection time)."""
    from multiaxis_eci.fits.fit_nc import build_qmatrix
    return build_qmatrix(data, K, variant)


class TestMIRTNonComp:
    # ── shared category→axis maps (models.qmatrix) ─────────────────────────
    def test_axes_as_list(self):
        assert axes_as_list(2) == [2]
        assert axes_as_list([1, 0]) == [1, 0]

    def test_qmatrix3_strict_simple_structure(self):
        for cat, v in QMATRIX_VARIANTS["qmatrix3"].items():
            assert len(axes_as_list(v)) == 1, f"{cat} loads >1 axis in qmatrix3"

    def test_qmatrix3x_crossloads_coding_and_science(self):
        q3, q3x = QMATRIX_VARIANTS["qmatrix3"], QMATRIX_VARIANTS["qmatrix3x"]
        crossed = {"Autonomous SWE", "Agentic Computer Use", "Biology", "Chemistry"}
        for cat in crossed:
            ax = axes_as_list(q3x[cat])
            assert len(ax) == 2 and 0 in ax              # gains reasoning (axis 0)
        for cat in q3:                                   # all others unchanged
            if cat not in crossed:
                assert axes_as_list(q3x[cat]) == axes_as_list(q3[cat])

    def test_qmatrix4_multimodal_axis(self):
        """qmatrix4 = qmatrix3 with Multimodal pulled out as a strict 4th axis.
        Kept OUT of QMATRIX_VARIANTS so the non-comp driver's 3-axis name mapping
        (OVERRIDE_MAPS) never sees a 4th index."""
        from multiaxis_eci.models.qmatrix import (AXIS_LABELS_K4, QMATRIX4_CAT_TO_AXIS,
                                  QMATRIX_VARIANTS)
        # strict simple structure: every category -> a single int axis
        assert all(isinstance(v, int) for v in QMATRIX4_CAT_TO_AXIS.values())
        assert set(QMATRIX4_CAT_TO_AXIS.values()) == {0, 1, 2, 3}
        assert len(AXIS_LABELS_K4) == 4
        assert QMATRIX4_CAT_TO_AXIS["Multimodal Understanding"] == 3   # own 4th axis
        assert QMATRIX4_CAT_TO_AXIS["Biology"] == 2                    # Bio/Chem stay
        assert QMATRIX4_CAT_TO_AXIS["Chemistry"] == 2                  #   in knowledge
        assert "qmatrix4" not in QMATRIX_VARIANTS                      # comp-only

    def test_qmatrix4x_crossloads(self):
        """qmatrix4x = qmatrix4 + qmatrix3x's cross-loads (coding & PhD-science
        also tap reasoning, axis 0); multimodal stays single (axis 3)."""
        from multiaxis_eci.models.qmatrix import (QMATRIX4_CAT_TO_AXIS, QMATRIX4X_CAT_TO_AXIS,
                                  QMATRIX_VARIANTS)
        crossed = {"Autonomous SWE", "Agentic Computer Use", "Biology", "Chemistry"}
        for cat in crossed:
            ax = axes_as_list(QMATRIX4X_CAT_TO_AXIS[cat])
            assert len(ax) == 2 and 0 in ax            # gains reasoning (axis 0)
        assert QMATRIX4X_CAT_TO_AXIS["Multimodal Understanding"] == 3   # stays single
        for cat in QMATRIX4_CAT_TO_AXIS:               # non-crossed unchanged vs qmatrix4
            if cat not in crossed:
                assert axes_as_list(QMATRIX4X_CAT_TO_AXIS[cat]) \
                    == axes_as_list(QMATRIX4_CAT_TO_AXIS[cat])
        assert "qmatrix4x" not in QMATRIX_VARIANTS     # comp-only

    # ── Q-matrix validation (_validate_qmatrix) ──────────────────────────
    def test_validate_qmatrix_accepts_multiloaded(self):
        out = _validate_qmatrix(np.array([[1, 1, 0], [0, 1, 0], [1, 0, 1]], float), 3)
        assert out.dtype == float and out.shape == (3, 3)

    def test_validate_qmatrix_rejects_bad_inputs(self):
        with pytest.raises(ValueError):                  # wrong n_benchmarks
            _validate_qmatrix(np.ones((3, 2)), 4)
        with pytest.raises(ValueError):                  # non-binary entry
            _validate_qmatrix(np.array([[2, 0], [1, 1]], float), 2)
        with pytest.raises(ValueError):                  # a benchmark loads no axis
            _validate_qmatrix(np.array([[0, 0], [1, 1]], float), 2)
        with pytest.raises(ValueError):                  # a dead axis (zero column)
            _validate_qmatrix(np.array([[1, 0], [1, 0]], float), 2)

    def test_validate_qmatrix_warns_on_strict_simple_structure(self):
        with pytest.warns(UserWarning, match="independent 1D"):
            _validate_qmatrix(np.array([[1, 0], [0, 1], [1, 0]], float), 3)

    # ── model construction ───────────────────────────────────────────────
    def test_build_creates_expected_vars(self, data_all):
        Q, _ = _bq(data_all)
        names = _model_var_names(build_mirt_nc_model(data_all, Q))
        for v in ["theta", "c", "sigma_b", "phi_b", "tau_c"]:
            assert v in names, f"missing {v}"
        assert "A" not in names and "tau_A" not in names  # restricted MLTM (slope=1)

    def test_free_discrimination_adds_slope(self, data_all):
        Q, _ = _bq(data_all)
        names = _model_var_names(build_mirt_nc_model(data_all, Q, free_discrimination=True))
        assert "A" in names and "tau_A" in names

    @pytest.mark.slow
    def test_tiny_sample_runs(self, data_all):
        Q, _ = _bq(data_all)
        with build_mirt_nc_model(data_all, Q):
            idata = pm.sample(draws=5, tune=5, chains=1, cores=1,
                              progressbar=False, random_seed=0,
                              return_inferencedata=True,
                              compute_convergence_checks=False)
        post = idata.posterior
        assert post["theta"].shape[-1] == 3
        assert post["c"].sizes["bench"] == data_all.n_benchmarks
        assert "Q" in idata.constant_data               # Q travels with the trace
        assert np.isfinite(post["theta"].values).all()

    # ── posterior predictive + shared helpers ────────────────────────────
    def test_pp_nc_shape_and_range(self, mirt_nc_synth, data_all):
        y_rep = posterior_predictive_mirt_nc(mirt_nc_synth, data_all)
        post = mirt_nc_synth.posterior
        S = post.sizes["chain"] * post.sizes["draw"]
        assert y_rep.shape == (S, data_all.n_obs)
        assert (y_rep >= 0.0).all() and (y_rep <= 1.0).all()

    def test_pp_nc_reproduces_product_link(self, mirt_nc_synth, data_all):
        """NC posterior predictive must reconstruct μ = ∏_{Q=1} σ(θ+c): the
        conjunctive product over LOADED axes only (Q=0 → factor 1)."""
        post = mirt_nc_synth.posterior
        K = post.sizes["latent"]
        theta = post["theta"].values.reshape(-1, data_all.n_models, K)
        c = post["c"].values.reshape(-1, data_all.n_benchmarks, K)
        Q = mirt_nc_synth.constant_data["Q"].values
        bi, mi = data_all.bench_idx, data_all.model_idx
        sig = 1.0 / (1.0 + np.exp(-(theta[:, mi, :] + c[:, bi, :])))     # (S, n_obs, K)
        gated = np.where(Q[bi][None].astype(bool), sig, 1.0)            # Q=0 -> factor 1
        mu_mean = np.clip(gated.prod(axis=-1), ECI_EPS, 1.0 - ECI_EPS).mean(axis=0)
        y_pred_mean = posterior_predictive_mirt_nc(mirt_nc_synth, data_all).mean(axis=0)
        assert np.corrcoef(y_pred_mean, mu_mean)[0, 1] > 0.99
        assert np.abs(y_pred_mean - mu_mean).mean() < 0.02

    def test_nc_qgate_unloaded_axis_is_factor_one(self):
        """A Q=0 axis must drop from the product (factor 1), NOT enter as σ(·)≈0.5
        — else every score is silently halved. θ=c=0 → loaded σ=0.5, so a
        one-axis benchmark predicts μ=0.5, not 0.5²=0.25 (the broken-gate value)."""
        import types
        S = 400
        theta = np.zeros((1, S, 1, 2))
        c = np.zeros((1, S, 1, 2))
        sigma_b = np.full((1, S, 1), 0.05)
        phi_b = 1.0 / (4.0 * sigma_b ** 2) - 1.0
        trace = az.from_dict(
            posterior={"theta": theta, "c": c, "sigma_b": sigma_b, "phi_b": phi_b},
            constant_data={"Q": np.array([[1.0, 0.0]])},               # loads axis 0 only
            coords={"model": [0], "bench": [0], "latent": ["a1", "a2"]},
            dims={"theta": ["model", "latent"], "c": ["bench", "latent"],
                  "sigma_b": ["bench"], "phi_b": ["bench"], "Q": ["bench", "latent"]})
        shim = types.SimpleNamespace(bench_idx=np.array([0]),
                                     model_idx=np.array([0]), n_obs=1)
        y = posterior_predictive_mirt_nc(trace, shim, max_draws=10_000)
        assert abs(float(y.mean()) - 0.5) < 0.02                       # 0.25 ⇒ broken gate

    def test_flatten_over_chains(self):
        a = np.arange(2 * 3 * 4).reshape(2, 3, 4)
        flat = _flatten_over_chains(a)
        assert flat.shape == (6, 4) and np.array_equal(flat, a.reshape(6, 4))

    def test_thin_sel(self):
        assert _thin_sel(100, 2000) is None              # no thinning when small
        sel = _thin_sel(5000, 2000)
        assert sel.shape == (2000,) and sel.min() >= 0 and sel.max() < 5000

    def test_beta_draw(self, data_all):
        S = 4
        mu = np.full((S, data_all.n_obs), 0.5)
        phi = np.full((S, data_all.n_benchmarks), 10.0)
        out = _beta_draw(mu, phi, data_all, seed=0)
        assert out.shape == (S, data_all.n_obs)
        assert (out > 0).all() and (out < 1).all()

    # ── identified r-hat (mirt_identified_rhat_nc) ────────────────────────
    def test_identified_rhat_nc_runs(self, mirt_nc_synth, data_all):
        out = mirt_identified_rhat_nc(mirt_nc_synth, data_all, n_obs_sample=50)
        for key in ["logmu_max_rhat", "logmu_mean_rhat",
                    "theta_max_rhat", "sigma_b_max_rhat"]:
            assert key in out and np.isfinite(out[key])

    # ── difficulty draws (nc_difficulty_draws) ────────────────────────────
    def test_nc_difficulty_draws_masks_offaxis(self, mirt_nc_synth):
        Q = mirt_nc_synth.constant_data["Q"].values
        b = nc_difficulty_draws(mirt_nc_synth.posterior, Q)
        assert b.shape[1:] == Q.shape                    # (S, B, K)
        assert np.isnan(b[:, Q == 0]).all()              # off-axis -> NaN
        assert np.isfinite(b[:, Q == 1]).all()           # loaded -> finite
        c = mirt_nc_synth.posterior["c"].values.reshape(b.shape)
        assert np.allclose(b[:, Q == 1], (-c)[:, Q == 1])  # restricted fit: b = -c

    def test_nc_difficulty_draws_free_discrim(self):
        import xarray as xr
        rng = np.random.default_rng(0)
        S, B, K = 5, 3, 2
        c = rng.normal(0.0, 1.0, (1, S, B, K))
        a = np.abs(rng.normal(1.0, 0.2, (1, S, B, K))) + 0.1
        post = xr.Dataset({"c": (("chain", "draw", "bench", "latent"), c),
                           "A": (("chain", "draw", "bench", "latent"), a)})
        Q = np.array([[1, 0], [1, 1], [0, 1]], float)
        b = nc_difficulty_draws(post, Q)
        cf, af = c.reshape(S, B, K), a.reshape(S, B, K)
        assert np.allclose(b[:, Q == 1], (-cf / af)[:, Q == 1])  # b = -c/a

    # ── Q-matrix construction (fits.fit_nc.build_qmatrix) ─────────────────
    def test_build_qmatrix_full_k3(self, data_all):
        Q, axes = _bq(data_all, 3, "full")
        assert axes == ["Reasoning", "Agentic", "Knowledge"]
        assert Q.shape == (data_all.n_benchmarks, 3)
        assert (Q.sum(axis=1) >= 1).all()                # every benchmark loads >=1
        assert (Q.sum(axis=1) >= 2).any()                # genuine conjunctions

    def test_build_qmatrix_strict_vs_cross(self, data_all):
        Qs, _ = _bq(data_all, 3, "qmatrix3")
        Qx, _ = _bq(data_all, 3, "qmatrix3x")
        assert (Qs.sum(axis=1) == 1).all()               # strict simple structure
        assert (Qx.sum(axis=1) == 2).any()               # cross-loaded rows exist

    def test_build_qmatrix_no_agentic_and_k4(self, data_all):
        Qn, axn = _bq(data_all, 3, "no-agentic")
        assert "Agentic" not in axn and Qn.shape[1] == len(axn) == 2
        Q4, ax4 = _bq(data_all, 4, "full")
        assert "Science" in ax4 and Q4.shape[1] == 4

    def test_comp_and_nc_qmatrix3_agree(self, data_all):
        """The compensatory anchors (models.qmatrix) and the non-comp Q
        (build_qmatrix) must pin each benchmark to the SAME axis — the whole
        point of the shared category map."""
        Q, _ = _bq(data_all, 3, "qmatrix3")
        comp = QMATRIX_VARIANTS["qmatrix3"]
        for bi, cat in enumerate(data_all.bench_category):
            comp_axis = axes_as_list(comp[str(cat)])[0]
            assert np.where(Q[bi] == 1.0)[0].tolist() == [comp_axis], f"{cat} mismatch"


class TestFigureSets:
    """The shared figure-set + comparison + dashboard assembler (Units 2–5)."""

    def _gof_inputs(self, n=30, seed=0):
        rng = np.random.default_rng(seed)
        scores = rng.uniform(0.05, 0.95, n)
        y_pred = np.clip(scores + rng.normal(0, 0.05, n), 0.01, 0.99)
        yrep = rng.uniform(0, 1, (200, n))
        pit = rng.uniform(0, 1, n)
        hover = [f"m{i}·b{i % 4}" for i in range(n)]
        bench_of_obs = [f"b{i % 4}" for i in range(n)]
        return scores, y_pred, yrep, pit, hover, bench_of_obs

    def test_build_gof_figures_keys(self):
        figs = viz.build_gof_figures(*self._gof_inputs())
        assert set(figs) == {"gof_pred_vs_observed", "gof_posterior_predictive",
                             "gof_pit", "gof_residuals"}
        assert "gof_pit_ecdf" in viz.build_gof_figures(*self._gof_inputs(), include_ecdf=True)

    def test_build_gof_bench_scores_vs_pred(self):
        s, yp, yr, pit, hov, bo = self._gof_inputs()
        model_of_obs = [f"m{i}" for i in range(len(s))]
        figs = viz.build_gof_figures(s, yp, yr, pit, hov, bo,
                                     model_of_obs=model_of_obs)
        fig = figs["gof_bench_scores_vs_pred"]
        n_bench = len(set(bo))
        assert len(fig.data) == 2 * n_bench           # obs + pred per benchmark
        assert len(fig.layout.updatemenus[0].buttons) == n_bench
        assert sum(tr.visible for tr in fig.data) == 2  # one benchmark shown
        # the default benchmark's predicted trace carries the interval band
        vis = [tr for tr in fig.data if tr.visible]
        pred = next(tr for tr in vis if tr.name.startswith("predicted"))
        assert pred.error_y.array is not None
        # observed y within each visible trace is sorted (ranked by observed score)
        obs = next(tr for tr in vis if tr.name == "observed score")
        assert list(obs.y) == sorted(obs.y)

    def test_build_gof_bench_icc(self):
        s, yp, yr, pit, hov, bo = self._gof_inputs()
        model_of_obs = [f"m{i}" for i in range(len(s))]
        eta = np.linspace(-3, 3, len(s))
        figs = viz.build_gof_figures(s, yp, yr, pit, hov, bo,
                                     model_of_obs=model_of_obs, eta_of_obs=eta)
        fig = figs["gof_bench_icc"]
        n_bench = len(set(bo))
        assert len(fig.data) == 2 * n_bench            # sigmoid + observed per benchmark
        assert len(fig.layout.updatemenus[0].buttons) == n_bench
        assert sum(tr.visible for tr in fig.data) == 2
        vis = [tr for tr in fig.data if tr.visible]
        curve = next(tr for tr in vis if tr.name == "fitted sigmoid")
        # the fitted curve is a monotone 0→1 sigmoid (no floor/ceiling given)
        assert list(curve.y) == sorted(curve.y)
        assert 0.0 <= min(curve.y) and max(curve.y) <= 1.0

    def test_build_gof_icc_floor_ceiling(self):
        s, yp, yr, pit, hov, bo = self._gof_inputs()
        b0 = sorted(set(bo))[0]
        figs = viz.build_gof_figures(
            s, yp, yr, pit, hov, bo, model_of_obs=[f"m{i}" for i in range(len(s))],
            eta_of_obs=np.linspace(-3, 3, len(s)),
            floor={b0: 0.25}, ceiling={b0: 0.9})
        fig = figs["gof_bench_icc"]
        # b0's sigmoid asymptotes at its floor/ceiling, not 0/1
        b0_curve = fig.data[2 * sorted(set(bo)).index(b0)]
        assert min(b0_curve.y) >= 0.25 - 1e-6
        assert max(b0_curve.y) <= 0.9 + 1e-6

    def test_build_gof_icc_needs_eta(self):
        s, yp, yr, pit, hov, bo = self._gof_inputs()
        figs = viz.build_gof_figures(s, yp, yr, pit, hov, bo,
                                     model_of_obs=[f"m{i}" for i in range(len(s))])
        assert "gof_bench_icc" not in figs

    def test_build_gof_residual_mask_subsets(self):
        s, yp, yr, pit, hov, bo = self._gof_inputs(n=20)
        mask = np.array([True] * 10 + [False] * 10)
        box = viz.build_gof_figures(s, yp, yr, pit, hov, bo, residual_mask=mask)["gof_residuals"]
        n_pts = sum(len(np.atleast_1d(tr.x)) for tr in box.data)
        assert n_pts == 10                       # only the masked observations

    def test_factor_corr_fig_rotated_subtitle(self):
        Phi = np.array([[1.0, 0.6], [0.6, 1.0]])
        Phi_raw = np.array([[1.0, 0.05], [0.05, 1.0]])
        rot = viz.factor_corr_fig(Phi, ["a1", "a2"], rotated=True, Phi_raw=Phi_raw)
        plain = viz.factor_corr_fig(Phi, ["a1", "a2"], rotated=False)
        assert len(rot.layout.annotations) >= 1  # raw-corr subtitle only when rotated
        assert not plain.layout.annotations

    def test_cmp_gof_fig_three_bars(self):
        tab = pd.DataFrame({"fit": ["a", "b"], "R2": [0.9, 0.95],
                            "RMSE": [0.06, 0.05], "MAE": [0.04, 0.03]})
        assert sum(1 for t in viz.cmp_gof_fig(tab).data if t.type == "bar") == 3

    def test_binary_qmatrix_all_ones_row(self):
        Q = np.array([[1, 1, 1], [1, 0, 0], [0, 1, 0]], float)   # unanchored row = all ones
        fig = viz.binary_qmatrix_fig(Q, ["a1", "a2", "a3"], ["b0", "b1", "b2"])
        assert fig.data[0].type == "heatmap"

    def test_delpd_se_math(self):
        d, se = viz.delpd_se(np.array([1., 2., 3., 4.]), np.array([0., 1., 2., 3.]))
        assert d == pytest.approx(4.0)            # diff of 1 each, summed
        assert se == pytest.approx(0.0, abs=1e-9)  # zero variance in the difference

    def _results(self, with_loo=True):
        rng = np.random.default_rng(1)
        out = []
        for i, (lab, typ) in enumerate([("1D", "baseline"), ("K=3 ARD", "exploratory"),
                                        ("K=3 skills", "confirmed")]):
            r = {"fit": lab, "name": lab, "type": typ, "K": 1 if i == 0 else 3,
                 "free_loadings": 75, "R2": 0.9 + 0.01 * i, "RMSE": 0.06, "MAE": 0.04,
                 "PIT_var": 0.05, "eta_rhat": 1.0 + 0.02 * i, "divergences": i,
                 "max_phi": "—" if i == 0 else round(0.5 + 0.1 * i, 3),
                 "pit": rng.uniform(0, 1, 30),
                 "per_bench_rmse": pd.Series([0.05, 0.06, 0.07], index=["b0", "b1", "b2"]),
                 "min_ess": 200.0 - 50 * i,
                 # Mixing is reported per draw, so a fit row carries the eta ESS
                 # and the kept-draw count it is divided by.
                 "eta_ess_min": 200.0 - 50 * i, "eta_ess_med": 4000.0 - 500 * i,
                 "n_draws_kept": 8000 - 1000 * i,
                 "tau_sorted": (np.array([1.0, 0.5, 0.3]) if i == 1 else None)}
            if with_loo:
                r.update({"loo_pointwise": rng.normal(0, 1, 30),
                          "waic_pointwise": rng.normal(0, 1, 30),
                          "loo_elpd": 100.0 - i, "loo_se": 5.0, "loo_p_eff": 10.0,
                          "waic_elpd": 100.0 - i, "waic_se": 5.0, "waic_p_eff": 10.0,
                          "pareto_k_good": 25, "pareto_k_ok": 3, "pareto_k_bad": 2,
                          "pareto_k_very_bad": 0, "pareto_k_max": 0.8, "pareto_k_mean": 0.3})
            out.append(r)
        return out

    def test_build_comparison_tables_and_figs(self):
        tables, figs = viz.build_comparison(self._results(with_loo=True))
        assert list(tables["gof_table"]["fit"]) == ["1D", "K=3 ARD", "K=3 skills"]
        assert "loo_waic_table" in tables and len(tables["loo_waic_table"]) == 3
        for k in ["cmp_per_benchmark_rmse", "cmp_gof", "cmp_convergence", "cmp_pit_ecdf",
                  "cmp_loo_waic", "cmp_pareto_k", "cmp_loo_vs_trust",
                  "cmp_tau_spectrum"]:
            assert k in figs, f"missing {k}"

    def test_build_comparison_without_loo_still_has_gof(self):
        tables, figs = viz.build_comparison(self._results(with_loo=False))
        assert "loo_waic_table" not in tables
        assert "cmp_loo_waic" not in figs and "cmp_gof" in figs

    def test_assemble_dashboard_self_contained(self):
        f = go.Figure(go.Bar(x=["a", "b"], y=[1, 2]))
        html = viz.assemble_dashboard(
            [{"id": "fit-x", "label": "Fit X", "type": "confirmed",
              "stat_line": "R²=0.9", "figures": {"gof": f}}],
            {"headline": "hi", "tables_html": "<table class='cmp'></table>",
             "figures": {"cmp_gof": f}})
        assert html.count("var PLOTS=") == 1                       # one Plotly payload
        # Self-contained: no external resource-LOADING tags (a <script src>/<link
        # href>). Substring 'src="http' does appear inside the inlined Plotly
        # bundle as a JS string literal (an unused mapbox icon loader), so we must
        # match actual tags, not raw substrings.
        import re
        assert not re.search(r'<script[^>]*\ssrc\s*=|<link[^>]*\shref\s*=', html)
        assert 'id="cmp"' in html and 'id="fit-x"' in html          # both sections
        assert "RENDERED[d.id]" in html                            # lazy render wired
        assert html.lstrip().lower().startswith("<!doctype html>")


# ───────────────────────── FitSpec ─────────────────────────────────────────
class TestFitSpec:
    """Fit identity: the tag, and recovering a spec from an existing trace."""

    @staticmethod
    def _idata(K=4, attrs=None, n_models=5, n_bench=3):
        """A trace stub carrying only what `from_trace` reads: attrs and dims."""
        import xarray as xr
        A = np.zeros((1, 2, n_bench, K))
        theta = np.zeros((1, 2, n_models, K))
        post = xr.Dataset({"A": (("chain", "draw", "bench", "latent"), A),
                           "theta": (("chain", "draw", "model", "latent"), theta)},
                          attrs=attrs or {})
        return az.InferenceData(posterior=post)

    def test_flagship_tag_literal(self):
        # The name every future flagship artefact is written under. Pooled noise
        # and the 3PL floors are defaults and carry no token; FrontierMath v1 and
        # AlgoTune are out of scope through the retirement list, so no `_drop`
        # either. Non-negative loadings are the default too, so no prior token.
        assert analysis.FLAGSHIP.tag == (
            "_humanmerge_lineageprior_lineagebm")
        assert analysis.FLAGSHIP.trace_path.name == (
            "trace_mirt_k4_humanmerge_lineageprior_lineagebm.nc")
        assert analysis.FLAGSHIP.pooled_noise and analysis.FLAGSHIP.floors
        assert analysis.FLAGSHIP.loading_prior == "normal"
        assert not analysis.FLAGSHIP.drop_benchmarks

    def test_spec_json_round_trip(self):
        spec = analysis.FitSpec(
            K=3, loading_prior="signed", link="loglog", human_merge=True,
            lineage_prior=True, lineage_bm=True, time_prior=True, theta_t=True,
            theta_pos=True, no_sg=True,
            apply_exclusions=True, cyber=True, simpleqa_original=True,
            drop_benchmarks=("FrontierMath v1", "AlgoTune"), private_bases=True,
            floors=True, ceiling_noise=True, known_se=True,
            pooled_noise=True)
        idata = self._idata(K=3, attrs={"mirt_spec": analysis.spec_json(spec)})
        got = analysis.FitSpec.from_trace(idata, spec.trace_path)
        assert got == spec

    def test_folder_tag_fallback(self):
        # No attrs at all: the folder tag carries the flags that never had one.
        idata = self._idata(K=3)
        p = (PROJECT_ROOT / "results" / "mirt_noSG_excluded_floors_ceilnoise"
             / "trace_mirt_k3_noSG_excluded_floors_ceilnoise.nc")
        spec = analysis.FitSpec.from_trace(idata, p)
        assert (spec.apply_exclusions, spec.no_sg, spec.floors,
                spec.ceiling_noise) == (True, True, True, True)
        assert spec.loading_prior == "normal" and spec.drop_benchmarks == ()

    def test_lossy_tokens_refused(self):
        idata = self._idata(K=3)
        with pytest.raises(ValueError):
            analysis.FitSpec.from_trace(
                idata, PROJECT_ROOT / "results" / "mirt_dropGBAEvalVPCT_floors"
                / "trace_mirt_k3_dropGBAEvalVPCT_floors.nc")
        # An unknown historical token is refused too, never partly parsed.
        with pytest.raises(ValueError, match="unrecognized tag token"):
            analysis.FitSpec.from_trace(
                idata, PROJECT_ROOT / "results" / "mirt_lineagehard_floors"
                / "trace_mirt_k3_lineagehard_floors.nc")

    def test_attrless_baseline_takes_folder_data_scope(self):
        # 2_fit.py's K=1 baseline used to carry no attrs and no tag in its filename. It
        # keeps the folder's DATA scope and gets the model-side flags from
        # 2_fit.py's baseline rule (floors/ceiling-noise/known_se forwarded, the priors
        # and the pooled noise not).
        idata = self._idata(K=1)
        p = (PROJECT_ROOT / "results"
             / "mirt_humanprior_lineageprior_cyber_floors_knownse_poolednoise"
             / "trace_mirt_k1.nc")
        spec = analysis.FitSpec.from_trace(idata, p)
        assert spec.K == 1
        assert (spec.cyber, spec.floors, spec.known_se) == (True, True, True)
        assert not (spec.human_prior or spec.lineage_prior or spec.pooled_noise)
        # floors ON emits no token; the baseline is fit pooled_noise=False
        # (2_fit.py's baseline rule), which the tag now names. Harmless for
        # folders: a baseline lives untagged in its K-fit's folder.
        assert spec.tag == "_cyber_knownse_unpooled"

    # `tag` writes the tokens and `_parse_tag` reads them from a second list, so
    # every flag has to survive the round trip or a folder tag silently loads the
    # wrong data scope. One case per boolean flag set AWAY from its default, and
    # one per loading prior; the dependent flag brings its prerequisite with it.
    # floors and pooled_noise default ON with no token; their opt-outs emit
    # `_nofloors` / `_unpooled` so a sensitivity run cannot overwrite the default
    # config's folder. Their legacy ON tokens are covered by
    # test_legacy_poolednoise_token_still_parses and
    # test_legacy_floors_token_still_parses.
    @pytest.mark.parametrize("kwargs", [
        pytest.param({f.name: not f.default}, id=f.name)
        for f in dataclasses.fields(analysis.FitSpec)
        if isinstance(f.default, bool)
    ] + [
        pytest.param({"loading_prior": p}, id=f"prior_{p}")
        for p in ("normal", "signed", "signedhs", "pt1", "bifactor")
    ])
    def test_tag_round_trips_through_parse_tag(self, kwargs):
        from multiaxis_eci.analysis.fitspec import _parse_tag
        prereq = {"lineage_bm": {"lineage_prior": True}}
        for f in list(kwargs):
            kwargs.update(prereq.get(f, {}))
        spec = analysis.FitSpec(K=2, **kwargs)
        # floors ON emits no token and an absent token parses as OFF (the legacy
        # meaning, kept for three attr-less traces on disk); floors OFF emits
        # `_nofloors` and round-trips exactly. Every other flag must survive.
        expected = spec if not spec.floors else dataclasses.replace(spec, floors=False)
        assert analysis.FitSpec(K=2, **_parse_tag(spec.tag)) == expected

    def test_legacy_floors_token_still_parses(self):
        # `tag` stopped emitting `_floors` once floors became the default, but
        # both directions of the legacy token must keep their ORIGINAL meaning:
        # present = on, ABSENT = off. Three attrless traces on disk sit in
        # folders with no token (mirt_humanprior, mirt_loglog) and were fit
        # without floors, so an absent token must not inherit the True default.
        from multiaxis_eci.analysis.fitspec import _parse_tag
        assert _parse_tag("_floors")["floors"] is True
        assert _parse_tag("_humanprior")["floors"] is False
        assert _parse_tag("")["floors"] is False
        # and the token is never written again
        assert "_floors" not in analysis.FitSpec(K=2, floors=True).tag

    def test_attrless_nofloors_folder_keeps_floors_off(self):
        # The end-to-end version of the case above, through from_trace: the
        # round-trip guard must accept a spec whose tag omits a token the folder
        # never carried, and the recovered scope must be the unclipped one.
        idata = self._idata(K=3, attrs={"mirt_loading_prior": "normal"})
        p = (PROJECT_ROOT / "results" / "mirt_humanprior"
             / "trace_mirt_k3_humanprior.nc")
        spec = analysis.FitSpec.from_trace(idata, p)
        assert spec.human_prior and not spec.floors
        assert spec.loading_prior == "normal"

    def test_legacy_poolednoise_token_still_parses(self):
        # Folders written while the tag carried the token must still resolve, and
        # they name the value the default already holds.
        from multiaxis_eci.analysis.fitspec import _parse_tag
        assert _parse_tag("_floors_poolednoise") == {
            "loading_prior": "normal", "floors": True, "pooled_noise": True}

    def test_legacy_unpooled_trace_in_tokenless_folder_resolves(self):
        # A --no-pooled-noise fit from before the `_unpooled` token: its attrs
        # say pooled_noise=False but its folder carries nothing. The recovered
        # spec's own tag now emits `_unpooled`, so the round-trip guard must
        # compare pooled_noise loosely (same treatment as floors) or every such
        # legacy trace becomes unreadable.
        legacy = analysis.FitSpec(K=3, human_prior=True, floors=True,
                                  pooled_noise=False)
        idata = self._idata(K=3, attrs={"mirt_spec": analysis.spec_json(legacy)})
        p = (PROJECT_ROOT / "results" / "mirt_humanprior_floors"
             / "trace_mirt_k3_humanprior_floors.nc")
        assert analysis.FitSpec.from_trace(idata, p) == legacy

    def test_optout_flags_get_their_own_folder(self):
        # --no-floors / --no-pooled-noise are sensitivity runs; without their own
        # tokens they would silently overwrite the default config's trace and
        # CSVs (same results_dir), and reuse its cached K=1 baseline.
        from multiaxis_eci.analysis.fitspec import _parse_tag
        base = analysis.FitSpec(K=2)
        nofl = dataclasses.replace(base, floors=False)
        unpl = dataclasses.replace(base, pooled_noise=False)
        assert nofl.tag == "_nofloors" and unpl.tag == "_unpooled"
        assert len({base.results_dir, nofl.results_dir, unpl.results_dir}) == 3
        assert _parse_tag("_nofloors")["floors"] is False
        assert _parse_tag("_unpooled")["pooled_noise"] is False

    def test_legacy_flagship_trace_name_resolves(self):
        # A trace whose `mirt_spec` attr names the drop flags sits in a folder
        # whose tag carries tokens the current grammar does not write, so the
        # round-trip guard must accept that pairing.
        import dataclasses as dc
        legacy = dc.replace(analysis.FLAGSHIP,
                            drop_benchmarks=("FrontierMath v1", "AlgoTune"))
        tag = ("_humanmerge_lineageprior_lineagebm_dropFrontierMathv1AlgoTune"
               "_floors_poolednoise")
        p = (PROJECT_ROOT / "results" / f"mirt{tag}" / f"trace_mirt_k4{tag}.nc")
        idata = self._idata(K=4, attrs={"mirt_spec": analysis.spec_json(legacy)})
        assert analysis.FitSpec.from_trace(idata, p) == legacy

    @pytest.mark.parametrize("folder,fname,attrs", [
        # folder flags disagree with the filename's
        ("mirt_floors_ceilnoise", "trace_mirt_k3_floors.nc", None),
        # attrs name a flag set the folder does not
        ("mirt_floors", "trace_mirt_k3_floors.nc",
         {"mirt_spec": None}),
        # attrs name a K the filename does not
        ("mirt_floors", "trace_mirt_k3_floors.nc",
         {"mirt_spec": "K4"}),
    ])
    def test_misfiled_trace_still_refused(self, folder, fname, attrs):
        # Field equality replaced a path-string compare; every mis-filing the
        # string compare caught must still raise.
        import dataclasses as dc
        if attrs and "mirt_spec" in attrs:
            spec = analysis.FitSpec(K=3, loading_prior="normal", floors=True)
            spec = (dc.replace(spec, K=4) if attrs["mirt_spec"] == "K4"
                    else dc.replace(spec, ceiling_noise=True))
            attrs = {"mirt_spec": analysis.spec_json(spec)}
        idata = self._idata(K=3, attrs=attrs)
        p = PROJECT_ROOT / "results" / folder / fname
        with pytest.raises(ValueError, match="round-trip failed"):
            analysis.FitSpec.from_trace(idata, p)

    def test_flag_conflicts_raise(self):
        with pytest.raises(ValueError):
            analysis.FitSpec(K=2, lineage_bm=True)



def _load_script(relpath):
    """Import a numbered script by path.

    The reproduction-path scripts are prefixed (`3_diagnostics/3_plot_mirt.py`),
    and a module name cannot start with a digit, so `import` cannot reach them.
    Everything importable lives in `multiaxis_eci/`; these are entry points, loaded here
    the only way Python allows.
    """
    import importlib.util
    path = PROJECT_ROOT / relpath
    name = "_script_" + path.stem.lstrip("0123456789_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def test_plot_mirt_folder_decision():
    """`--folder` must refuse the two filenames that are not fits of their own,
    and thin one draw per 2 GB so the 38 GB flagship fits in 26 GB of RAM."""
    folder_decision = _load_script("3_diagnostics/3_plot_mirt.py").folder_decision
    assert folder_decision("trace.nc", 1_694_555_488)[0] is not None
    assert folder_decision("trace_mirt_k1.nc", 141_356)[0] is not None
    assert folder_decision("trace_mirt_k2_loglog.nc", 20_333_520) == (None, 1)
    assert folder_decision("trace_mirt_k4_humanmerge.nc",
                           38_016_363_203) == (None, 19)
    # an explicit --thin wins over the size rule
    assert folder_decision("trace_mirt_k4_humanmerge.nc",
                           38_016_363_203, thin=3) == (None, 3)


def test_dashboard_json_registry_round_trip(tmp_path, monkeypatch):
    """A `--add`ed entry must come back out of the JSON as a usable card.

    `dataclasses.asdict` turns the spec's `drop_benchmarks` tuple into a list,
    which would make the spec unhashable and break the dashboard's per-scope
    data cache; the trace path must come back a Path, since a legacy card is
    the whole reason it is stored explicitly.
    """
    import dataclasses as dc
    import json as _json
    bd = _load_script("3_diagnostics/4_build_dashboard.py")

    spec = dc.replace(analysis.FLAGSHIP, drop_benchmarks=("FrontierMath v1",))
    reg = tmp_path / "dashboard_fits.json"
    reg.write_text(_json.dumps([{
        "name": "tmp_card", "label": "L", "type": "exploratory",
        "forecast": True, "spec": dc.asdict(spec),
        "trace_path": str(analysis.FLAGSHIP_TRACE)}]))
    monkeypatch.setattr(bd, "FITS_JSON", reg)

    entry, = bd._json_fits()
    assert entry["spec"] == spec and hash(entry["spec"]) == hash(spec)
    assert entry["trace_path"] == analysis.FLAGSHIP_TRACE
    assert bd._trace_path(entry) == analysis.FLAGSHIP_TRACE
    assert (entry["origin"], entry["forecast"]) == ("json", True)
    assert [f["name"] for f in bd.all_fits()][-1] == "tmp_card"


import re as _re


class TestLayoutPaths:
    """The 2026-08 restructure renamed data/ -> 1_data/, diagnostics/ ->
    3_diagnostics/, fit.py -> 2_fit.py and hoisted the library into
    multiaxis_eci/. Paths built piecewise (`ROOT / "data" / "curated"`) survive
    a rename silently: they still evaluate to a Path, and only blow up when
    something reads them. These lock the layout so the next rename fails here
    instead of in a user's run.

    This file is excluded from its own scans: the patterns below are literals
    in it, so it would always match itself.
    """

    _SELF = Path(__file__).resolve()
    _SKIP_PARTS = {".git", "__pycache__", ".pytest_cache", "results", "plots"}

    @classmethod
    def _files(cls, suffixes):
        # git-visible files only, when git is available: tracked plus
        # untracked-but-not-ignored (--others --exclude-standard), so a new
        # file is scanned BEFORE it is staged, while a maintainer's gitignored
        # local dirs (blogpost/, memo/, evals/*/out/) cannot raise false alarms
        # a fresh cloner can never reproduce. Falls back to the rglob sweep
        # when the tree is not a git checkout (a tarball download) — including
        # when git succeeds but sees nothing, which happens when the tarball
        # sits inside some ENCLOSING repo (a dotfiles repo over $HOME): exit 0,
        # zero rows, and the locks would pass vacuously.
        import subprocess
        try:
            listed = subprocess.run(
                ["git", "ls-files", "-z", "--cached", "--others",
                 "--exclude-standard"], cwd=PROJECT_ROOT, capture_output=True,
                check=True, text=True).stdout.split("\0")
            paths = [PROJECT_ROOT / t for t in listed if t]
        except (OSError, subprocess.CalledProcessError):
            paths = []
        if not paths:
            paths = [p for p in PROJECT_ROOT.rglob("*")
                     if not cls._SKIP_PARTS & set(p.parts)]
        return [p for p in paths
                if p.is_file() and p.suffix in suffixes
                and p.resolve() != cls._SELF]

    def test_no_prerename_path_literals(self):
        # `[/"]` after the name: catches both the exact segment (`/ "data"`)
        # and a longer literal (`/ "data/processed/..."`), which the
        # closing-quote-only form let through (found live in the post scripts).
        stale = _re.compile(r'/ *"(data|diagnostics)[/"]|"(diagnostics|eci)/')
        hits = [f"{p.relative_to(PROJECT_ROOT)}:{i}"
                for p in self._files({".py"})
                for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1)
                if stale.search(line)]
        assert not hits, f"pre-rename path literals: {hits}"

    def test_no_doubled_rename_artifacts(self):
        """A rename pass that replaces a long form and then a short one applies
        the second rule inside the first rule's output. It bit twice here:
        `diagnostics/x.py` -> `3_diagnostics/3_x.py` -> `3_3_diagnostics/3_x.py`,
        and `eci/data.py` -> `multiaxis_eci/data.py` ->
        `multiaxis_multiaxis_eci/data.py`. Both forms read as plausible until
        someone follows the path, so catch the shape rather than the instances:
        any token immediately repeated after an underscore.
        """
        # `\b` cannot follow the second token when an underscore does, which is
        # exactly the multiaxis_multiaxis_eci case, so the tail is `_` or a
        # non-word character rather than a word boundary.
        doubled = _re.compile(r"\b([123])_\1_(data|fit|diagnostics|pipeline)"
                              r"|\b([a-z]{3,})_\3(?=_|[^\w]|$)")
        hits = []
        for f in self._files({".py", ".md", ".sh", ".ipynb", ".json", ".toml", ".txt"}):
            for i, line in enumerate(f.read_text(encoding="utf-8", errors="ignore").splitlines(), 1):
                m = doubled.search(line)
                if m:
                    hits.append(f"{f.relative_to(PROJECT_ROOT)}:{i}: {m.group(0)}")
        assert not hits, f"doubled rename artifacts: {hits}"

    def test_layout_entry_points_exist(self):
        for rel in ["1_data", "1_data/1_pipeline/pipeline.ipynb", "1_data/curated",
                    "1_data/processed/benchmarks_merged.csv", "2_fit.py",
                    "3_diagnostics/4_build_dashboard.py", "3_diagnostics/dashboard_fits.json",
                    "multiaxis_eci/config.py", "multiaxis_eci/scripts.py",
                    "notebooks", "docs/cli.md", "LICENSE", "NOTICE.md"]:
            assert (PROJECT_ROOT / rel).exists(), f"missing: {rel}"

    def test_no_hardcoded_home_paths(self):
        """A personal home path in a docstring or a notebook output is both a
        leak and a command nobody else can run."""
        home = _re.compile(r"/Users/[a-z]|miniforge3/envs")
        hits = [str(p.relative_to(PROJECT_ROOT))
                for p in self._files({".py", ".md", ".sh", ".ipynb", ".toml"})
                if home.search(p.read_text(encoding="utf-8", errors="ignore"))]
        assert not hits, f"hardcoded home / env paths: {hits}"
