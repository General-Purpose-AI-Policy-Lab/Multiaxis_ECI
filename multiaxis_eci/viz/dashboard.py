"""Figure-set assemblers and the self-contained dashboard HTML writer."""
from __future__ import annotations

import numpy as np
import pandas as pd

from multiaxis_eci.viz.compare import (
    cmp_convergence_fig, cmp_gof_fig,
    cmp_loo_vs_trust_fig, cmp_loo_waic_fig, cmp_pareto_k_fig,
    cmp_per_benchmark_rmse_fig, cmp_pit_ecdf_fig, cmp_tau_spectrum_fig,
)
from multiaxis_eci.viz.core import capability_timeline_fig, forest_fig
from multiaxis_eci.viz.forecast import (
    capability_forecast_fig, crossover_dotwhisker_fig, exceedance_prob_fig,
)
from multiaxis_eci.viz.gof import (
    benchmark_icc_fig, benchmark_obs_vs_pred_fig,
    density_overlay_fig, pit_ecdf_fig, pit_hist_fig,
    pred_vs_obs_fig, residuals_per_benchmark_fig,
)
from multiaxis_eci.viz.mirt import binary_qmatrix_fig, factor_corr_fig, loadings_heatmap_fig

# ── shared figure-SETS (Unit 3): one GoF quad + one per-fit set, all callers ──

def _slug(s: str) -> str:
    return s.replace("/", "_").replace(" ", "")


def build_gof_figures(scores, y_pred_mean, yrep, pit, hover, bench_of_obs,
                      residual_mask=None, include_ecdf: bool = False,
                      model_of_obs=None, eta_of_obs=None,
                      floor=None, ceiling=None) -> dict:
    """Shared goodness-of-fit figure set — the core figures 2_fit.py, `plot_mirt`
    and the dashboard all read from here.

    `bench_of_obs` is the benchmark name for each observation; `residual_mask`
    (bool array) subsets which observations enter the per-benchmark residual box
    (the canonical preset passes ~zero_score_mask); `include_ecdf` adds the PIT ECDF;
    `model_of_obs` (model name per observation) adds the per-benchmark
    raw-scores-vs-predictions dropdown figure. `eta_of_obs` (per-observation
    A_b·θ_m − D_b) adds the per-benchmark item characteristic curve, with
    optional per-benchmark `floor`/`ceiling` asymptotes."""
    figs = {
        "gof_pred_vs_observed":     pred_vs_obs_fig(scores, y_pred_mean, hover),
        "gof_posterior_predictive": density_overlay_fig(yrep, scores),
        "gof_pit":                  pit_hist_fig(pit),
    }
    if include_ecdf:
        figs["gof_pit_ecdf"] = pit_ecdf_fig(pit)
    if model_of_obs is not None:
        figs["gof_bench_scores_vs_pred"] = benchmark_obs_vs_pred_fig(
            scores, yrep, model_of_obs, bench_of_obs)
        if eta_of_obs is not None:
            figs["gof_bench_icc"] = benchmark_icc_fig(
                eta_of_obs, scores, model_of_obs, bench_of_obs,
                floor=floor, ceiling=ceiling)
    m = (np.ones(len(scores), dtype=bool) if residual_mask is None
         else np.asarray(residual_mask, dtype=bool))
    idx = np.where(m)[0]
    resid_df = pd.DataFrame({"benchmark": [bench_of_obs[i] for i in idx],
                             "residual": (np.asarray(scores) - np.asarray(y_pred_mean))[m]})
    figs["gof_residuals"] = residuals_per_benchmark_fig(resid_df)
    return figs


def signed_display_frames(view, idata):
    """The two clean frames of a SIGNED-family fit — "raw" (no-transform PCA
    frame, where a bipolar contrast survives as one axis) and "oblique" (promax,
    the general factor shared into the skill axes) — or None for any other fit,
    whose loadings are already in an appropriate frame."""
    from multiaxis_eci.analysis import (align_rotations, mirt_factors_from_trace,
                          trace_loading_prior)
    if not (view.A is not None and view.K >= 2
            and trace_loading_prior(idata) == "signed"):
        return None
    A0, th0, _ = mirt_factors_from_trace(idata, rank_track=False)
    return {"raw": align_rotations(A0, th0, method="raw"),
            "oblique": align_rotations(A0, th0, method="promax")}


def build_axis_figures(view, data, raw, bench, signed_frames=None,
                       prefix: str = "", suffix: str = "",
                       axis_titles: dict | None = None,
                       human_labels: dict | None = None) -> dict:
    """Ability-timeline + loading figures for one posterior: per axis, the
    measured timeline (SD<0.4, low-obs dropped), its all-models companion, and
    the loading forest, plus the all-axis loading heatmap.

    `build_fit_figures` and a MODE-RESTRICTED view (the same fit sliced to one
    basin's chains) share these builders through this function. `prefix` keys
    the mode's figures apart from the whole-fit ones on the same card; `suffix`
    names the mode in every title, so each figure reads on its own. `axis_titles`
    (e.g. config.AXIS_TITLES) swaps the display text only — figure dict keys and
    axis widget titles keep `names[k]` so cache/anchor ids don't churn.
    `human_labels` reaches `capability_timeline_fig` unchanged (None = its
    default French tier labels, `{}` = the raw English tier names)."""
    from multiaxis_eci.analysis import (loadings_forest_df, mirt_human_axis_stats,
                          mirt_model_timeline_df)
    figs = {}
    K, names = view.K, view.names
    titles = axis_titles or {}
    # Signed fits: timelines and loadings in the OBLIQUE frame (dashboard
    # decision 2026-07-06 — the raw-frame timelines doubled every card for
    # little extra signal; the raw frame survives in the axis-strength forest).
    theta = signed_frames["oblique"].theta if signed_frames else view.theta
    A = signed_frames["oblique"].A if signed_frames else view.A
    tag = "_oblique" if signed_frames else ""
    for k in range(K):
        disp = titles.get(names[k], names[k])
        hstat = mirt_human_axis_stats(theta, k, data)
        tl = mirt_model_timeline_df(theta, k, data, raw)
        if not tl.empty:
            fig = capability_timeline_fig(tl, human_stats=hstat,
                                          human_labels=human_labels)
            fig.update_layout(title=dict(text=f"{disp} — measured (50% intervals){suffix}", x=0.5),
                              yaxis=dict(title=disp))
            figs[f"{prefix}timeline_{k+1}_{_slug(names[k])}{tag}"] = fig
        # ALL-models companion — every dated model, incl. sparse/extrapolated (wide CI).
        tl_all = mirt_model_timeline_df(theta, k, data, raw,
                                        sd_cap=None, drop_low_obs=False)
        if not tl_all.empty:
            fig_all = capability_timeline_fig(tl_all, human_stats=hstat,
                                              human_labels=human_labels)
            fig_all.update_layout(
                title=dict(text=f"{disp} — all models (50% intervals){suffix}", x=0.5),
                yaxis=dict(title=disp))
            figs[f"{prefix}timeline_{k+1}_{_slug(names[k])}{tag}_all"] = fig_all
    if A is not None and K >= 2:
        # Per-axis loading forests: which benchmarks load ± on that axis (sorted,
        # with HDI). The heatmap stays as the compact all-axis overview.
        for k in range(K):
            figs[f"{prefix}loadings_{k+1}_{_slug(names[k])}"] = forest_fig(
                loadings_forest_df(A, k, bench), "loading",
                f"{titles.get(names[k], names[k])} loadings{suffix}",
                reference=0.0, ref_label="0")
        # All-axis overview in axis SHARE, not raw loading: the fraction of a
        # benchmark's squared loading-row norm pointing along the axis. Purity
        # defines an axis better than steepness — on raw loadings a long
        # half-aligned row out-ranks a short pure one (see analysis.factors
        # loadings_table on bench_norm vs axis_share). Shares are sign-free,
        # so signed fits keep their loading signs in the per-axis forests above.
        med = np.median(A, axis=0)
        share = med**2 / np.maximum((med**2).sum(axis=1, keepdims=True), 1e-12)
        hm = loadings_heatmap_fig(
            share[None], names, bench, top_n=20,
            title=f"Benchmark axis share (top 20 per axis){suffix}")
        hm.data[0].colorbar.title = "axis share"
        hm.data[0].zmin, hm.data[0].zmax = 0.0, 1.0
        figs[f"{prefix}loadings_heatmap{tag}"] = hm
    return figs


def forecast_figures(view, data, raw, names, th_fc,
                     axis_titles: dict | None = None) -> dict:
    """Frontier-projection figure set: per axis a timeline overlay, a crossover
    'when' chart and an exceedance-probability curve.

    ONE definition for both the dashboard card and the forecast-only re-render
    (`3_diagnostics/forecast_only.py`), so the two cannot drift apart on
    fit_basis, sd_cap or the SOTA rule. `th_fc` is the ability array to forecast on, in the frame the
    caller's timelines already use. `axis_titles` swaps display text only,
    same convention as `build_axis_figures`.
    """
    figs = {}
    K = view.K
    titles = axis_titles or {}
    from multiaxis_eci.analysis import (mirt_crossover_df, mirt_frontier_forecast,
                          mirt_human_axis_stats, mirt_model_timeline_df)
    from multiaxis_eci.config import FORECAST_KW, FORECAST_NO_SOTA_AXES
    for k in range(K):
        disp = titles.get(names[k], names[k])
        # Informed cloud at SD < 0.4 (low-obs dropped): the forecast panel
        # shows only measured abilities — an extrapolated pre-2023 model at
        # prior-wide CI is not evidence about the frontier. The forecast
        # set uses 0.4, looser than the 0.3 of the measured timelines,
        # because the record fit below shares this cap and at 0.3 a thin
        # axis freezes at 2 records and the line detaches from the data
        # (axis 4, 2026-08-13). The cloud and the fit must share one cap so
        # every fitted point is a plotted point. SOTA releases bypass the
        # filter inside mirt_model_timeline_df and stay drawn with their
        # honest wide CI. All intervals on this figure set are 50% HDIs:
        # the whiskers, the human bands, the forecast band and the
        # crossover dates.
        tl = mirt_model_timeline_df(th_fc, k, data, raw, sd_cap=0.4,
                                    hdi_prob=0.5)
        if tl.empty:
            continue
        try:
            # Frontier trend on the record set (fit_basis="records",
            # SD < 0.4 on this axis, low-obs dropped, SOTA exempt from
            # both filters), fit from Oct 2024 on (the reasoning-model
            # cutoff). The exemption keeps the frontier releases in the
            # fit: on a thin axis they carry SD ~1.0-1.5 and their means
            # are mostly the lineage prior (successor + positive drift),
            # so the slope is part evidence and part prior. That is the
            # accepted cost of a line that tracks the visible frontier
            # instead of stopping at the last well-measured record.
            # back_start only matters to the regression bases; the
            # envelope (FORECAST_KW's default) ignores it and draws the
            # observed record steps from its own window start, with the
            # forward rate measured over the last rate_window years.
            back = pd.to_datetime(tl["release_date"]).min()
            from multiaxis_eci.config import FORECAST_BACKCAST_FLOOR
            fc = mirt_frontier_forecast(th_fc, k, data, raw,
                                        **dict(FORECAST_KW,
                                              sota_exempt=k not in FORECAST_NO_SOTA_AXES,
                                              back_start=back,
                                              backcast_floor=FORECAST_BACKCAST_FLOOR.get(f"axis{k + 1}")))
        except ValueError:
            continue
        hstat = mirt_human_axis_stats(th_fc, k, data, hdi_prob=0.5)
        cx = mirt_crossover_df(fc, th_fc, k, data, axis_name=names[k],
                               hdi_prob=0.5)
        slug = _slug(names[k])
        figs[f"forecast_{k+1}_{slug}"] = capability_forecast_fig(
            tl, hstat, fc, cx, axis_name=disp)
        figs[f"forecast_{k+1}_{slug}"].update_layout(
            title_text=f"Forecast — {disp} (50% intervals)")
        figs[f"forecast_{k+1}_{slug}_when"] = crossover_dotwhisker_fig(
            cx, axis_name=disp)
        figs[f"forecast_{k+1}_{slug}_when"].update_layout(
            title_text=f"Forecast — {disp} (crossover dates, 50% intervals)")
        figs[f"forecast_{k+1}_{slug}_prob"] = exceedance_prob_fig(
            fc, th_fc, k, data, axis_name=disp)
    return figs


def build_fit_figures(view, gof, yrep, data, raw, bench, mod, idata,
                      forecast: bool = False) -> dict:
    """Canonical per-fit MIRT figure set, gated by fit family (comp / nc).

    Returns a name→go.Figure dict instead of writing files, so the dashboard
    embeds them and the single-fit CLI saves them. `view` is a `analysis.FitView`. Diagnostic honesty: GoF/PIT come from `gof`
    (computed on the WHOLE fit); the timeline SD/informed filters only touch the
    timeline figures, never the returned metrics.

    `forecast=True` adds the frontier-projection figure set (per axis: timeline
    overlay + crossover 'when' chart + exceedance-probability curves) in the same
    oblique frame as the timelines. Opt-in per fit because it only makes sense
    where human tiers are in the fit."""
    from multiaxis_eci.analysis import (mirt_difficulty_timeline_df, mirt_human_axis_stats,
                          mirt_model_timeline_df, nc_difficulty_timeline_df,
                          tau_spectrum_df, trace_anchors, trace_loading_prior)
    figs = {}
    hover = [f"{mod[m]} · {bench[b]}" for m, b in zip(data.model_idx, data.bench_idx)]
    bench_of_obs = [bench[b] for b in data.bench_idx]
    # Item characteristic curve inputs — compensatory fits only: the sum link has
    # one logit η = A_b·θ_m − D_b per observation, the non-comp product link does
    # not. The fixed floor and the estimated ceiling, when the fit carries them,
    # set the curve asymptotes.
    eta_of_obs = floor = ceiling = None
    # loglog link: ICC eta needs alpha * log(A . theta_pos); skipped here like the nc family
    is_loglog = "alpha" in idata.posterior
    if not view.is_nc and view.A is not None and not is_loglog:
        import json
        A_mean, th_mean = view.A.mean(0), view.theta.mean(0)
        D_mean = idata.posterior["D"].mean(("chain", "draw")).values
        eta_of_obs = ((A_mean[data.bench_idx] * th_mean[data.model_idx]).sum(-1)
                      - D_mean[data.bench_idx])
        attrs = idata.posterior.attrs
        if "mirt_floor_c" in attrs:
            floor = dict(zip(bench, json.loads(attrs["mirt_floor_c"])))
        if "ceiling_d" in idata.posterior:
            # Estimated asymptote (--ceiling-noise): the fitted d is a posterior
            # variable, same source the PPC reads. Averaged over the chains this
            # card keeps, so a mode-restricted card draws its own mode's ceiling.
            ceiling = dict(zip(bench, idata.posterior["ceiling_d"]
                               .mean(("chain", "draw")).values))
    figs.update(build_gof_figures(data.scores, gof.y_pred_mean, yrep, gof.pit,
                                  hover, bench_of_obs,
                                  model_of_obs=[mod[m] for m in data.model_idx],
                                  eta_of_obs=eta_of_obs, floor=floor, ceiling=ceiling))

    K, names = view.K, view.names
    signed_frames = signed_display_frames(view, idata)
    is_signed = signed_frames is not None
    # Timelines and loadings come from one builder (so a mode view reuses it),
    # but they sit on either side of the difficulty timeline on the page. The
    # loading block is held back and re-inserted below to hold that fixed order:
    # a card whose trace is on a superseded data generation can never be
    # re-rendered, so a reordering here would leave the dashboard permanently
    # inconsistent between those cards and fresh ones.
    axis_figs = build_axis_figures(view, data, raw, bench, signed_frames)
    load_figs = {k: axis_figs.pop(k) for k in list(axis_figs)
                 if k.startswith("loadings")}
    figs.update(axis_figs)

    diff_df = (nc_difficulty_timeline_df(idata, data, raw) if view.is_nc
               else mirt_difficulty_timeline_df(idata, data, raw))
    figd = capability_timeline_fig(diff_df, human_stats=None)
    figd.update_layout(title=dict(text="Benchmark difficulty (50% intervals)", x=0.5),
                       yaxis=dict(title="difficulty (b = −c)" if view.is_nc else "difficulty (D)"))
    figs["timeline_difficulty"] = figd

    # Frontier forecast (opt-in per fit): extrapolate the running-best model per
    # axis and project when it crosses each human tier. Same oblique frame as the
    # timelines above; gated on humans being in the fit (nothing to cross otherwise).
    if forecast and not view.is_nc and K > 1 and data.is_human.any():
        th_fc = signed_frames["oblique"].theta if is_signed else view.theta
        figs.update(forecast_figures(view, data, raw, names, th_fc))

    figs.update(load_figs)          # held back above — legacy page position

    if K >= 2:
        # A signed fit's display Phi is the promax factor correlation, an
        # oblique quantity: title it as such and annotate the raw ability
        # correlation, instead of presenting promax under "(ability)".
        figs["factor_correlations"] = factor_corr_fig(
            view.Phi, names, rotated=view.phi_is_promax, Phi_raw=view.Phi_raw)

    if view.is_nc:
        Q = idata.constant_data["Q"].values
        qbench = idata.constant_data["Q"].coords["bench"].values.tolist()
        figs["qmatrix"] = binary_qmatrix_fig(Q, names, qbench,
                                             title="Q-matrix (conjunctive loadings)",
                                             multi_loaded=True)
    elif view.anchored:
        anchors = trace_anchors(idata)
        Q = np.ones((len(bench), K))
        for b, ax in anchors.items():
            if b not in bench:
                continue
            i = bench.index(b)
            Q[i, :] = 0.0
            Q[i, [ax] if isinstance(ax, int) else ax] = 1.0
        figs["qmatrix"] = binary_qmatrix_fig(Q, names, bench, title="Q-matrix (allowed loadings)")

    if (not view.is_nc and not view.anchored
            and view.tau is not None and K >= 2):
        # Shared-scale fits (signed/normal) have a FLAT tau_A by construction —
        # per-axis strength lives in the (aligned) loading column norms instead.
        tau_range = float((view.tau.max(axis=1) - view.tau.min(axis=1)).max())
        if is_signed:
            # Variance-ordered PCA-frame column norms — the honest spectrum
            # (matches the raw timelines/loadings frame, not varimax).
            strength, label, title = (np.linalg.norm(signed_frames["raw"].A, axis=1),
                                      "axis strength", "Axis strength")
        elif ((tau_range < 1e-6 or trace_loading_prior(idata) == "bifactor")
                and view.A is not None):
            # Bifactor's tau_A separates g from the specifics but not the
            # specifics from each other (one shared horseshoe scale), so read
            # strength off the loading columns as for the flat-tau fits.
            strength, label, title = (np.linalg.norm(view.A, axis=1), "axis strength",
                                      "Axis strength")
        else:
            strength, label, title = view.tau, "axis strength", "Axis strength"
        spec = tau_spectrum_df(strength).rename(columns={
            "tau_median": "mean", "tau_hdi_low": "hdi_low",
            "tau_hdi_high": "hdi_high", "axis": "name"})
        figs["axis_strength"] = forest_fig(
            spec[["name", "mean", "hdi_low", "hdi_high"]], label, title)
    return figs


# ── cross-fit comparison assembly (Unit 4): tables + all cmp_* figures ──────

def build_comparison(results: list):
    """Assemble the cross-fit comparison from the per-fit `results` dicts. Returns
    (tables, figures): tables = {gof_table, [loo_waic_table]}; figures = the cmp_*
    set — tables and figures are produced together in one pass."""
    gof_cols = ["fit", "type", "K", "free_loadings", "R2", "RMSE", "MAE",
                "PIT_var", "eta_rhat", "divergences", "max_phi"]
    gof_table = pd.DataFrame([{c: r[c] for c in gof_cols} for r in results])
    figs = {
        "cmp_per_benchmark_rmse": cmp_per_benchmark_rmse_fig(
            pd.DataFrame({r["fit"]: r["per_bench_rmse"] for r in results})),
        "cmp_gof":         cmp_gof_fig(gof_table),
        "cmp_convergence": cmp_convergence_fig(gof_table),
        "cmp_pit_ecdf":    cmp_pit_ecdf_fig(results),
    }
    tables = {"gof_table": gof_table}

    loo_results = [r for r in results if r.get("loo_pointwise") is not None]
    # ELPD is a SUM over observations, so comparing fits trained on different
    # observation counts is invalid (the bigger N just gets a bigger ELPD). Some
    # fits here DO differ (−SG drops 10 obs, −ARC-AGI drops a benchmark), so the
    # cross-fit ΔELPD figure is restricted to the MODAL obs group; the rest are
    # named in a footnote rather than plotted as silent NaN (invisible) rows.
    nobs = {r.get("n_obs") for r in loo_results} - {None}
    loo_cmp, loo_note = loo_results, None
    if len(nobs) > 1:
        from collections import Counter
        modal_n = Counter(r.get("n_obs") for r in loo_results
                          if r.get("n_obs")).most_common(1)[0][0]
        loo_cmp = [r for r in loo_results if r.get("n_obs") == modal_n]
        dropped = [r for r in loo_results if r.get("n_obs") != modal_n]
        loo_note = (f"ΔELPD omitted for fits on modified data (different obs, "
                    f"not comparable): " + ", ".join(r["name"] for r in dropped)
                    + f".  Shown fits share {modal_n} obs.")
        print(f"  LOO/WAIC ΔELPD figure restricted to the modal {modal_n}-obs group "
              f"({len(loo_cmp)} fits); omitted {[r['name'] for r in dropped]}.")
    if len(loo_results) >= 2:
        loo_cols = ["name", "loo_elpd", "loo_se", "loo_p_eff", "waic_elpd", "waic_se",
                    "waic_p_eff", "pareto_k_good", "pareto_k_ok", "pareto_k_bad",
                    "pareto_k_very_bad", "pareto_k_max", "pareto_k_mean"]
        loo_table = (pd.DataFrame([{c: r[c] for c in loo_cols} for r in loo_results])
                     .sort_values("loo_elpd", ascending=False).reset_index(drop=True))
        tables["loo_waic_table"] = loo_table
        if len(loo_cmp) >= 2:
            figs["cmp_loo_waic"] = cmp_loo_waic_fig(loo_cmp, note=loo_note)
        figs["cmp_pareto_k"] = cmp_pareto_k_fig(loo_results)
        df = gof_table.merge(loo_table[["name", "loo_elpd"]], left_on="fit",
                             right_on="name", how="left")
        # Same modal-obs restriction as the ΔELPD figure above: the trust
        # chart's first panel is `loo_elpd - max`, an ELPD-sum comparison,
        # invalid across fits trained on different observation counts.
        df = df[df["name"].isin({r["name"] for r in loo_cmp})]
        # min/median ESS on the identified eta, plotted against the kept-draw
        # count: cards keep different numbers of draws, so the draw ceiling is
        # what makes an absolute ESS readable.
        df["ess_min"] = df["fit"].map(
            {r["fit"]: r["eta_ess_min"] for r in results if r.get("eta_ess_min")})
        df["ess_med"] = df["fit"].map(
            {r["fit"]: r["eta_ess_med"] for r in results if r.get("eta_ess_med")})
        df["n_draws"] = df["fit"].map(
            {r["fit"]: r["n_draws_kept"] for r in results if r.get("n_draws_kept")})
        df = df.dropna(subset=["loo_elpd", "ess_min"]).sort_values("loo_elpd")
        if len(df):
            figs["cmp_loo_vs_trust"] = cmp_loo_vs_trust_fig(df)

    taus = {(r["fit"], r["type"]): r["tau_sorted"]
            for r in results if r.get("tau_sorted") is not None}
    if taus:
        figs["cmp_tau_spectrum"] = cmp_tau_spectrum_fig(taus)

    return tables, figs


# ── self-contained dashboard assembler (Unit 5) ─────────────────────────────

_DASH_CSS = """
body{margin:0;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;color:#222}
.layout{display:flex}
.sidebar{width:240px;min-width:240px;height:100vh;overflow-y:auto;position:sticky;top:0;
  background:#fafafa;border-right:1px solid #e5e5e5;padding:12px 0;box-sizing:border-box}
.navgroup{font-size:11px;text-transform:uppercase;color:#999;padding:12px 16px 4px;letter-spacing:.06em}
.navbroad{font-size:12px;color:#444;font-weight:600;padding:10px 16px 3px;line-height:1.35}
.navitem{display:block;padding:6px 16px;color:#127a56;text-decoration:none;font-size:13px;
  cursor:pointer;border-left:3px solid transparent}
.navitem.sub{padding-left:32px}
.navitem:hover{background:#f0f0f0}
.navitem.active{border-left-color:#1D9E75;background:#eef6f2;font-weight:600;color:#111}
.content{flex:1;min-width:0;padding:20px 28px;box-sizing:border-box}
.section{display:none}
.section h2{margin-top:0}
.plot{max-width:100%;overflow-x:auto;margin:6px 0 26px}
.headline{background:#f7f7f7;border-left:4px solid #1D9E75;padding:12px 16px;margin-bottom:16px;
  font-size:14px;line-height:1.55}
.statline{color:#555;font-size:13px;margin-bottom:12px}
table.cmp{border-collapse:collapse;font-size:12px;margin:10px 0 22px}
table.cmp th,table.cmp td{border:1px solid #ddd;padding:3px 8px;text-align:right;white-space:nowrap}
table.cmp th{background:#f2f2f2}
h3{font-size:12px;color:#777;margin:18px 0 2px;font-weight:600;letter-spacing:.02em}
"""

_DASH_JS = """
function renderSection(id){
  var sec=document.getElementById(id); if(!sec) return;
  var divs=sec.querySelectorAll('.plot');
  for(var i=0;i<divs.length;i++){
    var d=divs[i];
    if(!RENDERED[d.id]){
      var spec=PLOTS[d.id];
      if(spec){Plotly.newPlot(d,spec.data,spec.layout,{responsive:true,displaylogo:false});RENDERED[d.id]=true;}
    }else{Plotly.Plots.resize(d);}
  }
}
function showSection(id){
  var s=document.querySelectorAll('.section');
  for(var i=0;i<s.length;i++)s[i].style.display='none';
  var n=document.querySelectorAll('.navitem');
  for(var j=0;j<n.length;j++)n[j].classList.remove('active');
  var sec=document.getElementById(id); if(sec)sec.style.display='block';
  var nav=document.querySelector('.navitem[data-target="'+id+'"]'); if(nav)nav.classList.add('active');
  renderSection(id);
}
function initDashboard(){var h=(location.hash||'#cmp').slice(1);if(!document.getElementById(h))h='cmp';showSection(h);}
"""


def assemble_dashboard(fit_sections: list, comparison: dict,
                       title: str = "Capability-dimensionality dashboard") -> str:
    """Assemble ONE self-contained HTML dashboard string.

    Plotly JS is embedded once; each figure's JSON is embedded and rendered LAZILY
    (`Plotly.newPlot` on a section's first reveal, `resize` thereafter), so only the
    visible section's figures are ever live in the DOM.

    fit_sections : list of {id, label, type, stat_line, figures:{name: go.Figure}}
                   plus optional {group, nav, table_html} — a shared `group`
                   collapses consecutive fits under one broad sidebar title, each
                   listed by its short `nav` name; `table_html` (the posterior-mode
                   summary) goes between the stat line and the figures.
    comparison   : {headline: html, tables_html: html, figures:{name: go.Figure}}.
    """
    import plotly.io as pio
    from plotly.offline import get_plotlyjs
    plots_js = {}

    def reg(fig, dom_id):
        plots_js[dom_id] = pio.to_json(fig)
        return f'<div class="plot" id="{dom_id}"></div>'

    # comparison section (headline optional — omitted entirely when absent)
    cmp_body = []
    if comparison.get("headline"):
        cmp_body.append(f'<div class="headline">{comparison["headline"]}</div>')
    cmp_body.append(comparison.get("tables_html", ""))
    for name, fig in comparison.get("figures", {}).items():
        cmp_body.append(f'<h3>{name}</h3>' + reg(fig, f'cmp__{name}'))
    sections = [f'<section class="section" id="cmp"><h2>Comparison</h2>{"".join(cmp_body)}</section>']

    nav = ['<a class="navitem" data-target="cmp" href="#cmp" '
           'onclick="showSection(\'cmp\');return false;">Comparison</a>']
    groups = {}
    for s in fit_sections:
        groups.setdefault(s["type"], []).append(s)
    # Only these four types are rendered below, so a section carrying any other
    # would be silently dropped from both the nav and the HTML.
    known = ("data", "baseline", "exploratory", "confirmed")
    unknown = sorted(set(groups) - set(known))
    if unknown:
        raise ValueError(f"section type(s) {unknown} would not be rendered — "
                         f"known types: {list(known)}")
    for typ in known:
        prev_group = None
        for i, s in enumerate(groups.get(typ, [])):
            if i == 0:
                nav.append(f'<div class="navgroup">{typ.capitalize()}</div>')
            # Consecutive fits sharing a `group` (e.g. the modes of one fit) get
            # the group as a broad title once, then one indented entry each.
            g = s.get("group")
            if g and g != prev_group:
                nav.append(f'<div class="navbroad">{g}</div>')
            prev_group = g
            cls, text = (("navitem sub", s.get("nav") or s["label"]) if g
                         else ("navitem", s["label"]))
            nav.append(f'<a class="{cls}" data-target="{s["id"]}" href="#{s["id"]}" '
                       f'onclick="showSection(\'{s["id"]}\');return false;">{text}</a>')
            body = [f'<div class="statline">{s.get("stat_line", "")}</div>']
            if s.get("table_html"):
                body.append(s["table_html"])
            for name, fig in s["figures"].items():
                body.append(f'<h3>{name}</h3>' + reg(fig, f'{s["id"]}__{name}'))
            sections.append(f'<section class="section" id="{s["id"]}">'
                            f'<h2>{s["label"]}</h2>{"".join(body)}</section>')

    plotly_js = get_plotlyjs()
    plots_json = "{" + ",".join(f'"{k}":{v}' for k, v in plots_js.items()) + "}"
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{title}</title><style>{_DASH_CSS}</style>"
        f"<script>{plotly_js}</script>"
        f"<script>var PLOTS={plots_json};var RENDERED={{}};{_DASH_JS}</script>"
        "</head><body>"
        f'<div class="layout"><nav class="sidebar">{"".join(nav)}</nav>'
        f'<main class="content">{"".join(sections)}</main></div>'
        "<script>initDashboard();</script>"
        "</body></html>"
    )


def write_dashboard(html: str, path: "Path") -> None:
    """Write the assembled dashboard HTML string to `path` (creates parent dirs)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html)
