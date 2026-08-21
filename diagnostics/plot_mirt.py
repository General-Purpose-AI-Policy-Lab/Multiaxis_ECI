"""Plot a SINGLE MIRT fit in detail — run AFTER a fit.py / fits/* fit.

Lean single-fit deep-dive. It shares ALL rotation/identity handling
(`analysis.prepare_fit`) and every figure builder (`viz/`) with the dashboard
(`diagnostics/build_dashboard.py`), so there is no duplicated logic, and adds
the single-fit-only comparative views (axis frontiers, ability scatter-matrix)
and the K-vs-1D block.

Per-figure PNG/HTML are written to `plots/<out>/` (git-ignored) for iterating on
one freshly-fit trace without rebuilding the whole dashboard.

Run:
  ~/miniforge3/envs/pymc_env/bin/python diagnostics/plot_mirt.py --K 3 --anchors qmatrix3 --loading-prior normal
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis import (  # noqa: E402
    mirt_factors_from_trace, mirt_informed_mask,
    mirt_model_timeline_df, prepare_fit, trace_loading_prior,
)
from config import SG_MODEL_NAME  # noqa: E402
from data import (  # noqa: E402
    PROCESSED_FILE, clip_scores_to_floors, drop_model_benchmark_cells,
    drop_model_observations, load_benchmark_ceilings, load_benchmark_floors,
    load_eci_data,
)
from viz import (  # noqa: E402
    alignment_methods_fig, axes_frontier_fig, axes_scatter_matrix_fig,
    build_fit_figures,
    factor_vs_1d_fig, per_bench_r2_delta_fig, pit_ecdf_fig, pred_scatter_fig, save_fig,
)
from ppc import compute_gof, posterior_predictive_mirt  # noqa: E402

RESULTS_DIR = ROOT / "results" / "mirt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=4, help="which fitted trace to load")
    ap.add_argument("--anchors", choices=["none", "qmatrix3", "qmatrix3x"], default="none",
                    help="load the anchored trace for this K (matches fit.py's tag)")
    ap.add_argument("--loading-prior",
                    choices=["normal", "signed", "signedhs", "bifactor"],
                    default="signed",
                    help="load the trace for this loading prior (must match the fit)")
    ap.add_argument("--human-prior", action="store_true",
                    help="load the human-prior trace (from results/mirt_humanprior/)")
    ap.add_argument("--human-merge", action="store_true",
                    help="load a --human-merge fit: matches fit.py's _humanmerge tag")
    ap.add_argument("--lineage-prior", action="store_true",
                    help="load the lineage-prior trace from results/mirt_lineageprior/")
    ap.add_argument("--lineage-bm", action="store_true",
                    help="load a --lineage-bm fit: matches fit.py's _lineagebm tag")
    ap.add_argument("--time-prior", action="store_true",
                    help="load a --time-prior fit: matches fit.py's _timeprior tag")
    ap.add_argument("--post-2023", action="store_true",
                    help="load a --post-2023 fit: matches fit.py's _post2023 "
                         "tag and applies the same era filter to the data")
    ap.add_argument("--no-sg", action="store_true",
                    help="load a --no-sg fit: matches fit.py's _noSG tag and "
                         "drops the same Skilled Generalist observations so GoF "
                         "is scored on the data the trace was fit to")
    ap.add_argument("--no-sg-gpqa", action="store_true",
                    help="load a --no-sg-gpqa fit: matches fit.py's _noSGgpqa "
                         "tag and drops the same SG GPQA cells for GoF scoring")
    ap.add_argument("--no-sg-arcagi", action="store_true",
                    help="load a --no-sg-arcagi fit: matches fit.py's "
                         "_noSGarcagi tag and drops the same SG ARC-AGI cells")
    ap.add_argument("--floors", action="store_true",
                    help="load a --floors fit: matches fit.py's _floors tag, "
                         "clips the same below-floor scores, and scores PPC/GoF "
                         "with the fixed-c 3PL link the trace was fit with")
    ap.add_argument("--ceilings", action="store_true",
                    help="load a --ceilings fit: matches fit.py's _ceilings tag "
                         "and scores PPC/GoF with the fixed-d link")
    ap.add_argument("--ceiling-noise", action="store_true",
                    help="load a --ceiling-noise fit: matches fit.py's _ceilnoise "
                         "tag (the estimated ceiling is read off the trace "
                         "automatically by the PPC); combines with --ceilings")
    ap.add_argument("--known-se", action="store_true",
                    help="load a --known-se fit: matches fit.py's _knownse tag "
                         "and scores PPC/GoF with the same per-cell instrument "
                         "precision the trace was fit with")
    ap.add_argument("--pooled-noise", action="store_true",
                    help="load a --pooled-noise fit: matches fit.py's "
                         "_poolednoise tag (the pooled sigma_b is read off the "
                         "trace like any other, so PPC/GoF are unchanged)")
    ap.add_argument("--apply-exclusions", action="store_true",
                    help="load an --apply-exclusions fit: matches fit.py's "
                         "_excluded tag and applies curated exclusions to the "
                         "data so PPC/GoF score the same benchmark set")
    ap.add_argument("--axes", type=int, default=4, help="how many top axes to plot")
    ap.add_argument("--out", default="mirt", help="output subfolder under plots/")
    args = ap.parse_args()

    global RESULTS_DIR
    loading_prior = args.loading_prior
    tag = "" if loading_prior == "normal" else f"_{loading_prior}"
    if args.anchors != "none":
        tag += f"_a{args.anchors}"
    if args.human_merge:
        tag += "_humanmerge"
    elif args.human_prior:
        tag += "_humanprior"
    if args.lineage_prior:
        tag += "_lineageprior"
    if args.lineage_bm:
        tag += "_lineagebm"
    if args.time_prior:
        tag += "_timeprior"
    if args.post_2023:
        tag += "_post2023"
    if args.no_sg:
        tag += "_noSG"
    if args.no_sg_gpqa:
        tag += "_noSGgpqa"
    if args.no_sg_arcagi:
        tag += "_noSGarcagi"
    if args.apply_exclusions:
        tag += "_excluded"
    if args.floors:
        tag += "_floors"
    if args.ceilings:
        tag += "_ceilings"
    if args.ceiling_noise:
        tag += "_ceilnoise"
    if args.known_se:
        tag += "_knownse"
    if args.pooled_noise:
        tag += "_poolednoise"

    out = args.out
    if (args.lineage_prior or args.lineage_bm or args.time_prior
            or args.human_prior or args.human_merge or args.post_2023
            or args.no_sg or args.no_sg_gpqa or args.no_sg_arcagi
            or args.apply_exclusions or args.floors or args.ceilings
            or args.ceiling_noise or args.known_se or args.pooled_noise):
        sub = f"mirt{tag}"
        RESULTS_DIR = ROOT / "results" / sub
        if out == "mirt":
            out = sub
    plots_dir = ROOT / "plots" / out
    plots_dir.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(PROCESSED_FILE)
    idata = az.from_netcdf(RESULTS_DIR / f"trace_mirt_k{args.K}{tag}.nc")

    # Match the data scope the trace was fit on (full vs drop-low-obs). Wrong
    # scope → IndexError, so detect it by the `model` coord size. The
    # --post-2023 scope flag must mirror the fit's.
    scope_kw = dict(min_release_date="2024-01-01" if args.post_2023 else None)
    include_all = not args.apply_exclusions
    trace_n_models = idata.posterior.sizes.get("model")
    data = load_eci_data(include_all_benchmarks=include_all, drop_low_obs_models=False,
                         **scope_kw)
    if trace_n_models == data.n_models:
        print(f"  trace uses ALL {trace_n_models} models")
    else:
        data = load_eci_data(include_all_benchmarks=include_all, drop_low_obs_models=True,
                             **scope_kw)
        if data.n_models != trace_n_models:
            raise RuntimeError(
                f"trace has {trace_n_models} models but neither full nor drop-low-obs "
                f"data shape matches ({data.n_models})")
        print(f"  trace uses DROP-LOW-OBS data ({trace_n_models} models)")

    if args.no_sg:
        # Mirror the fit: drop the SG observations so PPC/GoF score the data the
        # trace was actually fit to (n_models unchanged, so the check above holds).
        data = drop_model_observations(data, [SG_MODEL_NAME])
        print(f"  --no-sg: scoring on data with '{SG_MODEL_NAME}' observations dropped")
    if args.no_sg_gpqa:
        gpqa = [b for b in data.blookup["benchmark"] if "GPQA" in b]
        data = drop_model_benchmark_cells(data, SG_MODEL_NAME, gpqa)
        print(f"  --no-sg-gpqa: scoring with '{SG_MODEL_NAME}' GPQA cells dropped")
    if args.no_sg_arcagi:
        arcagi = [b for b in data.blookup["benchmark"] if b.startswith("ARC-AGI")]
        data = drop_model_benchmark_cells(data, SG_MODEL_NAME, arcagi)
        print(f"  --no-sg-arcagi: scoring with '{SG_MODEL_NAME}' ARC-AGI cells dropped")
    # Floors/ceilings must mirror the fit exactly: clip the same scores and
    # score PPC/GoF with the same 3PL/4PL link the trace was fit with.
    floor_c = None
    if args.floors:
        floor_c = load_benchmark_floors(data)
        data = clip_scores_to_floors(data, floor_c)
        print("  --floors: scores clipped to chance floors; GoF uses the fixed-c 3PL link")
    ceiling_d = None
    if args.ceilings:
        ceiling_d = load_benchmark_ceilings(data)
        print("  --ceilings: GoF uses the fixed-d link")
    # Same rule for the noise split: the predictive Beta needs the per-cell
    # instrument precision the fit used, or GoF/PIT score a different likelihood.
    n_eff = data.n_eff if args.known_se else None
    if n_eff is not None:
        print(f"  --known-se: GoF uses the split noise on "
              f"{int(np.isfinite(n_eff).sum())} cells with a reported stderr")

    view = prepare_fit(idata, data)
    K = view.K
    n_axes = min(args.axes, K)
    names = view.names
    bench = data.blookup.sort_values("benchmark_idx")["benchmark"].tolist()
    mod = data.mlookup.sort_values("model_idx")["model"].tolist()
    # ── canonical per-fit figure set (shared with the dashboard) ──────────────
    yrep = posterior_predictive_mirt(idata, data, floor_c=floor_c,
                                     ceiling_d=ceiling_d, n_eff=n_eff)
    mu = posterior_predictive_mirt(idata, data, floor_c=floor_c,
                                   ceiling_d=ceiling_d, n_eff=n_eff,
                                   return_mean=True)
    gof = compute_gof(yrep, data, mu)
    figs = build_fit_figures(view, gof, yrep, data, raw, bench, mod, idata)
    figs["pit_ecdf"] = pit_ecdf_fig(gof.pit)

    # ── signed extras: the rotation-method comparison (four independent
    # post-hoc identifications of the same trace), read from the CSV the fit
    # driver / align_mirt wrote — no recompute. K-tagged name first (K=2 and
    # K=3 share a results dir), legacy untagged as fallback.
    if trace_loading_prior(idata) in ("signed", "signedhs"):
        for cand in (RESULTS_DIR / f"mirt_alignment_loadings_k{K}.csv",
                     RESULTS_DIR / "mirt_alignment_loadings.csv"):
            if cand.exists():
                figs["rotation_methods"] = alignment_methods_fig(pd.read_csv(cand))
                break

    # ── single-fit comparative views (informed timelines) ────────────────────
    if n_axes >= 2:
        axis_tl = {k: mirt_model_timeline_df(view.theta, k, data, raw) for k in range(n_axes)}
        figs["axes_timeline_compare"] = axes_frontier_fig(axis_tl, names, n_axes)
        keep = (mirt_informed_mask(view.theta)[:, :n_axes].all(axis=1)
                & ~data.is_human & ~data.is_low_obs)
        idx = np.where(keep)[0]
        if len(idx) > 5:
            org_by_model = (raw.dropna(subset=["organization"])
                            .groupby("model_version")["organization"].first())
            orgs = np.array([org_by_model.get(mod[i], "other") for i in idx])
            top = pd.Series(orgs).value_counts().head(7).index.tolist()
            tmean = view.theta.mean(axis=0)
            dfm = pd.DataFrame({names[k]: tmean[idx, k] for k in range(n_axes)})
            dfm["model"] = [mod[i] for i in idx]
            dfm["org"] = np.where(np.isin(orgs, top), orgs, "other")
            figs["axes_scatter_matrix"] = axes_scatter_matrix_fig(
                dfm, [names[k] for k in range(n_axes)])

    # ── K vs 1D block (reuses the cached K=1 baseline; no extra fit) ──────────
    base = RESULTS_DIR / "trace_mirt_k1.nc"
    if base.exists():
        idata_1d = az.from_netcdf(base)
        if idata_1d.posterior.sizes.get("model") != data.n_models:
            print(f"  skipping 1D comparison: K=1 baseline has "
                  f"{idata_1d.posterior.sizes.get('model')} models vs {data.n_models}.")
        else:
            _, tb_theta, _ = mirt_factors_from_trace(idata_1d)
            tb = tb_theta.mean(axis=0)[:, 0]
            a1 = view.theta.mean(axis=0)[:, 0]
            if np.corrcoef(a1, tb)[0, 1] < 0:
                tb = -tb
            r = float(np.corrcoef(a1, tb)[0, 1])
            figs["factor1_vs_1d"] = factor_vs_1d_fig(tb, a1, mod, r)

            # fit.py fits the K=1 baseline with the same likelihood options.
            pred_1d = posterior_predictive_mirt(idata_1d, data, floor_c=floor_c,
                                                ceiling_d=ceiling_d,
                                                n_eff=n_eff).mean(axis=0)
            resid_kd = data.scores - gof.y_pred_mean
            resid_1d = data.scores - pred_1d
            hover = [f"{mod[m]} · {bench[b]}" for m, b in zip(data.model_idx, data.bench_idx)]
            figs["pred_k_vs_k1"] = pred_scatter_fig(
                pred_1d, gof.y_pred_mean, np.abs(resid_1d) - np.abs(resid_kd), hover)

            var_y = pd.Series(data.scores).groupby(data.bench_idx).var()
            n_per = pd.Series(np.ones_like(data.scores)).groupby(data.bench_idx).sum()
            r2_1d = 1 - pd.Series(resid_1d ** 2).groupby(data.bench_idx).sum() / (var_y * n_per)
            r2_kd = 1 - pd.Series(resid_kd ** 2).groupby(data.bench_idx).sum() / (var_y * n_per)
            delta = (r2_kd - r2_1d)
            bench_df = pd.DataFrame({
                "name": [bench[i] for i in delta.index],
                "delta_r2": delta.values, "n_obs": n_per.values.astype(int),
            }).sort_values("delta_r2")
            figs["r2_delta_per_bench"] = per_bench_r2_delta_fig(bench_df)

    for name, fig in figs.items():
        save_fig(fig, f"mirt_{name}", plots_dir)
    print(f"  PPC: R²={gof.metrics['bayesian_r2']:.3f}  RMSE={gof.metrics['rmse']:.3f}  "
          f"MAE={gof.metrics['mae']:.3f}")
    print(f"figures → {plots_dir}")


if __name__ == "__main__":
    main()
