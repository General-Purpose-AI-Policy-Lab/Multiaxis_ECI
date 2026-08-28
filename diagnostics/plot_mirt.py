"""Plot a SINGLE MIRT fit in detail — run AFTER a fit.py / fits/* fit.

Single-fit deep-dive. Rotation/identity handling (`analysis.prepare_fit`) and
every figure builder (`viz/`) are the dashboard's
(`diagnostics/build_dashboard.py`); this script adds the single-fit-only
comparative views (axis frontiers, ability scatter-matrix) and the K-vs-1D
block.

The fit's flag set, its data scope and its results folder come from
`analysis.FitSpec.from_trace`, so a trace path is the only thing a caller has
to name. Per-figure PNG/HTML are written to `plots/<out>/` (git-ignored) for
iterating on one freshly-fit trace without rebuilding the whole dashboard.

Run:
  ~/miniforge3/envs/pymc_env/bin/python diagnostics/plot_mirt.py \
      --trace results/mirt_humanprior/trace_mirt_k3_humanprior.nc
  ~/miniforge3/envs/pymc_env/bin/python diagnostics/plot_mirt.py --folder results/

A multi-GB trace goes through `--thin` (the flagship uses
`analysis.FLAGSHIP_THIN`): every figure here is a median or an interval, and
the whole trace does not fit in RAM. `--folder` picks `--thin` from each file's
size and renders every trace in its own child process, so the OS reclaims the
memory between fits and one failure does not end the sweep.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import arviz as az

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis import (  # noqa: E402
    FitSpec, mirt_factors_from_trace, mirt_informed_mask,
    mirt_model_timeline_df, prepare_fit, trace_loading_prior,
)
from data import PROCESSED_FILE  # noqa: E402
from viz import (  # noqa: E402
    alignment_methods_fig, axes_frontier_fig, axes_scatter_matrix_fig,
    build_fit_figures,
    factor_vs_1d_fig, per_bench_r2_delta_fig, pit_ecdf_fig, pred_scatter_fig, save_fig,
)
from ppc import compute_gof, posterior_predictive_mirt  # noqa: E402

# Every posterior variable any figure or the PPC here reads. Names absent from a
# given trace are skipped, so this one list covers the linear and log-logistic
# links, the softplus-theta variant and the estimated ceiling.
PLOT_VARS = ("A", "theta", "theta_pos", "tau_A", "D", "phi_b", "alpha", "ceiling_d")


def _spec_of(trace_path) -> FitSpec:
    """The spec of a trace on disk. A lazy open reads attrs and dims only, no
    data, so this is cheap on a 38 GB file."""
    with xr.open_dataset(trace_path, group="posterior") as post:
        return FitSpec.from_trace(az.InferenceData(posterior=post), trace_path)


def plot_fit(trace_path, *, idata=None, axes=None, out=None, thin: int = 1,
             forecast: bool = False) -> Path:
    """Render the single-fit figure set for one trace. Returns the plots folder.

    `idata` lets a just-finished fit hand over the posterior it already holds.
    Otherwise only `PLOT_VARS` are read off disk, thinned by `thin`.

    `forecast` asks for the frontier-projection figures; the builder itself
    gates them on K > 1, the compensatory family and human tiers being in the
    fit, so a fit without those gets none.
    """
    trace_path = Path(trace_path)
    if idata is None:
        # The spec is known before deciding which variables to pull.
        spec = _spec_of(trace_path)
        idata = spec.open_posterior(keep=PLOT_VARS, thin=thin, path=trace_path)
    else:
        spec = FitSpec.from_trace(idata, trace_path)
    # Per-fit artefacts (the alignment CSV, the K=1 baseline) are looked up
    # BESIDE the trace, not under `spec.results_dir`: a trace fitted under an
    # earlier tag grammar sits in a folder the spec no longer derives, and its
    # own folder is the one that holds its files.
    results_dir = trace_path.parent
    plots_dir = Path(out) if out else spec.plots_dir
    plots_dir.mkdir(parents=True, exist_ok=True)
    if results_dir != spec.results_dir:
        print(f"  trace folder {results_dir.name!r} predates this spec's tag "
              f"{spec.tag!r}; per-fit files read from there, figures → {plots_dir}")

    raw = pd.read_csv(PROCESSED_FILE)
    data, floor_c, n_eff = spec.load_data(idata)

    view = prepare_fit(idata, data)
    K = view.K
    n_axes = min(axes or K, K)
    names = view.names
    bench = data.blookup.sort_values("benchmark_idx")["benchmark"].tolist()
    mod = data.mlookup.sort_values("model_idx")["model"].tolist()
    # ── canonical per-fit figure set (shared with the dashboard) ──────────────
    yrep = posterior_predictive_mirt(idata, data, floor_c=floor_c, n_eff=n_eff)
    mu = posterior_predictive_mirt(idata, data, floor_c=floor_c, n_eff=n_eff,
                                   return_mean=True)
    gof = compute_gof(yrep, data, mu)
    figs = build_fit_figures(view, gof, yrep, data, raw, bench, mod, idata,
                             forecast=forecast)
    figs["pit_ecdf"] = pit_ecdf_fig(gof.pit)

    # ── signed extras: the rotation-method comparison (four independent
    # post-hoc identifications of the same trace), read from the CSV
    # `diagnostics/align_mirt.py` wrote — no recompute. K-tagged name first
    # (K=2 and K=3 share a results dir), untagged as fallback.
    if trace_loading_prior(idata) == "signed":
        for cand in (results_dir / f"mirt_alignment_loadings_k{K}.csv",
                     results_dir / "mirt_alignment_loadings.csv"):
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
    base = results_dir / "trace_mirt_k1.nc"
    if base.exists() and base != trace_path:
        idata_1d = spec.open_posterior(keep=PLOT_VARS, thin=thin, path=base)
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
    return plots_dir


def folder_decision(name: str, size_bytes: int, thin: int | None = None):
    """(skip reason or None, thin) for one candidate trace under `--folder`.

    Two filenames are not fits of their own: `trace.nc` is the canonical index,
    and a bare `trace_mirt_k1.nc` is the helper baseline fit.py fits beside a
    K-axis fit, which `plot_fit` already reads from the K-axis trace's own
    folder. Thin keeps one draw per 2 GB of file, the ratio
    that fits the 38 GB flagship inside 26 GB of RAM. An explicit `thin` wins.
    """
    if name == "trace.nc":
        return "canonical index, not a MIRT trace", 1
    if name == "trace_mirt_k1.nc":
        return "K=1 helper baseline, plotted with its K-axis fit", 1
    return None, thin if thin else max(1, round(size_bytes / 2e9))


def sweep(folder, *, axes=None, thin=None, forecast=True, dry_run=False) -> int:
    """Render every MIRT trace under `folder`. Returns a process exit code.

    Traces sit one level down (`results/mirt{tag}/trace_*.nc`); the flat glob
    also accepts a single fit folder. Each render is a child process, so its
    memory goes back to the OS and a crash costs one fit. Exit is non-zero only
    when every candidate failed, since a skip is a decision, not a failure.
    """
    root = Path(folder)
    paths = sorted(set(root.glob("*/*.nc")) | set(root.glob("*.nc")))
    n_render = n_skip = n_fail = 0
    for p in paths:
        why, n = folder_decision(p.name, p.stat().st_size, thin)
        if why is None:
            try:
                target = _spec_of(p).plots_dir
            except Exception as e:
                why = f"unreadable fit spec: {e}"
        if why is not None:
            print(f"skip     {p}  ({why})")
            n_skip += 1
            continue
        if dry_run:
            print(f"render   {p}  thin={n}  → {target}")
            continue
        cmd = [sys.executable, __file__, "--trace", str(p), "--thin", str(n)]
        if forecast:
            cmd.append("--forecast")
        if axes:
            cmd += ["--axes", str(axes)]
        if subprocess.run(cmd).returncode == 0:
            print(f"rendered {p}  thin={n}  → {target}")
            n_render += 1
        else:
            print(f"failed   {p}  (child process exited non-zero)")
            n_fail += 1
    print(f"\n{len(paths)} candidates: {n_render} rendered, {n_skip} skipped, "
          f"{n_fail} failed")
    return 1 if n_fail and not n_render else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--trace", help="path to the fitted trace")
    src.add_argument("--folder", help="render every MIRT trace under DIR "
                                     "(one child process each, forecasts on)")
    ap.add_argument("--axes", type=int, default=None,
                    help="how many top axes to plot (default: all of them)")
    ap.add_argument("--out", default=None,
                    help="output folder (default: the fit's plots/mirt_k{K}{tag}/)")
    ap.add_argument("--thin", type=int, default=None,
                    help="keep every n-th draw — needed on a trace larger than "
                         "RAM. --folder picks it from the file size unless set")
    ap.add_argument("--forecast", action="store_true",
                    help="add the frontier-projection figures (K>1 fits with "
                         "human tiers only); on by default under --folder")
    ap.add_argument("--dry-run", action="store_true",
                    help="with --folder: print the per-trace decision, render "
                         "nothing")
    args = ap.parse_args()
    if args.folder:
        sys.exit(sweep(args.folder, axes=args.axes, thin=args.thin,
                       dry_run=args.dry_run))
    plot_fit(args.trace, axes=args.axes, out=args.out, thin=args.thin or 1,
             forecast=args.forecast)


if __name__ == "__main__":
    main()
