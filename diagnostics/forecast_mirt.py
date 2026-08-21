"""Forecast when the AI frontier outpaces each human tier, per axis.

Runs on a converged signed K=3 MIRT trace (default: the no-SG human+lineage
prior fit). For every axis it fits a per-draw linear trend to the frontier
(running-best model) and extrapolates it, then reports the projected crossover
date + CI against each human tier and whether that tier is already passed.

Emits, under plots/<out>/:
  * timeline_axis{k}_<slug>        — the plain capability timeline (unchanged)
  * forecast_axis{k}_<slug>        — timeline + forecast band + crossover markers
  * forecast_axis{k}_<slug>_when   — crossover date dot-whisker per tier
  * forecast_axis{k}_<slug>_prob   — P(frontier > tier) over time
and results/<out>/forecast_crossover.csv (all axes concatenated).

Run:
  ~/miniforge3/envs/pymc_env/bin/python diagnostics/forecast_mirt.py --K 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import arviz as az
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis import (  # noqa: E402
    mirt_crossover_df, mirt_frontier_forecast, mirt_human_axis_stats,
    mirt_model_timeline_df, prepare_fit,
)
from config import SG_MODEL_NAME  # noqa: E402
from data import PROCESSED_FILE, drop_model_observations, load_eci_data  # noqa: E402
from viz import (  # noqa: E402
    capability_forecast_fig, capability_timeline_fig, crossover_dotwhisker_fig,
    exceedance_prob_fig, save_fig,
)

DEFAULT_TRACE = ROOT / "results" / "mirt_noSG" / "trace_mirt_k3_signed_noSG_priors.nc"


def _slug(s: str) -> str:
    return s.replace("/", "_").replace(" ", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=str(DEFAULT_TRACE),
                    help="path to the MIRT trace (.nc)")
    ap.add_argument("--K", type=int, default=3, help="latent dimension of the trace")
    ap.add_argument("--no-sg", dest="no_sg", action="store_true", default=True,
                    help="drop Skilled Generalist obs to match the no-SG fit (default on)")
    ap.add_argument("--keep-sg", dest="no_sg", action="store_false",
                    help="do NOT drop SG obs (for traces fit with SG kept)")
    ap.add_argument("--fit-start", default="2023-01-01",
                    help="ignore frontier points before this date when fitting the trend")
    ap.add_argument("--horizon", default=None,
                    help="extrapolate to this date (default: last obs + 5y, capped 2032)")
    ap.add_argument("--drop-chains", default="",
                    help="comma-separated chain indices to drop before analysis "
                         "(0-indexed, arviz chain coords), e.g. mode-restricted readout")
    ap.add_argument("--out", default="mirt_noSG", help="subfolder under plots/ and results/")
    args = ap.parse_args()

    raw = pd.read_csv(PROCESSED_FILE)
    idata = az.from_netcdf(args.trace)
    if args.drop_chains.strip():
        drop = [int(c) for c in args.drop_chains.split(",") if c.strip() != ""]
        keep = [c for c in idata.posterior.chain.values if c not in drop]
        idata = idata.sel(chain=keep)
        print(f"  dropped chain(s) {drop}; kept {len(keep)} of {len(keep) + len(drop)}: {keep}")
    data = load_eci_data(include_all_benchmarks=True, drop_low_obs_models=False)
    if idata.posterior.sizes.get("model") != data.n_models:
        raise RuntimeError(
            f"trace has {idata.posterior.sizes.get('model')} models but the current "
            f"data loads {data.n_models}. The trace's model index must match the data "
            "exactly. This usually means the processed data has drifted since the fit "
            "(check `git status` for modified data/processed or data/curated files) — "
            "run against the data snapshot the trace was fit on, or use a trace fit on "
            "the current data.")
    if args.no_sg:
        data = drop_model_observations(data, [SG_MODEL_NAME])
        print(f"  dropped '{SG_MODEL_NAME}' observations to match the no-SG fit")

    view = prepare_fit(idata, data)
    plots_dir = ROOT / "plots" / args.out
    results_dir = ROOT / "results" / args.out
    results_dir.mkdir(parents=True, exist_ok=True)

    crossovers = []
    for k in range(view.K):
        name = view.names[k]
        slug = _slug(name)
        tl = mirt_model_timeline_df(view.theta, k, data, raw)
        hstat = mirt_human_axis_stats(view.theta, k, data)
        if tl.empty:
            print(f"  axis {k + 1} ({name}): no dated models — skipped")
            continue

        # Plain timeline, kept as its own figure (existing view, untouched).
        save_fig(capability_timeline_fig(tl, human_stats=hstat),
                 f"timeline_axis{k + 1}_{slug}", plots_dir)

        fc = mirt_frontier_forecast(view.theta, k, data, raw,
                                    horizon_date=args.horizon, fit_start=args.fit_start)
        cx = mirt_crossover_df(fc, view.theta, k, data, axis_name=name)
        crossovers.append(cx)

        save_fig(capability_forecast_fig(tl, hstat, fc, cx, axis_name=name),
                 f"forecast_axis{k + 1}_{slug}", plots_dir)
        save_fig(crossover_dotwhisker_fig(cx, axis_name=name),
                 f"forecast_axis{k + 1}_{slug}_when", plots_dir)
        save_fig(exceedance_prob_fig(fc, view.theta, k, data, axis_name=name),
                 f"forecast_axis{k + 1}_{slug}_prob", plots_dir)

        passed = cx[cx["status"].str.startswith("passed")]["tier"].tolist()
        future = cx[cx["status"] == "future"]
        print(f"  axis {k + 1} ({name}): frontier from {fc.last_obs_date.date()}"
              f" ({len(fc.frontier_names)} record-setters)")
        print(f"    already passed: {passed or '—'}")
        for _, r in future.iterrows():
            print(f"    {r['tier']}: ~{pd.Timestamp(r['crossover_date_median']).date()}"
                  f" (P now {r['p_passed_now']:.2f})")

    if crossovers:
        out_csv = results_dir / "forecast_crossover.csv"
        pd.concat(crossovers, ignore_index=True).to_csv(out_csv, index=False)
        print(f"  wrote {out_csv}")


if __name__ == "__main__":
    main()
