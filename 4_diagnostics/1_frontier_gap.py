"""Open-weights vs closed frontier (or US vs CN with --group country) on one K=1 canonical trace,
in ECI-H: records per group, the months-behind lag per open record read off the closed frontier
of the same posterior draw, one trend line per group with the gap and the lag it implies today,
and the human-tier crossings of each line. See analysis.frontier_gap.

Runs on one benchmark access scope: the fit on all benchmarks (canonical/) or on one access class
(canonical_public/, canonical_semi_private/, canonical_private/, from
`3_fit/fit.py --preset canonical --access CLASS`). Outputs land in the data generation's
comparisons/ as frontier_gap_<group>_<scope>_*; 2_plot_frontier_gap.py then puts the scopes
side by side.

  python 4_diagnostics/1_frontier_gap.py [--access all|public|semi_private|private] [--group openness|country]
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "2_model"))

from multiaxis_eci import config  # noqa: E402
from multiaxis_eci.analysis.frontier_gap import (  # noqa: E402
    GROUP_TITLES,
    SCOPES,
    save_scope,
    scope_gap,
)
from multiaxis_eci.analysis.stats import (  # noqa: E402
    _release_dates,
    capability_draws,
    eci_transform,
)
from multiaxis_eci.analysis.timelines import candidate_mask  # noqa: E402
from multiaxis_eci.data import PROCESSED_FILE, access_scope_drop_list, load_eci_data  # noqa: E402
from multiaxis_eci.persistence import load_trace  # noqa: E402
from multiaxis_eci.viz.core import save_fig  # noqa: E402
from multiaxis_eci.viz.frontier_gap import SCOPE_TITLES, frontier_panels_fig, lag_fig  # noqa: E402


@dataclass
class _MiniData:
    """Duck-typed ECIData stand-in with only the fields the analysis reads."""
    mlookup: pd.DataFrame
    n_models: int
    is_human: np.ndarray
    is_sota: np.ndarray
    is_low_obs: np.ndarray
    n_obs_per_model: np.ndarray


def load_matched(trace_path: Path, scope: str, allow_stale: bool):
    """Trace + rebuilt data on one common model list; theta0 is (S, n_common). A trace/data
    mismatch raises unless --allow-stale (name-intersection join)."""
    trace = load_trace(trace_path)
    drop = None if scope == "all" else access_scope_drop_list(False, scope)
    data = load_eci_data(drop_benchmarks=drop)
    theta_all = capability_draws(trace)                                  # (S, n_trace)
    trace_names = trace.posterior["theta"].coords["model"].values.tolist()
    data_names = data.mlookup.sort_values("model_idx")["model"].tolist()
    if trace_names != data_names:
        msg = (f"trace has {len(trace_names)} models, the rebuilt {scope} scope has "
               f"{len(data_names)}: they don't match (the trace predates the current data "
               "generation; re-fit before quoting numbers)")
        if not allow_stale:
            raise AssertionError(msg + ". Pass --allow-stale for a quick look that joins by "
                                 "model-name intersection.")
        print(f"WARNING: {msg}. --allow-stale: joining by name; every number below is a "
              "STALE-TRACE number, not for quoting.")
    trace_pos = {m: i for i, m in enumerate(trace_names)}
    data_pos = {m: i for i, m in enumerate(data_names)}
    common = [m for m in trace_names if m in data_pos]
    theta0 = theta_all[:, [trace_pos[m] for m in common]]
    rows = [data_pos[m] for m in common]
    mini = _MiniData(
        mlookup=pd.DataFrame({"model": common, "model_idx": np.arange(1, len(common) + 1)}),
        n_models=len(common), is_human=data.is_human[rows],
        is_sota=(data.is_sota[rows] if data.is_sota is not None else np.zeros(len(common), bool)),
        is_low_obs=data.is_low_obs[rows], n_obs_per_model=data.n_obs_per_model[rows])
    return mini, theta0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--access", default="all", choices=list(SCOPES),
                    help="benchmark access scope: the fit folder to read (default all = canonical/)")
    ap.add_argument("--group", default="openness", choices=list(GROUP_TITLES),
                    help="the two groups compared (default openness: closed vs open weights)")
    ap.add_argument("--results-dir", default=None, help="folder holding trace.nc (overrides --access's)")
    ap.add_argument("--allow-stale", action="store_true",
                    help="join trace and data by model name instead of refusing a mismatch")
    ap.add_argument("--fit-start", default=config.FORECAST_KW["fit_start"],
                    help="trend lines use frontier points released on or after this date "
                         f"(default {config.FORECAST_KW['fit_start']})")
    ap.add_argument("--top-k", type=int, default=config.FORECAST_KW["top_k"],
                    help="a release joins the trend fit when it is in the running top-k of "
                         f"medians at its date (default {config.FORECAST_KW['top_k']})")
    ap.add_argument("--min-obs", type=int, default=2,
                    help="a candidate needs at least this many scores in the scope (default 2: "
                         "the K=1 reading of the K-axis coverage rule, one benchmark alone is "
                         "not a full unit of evidence)")
    ap.add_argument("--today", default=None, help="pin the 'today' of the gap and the figures")
    args = ap.parse_args()

    folder, caption = SCOPES[args.access]
    results_dir = Path(args.results_dir) if args.results_dir else config.RESULTS_DIR / folder
    trace_path = results_dir / "trace.nc"
    print(f"Loading {trace_path} ...", flush=True)
    mini, theta0 = load_matched(trace_path, args.access, args.allow_stale)
    print(f"  {mini.n_models} models ({int(mini.is_human.sum())} human tiers, "
          f"{int(mini.is_sota.sum())} SOTA members), {theta0.shape[0]} draws")

    transform = eci_transform(theta0, mini)
    E = transform.a[:, None] + transform.b[:, None] * theta0                 # (S, n) ECI-H per draw
    model_dates, _ = _release_dates(pd.read_csv(PROCESSED_FILE))
    # The repository's candidate rule (dated, not a human tier, evaluated on the axis), plus the
    # K=1 reading of "one full unit of evidence": at least --min-obs scores in the scope.
    keep = candidate_mask(theta0[:, :, None], 0, mini, model_dates)
    keep &= mini.n_obs_per_model >= args.min_obs
    print(f"  candidates: {int(keep.sum())} (dated, non-human, at least {args.min_obs} scores)")
    res = scope_gap(E, mini, model_dates, keep, scope=args.access, kind=args.group,
                    fit_start=args.fit_start, top_k=args.top_k, today=args.today)

    paths = save_scope(res, config.COMPARISONS_DIR)
    for p in paths:
        print(f"wrote {p}")

    titles = GROUP_TITLES[args.group]
    figures_dir = config.COMPARISONS_DIR / config.FIGURES_DIRNAME
    stem = f"frontier_gap_{args.group}_{args.access}"
    fig = frontier_panels_fig({args.access: res}, [args.access], today=args.today,
                              group_titles=titles,
                              title=f"{titles[res.leader.label]} vs {titles[res.follower.label]} "
                                    f"(ECI-H): {caption}")
    save_fig(fig, stem, figures_dir)
    fig = lag_fig({args.access: res}, [args.access], today=args.today, label_scope=args.access,
                  follower_title=titles[res.follower.label].lower(),
                  leader_title=titles[res.leader.label].lower(),
                  title=f"How far behind: {SCOPE_TITLES[args.access].lower()}")
    save_fig(fig, f"{stem}_lag", figures_dir)
    print(f"wrote {figures_dir / stem}.png (+ _lag, html/)")

    print(f"\n── {caption}: {titles[res.leader.label]} vs {titles[res.follower.label]} ──")
    for g in (res.leader, res.follower):
        print(f"  {g.label}: {len(g.tl)} candidates, {len(g.records)} records, "
              f"{int(g.tl['in_fit'].sum())} points in the trend fit"
              + (f", slope {np.median(g.fit.b):+.1f} ECI-H/yr" if g.fit is not None else ", no line"))
    pd.set_option("display.width", 200)
    print(res.summary.round(2).to_string(index=False))
    print("\n  envelope lag per record (months behind, median [50%]):")
    for _, r in res.lag_df.iterrows():
        print(f"    {r['release_date']:%Y-%m-%d}  {r['name']:<40s} ECI-H {r['level_eci_median']:6.1f}  "
              f"{r['lag_months_median']:5.1f} [{r['lag_hdi50_low']:5.1f}, {r['lag_hdi50_high']:5.1f}]"
              + (f"  undefined in {r['frac_undefined']:.0%} of draws" if r['frac_undefined'] > 0.05 else ""))


if __name__ == "__main__":
    main()
