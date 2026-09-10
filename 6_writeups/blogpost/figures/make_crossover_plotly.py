"""Crossover dates per axis, three stacked panels, in the post's Plotly style.

The figure is `viz.crossover_panels_fig` at the `POST` scale, the builder the
dashboard and the per-fit figure folders use. Per human tier: the median
crossing date (dot), a THICK 50% bar and a THIN 80% bar, each bar split at the
today line — the share already behind us is green, the share still ahead is red. Crossings are solved per draw by
`analysis.mirt_crossover_df` on the record ENVELOPE (observed step date inside
the window, raw early-rate backcast below it, forward line above it), so both
interval widths come from the same per-draw distribution and both are HDIs,
the summary the whole forecast pipeline uses.

The envelope draws come from `make_trend_plotly.forecast` — the forecast
cache pickle BESIDE the trace (FORECAST_KW: envelope basis, per-axis SOTA
exemption), keyed by folder so another fit's forecast can never be reused. The human theta draws need the trace itself (all chains,
flagship thinning, axis identity checked), so the assembled table is cached
as `lw_crossover_50_80.csv` beside the trace and later runs read the CSV only.

The window is fixed at 2015-2030 so every variant of the figure is
comparable. Whatever runs past it (a whisker's tail, or a median outside the
window) is clipped at the edge and marked with a small dot plus the true date
in small type.

Usage:
    python 6_writeups/blogpost/figures/make_crossover_plotly.py [--trace FILE] [--tag ""]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "2_model"))
sys.path.insert(0, str(HERE))
from multiaxis_eci.analysis import (  # noqa: E402
    FLAGSHIP,
    FLAGSHIP_THIN,
    align_to_reference_loadings,
    prepare_fit,
    require_axis_titles,
)
from multiaxis_eci.analysis import FLAGSHIP_TRACE as TRACE
from multiaxis_eci.viz import POST, crossover_panels_fig  # noqa: E402
from multiaxis_eci.viz.core import save_html, save_print  # noqa: E402
from multiaxis_eci.viz.forecast import CROSSOVER_WINDOW  # noqa: E402

# None drops the in-figure title: the post's caption carries the
# description. Set a string to draw it on the canvas again.
TITLE = None

# Explicit literal, NOT list(TITLES): axis 4 (Legacy QA) is out of every
# forecast figure — its benchmarks carry no recent measurements, so a trend
# there is not meaningful. TITLES has 4 keys; only 3 are looked up here.
AXES = ["axis1", "axis2", "axis3"]
# Every panel backcasts (dates before an axis's first measured model are raw
# backward extrapolations of its early record trend); the post's caption
# carries that caveat, the panel titles do not.
X0, X1 = CROSSOVER_WINDOW            # the library's shared crossover window
# Interval masses, widest last. One value draws a single thick bar.
PROBS = (0.5, 0.8)

_DATE_COLS = ["crossover_date_median", "crossover_hdi_low", "crossover_hdi_high",
              "hdi80_low", "hdi80_high"]


def crossovers(trace: Path, cached: bool = False,
               chains: list[int] | None = None,
               probs: tuple = PROBS) -> pd.DataFrame:
    """Per (axis, tier): median crossing + 50% and 80% HDIs, one row each.

    Reads `lw_crossover_50_80.csv` beside `trace` when it exists; a rebuild
    takes the slope/intercept draws from the forecast cache and the human
    theta draws from the trace (all chains, flagship thinning, axis identity
    checked before any label is trusted).
    """
    group = "" if chains is None else "_c" + "".join(map(str, chains))
    mass = "_".join(f"{int(p * 100)}" for p in probs)
    cache = trace.parent / f"lw_crossover_{mass}{group}.csv"
    if cache.exists():
        print(f"  reused {cache}")
        return pd.read_csv(cache, parse_dates=_DATE_COLS)
    if cached:
        raise SystemExit(f"--cached but {cache} is missing — run without "
                         "--cached once to rebuild it from the trace.")
    from make_all import END
    from make_trend_plotly import forecast

    from multiaxis_eci.analysis import axis_forecast_inputs, crossover_table
    from multiaxis_eci.data import PROCESSED_FILE

    idata = FLAGSHIP.open_posterior(keep=["A", "theta", "tau_A"],
                                    thin=FLAGSHIP_THIN, chains=chains, path=trace)
    data, *_ = FLAGSHIP.load_data(idata)
    view = prepare_fit(idata, data)
    if chains is not None:
        view = align_to_reference_loadings(view, data, trace.parent / "mirt_loadings.csv")
    require_axis_titles(trace.parent, view, data)
    # The cached pickle holds the whole fit's slope/intercept draws, so a chain
    # subset has to run its own forecast; the settings are the same ones.
    if chains is None:
        per_axis = forecast(trace)
    else:
        raw = pd.read_csv(PROCESSED_FILE)
        per_axis = {n: axis_forecast_inputs(view.theta, view.names.index(n), data, raw, n,
                                             A_draws=view.A, horizon_date=END)
                    for n in AXES}

    parts = [crossover_table(per_axis[name]["fc"], view.theta, view.names.index(name), data,
                             name, probs=probs) for name in AXES]
    if len(probs) == 1:
        for cx in parts:
            cx["hdi80_low"] = cx["crossover_hdi_low"]
            cx["hdi80_high"] = cx["crossover_hdi_high"]
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(cache, index=False)
    print(f"  wrote {cache}")
    return out


def main(trace: Path = TRACE, tag: str = "", out_dir: Path = HERE,
         cached: bool = False, chains: list[int] | None = None,
         probs: tuple = PROBS) -> None:
    out = out_dir / f"forecast_crossover_plotly{tag}"
    cx = crossovers(trace, cached=cached, chains=chains, probs=probs)
    cx = cx[cx["axis"].isin(AXES)]
    dropped = cx["crossover_date_median"].isna()
    for _, r in cx[dropped].iterrows():
        print(f"  no crossing: {r['axis']} / {r['tier']} "
              f"(frac_positive_slope {r['frac_positive_slope']:.2f})")
    cx = cx[~dropped]
    for _, r in cx.iterrows():
        print(f"    {r['axis']} / {r['tier']}: median {r['crossover_date_median'].date()}, "
              f"{probs[0]:.0%} [{r['crossover_hdi_low'].date()}, "
              f"{r['crossover_hdi_high'].date()}]"
              + (f", {probs[1]:.0%} [{r['hdi80_low'].date()}, {r['hdi80_high'].date()}]"
                 if len(probs) > 1 else ""))
    # Fixed 2015-2030 window, shared by every variant of the figure so they are
    # all directly comparable; the builder clips and dates whatever runs past it.
    titles = require_axis_titles(trace.parent)      # confirmed names, checked at cache time
    fig = crossover_panels_fig(cx, AXES, titles, probs=probs, style=POST,
                               window=(X0, X1), title=TITLE)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_html(fig, out)
    save_print(fig, out)
    print(f"  wrote {out.with_suffix('.png')} and html/{out.stem}.html")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trace", type=Path, default=TRACE)
    p.add_argument("--tag", default="")
    p.add_argument("--chains", type=lambda s: [int(c) for c in s.split(",")],
                   help="restrict to these chains (default: the whole fit)")
    p.add_argument("--probs", default=",".join(str(x) for x in PROBS),
                   type=lambda v: tuple(float(x) for x in v.split(",")),
                   help="interval mass, or two masses widest last")
    p.add_argument("--cached", action="store_true",
                   help="never open the trace; fail if the CSV cache is missing")
    args = p.parse_args()
    main(args.trace, args.tag, cached=args.cached, chains=args.chains,
         probs=args.probs)
