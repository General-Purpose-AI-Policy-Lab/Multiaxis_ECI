"""Frontier-trend forecast, three stacked axes, in the post's Plotly style.

The figure is `viz.frontier_trend_fig` at the `POST` scale, the same builder the
dashboard and the per-fit figure folders use (dated models, frontier trend with
its 80% band, human tiers in Blues named in the right margin, today line, no
legend); this script only owns the flagship trace, the forecast cache and the
fixed 2023-2030 window.

Reads the flagship trace over ALL chains: the post's figures are
whole-posterior, never mode-restricted. The forecast itself is
`make_all.compute` (FORECAST_KW, envelope basis, per-axis SOTA exemption), and
its result is cached in `lw_forecast_cache_80.pkl` BESIDE the trace it came
from — keyed by folder, so pointing --trace elsewhere can never reuse another
fit's forecast. Axis identity is checked against `analysis.check_axis_identity` (config.AXIS_SIGNATURES)
before the cache is written, so a reused cache is a checked one.

Usage:
    python 6_writeups/blogpost/figures/make_trend_plotly.py [--trace FILE] [--tag ""]
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "2_model"))   # the pickle holds analysis.ForecastResult
sys.path.insert(0, str(HERE))
from multiaxis_eci.analysis import (  # noqa: E402
    FLAGSHIP,
    FLAGSHIP_THIN,
    prepare_fit,
    require_axis_titles,
)
from multiaxis_eci.analysis import FLAGSHIP_TRACE as TRACE
from multiaxis_eci.data import PROCESSED_FILE  # noqa: E402
from multiaxis_eci.viz import POST, frontier_trend_fig  # noqa: E402
from multiaxis_eci.viz.core import save_html, save_print  # noqa: E402

# None drops the in-figure title: the post's caption carries the
# description. Set a string to draw it on the canvas again, e.g.
# "Frontier trend per axis, against the human tiers".
TITLE = None

# Explicit literal, NOT list(TITLES): axis 4 (Legacy QA) is deliberately
# excluded from every forecast figure, its benchmarks carry no recent
# measurements. TITLES has 4 keys; only 3 are looked up here.
AXES = ["axis1", "axis2", "axis3"]
X0, X1 = "2023-01-01", "2030-01-01"        # same window on every row


def forecast(trace: Path, cached: bool = False,
             chains: list[int] | None = None) -> dict:
    """Per axis {"fc", "tl", "hs"}, from the pickle beside `trace` or rebuilt.

    A rebuild opens the trace over all chains at the flagship thinning, runs
    the axis-identity check (SystemExit before any mislabeled axis), and hands
    the view to `make_all.compute`, which owns the forecast settings.

    A chain SUBSET (`chains`) is never cached — the pickle beside the trace is
    the whole fit's — and its axes are permuted back onto the fit-level
    display frame first, exactly as in `make_crossover_plotly`.
    """
    if chains is not None:
        from make_all import compute

        from multiaxis_eci.analysis import align_to_reference_loadings

        idata = FLAGSHIP.open_posterior(keep=["A", "theta", "tau_A"],
                                        thin=FLAGSHIP_THIN, chains=chains,
                                        path=trace)
        data, *_ = FLAGSHIP.load_data(idata)
        view = prepare_fit(idata, data)
        view = align_to_reference_loadings(view, data, trace.parent / "mirt_loadings.csv")
        require_axis_titles(trace.parent, view, data)
        return compute(view, data, pd.read_csv(PROCESSED_FILE))
    cache = trace.parent / "lw_forecast_cache_80.pkl"
    if cache.exists():
        # A cache written before the package rename pickled the result as
        # `analysis.forecast.ForecastResult`; alias the old module path so
        # those caches keep loading on the multiaxis_eci layout.
        import multiaxis_eci.analysis as _a
        import multiaxis_eci.analysis.forecast as _af
        sys.modules.setdefault("analysis", _a)
        sys.modules.setdefault("analysis.forecast", _af)
        print(f"  reused {cache}")
        return pickle.loads(cache.read_bytes())
    if cached:
        raise SystemExit(f"--cached but {cache} is missing — run without "
                         "--cached once to rebuild it from the trace.")
    from make_all import compute

    idata = FLAGSHIP.open_posterior(keep=["A", "theta", "tau_A"],
                                    thin=FLAGSHIP_THIN, chains=None, path=trace)
    data, *_ = FLAGSHIP.load_data(idata)
    view = prepare_fit(idata, data)
    require_axis_titles(trace.parent, view, data)
    per_axis = compute(view, data, pd.read_csv(PROCESSED_FILE))
    cache.write_bytes(pickle.dumps(per_axis))
    print(f"  wrote {cache}")
    return per_axis


def main(trace: Path = TRACE, tag: str = "", out_dir: Path = HERE,
         cached: bool = False, chains: list[int] | None = None) -> None:
    out = out_dir / f"forecast_trend_plotly{tag}"
    per_axis = forecast(trace, cached=cached, chains=chains)
    titles = require_axis_titles(trace.parent)      # confirmed names, checked at cache time
    fig = frontier_trend_fig(per_axis, AXES, titles, style=POST, window=(X0, X1), title=TITLE)
    for name in AXES:
        d = per_axis[name]
        print(f"  {name}: {len(d['tl'])} models, slope median "
              f"{float(np.median(d['fc'].slope)):+.2f}/yr")
    out.parent.mkdir(parents=True, exist_ok=True)
    save_html(fig, out)
    save_print(fig, out)
    print(f"  wrote {out.with_suffix('.png')} and html/{out.stem}.html")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trace", type=Path, default=TRACE)
    p.add_argument("--tag", default="")
    p.add_argument("--cached", action="store_true",
                   help="never open the trace; fail if the cache is missing")
    p.add_argument("--chains", default=None,
                   help="comma-separated chain subset (e.g. 0,1,3,8); "
                        "default: all chains, through the cache")
    args = p.parse_args()
    main(args.trace, args.tag, cached=args.cached,
         chains=None if args.chains is None
         else [int(c) for c in args.chains.split(",")])
