"""Frontier-trend forecast, three stacked axes, in the post's Plotly style.

Same content as the matplotlib panel in `make_forecast_figs.trend_fig` (dated
models, frontier trend with its 50% band, human tiers, today line), drawn with
the palette and type sizes of `make_timeline_plotly.py`: teal models, orange
trend, human tiers in Blues with their names in the right margin, no legend.

Reads the flagship trace over ALL chains: the post's figures are
whole-posterior, never mode-restricted. The forecast itself is
`make_all.compute` (FORECAST_KW, records basis, per-axis SOTA exemption), and
its result is cached in `lw_forecast_cache_50.pkl` BESIDE the trace it came
from — keyed by folder, so pointing --trace elsewhere can never reuse another
fit's forecast. Axis identity is checked against `make_all.EXPECTED_TOPS`
before the cache is written, so a reused cache is a checked one.

Usage:
    python lw_post/figures/make_trend_plotly.py [--trace FILE] [--tag ""]
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors as pc
from plotly.subplots import make_subplots
import plotly.graph_objects as go

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))   # the pickle holds analysis.ForecastResult
sys.path.insert(0, str(HERE))
from analysis import (FLAGSHIP, FLAGSHIP_THIN,  # noqa: E402
                      FLAGSHIP_TRACE as TRACE, prepare_fit)
from config import AXIS_TITLES as TITLES  # noqa: E402
from data import PROCESSED_FILE  # noqa: E402
from viz.core import save_print  # noqa: E402


# None drops the in-figure title: the LessWrong caption carries the
# description. Set a string to draw it on the canvas again, e.g.
# "Frontier trend per axis, against the human tiers".
TITLE = None

# Explicit literal, NOT list(TITLES): axis 4 (Legacy QA) is deliberately
# excluded from every forecast figure, its benchmarks carry no recent
# measurements. TITLES has 4 keys; only 3 are looked up here.
AXES = ["axis1", "axis2", "axis3"]
X0, X1 = "2023-01-01", "2030-01-01"        # same window on every row
AI = "#20a39e"                              # timeline figure's model teal
TREND = "#ff9500"

# Post-scale sizing, shared with the timeline / forests / loadings figures:
# the figure is shared flat, so type must read without zooming. The taller
# canvas gives each row the headroom the axis-3 tier-label stack needs at this
# type size; save_print's scale=2 doubles the pixels.
FONT_TITLE = 42       # figure title (drawn only when TITLE is set)
FONT_AXIS = 36        # panel titles
FONT_TICK = 30        # tick labels and both axis captions
FONT_TIER = 26        # right-margin human tier names
MARKER = 13           # model points, matching the timeline figure
ERRBAR_W = 2.0        # model error-bar line width
TREND_W = 5           # trend median dash
REFLINE_W = 2.2       # tier hlines and the today vline
TIER_GAP = 0.066      # min label spacing, fraction of the row's y span
WIDTH, HEIGHT = 2200, 2500


def forecast(trace: Path, cached: bool = False) -> dict:
    """Per axis {"fc", "tl", "hs"}, from the pickle beside `trace` or rebuilt.

    A rebuild opens the trace over all chains at the flagship thinning, runs
    the axis-identity check (SystemExit before any mislabeled axis), and hands
    the view to `make_all.compute`, which owns the forecast settings.
    """
    cache = trace.parent / "lw_forecast_cache_50.pkl"
    if cache.exists():
        print(f"  reused {cache}")
        return pickle.loads(cache.read_bytes())
    if cached:
        raise SystemExit(f"--cached but {cache} is missing — run without "
                         "--cached once to rebuild it from the trace.")
    from make_all import check_axis_identity, compute

    idata = FLAGSHIP.open_posterior(keep=["A", "theta", "tau_A"],
                                    thin=FLAGSHIP_THIN, chains=None, path=trace)
    data, *_ = FLAGSHIP.load_data(idata)
    view = prepare_fit(idata, data)
    check_axis_identity(view, data)
    per_axis = compute(view, data, pd.read_csv(PROCESSED_FILE))
    cache.write_bytes(pickle.dumps(per_axis))
    print(f"  wrote {cache}")
    return per_axis


def _tier_labels(fig, hs: pd.DataFrame, row: int, yref: str, ylim: tuple[float, float]):
    """Tier lines in Blues (strongest = highest ability) plus right-margin names.

    Labels are nudged apart top-down against this row's own y range, so a label
    can sit slightly off its line where tiers crowd.
    """
    rows = hs.sort_values("mean", ascending=False).reset_index(drop=True)
    fracs = np.linspace(0.92, 0.35, len(rows)) if len(rows) > 1 else [0.7]
    colors = [pc.sample_colorscale("Blues", float(f))[0] for f in fracs]

    span = ylim[1] - ylim[0]
    gap = TIER_GAP * span
    ys, prev = [], np.inf
    for lvl in rows["mean"]:
        y = min(float(lvl), prev - gap)
        ys.append(y)
        prev = y
    # The stack is usually taller than the tier block itself (nine names at this
    # type size), so center it on the block instead of hanging it all below the
    # top tier, then keep it inside the row.
    shift = float(np.mean(rows["mean"])) - float(np.mean(ys))
    shift = min(shift, (ylim[1] - 0.02 * span) - ys[0])
    shift = max(shift, (ylim[0] + 0.02 * span) - ys[-1])
    ys = [y + shift for y in ys]

    for (_, r), y, col in zip(rows.iterrows(), ys, colors):
        fig.add_hline(y=float(r["mean"]), row=row, col=1, line=dict(
            color=col, width=REFLINE_W, dash="dash"), opacity=0.75)
        fig.add_annotation(x=1.005, y=y, xref="paper", yref=yref,
                           text=r["name"], showarrow=False, xanchor="left",
                           font=dict(size=FONT_TIER, color=col))


def main(trace: Path = TRACE, tag: str = "", out_dir: Path = HERE,
         cached: bool = False) -> None:
    out = out_dir / f"forecast_trend_plotly{tag}"
    per_axis = forecast(trace, cached=cached)
    today = pd.Timestamp.today().normalize().strftime("%Y-%m-%d")

    fig = make_subplots(rows=len(AXES), cols=1, shared_xaxes=True,
                        vertical_spacing=0.085,
                        subplot_titles=[TITLES[a] for a in AXES])

    for i, name in enumerate(AXES, start=1):
        d = per_axis[name]
        fc, tl, hs = d["fc"], d["tl"], d["hs"]
        gx = pd.to_datetime(fc.grid_dates)

        # 50% band: lower edge, then upper edge filled down to it.
        fig.add_trace(go.Scatter(x=gx, y=fc.lo, mode="lines", showlegend=False,
                                 line=dict(width=0), hoverinfo="skip"),
                      row=i, col=1)
        fig.add_trace(go.Scatter(x=gx, y=fc.hi, mode="lines", fill="tonexty",
                                 fillcolor="rgba(255,149,0,0.18)", showlegend=False,
                                 line=dict(width=0), hoverinfo="skip"),
                      row=i, col=1)

        # Dated models with their 50% intervals.
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(tl["release_date"]), y=tl["mean"], mode="markers",
            marker=dict(color=AI, size=MARKER, opacity=0.55,
                        line=dict(width=0)),
            error_y=dict(type="data", symmetric=False,
                         array=tl["hdi_high"] - tl["mean"],
                         arrayminus=tl["mean"] - tl["hdi_low"],
                         thickness=ERRBAR_W, width=0,
                         color="rgba(32,163,158,0.35)"),
            text=tl["name"], showlegend=False,
            hovertemplate="%{text}<br>%{x|%Y-%m-%d}: %{y:.2f}<extra></extra>"),
            row=i, col=1)

        # Trend median.
        fig.add_trace(go.Scatter(x=gx, y=fc.median, mode="lines", showlegend=False,
                                 line=dict(color=TREND, width=TREND_W, dash="dash"),
                                 hovertemplate="%{x|%Y-%m}: %{y:.2f}<extra></extra>"),
                      row=i, col=1)

        lo = min(float(tl["hdi_low"].min()), float(np.min(fc.lo)),
                 float(hs["mean"].min()))
        hi = max(float(tl["hdi_high"].max()), float(np.max(fc.hi)),
                 float(hs["mean"].max()))
        pad = 0.06 * (hi - lo)
        ylim = (lo - pad, hi + pad)

        fig.add_vline(x=today, row=i, col=1,
                      line=dict(color="#444", width=REFLINE_W, dash="dot"))
        yref = "y" if i == 1 else f"y{i}"
        _tier_labels(fig, hs, i, yref, ylim)

        fig.update_yaxes(title=dict(text="ability", font=dict(size=FONT_TICK)),
                         tickfont=dict(size=FONT_TICK), range=list(ylim),
                         gridcolor="#eeeeee", zeroline=False, row=i, col=1)
        fig.update_xaxes(range=[X0, X1], dtick="M12", tickformat="%Y",
                         tickfont=dict(size=FONT_TICK), gridcolor="#f4f4f4",
                         showticklabels=True, row=i, col=1)
        print(f"  {name}: slope median {np.median(fc.slope):+.3f}/yr, "
              f"trend at 2030 {fc.median[-1]:.2f} "
              f"[{fc.lo[-1]:.2f}, {fc.hi[-1]:.2f}] (50% HDI), "
              f"{len(tl)} dated models, earliest "
              f"{pd.to_datetime(tl['release_date']).min().date()}, y {ylim[0]:.2f}"
              f"..{ylim[1]:.2f}")

    fig.update_xaxes(title=dict(text="Release date", font=dict(size=FONT_TICK)),
                     row=len(AXES), col=1)
    for ann in fig.layout.annotations[:len(AXES)]:      # subplot titles
        ann.font = dict(size=FONT_AXIS)
    fig.update_layout(
        showlegend=False, template="plotly_white",
        height=HEIGHT, width=WIDTH, margin=dict(l=160, r=560, t=100, b=120))
    if TITLE is not None:
        fig.update_layout(title=dict(text=TITLE, x=0.5,
                                     font=dict(size=FONT_TITLE)),
                          margin_t=170)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out.with_suffix(".html"))
    save_print(fig, out)
    print(f"  wrote {out.with_suffix('.png')} and {out.with_suffix('.html')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trace", type=Path, default=TRACE)
    p.add_argument("--tag", default="")
    p.add_argument("--cached", action="store_true",
                   help="never open the trace; fail if the cache is missing")
    args = p.parse_args()
    main(args.trace, args.tag, cached=args.cached)
