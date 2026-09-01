"""Crossover dates per axis, three stacked panels, in the post's Plotly style.

Per human tier: the median crossing date (dot), a THICK 50% bar and a THIN 80%
bar, each bar split at the today line — the share already behind us is green,
the share still ahead is red. Crossings are solved per draw
(t* = (θ_h − intercept)/slope) by `analysis.mirt_crossover_df`, so both
interval widths come from the same per-draw distribution and both are HDIs,
the summary the whole forecast pipeline uses.

The slope/intercept draws come from `make_trend_plotly.forecast` — the
forecast cache pickle BESIDE the trace (records basis, FORECAST_KW, per-axis
SOTA exemption), keyed by folder so another fit's forecast can never be
reused. The human theta draws need the trace itself (all chains, flagship
thinning, axis identity checked), so the assembled table is cached as
`lw_crossover_50_80.csv` beside the trace and later runs read the CSV only.

The window is fixed at 2015-2030 so every variant of the figure is
comparable. Whatever runs past it (a whisker's tail, or a median outside the
window) is clipped at the edge and marked with a small dot plus the true date
in small type.

Usage:
    python blogpost/figures/make_crossover_plotly.py [--trace FILE] [--tag ""]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from multiaxis_eci.analysis import (FLAGSHIP, FLAGSHIP_THIN,  # noqa: E402
                      FLAGSHIP_TRACE as TRACE, prepare_fit)
from multiaxis_eci.config import AXIS_TITLES as TITLES  # noqa: E402
from multiaxis_eci.viz.core import FUTURE_COLOR as FUTURE, PASSED_COLOR as PAST, save_html, save_print  # noqa: E402


# None drops the in-figure title: the post's caption carries the
# description. Set a string to draw it on the canvas again.
TITLE = None

# Explicit literal, NOT list(TITLES): axis 4 (Legacy QA) is out of every
# forecast figure — its benchmarks carry no recent measurements, so a trend
# there is not meaningful. TITLES has 4 keys; only 3 are looked up here.
AXES = ["axis1", "axis2", "axis3"]
# Every panel backcasts (FORECAST_BACKCAST_FLOOR covers all three axes):
# dates before an axis's first measured model are backward extrapolations
# of its early record trend, floored at the figure's window start.
_NOTE = "(pre-data dates: backward extrapolation of the early record trend)"
BACKCAST = {"axis1": _NOTE, "axis2": _NOTE, "axis3": _NOTE}

TODAY_COLOR = "#444"

# Post-scale sizing, shared with the trend / forests / loadings figures.
FONT_TITLE = 42       # figure title (drawn only when TITLE is set)
FONT_AXIS = 36        # panel titles
FONT_NOTE = 26        # the backcast parenthetical inside a panel title
FONT_TICK = 30        # tick labels (tier rows, years) and the x caption
FONT_LEGEND = 30
MARKER = 17           # median dot
BAR50_W = 11          # the wider interval, drawn thick
BAR80_W = 4           # the narrower interval, drawn thin
# Interval masses, widest last. One value draws a single bar at BAR50_W.
PROBS = (0.5, 0.8)
WIDTH, HEIGHT = 2200, 2300

_DATE_COLS = ["crossover_date_median", "crossover_hdi_low", "crossover_hdi_high",
              "hdi80_low", "hdi80_high"]


def _display_frame(view, data, trace: Path):
    """Permute a chain subset's axes onto the whole fit's display frame.

    `prepare_fit` ranks the axes by loading energy WITHIN whatever draws it is
    given, and a chain group ranks them in its own order — on this fit the
    minority group puts Legacy QA above Agentic. Matching each column to the
    fit-level `mirt_loadings.csv` medians by correlation puts panel k back on
    the axis panel k carries everywhere else.
    """
    import dataclasses

    pooled = (pd.read_csv(trace.parent / "mirt_loadings.csv")
              .pivot(index="benchmark", columns="axis", values="loading_median"))
    bench = list(data.blookup.sort_values("benchmark_idx")["benchmark"])
    P = pooled.loc[bench].values
    M = np.median(view.require_A(), axis=0)
    corr = np.corrcoef(P.T, M.T)[:P.shape[1], P.shape[1]:]
    perm = corr.argmax(axis=1)
    if sorted(perm) != list(range(P.shape[1])):
        raise SystemExit(f"axis match is not a permutation: {perm}\n{corr.round(3)}")
    print("  display axis -> subset column: "
          + ", ".join(f"{k+1}->{s+1} (r {corr[k, s]:+.2f})"
                      for k, s in enumerate(perm)))
    return dataclasses.replace(view, theta=view.theta[:, :, perm],
                               A=view.A[:, :, perm])


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
    from multiaxis_eci.analysis import mirt_crossover_df, mirt_frontier_forecast
    from multiaxis_eci.config import (FORECAST_BACKCAST_FLOOR, FORECAST_KW,
                                      FORECAST_NO_SOTA_AXES)
    from multiaxis_eci.data import PROCESSED_FILE
    from make_all import END, check_axis_identity
    from make_trend_plotly import forecast

    idata = FLAGSHIP.open_posterior(keep=["A", "theta", "tau_A"],
                                    thin=FLAGSHIP_THIN, chains=chains, path=trace)
    data, *_ = FLAGSHIP.load_data(idata)
    view = prepare_fit(idata, data)
    if chains is not None:
        view = _display_frame(view, data, trace)
    check_axis_identity(view, data)
    # The cached pickle holds the whole fit's slope/intercept draws, so a chain
    # subset has to run its own forecast; the settings are the same ones.
    if chains is None:
        per_axis = forecast(trace)
    else:
        raw = pd.read_csv(PROCESSED_FILE)
        per_axis = {
            n: {"fc": mirt_frontier_forecast(
                    view.theta, view.names.index(n), data, raw,
                    **dict(FORECAST_KW, horizon_date=END,
                           sota_exempt=view.names.index(n)
                           not in FORECAST_NO_SOTA_AXES,
                           backcast_floor=FORECAST_BACKCAST_FLOOR.get(n)))}
            for n in AXES}

    parts = []
    for name in AXES:
        k = view.names.index(name)
        fc = per_axis[name]["fc"]
        # Same draws feed both widths; only the HDI mass differs.
        cx = mirt_crossover_df(fc, view.theta, k, data, axis_name=name,
                               hdi_prob=probs[0])
        if len(probs) > 1:
            wide = mirt_crossover_df(fc, view.theta, k, data, axis_name=name,
                                     hdi_prob=probs[1])[
                ["axis", "tier", "crossover_hdi_low", "crossover_hdi_high"]
            ].rename(columns={"crossover_hdi_low": "hdi80_low",
                              "crossover_hdi_high": "hdi80_high"})
            cx = cx.merge(wide, on=["axis", "tier"])
        else:
            cx["hdi80_low"] = cx["crossover_hdi_low"]
            cx["hdi80_high"] = cx["crossover_hdi_high"]
        parts.append(cx)
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
    if dropped.any():
        for _, r in cx[dropped].iterrows():
            print(f"  no crossing: {r['axis']} / {r['tier']} "
                  f"(frac_positive_slope {r['frac_positive_slope']:.2f})")
        cx = cx[~dropped]
    today = pd.Timestamp.today().normalize()

    # Fixed 2015-2030 window, shared by every variant of the figure so they
    # are all directly comparable. Whatever runs past it is clipped at the
    # edge and marked with a small dot plus the true date in small type. The
    # pad is canvas only (room for the edge dots and their labels); the ticks
    # stay on the years.
    x0d, x1d = pd.Timestamp("2015-01-01"), pd.Timestamp("2030-01-01")
    pad = pd.Timedelta(days=160)
    x0, x1 = x0d - pad, x1d + pad

    titles = [TITLES[a] + (f'  <span style="font-size:{FONT_NOTE}px">'
                           f'{BACKCAST[a]}</span>' if a in BACKCAST else "")
              for a in AXES]
    fig = make_subplots(rows=len(AXES), cols=1, shared_xaxes=True,
                        vertical_spacing=0.075, subplot_titles=titles)

    clipped = []
    for i, name in enumerate(AXES, start=1):
        rows = cx[cx.axis == name].sort_values("human_mean")   # lowest tier at bottom
        tiers = rows["tier"].tolist()
        print(f"  {TITLES[name]}")

        # Every bar is split at the today line: its past share is green, its
        # future share red — the color reads per segment, not per row. Batches:
        # thin 80% first, thick 50% on top, median dot last.
        bars = (("hdi80_low", "hdi80_high", BAR80_W),
                ("crossover_hdi_low", "crossover_hdi_high", BAR50_W))
        segs = {(w, c): ([], []) for _, _, w in bars for c in (PAST, FUTURE)}
        for lo_c, hi_c, w in (bars[1:] if len(probs) == 1 else bars):
            for _, r in rows.iterrows():
                start, end = max(r[lo_c], x0d), min(r[hi_c], x1d)
                if start > end:                      # bar entirely off-window
                    continue
                pieces = []
                if start < today:
                    pieces.append((start, min(end, today), PAST))
                if end > today:
                    pieces.append((max(start, today), end, FUTURE))
                for s, e, c in pieces:
                    sx, sy = segs[(w, c)]
                    sx += [s.isoformat(), e.isoformat(), None]
                    sy += [r["tier"], r["tier"], None]
        for (w, c), (sx, sy) in segs.items():
            if sx:
                fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines",
                                         line=dict(color=c, width=w),
                                         opacity=0.85, showlegend=False,
                                         hoverinfo="skip"), row=i, col=1)

        # Median dots, colored by their own side of today; a median outside
        # the window is left to the edge markers below.
        m_in = rows[(rows["crossover_date_median"] >= x0d)
                    & (rows["crossover_date_median"] <= x1d)]
        for col, grp in ((PAST, m_in[m_in["crossover_date_median"] <= today]),
                         (FUTURE, m_in[m_in["crossover_date_median"] > today])):
            if grp.empty:
                continue
            fig.add_trace(go.Scatter(
                x=grp["crossover_date_median"].dt.strftime("%Y-%m-%d"),
                y=grp["tier"], mode="markers",
                marker=dict(color=col, size=MARKER), showlegend=False,
                hovertemplate="%{y}<br>median %{x|%Y-%m-%d}<extra></extra>"),
                row=i, col=1)

        # Off-window overflow: one small dot at the edge per (row, side), with
        # the true date in small type above it — the median's date when the
        # median itself is out, otherwise the 80% endpoint's.
        for _, r in rows.iterrows():
            m = r["crossover_date_median"]
            for side, edge, lo_c, hi_c in (("left", x0d, "hdi80_low", None),
                                           ("right", x1d, None, "hdi80_high")):
                if side == "left":
                    over = r[lo_c] < edge or m < edge
                    true_date = m if m < edge else r[lo_c]
                    tpos = "top right"
                else:
                    over = r[hi_c] > edge or m > edge
                    true_date = m if m > edge else r[hi_c]
                    tpos = "top left"
                if not over:
                    continue
                col = PAST if edge <= today else FUTURE
                clipped.append((name, r["tier"], side, true_date,
                                (m < x0d) or (m > x1d)))
                fig.add_trace(go.Scatter(
                    x=[edge.isoformat()], y=[r["tier"]], mode="markers+text",
                    marker=dict(color=col, size=9),
                    text=[f"{true_date.year}"], textposition=tpos,
                    textfont=dict(size=19, color=col),
                    showlegend=False, hoverinfo="skip"), row=i, col=1)

        for _, r in rows.iterrows():
            print(f"    {r['tier']}: median {r['crossover_date_median'].date()}, "
                  f"50% [{r['crossover_hdi_low'].date()}, "
                  f"{r['crossover_hdi_high'].date()}], "
                  f"80% [{r['hdi80_low'].date()}, {r['hdi80_high'].date()}]")

        fig.add_vline(x=today.strftime("%Y-%m-%d"), row=i, col=1,
                      line=dict(color=TODAY_COLOR, width=2.5, dash="dot"))
        fig.update_yaxes(categoryorder="array", categoryarray=tiers,
                         tickfont=dict(size=FONT_TICK), showgrid=False,
                         range=[-0.7, len(tiers) - 0.3], row=i, col=1)
        fig.update_xaxes(range=[x0.strftime("%Y-%m-%d"), x1.strftime("%Y-%m-%d")],
                         tickformat="%Y", tickfont=dict(size=FONT_TICK),
                         gridcolor="#e9e9e9", showticklabels=True, row=i, col=1)

    # Legend proxies: colors, the two bar widths, the today line and (when any
    # whisker was cut) the clip arrow. Real traces stay legend-free so each
    # element is named exactly once.
    def proxy(rank, name, **kw):
        fig.add_trace(go.Scatter(x=[None], y=[None], name=name,
                                 legendrank=rank, **kw), row=1, col=1)

    proxy(1, "already behind us", mode="lines",
          line=dict(color=PAST, width=BAR50_W))
    proxy(2, "still ahead", mode="lines",
          line=dict(color=FUTURE, width=BAR50_W))
    proxy(3, "median", mode="markers", marker=dict(color="#888", size=MARKER))
    if len(probs) == 1:
        proxy(4, f"{probs[0]:.0%} interval", mode="lines",
              line=dict(color="#888", width=BAR50_W))
    else:
        proxy(4, f"{probs[0]:.0%} interval (thick)", mode="lines",
              line=dict(color="#888", width=BAR50_W))
        proxy(5, f"{probs[1]:.0%} interval (thin)", mode="lines",
              line=dict(color="#888", width=BAR80_W))
    proxy(6, "today", mode="lines",
          line=dict(color=TODAY_COLOR, width=2.5, dash="dot"))
    if clipped:
        proxy(7, "continues past the window (true date shown)", mode="markers",
              marker=dict(color="#888", size=9))

    fig.update_xaxes(title=dict(text="Crossing date", font=dict(size=FONT_TICK)),
                     row=len(AXES), col=1)
    for ann in fig.layout.annotations[:len(AXES)]:      # subplot titles
        ann.font = dict(size=FONT_AXIS)
    fig.update_layout(
        template="plotly_white",
        legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center",
                    x=0.5, font=dict(size=FONT_LEGEND), bgcolor="rgba(0,0,0,0)",
                    borderwidth=0),
        height=HEIGHT, width=WIDTH, margin=dict(l=520, r=90, t=120, b=260))
    if TITLE is not None:
        fig.update_layout(title=dict(text=TITLE, x=0.5,
                                     font=dict(size=FONT_TITLE)),
                          margin_t=190)

    out.parent.mkdir(parents=True, exist_ok=True)
    save_html(fig, out)
    save_print(fig, out)
    print(f"  fixed x window {x0d.date()} .. {x1d.date()}")
    for name, tier, side, true_date, median_out in clipped:
        what = "median" if median_out else "80% endpoint"
        print(f"  clipped {side}: {name} / {tier} — {what} {true_date.date()} "
              f"marked as a small dot at the edge")
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
