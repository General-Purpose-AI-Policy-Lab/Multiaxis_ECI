"""Figures of the two-group frontier gap (analysis.frontier_gap): the frontier panels per
benchmark access scope, the months-behind figure in the manner of Ihle (2026), the summary
table and the crossover panels (reusing viz.forecast.crossover_panels_fig)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from multiaxis_eci.analysis.forecast import _to_year
from multiaxis_eci.viz.core import _rgba, pretty_model_name
from multiaxis_eci.viz.forecast import (
    TODAY_COLOR,
    _spread_labels,
    _tier_labels,
    _today,
    dot_dash,
    trend_dash,
)
from multiaxis_eci.viz.style import DASHBOARD, FigureStyle, apply_fonts

# Okabe-Ito, validated for colour-vision deficiency: the leader group blue, the follower vermillion.
GROUP_COLORS = {"closed": "#0072B2", "open": "#D55E00", "US": "#0072B2", "CN": "#D55E00"}
# One hue per access scope; the amber and the pink are light on white, so every series is also
# named at its line end.
SCOPE_COLORS = {"all": "#0072B2", "public": "#009E73", "semi_private": "#E69F00",
                "private": "#CC79A7"}
SCOPE_TITLES = {"all": "All benchmarks", "public": "Public benchmarks",
                "semi_private": "Semi-private benchmarks", "private": "Private benchmarks"}
DEFAULT_WINDOW = ("2021-01-01", "2027-07-01")     # the panels: from the first closed records to the projection


def frontier_panels_fig(results: dict, scopes: list[str], *, style: FigureStyle = DASHBOARD,
                        window: tuple[str, str] = DEFAULT_WINDOW, today=None,
                        title: str | None = None, group_titles: dict | None = None,
                        y_range: tuple[float, float] | None = None) -> go.Figure:
    """One stacked panel per scope: both groups' candidates in ECI-H (records as diamonds with
    their 80% interval, the rest as dots), each group's trend line with its 80% band, the human
    tiers named in the right margin and the today line. One y-range for every panel, read off
    the candidates inside the window, so the panels compare at a glance."""
    today_s = _today(today).strftime("%Y-%m-%d")
    group_titles = group_titles or {}
    fig = make_subplots(rows=len(scopes), cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=[SCOPE_TITLES.get(s, s) for s in scopes])
    fig.update_annotations(font_size=style.font_axis)
    w0, w1 = pd.Timestamp(window[0]), pd.Timestamp(window[1])

    if y_range is None:
        lo_, hi_ = [], []
        for s in scopes:
            r = results[s]
            for g in (r.leader, r.follower):
                inside = g.tl[(g.tl["release_date"] >= w0) & (g.tl["release_date"] <= w1)]
                if len(inside):
                    lo_.append(float(inside["mean"].quantile(0.02)))
                    hi_.append(float(inside["hdi_high"].max()))
            lo_.append(float(r.humans["mean"].min())); hi_.append(float(r.humans["mean"].max()))
        lo, hi = min(lo_), max(hi_)
        pad = 0.06 * (hi - lo)
        y_range = (lo - pad, hi + 2.5 * pad)

    for i, s in enumerate(scopes, start=1):
        r = results[s]
        for g in (r.leader, r.follower):
            col = GROUP_COLORS.get(g.label, "#555")
            gname = group_titles.get(g.label, g.label)
            tl = g.tl
            cloud, recs = tl[~tl["is_record"]], tl[tl["is_record"]]
            fig.add_trace(go.Scatter(
                x=cloud["release_date"], y=cloud["mean"], mode="markers", name=gname,
                legendgroup=g.label, showlegend=(i == 1),
                marker=dict(color=col, size=style.marker - 1, opacity=0.35, line=dict(width=0)),
                text=cloud["name"],
                hovertemplate="%{text}<br>%{x|%Y-%m-%d}: ECI-H %{y:.1f}<extra>" + gname + "</extra>"),
                row=i, col=1)
            fig.add_trace(go.Scatter(
                x=recs["release_date"], y=recs["mean"], mode="markers",
                name=f"{gname}: frontier record", legendgroup=g.label, showlegend=(i == 1),
                marker=dict(color=col, size=style.marker + 3, symbol="diamond",
                            line=dict(width=1, color="white")),
                error_y=dict(type="data", symmetric=False, array=recs["hdi_high"] - recs["mean"],
                             arrayminus=recs["mean"] - recs["hdi_low"], thickness=style.errbar,
                             width=0, color=_rgba(col, 0.55)),
                text=recs["name"],
                hovertemplate="%{text}<br>%{x|%Y-%m-%d}: ECI-H %{y:.1f}<extra>record</extra>"),
                row=i, col=1)
            fc = g.forecast(horizon=window[1])
            if fc is None:
                continue
            gx = pd.to_datetime(fc.grid_dates)
            fig.add_trace(go.Scatter(x=list(gx) + list(gx[::-1]), y=list(fc.hi) + list(fc.lo[::-1]),
                                     fill="toself", fillcolor=_rgba(col, 0.12), line=dict(width=0),
                                     hoverinfo="skip", showlegend=False), row=i, col=1)
            fig.add_trace(go.Scatter(
                x=gx, y=fc.median, mode="lines", name=f"{gname}: trend (80% band)",
                legendgroup=g.label, showlegend=(i == 1),
                line=dict(color=col, width=style.trend, dash=trend_dash(style.trend)),
                hovertemplate="%{x|%Y-%m}: ECI-H %{y:.1f}<extra>" + gname + " trend</extra>"),
                row=i, col=1)
        fig.add_vline(x=today_s, row=i, col=1,
                      line=dict(color=TODAY_COLOR, width=style.refline, dash=dot_dash(style.refline)))
        _tier_labels(fig, r.humans, i, "y" if i == 1 else f"y{i}", y_range, style, {})
        fig.update_yaxes(title_text="ECI-H", range=list(y_range), gridcolor="#eeeeee",
                         zeroline=False, row=i, col=1)
    fig.update_xaxes(range=[window[0], window[1]], dtick="M12", tickformat="%Y",
                     gridcolor="#f4f4f4", showticklabels=True)
    fig.update_xaxes(title_text="Release date", row=len(scopes), col=1)
    fig.update_layout(template="plotly_white", width=style.width,
                      height=style.height_per_row * len(scopes) + 160,
                      legend=dict(orientation="h", yanchor="top", y=-0.11 / len(scopes) - 0.02,
                                  xanchor="left", x=0, font=dict(size=style.font_legend)),
                      margin=dict(l=int(style.font_tick * 5), r=int(style.font_tier * 21),
                                  t=100 if title else 70, b=int(style.font_tick * 9)))
    if title:
        fig.update_layout(title=dict(text=title, x=0.5))
    return apply_fonts(fig, style)


def _kernel_curves(years: np.ndarray, lags: np.ndarray, grid: np.ndarray,
                   bandwidth_years: float) -> np.ndarray:
    """(S, G) Gaussian-kernel smooth of the per-draw lags over the records' years, NaN lags
    dropped from the weights of their draw."""
    w = np.exp(-0.5 * ((grid[:, None] - years[None, :]) / bandwidth_years) ** 2)   # (G, R)
    finite = np.isfinite(lags)                                                     # (S, R)
    num = np.where(finite, lags, 0.0) @ w.T                                        # (S, G)
    den = finite.astype(float) @ w.T
    with np.errstate(invalid="ignore", divide="ignore"):
        # No curve where the records give the kernel less than one and a half points of weight:
        # a lone record (GPT-J in 2021) shows as its dot, not as a curve of its own.
        return np.where(den > 1.5, num / den, np.nan)


# Record names are set at this angle, reading up to the right: releases crowd into 2024-2026 and
# only the diagonal keeps a dozen names side by side in that stretch.
LABEL_ANGLE = 30.0
# Width of a glyph as a fraction of the font size, for the collision boxes below. Plotly measures
# text in the browser and we cannot, so this is the average of the default sans over the model
# names we set; erring high only spaces the names out a little more than needed.
GLYPH_W = 0.55
# The hairline from a name back to the interval bar it belongs to, and the colour of the names.
LEADER_COLOR = "#c2c2c2"
LABEL_COLOR = "#444444"
# The hollow markers mean the same thing in every scope, so their key is drawn in neutral grey.
KEY_COLOR = "#666666"
MARKER_KEY = (("backcast", "diamond-open", "crossing dated back"),
              ("censored", "triangle-up-open", "lower bound"))


def _rotated_bbox(w: float, h: float, up: bool) -> tuple[float, float, float, float]:
    """(dx0, dx1, dy0, dy1): the pixels a `w` x `h` text box turned `LABEL_ANGLE` counterclockwise
    takes around its anchor, in a y-up frame. A name above its dot is anchored at its bottom left
    and runs up to the right; one below is anchored at its top right and runs down to the left."""
    c, s = np.cos(np.radians(LABEL_ANGLE)), np.sin(np.radians(LABEL_ANGLE))
    if up:
        return -s * h, c * w, 0.0, s * w + c * h
    return -c * w, s * h, -(s * w + c * h), 0.0


def _overlap(a: tuple, b: tuple) -> float:
    """Area two (x0, x1, y0, y1) boxes share."""
    dx = min(a[1], b[1]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[2], b[2])
    return dx * dy if dx > 0 and dy > 0 else 0.0


def _stack_labels(anchors: list, boxes: list, ups: list, bounds: tuple[float, float], *,
                  obstacles: list | None = None, pad: float = 6.0, step: float = 5.0,
                  max_push: float = 200.0, dot_cost: float = 0.2) -> list[float]:
    """Vertical offsets (pixels) that keep the rotated record names off one another and off the
    dots already drawn.

    Names are placed in release order: each starts `pad` off the end of its interval bar and is
    pushed further from its dot until its box clears the names already placed, the `obstacles`
    (every scope's markers) and the edges of `bounds` (the plotting area, y-up). Where nothing is
    free the least-costly offset wins, so a crowded name is spread, never dropped; grazing a
    marker costs `dot_cost` of what covering another name costs, because a name over a name is
    the one thing that cannot be read.
    """
    placed, obstacles, offsets = [], list(obstacles or ()), []
    for (x, y), (dx0, dx1, dy0, dy1), up in zip(anchors, boxes, ups):
        best, best_cost, d = None, None, pad
        while d <= max_push:
            off = d if up else -d
            box = (x + dx0, x + dx1, y + off + dy0, y + off + dy1)
            if bounds[0] <= box[2] and box[3] <= bounds[1]:
                cost = (sum(_overlap(box, b) for b in placed)
                        + dot_cost * sum(_overlap(box, b) for b in obstacles))
                if cost == 0.0:
                    best, best_cost = off, 0.0
                    break
                if best_cost is None or cost < best_cost:
                    best, best_cost = off, cost
            d += step
        best = (pad if up else -pad) if best is None else best
        offsets.append(best)
        placed.append((x + dx0, x + dx1, y + best + dy0, y + best + dy1))
    return offsets


def _lag_curve(r, smooth_months: float, today_d: pd.Timestamp):
    """(dates, median, lo, hi) of one scope's smoothed lag curve, or None when it has no record.
    The curve rests on every dated lag; the censored lower bounds stay out of it."""
    df, draws = r.lag_df, r.lag_draws
    if not len(df):
        return None
    cens = df["censored"].astype(bool) if "censored" in df else pd.Series(False, index=df.index)
    years = _to_year(pd.DatetimeIndex(df.loc[~cens, "release_date"]))
    grid_d = pd.date_range(pd.Timestamp(df["release_date"].min()).replace(day=1), today_d, freq="MS")
    curves = _kernel_curves(years, draws[:, (~cens).to_numpy()], _to_year(grid_d),
                            smooth_months / 12.0)
    lo, hi = np.nanpercentile(curves, [10, 90], axis=0)
    return grid_d, np.nanmedian(curves, axis=0), lo, hi


def lag_fig(results: dict, scopes: list[str], *, style: FigureStyle = DASHBOARD,
            window: tuple[str, str] | None = None, today=None, smooth_months: float = 6.0,
            label_scope: str = "all", title: str | None = None,
            follower_title: str = "open-weights", leader_title: str = "closed") -> go.Figure:
    """Months behind the leader's frontier, per follower record and access scope: a dot with its
    50% interval per record, and a Gaussian-smoothed curve (bandwidth `smooth_months`) with its
    80% band across posterior draws per scope, named at its right end. The records of
    `label_scope` are named beside their dot, the names spread so they do not cover one another.

    The y-range and the plotting area are fixed here rather than left to Plotly, because the name
    placement works in pixels and has to know where the panel's edges are.
    """
    today_d = _today(today)
    curves = {s: _lag_curve(results[s], smooth_months, today_d) for s in scopes}
    drawn = [s for s in scopes if curves[s] is not None]
    if not drawn:
        # Every record of every scope needed the dating-back assumption (MAX_DATED_BACK_FRAC):
        # say so on the panel rather than raise out of a plotting driver.
        fig = go.Figure()
        fig.add_annotation(x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                           text="no record with a measured crossing in any scope",
                           font=dict(size=style.font_axis, color=LABEL_COLOR))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(template="plotly_white", width=style.width,
                          height=int(style.height_per_row * 1.8) + 60,
                          title=dict(text=title, x=0.5) if title else None)
        return apply_fonts(fig, style)
    if window is None:
        # No cut in time: the axis starts a little before the first record of any scope.
        first = min(pd.Timestamp(results[s].lag_df["release_date"].min()) for s in drawn)
        window = ((first - pd.Timedelta(days=120)).strftime("%Y-%m-%d"),
                  (today_d + pd.Timedelta(days=200)).strftime("%Y-%m-%d"))
    w0, w1 = pd.Timestamp(window[0]), pd.Timestamp(window[1])

    # One y-range over everything drawn (interval bars and bands), with air above for the names.
    vals = [0.0]
    for s in drawn:
        _, _, lo, hi = curves[s]
        d = results[s].lag_df
        vals += [np.nanmin(lo), np.nanmax(hi), float(d["lag_hdi50_low"].min()),
                 float(d["lag_hdi50_high"].max())]
    y_lo, y_hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    span = max(y_hi - y_lo, 1.0)
    y_range = (y_lo - 0.06 * span, y_hi + 0.12 * span)

    height = int(style.height_per_row * 1.8) + 60
    margin = dict(l=int(style.font_tick * 5), r=int(style.font_tier * 14),
                  t=int(style.font_title * 5.2) if title else int(style.font_legend * 4),
                  b=int(style.font_tick * 6))
    plot_w = style.width - margin["l"] - margin["r"]
    plot_h = height - margin["t"] - margin["b"]
    to_px_x = lambda t: (pd.Timestamp(t) - w0).days / max((w1 - w0).days, 1) * plot_w  # noqa: E731
    to_px_y = lambda v: (v - y_range[0]) / (y_range[1] - y_range[0]) * plot_h          # noqa: E731
    px_to_y = (y_range[1] - y_range[0]) / plot_h

    fig = go.Figure()
    ends = []            # (value at the right end, name, colour) per scope, for the margin names
    obstacles = []       # every scope's markers in pixels, so no record name lands on a dot
    shapes_seen = set()  # which marker shapes the data actually uses, for the key below
    for s in drawn:
        r = results[s]
        df = r.lag_df
        col = SCOPE_COLORS.get(s, "#555")
        name = SCOPE_TITLES.get(s, s)
        grid_d, med, lo, hi = curves[s]
        cens = df["censored"].astype(bool) if "censored" in df else pd.Series(False, index=df.index)
        back = df["backcast"].astype(bool) if "backcast" in df else pd.Series(False, index=df.index)
        fig.add_trace(go.Scatter(x=list(grid_d) + list(grid_d[::-1]), y=list(hi) + list(lo[::-1]),
                                 fill="toself", fillcolor=_rgba(col, 0.10), line=dict(width=0),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=grid_d, y=med, mode="lines", name=name, legendgroup=s,
                                 line=dict(color=col, width=style.trend),
                                 hovertemplate="%{x|%Y-%m}: %{y:.1f} months<extra>" + name + "</extra>"))

        def points(sub, symbol, hover, key, scope=s, col=col, name=name):
            if not len(sub):
                return
            shapes_seen.add(key)
            fig.add_trace(go.Scatter(
                x=sub["release_date"], y=sub["lag_months_median"], mode="markers",
                legendgroup=scope, name=f"{name}: records", showlegend=False,
                marker=dict(color=col, size=style.marker + 1, symbol=symbol,
                            line=dict(width=1.5 if key else 1, color=col if key else "white")),
                error_y=dict(type="data", symmetric=False,
                             array=sub["lag_hdi50_high"] - sub["lag_months_median"],
                             arrayminus=sub["lag_months_median"] - sub["lag_hdi50_low"],
                             thickness=style.errbar, width=0, color=_rgba(col, 0.6)),
                text=sub["name"],
                customdata=np.column_stack([(sub[f"frac_{key}"] * 100).round() if key else np.zeros(len(sub)),
                                            sub["first_leader_above"] if "first_leader_above" in sub else [""] * len(sub)]),
                hovertemplate="%{text}<br>%{x|%Y-%m-%d}: " + hover
                              + "<br>first closed model above: %{customdata[1]}<extra>" + name + "</extra>"))

        points(df[~cens & ~back], "circle", "%{y:.1f} months behind", "")
        # Backcast records: the closed frontier already exceeded them on its first measured day;
        # the crossing is dated back at the frontier's early rate (a hollow diamond).
        points(df[back & ~cens], "diamond-open",
               "%{y:.1f} months behind (backcast in %{customdata[0]:.0f}% of draws)", "backcast")
        # Censored records: nothing to date back with, the lag is a lower bound (hollow triangle).
        points(df[cens], "triangle-up-open",
               "at least %{y:.1f} months behind (bound in %{customdata[0]:.0f}% of draws)", "censored")
        if np.isfinite(med[-1]):
            ends.append((float(med[-1]), name, col))
        half = style.marker + 3
        obstacles += [(to_px_x(d) - half, to_px_x(d) + half, to_px_y(v) - half, to_px_y(v) + half)
                      for d, v in zip(df["release_date"], df["lag_months_median"])
                      if np.isfinite(v)]
        # The curve itself is an obstacle too, sampled along its length: a name across a trend
        # line hides the very thing the figure is about.
        obstacles += [(to_px_x(d) - 6, to_px_x(d) + 6, to_px_y(v) - 5, to_px_y(v) + 5)
                      for d, v in zip(grid_d[::2], med[::2]) if np.isfinite(v)]

    # The scopes read off their colour, the hollow markers off their shape: one grey key entry per
    # shape the data uses, after the four scope lines, rather than a coloured entry per scope.
    for key, symbol, text in MARKER_KEY:
        if key in shapes_seen:
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers", name=text, hoverinfo="skip",
                marker=dict(symbol=symbol, size=style.marker + 1, color=KEY_COLOR,
                            line=dict(width=1.5, color=KEY_COLOR))))
    if label_scope in drawn:
        _record_labels(fig, results[label_scope].lag_df, style, to_px_x, to_px_y, px_to_y,
                       (plot_w, plot_h), obstacles)
    _line_end_labels(fig, ends, style, y_range, plot_h)
    fig.add_hline(y=0, line=dict(color="#999", width=style.refline))
    fig.add_vline(x=today_d.strftime("%Y-%m-%d"),
                  line=dict(color=TODAY_COLOR, width=style.refline, dash=dot_dash(style.refline)))
    fig.add_annotation(x=today_d, y=1.0, yref="paper", text="today", showarrow=False,
                       xanchor="right", xshift=-4, yanchor="top",
                       font=dict(size=style.font_note, color=TODAY_COLOR))
    span_years = (w1 - w0).days / 365.25
    fig.update_xaxes(title_text=f"Release date of the {follower_title} record", range=list(window),
                     dtick="M12" if span_years > 5 else "M6", tickformat="%Y" if span_years > 5 else "%Y-%m",
                     gridcolor="#f4f4f4")
    fig.update_yaxes(title_text=f"Months behind the {leader_title} frontier (ECI-H)",
                     range=list(y_range), dtick=5, gridcolor="#eeeeee", zeroline=False)
    fig.update_layout(template="plotly_white", width=style.width, height=height, margin=margin,
                      legend=dict(orientation="h", yanchor="bottom", y=1.005, xanchor="left",
                                  font=dict(size=style.font_legend)))
    if title:
        fig.update_layout(title=dict(text=title, x=0.5, y=0.975, yanchor="top"))
    return apply_fonts(fig, style)


def _record_labels(fig: go.Figure, df: pd.DataFrame, style: FigureStyle,
                   to_px_x, to_px_y, px_to_y: float, plot: tuple[float, float],
                   obstacles: list) -> None:
    """Name every record of the labelled scope beside its dot: names take alternate sides, one
    that would run off an edge takes the other, and `_stack_labels` spreads them off each other
    and off `obstacles` (the markers of every scope). A hairline joins a name that had to move
    to the end of its interval bar."""
    rows = df[df["lag_months_median"].notna()].sort_values("release_date")
    if not len(rows):
        return
    plot_w, plot_h = plot
    font = max(style.font_note - 3, 8)
    texts = [("≥ " if bool(r.get("censored", False)) else "") + pretty_model_name(r["name"])
             for _, r in rows.iterrows()]
    sizes = [(GLYPH_W * font * len(t), 1.15 * font) for t in texts]
    xs = [to_px_x(r["release_date"]) for _, r in rows.iterrows()]
    tops = rows["lag_hdi50_high"].to_numpy(dtype=float)
    bots = rows["lag_hdi50_low"].to_numpy(dtype=float)
    ups = []
    for k, (x, (w, h)) in enumerate(zip(xs, sizes)):
        up = k % 2 == 0
        # A name above its dot runs up to the right, one below runs down to the left: near an
        # edge, take the side that keeps the name on the panel instead of in the margin.
        if up and x + _rotated_bbox(w, h, True)[1] > plot_w:
            up = False
        elif not up and x + _rotated_bbox(w, h, False)[0] < 0.0:
            up = True
        ups.append(up)
    anchors = [(x, to_px_y(t if up else b)) for x, t, b, up in zip(xs, tops, bots, ups)]
    boxes = [_rotated_bbox(w, h, up) for (w, h), up in zip(sizes, ups)]
    offsets = _stack_labels(anchors, boxes, ups, (0.0, plot_h), obstacles=obstacles)
    for (_, r), text, up, top, bot, off in zip(rows.iterrows(), texts, ups, tops, bots, offsets):
        base = top if up else bot
        y = base + off * px_to_y
        fig.add_annotation(x=r["release_date"], y=y, text=text, showarrow=False,
                           textangle=-LABEL_ANGLE, yanchor="bottom" if up else "top",
                           xanchor="left" if up else "right", font=dict(size=font, color=LABEL_COLOR))
        if abs(off) > 14.0:
            # Grey, not the scope's hue: a leader must not read as one more interval bar.
            fig.add_shape(type="line", x0=r["release_date"], x1=r["release_date"], y0=base, y1=y,
                          line=dict(color=LEADER_COLOR, width=max(1.0, style.refline * 0.7),
                                    dash="dot"))


def _line_end_labels(fig: go.Figure, ends: list, style: FigureStyle,
                     y_range: tuple[float, float], plot_h: float) -> None:
    """Each scope named in the right margin, level with its curve's last value; where two curves
    finish together the names spread just enough to stay apart, joined by a thin leader."""
    if not ends:
        return
    ends = sorted(ends, key=lambda e: -e[0])
    span = y_range[1] - y_range[0]
    gap = 1.35 * style.font_tier / plot_h * span
    ys = _spread_labels(np.array([e[0] for e in ends], dtype=float), gap,
                        y_range[0] + 0.02 * span, y_range[1] - 0.02 * span)
    for (level, name, col), y in zip(ends, ys):
        fig.add_annotation(x=1.008, xref="paper", y=y, text=name, showarrow=False, xanchor="left",
                           font=dict(size=style.font_tier, color=col))
        if abs(y - level) > 0.25 * gap:
            fig.add_shape(type="line", xref="paper", x0=1.0, x1=1.007, y0=level, y1=y,
                          line=dict(color=col, width=max(1, style.refline / 2)), opacity=0.8)


def summary_table_fig(cells: pd.DataFrame, title: str, *, width: int = 1500,
                      row_height: int = 30) -> go.Figure:
    """A plotly table from a frame of preformatted strings (first column = row label). The
    first column is wider, and the figure's height leaves room for its wrapped labels."""
    hdr = dict(fill_color="#1f3a5f", font=dict(color="white", size=14), align="left", height=32)
    cel = dict(fill_color=[["#f7f9fc", "white"] * (len(cells) // 2 + 1)], font=dict(size=13),
               align="left", height=row_height)
    ncol = len(cells.columns)
    fig = go.Figure(go.Table(columnwidth=[2.2] + [1.0] * (ncol - 1),
                             header=dict(values=list(cells.columns), **hdr),
                             cells=dict(values=[cells[c].tolist() for c in cells.columns], **cel)))
    fig.update_layout(width=width, height=int(row_height * 1.6 * (len(cells) + 1)) + 110,
                      margin=dict(l=20, r=20, t=60, b=10), title=dict(text=title, x=0.5, font=dict(size=15)))
    return fig
