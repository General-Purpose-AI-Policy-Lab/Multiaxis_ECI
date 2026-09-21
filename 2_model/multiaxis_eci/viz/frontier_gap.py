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
# The panels run to the projection's horizon; they start at the first candidate of either group,
# so the reader sees every release the lag is read against — the closed frontier's own first
# measured day included, which is what the early lags are dated from.
PANELS_HORIZON = "2027-07-01"


def frontier_panels_fig(results: dict, scopes: list[str], *, style: FigureStyle = DASHBOARD,
                        window: tuple[str, str] | None = None, today=None,
                        title: str | None = None, group_titles: dict | None = None,
                        y_range: tuple[float, float] | None = None,
                        scope_titles: dict | None = None) -> go.Figure:
    """One stacked panel per scope: both groups' candidates in ECI-H (records as diamonds with
    their 80% interval, the rest as dots), each group's trend line with its 80% band, the human
    tiers named in the right margin and the today line. One y-range for every panel, read off
    the candidates inside the window, so the panels compare at a glance.

    `window` defaults to the first candidate of any scope through `PANELS_HORIZON`: a panel that
    cut its early releases would hide the closed frontier's first measured day, which is the
    very point the early months-behind figures are dated from."""
    today_s = _today(today).strftime("%Y-%m-%d")
    group_titles = group_titles or {}
    scope_titles = scope_titles or SCOPE_TITLES
    if window is None:
        first = min(pd.Timestamp(g.tl["release_date"].min()) for s in scopes
                    for g in (results[s].leader, results[s].follower) if len(g.tl))
        window = ((first - pd.Timedelta(days=120)).strftime("%Y-%m-%d"), PANELS_HORIZON)
    fig = make_subplots(rows=len(scopes), cols=1, shared_xaxes=True, vertical_spacing=0.07,
                        subplot_titles=[scope_titles.get(s, s) for s in scopes])
    fig.update_annotations(font_size=style.font_axis)
    w0, w1 = pd.Timestamp(window[0]), pd.Timestamp(window[1])

    if y_range is None:
        lo_, hi_ = [], []
        for s in scopes:
            r = results[s]
            for g in (r.leader, r.follower):
                inside = g.tl[(g.tl["release_date"] >= w0) & (g.tl["release_date"] <= w1)]
                if len(inside):
                    # The cloud's weak tail may fall off the bottom of the panel: the 2%
                    # quantile keeps a handful of early small models from flattening the rest.
                    # A frontier record never falls off, interval included — it is the subject.
                    lo_.append(float(inside["mean"].quantile(0.02)))
                    recs_in = inside[inside["is_record"]]
                    if len(recs_in):
                        lo_.append(float(recs_in["hdi_low"].min()))
                    hi_.append(float(inside["hdi_high"].max()))
            if len(r.humans):        # a fit without the human baselines has no tiers to fit in
                lo_.append(float(r.humans["mean"].min()))
                hi_.append(float(r.humans["mean"].max()))
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
        if len(r.humans):
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


# Record names hang under the zero line, set vertically: a dozen releases of the same season fit
# side by side that way, and every name starts on the same line so the band reads as one list.
LABEL_ANGLE = 90.0
# Width of a glyph as a fraction of the font size. Plotly measures text in the browser and we
# cannot, so this is the average of the default sans over the model names we set; it sizes the
# band the names hang in and the room each one needs beside its neighbour.
GLYPH_W = 0.55
# The hairline from a name back to the interval bar it belongs to, and the colour of the names.
LEADER_COLOR = "#c2c2c2"
LABEL_COLOR = "#444444"
# The hollow markers mean the same thing in every scope, so their key is drawn in neutral grey.
KEY_COLOR = "#666666"
MARKER_KEY = (("", "circle", "crossing measured"),
              ("backcast", "diamond-open", "crossing dated back"),
              ("censored", "triangle-up-open", "lower bound"))


def _leader_px(style: FigureStyle) -> float:
    """Run of the leader between the end of a line and its name in the right margin. Short: the
    name belongs to the line, and every pixel of leader is a pixel the panel does not get."""
    return 0.9 * style.font_tier


def _label_font(style: FigureStyle) -> int:
    """Type size of the record names. Read off `font_tier`, the scale's margin-label size, rather
    than off `font_note`: the names are the figure's third register, not a footnote, and at the
    POST scale a footnote's size would be the one thing on the panel still asking to be zoomed."""
    return max(int(round(style.font_tier * 0.78)), 8)


def _record_label(raw: str) -> str:
    """The name a record is written under: `pretty_model_name` without the effort suffix.

    The band is as tall as its longest name, so an effort in parentheses — "(thinking)",
    "(unknown)", "(max)" — costs the panel that height for something the hover, the records CSV
    and the summary table all carry. One effort per family reaches the frontier, so the shorter
    name still names exactly one record.
    """
    return pretty_model_name(raw.split("_", 1)[0])


def _record_names(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """The records that carry a lag, in release order, with the name each one is labelled by."""
    rows = df[df["lag_months_median"].notna()].sort_values("release_date")
    return rows, [("≥ " if bool(r.get("censored", False)) else "") + _record_label(r["name"])
                  for _, r in rows.iterrows()]


# Glyphs are narrower down a vertical name than GLYPH_W's allowance for a horizontal row of
# them, and the band's height is dead space wherever it overshoots: measured over the model
# names we set, 0.5 covers the longest without leaving a hand's width of white under it.
BAND_GLYPH_W = 0.46


def _label_band_px(texts: list[str], font: int) -> float:
    """Pixels the name band needs: the longest name, plus the air between it and the data."""
    if not texts:
        return 0.0
    return BAND_GLYPH_W * font * max(len(t) for t in texts) + 0.4 * font


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
            follower_title: str = "open-weights", leader_title: str = "closed",
            scope_titles: dict | None = None) -> go.Figure:
    """Months behind the leader's frontier, per follower record and access scope: a dot with its
    50% interval per record, and a Gaussian-smoothed curve (bandwidth `smooth_months`) with its
    80% band across posterior draws per scope, named at its right end. The records of
    `label_scope` are named beside their dot, the names spread so they do not cover one another.

    The y-range and the plotting area are fixed here rather than left to Plotly (the margins are
    declared `autoexpand=False`), because the name band is measured in pixels and its height has
    to come off a plotting area whose size is known.
    """
    today_d = _today(today)
    scope_titles = scope_titles or SCOPE_TITLES
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
        # The axis opens a little before the first record of the scope the figure names. Another
        # scope may hold an older one — GPT-J on public benchmarks, August 2021, sits eighteen
        # months before anything else — and stretching the axis for a single point costs every
        # other point the room it needs. Records left of the window are off the figure, not out
        # of the series: they stay in the records CSV and in the curves.
        base = label_scope if label_scope in drawn else drawn[0]
        first = pd.Timestamp(results[base].lag_df["release_date"].min())
        # ... and closes on the first of January after today, so the axis ends on a labelled
        # year rather than trailing off a few months past the last one.
        window = ((first - pd.Timedelta(days=120)).strftime("%Y-%m-%d"),
                  f"{today_d.year + 1}-01-01")
    w0, w1 = pd.Timestamp(window[0]), pd.Timestamp(window[1])

    # One y-range over everything the window actually shows (interval bars and bands), with air
    # above; the names get a band of their own under the data, so the data region is what is left
    # of the panel. Records outside the window are left out of it too — reserving height for a
    # point that is off the left edge only flattens the ones on screen.
    vals = [0.0]
    for s in drawn:
        grid_d, _, lo, hi = curves[s]
        inside = (pd.DatetimeIndex(grid_d) >= w0) & (pd.DatetimeIndex(grid_d) <= w1)
        if inside.any():
            vals += [np.nanmin(lo[inside]), np.nanmax(hi[inside])]
        d = results[s].lag_df
        d = d[(d["release_date"] >= w0) & (d["release_date"] <= w1)]
        if len(d):
            vals += [float(d["lag_hdi50_low"].min()), float(d["lag_hdi50_high"].max())]
    y_lo, y_hi = float(np.nanmin(vals)), float(np.nanmax(vals))
    span = max(y_hi - y_lo, 1.0)

    height = int(style.height_per_row * 1.55) + 60
    # The right margin holds the scope names, which carry their benchmark count, and the leaders
    # that tie each one to its line: measured off the longest name, with a tenth in hand for the
    # French renders' own wording. Above the panel only the marker key is left, one row.
    name_px = GLYPH_W * style.font_tier * max(len(scope_titles.get(s, s)) for s in drawn)
    # The marker key sits in that margin too, and on a single-scope panel it is the longer of the
    # two: "All benchmarks" alone would size a margin the key then overflows.
    key_px = (GLYPH_W * style.font_legend * max(len(t) for _, _, t in MARKER_KEY)
              + 3.0 * style.font_legend)
    lead_px = _leader_px(style)
    margin = dict(l=int(style.font_tick * 5),
                  r=int(lead_px + 1.1 * max(name_px, key_px) + style.font_tier),
                  t=int(style.font_title * 2.8) if title else int(style.font_legend * 1.4),
                  b=int(style.font_tick * 1.6 + style.font_axis * 2.2),
                  autoexpand=False)                # these margins ARE the plotting area
    plot_w = style.width - margin["l"] - margin["r"]
    plot_h = height - margin["t"] - margin["b"]

    # The names hang from `band_top`, below both the zero line and the lowest thing drawn. Their
    # band is measured in pixels, so the y-range is solved for it: the data keeps the rest.
    font_lab = _label_font(style)
    band_px = 0.0
    if label_scope in drawn:
        band_px = _label_band_px(_record_names(results[label_scope].lag_df)[1], font_lab)
    band_px = min(band_px, 0.45 * plot_h)
    top = y_hi + 0.10 * span
    band_top = min(0.0, y_lo) - 0.015 * span
    scale = (plot_h - band_px) / max(top - band_top, 1e-9)          # pixels per month
    y_range = (band_top - band_px / scale, top)

    to_px_x = lambda t: (pd.Timestamp(t) - w0).days / max((w1 - w0).days, 1) * plot_w  # noqa: E731
    from_px_x = lambda px: w0 + pd.Timedelta(days=px / plot_w * (w1 - w0).days)        # noqa: E731

    fig = go.Figure()
    ends = []            # (value, name, colour, where the line ends) per scope, for the margin
    shapes_seen = set()  # which marker shapes the data actually uses, for the key below
    for s in drawn:
        r = results[s]
        df = r.lag_df
        col = SCOPE_COLORS.get(s, "#555")
        name = scope_titles.get(s, s)
        grid_d, med, lo, hi = curves[s]
        cens = df["censored"].astype(bool) if "censored" in df else pd.Series(False, index=df.index)
        back = df["backcast"].astype(bool) if "backcast" in df else pd.Series(False, index=df.index)
        fig.add_trace(go.Scatter(x=list(grid_d) + list(grid_d[::-1]), y=list(hi) + list(lo[::-1]),
                                 fill="toself", fillcolor=_rgba(col, 0.10), line=dict(width=0),
                                 hoverinfo="skip", showlegend=False))
        # No legend entry: the scope is named at the end of its own line, in the right margin.
        fig.add_trace(go.Scatter(x=grid_d, y=med, mode="lines", name=name, legendgroup=s,
                                 showlegend=False, line=dict(color=col, width=style.trend),
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
            # Where the line actually stops, in paper units: the curves end at today, well short
            # of the panel's right edge, and the leader has to start there to touch its line.
            x_end = (pd.Timestamp(grid_d[-1]) - w0).days / max((w1 - w0).days, 1)
            ends.append((float(med[-1]), name, col, x_end))

    # The scopes read off their colour, the hollow markers off their shape: one grey key entry per
    # shape the data uses, after the four scope lines, rather than a coloured entry per scope.
    for key, symbol, text in MARKER_KEY:
        if key in shapes_seen:
            filled = not symbol.endswith("-open")
            fig.add_trace(go.Scatter(
                x=[None], y=[None], mode="markers", name=text, hoverinfo="skip",
                marker=dict(symbol=symbol, size=style.marker + 1, color=KEY_COLOR,
                            line=dict(width=1.0 if filled else 1.5,
                                      color="white" if filled else KEY_COLOR))))
    if label_scope in drawn:
        _record_labels(fig, results[label_scope].lag_df, style, to_px_x, from_px_x, plot_w,
                       band_top)
    _line_end_labels(fig, ends, style, y_range, plot_w, plot_h)
    fig.add_hline(y=0, line=dict(color="#999", width=style.refline))
    fig.add_vline(x=today_d.strftime("%Y-%m-%d"),
                  line=dict(color=TODAY_COLOR, width=style.refline, dash=dot_dash(style.refline)))
    fig.add_annotation(x=today_d, y=1.0, yref="paper", text="today", showarrow=False,
                       xanchor="right", xshift=-4, yanchor="top",
                       font=dict(size=style.font_tier, color=TODAY_COLOR))
    # Dates by the year, horizontal: the record names below are the vertical text on this figure.
    # A window of a couple of years (the single-scope figures) gets its months instead.
    # One label per year past two and a half of them: horizontal "2025-07" labels every six
    # months run into each other well before the axis is five years long.
    span_years = (w1 - w0).days / 365.25
    by_year = span_years > 2.5
    fig.update_xaxes(title_text=f"Release date of the {follower_title} record", range=list(window),
                     dtick="M12" if by_year else "M6", tickformat="%Y" if by_year else "%Y-%m",
                     tickangle=0, gridcolor="#f4f4f4")
    # Ticks (and their gridlines) every four months, and only where there is data: the band of
    # names below carries none.
    step = 4.0
    ticks = np.arange(np.ceil(band_top / step) * step, np.floor(top / step) * step + step / 2, step)
    # Two lines: at the write-up's type scale the caption is longer than the panel is tall.
    fig.update_yaxes(title_text=f"Months behind the {leader_title}<br>frontier (ECI-H)",
                     range=list(y_range), tickvals=ticks, gridcolor="#eeeeee", zeroline=False)
    # The marker key goes in the right margin, above the scope names and flush with them: what a
    # dot and a hollow diamond mean is read beside the lines, not in a band over the panel.
    fig.update_layout(template="plotly_white", width=style.width, height=height, margin=margin,
                      legend=dict(orientation="v", yanchor="top", y=1.0, xanchor="left",
                                  x=1.0 + lead_px / plot_w, bgcolor="rgba(0,0,0,0)",
                                  font=dict(size=style.font_legend)))
    if title:
        fig.update_layout(title=dict(text=title, x=0.5, y=0.975, yanchor="top"))
    return apply_fonts(fig, style)


def _record_labels(fig: go.Figure, df: pd.DataFrame, style: FigureStyle, to_px_x, from_px_x,
                   plot_w: float, band_top: float) -> None:
    """Name every record of the labelled scope in the band under the data.

    Every name hangs from `band_top`, set vertically and top-aligned, so the band reads as one
    list however the releases crowd. Names are spread along x only far enough to stay off each
    other (`_spread_labels`, the rule the scope names follow in the right margin), and a grey
    hairline runs from the foot of each record's interval bar down to the name it belongs to.
    """
    rows, texts = _record_names(df)
    if not len(rows):
        return
    font = _label_font(style)
    xs = np.array([to_px_x(r["release_date"]) for _, r in rows.iterrows()], dtype=float)
    # `_spread_labels` takes positions in descending order: the releases run the other way.
    gap = 1.5 * font
    xs_out = _spread_labels(xs[::-1], gap, 0.0, plot_w)[::-1]
    for (_, r), text, x_px in zip(rows.iterrows(), texts, xs_out):
        x_lab = from_px_x(float(x_px))
        fig.add_annotation(x=x_lab, y=band_top, text=text, showarrow=False, textangle=LABEL_ANGLE,
                           xanchor="center", yanchor="top", font=dict(size=font, color=LABEL_COLOR))
        fig.add_shape(type="line", x0=r["release_date"], y0=float(r["lag_hdi50_low"]),
                      x1=x_lab, y1=band_top,
                      line=dict(color=LEADER_COLOR, width=max(1.0, style.refline * 0.7),
                                dash="dot"))


def _line_end_labels(fig: go.Figure, ends: list, style: FigureStyle,
                     y_range: tuple[float, float], plot_w: float, plot_h: float) -> None:
    """Each scope named in the right margin, level with its curve's last value.

    With no legend above the panel these names ARE the figure's key, so they are given room:
    where curves finish together the names spread far enough to read as separate lines, and a
    leader always runs from the end of the line itself — not from the panel's edge, which the
    curves stop well short of — to the name, whether the name had to move or not.
    """
    if not ends:
        return
    ends = sorted(ends, key=lambda e: -e[0])
    span = y_range[1] - y_range[0]
    gap = 1.8 * style.font_tier / plot_h * span
    ys = _spread_labels(np.array([e[0] for e in ends], dtype=float), gap,
                        y_range[0] + 0.02 * span, y_range[1] - 0.02 * span)
    lead = _leader_px(style) / plot_w                       # the leader's run, in paper units
    for (level, name, col, x_end), y in zip(ends, ys):
        fig.add_annotation(x=1.0 + lead, xref="paper", y=y, text=name, showarrow=False,
                           xanchor="left", font=dict(size=style.font_tier, color=col))
        fig.add_shape(type="line", xref="paper", x0=x_end, x1=1.0 + 0.8 * lead, y0=level, y1=y,
                      line=dict(color=col, width=max(1.0, style.refline * 0.8)), opacity=0.85)


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
