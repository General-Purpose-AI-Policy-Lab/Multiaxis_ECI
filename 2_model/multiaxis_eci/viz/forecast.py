"""Frontier-forecast figures: the trend panels and the crossover panels.

One design for the dashboard, the per-fit figure folders and the blog post: the builders take
a `FigureStyle` (viz.style) and the post's scripts pass `POST`. The trend panel is the post's
figure 2 (dated models with their intervals, the record envelope's band and median, the human
tiers as dashed lines named in the right margin, a today line); the crossover panel is its
figure 3 (per tier, the median crossing date and one or two interval bars split at the today
line, the share behind us in green and the share ahead in red, off-window ends clipped and
dated at the edge).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from multiaxis_eci.analysis.forecast import _to_year
from multiaxis_eci.viz.core import (
    FUTURE_COLOR,
    HUMAN_LEVEL_LABELS,
    MODEL_COLOR,
    PASSED_COLOR,
    _rgba,
    human_tier_palette,
)
from multiaxis_eci.viz.style import DASHBOARD, FigureStyle, apply_fonts

FORECAST_COLOR = "#ff9500"        # frontier extrapolation (the reasoning regime)
OTHER_COLOR = "#8064a2"           # the non-reasoning regime's line (the muted purple of its fit set)


def trend_dash(width: float) -> str:
    """Both projection lines share one short-dash pattern, scaled with the line width so the
    post's thick lines keep the dashboard's proportions (a 6px dash on a 5px line reads as a
    row of squares)."""
    return f"{max(6.0, 2.7 * width):.1f}px,{max(4.0, 1.8 * width):.1f}px"


def dot_dash(width: float) -> str:
    """A row of true dots for the today line: Plotly's named "dot" pattern stretches with the
    line width and prints as short dashes."""
    return f"{max(1.5, width):.1f}px,{max(3.0, 2 * width):.1f}px"
# The releases each regime's line was fitted on: muted red for the reasoning fit set, muted
# purple for the non-reasoning one (their markers and whiskers; every other release keeps
# MODEL_COLOR).
REASONING_FIT_COLOR = "#c0504d"
OTHER_FIT_COLOR = "#8064a2"
# Every crossover figure shares this x-range, so panels from different fits, chain groups
# and scopes read against the same years.
CROSSOVER_WINDOW = ("2015-01-01", "2030-01-01")
# Every trend panel stops at the same year too; its start follows the data (the post fixes 2023).
TREND_WINDOW_END = "2030-01-01"
TODAY_COLOR = "#444"

# Every string these figures draw, per language (English default, French for the
# `fr/` renders); tier names come from core.HUMAN_LEVEL_LABELS.
FORECAST_TEXT = {
    "en": {"trend": "Projected frontier", "others": "non-reasoning models", "release": "Release date",
           "ability": "ability",
           "crossing_axis": "Crossing date", "behind": "already behind us",
           "ahead": "still ahead", "median": "median", "interval": "{p:.0%} interval",
           "thick": "{p:.0%} interval (thick)", "thin": "{p:.0%} interval (thin)",
           "today": "today", "clipped": "continues past the window (date shown)"},
    "fr": {"trend": "Frontière projetée", "others": "modèles sans raisonnement",
           "release": "Date de sortie", "ability": "capacité",
           "crossing_axis": "Date de croisement", "behind": "déjà derrière nous",
           "ahead": "encore à venir", "median": "médiane", "interval": "intervalle à {p:.0%}",
           "thick": "intervalle à {p:.0%} (épais)", "thin": "intervalle à {p:.0%} (fin)",
           "today": "aujourd'hui", "clipped": "dépasse la fenêtre (date indiquée)"},
}


def _today(today) -> pd.Timestamp:
    return pd.Timestamp(today) if today is not None else pd.Timestamp.today().normalize()


def _spread_labels(levels: np.ndarray, gap: float, lo: float, hi: float) -> np.ndarray:
    """Label positions for `levels` (sorted descending): each label stays on its line unless two
    would overlap, in which case only the crowded ones move, symmetrically, until every pair is
    `gap` apart; the stack is then kept inside [lo, hi]."""
    y = levels.astype(float).copy()
    for _ in range(200):
        moved = False
        for a in range(len(y) - 1):
            short = gap - (y[a] - y[a + 1])
            if short > 1e-9:
                y[a] += short / 2
                y[a + 1] -= short / 2
                moved = True
        if not moved:
            break
    if len(y):
        y += max(0.0, lo - y[-1]) - max(0.0, y[0] - hi)
    return y


def _tier_labels(fig, hs: pd.DataFrame, row: int, yref: str, ylim: tuple[float, float],
                 style: FigureStyle, labels: dict, gap_frac: float | None = None) -> None:
    """Dashed tier lines in Blues (strongest darkest) plus their names in the right margin.

    A name sits level with its line whenever the neighbours leave room; where tiers crowd, the
    crowded names spread just enough to stay legible and a thin leader joins each displaced
    name to its line.
    """
    rows = hs.sort_values("mean", ascending=False).reset_index(drop=True)
    colors = human_tier_palette(len(rows))
    span = ylim[1] - ylim[0]
    plot_px = 0.8 * style.height_per_row                   # the panel's plotting height
    if gap_frac is None:
        # The type height plus a little air, in axis units, whatever the style's scale.
        gap_frac = 1.3 * style.font_tier / plot_px
    gap = gap_frac * span
    levels = rows["mean"].to_numpy(dtype=float)
    ys = _spread_labels(levels, gap, ylim[0] + 0.02 * span, ylim[1] - 0.02 * span)
    for (_, r), y, col in zip(rows.iterrows(), ys, colors):
        level = float(r["mean"])
        fig.add_hline(y=level, row=row, col=1,
                      line=dict(color=col, width=style.refline, dash="dash"), opacity=0.75)
        text = labels.get(r["name"], r["name"])
        fig.add_annotation(x=1.008, y=y, xref="paper", yref=yref, text=text, showarrow=False,
                           xanchor="left", font=dict(size=style.font_tier, color=col))
        if abs(y - level) > 0.25 * gap:
            # A thin leader from the line's end to the displaced name, drawn in the margin.
            fig.add_shape(type="line", xref="paper", yref=yref, x0=1.0, y0=level, x1=1.007,
                          y1=y, line=dict(color=col, width=max(1, style.refline / 2)),
                          opacity=0.8)


def frontier_trend_fig(per_axis: dict, axes: list[str], titles: dict | None = None, *,
                       style: FigureStyle = DASHBOARD, window: tuple[str, str] | None = None,
                       today=None, lang: str = "en", human_labels: dict | None = None,
                       title: str | None = None, y_label: str | None = None) -> go.Figure:
    """The frontier trend per axis, one stacked panel each. `y_label` names the ability scale.

    `per_axis[name]` holds `fc` (a ForecastResult: grid_dates, lo, median, hi, slope), `tl`
    (the candidates' timeline frame: release_date, mean, hdi_low, hdi_high, name) and `hs` (the
    human tiers: name, mean, and hdi_low / hdi_high, which band the bottom and top tiers). Each panel draws the dated models with their intervals, the
    forecast band (fc.lo to fc.hi) and its median, the non-reasoning regime's line and band in
    muted purple over its own span when `fc.other` is set, its median alone carried on as a
    thin line up to today so the change of slope shows, the releases each line was fitted on in
    that regime's colour (`fc.fit_names` muted red, `fc.other.fit_names` muted purple; markers
    and whiskers), the tiers as dashed lines named in the
    right margin, and the today line; no legend, the caption names the series. `window` fixes
    the x-range on every panel (the post uses 2023 to 2030); None starts at the first candidate
    and stops at `TREND_WINDOW_END` (2030), the crossover figures' right edge.
    """
    text = FORECAST_TEXT[lang]
    labels = HUMAN_LEVEL_LABELS[lang] if human_labels is None else human_labels
    titles = titles or {}
    today_s = _today(today).strftime("%Y-%m-%d")
    panel_titles = None if (title and len(axes) == 1) else [titles.get(a, a) for a in axes]
    fig = make_subplots(rows=len(axes), cols=1, shared_xaxes=True, vertical_spacing=0.085,
                        subplot_titles=panel_titles)
    fig.update_annotations(font_size=style.font_axis)          # panel titles
    x_min, x_max = None, None
    for i, name in enumerate(axes, start=1):
        d = per_axis[name]
        fc, tl, hs = d["fc"], d["tl"], d["hs"]
        gx = pd.to_datetime(fc.grid_dates)
        fig.add_trace(go.Scatter(x=gx, y=fc.lo, mode="lines", showlegend=False,
                                 line=dict(width=0), hoverinfo="skip"), row=i, col=1)
        fig.add_trace(go.Scatter(x=gx, y=fc.hi, mode="lines", fill="tonexty",
                                 fillcolor=_rgba("rgb(255,149,0)", 0.18), showlegend=False,
                                 line=dict(width=0), hoverinfo="skip"), row=i, col=1)
        other = getattr(fc, "other", None)
        if other is not None:
            # The non-reasoning regime: its own line and band, over its own span only.
            gxo = pd.to_datetime(other.grid_dates)
            fig.add_trace(go.Scatter(x=gxo, y=other.lo, mode="lines", showlegend=False,
                                     line=dict(width=0), hoverinfo="skip"), row=i, col=1)
            fig.add_trace(go.Scatter(x=gxo, y=other.hi, mode="lines", fill="tonexty",
                                     fillcolor=_rgba(OTHER_COLOR, 0.12), showlegend=False,
                                     line=dict(width=0), hoverinfo="skip"), row=i, col=1)
            fig.add_trace(go.Scatter(x=gxo, y=other.median, mode="lines", showlegend=False,
                                     name=text["others"],
                                     line=dict(color=OTHER_COLOR, width=style.trend * 0.8,
                                               dash=trend_dash(style.trend)),
                                     hovertemplate="%{x|%Y-%m}: %{y:.2f}<extra>"
                                                   + text["others"] + "</extra>"), row=i, col=1)
            # The others' median alone, carried on from its last release to today (no band):
            # the eye reads the change of slope against the reasoning line.
            ext = pd.date_range(gxo.max(), _today(today), freq="MS")
            if len(ext) >= 2:
                ext_y = np.median(other.intercept[:, None]
                                  + other.slope[:, None] * _to_year(ext)[None, :], axis=0)
                fig.add_trace(go.Scatter(x=ext, y=ext_y, mode="lines", showlegend=False,
                                         line=dict(color=OTHER_COLOR, width=style.trend * 0.5,
                                                   dash=trend_dash(style.trend)), opacity=0.7,
                                         hoverinfo="skip"), row=i, col=1)
        dates = pd.to_datetime(tl["release_date"])
        # The measured cloud, then the fit sets over it in their regime's colour: the releases
        # the reasoning line was fitted on (muted red) and those of the non-reasoning line
        # (muted purple), so the eye can tell what each line rests on.
        fit_sets = [(REASONING_FIT_COLOR, set(fc.fit_names or []))]
        if other is not None:
            fit_sets.append((OTHER_FIT_COLOR, set(other.fit_names or [])))
        in_fit = tl["name"].isin(set().union(*(names for _, names in fit_sets)))
        # The rest of the cloud is faint: since the gate is the coverage rule alone it holds
        # every effort of every family, several hundred points per axis.
        for col_, sub, alpha_m, alpha_e in (
                [(MODEL_COLOR, tl[~in_fit], 0.3, 0.18)]
                + [(c, tl[tl["name"].isin(n)], 0.9, 0.6) for c, n in fit_sets]):
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(sub["release_date"]), y=sub["mean"], mode="markers",
                marker=dict(color=col_, size=style.marker, opacity=alpha_m, line=dict(width=0)),
                error_y=dict(type="data", symmetric=False,
                             array=sub["hdi_high"] - sub["mean"],
                             arrayminus=sub["mean"] - sub["hdi_low"],
                             thickness=style.errbar, width=0, color=_rgba(col_, alpha_e)),
                text=sub["name"], showlegend=False,
                hovertemplate="%{text}<br>%{x|%Y-%m-%d}: %{y:.2f}<extra></extra>"), row=i, col=1)
        fig.add_trace(go.Scatter(x=gx, y=fc.median, mode="lines", showlegend=False,
                                 name=text["trend"],
                                 line=dict(color=FORECAST_COLOR, width=style.trend,
                                           dash=trend_dash(style.trend)),
                                 hovertemplate="%{x|%Y-%m}: %{y:.2f}<extra></extra>"),
                      row=i, col=1)
        # The y-range follows the data, not the projection: the models' intervals and the
        # human tier medians. A band or trend that climbs past the top tier by 2030 runs
        # off the panel rather than flattening everything else into the bottom half.
        # Tier intervals are wide on a K-axis fit (a tier is scored on a handful of
        # benchmarks); they are drawn, but they do not set the range either.
        has_hdi = {"hdi_low", "hdi_high"} <= set(hs.columns)
        lo = min(float(tl["hdi_low"].min()), float(hs["mean"].min()))
        hi = max(float(tl["hdi_high"].max()), float(hs["mean"].max()))
        pad = 0.06 * (hi - lo)
        # More air above than below: the trend and its band leave the panel through the top,
        # and the topmost tier name needs room.
        ylim = (lo - pad, hi + 3 * pad)
        if has_hdi and len(hs):
            # The bottom and top tiers of the axis carry their 80% interval, so the ladder's
            # bounds read as the uncertain quantities they are: a faint band when it is
            # narrower than a quarter of the panel, otherwise (a tier scored on a handful of
            # benchmarks, on a K-axis fit) a whisker with caps near the panel's right edge,
            # since a band that wide would only tint the whole panel.
            ranked = hs.sort_values("mean")
            pal = human_tier_palette(len(hs))
            x_end = pd.Timestamp(window[1] if window else TREND_WINDOW_END)
            for r_, col_, back in ((ranked.iloc[-1], pal[0], 100), (ranked.iloc[0], pal[-1], 220)):
                y0, y1, m = float(r_["hdi_low"]), float(r_["hdi_high"]), float(r_["mean"])
                name_ = labels.get(r_["name"], r_["name"])
                if y1 - y0 <= 0.25 * (ylim[1] - ylim[0]):
                    fig.add_hrect(y0=y0, y1=y1, row=i, col=1, fillcolor=_rgba(col_, 0.08),
                                  line_width=0, layer="below")
                else:
                    fig.add_trace(go.Scatter(
                        x=[x_end - pd.Timedelta(days=back)], y=[m], mode="markers",
                        marker=dict(color=col_, size=style.marker - 1, symbol="square"),
                        error_y=dict(type="data", symmetric=False, array=[y1 - m],
                                     arrayminus=[m - y0], color=col_, thickness=style.errbar,
                                     width=style.marker), showlegend=False, opacity=0.8,
                        hovertemplate=f"{name_}: 80% interval [{y0:.2f}, {y1:.2f}]"
                                      "<extra></extra>"), row=i, col=1)
        fig.add_vline(x=today_s, row=i, col=1,
                      line=dict(color=TODAY_COLOR, width=style.refline, dash=dot_dash(style.refline)))
        _tier_labels(fig, hs, i, "y" if i == 1 else f"y{i}", ylim, style, labels)
        fig.update_yaxes(title_text=y_label or text["ability"], range=list(ylim), gridcolor="#eeeeee",
                         zeroline=False, row=i, col=1)
        x_min = dates.min() if x_min is None else min(x_min, dates.min())
        x_max = gx.max() if x_max is None else max(x_max, gx.max())
    if window is None:
        window = ((x_min - pd.Timedelta(days=90)).strftime("%Y-%m-%d"), TREND_WINDOW_END)
    fig.update_xaxes(range=list(window), dtick="M12", tickformat="%Y", gridcolor="#f4f4f4",
                     showticklabels=True)
    fig.update_xaxes(title_text=text["release"], row=len(axes), col=1)
    fig.update_layout(showlegend=False, template="plotly_white", width=style.width,
                      height=style.height_per_row * len(axes) + (100 if title else 60),
                      margin=dict(l=int(style.font_tick * 5), r=int(style.font_tier * 21),
                                  t=100 if title else 70, b=int(style.font_tick * 4)))
    if title:
        fig.update_layout(title=dict(text=title, x=0.5))
    return apply_fonts(fig, style)


def crossover_panels_fig(cx: pd.DataFrame, axes: list[str], titles: dict | None = None, *,
                         probs: tuple[float, ...] = (0.5,), style: FigureStyle = DASHBOARD,
                         window: tuple[str, str] | None = CROSSOVER_WINDOW, today=None,
                         lang: str = "en", human_labels: dict | None = None,
                         title: str | None = None) -> go.Figure:
    """Crossing dates per axis and human tier, one stacked panel per axis.

    `cx` is `analysis.mirt_crossover_df` output for every axis (columns axis, tier, human_mean,
    crossover_date_median, crossover_hdi_low, crossover_hdi_high, and hdi80_low / hdi80_high
    when a second, wider mass was computed). `probs` names the bars, widest last: one value
    draws one thick bar, two draw the thick first-mass bar over the thin second-mass bar. Every
    bar is split at the today line, green behind, red ahead. `window` fixes the x-range, 2015
    to 2030 by default (`CROSSOVER_WINDOW`) so every crossover figure is comparable; None spans
    the data. Whatever runs past the window is clipped at
    the edge, marked by a small dot with the true year above it.
    """
    text = FORECAST_TEXT[lang]
    labels = HUMAN_LEVEL_LABELS[lang] if human_labels is None else human_labels
    titles = titles or {}
    today_d = _today(today)
    cx = cx[cx["axis"].isin(axes) & cx["crossover_date_median"].notna()].copy()
    for c in ("crossover_date_median", "crossover_hdi_low", "crossover_hdi_high",
              "hdi80_low", "hdi80_high"):
        if c in cx:
            cx[c] = pd.to_datetime(cx[c])
    wide = len(probs) > 1 and "hdi80_low" in cx
    if window is None:
        lo_c, hi_c = ("hdi80_low", "hdi80_high") if wide else ("crossover_hdi_low",
                                                                "crossover_hdi_high")
        x0d = max(cx[lo_c].min(), pd.Timestamp("2015-01-01")).normalize()
        x1d = min(cx[hi_c].max(), pd.Timestamp("2032-01-01")).normalize()
    else:
        x0d, x1d = pd.Timestamp(window[0]), pd.Timestamp(window[1])
    pad = pd.Timedelta(days=160)
    x0, x1 = x0d - pad, x1d + pad

    panel_titles = None if (title and len(axes) == 1) else [titles.get(a, a) for a in axes]
    fig = make_subplots(rows=len(axes), cols=1, shared_xaxes=True, vertical_spacing=0.075,
                        subplot_titles=panel_titles)
    fig.update_annotations(font_size=style.font_axis)          # panel titles
    bars = [("crossover_hdi_low", "crossover_hdi_high", style.bar_thick)]
    if wide:
        bars = [("hdi80_low", "hdi80_high", style.bar_thin)] + bars
    clipped = False
    for i, name in enumerate(axes, start=1):
        rows = cx[cx["axis"] == name].sort_values("human_mean")
        tiers = [labels.get(t, t) for t in rows["tier"]]
        rows = rows.assign(_label=tiers)
        segs: dict = {(w, c): ([], []) for _, _, w in bars for c in (PASSED_COLOR, FUTURE_COLOR)}
        for lo_c, hi_c, w in bars:
            for _, r in rows.iterrows():
                start, end = max(r[lo_c], x0d), min(r[hi_c], x1d)
                if start > end:
                    continue
                pieces = []
                if start < today_d:
                    pieces.append((start, min(end, today_d), PASSED_COLOR))
                if end > today_d:
                    pieces.append((max(start, today_d), end, FUTURE_COLOR))
                for s, e, c in pieces:
                    sx, sy = segs[(w, c)]
                    sx += [s.isoformat(), e.isoformat(), None]
                    sy += [r["_label"], r["_label"], None]
        for (w, c), (sx, sy) in segs.items():
            if sx:
                fig.add_trace(go.Scatter(x=sx, y=sy, mode="lines", line=dict(color=c, width=w),
                                         opacity=0.85, showlegend=False, hoverinfo="skip"),
                              row=i, col=1)
        med = rows["crossover_date_median"]
        m_in = rows[(med >= x0d) & (med <= x1d)]
        for col, grp in ((PASSED_COLOR, m_in[m_in["crossover_date_median"] <= today_d]),
                         (FUTURE_COLOR, m_in[m_in["crossover_date_median"] > today_d])):
            if grp.empty:
                continue
            fig.add_trace(go.Scatter(
                x=grp["crossover_date_median"].dt.strftime("%Y-%m-%d"), y=grp["_label"],
                mode="markers", marker=dict(color=col, size=style.marker_median),
                showlegend=False,
                hovertemplate="%{y}<br>" + text["median"] + " %{x|%Y-%m-%d}<extra></extra>"),
                row=i, col=1)
        lo_c, hi_c = bars[0][0], bars[0][1]
        for _, r in rows.iterrows():
            m = r["crossover_date_median"]
            for side, edge in (("left", x0d), ("right", x1d)):
                over = (r[lo_c] < edge or m < edge) if side == "left" else (r[hi_c] > edge
                                                                            or m > edge)
                if not over:
                    continue
                true = (m if (m < edge if side == "left" else m > edge)
                        else (r[lo_c] if side == "left" else r[hi_c]))
                col = PASSED_COLOR if edge <= today_d else FUTURE_COLOR
                clipped = True
                fig.add_trace(go.Scatter(
                    x=[edge.isoformat()], y=[r["_label"]], mode="markers+text",
                    marker=dict(color=col, size=max(style.marker_median // 2, 5)),
                    text=[f"{true.year}"], textposition="top right" if side == "left"
                    else "top left", textfont=dict(size=style.font_note, color=col),
                    showlegend=False, hoverinfo="skip"), row=i, col=1)
        fig.add_vline(x=today_d.strftime("%Y-%m-%d"), row=i, col=1,
                      line=dict(color=TODAY_COLOR, width=style.refline, dash=dot_dash(style.refline)))
        fig.update_yaxes(categoryorder="array", categoryarray=tiers, showgrid=False,
                         range=[-0.7, len(tiers) - 0.3], row=i, col=1)
        fig.update_xaxes(range=[x0.strftime("%Y-%m-%d"), x1.strftime("%Y-%m-%d")],
                         tickformat="%Y", dtick="M12", gridcolor="#e9e9e9",
                         showticklabels=True, row=i, col=1)

    def proxy(rank, name, **kw):
        fig.add_trace(go.Scatter(x=[None], y=[None], name=name, legendrank=rank,
                                 showlegend=True, **kw), row=1, col=1)

    proxy(1, text["behind"], mode="lines", line=dict(color=PASSED_COLOR, width=style.bar_thick))
    proxy(2, text["ahead"], mode="lines", line=dict(color=FUTURE_COLOR, width=style.bar_thick))
    proxy(3, text["median"], mode="markers", marker=dict(color="#888", size=style.marker_median))
    if wide:
        proxy(4, text["thick"].format(p=probs[0]), mode="lines",
              line=dict(color="#888", width=style.bar_thick))
        proxy(5, text["thin"].format(p=probs[1]), mode="lines",
              line=dict(color="#888", width=style.bar_thin))
    else:
        proxy(4, text["interval"].format(p=probs[0]), mode="lines",
              line=dict(color="#888", width=style.bar_thick))
    proxy(6, text["today"], mode="lines",
          line=dict(color=TODAY_COLOR, width=style.refline, dash=dot_dash(style.refline)))
    if clipped:
        proxy(7, text["clipped"], mode="markers", marker=dict(color="#888", size=9))
    fig.update_xaxes(title_text=text["crossing_axis"], title_font_size=style.font_tick,
                     row=len(axes), col=1)
    fig.update_layout(
        template="plotly_white", width=style.width,
        height=int(style.height_per_row * 0.92) * len(axes) + (160 if title else 120),
        legend=dict(orientation="h", yanchor="top", y=-0.05 * max(1.0, 3 / len(axes)),
                    xanchor="center", x=0.5, bgcolor="rgba(0,0,0,0)", borderwidth=0),
        margin=dict(l=int(style.font_tick * 17), r=int(style.font_tick * 3),
                    t=120 if title else 80, b=int(style.font_legend * 8)))
    if title:
        fig.update_layout(title=dict(text=title, x=0.5))
    return apply_fonts(fig, style)
