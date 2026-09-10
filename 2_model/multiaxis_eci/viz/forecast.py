"""Frontier-forecast figures: the trend panels, the crossover panels, the exceedance curves.

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

from multiaxis_eci.viz.core import (
    FUTURE_COLOR,
    HUMAN_LEVEL_LABELS,
    MODEL_COLOR,
    PASSED_COLOR,
    _rgba,
    human_tier_palette,
)
from multiaxis_eci.viz.style import DASHBOARD, FigureStyle, apply_fonts

FORECAST_COLOR = "#ff9500"        # frontier extrapolation
TODAY_COLOR = "#444"

# Every string these figures draw, per language (English default, French for the
# `fr/` renders); tier names come from core.HUMAN_LEVEL_LABELS.
FORECAST_TEXT = {
    "en": {"trend": "Projected frontier", "release": "Release date", "ability": "ability",
           "crossing_axis": "Crossing date", "behind": "already behind us",
           "ahead": "still ahead", "median": "median", "interval": "{p:.0%} interval",
           "thick": "{p:.0%} interval (thick)", "thin": "{p:.0%} interval (thin)",
           "today": "today", "clipped": "continues past the window (date shown)",
           "decisive": "0.975 (decisive)", "p_axis": "P(frontier > tier)"},
    "fr": {"trend": "Frontière projetée", "release": "Date de sortie", "ability": "capacité",
           "crossing_axis": "Date de croisement", "behind": "déjà derrière nous",
           "ahead": "encore à venir", "median": "médiane", "interval": "intervalle à {p:.0%}",
           "thick": "intervalle à {p:.0%} (épais)", "thin": "intervalle à {p:.0%} (fin)",
           "today": "aujourd'hui", "clipped": "dépasse la fenêtre (date indiquée)",
           "decisive": "0.975 (décisif)", "p_axis": "P(frontière > niveau)"},
}


def _today(today) -> pd.Timestamp:
    return pd.Timestamp(today) if today is not None else pd.Timestamp.today().normalize()


def _tier_labels(fig, hs: pd.DataFrame, row: int, yref: str, ylim: tuple[float, float],
                 style: FigureStyle, labels: dict, gap_frac: float = 0.066) -> None:
    """Dashed tier lines in Blues (strongest darkest) plus their names in the right margin.

    Names are nudged apart top-down against the row's own y range, then the stack is centred
    on the tier block and kept inside the row, so a name can sit a little off its line where
    tiers crowd.
    """
    rows = hs.sort_values("mean", ascending=False).reset_index(drop=True)
    colors = human_tier_palette(len(rows))
    span = ylim[1] - ylim[0]
    gap = gap_frac * span
    ys, prev = [], np.inf
    for lvl in rows["mean"]:
        y = min(float(lvl), prev - gap)
        ys.append(y)
        prev = y
    shift = float(np.mean(rows["mean"])) - float(np.mean(ys))
    shift = min(shift, (ylim[1] - 0.02 * span) - ys[0])
    shift = max(shift, (ylim[0] + 0.02 * span) - ys[-1])
    ys = [y + shift for y in ys]
    for (_, r), y, col in zip(rows.iterrows(), ys, colors):
        fig.add_hline(y=float(r["mean"]), row=row, col=1,
                      line=dict(color=col, width=style.refline, dash="dash"), opacity=0.75)
        fig.add_annotation(x=1.005, y=y, xref="paper", yref=yref,
                           text=labels.get(r["name"], r["name"]), showarrow=False,
                           xanchor="left", font=dict(size=style.font_tier, color=col))


def frontier_trend_fig(per_axis: dict, axes: list[str], titles: dict | None = None, *,
                       style: FigureStyle = DASHBOARD, window: tuple[str, str] | None = None,
                       today=None, lang: str = "en", human_labels: dict | None = None,
                       title: str | None = None) -> go.Figure:
    """The frontier trend per axis, one stacked panel each.

    `per_axis[name]` holds `fc` (a ForecastResult: grid_dates, lo, median, hi, slope), `tl`
    (the candidates' timeline frame: release_date, mean, hdi_low, hdi_high, name) and `hs` (the
    human tiers: name, mean). Each panel draws the dated models with their intervals, the
    forecast band (fc.lo to fc.hi) and its median, the tiers as dashed lines named in the
    right margin, and the today line; no legend, the caption names the series. `window` fixes
    the x-range on every panel (the post uses 2023 to 2030); None spans the data.
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
        dates = pd.to_datetime(tl["release_date"])
        fig.add_trace(go.Scatter(
            x=dates, y=tl["mean"], mode="markers",
            marker=dict(color=MODEL_COLOR, size=style.marker, opacity=0.55, line=dict(width=0)),
            error_y=dict(type="data", symmetric=False,
                         array=tl["hdi_high"] - tl["mean"], arrayminus=tl["mean"] - tl["hdi_low"],
                         thickness=style.errbar, width=0, color="rgba(32,163,158,0.35)"),
            text=tl["name"], showlegend=False,
            hovertemplate="%{text}<br>%{x|%Y-%m-%d}: %{y:.2f}<extra></extra>"), row=i, col=1)
        fig.add_trace(go.Scatter(x=gx, y=fc.median, mode="lines", showlegend=False,
                                 name=text["trend"],
                                 line=dict(color=FORECAST_COLOR, width=style.trend, dash="dash"),
                                 hovertemplate="%{x|%Y-%m}: %{y:.2f}<extra></extra>"),
                      row=i, col=1)
        lo = min(float(tl["hdi_low"].min()), float(np.min(fc.lo)), float(hs["mean"].min()))
        hi = max(float(tl["hdi_high"].max()), float(np.max(fc.hi)), float(hs["mean"].max()))
        pad = 0.06 * (hi - lo)
        ylim = (lo - pad, hi + pad)
        fig.add_vline(x=today_s, row=i, col=1,
                      line=dict(color=TODAY_COLOR, width=style.refline, dash="dot"))
        _tier_labels(fig, hs, i, "y" if i == 1 else f"y{i}", ylim, style, labels)
        fig.update_yaxes(title_text=text["ability"], range=list(ylim), gridcolor="#eeeeee",
                         zeroline=False, row=i, col=1)
        x_min = dates.min() if x_min is None else min(x_min, dates.min())
        x_max = gx.max() if x_max is None else max(x_max, gx.max())
    if window is None:
        window = ((x_min - pd.Timedelta(days=90)).strftime("%Y-%m-%d"),
                  (x_max + pd.Timedelta(days=30)).strftime("%Y-%m-%d"))
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
                         window: tuple[str, str] | None = None, today=None, lang: str = "en",
                         human_labels: dict | None = None, title: str | None = None) -> go.Figure:
    """Crossing dates per axis and human tier, one stacked panel per axis.

    `cx` is `analysis.mirt_crossover_df` output for every axis (columns axis, tier, human_mean,
    crossover_date_median, crossover_hdi_low, crossover_hdi_high, and hdi80_low / hdi80_high
    when a second, wider mass was computed). `probs` names the bars, widest last: one value
    draws one thick bar, two draw the thick first-mass bar over the thin second-mass bar. Every
    bar is split at the today line, green behind, red ahead. `window` fixes the x-range (the
    post uses 2015 to 2030); None spans the data. Whatever runs past the window is clipped at
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
                      line=dict(color=TODAY_COLOR, width=style.refline, dash="dot"))
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
          line=dict(color=TODAY_COLOR, width=style.refline, dash="dot"))
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


def exceedance_prob_fig(fc, theta_draws, k: int, data, *, axis_name: str,
                        human_labels: dict | None = None, lang: str = "en",
                        style: FigureStyle = DASHBOARD) -> go.Figure:
    """P(frontier > tier) over the forecast grid — one S-curve per human tier,
    with reference lines at 0.5 and 0.975 (decisive).

    `human_labels` defaults to the `lang` tier names (English = raw names)."""
    text = FORECAST_TEXT[lang]
    labels = HUMAN_LEVEL_LABELS[lang] if human_labels is None else human_labels
    from multiaxis_eci.analysis.forecast import _to_year, frontier_paths

    xg = _to_year(fc.grid_dates)
    # Through frontier_paths, NOT intercept + slope * t: an envelope result's
    # line is only valid beyond the last record — evaluated backward it would
    # extrapolate the recent rate over the observed window and misprice every
    # historical exceedance probability.
    f = frontier_paths(fc, xg)                                      # (S, G)
    gx = pd.to_datetime(fc.grid_dates).strftime("%Y-%m-%d")
    names = data.mlookup.sort_values("model_idx")["model"].tolist()
    humans = [(i, m) for i, m in enumerate(names) if data.is_human[i]]
    humans.sort(key=lambda im: theta_draws[:, im[0], k].mean())
    palette = human_tier_palette(len(humans))[::-1]

    fig = go.Figure()
    for (i, m), col in zip(humans, palette):
        th = theta_draws[:, i, k]
        p = (f > th[:, None]).mean(0)                               # (G,)
        fig.add_trace(go.Scatter(
            x=gx, y=p, mode="lines", line=dict(color=col, width=style.trend),
            name=labels.get(m, m),
            hovertemplate="P = %{y:.2f}<br>%{x|%Y-%m}<extra></extra>"))
    for yv, lab in [(0.5, "0.5"), (0.975, text["decisive"])]:
        fig.add_hline(y=yv, line=dict(color="#888", dash="dot", width=style.refline),
                      annotation_text=lab, annotation_position="right")
    fig.update_layout(
        title=dict(text=f"Forecast: {axis_name} (P exceed human)", x=0.5),
        xaxis=dict(type="date", title="Date", showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
        yaxis=dict(title=text["p_axis"], range=[0, 1],
                   showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
        template="plotly_white", height=style.height_per_row + 60, width=style.width,
        margin=dict(l=70, r=int(style.font_legend * 22), t=80, b=55),
        legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=1.01,
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="rgba(0,0,0,0.1)", borderwidth=1))
    return apply_fonts(fig, style)
