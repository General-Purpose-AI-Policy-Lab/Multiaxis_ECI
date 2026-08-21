"""Frontier-forecast figures: extrapolated timeline, crossover
dot-whisker, exceedance probability."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from viz.core import HUMAN_LEVEL_LABELS_FR, capability_timeline_fig

# ── Frontier forecasting figures ────────────────────────────────────────────

FORECAST_COLOR = "#ff9500"        # frontier extrapolation
_PASSED_COLOR  = "#2ca02c"        # tier already surpassed
_FUTURE_COLOR  = "#d62728"        # tier not yet reached


def _crossover_color(status: str) -> str:
    return _PASSED_COLOR if status.startswith("passed") else _FUTURE_COLOR


def capability_forecast_fig(timeline_df, human_stats, fc, crossover_df,
                            *, axis_name: str,
                            fit_points=None) -> go.Figure:
    """Timeline (points + human bands) with the frontier forecast band and a
    dashed vertical marker at each tier's projected crossover date. Built on top
    of `capability_timeline_fig`, so styling and legend are inherited verbatim.

    `fit_points` (optional timeline-schema DataFrame) marks the record-setters
    the trend was actually regressed on. The fit cap (SD < 0.4) is looser than
    the measured cloud's (SD < 0.3), so a fitted record can be absent from the
    cloud; the markers keep every fitted point visible."""
    fig = capability_timeline_fig(timeline_df, human_stats=human_stats)
    gx = pd.to_datetime(fc.grid_dates).strftime("%Y-%m-%d")

    fig.add_trace(go.Scatter(                                       # HDI band
        x=list(gx) + list(gx[::-1]),
        y=list(fc.hi) + list(fc.lo[::-1]),
        fill="toself", fillcolor="rgba(255,149,0,0.12)",
        line=dict(width=0), hoverinfo="skip", showlegend=False))
    line_name = ("Tendance projetée (tous modèles informés)"
                 if getattr(fc, "fit_basis", "frontier") == "informed"
                 else "Frontière projetée")
    fig.add_trace(go.Scatter(                                       # median line
        x=gx, y=fc.median, mode="lines",
        line=dict(color=FORECAST_COLOR, dash="dash", width=2.2),
        name=line_name,
        hovertemplate="tendance ≈ %{y:.2f}<br>%{x|%Y-%m}<extra></extra>"))

    for _, r in crossover_df.iterrows():
        d = r["crossover_date_median"]
        if pd.isna(d):
            continue
        passed = r["status"].startswith("passed")
        col = _crossover_color(r["status"])
        ts = pd.Timestamp(d)
        # strftime string (not a Timestamp) so kaleido can serialize the shape.
        # Label the line at the x-axis with the EXACT projected crossover date,
        # written vertically so adjacent markers don't collide.
        fig.add_vline(
            x=ts.strftime("%Y-%m-%d"),
            line=dict(color=col, dash="dot" if passed else "dash", width=1.2),
            annotation_text=ts.strftime("%Y-%m-%d"),
            annotation_position="bottom",
            annotation_textangle=-90,
            annotation_yanchor="top",
            annotation_font_size=9,
            annotation_font_color=col)

    # Extend x to the horizon (and to any crossover markers), capped at 2032.
    xmax = pd.to_datetime(fc.grid_dates).max()
    valid = crossover_df["crossover_date_median"].dropna()
    if len(valid):
        xmax = max(xmax, pd.to_datetime(valid).max())
    xmax = min(xmax, pd.Timestamp("2032-01-01"))
    xmin = pd.to_datetime(timeline_df["release_date"]).min()
    fig.update_xaxes(range=[xmin.strftime("%Y-%m-%d"), xmax.strftime("%Y-%m-%d")])
    # Pin y to the data (model whiskers + human bands), never to the forecast
    # band: a weakly identified slope gives a band tens of logits wide, and
    # autoscaling to it collapses every dot and tier into a flat stripe. The
    # band is clipped instead — its informative part is where the data live.
    ylo = timeline_df["hdi_low"].min()
    yhi = timeline_df["hdi_high"].max()
    if human_stats is not None and len(human_stats):
        ylo = min(ylo, human_stats["hdi_low"].min())
        yhi = max(yhi, human_stats["hdi_high"].max())
    pad = 0.06 * (yhi - ylo)
    fig.update_yaxes(range=[ylo - pad, yhi + pad])
    fig.update_layout(title=dict(
        text=f"Forecast — {axis_name}", x=0.5))
    return fig


def crossover_dotwhisker_fig(crossover_df, *, axis_name: str) -> go.Figure:
    """When the frontier is projected to reach each human tier: tier on Y,
    crossover date + 94% CI on X, coloured passed (vert) vs future (rouge)."""
    fig = go.Figure()
    for _, r in crossover_df.iterrows():
        label = HUMAN_LEVEL_LABELS_FR.get(r["tier"], r["tier"])
        if pd.isna(r["crossover_date_median"]):
            fig.add_trace(go.Scatter(
                x=[None], y=[label], mode="markers",
                marker=dict(color="#999", symbol="x"),
                name="pas de croisement projeté", showlegend=False,
                hovertemplate=f"{label} : pente ≤ 0<extra></extra>"))
            continue
        color = _crossover_color(r["status"])
        lo = pd.Timestamp(r["crossover_hdi_low"]).strftime("%Y-%m-%d")
        hi = pd.Timestamp(r["crossover_hdi_high"]).strftime("%Y-%m-%d")
        med = pd.Timestamp(r["crossover_date_median"]).strftime("%Y-%m-%d")
        fig.add_trace(go.Scatter(x=[lo, hi], y=[label, label], mode="lines",
                                 line=dict(color=color, width=2),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            x=[med], y=[label], mode="markers+text",
            marker=dict(color=color, size=10),
            text=[pd.Timestamp(r["crossover_date_median"]).strftime("%Y-%m-%d")],
            textposition="top center",
            textfont=dict(size=10, color=color),
            showlegend=False,
            hovertemplate=(f"{label}<br>médiane %{{x|%Y-%m}}"
                           f"<br>P(dépassé aujourd'hui) = {r['p_passed_now']:.2f}"
                           "<extra></extra>")))
    fig.add_vline(x=pd.Timestamp.today().normalize().strftime("%Y-%m-%d"),
                  line=dict(color="#444", dash="dot", width=1),
                  annotation_text="aujourd'hui", annotation_position="top")
    fig.update_layout(
        title=dict(text=f"Forecast — {axis_name} (crossover dates)", x=0.5),
        xaxis=dict(type="date", title="Date de croisement projetée",
                   showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
        yaxis=dict(title="", showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
        template="plotly_white", height=460, width=1000,
        margin=dict(l=230, r=60, t=80, b=55))
    return fig


def exceedance_prob_fig(fc, theta_draws, k: int, data, *, axis_name: str) -> go.Figure:
    """P(frontier > tier) over the forecast grid — one S-curve per human tier,
    with reference lines at 0.5 and 0.975 (decisive)."""
    from analysis.forecast import _to_year

    xg = _to_year(fc.grid_dates)
    f = fc.intercept[:, None] + fc.slope[:, None] * xg[None, :]     # (S, G)
    gx = pd.to_datetime(fc.grid_dates).strftime("%Y-%m-%d")
    names = data.mlookup.sort_values("model_idx")["model"].tolist()

    import plotly.colors as pc
    humans = [(i, m) for i, m in enumerate(names) if data.is_human[i]]
    humans.sort(key=lambda im: theta_draws[:, im[0], k].mean())
    palette = [pc.sample_colorscale("Blues", float(p))[0]
               for p in np.linspace(0.35, 0.92, len(humans) or 1)]

    fig = go.Figure()
    for (i, m), col in zip(humans, palette):
        th = theta_draws[:, i, k]
        p = (f > th[:, None]).mean(0)                               # (G,)
        fig.add_trace(go.Scatter(
            x=gx, y=p, mode="lines", line=dict(color=col, width=2),
            name=HUMAN_LEVEL_LABELS_FR.get(m, m),
            hovertemplate="P = %{y:.2f}<br>%{x|%Y-%m}<extra></extra>"))
    for yv, lab in [(0.5, "0.5"), (0.975, "0.975 (décisif)")]:
        fig.add_hline(y=yv, line=dict(color="#888", dash="dot", width=1),
                      annotation_text=lab, annotation_position="right")
    fig.update_layout(
        title=dict(text=f"Forecast — {axis_name} (P exceed human)", x=0.5),
        xaxis=dict(type="date", title="Date", showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
        yaxis=dict(title="P(frontière > niveau)", range=[0, 1],
                   showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
        template="plotly_white", height=520, width=1100,
        margin=dict(l=70, r=260, t=80, b=55),
        legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=1.01,
                    bgcolor="rgba(255,255,255,0.85)",
                    bordercolor="rgba(0,0,0,0.1)", borderwidth=1))
    return fig


