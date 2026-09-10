"""Side-by-side diagnostics of two chain groups of one fit (majority against minority).

A multimodal posterior splits its chains into groups that disagree on some part of the
solution. These figures put the two groups on the same panels so the disagreement can be
located: which chains sit in which log-density basin, which benchmarks load differently,
whether the models' abilities or only the human tiers move, and how far the projected
crossing dates drift. Both groups are expected in the fit's display frame
(`analysis.align_to_reference_loadings`), so axis k is the same axis on every panel.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from multiaxis_eci.viz.core import MODEL_COLOR
from multiaxis_eci.viz.forecast import CROSSOVER_WINDOW
from multiaxis_eci.viz.style import DASHBOARD, FigureStyle, apply_fonts

# Okabe-Ito blue and vermillion: the two chain groups, on every figure of this module.
GROUP_COLORS = ("#0072B2", "#D55E00")
HUMAN_COLOR = "#e69f00"


def _grid(n: int, titles: list[str], ncols: int = 2, **kw):
    nrows = -(-n // ncols)
    return make_subplots(rows=nrows, cols=ncols, subplot_titles=titles, **kw), nrows


def chain_logp_fig(modes_doc: dict, split: dict, *, style: FigureStyle = DASHBOARD) -> go.Figure:
    """Mean log-density of every chain relative to the best mode, coloured by group.

    The bar text names the mode `diagnose_chains` assigned the chain to. A group whose bars
    sit many nats below the other is a worse basin, not just a different axis solution.
    """
    delta = modes_doc.get("chain_delta_logp") or []
    mode_of = {c: m["label"] for m in modes_doc.get("modes", []) for c in m["chains"]}
    chains = list(range(len(delta)))
    colors = [GROUP_COLORS[0] if c in split["majority"] else GROUP_COLORS[1] for c in chains]
    fig = go.Figure(go.Bar(
        x=[str(c) for c in chains], y=delta, marker_color=colors,
        text=[f"mode {mode_of.get(c, '?')}" for c in chains], textposition="outside",
        hovertemplate="chain %{x}<br>Δ logp %{y:.1f}<extra></extra>", showlegend=False))
    for color, lab in zip(GROUP_COLORS, ("majority chains", "minority chains")):
        fig.add_trace(go.Bar(x=[None], y=[None], marker_color=color, name=lab))
    fig.update_layout(
        title=dict(text="Chain mean log-density below the best mode", x=0.5),
        xaxis_title="chain", yaxis_title="Δ mean logp (nats)", template="plotly_white",
        width=style.width, height=style.height_per_row, barmode="overlay",
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5))
    return apply_fonts(fig, style)


def loadings_compare_fig(A_a, A_b, bench: list[str], names: list[str], titles: dict | None = None,
                         labels=("majority", "minority"), top_n: int = 15, *,
                         style: FigureStyle = DASHBOARD) -> go.Figure:
    """Per axis, the `top_n` benchmarks by axis share (mean of the two groups): the median
    loading with its 95% interval in each group, one row per benchmark, dumbbell style.
    Rows where the two markers sit far apart are where the groups disagree on the axis."""
    titles = titles or {}
    med_a, med_b = np.median(A_a, 0), np.median(A_b, 0)
    share = np.zeros_like(med_a)
    for med in (med_a, med_b):
        share += med ** 2 / np.maximum((med ** 2).sum(1, keepdims=True), 1e-12) / 2
    fig, nrows = _grid(len(names), [titles.get(n, n) for n in names], horizontal_spacing=0.3,
                       vertical_spacing=0.12)
    fig.update_annotations(font_size=style.font_axis)
    for k in range(len(names)):
        row, col = k // 2 + 1, k % 2 + 1
        top = np.argsort(-share[:, k])[:top_n][::-1]
        ys = [bench[b] for b in top]
        for i, (A, lab, color) in enumerate(zip((A_a, A_b), labels, GROUP_COLORS)):
            med = np.median(A[:, top, k], 0)
            lo, hi = np.percentile(A[:, top, k], [2.5, 97.5], axis=0)
            fig.add_trace(go.Scatter(
                x=med, y=ys, mode="markers", name=lab, legendgroup=lab, showlegend=(k == 0),
                marker=dict(color=color, size=style.marker + 1, symbol="circle" if i == 0
                            else "diamond"),
                error_x=dict(type="data", symmetric=False, array=hi - med, arrayminus=med - lo,
                             color=color, thickness=style.errbar, width=0),
                hovertemplate="<b>%{y}</b><br>" + lab + " %{x:.2f}<extra></extra>"),
                row=row, col=col)
        for y, b in zip(ys, top):
            fig.add_trace(go.Scatter(x=[med_a[b, k], med_b[b, k]], y=[y, y], mode="lines",
                                     line=dict(color="#999", width=style.refline),
                                     showlegend=False, hoverinfo="skip"), row=row, col=col)
        fig.update_yaxes(categoryorder="array", categoryarray=ys, tickmode="array", tickvals=ys,
                         automargin=True, row=row, col=col)
        fig.update_xaxes(title_text="loading (median, 95% interval)", zeroline=True,
                         zerolinecolor="#222", row=row, col=col)
    fig.update_layout(
        title=dict(text=f"Loadings by chain group ({labels[0]} against {labels[1]})", x=0.5),
        template="plotly_white", width=style.width,
        height=nrows * max(int(style.height_per_row * 0.9), int(style.font_tick * 1.9) * top_n)
        + 160,
        legend=dict(orientation="h", yanchor="top", y=-0.05, xanchor="center", x=0.5),
        margin=dict(l=60, r=40, t=110, b=110))
    return apply_fonts(fig, style)


def abilities_compare_fig(theta_a, theta_b, data, names: list[str], titles: dict | None = None,
                          labels=("majority", "minority"), *,
                          style: FigureStyle = DASHBOARD) -> go.Figure:
    """Per axis, every test-taker's median ability in group A against group B, models as
    dots and human tiers as squares, with the identity line and the correlation. A cloud on
    the line means the groups agree on the models; tiers off the line locate a disagreement
    that touches only the human ladder."""
    titles = titles or {}
    models = data.mlookup.sort_values("model_idx")["model"].tolist()
    is_h = np.asarray(data.is_human, bool)
    med_a, med_b = np.median(theta_a, 0), np.median(theta_b, 0)
    fig, nrows = _grid(len(names), [titles.get(n, n) for n in names], horizontal_spacing=0.12,
                       vertical_spacing=0.14)
    fig.update_annotations(font_size=style.font_axis)
    for k in range(len(names)):
        row, col = k // 2 + 1, k % 2 + 1
        r = float(np.corrcoef(med_a[:, k], med_b[:, k])[0, 1])
        lo = float(min(med_a[:, k].min(), med_b[:, k].min()))
        hi = float(max(med_a[:, k].max(), med_b[:, k].max()))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", showlegend=False,
                                 line=dict(color="#888", dash="dash", width=style.refline),
                                 hoverinfo="skip"), row=row, col=col)
        for mask, color, symbol, lab in ((~is_h, MODEL_COLOR, "circle", "models"),
                                         (is_h, HUMAN_COLOR, "square", "human tiers")):
            fig.add_trace(go.Scatter(
                x=med_a[mask, k], y=med_b[mask, k], mode="markers", name=lab, legendgroup=lab,
                showlegend=(k == 0), text=[m for m, keep in zip(models, mask) if keep],
                marker=dict(color=color, size=style.marker + (3 if lab == "human tiers" else 0),
                            symbol=symbol, opacity=0.75),
                hovertemplate="<b>%{text}</b><br>" + f"{labels[0]} %{{x:.2f}}, {labels[1]} "
                              "%{y:.2f}<extra></extra>"), row=row, col=col)
        fig.add_annotation(xref=f"x{'' if k == 0 else k + 1} domain",
                           yref=f"y{'' if k == 0 else k + 1} domain", x=0.02, y=0.98,
                           text=f"r = {r:.3f}", showarrow=False, xanchor="left",
                           font=dict(size=style.font_tick))
        fig.update_xaxes(title_text=f"ability, {labels[0]} chains", row=row, col=col)
        fig.update_yaxes(title_text=f"ability, {labels[1]} chains", row=row, col=col)
    fig.update_layout(
        title=dict(text="Median abilities by chain group", x=0.5), template="plotly_white",
        width=style.width, height=nrows * style.height_per_row + 140,
        legend=dict(orientation="h", yanchor="top", y=-0.06, xanchor="center", x=0.5),
        margin=dict(l=70, r=40, t=110, b=110))
    return apply_fonts(fig, style)


def human_tiers_compare_fig(theta_a, theta_b, data, names: list[str], titles: dict | None = None,
                            labels=("majority", "minority"), prob: float = 0.95, *,
                            style: FigureStyle = DASHBOARD) -> go.Figure:
    """Per axis, each human tier's ability in the two groups: median and central `prob`
    interval, the groups dodged on the row. The blog post's chain-mode figure, drawn for any
    fit: tiers that swap order or jump between groups are the multimodality's signature."""
    titles = titles or {}
    models = data.mlookup.sort_values("model_idx")["model"].tolist()
    rows_h = np.flatnonzero(np.asarray(data.is_human, bool))
    q = [(1 - prob) / 2, 0.5, (1 + prob) / 2]
    fig, nrows = _grid(len(names), [titles.get(n, n) for n in names], horizontal_spacing=0.3,
                       vertical_spacing=0.12)
    fig.update_annotations(font_size=style.font_axis)
    order = rows_h[np.argsort(np.median(theta_a[:, rows_h, 0], 0))]
    tiers = [models[i] for i in order]
    for k in range(len(names)):
        row, col = k // 2 + 1, k % 2 + 1
        for i, (theta, lab, color, off) in enumerate(zip((theta_a, theta_b), labels, GROUP_COLORS,
                                                          (0.15, -0.15))):
            lo, med, hi = np.quantile(theta[:, order, k], q, axis=0)
            fig.add_trace(go.Scatter(
                x=med, y=np.arange(len(order)) + off, mode="markers", name=lab, legendgroup=lab,
                showlegend=(k == 0), text=tiers,
                marker=dict(color=color, size=style.marker + 2, symbol="circle" if i == 0
                            else "diamond"),
                error_x=dict(type="data", symmetric=False, array=hi - med, arrayminus=med - lo,
                             color=color, thickness=style.errbar, width=0),
                hovertemplate="<b>%{text}</b><br>" + lab + " %{x:.2f}<extra></extra>"),
                row=row, col=col)
        fig.update_yaxes(tickmode="array", tickvals=list(range(len(order))), ticktext=tiers,
                         range=[-0.7, len(order) - 0.3], row=row, col=col)
        fig.update_xaxes(title_text=f"ability (median, {prob:.0%} interval)", row=row, col=col)
    fig.update_layout(
        title=dict(text="Human tiers by chain group", x=0.5), template="plotly_white",
        width=style.width, height=nrows * style.height_per_row + 140,
        legend=dict(orientation="h", yanchor="top", y=-0.06, xanchor="center", x=0.5),
        margin=dict(l=60, r=40, t=110, b=110))
    return apply_fonts(fig, style)


def crossover_compare_fig(cx_a: pd.DataFrame, cx_b: pd.DataFrame, axes: list[str],
                          titles: dict | None = None, labels=("majority", "minority"), *,
                          today=None, style: FigureStyle = DASHBOARD) -> go.Figure:
    """Per axis, each tier's median crossing date with its 50% interval in the two groups,
    dodged on the row, on the shared 2015 to 2030 window. `cx_a` / `cx_b` are
    `analysis.crossover_table` outputs over the axes (columns axis, tier, human_mean,
    crossover_date_median, crossover_hdi_low, crossover_hdi_high)."""
    titles = titles or {}
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.today().normalize()
    x0, x1 = (pd.Timestamp(CROSSOVER_WINDOW[0]) - pd.Timedelta(days=160),
              pd.Timestamp(CROSSOVER_WINDOW[1]) + pd.Timedelta(days=160))
    fig = make_subplots(rows=len(axes), cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=[titles.get(a, a) for a in axes])
    fig.update_annotations(font_size=style.font_axis)
    for i, name in enumerate(axes, start=1):
        ra = cx_a[cx_a["axis"] == name].sort_values("human_mean")
        tiers = list(ra["tier"])
        for cx, lab, color, off in zip((cx_a, cx_b), labels, GROUP_COLORS, (0.15, -0.15)):
            r = cx[cx["axis"] == name].set_index("tier").reindex(tiers)
            med = pd.to_datetime(r["crossover_date_median"])
            lo = pd.to_datetime(r["crossover_hdi_low"]).clip(lower=x0, upper=x1)
            hi = pd.to_datetime(r["crossover_hdi_high"]).clip(lower=x0, upper=x1)
            ok = med.notna()
            y = np.arange(len(tiers)) + off
            for yy, lo_, hi_ in zip(y[ok], lo[ok], hi[ok]):
                fig.add_trace(go.Scatter(x=[lo_, hi_], y=[yy, yy], mode="lines",
                                         line=dict(color=color, width=style.bar_thin),
                                         showlegend=False, hoverinfo="skip"), row=i, col=1)
            fig.add_trace(go.Scatter(
                x=med[ok].clip(lower=x0, upper=x1), y=y[ok], mode="markers", name=lab,
                legendgroup=lab, showlegend=(i == 1), text=[t for t, k in zip(tiers, ok) if k],
                marker=dict(color=color, size=style.marker_median),
                hovertemplate="<b>%{text}</b><br>" + lab + " %{x|%Y-%m-%d}<extra></extra>"),
                row=i, col=1)
        fig.add_vline(x=today.strftime("%Y-%m-%d"), row=i, col=1,
                      line=dict(color="#444", width=style.refline, dash="dot"))
        fig.update_yaxes(tickmode="array", tickvals=list(range(len(tiers))), ticktext=tiers,
                         range=[-0.7, len(tiers) - 0.3], row=i, col=1)
        fig.update_xaxes(range=[x0.strftime("%Y-%m-%d"), x1.strftime("%Y-%m-%d")],
                         tickformat="%Y", dtick="M12", showticklabels=True, row=i, col=1)
    fig.update_xaxes(title_text="Crossing date (median, 50% interval)", row=len(axes), col=1)
    fig.update_layout(
        title=dict(text="Projected crossings by chain group", x=0.5), template="plotly_white",
        width=style.width, height=int(style.height_per_row * 0.9) * len(axes) + 160,
        legend=dict(orientation="h", yanchor="top", y=-0.05 * max(1.0, 3 / len(axes)),
                    xanchor="center", x=0.5),
        margin=dict(l=int(style.font_tick * 17), r=40, t=110, b=110))
    return apply_fonts(fig, style)


def build_chain_group_figures(view_a, view_b, data, raw, titles: dict | None, modes_doc: dict,
                              split: dict, labels=("majority", "minority")) -> dict:
    """The comparison set for two aligned chain-group views: log-density per chain, loadings,
    abilities, human tiers and (when the fit has human tiers) the projected crossings."""
    from multiaxis_eci.analysis import axis_forecast_inputs, crossover_table

    names = list(view_a.names)
    bench = data.blookup.sort_values("benchmark_idx")["benchmark"].tolist()
    figs = {
        "chain_logp": chain_logp_fig(modes_doc, split),
        "loadings_compare": loadings_compare_fig(view_a.require_A(), view_b.require_A(), bench,
                                                 names, titles, labels),
        "abilities_compare": abilities_compare_fig(view_a.theta, view_b.theta, data, names,
                                                   titles, labels),
        "human_tiers_compare": human_tiers_compare_fig(view_a.theta, view_b.theta, data, names,
                                                       titles, labels),
    }
    if np.asarray(data.is_human, bool).any():
        tables = []
        for view in (view_a, view_b):
            parts = []
            for k, name in enumerate(names):
                try:
                    fc = axis_forecast_inputs(view.theta, k, data, raw, name, A_draws=view.A)["fc"]
                except ValueError:
                    continue
                parts.append(crossover_table(fc, view.theta, k, data, name, probs=(0.5,)))
            tables.append(pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
                columns=["axis", "tier", "human_mean", "crossover_date_median",
                         "crossover_hdi_low", "crossover_hdi_high"]))
        axes = [n for n in names if (tables[0]["axis"] == n).any()]
        if axes:
            figs["crossover_compare"] = crossover_compare_fig(tables[0], tables[1], axes, titles,
                                                              labels)
    return figs
