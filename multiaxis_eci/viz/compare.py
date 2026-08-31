"""Cross-fit comparison figures (cmp_* family) and the alignment-method
comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── cross-fit comparison builders ──
_CMP_TYPE_COLOR = {"baseline": "#888780", "exploratory": "#D85A30", "confirmed": "#1D9E75"}


def cmp_per_benchmark_rmse_fig(df: pd.DataFrame,
                               title: str = "Per-benchmark fit (RMSE) — where fits differ") -> go.Figure:
    """Benchmark × fit RMSE heatmap. `df`: index=benchmark, columns=fit label."""
    df = df.loc[df.mean(axis=1).sort_values().index]
    fig = go.Figure(go.Heatmap(z=df.values, x=df.columns.tolist(), y=df.index.tolist(),
                               colorscale="RdYlGn_r", colorbar=dict(title="RMSE")))
    fig.update_layout(title=dict(text=title, x=0.5), template="plotly_white",
                      width=780, height=max(600, 14 * len(df)), xaxis_tickangle=-25)
    return fig


def cmp_gof_fig(tab: pd.DataFrame, title: str = "Goodness of fit") -> go.Figure:
    """R² (left axis, zoomed) + RMSE & MAE (right axis) grouped bars."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(name="R²", x=tab["fit"], y=tab["R2"], marker_color="#2ca02c",
                         text=tab["R2"], textposition="outside"), secondary_y=False)
    fig.add_trace(go.Bar(name="RMSE", x=tab["fit"], y=tab["RMSE"], marker_color="#d62728"),
                  secondary_y=True)
    fig.add_trace(go.Bar(name="MAE", x=tab["fit"], y=tab["MAE"], marker_color="#ff7f0e"),
                  secondary_y=True)
    fig.update_yaxes(title_text="Bayesian R²", range=[float(tab["R2"].min()) - 0.01, 1.0],
                     secondary_y=False)
    fig.update_yaxes(title_text="RMSE / MAE", rangemode="tozero", secondary_y=True)
    fig.update_layout(barmode="group", title=dict(text=title, x=0.5),
                      template="plotly_white", height=480, width=900, xaxis_tickangle=-20)
    return fig


def cmp_convergence_fig(tab: pd.DataFrame,
                        title: str = "Convergence — identified r̂ (lower = better mixed)") -> go.Figure:
    """Identified r̂ per fit with the 1.01 line; divergence count on each bar."""
    fig = go.Figure(go.Bar(
        x=tab["fit"], y=tab["eta_rhat"],
        marker_color=np.where(tab["eta_rhat"] > 1.01, "#d62728", "#2ca02c"),
        text=[f"r̂={r:.2f}<br>{int(d)} div" for r, d in zip(tab["eta_rhat"], tab["divergences"])],
        textposition="outside"))
    fig.add_hline(y=1.01, line=dict(color="green", dash="dash"),
                  annotation_text="converged (r̂ = 1.01)", annotation_position="top left")
    fig.update_layout(title=dict(text=title, x=0.5), yaxis_title="identified r̂",
                      template="plotly_white", height=480, width=900, xaxis_tickangle=-20)
    return fig


def cmp_pit_ecdf_fig(results: list,
                     title: str = "Calibration — PIT ECDF vs perfect (hugs diagonal = calibrated)") -> go.Figure:
    """PIT ECDF overlay vs the diagonal. `results`: list of dicts with 'fit' + 'pit'."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                             line=dict(color="#888", dash="dash")))
    for r in results:
        p = np.sort(np.asarray(r["pit"]))
        ecdf = np.arange(1, len(p) + 1) / len(p)
        fig.add_trace(go.Scatter(x=p, y=ecdf, mode="lines", name=r["fit"], line=dict(width=1.6)))
    fig.update_layout(title=dict(text=title, x=0.5), xaxis_title="PIT",
                      yaxis_title="empirical CDF", template="plotly_white", height=600, width=760)
    return fig


def delpd_se(point_a, point_b):
    """SE of ΔELPD = ELPD(A) − ELPD(B) via the per-observation difference variance
    (accounts for the correlation between pointwise ELPDs on the same obs).

    Pointwise differencing only makes sense when both fits scored the SAME
    observations. Fits on different observation sets (e.g. the converged
    no-Skilled-Generalist fit: 3,704 obs vs 3,714) are not comparable this
    way — return NaN so the comparison shows an honest gap instead of
    crashing or silently mis-pairing rows."""
    a, b = np.asarray(point_a), np.asarray(point_b)
    if a.shape != b.shape:
        return float("nan"), float("nan")
    diff = a - b
    n = len(diff)
    return float(diff.sum()), float(np.sqrt(n * diff.var(ddof=1)))


def cmp_loo_waic_fig(results: list,
                     title: str = "Predictive accuracy comparison (higher = better, 0 = best model)",
                     note: str = None) -> go.Figure:
    """ΔELPD vs the best model, with the SE of the DIFFERENCE. `results`: dicts with
    'name', 'loo_elpd', 'loo_pointwise', 'waic_pointwise'. All fits passed here
    must share the SAME observations (ΔELPD is a sum over obs — see build_comparison,
    which filters to the modal obs group); `note` names any fits it had to omit."""
    sr = sorted(results, key=lambda r: r["loo_elpd"], reverse=True)
    best = sr[0]
    rows_loo, rows_waic = [], []
    for r in sr:
        d_loo, se_loo = delpd_se(r["loo_pointwise"], best["loo_pointwise"])
        d_waic, se_waic = delpd_se(r["waic_pointwise"], best["waic_pointwise"])
        rows_loo.append((r["name"], d_loo, se_loo))
        rows_waic.append((r["name"], d_waic, se_waic))
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                        subplot_titles=["PSIS-LOO (ΔELPD ± SE)", "WAIC (ΔELPD ± SE)"],
                        horizontal_spacing=0.08)
    names = [r[0] for r in rows_loo]
    y = list(range(len(names)))
    for col, rows, color in [(1, rows_loo, "#2c7fb8"), (2, rows_waic, "#d95f02")]:
        deltas = [r[1] for r in rows]
        ses = [r[2] for r in rows]
        fig.add_trace(go.Scatter(
            x=deltas, y=y, mode="markers",
            error_x=dict(type="data", array=ses, thickness=1.5, width=5),
            marker=dict(size=10, color=color),
            text=[f"{n}<br>ΔELPD = {d:+.1f} ± {s:.1f}" for n, d, s in zip(names, deltas, ses)],
            hovertemplate="%{text}<extra></extra>", showlegend=False), row=1, col=col)
        fig.add_vline(x=0, line=dict(color="#888", dash="dash"), row=1, col=col)
    fig.update_yaxes(tickvals=y, ticktext=names, row=1, col=1)
    fig.update_xaxes(title_text="ΔELPD (vs best)", row=1, col=1)
    fig.update_xaxes(title_text="ΔELPD (vs best)", row=1, col=2)
    if note:
        fig.add_annotation(text=note, xref="paper", yref="paper", x=0, y=-0.12,
                           showarrow=False, align="left", font=dict(size=11, color="#888"))
    fig.update_layout(title=dict(text=title, x=0.5), template="plotly_white",
                      height=max(380, 80 * len(names)) + (60 if note else 0), width=1000,
                      margin=dict(b=90 if note else 60))
    return fig


def cmp_pareto_k_fig(results: list,
                     title: str = "LOO reliability — Pareto-k per observation") -> go.Figure:
    """Stacked bar of Pareto-k categories per fit."""
    names = [r["name"] for r in results]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="reliable (k < 0.5)", x=names,
                         y=[r["pareto_k_good"] for r in results], marker_color="#2ca02c"))
    fig.add_trace(go.Bar(name="marginal (0.5 ≤ k < 0.7)", x=names,
                         y=[r["pareto_k_ok"] for r in results], marker_color="#ff7f0e"))
    fig.add_trace(go.Bar(name="unreliable (0.7 ≤ k < 1.0)", x=names,
                         y=[r["pareto_k_bad"] for r in results], marker_color="#d62728"))
    fig.add_trace(go.Bar(name="failed (k ≥ 1.0)", x=names,
                         y=[r["pareto_k_very_bad"] for r in results], marker_color="#7f0000"))
    fig.update_layout(barmode="stack", title=dict(text=title, x=0.5),
                      yaxis_title="number of observations", template="plotly_white",
                      height=460, width=900, xaxis_tickangle=-15,
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


def cmp_loo_vs_trust_fig(df: pd.DataFrame,
                         title: str = "Fit vs trust — best ELPD/R² ⇒ worst r̂/ESS") -> go.Figure:
    """Four panels (LOO ΔELPD, R², eta r̂, eta ESS) per fit. `df` sorted by
    loo_elpd ascending; columns: fit, type, loo_elpd, R2, eta_rhat, ess_min,
    ess_med, n_draws.

    The ESS panel carries the kept-draw count as a third marker: cards keep
    4,000-54,000 draws, so an absolute ESS is only readable against its own
    ceiling (the gap from draws to ESS is the autocorrelation)."""
    colors = [_CMP_TYPE_COLOR.get(t, "#888780") for t in df["type"]]
    y = df["fit"].tolist()
    fig = make_subplots(rows=1, cols=4, shared_yaxes=True, horizontal_spacing=0.06,
                        subplot_titles=["LOO ΔELPD ↑", "R² ↑", "eta r̂ ↓",
                                        "eta ESS ↑ vs draws (log)"])
    fig.add_trace(go.Bar(x=df["loo_elpd"] - df["loo_elpd"].max(), y=y, orientation="h",
                         marker_color=colors, showlegend=False), row=1, col=1)
    fig.add_trace(go.Bar(x=df["R2"], y=y, orientation="h", marker_color=colors, showlegend=False), row=1, col=2)
    fig.add_trace(go.Bar(x=df["eta_rhat"], y=y, orientation="h", marker_color=colors, showlegend=False), row=1, col=3)
    # ESS on a LOG axis: markers, not bars — horizontal bars don't render
    # reliably on a log scale (the high-ESS fits silently vanish).
    fig.add_trace(go.Scatter(x=df["n_draws"], y=y, mode="markers", name="draws kept",
                             marker=dict(color="#888780", size=13,
                                         symbol="line-ns-open",
                                         line=dict(width=2, color="#888780"))),
                  row=1, col=4)
    fig.add_trace(go.Scatter(x=df["ess_med"], y=y, mode="markers", name="ESS median",
                             marker=dict(color=colors, size=11, symbol="circle-open",
                                         line=dict(width=2))), row=1, col=4)
    fig.add_trace(go.Scatter(x=df["ess_min"], y=y, mode="markers", name="ESS min",
                             marker=dict(color=colors, size=11)), row=1, col=4)
    fig.add_vline(x=1.01, line=dict(color="#0F6E56", dash="dash"), row=1, col=3)
    fig.add_vline(x=100, line=dict(color="#0F6E56", dash="dash"), row=1, col=4)
    fig.update_xaxes(range=[df["R2"].min() - 0.01, 1.0], row=1, col=2)
    fig.update_xaxes(type="log", row=1, col=4)
    for t, c in _CMP_TYPE_COLOR.items():
        fig.add_trace(go.Bar(x=[None], y=[None], marker_color=c, name=t), row=1, col=1)
    fig.update_layout(title=dict(text=title, x=0.5), template="plotly_white",
                      height=440, width=1150,
                      legend=dict(orientation="h", y=1.04, x=1, xanchor="right"))
    return fig


def cmp_tau_spectrum_fig(taus: dict,
                         title: str = "Axis strength (normalised to axis 1)") -> go.Figure:
    """τ_A spectrum for exploratory fits. `taus`: dict (label, type) → sorted τ array."""
    fig = go.Figure()
    for (label, typ), t in taus.items():
        if typ != "exploratory":
            continue
        t = np.asarray(t)
        fig.add_trace(go.Scatter(x=[f"axis{i+1}" for i in range(len(t))], y=t / t[0],
                                 mode="lines+markers", name=label))
    fig.update_layout(title=dict(text=title, x=0.5), xaxis_title="axis (sorted)",
                      yaxis_title="strength (÷ axis 1)", template="plotly_white", height=440, width=720)
    return fig


_ALIGN_METHOD_COLORS = {"varimax": "#0C447C", "wop": "#e67e22",
                        "matchalign": "#27ae60", "promax": "#8e44ad",
                        "geomin": "#16a085"}


def alignment_methods_fig(load_df: pd.DataFrame, top_n: int = 10,
                          title: str = "Rotation methods compared — same trace, four "
                                       "independent post-hoc identifications") -> go.Figure:
    """WHERE the different rotations live for a signed fit: per axis, the mean
    aligned loading of the top benchmarks under each alignment method
    (varimax / WOP / MatchAlign / promax). Bars that agree across colors =
    the axis is data-driven, not an artifact of the rotation criterion; bars
    that disagree = orientation the data leave soft. `load_df` is the long
    `mirt_alignment_loadings_k{K}.csv` written by `3_diagnostics/align_mirt.py`
    (columns: method, axis, benchmark, loading_median, hdi_low, hdi_high)."""
    axes_names = sorted(load_df["axis"].unique())
    methods = [m for m in _ALIGN_METHOD_COLORS if m in set(load_df["method"])]
    fig = make_subplots(rows=1, cols=len(axes_names),
                        subplot_titles=axes_names, horizontal_spacing=0.10)
    for c, ax_name in enumerate(axes_names, start=1):
        sub = load_df[load_df["axis"] == ax_name]
        # top benchmarks by any-method |median| — where the axis actually lives
        strength = sub.groupby("benchmark")["loading_median"].apply(
            lambda s: s.abs().max()).nlargest(top_n)
        order = strength.index.tolist()[::-1]
        for m in methods:
            d = (sub[sub["method"] == m].set_index("benchmark")
                 .reindex(order))
            fig.add_trace(go.Bar(
                y=order, x=d["loading_median"], orientation="h", name=m,
                marker_color=_ALIGN_METHOD_COLORS[m], showlegend=(c == 1),
                error_x=dict(array=(d["hdi_high"] - d["loading_median"]),
                             arrayminus=(d["loading_median"] - d["hdi_low"]),
                             thickness=0.8, width=0)),
                row=1, col=c)
        fig.add_vline(x=0, line=dict(color="black", width=1), row=1, col=c)
    fig.update_layout(barmode="group", template="plotly_white",
                      height=max(480, 34 * top_n + 140), width=440 * len(axes_names),
                      title=dict(text=title, x=0.5),
                      legend=dict(orientation="h", y=1.08, x=0),
                      font=dict(size=10))
    return fig


