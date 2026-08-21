"""Per-fit MIRT figures: loading heatmaps, factor correlation, Q-matrix,
K-vs-1D comparisons, axis frontiers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ── MIRT per-fit figure builders ──
# All pure: take arrays / DataFrames, return a go.Figure. No trace loading, no
# file I/O — so the dashboard can embed them and the CLI can save them.

def loadings_heatmap_fig(A: np.ndarray, names, bench,
                         title: str = "Benchmark loadings",
                         top_n: int | None = None) -> go.Figure:
    """Benchmark × axis median-loading heatmap, benchmarks grouped by dominant
    axis then by loading strength. `A` is the (S, B, K) loading draws.

    SIGNED loadings get a diverging colorscale centred on 0
    and |·|-based grouping — the one-sided Viridis + raw max/argmax of the
    non-negative fits would hide the negative tail and mis-bucket benchmarks
    whose strongest loading is negative.

    `top_n` (optional): keep only the union of each axis's `top_n` benchmarks
    by |loading|, so a dense full-benchmark heatmap stays readable. The
    per-axis colouring and grouping are unchanged; only the row set shrinks."""
    Lmed = np.median(A, axis=0)
    signed = bool((Lmed < 0.0).any())
    Lkey = np.abs(Lmed) if signed else Lmed
    order = np.lexsort((-Lkey.max(axis=1), Lkey.argmax(axis=1)))
    if top_n is not None:
        # Union of the strongest-loading benchmarks per axis (by |loading|).
        keep = set()
        for k in range(Lkey.shape[1]):
            keep.update(np.argsort(-Lkey[:, k])[:top_n].tolist())
        order = np.array([i for i in order if i in keep])
        if title == "Benchmark loadings":
            title = f"Benchmark loadings (top {top_n} per axis)"
    if signed:
        m = float(np.abs(Lmed).max())
        heat_kw = dict(colorscale="RdBu", reversescale=True,
                       zmid=0.0, zmin=-m, zmax=m)
    else:
        heat_kw = dict(colorscale="Viridis")
    ynames = [bench[i] for i in order]
    fig = go.Figure(go.Heatmap(
        z=Lmed[order], x=list(names), y=ynames,
        colorbar=dict(title="loading"), **heat_kw))
    # Force every benchmark tick to render — Plotly otherwise thins the
    # categorical y-axis when there are many rows, dropping most names.
    fig.update_layout(title=dict(text=title, x=0.5), template="plotly_white",
                      width=620, height=max(600, 18 * len(order)),
                      yaxis=dict(tickmode="array", tickvals=ynames,
                                 ticktext=ynames, automargin=True,
                                 tickfont=dict(size=9)))
    return fig


def factor_corr_fig(Phi: np.ndarray, names, rotated: bool = False,
                    Phi_raw: np.ndarray | None = None) -> go.Figure:
    """Factor / ability correlation heatmap. When `rotated`, the title flags promax
    (oblique) and a raw-ability-correlation subtitle is annotated so the two are
    never conflated (the K=2 "0.71 promax vs 0.05 raw" trap)."""
    K = len(names)
    title = ("Factor correlations (promax · oblique)" if rotated
             else "Factor correlations (ability)")
    fig = go.Figure(go.Heatmap(
        z=Phi, x=list(names), y=list(names), zmin=-1, zmax=1, zmid=0,
        colorscale="RdBu", reversescale=True,
        text=np.round(Phi, 2), texttemplate="%{text}"))
    fig.update_layout(title=dict(text=title, x=0.5), template="plotly_white",
                      height=480, width=560, yaxis=dict(autorange="reversed"))
    if rotated and Phi_raw is not None:
        od = Phi_raw[np.triu_indices(K, 1)]
        sub = (f"raw ability correlation: {od[0]:+.2f}" if K == 2 else
               f"raw ability correlation: mean |r| = {np.abs(od).mean():.2f}, "
               f"max = {np.abs(od).max():.2f}")
        fig.add_annotation(text=sub, showarrow=False, xref="paper", yref="paper",
                           x=0.5, y=-0.13, font=dict(size=12, color="#555"))
    return fig


def binary_qmatrix_fig(Q: np.ndarray, names, bench,
                       title: str = "Q-matrix", multi_loaded: bool = False) -> go.Figure:
    """Binary Q-matrix heatmap (which axes each benchmark may/does load). Rows
    grouped by first allowed axis; with `multi_loaded`, most-loaded rows first
    (the conjunctive non-comp Q). 1 = allowed/required, 0 = zeroed/dropped."""
    Q = np.asarray(Q, dtype=float)
    primary = Q.argmax(axis=1)
    if multi_loaded:
        order = sorted(range(len(bench)), key=lambda i: (primary[i], -Q[i].sum(), bench[i]))
    else:
        order = sorted(range(len(bench)), key=lambda i: (primary[i], bench[i]))
    fig = go.Figure(go.Heatmap(
        z=Q[order], x=list(names), y=[bench[i] for i in order],
        colorscale=[[0, "#f0f0f0"], [1, "#2c7fb8"]], showscale=False, xgap=1, ygap=1))
    fig.update_layout(title=dict(text=title, x=0.5), template="plotly_white",
                      width=560, height=max(600, 15 * len(bench)))
    return fig


def ppca_spectrum_fig(med, lo, hi, labels) -> go.Figure:
    """PPCA ARD τ spectrum bar (ranked per draw). Takes the precomputed
    median + 5/95 band + axis labels (trace read stays in the caller)."""
    med, lo, hi = np.asarray(med), np.asarray(lo), np.asarray(hi)
    return go.Figure(go.Bar(x=list(labels), y=med, error_y=dict(
        type="data", symmetric=False, array=hi - med, arrayminus=med - lo))
    ).update_layout(title=dict(text="PPCA axis strength (ARD τ, ranked per draw)", x=0.5),
                    xaxis_title="axis", yaxis_title="τ",
                    template="plotly_white", height=440, width=720)


# ── single-fit deep-dive builders (K vs 1D; used by the lean plot_mirt CLI) ──

def factor_vs_1d_fig(one_d: np.ndarray, axis1: np.ndarray, model_names, r: float) -> go.Figure:
    """Axis-1 ability vs 1D capability scatter (sanity: dominant axis ≈ 1D C)."""
    fig = go.Figure(go.Scatter(
        x=one_d, y=axis1, mode="markers",
        marker=dict(color="#1f77b4", size=5, opacity=0.55),
        text=list(model_names), hovertemplate="<b>%{text}</b><extra></extra>"))
    fig.update_layout(title=f"Axis 1 vs 1D capability (r = {r:.2f})",
                      xaxis_title="1D capability", yaxis_title="axis 1 ability",
                      template="plotly_white", height=560, width=620)
    return fig


def pred_scatter_fig(pred_1d: np.ndarray, pred_kd: np.ndarray, err_delta: np.ndarray,
                     hover, title: str = "K vs 1D predicted mean (per observation)") -> go.Figure:
    """K-axis vs K=1 predicted-mean scatter, colored by |err 1D| − |err K|."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pred_1d, y=pred_kd, mode="markers",
        marker=dict(size=4, opacity=0.4, color=err_delta, colorscale="RdBu", cmid=0,
                    showscale=True, colorbar=dict(title="|err 1D| − |err K|")),
        text=hover, hovertemplate="%{text}<br>1D=%{x:.2f} · K=%{y:.2f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                             line=dict(color="#888", dash="dash"), showlegend=False))
    fig.update_layout(title=title, xaxis_title="K=1 predicted mean",
                      yaxis_title="K predicted mean", template="plotly_white",
                      height=580, width=640)
    return fig


def per_bench_r2_delta_fig(bench_df: pd.DataFrame,
                           title: str = "Per-benchmark R² gain: K − K=1") -> go.Figure:
    """Per-benchmark ΔR² (K minus 1D) horizontal bar. bench_df: name, delta_r2, n_obs."""
    fig = go.Figure(go.Bar(
        x=bench_df["delta_r2"], y=bench_df["name"], orientation="h",
        marker=dict(color=np.where(bench_df["delta_r2"] >= 0, "#2ca02c", "#d62728")),
        text=bench_df["n_obs"], texttemplate="n=%{text}", textposition="outside",
        hovertemplate="<b>%{y}</b><br>ΔR² = %{x:.3f}<extra></extra>"))
    fig.add_vline(x=0, line=dict(color="#444"))
    fig.update_layout(title=title, xaxis_title="ΔR² (positive = K fits better)",
                      template="plotly_white", height=max(600, 15 * len(bench_df)), width=820)
    return fig


def axes_frontier_fig(axis_tl: dict, names, n_axes: int,
                      title: str = "Axis frontiers over time") -> go.Figure:
    """Overlay each axis's running frontier (cumulative max mean ability) vs date.
    `axis_tl` maps axis index k → a timeline DataFrame (name, release_date, mean)."""
    pal = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
    fig = go.Figure()
    for k in range(n_axes):
        tl = axis_tl.get(k)
        if tl is None or tl.empty:
            continue
        tl = tl.sort_values("release_date")
        frontier = np.maximum.accumulate(tl["mean"].to_numpy())
        fig.add_trace(go.Scatter(x=tl["release_date"], y=frontier, mode="lines",
                                 name=names[k], line=dict(color=pal[k % len(pal)], width=2)))
        fig.add_trace(go.Scatter(x=tl["release_date"], y=tl["mean"], mode="markers",
                                 showlegend=False,
                                 marker=dict(size=4, opacity=0.3, color=pal[k % len(pal)]),
                                 hovertext=tl["name"],
                                 hovertemplate="%{hovertext}<br>%{y:.2f}<extra></extra>"))
    fig.update_layout(title=title, xaxis_title="release date", yaxis_title="frontier ability",
                      template="plotly_white", height=560, width=900)
    return fig


def axes_scatter_matrix_fig(dfm: pd.DataFrame, dims,
                            title: str = "Model abilities across axes") -> go.Figure:
    """px scatter-matrix of model abilities across axes, colored by 'org' column."""
    import plotly.express as px
    fig = px.scatter_matrix(dfm, dimensions=list(dims), color="org",
                            hover_name="model", title=title)
    fig.update_traces(diagonal_visible=False, showupperhalf=False,
                      marker=dict(size=4, opacity=0.6))
    n = len(dims)
    fig.update_layout(template="plotly_white", height=250 * n, width=250 * n)
    return fig


