"""Per-fit MIRT figures: loading heatmaps, factor correlation, Q-matrix,
K-vs-1D comparisons, axis frontiers."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from multiaxis_eci.viz.core import AI_COLOR, HUMAN_COLOR

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


# ── per-axis grids (one panel per axis, one figure-level legend) ──

# The row kinds a per-axis forest can carry: marker color, marker symbol, legend
# label. `kind` is optional on the input frames — a frame without it is models.
FOREST_KINDS = (
    ("model",    AI_COLOR,    "circle",  "models"),
    ("frontier", AI_COLOR,    "diamond", "frontier releases (shown even when wide)"),
    ("human",    HUMAN_COLOR, "square",  "human tiers"),
)


def forest_grid_fig(dfs, titles, ncols: int = 2,
                    x_title: str = "ability (median, 94% interval)",
                    title: str = "Top models, frontier releases and human tiers per axis",
                    row_px: int = 22, width: int = 1180) -> go.Figure:
    """One forest per axis on a grid, sharing a single figure-level legend.

    Each frame is a `forest_fig` frame (name, mean, hdi_low, hdi_high) plus an
    optional `kind` column in {model, frontier, human}, which picks the marker
    and the legend entry. Rows are drawn in the frame's own order, bottom-up, so
    a caller that sorts ascending by `mean` gets the strongest row at the top.
    Panel y-axes are independent: the same name may sit at a different height in
    each panel, which is the point of a per-axis forest.
    """
    dfs = list(dfs)
    nrows = -(-len(dfs) // ncols)
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=list(titles),
                        horizontal_spacing=0.36 / ncols, vertical_spacing=0.10)
    seen: set[str] = set()
    tallest = 1
    for i, df in enumerate(dfs):
        row, col = i // ncols + 1, i % ncols + 1
        kinds = df["kind"] if "kind" in df else pd.Series("model", index=df.index)
        tallest = max(tallest, len(df))
        for kind, color, symbol, label in FOREST_KINDS:
            d = df[kinds == kind]
            if d.empty:
                continue
            fig.add_trace(go.Scatter(
                x=d["mean"], y=d["name"], mode="markers", name=label,
                legendgroup=kind, showlegend=label not in seen,
                marker=dict(color=color, size=7, symbol=symbol),
                error_x=dict(type="data", symmetric=False,
                             array=(d["hdi_high"] - d["mean"]).values,
                             arrayminus=(d["mean"] - d["hdi_low"]).values,
                             color=color, thickness=1.6, width=0),
                hovertemplate="<b>%{y}</b><br>%{x:.2f}<extra></extra>",
            ), row=row, col=col)
            seen.add(label)
        # Every row label must render: Plotly thins a long categorical axis.
        names = df["name"].tolist()
        fig.update_yaxes(categoryorder="array", categoryarray=names,
                         tickmode="array", tickvals=names, ticktext=names,
                         tickfont=dict(size=9), automargin=True,
                         row=row, col=col)
        fig.update_xaxes(title_text=x_title, row=row, col=col)
    fig.update_layout(
        title=dict(text=title, x=0.5), template="plotly_white",
        width=width, height=nrows * max(360, row_px * tallest) + 90,
        legend=dict(orientation="h", yanchor="bottom", y=-0.07, x=0.5,
                    xanchor="center"),
        margin=dict(l=60, r=30, t=90, b=90))
    return fig


def loadings_grid_fig(load_df: pd.DataFrame, titles: dict | None = None,
                      ncols: int = 2, top_n: int = 20,
                      title: str | None = None, width: int = 1180) -> go.Figure:
    """Per axis, the `top_n` benchmarks with the largest share of that axis.

    `load_df` is `analysis.loadings_table` output — one row per (axis,
    benchmark) with loading_median / hdi_low / hdi_high / axis_share — so the
    share definition (the fraction of a benchmark's squared loading row pointing
    along this axis) is the one the dashboard heatmap and the fit's own CSVs use.

    The bar is the loading, so steepness stays visible; the color is the share,
    so a short row pointing squarely along the axis reads as pure even though
    its bar is short. Ranking is by share, not by loading: a long half-aligned
    row would otherwise out-rank it and mis-name the axis.
    """
    titles = titles or {}
    axes = list(dict.fromkeys(load_df["axis"]))
    nrows = -(-len(axes) // ncols)
    fig = make_subplots(rows=nrows, cols=ncols, vertical_spacing=0.10,
                        horizontal_spacing=0.24 / ncols,
                        subplot_titles=[titles.get(a, a) for a in axes])
    for i, axis in enumerate(axes):
        row, col = i // ncols + 1, i % ncols + 1
        d = (load_df[load_df["axis"] == axis]
             .sort_values("axis_share", ascending=False, kind="stable")
             .head(top_n).iloc[::-1])            # weakest share at the bottom
        fig.add_trace(go.Bar(
            x=d["loading_median"], y=d["benchmark"], orientation="h",
            showlegend=False, marker=dict(color=d["axis_share"], coloraxis="coloraxis"),
            error_x=dict(type="data", symmetric=False,
                         array=(d["hdi_high"] - d["loading_median"]).values,
                         arrayminus=(d["loading_median"] - d["hdi_low"]).values,
                         color="#333", thickness=1.1, width=3),
            customdata=d["axis_share"],
            hovertemplate="<b>%{y}</b><br>loading %{x:.2f}"
                          "<br>axis share %{customdata:.2f}<extra></extra>",
        ), row=row, col=col)
        names = d["benchmark"].tolist()
        # The share is printed after the name as well as encoded in the color: a
        # static export has no hover, and the exact number is the ranking key.
        fig.update_yaxes(
            categoryorder="array", categoryarray=names,
            tickmode="array", tickvals=names,
            ticktext=[f"{b}  {s:.2f}" for b, s in zip(names, d["axis_share"])],
            tickfont=dict(size=9), automargin=True, row=row, col=col)
        fig.update_xaxes(title_text="loading (median, 94% interval)",
                         zeroline=True, zerolinecolor="#222", row=row, col=col)
    fig.update_layout(
        title=dict(text=title or f"The {top_n} benchmarks that define each axis, "
                                 "ranked by axis share", x=0.5),
        template="plotly_white", width=width,
        height=nrows * max(360, 22 * top_n) + 80,
        coloraxis=dict(colorscale="Viridis", cmin=0.0, cmax=1.0,
                       colorbar=dict(title="axis share", len=0.6)),
        margin=dict(l=60, r=30, t=90, b=60))
    return fig


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


