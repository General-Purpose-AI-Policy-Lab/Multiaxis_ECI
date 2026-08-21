"""Goodness-of-fit figures: PIT, density overlay, predicted-vs-observed,
per-benchmark residuals, per-benchmark raw scores vs predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from config import DENSITY_SEED

# ── Goodness of fit ───────────────────────────────────────────────────────
def pit_hist_fig(pit: np.ndarray, n_bins: int = 20) -> go.Figure:
    n = len(pit)
    p = 1 / n_bins
    se = np.sqrt(p * (1 - p) / n)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=pit, nbinsx=n_bins, marker_color="#4682B4", opacity=0.85,
        histnorm="probability density",
        hovertemplate="PIT bin %{x}<br>density: %{y:.3f}<extra></extra>",
    ))
    fig.add_hrect(
        y0=(p - 1.96 * se) * n_bins, y1=(p + 1.96 * se) * n_bins,
        line_width=0, fillcolor="crimson", opacity=0.10,
        annotation_text="95% band under H0",
        annotation_position="bottom right",
    )
    fig.add_hline(y=1.0, line_dash="dash", line_color="crimson",
                  annotation_text="Uniform(0,1)", annotation_position="top right")
    fig.update_layout(
        title="PIT histogram",
        xaxis_title="PIT u_n", yaxis_title="density",
        template="plotly_white",
        height=420, width=820,
        margin=dict(l=55, r=20, t=55, b=45),
    )
    return fig


def pit_ecdf_fig(pit: np.ndarray) -> go.Figure:
    sorted_pit = np.sort(pit)
    ecdf_y = np.arange(1, len(pit) + 1) / len(pit)
    ks_band = 1.358 / np.sqrt(len(pit))

    u = np.linspace(0, 1, 200)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([u, u[::-1]]),
        y=np.concatenate([np.clip(u - ks_band, 0, 1),
                          np.clip(u[::-1] + ks_band, 0, 1)]),
        fill="toself", fillcolor="rgba(220,20,60,0.10)",
        line=dict(width=0), name="95% KS band", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="crimson", dash="dash"), name="Uniform(0,1)",
    ))
    fig.add_trace(go.Scatter(
        x=sorted_pit, y=ecdf_y, mode="lines",
        line=dict(color="#1f77b4", width=2), name="PIT empirical CDF",
    ))
    fig.update_layout(
        title="PIT empirical CDF",
        xaxis_title="u", yaxis_title="F(u)",
        template="plotly_white",
        height=480, width=640,
        margin=dict(l=55, r=20, t=55, b=45),
    )
    return fig


def density_overlay_fig(y_rep_flat: np.ndarray,
                          scores: np.ndarray,
                          n_samples: int = 200,
                          seed: int = DENSITY_SEED) -> go.Figure:
    rng = np.random.default_rng(seed)
    sample_idx = rng.choice(y_rep_flat.shape[0], size=n_samples, replace=False)
    x_grid = np.linspace(0, 1, 80)
    x_mid  = (x_grid[:-1] + x_grid[1:]) / 2

    fig = go.Figure()
    for j, s in enumerate(sample_idx):
        h, _ = np.histogram(y_rep_flat[s], bins=x_grid, density=True)
        fig.add_trace(go.Scatter(
            x=x_mid, y=h, mode="lines",
            line=dict(color="#4682B4", width=0.5),
            opacity=0.20, showlegend=(j == 0),
            name="posterior predictive draws" if j == 0 else None,
            hoverinfo="skip",
        ))
    h_obs, _ = np.histogram(scores, bins=x_grid, density=True)
    fig.add_trace(go.Scatter(
        x=x_mid, y=h_obs, mode="lines",
        line=dict(color="crimson", width=2.2), name="observed scores",
    ))
    fig.update_layout(
        title="Posterior predictive density",
        xaxis_title="score", yaxis_title="density",
        template="plotly_white",
        height=460, width=820,
        margin=dict(l=55, r=20, t=55, b=45),
    )
    return fig


def pred_vs_obs_fig(scores: np.ndarray,
                     y_pred_mean: np.ndarray,
                     hover_labels: list[str]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        line=dict(color="gray", dash="dash"), name="y_pred = y_obs",
    ))
    fig.add_trace(go.Scatter(
        x=scores, y=y_pred_mean, mode="markers",
        marker=dict(color="#1f77b4", size=5, opacity=0.55, line=dict(width=0)),
        text=hover_labels,
        hovertemplate="<b>%{text}</b><br>obs=%{x:.3f}<br>pred=%{y:.3f}<extra></extra>",
        name="posterior mean prediction",
    ))
    fig.update_layout(
        title="Predicted vs observed",
        xaxis_title="observed score", yaxis_title="posterior predictive mean",
        template="plotly_white",
        height=560, width=620,
        margin=dict(l=60, r=20, t=55, b=50),
        xaxis=dict(range=[0, 1]), yaxis=dict(range=[0, 1]),
    )
    return fig


def benchmark_obs_vs_pred_fig(scores: np.ndarray,
                              yrep: np.ndarray,
                              model_of_obs: list[str],
                              bench_of_obs: list[str],
                              interval: tuple[int, int] = (5, 95)) -> go.Figure:
    """Raw scores vs posterior predictive, one benchmark at a time (dropdown).

    For the selected benchmark, models are ranked by observed score; each rank
    carries the observed raw score and the predictive median with its
    `interval` percentile band. Systematic misfit reads as the two marker sets
    separating; the band width shows how much of the gap the model attributes
    to benchmark noise. Opens on the most-observed benchmark.
    """
    scores = np.asarray(scores)
    lo, med, hi = np.percentile(yrep, [interval[0], 50, interval[1]], axis=0)
    band = interval[1] - interval[0]

    obs_of_bench: dict[str, list[int]] = {}
    for i, b in enumerate(bench_of_obs):
        obs_of_bench.setdefault(b, []).append(i)
    benches = sorted(obs_of_bench)
    default = max(range(len(benches)), key=lambda j: len(obs_of_bench[benches[j]]))

    fig = go.Figure()
    for j, b in enumerate(benches):
        idx = np.asarray(obs_of_bench[b])
        idx = idx[np.argsort(scores[idx])]
        rank = np.arange(1, len(idx) + 1)
        models = [model_of_obs[i] for i in idx]
        fig.add_trace(go.Scatter(
            x=rank, y=med[idx], mode="markers", visible=(j == default),
            marker=dict(color="#4682B4", size=6),
            error_y=dict(type="data", array=hi[idx] - med[idx],
                         arrayminus=med[idx] - lo[idx],
                         color="rgba(70,130,180,0.45)", thickness=1.2, width=0),
            text=models,
            hovertemplate="<b>%{text}</b><br>predicted=%{y:.3f}<extra></extra>",
            name=f"predicted (median, {band}% interval)",
        ))
        fig.add_trace(go.Scatter(
            x=rank, y=scores[idx], mode="markers", visible=(j == default),
            marker=dict(color="crimson", size=6, symbol="diamond"),
            text=models,
            hovertemplate="<b>%{text}</b><br>observed=%{y:.3f}<extra></extra>",
            name="observed score",
        ))

    buttons = []
    for j, b in enumerate(benches):
        vis = [False] * (2 * len(benches))
        vis[2 * j] = vis[2 * j + 1] = True
        buttons.append(dict(
            label=f"{b} ({len(obs_of_bench[b])})", method="update",
            args=[{"visible": vis},
                  {"title.text": f"Raw scores vs predictions — {b}"}],
        ))
    fig.update_layout(
        title=f"Raw scores vs predictions — {benches[default]}",
        updatemenus=[dict(buttons=buttons, active=default,
                          x=1.0, xanchor="right", y=1.18, yanchor="top")],
        xaxis_title="models, ranked by observed score",
        yaxis=dict(title="score", range=[-0.02, 1.02]),
        template="plotly_white",
        height=520, width=980,
        margin=dict(l=55, r=20, t=95, b=50),
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def benchmark_icc_fig(eta_of_obs: np.ndarray,
                      scores: np.ndarray,
                      model_of_obs: list[str],
                      bench_of_obs: list[str],
                      floor: dict | None = None,
                      ceiling: dict | None = None) -> go.Figure:
    """Item characteristic curve per benchmark (dropdown).

    The fitted response sigmoid μ = c + (d−c)·σ(η) drawn against the observed
    scores, where η = A_b·θ_m − D_b (from posterior-mean loadings, ability, and
    difficulty) places each model on the benchmark's logit scale. Dots hugging
    the curve mean the model predicts that benchmark well; a vertical bias or
    wide scatter off it flags misfit. `floor`/`ceiling` are per-benchmark
    asymptotes (default 0/1). Opens on the most-observed benchmark.
    """
    eta = np.asarray(eta_of_obs, float)
    scores = np.asarray(scores, float)
    floor, ceiling = floor or {}, ceiling or {}

    obs_of_bench: dict[str, list[int]] = {}
    for i, b in enumerate(bench_of_obs):
        obs_of_bench.setdefault(b, []).append(i)
    benches = sorted(obs_of_bench)
    default = max(range(len(benches)), key=lambda j: len(obs_of_bench[benches[j]]))

    fig = go.Figure()
    for j, b in enumerate(benches):
        idx = np.asarray(obs_of_bench[b])
        c, d = floor.get(b, 0.0), ceiling.get(b, 1.0)
        grid = np.linspace(eta[idx].min() - 1.0, eta[idx].max() + 1.0, 100)
        fig.add_trace(go.Scatter(
            x=grid, y=c + (d - c) / (1.0 + np.exp(-grid)), mode="lines",
            visible=(j == default), line=dict(color="#4682B4", width=2),
            name="fitted sigmoid", hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=eta[idx], y=scores[idx], mode="markers", visible=(j == default),
            marker=dict(color="crimson", size=6, symbol="diamond"),
            text=[model_of_obs[i] for i in idx],
            hovertemplate="<b>%{text}</b><br>η=%{x:.2f}<br>observed=%{y:.3f}<extra></extra>",
            name="observed score",
        ))

    buttons = []
    for j, b in enumerate(benches):
        vis = [False] * (2 * len(benches))
        vis[2 * j] = vis[2 * j + 1] = True
        buttons.append(dict(
            label=f"{b} ({len(obs_of_bench[b])})", method="update",
            args=[{"visible": vis},
                  {"title.text": f"Item characteristic curve — {b}"}],
        ))
    fig.update_layout(
        title=f"Item characteristic curve — {benches[default]}",
        updatemenus=[dict(buttons=buttons, active=default,
                          x=1.0, xanchor="right", y=1.18, yanchor="top")],
        xaxis=dict(title="ability − difficulty  (A·θ − D, logits)"),
        yaxis=dict(title="score", range=[-0.02, 1.02]),
        template="plotly_white",
        height=520, width=980,
        margin=dict(l=55, r=20, t=95, b=50),
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def residuals_per_benchmark_fig(resid_df: pd.DataFrame) -> go.Figure:
    order = (resid_df.groupby("benchmark")["residual"]
             .mean().sort_values().index.tolist())
    fig = go.Figure()
    for b in order:
        fig.add_trace(go.Box(
            x=resid_df.loc[resid_df["benchmark"] == b, "residual"],
            name=b,
            boxpoints="all", jitter=0.4, pointpos=0,
            marker=dict(size=3, opacity=0.5, color="#1f77b4"),
            line=dict(width=1),
            showlegend=False,
        ))
    fig.add_vline(x=0, line_dash="dash", line_color="crimson")
    fig.update_layout(
        title="Residuals per benchmark",
        xaxis_title="residual",
        template="plotly_white",
        height=820, width=900,
        margin=dict(l=220, r=30, t=55, b=45),
    )
    return fig


