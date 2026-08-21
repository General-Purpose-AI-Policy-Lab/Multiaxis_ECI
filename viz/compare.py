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


def cmp_axis_reproducibility_fig(
        rows: list,
        title: str = "Do independent chains find the same axes? "
                     "(1D + exploratory fits)") -> go.Figure:
    """Cross-chain axis reproducibility per fit — the diagnosis behind the r̂.

    One bar group per fit, one bar per ranked axis: median |corr| of per-chain
    mean abilities (1 = every chain found the same axis; 0 = each run invents
    its own). `rows`: dicts with fit, axis_repro (list, strongest axis first),
    eta_rhat, div_pct. r̂ says THAT chains disagree; this says WHICH axis."""
    shades = ["#0C447C", "#378ADD", "#85B7EB", "#c9dff5"]
    fig = go.Figure()
    max_k = max(len(r["axis_repro"]) for r in rows)
    for k in range(max_k):
        fig.add_trace(go.Bar(
            name=f"axis {k + 1} (ranked)",
            x=[r["fit"] for r in rows],
            y=[r["axis_repro"][k] if k < len(r["axis_repro"]) else None
               for r in rows],
            marker_color=shades[min(k, len(shades) - 1)]))
    fig.add_hline(y=0.95, line=dict(color="green", dash="dash"),
                  annotation_text="reproducible", annotation_position="top right")
    fig.update_layout(
        barmode="group", template="plotly_white", height=520, width=940,
        title=dict(text=title, x=0.5),
        yaxis=dict(title="cross-chain |corr| of axis abilities", range=[0, 1.08]),
        xaxis=dict(tickvals=[r["fit"] for r in rows],
                   ticktext=[f"{r['fit']}<br><sub>r̂ {r['eta_rhat']:.2f} · "
                             f"{r['div_pct']:.0f}% div</sub>" for r in rows]),
        legend=dict(orientation="h", y=1.06, x=0))
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


def cmp_loo_ladder_fig(df: pd.DataFrame,
                       title: str = "LOO ladder, per mode (curated chain subsets)"
                       ) -> go.Figure:
    """Render the hand-curated per-mode LOO ladder CSV
    (results/comparisons/loo_cv_full.csv) — rows are fits OR chain-subset
    modes of a fit, defined offline; this figure never recomputes them.

    Columns used: model, n_chains, elpd_loo, se, eta_rhat, group, rank,
    elpd_diff, dse (plus p_loo / k>0.7 / k_max / ess_* for hover).

    Left panel: absolute ELPD ± SE for every row, colored by data group —
    absolute values are only comparable WITHIN a group (the floor-clipped fit
    scores different observed values, so its ELPD lives on its own scale).
    Right panel: ΔELPD ± paired dSE vs the group-best row, only for rows the
    CSV ranks (the comparable group). Open markers flag rows whose convergence
    is not demonstrated (eta r̂ > 1.01, or a single-chain mode where r̂ is
    undefined)."""
    d = df.copy()
    labels = [f"{m} · {int(n)}ch" for m, n in zip(d["model"], d["n_chains"])]
    y = list(range(len(d)))[::-1]                       # first CSV row on top
    group_color = {g: c for g, c in zip(d["group"].unique(),
                                        ["#2c7fb8", "#d95f02", "#2ca02c"])}
    rhat = pd.to_numeric(d["eta_rhat"], errors="coerce")
    open_marker = (rhat > 1.01) | rhat.isna() | (d["n_chains"] < 2)
    hover = [
        (f"{m}<br>ELPD {e:.1f} ± {s:.1f} · p_loo {p:.0f}"
         f"<br>Pareto k>0.7: {int(kb)} · k_max {km:.2f}"
         f"<br>eta r̂ {r if np.isfinite(r) else 'n/a (single chain)'}")
        for m, e, s, p, kb, km, r in zip(d["model"], d["elpd_loo"], d["se"],
                                         d["p_loo"], d["k>0.7"], d["k_max"], rhat)]
    fig = make_subplots(rows=1, cols=2, shared_yaxes=True,
                        subplot_titles=["ELPD ± SE (per data group)",
                                        "ΔELPD ± paired dSE (vs group best)"],
                        horizontal_spacing=0.08)
    for g in d["group"].unique():
        m = (d["group"] == g).values
        fig.add_trace(go.Scatter(
            x=d["elpd_loo"][m], y=[yy for yy, keep in zip(y, m) if keep],
            mode="markers",
            error_x=dict(type="data", array=d["se"][m], thickness=1.5, width=5),
            marker=dict(size=10, color=group_color[g],
                        symbol=["circle-open" if o else "circle"
                                for o in open_marker[m]],
                        line=dict(width=2, color=group_color[g])),
            text=[h for h, keep in zip(hover, m) if keep],
            hovertemplate="%{text}<extra></extra>", showlegend=False),
            row=1, col=1)
    ranked = d["rank"].notna().values
    fig.add_trace(go.Scatter(
        x=(-d["elpd_diff"][ranked]), y=[yy for yy, keep in zip(y, ranked) if keep],
        mode="markers",
        error_x=dict(type="data", array=d["dse"][ranked], thickness=1.5, width=5),
        marker=dict(size=10, color=[group_color[g] for g in d["group"][ranked]],
                    symbol=["circle-open" if o else "circle"
                            for o in open_marker[ranked]],
                    line=dict(width=2,
                              color=[group_color[g] for g in d["group"][ranked]])),
        text=[h for h, keep in zip(hover, ranked) if keep],
        hovertemplate="%{text}<extra></extra>", showlegend=False), row=1, col=2)
    # legend-only entries: fixed filled markers so the legend reads as the
    # group key (the per-point open/filled symbol encodes convergence, not group)
    for g in d["group"].unique():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                                 marker=dict(size=10, color=group_color[g]),
                                 name=g), row=1, col=1)
    fig.add_vline(x=0, line=dict(color="#888", dash="dash"), row=1, col=2)
    fig.update_yaxes(tickvals=y, ticktext=labels, row=1, col=1)
    fig.update_xaxes(title_text="ELPD (comparable within a group only)", row=1, col=1)
    fig.update_xaxes(title_text="ΔELPD (0 = group best)", row=1, col=2)
    fig.add_annotation(
        text="Curated ladder (loo_cv_full.csv): modes are hand-defined chain subsets.<br>"
             "Open marker = convergence not demonstrated (eta r̂ > 1.01 or "
             "single-chain mode).<br>"
             "Floor-clipped rows score different observed values; no ΔELPD "
             "across groups.",
        xref="paper", yref="paper", x=0, y=-0.26, showarrow=False, align="left",
        font=dict(size=11, color="#888"))
    fig.update_layout(title=dict(text=title, x=0.5), template="plotly_white",
                      height=max(400, 80 * len(d)) + 90, width=1050,
                      margin=dict(b=130),
                      legend=dict(orientation="h", yanchor="bottom", y=1.06,
                                  xanchor="right", x=1))
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


def cmp_axis_match_fig(df: pd.DataFrame,
                       title: str = "Do the discovered axes match the hypothesized skills? "
                                    "(|corr| of loading vectors vs the confirmed Q-matrix axes)") -> go.Figure:
    """Heatmap: each fit's axes (rows) vs the confirmed Q-matrix skill axes
    (cols), scored by |corr| of the mean loading vectors over the shared
    benchmark set. Green row = a discovered axis that lines up with one
    hypothesized skill; a row with no green is structure the category scheme
    does not explain (e.g. a signed contrast axis); a row green in TWO columns
    is a blend. `df`: rows = "fit · axis", cols = skill axis names."""
    fig = go.Figure(go.Heatmap(
        z=df.values, x=list(df.columns), y=list(df.index),
        zmin=0, zmax=1, colorscale="RdYlGn",
        text=np.round(df.values, 2), texttemplate="%{text}",
        colorbar=dict(title="|corr|")))
    fig.update_layout(title=dict(text=title, x=0.5), template="plotly_white",
                      height=max(420, 26 * len(df) + 160), width=760,
                      yaxis=dict(autorange="reversed"))
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
    `mirt_alignment_loadings_k{K}.csv` written by fit.py/align_mirt
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


def cmp_axis_collapse_fig(d3: pd.DataFrame,
                          title: str = "Axis distinctness — max |Φ| off-diagonal (→1 = axes collapse)") -> go.Figure:
    """Max |Φ| off-diagonal per fit. `d3`: df with fit, type, max_phi (float)."""
    fig = go.Figure(go.Bar(x=d3["fit"], y=d3["max_phi"],
                           marker_color=[_CMP_TYPE_COLOR.get(t, "#888780") for t in d3["type"]]))
    fig.update_layout(title=dict(text=title, x=0.5), yaxis_title="max |Φ|",
                      template="plotly_white", height=440, width=820, xaxis_tickangle=-25)
    return fig


