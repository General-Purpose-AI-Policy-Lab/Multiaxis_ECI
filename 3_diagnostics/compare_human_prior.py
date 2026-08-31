"""Ordered-human-prior vs independent-θ humans — head-to-head, confirmed Q-matrix.

Pairs each ordered-human-prior skill fit (results/mirt_humanprior/) with its
independent-θ counterpart (results/mirt/) on the same data. Because both fits pin
the axes with the same Q-matrix, the axes ARE comparable across fits (unlike the
exploratory fits), so the per-axis tier comparison is meaningful.

Questions, one figure each (per K, in k<K>/):
  1. cmp_tier_ability_axes  — θ tier ability per axis: does each axis respect
     the tier partial order (config.HUMAN_ORDER parent chains)? The independent
     fit inverts on sparse axes; the ordered fit cannot (monotone by
     construction along each chain; branch pairs are unconstrained).
  2. cmp_score_monotonicity — # benchmarks where a weaker tier out-scores a
     COMPARABLE stronger one (an ancestor in the partial order) in predicted
     score (axis-invariant). Ordered drives it to 0.
  3. cmp_tier_score_examples — predicted score vs tier for the best-observed
     benchmarks, ×=observed.
  4. cmp_gof                — Bayesian R² / RMSE; enforcing the order must not cost fit.

Plus one shared figure (in comparison/):
  cmp_human_prior_loo       — divergences and PSIS-LOO ELPD, ordered vs independent,
     for both K. Shows the prior is convergence- and prediction-neutral.

Run:  ~/miniforge3/envs/pymc_env/bin/python 3_diagnostics/compare_human_prior.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from multiaxis_eci.analysis import post_stats, trace_axis_names  # noqa: E402
from multiaxis_eci.config import HUMAN_ORDER  # noqa: E402
from multiaxis_eci.data import load_eci_data  # noqa: E402
from multiaxis_eci.persistence import save_df  # noqa: E402
from multiaxis_eci.viz import save_fig  # noqa: E402
from multiaxis_eci.ppc import compute_gof, posterior_predictive_mirt  # noqa: E402

OUT = ROOT / "results" / "mirt_humanprior" / "comparison"
MONO_EPS = 1e-3
PAIRS = [
    {"K": 3, "label": "K=3 skills",
     "independent": ROOT / "results" / "mirt" / "trace_mirt_k3_noard_aqmatrix3.nc",
     "ordered": ROOT / "results" / "mirt_humanprior" / "trace_mirt_k3_noard_aqmatrix3_humanprior.nc"},
    {"K": 4, "label": "K=4 skills + multimodal",
     "independent": ROOT / "results" / "mirt" / "trace_mirt_k4_noard_aqmatrix4.nc",
     "ordered": ROOT / "results" / "mirt_humanprior" / "trace_mirt_k4_noard_aqmatrix4_humanprior.nc"},
]


def _expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def _tier_indices(data):
    names = data.mlookup.sort_values("model_idx")["model"].tolist()
    idx, present = [], []
    for tier in HUMAN_ORDER:
        if tier in names:
            idx.append(names.index(tier))
            present.append(tier)
    return idx, present


def summarize(path, data):
    """Per-fit summary: tier θ per axis, predicted μ per tier, GoF, divergences, LOO."""
    idata = az.from_netcdf(str(path))
    post = idata.posterior
    K = int(post.sizes["latent"])
    S = int(post.sizes["chain"] * post.sizes["draw"])
    step = max(1, S // 1000)

    A = post["A"].values.reshape(S, data.n_benchmarks, K)[::step]
    D = post["D"].values.reshape(S, data.n_benchmarks)[::step]
    theta = post["theta"].values.reshape(S, data.n_models, K)[::step]
    t_idx, t_names = _tier_indices(data)

    theta_stats = {(name, k): post_stats(theta[:, ti, k])
                   for ti, name in zip(t_idx, t_names) for k in range(K)}
    mu_mean = {name: _expit(np.einsum("sbk,sk->sb", A, theta[:, ti, :]) - D).mean(axis=0)
               for ti, name in zip(t_idx, t_names)}

    yrep = posterior_predictive_mirt(idata, data)
    mu = posterior_predictive_mirt(idata, data, return_mean=True)
    gof = compute_gof(yrep, data, mu)
    human_obs = data.is_human[data.model_idx]
    r = data.scores - gof.y_pred_mean
    rmse_hum = float(np.sqrt((r[human_obs] ** 2).mean())) if human_obs.any() else float("nan")
    div = int(idata.sample_stats["diverging"].sum()) if "diverging" in idata.sample_stats else -1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loo = az.loo(idata)
    axis_names = trace_axis_names(idata, K)
    del idata
    return {
        "K": K, "axis_names": axis_names, "tier_names": t_names,
        "theta_stats": theta_stats, "mu_mean": mu_mean,
        "r2": float(gof.metrics["bayesian_r2"]), "rmse_all": float(gof.metrics["rmse"]),
        "rmse_human": rmse_hum, "div": div,
        "loo_elpd": float(loo.elpd_loo), "loo_se": float(loo.se),
    }


def _comparable_pairs(tier_names):
    """(weaker, stronger) tier pairs the partial order actually constrains —
    every ancestor→descendant pair in HUMAN_ORDER. Tiers on different branches
    (e.g. Top Performer vs Committee of Domain Experts) are incomparable and a
    score flip between them is NOT a violation."""
    pairs = []
    for t in tier_names:
        p = HUMAN_ORDER.get(t)
        while p is not None:
            if p in tier_names:
                pairs.append((p, t))
            p = HUMAN_ORDER.get(p)
    return pairs


def _violations(mu_mean, tier_names):
    pairs = _comparable_pairs(tier_names)
    if not pairs:
        return 0, 0
    drops = np.vstack([mu_mean[w] - mu_mean[s] > MONO_EPS for w, s in pairs])
    return int(drops.any(axis=0).sum()), int(drops.sum())


def fig_tier_ability(orig, ordd):
    """θ tier ability per axis, independent (dotted) vs ordered (solid)."""
    K = ordd["K"]
    tiers, x = ordd["tier_names"], list(range(len(ordd["tier_names"])))
    fig = make_subplots(rows=1, cols=K, shared_yaxes=True, subplot_titles=ordd["axis_names"])
    for k in range(K):
        for fit, color, dash, tag in [(orig, "#d95f02", "dot", "independent"),
                                      (ordd, "#2c7fb8", "solid", "ordered")]:
            means = np.array([fit["theta_stats"][(t, k)][0] for t in tiers])
            los = np.array([fit["theta_stats"][(t, k)][1] for t in tiers])
            his = np.array([fit["theta_stats"][(t, k)][2] for t in tiers])
            fig.add_trace(go.Scatter(
                x=x, y=means, mode="lines+markers", name=tag, legendgroup=tag,
                showlegend=(k == 0), line=dict(color=color, dash=dash),
                error_y=dict(type="data", symmetric=False, array=his - means,
                             arrayminus=means - los, thickness=1, width=3, color=color)),
                row=1, col=k + 1)
        fig.update_xaxes(tickvals=x, ticktext=[t.replace(" ", "<br>") for t in tiers],
                         row=1, col=k + 1)
    fig.update_layout(
        title=dict(text="Human-tier ability per axis — independent (dotted) vs ordered (solid)", x=0.5),
        template="plotly_white", height=460, width=max(560, 320 * K), yaxis_title="θ (ability)")
    return fig


def fig_score_monotonicity(orig, ordd):
    o_b, o_s = _violations(orig["mu_mean"], ordd["tier_names"])
    n_b, n_s = _violations(ordd["mu_mean"], ordd["tier_names"])
    fig = go.Figure(go.Bar(
        x=["independent", "ordered"], y=[o_b, n_b], marker_color=["#d95f02", "#2c7fb8"],
        text=[f"{o_b} benchmarks<br>({o_s} down-steps)", f"{n_b} benchmarks<br>({n_s} down-steps)"],
        textposition="outside"))
    fig.update_layout(
        title=dict(text="Predicted-score ordering violations across tiers (lower = better)", x=0.5),
        yaxis_title="# benchmarks where a weaker tier out-scores a stronger one",
        template="plotly_white", height=460, width=620)
    return fig, (o_b, n_b)


def fig_tier_examples(orig, ordd, data, n=6):
    """Predicted μ vs tier for the n best-observed benchmarks, ×=observed."""
    bench = data.blookup.sort_values("benchmark_idx")["benchmark"].tolist()
    tiers = ordd["tier_names"]
    tier_pos = {data.mlookup.sort_values("model_idx")["model"].tolist().index(t): t for t in tiers}
    obs = {}
    for mi, bi, sc in zip(data.model_idx, data.bench_idx, data.scores):
        if mi in tier_pos:
            obs.setdefault(bi, {})[tier_pos[mi]] = sc
    ranked = sorted(obs, key=lambda b: len(obs[b]), reverse=True)[:n] or list(range(min(n, len(bench))))

    rows = (len(ranked) + 2) // 3
    fig = make_subplots(rows=rows, cols=3, subplot_titles=[bench[b][:38] for b in ranked])
    x = list(range(len(tiers)))
    for i, b in enumerate(ranked):
        rr, cc = i // 3 + 1, i % 3 + 1
        for fit, color, dash, tag in [(orig, "#d95f02", "dot", "independent"),
                                      (ordd, "#2c7fb8", "solid", "ordered")]:
            fig.add_trace(go.Scatter(x=x, y=[fit["mu_mean"][t][b] for t in tiers],
                          mode="lines+markers", name=tag, legendgroup=tag,
                          showlegend=(i == 0), line=dict(color=color, dash=dash)), row=rr, col=cc)
        if b in obs:
            fig.add_trace(go.Scatter(x=[tiers.index(t) for t in obs[b]], y=list(obs[b].values()),
                          mode="markers", name="observed", legendgroup="observed",
                          showlegend=(i == 0), marker=dict(color="#111", size=9, symbol="x")),
                          row=rr, col=cc)
        fig.update_xaxes(tickvals=x, ticktext=[t.split()[0] for t in tiers], row=rr, col=cc)
    fig.update_layout(title=dict(text="Predicted score vs human tier (×=observed)", x=0.5),
                      template="plotly_white", height=300 * rows, width=1000)
    return fig


def fig_gof(orig, ordd):
    cats = ["R² (all)", "RMSE (all)", "RMSE (human rows)"]
    fig = go.Figure()
    for fit, name, color in [(orig, "independent", "#d95f02"), (ordd, "ordered", "#2c7fb8")]:
        vals = [fit["r2"], fit["rmse_all"], fit["rmse_human"]]
        fig.add_trace(go.Bar(name=name, x=cats, y=vals, marker_color=color,
                             text=[f"{v:.3f}" for v in vals], textposition="outside"))
    fig.update_layout(title=dict(text="Cost of enforcing the order — fit should be ≈unchanged", x=0.5),
                      barmode="group", template="plotly_white", height=460, width=720,
                      yaxis_rangemode="tozero")
    return fig


def fig_div_loo(rows):
    """Shared: divergences and PSIS-LOO ELPD, ordered vs independent, both K.
    The prior should move neither — it constrains 5 human tiers' θ, nothing that
    drives mixing or prediction."""
    labels = [r["label"] for r in rows]
    fig = make_subplots(rows=1, cols=2, subplot_titles=["divergences", "PSIS-LOO ELPD"])
    for col, key, err in [(1, "div", None), (2, "loo_elpd", "loo_se")]:
        for name, color in [("independent", "#d95f02"), ("ordered", "#2c7fb8")]:
            ey = dict(type="data", array=[r[name][err] for r in rows]) if err else None
            fig.add_trace(go.Bar(name=name, x=labels, y=[r[name][key] for r in rows],
                                 marker_color=color, error_y=ey, showlegend=(col == 1)),
                          row=1, col=col)
    fig.update_layout(barmode="group", template="plotly_white", height=440, width=900,
                      title=dict(text="Ordered human prior — effect on convergence & prediction", x=0.5))
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_eci_data(include_all_benchmarks=True)

    table, loo_rows = [], []
    for pair in PAIRS:
        if not pair["independent"].exists() or not pair["ordered"].exists():
            print(f"{pair['label']}: SKIP — missing a trace", flush=True)
            continue
        print(f"{pair['label']}: independent → {pair['independent'].name}", flush=True)
        orig = summarize(pair["independent"], data)
        print(f"{pair['label']}: ordered     → {pair['ordered'].name}", flush=True)
        ordd = summarize(pair["ordered"], data)

        sub = OUT / f"k{pair['K']}"
        sub.mkdir(parents=True, exist_ok=True)
        save_fig(fig_tier_ability(orig, ordd), "cmp_tier_ability_axes", sub)
        mono_fig, (o_v, n_v) = fig_score_monotonicity(orig, ordd)
        save_fig(mono_fig, "cmp_score_monotonicity", sub)
        save_fig(fig_tier_examples(orig, ordd, data), "cmp_tier_score_examples", sub)
        save_fig(fig_gof(orig, ordd), "cmp_gof", sub)

        loo_rows.append({"label": pair["label"], "independent": orig, "ordered": ordd})
        table.append({
            "fit": pair["label"], "viol_independent": o_v, "viol_ordered": n_v,
            "R2_independent": round(orig["r2"], 4), "R2_ordered": round(ordd["r2"], 4),
            "div_independent": orig["div"], "div_ordered": ordd["div"],
            "loo_independent": round(orig["loo_elpd"], 1), "loo_ordered": round(ordd["loo_elpd"], 1),
            "RMSE_human_independent": round(orig["rmse_human"], 4),
            "RMSE_human_ordered": round(ordd["rmse_human"], 4)})
        print(f"  violations: indep={o_v} ordered={n_v} | R²: {orig['r2']:.3f}→{ordd['r2']:.3f} "
              f"| div: {orig['div']}→{ordd['div']} | LOO: {orig['loo_elpd']:.0f}→{ordd['loo_elpd']:.0f}",
              flush=True)

    if not table:
        print("\nNothing compared — fit the confirmed human-prior models first.")
        return
    save_fig(fig_div_loo(loo_rows), "cmp_human_prior_loo", OUT)
    tab = pd.DataFrame(table)
    save_df(tab, OUT / "comparison_table.csv")
    print("\n" + tab.to_string(index=False), flush=True)

    (OUT / "README.md").write_text(
        "# Ordered vs independent human baselines (confirmed Q-matrix)\n\n"
        "Each ordered-human-prior skill fit (`results/mirt_humanprior/`) paired with "
        "its independent-θ counterpart (`results/mirt/`) on the same data. The "
        "Q-matrix pins the axes, so axes are comparable across the two fits.\n\n"
        "## Summary\n\n" + tab.to_string(index=False) + "\n\n"
        "- `viol_*` — # benchmarks where a weaker tier out-scores a COMPARABLE "
        "stronger one (an ancestor in the config.HUMAN_ORDER partial order) in "
        "predicted score; the prior drives it to 0. Incomparable branch pairs "
        "(e.g. Top Performer vs Committee of Domain Experts) are not counted.\n"
        "- `R2_*`, `div_*`, `loo_*` — fit, convergence, prediction: ≈unchanged "
        "(the prior only constrains the human tiers' θ).\n\n"
        "Per-K figures in `k<K>/`: `cmp_tier_ability_axes`, `cmp_score_monotonicity`, "
        "`cmp_tier_score_examples`, `cmp_gof`. Shared: `cmp_human_prior_loo` "
        "(divergences + PSIS-LOO, ordered vs independent).\n")
    print(f"\nComparison → {OUT}", flush=True)


if __name__ == "__main__":
    main()
