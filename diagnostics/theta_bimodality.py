"""Per-test-taker theta bimodality on a fitted MIRT trace — no re-sampling.

`diagnose_chains.py` answers "are the chains one solution?" and reads only
rotation-invariant quantities (logp, D, chain-mean loadings A). On the flagship
K=4 fit it answers ONE mode, all 12 chains. This script answers the different
question that the fit-level verdict cannot see: is an individual test-taker's
per-axis theta a well-summarized quantity?

It is not, for a minority of takers. Their per-chain theta means split into two
lumps on an axis while the 4-axis SUM stays pinned — chains park the same total
ability in different axes. That is a flat trade-off ridge in the parametrization,
not a second solution: logp and loadings are invariant along it, which is exactly
why the mode detector is right to report one mode.

Two signals per (taker, axis):
  1. chain_split — sort the per-chain theta means, take the largest gap. Flag it
     when the gap exceeds 3x the median WITHIN-chain sd and both sides hold >= 2
     chains. Scaling by the within-chain sd is what makes "gap" mean "the chains
     do not overlap" rather than "the numbers differ".
  2. outside_50 — the pooled posterior MEAN falls outside its own 50% HDI
     (highest-density interval). A one-line consequence of two lumps: the mean
     lands in the empty valley between them, while the shortest 50% interval
     sits inside one lump. Read on the HDI and NOT on the central 25-75%
     quantile interval, which spans the valley and therefore contains the mean
     for all 835 takers on all 4 axes — it cannot see the effect at all.
     `outside_50_central` keeps that comparison in the table.

Per taker, `ridge_ratio = sd(per-chain means of the axis SUM) / mean over axes of
sd(per-chain means of that axis)`. Small means the total is pinned while the
axes swap, which is the ridge signature.

The same scan runs on the loading matrix A, which has the identical
(chain, draw, unit, axis) shape, to check whether benchmark loadings split too.

Outputs:
  results/<fit>/theta_bimodality.csv   — one row per taker x axis.
  results/<fit>/bimodality.html        — three figures, no prose: a gap heatmap
                                         over the flagged takers, and one
                                         dropdown of per-chain-lump histograms
                                         each for takers and for benchmarks.

Axes are named axis1..axisK. The trace carries no `mirt_axis_names`, so any
semantic name would be invented.

Run:
  ~/miniforge3/envs/pymc_env/bin/python diagnostics/theta_bimodality.py
  ~/miniforge3/envs/pymc_env/bin/python diagnostics/theta_bimodality.py \
      --trace results/mirt_signed/trace_mirt_k3_signed.nc
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.stats import _release_dates  # noqa: E402
from analysis.timelines import mirt_informed_mask  # noqa: E402
from data import PROCESSED_FILE  # noqa: E402
from diagnostics.diagnose_chains import _load_matching_data, modes_path  # noqa: E402
from lineage import build_lineage_structure  # noqa: E402

DEFAULT_TRACE = (ROOT / "results"
                 / "mirt_humanmerge_lineageprior_lineagebm_dropFrontierMathv1AlgoTune_floors_poolednoise"
                 / "trace_mirt_k4_humanmerge_lineageprior_lineagebm_dropFrontierMathv1AlgoTune_floors_poolednoise.nc")

# a chain-mean gap this many within-chain sds wide separates two lumps
GAP_SDS = 3.0
# min chains on each side of the gap — 1 chain apart is an outlier, not a lump
MIN_SIDE = 2
# posterior SD below which an ability counts as data-informed (timelines' default)
SD_CAP = 0.4
# draw stride for the figure histograms: 600 draws per chain resolve two lumps,
# and the page ships binned COUNTS, so the stride costs smoothness, not weight
FIG_STRIDE = 10

# Okabe-Ito, colorblind-safe: the two lumps and the agreeing case
C_LOW, C_HIGH, C_ONE = "#009E73", "#D55E00", "#7f7f7f"


def _open_posterior(path):
    """The posterior group only. The flagship trace is 20 GB on disk and theta is
    2.4 GB of it; `az.from_netcdf` would pull every other variable too."""
    import xarray as xr
    return az.InferenceData(posterior=xr.open_dataset(path, group="posterior"))


def taker_axis_stats(theta: np.ndarray) -> dict:
    """Per (model, axis) statistics from theta (chain, draw, M, K).

    Returns arrays shaped (M, K) unless noted. `chain_split` marks the two-lump
    case; `lump_lo`/`lump_hi`/`hi_share` are NaN where it is not flagged."""
    C, Dn, M, K = theta.shape
    flat = theta.reshape(C * Dn, M, K)          # view, no copy
    cmean = theta.mean(axis=1)                  # (C, M, K) per-chain means
    csd = np.median(theta.std(axis=1), axis=0)  # (M, K) median within-chain sd

    mean = flat.mean(axis=0)
    sd = flat.std(axis=0)
    q25 = np.empty((M, K)); med = np.empty((M, K)); q75 = np.empty((M, K))
    hdi = np.empty((M, K, 2))
    for k in range(K):                          # per axis: caps the sort copies
        q25[:, k], med[:, k], q75[:, k] = np.quantile(
            flat[:, :, k], [0.25, 0.5, 0.75], axis=0)
        hdi[:, k] = np.asarray(az.hdi(theta[:, :, :, k], hdi_prob=0.5))

    srt = np.sort(cmean, axis=0)                # (C, M, K)
    gaps = np.diff(srt, axis=0)                 # (C-1, M, K)
    # only cuts leaving >= MIN_SIDE chains per side are eligible
    ok = np.zeros(C - 1, bool)
    ok[MIN_SIDE - 1:C - MIN_SIDE] = True
    cut = np.where(ok[:, None, None], gaps, -np.inf).argmax(axis=0)   # (M, K)
    gap = np.take_along_axis(gaps, cut[None], axis=0)[0]
    split = ok[cut] & (gap > GAP_SDS * csd)

    n_hi = C - 1 - cut                          # chains above the cut
    lo = np.where(np.arange(C)[:, None, None] <= cut[None], srt, np.nan)
    hi = np.where(np.arange(C)[:, None, None] > cut[None], srt, np.nan)
    nan = np.full((M, K), np.nan)
    return {
        "mean": mean, "median": med, "sd": sd,
        "chain_split": split,
        "outside_50": (mean < hdi[:, :, 0]) | (mean > hdi[:, :, 1]),
        "outside_50_central": (mean < q25) | (mean > q75),
        "lump_lo": np.where(split, np.nanmean(lo, axis=0), nan),
        "lump_hi": np.where(split, np.nanmean(hi, axis=0), nan),
        "hi_share": np.where(split, n_hi / C, nan),
        "chain_means": cmean,
        # sd over chains of the per-chain mean of the axis SUM (mean of a sum is
        # the sum of the means, so no (C, draw, M) temporary is needed)
        "sum_spread": cmean.sum(axis=2).std(axis=0),            # (M,)
        "axis_spread": cmean.std(axis=0).mean(axis=1),          # (M,)
    }


def build_table(stats, data, names, informed, chained, dates) -> pd.DataFrame:
    M, K = stats["mean"].shape
    ridge = stats["sum_spread"] / stats["axis_spread"]
    is_sota = (data.is_sota if data.is_sota is not None
               else np.zeros(M, bool))
    rows = []
    for m in range(M):
        for k in range(K):
            rows.append({
                "model": names[m], "axis": f"axis{k + 1}",
                "n_obs": int(data.n_obs_per_model[m]),
                "is_human": bool(data.is_human[m]),
                "is_sota": bool(is_sota[m]),
                "is_low_obs": bool(data.is_low_obs[m]),
                "chained": bool(chained[m]),
                "informed": bool(informed[m, k]),
                "release_date": dates.get(names[m], pd.NaT),
                "mean": stats["mean"][m, k], "median": stats["median"][m, k],
                "sd": stats["sd"][m, k],
                "chain_split": bool(stats["chain_split"][m, k]),
                "outside_50": bool(stats["outside_50"][m, k]),
                "outside_50_central": bool(stats["outside_50_central"][m, k]),
                "lump_lo": stats["lump_lo"][m, k],
                "lump_hi": stats["lump_hi"][m, k],
                "hi_share": stats["hi_share"][m, k],
                "sum_spread": stats["sum_spread"][m],
                "axis_spread": stats["axis_spread"][m],
                "ridge_ratio": ridge[m],
            })
    df = pd.DataFrame(rows)
    df["lump_gap"] = df["lump_hi"] - df["lump_lo"]
    return df


# ── figures ─────────────────────────────────────────────────
def _hi_chains(chain_means) -> set:
    """Chains above the largest gap in the sorted per-chain means."""
    order = np.argsort(chain_means)
    cut = int(np.diff(chain_means[order]).argmax())
    return set(int(c) for c in order[cut + 1:])


def gap_heatmap_fig(df, K, kind: str = "test-takers") -> go.Figure:
    """Who splits and where: rows are the flagged takers by descending widest gap,
    columns the axes, cell the distance between the two lumps. A blank cell is an
    axis whose chains agree."""
    flagged = set(df.loc[df["chain_split"], "model"])
    piv = (df[df["model"].isin(flagged)]
           .pivot(index="model", columns="axis", values="lump_gap")
           .reindex(columns=[f"axis{k + 1}" for k in range(K)]))
    piv = piv.loc[piv.max(axis=1).sort_values().index]      # plotly draws y upward
    # Log color: the human tiers sit at gaps of 6 to 8 and every machine taker
    # between 0.9 and 2.7, so a linear scale paints 101 of 108 rows one colour.
    # Ticks carry the real gap, so nothing is hidden.
    ticks = [1, 1.5, 2, 3, 5, 8]
    fig = go.Figure(go.Heatmap(
        z=np.log10(piv.values), customdata=piv.values,
        x=list(piv.columns), y=list(piv.index), colorscale="Viridis",
        colorbar=dict(title="lump<br>gap", tickvals=np.log10(ticks),
                      ticktext=[str(t) for t in ticks]),
        hovertemplate="%{y}<br>%{x}<br>lump gap %{customdata:.2f}<extra></extra>"))
    fig.update_layout(
        template="plotly_white", height=max(320, 15 * len(piv) + 150),
        title=f"Lump gap, the {len(piv)} {kind} with a split "
              "(blank = chains agree)",
        xaxis_title="latent axis", margin=dict(l=290, r=20, t=60, b=45))
    fig.update_yaxes(tickfont=dict(size=9))
    return fig


def dropdown_hist_fig(units, col_titles, title, xtitle, bins=40,
                      height=440) -> go.Figure:
    """One dropdown entry per unit. Each panel gets two pre-binned bar traces,
    the draws of the chains in the lower lump and those in the upper lump, so the
    page carries histogram counts instead of millions of raw draws."""
    ncols = len(col_titles)
    per_unit = 2 * ncols
    fig = make_subplots(rows=1, cols=ncols, subplot_titles=col_titles,
                        horizontal_spacing=0.05)
    # legend-only traces: the real ones are too many to label individually
    for col, lab in ((C_LOW, "lower lump"), (C_HIGH, "upper lump"),
                     (C_ONE, "one lump (chains agree)")):
        fig.add_trace(go.Bar(x=[None], y=[None], marker_color=col, name=lab,
                             showlegend=True), row=1, col=1)
    for i, u in enumerate(units):
        for c, (draws, split) in enumerate(u["panels"]):
            edges = np.histogram_bin_edges(draws, bins)
            mid = (edges[:-1] + edges[1:]) / 2
            wide = float(edges[1] - edges[0])
            hi = _hi_chains(draws.mean(axis=1)) if split else set()
            lo_rows = [r for r in range(draws.shape[0]) if r not in hi]
            for rows_, col in ((lo_rows, C_LOW if split else C_ONE),
                               (sorted(hi), C_HIGH)):
                y = (np.histogram(draws[rows_].ravel(), edges)[0]
                     if rows_ else np.zeros(len(mid), int))
                fig.add_trace(go.Bar(
                    x=mid, y=y, width=wide, marker_color=col, marker_line_width=0,
                    showlegend=False, visible=(i == 0),
                    hovertemplate="%{x:.2f}<br>%{y} draws<extra></extra>",
                ), row=1, col=c + 1)
    n_real = len(units) * per_unit
    buttons = [dict(
        label=u["label"], method="update",
        # flat key: a nested {"title": {...}} would replace the whole title
        # object and drop its container anchoring
        args=[{"visible": [True] * 3 + [j // per_unit == i for j in range(n_real)]},
              {"title.text": f"{title}: {u['label']}"}])
        for i, u in enumerate(units)]
    fig.update_xaxes(title_text=xtitle, autorange=True)
    fig.update_yaxes(title_text="draws", col=1)
    # stacked above the plot area: title (container-anchored), dropdown, legend
    fig.update_layout(
        template="plotly_white", height=height, barmode="overlay", bargap=0,
        title=dict(text=f"{title}: {units[0]['label']}", yref="container",
                   y=0.97, yanchor="top", x=0, xref="paper"),
        legend=dict(orientation="h", y=1.22, yanchor="bottom", x=0),
        margin=dict(l=60, r=20, t=185, b=55),
        updatemenus=[dict(buttons=buttons, x=0, xanchor="left", y=1.55,
                          yanchor="top", showactive=True)])
    return fig


def _units(df, key, draws_of, K, per_axis: bool):
    """Dropdown entries, widest gap first. `per_axis=False` gives one entry per
    unit with all K axes side by side; True gives one entry per flagged
    unit-axis pair with that single axis."""
    flagged = df[df["chain_split"]].sort_values("lump_gap", ascending=False)
    out = []
    if per_axis:
        for _, r in flagged.iterrows():
            k = int(r["axis"][4:]) - 1
            out.append({"label": f"{r[key]} | {r['axis']} | gap {r['lump_gap']:.2f}",
                        "panels": [(draws_of(r[key], k), True)]})
        return out
    for name in flagged[key].drop_duplicates():
        sub = df[df[key] == name].sort_values("axis")
        split = sub["chain_split"].values
        axes = ",".join(f"axis{k + 1}" for k in range(K) if split[k])
        out.append({
            "label": f"{name} | {axes} | gap {sub['lump_gap'].max():.2f}",
            "panels": [(draws_of(name, k), bool(split[k])) for k in range(K)]})
    return out


CSS = """
/* background pinned: a viewer in dark mode would otherwise put dark text on a
   dark ground while the plotly_white figures stay white */
html,body{background:#fff}
body{font:15px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;
 max-width:1180px;margin:2rem auto;padding:0 1.4rem;color:#1c1c1c}
h1{font-size:1.4rem;margin-bottom:.2rem}
p.sub{color:#555;font-size:13px;margin-top:0}
figure{margin:2.6rem 0}
"""


def build_html(figs, subtitle, bench_note) -> str:
    body = [pio.to_html(figs["heatmap"], full_html=False, include_plotlyjs=False),
            pio.to_html(figs["models"], full_html=False, include_plotlyjs=False)]
    body.append(pio.to_html(figs["benchmarks"], full_html=False,
                            include_plotlyjs=False)
                if figs.get("benchmarks") is not None else f"<p>{bench_note}</p>")
    return "\n".join(
        ['<meta charset="utf-8">', f"<style>{CSS}</style>",
         f"<script>{get_plotlyjs()}</script>",
         "<h1>Where the chains disagree</h1>",
         f'<p class="sub">{subtitle}</p>']
        + [f"<figure>{b}</figure>" for b in body])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", default=str(DEFAULT_TRACE),
                    help="path to the trace .nc (default: the flagship K=4 fit)")
    args = ap.parse_args()

    print(f"Loading {args.trace} ...", flush=True)
    idata = _open_posterior(args.trace)
    post = idata.posterior
    theta = post["theta"].values                      # (chain, draw, M, K)
    C, Dn, M, K = theta.shape
    print(f"  theta {theta.shape}")

    data = _load_matching_data(idata)
    names = data.mlookup.sort_values("model_idx")["model"].tolist()
    assert len(names) == M, f"{len(names)} data models vs {M} trace rows"

    informed = mirt_informed_mask(theta.reshape(C * Dn, M, K), SD_CAP)
    ls = build_lineage_structure(data.mlookup.sort_values("model_idx"))
    chained = np.zeros(M, bool)
    if ls is not None:
        chained[ls.row_idx] = True
    dates, _ = _release_dates(pd.read_csv(PROCESSED_FILE, parse_dates=["release_date"]))

    stats = taker_axis_stats(theta)
    df = build_table(stats, data, names, informed, chained, dates)

    out_dir = Path(args.trace).parent
    csv_path = out_dir / "theta_bimodality.csv"
    df.to_csv(csv_path, index=False)

    # The SAME scan on the loading matrix: A is (chain, draw, bench, K), the
    # identical shape family, so the taker scan applies unchanged.
    A = post["A"].values
    bnames = list(post["bench"].values)
    astats = taker_axis_stats(A)
    adf = pd.DataFrame([
        {"benchmark": bnames[b], "axis": f"axis{k + 1}",
         "chain_split": bool(astats["chain_split"][b, k]),
         "lump_gap": astats["lump_hi"][b, k] - astats["lump_lo"][b, k]}
        for b in range(len(bnames)) for k in range(K)])
    n_bench_split = int(adf["chain_split"].sum())

    idx = {n: i for i, n in enumerate(names)}
    bidx = {n: i for i, n in enumerate(bnames)}
    figs = {
        "heatmap": gap_heatmap_fig(df, K),
        "models": dropdown_hist_fig(
            _units(df, "model", lambda n, k: theta[:, ::FIG_STRIDE, idx[n], k],
                   K, per_axis=False),
            [f"axis{k + 1}" for k in range(K)],
            "Ability draws by chain lump", "theta"),
        "benchmarks": (dropdown_hist_fig(
            _units(adf, "benchmark", lambda n, k: A[:, ::FIG_STRIDE, bidx[n], k],
                   K, per_axis=True),
            [""], "Loading draws by chain lump", "loading A")
            if n_bench_split else None),
    }
    mp = modes_path(args.trace)
    modes = json.loads(mp.read_text()) if mp.exists() else None
    n_ch = modes["n_chains"] if modes else C
    n_mode = len(modes["modes"]) if modes else "?"
    html_path = out_dir / "bimodality.html"
    html_path.write_text(build_html(
        figs,
        f"{Path(args.trace).name} | K={K}, {n_ch} chains, {n_mode} posterior mode, "
        f"{M} test-takers, {len(bnames)} benchmarks. A split = the chains "
        f"disagree in two lumps (gap &gt; {GAP_SDS:g}x the median within-chain "
        f"sd, at least {MIN_SIDE} chains a side).",
        f"No benchmark loading splits: 0 of {len(bnames) * K} benchmark-axis "
        "cells flagged."))

    # stdout summary
    worst = (df[df["chain_split"]].groupby("model")["lump_gap"].max()
             .sort_values(ascending=False))
    per = df.groupby("model").agg(split=("chain_split", "any"),
                                  n_obs=("n_obs", "first"))
    print(f"\n  test-takers {M} | axes {K} | chains {C}")
    print("  axis    chain_split   outside_50(HDI)   outside_50(central)")
    for k in range(K):
        print(f"  axis{k + 1}   {int(stats['chain_split'][:, k].sum()):9d}"
              f"   {int(stats['outside_50'][:, k].sum()):15d}"
              f"   {int(stats['outside_50_central'][:, k].sum()):19d}")
    hum = set(df.loc[df["is_human"], "model"])
    print(f"\n  top 8 lump gaps (machine test-takers only; "
          f"{len(set(worst.index) & hum)} human tiers also split):")
    for name, gap in worst[~worst.index.isin(hum)].head(8).items():
        m = names.index(name)
        ax = ",".join(f"a{k + 1}" for k in range(K)
                      if stats["chain_split"][m, k])
        print(f"    {name:42s} gap {gap:5.2f}  split on {ax}  "
              f"n_obs {int(data.n_obs_per_model[m])}  "
              f"ridge {stats['sum_spread'][m] / stats['axis_spread'][m]:.3f}")
    if n_bench_split:
        print(f"\n  benchmark loadings: {n_bench_split} of {len(bnames) * K} "
              "benchmark-axis cells split. Widest:")
        for _, r in (adf[adf["chain_split"]].sort_values("lump_gap", ascending=False)
                     .head(8).iterrows()):
            print(f"    {r['benchmark']:42s} {r['axis']}  gap {r['lump_gap']:.2f}")
    else:
        print(f"\n  benchmark loadings: no split "
              f"(0 of {len(bnames) * K} benchmark-axis cells)")
    sp = per[per["split"]]
    ns = per[~per["split"]]
    rr = df.groupby("model")["ridge_ratio"].first()
    print(f"\n  median ridge_ratio: split {rr[sp.index].median():.3f} "
          f"(n={len(sp)})  non-split {rr[ns.index].median():.3f} (n={len(ns)})")
    print(f"  median n_obs: split {sp['n_obs'].median():.0f}  "
          f"non-split {ns['n_obs'].median():.0f}")
    print(f"\n  {csv_path}  ({len(df)} rows)")
    print(f"  {html_path}  ({html_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
