"""Per-test-taker theta bimodality on a fitted MIRT trace — no re-sampling.

`diagnose_chains.py` answers "are the chains one solution?" and reads only
rotation-invariant quantities (logp, D, chain-mean loadings A). On the flagship
K=4 fit it answers ONE mode, all 12 chains. This script answers the different
question that the fit-level verdict cannot see: is an individual test-taker's
per-axis theta a well-summarized quantity?

Most of the apparent disagreement is an AXIS PERMUTATION. The chains find the
same four axes and file them in different raw columns. `axis_permutations`
recovers each chain's filing by matching its mean loading columns to the
all-chain grand mean, and everything downstream runs on the aligned draws. That
takes the split takers from 145 to 29 of 835.

What survives the alignment is the real disagreement: a small chain subset docks
the human tiers on a different axis. That is what the page reports.

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
  results/<fit>/theta_bimodality.csv   — one row per taker x axis, the whole
                                         scan (machines and humans, every axis),
                                         on the ALIGNED draws.
  results/<fit>/bimodality.html        — four sections, no prose. Chain
                                         alignment: which raw slot each chain
                                         filed each common axis in. One human
                                         story: the tier profiles, per chain and
                                         summarized over the two chain groups.
                                         What still splits: the takers left over
                                         after alignment, and how the residual
                                         chains differ. Forecast, aligned: the
                                         frontier trend per axis against the
                                         human targets of both groups, the
                                         record distributions, and the crossover
                                         dates. The CSV and the stdout summary
                                         carry the whole diagnostic.

Axes are named axis1..axisK. The trace carries no `mirt_axis_names`, so any
semantic name would be invented.

Run:
  python 3_diagnostics/theta_bimodality.py
  python 3_diagnostics/theta_bimodality.py \
      --trace results/mirt/trace_mirt_k2.nc
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from multiaxis_eci.analysis.forecast import mirt_crossover_df, mirt_frontier_forecast  # noqa: E402
from multiaxis_eci.analysis.stats import _release_dates  # noqa: E402
from multiaxis_eci.analysis.timelines import mirt_informed_mask, mirt_model_timeline_df  # noqa: E402
from multiaxis_eci.config import FORECAST_KW  # noqa: E402
from multiaxis_eci.data import PROCESSED_FILE  # noqa: E402
from multiaxis_eci.scripts import load as _load_script  # noqa: E402
_dc = _load_script("3_diagnostics/diagnose_chains.py")
_load_matching_data, modes_path = _dc._load_matching_data, _dc.modes_path
from multiaxis_eci.lineage import build_lineage_structure  # noqa: E402

from multiaxis_eci.analysis import FLAGSHIP_TRACE  # noqa: E402

DEFAULT_TRACE = FLAGSHIP_TRACE

# a chain-mean gap this many within-chain sds wide separates two lumps
GAP_SDS = 3.0
# min chains on each side of the gap — 1 chain apart is an outlier, not a lump
MIN_SIDE = 2
# posterior SD below which an ability counts as data-informed
SD_CAP = 0.4
# draw stride for the figure histograms: 600 draws per chain resolve two lumps,
# and the page ships binned COUNTS, so the stride costs smoothness, not weight
FIG_STRIDE = 10

# Okabe-Ito, colorblind-safe: the chains that are not the subject of a figure
C_ONE = "#7f7f7f"
# one colour per chain, the same in every figure on the page
CHAIN_COLORS = px.colors.qualitative.Safe
# one colour per chain SUBSET (a group of chains, never a single chain). After
# alignment there are exactly two: the majority and the chains that still differ.
# Okabe-Ito blue and reddish-purple, the same pair in every figure on the page.
SUB_POOL, SUB_MAJ, SUB_MIN = "#000000", "#0072B2", "#CC79A7"
# the four tiers drawn in the static tier panel
PANEL_TIERS = ["Average Human", "Skilled Generalist", "Domain Expert",
               "Top Performer"]
# the two tiers drawn as forecast targets, and the colour that names each. The
# chain group is the dash there (solid majority, dashed minority), so the tier
# cannot also be one.
TARGET_TIERS = {"Domain Expert": "#009E73", "Top Performer": "#D55E00"}
# stride for the loading-column match: 500 draws per chain already give a
# chain-mean loading matrix stable to 3 decimals
PERM_STRIDE = 40
# the permutation table this fit is known to produce, printed against the
# computed one. A mismatch is reported and the COMPUTED table is used.
KNOWN_PERMS = [(0, 2, 3, 1), (0, 1, 2, 3), (0, 3, 1, 2), (0, 1, 2, 3),
               (0, 1, 3, 2), (2, 1, 0, 3), (0, 3, 2, 1), (1, 0, 2, 3),
               (3, 2, 1, 0), (1, 3, 2, 0)]
# the forecast the dashboard draws: records only, fit from the reasoning-model
# cutoff, every fitted model measured on the axis (no SOTA exemption)
FC_KW = dict(FORECAST_KW, sota_exempt=False)


def _open_posterior(path):
    """The posterior group only. The flagship trace is 20 GB on disk and theta is
    2.4 GB of it; `az.from_netcdf` would pull every other variable too."""
    import xarray as xr
    return az.InferenceData(posterior=xr.open_dataset(path, group="posterior"))


def chain_split_flags(cmean, csd):
    """The two-lump flag from the per-chain means (C, M, K) and the median
    within-chain sd (M, K). Returns (split, sorted means, cut index), the cut
    being the position of the widest eligible gap.

    Split out of `taker_axis_stats` so the raw-column split count can be taken
    before alignment without paying for the quantiles and the HDIs."""
    C = cmean.shape[0]
    srt = np.sort(cmean, axis=0)                # (C, M, K)
    gaps = np.diff(srt, axis=0)                 # (C-1, M, K)
    # only cuts leaving >= MIN_SIDE chains per side are eligible
    ok = np.zeros(C - 1, bool)
    ok[MIN_SIDE - 1:C - MIN_SIDE] = True
    cut = np.where(ok[:, None, None], gaps, -np.inf).argmax(axis=0)   # (M, K)
    gap = np.take_along_axis(gaps, cut[None], axis=0)[0]
    return ok[cut] & (gap > GAP_SDS * csd), srt, cut


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

    split, srt, cut = chain_split_flags(cmean, csd)
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


# ── r-hat ───────────────────────────────────────────────────
def _rhat(x) -> np.ndarray:
    """r-hat over a (chain, draw, ...) array, returned with the trailing shape.

    `az.rhat` takes a bare ndarray only when it is (chain, draw); anything wider
    has to go through a dataset first."""
    return np.asarray(az.rhat(az.convert_to_dataset(np.asarray(x)))["x"])


def _ess(x) -> np.ndarray:
    """Bulk (rank-normalized) ESS over a (chain, draw, ...) array, same wrapping
    reason as `_rhat`: az.ess takes a bare ndarray only when it is (chain, draw)."""
    return np.asarray(az.ess(az.convert_to_dataset(np.asarray(x)))["x"])


def cell_ess(theta, cells, majority, minority) -> dict:
    """Bulk ESS per (model, axis) cell: all chains, the majority only, the
    minority only.

    On the FULL draws, never the figure stride: ESS counts how many independent
    draws the chains are worth, and striding would answer a different question.
    A subset holds fewer raw draws than the pooled set, so the three numbers are
    comparable only after the per-1,000-draw normalization the summary prints.

    The cell is sliced BEFORE the chain subset: `theta[majority]` would copy a
    4 GB slab, `theta[:, :, m, k]` is 1.6 MB."""
    out = {}
    for m, k in cells:
        x = theta[:, :, m, k]                     # (C, draw), one cell
        out[(m, k)] = (float(_ess(x)), float(_ess(x[majority])),
                       float(_ess(x[minority])))
    return out


def _rhat_chunked(x, chunk: int = 100) -> np.ndarray:
    """r-hat over a (chain, draw, unit, ...) array, `chunk` units at a time.

    One call on the whole of theta would rank-transform 5.3 GB into fresh
    float64 copies and exhaust the machine. Chunking bounds the temporaries and
    changes no number: r-hat is per cell."""
    return np.concatenate([_rhat(x[:, :, s:s + chunk])
                           for s in range(0, x.shape[2], chunk)])


def eta_rhat_scan(theta, A, D, model_idx, bench_idx, block: int = 250):
    """r-hat of eta = sum_k A_bk theta_mk - D_b, one value per observation.

    eta is the identified quantity: it is what the likelihood sees, so it is
    invariant along the axis trade-off ridge that splits theta. Built blockwise
    at float32, one axis at a time, to keep every temporary near 200 MB."""
    K = theta.shape[3]
    out = np.empty(len(model_idx))
    for s in range(0, len(model_idx), block):
        m_idx, b_idx = model_idx[s:s + block], bench_idx[s:s + block]
        eta = -D[:, :, b_idx].astype(np.float32)
        for k in range(K):
            eta += (A[:, :, b_idx, k].astype(np.float32)
                    * theta[:, :, m_idx, k].astype(np.float32))
        out[s:s + block] = _rhat(eta)
    return out


# ── axis alignment ──────────────────────────────────────────
def axis_permutations(A, stride: int = PERM_STRIDE):
    """Per chain, the raw column each COMMON axis was filed in.

    The chains agree on WHICH four axes exist and disagree on the order they sit
    in, which is the label-switching a rotation-free non-negative loading prior
    leaves free. Each chain's mean loading column is matched to the all-chain
    grand-mean columns by correlation over the benchmarks, and the best of the
    K! orderings wins. K=4 gives 24 orderings, so the brute force is free.

    Convention: common axis k lives in raw slot perm[k], so the aligned array is
    `x[c][..., list(perm)]`.

    Returns (perms, matched, identity): the per-chain permutation, its mean
    matched correlation, and the mean correlation the raw order would give."""
    cm = A[:, ::stride].mean(axis=1)                      # (C, B, K)
    z = (cm - cm.mean(axis=1, keepdims=True)) / (cm.std(axis=1, keepdims=True) + 1e-12)
    g = z.mean(axis=0)                                    # (B, K) grand mean
    g = (g - g.mean(axis=0)) / (g.std(axis=0) + 1e-12)
    K = cm.shape[2]
    perms, matched, identity = [], [], []
    for c in range(cm.shape[0]):
        corr = z[c].T @ g / z.shape[1]                    # (raw slot, common axis)
        best = max(itertools.permutations(range(K)),
                   key=lambda p: sum(corr[p[k], k] for k in range(K)))
        perms.append(best)
        matched.append(float(np.mean([corr[best[k], k] for k in range(K)])))
        identity.append(float(np.mean(np.diag(corr))))
    return perms, np.array(matched), np.array(identity)


def apply_permutations(x, perms) -> None:
    """Reorder the axis dimension of a (chain, draw, unit, K) array in place.

    One chain at a time: a whole-array fancy index would double the 5 GB theta."""
    for c, p in enumerate(perms):
        if list(p) != list(range(len(p))):
            x[c] = x[c][:, :, list(p)]


def residual_groups(chain_means, is_human):
    """The chains that still disagree about the humans after alignment.

    Per chain, the mean absolute distance of its human-tier profile from the
    cross-chain median profile, cut at the largest gap in that scalar. Returns
    (majority, minority) chain lists."""
    prof = chain_means[:, is_human, :].reshape(chain_means.shape[0], -1)
    d = np.abs(prof - np.median(prof, axis=0)).mean(axis=1)
    minority = sorted(_hi_chains(d))
    return [c for c in range(len(d)) if c not in minority], minority


# ── figures ─────────────────────────────────────────────────
def _chain_color(c: int) -> str:
    return CHAIN_COLORS[c % len(CHAIN_COLORS)]


def _rgba(color: str, alpha: float) -> str:
    """'#rrggbb' -> 'rgba(r,g,b,a)'. A plotly fill carries its opacity in the
    colour string; trace-level `opacity` would fade the outline too."""
    r, g, b = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def _chain_hist_traces(draws, bins: int = 60, showlegend: bool = False):
    """One step-line histogram per chain over shared bin edges.

    Lines and not bars: ten overlaid chains stay readable, and the page ships
    `bins` points per chain instead of the raw draws."""
    draws = np.asarray(draws)
    edges = np.histogram_bin_edges(draws, bins)
    mid = (edges[:-1] + edges[1:]) / 2
    return [go.Scatter(
        x=mid, y=np.histogram(row, edges)[0], mode="lines",
        line=dict(color=_chain_color(c), width=1.3, shape="hvh"),
        name=f"chain {c}", legendgroup=f"chain{c}", showlegend=showlegend,
        hovertemplate=f"chain {c}<br>%{{x:.2f}}<br>%{{y}} draws<extra></extra>")
        for c, row in enumerate(draws)]


def _hi_chains(chain_means) -> set:
    """Chains above the largest gap in the sorted per-chain means."""
    order = np.argsort(chain_means)
    cut = int(np.diff(chain_means[order]).argmax())
    return set(int(c) for c in order[cut + 1:])


def _ess_tag(e) -> str:
    """The three ESS numbers as they appear in every dropdown label."""
    return f"ESS pooled {e[0]:.0f} / maj {e[1]:.0f} / min {e[2]:.0f}"


def lump_dist_fig(theta, entries, essmap, names, majority, minority,
                  bins: int = 60) -> go.Figure | None:
    """The two lumps as distributions, one flagged cell in the dropdown.

    `entries` are (row, axis, note). Each group's draws are pooled across its own
    chains and drawn as a filled density; the all-chain density sits faintly
    behind, so the reader sees the shape pooling produces from the two.

    Density and not counts: the majority holds four times the minority's draws,
    so counts would compare group sizes instead of locations."""
    if not entries:
        return None
    maj_lab = f"majority ({len(majority)} chains)"
    min_lab = "minority (chains " + ", ".join(map(str, minority)) + ")"
    groups = [(list(range(theta.shape[0])), C_ONE, "pooled", "all chains pooled",
               0.18),
              (list(majority), SUB_MAJ, "maj", maj_lab, 0.40),
              (list(minority), SUB_MIN, "min", min_lab, 0.40)]
    title = "The two lumps, mode by mode"
    fig = go.Figure()
    # legend proxies: three entries for the whole figure, not three per cell
    for _, color, grp, lab, alpha in groups:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines", name=lab, legendgroup=grp,
            fill="toself", fillcolor=_rgba(color, alpha),
            line=dict(color=color, width=1.4), hoverinfo="skip"))
    labels = []
    for i, (m, k, note) in enumerate(entries):
        draws = theta[:, ::FIG_STRIDE, m, k]
        edges = np.histogram_bin_edges(draws, bins)
        mid = (edges[:-1] + edges[1:]) / 2
        dens = [np.histogram(draws[rows].ravel(), edges, density=True)[0]
                for rows, *_ in groups]
        top = max(d.max() for d in dens) * 1.03
        for (rows, color, grp, lab, alpha), y in zip(groups, dens):
            fig.add_trace(go.Scatter(
                x=mid, y=y, mode="lines", fill="tozeroy",
                line=dict(color=color, width=1.4, shape="hvh"),
                fillcolor=_rgba(color, alpha), name=lab, legendgroup=grp,
                showlegend=False, visible=(i == 0),
                hovertemplate=f"{lab}<br>%{{x:.2f}}<br>density %{{y:.2f}}"
                              "<extra></extra>"))
        for rows, color, grp, lab, _ in groups[1:]:      # the two mode means
            mu = float(draws[rows].mean())
            fig.add_trace(go.Scatter(
                x=[mu, mu], y=[0, top], mode="lines", name=lab, legendgroup=grp,
                showlegend=False, visible=(i == 0),
                line=dict(color=color, width=1.2, dash="dash"),
                hovertemplate=f"{lab} mean {mu:.2f}<extra></extra>"))
        labels.append(f"{names[m]} · axis {k + 1}{note} · "
                      f"{_ess_tag(essmap[(m, k)])}")
    per = len(groups) + 2                # 3 densities + 2 mean lines per cell
    n_real = len(entries) * per
    buttons = [dict(
        label=lab, method="update",
        args=[{"visible": [True] * len(groups)
                          + [j // per == i for j in range(n_real)]},
              {"title.text": f"{title}: {lab}"}])
        for i, lab in enumerate(labels)]
    fig.update_layout(
        template="plotly_white", height=470,
        title=dict(text=f"{title}: {labels[0]}", yref="container", y=0.97,
                   yanchor="top", x=0, xref="paper"),
        xaxis_title="theta", yaxis_title="density",
        legend=dict(orientation="h", y=1.18, yanchor="bottom", x=0,
                    font=dict(size=11)),
        margin=dict(l=70, r=20, t=185, b=55),
        updatemenus=[dict(buttons=buttons, active=0, x=0, xanchor="left",
                          y=1.5, yanchor="top", showactive=True)])
    return fig


def ess_scatter_fig(essmap, split_cells, rec_cells, names) -> go.Figure | None:
    """Pooled ESS against within-majority ESS, one point per cell.

    Every split cell sits far above the y=x line: the 8 majority chains alone
    are worth two to three orders of magnitude more effective draws than all 10
    together, because pooling two lumps inflates the between-chain variance ESS
    divides by. Record cells straddle the line and are the honest control: the
    axis 1 and axis 2 records sit on it, most axis 3 and axis 4 records sit
    above it, so the loss is not confined to the flagged cells."""
    series = [("split cells", split_cells, SUB_MIN, 8),
              ("record cells", rec_cells, C_ONE, 6)]
    series = [(lab, [c for c in cells if c in essmap], col, sz)
              for lab, cells, col, sz in series]
    if not any(cells for _, cells, _, _ in series):
        return None
    fig = go.Figure()
    vals = [v for _, cells, _, _ in series for c in cells
            for v in essmap[c][:2]]
    lim = [min(vals) * 0.8, max(vals) * 1.25]
    fig.add_trace(go.Scatter(
        x=lim, y=lim, mode="lines", name="y = x",
        line=dict(color="#999", width=1, dash="dash"), hoverinfo="skip"))
    for lab, cells, color, size in series:
        fig.add_trace(go.Scatter(
            x=[essmap[c][0] for c in cells], y=[essmap[c][1] for c in cells],
            mode="markers", name=f"{lab} ({len(cells)})",
            text=[f"{names[m]} · axis {k + 1}" for m, k in cells],
            marker=dict(size=size, color=color, opacity=0.85,
                        line=dict(width=0.5, color="#fff")),
            hovertemplate="%{text}<br>pooled ESS %{x:.0f}"
                          "<br>majority ESS %{y:.0f}<extra></extra>"))
    fig.update_xaxes(type="log", title_text="bulk ESS, all chains pooled",
                     range=list(np.log10(lim)))
    fig.update_yaxes(type="log", title_text="bulk ESS, majority chains only",
                     range=list(np.log10(lim)))
    fig.update_layout(
        template="plotly_white", height=520,
        title=dict(text="Pooling two lumps costs effective samples",
                   yref="container", y=0.97, yanchor="top", x=0, xref="paper"),
        legend=dict(orientation="h", y=1.01, yanchor="bottom", x=0,
                    font=dict(size=11)),
        margin=dict(l=80, r=30, t=110, b=60))
    return fig


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


def centered_chain_means(chain_means, rows) -> np.ndarray:
    """(C, K) per-chain mean theta, centered per (taker, axis) on the chain
    average and then averaged over `rows`. Centering is what lets takers sitting
    at different ability levels be averaged together."""
    cm = chain_means[:, rows, :]
    return (cm - cm.mean(axis=0, keepdims=True)).mean(axis=1)


def chain_tier_fig(chain_means, names, is_human, K,
                   default: str = "Domain Expert") -> go.Figure | None:
    """One line per chain across the axes, for the human tier in the dropdown.

    The chains of a story dip together on that story's axis; a chain in no story
    stays flat. Ten legend-only proxies carry the chain colours, so the legend
    holds 10 entries and not 10 per tier."""
    tiers = [n for i, n in enumerate(names) if is_human[i]]
    if not tiers:
        return None
    C = chain_means.shape[0]
    x = [f"axis {j + 1}" for j in range(K)]
    d = tiers.index(default) if default in tiers else 0
    fig = go.Figure()
    for c in range(C):
        # x on a real category, y empty: an all-None x would type the axis
        # linear and the category traces would not draw
        fig.add_trace(go.Scatter(
            x=[x[0]], y=[None], mode="lines+markers", name=f"chain {c}",
            line=dict(color=_chain_color(c), width=1.4),
            marker=dict(size=8, color=_chain_color(c)), hoverinfo="skip"))
    for t, tier in enumerate(tiers):
        y = chain_means[:, names.index(tier), :]
        for c in range(C):
            fig.add_trace(go.Scatter(
                x=x, y=y[c], mode="lines+markers", showlegend=False,
                visible=(t == d),
                line=dict(color=_chain_color(c), width=1.4),
                marker=dict(size=8, color=_chain_color(c)),
                hovertemplate=f"chain {c}<br>%{{x}}<br>theta %{{y:.2f}}"
                              "<extra></extra>"))
    n_real = len(tiers) * C
    buttons = [dict(
        label=tier, method="update",
        args=[{"visible": [True] * C + [j // C == t for j in range(n_real)]},
              {"title.text": f"Per-chain ability profile: {tier}"}])
        for t, tier in enumerate(tiers)]
    fig.update_layout(
        template="plotly_white", height=470,
        title=dict(text=f"Per-chain ability profile: {tiers[d]}",
                   yref="container", y=0.97, yanchor="top", x=0, xref="paper"),
        xaxis_title="latent axis", yaxis_title="per-chain mean theta",
        legend=dict(orientation="h", y=1.18, yanchor="bottom", x=0,
                    font=dict(size=11)),
        margin=dict(l=75, r=20, t=185, b=55),
        updatemenus=[dict(buttons=buttons, active=d, x=0, xanchor="left",
                          y=1.5, yanchor="top", showactive=True)])
    return fig


def alignment_heatmap_fig(perms, matched, n_raw, n_aligned, M) -> go.Figure:
    """Which raw column each chain filed each common axis in.

    The cell text is the raw slot, the colour is that slot, so a row that is not
    0,1,2,3 is a chain whose columns are permuted. The row label carries the mean
    correlation the match reaches."""
    K = len(perms[0])
    z = np.array([list(p) for p in perms], float)
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"common axis {k + 1}" for k in range(K)],
        y=[f"chain {c} · r {matched[c]:.2f}" for c in range(len(perms))],
        zmin=0, zmax=K - 1, colorscale="Viridis",
        texttemplate="%{z:.0f}", textfont=dict(size=12),
        colorbar=dict(title="raw<br>slot", tickvals=list(range(K))),
        hovertemplate="%{y}<br>%{x}<br>raw slot %{z:.0f}<extra></extra>"))
    fig.update_layout(
        template="plotly_white", height=60 + 30 * len(perms) + 120,
        # a literal arrow: plotly title text is not HTML-entity decoded
        title=f"Axis filing per chain: splits {n_raw} → {n_aligned} "
              f"of {M} after alignment",
        xaxis_title="common axis", margin=dict(l=140, r=20, t=70, b=55))
    fig.update_yaxes(autorange="reversed")     # chain 0 on top, reading order
    return fig


def tier_panel_fig(chain_means, names, majority, minority, K) -> go.Figure | None:
    """Four tiers, each profiled over the axes twice: the majority chains and the
    chains that still differ. One panel per tier, shared y-range."""
    tiers = [t for t in PANEL_TIERS if t in names]
    if not tiers or not minority:
        return None
    x = [f"axis {j + 1}" for j in range(K)]
    fig = make_subplots(rows=1, cols=len(tiers), shared_yaxes=True,
                        horizontal_spacing=0.03, subplot_titles=tiers)
    for col, tier in enumerate(tiers, start=1):
        y = chain_means[:, names.index(tier), :]
        for chains, color, dash, lab in (
                (majority, SUB_MAJ, "solid", f"majority chains ({len(majority)})"),
                (minority, SUB_MIN, "dash",
                 "chains " + ", ".join(map(str, minority)))):
            fig.add_trace(go.Scatter(
                x=x, y=y[chains].mean(axis=0), mode="lines+markers", name=lab,
                legendgroup=lab, showlegend=(col == 1),
                line=dict(color=color, width=1.8, dash=dash),
                marker=dict(size=8, color=color),
                hovertemplate=f"{lab}<br>%{{x}}<br>theta %{{y:.2f}}"
                              "<extra></extra>"), row=1, col=col)
    fig.add_hline(y=0, line=dict(color="#999", dash="dash", width=1))
    fig.update_yaxes(title_text="mean theta over the group", col=1)
    fig.update_annotations(font_size=12)
    fig.update_layout(
        template="plotly_white", height=430,
        title=dict(text="Tier profiles, two chain groups", yref="container",
                   y=0.97, yanchor="top", x=0, xref="paper"),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0,
                    font=dict(size=11)),
        margin=dict(l=80, r=20, t=135, b=55))
    return fig


def residual_dots_fig(chain_means, row_sets, minority, K,
                      height: int = 360) -> go.Figure | None:
    """Where the residual chains park ability, against the other chains.

    One panel per row set (the human tiers, the takers that still split). Each
    dot is one chain's mean theta on one axis, centered per (taker, axis) on the
    chain average and averaged over the rows, so takers at different ability
    levels can share a panel."""
    row_sets = [(lab, rows) for lab, rows in row_sets if len(rows)]
    if not row_sets or not minority:
        return None
    C = chain_means.shape[0]
    x = [f"axis {j + 1}" for j in range(K)]
    others = [c for c in range(C) if c not in minority]
    fig = make_subplots(rows=1, cols=len(row_sets), shared_yaxes=True,
                        horizontal_spacing=0.05,
                        subplot_titles=[f"{lab} (n={len(r)})"
                                        for lab, r in row_sets])
    for col, (_, rows) in enumerate(row_sets, start=1):
        z = centered_chain_means(chain_means, rows)
        fig.add_trace(go.Scatter(
            x=[x[j] for _ in others for j in range(K)],
            y=[z[c, j] for c in others for j in range(K)],
            mode="markers", name=f"other {len(others)} chains",
            legendgroup="others", showlegend=(col == 1),
            marker=dict(size=6, color=C_ONE, opacity=0.7),
            hovertemplate="%{x}<br>%{y:+.2f}<extra></extra>"), row=1, col=col)
        for c in minority:
            fig.add_trace(go.Scatter(
                x=x, y=z[c], mode="markers", name=f"chain {c}",
                legendgroup=f"chain{c}", showlegend=(col == 1),
                marker=dict(size=11, color=_chain_color(c)),
                hovertemplate=f"chain {c}<br>%{{x}}<br>%{{y:+.2f}}"
                              "<extra></extra>"), row=1, col=col)
    fig.add_hline(y=0, line=dict(color="#333", dash="dash", width=1))
    fig.update_yaxes(title_text="centered mean theta", col=1)
    fig.update_annotations(font_size=12)
    fig.update_layout(
        template="plotly_white", height=height,
        title=dict(text="Chains " + ", ".join(map(str, minority))
                        + " against the rest", yref="container", y=0.96,
                   yanchor="top", x=0, xref="paper"),
        legend=dict(orientation="h", y=1.01, yanchor="bottom", x=0,
                    font=dict(size=11)),
        margin=dict(l=80, r=20, t=120, b=50))
    return fig


# ── forecast ────────────────────────────────────────────────
# The alignment is a nuisance only if it moves a published number. The published
# numbers are the per-axis frontier trend and the human-crossover dates, both
# read off a linear fit in release date. Both are drawn here on the aligned
# draws, with the human targets read on the same chain group as the trend, so a
# group that docks the tiers on a different axis reads a different date.


def _dstr(x):
    """Timestamps → 'YYYY-MM-DD' strings. Plotly + kaleido serialize a date
    axis from strings; raw Timestamps raise at write_image time."""
    return pd.to_datetime(x).strftime("%Y-%m-%d")


def forecast_probe(tk, data, raw_df, names, dates, subsets, label: str) -> dict:
    """One axis of `tk` (chain, draw, M, 1) forecast pooled, then refit on chain
    subsets over the SAME frozen record set.

    Freezing matters: the record set is picked by posterior means, so letting
    each subset re-pick it would compare two different regressions instead of
    the same regression read on two halves of the posterior. The big subset
    arrays live and die inside this call.

    `subsets` are (label, chains, colour, dash). Returns the pooled forecast,
    its record points and crossovers, and per subset a slope, a median line and
    its crossovers. Raises ValueError like any other forecast caller."""
    M = tk.shape[2]
    flat = tk.reshape(-1, M, 1)
    tl = mirt_model_timeline_df(flat, 0, data, raw_df, sd_cap=SD_CAP, hdi_prob=0.5)
    if tl.empty:
        raise ValueError(f"{label}: no dated informed model")
    kw = dict(FC_KW, back_start=pd.to_datetime(tl["release_date"]).min())
    fc = mirt_frontier_forecast(flat, 0, data, raw_df, **kw)
    frozen = fc.fit_names

    pos = {n: i for i, n in enumerate(names)}
    rec_x = [dates[n] for n in frozen]
    rec_y = [float(flat[:, pos[n], 0].mean()) for n in frozen]

    subs = []
    for slabel, chains, color, dash in subsets:
        sub = tk[list(chains)].reshape(-1, M, 1)
        sfc = mirt_frontier_forecast(sub, 0, data, raw_df, fit_names=frozen, **kw)
        subs.append({
            "label": slabel, "color": color, "dash": dash, "fc": sfc,
            "slope": float(np.median(sfc.slope)),
            "cx": mirt_crossover_df(sfc, sub, 0, data, axis_name=label,
                                    hdi_prob=0.5)})
        del sub
    return {"label": label, "fc": fc, "frozen": frozen,
            "slope": float(np.median(fc.slope)),
            "records": (rec_x, rec_y), "subs": subs,
            "cx": mirt_crossover_df(fc, flat, 0, data, axis_name=label,
                                    hdi_prob=0.5)}


def _pooled_crossover_base(probes, title):
    """Pooled median dot and 50% HDI whisker per (axis, tier) crossing row, plus
    the layout. Returns (figure, row labels); (None, []) when nothing crosses.
    Callers add their own subset markers on the same rows."""
    labs, med, lo, hi = [], [], [], []
    for p in probes:
        for _, r in p["cx"].iterrows():
            if r["status"] == "no_crossing":
                continue
            labs.append(f"{p['label']} · {r['tier']}")
            med.append(r["crossover_date_median"])
            lo.append(r["crossover_hdi_low"])
            hi.append(r["crossover_hdi_high"])
    if not labs:
        return None, []
    fig = go.Figure()
    wx, wy = [], []
    for lab, a, b in zip(labs, lo, hi):
        wx += [_dstr(a), _dstr(b), None]
        wy += [lab, lab, None]
    fig.add_trace(go.Scatter(x=wx, y=wy, mode="lines", name="pooled 50% HDI",
                             line=dict(color="#bbb", width=6), hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=_dstr(med), y=labs, mode="markers", name="pooled median",
        marker=dict(size=9, color=SUB_POOL, symbol="circle"),
        hovertemplate="%{y}<br>%{x}<extra></extra>"))
    fig.update_layout(
        template="plotly_white", height=max(360, 26 * len(labs) + 190),
        title=dict(text=title, yref="container", y=0.985, yanchor="top",
                   x=0, xref="paper"),
        xaxis_title="crossover date", margin=dict(l=250, r=20, t=115, b=55),
        legend=dict(orientation="h", y=1.01, yanchor="bottom", x=0))
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=10))
    return fig, labs


def mode_crossover_dotwhisker_fig(probes) -> go.Figure:
    """When the trend reaches each human tier, pooled and per chain group. Trend
    and target both come from the group's own draws, so a group that docks the
    tiers on another axis reads another date."""
    fig, labs = _pooled_crossover_base(
        probes, "Human crossover dates, pooled and per group")
    if fig is None:
        return None
    syms = ["triangle-up", "triangle-down"]
    kinds = {}
    for p in probes:
        for i, s in enumerate(p["subs"]):
            x, y = kinds.setdefault(
                s["label"], ([], [], s["color"], syms[i % len(syms)]))[:2]
            for _, r in s["cx"].iterrows():
                lab = f"{p['label']} · {r['tier']}"
                if r["status"] != "no_crossing" and lab in labs:
                    x.append(_dstr(r["crossover_date_median"]))
                    y.append(lab)
    for key, (x, y, color, sym) in kinds.items():
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers", name=key,
            marker=dict(size=8, color=color, symbol=sym,
                        line=dict(width=1, color=color)),
            hovertemplate="%{y}<br>%{x}<extra></extra>"))
    return fig


def aligned_forecast_fig(probes, chain_means, names, majority, minority,
                         K) -> go.Figure | None:
    """One panel per axis: the frozen records, the pooled frontier trend with its
    50% band, and the two top human tiers at the level each chain group gives
    them. The crossing sits where trend and target meet."""
    tiers = [t for t in TARGET_TIERS if t in names]
    if not probes or not tiers:
        return None
    ncols = min(2, len(probes))
    nrows = -(-len(probes) // ncols)
    fig = make_subplots(rows=nrows, cols=ncols, horizontal_spacing=0.07,
                        vertical_spacing=0.12,
                        subplot_titles=[p["label"] for p in probes])
    for i, p in enumerate(probes):
        row, col, first = i // ncols + 1, i % ncols + 1, (i == 0)
        fc, gx = p["fc"], _dstr(p["fc"].grid_dates)
        for y, name, fill in ((fc.hi, None, None),
                              (fc.lo, "pooled 50% band", "tonexty")):
            fig.add_trace(go.Scatter(
                x=gx, y=y, mode="lines", line=dict(width=0), fill=fill,
                fillcolor="rgba(0,0,0,0.10)", name=name or "band",
                legendgroup="band", showlegend=bool(first and name),
                hoverinfo="skip"), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=_dstr(p["records"][0]), y=p["records"][1], mode="markers",
            marker=dict(size=6, color="#888"), name="frozen records",
            legendgroup="rec", showlegend=first, text=p["frozen"],
            hovertemplate="%{text}<br>%{x}<br>theta %{y:.2f}<extra></extra>"),
            row=row, col=col)
        fig.add_trace(go.Scatter(
            x=gx, y=fc.median, mode="lines", name="pooled trend",
            legendgroup="trend", showlegend=first,
            line=dict(color=SUB_POOL, width=2.2),
            hovertemplate="%{x}<br>theta %{y:.2f}<extra></extra>"),
            row=row, col=col)
        for tier in tiers:
            for chains, dash, glab in (
                    (majority, "solid", f"majority chains ({len(majority)})"),
                    (minority, "dash",
                     "chains " + ", ".join(map(str, minority)))):
                lvl = float(chain_means[chains, names.index(tier), i].mean())
                lab = f"{tier} · {glab}"
                fig.add_trace(go.Scatter(
                    x=[gx[0], gx[-1]], y=[lvl, lvl], mode="lines", name=lab,
                    legendgroup=lab, showlegend=first,
                    line=dict(color=TARGET_TIERS[tier], width=1.5, dash=dash),
                    hovertemplate=f"{lab}<br>theta {lvl:.2f}<extra></extra>"),
                    row=row, col=col)
    fig.update_yaxes(title_text="theta", col=1)
    fig.update_annotations(font_size=12)
    fig.update_layout(
        template="plotly_white", height=370 * nrows + 150,
        title=dict(text="Frontier trend and human targets, aligned",
                   yref="container", y=0.98, yanchor="top", x=0, xref="paper"),
        legend=dict(y=1, yanchor="top", x=1.02, font=dict(size=11)),
        margin=dict(l=70, r=270, t=110, b=55))
    return fig


def record_entries(probes, names):
    """(axis, name, row) for every frontier record a trend is fit on. A record
    appears once per axis it sets, since the coordinate shown differs."""
    pos = {n: i for i, n in enumerate(names)}
    return [(k, n, pos[n]) for k, p in enumerate(probes) for n in p["frozen"]]


def record_dist_fig(theta, probes, names, dates, essmap,
                    default_axis: int = 2) -> go.Figure | None:
    """Per-chain theta of the frontier records, one record in the dropdown.

    The records are the models the trend is fit on, so they are the takers whose
    ability the forecast actually reads. None of them splits: the filled trace is
    every chain's draws pooled and it stays a single hump, with no valley between
    two lumps. The chain lines drawn on top of it are still OFFSET from each
    other, and that offset is what collapses the pooled ESS on axes 3 and 4 —
    on 28 of 46 record cells the within-majority ESS is more than 3x the pooled
    one. One hump is therefore a statement about shape, not about precision.
    The dropdown label carries the same three bulk ESS numbers the lump figure
    shows, computed on the full draws and not the strided ones the histogram
    bins."""
    if not probes:
        return None
    entries = record_entries(probes, names)
    if not entries:
        return None

    def released(n):
        t = pd.to_datetime(dates.get(n, pd.NaT))
        return pd.Timestamp.min if pd.isna(t) else t

    # default entry: the newest record on the axis the human stories hang on
    d = max(range(len(entries)),
            key=lambda i: (entries[i][0] == default_axis, released(entries[i][1])))

    C = theta.shape[0]
    title = "Records: one hump, chains offset inside it"
    fig = go.Figure()
    # legend proxies: one entry per chain plus the pooled shape, always visible.
    # The real traces carry the same legendgroups, so a click hides that chain in
    # whichever record is on screen, and the legend keeps C+1 entries instead of
    # C+1 per record.
    for c in range(C):
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines", name=f"chain {c}",
            legendgroup=f"chain{c}", line=dict(color=_chain_color(c), width=1.3),
            hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="lines", name="all chains combined",
        legendgroup="pooled", fill="toself", fillcolor="rgba(120,120,120,0.40)",
        line=dict(color="#4d4d4d", width=1.6), hoverinfo="skip"))
    peaks = None
    for i, (k, _, m) in enumerate(entries):
        draws = theta[:, ::FIG_STRIDE, m, k]
        # same edges `_chain_hist_traces` picks, so the pooled area and the chain
        # lines share bins
        edges = np.histogram_bin_edges(draws, 60)
        y = np.histogram(draws.ravel(), edges)[0]
        # pooled first: the chain lines are drawn on top of the filled area
        fig.add_trace(go.Scatter(
            x=(edges[:-1] + edges[1:]) / 2, y=y, mode="lines",
            line=dict(color="#4d4d4d", width=1.6, shape="hvh"),
            fill="tozeroy", fillcolor="rgba(120,120,120,0.40)",
            name="all chains combined", legendgroup="pooled", showlegend=False,
            visible=(i == d),
            hovertemplate="all chains<br>%{x:.2f}<br>%{y} draws<extra></extra>"))
        for tr in _chain_hist_traces(draws):
            tr.visible = (i == d)
            fig.add_trace(tr)
        if i == d:
            # Modes of a SMOOTHED density, not of the raw bin counts: 60-bin
            # counts jitter enough to invent a second local maximum on a shape
            # that has no valley, which is what the figure's claim is about.
            flat = draws.ravel()
            dens = gaussian_kde(flat[::5])(np.linspace(flat.min(), flat.max(), 400))
            peaks = int(((dens[1:-1] >= dens[:-2]) & (dens[1:-1] > dens[2:])
                         & (dens[1:-1] > 0.1 * dens.max())).sum())
    labels = [f"axis {k + 1} · {n} · {_ess_tag(essmap[(m, k)])}"
              for k, n, m in entries]
    print(f"  record cells {len(entries)} | default '{labels[d]}' pooled "
          f"local maxima >10% of peak: {peaks}")
    print("  axis  record                                     release     "
          "ESS pooled      maj      min")
    for k, n, m in sorted(entries, key=lambda t: (t[0], released(t[1]))):
        rel = released(n)
        rel = "" if rel == pd.Timestamp.min else rel.strftime("%Y-%m-%d")
        p, a, b = essmap[(m, k)]
        print(f"    axis{k + 1}  {n:42s} {rel:10s} {p:10.0f} {a:8.0f} {b:8.0f}")
    n_proxy = C + 1                   # C chain proxies + the pooled one
    per = C + 1                       # per entry: 1 pooled area + C chain lines
    n_real = len(entries) * per
    buttons = [dict(
        label=lab, method="update",
        args=[{"visible": [True] * n_proxy
                          + [j // per == i for j in range(n_real)]},
              {"title.text": f"{title}: {lab}"}])
        for i, lab in enumerate(labels)]
    fig.update_layout(
        template="plotly_white", height=470,
        title=dict(text=f"{title}: {labels[d]}", yref="container", y=0.97,
                   yanchor="top", x=0, xref="paper"),
        xaxis_title="theta", yaxis_title="draws",
        legend=dict(orientation="h", y=1.18, yanchor="bottom", x=0,
                    font=dict(size=11)),
        margin=dict(l=70, r=20, t=185, b=55),
        updatemenus=[dict(buttons=buttons, active=d, x=0, xanchor="left",
                          y=1.5, yanchor="top", showactive=True)])
    return fig


def forecast_section(theta, data, raw_df, names, dates, majority, minority, K):
    """One probe per axis: the pooled trend on the frozen record set, refit on
    each of the two chain groups.

    Each axis is copied out at float32 one at a time: the pooled trace is 2.7 GB
    of float64 and a subset is a fancy-index COPY, so slicing the axis first
    keeps every temporary near 300 MB."""
    subs = [(f"majority chains ({len(majority)})", majority, SUB_MAJ, "solid"),
            ("chains " + ", ".join(map(str, minority)), minority, SUB_MIN,
             "dash")]
    probes = []
    for k in range(K):
        tk = np.ascontiguousarray(theta[:, :, :, k:k + 1], dtype=np.float32)
        probes.append(forecast_probe(tk, data, raw_df, names, dates, subs,
                                     f"axis {k + 1}"))
        del tk
    return probes


CSS = """
/* background pinned: a viewer in dark mode would otherwise put dark text on a
   dark ground while the plotly_white figures stay white */
html,body{background:#fff}
body{font:15px/1.5 -apple-system,Segoe UI,Helvetica,Arial,sans-serif;
 max-width:1180px;margin:2rem auto;padding:0 1.4rem;color:#1c1c1c}
h1{font-size:1.4rem;margin-bottom:.2rem}
h2{font-size:1.1rem;margin:3rem 0 .4rem;padding-top:1rem;
 border-top:1px solid #e3e3e3;color:#333}
p.sub{color:#555;font-size:13px;margin-top:0}
figure{margin:2.6rem 0}
"""


def build_html(sections, subtitle) -> str:
    """`sections` is a list of (heading, [figure or html fragment, ...]).
    A None item is dropped, so a caller can pass a figure that did not build."""
    out = ['<meta charset="utf-8">', f"<style>{CSS}</style>",
           f"<script>{get_plotlyjs()}</script>",
           "<h1>The real story: chains aligned</h1>",
           f'<p class="sub">{subtitle}</p>']
    for heading, items in sections:
        items = [i for i in items if i is not None]
        if not items:
            continue
        out.append(f"<h2>{heading}</h2>")
        out += [f"<figure>{i if isinstance(i, str) else pio.to_html(i, full_html=False, include_plotlyjs=False)}</figure>"
                for i in items]
    return "\n".join(out)


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
    A = post["A"].values                              # (chain, draw, bench, K)
    bnames = list(post["bench"].values)
    print(f"  theta {theta.shape} | A {A.shape}")

    data = _load_matching_data(idata)
    names = data.mlookup.sort_values("model_idx")["model"].tolist()
    assert len(names) == M, f"{len(names)} data models vs {M} trace rows"

    # ── align the axes before anything reads them ──────────────
    # The chains agree on the four axes and file them in different raw columns,
    # so a raw-column scan reports a permutation as a disagreement. Every number
    # below, the CSV included, is on the aligned draws.
    raw_split = int(chain_split_flags(
        theta.mean(axis=1), np.median(theta.std(axis=1), axis=0))[0]
        .any(axis=1).sum())
    perms, matched, ident = axis_permutations(A)
    print("  axis alignment: common axis 1..K -> raw slot, by loading-column "
          "correlation")
    for c in range(C):
        print(f"    chain {c}  {tuple(perms[c])}  matched {matched[c]:.3f}"
              f"  identity {ident[c]:.3f}")
    known = [tuple(p) for p in KNOWN_PERMS]
    if len(known) != C:
        print(f"  no recorded permutation table for {C} chains")
    elif [tuple(p) for p in perms] == known:
        print("  permutation table matches the recorded one")
    else:
        print("  WARNING: permutation table DIFFERS from the recorded one, "
              "proceeding with the computed one")
        print(f"    recorded {known}")
        print(f"    computed {[tuple(p) for p in perms]}")
    apply_permutations(theta, perms)
    apply_permutations(A, perms)

    informed = mirt_informed_mask(theta.reshape(C * Dn, M, K), SD_CAP)
    ls = build_lineage_structure(data.mlookup.sort_values("model_idx"))
    chained = np.zeros(M, bool)
    if ls is not None:
        chained[ls.row_idx] = True
    raw_df = pd.read_csv(PROCESSED_FILE, parse_dates=["release_date"])
    dates, _ = _release_dates(raw_df)

    stats = taker_axis_stats(theta)
    df = build_table(stats, data, names, informed, chained, dates)
    aligned_split = int(stats["chain_split"].any(axis=1).sum())
    print(f"  split takers: raw {raw_split}/{M} -> aligned {aligned_split}/{M}")

    # The r-hat and eta scans are not run: the page reports no r-hat, and the
    # cached values are keyed to the raw columns, which no longer exist here.
    out_dir = Path(args.trace).parent
    print("  r-hat scans skipped: the page reads aligned theta")

    # The SAME scan on the loading matrix: A is (chain, draw, bench, K), the
    # identical shape family, so the taker scan applies unchanged.
    astats = taker_axis_stats(A)
    adf = pd.DataFrame([
        {"benchmark": bnames[b], "axis": f"axis{k + 1}",
         "chain_split": bool(astats["chain_split"][b, k]),
         "lump_gap": astats["lump_hi"][b, k] - astats["lump_lo"][b, k]}
        for b in range(len(bnames)) for k in range(K)])
    n_bench_split = int(adf["chain_split"].sum())

    # the grand-mean loading columns name the axes: the top benchmarks per
    # common axis are what "axis 1" means on this page
    gmean = A.mean(axis=(0, 1))                       # (bench, K)
    print("\n  grand-mean loadings, top 8 per common axis")
    for k in range(K):
        print(f"    axis{k + 1}: " + ", ".join(
            f"{bnames[b]} {gmean[b, k]:.2f}"
            for b in np.argsort(gmean[:, k])[::-1][:8]))

    cm = stats["chain_means"]
    majority, minority = residual_groups(cm, data.is_human)
    print(f"\n  residual groups: majority {majority}  minority {minority}")

    # rows for the residual comparison: the human tiers, and the machine takers
    # that still split, widest gap first
    hum_rows = list(np.flatnonzero(data.is_human))
    mach = df[df["chain_split"] & ~df["is_human"]]
    top_split = [names.index(n) for n in (mach.groupby("model")["lump_gap"].max()
                                          .sort_values(ascending=False)
                                          .head(12).index)]
    # Does the residual disagreement reach the published numbers? Skipped under
    # --theta-pos: the forecast reads raw theta, which is not the ability the
    # likelihood sees there. Any forecast failure (too few dated records on an
    # axis) drops the section rather than the run. Run BEFORE the sections are
    # assembled: the ESS figure in "What still splits" needs the record cells.
    theta_pos = "theta_pos" in post
    probes = None
    if theta_pos:
        print("  forecast section skipped: --theta-pos fit")
    else:
        print("  forecast per axis ...", flush=True)
        try:
            probes = forecast_section(theta, data, raw_df, names, dates,
                                      majority, minority, K)
        except ValueError as e:
            print(f"  forecast section skipped: {e}")

    # ── ESS per cell, pooled and per mode ──────────────────────
    # Split cells first (widest lump gap first, so the dropdown opens on the
    # clearest one), then the human tiers on the axis the splits concentrate on.
    # Those tiers disagree the same way but under the flag threshold, which is
    # why they are labelled sub-threshold rather than dropped.
    split_cells = [(int(m), int(k)) for m, k in np.argwhere(stats["chain_split"])]
    split_cells.sort(key=lambda c: -(stats["lump_hi"][c] - stats["lump_lo"][c]))
    split_axis = (int(np.bincount([k for _, k in split_cells],
                                  minlength=K).argmax())
                  if split_cells else K - 1)
    sub_cells = [(int(m), split_axis) for m in np.flatnonzero(data.is_human)
                 if not stats["chain_split"][m, split_axis]]
    rec_cells = ([(m, k) for k, _, m in record_entries(probes, names)]
                 if probes else [])
    print(f"  bulk ESS on {len(set(split_cells + sub_cells + rec_cells))} cells "
          "(pooled / majority / minority) ...", flush=True)
    essmap = cell_ess(theta, dict.fromkeys(split_cells + sub_cells + rec_cells),
                      majority, minority)
    lump_entries = ([(m, k, "") for m, k in split_cells]
                    + [(m, k, " (sub-threshold)") for m, k in sub_cells])

    sections = [
        ("Chain alignment",
         [alignment_heatmap_fig(perms, matched, raw_split, aligned_split, M)]),
        ("One human story",
         [chain_tier_fig(cm, names, data.is_human, K),
          tier_panel_fig(cm, names, majority, minority, K)]),
        ("What still splits",
         [lump_dist_fig(theta, lump_entries, essmap, names, majority, minority),
          gap_heatmap_fig(df, K) if aligned_split else None,
          residual_dots_fig(cm, [("human tiers", hum_rows),
                                 ("residual split takers", top_split)],
                            minority, K),
          ess_scatter_fig(essmap, split_cells, rec_cells, names)]),
    ]
    if probes is not None:
        sections.append((
            "Forecast, aligned",
            [aligned_forecast_fig(probes, cm, names, majority, minority, K),
             record_dist_fig(theta, probes, names, dates, essmap),
             mode_crossover_dotwhisker_fig(probes)]))

    csv_path = out_dir / "theta_bimodality.csv"
    df.to_csv(csv_path, index=False)

    mp = modes_path(args.trace)
    modes = json.loads(mp.read_text()) if mp.exists() else None
    n_ch = modes["n_chains"] if modes else C
    n_mode = len(modes["modes"]) if modes else "?"
    html_path = out_dir / "bimodality.html"
    html_path.write_text(build_html(
        sections,
        f"{Path(args.trace).name} | K={K}, {n_ch} chains, {n_mode} posterior mode, "
        f"{M} test-takers, {len(bnames)} benchmarks. Axes aligned per chain by "
        f"the loading columns. A split = the chains disagree in two lumps "
        f"(gap &gt; {GAP_SDS:g}x the median within-chain sd, at least "
        f"{MIN_SIDE} chains a side)."))

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
    print("\n  human tiers, mean theta per group")
    for tier in PANEL_TIERS:
        if tier not in names:
            continue
        y = cm[:, names.index(tier), :]
        maj = ", ".join(f"{v:+.2f}" for v in y[majority].mean(axis=0))
        mn = ", ".join(f"{v:+.2f}" for v in y[minority].mean(axis=0))
        print(f"    {tier:22s} majority ({maj})   minority ({mn})")
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
    n_split = int(df["chain_split"].sum())
    print(f"\n  informed gate: {int(df.loc[df['chain_split'], 'informed'].sum())}"
          f"/{n_split} split taker-axis rows pass SD<{SD_CAP:g}")

    # ── bulk ESS, split cells ──────────────────────────────────
    print(f"\n  bulk ESS, the {len(split_cells)} split cells")
    print("    model                                      axis    pooled"
          "      maj      min")
    for m, k in split_cells:
        p, a, b = essmap[(m, k)]
        print(f"    {names[m]:42s} axis{k + 1} {p:9.0f} {a:8.0f} {b:8.0f}")
    if sub_cells:
        print(f"  bulk ESS, the {len(sub_cells)} human tiers on "
              f"axis{split_axis + 1} (sub-threshold)")
        for m, k in sub_cells:
            p, a, b = essmap[(m, k)]
            print(f"    {names[m]:42s} axis{k + 1} {p:9.0f} {a:8.0f} {b:8.0f}")

    # A subset holds fewer raw draws than the pooled set, so the raw comparison
    # is unfair by construction. Per 1,000 draws removes the draw count and
    # leaves the sampling-efficiency effect only.
    n_raw = (C * Dn, len(majority) * Dn, len(minority) * Dn)
    for lab, cells in (("split", split_cells), ("record", rec_cells)):
        if not cells:
            continue
        e = np.array([essmap[c] for c in cells])
        med, medn = np.median(e, axis=0), np.median(e / np.array(n_raw) * 1000,
                                                    axis=0)
        print(f"  median ESS, {lab} cells (n={len(cells)}): raw pooled "
              f"{med[0]:.0f} / maj {med[1]:.0f} / min {med[2]:.0f}   "
              f"per 1,000 draws pooled {medn[0]:.1f} / maj {medn[1]:.1f} / "
              f"min {medn[2]:.1f}")
    if probes is not None:
        print("  frontier slope (theta/yr) on the frozen record set")
        for p in probes:
            subs = "  ".join(f"{s['label']} {s['slope']:.3f}" for s in p["subs"])
            print(f"    {p['label']}  pooled {p['slope']:.3f}   {subs}")
        print("  crossover median dates")
        for p in probes:
            for _, r in p["cx"].iterrows():
                if r["status"] == "no_crossing" or r["tier"] not in TARGET_TIERS:
                    continue
                subs = "  ".join(
                    f"{s['label']} "
                    + str(s["cx"].set_index("tier").loc[r["tier"],
                                                        "crossover_date_median"])[:10]
                    for s in p["subs"])
                print(f"    {p['label']} · {r['tier']:16s} pooled "
                      f"{str(r['crossover_date_median'])[:10]}   {subs}")
    print(f"\n  {csv_path}  ({len(df)} rows)")
    print(f"  {html_path}  ({html_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
