"""Human tiers per axis, the majority chains against the chains that differ.

The non-negative loading prior fixes the rotation but not the axis ORDER, so the
chains file the same four axes in different raw columns. Two alignment steps run
before a single number is read:

  1. `theta_bimodality.axis_permutations` matches every chain's mean loading
     columns to the all-chain grand mean, which puts all chains in one common
     RAW column order.
  2. that raw order is matched to `prepare_fit`'s per-draw energy-rank display
     frame by loading-column correlation, so panel k carries the same axis
     identity as every other figure in the post.

The chains are then cut into two groups by how far each one's human profile sits
from the cross-chain median (`theta_bimodality.residual_groups`); the groups are
COMPUTED here, never assumed. Only the human rows of theta are pulled, so this
costs 60 MB and not the 5 GB of the whole ability block.

Statistics are the project's `post_stats` convention: posterior median and a 95%
CENTRAL quantile interval, not an HDI.

Usage:
    python lw_post/figures/make_human_modes_plotly.py [--trace FILE] [--tag ""]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import xarray as xr
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "3_diagnostics"))
sys.path.insert(0, str(HERE))

from multiaxis_eci.analysis import (FLAGSHIP, FLAGSHIP_THIN,  # noqa: E402
                      FLAGSHIP_TRACE as TRACE, prepare_fit)
from multiaxis_eci.config import AXIS_TITLES  # noqa: E402
from make_all import two_column_layout  # noqa: E402
from theta_bimodality import PERM_STRIDE, axis_permutations, residual_groups  # noqa: E402
from multiaxis_eci.viz.core import save_print  # noqa: E402


# None drops the in-figure title: the LessWrong caption carries the
# description. Set a string to draw it on the canvas again, e.g.
# "Human tiers, the two chain groups of the flagship fit".
TITLE = None

# Okabe-Ito blue / vermillion: the two chain groups. The diagnostics page's own
# pair (blue / reddish-purple) is two pinks on paper.
MAJ_COLOR, MIN_COLOR = "#0072B2", "#D55E00"
PROB = 0.95           # central quantile interval, the post_stats convention
DODGE = 0.18          # half the vertical gap between the two groups of a tier

# Post-scale sizing, shared with the trend / forests / crossover figures.
FONT_TITLE = 42       # figure title (drawn only when TITLE is set)
FONT_AXIS = 36        # panel titles
FONT_TICK = 30        # tick labels (tier rows, theta) and the x caption
FONT_LEGEND = 30
MARKER = 15
ERRBAR_W = 3.0
# 9 tier rows per panel at tick size, two panel columns. The left margin holds
# the left column's row names and the inter-column gutter holds the right
# column's, so both are sized from the longest tier name (31 characters).
# The canvas is wider than the other figures' 2100 because the axis-2 panel
# title is 42 characters at panel-title size: centred on a narrow right
# column it needs the paper to the right of that column to stay on the page.
WIDTH, HEIGHT = 2400, 1900
COL_DOM = [(0.0, 0.28), (0.70, 0.95)]


def main(trace: Path = TRACE, tag: str = "", out_dir: Path = HERE) -> None:
    # prepare_fit needs the whole ability block; the tier figure itself needs
    # only the human rows, so the trace is opened twice with different subsets.
    idata = FLAGSHIP.open_posterior(keep=["A", "theta", "tau_A"],
                                    thin=FLAGSHIP_THIN, chains=None, path=trace)
    data, *_ = FLAGSHIP.load_data(idata)     # raises if the coords disagree
    print(f"  data scope: {data.n_obs} obs, {data.n_models} test-takers, "
          f"{data.n_benchmarks} benchmarks")
    names = data.mlookup.sort_values("model_idx")["model"].tolist()
    rows = np.flatnonzero(data.is_human)
    tiers = [names[i] for i in rows]

    with xr.open_dataset(trace, group="posterior") as post:
        A = post["A"].isel(draw=slice(None, None, PERM_STRIDE)).values
        th = post["theta"].isel(model=list(rows)).values      # (C, D, T, K)
    perms, matched, _ = axis_permutations(A, stride=1)
    for c, pm in enumerate(perms):
        th[c] = th[c][:, :, list(pm)]
    C, _, T, K = th.shape
    print(f"  {C} chains, {T} human tiers, {K} axes; "
          f"matched loading corr {matched.mean():.3f}")

    # residual_groups expects (C, M, K) with a human mask over M. Its distance
    # is summed over all axes, so it is invariant to the display reordering
    # below and can be taken before it.
    majority, minority = residual_groups(th.mean(axis=1), np.ones(T, bool))
    print(f"  majority chains {majority} | differing chains {minority}")
    if not minority:
        raise SystemExit("no chain group differs on the human tiers")

    # The alignment leaves the axes in the raw column order; every other figure
    # reports them in prepare_fit's per-draw energy rank order. Match the two by
    # loading-column correlation, or the panels carry the wrong axis names.
    Aal = A.copy()
    for c, pm in enumerate(perms):
        Aal[c] = Aal[c][:, :, list(pm)]
    raw_med = np.median(Aal[majority].reshape(-1, A.shape[2], A.shape[3]), axis=0)
    rep_med = np.median(prepare_fit(
        FLAGSHIP.open_posterior(keep=["A", "theta", "tau_A"], thin=FLAGSHIP_THIN,
                                chains=majority, path=trace),
        data).require_A(), axis=0)
    zc = lambda X: (X - X.mean(0)) / (X.std(0) + 1e-12)
    corr = zc(rep_med).T @ zc(raw_med) / raw_med.shape[0]
    slot = [int(np.argmax(np.abs(corr[k]))) for k in range(corr.shape[0])]
    assert sorted(slot) == list(range(corr.shape[0])), f"ambiguous match {slot}"
    print("  reported axis -> aligned raw slot: "
          + ", ".join(f"{k + 1}->{s + 1} (r {corr[k, s]:+.2f})"
                      for k, s in enumerate(slot)))
    th = th[:, :, :, slot]

    order = np.argsort(th[majority].mean(axis=(0, 1))[:, 0])   # by axis-1 level
    q = [(1 - PROB) / 2, (1 + PROB) / 2]
    labels = (f"majority chains ({len(majority)})",
              f"minority chains ({len(minority)}: "
              + ", ".join(map(str, minority)) + ")")

    titles = [AXIS_TITLES[f"axis{k + 1}"] for k in range(K)]
    fig = make_subplots(rows=2, cols=2, subplot_titles=titles,
                        vertical_spacing=0.10, horizontal_spacing=0.02)
    for k in range(K):
        r, c = k // 2 + 1, k % 2 + 1
        for grp, color, off, lab in ((majority, MAJ_COLOR, DODGE, labels[0]),
                                     (minority, MIN_COLOR, -DODGE, labels[1])):
            d = th[grp][:, :, :, k].reshape(-1, T)             # (draws, tier)
            med = np.median(d, axis=0)[order]
            lo, hi = np.quantile(d, q, axis=0)[:, order]
            fig.add_trace(go.Scatter(
                x=med, y=np.arange(T) + off, mode="markers",
                marker=dict(color=color, size=MARKER),
                error_x=dict(type="data", symmetric=False, array=hi - med,
                             arrayminus=med - lo, color=color,
                             thickness=ERRBAR_W, width=0),
                name=lab, legendgroup=lab, showlegend=(k == 0),
                hovertemplate="%{text}<br>theta %{x:.2f}<extra></extra>",
                text=[tiers[i] for i in order]), row=r, col=c)
        fig.add_vline(x=0, row=r, col=c, line=dict(color="#444", width=2))
        fig.update_yaxes(tickmode="array", tickvals=list(range(T)),
                         ticktext=[tiers[i] for i in order],
                         tickfont=dict(size=FONT_TICK), showgrid=False,
                         range=[-0.7, T - 0.3], row=r, col=c)
        fig.update_xaxes(tickfont=dict(size=FONT_TICK), gridcolor="#e9e9e9",
                         row=r, col=c)
    fig.update_xaxes(title=dict(text=f"theta (median, {PROB:.0%} interval)",
                                font=dict(size=FONT_TICK)), row=2)

    fig.update_layout(
        template="plotly_white", width=WIDTH, height=HEIGHT,
        margin=dict(l=520, r=200, t=110, b=190),
        legend=dict(orientation="h", yanchor="top", y=-0.07, xanchor="center",
                    x=0.5, font=dict(size=FONT_LEGEND),
                    bgcolor="rgba(0,0,0,0)", borderwidth=0))
    if TITLE is not None:
        fig.update_layout(title=dict(text=TITLE, x=0.5,
                                     font=dict(size=FONT_TITLE)), margin_t=190)
    # The gutter here holds a 31-character tier name.
    two_column_layout(fig, COL_DOM, AXIS_TITLES, k=K)
    for ann in fig.layout.annotations[:K]:
        ann.font = dict(size=FONT_AXIS)

    out = out_dir / f"human_modes_plotly{tag}"
    fig.write_html(out.with_suffix(".html"))
    save_print(fig, out)
    gap = th[minority].mean(axis=(0, 1)) - th[majority].mean(axis=(0, 1))
    for k in range(K):
        j = int(np.abs(gap[:, k]).argmax())
        print(f"  {titles[k]}: signed tier mean {gap[:, k].mean():+.2f}, "
              f"largest {gap[j, k]:+.2f} on {tiers[j]}")
    print(f"  wrote {out.with_suffix('.png')} and {out.with_suffix('.html')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trace", type=Path, default=TRACE)
    p.add_argument("--tag", default="")
    args = p.parse_args()
    main(args.trace, args.tag)
