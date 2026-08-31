"""Every test-taker the flagship fit splits on, majority chains against the rest.

`diagnostics/theta_bimodality.py` flags a taker as split when the gap in its
sorted per-chain theta means is wider than 3 within-chain sds with at least two
chains each side. Nearly all of the flags land on one axis, the agentic one, and
this figure draws that axis only.

The frame matters. The flag is computed on draws put in ONE common raw column
order by `theta_bimodality.axis_permutations`, which matches every chain's mean
loading columns to the all-chain grand mean. The display frame used by the rest
of the post is a different ordering of the same four axes, so the CSV's axis
LABEL cannot be trusted against the post's. The agentic column is therefore
identified by its own top-loading benchmarks (GBAEval, ProofBench, Remote Labor
Index) and the CSV label is checked against that, never assumed.

The chains are cut into two groups by how far each one's human profile sits from
the cross-chain median (`theta_bimodality.residual_groups`); the groups are
COMPUTED here, never assumed.

Machine rows only. The human tiers split the same way and carry their own
figure, where the tier profile is the subject rather than one row among models.

Statistics are the project's `post_stats` convention: posterior median and a 95%
CENTRAL quantile interval, not an HDI.

Usage:
    python blogpost/figures/make_split_takers_plotly.py [--trace FILE] [--tag ""]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xarray as xr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "3_diagnostics"))

from multiaxis_eci.analysis import (FLAGSHIP, FLAGSHIP_THIN,  # noqa: E402
                      FLAGSHIP_TRACE as TRACE)
from theta_bimodality import PERM_STRIDE, axis_permutations, residual_groups  # noqa: E402
from multiaxis_eci.viz.core import save_print  # noqa: E402


# None drops the in-figure title: the post's caption carries the
# description. Set a string to draw it on the canvas again, e.g.
# "The test-takers the flagship fit splits on".
TITLE = None

# Benchmarks that name the agentic axis. The column carrying them at the top of
# its loadings is the one the figure draws.
AGENTIC_MARKERS = ("GBAEval", "ProofBench", "Remote Labor Index")
# the winning column must rank every marker inside this many benchmarks
MARKER_RANK_CAP = 15

# Okabe-Ito blue / vermillion: the two chain groups.
MAJ_COLOR, MIN_COLOR = "#0072B2", "#D55E00"
PROB = 0.95           # central quantile interval, the post_stats convention
DODGE = 0.18          # half the vertical gap between the two groups of a row

# Post-scale sizing, shared with the human-modes / trend / forests figures.
FONT_TITLE = 42
FONT_AXIS = 36        # x caption
FONT_TICK = 30        # tick labels (taker rows, theta)
FONT_LEGEND = 30
MARKER = 15
ERRBAR_W = 3.0
ROW_PX = 78           # vertical pitch per taker row, so 30px names never touch
# One column, so the left margin is the only place a row name can sit: 26
# characters at tick size (deepseek-coder-33b-base) plus the legend's own width.
WIDTH, MARGIN_L = 2100, 520


def main(trace: Path = TRACE, tag: str = "", out_dir: Path = HERE) -> None:
    idata = FLAGSHIP.open_posterior(keep=["A", "theta", "tau_A"],
                                    thin=FLAGSHIP_THIN, chains=None, path=trace)
    data, *_ = FLAGSHIP.load_data(idata)     # raises if the coords disagree
    print(f"  data scope: {data.n_obs} obs, {data.n_models} test-takers, "
          f"{data.n_benchmarks} benchmarks")
    names = data.mlookup.sort_values("model_idx")["model"].tolist()
    bench = data.blookup.sort_values("benchmark_idx")["benchmark"].tolist()

    with xr.open_dataset(trace, group="posterior") as post:
        A = post["A"].isel(draw=slice(None, None, PERM_STRIDE)).values
    perms, matched, _ = axis_permutations(A, stride=1)
    Aal = A.copy()
    for c, pm in enumerate(perms):
        Aal[c] = Aal[c][:, :, list(pm)]
    K = A.shape[3]
    print(f"  {A.shape[0]} chains, {K} axes; "
          f"matched loading corr {matched.mean():.3f}")

    # Which aligned raw column is the agentic axis: the one that ranks the
    # agentic benchmarks highest. A per-axis loading scale is free, so the
    # comparison is a RANK within each column, never a level across columns.
    med_A = np.median(Aal.reshape(-1, A.shape[2], K), axis=0)
    rank = np.empty((K, len(AGENTIC_MARKERS)), int)
    for k in range(K):
        order_b = [bench[i] for i in np.argsort(-med_A[:, k])]
        rank[k] = [order_b.index(m) for m in AGENTIC_MARKERS]
        print(f"  aligned axis{k + 1} top loadings: {', '.join(order_b[:6])}"
              f" | marker ranks {[int(r) for r in rank[k]]}")
    agentic = int(rank.sum(axis=1).argmin())
    if rank[agentic].max() >= MARKER_RANK_CAP:
        raise SystemExit(f"no aligned column carries {AGENTIC_MARKERS}: {rank}")
    print(f"  agentic axis = aligned column axis{agentic + 1}, "
          f"marker ranks {[int(r) for r in rank[agentic]]} of {len(bench)} benchmarks")

    sp = pd.read_csv(trace.parent / "theta_bimodality.csv")
    sp = sp[sp.chain_split]
    print("  split flags per axis: "
          + ", ".join(f"{a} {n}" for a, n in sp.axis.value_counts().items()))
    # Models only: the human tiers carry the same split, on their own figure.
    sp = sp[(sp.axis == f"axis{agentic + 1}") & (~sp.is_human)]
    want = list(sp.model)
    rows = [names.index(m) for m in want]

    with xr.open_dataset(trace, group="posterior") as post:
        th_all = post["theta"].values                       # (C, D, M, K)
    for c, pm in enumerate(perms):
        th_all[c] = th_all[c][:, :, list(pm)]
    # residual_groups expects (C, M, K) with a human mask over M, and sums its
    # distance over all axes, so it runs on the whole aligned ability block.
    majority, minority = residual_groups(th_all.mean(axis=1), data.is_human)
    print(f"  majority chains {majority} | differing chains {minority}")
    if not minority:
        raise SystemExit("no chain group differs on the human tiers")
    th = th_all[:, :, rows, agentic]                        # (C, D, n)

    q = [(1 - PROB) / 2, (1 + PROB) / 2]
    stat = {}
    for lab, grp in (("maj", majority), ("min", minority)):
        d = th[grp].reshape(-1, len(want))
        stat[lab] = (np.median(d, axis=0), *np.quantile(d, q, axis=0))
    delta = stat["min"][0] - stat["maj"][0]
    order = np.argsort(delta)                  # row 0 at the BOTTOM of the axis
    n = len(want)

    labels = (f"majority chains ({len(majority)})",
              f"minority chains ({len(minority)}: "
              + ", ".join(map(str, minority)) + ")")
    fig = go.Figure()
    for lab, color, off, name in (("maj", MAJ_COLOR, DODGE, labels[0]),
                                  ("min", MIN_COLOR, -DODGE, labels[1])):
        med, lo, hi = (v[order] for v in stat[lab])
        fig.add_trace(go.Scatter(
            x=med, y=np.arange(n) + off, mode="markers",
            marker=dict(color=color, size=MARKER),
            error_x=dict(type="data", symmetric=False, array=hi - med,
                         arrayminus=med - lo, color=color,
                         thickness=ERRBAR_W, width=0),
            name=name, hovertemplate="%{text}<br>theta %{x:.2f}<extra></extra>",
            text=[want[i] for i in order]))
    fig.add_vline(x=0, line=dict(color="#444", width=2))

    fig.update_yaxes(tickmode="array", tickvals=list(range(n)),
                     ticktext=[want[i] for i in order],
                     tickfont=dict(size=FONT_TICK), showgrid=False,
                     range=[-0.7, n - 0.3])
    fig.update_xaxes(tickfont=dict(size=FONT_TICK), gridcolor="#e9e9e9",
                     title=dict(text="ability on the agentic axis "
                                     f"(median, {PROB:.0%} interval)",
                                font=dict(size=FONT_AXIS)))
    fig.update_layout(
        template="plotly_white", width=WIDTH, height=ROW_PX * n + 420,
        margin=dict(l=MARGIN_L, r=120, t=80, b=250),
        legend=dict(orientation="h", yanchor="top", y=-0.10, xanchor="center",
                    x=0.5, font=dict(size=FONT_LEGEND),
                    bgcolor="rgba(0,0,0,0)", borderwidth=0))
    if TITLE is not None:
        fig.update_layout(title=dict(text=TITLE, x=0.5,
                                     font=dict(size=FONT_TITLE)), margin_t=190)

    out = out_dir / f"split_takers_agentic_plotly{tag}"
    fig.write_html(out.with_suffix(".html"))
    save_print(fig, out)
    print(f"  {n} split models, plot order "
          "bottom to top:")
    for i in order:
        print(f"    {want[i]:<28} maj {stat['maj'][0][i]:+.2f}  "
              f"min {stat['min'][0][i]:+.2f}  delta {delta[i]:+.2f}")
    print(f"  minority minus majority: mean {delta.mean():+.2f} "
          f"(down {int((delta < 0).sum())}, up {int((delta > 0).sum())})")
    print(f"  wrote {out.with_suffix('.png')} and {out.with_suffix('.html')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trace", type=Path, default=TRACE)
    p.add_argument("--tag", default="")
    args = p.parse_args()
    main(args.trace, args.tag)
