"""Reference render of the lineage prior's structure, as a multi-page PDF.

Page 1 puts every chain on one calendar axis. Then one page per chain lists its
nodes in date order with, per node, the offset groups that share that node's psi
and the release gap in years on the edge coming in — the dt the Brownian step
scales with (mean drift*dt, sd s*sqrt(dt)).

The tree is read back out of the fitted structure (B, C, delta_dt), not
re-derived from the map file, so the page shows what the model builds after the
data join: chains that lost too many nodes to the join are absent, and node
positions are the ones the psi vector actually uses.

Run:
  ~/miniforge3/envs/pymc_env/bin/python 3_diagnostics/plot_lineage.py
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from multiaxis_eci.data import load_eci_data          # noqa: E402
from multiaxis_eci.lineage import LINEAGE_MAP, build_lineage_structure  # noqa: E402

SPINE, BRANCH, EDGE, GREY = "#1f77b4", "#d95f02", "#444444", "#777777"


def collect(data, csv_path=LINEAGE_MAP):
    """One row per node: chain, label, date, parent, incoming dt, offset groups.

    Rebuilds the map frame the same way build_lineage_structure does (dedup on
    raw_string first, then the in_chain filter) so the rows line up with
    row_idx element by element; the assert below is what enforces that.
    """
    ls = build_lineage_structure(data.mlookup, csv_path=csv_path)
    if ls is None:
        raise SystemExit("no usable chain in the map for this data scope")

    m = pd.read_csv(csv_path, keep_default_na=False).drop_duplicates("raw_string")
    m = m[m["in_chain"].str.lower() == "yes"]
    row_of = {n: i for i, n in enumerate(data.mlookup["model"])}
    m = m[m["raw_string"].isin(row_of) & m["chain"].isin(ls.chain_names)]
    m = m.assign(row=m["raw_string"].map(row_of)).sort_values("row")
    assert np.array_equal(m["row"].to_numpy(), ls.row_idx), \
        "map frame is out of step with the fitted structure"
    m = m.assign(node_i=ls.node_idx, grp=ls.offset_group)

    chain_of = np.argmax(ls.B, axis=1)
    C = ls.C.astype(bool)
    depth = C.sum(1)

    # Parent from C alone: the one node of the same chain whose delta set is
    # C[i] minus exactly one mark. That missing mark is the step INTO i, so it
    # also names i's own delta column and therefore its gap.
    nodes = []
    for i in range(ls.n_nodes):
        rows = m[m["node_i"] == i]
        parent, dt = -1, np.nan   # -1 = founder (no incoming step)
        if depth[i]:
            cand = [j for j in range(ls.n_nodes)
                    if chain_of[j] == chain_of[i] and depth[j] == depth[i] - 1
                    and not (C[j] & ~C[i]).any()]
            assert len(cand) == 1, f"node {i} has {len(cand)} candidate parents"
            parent = cand[0]
            own = np.flatnonzero(C[i] & ~C[parent])
            assert own.size == 1, f"node {i} owns {own.size} deltas"
            dt = ls.delta_dt[own[0]]
        groups = [(v, len(g)) for (_, v), g in
                  rows.groupby(["grp", "variant"], sort=True)]
        nodes.append(dict(
            i=i, chain=ls.chain_names[chain_of[i]], label=rows["node"].iloc[0],
            vendor=rows["vendor"].iloc[0], date=pd.Timestamp(rows["node_date"].iloc[0]),
            # display only: an explicitly parented node opens a side lane
            branched=bool(rows["parent"].iloc[0]) if "parent" in rows else False,
            parent=parent, dt=dt, groups=groups, n_models=len(rows)))
    return ls, pd.DataFrame(nodes)


def _lanes(nd):
    """Lane 0 = the spine; an explicitly parented node and its descendants sit
    one lane out, so a side line never overprints the spine it hangs off."""
    lane = {}
    for r in nd.itertuples():
        base = 0 if r.parent < 0 else lane[r.parent]
        lane[r.i] = base + 1 if r.branched else base
    return lane


def overview(nd, ls, n_models):
    order = (nd.groupby("chain").agg(vendor=("vendor", "first"),
                                     first=("date", "min")).sort_values(["vendor", "chain"]))
    fig, ax = plt.subplots(figsize=(11, 0.30 * len(order) + 1.9))
    y_of = {c: -k for k, c in enumerate(order.index)}
    lane = _lanes(nd)
    for r in nd.itertuples():
        y = y_of[r.chain] + 0.22 * lane[r.i]
        if r.parent >= 0:
            p = nd.iloc[r.parent]
            ax.plot([p.date, r.date], [y_of[r.chain] + 0.22 * lane[r.parent], y],
                    "-" if lane[r.i] == 0 else "--", c=EDGE, lw=0.8, zorder=1)
        ax.plot(r.date, y, "o", ms=4.2 if lane[r.i] == 0 else 3.4,
                c=SPINE if lane[r.i] == 0 else BRANCH, zorder=2)
    for c, y in y_of.items():
        sub = nd[nd.chain == c]
        ax.text(sub.date.max() + pd.Timedelta(days=40), y,
                f"{len(sub)}n / {sub.n_models.sum()}m", va="center", fontsize=5.5, c=GREY)
    ax.set_yticks(list(y_of.values()))
    ax.set_yticklabels([f"{order.vendor[c]} · {c}" for c in y_of], fontsize=6)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_ylim(-len(order) + 0.4, 0.9)
    ax.grid(axis="x", lw=0.4, alpha=0.3)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.plot([], [], "o-", c=SPINE, ms=4.2, lw=0.8, label="spine node / release step")
    ax.plot([], [], "o--", c=BRANCH, ms=3.4, lw=0.8, label="branched node / branch step")
    ax.legend(fontsize=6, loc="upper left", frameon=False)
    ax.set_title(f"Lineage prior: {ls.n_chains} chains, {ls.n_nodes} nodes, "
                 f"{ls.n_deltas} steps, {ls.n_groups} offset groups over "
                 f"{n_models} test-takers\n"
                 f"(n = nodes, m = test-takers; median gap "
                 f"{np.median(ls.delta_dt):.3f} yr)", fontsize=8.5)
    fig.tight_layout()
    return fig


def chain_page(nd, chain):
    sub = nd[nd.chain == chain]
    lane = _lanes(nd)
    wrapped = {r.i: textwrap.wrap(" · ".join(f"{v}×{n}" for v, n in r.groups), 68) or [""]
               for r in sub.itertuples()}
    rows = {}
    y = 0.0
    for r in sub.itertuples():                    # index order == date order
        rows[r.i] = y
        y -= 0.85 + 0.55 * (len(wrapped[r.i]) - 1)

    fig, ax = plt.subplots(figsize=(8.3, max(2.4, 0.42 * -y + 1.3)))
    for r in sub.itertuples():
        x, yy = 0.55 * lane[r.i], rows[r.i]
        if r.parent >= 0:
            xp, yp = 0.55 * lane[r.parent], rows[r.parent]
            ax.plot([xp, xp, x], [yp, yy, yy], "-" if lane[r.i] == lane[r.parent] else "--",
                    c=EDGE, lw=0.9, zorder=1)
            # on the connector itself, masked, so it never lands on a variant
            # line: mid-drop for a spine step, mid-run for a branch step
            lx, ly = ((xp, (yp + yy) / 2) if lane[r.i] == lane[r.parent]
                      else ((xp + x) / 2, yy))
            ax.text(lx, ly, f"Δ{r.dt:.2f} yr", fontsize=5.5, c=GREY,
                    ha="center", va="center", zorder=3,
                    bbox=dict(fc="white", ec="none", pad=0.8))
        ax.plot(x, yy, "o", ms=6, c=SPINE if lane[r.i] == 0 else BRANCH, zorder=2)
        ax.text(-0.15, yy, r.date.strftime("%Y-%m-%d"), fontsize=6, ha="right", va="center")
        ax.text(x + 0.28, yy, f"{r.label}   ({r.n_models} test-taker"
                              f"{'s' if r.n_models > 1 else ''})",
                fontsize=7.5, va="center", fontweight="bold")
        for k, line in enumerate(wrapped[r.i]):
            ax.text(x + 0.29, yy - 0.34 - 0.5 * k, line, fontsize=6, c=GREY, va="center")
    ax.set_xlim(-1.15, 4.2)
    ax.set_ylim(y - 0.5, 1.0)
    ax.axis("off")
    ax.set_title(f"chain '{chain}' ({sub.vendor.iloc[0]}) — {len(sub)} nodes, "
                 f"{len(sub) - 1} steps, {sub.n_models.sum()} test-takers\n"
                 f"psi[node] = founder + Σ steps; theta = psi[node] + offset[group]",
                 fontsize=8.5, loc="left")
    fig.tight_layout()
    return fig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "plots" / "lineage_map.pdf"))
    ap.add_argument("--apply-exclusions", action="store_true",
                    help="canonical benchmark scope instead of the exploration scope")
    args = ap.parse_args()

    data = load_eci_data(include_all_benchmarks=not args.apply_exclusions)
    ls, nd = collect(data)
    print(f"{ls.n_chains} chains / {ls.n_nodes} nodes / {ls.n_deltas} steps / "
          f"{ls.n_groups} offset groups / {int(nd.n_models.sum())} chained test-takers")
    tied = [(r.label, r.chain) for r in nd.itertuples()
            for v, n in r.groups if n > 1]
    print(f"offset groups holding >1 test-taker (identical theta): {len(tied)}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(out) as pdf:
        pdf.savefig(overview(nd, ls, int(nd.n_models.sum())))
        plt.close("all")
        for chain in sorted(nd.chain.unique()):
            pdf.savefig(chain_page(nd, chain))
            plt.close("all")
    print(f"{1 + nd.chain.nunique()} pages → {out}")


if __name__ == "__main__":
    main()
