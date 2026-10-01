"""ECI-H along the lineages, to catch filing mistakes in 1_curated/lineage_map.csv.

Every parent -> child edge of the map (the previous spine node, or the explicit `parent`) is read
off a canonical fit's per-run ECI-H table (`all_models_eci.csv`). A release rarely scores well
below the one it continues, so a large drop usually means the map is wrong somewhere: a smaller
model read as a larger one (Qwen3 1.7B on the 7B chain, 2026-10-01), a code or preview model on a
general spine. Drops that were investigated and found genuine (Gemini 2.5 Pro's 05-06 preview) are
listed with their reason in 1_curated/lineage_reviewed_exceptions.csv (`check` = regression);
tests/test_lineage_map.py fails on any other.

Two comparisons per edge: the best run of each node (highest posterior mean), and each variant
the two nodes share (effort against the same effort).
"""
from __future__ import annotations

import pandas as pd

from multiaxis_eci.lineage import LINEAGE_MAP

# A drop is flagged when the child loses at least MIN_DROP ECI-H points (the frontier gains about
# 20 a year), its whole credible interval lies below the parent's posterior mean, and the parent's
# lies above the child's: a run measured on one or two scores cannot raise a flag on its own.
MIN_DROP = 5.0


def edges(lineage: pd.DataFrame) -> pd.DataFrame:
    """(chain, parent, child, parent_date, child_date) for every edge of the chained nodes."""
    y = lineage[lineage["in_chain"].astype(str).str.lower() == "yes"]
    rows = []
    for chain, g in y.groupby("chain"):
        nodes = (g.groupby("node").agg(date=("node_date", "min"), parent=("parent", "first"))
                  .reset_index().sort_values(["date", "node"]))
        date = dict(zip(nodes["node"], nodes["date"]))
        prev = None
        for r in nodes.itertuples():
            if pd.isna(r.parent):
                if prev is not None:
                    rows.append((chain, prev, r.node, date[prev], date[r.node]))
                prev = r.node
            else:
                rows.append((chain, r.parent, r.node, date[r.parent], date[r.node]))
    return pd.DataFrame(rows, columns=["chain", "parent", "child", "parent_date", "child_date"])


def comparisons(eci: pd.DataFrame, lineage: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every (edge, best or shared variant) comparison, with both runs' ECI-H and the drop."""
    lineage = pd.read_csv(LINEAGE_MAP) if lineage is None else lineage
    e = eci.set_index("name")
    y = lineage[(lineage["in_chain"].astype(str).str.lower() == "yes")
                & lineage["raw_string"].isin(e.index)]
    runs = {k: g for k, g in y.groupby(["chain", "node"])}
    rows = []
    for ed in edges(lineage).itertuples():
        gp, gc = runs.get((ed.chain, ed.parent)), runs.get((ed.chain, ed.child))
        if gp is None or gc is None:
            continue
        def best(g):
            return e.loc[list(g["raw_string"]), "mean"].idxmax()
        pairs = [("best", best(gp), best(gc))]
        for v in sorted(set(gp["variant"]) & set(gc["variant"])):
            pair = (gp.loc[gp["variant"] == v, "raw_string"].iloc[0],
                    gc.loc[gc["variant"] == v, "raw_string"].iloc[0])
            if pair != pairs[0][1:]:          # the best runs again: one comparison is enough
                pairs.append((v, *pair))
        for kind, rp, rc in pairs:
            p, c = e.loc[rp], e.loc[rc]
            rows.append(dict(chain=ed.chain, parent=ed.parent, child=ed.child, variant=kind,
                             parent_run=rp, child_run=rc, parent_eci=p["mean"],
                             child_eci=c["mean"], parent_hdi_low=p["hdi_low"],
                             child_hdi_high=c["hdi_high"],
                             drop=p["mean"] - c["mean"], n_parent=int(p["n_obs"]),
                             n_child=int(c["n_obs"])))
    return pd.DataFrame(rows)


def regressions(eci: pd.DataFrame, lineage: pd.DataFrame | None = None,
                min_drop: float = MIN_DROP) -> pd.DataFrame:
    """The comparisons whose drop is large and credible (see MIN_DROP), largest first."""
    c = comparisons(eci, lineage)
    if c.empty:
        return c
    flag = ((c["drop"] >= min_drop) & (c["child_hdi_high"] < c["parent_eci"])
            & (c["parent_hdi_low"] > c["child_eci"]))
    return c[flag].sort_values("drop", ascending=False).reset_index(drop=True)
