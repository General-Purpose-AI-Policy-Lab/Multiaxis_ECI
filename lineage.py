"""Release-lineage structure for the compensatory MIRT lineage prior.

Reads the reviewed alias map (data/curated/lineage_map.csv) and turns it into
the index arrays the model needs to build, per chained test-taker,

    theta[m]  = psi[node(m)] + offset[group(m)]
    psi[node] = founder(chain) + cumulative soft increments along the chain

Pure numpy/pandas — no PyMC. See LINEAGE_PLAN.md.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import DATA_DIR

LINEAGE_MAP = DATA_DIR / "curated" / "lineage_map.csv"


@dataclass(frozen=True)
class LineageStructure:
    """Everything the model layer needs; nothing it doesn't.

    Rows are 0-indexed into the model's theta order (sorted unique model_version).
    ``B @ founders + C @ deltas`` gives psi per node without a per-chain loop:
    B is one-hot node->founder, C accumulates a chain's deltas up to each node
    (founder rows are all-zero).
    """
    row_idx: np.ndarray        # (R,) chained model rows
    B: np.ndarray              # (n_nodes, n_chains) node -> its chain's founder
    C: np.ndarray              # (n_nodes, n_deltas) within-chain cumulative increments
    node_idx: np.ndarray       # (R,) node per row -> gather psi
    offset_group: np.ndarray   # (R,) (node,variant) group per row -> gather offset
    delta_dt: np.ndarray       # (n_deltas,) years between the two nodes each delta bridges
    delta_chain: np.ndarray    # (n_deltas,) chain index (chain_names order) per delta column; post-fit reads only
    founder_date: np.ndarray   # (n_chains,) datetime64 release date of each chain's founder
    n_chains: int
    n_nodes: int
    n_deltas: int
    n_groups: int
    chain_names: list[str]


def build_lineage_structure(mlookup: pd.DataFrame,
                            csv_path=LINEAGE_MAP,
                            collapse_variants: bool = False) -> LineageStructure | None:
    """Build the lineage structure for the models in ``mlookup`` (column
    ``model``, in theta-row order). Returns ``None`` if no usable chain survives.

    A chain is kept if >=2 of its nodes survive the join to the data (steps to
    order), or if its single node keeps >=2 (node, variant) groups — no deltas,
    variant offsets only. All surviving nodes must be dated. A node whose
    only variants were dropped from the data disappears and positions are
    renumbered, so C never carries a gap.
    """
    if not csv_path.exists():
        return None
    m = pd.read_csv(csv_path).drop_duplicates("raw_string")
    m = m[m["in_chain"].astype(str).str.lower() == "yes"].copy()

    if collapse_variants:
        # The caller aggregated evaluation settings, so a model's rows now carry
        # its BASE name and every `raw_string` with an effort/token/provider
        # suffix would silently miss the join, taking whole chains with it. Map
        # the keys the same way. Safe because no base model spans two
        # (chain, node) pairs — variants of one release always share a node —
        # and `variant` stops meaning anything once the settings are merged.
        from data import _effort_base
        m["raw_string"] = m["raw_string"].map(_effort_base)
        m["variant"] = "bare"
        m = m.drop_duplicates("raw_string")

    row_of = {name: i for i, name in enumerate(mlookup["model"].tolist())}
    m = m[m["raw_string"].isin(row_of)]
    if m.empty:
        return None
    m["row"] = m["raw_string"].map(row_of)
    m["date"] = pd.to_datetime(m["node_date"], errors="coerce")

    # A chain earns its keep two ways: >=2 nodes (there are steps to order), or
    # ONE node with >=2 (node, variant) groups — zero deltas, but the variant
    # offsets still tie the release's effort rungs to one ability, which is the
    # whole point for a first-release vendor with no predecessor to parent onto
    # (Inkling). A lone bare release still drops: one row, nothing to tie.
    keep = [c for c, g in m.groupby("chain")
            if (g["node"].nunique() >= 2
                or g[["node", "variant"]].drop_duplicates().shape[0] >= 2)
            and g["date"].notna().all()]
    m = m[m["chain"].isin(keep)]
    if m.empty:
        return None

    # Global node order: chains sorted by name, nodes within a chain by date.
    # Contiguous per chain so each chain's delta columns are a clean block.
    #
    # A node's predecessor is the previous SPINE node by date. Date is the only
    # ordering signal there (anti-circularity rule), so two spine nodes sharing a
    # date leave the order — and which node founds the chain — ambiguous; raise
    # rather than let a lexical name tiebreak pick it silently.
    #
    # The optional `parent` column names a predecessor explicitly, which makes
    # the chain a TREE: a side line (a different product line built off a known
    # release) hangs off its branch point instead of being spliced into the
    # spine. Spliced in, it both breaks the spine's own step in two and claims a
    # near-zero gap to whichever releases happen to bracket it, which under the
    # Brownian prior (step sd scales with sqrt(dt)) is a strong similarity claim
    # made on calendar proximity alone. A parented node is off the spine, so it
    # never orders anything and may share a date with its siblings.
    has_parent = "parent" in m.columns
    chain_names = sorted(m["chain"].unique())
    node_keys: list[tuple[str, str]] = []
    date_of: dict[tuple[str, str], pd.Timestamp] = {}
    parent_of: dict[tuple[str, str], tuple[str, str] | None] = {}
    for c in chain_names:
        cols = ["node", "date"] + (["parent"] if has_parent else [])
        nd_sorted = (m[m["chain"] == c][cols]
                     .drop_duplicates("node").sort_values("date"))
        parents = (nd_sorted["parent"] if has_parent
                   else pd.Series(pd.NA, index=nd_sorted.index))
        spine = nd_sorted[parents.isna()]
        tied = spine[spine["date"].duplicated(keep=False)]
        if not tied.empty:
            raise ValueError(
                f"chain {c!r} has distinct spine nodes sharing a date, so their order "
                f"is ambiguous:\n{tied.sort_values('date').to_string(index=False)}\n"
                f"Give one an explicit `parent` to move it off the spine, or split "
                f"the chain.")
        node_keys.extend((c, nd) for nd in nd_sorted["node"])
        date_of.update({(c, nd): d for nd, d in zip(nd_sorted["node"], nd_sorted["date"])})
        prev = None
        for nd in spine["node"]:
            parent_of[(c, nd)] = None if prev is None else (c, prev)
            prev = nd
        for nd, p in zip(nd_sorted["node"], parents):
            if pd.notna(p):
                if (c, p) not in set(zip([c] * len(nd_sorted), nd_sorted["node"])):
                    raise ValueError(
                        f"chain {c!r}: node {nd!r} names parent {p!r}, which is not a "
                        f"node of that chain (in this fit's data).")
                parent_of[(c, nd)] = (c, p)
    node_pos = {k: i for i, k in enumerate(node_keys)}
    chain_pos = {c: i for i, c in enumerate(chain_names)}
    n_nodes, n_chains = len(node_keys), len(chain_names)

    # B and C. Every non-founder node owns exactly one delta, the step coming
    # into it, so n_deltas is unchanged by branching. A node's C row marks every
    # delta on its path back to the founder: a prefix of the date order on a
    # plain chain, the branch path on a tree.
    n_deltas = n_nodes - n_chains
    B = np.zeros((n_nodes, n_chains))
    C = np.zeros((n_nodes, n_deltas))
    delta_dt = np.zeros(n_deltas)
    delta_chain = np.zeros(n_deltas, dtype=int)
    founder_date = np.empty(n_chains, dtype="datetime64[ns]")
    col = 0
    for c in chain_names:
        chain_node_keys = [k for k in node_keys if k[0] == c]
        founders = [k for k in chain_node_keys if parent_of[k] is None]
        if len(founders) != 1:
            raise ValueError(
                f"chain {c!r} needs exactly one unparented founder, found "
                f"{[k[1] for k in founders]}.")
        # The reviewed map date, the same signal that ordered the spine — not the
        # processed file's release_date, which can disagree or be missing.
        founder_date[chain_pos[c]] = date_of[founders[0]]
        delta_col = {}
        for k in chain_node_keys:
            if parent_of[k] is not None:
                delta_col[k] = col
                col += 1
        for k in chain_node_keys:
            ni = node_pos[k]
            B[ni, chain_pos[c]] = 1.0
            walk, seen = k, set()
            while parent_of[walk] is not None:
                if walk in seen:
                    raise ValueError(f"chain {c!r} has a parent cycle through {walk[1]!r}.")
                seen.add(walk)
                C[ni, delta_col[walk]] = 1.0
                walk = parent_of[walk]
        for k, dc in delta_col.items():
            dt_years = (date_of[k] - date_of[parent_of[k]]).days / 365.25
            if dt_years <= 0:
                raise ValueError(
                    f"chain {c!r}: {k[1]!r} ({date_of[k].date()}) is not after its "
                    f"parent {parent_of[k][1]!r} ({date_of[parent_of[k]].date()}); "
                    f"gap {dt_years:.3f} years.")
            delta_dt[dc] = dt_years
            delta_chain[dc] = chain_pos[c]

    # Offset groups = unique (chain, node, variant); alias rows share a group and
    # therefore an offset, so they collapse to identical theta.
    m = m.sort_values("row")
    grp_keys = sorted(set(zip(m["chain"], m["node"], m["variant"])))
    grp_pos = {k: i for i, k in enumerate(grp_keys)}

    row_idx = m["row"].to_numpy()
    node_idx = np.array([node_pos[(c, nd)] for c, nd in zip(m["chain"], m["node"])])
    offset_group = np.array([grp_pos[(c, nd, v)]
                             for c, nd, v in zip(m["chain"], m["node"], m["variant"])])

    return LineageStructure(
        row_idx=row_idx, B=B, C=C, node_idx=node_idx, offset_group=offset_group,
        delta_dt=delta_dt, delta_chain=delta_chain, founder_date=founder_date,
        n_chains=n_chains, n_nodes=n_nodes, n_deltas=n_deltas,
        n_groups=len(grp_keys), chain_names=chain_names)
