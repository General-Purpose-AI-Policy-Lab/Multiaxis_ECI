"""Rows of the per-axis forests: top models, the newest releases of their lines and human tiers."""
from __future__ import annotations

from textwrap import shorten

import numpy as np
import pandas as pd

from multiaxis_eci.analysis.stats import _release_dates
from multiaxis_eci.analysis.timelines import candidate_mask
from multiaxis_eci.config import FOREST_PINNED_RELEASES
from multiaxis_eci.data import ECIData, model_product
from multiaxis_eci.lineage import build_lineage_structure


def line_tips(data: ECIData) -> tuple[dict[int, int], list[tuple[int, np.ndarray]]]:
    """The lines of `1_curated/lineage_map.csv` and their current tips, on this fit's rows.

    Returns (chain of each chained model row, [(chain, model rows of the node)] for every tip
    node). A tip is a node no other node names as its predecessor: the newest release of the
    spine, and the end of each side line (Mythos 5 off Mythos Preview, beside Fable 5.1).
    Read off the reviewed map, so a new release joins as soon as the map files it.
    """
    lin = build_lineage_structure(data.mlookup)
    if lin is None or lin.node_parent is None:
        return {}, []
    node_chain = lin.B.argmax(axis=1)
    chain_of = {int(r): int(node_chain[n]) for r, n in zip(lin.row_idx, lin.node_idx)}
    tips = sorted(set(range(lin.n_nodes)) - set(lin.node_parent[lin.node_parent >= 0].tolist()))
    return chain_of, [(int(node_chain[n]), lin.row_idx[lin.node_idx == n]) for n in tips]


def forest_frames(view, data: ECIData, raw_df: pd.DataFrame | None = None, n_top: int = 11,
                  pinned: set[str] = FOREST_PINNED_RELEASES, hdi_prob: float = 0.95,
                  by_family: bool = False, **gate) -> list[pd.DataFrame]:
    """Per axis, the forest rows: the top `n_top` models by median ability among the timeline
    candidates (`candidate_mask`, the forecast's gate, so the forest and the forecast can
    never show different frontiers), the frontier releases, and every human tier.

    The frontier releases are the tips (`line_tips`) of every lineage line with a model among
    the top rows, each as its best effort on the axis, joined whatever the coverage gate says
    as long as the release was scored on a benchmark that loads on the axis: the newest
    release of a leading line is the headline of the figure, so it is drawn with its wide
    interval rather than dropped for having few scores yet. Being read off the lineage map,
    the set follows every data update by itself; `pinned` adds named runs on top (empty by
    default, config.FOREST_PINNED_RELEASES).

    `by_family` collapses the candidates to one row per release (`data.model_product`: base
    model plus snapshot, a pro mode or access tier apart), keeping the effort with the highest median on the axis, so the forest
    ranks releases rather than reasoning efforts; the row keeps the winning effort's name.

    Undated models cannot be candidates; without `raw_df` every model counts as dated. Rows are
    ascending by median, so the strongest lands at the top of a panel. Labels are shortened at
    word boundaries; a label that collides after shortening keeps its full text, because Plotly
    merges duplicate categories into one row.
    """
    from multiaxis_eci.viz.core import pretty_model_name  # lazy: viz imports analysis

    names = data.mlookup.sort_values("model_idx")["model"].tolist()
    is_h = np.asarray(data.is_human, dtype=bool)
    if raw_df is not None:
        dates, _ = _release_dates(raw_df)
    else:
        dates = pd.Series(pd.Timestamp("2000-01-01"), index=names)
    q_lo, q_hi = (1 - hdi_prob) / 2 * 100, (1 + hdi_prob) / 2 * 100
    chain_of, tips = line_tips(data)
    frames = []
    for k in range(view.K):
        th = view.theta[:, :, k]
        med = np.median(th, axis=0)
        lo, hi = np.percentile(th, [q_lo, q_hi], axis=0)
        ok = candidate_mask(view.theta, k, data, dates, **gate)
        ranked = [i for i in np.argsort(-med) if ok[i]]
        if by_family:
            seen: set[str] = set()
            ranked = [i for i in ranked
                      if not (model_product(names[i]) in seen or seen.add(model_product(names[i])))]
        top = ranked[:n_top]
        shown = {model_product(names[i]) if by_family else names[i] for i in top}
        key = model_product if by_family else (lambda name: name)
        # Scored on a benchmark that loads on the axis, if only a little.
        seen = candidate_mask(view.theta, k, data, dates, **{**gate, "min_coverage": 1e-9})
        lines = {chain_of[i] for i in top if i in chain_of}
        frontier = []
        for chain, rows in tips:
            rows = [int(i) for i in rows if seen[i]]
            if chain not in lines or not rows or any(key(names[i]) in shown for i in rows):
                continue
            best = max(rows, key=lambda i: med[i])
            frontier.append(best)
            shown.add(key(names[best]))
        for p in sorted(pinned):
            members = [i for i in range(len(names)) if not is_h[i]
                       and (model_product(names[i]) == model_product(p) if by_family
                            else names[i] == p)]
            if members and not any(key(names[i]) in shown for i in members):
                best = max(members, key=lambda i: med[i])
                frontier.append(best)
                shown.add(key(names[best]))
        rows = ([(i, "model") for i in top]
                + [(i, "frontier") for i in frontier]
                + [(i, "human") for i in np.where(is_h)[0]])
        rows.sort(key=lambda r: med[r[0]])
        full = [names[i] if is_h[i] else pretty_model_name(names[i]) for i, _ in rows]
        short = [shorten(f, width=34, placeholder=" …") for f in full]
        dup = {s for s in short if short.count(s) > 1}
        frames.append(pd.DataFrame({
            "name": [f if s in dup else s for s, f in zip(short, full)],
            "kind": [kind for _, kind in rows],
            "mean": [med[i] for i, _ in rows],
            "hdi_low": [lo[i] for i, _ in rows],
            "hdi_high": [hi[i] for i, _ in rows]}))
    return frames
