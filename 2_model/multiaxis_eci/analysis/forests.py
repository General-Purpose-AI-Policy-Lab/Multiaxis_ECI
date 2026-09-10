"""Rows of the per-axis forests: top models, pinned frontier releases and human tiers."""
from __future__ import annotations

from textwrap import shorten

import numpy as np
import pandas as pd

from multiaxis_eci.analysis.stats import _release_dates
from multiaxis_eci.analysis.timelines import candidate_mask
from multiaxis_eci.config import FOREST_PINNED_RELEASES
from multiaxis_eci.data import ECIData, model_family


def forest_frames(view, data: ECIData, raw_df: pd.DataFrame | None = None, n_top: int = 11,
                  pinned: set[str] = FOREST_PINNED_RELEASES, hdi_prob: float = 0.95,
                  by_family: bool = False, **gate) -> list[pd.DataFrame]:
    """Per axis, the forest rows: the top `n_top` models by median ability among the timeline
    candidates (`candidate_mask` with the forecast's gate, so the forest and the forecast can
    never show different frontiers), the `pinned` frontier releases, and every human tier.

    `by_family` collapses the candidates to one row per release (`data.model_family`: base
    model plus snapshot), keeping the effort with the highest median on the axis, so the forest
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
                      if not (model_family(names[i]) in seen or seen.add(model_family(names[i])))]
        top = ranked[:n_top]
        if by_family:
            # A pinned release is a family here too: it joins as its best effort, and only
            # when no effort of that family is already among the top rows.
            shown = {model_family(names[i]) for i in top}
            pinned_fams = {model_family(p) for p in pinned} - shown
            frontier = []
            for fam_name in sorted(pinned_fams):
                members = [i for i in range(len(names))
                           if not is_h[i] and model_family(names[i]) == fam_name]
                if members:
                    frontier.append(max(members, key=lambda i: med[i]))
        else:
            frontier = [i for i in range(len(names))
                        if names[i] in pinned and not is_h[i] and i not in top]
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
