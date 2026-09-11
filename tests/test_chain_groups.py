"""Chain-group comparison figures draw both groups on every panel."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from multiaxis_eci.viz.chain_groups import (
    abilities_compare_fig,
    chain_logp_fig,
    crossover_compare_fig,
    human_tiers_compare_fig,
    loadings_compare_fig,
)


@dataclass
class _Data:
    mlookup: pd.DataFrame
    blookup: pd.DataFrame
    is_human: np.ndarray


def _fixture():
    rng = np.random.default_rng(1)
    S, B, M, K = 40, 6, 8, 2
    A_a = np.abs(rng.normal(1, 0.2, (S, B, K)))
    A_b = A_a + rng.normal(0, 0.05, (S, B, K))
    th_a = rng.normal(0, 1, (S, M, K))
    th_b = th_a + rng.normal(0, 0.1, (S, M, K))
    names = ["axis1", "axis2"]
    bench = [f"b{i}" for i in range(B)]
    data = _Data(pd.DataFrame({"model": [f"m{i}" for i in range(M - 2)] + ["Average Human",
                                                                          "Top Performer"],
                               "model_idx": np.arange(1, M + 1)}),
                 pd.DataFrame({"benchmark": bench, "benchmark_idx": np.arange(1, B + 1)}),
                 np.array([False] * (M - 2) + [True, True]))
    return A_a, A_b, th_a, th_b, names, bench, data


def test_group_builders():
    A_a, A_b, th_a, th_b, names, bench, data = _fixture()
    titles = {"axis1": "Axis 1: Fluid", "axis2": "Axis 2"}
    doc = {"chain_delta_logp": [0.0, -3.0, -9.5, -1.0],
           "modes": [{"label": "A", "chains": [0, 3]}, {"label": "B", "chains": [1, 2]}]}
    split = {"majority": [0, 3], "minority": [1, 2], "n_chains": 4}
    lp = chain_logp_fig(doc, split)
    assert list(lp.data[0].marker.color)[:2] == ["#0072B2", "#D55E00"]

    ld = loadings_compare_fig(A_a, A_b, bench, names, titles, top_n=4)
    assert [a.text for a in ld.layout.annotations][:2] == ["Axis 1: Fluid", "Axis 2"]
    assert sum(t.showlegend for t in ld.data if t.showlegend) == 2      # one entry per group
    assert len([t for t in ld.data if t.mode == "markers"]) == 4        # 2 groups x 2 axes

    ab = abilities_compare_fig(th_a, th_b, data, names, titles)
    r_notes = [a for a in ab.layout.annotations if a.text.startswith("r = ")]
    assert len(r_notes) == 2 and float(r_notes[0].text[4:]) > 0.9
    squares = [t for t in ab.data if t.marker.symbol == "square"]
    assert len(squares) == 2 and len(squares[0].x) == 2                 # the two tiers

    ht = human_tiers_compare_fig(th_a, th_b, data, names, titles)
    assert list(ht.layout.yaxis.ticktext) in (["Average Human", "Top Performer"],
                                              ["Top Performer", "Average Human"])
    assert isinstance(ht, go.Figure)

    cx = pd.DataFrame({"axis": ["axis1", "axis1"], "tier": ["Average Human", "Top Performer"],
                       "human_mean": [0.0, 2.0],
                       "crossover_date_median": ["2025-06-01", "2029-01-01"],
                       "crossover_hdi_low": ["2024-01-01", "2027-01-01"],
                       "crossover_hdi_high": ["2026-01-01", "2035-01-01"]})
    cx_b = cx.assign(crossover_date_median=["2026-01-01", None])
    cf = crossover_compare_fig(cx, cx_b, ["axis1"], titles, today="2026-09-10")
    markers = [t for t in cf.data if t.mode == "markers"]
    assert [len(t.x) for t in markers] == [2, 1]                        # the None row is skipped
    assert cf.layout.xaxis.range == ("2014-07-25", "2035-06-10")
