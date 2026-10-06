"""Human tiers are placed only on the axes their scores measure, carried along the tier order."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from multiaxis_eci.analysis import ability_label, human_unit_affine
from multiaxis_eci.analysis.human_ident import (
    drop_unidentified,
    human_axis_contraction,
    human_scale_axes,
    identified_tiers,
)


@dataclass
class _Data:
    mlookup: pd.DataFrame
    is_human: np.ndarray
    model_idx: np.ndarray
    bench_idx: np.ndarray


ORDER = {"Average Human": None, "Domain Expert": "Average Human",
         "Top Performer": "Domain Expert"}


def _fit(human_obs):
    """Two axes; benchmark 0 measures axis 0 alone, benchmark 1 is mostly axis 0 with a small
    axis-1 loading, benchmark 2 measures axis 1. `human_obs` lists (tier, benchmark)."""
    names = ["m0", "m1", "Average Human", "Domain Expert", "Top Performer"]
    data = _Data(pd.DataFrame({"model": names, "model_idx": np.arange(5)}),
                 np.array([False, False, True, True, True]),
                 np.array([0, 0, 1, 1, 1] + [names.index(t) for t, _ in human_obs]),
                 np.array([0, 2, 0, 1, 2] + [b for _, b in human_obs]))
    A = np.tile(np.array([[2.0, 0.0], [1.5, 0.5], [0.0, 2.0]]), (10, 1, 1))
    theta = np.zeros((10, 5, 2))
    return A, theta, data


def test_only_dominant_benchmarks_place_a_tier():
    # Many scores on benchmark 1 (axis share 0.9 on axis 0, 0.1 on axis 1): with the full
    # loadings they would pin axis 1 too; kept to the dominant axis they do not.
    A, theta, data = _fit([("Average Human", 1)] * 6 + [("Domain Expert", 0)] * 3)
    c = human_axis_contraction(A, theta, data, ORDER)
    assert list(c.index) == ["Average Human", "Domain Expert", "Top Performer"]
    assert (c[0] > 0.5).all()                       # Top Performer through its parents
    assert np.allclose(c[1], 0.0)
    assert identified_tiers(c, 1) == set()
    assert not human_scale_axes(c, 2)[1] and human_scale_axes(c, 2)[0]
    full = human_axis_contraction(A, theta, data, ORDER, share_min=0.0)
    assert (full[1] > 0.5).all()                    # the extrapolation the threshold removes


def test_information_flows_down_the_order():
    # Only the Top Performer is scored on axis 1: its parents learn through the increments,
    # less than it does.
    A, theta, data = _fit([("Average Human", 0)] * 3 + [("Top Performer", 2)] * 5)
    c = human_axis_contraction(A, theta, data, ORDER)
    assert c.loc["Top Performer", 1] > c.loc["Domain Expert", 1] > c.loc["Average Human", 1] > 0


def test_tiers_and_scale_follow_the_contraction():
    A, theta, data = _fit([("Average Human", 0)] * 3)
    c = human_axis_contraction(A, theta, data, ORDER)
    hs = pd.DataFrame({"name": ["Average Human", "Top Performer", "m0"], "mean": [0, 1, 2]})
    assert list(drop_unidentified(hs, 1, c)["name"]) == ["m0"]
    assert drop_unidentified(hs, 0, c).equals(hs)
    assert drop_unidentified(hs, 1, None) is hs     # no contraction: every tier drawn

    rng = np.random.default_rng(0)
    th = rng.normal(0, 1, (50, 5, 2))
    th[:, 2, :] = rng.normal(-1.0, 0.1, (50, 2))
    th[:, 4, :] = rng.normal(2.0, 0.1, (50, 2))
    origin, unit = human_unit_affine(th, data, human_scale_axes(c, 2))
    assert origin[1] == 0.0 and unit[1] == 1.0      # fitted scale on the unplaced axis
    np.testing.assert_allclose(origin[0], np.median(th[:, 2, 0]))
    assert human_unit_affine(th, data, np.array([False, False])) is None
    assert ability_label((origin, unit), human=False) == "ability (fitted scale)"


def test_no_order_no_test():
    A, theta, data = _fit([("Average Human", 0)])
    assert human_axis_contraction(A, theta, data, None) is None
    assert human_axis_contraction(None, theta, data, ORDER) is None
