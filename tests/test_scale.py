"""Human units: Average Human at 0 and Top Performer at 1 on every axis, on finished summaries."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from multiaxis_eci.analysis import (
    ability_label,
    human_unit_affine,
    rescale_forecast,
    rescale_frame,
    rescale_theta,
)
from multiaxis_eci.analysis.forecast import ForecastResult


@dataclass
class _Data:
    mlookup: pd.DataFrame


def test_human_units():
    names = ["m0", "m1", "Average Human", "Top Performer"]
    rng = np.random.default_rng(0)
    theta = rng.normal(0, 1, (50, 4, 2))
    theta[:, 2, :] = rng.normal(-1.0, 0.1, (50, 2))
    theta[:, 3, :] = rng.normal(2.0, 0.1, (50, 2))
    data = _Data(pd.DataFrame({"model": names, "model_idx": np.arange(1, 5)}))
    affine = human_unit_affine(theta, data)
    assert affine is not None
    origin, unit = affine
    np.testing.assert_allclose(origin, np.median(theta[:, 2, :], 0))
    np.testing.assert_allclose(unit, np.median(theta[:, 3, :], 0) - origin)
    hs = pd.DataFrame({"name": ["Average Human", "Top Performer"],
                       "mean": np.median(theta[:, 2:, 0], 0), "hdi_low": [0, 0], "hdi_high": [1, 1]})
    out = rescale_frame(hs, 0, affine)
    np.testing.assert_allclose(out["mean"], [0.0, 1.0])
    assert hs["mean"].iloc[1] != 1.0                                # the original is untouched
    scaled = rescale_theta(theta, affine)
    np.testing.assert_allclose(np.median(scaled[:, 2, :], 0), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.median(scaled[:, 3, :], 0), 1.0, atol=1e-12)
    fc = ForecastResult(grid_dates=np.array(["2026-01-01", "2027-01-01"], dtype="datetime64[D]"),
                        median=np.array([2.0, 5.0]), lo=np.array([1.0, 4.0]), hi=np.array([3.0, 6.0]),
                        slope=np.full(5, 3.0), intercept=np.zeros(5), frontier_names=[],
                        last_obs_date=pd.Timestamp("2026-01-01"))
    f2 = rescale_forecast(fc, 0, affine)
    np.testing.assert_allclose(f2.slope, 3.0 / unit[0])
    np.testing.assert_allclose(f2.median, (fc.median - origin[0]) / unit[0])
    assert ability_label(affine).startswith("ability (Average Human = 0")
    # No anchors in the fit: no scale, frames pass through, the caption stays generic.
    data2 = _Data(pd.DataFrame({"model": ["a", "b", "c", "d"], "model_idx": np.arange(1, 5)}))
    assert human_unit_affine(theta, data2) is None
    assert rescale_frame(hs, 0, None) is hs and ability_label(None) == "ability"
