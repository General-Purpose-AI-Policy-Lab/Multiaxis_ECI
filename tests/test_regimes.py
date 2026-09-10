"""Two-regime forecast: top-k frontier, weighted line on medians, crossings on the piecewise frontier."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from multiaxis_eci.analysis import (
    frontier_topk,
    is_reasoning,
    regime_crossover_df,
    two_regime_forecast,
    weighted_line_fit,
)


def test_frontier_topk_and_reasoning_flag():
    tl = pd.DataFrame({"name": ["a", "b", "c", "d", "e"],
                       "release_date": pd.to_datetime(["2024-01-01", "2024-06-01", "2025-01-01",
                                                       "2025-06-01", "2026-01-01"]),
                       "mean": [0.0, 1.0, 0.5, 2.0, 1.2]})
    assert frontier_topk(tl, 1) == ["a", "b", "d"]                 # running max, records only
    assert frontier_topk(tl, 2) == ["a", "b", "c", "d", "e"]       # e beats c: second best so far
    assert is_reasoning("o3-2025-04-16_high") and is_reasoning("DeepSeek-R1")
    assert is_reasoning("gpt-5-2025-08-07") and not is_reasoning("gpt-4o-2024-05-13")
    assert not is_reasoning("Llama-3-70b")


def test_weighted_line_fit_recovers_a_slope():
    rng = np.random.default_rng(0)
    t = np.linspace(2024.0, 2026.5, 12)
    sd = np.full(12, 0.15)
    y = -1.0 + 0.8 * (t - 2024.0) + rng.normal(0, 0.15, 12)
    fit = weighted_line_fit(t, y, sd, [f"m{i}" for i in range(12)])
    assert abs(np.median(fit.b) - 0.8) < 0.15
    assert fit.t0 == 2024.0 and fit.names[0] == "m0"
    # A precise point pulls the line more than a wide one.
    y2 = y.copy(); y2[-1] += 2.0
    loose = weighted_line_fit(t, y2, np.where(np.arange(12) == 11, 2.0, 0.15), fit.names)
    tight = weighted_line_fit(t, y2, sd, fit.names)
    assert np.median(loose.b) < np.median(tight.b)


@dataclass
class _Data:
    mlookup: pd.DataFrame
    is_human: np.ndarray


def test_two_regime_forecast_and_crossings():
    rng = np.random.default_rng(1)
    # Others: 2022-2024 at slope 0.3; reasoning: from 2024-09, higher and at slope 0.9.
    others = [(f"llama-{i}", pd.Timestamp("2022-01-01") + pd.DateOffset(months=4 * i),
               -1.0 + 0.3 * i / 3) for i in range(8)]
    reason = [(f"o3-2025-01-01_v{i}", pd.Timestamp("2024-09-01") + pd.DateOffset(months=3 * i),
               0.2 + 0.9 * i / 4) for i in range(8)]
    names = [n for n, _, _ in others + reason] + ["Average Human", "Top Performer"]
    level = np.array([m for _, _, m in others + reason] + [0.0, 2.5])
    S = 300
    theta = level[None, :, None] + rng.normal(0, 0.1, (S, len(names), 1))
    data = _Data(pd.DataFrame({"model": names, "model_idx": np.arange(1, len(names) + 1)}),
                 np.array([False] * 16 + [True, True]))
    tl = pd.DataFrame({"name": names[:16], "release_date": [d for _, d, _ in others + reason],
                       "mean": np.median(theta[:, :16, 0], 0)})
    fc = two_regime_forecast(theta, 0, data, tl, top_k=2, hdi_prob=0.8, horizon_date="2030-01-01")
    assert fc.fit_basis == "regimes" and fc.other is not None
    assert abs(np.median(fc.slope) - 0.9) < 0.2 and abs(np.median(fc.other.slope) - 0.3) < 0.15
    assert pd.Timestamp(fc.grid_dates[0]) == pd.Timestamp("2024-09-01")          # from the switch
    assert pd.Timestamp(fc.other.grid_dates[-1]) <= pd.Timestamp("2024-06-01")   # to its last point
    assert abs(fc.switch_year - 2024.67) < 0.01
    cx = regime_crossover_df(fc, theta, 0, data, axis_name="axis1", today="2026-09-10")
    top = cx[cx.tier == "Top Performer"].iloc[0]
    avg = cx[cx.tier == "Average Human"].iloc[0]
    # Top Performer (2.5) is ahead on the reasoning line: 0.2 + 0.9 (t - 2024.67) = 2.5 -> ~2027-02.
    assert top.status == "future" and pd.Timestamp("2026-10-01") < top.crossover_date_median < pd.Timestamp("2027-08-01")
    # Average Human (0.0) was above the others' line when reasoning arrived at 0.2: the
    # piecewise frontier crosses it at the switch itself, never later.
    assert avg.status == "passed_ci"
    assert pd.Timestamp("2024-08-01") < avg.crossover_date_median <= pd.Timestamp("2024-09-02")
    with pytest.raises(ValueError):
        two_regime_forecast(theta, 0, data, tl[tl["name"].str.startswith("llama")], top_k=2)
