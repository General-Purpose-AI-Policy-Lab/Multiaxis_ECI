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


def test_rgba_accepts_hex_and_rgb():
    from multiaxis_eci.viz.core import _rgba
    assert _rgba("#c0504d", 0.6) == "rgba(192,80,77,0.6)"
    assert _rgba("rgb(1,2,3)", 0.5) == "rgba(1,2,3,0.5)"


def test_trend_fig_colours_fit_sets():
    """The releases each regime's line was fitted on are drawn in that regime's colour, the
    rest of the cloud in the model colour."""
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    from multiaxis_eci.analysis.forecast import ForecastResult
    from multiaxis_eci.viz import frontier_trend_fig
    from multiaxis_eci.viz.core import MODEL_COLOR
    from multiaxis_eci.viz.forecast import OTHER_FIT_COLOR, REASONING_FIT_COLOR

    grid = pd.date_range("2024-01-01", "2030-01-01", freq="MS").values
    n = len(grid)
    other = ForecastResult(grid_dates=grid[:12], median=np.zeros(12), lo=-np.ones(12),
                           hi=np.ones(12), slope=np.ones(3), intercept=np.zeros(3),
                           frontier_names=["a", "b"], last_obs_date=pd.Timestamp("2024-12-01"),
                           fit_names=["a", "b"], fit_basis="regimes", kind="line")
    fc = ForecastResult(grid_dates=grid, median=np.linspace(0, 2, n), lo=np.linspace(-1, 1, n),
                        hi=np.linspace(1, 3, n), slope=np.ones(3), intercept=np.zeros(3),
                        frontier_names=["a", "b", "c", "d"],
                        last_obs_date=pd.Timestamp("2026-01-01"), fit_names=["c", "d"],
                        fit_basis="regimes", kind="line", other=other, switch_year=2025.0)
    tl = pd.DataFrame({"name": list("abcde"),
                       "release_date": pd.to_datetime(["2024-01-01", "2024-06-01", "2025-01-01",
                                                       "2025-09-01", "2025-03-01"]),
                       "mean": [0.1, 0.2, 1.0, 1.4, 0.3],
                       "hdi_low": [0, 0.1, 0.8, 1.2, 0.1], "hdi_high": [0.2, 0.3, 1.2, 1.6, 0.5]})
    hs = pd.DataFrame({"name": ["Average Human", "Top Performer"], "mean": [0.0, 1.0],
                       "hdi_low": [-0.2, 0.8], "hdi_high": [0.2, 1.2]})
    fig = frontier_trend_fig({"axis1": dict(fc=fc, tl=tl, hs=hs)}, ["axis1"],
                             today="2026-01-01")
    by_colour = {}
    for tr in fig.data:
        if isinstance(tr, go.Scatter) and tr.mode == "markers" and tr.text is not None:
            by_colour.setdefault(tr.marker.color, set()).update(tr.text)
    assert by_colour[REASONING_FIT_COLOR] == {"c", "d"}
    assert by_colour[OTHER_FIT_COLOR] == {"a", "b"}
    assert by_colour[MODEL_COLOR] == {"e"}
    assert "192,80,77" in {tr.error_y.color for tr in fig.data if tr.mode == "markers"
                           and tr.marker.color == REASONING_FIT_COLOR}.pop()
