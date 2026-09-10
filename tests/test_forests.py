"""Forest rows: top candidates, pinned releases, human tiers, and one row per family."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from multiaxis_eci.analysis import forest_frames


@dataclass
class _View:
    theta: np.ndarray
    K: int


@dataclass
class _Data:
    mlookup: pd.DataFrame
    is_human: np.ndarray
    is_low_obs: np.ndarray

    @property
    def n_models(self):
        return len(self.mlookup)

    @property
    def is_sota(self):
        return np.zeros(self.n_models, bool)


def test_forest_rows_and_family_collapse():
    names = ["gpt-5-2025-08-07_high", "gpt-5-2025-08-07_medium", "gpt-5-2025-08-07",
             "claude-opus-4-7_max", "claude-opus-4-7_low", "llama-3-70b", "Average Human"]
    rng = np.random.default_rng(0)
    level = np.array([2.0, 1.8, 1.5, 1.9, 1.2, 0.3, 1.0])
    theta = level[None, :, None] + rng.normal(0, 0.05, (200, len(names), 1))
    view = _View(theta, 1)
    data = _Data(pd.DataFrame({"model": names, "model_idx": np.arange(1, len(names) + 1)}),
                 np.array([False] * 6 + [True]), np.zeros(len(names), bool))
    gate = dict(sd_cap=None, drop_low_obs=False, sota_exempt=False)
    (df,) = forest_frames(view, data, n_top=4, pinned={"llama-3-70b"}, **gate)
    # Ascending by median: the strongest lands at the top of the panel; the pinned
    # release is a frontier row even outside the top n; the tier is a human row.
    assert list(df["kind"]) == ["frontier", "human", "model", "model", "model", "model"]
    assert df["mean"].is_monotonic_increasing
    # Pinned releases are families too: a pinned effort whose family already has a row
    # adds nothing; a pinned family outside the top rows joins as its best effort.
    (fam,) = forest_frames(view, data, n_top=2, pinned={"gpt-5-2025-08-07", "llama-3-70b"},
                           by_family=True, **gate)
    assert list(fam["kind"]) == ["frontier", "human", "model", "model"]
    assert fam.loc[fam["kind"] == "frontier", "name"].item() == "Llama 3 70b"
    (fam,) = forest_frames(view, data, n_top=4, pinned=set(), by_family=True, **gate)
    models = fam[fam["kind"] == "model"]
    # One row per release, its best effort: gpt-5 high, claude opus 4.7 max, llama.
    assert len(models) == 3
    assert list(models.sort_values("mean", ascending=False)["name"]) == [
        "GPT 5 (high)", "Claude Opus 4 7 (max)", "Llama 3 70b"]
