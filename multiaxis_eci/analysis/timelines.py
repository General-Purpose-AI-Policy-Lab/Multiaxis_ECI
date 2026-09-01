"""Release-date timeline frames for abilities and difficulties."""
from __future__ import annotations

import numpy as np
import pandas as pd

from multiaxis_eci import config
from multiaxis_eci.analysis.convergence import nc_difficulty_draws
from multiaxis_eci.analysis.stats import _release_dates, forest_stats_from_draws, post_stats
from multiaxis_eci.data import ECIData

def mirt_informed_mask(theta_canon: np.ndarray, sd_cap: float = 0.33) -> np.ndarray:
    """(M, K) bool: True where a model's per-axis posterior SD < sd_cap.

    Filters out models whose ability is extrapolated rather than measured
    (sparse old models can land at misleadingly high ability with wide CIs).

    Default 0.4 (history 0.6 → 0.3 → 0.4): 0.6 corresponds to a 95% CI width
    of ~2.35 — far too wide on this scale, it admitted ~10 pre-2023 models at
    frontier levels. 0.3 removed them but froze thin axes' record chains
    (axis 4: 2 post-Oct-2024 records, the forecast line detached from the
    data). 0.4 (95% CI width ~1.57) keeps the ghosts out while giving every
    axis a usable record set — the measured-model definition shared by every
    graph (2026-08-13).
    """
    return theta_canon.std(axis=0) < sd_cap


def mirt_model_timeline_df(theta_canon: np.ndarray, k: int,
                           data: ECIData, raw_df: pd.DataFrame,
                           sd_cap: float | None = 0.33,
                           drop_low_obs: bool = True,
                           hdi_prob: float = 0.5) -> pd.DataFrame:
    """Model-ability timeline for axis k (kind='model'); humans excluded
    (they have no release date and are drawn as reference bands instead).

    Two filters, both toggleable so callers can build an 'all models' view and
    a clean 'informed' view from the same trace:
      * drop_low_obs — skip models flagged is_low_obs (< LOW_OBS_THRESHOLD obs).
      * sd_cap — skip models whose axis-k ability is not data-informed
        (posterior SD >= sd_cap; see mirt_informed_mask). Pass None to disable.
    Defaults give the clean view; pass sd_cap=None, drop_low_obs=False for all
    dated models (including the sparse old ones with near-prior-wide CIs).

    SOTA models (data.is_sota) are EXEMPT from both filters and always shown,
    even when sparse and wide (e.g. Fable 5 / Mythos): a frontier release is the
    headline of the timeline, and its uncertainty is communicated honestly by the
    drawn CI rather than by silently dropping the point. Humans and the
    missing-release-date guard still apply to every model."""
    model_dates, _ = _release_dates(raw_df)
    names = data.mlookup.sort_values("model_idx")["model"].tolist()
    informed = (mirt_informed_mask(theta_canon, sd_cap)[:, k] if sd_cap is not None
                else np.ones(data.n_models, dtype=bool))
    sota = (data.is_sota if data.is_sota is not None
            else np.zeros(data.n_models, dtype=bool))
    rows = []
    for i, m in enumerate(names):
        if data.is_human[i] or m not in model_dates.index:
            continue
        if not sota[i]:                       # SOTA bypasses the low-obs / sd_cap filters
            if drop_low_obs and data.is_low_obs[i]:
                continue
            if not informed[i]:
                continue
        mean, lo, hi = post_stats(theta_canon[:, i, k], hdi_prob=hdi_prob)
        rows.append({"name": m, "kind": "model", "release_date": model_dates[m],
                     "mean": mean, "hdi_low": lo, "hdi_high": hi})
    out = pd.DataFrame(rows)
    out["release_date"] = pd.to_datetime(out["release_date"])
    return out


def mirt_difficulty_timeline_df(trace, data: ECIData,
                                raw_df: pd.DataFrame,
                                hdi_prob: float = 0.5) -> pd.DataFrame:
    """Shared benchmark intercept D_b timeline (kind='benchmark'). D_b is
    sign-invariant and axis-independent, so no canonicalization is needed."""
    D = trace.posterior["D"].values.reshape(-1, data.n_benchmarks)
    _, bench_dates = _release_dates(raw_df)
    names = data.blookup.sort_values("benchmark_idx")["benchmark"].tolist()
    rows = []
    for j, b in enumerate(names):
        if b not in bench_dates.index:
            continue
        mean, lo, hi = post_stats(D[:, j], hdi_prob=hdi_prob)
        rows.append({"name": b, "kind": "benchmark", "release_date": bench_dates[b],
                     "mean": mean, "hdi_low": lo, "hdi_high": hi})
    out = pd.DataFrame(rows)
    out["release_date"] = pd.to_datetime(out["release_date"])
    return out


def mirt_human_axis_stats(theta_canon: np.ndarray, k: int,
                          data: ECIData,
                          hdi_prob: float = 0.5) -> pd.DataFrame:
    """Per-axis human-group theta_k stats for the timeline reference bands
    (schema: name, mean, hdi_low, hdi_high, n_obs)."""
    if not data.is_human.any():
        return pd.DataFrame(columns=["name", "mean", "hdi_low", "hdi_high", "n_obs"])
    names = data.mlookup.sort_values("model_idx")["model"].tolist()
    rows = []
    for i, m in enumerate(names):
        if not data.is_human[i]:
            continue
        mean, lo, hi = post_stats(theta_canon[:, i, k], hdi_prob=hdi_prob)
        rows.append({"name": m, "mean": mean, "hdi_low": lo, "hdi_high": hi,
                     "n_obs": int(data.n_obs_per_model[i])})
    return pd.DataFrame(rows).sort_values("mean").reset_index(drop=True)


def loadings_forest_df(A_canon: np.ndarray, k: int,
                       bench_names: list[str]) -> pd.DataFrame:
    """Per-benchmark loading on axis k as a forest_fig frame (name/mean/hdi)."""
    return forest_stats_from_draws(A_canon[:, :, k], bench_names)


def nc_difficulty_timeline_df(idata, data: ECIData,
                              raw_df: pd.DataFrame) -> pd.DataFrame:
    """Benchmark-difficulty timeline for the NON-compensatory fit (same schema as
    `mirt_difficulty_timeline_df`, so it drops straight into `capability_timeline_fig`).

    NC has no single difficulty D — difficulty is per (benchmark, axis) as
    b = −c/a (a = 1 in the restricted fit). We collapse each benchmark to one
    scalar by averaging b over the axes it actually loads (Q = 1); the shared
    helper masks off-axis cells to NaN, so nanmean = mean over loaded axes."""
    post = idata.posterior
    Q = idata.constant_data["Q"].values                      # (B, K)
    bench = post["c"].coords["bench"].values.tolist()
    b_draw = np.nanmean(nc_difficulty_draws(post, Q), axis=2)   # (S, B)
    _, bench_dates = _release_dates(raw_df)
    rows = []
    for j, bm in enumerate(bench):
        if bm not in bench_dates.index:
            continue
        mean, lo, hi = post_stats(b_draw[:, j], hdi_prob=0.5)
        rows.append({"name": bm, "kind": "benchmark", "release_date": bench_dates[bm],
                     "mean": mean, "hdi_low": lo, "hdi_high": hi})
    out = pd.DataFrame(rows)
    out["release_date"] = pd.to_datetime(out["release_date"])
    return out


