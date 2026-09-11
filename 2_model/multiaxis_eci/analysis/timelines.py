"""Release-date timeline frames for abilities and difficulties."""
from __future__ import annotations

import numpy as np
import pandas as pd

from multiaxis_eci.analysis.convergence import nc_difficulty_draws
from multiaxis_eci.analysis.stats import _release_dates, forest_stats_from_draws, post_stats
from multiaxis_eci.config import INFORMED_SD_CAP, MIN_AXIS_COVERAGE
from multiaxis_eci.data import ECIData


def mirt_informed_mask(theta_canon: np.ndarray, sd_cap: float = INFORMED_SD_CAP) -> np.ndarray:
    """(M, K) bool: True where a model's per-axis posterior SD < sd_cap.

    Filters out models whose ability is extrapolated rather than measured
    (sparse old models can land at misleadingly high ability with wide CIs).

    Default 0.33 (history 0.6 → 0.3 → 0.4 → 0.33): 0.6 corresponds to a 95%
    CI width of ~2.35 — far too wide on this scale, it admitted ~10 pre-2023
    models at frontier levels. 0.3 removed them but froze thin axes' record
    chains (axis 4: 2 post-Oct-2024 records, the forecast line detached from
    the data). 0.4 (95% CI width ~1.57) kept the ghosts out while giving every
    axis a usable record set (2026-08-13); the envelope basis then let the cap
    tighten to 0.33, the measured-model definition shared by every graph and
    by `config.FORECAST_KW`.
    """
    return theta_canon.std(axis=0) < sd_cap


def axis_coverage(A_draws: np.ndarray, k: int, data: ECIData) -> np.ndarray:
    """(M,) how much a test-taker's own evaluations say about axis k: the sum of the axis-k
    shares (median loadings) of every benchmark it was scored on.

    A benchmark's share is the fraction of its squared loading row pointing along k, so a
    release scored on one pure axis-k benchmark has coverage 1, and one scored only on
    benchmarks of other axes has coverage near 0 whatever its number of observations. The
    count is per test-taker, not per family (user decision 2026-09-11): a reasoning effort
    that was never run on the axis does not inherit its siblings' evidence.
    """
    med = np.median(A_draws, axis=0)                                    # (B, K)
    share = med ** 2 / np.maximum((med ** 2).sum(axis=1, keepdims=True), 1e-12)
    obs = np.zeros((data.n_models, data.n_benchmarks), dtype=bool)
    obs[np.asarray(data.model_idx), np.asarray(data.bench_idx)] = True
    return obs @ share[:, k]


def candidate_mask(theta_canon: np.ndarray, k: int, data: ECIData, model_dates: pd.Series, *,
                   A_draws: np.ndarray | None = None, min_coverage: float = MIN_AXIS_COVERAGE,
                   sd_cap: float | None = None, drop_low_obs: bool = False) -> np.ndarray:
    """(M,) bool: the test-takers a timeline, a forecast or a frontier may draw on for axis k.

    The one guard shared by the measured timelines, the forecast candidates, the frontier gap
    and the post's forests: dated, not a human tier, and evaluated on the axis, i.e. its own
    `axis_coverage` (the axis shares of the benchmarks it was scored on) reaches
    `min_coverage` (config.MIN_AXIS_COVERAGE). Without `A_draws` (a K=1 fit) every scored
    release passes. That is the whole rule (user decision 2026-09-11): no posterior-SD cap, no
    low-observation flag, no SOTA exemption; the thinly-measured releases it admits carry
    their wide intervals on the figures and weigh little in the trend fits. `sd_cap` (drop
    when the axis posterior SD reaches it) and `drop_low_obs` (drop `is_low_obs` models)
    remain as optional tightenings for the K=1 tools that still want them.
    """
    names = data.mlookup.sort_values("model_idx")["model"].tolist()
    theta_k = theta_canon[..., k] if theta_canon.ndim == 3 else theta_canon
    keep = np.array([m in model_dates.index for m in names], dtype=bool)
    keep &= ~np.asarray(data.is_human, dtype=bool)
    if A_draws is not None:
        keep &= axis_coverage(A_draws, k, data) >= min_coverage
    if sd_cap is not None:
        keep &= theta_k.std(axis=0) < sd_cap
    if drop_low_obs and data.is_low_obs is not None:
        keep &= ~np.asarray(data.is_low_obs, dtype=bool)
    return keep


def mirt_model_timeline_df(theta_canon: np.ndarray, k: int,
                           data: ECIData, raw_df: pd.DataFrame,
                           sd_cap: float | None = None,
                           drop_low_obs: bool = False,
                           A_draws: np.ndarray | None = None,
                           hdi_prob: float = 0.5,
                           min_coverage: float = MIN_AXIS_COVERAGE) -> pd.DataFrame:
    """Model-ability timeline for axis k (kind='model'); humans excluded
    (they have no release date and are drawn as reference bands instead).

    The rows are `candidate_mask`'s: dated, not human, and with `A_draws` (the loadings)
    evaluated on the axis (own coverage of at least `min_coverage`); `min_coverage=0.0`
    gives the all-models view. `sd_cap` and `drop_low_obs` are the optional tightenings
    the K=1 tools pass; the K-axis figures leave them off."""
    model_dates, _ = _release_dates(raw_df)
    names = data.mlookup.sort_values("model_idx")["model"].tolist()
    keep = candidate_mask(theta_canon, k, data, model_dates, A_draws=A_draws,
                          min_coverage=min_coverage, sd_cap=sd_cap, drop_low_obs=drop_low_obs)
    rows = []
    for i, m in enumerate(names):
        if not keep[i]:
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


