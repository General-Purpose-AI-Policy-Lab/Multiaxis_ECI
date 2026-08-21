"""ECI-scale statistics: capability draws, the affine anchor transform,
and the SOTA / all-models / timeline / human / forest summary tables."""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass

import config  # attribute access so runtime overrides (e.g., --eci-data-only) propagate
from data import ECIData, find_model_idx

@dataclass
class ECITransform:
    """Per-draw affine map from C to ECI: ECI = a + b * C."""
    a: np.ndarray   # (n_samples,)
    b: np.ndarray   # (n_samples,)

    def apply(self, C: np.ndarray) -> np.ndarray:
        return self.a + self.b * C


def post_stats(samples: np.ndarray, hdi_prob: float = 0.94):
    """Posterior median plus the central `hdi_prob` interval.

    The interval cuts equal tails, so the median always sits inside it. A
    posterior mean with an HDI can separate: on the bimodal trade-off ridges
    of the exploration fits the mean lands between the modes and falls outside
    its own 50% HDI, which draws a timeline dot away from its own bar.

    Downstream columns keep the names `mean` / `hdi_low` / `hdi_high` and carry
    the median and the two quantiles.
    """
    lo, hi = np.quantile(samples, [(1 - hdi_prob) / 2, (1 + hdi_prob) / 2])
    return float(np.median(samples)), float(lo), float(hi)


def capability_draws(trace) -> np.ndarray:
    """Overall-capability draws, flattened to (n_samples, n_models).

    Reads `C` from a 1D trace, or `theta[..., 0]` from a K=1 MIRT trace —
    the two parameterizations of the same scalar capability. Raises on a
    multi-axis trace: there is no single "overall" axis to report there.
    """
    post = trace.posterior
    if "C" in post:
        C = post["C"].values
        return C.reshape(-1, C.shape[-1])
    theta = post["theta"].values                    # (chain, draw, model, K)
    if theta.shape[-1] != 1:
        raise ValueError(
            f"capability_draws needs a 1D or K=1 trace; this trace has "
            f"K={theta.shape[-1]} axes.")
    return theta[..., 0].reshape(-1, theta.shape[-2])


def flat_C(trace) -> np.ndarray:
    """Flatten posterior capability across chain × draw → (n_samples, n_models)."""
    return capability_draws(trace)


def eci_transform(C_flat: np.ndarray, data: ECIData) -> ECITransform:
    """Per-draw ECI affine from the two-anchor convention.

    When config.RAW_C_MODE is True, returns the identity transform (a=0, b=1
    per draw) so ECI = C in all downstream output.
    """
    if config.RAW_C_MODE:
        n = C_flat.shape[0]
        return ECITransform(a=np.zeros(n), b=np.ones(n))
    low_idx  = find_model_idx(data.mlookup, config.ANCHOR_LOW[0])
    high_idx = find_model_idx(data.mlookup, config.ANCHOR_HIGH[0])
    span = config.ANCHOR_HIGH[1] - config.ANCHOR_LOW[1]
    b = span / (C_flat[:, high_idx] - C_flat[:, low_idx])
    a = config.ANCHOR_LOW[1] - b * C_flat[:, low_idx]
    return ECITransform(a=a, b=b)


def sota_stats_df(trace, data: ECIData, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Per-SOTA-model posterior mean + 94% HDI for C and ECI.

    Release dates come from the dataset (earliest per model), with
    `config.RELEASE_DATES` filling dateless previews. Models listed in
    `config.SOTA_MODELS` that have no rows in the filtered dataset (e.g. all
    their benchmarks fell into a dropped category) are skipped with a warning
    rather than crashing the pipeline.
    """
    C_flat = flat_C(trace)
    transform = eci_transform(C_flat, data)
    available = set(data.mlookup["model"].values)
    model_dates, _ = _release_dates(raw_df)

    rows = []
    for name in config.SOTA_MODELS:
        if name not in available:
            print(f"  [sota_stats_df] skipping {name!r} — no surviving rows in the filtered dataset")
            continue
        idx = find_model_idx(data.mlookup, name)
        c_samples   = C_flat[:, idx]
        eci_samples = transform.apply(c_samples)
        c_mean, c_lo, c_hi = post_stats(c_samples)
        e_mean, e_lo, e_hi = post_stats(eci_samples)
        rows.append({
            "model":        name,
            "release_date": model_dates.get(name),
            "C_mean":       c_mean,
            "C_hdi_low":    c_lo,
            "C_hdi_high":   c_hi,
            "ECI_mean":     e_mean,
            "ECI_hdi_low":  e_lo,
            "ECI_hdi_high": e_hi,
        })
    return pd.DataFrame(rows)


def all_models_stats_df(trace, data: ECIData,
                          metric: str = "C") -> pd.DataFrame:
    """Posterior median + central 94% interval for every model.

    metric='C'   → raw capability (the default — interpretable as logit units
                   relative to whatever anchor the trace was anchored at).
    metric='ECI' → applies the affine anchor transform to C per draw before
                   summarizing. Use only when comparability with the public
                   index is needed.

    Sorted ascending by mean so the tall plot reads bottom-up.
    """
    C_flat = flat_C(trace)
    if metric == "ECI":
        transform = eci_transform(C_flat, data)
        samples = transform.a[:, None] + transform.b[:, None] * C_flat
    elif metric == "C":
        samples = C_flat
    else:
        raise ValueError(f"metric must be 'C' or 'ECI', got {metric!r}")

    # Same summary as post_stats, vectorized over models: median and the
    # equal-tailed 94% interval, down the draw axis.
    means = np.median(samples, axis=0)
    lo, hi = np.quantile(samples, [0.03, 0.97], axis=0)

    df = pd.DataFrame({
        "name":      data.mlookup["model"].values,
        "mean":      means,
        "hdi_low":   lo,
        "hdi_high":  hi,
        "n_obs":     data.n_obs_per_model,
        "is_low_obs": data.is_low_obs,
    }).sort_values("mean").reset_index(drop=True)
    return df


def timeline_stats_df(trace, data: ECIData, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Per-entity posterior summary with release_date attached.

    Returns one DataFrame with rows for both models (kind='model', value=C)
    and benchmarks (kind='benchmark', value=D). Used by the capability
    timeline plot. release_date for a benchmark is the earliest release_date
    among models evaluated against it.
    """
    C = capability_draws(trace)
    D = trace.posterior["D"].values.reshape(-1, data.n_benchmarks)

    # Same date source as sota_stats_df / the MIRT timelines: dataset dates
    # with config.RELEASE_DATES filling dateless previews, so a SOTA release
    # that appears in sota.csv is never silently absent from the timeline.
    model_dates, bench_dates = _release_dates(raw_df)

    rows = []
    for i, m in enumerate(data.mlookup["model"].values):
        if m not in model_dates.index:
            continue
        mean, lo, hi = post_stats(C[:, i])
        rows.append({"name": m, "kind": "model",
                     "release_date": model_dates[m],
                     "mean": mean, "hdi_low": lo, "hdi_high": hi,
                     "n_obs": int(data.n_obs_per_model[i]),
                     "is_low_obs": bool(data.is_low_obs[i])})
    for j, b in enumerate(data.blookup["benchmark"].values):
        if b not in bench_dates.index:
            continue
        mean, lo, hi = post_stats(D[:, j])
        rows.append({"name": b, "kind": "benchmark",
                     "release_date": bench_dates[b],
                     "mean": mean, "hdi_low": lo, "hdi_high": hi,
                     "n_obs": int((data.bench_idx == j).sum()),
                     "is_low_obs": False})
    out = pd.DataFrame(rows)
    out["release_date"] = pd.to_datetime(out["release_date"])
    return out


def human_stats_df(trace, data: ECIData) -> pd.DataFrame:
    """Posterior mean + 94% HDI of C for each fitted human group.

    Humans are now full test-takers in the IRT (added to the dataset as
    rows in load_eci_data). This returns one row per group: name, mean,
    hdi_low, hdi_high, n_obs.
    """
    if not data.is_human.any():
        return pd.DataFrame(columns=["name", "mean", "hdi_low", "hdi_high", "n_obs"])
    C_flat = flat_C(trace)
    rows = []
    for i, m in enumerate(data.mlookup["model"].values):
        if not data.is_human[i]:
            continue
        mean, lo, hi = post_stats(C_flat[:, i])
        rows.append({"name": m, "mean": mean, "hdi_low": lo, "hdi_high": hi,
                     "n_obs": int(data.n_obs_per_model[i])})
    return pd.DataFrame(rows).sort_values("mean").reset_index(drop=True)


def forest_stats_from_draws(draws: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Posterior mean + 94% HDI per entry of pre-flattened (n_samples, n) draws, sorted."""
    rows = []
    for i, n in enumerate(names):
        mean, lo, hi = post_stats(draws[:, i])
        rows.append({"name": n, "mean": mean, "hdi_low": lo, "hdi_high": hi})
    return pd.DataFrame(rows).sort_values("mean").reset_index(drop=True)


def forest_stats_df(trace, var_name: str, names: list[str]) -> pd.DataFrame:
    """Posterior mean + 94% HDI for every entry of a vector-valued variable, sorted."""
    post = trace.posterior[var_name].values
    return forest_stats_from_draws(post.reshape(-1, post.shape[-1]), names)


def _release_dates(raw_df: pd.DataFrame):
    """Earliest release_date per model/benchmark; fills gaps from config.RELEASE_DATES."""
    valid = raw_df.dropna(subset=["release_date"])
    model_dates = valid.groupby("model_version")["release_date"].min()
    bench_dates = valid.groupby("benchmark")["release_date"].min()
    curated = {m: d for m, d in config.RELEASE_DATES.items() if m not in model_dates.index}
    if curated:
        model_dates = pd.concat([model_dates, pd.Series(curated)])
    return model_dates, bench_dates
