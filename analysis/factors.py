"""Reading factors off a fitted trace: anchor/prior introspection,
canonicalized loadings and abilities, loading and score tables."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from analysis.rotation import canonicalize_factors

def trace_anchors(trace) -> dict:
    """Anchor map stored on a MIRT trace, or {} if the fit was unanchored.

    fit.py writes the anchor dict (benchmark -> axis index) to
    `trace.posterior.attrs["mirt_anchors"]` as JSON at save time (Option C:
    the metadata travels with the trace, so post-processing knows whether the
    axes are pre-identified without the caller having to remember a flag)."""
    raw = trace.posterior.attrs.get("mirt_anchors", "") if hasattr(trace, "posterior") else ""
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


def trace_axis_names(trace, K: int) -> list[str]:
    """Human-readable per-axis names stored on the trace, else ['axis1'..'axisK'].

    Confirmatory fits (e.g. the qmatrix3 skill Q-matrix) write the axis
    identities to `trace.posterior.attrs["mirt_axis_names"]` (JSON list) at save
    time, so plots can label axes by skill instead of by index."""
    raw = trace.posterior.attrs.get("mirt_axis_names", "") if hasattr(trace, "posterior") else ""
    try:
        names = json.loads(raw) if raw else []
    except (json.JSONDecodeError, TypeError):
        names = []
    if len(names) == K:
        return list(names)
    return [f"axis{k + 1}" for k in range(K)]


def trace_loading_prior(trace) -> str:
    """Loading prior recorded on a MIRT trace ('' if absent).

    Fit drivers write `posterior.attrs["mirt_loading_prior"]` at save time so
    post-processing (prepare_fit) can pick the right rotation/sign handling
    without the caller passing flags — same pattern as `trace_anchors`."""
    if not hasattr(trace, "posterior"):
        return ""
    return str(trace.posterior.attrs.get("mirt_loading_prior", ""))


def mirt_factors_from_trace(trace, rank_track: bool | None = None):
    """Pull (A, theta, tau_A) from a MIRT trace and canonicalize them.

    Returns (A, theta, tau). Flattens chain × draw. `rank_track` defaults to
    None = auto: rank-track unless the trace carries an anchor map (anchored
    fits have pre-identified axes — see `trace_anchors` / `canonicalize_factors`),
    is a signed-free fit (align_rotations owns the permutation there; the
    energy ranking would scramble axes before alignment sees them), or is a
    bifactor fit (axis1 is the general column BY CONSTRUCTION — ranking by
    energy could demote it, and the specifics are near-tied by design, so
    per-draw ranking would shuffle them).
    """
    if rank_track is None:
        rank_track = (not trace_anchors(trace)
                      and trace_loading_prior(trace)
                      not in ("signed", "bifactor"))
    A = trace.posterior["A"].values
    theta = trace.posterior["theta"].values
    tau = trace.posterior["tau_A"].values
    S = A.shape[0] * A.shape[1]
    A = A.reshape(S, A.shape[2], A.shape[3])
    theta = theta.reshape(S, theta.shape[2], theta.shape[3])
    tau = tau.reshape(S, tau.shape[2])
    return canonicalize_factors(A, theta, tau, rank_track=rank_track)


def tau_spectrum_df(tau_canon: np.ndarray) -> pd.DataFrame:
    """Per-axis tau_A summary (rank-sorted) + consecutive gap ratios.

    `tau_canon` is the (S, K) tau returned by canonicalize_factors (already
    sorted descending per draw). The gap ratio axis_k / axis_{k+1} flags where
    the live axes give way to the prior-shrunk tail.
    """
    med = np.median(tau_canon, axis=0)
    lo, hi = np.percentile(tau_canon, [3, 97], axis=0)
    K = tau_canon.shape[1]
    rows = []
    for k in range(K):
        ratio = med[k] / med[k + 1] if k < K - 1 else np.nan
        rows.append({
            "axis":         f"axis{k + 1}",
            "tau_median":   float(med[k]),
            "tau_hdi_low":  float(lo[k]),
            "tau_hdi_high": float(hi[k]),
            "gap_ratio_to_next": float(ratio) if np.isfinite(ratio) else np.nan,
        })
    return pd.DataFrame(rows)


def loadings_table(A_canon: np.ndarray,
                   bench_names: list[str],
                   bench_category=None,
                   hdi=(3, 97)) -> pd.DataFrame:
    """Long-format loadings table: one row per (axis, benchmark).

    Columns: axis, benchmark, category, loading_median, hdi_low, hdi_high,
    bench_norm, axis_share. Sorted by axis then descending loading. `A_canon`
    is the (S, B, K) rotated loading draws — median/HDI are taken per entry,
    so the rotated loadings carry uncertainty (the rotation transform is
    fixed, applied to every draw).

    A loading conflates two things: how steep the benchmark is overall
    (bench_norm, the length of its median loading row — MDISC) and how much of
    that steepness points along this axis (axis_share, the entry's fraction of
    the squared row norm). A long row that is half-aligned can out-rank a
    short row that points squarely along the axis, so read axis_share next to
    loading_median before naming an axis by its top loaders.
    """
    med = np.median(A_canon, axis=0)
    lo, hi = np.percentile(A_canon, hdi, axis=0)
    norm = np.sqrt((med ** 2).sum(axis=1))
    share = med ** 2 / np.maximum((med ** 2).sum(axis=1, keepdims=True), 1e-12)
    B, K = med.shape
    rows = []
    for k in range(K):
        for b in range(B):
            rows.append({
                "axis":           f"axis{k + 1}",
                "benchmark":      bench_names[b],
                "category":       None if bench_category is None else str(bench_category[b]),
                "loading_median": float(med[b, k]),
                "hdi_low":        float(lo[b, k]),
                "hdi_high":       float(hi[b, k]),
                "bench_norm":     float(norm[b]),
                "axis_share":     float(share[b, k]),
            })
    df = pd.DataFrame(rows)
    return df.sort_values(["axis", "loading_median"],
                          ascending=[True, False]).reset_index(drop=True)


def factor_scores_df(theta_canon: np.ndarray,
                     model_names: list[str],
                     is_human=None) -> pd.DataFrame:
    """Per-model posterior-mean ability on each canonicalized axis (wide)."""
    mean = theta_canon.mean(axis=0)            # (M, K)
    K = mean.shape[1]
    out = {"model": list(model_names)}
    for k in range(K):
        out[f"theta{k + 1}_mean"] = mean[:, k]
    df = pd.DataFrame(out)
    if is_human is not None:
        df["is_human"] = np.asarray(is_human)
    return df


