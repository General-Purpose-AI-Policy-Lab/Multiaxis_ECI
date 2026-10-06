"""Which human tiers the data place on which axis.

A human tier is a test-taker scored on a handful of benchmarks, tied to the other tiers by the
ordering prior (models.mirt._human_block). Its posterior on an axis can be narrow without any
of its scores measuring that axis: a compensatory fit pins theta_k through the small k-loadings
of benchmarks that mostly measure other axes, by linear extrapolation. On the 2026-10-05 K=4
flagship no human score is on a cyber benchmark, and the tiers still sit 5 to 11 model sd below
every model on the cyber axis with posterior sd around 1, so neither the posterior sd nor the
model-based information flags them.

The test here asks how much the scores on benchmarks that MOSTLY measure axis k (axis share >=
`config.HUMAN_AXIS_SHARE_MIN`) tell about each tier on k, once carried along the tier order:

  * each human score contributes Fisher information w * a a^T on its tier, where `a` keeps only
    the loading components whose axis share passes the threshold and `w` is the Beta
    likelihood's information on the logit at the tier's fitted mean (a Beta-binomial score is
    read as Beta; the floor asymptote is ignored);
  * the ordering prior is approximated by a Gaussian: a root Normal(0, PRIOR_HUMAN_ROOT^2), one
    increment of variance (1 - 2/pi) PRIOR_DELTA_HUMAN^2 per tier (the HalfNormal's variance),
    and a multi-parent tier read as the mean of its paths rather than their max. Information then
    flows up and down the order, damped by the increments;
  * the result is the contraction 1 - var_post / var_prior per tier and axis.

The Gaussian reading understates the trace's posterior sd; it ranks the tiers, it is not an
interval to draw. Axis-agnostic inputs only (medians of D and phi_b, the order), so a view
re-ordered onto a reference frame keeps its answer.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.special import expit, polygamma

from multiaxis_eci.config import (
    HUMAN_AXIS_CONTRACTION_MIN,
    HUMAN_AXIS_SHARE_MIN,
    HUMAN_UNIT_ANCHORS,
    PRIOR_DELTA_HUMAN,
    PRIOR_HUMAN_ROOT,
)
from multiaxis_eci.data import ECIData


def trace_human_order(idata):
    """The tier order a fit was sampled with (posterior attr `mirt_human_order`), or None."""
    raw = idata.posterior.attrs.get("mirt_human_order")
    return json.loads(raw) if raw else None


def _beta_logit_information(mu: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Fisher information on eta = logit(mu) of one Beta(mu phi, (1 - mu) phi) score."""
    return (phi ** 2 * (mu * (1 - mu)) ** 2
            * (polygamma(1, mu * phi) + polygamma(1, (1 - mu) * phi)))


def _prior_covariance(R: np.ndarray, P: np.ndarray, groups: list) -> np.ndarray:
    """Gaussian stand-in for the ordering prior: (n_tiers, n_tiers), the same on every axis."""
    paths = (R @ R.T) * PRIOR_HUMAN_ROOT ** 2 + (P @ P.T) * (1 - 2 / np.pi) * PRIOR_DELTA_HUMAN ** 2
    M = np.zeros((len(groups), R.shape[0]))
    for t, g in enumerate(groups):
        M[t, g] = 1.0 / len(g)
    return M @ paths @ M.T


def human_axis_contraction(A_draws: np.ndarray, theta_draws: np.ndarray, data: ECIData,
                           human_order, D: np.ndarray | None = None,
                           phi: np.ndarray | None = None,
                           share_min: float = HUMAN_AXIS_SHARE_MIN) -> pd.DataFrame | None:
    """Per tier (rows, the tier names) and axis (columns 0..K-1), the prior-to-posterior
    contraction of theta_k that the tier order and the human scores on benchmarks mostly
    measuring k give. None when the fit has no loadings, no order or fewer than two tiers.

    `D` and `phi` are per-benchmark medians; without them every score weighs as at mu = 1/2
    with unit precision (the ranking of tiers, not the threshold, is then what reads)."""
    from multiaxis_eci.models.mirt import _human_structure

    if A_draws is None or not human_order or not data.is_human.any():
        return None
    names = data.mlookup.sort_values("model_idx")["model"].tolist()
    struct = _human_structure(human_order, names)
    if struct is None:
        return None
    rows, R, P, _, groups = struct
    A = np.median(A_draws, axis=0)                                   # (B, K)
    K = A.shape[1]
    share = A ** 2 / np.maximum((A ** 2).sum(axis=1, keepdims=True), 1e-12)
    A_kept = np.where(share >= share_min, A, 0.0)
    theta = np.median(theta_draws, axis=0)                           # (M, K)

    tier_of_model = {int(m): t for t, m in enumerate(rows)}
    info = np.zeros((len(rows), K, K))
    for m, b in zip(data.model_idx, data.bench_idx):
        t = tier_of_model.get(int(m))
        if t is None:
            continue
        if D is None or phi is None:
            w = 1.0
        else:
            mu = np.clip(expit(A[b] @ theta[m] - D[b]), 1e-6, 1 - 1e-6)
            w = float(_beta_logit_information(mu, phi[b]))
        info[t] += w * np.outer(A_kept[b], A_kept[b])

    S0 = _prior_covariance(R, P, groups)
    n = len(rows)
    prec = np.kron(np.linalg.inv(S0), np.eye(K))
    for t in range(n):
        prec[t * K:(t + 1) * K, t * K:(t + 1) * K] += info[t]
    var_post = np.diag(np.linalg.inv(prec)).reshape(n, K)
    contraction = 1.0 - var_post / np.diag(S0)[:, None]
    return pd.DataFrame(contraction, index=[names[m] for m in rows], columns=range(K))


def view_human_contraction(view, data: ECIData, theta: np.ndarray | None = None,
                           A: np.ndarray | None = None) -> pd.DataFrame | None:
    """`human_axis_contraction` on a FitView (its own frame unless `theta` / `A` are given)."""
    return human_axis_contraction(view.A if A is None else A,
                                  view.theta if theta is None else theta, data,
                                  view.human_order, D=view.D_med, phi=view.phi_med)


def identified_tiers(contraction: pd.DataFrame | None, k: int,
                     min_contraction: float = HUMAN_AXIS_CONTRACTION_MIN) -> set | None:
    """The tiers drawn on axis k; None (draw every tier) when no contraction was computed."""
    if contraction is None:
        return None
    return set(contraction.index[contraction[k] >= min_contraction])


def drop_unidentified(df: pd.DataFrame, k: int, contraction: pd.DataFrame | None,
                      col: str = "name") -> pd.DataFrame:
    """`df` without the human tiers not placed on axis k (rows of other names are kept)."""
    keep = identified_tiers(contraction, k)
    if keep is None or df is None or len(df) == 0 or col not in df:
        return df
    hidden = set(contraction.index) - keep
    return df[~df[col].isin(hidden)].reset_index(drop=True)


def human_scale_axes(contraction: pd.DataFrame | None, K: int) -> np.ndarray:
    """(K,) bool: the axes on which both HUMAN_UNIT_ANCHORS tiers are drawn, so the human-unit
    scale holds there. Every axis when no contraction was computed."""
    if contraction is None:
        return np.ones(K, dtype=bool)
    return np.array([all(t in (identified_tiers(contraction, k) or ())
                         for t in HUMAN_UNIT_ANCHORS) for k in range(K)])
