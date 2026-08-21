"""Non-compensatory (conjunctive) multidimensional 2PL Beta-IRT.

mu = prod_k sigmoid(a_k*theta[m,k] + c[b,k])^Q[b,k] — weakness on one axis
can't be bought back by strength on another. Same Beta likelihood as models/mirt.py.
Slope fixed at 1 by default (restricted MLTM); free_discrimination=True frees it.
Q-matrix gates axes via exponent (Q=0 → factor=1), not via zero loading.
"""
from __future__ import annotations

import warnings

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from config import ECI_EPS, NC_C_OFFSET, PRIOR_SIGMA_B, PRIOR_TAU_C
from data import ECIData


def _validate_qmatrix(qmatrix: np.ndarray, n_benchmarks: int) -> np.ndarray:
    """Check the Q-matrix and return it as a float (n_benchmarks, K) array."""
    Q = np.asarray(qmatrix)
    if Q.ndim != 2 or Q.shape[0] != n_benchmarks:
        raise ValueError(
            f"qmatrix must be (n_benchmarks={n_benchmarks}, K); got {Q.shape}")
    if not np.isin(Q, (0, 1)).all():
        raise ValueError("qmatrix entries must all be 0 or 1")
    row_sums = Q.sum(axis=1)
    if (row_sums < 1).any():
        bad = np.where(row_sums < 1)[0].tolist()
        raise ValueError(
            f"every benchmark must load >=1 axis; rows with none: {bad} "
            "(a zero-row gives mu = exp(0) = 1 for all models — degenerate)")
    col_sums = Q.sum(axis=0)
    if (col_sums < 1).any():
        dead = np.where(col_sums < 1)[0].tolist()
        raise ValueError(f"every axis must be loaded by >=1 benchmark; dead axes: {dead}")
    if (row_sums == 1).all():
        # Strict simple structure is fine for the compensatory SUM link (it's the
        # confirmatory qmatrix3), but DEGENERATE for this product link: with one
        # factor per benchmark, mu = sigmoid(theta+c) — a single term — so the K
        # axes share no data and decouple into K independent 1D IRTs (no
        # conjunction binds; cross-axis factor correlations are pure noise). Warn
        # rather than raise: it is a valid model, just not a conjunctive one.
        warnings.warn(
            "non-compensatory Q-matrix is strict simple structure (every benchmark "
            "loads exactly one axis) — the product link degenerates to K independent "
            "1D IRTs (no conjunction). Use a multi-loaded Q (e.g. fits/fit_nc.py "
            "--qvariant full / qmatrix3x) to actually test conjunction.",
            stacklevel=2)
    return Q.astype(float)


def build_mirt_nc_model(data: ECIData, qmatrix: np.ndarray,
                        free_discrimination: bool = False) -> pm.Model:
    """Build the K-factor NON-COMPENSATORY (conjunctive) Beta-MIRT model.

    Parameters
    ----------
    data : ECIData
        The fitted dataset (uses scores, model_idx, bench_idx, lookups).
    qmatrix : np.ndarray, shape (n_benchmarks, K)
        0/1 loading map (built from benchmark categories by the driver). K is
        inferred from its second dimension. Each benchmark must load >=1 axis
        and each axis must be loaded by >=1 benchmark.
    free_discrimination : bool, default False
        False -> restricted MLTM, slope a_k = 1 (phase 1, clean identification).
        True  -> free per-axis slope a_k with a tight prior near 1 (phase 2).

    Trace variables of interest: `theta` (model, latent), `c` (bench, latent)
    easiness intercept, `sigma_b`, `phi_b`, plus `A` and `tau_A` only when
    `free_discrimination=True`. The Q-matrix is saved in `constant_data["Q"]`.
    """
    Q = _validate_qmatrix(qmatrix, data.n_benchmarks)
    K = Q.shape[1]

    coords = {
        "model":  data.mlookup["model"].tolist(),
        "bench":  data.blookup["benchmark"].tolist(),
        "latent": [f"axis{k + 1}" for k in range(K)],
    }
    with pm.Model(coords=coords) as model:
        Q_data = pm.Data("Q", Q, dims=("bench", "latent"))
        sigma_b = pm.LogNormal("sigma_b", dims="bench", **PRIOR_SIGMA_B)
        tau_c = pm.LogNormal("tau_c", **PRIOR_TAU_C)

        # ZeroSumNormal constrains trailing axis → transpose to (model, latent).
        theta_t = pm.ZeroSumNormal("theta_t", sigma=1.0, dims=("latent", "model"))
        theta = pm.Deterministic("theta", theta_t.T, dims=("model", "latent"))

        c_z = pm.Normal("c_z", 0.0, 1.0, dims=("bench", "latent"))
        c = pm.Deterministic("c", NC_C_OFFSET + c_z * tau_c, dims=("bench", "latent"))

        if free_discrimination:
            A_z   = pm.HalfNormal("A_z", sigma=1.0, dims=("bench", "latent"))
            tau_A = pm.LogNormal("tau_A", mu=0.0, sigma=0.3, dims="latent")
            A = pm.Deterministic("A", A_z * Q_data * tau_A, dims=("bench", "latent"))
            slope_obs = A[data.bench_idx]            # (n_obs, K)
        else:
            slope_obs = 1.0

        theta_obs = theta[data.model_idx]            # (n_obs, K)
        c_obs     = c[data.bench_idx]                # (n_obs, K)
        Q_obs     = Q_data[data.bench_idx]           # (n_obs, K)

        z = slope_obs * theta_obs + c_obs            # (n_obs, K)
        log_terms = -pt.softplus(-z)                 # log sigmoid(z), stable
        log_mu = (Q_obs * log_terms).sum(axis=-1)    # (n_obs,)
        mu_n = pt.clip(pt.exp(log_mu), ECI_EPS, 1.0 - ECI_EPS)

        phi   = pm.Deterministic("phi_b", 1.0 / (4.0 * sigma_b**2) - 1.0, dims="bench")
        phi_n = phi[data.bench_idx]

        a = mu_n * phi_n
        b = (1.0 - mu_n) * phi_n

        clipped = np.clip(data.scores, ECI_EPS, 1.0 - ECI_EPS)
        pm.Beta("obs", alpha=a, beta=b, observed=clipped)

    return model
