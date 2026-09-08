"""Posterior predictive checks and goodness-of-fit metrics.

PIT excludes the boundary-clipped rows on BOTH sides (exact zeros and scores
at/above 1 - ECI_EPS): the model never saw those observed values, and an
exact-1 row would land at PIT = 1 by construction (every Beta draw is < 1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import expit
from scipy.stats import kstest

from multiaxis_eci.config import ECI_EPS, PIT_TIE_SEED, PPC_SEED
from multiaxis_eci.data import ECIData


@dataclass
class GoFResults:
    y_rep_flat: np.ndarray     # (n_samples, n_obs) posterior predictive draws
    y_pred_mean: np.ndarray    # (n_obs,) posterior predictive mean per observation
    pit: np.ndarray            # (n_non_boundary,) PIT values, boundary rows excluded
    metrics: dict


def _flatten_over_chains(arr: np.ndarray) -> np.ndarray:
    return arr.reshape(-1, *arr.shape[2:])


def _thin_sel(n_samples: int, max_draws: int):
    """Even subsample index of size max_draws when n_samples exceeds it, else
    None — the (n_samples, n_obs) broadcast is the PPC memory bottleneck."""
    if n_samples > max_draws:
        return np.linspace(0, n_samples - 1, max_draws).astype(int)
    return None


def _pull_flat(post, names: list[str], max_draws: int) -> list[np.ndarray]:
    """Flatten (chain, draw, ...) → (S, ...) for each named posterior variable
    and thin them all with one shared even subsample — the common head of
    every posterior predictive below."""
    arrs = [_flatten_over_chains(post[n].values) for n in names]
    sel = _thin_sel(arrs[0].shape[0], max_draws)
    if sel is not None:
        arrs = [a[sel] for a in arrs]
    return arrs


def _beta_draw(mu: np.ndarray, phi_flat: np.ndarray, data: ECIData,
               seed: int, n_eff: np.ndarray | None = None) -> np.ndarray:
    """Draw Beta(mu*phi, (1-mu)*phi) per observation — the shared tail of every
    MIRT posterior predictive. mu: (S, n_obs); phi_flat: (S, n_benchmarks).

    n_eff: per-observation effective test length (data.n_eff) for a fit built
    with the known_se noise split. phi+1 is a test length, so the known
    instrument part and the estimated excess add as relative variances:
    1/(1+phi_n) = 1/n_eff + 4*sigma_b^2, and 4*sigma_b^2 == 1/(1+phi_b). A cell
    with n_eff = inf keeps phi_b.
    """
    phi_obs = phi_flat[:, data.bench_idx]
    if n_eff is not None:
        phi_obs = 1.0 / (1.0 / np.asarray(n_eff, dtype=np.float64)
                         + 1.0 / (1.0 + phi_obs)) - 1.0
    rng = np.random.default_rng(seed)
    return rng.beta(mu * phi_obs, (1.0 - mu) * phi_obs)


def posterior_predictive_mirt(trace, data: ECIData,
                              seed: int = PPC_SEED,
                              max_draws: int = 2000,
                              floor_c: np.ndarray | None = None,
                              n_eff: np.ndarray | None = None,
                              return_mean: bool = False) -> np.ndarray:
    """MIRT posterior predictive: mu = sigmoid(sum_k A theta - D), draw Beta.

    Posterior predictive for the k-factor compensatory model (models/mirt.py). Thins to `max_draws` posterior samples — the (n_samples,
    n_obs) broadcast is the memory bottleneck — and accumulates eta over the K
    axes to avoid materializing an (n_samples, n_obs, K) array.

    floor_c: per-benchmark chance floor (blookup order). When given, applies the
    same fixed-c 3PL link as the model, mu = c_b + (1 - c_b) * sigmoid(eta), so
    GoF/PIT match a floored fit. Omit for a plain 2PL trace.

    A ceiling_noise trace carries its estimated `ceiling_d` in the posterior
    and is detected automatically; there is no fixed-ceiling argument.

    n_eff: per-observation effective test length (data.n_eff) for a trace fit
    with the known_se noise split — the predictive Beta then uses the same
    per-cell precision the model did, so GoF and PIT score that likelihood.
    Omit for a trace fit without the split.

    return_mean: return the per-observation fitted means mu = E[y|theta] per
    draw (shape S x n_obs) instead of Beta draws — the noise-free fitted values
    the Gelman Bayesian R² needs.
    """
    post = trace.posterior
    soft_d = "ceiling_d" in post
    ceil_names = ["ceiling_d"] if soft_d else []
    if "alpha" in post:
        A, tp, alpha, phi, *rest = _pull_flat(
            post, ["A", "theta_pos", "alpha", "phi_b"] + ceil_names, max_draws)
        K = A.shape[-1]

        s = np.zeros_like(A[:, data.bench_idx, 0])
        for k in range(K):
            s += A[:, data.bench_idx, k] * tp[:, data.model_idx, k]
        # loglog link: difficulty lives in A's row scale, so no D subtraction;
        # must match models/mirt.py bit for bit or GoF/PIT score a different
        # likelihood than the fit.
        eta = alpha[:, data.bench_idx] * np.log(s)
    else:
        # theta_pos fit: the likelihood reads softplus(theta), not theta
        # (models/mirt.py). Raw theta stays the reported ability, but scoring the
        # predictive with it would evaluate a different link than the fit.
        th_name = "theta_pos" if "theta_pos" in post else "theta"
        names = ["A", th_name, "D", "phi_b"] + ceil_names
        A, theta, D, phi, *rest = _pull_flat(post, names, max_draws)
        K = A.shape[-1]

        eta = -D[:, data.bench_idx]                                  # (S, n_obs)
        for k in range(K):
            eta += A[:, data.bench_idx, k] * theta[:, data.model_idx, k]
    mu = expit(eta)
    if soft_d:
        # ceiling_noise trace: apply the same mu = c + (d - c) * sigmoid link
        # the model used, with the per-draw estimated ceiling d.
        d = rest[0]                                              # (S, B)
        c_obs = np.asarray(floor_c)[data.bench_idx] if floor_c is not None else 0.0
        mu = c_obs + (d[:, data.bench_idx] - c_obs) * mu
    elif floor_c is not None:
        c_obs = np.asarray(floor_c)[data.bench_idx]
        mu = c_obs + (1.0 - c_obs) * mu
    if return_mean:
        return mu
    return _beta_draw(mu, phi, data, seed, n_eff=n_eff)


def posterior_predictive_mirt_nc(trace, data: ECIData,
                                 seed: int = PPC_SEED,
                                 max_draws: int = 2000,
                                 return_mean: bool = False) -> np.ndarray:
    """Non-compensatory MIRT posterior predictive.

    mu = prod_k sigmoid(a_k*theta[m,k] + c[b,k])^Q[b,k], then draw
    Beta(mu*phi, (1-mu)*phi). Mirrors `posterior_predictive_mirt` but with the
    conjunctive PRODUCT link, the per-(bench,axis) easiness intercept `c`
    (there is no single difficulty `D`), and the Q-matrix gate read from
    `constant_data["Q"]`. Reads `A` only if present (free-discrimination fit);
    otherwise the slope is fixed at 1 (restricted MLTM). Accumulates in log
    space and clips mu, exactly as the model does.
    """
    post = trace.posterior
    Q = trace.constant_data["Q"].values             # (B, K), 0/1
    has_A = "A" in post
    names = ["theta", "c", "phi_b"] + (["A"] if has_A else [])
    theta, c, phi, *rest = _pull_flat(post, names, max_draws)
    A = rest[0] if has_A else None
    K = theta.shape[-1]

    Qb = Q[data.bench_idx]                          # (n_obs, K)
    log_mu = np.zeros((theta.shape[0], data.n_obs))
    for k in range(K):
        z = c[:, data.bench_idx, k]                 # (S, n_obs)
        if has_A:
            z = z + A[:, data.bench_idx, k] * theta[:, data.model_idx, k]
        else:
            z = z + theta[:, data.model_idx, k]
        log_sig = -np.logaddexp(0.0, -z)            # log sigmoid(z), stable
        log_mu += Qb[:, k] * log_sig                # gate: unloaded axes drop out
    mu = np.clip(np.exp(log_mu), ECI_EPS, 1.0 - ECI_EPS)
    if return_mean:
        return mu
    return _beta_draw(mu, phi, data, seed)


def posterior_predictive_mirt_sparse(trace, data: ECIData,
                                     seed: int = PPC_SEED,
                                     max_draws: int = 2000,
                                     return_mean: bool = False) -> np.ndarray:
    """Sparse-gate non-compensatory MIRT posterior predictive.

    mu = prod_k sigmoid(theta[m,k] + c[b,k]) ^ g[b,k], then draw
    Beta(mu*phi, (1-mu)*phi). The gate g is read per draw from the posterior and
    the slope is 1. Accumulates in log space and clips mu, as the model does.
    """
    theta, c, g, phi = _pull_flat(trace.posterior,
                                  ["theta", "c", "g", "phi_b"], max_draws)
    K = theta.shape[-1]

    log_mu = np.zeros((theta.shape[0], data.n_obs))
    for k in range(K):
        z = theta[:, data.model_idx, k] + c[:, data.bench_idx, k]  # (S, n_obs)
        log_sig = -np.logaddexp(0.0, -z)                           # log sigmoid(z)
        log_mu += g[:, data.bench_idx, k] * log_sig                # continuous gate
    mu = np.clip(np.exp(log_mu), ECI_EPS, 1.0 - ECI_EPS)
    if return_mean:
        return mu
    return _beta_draw(mu, phi, data, seed)


def posterior_predictive_mirt_interaction(trace, data: ECIData,
                                          seed: int = PPC_SEED,
                                          max_draws: int = 2000,
                                          return_mean: bool = False,
                                          floor_c: np.ndarray | None = None) -> np.ndarray:
    """Semi-compensatory (interaction) MIRT posterior predictive
    (models/mirt_interaction.py).

    eta = sum_k A[b,k] theta[m,k] + sum_{j<k} gamma[b,jk] tp[m,j] tp[m,k] - D[b],
    mu = sigmoid(eta) clipped to [ECI_EPS, 1 - ECI_EPS] — the model clips its
    mean (models/mirt_interaction.py), so the predictive must use the same
    link or GoF/PIT score a different likelihood than the fit. The linear term
    reads raw `theta` and the product reads `tp` = `theta_pos` = softplus(theta),
    matching the two scales the model uses. Draw Beta; gamma is read per draw.
    floor_c: per-benchmark chance floor (blookup order) for a fit built with the
    fixed-c 3PL link — same floor transform before the clip, omit for a plain fit.
    """
    import itertools
    A, theta, tp, D, gamma, phi = _pull_flat(
        trace.posterior, ["A", "theta", "theta_pos", "D", "gamma", "phi_b"], max_draws)
    K = A.shape[-1]
    pairs = list(itertools.combinations(range(K), 2))

    eta = -D[:, data.bench_idx]                                  # (S, n_obs)
    for k in range(K):
        eta += A[:, data.bench_idx, k] * theta[:, data.model_idx, k]
    for p, (j, k) in enumerate(pairs):
        eta += (gamma[:, data.bench_idx, p]
                * tp[:, data.model_idx, j] * tp[:, data.model_idx, k])
    mu = expit(eta)
    if floor_c is not None:
        c_obs = np.asarray(floor_c)[data.bench_idx]
        mu = c_obs + (1.0 - c_obs) * mu
    mu = np.clip(mu, ECI_EPS, 1.0 - ECI_EPS)
    if return_mean:
        return mu
    return _beta_draw(mu, phi, data, seed)


def boundary_mask(data: ECIData) -> np.ndarray:
    """(n_obs,) bool — rows whose observed score was moved by the fit-time
    boundary clip: exact zeros (zero_score_mask) and scores at/above
    1 - ECI_EPS. PIT is degenerate there — for y = 1, P(Y_rep ≤ y) = 1 for
    every Beta draw — so these rows are excluded from PIT and its KS/variance
    summaries (persistence.save_pit uses the same mask to stay row-aligned)."""
    return data.zero_score_mask | (data.scores >= 1.0 - ECI_EPS)


def pit_values(y_rep_flat: np.ndarray,
                scores: np.ndarray,
                exclude_mask: np.ndarray,
                tie_seed: int = PIT_TIE_SEED) -> np.ndarray:
    """PIT u_n = P(Y_rep ≤ y_n), skipping the rows in `exclude_mask`
    (the boundary-clipped observations — see `boundary_mask`)."""
    nonzero = ~exclude_mask
    y_rep_u = y_rep_flat[:, nonzero]
    y_obs_u = scores[nonzero][None, :]

    rng = np.random.default_rng(tie_seed)
    tie_break = rng.uniform(size=y_rep_u.shape)
    return ((y_rep_u < y_obs_u) + tie_break * (y_rep_u == y_obs_u)).mean(axis=0)


def _bayesian_r2(mu_flat: np.ndarray | None, scores: np.ndarray,
                 resid: np.ndarray) -> float:
    """Gelman, Goodrich, Gabry & Vehtari (2019) Bayesian R².

    Per posterior draw s: the variance across observations of the fitted means
    mu_s = E[y | theta_s] over that plus the residual variance
    Var_i(y_i - mu_{s,i}); summarised by the posterior median (the per-draw R²
    is right-skewed). Built on the fitted means, not the noisy predictive draws,
    so it is the fraction of outcome variance the model explains and stays in
    [0, 1]. `mu_flat` is (S, n_obs).

    Falls back to the classical 1 - Var(resid)/Var(y) on the predictive mean
    when per-draw fitted means are not supplied; the two coincide only when
    fitted values and residuals are uncorrelated, which a shrunk posterior fit
    does not guarantee, so the fallback is an approximation.
    """
    if mu_flat is None:
        return float(1.0 - np.var(resid) / np.var(scores))
    var_fit = mu_flat.var(axis=1)
    var_res = (scores[None, :] - mu_flat).var(axis=1)
    return float(np.median(var_fit / (var_fit + var_res)))


def compute_gof(y_rep_flat: np.ndarray, data: ECIData,
                mu_flat: np.ndarray | None = None) -> GoFResults:
    y_pred_mean = y_rep_flat.mean(axis=0)
    pit = pit_values(y_rep_flat, data.scores, boundary_mask(data))
    resid = data.scores - y_pred_mean

    ks_stat, ks_p = kstest(pit, "uniform")

    # Posterior probability that the zero-score rows would be replicated at or
    # below the diagnostic threshold — calibration check for the boundary points.
    if data.zero_score_mask.any():
        zero_below_c = float(
            (y_rep_flat[:, data.zero_score_mask] <= data.zero_diag_threshold).mean()
        )
    else:
        zero_below_c = float("nan")

    metrics = {
        "n_obs":                       int(data.n_obs),
        "n_nonzero_score":             int((~data.zero_score_mask).sum()),
        "n_zero_score":                int(data.zero_score_mask.sum()),
        "n_one_score":                 int((data.scores >= 1.0 - ECI_EPS).sum()),
        "zero_diag_threshold":         float(data.zero_diag_threshold),
        "rmse":                        float(np.sqrt(np.mean(resid ** 2))),
        "mae":                         float(np.mean(np.abs(resid))),
        "bayesian_r2":                 _bayesian_r2(mu_flat, data.scores, resid),
        "ks_stat":                     float(ks_stat),
        "ks_p":                        float(ks_p),
        "pit_mean":                    float(pit.mean()),
        "pit_var":                     float(pit.var()),
        "zero_pred_below_threshold":   zero_below_c,
    }
    return GoFResults(
        y_rep_flat=y_rep_flat, y_pred_mean=y_pred_mean, pit=pit, metrics=metrics,
    )
