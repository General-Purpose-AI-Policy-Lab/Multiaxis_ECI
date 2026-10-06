"""The Silva-Zanella mixture estimator recovers exact leave-one-out on a model where it is known."""
from __future__ import annotations

import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm

from multiaxis_eci.scripts import load

mixture_loo = load("4_diagnostics/robust_loo.py").mixture_loo


def test_mixture_loo_matches_exact_loo_on_a_grid():
    """theta on a fine grid, y_i ~ N(theta, 1), one outlier: draws from the exact mixture
    q ∝ p(theta|y) sum_j 1/p(y_j|theta) give every elpd_i within Monte Carlo error."""
    rng = np.random.default_rng(0)
    y = np.append(rng.normal(0.0, 1.0, 30), 6.0)            # the outlier is where PSIS fails
    grid = np.linspace(-5, 9, 4001)
    lp = norm.logpdf(grid, 0, 3)
    ll = norm.logpdf(y[None, :], grid[:, None], 1.0)         # (grid, obs)
    log_post = lp + ll.sum(1)
    exact = np.array([logsumexp(log_post - ll[:, i] + ll[:, i])
                      - logsumexp(log_post - ll[:, i]) for i in range(len(y))])
    log_q = log_post + logsumexp(-ll, axis=1)
    q = np.exp(log_q - logsumexp(log_q))
    draws = rng.choice(len(grid), size=20000, p=q)
    elpd, ess = mixture_loo(ll[draws])
    np.testing.assert_allclose(elpd, exact, atol=0.03)
    assert ess.min() > 1000, "the mixture keeps every observation's weights usable"


def test_balanced_mixture_keeps_every_observation_usable():
    """Outliers of growing size take the unweighted mixture; weighting each component by a first
    estimate of its leave-one-out density (here the exact one, shifted, as any c_j > 0 is valid)
    gives every observation many effective draws and stays exact."""
    rng = np.random.default_rng(1)
    y = np.concatenate([rng.normal(0.0, 0.3, 200), [3.0, 4.0, 6.0, 9.0]])
    grid = np.linspace(-4, 12, 8001)
    ll = norm.logpdf(y[None, :], grid[:, None], 0.3)
    log_post = norm.logpdf(grid, 0, 3) + ll.sum(1)
    exact = np.array([logsumexp(log_post) - logsumexp(log_post - ll[:, i]) for i in range(len(y))])
    out = {}
    for name, log_c in (("plain", np.zeros(len(y))), ("balanced", exact + 0.5)):
        log_q = log_post + logsumexp(log_c[None, :] - ll, axis=1)
        q = np.exp(log_q - logsumexp(log_q))
        draws = rng.choice(len(grid), size=4000, p=q)
        out[name] = mixture_loo(ll[draws], log_c)
    np.testing.assert_allclose(out["balanced"][0], exact, atol=0.15)
    assert np.median(out["balanced"][1]) > 10 * np.median(out["plain"][1])

