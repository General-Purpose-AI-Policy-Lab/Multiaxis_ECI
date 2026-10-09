"""The integrated predictive of one latent block matches exact integration on a grid."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.special import logsumexp
from scipy.stats import norm

from multiaxis_eci.scripts import load

_il = load("4_diagnostics/integrated_loo.py")


@pytest.mark.parametrize("beta", [None, 1.0, 5.0], ids=["linear", "semicomp", "noncomp"])
def test_block_predictive_matches_grid_integration(beta):
    """A 2-D block with six Beta-binomial scores and a Gaussian prior: the Laplace cavity
    predictive of each score is close to the exact leave-one-out integral over the block, on the
    linear link and on the softmin links (Gauss-Newton Laplace, linearized cavity)."""
    f = {"beta": None if beta is None else np.array([beta]), "offset": np.array([[0.1, -0.1]])}
    rng = np.random.default_rng(0)
    J, K = 6, 2
    A = np.abs(rng.normal(1.0, 0.4, size=(J, K)))
    D = rng.normal(0.0, 0.5, size=J)
    off = rng.normal(0.0, 0.1, size=(J, K))
    n_items = np.full(J, 40.0)
    truth = np.array([0.6, -0.3])
    eta_true = _il.eta_jac(f, A[None], (truth + off)[None], D[None])[0][0]
    c = np.full(J, 0.1)
    phi = np.full(J, 30.0)
    mu = c + (1 - c) / (1 + np.exp(-eta_true))
    k = rng.binomial(40, mu).astype(float)
    y = np.clip(k / 40, 0.005, 0.995)
    m0, p0 = np.array([0.2, 0.0]), np.array([1.5, 0.8])
    args = (c, phi, n_items, k, y)

    approx = _il.block_predictive(f, A[None], off[None], D[None], tuple(a[None] for a in args),
                                  m0[None], p0[None], truth[None])[0]

    g = np.linspace(-4, 4, 401)
    X, Y = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel()], 1)                             # (G, 2)
    log_prior = norm.logpdf(pts, m0, 1 / np.sqrt(p0)).sum(1)
    G = len(pts)
    fg = {"beta": None if beta is None else np.full(G, beta), "offset": np.repeat(f["offset"], G, 0)}
    eta = _il.eta_jac(fg, np.broadcast_to(A, (G, J, K)), pts[:, None, :] + off[None],
                      np.broadcast_to(D, (G, J)))[0]
    ll = _il.obs_loglik(eta, *args)                                       # (G, J)
    tot = log_prior + ll.sum(1)
    exact = np.array([logsumexp(tot) - logsumexp(tot - ll[:, i]) for i in range(J)])
    print(np.abs(approx - exact).max())
    # The softmin links are scored by importance sampling (integrated_loo.IS_DRAWS): Monte Carlo
    # error on one posterior draw, about 0.05 nats at most here.
    np.testing.assert_allclose(approx, exact, atol=0.05 if beta is None else 0.08)
