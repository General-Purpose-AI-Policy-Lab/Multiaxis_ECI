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


@pytest.mark.parametrize("n_scores", [6, 2], ids=["well-measured", "thin"])
def test_block_predictive_skew_prior_and_ceiling_match_grid(n_scores):
    """The skew-normal block (the K-axis default since 2026-10-20) and a --ceiling-noise ceiling:
    the Laplace cavity predictive, with the skew-normal prior's exact gradient and curvature,
    stays close to the exact leave-one-out integral, on a block pinned by its scores and on a
    thin one whose posterior is mostly its skewed prior."""
    from scipy.stats import skewnorm
    f = {"beta": None, "offset": np.array([[0.0, 0.0]])}
    rng = np.random.default_rng(1)
    J, K = n_scores, 2
    A = np.abs(rng.normal(1.0, 0.4, size=(J, K)))
    D = rng.normal(0.0, 0.5, size=J)
    off = np.zeros((J, K))
    n_items = np.full(J, 40.0)
    truth = np.array([0.5, -0.4])
    c, d, phi = np.full(J, 0.1), np.full(J, 0.9), np.full(J, 30.0)
    eta_true = (A * (truth + off)).sum(1) - D
    k = rng.binomial(40, c + (d - c) / (1 + np.exp(-eta_true))).astype(float)
    y = np.clip(k / 40, 0.005, 0.995)
    alpha = np.array([-6.0, 1.5])
    xi, omega = _il.skew_params(alpha)
    args = (c, phi, n_items, k, y, d)
    approx = _il.block_predictive(f, A[None], off[None], D[None], tuple(a[None] for a in args),
                                  np.zeros((1, K)), np.zeros((1, K)), truth[None],
                                  skew=(xi[None], omega[None], alpha[None]))[0]
    g = np.linspace(-5, 4, 451)
    X, Y = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel()], 1)
    log_prior = sum(skewnorm.logpdf(pts[:, i], alpha[i], loc=xi[i], scale=omega[i]) for i in range(K))
    eta = pts @ A.T - D
    ll = _il.obs_loglik(eta, *args)
    tot = log_prior + ll.sum(1)
    exact = np.array([logsumexp(tot) - logsumexp(tot - ll[:, i]) for i in range(J)])
    print(np.abs(approx - exact).max())
    # Scored by importance sampling on the Laplace fit (a pure Laplace cavity missed by up to
    # 0.11 nats here): unbiased to about 0.01 over IS seeds, up to 0.07 on one posterior draw
    # at IS_DRAWS = 1024, the same Monte Carlo error as the softmin links.
    np.testing.assert_allclose(approx, exact, atol=0.08)


def test_trace_lineage_reads_the_frozen_map_and_checks_the_chains(tmp_path):
    """A model rebuilt from a trace reads the lineage map frozen beside it at fit time, and a map
    whose chains are not the trace's raises instead of re-indexing the lineage draws."""
    import json
    from types import SimpleNamespace

    import pandas as pd

    from multiaxis_eci.lineage import FITTED_MAP_NAME, LINEAGE_MAP, trace_lineage_structure
    lm = pd.read_csv(LINEAGE_MAP)
    mlookup = pd.DataFrame({"model": lm["raw_string"].drop_duplicates().tolist()})
    full = trace_lineage_structure(mlookup, tmp_path / "trace_mirt_k4_x.nc")
    post = SimpleNamespace(attrs={"mirt_lineage_chains": json.dumps(list(full.chain_names))})
    assert trace_lineage_structure(mlookup, tmp_path / "trace_mirt_k4_x.nc", post).n_chains == full.n_chains
    # the fit's frozen map lacks one chain: it is read, and the trace's chain list rejects it
    drop = full.chain_names[-1]
    lm[lm["chain"] != drop].to_csv(tmp_path / FITTED_MAP_NAME, index=False)
    assert trace_lineage_structure(mlookup, tmp_path / "trace_mirt_k4_x.nc").n_chains == full.n_chains - 1
    with pytest.raises(ValueError, match="changed after the fit"):
        trace_lineage_structure(mlookup, tmp_path / "trace_mirt_k4_x.nc", post)
