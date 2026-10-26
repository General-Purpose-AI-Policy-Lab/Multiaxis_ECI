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


@pytest.mark.parametrize("n_scores,child", [(6, False), (2, False), (3, True)],
                         ids=["well-measured", "thin", "founder-with-child"])
def test_block_predictive_general_prior_matches_grid(n_scores, child):
    """The higher-order prior (`--theta-general`): theta = lambda g + sqrt(1 - lambda^2) s, s
    skew-normal, integrated as the block (s, g) seen through theta = M x (integrated_loo
    `_general_blocks`), against the exact integral over theta on a grid, whose prior density is
    the integral over g; with a child's correlated Gaussian step factor for a chain founder."""
    from scipy.stats import multivariate_normal, skewnorm
    f = {"beta": None, "offset": np.array([[0.0, 0.0]])}
    rng = np.random.default_rng(2)
    J, K = n_scores, 2
    A = np.abs(rng.normal(1.0, 0.4, size=(J, K)))
    D = rng.normal(0.0, 0.5, size=J)
    off = rng.normal(0.0, 0.1, size=(J, K))
    n_items = np.full(J, 40.0)
    truth = np.array([0.5, 0.1])
    c, d, phi = np.full(J, 0.1), np.full(J, 0.95), np.full(J, 30.0)
    eta_true = (A * (truth + off)).sum(1) - D
    k = rng.binomial(40, c + (d - c) / (1 + np.exp(-eta_true))).astype(float)
    y = np.clip(k / 40, 0.005, 0.995)
    alpha, lam = np.array([-5.0, 2.0]), np.array([0.7, 0.5])
    xi, omega = _il.skew_params(alpha)
    cc = np.sqrt(1 - lam ** 2)
    M = np.concatenate([np.diag(cc), lam[:, None]], 1)                  # (K, K+1)
    R = np.outer(lam, lam) + np.diag(1 - lam ** 2)
    Q = np.linalg.inv(np.diag([0.6, 0.4]) @ R @ np.diag([0.6, 0.4]))    # a child's step precision
    mu_ch = np.array([0.9, 0.6])
    P0 = (M.T @ Q @ M) if child else np.zeros((K + 1, K + 1))
    h = (M.T @ Q @ mu_ch) if child else np.zeros(K + 1)
    skew_x = (np.r_[xi, 0.0][None], np.r_[omega, 1.0][None], np.r_[alpha, 0.0][None])
    D_eff = D - (A * off).sum(1)
    approx = _il.block_predictive(f, (A @ M)[None], np.zeros((1, J, K + 1)), D_eff[None],
                                  tuple(a[None] for a in (c, phi, n_items, k, y, d)), None,
                                  (P0[None], h[None]), np.r_[truth / cc, 0.0][None],
                                  skew=skew_x, iters=25)[0]
    g = np.linspace(-5, 4, 301)
    X, Y = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel()], 1)
    gq, wq = np.polynomial.hermite_e.hermegauss(80)
    lw = np.log(wq / np.sqrt(2 * np.pi))
    dens = sum(skewnorm.logpdf((pts[:, i, None] - lam[i] * gq) / cc[i], alpha[i], loc=xi[i],
                               scale=omega[i]) - np.log(cc[i]) for i in range(K))
    log_prior = logsumexp(dens + lw, axis=1)
    if child:
        log_prior = log_prior + multivariate_normal(mu_ch, np.linalg.inv(Q)).logpdf(pts)
    eta = np.einsum("gjk,jk->gj", pts[:, None, :] + off[None], A) - D
    ll = _il.obs_loglik(eta, c, phi, n_items, k, y, d)
    tot = log_prior + ll.sum(1)
    exact = np.array([logsumexp(tot) - logsumexp(tot - ll[:, i]) for i in range(J)])
    print(np.abs(approx - exact).max())
    np.testing.assert_allclose(approx, exact, atol=0.08)


def _toy_lineage_fit(general: bool, rng):
    """A K=2 Brownian-lineage MIRT on seven test-takers: chain c0 founder -> n1 -> n2, chain c1
    founder -> n4, and two models outside the chains, every one scored on three benchmarks.
    Returns the PyMC model, the data and the lineage structure."""
    import pandas as pd

    from multiaxis_eci.data import ECIData
    from multiaxis_eci.lineage import LineageStructure
    from multiaxis_eci.models.mirt import build_mirt_model
    n_models, n_bench = 7, 3
    model_idx = np.repeat(np.arange(n_models), n_bench)
    bench_idx = np.tile(np.arange(n_bench), n_models)
    n = len(model_idx)
    data = ECIData(
        scores=rng.uniform(0.1, 0.9, n), zero_score_mask=np.zeros(n, bool), model_idx=model_idx,
        bench_idx=bench_idx,
        mlookup=pd.DataFrame({"model": [f"m{i}" for i in range(n_models)],
                              "model_idx": np.arange(1, n_models + 1)}),
        blookup=pd.DataFrame({"benchmark": [f"b{j}" for j in range(n_bench)],
                              "benchmark_idx": np.arange(1, n_bench + 1)}),
        n_models=n_models, n_benchmarks=n_bench, n_obs=n, zero_diag_threshold=0.01,
        n_obs_per_model=np.full(n_models, n_bench), is_low_obs=np.zeros(n_models, bool),
        excluded_benchmarks=set(), is_human=np.zeros(n_models, bool), bench_category=None,
        is_sota=np.zeros(n_models, bool))
    lin = LineageStructure(
        row_idx=np.arange(5), node_idx=np.arange(5), offset_group=np.arange(5),
        B=np.array([[1, 0], [1, 0], [1, 0], [0, 1], [0, 1]], float),
        C=np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0], [0, 0, 1]], float),
        delta_dt=np.array([0.5, 1.2, 0.8]), delta_chain=np.array([0, 0, 1]),
        founder_date=np.array(["2024-01-01", "2024-06-01"], dtype="datetime64[D]"),
        n_chains=2, n_nodes=5, n_deltas=3, n_groups=5, chain_names=["c0", "c1"])
    model = build_mirt_model(data, 2, lineage=lin, lineage_bm=True, theta_skew=True,
                             theta_general=general)
    return model, data, lin


@pytest.mark.parametrize("general", [False, True], ids=["skew-normal", "higher-order"])
def test_block_priors_match_the_model(general):
    """Each block's conditional prior in `block_priors` is the PyMC model's own: moving a block
    while every other parameter stays put (the release levels of its parent and children
    included) changes the model's prior log-density exactly as the block's Gaussian and
    skew-normal factors say, and changes the abilities of the block's scores only. Under the
    skew-normal block a chain founder has no incoming step (it took a spurious one from the last
    node of the map until 2026-10-26, which this test would have caught)."""
    rng = np.random.default_rng(3)
    model, data, lin = _toy_lineage_fit(general, rng)
    point = {k: v + rng.normal(0, 0.7, np.shape(v)) for k, v in model.initial_point().items()}
    if "theta_skew_alpha" in point:
        point["theta_skew_alpha"] = np.array([-4.0, 2.5])
    names = ["A", "theta", "D", "phi_b", "theta_skew_alpha", "lin_base", "lin_delta_z",
             "lin_drift", "lin_spread"] + (["theta_general_lambda"] if general else [])
    # the variables as functions of the point (left as random variables they would be drawn)
    evaluate = model.compile_fn(model.replace_rvs_by_values([model[v] for v in names]),
                                inputs=model.value_vars, on_unused_input="ignore")
    logp_all = model.compile_logp()
    logp_obs = model.compile_logp(vars=model.observed_RVs)

    def values(pt):
        return dict(zip(names, evaluate(pt)))

    def prior_logp(pt):
        return float(logp_all(pt) - logp_obs(pt))

    v0 = values(point)
    f = _il.assemble_fit(lambda v: None if v not in v0 else np.asarray(v0[v])[None], 1, data,
                         lin, np.array([], dtype=int), None, None, general=general)
    parent, d_in, children = _il.lineage_graph(lin)
    lam = v0["theta_general_lambda"] if general else np.zeros(2)
    L = np.linalg.cholesky(np.outer(lam, lam) + np.diag(1 - lam ** 2))
    spread = v0["lin_spread"]
    # rows of the exchangeable block: the models outside the chains, then the chain founders
    rest = np.setdiff1d(np.arange(data.n_models), lin.row_idx)

    def raw_step(col, d_psi):
        """the change of lin_delta_z's row col that moves its step by d_psi"""
        return np.linalg.solve(L, d_psi / (spread * np.sqrt(lin.delta_dt[col])))

    blocks = list(_il.block_priors(f))
    assert len(blocks) == lin.n_nodes + rest.size
    # every score sits in exactly one block
    np.testing.assert_array_equal(np.sort(np.concatenate([b["obs"] for b in blocks])),
                                  np.arange(data.n_obs))
    for blk in blocks:
        rows = np.unique(data.model_idx[blk["obs"]])
        u = rng.normal(0, 0.3, blk["x0"].shape[1])
        pt = {k: np.array(v, copy=True) for k, v in point.items()}
        # the model coordinates of the block, and the children's steps that keep them in place
        if blk["M"] is None and rows[0] in lin.row_idx and parent[lin.node_idx[rows[0]]] >= 0:
            n = lin.node_idx[rows[0]]
            d_theta = u
            pt["lin_delta_z"][d_in[n]] += raw_step(d_in[n], u)
        else:
            if rows[0] in lin.row_idx:
                n = lin.node_idx[rows[0]]
                cell = rest.size + int(lin.B[n].argmax())
            else:
                n, cell = None, int(np.flatnonzero(rest == rows[0])[0])
            pt["theta_skew"][cell] += u[:2]
            d_theta = u[:2]
            if general:
                pt["theta_general_g"][cell] += u[2]
                d_theta = blk["M"][0] @ u
        if n is not None:
            for ch in children[n]:
                pt["lin_delta_z"][d_in[ch]] -= raw_step(d_in[ch], d_theta)
        moved = np.flatnonzero(np.abs(values(pt)["theta"] - v0["theta"]).sum(1) > 1e-9)
        np.testing.assert_array_equal(moved, rows)

        P0, h = blk["prior"][0][0], blk["prior"][1][0]

        def factor(x, blk=blk, P0=P0, h=h):
            out = -0.5 * x @ P0 @ x + h @ x
            if blk["skew"] is not None:
                out += _il.skew_terms(x[None], tuple(s[:1] for s in blk["skew"]))[0].sum()
            return out
        x0 = blk["x0"][0] if blk["M"] is None else np.r_[point["theta_skew"][cell],
                                                          point["theta_general_g"][cell]]
        np.testing.assert_allclose(prior_logp(pt) - prior_logp(point),
                                   factor(x0 + u) - factor(x0), atol=1e-8)


def test_psis_elpd_matches_importance_sampling_on_light_tails():
    """Where the importance weights 1/p have light tails, PSIS-LOO is the plain importance
    estimate, -log mean(1/p) per observation, with a small Pareto k (draws along the last axis
    of the log weights arviz expects, observations first)."""
    rng = np.random.default_rng(4)
    ll = np.array([-1.0, -2.5, -0.3]) + rng.normal(0, 0.05, (4000, 3))
    elpd, k = _il.psis_elpd(ll)
    from scipy.special import logsumexp as lse
    np.testing.assert_allclose(elpd, -(lse(-ll, axis=0) - np.log(len(ll))), atol=1e-3)
    assert np.all(k < 0.5)
