"""Integrated leave-one-out of a K-axis MIRT fit: each score's own latent block integrated out.

PSIS-LOO on the K=4 flagship leaves 14% of the observations with a Pareto k above 0.7. A score
moves the ability of its model a lot when the model has few scores, so the full posterior covers
the leave-one-out posterior poorly in exactly that direction. The remedy the literature gives for
latent-variable models (Vehtari et al. 2016, JMLR; Merkle, Furr & Rabe-Hesketh 2019; the `loo`
documentation for hierarchical models) is to integrate that direction out, and importance-sample
only what is left:

    p(y_i | y_-i) = E_{phi ~ p(phi | y_-i)} p(y_i | y_-i, phi),

with phi every parameter but the score's latent block, so the weights are 1 / p(y_i | y_-i, phi_s)
(Pareto-smoothed), much flatter than 1 / p(y_i | theta_s).

The latent block of a score is
  * for a model of a release chain: its release's level psi (K-vector), which every variant of
    the release shares. Its conditional prior given phi is Gaussian: the Brownian lineage step
    from its parent, N(psi_parent + drift*dt, spread^2*dt), times one such factor per child
    (a founder's own base is one entry of the ZeroSumNormal, taken as N(0, 1));
  * for a model outside the chains: its theta, prior N(0, I) (the ZeroSumNormal entry);
  * human tiers (ordered prior) are not integrated: their scores keep plain PSIS.
Under the skew-normal theta block (the K-axis default since 2026-10-20) the founders' bases and the
models outside the chains have a skew-normal prior per axis, SkewNormal(xi_k, omega_k, alpha_k) at
the draw's alpha, and a founder no incoming step (until 2026-10-26 it took a spurious one, from the
last node of the map: about 2,900 nats on a chain of the 2026-10-19 flagship). Under the
higher-order prior (--theta-general) the steps are correlated across axes and a cell is
lambda g + sqrt(1 - lambda^2) s, integrated as (s, g) (`block_priors`). Under --ceiling-noise the
curve stops at the draw's ceilings, mu = c + (d - c) sigmoid(eta). Under --general-sums every block
is built in (g, z) and the reported ability is its image theta = T x (theta_0 = g, theta_k =
g + sigma_k z_k): the blocks are integrated in x, where their priors are the ones above, and
the scores see them through the loadings A T. Each block's prior is checked
against the PyMC model's own (tests/test_integrated_loo.py).
For each draw, the block's conditional posterior (prior times every score of the block) is
approximated by a Laplace fit; removing score i gives its cavity Gaussian in closed form, and
since score i sees the block only through eta_i = A_i . (psi + offset) - D_i, the predictive
density p(y_i | y_-i, phi_s) is a one-dimensional Gauss-Hermite integral over eta_i. Nothing
is refitted, so every posterior mode of a trace is scored from its own chains. On the softmin
links (models/mirt.softmin_eta, `--link semicomp/noncomp`) eta is not linear in the block, and
each block is integrated by importance sampling on its Laplace fit instead (`softmin_block_is`).

    python 4_diagnostics/integrated_loo.py [--trace TRACE] [--groups A2:2,4,5 A1:0,1 B:7,8,9]

Every chain is scored alone and a group of chains (a mode, by default those of the modes file)
by the average of its chains: PSIS is optimistic with few draws where k is high, so groups are
compared at equal draws. Output in `diagnostics/` beside the trace:
`integrated_loo_<group>.csv` (elpd_i, the largest Pareto k over its chains, and the conditional
PSIS elpd_i and k) and `integrated_loo.json` (totals, per-chain totals and the Monte Carlo error
they imply, paired differences to the first group on every observation and on those with k < 0.7
in every group).

Checked on the flagship (2026-10-06): the numpy likelihood reproduces the trace's to 1e-9; on 721
real (score, draw) pairs the Laplace-cavity predictive matches a 200,000-sample importance
integral to +0.001 nats on average (90% within 0.03); where conditional PSIS is reliable
(k < 0.5) the two estimates agree, the gap shrinking with the draws (-0.013 nats per score at
400 draws, -0.006 at 1,200).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import expit, gammaln, log_ndtr, logsumexp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "2_model"))

from multiaxis_eci import config  # noqa: E402

GH_X, GH_W = np.polynomial.hermite_e.hermegauss(40)   # probabilists' Hermite: weight exp(-x^2/2)
LOG_GH_W = np.log(GH_W / np.sqrt(2 * np.pi))


# ── likelihood, in numpy, as models/mirt.py writes it ───────────────────────────────────────

def obs_loglik(eta, c, phi, n_items, k, y, d=1.0):
    """log p(y | eta) of the fixed-c 3PL Beta-binomial (n_items > 0) / Beta (n_items == 0)
    likelihood, capped at the ceiling d (1 without --ceiling-noise); every argument broadcasts."""
    mu = c + (d - c) * expit(eta)
    a, b = mu * phi, (1.0 - mu) * phi
    is_b = n_items > 0
    n = np.where(is_b, n_items, 1.0)
    kk = np.where(is_b, k, 0.5)
    log_beta_ab = gammaln(a) + gammaln(b) - gammaln(a + b)
    binom = (gammaln(n + 1.0) - gammaln(kk + 1.0) - gammaln(n - kk + 1.0)
             + gammaln(kk + a) + gammaln(n - kk + b) - gammaln(n + a + b) - log_beta_ab)
    beta = (a - 1.0) * np.log(y) + (b - 1.0) * np.log1p(-y) - log_beta_ab
    return np.where(is_b, binom, beta)


def eta_derivatives(eta, args, h: float = 1e-3):
    """First and second derivative of obs_loglik in eta, by central differences."""
    f0, fp, fm = obs_loglik(eta, *args), obs_loglik(eta + h, *args), obs_loglik(eta - h, *args)
    return f0, (fp - fm) / (2 * h), (fp - 2 * f0 + fm) / h ** 2


# ── the skew-normal prior of the exchangeable block ─────────────────────────────────────────

def skew_params(alpha):
    """(xi, omega) of the standardised skew-normal of shape alpha (models/mirt._theta_skew_cells)."""
    delta = alpha / np.sqrt(1 + alpha ** 2)
    omega = 1.0 / np.sqrt(1 - 2 * delta ** 2 / np.pi)
    return -omega * delta * np.sqrt(2 / np.pi), omega


def skew_terms(x, skew):
    """log density (up to its constant in x), gradient and second derivative of the skew-normal
    prior at x, per coordinate: z = (x - xi) / omega, t = alpha z, lambda = phi(t) / Phi(t),
    d/dx = (-z + alpha lambda) / omega, d2/dx2 = (-1 - alpha^2 lambda (t + lambda)) / omega^2."""
    xi, omega, alpha = skew
    z = (x - xi) / omega
    t = alpha * z
    log_cdf = log_ndtr(t)
    lam = np.exp(-0.5 * t ** 2 - 0.5 * np.log(2 * np.pi) - log_cdf)
    return (-0.5 * z ** 2 + log_cdf - np.log(omega),
            (-z + alpha * lam) / omega,
            (-1.0 - alpha ** 2 * lam * (t + lam)) / omega ** 2)


# ── one latent block, every draw at once ─────────────────────────────────────────────────────

def eta_jac(f, A_o, xo, D_o):
    """eta (S, J) of scores whose (offset-included) abilities are xo (S, J, K), and its
    gradient in xo (S, J, K): A_o for the linear link, s * softmax weights for the softmin
    links (models/mirt.softmin_eta, the per-axis offset f["offset"] included)."""
    if f["beta"] is None:
        return np.einsum("sjk,sjk->sj", A_o, xo) - D_o, A_o
    beta = f["beta"][:, None, None] if np.ndim(f["beta"]) else f["beta"]
    z = xo + f["offset"][:, None, :]
    s = A_o.sum(-1)
    lw = np.log(A_o + 1e-12) - np.log(s)[..., None] - beta * z
    lse = logsumexp(lw, axis=-1)
    pi = np.exp(lw - lse[..., None])
    b = beta[..., 0] if np.ndim(beta) else beta
    return -s * lse / b - D_o, s[..., None] * pi


def _gauss_natural(m0, p0, K):
    """The block's Gaussian prior factor as (P0 (S, K, K), h (S, K)), log factor -x'P0x/2 + h'x:
    from a mean and a diagonal precision (S, K), or from p0 = (P0, h) already full, the form the
    higher-order prior gives (`general_blocks`), whose precision can be singular in the block."""
    if isinstance(p0, tuple):
        return p0
    return p0[:, :, None] * np.eye(K), p0 * m0


def block_predictive(f, A_o, off, D_o, args, m0, p0, x0, iters: int = 10, skew=None):
    """log p(y_i | y_-i, phi_s) for every score i of one latent block and every draw s.

    A_o (S, J, K) loadings of the block's J scores, off (S, J, K) the per-score offset added to
    the block value (a variant's offset from its release level; 0 outside the chains), D_o (S, J)
    difficulties, args the per-score likelihood inputs broadcast to (S, J), m0 / p0 (S, K) the
    block's Gaussian prior mean and precision (diagonal), x0 (S, K) a starting value (the draw's
    own block value), skew None or (xi, omega, alpha) (S, K) of a skew-normal prior factor on
    top of the Gaussian one (`skew_terms`). p0 may instead be (P0, h), a full precision and its
    natural mean term (`_gauss_natural`; m0 is then unused). Returns (S, J). On a softmin link eta is not linear in the block: the
    Laplace fit is Gauss-Newton on its gradient J (`eta_jac`) and serves as the proposal of
    `softmin_block_is`; on the linear link J = A_o and the cavity is exact as before.
    """
    K = x0.shape[1]
    P0, h = _gauss_natural(m0, p0, K)
    eye = np.eye(K)
    x = x0.copy()
    for _ in range(iters):
        eta, J = eta_jac(f, A_o, x[:, None, :] + off, D_o)
        _, d1, d2 = eta_derivatives(eta, args)
        w = np.maximum(-d2, 1e-8)                                 # curvature, kept positive
        grad = np.einsum("sj,sjk->sk", d1, J) + h - np.einsum("skl,sl->sk", P0, x)
        pp = P0
        if skew is not None:
            _, g1, g2 = skew_terms(x, skew)
            grad, pp = grad + g1, P0 + np.maximum(-g2, 1e-8)[:, :, None] * eye
        P = np.einsum("sj,sjk,sjl->skl", w, J, J) + pp
        step = np.clip(np.linalg.solve(P, grad[..., None])[..., 0], -1.0, 1.0)   # damped
        x = x + step
        if np.abs(step).max() < 1e-6:
            break
    eta, J = eta_jac(f, A_o, x[:, None, :] + off, D_o)
    _, d1, d2 = eta_derivatives(eta, args)
    w = np.maximum(-d2, 1e-8)
    pp = P0 if skew is None else P0 + np.maximum(-skew_terms(x, skew)[2], 1e-8)[:, :, None] * eye
    P = np.einsum("sj,sjk,sjl->skl", w, J, J) + pp
    Sigma = np.linalg.inv(P)
    if f["beta"] is not None or skew is not None:
        # a softmin link, or a skew-normal prior: the conditional posterior is skewed, and the
        # Laplace cavity misses the exact integral by up to 0.1 nats a score (tests)
        return softmin_block_is(f, A_o, off, D_o, args, P0, h, x, Sigma, skew=skew)
    v_full = np.einsum("sjk,skl,sjl->sj", J, Sigma, J)            # J_i' Sigma J_i
    # Cavity of score i (Sherman-Morrison): drop its curvature and its pull on the mode.
    v_cav = v_full / (1.0 - w * v_full)
    m_cav = eta - d1 * v_cav
    nodes = m_cav[..., None] + np.sqrt(v_cav)[..., None] * GH_X       # (S, J, Q)
    ll = obs_loglik(nodes, *[a[..., None] for a in args])
    return logsumexp(ll + LOG_GH_W, axis=-1)


# Importance sampling of a softmin-link block. Against exact grid integration on 2-D blocks
# (tests/test_integrated_loo.py), the largest per-score error on ONE posterior draw is about
# 0.1 nats at 256 draws, 0.05 at 1,024 with the proposal covariance doubled in scale, 0.03 at
# 4,096; averaging over the posterior draws shrinks it further.
IS_DRAWS = 1024       # importance draws per posterior draw and block
IS_INFLATE = 2.0      # proposal scale over the Laplace covariance
IS_CHUNK = 16         # posterior draws per chunk (memory: chunk x IS_DRAWS x J x K)
_IS_Z = np.random.default_rng(20261007).standard_t(5, size=(IS_DRAWS, 8))


def softmin_block_is(f, A_o, off, D_o, args, P0, h, x, Sigma, skew=None):
    """The integrated predictive of a softmin-link block by importance sampling: the block's
    conditional posterior is far from Gaussian there (the minimum switches axes away from the
    mode, so a Laplace or cubature cavity misses it by 0.1 to 0.4 nats a score), so the exact
    integral is estimated instead, with a proposal of independent t(5) coordinates on the Laplace fit (covariance
    inflated by IS_INFLATE**2) and every score of the block scored on the same draws:
    log p(y_i | y_-i) = LSE(log w) - LSE(log w - ll_i), w = prior * prod_j p(y_j | x) / q(x).
    The Gaussian prior factor comes in natural form, P0 (S, K, K) and h (S, K)."""
    from multiaxis_eci.models.mirt import softmin_eta
    S, J, K = A_o.shape
    z = _IS_Z[:, :K]
    out = np.empty((S, J))
    for a in range(0, S, IS_CHUNK):
        sl = slice(a, a + IS_CHUNK)
        L = np.linalg.cholesky(Sigma[sl] * IS_INFLATE ** 2 + 1e-10 * np.eye(K))   # (s, K, K)
        xs = x[sl, None, :] + np.einsum("mk,slk->sml", z, L)                 # (s, M, K)
        # log q: independent t(5) coordinates through L, up to a constant shared by every draw
        logdet = np.log(np.diagonal(L, axis1=-2, axis2=-1)).sum(-1)          # (s,)
        logq = -logdet[:, None] - 3.0 * np.log1p(z ** 2 / 5).sum(-1)[None, :]
        logprior = (-0.5 * np.einsum("smk,skl,sml->sm", xs, P0[sl], xs)
                    + np.einsum("smk,sk->sm", xs, h[sl]))
        if skew is not None:
            logprior = logprior + skew_terms(xs, tuple(s_[sl, None, :] for s_ in skew))[0].sum(-1)
        if f["beta"] is None:                                                 # linear link
            th = xs[:, :, None, :] + off[sl, None]
            eta = np.einsum("smjk,sjk->smj", th, A_o[sl]) - D_o[sl, None, :]
        else:
            beta = f["beta"][sl, None, None]
            th = xs[:, :, None, :] + off[sl, None] + f["offset"][sl, None, None, :]
            eta = softmin_eta(A_o[sl, None], th, D_o[sl, None], beta)        # (s, M, J)
        ll = obs_loglik(eta, *[np.asarray(g)[sl, None] for g in args])
        lw = logprior + ll.sum(-1) - logq                                     # (s, M)
        out[sl] = logsumexp(lw, axis=1)[:, None] - logsumexp(lw[..., None] - ll, axis=1)
    return out


# ── the fit's blocks ────────────────────────────────────────────────────────────────────────

def lineage_graph(lin):
    """Per node: parent node (-1 for a founder), the delta column into it, and its children."""
    n_nodes = lin.n_nodes
    path = [set(np.flatnonzero(lin.C[n])) for n in range(n_nodes)]
    chain = lin.B.argmax(axis=1)
    parent = np.full(n_nodes, -1)
    d_in = np.full(n_nodes, -1)
    for n in range(n_nodes):
        if not path[n]:
            continue
        for q in range(n_nodes):
            if chain[q] == chain[n] and len(path[q]) == len(path[n]) - 1 and path[q] < path[n]:
                parent[n] = q
                d_in[n] = next(iter(path[n] - path[q]))
                break
        if parent[n] < 0:
            raise ValueError(f"node {n}: no parent found in its chain")
    children = [np.flatnonzero(parent == n) for n in range(n_nodes)]
    return parent, d_in, children


def soft_link(spec) -> bool:
    return spec.link in ("semicomp", "noncomp", "semicomp0")


def load_fit(trace_path: Path, chains: list[int], lineage_map: Path | None = None):
    from multiaxis_eci.analysis.fitspec import FitSpec, sampled_human_order
    from multiaxis_eci.models.mirt import trace_softmin
    from multiaxis_eci.lineage import trace_lineage_structure
    from multiaxis_eci.models.mirt import _human_structure

    post_all = xr.open_dataset(trace_path, group="posterior")
    spec = FitSpec.from_trace(az.InferenceData(posterior=post_all), trace_path)
    if spec.time_prior or spec.theta_pos or spec.theta_t \
            or spec.link not in ("linear", "semicomp", "noncomp", "semicomp0") \
            or spec.known_se or spec.censor_bounds or not spec.lineage_bm or spec.private_bases:
        # private_bases: the founders' bases are private Normal(0, 1), not the exchangeable
        # block's cells `block_priors` gives them
        raise SystemExit("integrated_loo covers the flagship's specification only")
    data, floor_c, _ = spec.load_data(az.InferenceData(posterior=post_all))
    # the map the trace was sampled with (frozen beside it), its chains checked against the trace
    lin = trace_lineage_structure(data.mlookup, trace_path, post_all, csv_path=lineage_map)
    names = data.mlookup.sort_values("model_idx")["model"].tolist()
    hs = _human_structure(sampled_human_order(post_all, spec), names)
    human_rows = np.asarray(hs[0]) if hs is not None else np.array([], dtype=int)
    p = post_all.isel(chain=chains)
    S = p.sizes["chain"] * p.sizes["draw"]

    def get(v):
        return p[v].values.reshape(S, *p[v].shape[2:]) if v in p else None

    soft = trace_softmin(p)
    return assemble_fit(get, S, data, lin, human_rows, floor_c, spec.binomial_n(data),
                        general=spec.theta_general, soft=soft, general_sums=spec.general_sums)


def assemble_fit(get, S: int, data, lin, human_rows, floor_c, binomial_n, general: bool = False,
                 soft=None, general_sums: bool = False) -> dict:
    """The arrays the integrated LOO reads, from `get(name)` -> (S, ...) draws of a model
    variable (None when absent): the trace's posterior in `load_fit`, a point of the PyMC model
    in the tests. The release levels psi are rebuilt from the lineage variables as
    models/mirt._lineage_block builds them. Under general_sums `theta` and `A` are returned in
    the blocks' (g, z) coordinates: x = T^-1 theta and A T, so eta = A theta - D is unchanged."""
    A, theta, D, phi = get("A"), get("theta"), get("D"), get("phi_b")
    if general_sums:
        # theta = T x, T = [[1, 0], [1, diag(sigma)]] (models/mirt._assemble_theta)
        if soft is not None:
            raise SystemExit("integrated_loo: --general-sums on a softmin link is not covered")
        K = A.shape[2]
        T = np.zeros((S, K, K))
        T[:, :, 0] = 1.0
        T[:, 1:, 1:] = get("theta_dev_sd")[:, :, None] * np.eye(K - 1)
        A = np.einsum("sbk,skl->sbl", A, T)
        theta = np.einsum("skl,snl->snk", np.linalg.inv(T), theta)
    ceil = get("ceiling_d")
    ceil = np.ones_like(D) if ceil is None else ceil
    skew = None
    if get("theta_skew_alpha") is not None:
        alpha = get("theta_skew_alpha")
        xi, omega = skew_params(alpha)
        skew = (xi, omega, alpha)
    base, z = get("lin_base"), get("lin_delta_z")
    drift, spread = get("lin_drift"), get("lin_spread")
    dt = lin.delta_dt
    lam = R_inv = None
    if general:
        # Higher-order prior (models/mirt._general_mix): the steps' standard-normal cells are
        # mixed by chol(R), R = lam lam' + diag(1 - lam^2), before the drift and spread apply.
        if soft is not None:
            raise SystemExit("integrated_loo: --theta-general on a softmin link is not covered")
        if skew is None:
            raise SystemExit("integrated_loo: --theta-general is built on the skew-normal block")
        lam = get("theta_general_lambda")
        R = lam[:, :, None] * lam[:, None, :] + (1.0 - lam ** 2)[:, :, None] * np.eye(lam.shape[1])
        z = np.einsum("skl,sdl->sdk", np.linalg.cholesky(R), z)
        R_inv = np.linalg.inv(R)
    delta = drift[:, None, :] * dt[None, :, None] + spread[:, None, :] * np.sqrt(dt)[None, :, None] * z
    psi = np.einsum("nc,sck->snk", lin.B, base) + np.einsum("nd,sdk->snk", lin.C, delta)

    n_items = np.where(np.isfinite(binomial_n) & (binomial_n > 0), binomial_n, 0.0) \
        if binomial_n is not None else np.zeros(data.n_obs)
    k = np.clip(data.scores, 0.0, 1.0) * n_items
    y = np.clip(data.scores, config.CONTINUOUS_EPS, 1.0 - config.CONTINUOUS_EPS)
    c = floor_c[data.bench_idx] if floor_c is not None else np.zeros(data.n_obs)
    beta = offset = None
    if soft is not None:
        beta, offset = soft
        # One beta per draw either way: "noncomp" fixes it (config.NONCOMP_BETA), so it is
        # broadcast to the draws rather than kept a scalar the per-draw slicing cannot index.
        beta = np.asarray(beta).reshape(S) if np.ndim(beta) else np.full(S, float(beta))
        offset = offset.reshape(S, -1)
    return dict(data=data, lin=lin, human_rows=human_rows, A=A, theta=theta, D=D, phi=phi,
                beta=beta, offset=offset, ceil=ceil, skew=skew, lam=lam, R_inv=R_inv,
                psi=psi, drift=drift, spread=spread, n_items=n_items, k=k, y=y, c=c, S=S)


def conditional_loglik(f) -> np.ndarray:
    """log p(y_n | theta_s), (S, n_obs), recomputed in numpy from the posterior."""
    d = f["data"]
    eta, _ = eta_jac(f, f["A"][:, d.bench_idx], f["theta"][:, d.model_idx], f["D"][:, d.bench_idx])
    return obs_loglik(eta, f["c"], f["phi"][:, d.bench_idx], f["n_items"], f["k"], f["y"],
                      f["ceil"][:, d.bench_idx])


def block_priors(f):
    """Every latent block the integrated LOO integrates out, with its conditional prior given phi:
    one dict per block with `obs` (the block's scores), `x0` (S, K') a start (the draw's value),
    `off` (S, J, K) the per-score offset added to the abilities, `prior` = (P0, h) the Gaussian
    factor in natural form (S, K', K') and (S, K'), `skew` None or the (xi, omega, alpha) (S, K')
    of a skew-normal factor per coordinate, `M` None (the block is the abilities, K' = K) or
    (S, K, K') with abilities M x (the higher-order prior's (s, g), K' = K + 1), and `iters` the
    Newton budget. tests/test_integrated_loo.py checks each prior against the PyMC model's own.

    A release level psi_n: its parent's level plus a Brownian step N(drift dt, spread^2 dt), and
    one such factor per child; under the higher-order prior the step's cells are mixed by
    chol(R), R = lambda lambda' + diag(1 - lambda^2), so a step factor has the full precision
    D^-1 R^-1 D^-1, D = diag(spread sqrt(dt)). A chain founder has no incoming step: its own base
    is the exchangeable block's cell, N(0, 1) (the ZeroSumNormal entry) under the Gaussian
    block, the skew-normal cell under the skew-normal one. A model outside the chains: that cell
    alone. Under the higher-order prior a cell is lambda g + c s, c = sqrt(1 - lambda^2), g ~ N(0, 1)
    and s skew-normal; its prior has no closed form in theta, so the block integrated is
    x = (s, g), whose prior is exact (g as a skew-normal of shape 0), seen through theta = M x,
    M = [diag(c) | lambda], a founder's children factors pulled back by M. Human tiers are not
    integrated."""
    d, lin, S = f["data"], f["lin"], f["S"]
    K = f["A"].shape[2]
    parent, d_in, children = lineage_graph(lin)
    lam, skew = f["lam"], f["skew"]
    eye = np.eye(K)
    if lam is not None:
        c = np.sqrt(1.0 - lam ** 2)
        M = np.concatenate([c[:, :, None] * eye, lam[:, :, None]], axis=2)
        xi, omega, alpha = skew
        cell_skew = (np.concatenate([xi, np.zeros((S, 1))], 1),
                     np.concatenate([omega, np.ones((S, 1))], 1),
                     np.concatenate([alpha, np.zeros((S, 1))], 1))

        def cell_start(theta_value):
            """(s, g) with g = 0 giving theta_value"""
            return np.concatenate([theta_value / c, np.zeros((S, 1))], 1)

        def step_precision(col):
            sd = f["spread"] * np.sqrt(lin.delta_dt[col])
            return f["R_inv"] / (sd[:, :, None] * sd[:, None, :])
    else:
        def step_precision(col):
            return eye / (f["spread"] ** 2 * lin.delta_dt[col])[:, :, None]

    def cell_block(obs, theta_value, off, P0, h):
        """the exchangeable cell (a founder's base, a model outside the chains), with the
        Gaussian factor (P0, h) of a founder's children in theta"""
        if lam is not None:
            return dict(obs=obs, x0=cell_start(theta_value), off=off, skew=cell_skew, M=M,
                        prior=(np.einsum("skl,skm,smn->sln", M, P0, M),
                               np.einsum("skl,sk->sl", M, h)), iters=25)
        if skew is None:                                              # the ZeroSumNormal entry
            P0, h = P0 + eye, h
        return dict(obs=obs, x0=theta_value, off=off, skew=skew, M=None, prior=(P0, h),
                    iters=10)

    rows_of_node = [lin.row_idx[lin.node_idx == n] for n in range(lin.n_nodes)]
    for n in range(lin.n_nodes):
        obs = np.flatnonzero(np.isin(d.model_idx, rows_of_node[n]))
        if obs.size == 0:
            continue
        P0, h = np.zeros((S, K, K)), np.zeros((S, K))
        steps = [(parent[n], d_in[n], +1)] if parent[n] >= 0 else []
        steps += [(ch, d_in[ch], -1) for ch in children[n]]
        for other, col, sign in steps:
            # the parent's step into n (sign +1), or n's step into a child (sign -1)
            Q = step_precision(col)
            mean = f["psi"][:, other] + sign * f["drift"] * lin.delta_dt[col]
            P0 = P0 + Q
            h = h + np.einsum("skl,sl->sk", Q, mean)
        off = f["theta"][:, d.model_idx[obs]] - f["psi"][:, n][:, None, :]
        if parent[n] >= 0:
            yield dict(obs=obs, x0=f["psi"][:, n], off=off, skew=None, M=None, prior=(P0, h),
                       iters=10)
        else:
            yield cell_block(obs, f["psi"][:, n], off, P0, h)

    claimed = np.concatenate([f["human_rows"], lin.row_idx])
    for m in np.setdiff1d(np.arange(d.n_models), claimed):
        obs = np.flatnonzero(d.model_idx == m)
        if obs.size == 0:
            continue
        yield cell_block(obs, f["theta"][:, m], np.zeros((S, obs.size, K)),
                         np.zeros((S, K, K)), np.zeros((S, K)))


def integrated_loglik(f) -> tuple[np.ndarray, np.ndarray]:
    """log p(y_n | y_-n, phi_s), (S, n_obs), and a mask of the observations it integrates
    (False: human tiers, left at the conditional log-likelihood). Each block of `block_priors`
    is integrated by `block_predictive`; where the block is x with abilities M x, its scores see
    it through the loadings A M, the offsets moved into the difficulty (exact on the linear link)."""
    d, S = f["data"], f["S"]
    out = conditional_loglik(f)
    done = np.zeros(d.n_obs, dtype=bool)
    for blk in block_priors(f):
        obs, off = blk["obs"], blk["off"]
        b = d.bench_idx[obs]
        args = (f["c"][obs][None, :], f["phi"][:, b], f["n_items"][obs][None, :],
                f["k"][obs][None, :], f["y"][obs][None, :], f["ceil"][:, b])
        args = tuple(np.broadcast_to(a, (S, len(obs))) for a in args)
        A_o, D_o = f["A"][:, b], f["D"][:, b]
        if blk["M"] is not None:
            D_o = D_o - np.einsum("sjk,sjk->sj", A_o, off)
            A_o = np.einsum("sjk,skl->sjl", A_o, blk["M"])
            off = np.zeros(A_o.shape)
        with np.errstate(all="ignore"):
            res = block_predictive(f, A_o, off, D_o, args, None, blk["prior"], blk["x0"],
                                   skew=blk["skew"], iters=blk["iters"])
        bad = ~np.isfinite(res)
        # A rare draw whose Newton fit runs off (extreme eta) keeps its conditional value.
        res[bad] = out[:, obs][bad]
        f["n_fallback"] = f.get("n_fallback", 0) + int(bad.sum())
        out[:, obs] = res
        done[obs] = True
    return out, done


def psis_elpd(loglik: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PSIS-LOO elpd_i and Pareto k from (S, n) log densities used as 1/p weights."""
    lw, k = az.psislw(-loglik.T)
    return logsumexp(lw + loglik.T, axis=1), k


def score_chain(trace_path: Path, chain: int, stored, lineage_map: Path | None = None) -> dict:
    """Integrated and conditional PSIS-LOO of one chain (pointwise elpd and k)."""
    f = load_fit(trace_path, [chain], lineage_map)
    cond = conditional_loglik(f)
    gap = float(np.abs(cond - stored.isel(chain=[chain]).values.reshape(f["S"], -1)).max())
    if gap > 1e-6:
        raise SystemExit(f"chain {chain}: the numpy likelihood does not reproduce the fit ({gap})")
    integ, done = integrated_loglik(f)
    e_int, k_int = psis_elpd(integ)
    e_cond, k_cond = psis_elpd(cond)
    return dict(f=f, e_int=e_int, k_int=k_int, e_cond=e_cond, k_cond=k_cond, done=done,
                fallback=int(f.get("n_fallback", 0)))


def _flagship_trace() -> Path:
    from multiaxis_eci.analysis.fitspec import FLAGSHIP_TRACE
    return FLAGSHIP_TRACE


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trace", type=Path, default=_flagship_trace())
    ap.add_argument("--groups", nargs="+", default=None, metavar="NAME:C1,C2",
                    help="chain groups to compare (default: the modes of the modes file); "
                         "the first is the reference")
    ap.add_argument("--lineage-map", type=Path, default=None,
                    help="the lineage map the trace was fitted with, when 1_curated/lineage_map.csv "
                         "has changed since (default: the current one)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="where the tables go (default: diagnostics/ beside the trace)")
    args = ap.parse_args()
    if args.groups:
        groups = {g.split(":")[0]: [int(c) for c in g.split(":")[1].split(",")]
                  for g in args.groups}
    else:
        modes_file = args.trace.parent / f"mirt_modes_{args.trace.stem}.json"
        groups = {m["label"]: m["chains"] for m in json.loads(modes_file.read_text())["modes"]}
    out_dir = args.out_dir or args.trace.parent / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    stored = xr.open_dataset(args.trace, group="log_likelihood")["obs"]

    # One chain at a time: PSIS is optimistic with few draws where k is high, so groups of
    # different sizes are compared on the AVERAGE of their chains' estimates, at equal draws
    # (on the flagship, pooling mode A's five chains moved its total by -65, almost all from
    # the observations still above k = 0.5).
    per_chain = {}
    for c in sorted({c for ch in groups.values() for c in ch}):
        per_chain[c] = score_chain(args.trace, c, stored, args.lineage_map)
        print(f"chain {c}: elpd_int {per_chain[c]['e_int'].sum():.1f}", flush=True)
    f = next(iter(per_chain.values()))["f"]
    d = f["data"]
    names = d.mlookup.sort_values("model_idx")["model"].to_numpy()
    bench = d.blookup.sort_values("benchmark_idx")["benchmark"].to_numpy()
    results, pointwise = {}, {}
    for name, chains in groups.items():
        r = [per_chain[c] for c in chains]
        e_int = np.mean([x["e_int"] for x in r], axis=0)
        k_int = np.max([x["k_int"] for x in r], axis=0)
        e_cond = np.mean([x["e_cond"] for x in r], axis=0)
        k_cond = np.max([x["k_cond"] for x in r], axis=0)
        pd.DataFrame({"model": names[d.model_idx], "benchmark": bench[d.bench_idx],
                      "integrated": r[0]["done"], "elpd_int": e_int, "k_int_max": k_int,
                      "elpd_psis": e_cond, "k_psis_max": k_cond}
                     ).to_csv(out_dir / f"integrated_loo_{name}.csv", index=False)
        pointwise[name] = (e_int, k_int)
        totals = [float(x["e_int"].sum()) for x in r]
        n = len(e_int)
        results[name] = {
            "chains": chains, "draws_per_chain": f["S"],
            "elpd_int": float(e_int.sum()), "se_int": float(np.sqrt(n) * e_int.std()),
            "elpd_int_per_chain": [round(t, 1) for t in totals],
            "mc_se": float(np.std(totals, ddof=1) / np.sqrt(len(totals))) if len(r) > 1 else None,
            "n_k_int_gt_0.7": int((k_int > 0.7).sum()), "n_k_int_gt_1": int((k_int > 1).sum()),
            "elpd_psis": float(e_cond.sum()), "n_k_psis_gt_0.7": int((k_cond > 0.7).sum()),
            "n_not_integrated": int((~r[0]["done"]).sum()),
            "n_draw_obs_fallback": int(sum(x["fallback"] for x in r)),
        }
    names_g = list(groups)
    ref = names_g[0]
    reliable = np.all([pointwise[g][1] < 0.7 for g in names_g], axis=0)
    for g in names_g[1:]:
        diff = pointwise[g][0] - pointwise[ref][0]
        results[g][f"diff_to_{ref}"] = float(diff.sum())
        results[g][f"se_diff_to_{ref}"] = float(np.sqrt(len(diff)) * diff.std())
        dr = diff[reliable]
        results[g][f"diff_to_{ref}_k_lt_0.7"] = float(dr.sum())
        results[g][f"se_diff_to_{ref}_k_lt_0.7"] = float(np.sqrt(len(dr)) * dr.std())
    results["_n_reliable_everywhere"] = int(reliable.sum())
    (out_dir / "integrated_loo.json").write_text(json.dumps(results, indent=2))
    print(pd.DataFrame({k: v for k, v in results.items() if not k.startswith("_")}).T
          .drop(columns=["chains", "elpd_int_per_chain"]).round(1).to_string())
    print(f"observations with k < 0.7 in every group: {reliable.sum()}")


if __name__ == "__main__":
    main()
