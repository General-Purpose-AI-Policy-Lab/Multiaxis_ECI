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
For each draw, the block's conditional posterior (prior times every score of the block) is
approximated by a Laplace fit; removing score i gives its cavity Gaussian in closed form, and
since score i sees the block only through eta_i = A_i . (psi + offset) - D_i, the predictive
density p(y_i | y_-i, phi_s) is a one-dimensional Gauss-Hermite integral over eta_i. Nothing
is refitted, so every posterior mode of a trace is scored from its own chains.

    python 4_diagnostics/integrated_loo.py [--trace TRACE] [--groups A2:2,4,5 A1:0,1 B:7,8,9]

Every chain is scored alone and a group of chains (a mode, by default those of the modes file)
by the average of its chains: PSIS is optimistic with few draws where k is high, so groups are
compared at equal draws. Output in the data generation's `diagnostics/`:
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
from scipy.special import expit, gammaln, logsumexp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "2_model"))

from multiaxis_eci import config  # noqa: E402

GH_X, GH_W = np.polynomial.hermite_e.hermegauss(40)   # probabilists' Hermite: weight exp(-x^2/2)
LOG_GH_W = np.log(GH_W / np.sqrt(2 * np.pi))


# ── likelihood, in numpy, as models/mirt.py writes it ───────────────────────────────────────

def obs_loglik(eta, c, phi, n_items, k, y):
    """log p(y | eta) of the fixed-c 3PL Beta-binomial (n_items > 0) / Beta (n_items == 0)
    likelihood; every argument broadcasts."""
    mu = c + (1.0 - c) * expit(eta)
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


# ── one latent block, every draw at once ─────────────────────────────────────────────────────

def block_predictive(A_o, off, D_o, args, m0, p0, x0, iters: int = 10):
    """log p(y_i | y_-i, phi_s) for every score i of one latent block and every draw s.

    A_o (S, J, K) loadings of the block's J scores, off (S, J, K) the per-score offset added to
    the block value (a variant's offset from its release level; 0 outside the chains), D_o (S, J)
    difficulties, args the per-score likelihood inputs broadcast to (S, J), m0 / p0 (S, K) the
    block's Gaussian prior mean and precision (diagonal), x0 (S, K) a starting value (the draw's
    own block value). Returns (S, J).
    """
    x = x0.copy()
    for _ in range(iters):
        eta = np.einsum("sjk,sjk->sj", A_o, x[:, None, :] + off) - D_o
        _, d1, d2 = eta_derivatives(eta, args)
        w = np.maximum(-d2, 1e-8)                                 # curvature, kept positive
        grad = np.einsum("sj,sjk->sk", d1, A_o) - p0 * (x - m0)
        P = np.einsum("sj,sjk,sjl->skl", w, A_o, A_o) + p0[:, :, None] * np.eye(x.shape[1])
        step = np.clip(np.linalg.solve(P, grad[..., None])[..., 0], -1.0, 1.0)   # damped
        x = x + step
        if np.abs(step).max() < 1e-6:
            break
    eta = np.einsum("sjk,sjk->sj", A_o, x[:, None, :] + off) - D_o
    _, d1, d2 = eta_derivatives(eta, args)
    w = np.maximum(-d2, 1e-8)
    P = np.einsum("sj,sjk,sjl->skl", w, A_o, A_o) + p0[:, :, None] * np.eye(x.shape[1])
    Sigma = np.linalg.inv(P)
    v_full = np.einsum("sjk,skl,sjl->sj", A_o, Sigma, A_o)        # A_i' Sigma A_i
    # Cavity of score i (Sherman-Morrison): drop its curvature and its pull on the mode.
    v_cav = v_full / (1.0 - w * v_full)
    m_cav = eta - d1 * v_cav
    nodes = m_cav[..., None] + np.sqrt(v_cav)[..., None] * GH_X       # (S, J, Q)
    ll = obs_loglik(nodes, *[a[..., None] for a in args])
    return logsumexp(ll + LOG_GH_W, axis=-1)


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


def load_fit(trace_path: Path, chains: list[int]):
    from multiaxis_eci.analysis.fitspec import FitSpec
    from multiaxis_eci.lineage import build_lineage_structure
    from multiaxis_eci.models.mirt import _human_structure

    post_all = xr.open_dataset(trace_path, group="posterior")
    spec = FitSpec.from_trace(az.InferenceData(posterior=post_all), trace_path)
    if spec.time_prior or spec.theta_pos or spec.theta_t or spec.link != "linear" \
            or spec.ceiling_noise or spec.known_se or spec.censor_bounds or not spec.lineage_bm:
        raise SystemExit("integrated_loo covers the flagship's specification only")
    data, floor_c, _ = spec.load_data(az.InferenceData(posterior=post_all))
    lin = build_lineage_structure(data.mlookup)
    names = data.mlookup.sort_values("model_idx")["model"].tolist()
    hs = _human_structure(spec.human_order, names)
    human_rows = np.asarray(hs[0]) if hs is not None else np.array([], dtype=int)
    p = post_all.isel(chain=chains)
    S = p.sizes["chain"] * p.sizes["draw"]

    def flat(v):
        return p[v].values.reshape(S, *p[v].shape[2:])

    A, theta, D, phi = flat("A"), flat("theta"), flat("D"), flat("phi_b")
    base, z = flat("lin_base"), flat("lin_delta_z")
    drift, spread = flat("lin_drift"), flat("lin_spread")
    dt = lin.delta_dt
    delta = drift[:, None, :] * dt[None, :, None] + spread[:, None, :] * np.sqrt(dt)[None, :, None] * z
    psi = np.einsum("nc,sck->snk", lin.B, base) + np.einsum("nd,sdk->snk", lin.C, delta)

    binomial_n = spec.binomial_n(data)
    n_items = np.where(np.isfinite(binomial_n) & (binomial_n > 0), binomial_n, 0.0) \
        if binomial_n is not None else np.zeros(data.n_obs)
    k = np.clip(data.scores, 0.0, 1.0) * n_items
    y = np.clip(data.scores, config.CONTINUOUS_EPS, 1.0 - config.CONTINUOUS_EPS)
    c = floor_c[data.bench_idx] if floor_c is not None else np.zeros(data.n_obs)
    return dict(data=data, lin=lin, human_rows=human_rows, A=A, theta=theta, D=D, phi=phi,
                psi=psi, drift=drift, spread=spread, n_items=n_items, k=k, y=y, c=c, S=S)


def conditional_loglik(f) -> np.ndarray:
    """log p(y_n | theta_s), (S, n_obs), recomputed in numpy from the posterior."""
    d = f["data"]
    eta = np.einsum("sok,sok->so", f["A"][:, d.bench_idx], f["theta"][:, d.model_idx]) \
        - f["D"][:, d.bench_idx]
    return obs_loglik(eta, f["c"], f["phi"][:, d.bench_idx], f["n_items"], f["k"], f["y"])


def integrated_loglik(f) -> tuple[np.ndarray, np.ndarray]:
    """log p(y_n | y_-n, phi_s), (S, n_obs), and a mask of the observations it integrates
    (False: human tiers, left at the conditional log-likelihood)."""
    d, lin, S = f["data"], f["lin"], f["S"]
    K = f["A"].shape[2]
    out = conditional_loglik(f)
    done = np.zeros(d.n_obs, dtype=bool)
    parent, d_in, children = lineage_graph(lin)
    spread2 = f["spread"] ** 2                                        # (S, K)

    def run(obs, block_value, off, m0, p0):
        b = d.bench_idx[obs]
        args = (f["c"][obs][None, :], f["phi"][:, b], f["n_items"][obs][None, :],
                f["k"][obs][None, :], f["y"][obs][None, :])
        args = tuple(np.broadcast_to(a, (S, len(obs))) for a in args)
        with np.errstate(all="ignore"):
            res = block_predictive(f["A"][:, b], off, f["D"][:, b], args, m0, p0, block_value)
        bad = ~np.isfinite(res)
        # A rare draw whose Newton fit runs off (extreme eta) keeps its conditional value.
        res[bad] = out[:, obs][bad]
        f["n_fallback"] = f.get("n_fallback", 0) + int(bad.sum())
        out[:, obs] = res
        done[obs] = True

    # Release levels: every score of every model of the node.
    rows_of_node = [lin.row_idx[lin.node_idx == n] for n in range(lin.n_nodes)]
    for n in range(lin.n_nodes):
        obs = np.flatnonzero(np.isin(d.model_idx, rows_of_node[n]))
        if obs.size == 0:
            continue
        p0 = np.zeros((S, K))
        num = np.zeros((S, K))
        if parent[n] < 0:
            p0 += 1.0                                                 # founder: ZSN entry
        else:
            v = spread2 * lin.delta_dt[d_in[n]]
            p0 += 1.0 / v
            num += (f["psi"][:, parent[n]] + f["drift"] * lin.delta_dt[d_in[n]]) / v
        for ch in children[n]:
            v = spread2 * lin.delta_dt[d_in[ch]]
            p0 += 1.0 / v
            num += (f["psi"][:, ch] - f["drift"] * lin.delta_dt[d_in[ch]]) / v
        m0 = num / p0
        off = f["theta"][:, d.model_idx[obs]] - f["psi"][:, n][:, None, :]
        run(obs, f["psi"][:, n], off, m0, p0)

    # Models outside the chains and the human tiers: their own theta, prior N(0, I).
    claimed = np.concatenate([f["human_rows"], lin.row_idx])
    for m in np.setdiff1d(np.arange(d.n_models), claimed):
        obs = np.flatnonzero(d.model_idx == m)
        if obs.size == 0:
            continue
        run(obs, f["theta"][:, m], np.zeros((S, obs.size, K)), np.zeros((S, K)),
            np.ones((S, K)))
    return out, done


def psis_elpd(loglik: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """PSIS-LOO elpd_i and Pareto k from (S, n) log densities used as 1/p weights."""
    lw, k = az.psislw(-loglik.T)
    return logsumexp(lw + loglik.T, axis=1), k


def score_chain(trace_path: Path, chain: int, stored) -> dict:
    """Integrated and conditional PSIS-LOO of one chain (pointwise elpd and k)."""
    f = load_fit(trace_path, [chain])
    cond = conditional_loglik(f)
    gap = float(np.abs(cond - stored.isel(chain=[chain]).values.reshape(f["S"], -1)).max())
    if gap > 1e-6:
        raise SystemExit(f"chain {chain}: the numpy likelihood does not reproduce the fit ({gap})")
    integ, done = integrated_loglik(f)
    e_int, k_int = psis_elpd(integ)
    e_cond, k_cond = psis_elpd(cond)
    return dict(f=f, e_int=e_int, k_int=k_int, e_cond=e_cond, k_cond=k_cond, done=done,
                fallback=int(f.get("n_fallback", 0)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trace", type=Path, default=config.RESULTS_DIR
                    / "mirt_humanmerge_lineageprior_lineagebm"
                    / "trace_mirt_k4_humanmerge_lineageprior_lineagebm.nc")
    ap.add_argument("--groups", nargs="+", default=None, metavar="NAME:C1,C2",
                    help="chain groups to compare (default: the modes of the modes file); "
                         "the first is the reference")
    args = ap.parse_args()
    if args.groups:
        groups = {g.split(":")[0]: [int(c) for c in g.split(":")[1].split(",")]
                  for g in args.groups}
    else:
        modes_file = args.trace.parent / f"mirt_modes_{args.trace.stem}.json"
        groups = {m["label"]: m["chains"] for m in json.loads(modes_file.read_text())["modes"]}
    out_dir = config.RESULTS_DIR / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    stored = xr.open_dataset(args.trace, group="log_likelihood")["obs"]

    # One chain at a time: PSIS is optimistic with few draws where k is high, so groups of
    # different sizes are compared on the AVERAGE of their chains' estimates, at equal draws
    # (on the flagship, pooling mode A's five chains moved its total by -65, almost all from
    # the observations still above k = 0.5).
    per_chain = {}
    for c in sorted({c for ch in groups.values() for c in ch}):
        per_chain[c] = score_chain(args.trace, c, stored)
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
