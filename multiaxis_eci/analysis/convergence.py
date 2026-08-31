"""Convergence on identified quantities (eta, D, sigma_b) for each
MIRT family — raw r-hat on A/theta is permutation-inflated and ignored."""
from __future__ import annotations

import arviz as az
import numpy as np
import xarray as xr

from multiaxis_eci.data import ECIData

def convergence(idata) -> dict:
    """Global max r-hat / min ESS / divergences (nan-safe for masked entries).

    The one copy every fit driver imports — it drifted when each driver
    carried its own (2_fit.py's lacked the nan-safety, so pt1's constant
    tau_A printed NaN as the max r-hat)."""
    rh = az.rhat(idata)
    ess = az.ess(idata)
    max_rhat = float(np.nanmax([np.nanmax(v.values) for v in rh.data_vars.values()]))
    min_ess = float(np.nanmin([np.nanmin(v.values) for v in ess.data_vars.values()]))
    div = int(idata.sample_stats["diverging"].sum()) if "diverging" in idata.sample_stats else -1
    n_draws = int(idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"])
    return {"max_rhat": max_rhat, "min_ess": min_ess, "divergences": div, "n_draws": n_draws}


def _rhat_subset_idx(data: ECIData, n_obs_sample: int, seed: int):
    """Random (bench_idx, model_idx) subset for a prediction-level r-hat. Shared
    by the compensatory and non-comp reporters so both sample the same way."""
    rng = np.random.default_rng(seed)
    n = min(n_obs_sample, data.n_obs)
    sel = rng.choice(data.n_obs, size=n, replace=False)
    return data.bench_idx[sel], data.model_idx[sel]


def _pred_rhat_summary(pred, name: str) -> dict:
    """max / mean / frac>1.01 of r-hat for a (chain, draw, obs) predictor DataArray."""
    r = az.rhat(pred.to_dataset(name=name))[name].values
    return {f"{name}_max_rhat":  float(np.nanmax(r)),
            f"{name}_mean_rhat": float(np.nanmean(r)),
            f"{name}_frac_gt_1.01": float(np.nanmean(r > 1.01))}


def _identified_eta(post, data: ECIData, n_obs_sample: int, seed: int):
    """eta over a random observation subset, in the scale the likelihood used.

    theta_pos when the fit has it (the likelihood reads softplus(theta) there).
    On a loglog fit ("alpha" in the posterior) eta is alpha*log(A . theta_pos):
    difficulty sits inside A's row scale, so there is no D term.
    """
    th = post["theta_pos"] if "theta_pos" in post else post["theta"]
    bi, mi = _rhat_subset_idx(data, n_obs_sample, seed)
    A_sel = post["A"].isel(bench=("obs", bi))
    th_sel = th.isel(model=("obs", mi))
    if "alpha" in post:
        return post["alpha"].isel(bench=("obs", bi)) * np.log(
            (A_sel * th_sel).sum("latent"))
    return (A_sel * th_sel).sum("latent") - post["D"].isel(bench=("obs", bi))


def mirt_identified_ess(trace, data: ECIData, n_obs_sample: int = 400,
                        seed: int = 0) -> dict:
    """Bulk ESS on the identified eta subset — the ESS companion to
    `mirt_identified_rhat`, same observation sample, same invariance argument
    (raw A/theta ESS is deflated by label switching that eta cannot see)."""
    eta = _identified_eta(trace.posterior, data, n_obs_sample, seed)
    ess = az.ess(eta.to_dataset(name="eta"))["eta"].values
    return {"eta_ess_min": float(np.nanmin(ess)),
            "eta_ess_med": float(np.nanmedian(ess))}


def _param_max_rhats(post, names) -> dict:
    """max r-hat for each named posterior variable that is present."""
    out = {}
    for v in names:
        if v in post:
            out[f"{v}_max_rhat"] = float(np.nanmax(az.rhat(post[v].to_dataset(name=v))[v].values))
    return out


def mirt_identified_rhat(trace, data: ECIData, n_obs_sample: int = 400,
                         seed: int = 0) -> dict:
    """Convergence on IDENTIFIED quantities — the honest MIRT verdict.

    Raw r-hat on A/theta/tau_A is meaningless: those carry the axis-permutation
    label-switching symmetry, so chains that found the same fit disagree on
    naming and r-hat explodes (see the section header). The fix is to score
    convergence on quantities invariant to relabelling:
      * eta = sum_k A[b,k]*theta[m,k] - D[b]  (the linear predictor) over a
        random subset of observations — invariant to permutation/rotation/sign,
      * D, sigma_b, tau_CD — per-benchmark / global, no axis label at all.

    On a theta_pos fit the likelihood reads softplus(theta), so eta is built
    from `theta_pos` there — scoring the raw scale would judge convergence on a
    quantity the model never evaluated. On a loglog fit ("alpha" in the
    posterior) eta is alpha*log(A . theta_pos) instead, with no D term.

    Returns max/mean r-hat for eta and max r-hat for each identified parameter.
    """
    post = trace.posterior
    out = _pred_rhat_summary(_identified_eta(post, data, n_obs_sample, seed), "eta")
    out.update(_param_max_rhats(post, ("D", "sigma_b", "tau_CD", "alpha")))
    return out


def mirt_identified_rhat_interaction(trace, data: ECIData, n_obs_sample: int = 400,
                                     seed: int = 0) -> dict:
    """Convergence for the semi-compensatory MIRT (models/mirt_interaction.py).

    Same identified headline as `mirt_identified_rhat`, but eta includes the
    pairwise interaction term, which reads the positive ability scale:
      eta = sum_k A theta + sum_{j<k} gamma tp_j tp_k - D,  tp = softplus(theta).
    Both scales are read off the trace so eta is the one the likelihood saw.
    PLT founders fix the rotation, so A/theta r-hat is meaningful here; the
    prediction-level eta and D/sigma_b/tau_CD are still the honest verdict.
    """
    import itertools
    post = trace.posterior
    A, th, tp, D, gamma = (post["A"], post["theta"], post["theta_pos"],
                           post["D"], post["gamma"])
    K = int(post.sizes["latent"])
    pairs = list(itertools.combinations(range(K), 2))
    bi, mi = _rhat_subset_idx(data, n_obs_sample, seed)

    thm = th.isel(model=("obs", mi))                     # (chain,draw,obs,latent)
    tpm = tp.isel(model=("obs", mi))
    eta = (A.isel(bench=("obs", bi)) * thm).sum("latent") - D.isel(bench=("obs", bi))
    gb = gamma.isel(bench=("obs", bi))                   # (chain,draw,obs,pair)
    for p, (j, k) in enumerate(pairs):
        eta = eta + gb.isel(pair=p) * tpm.isel(latent=j) * tpm.isel(latent=k)
    out = _pred_rhat_summary(eta, "eta")
    out.update(_param_max_rhats(post, ("D", "sigma_b", "tau_CD")))
    return out


def mirt_identified_rhat_nc(trace, data: ECIData, n_obs_sample: int = 400,
                            seed: int = 0) -> dict:
    """Convergence for the NON-COMPENSATORY model (models/mirt_nc.py).

    A fixed Q-matrix removes BOTH rotation (the product link is not
    rotation-invariant) AND permutation (each axis column has a distinct loading
    pattern, so the axes are not exchangeable). Unlike the compensatory model the
    raw parameters are therefore already identified — r-hat on theta/c is honest.
    The headline is still the prediction-level log_mu over a random subset of
    observations (link-correct; summarises whether the chains agree on what they
    predict): log_mu = sum_k Q[b,k] * log sigmoid(a_k*theta + c). We also report
    max r-hat for the identified parameters theta, sigma_b, tau_c. Reads A
    only if present (free-discrimination fit).
    """
    post = trace.posterior
    theta, c = post["theta"], post["c"]
    Q = trace.constant_data["Q"].values                  # (B, K)
    bi, mi = _rhat_subset_idx(data, n_obs_sample, seed)

    z = c.isel(bench=("obs", bi))                         # (chain,draw,obs,latent)
    if "A" in post:
        z = z + post["A"].isel(bench=("obs", bi)) * theta.isel(model=("obs", mi))
    else:
        z = z + theta.isel(model=("obs", mi))
    log_sig = -np.logaddexp(0.0, -z.values)              # numpy (chain,draw,obs,latent)
    log_mu = xr.DataArray((log_sig * Q[bi]).sum(axis=-1),  # gate by Q -> (chain,draw,obs)
                          dims=("chain", "draw", "obs"))

    out = _pred_rhat_summary(log_mu, "logmu")
    out.update(_param_max_rhats(post, ("theta", "sigma_b", "tau_c")))
    return out


def nc_difficulty_draws(post, Q: np.ndarray) -> np.ndarray:
    """Per-(benchmark, axis) difficulty draws for the non-compensatory fit.

    Difficulty is b = -c/a, where a = A if the slope was freed
    (free_discrimination) else 1. Off-axis cells (Q=0) are set to NaN — they
    don't enter the likelihood, so any summary must exclude them (per cell:
    filter finite; per benchmark: nanmean over axes). Returns (S, B, K) with
    S = chain*draw. Shared by fits.fit_nc.difficulty_table (per-cell table) and
    analysis.timelines.nc_difficulty_timeline_df (per-benchmark scalar)."""
    B, K = Q.shape
    c = post["c"].values.reshape(-1, B, K)
    mask = Q[None].astype(bool)
    if "A" in post:
        a = post["A"].values.reshape(-1, B, K)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(mask, -c / a, np.nan)
    return np.where(mask, -c, np.nan)


def mirt_identified_rhat_sparse(trace, data: ECIData, n_obs_sample: int = 400,
                                seed: int = 0) -> dict:
    """Convergence for the sparse-gate non-comp (models/mirt_sparse.py).

    Headline: prediction-level log_mu over a random subset of observations,
    log_mu = sum_k g[b,k] * log sigmoid(theta[m,k] + c[b,k]), with the gate g read
    per draw and the slope at 1. Also returns max r-hat for the identified params
    theta, sigma_b, tau_c.
    """
    post = trace.posterior
    theta, c, g = post["theta"], post["c"], post["g"]
    bi, mi = _rhat_subset_idx(data, n_obs_sample, seed)

    z = c.isel(bench=("obs", bi)) + theta.isel(model=("obs", mi))
    log_sig = -np.logaddexp(0.0, -z.values)               # (chain,draw,obs,latent)
    gb = g.isel(bench=("obs", bi)).values                 # (chain,draw,obs,latent)
    log_mu = xr.DataArray((log_sig * gb).sum(axis=-1),
                          dims=("chain", "draw", "obs"))
    out = _pred_rhat_summary(log_mu, "logmu")
    out.update(_param_max_rhats(post, ("theta", "sigma_b", "tau_c")))
    return out

