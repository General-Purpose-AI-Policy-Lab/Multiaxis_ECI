"""Leave-one-out per posterior mode by mixture importance sampling (Silva & Zanella).

PSIS-LOO (Vehtari, Gelman & Gabry 2017) reweights the full-data posterior draws by 1/p(y_i|theta)
for each observation. On the K=4 flagship 14% of the observations get a Pareto k above 0.7: their
weights have no usable variance and their elpd_i is optimistic, the more so the fewer the draws.
Comparing the posterior modes on PSIS-LOO then mostly compares those unreliable terms.

Silva & Zanella (2024, "Robust leave-one-out cross-validation for high-dimensional Bayesian
models", JASA) sample once from a mixture of the leave-one-out posteriors,

    q(theta) ∝ p(theta | y) * sum_j 1 / p(y_j | theta),

and estimate every observation's leave-one-out predictive density from those draws:

    p(y_i | y_-i) ≈ sum_s 1/L_s  /  sum_s (1/p(y_i|theta_s)) / L_s,   L_s = sum_j 1/p(y_j|theta_s).

Each weight (1/p_is)/L_s is at most 1, so no draw can dominate the way PSIS weights do.

That mixture weights the leave-one-out posterior of j by 1/p(y_j | y_-j), and our observation
densities span many orders of magnitude (a Beta density is large on a well-predicted score, tiny
on an outlier): on the flagship the few worst-predicted observations took the whole mixture and
every other observation kept a median of 15 effective draws out of 4,000 (2026-10-06). By default
each component is therefore weighted by a first estimate of its leave-one-out density,
c_j ≈ p(y_j | y_-j) (the PSIS estimate of the starting chain):

    q(theta) ∝ p(theta | y) * sum_j c_j / p(y_j | theta),   L_s = sum_j c_j / p(y_j | theta_s),

with the same estimator. This extension is ours, not the paper's: any c_j > 0 keeps the estimator
consistent (the mixture still covers every leave-one-out posterior) and only changes its variance;
on a grid model with outliers of growing size it raised the median effective sample size from
140 to 3,988 of 4,000 draws and cut the mean error tenfold. `--plain` runs the paper's mixture.

The mixture is the flagship's own model plus one `pm.Potential` (log sum_j c_j/p(y_j|theta)),
rebuilt from the trace's `mirt_spec`, and its chains start from a draw of one chain of the mode
under study (no jitter), so each run estimates the leave-one-out performance of that mode.

    python 4_diagnostics/robust_loo.py --trace TRACE --mode A:5 [--chains 4 --draws 1000]
    python 4_diagnostics/robust_loo.py --trace TRACE --compare A B C

Outputs in the data generation's `diagnostics/`: `robust_loo_<label>.csv` (elpd_i, the effective
sample size of its weights, the PSIS elpd_i and k of the original chains for comparison),
`robust_loo_<label>.json` (totals, the total per mixture chain as a Monte Carlo check, which mode
the mixture chains stayed in) and `robust_loo_<label>_loglik.npz` (the mixture draws'
log-likelihood, float32, to recompute anything above).
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "2_model"))

from multiaxis_eci import config  # noqa: E402


def mixture_loo(loglik: np.ndarray, log_c: np.ndarray | None = None
                ) -> tuple[np.ndarray, np.ndarray]:
    """Silva-Zanella elpd_i and the effective sample size of each observation's weights.

    `loglik` is (draws, observations): log p(y_i | theta_s) at draws from the mixture q, whose
    components were weighted by `log_c` (None: the paper's unweighted mixture).
    """
    log_c = np.zeros(loglik.shape[1]) if log_c is None else np.asarray(log_c)
    log_l = logsumexp(log_c - loglik, axis=1)                # log L_s
    log_w = -loglik - log_l[:, None]                         # log of (1/p_is) / L_s
    elpd = logsumexp(-log_l) - logsumexp(log_w, axis=0)
    w = np.exp(log_w - log_w.max(axis=0))
    ess = w.sum(axis=0) ** 2 / (w ** 2).sum(axis=0)
    return elpd, ess


def subset_observations(data, keep: np.ndarray):
    """`data` restricted to the observations where `keep` is True, every model and benchmark
    kept, so a model built on it has the same parameters as the full fit's."""
    import dataclasses
    idx = np.flatnonzero(keep)
    return dataclasses.replace(
        data, scores=data.scores[idx], zero_score_mask=data.zero_score_mask[idx],
        model_idx=data.model_idx[idx], bench_idx=data.bench_idx[idx], n_obs=len(idx),
        n_obs_per_model=np.bincount(data.model_idx[idx], minlength=data.n_models),
        n_eff=None if data.n_eff is None else data.n_eff[idx])


def build_model(trace_path: Path, keep: np.ndarray | None = None):
    """The fit's PyMC model, rebuilt from the trace's spec as 3_fit/fit.py builds it, on every
    observation or on those where `keep` is True. Returns (model, data, post): `data` is the
    FULL data scope either way, `post` the trace's posterior."""
    from multiaxis_eci.analysis.fitspec import FitSpec, sampled_human_order
    from multiaxis_eci.data import (
        load_benchmark_ceilings,
        load_boundary_eps,
        load_observed_best,
        release_time_covariate,
    )
    from multiaxis_eci.lineage import trace_lineage_structure
    from multiaxis_eci.models.mirt import build_mirt_model

    post = xr.open_dataset(trace_path, group="posterior")
    spec = FitSpec.from_trace(az.InferenceData(posterior=post), trace_path)
    data, floor_c, _ = spec.load_data(az.InferenceData(posterior=post))
    lineage = (trace_lineage_structure(data.mlookup, trace_path, post) if spec.lineage_prior
               else None)
    time_t = release_time_covariate(data.mlookup, lineage) if spec.time_prior else None
    # Per-observation inputs are read on the full data, then cut with it.
    binomial_n = spec.binomial_n(data)
    censor_eps = load_boundary_eps(data) if spec.censor_bounds else None
    fit_data = data
    if keep is not None:
        fit_data = subset_observations(data, keep)
        binomial_n = None if binomial_n is None else np.asarray(binomial_n)[keep]
        censor_eps = None if censor_eps is None else np.asarray(censor_eps)[keep]
    model = build_mirt_model(
        fit_data, spec.K, loading_prior=spec.loading_prior, human_order=sampled_human_order(post, spec),
        human_delta_sd=spec.human_delta_sd, human_root_sd=spec.human_root_sd,
        lineage=lineage, lineage_bm=spec.lineage_bm, effort_means=spec.effort_means,
        floor_c=floor_c, ceiling_noise=spec.ceiling_noise, known_se=spec.known_se,
        known_ceiling=load_benchmark_ceilings(fit_data) if spec.ceiling_noise else None,
        ceiling_lower=load_observed_best(fit_data) if spec.ceiling_noise else None,
        censor_eps=censor_eps, binomial_n=binomial_n, pooled_noise=spec.pooled_noise,
        sigma_b_floor=spec.sigma_b_floor,
        shared_base_zsn=not spec.private_bases, time_t=time_t, theta_t_cells=spec.theta_t,
        theta_skew=spec.theta_skew,
        theta_pos=spec.theta_pos, link=spec.link)
    # The builder's initial values only steer a cold start, and PyMC cannot compute a
    # log-likelihood on a model that carries them (as in fit.py); the runs here start from a
    # posterior draw instead.
    model.rvs_to_initial_values = dict.fromkeys(model.rvs_to_initial_values)
    return model, data, post


def check_rebuild(model, post, trace_path: Path, chain: int, n_draws: int = 5) -> float:
    """Largest gap between the rebuilt model's log-likelihood and the trace's own, on a few
    draws of `chain`: the rebuild is only trusted when it reproduces the fitted likelihood."""
    import pymc as pm
    free = [rv.name for rv in model.free_RVs]
    missing = [v for v in free if v not in post]
    if missing:
        raise SystemExit(f"trace lacks free variables {missing}")
    sub = post[free].isel(chain=[chain], draw=slice(0, n_draws))
    ll = pm.compute_log_likelihood(az.InferenceData(posterior=sub), model=model,
                                   var_names=["obs"], progressbar=False, extend_inferencedata=False)
    ref = xr.open_dataset(trace_path, group="log_likelihood")["obs"].isel(
        chain=[chain], draw=slice(0, n_draws)).values
    return float(np.abs(ll["obs"].values - ref).max())


def mean_loadings(post) -> np.ndarray:
    return post["A"].mean("draw").values                     # (chain, bench, K)


def best_match(L: np.ndarray, ref: np.ndarray) -> list[float]:
    """Per-axis correlation of loadings L with ref, under the best axis permutation."""
    K = ref.shape[1]
    perm = max(itertools.permutations(range(K)),
               key=lambda p: sum(np.corrcoef(ref[:, k], L[:, p[k]])[0, 1] for k in range(K)))
    return [float(np.corrcoef(ref[:, k], L[:, perm[k]])[0, 1]) for k in range(K)]


def run_mode(trace_path: Path, label: str, chain: int, chains: int, draws: int, tune: int,
             seed: int, out_dir: Path, balanced: bool = True) -> None:
    import nutpie
    import pymc as pm
    import pytensor.tensor as pt

    from multiaxis_eci.models.mirt import align_effort_sign

    model, data, post = build_model(trace_path)
    gap = check_rebuild(model, post, trace_path, chain)
    print(f"rebuilt model reproduces the trace's log-likelihood to {gap:.2e}", flush=True)
    if gap > 1e-6:
        raise SystemExit("the rebuilt model does not match the fit; stopping")

    # The starting chain's PSIS-LOO: reported beside the result, and the component weights
    # log c_j of the balanced mixture.
    ref_ll = xr.open_dataset(trace_path, group="log_likelihood")["obs"].isel(chain=chain).values
    lw, k = az.psislw(-ref_ll.T)
    psis = logsumexp(lw + ref_ll.T, axis=1)
    log_c = psis if balanced else np.zeros_like(psis)

    # Start every chain at the last draw of the chosen chain, no jitter, so the run studies
    # that mode; the mixture's heavier tails are what it then explores around it.
    free = [rv.name for rv in model.free_RVs]
    start = {v: post[v].isel(chain=chain, draw=-1).values for v in free}
    with model:
        ll = model.logp(vars=[model["obs"]], sum=False)[0]
        pm.Potential("silva_zanella_mixture", pt.logsumexp(pt.as_tensor(log_c) - ll))
    compiled = nutpie.compile_pymc_model(model, initial_points=start, jitter_rvs=set())
    t0 = time.time()
    idata = nutpie.sample(compiled, draws=draws, tune=tune, chains=chains, seed=seed,
                          target_accept=0.95, low_rank_modified_mass_matrix=True,
                          save_warmup=False, progress_bar=False)
    print(f"mixture sampled in {(time.time() - t0) / 60:.0f} min", flush=True)
    idata.posterior = idata.posterior.drop_vars(
        [v for v in idata.posterior.data_vars if v.endswith("__")])
    align_effort_sign(idata.posterior)
    pm.compute_log_likelihood(idata, model=model, var_names=["obs"], progressbar=False)
    ll_cd = idata.log_likelihood["obs"].values               # (chain, draw, obs)
    loglik = ll_cd.reshape(-1, ll_cd.shape[-1])

    elpd, ess = mixture_loo(loglik, log_c)
    # Monte Carlo check: the same estimate from each mixture chain alone.
    per_chain = [float(mixture_loo(ll_cd[c], log_c)[0].sum()) for c in range(ll_cd.shape[0])]

    ref_A = mean_loadings(post)[chain]
    stay = [min(best_match(L, ref_A)) for L in mean_loadings(idata.posterior)]
    div = idata.sample_stats["diverging"].sum("chain").sum().item() \
        if "diverging" in idata.sample_stats else -1
    bench = data.blookup.sort_values("benchmark_idx")["benchmark"].values[data.bench_idx]
    model_n = data.mlookup.sort_values("model_idx")["model"].values[data.model_idx]
    df = pd.DataFrame({"model": model_n, "benchmark": bench, "elpd_sz": elpd, "ess_sz": ess,
                       "elpd_psis_chain": psis, "pareto_k_chain": k})
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"robust_loo_{label}.csv", index=False)
    np.savez_compressed(out_dir / f"robust_loo_{label}_loglik.npz",
                        loglik=ll_cd.astype(np.float32), log_c=log_c)
    n = len(elpd)
    summary = {
        "mode": label, "start_chain": chain, "chains": chains, "draws": draws, "tune": tune,
        "mixture": "balanced (c_j = PSIS of the start chain)" if balanced else "plain",
        "elpd_sz": float(elpd.sum()), "se_sz": float(np.sqrt(n) * elpd.std()),
        "elpd_sz_per_mixture_chain": [round(x, 1) for x in per_chain],
        "elpd_psis_start_chain": float(psis.sum()),
        "n_pareto_k_gt_0.7_start_chain": int((k > 0.7).sum()),
        "ess_sz_min": float(ess.min()), "ess_sz_median": float(np.median(ess)),
        "n_ess_sz_lt_50": int((ess < 50).sum()),
        "mixture_divergences": int(div),
        "mixture_chain_min_axis_corr_to_start": [round(x, 3) for x in stay],
    }
    (out_dir / f"robust_loo_{label}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def compare(out_dir: Path, labels: list[str]) -> None:
    """Paired elpd differences between modes, on every observation and by benchmark."""
    dfs = {lab: pd.read_csv(out_dir / f"robust_loo_{lab}.csv") for lab in labels}
    ref = labels[0]
    n = len(dfs[ref])
    print("mode  elpd_sz    se   | diff to", ref, "  se_diff | PSIS elpd (start chain)")
    for lab in labels:
        d = dfs[lab].elpd_sz - dfs[ref].elpd_sz
        s = json.loads((out_dir / f"robust_loo_{lab}.json").read_text())
        print(f"{lab:4s} {s['elpd_sz']:9.1f} {s['se_sz']:6.1f} | {d.sum():+8.1f} "
              f"{np.sqrt(n) * d.std():8.1f} | {s['elpd_psis_start_chain']:9.1f}")
    for lab in labels[1:]:
        d = dfs[lab].assign(diff=dfs[lab].elpd_sz - dfs[ref].elpd_sz)
        g = d.groupby("benchmark")["diff"].sum().sort_values()
        print(f"\n{lab} - {ref} by benchmark: {ref} better on", g.head(6).round(1).to_dict())
        print(f"  {lab} better on", g.tail(6)[::-1].round(1).to_dict())


def _flagship_trace() -> Path:
    from multiaxis_eci.analysis.fitspec import FLAGSHIP_TRACE
    return FLAGSHIP_TRACE


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trace", type=Path, default=_flagship_trace())
    ap.add_argument("--mode", action="append", default=[], metavar="LABEL:CHAIN",
                    help="a mode to score, started from the last draw of CHAIN")
    ap.add_argument("--compare", nargs="+", metavar="LABEL",
                    help="compare modes already scored (the first is the reference)")
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--draws", type=int, default=1000)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--plain", action="store_true",
                    help="the paper's unweighted mixture instead of the balanced one")
    args = ap.parse_args()
    out_dir = config.RESULTS_DIR / "diagnostics"
    for item in args.mode:
        label, chain = item.split(":")
        run_mode(args.trace, label, int(chain), args.chains, args.draws, args.tune, args.seed,
                 out_dir, balanced=not args.plain)
    if args.compare:
        compare(out_dir, args.compare)


if __name__ == "__main__":
    main()
