"""Multi-start MAP search on the K=4 flagship model with the sigma_b floor.

The 2026-10-12 sampler study found noise-collapse spikes: optima and trapped NUTS chains where
many benchmarks' sigma_b fell to <= 1e-3 and the log posterior sat 3,000 to 5,000 nats above the
sampled modes. The floor sigma_b >= 0.005 (config.SIGMA_B_FLOOR) was added afterwards. This
script checks whether optima still collapse onto that floor.

The model is the flagship as 3_fit/fit.py builds it (FitSpec K=4, lineage prior, Brownian steps;
models only, skew-normal theta, estimated ceilings, floors, Beta-binomial, pooled noise are the
spec's defaults). Each start is the jittered initial point a NUTS chain starts from. The
objective is the unconstrained log density NUTS samples (Jacobian included), maximised by
L-BFGS. One row per start: log density, log-likelihood, log prior, and the noise summary.

    python 6_writeups/preprint/step0/map_search/map_search.py --starts 60
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "2_model"))

import numpy as np
import pandas as pd
import pymc as pm
from pymc.initial_point import make_initial_point_fn
from scipy.optimize import minimize

from multiaxis_eci.analysis.fitspec import FitSpec
from multiaxis_eci.data import load_benchmark_ceilings, load_observed_best
from multiaxis_eci.lineage import build_lineage_structure
from multiaxis_eci.models.mirt import build_mirt_model

OUT = Path(__file__).resolve().parent


def build():
    spec = FitSpec(K=4, lineage_prior=True, lineage_bm=True)
    assert spec.tag == "_lineageprior_lineagebm", spec.tag
    data, floor_c, _ = spec.load_data()
    model = build_mirt_model(
        data, spec.K, loading_prior=spec.loading_prior,
        human_delta_sd=spec.human_delta_sd, human_root_sd=spec.human_root_sd,
        lineage=build_lineage_structure(data.mlookup), lineage_bm=spec.lineage_bm,
        effort_means=spec.effort_means, floor_c=floor_c,
        ceiling_noise=spec.ceiling_noise, known_se=spec.known_se,
        known_ceiling=load_benchmark_ceilings(data), ceiling_lower=load_observed_best(data),
        binomial_n=spec.binomial_n(data), pooled_noise=spec.pooled_noise,
        sigma_b_floor=spec.sigma_b_floor, shared_base_zsn=not spec.private_bases,
        theta_skew=spec.theta_skew, link=spec.link)
    return spec, data, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starts", type=int, default=60)
    ap.add_argument("--seed", type=int, default=20261102)
    ap.add_argument("--maxiter", type=int, default=20000)
    ap.add_argument("--out", default="map_starts.csv")
    args = ap.parse_args()

    spec, data, model = build()
    print(f"{data.n_obs} obs / {data.n_models} models / {data.n_benchmarks} benchmarks; "
          f"sigma_b floor {spec.sigma_b_floor}", flush=True)
    f = model.logp_dlogp_function(ravel_inputs=True)       # NUTS's own density and gradient
    f.set_extra_values({})
    names = [v.name for v in model.value_vars]
    loglik = model.compile_logp(vars=model.observed_RVs, jacobian=True, sum=True)
    # RVs replaced by their value variables: compiled as RVs, they would be fresh prior draws
    noise = model.compile_fn(
        model.replace_rvs_by_values([model["sigma_b"], model["sigma_b_mu"], model["sigma_b_tau"]]),
        inputs=model.value_vars, on_unused_input="ignore")
    ip_fn = make_initial_point_fn(model=model, jitter_rvs=set(model.free_RVs),
                                  return_transformed=True)
    shapes = [np.shape(v) for v in ip_fn(0).values()]
    keys = list(ip_fn(0).keys())
    assert keys == names, (keys[:5], names[:5])

    def unravel(x):
        out, i = {}, 0
        for k, s in zip(keys, shapes):
            n = int(np.prod(s))
            out[k] = x[i:i + n].reshape(s)
            i += n
        return out

    def neg(x):
        lp, g = f(x)
        if not np.isfinite(lp):
            return 1e12, np.zeros_like(x)
        return -lp, -g

    floor = spec.sigma_b_floor
    bench = data.blookup.sort_values("benchmark_idx")["benchmark"].to_numpy()
    rows, sig_rows, optima = [], [], []
    rng = np.random.default_rng(args.seed)
    for s in range(args.starts):
        # a jittered point can fall outside the model's support (density 0): redraw, as PyMC does
        x0 = None
        while x0 is None or not np.isfinite(f(x0)[0]):
            x0 = np.concatenate([np.ravel(v) for v in ip_fn(int(rng.integers(2**31))).values()])
        t0 = time.time()
        r = minimize(neg, x0, jac=True, method="L-BFGS-B",
                     options={"maxiter": args.maxiter, "maxfun": 2 * args.maxiter})
        pt = unravel(r.x)
        sig, mu_s, tau_s = noise(pt)
        ll = float(loglik(pt))
        at_floor = sig < floor * 1.01
        rows.append({"start": s, "logp": -r.fun, "loglik": ll, "logprior": -r.fun - ll,
                     "min_sigma_b": sig.min(), "n_sigma_b_at_floor": int(at_floor.sum()),
                     "n_sigma_b_below_0.01": int((sig < 0.01).sum()),
                     "sigma_b_mu": float(mu_s), "sigma_b_tau": float(tau_s),
                     "at_floor": ";".join(bench[at_floor]), "iters": r.nit,
                     "converged": bool(r.success), "message": str(r.message)[:60],
                     "seconds": round(time.time() - t0, 1)})
        sig_rows.append(sig)
        optima.append(r.x)
        # the optimum in NUTS's unconstrained coordinates, so a chain can start from it
        np.savez_compressed(OUT / args.out.replace(".csv", "_optima.npz"), x=np.array(optima),
                            keys=np.array(keys), sizes=np.array([np.prod(s) for s in shapes]))
        print({k: rows[-1][k] for k in ("start", "logp", "loglik", "min_sigma_b",
                                        "n_sigma_b_at_floor", "iters", "seconds")}, flush=True)
        pd.DataFrame(rows).to_csv(OUT / args.out, index=False)
        pd.DataFrame(np.array(sig_rows), columns=bench).to_csv(
            OUT / args.out.replace(".csv", "_sigma_b.csv"), index_label="start")


if __name__ == "__main__":
    main()
