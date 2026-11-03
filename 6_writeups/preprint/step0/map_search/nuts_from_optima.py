"""Short NUTS chains started from the best MAP optima of map_search.py.

In the 2026-10-12 sampler study, 14 of 30 such chains stayed in noise-collapse spikes
(sigma_b <= 1e-3 on many benchmarks) on the model without the sigma_b floor. This reruns that
check on the floored flagship: one PyMC NUTS chain per optimum, started exactly there (no
jitter). Per chain: median log density and log-likelihood, the noise population (sigma_b_mu,
sigma_b_tau), the smallest sigma_b and how many benchmarks sit within 20% of the floor.

    python 6_writeups/preprint/step0/map_search/nuts_from_optima.py --top 10 --tune 1800 --draws 300
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd
import pymc as pm

from map_search import OUT, build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--tune", type=int, default=1800)
    ap.add_argument("--draws", type=int, default=300)
    ap.add_argument("--seed", type=int, default=191000)
    args = ap.parse_args()

    runs = []
    for csv in sorted(glob.glob(str(OUT / "map_starts_seed[0-9].csv"))):
        df = pd.read_csv(csv)
        opt = np.load(csv.replace(".csv", "_optima.npz"))
        df["x"] = list(opt["x"][: len(df)])
        df["file"] = csv
        runs.append(df)
    allstarts = pd.concat(runs).sort_values("logp", ascending=False).head(args.top)
    keys, sizes = list(opt["keys"]), opt["sizes"].astype(int)

    spec, data, model = build()
    # value vars (unconstrained) -> each free RV's constrained value, which initvals expects
    to_rv = model.compile_fn(
        [model.rvs_to_transforms[rv].backward(model.rvs_to_values[rv], *rv.owner.inputs)
         if model.rvs_to_transforms.get(rv) is not None else model.rvs_to_values[rv]
         for rv in model.free_RVs],
        inputs=model.value_vars, on_unused_input="ignore")
    shapes = {v.name: model.initial_point()[v.name].shape for v in model.value_vars}
    initvals = []
    for x in allstarts.x:
        pt, i = {}, 0
        for k, n in zip(keys, sizes):
            pt[k] = x[i:i + n].reshape(shapes[k])
            i += n
        initvals.append({rv.name: val for rv, val in zip(model.free_RVs, to_rv(pt))})
    model.rvs_to_initial_values = dict.fromkeys(model.rvs_to_initial_values)

    with model:
        idata = pm.sample(draws=args.draws, tune=args.tune, chains=args.top, cores=args.top,
                          initvals=initvals, init="adapt_diag", random_seed=args.seed,
                          target_accept=0.95,             # the K-axis fits' setting (fit.py)
                          nuts_sampler="pymc", progressbar=True,
                          var_names=[rv.name for rv in model.free_RVs] + ["sigma_b"])
        idata.to_netcdf(OUT / "nuts_from_optima_trace.nc")      # saved before any readout
        ll = pm.compute_log_likelihood(idata, var_names=["obs"], progressbar=False,
                                       extend_inferencedata=False)
    floor = spec.sigma_b_floor
    sig = idata.posterior["sigma_b"].values                    # (chain, draw, bench)
    lp = idata.sample_stats["lp"].values
    llsum = ll["obs"].sum([d for d in ll["obs"].dims if d not in ("chain", "draw")]).values
    rows = []
    for c in range(args.top):
        rows.append({"chain": c, "map_start_logp": allstarts.logp.iloc[c],
                     "logp_median": np.median(lp[c]), "loglik_median": np.median(llsum[c]),
                     "sigma_b_mu_median": np.median(idata.posterior["sigma_b_mu"].values[c]),
                     "sigma_b_tau_median": np.median(idata.posterior["sigma_b_tau"].values[c]),
                     "min_sigma_b": sig[c].min(),
                     "n_bench_median_within_20pct_of_floor":
                         int((np.median(sig[c], axis=0) < 1.2 * floor).sum()),
                     "divergences": int(idata.sample_stats["diverging"].values[c].sum())})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "nuts_from_optima.csv", index=False)
    print(out.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
