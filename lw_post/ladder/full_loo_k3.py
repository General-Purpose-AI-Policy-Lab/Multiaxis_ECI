"""Full-draw PSIS-LOO for K=3 (both priors) + paired delta against the K=4
both-priors reference, matching the full-draw methodology already used for
the other rows of comparison.csv (K1 baseline, K4 bare, K4 human-only,
K4 reference itself): no thinning, reff from the ESS of the pointwise log
density (r_eff_log_lik convention), paired delta restricted to rows where
BOTH fits have Pareto-k < 0.7.

Also recomputes the K4 reference full-draw LOO as a reproduction check
against the already-staged 7,710.4 +- 76.3.

Run: python lw_post/ladder/full_loo_k3.py
"""
from __future__ import annotations

import gc
import json
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
FLAG = ROOT / ("results/mirt_humanmerge_lineageprior_lineagebm_"
               "dropFrontierMathv1AlgoTune_floors_poolednoise")
K3_TRACE = FLAG / ("trace_mirt_k3_humanmerge_lineageprior_lineagebm_"
                    "dropFrontierMathv1AlgoTune_floors_poolednoise.nc")
K4_TRACE = FLAG / ("trace_mirt_k4_humanmerge_lineageprior_lineagebm_"
                    "dropFrontierMathv1AlgoTune_floors_poolednoise.nc")
OUT = Path(__file__).resolve().parent


def full_loo(path: Path):
    """PSIS-LOO on every draw (no thinning). float32 to fit the 10x20,000
    reference trace's log_likelihood (float64 on disk -> 8 GB; float32 -> 4 GB)
    in memory alongside arviz's internal psislw copies."""
    ll = xr.open_dataset(path, group="log_likelihood")
    obs = ll["obs"].astype("float32").load()
    ll.close()
    idata = az.InferenceData(log_likelihood=obs.to_dataset(name="obs"))
    n = obs.sizes["chain"] * obs.sizes["draw"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reff = float(np.nanmean(az.ess(obs.to_dataset(name="obs"),
                                       method="mean")["obs"].values) / n)
        loo = az.loo(idata, pointwise=True, reff=min(reff, 1.0))
    loo_i = getattr(loo, "loo_i", None)
    if loo_i is None:
        loo_i = loo.elpd_i
    elpd_i = np.asarray(loo_i.values).ravel()
    pk = np.asarray(loo.pareto_k.values).ravel()
    out = {
        "elpd_loo": float(loo.elpd_loo), "elpd_se": float(loo.se),
        "p_loo": float(loo.p_loo),
        "frac_k_ge_0p7": float(np.mean(pk >= 0.7)),
        "frac_k_ge_1": float(np.mean(pk >= 1.0)),
        "loo_draws": int(n), "reff": round(reff, 3),
    }
    del obs, idata
    gc.collect()
    return out, elpd_i, pk


def obs_vector(path: Path) -> np.ndarray:
    od = xr.open_dataset(path, group="observed_data")
    v = np.asarray(od["obs"].values).ravel()
    od.close()
    return v


def main():
    obs_k3, obs_k4 = obs_vector(K3_TRACE), obs_vector(K4_TRACE)
    same = obs_k3.size == obs_k4.size and np.array_equal(obs_k3, obs_k4)
    print(f"observed_data identical between K3 and K4 reference: {same} "
          f"({obs_k3.size} vs {obs_k4.size} obs)", flush=True)
    assert same, "observed vectors differ -- cannot pair"

    print("\n== K4 reference (reproduction check) ==", flush=True)
    ref_summary, ref_elpd_i, ref_pk = full_loo(K4_TRACE)
    print(ref_summary, flush=True)

    print("\n== K3 both priors ==", flush=True)
    k3_summary, k3_elpd_i, k3_pk = full_loo(K3_TRACE)
    print(k3_summary, flush=True)

    ok = (k3_pk < 0.7) & (ref_pk < 0.7)
    diff = k3_elpd_i[ok] - ref_elpd_i[ok]
    paired = {
        "paired_rows_clean": int(ok.sum()),
        "paired_delta_vs_ref": float(diff.sum()),
        "paired_delta_se": float(np.sqrt(diff.size * np.var(diff))),
    }
    print("\n== paired delta (K3 - K4 reference), k<0.7 on both ==", flush=True)
    print(paired, flush=True)

    json.dump({"k4_reference_reproduction": ref_summary,
               "k3_full_priors": k3_summary,
               "paired_delta_k3_vs_ref": paired},
              open(OUT / "full_loo_k3_result.json", "w"), indent=2)


if __name__ == "__main__":
    main()
