"""Is the ladder's LOO robust to the two knobs the memory budget forced?

  (a) draw thinning — the flagship log-likelihood is 10 x 20,000 x 5,004;
  (b) reff — arviz derives it from the posterior group (38 GB on the flagship),
      so ladder.py uses the relative ESS of the pointwise log densities.

For each fit this recomputes elpd at the ladder thinning and at twice-coarser
thinning, and at reff = 0.05 / 0.5 / 1.0, then re-derives the headline paired
deltas from the coarser pointwise densities.

Run: ~/miniforge3/envs/pymc_env/bin/python lw_post/ladder/sensitivity.py
"""
from __future__ import annotations

import gc
import sys
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ladder import FITS, TARGET_DRAWS_PER_CHAIN, OUT  # noqa: E402


def loo_at(obs, reff):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loo = az.loo(az.InferenceData(log_likelihood=obs.to_dataset(name="obs")),
                     pointwise=True, reff=reff)
    li = getattr(loo, "loo_i", None)
    li = loo.elpd_i if li is None else li
    return loo, np.asarray(li.values).ravel(), np.asarray(loo.pareto_k.values).ravel()


def main():
    rows, li_coarse, kk_coarse = [], {}, {}
    for step, (label, path) in FITS.items():
        ll = xr.open_dataset(path, group="log_likelihood")
        base = max(1, ll.sizes["draw"] // TARGET_DRAWS_PER_CHAIN)
        obs = ll["obs"].isel(draw=slice(None, None, base)).load()
        ll.close()
        n = obs.sizes["chain"] * obs.sizes["draw"]
        reff_ll = min(1.0, float(np.nanmean(
            az.ess(obs.to_dataset(name="obs"), method="mean")["obs"].values) / n))
        for tag, o, reff in [("thin x1, reff=ll", obs, reff_ll),
                             ("thin x1, reff=0.05", obs, 0.05),
                             ("thin x1, reff=0.5", obs, 0.5),
                             ("thin x1, reff=1.0", obs, 1.0),
                             ("thin x2, reff=ll", obs.isel(draw=slice(None, None, 2)),
                              reff_ll)]:
            loo, li, kk = loo_at(o, reff)
            rows.append({"step": step, "variant": tag,
                         "draws": int(o.sizes["chain"] * o.sizes["draw"]),
                         "reff": round(reff, 3),
                         "elpd_loo": round(float(loo.elpd_loo), 2),
                         "se": round(float(loo.se), 2),
                         "p_loo": round(float(loo.p_loo), 2),
                         "k_gt_0.7": int((kk > 0.7).sum())})
            if tag.startswith("thin x2"):
                li_coarse[step], kk_coarse[step] = li, kk
            print(rows[-1], flush=True)
        del obs
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "ladder_loo_sensitivity.csv", index=False)

    pr = []
    for a, b in [("1", "0"), ("2", "1"), ("3", "2"), ("3", "0"), ("3", "x")]:
        d = li_coarse[a] - li_coarse[b]
        ok = (kk_coarse[a] < 0.7) & (kk_coarse[b] < 0.7)
        pr.append({"pair": f"step{a} - step{b}",
                   "delta_elpd": round(float(d.sum()), 2),
                   "se": round(float(np.sqrt(d.size * np.var(d))), 2),
                   "delta_elpd_k_ok": round(float(d[ok].sum()), 2),
                   "n_k_ok_rows": int(ok.sum())})
    dfp = pd.DataFrame(pr)
    dfp.to_csv(OUT / "ladder_pairwise_thin2.csv", index=False)
    print("\ncoarse-thin paired deltas:\n" + dfp.to_string(index=False))
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
