"""Which ceilings the data identify, and whether the ceilings improve the fit.

1. Information: for each benchmark, the posterior sd of its gap e_b = 1 - d_b against the sd of
   its prior (0.25 * Beta(1.43, 4.67) truncated at 1 - best observed score, sampled). A ratio
   near 1 is a ceiling the prior sets; well below 1, one the scores set.
2. PSIS-LOO, observation by observation, of the flagship (no ceiling) and of the --ceiling-noise
   fit, each on its primary mode's chains, summed per benchmark: where the ceilings help or hurt.

Writes information_and_loo.csv beside this script.
"""
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "2_model"))
from multiaxis_eci.analysis.fitspec import FitSpec, best_mode_chains  # noqa: E402
from multiaxis_eci.config import PRIOR_CEILING_GAP as G  # noqa: E402
from multiaxis_eci.data import load_benchmark_ceilings, load_observed_best  # noqa: E402

CEIL = HERE.parent / "trace_mirt_k4_lineageprior_lineagebm.nc"
FLAG = (HERE.parents[1] / "mirt_lineageprior_lineagebm_noceiling"
        / "trace_mirt_k4_lineageprior_lineagebm_noceiling.nc")


def spec_data(trace):
    with xr.open_dataset(trace, group="posterior") as p:
        spec = FitSpec.from_trace(az.InferenceData(posterior=p.isel(draw=slice(0, 1))), trace)
    return spec.load_data()[0]


data = spec_data(CEIL)
bench = data.blookup.sort_values("benchmark_idx")["benchmark"].tolist()
best = load_observed_best(data)
known = load_benchmark_ceilings(data)

# 1. prior and posterior spread of the gap
rng = np.random.default_rng(0)
raw = rng.beta(G["alpha"], G["beta"], 400_000)
post = xr.open_dataset(CEIL, group="posterior")
chains = best_mode_chains(CEIL)
gap = 1 - post["ceiling_d"].isel(chain=chains).values.reshape(-1, len(bench))
rows = []
for b, name in enumerate(bench):
    top = min(1.0, max(0.0, (1 - best[b]) / G["scale"]))
    if np.isfinite(known[b]) or top <= 1e-9:
        rows.append(dict(benchmark=name, prior_sd=0.0, post_sd=0.0, info_ratio=np.nan,
                         prior_mean=1 - (known[b] if np.isfinite(known[b]) else 1.0)))
        continue
    pr = G["scale"] * raw[raw <= top]
    rows.append(dict(benchmark=name, prior_mean=pr.mean(), prior_sd=pr.std(),
                     post_mean=gap[:, b].mean(), post_sd=gap[:, b].std(),
                     info_ratio=gap[:, b].std() / pr.std()))
info = pd.DataFrame(rows)
info["best_observed"] = best
info["shift_in_prior_sd"] = (info["post_mean"] - info["prior_mean"]) / info["prior_sd"]

# 2. PSIS-LOO per observation, primary modes
def pointwise(trace):
    ll = xr.open_dataset(trace, group="log_likelihood")["obs"]
    ch = best_mode_chains(trace)
    ll = ll.isel(chain=ch) if ch is not None else ll
    v = ll.values
    lo = az.loo(az.from_dict(posterior={"_": np.zeros(v.shape[:2])}, log_likelihood={"obs": v}),
                pointwise=True)
    return np.asarray(lo.loo_i), np.asarray(lo.pareto_k), float(lo.elpd_loo), float(lo.se)


assert spec_data(FLAG).n_obs == data.n_obs
e_c, k_c, tot_c, se_c = pointwise(CEIL)
e_f, k_f, tot_f, se_f = pointwise(FLAG)
diff = e_c - e_f
se_diff = np.sqrt(len(diff)) * diff.std()
good = (k_c < 0.7) & (k_f < 0.7)
print(f"PSIS-LOO elpd: ceiling-noise {tot_c:.1f} (se {se_c:.1f}), flagship {tot_f:.1f} (se {se_f:.1f}); "
      f"difference {diff.sum():+.1f} (se {se_diff:.1f})")
print(f"  on the {good.sum()} observations with k < 0.7 in both: {diff[good].sum():+.1f} "
      f"(se {np.sqrt(good.sum()) * diff[good].std():.1f}); k > 0.7: {(k_c > 0.7).sum()} / {(k_f > 0.7).sum()}")
per = (pd.DataFrame({"b": data.bench_idx, "d": diff, "good": good})
       .groupby("b").agg(elpd_diff=("d", "sum"), elpd_diff_reliable=("d", lambda s: s[good[s.index]].sum()),
                         n_obs=("d", "size")))
per.index = [bench[i] for i in per.index]
t = info.set_index("benchmark").join(per)
t.to_csv(HERE / "information_and_loo.csv", float_format="%.4f")

est = t[t["prior_sd"] > 0]
print(f"\nestimated ceilings: {len(est)}; data-informed (posterior sd < 60% of the prior's): "
      f"{(est['info_ratio'] < 0.6).sum()}; prior-dominated (> 85%): {(est['info_ratio'] > 0.85).sum()}")
cols = ["best_observed", "prior_mean", "post_mean", "info_ratio", "shift_in_prior_sd", "elpd_diff", "n_obs"]
print("\nmost data-informed ceilings:")
print(est.sort_values("info_ratio")[cols].head(15).to_string(float_format="%.3f"))
print("\nlargest elpd gains / losses of the ceilings, per benchmark:")
s = t.sort_values("elpd_diff")
print(pd.concat([s.tail(10), s.head(8)])[cols].to_string(float_format="%.3f"))
