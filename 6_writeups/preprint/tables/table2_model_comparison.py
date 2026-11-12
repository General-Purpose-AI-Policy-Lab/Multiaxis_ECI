"""Table 2: the single-axis baseline against the multi-axis model, on the same 5,487 scores.

In-sample fit (RMSE, Bayesian R2) is read from the fit folder's mirt_gof_k1.json and mirt_gof_k4.json.
Out of sample: PSIS-LOO per score from each trace's stored pointwise log-likelihood, at equal draws
(the baseline's 10 chains against the first 10 chains of mode A and of mode C, 1,000 saved draws each).
The paired difference to the baseline is given on every score and on the scores whose Pareto k is at
most 0.7 in both fits, with its standard error sqrt(n) * sd(difference). No relative-efficiency
correction is applied to the weights. Nothing is refitted.

Writes table2_model_comparison.csv and table2_loo_pointwise.csv beside this file.

    python 6_writeups/preprint/tables/table2_model_comparison.py
"""
import json
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "2_model"))
from multiaxis_eci.analysis import FLAGSHIP_TRACE, best_mode_chains, secondary_mode_chains  # noqa: E402

FIT, N_CHAINS, K_MAX = FLAGSHIP_TRACE.parent, 10, 0.7


def psis_loo(trace, chains):
    """(elpd_i, Pareto k_i) of a trace's chains."""
    with xr.open_dataset(trace, group="log_likelihood") as g:
        ll = g["obs"].isel(chain=chains).load().values.astype("float64")
    ll = ll.reshape(-1, ll.shape[-1]).T                      # (n_obs, draws)
    lw, k = az.psislw(-ll)
    return logsumexp(np.asarray(lw) + ll, axis=1), np.asarray(k)


fits = {
    "single-axis baseline": (FIT / "trace_mirt_k1.nc", list(range(N_CHAINS)), "mirt_gof_k1.json"),
    "multi-axis, mode A": (FLAGSHIP_TRACE, best_mode_chains(FLAGSHIP_TRACE)[:N_CHAINS], "mirt_gof_k4.json"),
    "multi-axis, mode C": (FLAGSHIP_TRACE, secondary_mode_chains(FLAGSHIP_TRACE)[:N_CHAINS], None),
}
point, rows = {}, []
for name, (trace, chains, gof) in fits.items():
    elpd, k = psis_loo(trace, chains)
    point[name] = (elpd, k)
    g = json.loads((FIT / gof).read_text()) if gof else {}
    rows.append(dict(model=name, chains=str(chains), rmse_in_sample=g.get("rmse"), r2_in_sample=g.get("bayesian_r2"),
                     elpd_loo=elpd.sum(), n_scores=len(elpd), n_pareto_k_high=int((k > K_MAX).sum())))
base_e, base_k = point["single-axis baseline"]
for row in rows:
    e, k = point[row["model"]]
    d, ok = e - base_e, (k <= K_MAX) & (base_k <= K_MAX)
    row.update(diff_all=d.sum(), se_all=np.sqrt(len(d)) * d.std(ddof=1), n_reliable=int(ok.sum()),
               diff_reliable=d[ok].sum(), se_reliable=np.sqrt(ok.sum()) * d[ok].std(ddof=1))
out = pd.DataFrame(rows)
out.to_csv(Path(__file__).with_name("table2_model_comparison.csv"), index=False)
pd.DataFrame({f"{n} {c}": v for n, (e, k) in point.items() for c, v in (("elpd", e), ("pareto_k", k))}
             ).to_csv(Path(__file__).with_name("table2_loo_pointwise.csv"), index_label="obs")
pd.set_option("display.width", 250, "display.max_columns", 20)
print(out.drop(columns="chains").round(3).to_string(index=False))
