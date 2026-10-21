"""Which ceilings of the ceiling flagship are held up by a human baseline rather than by a model.

The ceiling prior is truncated at the best performance observed on each benchmark, a model score
or a human baseline (data.load_observed_best). Where the human row is the higher of the two, it
alone sets the lower bound d_b >= best human. For those benchmarks: the best model score, the best
human score (tier, note, source), the posterior ceiling (median, 5-95%) of the mode A chains, and
how far the bound sits above the models. Writes human_bound_ceilings.csv.
"""
import json
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "2_model"))
from multiaxis_eci.analysis.fitspec import FitSpec  # noqa: E402
from multiaxis_eci.data import HUMAN_FILE, load_benchmark_ceilings  # noqa: E402

TRACE = HERE.parent / "trace_mirt_k4_lineageprior_lineagebm.nc"
modes = json.loads((HERE.parent / "mirt_modes_trace_mirt_k4_lineageprior_lineagebm.json").read_text())
chains = [m for m in modes["modes"] if m["label"] == "A"][0]["chains"]

with xr.open_dataset(TRACE, group="posterior") as post:
    spec = FitSpec.from_trace(az.InferenceData(posterior=post), TRACE)
    data, _, _ = spec.load_data(az.InferenceData(posterior=post))
    d = post["ceiling_d"].isel(chain=chains).values.reshape(-1, post.sizes["bench"])
benchmarks = list(data.blookup.sort_values("benchmark_idx")["benchmark"])
model_best = np.full(len(benchmarks), -np.inf)
np.maximum.at(model_best, np.asarray(data.bench_idx), np.asarray(data.scores, dtype=float))
known = load_benchmark_ceilings(data)

hb = pd.read_csv(HUMAN_FILE)
hb["benchmark"] = hb["benchmark"].str.strip()
top = hb.sort_values("score").groupby("benchmark").tail(1).set_index("benchmark")

rows = []
for i, b in enumerate(benchmarks):
    if b not in top.index:
        continue
    h = top.loc[b]
    rows.append(dict(
        benchmark=b, best_model=model_best[i], best_human=h["score"], tier=h["group"],
        note=h["note"], source=h["source"],
        binds=bool(h["score"] > model_best[i]),
        known_ceiling=bool(np.isfinite(known[i])) if known is not None else False,
        d_median=np.median(d[:, i]), d_q05=np.quantile(d[:, i], 0.05), d_q95=np.quantile(d[:, i], 0.95),
    ))
out = pd.DataFrame(rows)
out["human_above_models"] = out["best_human"] - out["best_model"]
out["d_above_models"] = out["d_median"] - out["best_model"]
out = out.sort_values(["binds", "human_above_models"], ascending=[False, False])
out.to_csv(HERE / "human_bound_ceilings.csv", index=False, float_format="%.3f")
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 40)
print(out[["benchmark", "best_model", "best_human", "tier", "note", "known_ceiling", "d_median", "d_q05", "d_q95"]]
      .to_string(index=False, float_format="%.3f"))
print(f"\n{out['binds'].sum()} of {len(out)} benchmarks with a human row have it above every model")
