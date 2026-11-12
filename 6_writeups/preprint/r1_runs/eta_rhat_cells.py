"""Per-observation eta r-hat over all 50 chains and over mode A, with the model and benchmark of each
cell, so the unconverged predictions can be located. Writes eta_rhat_by_obs.csv beside this file."""
import json, sys
from pathlib import Path
import arviz as az, numpy as np, pandas as pd, xarray as xr
sys.path.insert(0, str(Path(__file__).resolve().parent))
import aligned_loadings as al

data = al.load_data()
with xr.open_dataset(al.TRACE, group="posterior") as p:
    post = p[["A", "theta", "D"]].load().astype("float64")
modes = json.loads((al.FIT / "mirt_modes_trace_mirt_k4_lineageprior_lineagebm.json").read_text())
a_chains = [c for m in modes["modes"] if m["label"] == "A" for c in m["chains"]]
bi, mi = xr.DataArray(data.bench_idx, dims="obs"), xr.DataArray(data.model_idx, dims="obs")
eta = (post["A"].isel(bench=bi) * post["theta"].isel(model=mi)).sum("latent") - post["D"].isel(bench=bi)
r_all = az.rhat(eta.to_dataset(name="e"))["e"].values
r_a = az.rhat(eta.isel(chain=a_chains).to_dataset(name="e"))["e"].values
sd = eta.std(("chain", "draw")).values
spread = eta.mean("draw").values
df = pd.DataFrame({"model": post["model"].values[data.model_idx], "benchmark": post["bench"].values[data.bench_idx],
                   "score": data.scores, "rhat_all": r_all, "rhat_modeA": r_a,
                   "chain_mean_range_in_sd": (spread.max(0) - spread.min(0)) / sd})
df.to_csv(al.OUT / "eta_rhat_by_obs.csv", index=False)
bad = df[df.rhat_all > 1.05]
print(f"eta cells over 1.05 (all chains): {len(bad)} / {len(df)}, on {bad.model.nunique()} / {df.model.nunique()} models "
      f"and {bad.benchmark.nunique()} / {df.benchmark.nunique()} benchmarks")
print("\nbenchmarks with the most cells over 1.05 (count / cells on that benchmark):")
n = df.groupby("benchmark").size()
print((bad.groupby("benchmark").size().sort_values(ascending=False).head(12).astype(str) + " / "
       + n.reindex(bad.groupby("benchmark").size().sort_values(ascending=False).head(12).index).astype(str)).to_string())
print("\nmodels with the most cells over 1.05:")
nm = df.groupby("model").size()
top = bad.groupby("model").size().sort_values(ascending=False).head(12)
print((top.astype(str) + " / " + nm.reindex(top.index).astype(str)).to_string())
print("\nworst 10 cells:")
print(df.nlargest(10, "rhat_all").round(3).to_string(index=False))
