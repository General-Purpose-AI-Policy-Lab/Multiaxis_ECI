"""Identified convergence of the flagship inside mode A and inside mode C.

Same quantities and same statistics as r1_runs/convergence_identified.csv (eta per observation, D and
sigma_b per benchmark: rank r-hat, bulk and tail ESS), plus the divergences on each mode's saved
draws. One mode is loaded at a time. Writes convergence_by_mode.csv beside this file.
"""
import json
import sys
from pathlib import Path

import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "r1_runs"))
import aligned_loadings as al  # noqa: E402

data = al.load_data()
modes = json.loads((al.FIT / f"mirt_modes_{al.TRACE.stem}.json").read_text())["modes"]
bi, mi = xr.DataArray(data.bench_idx, dims="obs"), xr.DataArray(data.model_idx, dims="obs")
rows = []
for label in ("A", "C"):
    chains = next(m["chains"] for m in modes if m["label"] == label)
    with xr.open_dataset(al.TRACE, group="posterior") as p:
        post = p[["A", "theta", "D", "sigma_b"]].isel(chain=chains).load().astype("float64")
    with xr.open_dataset(al.TRACE, group="sample_stats") as s:
        div = int(s["diverging"].isel(chain=chains).sum())
    eta = (post["A"].isel(bench=bi) * post["theta"].isel(model=mi)).sum("latent") - post["D"].isel(bench=bi)
    n = len(chains) * post.sizes["draw"]
    for name, da in (("eta", eta), ("D", post["D"]), ("sigma_b", post["sigma_b"])):
        row = al.stat_row(name, *al.summarize(da, name))
        row.update(mode=label, chains=len(chains), divergences=f"{div} / {n}")
        rows.append(row)
    del post, eta
out = pd.DataFrame(rows)
out.to_csv(Path(__file__).with_name("convergence_by_mode.csv"), index=False)
print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}" if x < 10 else f"{x:.0f}"))
