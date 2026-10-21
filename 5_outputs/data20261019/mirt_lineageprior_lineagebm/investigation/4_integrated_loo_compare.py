"""The four combinations (axis structure x ceilings) under integrated LOO.

Reads the per-observation tables `4_diagnostics/integrated_loo.py` wrote for the flagship and for
the --ceiling-noise fit (diagnostics/integrated_loo_<mode>.csv: each mode's elpd averaged over its
chains at equal draws, and the largest Pareto k over them), and compares every pair on the
observations whose k is below 0.7 in both, then on all of them; per benchmark, the ceiling effect
at equal structure. Writes integrated_loo_compare.csv beside this script.
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
GEN = HERE.parents[1]
TABLES = {
    "flagship A (merged + visual, no ceilings)": GEN / "mirt_lineageprior_lineagebm_noceiling/diagnostics/integrated_loo_A.csv",
    "flagship B (agentic apart, no ceilings)": GEN / "mirt_lineageprior_lineagebm_noceiling/diagnostics/integrated_loo_B.csv",
    "ceilings A (agentic apart)": GEN / "mirt_lineageprior_lineagebm/diagnostics/integrated_loo_A.csv",
    "ceilings B (merged + visual)": GEN / "mirt_lineageprior_lineagebm/diagnostics/integrated_loo_B.csv",
}
t = {k: pd.read_csv(v) for k, v in TABLES.items()}
first = next(iter(t.values()))
for k, v in t.items():
    assert (v["model"].values == first["model"].values).all() and \
           (v["benchmark"].values == first["benchmark"].values).all(), k
for k, v in t.items():
    print(f"{k:44s} elpd_int {v['elpd_int'].sum():10.1f}   k > 0.7: {(v['k_int_max'] > 0.7).sum()}")
rows = []
for a, b in combinations(t, 2):
    d = t[a]["elpd_int"].values - t[b]["elpd_int"].values
    g = (t[a]["k_int_max"].values < 0.7) & (t[b]["k_int_max"].values < 0.7)
    rows.append(dict(a=a, b=b, diff_all=d.sum(), se_all=np.sqrt(len(d)) * d.std(),
                     diff_reliable=d[g].sum(), se_reliable=np.sqrt(g.sum()) * d[g].std(), n_reliable=int(g.sum())))
cmp = pd.DataFrame(rows)
cmp.to_csv(HERE / "integrated_loo_compare.csv", index=False, float_format="%.2f")
print("\n" + cmp.to_string(index=False, float_format="%.1f"))
# the ceilings at equal structure, per benchmark, on the reliable observations
for a, b in (("ceilings B (merged + visual)", "flagship A (merged + visual, no ceilings)"),
             ("ceilings A (agentic apart)", "flagship B (agentic apart, no ceilings)")):
    g = (t[a]["k_int_max"] < 0.7) & (t[b]["k_int_max"] < 0.7)
    per = (pd.DataFrame({"benchmark": first["benchmark"], "d": t[a]["elpd_int"] - t[b]["elpd_int"]})[g]
           .groupby("benchmark")["d"].sum().sort_values())
    print(f"\n{a} - {b}, per benchmark (reliable observations): gains")
    print(per.tail(8).round(1).to_string())
    print("losses"); print(per.head(6).round(1).to_string())
