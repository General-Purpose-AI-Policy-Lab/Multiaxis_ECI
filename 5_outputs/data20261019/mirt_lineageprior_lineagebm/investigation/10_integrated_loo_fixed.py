"""Integrated LOO of the flagship's five modes, recomputed with the founder fix of 2026-10-26.

Until that day `4_diagnostics/integrated_loo.py` gave a chain founder, under the skew-normal block,
a spurious incoming step from the last node of the lineage map; every integrated LOO of a
skew-normal fit before it is off (about 2,900 nats on one chain of this flagship). The modes were
rescored in separate runs (mode C in two halves), each writing `integrated_loo_<group>.csv`
(per score: elpd averaged over the group's chains, largest Pareto k): this script joins them,
puts the per-mode tables in the fit's `diagnostics/`, and writes the paired differences to mode A,
on every score and on the scores with k < 0.7 in every mode, to integrated_loo_5modes_50chains.json.

    python 5_outputs/data20261019/mirt_lineageprior_lineagebm/investigation/10_integrated_loo_fixed.py RUNS_DIR
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
runs = Path(sys.argv[1])
modes = json.loads((HERE.parent / "mirt_modes_trace_mirt_k4_lineageprior_lineagebm.json").read_text())
chains = {m["label"]: m["chains"] for m in modes["modes"]}


def table(path):
    return pd.read_csv(path)


tabs = {"A": table(runs / "flag" / "integrated_loo_flagA.csv"),
        "B": table(runs / "flag_BDE" / "integrated_loo_B.csv"),
        "D": table(runs / "flag_BDE" / "integrated_loo_D.csv"),
        "E": table(runs / "flag_BDE" / "integrated_loo_E.csv")}
c1, c2 = table(runs / "flag_C1" / "integrated_loo_C1.csv"), table(runs / "flag_C2" / "integrated_loo_C2.csv")
n1, n2 = 10, 9                                       # chains in each half of mode C
C = c1.copy()
for col in ("elpd_int", "elpd_psis"):
    C[col] = (n1 * c1[col] + n2 * c2[col]) / (n1 + n2)
for col in ("k_int_max", "k_psis_max"):
    C[col] = np.maximum(c1[col], c2[col])
tabs["C"] = C
ref = tabs["A"]
for t in tabs.values():
    assert (t["model"].values == ref["model"].values).all()
    assert (t["benchmark"].values == ref["benchmark"].values).all()
out_dir = HERE.parent / "diagnostics"
out_dir.mkdir(exist_ok=True)
for lab, t in tabs.items():
    t.to_csv(out_dir / f"integrated_loo_{lab}.csv", index=False)

reliable = np.all([t["k_int_max"].values < 0.7 for t in tabs.values()], axis=0)
res = {"note": "recomputed 2026-10-26 with the founder fix of 4_diagnostics/integrated_loo.py",
       "n_obs": int(len(ref)), "n_reliable_all_modes": int(reliable.sum())}
for lab in ("A", "B", "C", "D", "E"):
    t = tabs[lab]
    r = {"chains": chains[lab], "elpd_int": round(float(t["elpd_int"].sum()), 1),
         "n_k_int_gt_0.7": int((t["k_int_max"] > 0.7).sum()),
         "elpd_psis": round(float(t["elpd_psis"].sum()), 1)}
    if lab != "A":
        d = t["elpd_int"].values - ref["elpd_int"].values
        dr = d[reliable]
        r.update(diff_to_A=round(float(d.sum()), 1), se_diff_to_A=round(float(np.sqrt(len(d)) * d.std()), 1),
                 diff_to_A_k_lt_0_7=round(float(dr.sum()), 1),
                 se_diff_to_A_k_lt_0_7=round(float(np.sqrt(len(dr)) * dr.std()), 1))
    res[lab] = r
(HERE / "integrated_loo_5modes_50chains.json").write_text(json.dumps(res, indent=1) + "\n")
print(json.dumps(res, indent=1))
