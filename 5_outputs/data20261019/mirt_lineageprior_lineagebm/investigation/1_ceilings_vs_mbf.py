"""Ceilings of the --ceiling-noise sensitivity fit against benchmark-forecasting's upper asymptotes.

Ours: d_b = 1 - e_b of the K=4 fit (primary mode's chains), bounded below by the best performance
observed on the benchmark (models, and human baselines). MBF: L_b of its main fit
(harvey_joint_skew_Lobs_Lceil_Lb1, cutoff 2026-10-01), the asymptote of a sigmoid through the
frontier's scores over time, bounded below by the best score at its cutoff. Writes
ceilings_vs_mbf.csv and ceilings_vs_mbf.png beside this script.

    python 5_outputs/data20261019/mirt_lineageprior_lineagebm/investigation/1_ceilings_vs_mbf.py
"""
import sys
from pathlib import Path

import arviz as az
import matplotlib
import numpy as np
import pandas as pd
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "2_model"))
from multiaxis_eci.analysis.fitspec import FitSpec, best_mode_chains  # noqa: E402
from multiaxis_eci.data import load_benchmark_ceilings, load_observed_best  # noqa: E402

TRACE = HERE.parent / "trace_mirt_k4_lineageprior_lineagebm.nc"
MBF = (ROOT.parent / "Model_Benchmark_forecasting" / "3_outputs" / "cutoff20261001" / "fits"
       / "harvey_joint_skew_Lobs_Lceil_Lb1_cutoff20261001_nutpie_dcfb7b1.nc")

post = xr.open_dataset(TRACE, group="posterior")
spec = FitSpec.from_trace(az.InferenceData(posterior=post.isel(draw=slice(0, 1))), TRACE)
data, _, _ = spec.load_data()
bench = data.blookup.sort_values("benchmark_idx")["benchmark"].tolist()
chains = best_mode_chains(TRACE)
d = post["ceiling_d"].isel(chain=chains).values.reshape(-1, len(bench))
best = load_observed_best(data)
known = load_benchmark_ceilings(data)
n_takers = pd.Series(data.model_idx).groupby(data.bench_idx).nunique().reindex(range(len(bench))).values
ours = pd.DataFrame({
    "benchmark": bench, "n_takers": n_takers,
    "best_observed": best, "known_ceiling": known,
    "d_median": np.median(d, 0), "d_q10": np.quantile(d, 0.1, 0), "d_q90": np.quantile(d, 0.9, 0),
    # how much of the posterior sits within 0.01 of the bound: the ceiling pressed onto the best score
    "p_at_bound": (d <= np.minimum(best, 1.0)[None, :] + 0.01).mean(0),
})

m = xr.open_dataset(MBF, group="posterior")
L = m["L"].values.reshape(-1, m.sizes["benchmark"])
mbf = pd.DataFrame({
    "benchmark": [str(b) for b in m["benchmark"].values],
    "L_median": np.median(L, 0), "L_q10": np.quantile(L, 0.1, 0), "L_q90": np.quantile(L, 0.9, 0),
    "L_min": L.min(0),                       # its lower bound (L_floor) up to sampling
    "L_pinned": L.std(0) < 1e-9,             # a known ceiling, or a score of 1.0 reached
})
t = ours.merge(mbf, on="benchmark", how="outer", indicator=True)
t["in"] = t["_merge"].map({"both": "both", "left_only": "ours only", "right_only": "MBF only"})
t = t.drop(columns="_merge")
both = t[(t["in"] == "both") & t["known_ceiling"].isna() & ~t["L_pinned"].fillna(False).astype(bool)]
both = both[both["best_observed"] < 1 - 1e-9]                       # estimated on both sides
t.to_csv(HERE / "ceilings_vs_mbf.csv", index=False, float_format="%.4f")

gap_o, gap_m = 1 - both["d_median"], 1 - both["L_median"]
print(f"benchmarks: ours {len(ours)}, MBF {len(mbf)}, in both and estimated on both sides {len(both)}")
print(f"median gap: ours {gap_o.median():.3f}, MBF {gap_m.median():.3f}; "
      f"corr(gap ours, gap MBF) = {np.corrcoef(gap_o, gap_m)[0, 1]:.2f} (Spearman "
      f"{pd.Series(gap_o.values).corr(pd.Series(gap_m.values), method='spearman'):.2f})")
print(f"ours pressed on the bound (p_at_bound > 0.5): {(both['p_at_bound'] > 0.5).sum()} of {len(both)}; "
      f"MBF at its floor (L_median - L_min < 0.01): {((both['L_median'] - both['L_min']) < 0.01).sum()}")
print(f"ours below MBF by more than 0.03: {((both['d_median'] - both['L_median']) < -0.03).sum()}; "
      f"above by more than 0.03: {((both['d_median'] - both['L_median']) > 0.03).sum()}")
cols = ["benchmark", "n_takers", "best_observed", "d_median", "p_at_bound", "L_median", "L_min"]
print("\nlargest gaps, ours:")
print(both.sort_values("d_median")[cols].head(15).to_string(index=False, float_format="%.3f"))
print("\nlargest disagreements (ours - MBF):")
dd = both.assign(diff=both["d_median"] - both["L_median"]).sort_values("diff")
print(pd.concat([dd.head(8), dd.tail(8)])[cols + ["diff"]].to_string(index=False, float_format="%.3f"))

fig, ax = plt.subplots(figsize=(7, 7))
pressed = both["p_at_bound"] > 0.5
for mask, colour, label in ((~pressed, "#2a6fdb", "ours above its bound"),
                            (pressed, "#d1495b", "ours pressed on the best observed score")):
    s = both[mask]
    ax.errorbar(s["L_median"], s["d_median"],
                xerr=[s["L_median"] - s["L_q10"], s["L_q90"] - s["L_median"]],
                yerr=[s["d_median"] - s["d_q10"], s["d_q90"] - s["d_median"]],
                fmt="o", ms=4, lw=0.6, color=colour, alpha=0.8, label=label)
for r in both.itertuples():
    if abs(r.d_median - r.L_median) > 0.04 or r.d_median < 0.9:
        ax.annotate(r.benchmark, (r.L_median, r.d_median), fontsize=6, alpha=0.8)
ax.plot([0.75, 1], [0.75, 1], color="grey", lw=0.8, ls="--")
ax.set(xlim=(0.74, 1.005), ylim=(0.74, 1.005), xlabel="MBF upper asymptote L (median, 80%)",
       ylabel="Multiaxis ceiling d (median, 80%)",
       title="Benchmark ceilings: --ceiling-noise K=4 fit vs benchmark-forecasting")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(HERE / "ceilings_vs_mbf.png", dpi=150)
print(f"\n-> {HERE / 'ceilings_vs_mbf.csv'}, ceilings_vs_mbf.png")
