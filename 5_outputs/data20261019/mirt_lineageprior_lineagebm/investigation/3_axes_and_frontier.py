"""Does an axis of the flagship stand in for missing ceilings?

Hypothesis: without an upper asymptote, benchmarks that saturate below 1 can only be fitted by an
axis whose frontier stops rising; the flagship's mode A has such an axis (visual & commonsense), and
the --ceiling-noise fit, which explains saturation by d_b < 1, does not need it.

For each solution (flagship mode A, flagship mode B, ceiling-noise mode A): per axis, the running
maximum of the models' median abilities (models' sd units) at 2024-01-01, 2025-01-01 and today,
and the benchmarks that load on it. Then, for the saturated benchmarks (best score within 0.1 of
their observed best since 2025), the axis that carries them in each solution.
Also PSIS-LOO of the ceiling-noise fit against flagship mode B (same structure, no ceilings), so the
structure and the ceilings can be told apart.
"""
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
GEN = HERE.parents[1]
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "2_model"))
from multiaxis_eci.analysis.fitspec import best_mode_chains, secondary_mode_chains  # noqa: E402

FLAG = GEN / "mirt_lineageprior_lineagebm_noceiling"
CEIL = GEN / "mirt_lineageprior_lineagebm"
SOLUTIONS = {"flagship A": FLAG, "flagship B": FLAG / "secondary_mode", "ceiling-noise A": CEIL}
models = pd.read_csv(ROOT / "0_input" / "models.csv").drop_duplicates("model_version")
date = pd.to_datetime(models.set_index("model_version")["publication_date"], errors="coerce")


def frontier_table(folder: Path) -> pd.DataFrame:
    fs = pd.read_csv(folder / "mirt_factor_scores.csv")
    fs = fs[~fs.get("is_human", pd.Series(False, index=fs.index)).astype(bool)]
    mcol = "model" if "model" in fs.columns else fs.columns[0]
    axes = [c for c in fs.columns if c.startswith("theta") and c.endswith("_mean")]
    fs = fs.assign(date=fs[mcol].map(date)).dropna(subset=["date"])
    L = pd.read_csv(folder / "mirt_loadings.csv").pivot(index="benchmark", columns="axis",
                                                          values="loading_median")
    share = L ** 2 / (L ** 2).values.sum(1, keepdims=True)
    rows = []
    for a in axes:
        z = (fs[a] - fs[a].mean()) / fs[a].std()
        at = {d: z[fs["date"] <= d].max() for d in ("2024-01-01", "2025-01-01", "2026-10-19")}
        ax = "axis" + a[len("theta"):-len("_mean")]
        rows.append(dict(axis=ax, f2024=at["2024-01-01"], f2025=at["2025-01-01"], today=at["2026-10-19"],
                         gain_since_2024=at["2026-10-19"] - at["2024-01-01"],
                         top=", ".join(share[ax].sort_values(ascending=False).index[:6])))
    return pd.DataFrame(rows), share


shares = {}
for name, folder in SOLUTIONS.items():
    t, shares[name] = frontier_table(folder)
    print(f"\n== {name}: frontier (running max of median ability, models' sd units)")
    print(t.to_string(index=False, float_format="%.2f"))

# saturated benchmarks: best score in 2026 within 0.05 of the best ever, and above 0.85
sc = pd.read_csv(ROOT / "0_input" / "all_scores_flat.csv", parse_dates=["release_date"])
best = sc.groupby("benchmark")["score"].max()
recent = sc[sc["release_date"] >= "2025-07-01"].groupby("benchmark")["score"].max()
early = sc[sc["release_date"] < "2025-01-01"].groupby("benchmark")["score"].max()
sat = best[(best > 0.85) & ((recent - early).reindex(best.index) < 0.08)].index
print(f"\nsaturated benchmarks (best > 0.85, < 0.08 gained by 2025-07+ releases over pre-2025 ones): {len(sat)}")
rows = []
for b in sat:
    r = {"benchmark": b, "best": best[b]}
    for name, sh in shares.items():
        if b in sh.index:
            r[name] = f"{sh.loc[b].idxmax()} ({sh.loc[b].max():.2f})"
    rows.append(r)
print(pd.DataFrame(rows).to_string(index=False, float_format="%.3f"))


# LOO: ceiling-noise A against flagship B
def loo_i(trace, chains):
    ll = xr.open_dataset(trace, group="log_likelihood")["obs"].isel(chain=chains).values
    lo = az.loo(az.from_dict(posterior={"_": np.zeros(ll.shape[:2])}, log_likelihood={"obs": ll}),
                pointwise=True)
    return np.asarray(lo.loo_i), np.asarray(lo.pareto_k)


ft = FLAG / "trace_mirt_k4_lineageprior_lineagebm_noceiling.nc"
ct = CEIL / "trace_mirt_k4_lineageprior_lineagebm.nc"
e_c, k_c = loo_i(ct, best_mode_chains(ct))
for lab, ch in (("flagship A", best_mode_chains(ft)), ("flagship B", secondary_mode_chains(ft))):
    e_f, k_f = loo_i(ft, ch)
    d = e_c - e_f
    g = (k_c < 0.7) & (k_f < 0.7)
    print(f"ceiling-noise A - {lab}: {d.sum():+.1f} (se {np.sqrt(len(d)) * d.std():.1f}); "
          f"k<0.7 in both ({g.sum()}): {d[g].sum():+.1f} (se {np.sqrt(g.sum()) * d[g].std():.1f})")
