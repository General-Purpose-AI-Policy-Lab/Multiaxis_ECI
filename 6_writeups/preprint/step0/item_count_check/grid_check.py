"""How well the item counts N_b match the scores we actually have.

A score that is a share of N_b questions, from one run, sits on the grid j / N_b. For each
item-scored benchmark in the flagship scope: the share of its scores within 0.0005 of a grid point
(a 3-decimal rounding tolerance), against the share expected by chance (2 * 0.0005 * N_b,
capped at 1), plus the item-count file's verification stamp.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "2_model"))
import numpy as np
import pandas as pd
from multiaxis_eci.analysis.fitspec import FitSpec
from multiaxis_eci.data import N_ITEMS_FILE

spec = FitSpec(K=4, lineage_prior=True, lineage_bm=True)
data, _, _ = spec.load_data()
names = np.array(data.blookup.sort_values("benchmark_idx")["benchmark"])   # the order load_binomial_n uses
df = pd.DataFrame({"benchmark": names[data.bench_idx],
                   "score": np.asarray(data.scores, float)})
tab = pd.read_csv(N_ITEMS_FILE)
tab = tab[tab.benchmark.isin(df.benchmark.unique())]
tol = 5e-4
rows = []
for _, t in tab.iterrows():
    s = df.loc[df.benchmark == t.benchmark, "score"].to_numpy()
    row = dict(benchmark=t.benchmark, scoring=t.scoring, n_items=t.n_items,
               verification=t.verification, n_scores=len(s))
    if t.scoring == "items":
        N = float(t.n_items)
        on = np.abs(s * N - np.round(s * N)) <= tol * N
        row.update(on_grid=on.mean(), chance=min(1.0, 2 * tol * N))
    rows.append(row)
out = pd.DataFrame(rows).sort_values(["scoring", "benchmark"])
out.to_csv(Path(__file__).with_name("grid_check.csv"), index=False)
it = out[out.scoring == "items"]
print("benchmarks in scope:", len(out), out.scoring.value_counts().to_dict())
print("item-scored, verification stamp:", it.verification.fillna("none").value_counts().to_dict())
print("observations on item-scored benchmarks:", int(it.n_scores.sum()), "of", len(df))
good = it[(it.chance < 0.5)]
print(f"testable (chance on-grid < 0.5): {len(good)} / {len(it)}")
print("  on-grid >= 0.9:", int((good.on_grid >= 0.9).sum()), " 0.5-0.9:",
      int(((good.on_grid >= 0.5) & (good.on_grid < 0.9)).sum()), " < 0.5:", int((good.on_grid < 0.5).sum()))
print(good.sort_values("on_grid")[["benchmark", "n_items", "verification", "n_scores", "on_grid", "chance"]]
      .head(15).to_string(index=False))
