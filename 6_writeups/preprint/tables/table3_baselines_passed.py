"""Table 3: human baselines against the frontier, per axis, for modes A and C of the multi-axis model.

Reads flagship_analysis/crossings_mode{A,C}.csv (the frontier trend's crossing of each placed human
baseline) and each mode's human_placement.json (the human scores a baseline rests on, per axis).
A baseline counts as passed when the probability that the frontier is above it today is at least
0.95; otherwise the row gives the median crossing date and its 80 % interval. No date is printed
for a passed baseline, so no date before the data appears.

Writes table3_baselines_passed.csv beside this file.

    python 6_writeups/preprint/tables/table3_baselines_passed.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "2_model"))
from multiaxis_eci.analysis import FLAGSHIP_TRACE, secondary_frame_dir  # noqa: E402

HERE, P_PASSED = Path(__file__).parent, 0.95
frames = {"A": FLAGSHIP_TRACE.parent, "C": secondary_frame_dir(FLAGSHIP_TRACE)}
rows = []
for mode, frame in frames.items():
    cx = pd.read_csv(HERE.parent / "flagship_analysis" / f"crossings_mode{mode}.csv")
    placed = {a["axis"]: a for a in json.loads((frame / "human_placement.json").read_text())["axes"]}
    for r in cx.itertuples():
        scores = sum(c["tier"] == r.tier for c in placed[r.axis]["consistency"])
        passed = r.p_passed_now >= P_PASSED
        rows.append(dict(
            mode=mode, chains=r.n_chains, axis=r.axis_title.split(": ", 1)[1], baseline=r.tier,
            human_scores=scores, probability_passed=round(r.p_passed_now, 2),
            result="passed" if passed else r.crossover_date_median[:7],
            interval_80="" if passed else f"{r.hdi80_low[:7]} to {r.hdi80_high[:7]}"))
out = pd.DataFrame(rows)
out.to_csv(HERE / "table3_baselines_passed.csv", index=False)
print(out.to_string(index=False))
print("\nscores per axis:", out.groupby(["mode", "axis"], sort=False).human_scores.sum().to_dict())
