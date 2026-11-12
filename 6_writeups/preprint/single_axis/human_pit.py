"""Human scores under the single-axis fit with humans inside (canonical_humanmerge).

Reads the fit's LOO-PIT table (the position of each observed score in its leave-one-out predictive
distribution) and counts the scores that fall in the bottom or top 5 %, for human rows and for AI
rows. The table leaves out the scores of exactly 0 or 1 (86 of the fit's 4,410, 2 of them human), so the
counts are over 38 of the 40 human scores and 4,286 of the 4,370 AI scores. Writes human_pit.csv (one
row per human score) beside this file.

    python 6_writeups/preprint/single_axis/human_pit.py
"""
from pathlib import Path

import pandas as pd

FIT = Path(__file__).resolve().parents[3] / "5_outputs/data20261019/canonical_humanmerge"
pit = pd.read_csv(FIT / "pit.csv")
humans = set(pd.read_csv(FIT / "human_groups_eci.csv")["name"])
is_human = pit["model"].isin(humans)
extreme = (pit["pit"] < 0.05) | (pit["pit"] > 0.95)
reliable = pit["pareto_k"] <= 0.7          # the leave-one-out estimate is trusted up to this k
for label, rows in (("human", is_human), ("AI", ~is_human)):
    print(f"{label}: {int((extreme & rows).sum())} / {int(rows.sum())} scores with LOO-PIT below 0.05 or "
          f"above 0.95; {int((extreme & rows & reliable).sum())} / {int((rows & reliable).sum())} "
          "among the scores with Pareto k at most 0.7")
out = pit[is_human].sort_values("pit")[["model", "benchmark", "score", "pit", "pareto_k"]]
out.to_csv(Path(__file__).with_name("human_pit.csv"), index=False)
print(out.to_string(index=False))
