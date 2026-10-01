"""ECI-H along every lineage edge of 1_curated/lineage_map.csv, read off a canonical fit.

Prints the edges by drop, the flagged ones first (analysis.lineage_check: at least MIN_DROP points
and credible on both sides), and marks those already reviewed in
1_curated/lineage_reviewed_exceptions.csv. tests/test_lineage_map.py fails on a flagged edge
that is not reviewed: fix the map, or investigate and list it with its reason.

  python 4_diagnostics/lineage_regressions.py [--fit 5_outputs/<generation>/canonical_humanmerge]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "2_model"))

from multiaxis_eci import config  # noqa: E402
from multiaxis_eci.analysis.lineage_check import MIN_DROP, comparisons, regressions  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default=str(config.RESULTS_DIR / "canonical_humanmerge"))
    ap.add_argument("--top", type=int, default=25, help="unflagged edges to show, worst first")
    args = ap.parse_args()
    eci = pd.read_csv(Path(args.fit) / "all_models_eci.csv")
    exc = pd.read_csv(ROOT / "1_curated" / "lineage_reviewed_exceptions.csv",
                      keep_default_na=False, dtype=str)
    ok = set(map(tuple, exc.loc[exc["check"] == "regression",
                                ["chain", "node", "variant", "raw_strings"]].values))
    pd.set_option("display.width", 250)
    cols = ["chain", "parent", "child", "variant", "parent_run", "child_run", "parent_eci",
            "child_eci", "drop", "n_parent", "n_child"]
    flagged = regressions(eci)
    flagged["reviewed"] = [(r.chain, r.child, r.variant, f"{r.parent_run}|{r.child_run}") in ok
                           for r in flagged.itertuples()]
    print(f"{args.fit}: flagged (drop >= {MIN_DROP:g} ECI-H, credible): {len(flagged)}, "
          f"{int((~flagged['reviewed']).sum())} not reviewed")
    print(flagged[cols + ["reviewed"]].round(1).to_string(index=False))
    allc = comparisons(eci).sort_values("drop", ascending=False)
    print(f"\nLargest drops, flagged or not ({args.top}):")
    print(allc[cols].head(args.top).round(1).to_string(index=False))


if __name__ == "__main__":
    main()
