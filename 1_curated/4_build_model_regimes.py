"""Reasoning regime of every model of 0_input/models.csv -> 1_curated/model_regimes.csv.

The regime is decided in code (`analysis.regimes`: the run's own mode, then its release's
evidence or the name patterns of `config`, the bare run of a hybrid release left
undetermined). This table is the REVIEWED snapshot of that decision:
tests/test_model_metadata.py fails as soon as the code classifies a model differently from the
table, a new model included, so a pattern that catches a new release by accident (a Flash-Lite
under `gemini-2.5`) is read before it reaches a forecast. After a sync, run this script, read
the diff of the table (`git diff 1_curated/model_regimes.csv`), fix the rules where a row is
wrong, and commit the table with the data.

  python 1_curated/4_build_model_regimes.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "2_model"))

from multiaxis_eci.analysis.regimes import regime_table  # noqa: E402

OUT = ROOT / "1_curated" / "model_regimes.csv"


def main() -> pd.DataFrame:
    out = regime_table()
    OUT.write_text(out.to_csv(index=False))
    print(f"wrote {OUT}: {out['regime'].value_counts().to_dict()}")
    return out


if __name__ == "__main__":
    main()
