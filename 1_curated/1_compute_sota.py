"""Recompute the SOTA list and write 1_curated/sota_families.txt.

SOTA = (world frontier records of the recent era) ∪ (every family within NEAR_ECI points of
the best model of each organisation present in that frontier). The file lists families (a
release: base model plus snapshot, `data.model_family`), so every reasoning effort of a SOTA
release is shown; the canonical fit's SOTA table keeps the best effort of each.
Read off the canonical K=1 fit of the current data generation (`all_models_eci.csv`,
`timeline.csv`): no separate sampling. The list drives the timelines, the country frontier's
record candidates and the SOTA table; it decides nothing about what a fit sees, so reading it
off the canonical fit is not circular. Candidates need MIN_OBS observations, Epoch's rule for a
record-setter and the repository's low-observation threshold (user decisions 2026-09-09).

  python 1_curated/1_compute_sota.py [--results-dir DIR]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "2_model"))

from multiaxis_eci import config  # noqa: E402
from multiaxis_eci.data import MODELS_FILE, model_family  # noqa: E402

WINDOW_MONTHS = 24        # "recent era": records set in the last 24 months before the newest release
MIN_OBS = config.LOW_OBS_THRESHOLD   # Epoch's >= 4-benchmark rule for record-setters
NEAR_ECI = 10.0           # a family counts as top-line when within this many ECI points of its organisation's best
OUT = ROOT / "1_curated" / "sota_families.txt"


def frontier_records(d: pd.DataFrame, since: pd.Timestamp) -> list[str]:
    """Running-max record-setters of median ECI by release date, kept when released after `since`."""
    run, out = -np.inf, []
    for _, r in d.sort_values(["release_date", "mean"]).iterrows():
        if r["mean"] > run + 1e-9:
            run = r["mean"]
            if r["release_date"] >= since:
                out.append(r["name"])
    return out


def compute(results_dir: Path) -> pd.DataFrame:
    """One row per selected family: best effort, ECI, release date, organisation, why."""
    eci = pd.read_csv(results_dir / "all_models_eci.csv")
    tl = pd.read_csv(results_dir / "timeline.csv")
    humans = set(pd.read_csv(results_dir / "human_groups.csv")["name"])
    models = pd.read_csv(MODELS_FILE, dtype=str).fillna("")
    org = dict(zip(models["model_version"], models["organization"], strict=True))

    d = eci.merge(tl.loc[tl["kind"] == "model", ["name", "release_date"]], on="name")
    d = d[~d["name"].isin(humans) & (d["n_obs"] >= MIN_OBS)].copy()
    d["release_date"] = pd.to_datetime(d["release_date"])
    d["family"] = d["name"].map(model_family)
    d["org"] = d["name"].map(org).fillna("")

    cutoff = d["release_date"].max() - pd.DateOffset(months=WINDOW_MONTHS)
    records = set(frontier_records(d, cutoff))
    frontier_orgs = {org.get(m, "") for m in records} - {""}

    # Best effort per family, then the two rules.
    fam = (d.sort_values("mean", ascending=False).drop_duplicates("family")
             .rename(columns={"name": "best_model"}))
    fam["release_date"] = fam["family"].map(d.groupby("family")["release_date"].min())
    fam["is_record"] = fam["family"].isin({model_family(m) for m in records})
    org_best = fam.groupby("org")["mean"].transform("max")
    fam["is_top_line"] = fam["org"].isin(frontier_orgs) & (fam["mean"] >= org_best - NEAR_ECI)
    out = fam[fam["is_record"] | fam["is_top_line"]].sort_values("release_date", ascending=False)
    out.attrs["cutoff"] = cutoff
    out.attrs["frontier_orgs"] = sorted(frontier_orgs)
    return out[["best_model", "family", "org", "release_date", "mean", "n_obs", "is_record", "is_top_line"]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=str(config.RESULTS_DIR / "canonical"),
                    help="a canonical fit folder (default: the current data generation's)")
    args = ap.parse_args()
    out = compute(Path(args.results_dir))
    OUT.write_text("# SOTA families (release = base model + snapshot), newest first; written by "
                   "1_curated/1_compute_sota.py, do not edit\n" + "\n".join(out["family"]) + "\n")
    print(f"records since {out.attrs['cutoff'].date()}; frontier organisations: "
          f"{', '.join(out.attrs['frontier_orgs'])}")
    print(f"{int(out['is_record'].sum())} record families, {int(out['is_top_line'].sum())} top-line "
          f"families, {len(out)} entries -> {OUT.relative_to(ROOT)}\n")
    for _, r in out.iterrows():
        why = "+".join(t for t, on in (("record", r.is_record), ("top-line", r.is_top_line)) if on)
        print(f"  {r.release_date.date()}  {r.best_model:40s} {r.org:18s} ECI {r['mean']:6.1f}"
              f"  n={int(r.n_obs):3d}  [{why}]")


if __name__ == "__main__":
    main()
