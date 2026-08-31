"""Read-only checker for the curated benchmark chance floors.

`1_data/curated/benchmark_lower_bounds.csv` is the hand-maintained ground truth
(one reviewed row per benchmark: value, reason, source URL — edit it directly).
This script validates it against the fit data and PRINTS three review tables:

  1. coverage — every fit benchmark must have a floor row (a missing row is an
     inert 0.0 floor at fit time, warned by data.load_benchmark_floors);
  2. format consistency — a floor should match the format its own reason states
     (e.g. a "4-option MCQ" reason implies 0.25);
  3. below-floor tally — who scores under each floor (human / old-weak /
     frontier). A frontier model below a floor usually means the floor is wrong
     (though not always: at-chance performance on a hard task is legitimate,
     e.g. VPCT's three buckets).

By default it writes nothing. `--write-clips` refreshes
`1_data/curated/benchmark_score_clips.csv` — the reviewed row-level list of
below-floor scores that `data.clip_scores_to_floors` APPLIES at fit time
(same draft-then-review convention as `lineage_map.csv`): run it after a data
or floor change, review the diff, commit. The floors file itself is never
written; it was originally bootstrapped from a hand-built spreadsheet via a
generator layer (drafts/corrections/aliases) retired on 2026-07-13 after a
source-by-source review of all 85 floors made the curated file authoritative.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from multiaxis_eci import config
from multiaxis_eci.data import BENCHMARK_CLIPS_FILE, BENCHMARK_FLOORS_FILE, load_eci_data

PROCESSED = ROOT / "data" / "processed" / "benchmarks_merged.csv"
HUMAN_FILE = ROOT / "data" / "curated" / "human_baselines.csv"

# a model dated on/after this counts as recent for the below-floor who-split
FRONTIER_DATE = "2025-06-01"


def expected_from_reason(reason):
    """Floor implied by the stated format, or nan if not a clean rule."""
    r = str(reason).lower()
    if "binary" in r or "yes/no" in r or "yes / no" in r or "true/false" in r or "true or false" in r:
        return 0.5
    for n in (10, 6, 5, 4, 3, 2):
        if f"{n}-option" in r or f"{n} option" in r or f"{n}-label" in r or f"{n}-choice" in r:
            return round(1.0 / n, 3)
    if "three buckets" in r:
        return round(1.0 / 3, 3)
    return np.nan


def format_check(floors):
    """Flag rows where the stated format implies a different floor."""
    out = []
    for _, r in floors.iterrows():
        exp = expected_from_reason(r["reason"])
        ok = np.isnan(exp) or abs(exp - r["lower_bound"]) <= 1e-3
        out.append("" if ok else f"reason implies {exp:.3f}")
    floors = floors.copy()
    floors["format_flag"] = out
    return floors


def score_table():
    """All (benchmark, model, score) rows: machine + human, with era + sota."""
    m = pd.read_csv(PROCESSED)[["model_version", "benchmark", "score", "release_date"]].copy()
    m["is_human"] = False
    h = pd.read_csv(HUMAN_FILE).rename(columns={"group": "model_version"})[["model_version", "benchmark", "score"]].copy()
    h["release_date"] = np.nan
    h["is_human"] = True
    df = pd.concat([m, h], ignore_index=True)
    df["date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["sota"] = df["model_version"].isin(set(config.SOTA_MODELS))
    return df


def who(row):
    if row["is_human"]:
        return "human"
    if row["sota"] or (pd.notna(row["date"]) and row["date"] >= pd.Timestamp(FRONTIER_DATE)):
        return "frontier"
    return "old/weak"


def below_floor_audit(floors, scores):
    """Per benchmark: how many scores fall below the floor and who."""
    fmap = dict(zip(floors["benchmark"], floors["lower_bound"]))
    rows = []
    for b, floor in fmap.items():
        s = scores[scores["benchmark"] == b]
        n = len(s)
        if n == 0 or np.isnan(floor) or floor <= 0.0:
            rows.append((b, floor, n, 0, 0.0, np.nan, 0, 0, 0, "valid" if floor == 0 else "no-data"))
            continue
        bel = s[s["score"] < floor]
        kinds = bel.apply(who, axis=1)
        n_h = int((kinds == "human").sum())
        n_o = int((kinds == "old/weak").sum())
        n_f = int((kinds == "frontier").sum())
        frac = len(bel) / n
        verdict = "suspect" if (n_f > 0 or frac > 0.5) else "valid"
        rows.append((b, floor, n, len(bel), round(frac, 3),
                     round(float(bel["score"].min()), 3) if len(bel) else np.nan,
                     n_h, n_o, n_f, verdict))
    cols = ["benchmark", "floor", "n_scores", "n_below", "frac_below", "min_below",
            "n_human_below", "n_oldweak_below", "n_frontier_below", "verdict"]
    return pd.DataFrame(rows, columns=cols)


def clip_rows(floors, scores):
    """One row per below-floor observation, with the clip-to-floor transform.

    original_score is written at full precision — data.clip_scores_to_floors
    matches it exactly against the loaded data to detect drift.
    """
    fmap = dict(zip(floors["benchmark"], floors["lower_bound"]))
    out = []
    for _, s in scores.iterrows():
        floor = fmap.get(s["benchmark"], np.nan)
        if np.isnan(floor) or floor <= 0.0 or s["score"] >= floor:
            continue
        out.append((s["benchmark"], s["model_version"], bool(s["is_human"]),
                    float(s["score"]), float(floor), float(floor),
                    s["release_date"] if pd.notna(s["release_date"]) else "",
                    who(s)))
    cols = ["benchmark", "model_version", "is_human", "original_score",
            "lower_bound", "clipped_score", "release_date", "taker_class"]
    return pd.DataFrame(out, columns=cols).sort_values(["benchmark", "original_score"])


def main(write_clips=False):
    data = load_eci_data(include_all_benchmarks=True)
    fit_benches = list(data.blookup["benchmark"].values)
    floors = pd.read_csv(BENCHMARK_FLOORS_FILE)

    pd.set_option("display.max_rows", None, "display.width", 200)

    missing = sorted(set(fit_benches) - set(floors["benchmark"]))
    extra = sorted(set(floors["benchmark"]) - set(fit_benches))
    print(f"\n=== coverage: {len(fit_benches)} fit benchmarks, "
          f"{len(floors)} curated rows ===")
    print(f"  fit benchmarks with NO floor row (inert 0.0 at fit time): "
          f"{missing if missing else 'none'}")
    print(f"  curated rows for benchmarks not in the fit: "
          f"{extra if extra else 'none'}")

    floors = format_check(floors)
    flagged = floors[floors["format_flag"] != ""]
    print(f"\n=== format-consistency flags ({len(flagged)}) ===")
    print(flagged[["benchmark", "lower_bound", "reason", "format_flag"]].to_string(index=False) if len(flagged) else "  none")

    scores = score_table()
    audit = below_floor_audit(floors, scores)
    have_below = audit[audit["n_below"] > 0].sort_values("verdict", ascending=False)
    print(f"\n=== BELOW-FLOOR AUDIT (benchmarks with any below-floor score: {len(have_below)}) ===")
    print(have_below.to_string(index=False) if len(have_below) else "  none")

    suspect = audit[audit["verdict"] == "suspect"]
    print(f"\n=== SUSPECT floors (frontier below, or >50% below) — review against the source ({len(suspect)}) ===")
    print(suspect.to_string(index=False) if len(suspect) else "  none")

    clips = clip_rows(floors, scores)
    print(f"\n=== SCORES UNDER THEIR FLOOR — what the --floors clip raises ({len(clips)} rows) ===")
    print(clips.to_string(index=False) if len(clips) else "  none")

    if write_clips:
        clips.to_csv(BENCHMARK_CLIPS_FILE, index=False)
        print(f"\nwrote {BENCHMARK_CLIPS_FILE} — review the diff, then commit")

    return floors, audit, clips


if __name__ == "__main__":
    main(write_clips="--write-clips" in sys.argv)
