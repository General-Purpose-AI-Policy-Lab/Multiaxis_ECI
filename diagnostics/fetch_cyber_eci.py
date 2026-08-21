"""Fetch Epoch's cyber-ECI benchmark table into a curated additive file.

Epoch runs a second, cyber-specific ECI on its own data path
(`epoch.ai/data/cyber/`) with 20 benchmarks that are NOT in the benchmark ZIP
or the benchmarks hub, so the pipeline notebook never sees them. This script
pulls that table, keeps the benchmarks whose scores still discriminate, maps
Epoch's display names onto our versioned model IDs, and writes
`data/curated/cyber_benchmarks.csv` in the processed-file schema.

The output is a curated file, not a pipeline stage: `data.py` appends it at fit
time behind `--cyber`. Epoch states several of these values were read off
published plots, a weaker provenance than the ZIP feeds. The file records no
fetch date, so staleness against a refreshed processed table is not detectable
from the file alone.

Usage:  python -m diagnostics.fetch_cyber_eci [--out PATH] [--dry-run]
"""
from __future__ import annotations

import argparse
import io
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data import PROCESSED_DIR, CURATED_DIR, _effort_base  # noqa: E402

CYBER_BENCHMARKS = "https://epoch.ai/data/cyber/eci_benchmarks.csv"
CYBER_SCORES = "https://epoch.ai/data/cyber/eci_scores.csv"
GENERAL_BENCHMARKS = "https://epoch.ai/data/eci_benchmarks.csv"

# Discrimination gates. A benchmark whose scores pile against either bound
# carries almost no information about ability differences.
SATURATED_MEAN = 0.75      # mean score at or above this
SATURATED_P90 = 1 / 3      # or this share of scores >= 0.90
FLOOR_MEAN = 0.20          # mean score below this
FLOOR_ZEROS = 1 / 3        # and this share of scores exactly 0

# Epoch's cyber table carries display names only, with no effort or context
# setting, so a mapped ID re-targets the `_unknown` variant where our table has
# one. These overrides cover the display names Epoch's general ECI does not
# link; each matches our table on release date exactly.
MANUAL_ALIASES = {
    "Claude Mythos Preview (Early)": "claude-mythos-preview-early",
    "GPT-3.5 Turbo": "gpt-3.5-turbo-0613",
    "GPT-4o": "gpt-4o-2024-05-13",
    "GPT-5.1 Codex Max": "gpt-5.1-codex-max",
    "GPT-5.2 Codex": "gpt-5.2-codex",
    "Gemini 1.5 Pro": "gemini-1.5-pro-001",
    "Grok 4.1": "grok-4-1",
    "MiniMax M2.7": "MiniMax-M2.7",
}

# Deliberately not mapped. Mythos 5 is NOT aliased to claude-fable-5: they
# differ in deployed safeguards.
DROP_MODELS = {
    "GPT-5.4-Cyber": "cyber-specialised variant, not a general test-taker",
    "GPT-5.5-Cyber": "cyber-specialised variant, not a general test-taker",
    "Claude Mythos 5": "not in our table; cyber-only test-taker",
    "Claude Mythos Preview (April)": "not in our table; cyber-only test-taker",
}

CATEGORY = "Agentic Computer Use"  # matches the two cyber benchmarks we already carry


def _read_csv(url: str) -> pd.DataFrame:
    with urllib.request.urlopen(url, timeout=120) as r:
        return pd.read_csv(io.BytesIO(r.read()))


def classify(cy: pd.DataFrame) -> pd.DataFrame:
    """Per-benchmark discrimination verdict."""
    g = cy.groupby("benchmark").performance.agg(
        n="size", mean="mean",
        p90=lambda s: (s >= 0.90).mean(),
        zeros=lambda s: (s == 0).mean())
    saturated = (g["mean"] >= SATURATED_MEAN) | (g["p90"] >= SATURATED_P90)
    floored = (g["mean"] < FLOOR_MEAN) & (g["zeros"] >= FLOOR_ZEROS)
    g["verdict"] = "keep"
    g.loc[floored, "verdict"] = "floor-bound"
    g.loc[saturated, "verdict"] = "saturated"   # saturation wins if both fire
    return g


def build_alias_map(cy: pd.DataFrame) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """cyber display name -> our model_version, preferring the `_unknown` variant."""
    ours = set(pd.read_csv(PROCESSED_DIR / "benchmarks_merged.csv")["model_version"])
    general = _read_csv(GENERAL_BENCHMARKS)
    scores = _read_csv(CYBER_SCORES)
    epoch_mv = general.drop_duplicates("Model").set_index("Model")["model_version"].to_dict()
    to_general = scores.set_index("Model")["general_model_name"].to_dict()

    alias, unresolved = {}, []
    for name in sorted(cy["Model"].unique()):
        if name in DROP_MODELS:
            continue
        if name in MANUAL_ALIASES:
            target = MANUAL_ALIASES[name]
            if target not in ours:
                unresolved.append((name, f"manual alias {target!r} missing from our table"))
                continue
            alias[name] = target
            continue
        general_name = to_general.get(name)
        mv = epoch_mv.get(general_name) if isinstance(general_name, str) else None
        if not isinstance(mv, str):
            unresolved.append((name, "no link in Epoch's general ECI and no manual alias"))
            continue
        base = _effort_base(mv)
        target = next((c for c in (f"{base}_unknown", base, mv) if c in ours), None)
        if target is None:
            unresolved.append((name, f"Epoch id {mv!r} has no counterpart in our table"))
            continue
        alias[name] = target
    return alias, unresolved


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=CURATED_DIR / "cyber_benchmarks.csv")
    ap.add_argument("--dry-run", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    cy = _read_csv(CYBER_BENCHMARKS)
    cy["date"] = pd.to_datetime(cy["date"], errors="coerce")
    print(f"Epoch cyber ECI: {len(cy)} obs / {cy.Model.nunique()} models "
          f"/ {cy.benchmark.nunique()} benchmarks")

    verdicts = classify(cy)
    print("\nDiscrimination gates "
          f"(saturated: mean >= {SATURATED_MEAN} or >= {SATURATED_P90:.0%} of scores >= 0.90; "
          f"floor-bound: mean < {FLOOR_MEAN} and >= {FLOOR_ZEROS:.0%} exact zeros)")
    print(verdicts.round(3).sort_values(["verdict", "n"], ascending=[True, False]).to_string())

    keep = set(verdicts.index[verdicts.verdict == "keep"])
    ours_bench = set(pd.read_csv(PROCESSED_DIR / "benchmarks_merged.csv")["benchmark"])
    already = keep & ours_bench
    new_bench = keep - ours_bench
    if already:
        print(f"\nalready in our table, left to the pipeline: {sorted(already)}")
    print(f"new benchmarks to add ({len(new_bench)}): {sorted(new_bench)}")

    alias, unresolved = build_alias_map(cy)
    print(f"\nmodels mapped: {len(alias)}")
    for name, why in DROP_MODELS.items():
        print(f"  dropped  {name:32s} {why}")
    for name, why in unresolved:
        # An unmapped model only costs rows if it scored on a kept benchmark.
        lost = sorted(set(cy.loc[cy.Model == name, "benchmark"]) & new_bench)
        cost = f"costs {len(lost)} rows: {lost}" if lost else "no rows on kept benchmarks"
        print(f"  UNMAPPED {name:32s} {why} ({cost})")

    out = cy[cy.benchmark.isin(new_bench) & cy.Model.isin(alias)].copy()
    out["model_version"] = out["Model"].map(alias)

    # Two display names can land on one model_version. Resolve with the max, the
    # pipeline's DEDUP_POLICY, and say so rather than dropping a row silently.
    dup = out.duplicated(["model_version", "benchmark"], keep=False)
    if dup.any():
        print(f"\n  {int(dup.sum())} rows collide on (model_version, benchmark); "
              f"resolving with max (pipeline DEDUP_POLICY)")
        for (mv, b), grp in out[dup].groupby(["model_version", "benchmark"]):
            print(f"    {mv} / {b}: {sorted(grp.Model)} -> {grp.performance.max():.3f}")
        out = (out.sort_values("performance")
                  .drop_duplicates(["model_version", "benchmark"], keep="last"))

    tidy = pd.DataFrame({
        "model_version": out["model_version"],
        "score": out["performance"],
        "release_date": out["date"].dt.strftime("%Y-%m-%d"),
        "organization": "",          # filled from our table at load time
        "benchmark": out["benchmark"],
        "stderr": pd.NA,             # carried but unused by the likelihood
        "source": "Epoch AI cyber ECI (epoch.ai/data/cyber)",
        "category": CATEGORY,
    }).sort_values(["benchmark", "model_version"]).reset_index(drop=True)

    assert tidy["score"].between(0, 1).all(), "scores must be proportions"
    print(f"\n{len(tidy)} rows / {tidy.model_version.nunique()} models "
          f"/ {tidy.benchmark.nunique()} benchmarks")
    print(tidy.groupby("benchmark").size().to_string())

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return 0
    tidy.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
