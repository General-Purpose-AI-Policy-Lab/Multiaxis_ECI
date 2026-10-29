"""Every model and benchmark of the synced data is covered by the curated tables a fit or an
analysis reads, so a sync cannot slip a new entry past them. The lineage map and the regimes have
their own tests (test_lineage_map.py, test_model_metadata.py); the full audit of a sync, with the
judgement calls, is 1_curated/0_audit_sync.py and the `data-refresh` skill."""
import subprocess
import sys
from pathlib import Path

import pandas as pd

from multiaxis_eci.config import HUMAN_ORDER_MERGED

ROOT = Path(__file__).resolve().parents[1]
SCORES = pd.read_csv(ROOT / "0_input" / "all_scores_flat.csv")
HUMANS = pd.read_csv(ROOT / "0_input" / "human_baselines.csv")


def test_every_benchmark_has_an_item_count():
    """Without a row the fit silently falls back to the Beta likelihood for that benchmark."""
    have = set(pd.read_csv(ROOT / "1_curated" / "benchmark_n_items.csv")["benchmark"])
    missing = sorted(set(SCORES["benchmark"]) - have)
    assert not missing, f"add to 1_curated/benchmark_n_items.csv (N or `continuous`): {missing}"


def test_every_model_has_an_openness_label():
    """The frontier analyses refuse an unlabelled candidate; label every model at sync time."""
    lab = pd.read_csv(ROOT / "1_curated" / "model_openness.csv").set_index("model_version")["openness"]
    bad = sorted(m for m in set(SCORES["model_version"]) if lab.get(m) not in ("open", "closed"))
    assert not bad, ("run 1_curated/3_build_model_openness.py --pipeline <checkout of the synced "
                     f"run>, then research what is left (model_openness_overrides.csv): {bad}")


def test_every_human_tier_is_in_the_order():
    """A stray tier name forks a tier and silently loses the ordered-human prior."""
    stray = sorted(set(HUMANS["group"].str.strip()) - set(HUMAN_ORDER_MERGED))
    assert not stray, f"human_baselines.csv tiers outside config.HUMAN_ORDER_MERGED: {stray}"


def test_every_human_benchmark_is_scored():
    """A baseline on a benchmark with no model score places nothing."""
    orphans = sorted(set(HUMANS["benchmark"]) - set(SCORES["benchmark"]))
    assert not orphans, f"human baselines on benchmarks without scores: {orphans}"


def test_sync_audit_has_no_blocking_finding(tmp_path):
    """The audit of the current sync against the previous generation reports no BLOCK."""
    out = subprocess.run([sys.executable, str(ROOT / "1_curated" / "0_audit_sync.py"),
                          "--out", str(tmp_path / "audit.md")],
                         cwd=ROOT, capture_output=True, text=True)
    blocks = [ln for ln in (tmp_path / "audit.md").read_text().split("## BLOCK", 1)[-1]
              .split("\n## ", 1)[0].splitlines() if ln.startswith("- ")] if out.returncode else []
    assert out.returncode == 0, "BLOCK findings:\n" + "\n".join(blocks) + out.stderr[-2000:]


def test_human_axis_content_is_well_formed():
    """One judgement per (axis title, benchmark, tier), keep in yes / no, every row six fields (an
    unquoted comma in a reason shifts the columns and stops the placement); a tier row is an
    exclusion of a pair that stays judged as a whole."""
    import csv
    with open(ROOT / "1_curated" / "human_axis_content.csv", newline="") as f:
        rows = list(csv.reader(f))
    bad = [i + 1 for i, r in enumerate(rows) if len(r) != 6]
    assert not bad, f"human_axis_content.csv lines with a field count other than 6: {bad}"
    df = pd.read_csv(ROOT / "1_curated" / "human_axis_content.csv", keep_default_na=False)
    assert set(df["keep"]) <= {"yes", "no"}
    tiered = df[df.tier != ""]
    assert (tiered.keep == "no").all(), "a tier row only takes a tier's score off"
    whole = set(zip(df.axis_title[df.tier == ""], df.benchmark[df.tier == ""]))
    assert set(zip(tiered.axis_title, tiered.benchmark)) <= whole
    dup = df[df.duplicated(["axis_title", "benchmark", "tier"], keep=False)]
    assert dup.empty, f"judged twice: {dup[['axis_title', 'benchmark']].values.tolist()}"
