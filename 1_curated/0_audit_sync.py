"""Audit a sync: what the new data generation brings, whether every table downstream covers it,
and what looks unexpected. Run it right after `python -m multiaxis_eci sync`, then again after the
builders, until it reports no BLOCK; the procedure around it is the `data-refresh` skill
(.claude/skills/data-refresh/SKILL.md).

The reference is the previous data generation: 0_input/ at HEAD when the sync is not committed
yet, else at the parent of the last commit that changed 0_input/provenance.json (`--ref` names
another revision). Every finding has a level:

  BLOCK   something a fit would get wrong or silently skip: fix it before any fit
  DECIDE  a judgement call for the user (a benchmark entering the index, a human baseline at
          1.0, an openness label inside its 30-day window...): ask, record the answer
  CHECK   unexpected, to investigate; usually explained (a revision, a genuine jump)
  INFO    what changed, for the record

The report goes to stdout and to 5_outputs/data<YYYYMMDD>/sync_audit.md (commit it with the
sync: it is the trace of what was looked at). Exit status 1 while any BLOCK remains.

  python 1_curated/0_audit_sync.py [--ref REV] [--pipeline PATH] [--out FILE]
"""
from __future__ import annotations

import argparse
import difflib
import io
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "2_model"))

from multiaxis_eci import config  # noqa: E402
from multiaxis_eci.config import HUMAN_ORDER_MERGED  # noqa: E402
from multiaxis_eci.data import model_family  # noqa: E402

INPUT = Path("0_input")
CURATED = ROOT / "1_curated"
LEVELS = ("BLOCK", "DECIDE", "CHECK", "INFO")
EFFORT_ORDER = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]
EFFORT_RE = re.compile(r"_(" + "|".join(EFFORT_ORDER) + r")$")

# Thresholds of the CHECK rules (scores are shares in [0, 1]).
THIN_BENCHMARK = 5        # fewer test-takers than this: thin benchmark
FRONTIER_JUMP = 0.15      # a benchmark's best score rising by more than this since the last sync
REVISION = 0.05           # an existing (model, benchmark) score moving by more than this
EFFORT_INVERSION = 0.15   # a higher effort of one release scoring this much below a lower one
LINEAGE_DROP = 0.20       # a release scoring this much below its lineage parent's best run
CHANCE_GAP = 0.15         # floor 0 but even the weakest tenth of models scores above this
CHANCE_MIN_TAKERS = 15    # ... on a panel this large (a frontier-only panel starts high anyway)


@dataclass
class Finding:
    level: str
    topic: str
    subject: str
    message: str


class Audit:
    def __init__(self):
        self.findings: list[Finding] = []

    def add(self, level, topic, subject, message):
        self.findings.append(Finding(level, topic, subject, message))

    def count(self, level):
        return sum(f.level == level for f in self.findings)


# ── inputs ────────────────────────────────────────────────────────────────────────────────────

def git(*args, cwd=ROOT) -> str:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True).stdout


def resolve_ref(ref: str | None) -> str:
    if ref:
        return ref
    if subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(INPUT / "provenance.json")],
                      cwd=ROOT).returncode:
        return "HEAD"                                   # synced, not committed yet
    last = git("log", "-n1", "--format=%H", "--", str(INPUT / "provenance.json")).strip()
    return f"{last}~1"


def read_at(ref: str, path: Path) -> pd.DataFrame:
    out = subprocess.run(["git", "show", f"{ref}:{path.as_posix()}"], cwd=ROOT,
                         capture_output=True, text=True)
    if out.returncode:
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(out.stdout))


def read_now(path: Path) -> pd.DataFrame:
    p = ROOT / path
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def curated_names(filename: str) -> set[str]:
    p = CURATED / filename
    if not p.exists():
        return set()
    return {ln.split("#", 1)[0].strip() for ln in p.read_text().splitlines()} - {""}


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def effort_of(model_version: str) -> str | None:
    m = EFFORT_RE.search(model_version)
    return m.group(1) if m else None


# ── checks ────────────────────────────────────────────────────────────────────────────────────

def check_provenance(a: Audit, pipeline: Path):
    shown = os.path.relpath(pipeline, ROOT)          # no home path in the committed report
    prov = json.loads((ROOT / INPUT / "provenance.json").read_text())
    commit = prov.get("pipeline_commit", "")
    run = str(prov.get("pipeline_run_date", ""))
    a.add("INFO", "provenance", run or "?", f"pipeline commit {commit[:9]}, run {run}")
    if not pipeline.exists():
        a.add("CHECK", "provenance", shown, "pipeline checkout not found: the openness "
              "builder reads Epoch's metadata from it (pass --pipeline)")
        return
    on_remote = git("branch", "-r", "--contains", commit, cwd=pipeline).strip() if commit else ""
    if not on_remote:
        a.add("CHECK", "provenance", commit[:9], "the synced pipeline commit is on no remote branch: "
              "merge and push the pipeline branch before the refit reaches main, or 0_input names "
              "a commit nobody else can check out")
    runs = sorted(p.name for p in (pipeline / "2_database").glob("[0-9]" * 8))
    if runs and run and runs[-1] != run.replace("-", ""):
        a.add("CHECK", "provenance", shown, f"the pipeline checkout's latest run is "
              f"{runs[-1]}, the synced one {run}: run 3_build_model_openness.py with --pipeline "
              "pointing at the checkout of the synced run, or Epoch's labels of the new models are "
              "missed (they then come out `unknown`)")


def check_benchmarks(a: Audit, ref: str, scores, scores_ref, bench, bench_ref, humans):
    new_b = sorted(set(bench["name"]) - set(bench_ref.get("name", pd.Series(dtype=str))))
    gone = sorted(set(bench_ref.get("name", pd.Series(dtype=str))) - set(bench["name"]))
    for b in gone:
        a.add("CHECK", "benchmark removed", b, "in the previous generation, not in this one: a "
              "retirement or a rename upstream? (a rename needs every curated table renamed too)")
    excluded = curated_names("excluded_benchmarks.txt")
    n_items = read_now(Path("1_curated/benchmark_n_items.csv"))
    have_n = set(n_items.get("benchmark", []))
    old_names = sorted(set(bench_ref.get("name", pd.Series(dtype=str))))
    meta = bench.set_index("name")
    by_b = scores.groupby("benchmark")
    for b in new_b:
        row = meta.loc[b]
        s = by_b.get_group(b)["score"] if b in by_b.groups else pd.Series(dtype=float)
        n_models = int(by_b.get_group(b)["model_version"].nunique()) if b in by_b.groups else 0
        a.add("INFO", "new benchmark", b, f"{n_models} test-takers, category {row.get('category')}, "
              f"floor {row.get('floor')}, ceiling {row.get('ceiling')}, access {row.get('access')}, "
              f"scores {s.min():.3f}-{s.max():.3f}" if len(s) else "no score")
        if pd.isna(row.get("floor")):
            a.add("BLOCK", "new benchmark", b, "no chance floor in the pipeline's benchmarks table")
        if pd.isna(row.get("access")) or not str(row.get("access")).strip():
            a.add("CHECK", "new benchmark", b, "no access class: the frontier analyses cannot place it")
        if pd.isna(row.get("release_date")):
            a.add("CHECK", "new benchmark", b, "no release date in the pipeline")
        if b not in have_n:
            a.add("BLOCK", "new benchmark", b, "no row in 1_curated/benchmark_n_items.csv: the fit "
                  "falls back to the Beta likelihood with a warning; add N (items) or `continuous`")
        if b in excluded:
            a.add("INFO", "new benchmark", b, "listed in excluded_benchmarks.txt: out of the canonical index")
        else:
            a.add("DECIDE", "new benchmark", b, "enters the canonical index (not in "
                  "excluded_benchmarks.txt): confirm it is neither built to be easy for humans nor "
                  "mainly visual (the 2026-09-29 rule)")
        if 0 < n_models < THIN_BENCHMARK:
            a.add("CHECK", "new benchmark", b, f"only {n_models} test-takers")
        if n_models >= CHANCE_MIN_TAKERS and float(row.get("floor") or 0) == 0 \
                and s.quantile(0.1) > CHANCE_GAP:
            a.add("CHECK", "new benchmark", b, f"floor 0 but the weakest tenth scores above "
                  f"{s.quantile(0.1):.2f}: is there a chance level (multiple choice)?")
        close = [o for o in old_names if o != b and (
            difflib.SequenceMatcher(None, norm(o), norm(b)).ratio() >= 0.75
            or (len(norm(o)) >= 6 and (norm(o) in norm(b) or norm(b) in norm(o))))]
        if close:
            a.add("CHECK", "new benchmark", b, f"close to existing {close}: a duplicate series, a "
                  "split or a new version of one of them? (pooling, versions and retirements are "
                  "decisions)")
        for _, h in humans[humans["benchmark"] == b].iterrows():
            a.add("DECIDE", "human baseline", f"{b} / {h['group']}", f"new baseline {h['score']}: "
                  "accept? (comparable conditions, not 1.0 by construction)")
    # every benchmark: the best score against the recorded ceiling, and frontier jumps
    best = scores.groupby("benchmark")["score"].max()
    best_ref = scores_ref.groupby("benchmark")["score"].max() if len(scores_ref) else pd.Series(dtype=float)
    for b, mx in best.items():
        ceil = meta["ceiling"].get(b, np.nan) if b in meta.index else np.nan
        if pd.notna(ceil) and mx > float(ceil) + 1e-9:
            a.add("BLOCK", "ceiling", b, f"best score {mx:.3f} above the recorded ceiling {ceil}")
        if b in best_ref.index and mx - best_ref[b] > FRONTIER_JUMP:
            top = scores[(scores["benchmark"] == b)].nlargest(3, "score")
            who = ", ".join(f"{r.model_version} {r.score:.3f}" for r in top.itertuples())
            a.add("CHECK", "frontier jump", b, f"best score {best_ref[b]:.3f} -> {mx:.3f} ({who}): "
                  "same metric and harness as before? genuine new frontier models?")
    return new_b


def check_clips(a: Audit, scores, bench, humans):
    """The reviewed clips (models and human tiers alike) against the scores under their floor."""
    floors = bench.set_index("name")["floor"]
    s = pd.concat([scores[["benchmark", "model_version", "score"]],
                   humans.rename(columns={"group": "model_version"})[["benchmark", "model_version", "score"]]])
    s = s.assign(floor=s["benchmark"].map(floors))
    below = s[s["score"] < s["floor"] - 1e-12]
    clips = read_now(Path("1_curated/benchmark_score_clips.csv"))
    keys = set(zip(clips.get("benchmark", []), clips.get("model_version", [])))
    missing = [(r.benchmark, r.model_version, r.score, r.floor) for r in below.itertuples()
               if (r.benchmark, r.model_version) not in keys]
    for b, m, v, f in missing:
        a.add("BLOCK", "below floor", f"{b} / {m}", f"score {v:.4f} under the floor {f} with no "
              "reviewed clip: run 4_diagnostics/audit_lower_bounds.py --write-clips, read the diff")
    now = set(zip(below["benchmark"], below["model_version"]))
    for b, m in sorted(keys - now):
        a.add("CHECK", "below floor", f"{b} / {m}", "clip row matching no below-floor score any "
              "more (the score or the floor moved): refresh the clips file")


def check_models(a: Audit, scores, scores_ref, models, models_ref):
    new_m = sorted(set(models["model_version"]) - set(models_ref.get("model_version", [])))
    scored = set(scores["model_version"])
    new_m = [m for m in new_m if m in scored]
    gone = set(models_ref.get("model_version", [])) - set(models["model_version"])
    n_ref = scores_ref.groupby("model_version").size() if len(scores_ref) else pd.Series(dtype=int)
    for m in sorted(gone):
        if n_ref.get(m, 0) >= 3:
            a.add("CHECK", "model removed", m, f"had {n_ref[m]} scores, gone: an alias change upstream?"
                  " (its curated rows - lineage, openness, overrides - may now be orphans)")
    if not new_m:
        return new_m
    a.add("INFO", "new models", f"{len(new_m)}", ", ".join(new_m))
    meta = models.set_index("model_version")
    lineage = read_now(Path("1_curated/lineage_map.csv")).set_index("raw_string")
    lineage_ref = read_at(REF, Path("1_curated/lineage_map.csv"))
    openness = read_now(Path("1_curated/model_openness.csv")).set_index("model_version")["openness"]
    regimes = set(read_now(Path("1_curated/model_regimes.csv"))["model_version"])
    countries = read_now(Path("1_curated/organization_countries.csv"))
    org_country = set(countries["organization"]) if len(countries) else set()
    old_chains = set(lineage_ref.get("chain", pd.Series(dtype=str)).dropna())
    n_obs = scores.groupby("model_version").size()
    for m in new_m:
        r = meta.loc[m]
        if m not in lineage.index:
            a.add("BLOCK", "lineage", m, "not in lineage_map.csv: run 1_curated/2_build_lineage_map.py, "
                  "then pytest tests/test_lineage_map.py")
        else:
            lr = lineage.loc[m]
            if str(lr["in_chain"]) == "no" and n_obs.get(m, 0) >= 3:
                a.add("CHECK", "lineage", m, f"left out of every chain ({n_obs[m]} scores): right?")
            elif pd.notna(lr["chain"]) and lr["chain"] not in old_chains:
                a.add("DECIDE", "lineage", m, f"founds a new chain `{lr['chain']}` (node {lr['node']}): "
                      "a new line, or a node of an existing one?")
        lab = openness.get(m, "missing")
        if lab not in ("open", "closed"):
            a.add("BLOCK", "openness", m, f"openness `{lab}`: research it (weights within 30 days of "
                  "release?) and add a row to model_openness_overrides.csv")
        if m not in regimes:
            a.add("BLOCK", "regime", m, "no row in model_regimes.csv: run 4_build_model_regimes.py, "
                  "read the diff")
        if pd.isna(r.get("publication_date")):
            a.add("CHECK", "release date", m, f"undated ({n_obs.get(m, 0)} scores): left out of the "
                  "dated views, the timelines and the frontier analyses")
        if pd.isna(r.get("organization")):
            a.add("CHECK", "organization", m, "no organization")
        elif pd.isna(r.get("country")) and r["organization"] not in org_country:
            a.add("CHECK", "country", m, f"no country for {r['organization']}: add it to "
                  "organization_countries.csv (the US-CN analysis reads it)")
    # near-duplicate names (an alias the pipeline missed), families compared
    fams_old = {}
    for m in set(models_ref.get("model_version", [])):
        fams_old.setdefault(norm(model_family(m)), set()).add(model_family(m))
    for f in sorted({model_family(m) for m in new_m}):
        same = fams_old.get(norm(f), set()) - {f}
        if same:
            a.add("CHECK", "alias", f, f"same name as existing {sorted(same)} up to case and "
                  "punctuation: one release under two spellings? (an alias in the pipeline)")
    # effort inversions within a release, and drops against the lineage parent
    s = scores[scores["model_version"].isin(new_m)]
    for (_fam, b), g in scores.assign(fam=scores["model_version"].map(model_family)) \
            .groupby(["fam", "benchmark"]):
        if not g["model_version"].isin(new_m).any():
            continue
        g = g.assign(e=g["model_version"].map(effort_of)).dropna(subset=["e"])
        g = g.assign(rank=g["e"].map(EFFORT_ORDER.index)).sort_values("rank")
        if len(g) < 2:
            continue
        lo_max = g["score"].cummax().shift()
        bad = g[(lo_max - g["score"]) > EFFORT_INVERSION]
        for r in bad.itertuples():
            a.add("CHECK", "effort inversion", f"{b} / {r.model_version}", f"{r.score:.3f}, "
                  f"{lo_max[r.Index] - r.score:.2f} below a lower effort of the same release: "
                  "truncated run, wrong effort label?")
    if "parent" in lineage.columns:
        node_models = lineage.reset_index().dropna(subset=["chain", "node"])
        by_node = node_models.groupby(["chain", "node"])["raw_string"].apply(set)
        for m in new_m:
            if m not in lineage.index or pd.isna(lineage.loc[m, "parent"]):
                continue
            ch, par = lineage.loc[m, "chain"], lineage.loc[m, "parent"]
            parents = {p for p in by_node.get((ch, par), set()) if effort_of(p) == effort_of(m)}
            ps = scores[scores["model_version"].isin(parents)].groupby("benchmark")["score"].max()
            for r in s[s["model_version"] == m].itertuples():
                if r.benchmark in ps.index and ps[r.benchmark] - r.score > LINEAGE_DROP:
                    a.add("CHECK", "lineage drop", f"{r.benchmark} / {m}", f"{r.score:.3f} against "
                          f"{ps[r.benchmark]:.3f} for its parent node `{par}` at the same effort: a regression, a "
                          "different harness, or the wrong parent?")
    return new_m


def check_revisions(a: Audit, scores, scores_ref):
    if not len(scores_ref):
        return
    key = ["model_version", "benchmark"]
    now = scores.groupby(key)["score"].mean()
    old = scores_ref.groupby(key)["score"].mean()
    both = pd.concat([old.rename("old"), now.rename("new")], axis=1, join="inner")
    moved = both[(both["new"] - both["old"]).abs() > REVISION]
    for (m, b), r in moved.sort_values("new").iterrows():
        a.add("CHECK", "score revised", f"{b} / {m}", f"{r['old']:.3f} -> {r['new']:.3f}: an "
              "upstream revision (source page, METR/AISI update) or an identity change?")
    lost = old.index.difference(now.index)
    still = set(scores["model_version"])
    lost = [k for k in lost if k[0] in still]
    if lost:
        per_b = pd.Series([b for _, b in lost]).value_counts()
        a.add("CHECK", "scores dropped", f"{len(lost)}", "scores of models still present are gone: "
              + ", ".join(f"{b} ({n})" for b, n in per_b.head(12).items()))
    a.add("INFO", "scores", f"{len(now)}", f"{len(now) - len(old):+d} (model, benchmark) cells; "
          f"{len(both)} carried over, {len(moved)} revised by more than {REVISION}")


def check_humans(a: Audit, humans, humans_ref, new_b):
    tiers = set(HUMAN_ORDER_MERGED)
    key = ["benchmark", "group"]
    for r in humans.itertuples():
        if r.group not in tiers:
            a.add("BLOCK", "human baseline", f"{r.benchmark} / {r.group}", "tier not in "
                  "config.HUMAN_ORDER_MERGED: a stray name forks the tier")
    if not len(humans_ref):
        return
    m = humans.merge(humans_ref, on=key, how="outer", suffixes=("", "_ref"), indicator=True)
    for _, r in m.iterrows():
        subj = f"{r['benchmark']} / {r['group']}"
        added = r["_merge"] == "left_only"
        if added and r["benchmark"] not in new_b:
            a.add("DECIDE", "human baseline", subj, f"new baseline {r['score']} on an existing "
                  "benchmark: accept? (comparable conditions, not 1.0 by construction)")
        elif r["_merge"] == "right_only":
            a.add("CHECK", "human baseline", subj, f"baseline {r['score_ref']} removed upstream")
        elif r["_merge"] == "both" and abs(r["score"] - r["score_ref"]) > 1e-9:
            a.add("CHECK", "human baseline", subj, f"{r['score_ref']} -> {r['score']}")
        if added and r["score"] >= 1.0 - 1e-9:
            a.add("DECIDE", "human baseline", subj, "a human score of 1.0: by construction (answer "
                  "key, validators) under conditions unlike the models'? Such rows are rejected")


def check_curated_rows(a: Audit):
    over = read_now(Path("1_curated/model_openness_overrides.csv"))
    for r in over.itertuples():
        if re.search(r"re-check|recheck|pending|window open", str(r.note), re.I):
            a.add("CHECK", "openness to re-check", r.model_version, str(r.note))


def check_placement_judgements(a: Audit, humans):
    """Benchmarks with human scores that the per-axis placement has not judged on the named axes."""
    names = sorted(config.OUTPUTS_DIR.glob("data*/mirt_lineageprior_lineagebm/axis_names.json"))
    if not names:
        return
    axes = json.loads(names[-1].read_text()).get("axes", {})
    titles = [ax.get("title") or ax.get("title_en") for ax in axes.values()]
    judged = read_now(Path("1_curated/human_axis_content.csv"))
    pairs = set(zip(judged.get("axis_title", []), judged.get("benchmark", [])))
    todo = sorted({b for b in humans["benchmark"] for t in titles if t and (t, b) not in pairs})
    if todo:
        a.add("DECIDE", "human placement", f"{len(todo)} benchmarks", "human scores not yet judged "
              f"against the named axes ({names[-1].parent.name}), at placement time: " + ", ".join(todo))


# ── report ────────────────────────────────────────────────────────────────────────────────────

def render(a: Audit, ref: str) -> str:
    lines = [f"# Sync audit, data generation {config.DATA_TAG}", "",
             f"Reference: 0_input/ at `{ref}`. Produced by `1_curated/0_audit_sync.py`; the "
             "procedure is `.claude/skills/data-refresh/SKILL.md`.", "",
             "| level | findings |", "|---|---|"]
    lines += [f"| {lv} | {a.count(lv)} |" for lv in LEVELS]
    for lv in LEVELS:
        fs = [f for f in a.findings if f.level == lv]
        if not fs:
            continue
        lines += ["", f"## {lv}", ""]
        for topic in dict.fromkeys(f.topic for f in fs):
            lines.append(f"**{topic}**")
            lines += [f"- `{f.subject}`: {f.message}" for f in fs if f.topic == topic]
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main():
    global REF
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ref", default=None, help="git revision of the previous generation")
    ap.add_argument("--pipeline", type=Path, default=config.PIPELINE_DIR)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    REF = resolve_ref(args.ref)
    a = Audit()
    scores, scores_ref = read_now(INPUT / "all_scores_flat.csv"), read_at(REF, INPUT / "all_scores_flat.csv")
    bench, bench_ref = read_now(INPUT / "benchmarks.csv"), read_at(REF, INPUT / "benchmarks.csv")
    models, models_ref = read_now(INPUT / "models.csv"), read_at(REF, INPUT / "models.csv")
    humans, humans_ref = read_now(INPUT / "human_baselines.csv"), read_at(REF, INPUT / "human_baselines.csv")
    check_provenance(a, args.pipeline)
    new_b = check_benchmarks(a, REF, scores, scores_ref, bench, bench_ref, humans)
    check_clips(a, scores, bench, humans)
    check_models(a, scores, scores_ref, models, models_ref)
    check_revisions(a, scores, scores_ref)
    check_humans(a, humans, humans_ref, new_b)
    check_curated_rows(a)
    check_placement_judgements(a, humans)
    report = render(a, REF)
    out = args.out or (config.RESULTS_DIR / "sync_audit.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report)
    print(report)
    print(f"-> {out}")
    sys.exit(1 if a.count("BLOCK") else 0)


REF = "HEAD"
if __name__ == "__main__":
    main()
