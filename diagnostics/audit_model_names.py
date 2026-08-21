"""Read-only audit of upstream model IDENTITY against its own display name.

Our test-taker identity is the upstream `Model version` string, and the effort
tag is nothing more than that string's suffix (`data._EFFORT_SUFFIX_RE`). The
leaderboard's own label lives in a separate `Name` column that the merge never
reads, so an upstream row whose label CONTRADICTS its id passes straight into
the fit as the wrong test-taker. Two ways that shows up, both observed:

  * a stated effort filed as another effort, or as `_unknown` — the label says
    "GPT 5.6 Sol (xHigh)" while the id says `gpt-5.6-sol_unknown`, so a real
    xhigh run founds a spurious unknown-effort test-taker beside the ladder;
  * a DIFFERENT MODEL filed as its sibling — the label says "Inkling Small"
    while the id says `Inkling`, so the small model's score is attributed to
    the flagship.

The checks, over the snapshot files, the alias map, the lineage map and the
processed file:

  1. effort: the label's stated effort vs the id's suffix;
  2. tier: a size/tier word (small/mini/nano/lite/air/flash/pro/turbo) present
     in one of the two and absent from the other;
  3. effort collapse: one id carrying several stated efforts — dedup then
     keeps one score per cell, so a whole ladder silently becomes one row;
  4. fan-in: one id whose labels disagree about a tier;
  5. aliases: `canonical/model_aliases.csv` rows whose variant wording implies
     an effort or tier the canonical id does not carry;
  6. lineage bookkeeping: variant tag vs id suffix, nodes mixing tier words,
     effort ladders split across in_chain;
  7. sibling dominance (data-side, processed file): a cut-down release
     outscoring its parent on shared benchmarks. Naming checks find a
     contradiction in the STRINGS; this one finds its consequence in the
     NUMBERS, which is how a mislabelled row is confirmed rather than guessed.

The snapshot checks read RAW upstream files, UPSTREAM of the pipeline's own
Name-config applier (section 03, which folds "(High)"-style labels into bare
and `_unknown` ids), the curated drops, and the feed-priority dedup — so a
flag is a candidate, not a defect in the fit: cross-check that the row
actually reaches `data/processed` under the wrong id before correcting it
(on 2026-08-06, 8 of 60 effort flags did; the rest were already resolved
downstream).

Writes nothing by default. `--write-queue` refreshes
`data/pipeline/output/name_audit.csv`, one row per flagged (file, id, label)
for review — same draft-then-review convention as `lineage_map.csv`. A flag is
a decision, not a defect: alias the row, correct its id, or accept the upstream
naming and leave it.

Run after every refresh:
  ~/miniforge3/envs/pymc_env/bin/python diagnostics/audit_model_names.py
"""
from pathlib import Path
import glob
import re
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data import _EFFORT_SUFFIX_RE  # noqa: E402  the one effort vocabulary

SNAPSHOTS = ROOT / "data" / "pipeline" / "snapshots"
ALIASES = ROOT / "data" / "pipeline" / "canonical" / "model_aliases.csv"
PROCESSED = ROOT / "data" / "processed" / "benchmarks_merged.csv"
QUEUE = ROOT / "data" / "pipeline" / "output" / "name_audit.csv"

# Label effort spellings -> the suffix vocabulary. Ordered longest-first so
# "x-high" is consumed before the "high" inside it would match.
EFFORT_SPELLINGS = [
    ("xhigh", "xhigh"), ("x-high", "xhigh"), ("extra high", "xhigh"),
    ("minimal", "minimal"), ("medium", "medium"), ("high", "high"),
    ("low", "low"), ("none", "none"), ("max", "max"),
]

# Words that name a DIFFERENT RELEASE rather than a run setting. `max`/`high`
# are deliberately absent: upstream uses them as effort words, so a tier test
# on them would fire on every effort variant.
TIER_WORDS = ["small", "mini", "nano", "lite", "air", "flash", "pro", "turbo"]

# Of those, the ones that mean a CUT-DOWN of another release. Only these carry
# an expectation of scoring BELOW their parent, so only these are read as
# evidence of a mislabelled row when they beat it. `pro`/`turbo` are upgrades:
# gpt-5.4-pro outscoring gpt-5.4 on 13 of 13 benchmarks is the product working.
CUTDOWN_WORDS = ["small", "mini", "nano", "lite", "air", "flash"]

def latest_snapshot():
    dates = sorted(p for p in SNAPSHOTS.glob("[0-9]*-[0-9]*-[0-9]*") if p.is_dir())
    if not dates:
        raise SystemExit(f"no snapshot under {SNAPSHOTS}")
    return dates[-1]


def norm(s):
    """Lowercase alphanumeric-only form, for comparing an id to a label."""
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def tokens(s):
    """Lowercase token set, split on separators only.

    Tier tests run on TOKENS, never on the concatenated form: `mini` is a
    substring of `minimal`, and `nano` spans the join in `luna_none`, so a
    substring test reports both as tier mismatches when neither is one.
    camelCase is deliberately NOT split either, or `MiniMax` yields a `mini`
    its own lowercase spelling does not; every real tier word upstream is
    already separated by a space, hyphen or underscore.
    """
    return {t for t in re.split(r"[^A-Za-z0-9]+", str(s).lower()) if t}


def tier_words(s):
    """Tier words present as whole tokens."""
    return {w for w in TIER_WORDS if w in tokens(s)}


def label_effort(name):
    """Effort the LABEL states, or None. Reads the trailing parenthetical only,
    so a model whose own name contains an effort word is not misread."""
    m = re.search(r"\(([^)]*)\)\s*$", str(name))
    if not m:
        return None
    inner = m.group(1).lower()
    ctx = re.search(r"(\d+)\s*k\b", inner)
    if ctx:
        return f"{ctx.group(1)}K"
    for spelling, tag in EFFORT_SPELLINGS:
        if re.search(rf"(?<![a-z]){re.escape(spelling)}(?![a-z])", inner):
            return tag
    return None


def label_effort_any(name, mv=None):
    """Effort the label states in EITHER style: a parenthetical, or a trailing
    dash/space token. Scraped boards use the second form ("inkling-max",
    "Fable 5 Extra High", "claude-4.8-opus-high"), which a parenthetical-only
    reader cannot see, so a stated effort on a bare id looks like no effort at
    all. Two tail tokens are tested before one, for "Extra High" -> xhigh.

    A tail token is IGNORED when the id's own base name already contains it:
    `gpt-5.1-codex-max` and `qwen3.7-max` are product names whose last word
    happens to be an effort word, and reading it as a run setting invents a
    conflict on every such release. The parenthetical form needs no such guard,
    because a name never carries its own trailing parenthetical.
    """
    paren = label_effort(name)
    if paren:
        return paren
    tk = [t for t in re.split(r"[^A-Za-z0-9]+", str(name).lower()) if t]
    own = tokens(_EFFORT_SUFFIX_RE.sub("", str(mv))) if mv is not None else set()
    for n in (2, 1):
        if len(tk) >= n:
            tail = " ".join(tk[-n:])
            for spelling, tag in EFFORT_SPELLINGS:
                if tail == spelling and not (own & set(tail.split())):
                    return tag
    return None


def id_effort(mv):
    """Effort the ID carries, or None — the same suffix the fit reads."""
    m = _EFFORT_SUFFIX_RE.search(str(mv))
    return m.group(0).lstrip("_") if m else None


def strip_label_effort(name):
    """Label without its trailing parenthetical, for identity comparison."""
    return re.sub(r"\s*\([^)]*\)\s*$", "", str(name)).strip()


def load_pairs(snapshot):
    """Every (file, id, label) the snapshot publishes, from the Epoch CSVs.

    The live feed carries no display-name column and the SEAL boards are read
    from raw HTML by the pipeline itself, so both are covered instead by the
    alias check: a scraped spelling reaches the fit only through an alias row.
    """
    rows = []
    for f in sorted(glob.glob(str(snapshot / "epoch" / "*.csv"))):
        try:
            df = pd.read_csv(f, low_memory=False)
        except Exception:
            continue
        if not {"Model version", "Name"} <= set(df.columns):
            continue
        sub = df[["Model version", "Name"]].dropna().astype(str)
        sub = sub[sub["Name"].str.strip() != ""]
        sub["file"] = Path(f).name
        rows.append(sub)
    if not rows:
        raise SystemExit(f"no Epoch CSV under {snapshot} carries both columns")
    out = pd.concat(rows, ignore_index=True)
    return out.rename(columns={"Model version": "mv", "Name": "name"})


def check_effort(pairs):
    """Label states an effort the id does not carry."""
    p = pairs.copy()
    p["label_effort"] = [label_effort_any(n, m) for n, m in zip(p["name"], p["mv"])]
    p["id_effort"] = p["mv"].map(id_effort)
    bad = p[p["label_effort"].notna()
            & (p["label_effort"] != p["id_effort"])].copy()
    bad["kind"] = bad["id_effort"].map(
        lambda e: "stated effort filed as unknown" if e == "unknown"
        else ("stated effort, id has none" if e is None else "effort contradiction"))
    return bad[["file", "mv", "name", "label_effort", "id_effort", "kind"]]


def check_tier(pairs):
    """A tier word appears in the label or the id but not in both.

    The label keeps its parenthetical here, unlike the effort check: upstream
    writes the tier inside it ("Gemini 3.1 (Pro)"), so stripping it would
    report every such row as a missing tier.
    """
    rows = []
    for _, r in pairs.iterrows():
        in_name, in_mv = tier_words(r["name"]), tier_words(r["mv"])
        for w in sorted(in_name ^ in_mv):
            rows.append(dict(file=r["file"], mv=r["mv"], name=r["name"], tier=w,
                             kind=("label says tier, id does not"
                                   if w in in_name else "id says tier, label does not")))
    return pd.DataFrame(rows)


def check_fan(pairs):
    """One id whose labels DISAGREE about a tier word.

    Restricted to two-sided tier conflicts: each label names a tier the other
    omits ("Phi-3-mini" vs "Phi-3-small"). Effort disagreement is check 3's
    job, spelling variety is the alias map's, and one-sided omission ("Gemini
    3.5 Flash" beside "Gemini 3.5 Flash Lite") is the board abbreviating —
    flagging any of those buries the real conflicts. The mirror check (one
    label, several ids) was tried and produced only the effort-variant
    convention working as designed, so it is deliberately absent.
    """
    rows = []
    for mv, g in pairs.groupby("mv"):
        sets = {frozenset(tier_words(n)) for n in g["name"]} - {frozenset()}
        if any((a - b) and (b - a) for a in sets for b in sets):
            rows.append(dict(mv=mv,
                             tiers=" vs ".join(sorted(",".join(sorted(s)) for s in sets)),
                             labels=" | ".join(sorted(
                                 n for n in g["name"].unique() if tier_words(n)))))
    return pd.DataFrame(rows)


def check_aliases():
    """Alias rows whose variant wording implies an effort or tier the canonical
    id lacks. An empty canonical is a deliberate drop, not a mapping."""
    if not ALIASES.exists():
        return pd.DataFrame()
    al = pd.read_csv(ALIASES, keep_default_na=False, dtype=str)
    rows = []
    for _, r in al.iterrows():
        variant, canon = r["variant"], r["canonical"]
        if not canon.strip():
            continue
        le, ie = label_effort(variant), id_effort(canon)
        if le and le != ie:
            rows.append(dict(variant=variant, canonical=canon, issue=f"effort {le} vs id {ie}"))
        nv, nc = norm(strip_label_effort(variant)), norm(canon)
        for w in TIER_WORDS:
            if (w in nv) != (w in nc):
                rows.append(dict(variant=variant, canonical=canon,
                                 issue=f"tier '{w}' on one side only"))
    return pd.DataFrame(rows)


def check_effort_collapse(pairs):
    """Ids carrying TWO OR MORE distinct stated efforts under one identity.

    The worst class, because it is silent and lossy: the board ran a ladder,
    every rung arrives under one id, and the dedup step then keeps a single
    score per (model, benchmark). With `DEDUP_POLICY="max"` that is the BEST
    rung, so the test-taker is scored at its top effort while its own ladder
    disappears — no variant offsets, no effort structure, and an upward bias in
    the retained score.
    """
    p = pairs.copy()
    p["label_effort"] = [label_effort_any(n, m) for n, m in zip(p["name"], p["mv"])]
    rows = []
    for mv, g in p.groupby("mv"):
        # pd.notna, not truthiness: a missing effort arrives as NaN, and NaN is
        # truthy, so a bare `if e` lets a float into a set of strings.
        efforts = sorted({e for e in g["label_effort"] if pd.notna(e)})
        if len(efforts) > 1:
            rows.append(dict(mv=mv, id_effort=id_effort(mv), n_efforts=len(efforts),
                             stated=", ".join(efforts),
                             labels=" | ".join(sorted(g["name"].unique()))))
    return pd.DataFrame(rows)


def check_lineage():
    """The release map's own effort and tier bookkeeping.

    Three ways the map can disagree with the ids it files:
      a. `variant` says `effort:X` while the raw_string's suffix says Y, so the
         variant-offset group ties the wrong rows together;
      b. one node mixes tier words, meaning a cut-down release sits on its
         flagship's node and shares that node's psi;
      c. a base model's effort variants are split across in_chain yes/no, so
         part of a ladder carries the lineage prior and part runs free.
    """
    path = ROOT / "data" / "curated" / "lineage_map.csv"
    if not path.exists():
        return (pd.DataFrame(),) * 3
    m = pd.read_csv(path, keep_default_na=False, dtype=str)

    bad_variant = []
    for _, r in m.iterrows():
        v = r.get("variant", "")
        if not v.startswith("effort:"):
            continue
        stated, actual = v.split(":", 1)[1], id_effort(r["raw_string"])
        if stated != (actual or "none-in-id"):
            bad_variant.append(dict(raw_string=r["raw_string"], chain=r["chain"],
                                    node=r["node"], variant=v, id_suffix=actual))

    mixed_node = []
    chained = m[m["in_chain"].str.lower() == "yes"]
    for (chain, node), g in chained.groupby(["chain", "node"]):
        sets = {frozenset(tier_words(s)) for s in g["raw_string"]}
        if len(sets) > 1:
            mixed_node.append(dict(chain=chain, node=node, n_models=len(g),
                                   tiers=" | ".join(sorted(
                                       ",".join(sorted(s)) or "(none)" for s in sets)),
                                   models=" | ".join(sorted(g["raw_string"]))))

    split_ladder = []
    m["base"] = m["raw_string"].map(lambda s: _EFFORT_SUFFIX_RE.sub("", str(s)))
    for base, g in m.groupby("base"):
        flags = set(g["in_chain"].str.lower())
        if len(flags) > 1:
            split_ladder.append(dict(
                base=base,
                in_chain=" | ".join(sorted(g.loc[g.in_chain.str.lower() == "yes", "raw_string"])),
                free=" | ".join(sorted(g.loc[g.in_chain.str.lower() != "yes", "raw_string"]))))

    return (pd.DataFrame(bad_variant), pd.DataFrame(mixed_node),
            pd.DataFrame(split_ladder))


def check_sibling_dominance(min_shared=2):
    """Data-side confirmation: a size-suffixed sibling beating its parent.

    A `-small`/`-mini`/`-nano`/`-lite`/`-air`/`-flash` release scoring ABOVE
    the model it is a cut-down of, on most of the benchmarks they share, is the
    numerical signature of a mislabelled row. Effort variants are collapsed to
    their base id first, so an xhigh-vs-low comparison is not read as a family
    inversion. Ties are ignored.
    """
    if not PROCESSED.exists():
        return pd.DataFrame()
    df = pd.read_csv(PROCESSED)
    df["base"] = df["model_version"].map(lambda m: _EFFORT_SUFFIX_RE.sub("", str(m)))
    best = df.groupby(["base", "benchmark"])["score"].max()
    bases = sorted({b for b in df["base"].unique()})
    rows = []
    for child in bases:
        cn = norm(child)
        for w in CUTDOWN_WORDS:
            if w not in tokens(child):
                continue
            parent_norm = cn.replace(w, "")
            for parent in bases:
                if parent == child or norm(parent) != parent_norm:
                    continue
                shared = (best.loc[child].index.intersection(best.loc[parent].index)
                          if child in best.index.get_level_values(0)
                          and parent in best.index.get_level_values(0) else [])
                if len(shared) < min_shared:
                    continue
                d = [(b, best.loc[(child, b)], best.loc[(parent, b)]) for b in shared]
                wins = sum(1 for _, c, p in d if c > p)
                if wins > len(d) / 2:
                    rows.append(dict(
                        child=child, parent=parent, shared=len(d), child_wins=wins,
                        mean_diff=round(sum(c - p for _, c, p in d) / len(d), 4),
                        benchmarks=", ".join(b for b, _, _ in d)))
    return pd.DataFrame(rows)


def main(write_queue=False):
    snapshot = latest_snapshot()
    pairs = load_pairs(snapshot)
    print(f"=== snapshot {snapshot.name}: {len(pairs)} (id, label) rows "
          f"over {pairs['file'].nunique()} files, {pairs['mv'].nunique()} distinct ids")

    eff = check_effort(pairs)
    print(f"\n=== 1. EFFORT: label states an effort the id does not carry "
          f"({len(eff)} rows, {eff['mv'].nunique() if len(eff) else 0} ids) ===")
    if len(eff):
        # dropna=False: an id with NO effort suffix has id_effort None, and the
        # default would silently drop that whole class from the table.
        g = (eff.groupby(["mv", "name", "label_effort", "id_effort", "kind"], dropna=False)
             .size().rename("rows").reset_index().sort_values(["kind", "mv"]))
        print(g.to_string(index=False))
    else:
        print("  none")

    tier = check_tier(pairs)
    print(f"\n=== 2. TIER: size word on one side only ({len(tier)} rows) ===")
    if len(tier):
        g = (tier.groupby(["mv", "name", "tier", "kind"]).size()
             .rename("rows").reset_index().sort_values("rows", ascending=False))
        print(g.to_string(index=False))
    else:
        print("  none")

    coll = check_effort_collapse(pairs)
    print(f"\n=== 3. EFFORT COLLAPSE: one id carrying several stated efforts "
          f"({len(coll)} ids) — dedup keeps one score per cell, so the ladder is lost ===")
    print(coll[["mv", "id_effort", "n_efforts", "stated"]].to_string(index=False)
          if len(coll) else "  none")

    fan_in = check_fan(pairs)
    print(f"\n=== 4. FAN-IN: one id, labels disagreeing on tier ({len(fan_in)}) ===")
    print(fan_in.to_string(index=False) if len(fan_in) else "  none")

    al = check_aliases()
    print(f"\n=== 5. ALIASES: variant wording vs canonical id ({len(al)}) ===")
    print(al.to_string(index=False) if len(al) else "  none")

    bad_variant, mixed_node, split_ladder = check_lineage()
    print(f"\n=== 6a. LINEAGE variant vs id suffix ({len(bad_variant)}) ===")
    print(bad_variant.to_string(index=False) if len(bad_variant) else "  none")
    print(f"\n=== 6b. LINEAGE node mixing tier words ({len(mixed_node)}) ===")
    print(mixed_node.to_string(index=False) if len(mixed_node) else "  none")
    print(f"\n=== 6c. LINEAGE ladder split across in_chain ({len(split_ladder)}) ===")
    print(split_ladder.to_string(index=False) if len(split_ladder) else "  none")

    dom = check_sibling_dominance()
    print(f"\n=== 7. SIBLING DOMINANCE: cut-down release outscoring its parent ({len(dom)}) ===")
    print(dom.to_string(index=False) if len(dom) else "  none")

    if write_queue:
        parts = []
        for name, d, cols in [
                ("effort", eff, ["file", "mv", "name", "kind"]),
                ("tier", tier, ["file", "mv", "name", "kind"])]:
            if len(d):
                q = d[cols].copy()
                q.insert(0, "check", name)
                parts.append(q)
        q = (pd.concat(parts, ignore_index=True).drop_duplicates()
             if parts else pd.DataFrame(columns=["check", "file", "mv", "name", "kind"]))
        QUEUE.parent.mkdir(parents=True, exist_ok=True)
        q.to_csv(QUEUE, index=False)
        print(f"\nwrote {QUEUE} ({len(q)} rows) — review, then alias or correct")

    return eff, tier, fan_in, al, dom


if __name__ == "__main__":
    main(write_queue="--write-queue" in sys.argv)
