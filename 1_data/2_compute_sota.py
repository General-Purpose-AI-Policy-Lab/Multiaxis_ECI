"""Recompute the data-driven SOTA list and write 1_data/curated/sota_models.txt.

SOTA = (frontier envelope on overall 1D capability) ∪ (each flagship lineage's
current leader), restricted to the recent era. Epoch-faithful — the frontier is
"the highest-capability model accessible at each date" (record-setters of the 1D
Beta-IRT C), and the flagship union guarantees every major vendor's current
top-line model is always shown even when it hasn't set a new all-time record.

Refresh: re-run after a data/fit refresh. Reviewed like the other curated maps.

  python 1_data/2_compute_sota.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pymc as pm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from multiaxis_eci.analysis import _release_dates, capability_draws  # noqa: E402
from multiaxis_eci.config import SAMPLE_KW  # noqa: E402
from multiaxis_eci.data import _effort_base, load_eci_data  # noqa: E402
from multiaxis_eci.models.mirt import build_mirt_model  # noqa: E402

WINDOW_MONTHS = 24        # "recent era" cutoff for both the envelope and flagships
MIN_OBS = 4               # Epoch's ≥4-benchmark rule for envelope record-setters
OUT = ROOT / "1_data" / "curated" / "sota_models.txt"

# Flagship lineage chains (the frontier tier). Small / mini / flash / open /
# fast lines are intentionally excluded — a chain is flagship if it is the
# vendor's top capability line. Names match lineage_map.csv `chain`.
FLAGSHIP_CHAINS = {
    "gpt", "o-flagship", "o-pro", "codex", "pro",   # OpenAI frontier lines
    "opus", "sonnet",                                # Anthropic
    "gemini-pro",                                    # Google DeepMind
    "grok",                                          # xAI
    "deepseek-v", "deepseek-r",                      # DeepSeek
    "qwen-max",                                      # Alibaba
    "glm", "kimi", "minimax",                        # Zhipu / Moonshot / MiniMax
    "mistral-large",                                 # Mistral
    "llama-70b",                                     # Meta (largest in-data line)
}

# Explicit always-include pins (prefix-matched → best-C variant present in data).
# The flagship logic picks only a chain's LATEST node; a pin forces an earlier
# model in too — e.g. Opus 4.8, which the opus chain's latest node (Fable 5)
# otherwise supersedes.
PINNED = ["claude-opus-4-8"]


def main():
    data = load_eci_data()                      # canonical 1D scope
    kw = dict(SAMPLE_KW)
    kw.update(draws=1500, tune=1000, chains=4, cores=4, progressbar=False,
              nuts_sampler="nutpie")
    with build_mirt_model(data, 1, loading_prior="normal"):
        trace = pm.sample(**kw)
    C = capability_draws(trace).mean(0)         # (M,) posterior-mean overall capability
    names = data.mlookup.sort_values("model_idx")["model"].tolist()
    c_of = dict(zip(names, C))
    model_dates, _ = _release_dates(pd.read_csv(ROOT / "1_data/processed/benchmarks_merged.csv"))

    date_of = {m: pd.to_datetime(d) for m, d in model_dates.items()}
    dated = [(date_of[m], m) for i, m in enumerate(names)
             if not data.is_human[i] and m in date_of]
    max_date = max(d for d, _ in dated)
    cutoff = max_date - pd.DateOffset(months=WINDOW_MONTHS)

    # ── frontier envelope: all-time running max of C, keep recent record-setters
    df = pd.DataFrame([(d, m, c_of[m], int(data.n_obs_per_model[names.index(m)]))
                       for d, m in dated], columns=["date", "model", "C", "n_obs"])
    df = df[df["n_obs"] >= MIN_OBS].sort_values("date")
    run, envelope = -np.inf, []
    for _, r in df.iterrows():
        if r.C > run + 1e-9:
            run = r.C
            if r.date >= cutoff:
                envelope.append(r.model)

    # ── flagship union: each flagship chain's latest node, best in-data variant
    lm_full = pd.read_csv(ROOT / "1_data/curated/lineage_map.csv")
    lin_date = dict(zip(lm_full["raw_string"],
                        pd.to_datetime(lm_full["node_date"], errors="coerce")))
    lm = lm_full[(lm_full["in_chain"].astype(str).str.lower() == "yes")
                 & lm_full["chain"].isin(FLAGSHIP_CHAINS)].copy()
    lm["date"] = pd.to_datetime(lm["node_date"], errors="coerce")
    lm = lm[lm["raw_string"].isin(c_of)]        # present in data
    flagships = []
    for chain, g in lm.groupby("chain"):
        latest = g[g["date"] == g["date"].max()]
        if latest.empty:                        # every in-data node undated
            continue                            # (NaT max matches nothing)
        if latest["date"].iloc[0] < cutoff:     # skip dead chains
            continue
        best = max(latest["raw_string"], key=lambda s: c_of[s])   # top variant
        flagships.append(best)
        date_of.setdefault(best, latest["date"].iloc[0])          # lineage date fallback

    # explicit pins: best-C in-data variant of each pinned stem
    pins = []
    for stem in PINNED:
        cands = [m for m in c_of if m.startswith(stem)]
        if cands:
            best = max(cands, key=lambda s: c_of[s])
            pins.append(best)
            date_of.setdefault(best, lin_date.get(best, pd.Timestamp.min))

    # every selected effort variant also pulls in its bare BASE config when it
    # is in the data — the timelines should show the base model alongside its
    # reasoning variant, not only the best-C effort config
    picked = set(envelope) | set(flagships) | set(pins)
    bases = set()
    for m in picked:
        b = _effort_base(m)
        if b != m and b in c_of and b not in picked:
            bases.add(b)
            date_of.setdefault(b, date_of.get(m, pd.Timestamp.min))

    sota = sorted(picked | bases,
                  key=lambda m: date_of.get(m, pd.Timestamp.min), reverse=True)  # newest first
    OUT.write_text("\n".join(sota) + "\n")

    print(f"cutoff {cutoff.date()} (max release {max_date.date()})")
    print(f"envelope: {len(envelope)} | flagships: {len(flagships)} | pins: {len(pins)}"
          f" | bases: {len(bases)} | union: {len(sota)}")
    print(f"\nwrote {len(sota)} models -> {OUT.relative_to(ROOT)}:")
    for m in sota:
        tag = []
        if m in envelope: tag.append("envelope")
        if m in flagships: tag.append("flagship")
        if m in pins: tag.append("pinned")
        if m in bases: tag.append("base")
        print(f"  {str(date_of[m].date())}  {m:42s} C={c_of[m]:+.2f}  [{'+'.join(tag)}]")


if __name__ == "__main__":
    main()
