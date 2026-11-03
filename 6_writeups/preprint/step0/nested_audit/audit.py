"""Nested benchmark audit: do benchmarks that share questions or a family share their luck?

Per pair: models in common, raw score correlation (Spearman), and Yen's Q3, the Pearson
correlation of the two benchmarks' residuals (score minus posterior-mean prediction) over the
common models. Q3 is read against every benchmark pair with >= MIN_N common models in the same
fit: a pair counts as dependent only in the top tail of that distribution.

Fits: the canonical K=1 trace (models only; humans dropped from the residuals) and the three
saved modes of the 5-chain K=4 flagship refit (indication only). All data generation 20261019.

    python 6_writeups/preprint/step0/nested_audit/audit.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import expit
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
CANON = ROOT / "deliverables/canonical_sciencespo_20261022/source/5_outputs/data20261019/canonical_humanmerge"
REFIT = ROOT / "deliverables/flagship_refit_20261021"
FLAG = REFIT / "source/5_outputs/data20261019/mirt_lineageprior_lineagebm"
MODES = {m: REFIT / f"exploratory_modes/mode_{m}/mirt_lineageprior_lineagebm" for m in "ABC"}
TRACE = "trace_mirt_k4_lineageprior_lineagebm.nc"
MIN_N = 10          # common models below which a correlation is listed but not read
N_DRAWS = 500       # thinned posterior draws for the predicted means

# (a, b, class, relation). Class A: the two share questions. Class B: same family, other questions.
PAIRS = [
    ("GPQA Diamond", "GPQA Diamond Biology", "A", "subset: 19 Bio questions of Diamond's 198"),
    ("GPQA Diamond", "GPQA Diamond Chemistry", "A", "subset: 93 Chem questions of Diamond's 198"),
    ("GPQA Main Biology", "GPQA Diamond Biology", "A", "subset: Diamond Bio 19 inside Main Bio 78"),
    ("GPQA Main Chemistry", "GPQA Diamond Chemistry", "A", "subset: Diamond Chem 93 inside Main Chem 183"),
    ("GPQA Diamond", "GPQA Main Biology", "A", "overlap: 19 of Main Bio's 78 are in Diamond"),
    ("GPQA Diamond", "GPQA Main Chemistry", "A", "overlap: 93 of Main Chem's 183 are in Diamond"),
    ("MMLU", "MMLU Biology", "A", "subset: 536 of 14,042"),
    ("MMLU", "MMLU Chemistry", "A", "subset: 230 of 14,042"),
    ("MMLU-Pro", "MMLU Pro Biology", "A", "subset: 717 of 12,032"),
    ("MMLU-Pro", "MMLU Pro Chemistry", "A", "subset: 1,132 of 12,032"),
    ("Cybench", "CAISI Cybench", "A", "same 40 tasks, other harness"),
    ("CyScenarioBench", "CyScenarioBench (Anthropic)", "A", "same challenge set, 9-11 vs 9-10 challenges"),
    ("CAISI CTF-Archive", "CAISI CTF-Archive-Diamond", "A", "same CAISI CTF archive; overlap not documented"),
    ("GPQA Diamond Biology", "GPQA Diamond Chemistry", "B", "sibling subsets"),
    ("GPQA Main Biology", "GPQA Main Chemistry", "B", "sibling subsets"),
    ("MMLU Biology", "MMLU Chemistry", "B", "sibling subsets"),
    ("MMLU Pro Biology", "MMLU Pro Chemistry", "B", "sibling subsets"),
    ("WMDP Biology", "WMDP Chemistry", "B", "sibling subsets"),
    ("MMLU", "MMLU-Pro", "B", "MMLU-Pro reuses filtered MMLU questions, 10 options"),
    ("LAB-Bench Cloning", "LAB-Bench LitQA2", "B", "sibling subsets"),
    ("LAB-Bench Cloning", "LAB-Bench Protocol", "B", "sibling subsets"),
    ("LAB-Bench Cloning", "LAB-Bench SeqQA", "B", "sibling subsets"),
    ("LAB-Bench LitQA2", "LAB-Bench Protocol", "B", "sibling subsets"),
    ("LAB-Bench LitQA2", "LAB-Bench SeqQA", "B", "sibling subsets"),
    ("LAB-Bench Protocol", "LAB-Bench SeqQA", "B", "sibling subsets"),
    ("AISI CTF Suites - Technical Non-Expert", "AISI CTF Suites - Apprentice", "B", "tiers"),
    ("AISI CTF Suites - Technical Non-Expert", "AISI CTF Suites - Practitioner", "B", "tiers"),
    ("AISI CTF Suites - Technical Non-Expert", "AISI CTF Suites - Expert", "B", "tiers"),
    ("AISI CTF Suites - Apprentice", "AISI CTF Suites - Practitioner", "B", "tiers"),
    ("AISI CTF Suites - Apprentice", "AISI CTF Suites - Expert", "B", "tiers"),
    ("AISI CTF Suites - Practitioner", "AISI CTF Suites - Expert", "B", "tiers"),
    ("AISI Cyber Ranges - Cooling Tower", "AISI Cyber Ranges - The Last Ones", "B", "two ranges, same harness"),
    ("FrontierMath Tiers 1-3 v2", "FrontierMath Tier 4 v2", "B", "tiers"),
    ("FrontierMath Tiers 1-3 v2", "FrontierMath Erdos", "B", "same programme"),
    ("FrontierMath Tier 4 v2", "FrontierMath Erdos", "B", "same programme"),
    ("ARC-AGI", "ARC-AGI-2", "B", "versions"),
    ("OSWorld", "OSWorld 2.0", "B", "versions"),
    ("WeirdML v2", "WeirdML v3", "B", "versions"),
    ("CL-bench", "CL-bench Life", "B", "same authors; overlap not documented"),
    ("InterCode-CTF", "NL2Bash", "B", "same source (Lyptus task horizons)"),
]


# Families for the family-level read: every pair among the members, not only the listed ones.
# RAND = the bio/chem rows digitised from RAND RR-A3797-1 (one evaluation campaign).
FAMILIES = {
    "RAND bio/chem": ["GPQA Diamond Biology", "GPQA Diamond Chemistry", "GPQA Main Biology",
                      "GPQA Main Chemistry", "MMLU Biology", "MMLU Chemistry", "MMLU Pro Biology",
                      "MMLU Pro Chemistry", "WMDP Biology", "WMDP Chemistry", "LAB-Bench Cloning",
                      "LAB-Bench LitQA2", "LAB-Bench Protocol", "LAB-Bench SeqQA", "BioLP-bench"],
    "AISI CTF tiers": ["AISI CTF Suites - Technical Non-Expert", "AISI CTF Suites - Apprentice",
                       "AISI CTF Suites - Practitioner", "AISI CTF Suites - Expert"],
    "FrontierMath": ["FrontierMath Tiers 1-3 v2", "FrontierMath Tier 4 v2", "FrontierMath Erdos"],
}


def predicted_mean(trace_path, floors: bool):
    """Posterior-mean fitted score for every (model, benchmark), on N_DRAWS thinned draws.
    K=4: mu = c + (d - c) sigmoid(A.theta - D), c the trace's floors, d its estimated ceiling.
    K=1 canonical: mu = sigmoid(A theta - D) (no floor, no ceiling)."""
    post = xr.open_dataset(trace_path, group="posterior")
    n = post.sizes["chain"] * post.sizes["draw"]
    step = max(1, n // N_DRAWS)
    pick = lambda v: post[v].stack(s=("chain", "draw")).isel(s=slice(None, None, step)).values
    A, th, D = pick("A"), pick("theta"), pick("D")                  # (B,K,S) (M,K,S) (B,S)
    eta = np.einsum("bks,mks->mbs", A, th) - D[None]
    mu = expit(eta)
    if floors:
        c = np.array(json.loads(post.attrs["mirt_floor_c"]))[None, :, None]
        d = pick("ceiling_d")[None]
        mu = c + (d - c) * mu
    return pd.DataFrame(mu.mean(-1), index=post["model"].values, columns=post["bench"].values)


def matrices(cells, pred):
    """Score and residual matrices (models x benchmarks) over the observed cells."""
    cells = cells[cells.model.isin(pred.index) & cells.benchmark.isin(pred.columns)].copy()
    cells["pred"] = pred.values[pred.index.get_indexer(cells.model), pred.columns.get_indexer(cells.benchmark)]
    score = cells.pivot_table(index="model", columns="benchmark", values="score")
    resid = cells.assign(r=cells.score - cells.pred).pivot_table(index="model", columns="benchmark", values="r")
    return score, resid


def all_q3(resid):
    """Q3 and common-model count for every benchmark pair."""
    R = resid.to_numpy()
    obs = (~np.isnan(R)).astype(float)
    N = obs.T @ obs
    Q = resid.corr(min_periods=3).to_numpy()
    i, j = np.triu_indices(R.shape[1], 1)
    return pd.DataFrame({"a": resid.columns[i], "b": resid.columns[j], "n": N[i, j], "q3": Q[i, j]})


def dominant_axis(path):
    lo = pd.read_csv(path / "mirt_loadings.csv")
    top = lo.loc[lo.groupby("benchmark").axis_share.idxmax()].set_index("benchmark")
    return top.axis, top.axis_share


def main():
    humans = set(pd.read_csv(CANON / "human_groups.csv").name)
    k1_cells = pd.read_csv(CANON / "pit.csv")[["model", "benchmark", "score"]]
    k1_cells = k1_cells[~k1_cells.model.isin(humans)]
    k4_cells = pd.read_csv(FLAG / "mirt_residuals.csv")[["model", "benchmark", "score"]]

    fits = {"K1": (CANON / "trace.nc", k1_cells, False)}
    fits.update({f"K4{m}": (p / TRACE, k4_cells, True) for m, p in MODES.items()})

    rows = pd.DataFrame(PAIRS, columns=["a", "b", "cls", "relation"])
    baselines, fam_rows = {}, []
    for name, (path, cells, floors) in fits.items():
        print(f"{name}: predicted means from {path.name}", flush=True)
        score, resid = matrices(cells, predicted_mean(path, floors))
        q = all_q3(resid)
        base = q[(q.n >= MIN_N) & q.q3.notna()].q3.to_numpy()
        baselines[name] = {"n_pairs": len(base), "mean": float(base.mean()),
                           "p50": float(np.median(base)), "p95": float(np.quantile(base, 0.95)),
                           "p99": float(np.quantile(base, 0.99))}
        key = {frozenset(ab): r for ab, r in zip(zip(q.a, q.b), q.itertuples())}
        n, q3, pct, rho = [], [], [], []
        for a, b in zip(rows.a, rows.b):
            r = key.get(frozenset((a, b)))
            if r is None:                                   # pair outside this fit's scope
                n.append(0); q3.append(np.nan); pct.append(np.nan); rho.append(np.nan)
                continue
            n.append(int(r.n)); q3.append(r.q3)
            pct.append(100 * (base < r.q3).mean() if np.isfinite(r.q3) else np.nan)
            both = score[[a, b]].dropna()
            rho.append(spearmanr(both[a], both[b])[0] if len(both) >= 3 else np.nan)
        rows[f"{name}_n"], rows[f"{name}_rho"] = n, rho
        rows[f"{name}_q3"], rows[f"{name}_pct"] = q3, pct
        p95 = baselines[name]["p95"]
        for fam, members in FAMILIES.items():
            w = q[q.a.isin(members) & q.b.isin(members) & (q.n >= MIN_N) & q.q3.notna()]
            fam_rows.append({"fit": name, "family": fam, "pairs_read": len(w),
                             "mean_q3": w.q3.mean(), "baseline_mean_q3": baselines[name]["mean"],
                             "n_above_p95": int((w.q3 > p95).sum())})
        # Same content, other source: RAND rows against the Epoch/Kaggle science columns that
        # contain their questions. Separates a shared-campaign effect from a shared-content one.
        rand, sci = FAMILIES["RAND bio/chem"], ["GPQA Diamond", "MMLU", "MMLU-Pro"]
        w = q[((q.a.isin(rand) & q.b.isin(sci)) | (q.b.isin(rand) & q.a.isin(sci)))
              & (q.n >= MIN_N) & q.q3.notna()]
        fam_rows.append({"fit": name, "family": "RAND vs GPQA Diamond/MMLU/MMLU-Pro",
                         "pairs_read": len(w), "mean_q3": w.q3.mean(),
                         "baseline_mean_q3": baselines[name]["mean"],
                         "n_above_p95": int((w.q3 > p95).sum())})

    axis_rows = []
    for m, p in MODES.items():
        ax, share = dominant_axis(p)
        rows[f"K4{m}_axes"] = [f"{ax.get(a, '?')}/{ax.get(b, '?')}" for a, b in zip(rows.a, rows.b)]
        lo = pd.read_csv(p / "mirt_loadings.csv")
        for k, g in lo.groupby("axis"):
            top = g.nlargest(10, "axis_share").benchmark      # the benchmarks the axis owns
            axis_rows.append({"mode": m, "axis": k, **{fam: int(top.isin(mem).sum())
                                                        for fam, mem in FAMILIES.items()},
                              "top10": "; ".join(top)})
    pd.DataFrame(fam_rows).round(3).to_csv(OUT / "families.csv", index=False)
    pd.DataFrame(axis_rows).to_csv(OUT / "axis_top10_families.csv", index=False)
    print(pd.DataFrame(fam_rows).round(3).to_string(index=False))
    print(pd.DataFrame(axis_rows).drop(columns="top10").to_string(index=False))
    rows.round(3).to_csv(OUT / "pairs.csv", index=False)
    json.dump(baselines, open(OUT / "baselines.json", "w"), indent=1)
    print(json.dumps(baselines, indent=1))
    cols = ["a", "b", "cls"] + [f"{f}_{s}" for f in fits for s in ("n", "q3", "pct")]
    print(rows[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
