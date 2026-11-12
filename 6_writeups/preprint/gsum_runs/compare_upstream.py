"""Compare a local gsum trace with the upstream 50-chain fit, from what upstream left on disk:
its modes file (per-chain log density, mode labels, the mode D selection) and its loadings table
(written on mode D's chains). Usage: compare_upstream.py <trace.nc> <local modes.json> > out.txt

Same chain index is read as same seed and chain number (upstream seeds are not recorded; the
order 44-48 is assumed, as in R1). Loadings: per-chain posterior-median A against the upstream
mode D table, per-axis |corr| under the axis permutation that maximises the minimum. A mode's
pooled median is taken after each of its chains is put on the axis order of the mode's
best-logp chain (the raw trace keeps each chain's own order of the specifics).
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

UP = Path("5_outputs/data20261019/traces_backup/"
          "mirt_bifactor_lineageprior_lineagebm_gsum_upstream_20261108")
trace, local_modes = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())

up = json.loads((UP / "mirt_modes_trace_mirt_k4_bifactor_lineageprior_lineagebm_gsum.json").read_text())
up_label = {c: m["label"] for m in up["modes"] for c in m["chains"]}
m0 = up["modes"][0]   # chain logp = best chain's logp + chain_delta_logp; recover the best chain's
up_best = m0["mean_logp"] - np.mean([up["chain_delta_logp"][c] for c in m0["chains"]])
up_logp = up_best + np.array(up["chain_delta_logp"])
loc_label = {c: m["label"] for m in local_modes["modes"] for c in m["chains"]}

with xr.open_dataset(trace, group="posterior") as post:
    A = post["A"].median("draw").values                       # (chain, bench, K)
    benches = [str(b) for b in post["bench"].values]
with xr.open_dataset(trace, group="sample_stats") as ss:
    logp = ss["lp" if "lp" in ss else "logp"].mean("draw").values
n_chain, _, K = A.shape

t = pd.read_csv(UP / "mirt_loadings.csv").pivot(index="benchmark", columns="axis", values="loading_median")
assert sorted(t.index) == sorted(benches), "benchmark sets differ"
U = t.loc[benches, [f"axis{k+1}" for k in range(K)]].values   # upstream mode D, (bench, K)
perms = np.array(list(itertools.permutations(range(K))))


def best_perm(ref, a):
    """|corr| matrix (ref axis x a axis) and the permutation of a's axes maximising the min."""
    C = np.abs(np.corrcoef(ref.T, a.T)[:K, K:])
    return C, perms[C[np.arange(K), perms].min(1).argmax()]


def match(a):
    """Per-axis |corr| of upstream axis k with local axis perm[k], best permutation; and the
    |corr| of the two g columns (axis 1 with axis 1), whatever the permutation."""
    C, best = best_perm(U, a)
    return C[np.arange(K), best], tuple(int(p) + 1 for p in best), C[0, 0]


print(f"trace {trace.name}: {n_chain} chains; upstream: {up['n_chains']} chains, mode D = "
      f"{up['selected_chains']['chains']}")
print("axis 1 is g in both; 'perm' lists the local axis matched to upstream axes 1-4\n")
print("chain | local logp (mode) | upstream logp (mode) | min axis corr to upstream mode D | per-axis corr | perm | g-to-g corr")
rows = []
for c in range(n_chain):
    corr, perm, gg = match(A[c])
    rows.append((c, loc_label.get(c, "?"), logp[c], corr.min()))
    print(f"{c:5d} | {logp[c]:10.1f} ({loc_label.get(c, '?')}) | {up_logp[c]:10.1f} ({up_label[c]}) | "
          f"{corr.min():.3f} | {np.round(corr, 3).tolist()} | {perm} | {gg:.3f}")

d = np.abs(logp - up_logp[:n_chain])
print(f"\n|local - upstream| chain logp: median {np.median(d):.1f}, max {d.max():.1f} nats "
      f"({n_chain} chains)")
same = sum(loc_label.get(c) == up_label[c] for c in range(n_chain))
print(f"same mode letter as the upstream twin: {same} / {n_chain} "
      "(letters are re-derived per chain set, so this is indicative only)")

print("\nlocal modes vs upstream mode D loadings")
print("mode | chains | mean logp | Δlogp | min axis corr per chain: min / median / max | "
      "pooled-median corr per axis, perm | g-to-g corr")
for m in local_modes["modes"]:
    ch = m["chains"]
    per = np.array([r[3] for r in rows if r[0] in ch])
    ref = A[ch[int(np.argmax(logp[ch]))]]
    aligned = np.stack([A[c][:, best_perm(ref, A[c])[1]] for c in ch])
    pooled, perm, gg = match(np.median(aligned, axis=0))
    print(f"   {m['label']} | {len(ch):2d} | {m['mean_logp']:10.1f} | {m['delta_logp']:6.1f} | "
          f"{per.min():.3f} / {np.median(per):.3f} / {per.max():.3f} | "
          f"{np.round(pooled, 3).tolist()} perm {perm} | {gg:.3f}")
print(f"\nupstream modes (chains, mean logp, Δ): "
      + ", ".join(f"{m['label']} {len(m['chains'])} {m['mean_logp']:.1f} {m['delta_logp']}" for m in up["modes"]))
