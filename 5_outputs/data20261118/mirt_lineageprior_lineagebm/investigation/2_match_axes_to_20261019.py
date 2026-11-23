"""Match the axes of each mode of the 2026-11-18 flagship to the 2026-10-19 flagship
(its named display frame and each of its modes). Reads only the loadings A of both traces
(fetch them from their releases first).

    python 5_outputs/data20261118/mirt_lineageprior_lineagebm/investigation/2_match_axes_to_20261019.py
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
NEW = str(HERE.parent) + "/"                                     # this fit's folder
OLD = str(HERE.parents[2] / "data20261019" / "mirt_lineageprior_lineagebm") + "/"
T = "trace_mirt_k4_lineageprior_lineagebm.nc"
M = "mirt_modes_trace_mirt_k4_lineageprior_lineagebm.json"

def chain_means(folder):
    ds = xr.open_dataset(folder + T, group="posterior")
    A = ds["A"].mean("draw").values.astype(float)          # (chain, bench, K)
    return A, [str(b) for b in ds["bench"].values]

def align(ref, other):
    """Columns of `other` permuted (and sign-flipped) onto `ref`; returns aligned, matched r."""
    K = ref.shape[1]
    c = np.corrcoef(ref.T, other.T)[:K, K:]
    _, perm = linear_sum_assignment(-np.abs(c))
    r = c[np.arange(K), perm]
    return other[:, perm] * np.sign(r), np.abs(r), perm

def mode_means(folder):
    A, bench = chain_means(folder)
    modes = json.load(open(folder + M))["modes"]
    out = {}
    for m in modes:
        ch = m["chains"]; ref = A[ch[0]]
        out[m["label"]] = (np.mean([align(ref, A[c])[0] for c in ch], axis=0), len(ch), m["delta_logp"])
    return out, bench

new, nb = mode_means(NEW)
old, ob = mode_means(OLD)
common = [b for b in nb if b in ob]
ni = [nb.index(b) for b in common]; oi = [ob.index(b) for b in common]
print(f"benchmarks: new {len(nb)}, old {len(ob)}, common {len(common)}")
print("only new:", sorted(set(nb) - set(ob)))
print("only old:", sorted(set(ob) - set(nb)))

print("\n== new mode x old mode: min / mean matched |corr| over the 4 axes (common benchmarks) ==")
rows = []
for ln, (An, n_n, dn) in new.items():
    row = {}
    for lo, (Ao, n_o, _do) in old.items():
        _, r, _ = align(Ao[oi], An[ni])
        row[f"old {lo} ({n_o})"] = f"{r.min():.2f}/{r.mean():.2f}"
    rows.append(pd.Series(row, name=f"new {ln} ({n_n}, dlogp {dn})"))
print(pd.DataFrame(rows).to_string())

# named display frame of the old fit
names = json.load(open(OLD + "axis_names.json"))["axes"]
L = pd.read_csv(OLD + "mirt_loadings.csv").pivot(index="benchmark", columns="axis", values="loading_median")
cols = sorted(L.columns, key=lambda c: int(c[4:]))
P = L.loc[common, cols].values
print("\n== each new mode against the NAMED axes of the 2026-10-19 fit ==")
for ln, (An, n_n, dn) in new.items():
    al, r, perm = align(P, An[ni])
    full = An[:, perm] * np.sign(np.corrcoef(P.T, An[ni].T)[:4, 4:][np.arange(4), perm])
    share = full ** 2 / np.maximum((full ** 2).sum(axis=1, keepdims=True), 1e-12)
    print(f"\n-- new mode {ln} ({n_n} chains, dlogp {dn}) --")
    for k, c in enumerate(cols):
        top = [nb[b] for b in np.argsort(-share[:, k])[:8]]
        oldtop = names[c]["top_benchmarks"]
        sig = names[c]["signature"]
        keep = [s for s in sig if s in top]
        gone = [s for s in sig if s not in nb]
        rank = {s: int(np.where(np.argsort(-share[:, k]) == nb.index(s))[0][0]) + 1 for s in sig if s in nb}
        print(f"  {c} {names[c]['title']!r}: r = {r[k]:.2f}; common top-8 with old: {len(set(top) & set(oldtop))}/8")
        print(f"     new top-8 by share: {top}")
        print(f"     signature ranks by share: {rank}" + (f"; no longer in data: {gone}" if gone else ""))
