"""What the K=3 posterior split trades against what.

diagnose_chains.py labels the split (mode A = 10 chains, mode B = chains 8,9,
dlogp -77.1). This says WHERE the two solutions differ: per-mode mean loadings
after the same permutation/sign match diagnose_chains uses, the benchmarks whose
axis share moves most, and the takers whose ability moves most.

Run: python blogpost/ladder/k3_modes.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from multiaxis_eci.data import clip_scores_to_floors, load_benchmark_floors, load_eci_data  # noqa: E402
from ladder import FITS, OUT  # noqa: E402

MODE_A = [0, 1, 2, 3, 4, 5, 6, 7, 10, 11]
MODE_B = [8, 9]


def match(ref, other):
    """Best permutation + sign alignment of `other`'s columns onto `ref`."""
    K = ref.shape[1]
    best = None
    for perm in itertools.permutations(range(K)):
        cand = other[:, perm]
        signs = np.sign([np.corrcoef(ref[:, k], cand[:, k])[0, 1] for k in range(K)])
        signs[signs == 0] = 1
        cand = cand * signs
        score = min(np.corrcoef(ref[:, k], cand[:, k])[0, 1] for k in range(K))
        if best is None or score > best[0]:
            best = (score, perm, signs, cand)
    return best


def main():
    path = FITS["x"][1]
    post = xr.open_dataset(path, group="posterior")
    A = post["A"].astype("float32").load().values          # (c,d,B,K)
    th = post["theta"].astype("float32").load().values     # (c,d,M,K)
    post.close()

    # Within a mode the chains still disagree on axis LABELS (the normal
    # loading prior is permutation- but not rotation-invariant), so a raw
    # group mean blends axes 1 and 2 into one column. Match every chain onto
    # the mode's first chain before averaging.
    def mode_mean(chains):
        ref = A[chains[0]].mean(0)
        As, Ts = [ref], [th[chains[0]].mean(0)]
        for c in chains[1:]:
            _, perm, signs, Ac = match(ref, A[c].mean(0))
            As.append(Ac)
            Ts.append(th[c].mean(0)[:, list(perm)] * signs)
        return np.mean(As, 0), np.mean(Ts, 0)

    Aa, Ta = mode_mean(MODE_A)
    Ab, Tb = mode_mean(MODE_B)
    score, perm, signs, Ab_al = match(Aa, Ab)
    Tb_al = Tb[:, list(perm)] * signs
    print(f"K=3 mode match: min |corr| {score:.3f}  perm {perm}  signs {signs}")

    data = load_eci_data(include_all_benchmarks=True,
                         drop_benchmarks=["FrontierMath v1", "AlgoTune"])
    data = clip_scores_to_floors(data, load_benchmark_floors(data))
    bench = (data.blookup.sort_values("benchmark_idx")["benchmark"]
             .to_numpy())
    models = data.mlookup.sort_values("model_idx")["model"].to_numpy()
    nobs = np.bincount(data.bench_idx, minlength=len(bench))

    K = Aa.shape[1]
    sh_a = Aa / Aa.sum(1, keepdims=True)
    sh_b = Ab_al / Ab_al.sum(1, keepdims=True)
    db = pd.DataFrame({"benchmark": bench, "n_obs": nobs,
                       **{f"A{k+1}_modeA": Aa[:, k].round(3) for k in range(K)},
                       **{f"A{k+1}_modeB": Ab_al[:, k].round(3) for k in range(K)},
                       **{f"share{k+1}_A": sh_a[:, k].round(3) for k in range(K)},
                       **{f"share{k+1}_B": sh_b[:, k].round(3) for k in range(K)}})
    db["max_share_shift"] = np.abs(sh_a - sh_b).max(1).round(3)
    db = db.sort_values("max_share_shift", ascending=False)
    db.to_csv(OUT / "k3_mode_loadings.csv", index=False)
    print("\ntop-15 benchmarks by axis-share shift between modes:")
    print(db.head(15)[["benchmark", "n_obs"] +
                      [f"share{k+1}_A" for k in range(K)] +
                      [f"share{k+1}_B" for k in range(K)] +
                      ["max_share_shift"]].to_string(index=False))

    dm = pd.DataFrame({"model": models,
                       "n_obs": data.n_obs_per_model,
                       "is_human": data.is_human,
                       **{f"theta{k+1}_A": Ta[:, k].round(3) for k in range(K)},
                       **{f"theta{k+1}_B": Tb_al[:, k].round(3) for k in range(K)}})
    dm["max_abs_shift"] = np.abs(Ta - Tb_al).max(1).round(3)
    dm = dm.sort_values("max_abs_shift", ascending=False)
    dm.to_csv(OUT / "k3_mode_theta.csv", index=False)
    print("\ntop-12 takers by ability shift between modes:")
    print(dm.head(12).to_string(index=False))
    print("\nhuman tiers:")
    print(dm[dm["is_human"]].to_string(index=False))
    print("\nper-axis mean |theta shift| :",
          np.abs(Ta - Tb_al).mean(0).round(3),
          "| per-axis mean loading, A:", Aa.mean(0).round(3),
          "B:", Ab_al.mean(0).round(3))


if __name__ == "__main__":
    main()
