"""Convergence on the identified quantities and the loadings with every chain's axes put in one order.

1. eta (all 5,487 observations), D and sigma_b over the 50 chains: rank r-hat, bulk and tail ESS.
2. Each chain's axes permuted onto the upstream flagship's named axes (its mode-A loading table): the
   permutation maximising the worst-axis |corr| of the chain's mean loadings. Axes never swap inside a
   chain (checked on 100-draw blocks), so one permutation per chain aligns all its draws.
3. Aligned A and theta pooled over all chains: per-cell median, interval, r-hat, ESS. On aligned draws
   r-hat measures real disagreement between chains, not labels.
4. Per mode: aligned loadings and the K x K remix M with L_mode ~= L_A M (least squares). Off-diagonal
   mass in M is the ridge: A M with theta M^-T leaves eta unchanged, so the data cannot pick between them.

Writes next to this file: convergence_identified.csv, chain_alignment.csv, aligned_loadings_by_chain.csv,
aligned_loadings_pooled.csv, aligned_theta_rhat.csv, mode_remix.csv. Read-only on the trace.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "2_model"))
sys.path.insert(0, str(REPO / "3_fit"))
OUT = Path(__file__).resolve().parent
FIT = REPO / "5_outputs/data20261019/mirt_lineageprior_lineagebm"
UP = REPO / "5_outputs/data20261019/traces_backup/mirt_lineageprior_lineagebm_upstream_20261103"
TRACE = FIT / "trace_mirt_k4_lineageprior_lineagebm.nc"


def load_data():
    """The fit's own data, built by fit.py's parser and FitSpec exactly as the run built it."""
    import argparse

    import fit
    from multiaxis_eci.analysis.fitspec import FitSpec
    box = {}
    fit.run_exploration = lambda args, parser: box.update(spec=FitSpec.from_args(args, parser))
    sys.argv = ["fit.py", "--K", "4", "--lineage-prior", "--lineage-bm"]
    fit.main()
    data, _, _ = box["spec"].load_data()
    return data


def summarize(da, name):
    """rank r-hat, bulk and tail ESS of a (chain, draw, ...) DataArray, one row per cell."""
    ds = da.to_dataset(name=name)
    r = az.rhat(ds)[name].values.ravel()
    eb = az.ess(ds, method="bulk")[name].values.ravel()
    et = az.ess(ds, method="tail")[name].values.ravel()
    return r, eb, et


def stat_row(name, r, eb, et):
    return {"quantity": name, "cells": r.size,
            "rhat_max": r.max(), "rhat_median": np.median(r),
            "rhat_gt_1.01": f"{(r > 1.01).sum()} / {r.size}", "rhat_gt_1.05": f"{(r > 1.05).sum()} / {r.size}",
            "bulk_ess_min": eb.min(), "bulk_ess_median": np.median(eb), "tail_ess_min": et.min()}


def best_perm(ref, L):
    """Permutation p (L's column p[k] becomes axis k) maximising min_k |corr(ref[:, k], L[:, p[k]])|."""
    K = ref.shape[1]
    C = np.abs(np.corrcoef(ref.T, L.T)[:K, K:])
    perms = np.array(list(itertools.permutations(range(K))))
    s = C[np.arange(K), perms].min(1)
    i = int(s.argmax())
    return perms[i], float(s[i]), C[np.arange(K), perms[i]]


def main():
    data = load_data()
    with xr.open_dataset(TRACE, group="posterior") as p:
        post = p[["A", "theta", "D", "sigma_b"]].load().astype("float64")
    with xr.open_dataset(TRACE, group="sample_stats") as s:
        div_saved = int(s["diverging"].sum())
    modes = json.loads((FIT / "mirt_modes_trace_mirt_k4_lineageprior_lineagebm.json").read_text())
    mode_of = {c: m["label"] for m in modes["modes"] for c in m["chains"]}
    n_chain, n_draw = post.sizes["chain"], post.sizes["draw"]
    benches = [str(b) for b in post["bench"].values]
    K = post.sizes["latent"]

    # ── 1. identified convergence ────────────────────────────────────────────
    bi = xr.DataArray(data.bench_idx, dims="obs")
    mi = xr.DataArray(data.model_idx, dims="obs")
    eta = (post["A"].isel(bench=bi) * post["theta"].isel(model=mi)).sum("latent") - post["D"].isel(bench=bi)
    rows = []
    a_chains = [c for c in range(n_chain) if mode_of[c] == "A"]
    for label, sel in (("all 50 chains", slice(None)), (f"mode A only ({len(a_chains)} chains)", a_chains)):
        for name, da in (("eta", eta), ("D", post["D"]), ("sigma_b", post["sigma_b"])):
            row = stat_row(name, *summarize(da.isel(chain=sel), name))
            row["chains"] = label
            rows.append(row)
    conv = pd.DataFrame(rows)
    conv.to_csv(OUT / "convergence_identified.csv", index=False)
    print(f"saved draws: {n_chain} chains x {n_draw}; divergences on saved draws {div_saved} / {n_chain * n_draw}")
    print(conv.to_string(index=False, float_format=lambda x: f"{x:.3f}" if x < 10 else f"{x:.0f}"))

    # ── 2. per-chain alignment onto the upstream named axes ─────────────────
    names = json.loads((UP / "axis_names.json").read_text())["axes"]
    up = (pd.read_csv(UP / "mirt_loadings.csv")
          .pivot(index="benchmark", columns="axis", values="loading_median").reindex(benches))
    axes = [f"axis{k + 1}" for k in range(K)]
    titles = [names[a]["title"] for a in axes]
    ref = up[axes].values
    Am = post["A"].mean("draw").values                      # chain, bench, latent
    perms, align = [], []
    for c in range(n_chain):
        p, score, per_axis = best_perm(ref, Am[c])
        perms.append(p)
        align.append({"chain": c, "seed": 44 + c // 10, "mode": mode_of[c], "permutation": p.tolist(),
                      "min_corr": score, **{f"corr_{a}": v for a, v in zip(axes, per_axis)}})
    align = pd.DataFrame(align)
    align.to_csv(OUT / "chain_alignment.csv", index=False)
    print("\nalignment to upstream axes (min over axes of |corr| of chain-mean loadings), by mode:")
    print(align.groupby("mode")[["min_corr"] + [f"corr_{a}" for a in axes]].median().round(3))

    perms = np.array(perms)                                 # chain, K: chain's column for axis k
    idx = xr.DataArray(perms, dims=("chain", "latent"))
    A_al = post["A"].isel(latent=idx)                       # chain, draw, bench, latent (aligned)
    th_al = post["theta"].isel(latent=idx)

    # ── 3. aligned loadings, pooled ──────────────────────────────────────────
    r, eb, et = summarize(A_al, "A")
    q = A_al.quantile([0.05, 0.5, 0.95], dim=("chain", "draw")).values   # q, bench, latent
    chain_med = A_al.median("draw")                          # chain, bench, latent
    between = chain_med.std("chain").values
    within = A_al.std("draw").mean("chain").values
    pooled = pd.DataFrame([
        {"benchmark": b, "axis": axes[k], "axis_title": titles[k],
         "median": q[1, i, k], "q05": q[0, i, k], "q95": q[2, i, k],
         "rhat": r.reshape(len(benches), K)[i, k], "bulk_ess": eb.reshape(len(benches), K)[i, k],
         "between_chain_sd": between[i, k], "within_chain_sd": within[i, k],
         **{f"median_mode_{m}": float(chain_med.isel(chain=[c for c in range(n_chain) if mode_of[c] == m])
                                      .median("chain").values[i, k])
            for m in sorted(set(mode_of.values()))}}
        for i, b in enumerate(benches) for k in range(K)])
    pooled.to_csv(OUT / "aligned_loadings_pooled.csv", index=False)
    pd.DataFrame([{"chain": c, "mode": mode_of[c], "benchmark": b, "axis": axes[k], "median": chain_med.values[c, i, k]}
                  for c in range(n_chain) for i, b in enumerate(benches) for k in range(K)]
                 ).to_csv(OUT / "aligned_loadings_by_chain.csv", index=False)
    print(f"\naligned loadings, all 50 chains: r-hat median {np.median(r):.3f}, max {r.max():.3f}; "
          f"cells over 1.05: {(r > 1.05).sum()} / {r.size}; bulk ESS min {eb.min():.0f}, median {np.median(eb):.0f}")
    print("per axis: r-hat median / max, cells over 1.05")
    for k in range(K):
        rk = r.reshape(len(benches), K)[:, k]
        print(f"  {axes[k]} {titles[k]}: {np.median(rk):.3f} / {rk.max():.3f}, {(rk > 1.05).sum()} / {len(rk)}")
    for k in range(K):
        top = pooled[pooled.axis == axes[k]].nlargest(8, "median")
        print(f"\n{axes[k]} {titles[k]}: top pooled loadings (median [5%, 95%], r-hat)")
        for _, t in top.iterrows():
            print(f"  {t.benchmark:32s} {t['median']:.2f} [{t.q05:.2f}, {t.q95:.2f}]  r-hat {t.rhat:.2f}")
    worst = pooled.nlargest(12, "rhat")
    print("\nleast converged aligned loading cells (between-chain sd against within-chain sd):")
    for _, t in worst.iterrows():
        print(f"  {t.axis} {t.benchmark:32s} r-hat {t.rhat:.2f}  between {t.between_chain_sd:.2f}  "
              f"within {t.within_chain_sd:.2f}  " +
              " ".join(f"{m}:{t[f'median_mode_{m}']:.2f}" for m in sorted(set(mode_of.values()))))

    rt, ebt, _ = summarize(th_al, "theta")
    rt = rt.reshape(post.sizes["model"], K)
    pd.DataFrame(rt, columns=axes, index=[str(m) for m in post["model"].values]).to_csv(OUT / "aligned_theta_rhat.csv")
    print("\naligned abilities: r-hat median / max, models over 1.05, per axis")
    for k in range(K):
        print(f"  {axes[k]}: {np.median(rt[:, k]):.3f} / {rt[:, k].max():.3f}, {(rt[:, k] > 1.05).sum()} / {rt.shape[0]}")

    # ── 4. the ridge: each mode's loadings as a remix of mode A's ───────────
    LA = chain_med.isel(chain=a_chains).median("chain").values
    remix = []
    print("\nremix M with L_mode ~= L_A M (rows: mode A axis, columns: mode axis); R^2 of the fit")
    for m in sorted(set(mode_of.values()) - {"A"}):
        Lm = chain_med.isel(chain=[c for c in range(n_chain) if mode_of[c] == m]).median("chain").values
        M, *_ = np.linalg.lstsq(LA, Lm, rcond=None)
        r2 = 1 - ((Lm - LA @ M) ** 2).sum() / ((Lm - Lm.mean(0)) ** 2).sum()
        diff_explained = 1 - ((Lm - LA @ M) ** 2).sum() / ((Lm - LA) ** 2).sum()
        print(f"  mode {m}: R^2 {r2:.3f}; share of the A-to-{m} difference that is a remix {diff_explained:.2f}")
        print(pd.DataFrame(M, index=axes, columns=axes).round(2).to_string())
        for i in range(K):
            remix.append({"mode": m, "from_axis": axes[i], **{a: M[i, j] for j, a in enumerate(axes)},
                          "r2": r2, "remix_share_of_difference": diff_explained})
    pd.DataFrame(remix).to_csv(OUT / "mode_remix.csv", index=False)


if __name__ == "__main__":
    main()
