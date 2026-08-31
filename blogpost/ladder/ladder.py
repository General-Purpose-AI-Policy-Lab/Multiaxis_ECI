"""Assumption-ladder table: convergence + LOO + paired ELPD deltas.

Five fits, one scope (exploration minus FrontierMath v1 and AlgoTune, floors,
pooled noise, 5,004 obs). Convergence is scored on identified quantities only
(eta, D, sigma_b) via analysis.convergence, because raw r-hat on A/theta is
permutation-inflated.

LOO is PSIS-LOO on thinned draws (uniform stride, ~1500 per chain) so the
biggest trace (10 x 20,000 x 5,004) fits in memory. Paired deltas use the
project convention: positional row matching after an assert that the observed
vectors are byte-identical, plus a sensitivity restricted to rows where BOTH
sides have Pareto-k < 0.7.

Run: python blogpost/ladder/ladder.py
"""
from __future__ import annotations

import gc
import json
import sys
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from multiaxis_eci.analysis.convergence import _rhat_subset_idx  # noqa: E402
from multiaxis_eci.config import ECI_EPS  # noqa: E402
from multiaxis_eci.data import clip_scores_to_floors, load_benchmark_floors, load_eci_data  # noqa: E402

OUT = Path(__file__).resolve().parent
FLAG = ROOT / ("results/mirt_humanmerge_lineageprior_lineagebm_"
               "dropFrontierMathv1AlgoTune_floors_poolednoise")

# step -> (label, trace path). Order is the ladder.
FITS = {
    "0": ("K=1 baseline", FLAG / "trace_mirt_k1.nc"),
    "1": ("K=4, no ability priors",
          ROOT / "results/mirt_dropFrontierMathv1AlgoTune_floors_poolednoise"
               / "trace_mirt_k4_dropFrontierMathv1AlgoTune_floors_poolednoise.nc"),
    "2": ("K=4 + human order",
          ROOT / "results/mirt_humanmerge_dropFrontierMathv1AlgoTune_floors_poolednoise"
               / "trace_mirt_k4_humanmerge_dropFrontierMathv1AlgoTune_floors_poolednoise.nc"),
    "3": ("K=4 + both priors (flagship)",
          FLAG / ("trace_mirt_k4_humanmerge_lineageprior_lineagebm_"
                  "dropFrontierMathv1AlgoTune_floors_poolednoise.nc")),
    "x": ("K=3 + both priors",
          FLAG / ("trace_mirt_k3_humanmerge_lineageprior_lineagebm_"
                  "dropFrontierMathv1AlgoTune_floors_poolednoise.nc")),
}
TARGET_DRAWS_PER_CHAIN = 1500
N_OBS_SAMPLE = 400          # same subset size analysis/convergence.py defaults to


def expected_obs():
    """The observed vector every fit on this scope must carry."""
    d = load_eci_data(include_all_benchmarks=True,
                      drop_benchmarks=["FrontierMath v1", "AlgoTune"])
    d = clip_scores_to_floors(d, load_benchmark_floors(d))
    return d, np.clip(d.scores, ECI_EPS, 1.0 - ECI_EPS)


def identified_convergence(path: Path, data) -> dict:
    """eta / D / sigma_b r-hat + eta bulk ESS, the way mirt_identified_rhat does
    it, but reading only the slices needed so the 38 GB trace fits in RAM."""
    bi, mi = _rhat_subset_idx(data, N_OBS_SAMPLE, seed=0)
    post = xr.open_dataset(path, group="posterior")
    uniq, inv = np.unique(mi, return_inverse=True)
    A = post["A"].astype("float32").load().values                 # (c,d,B,K)
    th = post["theta"].isel(model=uniq).astype("float32").load().values
    D = post["D"].astype("float32").load().values                 # (c,d,B)
    # chunk over obs: the fancy-index copies of A/theta would otherwise be
    # several GB each on the 10 x 20,000 trace.
    eta = np.empty(A.shape[:2] + (bi.size,), dtype="float32")
    for s in range(0, bi.size, 50):
        e = slice(s, min(s + 50, bi.size))
        eta[..., e] = (A[:, :, bi[e], :] * th[:, :, inv[e], :]).sum(-1) - D[:, :, bi[e]]
    del A, th, D
    gc.collect()
    eta = xr.DataArray(eta, dims=("chain", "draw", "obs")).to_dataset(name="eta")
    r = az.rhat(eta)["eta"].values
    ess = az.ess(eta)["eta"].values
    out = {"eta_max_rhat": float(np.nanmax(r)),
           "eta_frac_rhat_gt_1.01": float(np.nanmean(r > 1.01)),
           "eta_ess_min": float(np.nanmin(ess)),
           "eta_ess_med": float(np.nanmedian(ess))}
    del eta
    gc.collect()
    for v in ("D", "sigma_b", "tau_CD"):
        if v in post:
            x = post[v].astype("float64").load()
            out[f"{v}_max_rhat"] = float(np.nanmax(az.rhat(x.to_dataset(name=v))[v].values))
    post.close()
    gc.collect()
    return out


def sampler_stats(path: Path) -> dict:
    ss = xr.open_dataset(path, group="sample_stats")
    div = ss["diverging"].load().values
    n_chain, n_draw = div.shape
    ss.close()
    return {"chains": int(n_chain), "draws_per_chain": int(n_draw),
            "total_draws": int(div.size), "divergences": int(div.sum())}


def thinned_loo(path: Path):
    """PSIS-LOO on a uniform stride of draws. Returns (loo, loo_i, pareto_k, step)."""
    ll = xr.open_dataset(path, group="log_likelihood")
    step = max(1, ll.sizes["draw"] // TARGET_DRAWS_PER_CHAIN)
    obs = ll["obs"].isel(draw=slice(None, None, step)).load()
    ll.close()
    idata = az.InferenceData(log_likelihood=obs.to_dataset(name="obs"))
    # arviz derives reff from the posterior group, which is 38 GB on the
    # flagship. Use the relative ESS of the pointwise log densities instead —
    # the r_eff_log_lik convention of the loo R package, and the same number
    # for every fit here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        n = obs.sizes["chain"] * obs.sizes["draw"]
        reff = float(np.nanmean(az.ess(obs.to_dataset(name="obs"),
                                       method="mean")["obs"].values) / n)
        loo = az.loo(idata, pointwise=True, reff=min(reff, 1.0))
    loo_i = getattr(loo, "loo_i", None)
    if loo_i is None:
        loo_i = loo.elpd_i
    out = (loo, np.asarray(loo_i.values).ravel(),
           np.asarray(loo.pareto_k.values).ravel(), step,
           int(obs.sizes["draw"]) * int(obs.sizes["chain"]), round(reff, 3))
    del obs, idata
    gc.collect()
    return out


def modes_summary(path: Path) -> dict:
    p = path.parent / f"mirt_modes_{path.stem}.json"
    if not p.exists():
        return {"n_modes": np.nan, "verdict": "no modes JSON"}
    m = json.loads(p.read_text())
    modes = m["modes"]
    if len(modes) == 1:
        return {"n_modes": 1, "verdict": f"one solution ({m['n_chains']} chains)"}
    parts = [f"{d['label']}={d['chains']} dlogp {d['delta_logp']}" for d in modes]
    return {"n_modes": len(modes),
            "verdict": "split: " + " | ".join(parts)}


def pair_delta(a, b, li, kk):
    """Paired pointwise dELPD (a - b), full rows and k<0.7-on-both rows."""
    diff = li[a] - li[b]
    delta, se = float(diff.sum()), float(np.sqrt(diff.size * np.var(diff)))
    ok = (kk[a] < 0.7) & (kk[b] < 0.7)
    d_ok = diff[ok]
    se_ok = float(np.sqrt(d_ok.size * np.var(d_ok))) if d_ok.size else np.nan
    return {"pair": f"step{a} - step{b}", "n_shared": int(diff.size),
            "delta_elpd": round(delta, 2), "se": round(se, 2),
            "z": round(delta / se, 2) if se else np.nan,
            "n_k_ok_rows": int(ok.sum()),
            "delta_elpd_k_ok": round(float(d_ok.sum()), 2),
            "se_k_ok": round(se_ok, 2),
            "z_k_ok": round(float(d_ok.sum()) / se_ok, 2) if se_ok else np.nan}


def main():
    data, expected = expected_obs()
    print(f"scope: {data.n_obs} obs / {data.n_models} takers / "
          f"{data.n_benchmarks} benchmarks", flush=True)

    # --- observed-data identity gate ------------------------------------
    ok_fits, obs_hash = [], {}
    for step, (label, path) in FITS.items():
        if not path.exists():
            print(f"  [MISSING] step {step}: {path}")
            continue
        od = xr.open_dataset(path, group="observed_data")
        got = np.asarray(od["obs"].values).ravel()
        od.close()
        same = got.size == expected.size and np.array_equal(got, expected)
        obs_hash[step] = (got.size, float(got.sum()))
        print(f"  step {step}: {got.size} obs, byte-identical to current data: {same}")
        if not same:
            print(f"    [STOP] step {step} scope differs — excluded from comparisons")
            continue
        ok_fits.append(step)

    rows, li, kk, elpds = [], {}, {}, {}
    for step in ok_fits:
        label, path = FITS[step]
        print(f"\n== step {step}: {label}", flush=True)
        r = {"step": step, "fit": label, "trace": path.name}
        r.update(sampler_stats(path))
        r.update(modes_summary(path))
        r.update(identified_convergence(path, data))
        print("   convergence:", {k: v for k, v in r.items()
                                  if "rhat" in k or "ess" in k}, flush=True)
        loo, loo_i, pk, step_thin, ndraw, reff = thinned_loo(path)
        li[step], kk[step], elpds[f"step{step}"] = loo_i, pk, loo
        r.update({"loo_thin_step": step_thin, "loo_draws_used": ndraw,
                  "loo_reff": reff,
                  "elpd_loo": round(float(loo.elpd_loo), 2),
                  "se_elpd": round(float(loo.se), 2),
                  "p_loo": round(float(loo.p_loo), 2),
                  "k_le_0.5": int((pk <= 0.5).sum()),
                  "k_0.5_0.7": int(((pk > 0.5) & (pk <= 0.7)).sum()),
                  "k_gt_0.7": int((pk > 0.7).sum())})
        print(f"   loo: elpd {r['elpd_loo']} +- {r['se_elpd']}  p_loo {r['p_loo']}"
              f"  k>0.7 {r['k_gt_0.7']}  (thin {step_thin}, {ndraw} draws)", flush=True)
        rows.append(r)
        gc.collect()

    df = pd.DataFrame(rows)

    # --- paired deltas ---------------------------------------------------
    pairs = [("1", "0"), ("2", "1"), ("3", "2"), ("3", "0"),
             ("2", "0"), ("3", "1"), ("x", "0"), ("3", "x")]
    pd_rows = [pair_delta(a, b, li, kk) for a, b in pairs
               if a in li and b in li]
    dfp = pd.DataFrame(pd_rows)
    dfp.to_csv(OUT / "ladder_pairwise.csv", index=False)
    print("\n" + dfp.to_string(index=False))

    # az.compare on the identical-data set — valid because the observed vector
    # gate above proved every member carries byte-identical observations.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cmp = az.compare(elpds, ic="loo")
    cmp.to_csv(OUT / "ladder_compare.csv")
    print("\naz.compare:\n" + cmp.to_string())

    # delta vs step 0 columns on the main table
    d0 = {r["pair"].split(" - ")[0].replace("step", ""): r
          for r in pd_rows if r["pair"].endswith("step0")}
    df["delta_elpd_vs_step0"] = df["step"].map(
        lambda s: d0[s]["delta_elpd"] if s in d0 else 0.0)
    df["se_delta_vs_step0"] = df["step"].map(
        lambda s: d0[s]["se"] if s in d0 else 0.0)
    df["delta_elpd_vs_step0_k_ok"] = df["step"].map(
        lambda s: d0[s]["delta_elpd_k_ok"] if s in d0 else 0.0)

    cols = ["step", "fit", "trace", "chains", "draws_per_chain", "total_draws",
            "divergences", "eta_ess_min", "eta_ess_med", "eta_max_rhat",
            "eta_frac_rhat_gt_1.01", "D_max_rhat", "sigma_b_max_rhat",
            "n_modes", "verdict", "elpd_loo", "se_elpd", "p_loo",
            "k_le_0.5", "k_0.5_0.7", "k_gt_0.7",
            "delta_elpd_vs_step0", "se_delta_vs_step0",
            "delta_elpd_vs_step0_k_ok", "loo_thin_step", "loo_draws_used",
            "loo_reff"]
    df = df[[c for c in cols if c in df.columns]]
    df.to_csv(OUT / "ladder_table.csv", index=False)
    print("\n" + df.to_string(index=False))
    np.savez(OUT / "loo_pointwise.npz",
             **{f"elpd_i_step{s}": li[s] for s in li},
             **{f"pareto_k_step{s}": kk[s] for s in kk})
    json.dump({s: {"n_obs": obs_hash[s][0], "obs_sum": obs_hash[s][1]}
               for s in obs_hash}, open(OUT / "obs_check.json", "w"), indent=1)


if __name__ == "__main__":
    main()
