"""Fit the semi-compensatory MIRT (compensatory + pairwise ability interactions)
and write its artefacts. Driver for models/mirt_interaction.py.

Default loadings are signed and free, identified by three PLT founders (one per
axis): ARC-AGI-2 -> reasoning, MMLU -> knowledge, OS World (Screenshot) -> agentic.
--loading-prior normal switches to non-negative HalfNormal loadings with no
founders (non-negativity pins the frame). --floors applies the fixed-c 3PL link
(chance floors from 1_data/curated/benchmark_lower_bounds.csv, scores clipped up to
the floor), as in fit.py --floors. Each benchmark gets a per-pair interaction
coefficient gamma (DeMars a_3), non-negative and acting on softplus abilities:
gamma > 0 means the item rewards having BOTH abilities ("needs both" /
conjunctive), gamma = 0 is compensatory.

Because gamma >= 0 by construction, an HDI-vs-zero test is vacuous. The readout
is `p_above_prior` = P(gamma > prior median): 0.5 means the posterior still is
the prior (the data say nothing), values towards 1 mean conjunction. The verdict
proper is LOO against the matched --gamma-pooling none control.

Honest note: free signed loadings make this the exploratory (multimodal) regime;
PLT fixes rotation but not the near-tied-axes basins, so expect to read gamma
mode-restricted there.

Run:
  ~/miniforge3/envs/pymc_env/bin/python fits/fit_interaction.py --prior-check
  ~/miniforge3/envs/pymc_env/bin/python fits/fit_interaction.py --human-prior --lineage-prior
  ~/miniforge3/envs/pymc_env/bin/python fits/fit_interaction.py --loading-prior normal \\
      --floors --human-prior --lineage-prior --gamma-pooling pooled
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from multiaxis_eci.analysis import factor_scores_df, mirt_identified_rhat_interaction  # noqa: E402
from multiaxis_eci.config import HUMAN_ORDER, SAMPLE_KW, SG_MODEL_NAME  # noqa: E402
from multiaxis_eci.data import (  # noqa: E402
    clip_scores_to_floors, drop_model_benchmark_cells, drop_model_observations,
    load_benchmark_floors, load_eci_data)
from multiaxis_eci.models.mirt_interaction import INTERACTION_SCALE, build_mirt_interaction_model  # noqa: E402
from multiaxis_eci.persistence import save_df, save_json, save_trace  # noqa: E402
from multiaxis_eci.ppc import compute_gof, posterior_predictive_mirt_interaction  # noqa: E402

RESULTS_DIR = ROOT / "results" / "mirt_interaction"
PLOTS_DIR = ROOT / "plots" / "mirt_interaction"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

AXES = ["Reasoning", "Knowledge", "Agentic"]
PLT_FOUNDERS = ["ARC-AGI-2", "MMLU", "OS World (Screenshot)"]   # one per axis, in order


def convergence(idata):
    """Global max r-hat / min ESS / divergences (nan-safe for masked entries)."""
    rh = az.rhat(idata)
    ess = az.ess(idata)
    max_rhat = float(np.nanmax([np.nanmax(v.values) for v in rh.data_vars.values()]))
    min_ess = float(np.nanmin([np.nanmin(v.values) for v in ess.data_vars.values()]))
    div = int(idata.sample_stats["diverging"].sum()) if "diverging" in idata.sample_stats else -1
    n_draws = int(idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"])
    return {"max_rhat": max_rhat, "min_ess": min_ess, "divergences": div, "n_draws": n_draws}


def prior_median(interaction_scale):
    """Median of the gamma prior, |z| * scale with z ~ HalfNormal(1).

    The 0.75 quantile of the standard normal, so P(gamma > this) = 0.5 under a
    posterior that has learned nothing. That makes p_above_prior scale-free: the
    no-signal reading is 0.5 whatever --interaction-scale is set to.
    """
    from scipy.stats import norm
    return float(norm.ppf(0.75)) * interaction_scale


def _gamma_row(col, pmed):
    """Summary of one gamma posterior column against its prior median."""
    lo, hi = az.hdi(col, hdi_prob=0.94)
    return {"gamma_median": float(np.median(col)), "hdi_low": float(lo),
            "hdi_high": float(hi), "p_above_prior": float((col > pmed).mean())}


def gamma_table(idata, data, K, interaction_scale):
    """Per-(benchmark, axis-pair) interaction gamma: median, 94% HDI, p_above_prior.

    Masked (founder off-triangle) cells are skipped. gamma >= 0 by construction,
    so there is no HDI-vs-zero verdict: p_above_prior near 0.5 means the cell is
    prior-dominated, near 1 means the data pushed it up (conjunctive).
    """
    pairs = list(itertools.combinations(range(K), 2))
    pair_names = [f"{AXES[j]}x{AXES[k]}" for (j, k) in pairs]
    g = idata.posterior["gamma"].values.reshape(-1, data.n_benchmarks, len(pairs))
    imask = idata.constant_data["inter_mask"].values
    bench = data.blookup["benchmark"].tolist()
    pmed = prior_median(interaction_scale)
    rows = []
    for bi in range(data.n_benchmarks):
        for p in range(len(pairs)):
            if imask[bi, p] == 0.0:
                continue
            rows.append({"benchmark": bench[bi], "pair": pair_names[p],
                         "category": str(data.bench_category[bi]),
                         **_gamma_row(g[:, bi, p], pmed)})
    return (pd.DataFrame(rows)
            .sort_values("gamma_median", ascending=False).reset_index(drop=True))


def gamma_pooled_table(idata, K, interaction_scale):
    """One shared gamma per axis-pair (pooled fit): median, 94% HDI, p_above_prior.

    Reads `gamma_pooled` (pair,) — the stage-1 readout of whether benchmarks need
    both skills on average per axis-pair. Columns as in gamma_table.
    """
    pairs = list(itertools.combinations(range(K), 2))
    pair_names = [f"{AXES[j]}x{AXES[k]}" for (j, k) in pairs]
    gp = idata.posterior["gamma_pooled"].values.reshape(-1, len(pairs))
    pmed = prior_median(interaction_scale)
    return pd.DataFrame([{"pair": pair_names[p], **_gamma_row(gp[:, p], pmed)}
                         for p in range(len(pairs))])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--tune", type=int, default=2000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--target-accept", type=float, default=0.9)
    ap.add_argument("--interaction-scale", type=float, default=INTERACTION_SCALE,
                    help="prior SD of the interaction coefficient gamma (DeMars a_3)")
    ap.add_argument("--gamma-pooling", default="benchmark",
                    choices=["benchmark", "pooled", "none"],
                    help="benchmark=per-(bench,pair); pooled=one gamma per pair "
                         "(stage-1); none=gamma≡0 compensatory baseline")
    ap.add_argument("--sampler", default="nutpie", choices=["pymc", "nutpie", "numpyro"])
    ap.add_argument("--loading-prior", default="signed", choices=["signed", "normal"],
                    help="signed=free cells + PLT founders (default); normal="
                         "non-negative HalfNormal cells, no founders")
    ap.add_argument("--floors", action="store_true",
                    help="fixed-c 3PL: clip below-floor scores up to each "
                         "benchmark's chance floor and set mu = c + (1-c)*sigmoid "
                         "(floors from 1_data/curated/benchmark_lower_bounds.csv)")
    ap.add_argument("--human-prior", action="store_true")
    ap.add_argument("--lineage-prior", action="store_true")
    ap.add_argument("--drop-low-obs", action="store_true")
    ap.add_argument("--no-sg", action="store_true",
                    help="drop all Skilled Generalist observations (tier kept, prior-only)")
    ap.add_argument("--no-sg-gpqa", action="store_true",
                    help="drop only the Skilled Generalist's GPQA cells")
    ap.add_argument("--prior-check", action="store_true",
                    help="sample the prior predictive, report the score distribution, stop")
    args = ap.parse_args()

    K = len(AXES)
    data = load_eci_data(include_all_benchmarks=True, drop_low_obs_models=args.drop_low_obs)
    if args.no_sg and args.no_sg_gpqa:
        ap.error("--no-sg and --no-sg-gpqa are mutually exclusive")
    if args.no_sg:
        n0 = data.n_obs
        data = drop_model_observations(data, [SG_MODEL_NAME])
        print(f"--no-sg: dropped {n0 - data.n_obs} '{SG_MODEL_NAME}' obs (tier kept, prior-only)")
    if args.no_sg_gpqa:
        n0 = data.n_obs
        gpqa = [b for b in data.blookup["benchmark"] if "GPQA" in b]
        data = drop_model_benchmark_cells(data, SG_MODEL_NAME, gpqa)
        print(f"--no-sg-gpqa: dropped {n0 - data.n_obs} '{SG_MODEL_NAME}' GPQA cells {gpqa}")
    floor_c = None
    if args.floors:
        floor_c = load_benchmark_floors(data)
        n_before = int((data.scores < floor_c[data.bench_idx]).sum())
        data = clip_scores_to_floors(data, floor_c)
        print(f"--floors: fixed-c 3PL; clipped {n_before} below-floor scores up "
              f"to their benchmark chance floor")
    human_order = HUMAN_ORDER if args.human_prior else None
    lineage_struct = None
    if args.lineage_prior:
        from multiaxis_eci.lineage import build_lineage_structure
        lineage_struct = build_lineage_structure(data.mlookup)

    founders = PLT_FOUNDERS if args.loading_prior == "signed" else None
    print(f"\nINTERACTION MIRT  K={K}  axes={AXES}")
    print(f"data: {data.n_models} models, {data.n_benchmarks} benchmarks, {data.n_obs} obs")
    if founders:
        print(f"PLT founders: {dict(zip(founders, AXES))}")
    else:
        print("loading prior: normal (non-negative cells, no founders)")
    print(f"priors: human={bool(human_order)}  lineage={lineage_struct is not None}  "
          f"interaction_scale={args.interaction_scale}  gamma_pooling={args.gamma_pooling}"
          f"  floors={args.floors}")

    model = build_mirt_interaction_model(data, founders, K=K, human_order=human_order,
                                         lineage=lineage_struct,
                                         interaction_scale=args.interaction_scale,
                                         gamma_pooling=args.gamma_pooling,
                                         loading_prior=args.loading_prior,
                                         floor_c=floor_c)

    if args.prior_check:
        with model:
            pp = pm.sample_prior_predictive(draws=500, random_seed=42)
        obs = pp.prior_predictive["obs"].values
        print("\nPRIOR-PREDICTIVE score distribution:")
        print(f"  mean={obs.mean():.3f}  median={np.median(obs):.3f}  "
              f"frac<0.05={(obs < 0.05).mean():.3f}  frac>0.95={(obs > 0.95).mean():.3f}")
        return

    sample_kw = {**SAMPLE_KW, "draws": args.draws, "tune": args.tune,
                 "chains": args.chains, "cores": args.chains,
                 "target_accept": args.target_accept, "nuts_sampler": args.sampler,
                 "progressbar": True}
    with model:
        idata = pm.sample(**sample_kw)
        pm.compute_log_likelihood(idata)

    idata.posterior.attrs["mirt_axis_names"] = json.dumps(AXES)
    if founders:
        idata.posterior.attrs["mirt_plt_founders"] = json.dumps(founders)
    idata.posterior.attrs["mirt_gamma_pooling"] = args.gamma_pooling
    idata.posterior.attrs["mirt_loading_prior"] = args.loading_prior
    idata.posterior.attrs["mirt_interaction_scale"] = args.interaction_scale
    if floor_c is not None:
        idata.posterior.attrs["mirt_floor_c"] = json.dumps(floor_c.tolist())
    gptag = {"benchmark": "", "pooled": "_gpooled", "none": "_gnone"}[args.gamma_pooling]
    tag = ("_interaction" + gptag
           + ("_normal" if args.loading_prior == "normal" else "")
           + ("_hp" if human_order else "") + ("_lp" if lineage_struct is not None else "")
           + ("_noSG" if args.no_sg else "") + ("_noSGgpqa" if args.no_sg_gpqa else "")
           + ("_floors" if args.floors else ""))
    save_trace(idata, RESULTS_DIR / f"trace_mirt{tag}.nc")

    conv = convergence(idata)
    idr = mirt_identified_rhat_interaction(idata, data)
    y_rep = posterior_predictive_mirt_interaction(idata, data, floor_c=floor_c)
    mu = posterior_predictive_mirt_interaction(idata, data, return_mean=True,
                                               floor_c=floor_c)
    gof = compute_gof(y_rep, data, mu)

    theta = idata.posterior["theta"].values.reshape(-1, data.n_models, K)
    fscores = factor_scores_df(theta, data.mlookup["model"].tolist(), data.is_human)

    save_df(fscores, RESULTS_DIR / f"mirt_interaction_factor_scores{tag}.csv")
    save_json(gof.metrics, RESULTS_DIR / f"mirt_interaction_gof{tag}.json")
    if args.gamma_pooling == "benchmark":
        gam = gamma_table(idata, data, K, args.interaction_scale)
        save_df(gam, RESULTS_DIR / f"mirt_interaction_gamma{tag}.csv")
    elif args.gamma_pooling == "pooled":
        gam = gamma_pooled_table(idata, K, args.interaction_scale)
        save_df(gam, RESULTS_DIR / f"mirt_interaction_gamma{tag}.csv")
    else:
        gam = None   # "none": gamma is identically zero, nothing to report

    print("\n── convergence (global; A/theta rotation-identified by PLT) ──")
    print(f"  max r-hat={conv['max_rhat']:.4f}  min ESS={conv['min_ess']:.0f}  "
          f"divergences={conv['divergences']}/{conv['n_draws']}")
    print("── identified (eta incl. interaction + D/sigma_b) ──")
    print(f"  eta max r-hat={idr['eta_max_rhat']:.4f}  D max r-hat={idr.get('D_max_rhat', float('nan')):.4f}")
    print("── fit ──")
    print(f"  Bayesian R²={gof.metrics['bayesian_r2']:.4f}  RMSE={gof.metrics['rmse']:.4f}  "
          f"MAE={gof.metrics['mae']:.4f}")
    print("── interaction gamma (>= 0; p = P(gamma > prior median), 0.5 = no signal) ──")
    if args.gamma_pooling == "none":
        print("  gamma ≡ 0 (compensatory baseline — matched LOO control for the pooled fit)")
    else:
        pmed = prior_median(args.interaction_scale)
        print(f"  prior median {pmed:.3f} logits-per-unit-product")
    if args.gamma_pooling == "pooled":
        for _, r in gam.iterrows():
            print(f"    {r.pair:22s} gamma={r.gamma_median:.3f} "
                  f"[{r.hdi_low:.3f}, {r.hdi_high:.3f}]  p={r.p_above_prior:.2f}")
    elif args.gamma_pooling == "benchmark":
        conj = gam[gam.p_above_prior > 0.90]
        print(f"  pushed up by the data (p>0.90): {len(conj)}  |  prior-dominated "
              f"(p<0.60): {int((gam.p_above_prior < 0.60).sum())}  of {len(gam)} cells")
        gpqa_rk = gam[(gam.benchmark.str.contains("GPQA")) & (gam.pair == "ReasoningxKnowledge")]
        for _, r in pd.concat([gpqa_rk, conj.head(8)]).iterrows():
            print(f"    {r.benchmark:26s} {r.pair:22s} gamma={r.gamma_median:.3f} "
                  f"[{r.hdi_low:.3f}, {r.hdi_high:.3f}]  p={r.p_above_prior:.2f}")
    print(f"\nartefacts -> {RESULTS_DIR}/  (tag '{tag}')")


if __name__ == "__main__":
    main()
