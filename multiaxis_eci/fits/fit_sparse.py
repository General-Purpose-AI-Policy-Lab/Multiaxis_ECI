"""Fit the sparse-gate non-compensatory Beta-MIRT and write its artefacts.

Driver for models/mirt_sparse.py. Three pure anchors form an identity block
(ARC-AGI-2 -> reasoning, GSM8K -> knowledge, OS World (Screenshot) -> agentic); a
regularized ("Finnish") horseshoe estimates every other benchmark's per-axis
gates. Discrimination is fixed a=1. --human-prior / --lineage-prior add the theta
ordering priors.

What it does:
  1. Load full-benchmark data (include_all_benchmarks=True), humans included.
  2. Build the sparse-gate model with category-seeded gate initial values.
  3. --prior-check: sample the prior predictive, report the score distribution, stop.
  4. Otherwise fit (nutpie), judge convergence on identified quantities
     (analysis.mirt_identified_rhat_sparse), run the PPC + GoF, and write factor
     scores, the surviving-gate structure, and GoF under results/mirt_sparse/.

Run:
  python fits/fit_sparse.py --prior-check
  python fits/fit_sparse.py --human-prior --lineage-prior
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from multiaxis_eci.analysis import factor_scores_df, mirt_identified_rhat_sparse  # noqa: E402
from multiaxis_eci.config import HUMAN_ORDER, RH_TAU_SCALE, SAMPLE_KW, SG_MODEL_NAME  # noqa: E402
from multiaxis_eci.data import (  # noqa: E402
    drop_model_benchmark_cells, drop_model_observations, load_eci_data)
from multiaxis_eci.lineage import build_lineage_structure  # noqa: E402
from multiaxis_eci.models.mirt_sparse import build_mirt_sparse_model  # noqa: E402
from multiaxis_eci.persistence import save_df, save_json, save_trace  # noqa: E402
from multiaxis_eci.ppc import compute_gof, posterior_predictive_mirt_sparse  # noqa: E402

RESULTS_DIR = ROOT / "results" / "mirt_sparse"
PLOTS_DIR = ROOT / "plots" / "mirt_sparse"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Identity-block anchors: one pure benchmark per axis.
AXES = ["Reasoning", "Knowledge", "Agentic"]
ANCHORS = {"ARC-AGI-2": 0, "MMLU": 1, "OS World (Screenshot)": 2}

GATE_SURVIVE = 0.2   # posterior-median gate above this counts as an active axis


def convergence(idata):
    """Global max r-hat / min ESS / divergences (nan-safe for masked entries)."""
    rh = az.rhat(idata)
    ess = az.ess(idata)
    max_rhat = float(np.nanmax([np.nanmax(v.values) for v in rh.data_vars.values()]))
    min_ess = float(np.nanmin([np.nanmin(v.values) for v in ess.data_vars.values()]))
    div = int(idata.sample_stats["diverging"].sum()) if "diverging" in idata.sample_stats else -1
    n_draws = int(idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"])
    return {"max_rhat": max_rhat, "min_ess": min_ess, "divergences": div, "n_draws": n_draws}


def gate_table(idata, data, K):
    """Posterior gate g per (non-anchor benchmark, axis): median, 94% HDI, and a
    `survived` flag (median > GATE_SURVIVE). Anchor rows are excluded (fixed gate)."""
    g = idata.posterior["g"].values.reshape(-1, data.n_benchmarks, K)  # (S, B, K)
    fm = idata.constant_data["free_cell_mask"].values                  # (B, K)
    bench = data.blookup["benchmark"].tolist()
    rows = []
    for bi in range(data.n_benchmarks):
        for k in range(K):
            if fm[bi, k] == 0.0:
                continue
            gk = g[:, bi, k]
            lo, hi = az.hdi(gk, hdi_prob=0.94)
            med = float(np.median(gk))
            rows.append({"benchmark": bench[bi], "axis": AXES[k],
                         "category": str(data.bench_category[bi]),
                         "gate_median": med, "hdi_low": float(lo), "hdi_high": float(hi),
                         "survived": bool(med > GATE_SURVIVE)})
    return (pd.DataFrame(rows)
            .sort_values(["axis", "gate_median"], ascending=[True, False])
            .reset_index(drop=True))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--tune", type=int, default=2000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--target-accept", type=float, default=0.99)
    ap.add_argument("--gate-tau0", type=float, default=RH_TAU_SCALE,
                    help="global horseshoe scale on the gates; smaller = stronger sparsity")
    ap.add_argument("--sampler", default="nutpie", choices=["pymc", "nutpie", "numpyro"])
    ap.add_argument("--human-prior", action="store_true")
    ap.add_argument("--lineage-prior", action="store_true")
    ap.add_argument("--drop-low-obs", action="store_true")
    ap.add_argument("--no-sg", action="store_true",
                    help="drop all Skilled Generalist observations (tier kept, prior-only theta)")
    ap.add_argument("--no-sg-gpqa", action="store_true",
                    help="drop only the Skilled Generalist's GPQA cells (its ARC-vs-GPQA straddle)")
    ap.add_argument("--prior-check", action="store_true",
                    help="sample the prior predictive, report the score distribution, stop")
    args = ap.parse_args()

    K = len(AXES)
    data = load_eci_data(include_all_benchmarks=True, drop_low_obs_models=args.drop_low_obs)
    if args.no_sg and args.no_sg_gpqa:
        ap.error("--no-sg and --no-sg-gpqa are mutually exclusive "
                 "(one drops all SG obs, the other only its GPQA cells)")
    if args.no_sg:
        n0 = data.n_obs
        data = drop_model_observations(data, [SG_MODEL_NAME])
        print(f"--no-sg: dropped {n0 - data.n_obs} '{SG_MODEL_NAME}' obs (tier kept, prior-only)")
    if args.no_sg_gpqa:
        n0 = data.n_obs
        gpqa = [b for b in data.blookup["benchmark"] if "GPQA" in b]
        data = drop_model_benchmark_cells(data, SG_MODEL_NAME, gpqa)
        print(f"--no-sg-gpqa: dropped {n0 - data.n_obs} '{SG_MODEL_NAME}' GPQA cells {gpqa}")
    human_order = HUMAN_ORDER if args.human_prior else None
    lineage = build_lineage_structure(data.mlookup) if args.lineage_prior else None

    print(f"\nSPARSE-GATE NON-COMP MIRT  K={K}  axes={AXES}")
    print(f"data: {data.n_models} models, {data.n_benchmarks} benchmarks, {data.n_obs} obs")
    print(f"anchors: {ANCHORS}")
    print(f"priors: human={bool(human_order)}  lineage={lineage is not None}  "
          f"gate_tau0={args.gate_tau0}")

    model = build_mirt_sparse_model(data, ANCHORS, K=K, human_order=human_order,
                                    lineage=lineage, gate_tau0=args.gate_tau0)

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
    idata.posterior.attrs["mirt_sparse_anchors"] = json.dumps(ANCHORS)
    tag = ("_sparse" + ("_hp" if human_order else "") + ("_lp" if lineage is not None else "")
           + ("_noSG" if args.no_sg else "") + ("_noSGgpqa" if args.no_sg_gpqa else ""))
    save_trace(idata, RESULTS_DIR / f"trace_mirt{tag}.nc")

    conv = convergence(idata)
    idr = mirt_identified_rhat_sparse(idata, data)
    y_rep = posterior_predictive_mirt_sparse(idata, data)
    mu = posterior_predictive_mirt_sparse(idata, data, return_mean=True)
    gof = compute_gof(y_rep, data, mu)

    theta = idata.posterior["theta"].values.reshape(-1, data.n_models, K)
    fscores = factor_scores_df(theta, data.mlookup["model"].tolist(), data.is_human)
    gates = gate_table(idata, data, K)

    save_df(fscores, RESULTS_DIR / f"mirt_sparse_factor_scores{tag}.csv")
    save_df(gates, RESULTS_DIR / f"mirt_sparse_gates{tag}.csv")
    save_json(gof.metrics, RESULTS_DIR / f"mirt_sparse_gof{tag}.json")

    print("\n── convergence (global) ──")
    print(f"  max r-hat={conv['max_rhat']:.4f}  min ESS={conv['min_ess']:.0f}  "
          f"divergences={conv['divergences']}/{conv['n_draws']}")
    print("── identified (log_mu + params) ──")
    print(f"  log_mu max r-hat={idr['logmu_max_rhat']:.4f}  "
          f"theta max r-hat={idr.get('theta_max_rhat', float('nan')):.4f}")
    print("── fit ──")
    print(f"  Bayesian R²={gof.metrics['bayesian_r2']:.4f}  RMSE={gof.metrics['rmse']:.4f}  "
          f"MAE={gof.metrics['mae']:.4f}")
    print(f"── discovered structure (surviving gates, median > {GATE_SURVIVE}) ──")
    surv = gates[gates["survived"]]
    for ax in AXES:
        n = int((surv["axis"] == ax).sum())
        tot = int((gates["axis"] == ax).sum())
        print(f"  {ax}: {n}/{tot} non-anchor benchmarks load this axis")
    print(f"\nartefacts -> {RESULTS_DIR}/  (tag '{tag}')")


if __name__ == "__main__":
    main()
