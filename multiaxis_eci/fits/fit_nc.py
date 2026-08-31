"""Fit the NON-COMPENSATORY (conjunctive) Beta-MIRT and write its artefacts.

Companion to the root fit.py (compensatory). Same data, same Beta
likelihood — the link is the conjunctive product (see models/mirt_nc.py). The
axes are pinned by a category-seeded Q-MATRIX (organic per-category
multi-loading), so there is NO rotation and NO permutation to undo: theta is
read out directly and the factor correlation is the empirical correlation of
posterior-mean abilities.

Phase 1 (default) is the restricted MLTM: discrimination fixed at 1, which tests
whether the imposed axes SEPARATE. Phase 2 (--free-discrim) frees the per-axis
slope to chase fit once the structure is settled.

What it does:
  1. Load full-benchmark data (include_all_benchmarks=True), humans included.
  2. Build the Q-matrix from benchmark categories (--K {3,4}, --qvariant).
  3. --prior-check: sample the prior predictive and report the implied score
     distribution (calibrates the NC_C_OFFSET / tau_c intercept prior); then stop.
  4. Otherwise fit, judge convergence on identified quantities
     (analysis.mirt_identified_rhat_nc — log_mu + theta/sigma_b/tau_c),
     run the PPC + GoF, and write factor scores / factor correlation / per-axis
     difficulty / residuals / GoF under results/mirt_nc/.

Run:
  python fits/fit_nc.py --K 3 --prior-check
  python fits/fit_nc.py --K 3
  python fits/fit_nc.py --K 4
  python fits/fit_nc.py --K 3 --qvariant no-agentic
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

from multiaxis_eci.analysis import (  # noqa: E402
    factor_scores_df, mirt_identified_rhat_nc, nc_difficulty_draws,
)
from multiaxis_eci.config import SAMPLE_KW  # noqa: E402
from multiaxis_eci.data import load_eci_data  # noqa: E402
from multiaxis_eci.models.qmatrix import QMATRIX_VARIANTS, axes_as_list  # noqa: E402
from multiaxis_eci.models.mirt_nc import build_mirt_nc_model  # noqa: E402
from multiaxis_eci.persistence import save_df, save_json, save_trace  # noqa: E402
from multiaxis_eci.ppc import compute_gof, posterior_predictive_mirt_nc  # noqa: E402

RESULTS_DIR = ROOT / "results" / "mirt_nc"
PLOTS_DIR   = ROOT / "plots" / "mirt_nc"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Category -> axis map: organic per-category multi-loading ────────────────
# Single-loadings where a category is clean (gives each axis enough pure
# anchoring benchmarks for identification); deliberate multi-loadings where the
# conjunction is real (bio/chem need reasoning AND specialised knowledge; SWE
# needs reasoning AND agentic execution; Core AGI needs all). K=4 splits a
# dedicated Science axis out of Biology/Chemistry.
AXES = {
    3: ["Reasoning", "Agentic", "Knowledge"],
    4: ["Reasoning", "Agentic", "Knowledge", "Science"],
}
CAT_TO_AXES = {
    3: {
        "High End Math Reasoning":       ["Reasoning"],
        "General Reasoning":             ["Reasoning"],
        "Multimodal Understanding":      ["Reasoning"],
        "Agentic Computer Use":          ["Agentic"],
        "Advanced Language and Writing": ["Knowledge"],
        "Domain Specific Questions":     ["Knowledge", "Reasoning"],
        "Biology":                       ["Knowledge", "Reasoning"],
        "Chemistry":                     ["Knowledge", "Reasoning"],
        "Autonomous SWE":                ["Agentic", "Reasoning"],
        "Core AGI Progress":             ["Reasoning", "Agentic", "Knowledge"],
    },
}
# K=4: identical to K=3 except Biology/Chemistry move onto a dedicated Science axis.
CAT_TO_AXES[4] = {**CAT_TO_AXES[3],
                  "Biology":   ["Science", "Reasoning"],
                  "Chemistry": ["Science", "Reasoning"]}

# Ports of the compensatory confirmatory Q-matrices, derived from the SINGLE
# source of truth (models.qmatrix.QMATRIX_VARIANTS) so comp and non-comp fit the
# SAME structure. That map is index-based; here we attach the non-comp axis
# names (AXES[3] = [Reasoning, Agentic, Knowledge], same order as the indices).
#   qmatrix3  = strict simple structure (each benchmark loads exactly ONE axis).
#     WARNING: in the non-comp PRODUCT this is degenerate — one factor per
#     benchmark, so it reduces to three independent 1D IRTs (no conjunction);
#     build_mirt_nc_model warns at fit time.
#   qmatrix3x = qmatrix3 + cross-loads coding & PhD-science onto Reasoning, so the
#     product actually conjoins (the non-comp version of the cross-loaded fit).
OVERRIDE_MAPS = {
    variant: {cat: [AXES[3][i] for i in axes_as_list(v)] for cat, v in cmap.items()}
    for variant, cmap in QMATRIX_VARIANTS.items()
}


def _axis_layout(K: int, variant: str):
    """Return (axis_names, category->axis_names) for the chosen variant.

    variant='no-agentic' drops the Agentic axis and re-routes its loadings onto
    Reasoning — the sensitivity fit that tests whether a separate agentic axis
    is warranted.
    """
    if variant in OVERRIDE_MAPS:
        if K != 3:
            raise ValueError(f"{variant} is defined for K=3 only (got K={K})")
        return list(AXES[3]), {c: list(v) for c, v in OVERRIDE_MAPS[variant].items()}
    axes = list(AXES[K])
    mapping = {c: list(v) for c, v in CAT_TO_AXES[K].items()}
    if variant == "no-agentic":
        axes = [a for a in axes if a != "Agentic"]
        mapping = {cat: sorted({("Reasoning" if a == "Agentic" else a) for a in axl},
                               key=axes.index)
                   for cat, axl in mapping.items()}
    return axes, mapping


def build_qmatrix(data, K: int, variant: str = "full"):
    """Build the (n_benchmarks, K_eff) 0/1 Q-matrix from benchmark categories.

    Returns (Q, axis_names). K_eff = len(axis_names) (one fewer than K for the
    no-agentic variant). Raises if any benchmark category is not in the map.
    """
    axes, mapping = _axis_layout(K, variant)
    cats = {str(c) for c in np.unique(data.bench_category)}
    missing = cats - set(mapping)
    if missing:
        raise ValueError(f"benchmark categories missing from K={K} Q-map: {sorted(missing)}")
    ax_idx = {a: i for i, a in enumerate(axes)}
    Q = np.zeros((data.n_benchmarks, len(axes)))
    for bi, cat in enumerate(data.bench_category):
        for a in mapping[str(cat)]:
            Q[bi, ax_idx[a]] = 1.0
    return Q, axes


def convergence(idata) -> dict:
    """Global max r-hat / min ESS / divergences (nan-safe for masked entries)."""
    rh = az.rhat(idata)
    ess = az.ess(idata)
    max_rhat = float(np.nanmax([np.nanmax(v.values) for v in rh.data_vars.values()]))
    min_ess  = float(np.nanmin([np.nanmin(v.values) for v in ess.data_vars.values()]))
    div = int(idata.sample_stats["diverging"].sum()) if "diverging" in idata.sample_stats else -1
    n_draws = int(idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"])
    return {"max_rhat": max_rhat, "min_ess": min_ess, "divergences": div, "n_draws": n_draws}


def difficulty_table(trace, data, axes) -> pd.DataFrame:
    """Per-axis difficulty b = -c/a for the loaded (Q=1) benchmark-axis cells.

    The model samples the easiness intercept c (clean geometry); the
    interpretable difficulty is b = -c / a (a = 1 in the restricted fit). Only
    cells the benchmark actually loads are reported.
    """
    K = len(axes)
    Q = trace.constant_data["Q"].values
    b = nc_difficulty_draws(trace.posterior, Q)              # (S, B, K), off-axis NaN
    bench_names = data.blookup["benchmark"].tolist()
    rows = []
    for bi in range(data.n_benchmarks):
        for k in range(K):
            if Q[bi, k] != 1.0:
                continue
            bk = b[:, bi, k]
            bk = bk[np.isfinite(bk)]
            if bk.size == 0:
                continue
            lo, hi = az.hdi(bk, hdi_prob=0.94)
            rows.append({"axis": axes[k], "benchmark": bench_names[bi],
                         "category": str(data.bench_category[bi]),
                         "difficulty_median": float(np.median(bk)),
                         "hdi_low": float(lo), "hdi_high": float(hi)})
    return pd.DataFrame(rows).sort_values(["axis", "difficulty_median"]).reset_index(drop=True)


def _residuals_df(data, y_pred_mean) -> pd.DataFrame:
    models = np.asarray(data.mlookup["model"].tolist())[data.model_idx]
    benches = np.asarray(data.blookup["benchmark"].tolist())[data.bench_idx]
    return pd.DataFrame({"model": models, "benchmark": benches,
                         "score": data.scores, "pred_mean": y_pred_mean,
                         "residual": data.scores - y_pred_mean})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--K", type=int, default=3, choices=[3, 4])
    ap.add_argument("--qvariant", default="full",
                    choices=["full", "qmatrix3", "qmatrix3x", "no-agentic"])
    ap.add_argument("--free-discrim", action="store_true",
                    help="phase 2: free the per-axis slope (tight prior near 1)")
    ap.add_argument("--draws", type=int, default=2000)
    ap.add_argument("--tune", type=int, default=2000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--target-accept", type=float, default=0.99)
    ap.add_argument("--sampler", default="nutpie", choices=["pymc", "nutpie", "numpyro"])
    ap.add_argument("--drop-low-obs", action="store_true")
    ap.add_argument("--prior-check", action="store_true",
                    help="sample the prior predictive, report the score distribution, stop")
    args = ap.parse_args()

    data = load_eci_data(include_all_benchmarks=True, drop_low_obs_models=args.drop_low_obs)
    Q, axes = build_qmatrix(data, args.K, args.qvariant)
    Keff = len(axes)

    print(f"\nNON-COMPENSATORY MIRT  K={args.K} ({args.qvariant}) -> {Keff} axes: {axes}")
    print(f"data: {data.n_models} models, {data.n_benchmarks} benchmarks, {data.n_obs} obs")
    loads = Q.sum(axis=0).astype(int)
    print("benchmarks loading each axis:",
          ", ".join(f"{a}={n}" for a, n in zip(axes, loads)))
    multi = int((Q.sum(axis=1) >= 2).sum())
    print(f"multi-loaded (conjunctive) benchmarks: {multi}/{data.n_benchmarks}  "
          f"| discrimination: {'free' if args.free_discrim else 'fixed=1'}")

    model = build_mirt_nc_model(data, Q, free_discrimination=args.free_discrim)

    if args.prior_check:
        with model:
            pp = pm.sample_prior_predictive(draws=500, random_seed=42)
        obs = pp.prior_predictive["obs"].values
        print("\nPRIOR-PREDICTIVE score distribution:")
        print(f"  mean={obs.mean():.3f}  median={np.median(obs):.3f}  "
              f"frac<0.05={(obs < 0.05).mean():.3f}  frac>0.95={(obs > 0.95).mean():.3f}")
        print("  (want mean ~0.4-0.6 and not floor-collapsed; tune NC_C_OFFSET / PRIOR_TAU_C if off)")
        return

    sample_kw = {**SAMPLE_KW, "draws": args.draws, "tune": args.tune,
                 "chains": args.chains, "cores": args.chains,
                 "target_accept": args.target_accept, "nuts_sampler": args.sampler,
                 "progressbar": True}
    with model:
        idata = pm.sample(**sample_kw)
        pm.compute_log_likelihood(idata)

    # Axis identities travel with the trace (analysis.trace_axis_names reads these).
    idata.posterior.attrs["mirt_axis_names"] = json.dumps(axes)
    idata.posterior.attrs["mirt_qvariant"]   = args.qvariant
    vtag = {"full": "", "no-agentic": "_noag",
            "qmatrix3": "_qm3", "qmatrix3x": "_qm3x"}[args.qvariant]
    tag = f"_k{args.K}{vtag}" + ("_free" if args.free_discrim else "")
    save_trace(idata, RESULTS_DIR / f"trace_mirt_nc{tag}.nc")

    conv = convergence(idata)
    idr = mirt_identified_rhat_nc(idata, data)
    y_rep = posterior_predictive_mirt_nc(idata, data)
    mu = posterior_predictive_mirt_nc(idata, data, return_mean=True)
    gof = compute_gof(y_rep, data, mu)

    theta = idata.posterior["theta"].values.reshape(-1, data.n_models, Keff)
    fscores = factor_scores_df(theta, data.mlookup["model"].tolist(), data.is_human)
    Phi = np.corrcoef(theta.mean(axis=0).T)
    Phi_df = (pd.DataFrame(np.atleast_2d(Phi), index=axes, columns=axes)
              .round(4).reset_index().rename(columns={"index": "axis"}))
    diff = difficulty_table(idata, data, axes)
    resid = _residuals_df(data, gof.y_pred_mean)

    save_df(fscores, RESULTS_DIR / f"mirt_nc_factor_scores{tag}.csv")
    save_df(Phi_df,  RESULTS_DIR / f"mirt_nc_factor_corr{tag}.csv")
    save_df(diff,    RESULTS_DIR / f"mirt_nc_difficulty{tag}.csv")
    save_df(resid,   RESULTS_DIR / f"mirt_nc_residuals{tag}.csv")
    save_json(gof.metrics, RESULTS_DIR / f"mirt_nc_gof{tag}.json")

    print("\n── convergence (global) ──")
    print(f"  max r-hat={conv['max_rhat']:.4f}  min ESS={conv['min_ess']:.0f}  "
          f"divergences={conv['divergences']}/{conv['n_draws']}")
    print("── identified (log_mu + identified params) ──")
    print(f"  log_mu max r-hat={idr['logmu_max_rhat']:.4f}  "
          f"theta max r-hat={idr.get('theta_max_rhat', float('nan')):.4f}")
    print("── fit ──")
    print(f"  Bayesian R²={gof.metrics['bayesian_r2']:.4f}  RMSE={gof.metrics['rmse']:.4f}  "
          f"MAE={gof.metrics['mae']:.4f}  PIT mean={gof.metrics['pit_mean']:.3f}")
    print("── factor correlation Phi (off-diagonal = axis overlap) ──")
    with pd.option_context("display.float_format", lambda v: f"{v:.3f}"):
        print(Phi_df.to_string(index=False))
    print(f"\nartefacts -> {RESULTS_DIR}/  (tag '{tag}')")


if __name__ == "__main__":
    main()
