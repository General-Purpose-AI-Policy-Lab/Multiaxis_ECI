"""One CLI for the compensatory Beta-MIRT family.

Two modes:

  python fit.py --preset canonical
      The headline ECI pipeline: K=1, "normal" loading prior, curated
      benchmark exclusions, humans as test-takers. Produces the anchored ECI
      scale, SOTA table, forests, timeline, PPC/GoF and figures under
      results/canonical/ and plots/canonical/.

  python fit.py --K 3 --loading-prior signed [--human-prior --lineage-prior ...]
      Exploration: the K-axis MIRT on the full benchmark set, promax
      post-processing, per-config results/mirt{tag}/ folders.

The non-compensatory / sparse-gate / interaction families have their own
drivers in fits/.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

import config  # attribute access so --eci-data-only / --raw-c overrides propagate
from analysis import (
    align_rotations, all_models_stats_df, apply_rotation, factor_corr_df,
    factor_scores_df, forest_stats_df, forest_stats_from_draws, human_stats_df,
    loadings_table, mirt_factors_from_trace, mirt_identified_rhat,
    nonneg_rotate, promax_rotate, sota_stats_df, tau_spectrum_df,
    timeline_stats_df,
)
from config import (
    DATA_DIR, HUMAN_ORDER, HUMAN_ORDER_MERGED, SAMPLE_KW, SG_MODEL_NAME,
)
from data import (
    apply_item_count_n_eff, clip_scores_to_floors, drop_model_benchmark_cells,
    drop_model_observations, drop_zero_scores, load_benchmark_ceilings,
    load_benchmark_floors, load_eci_data, release_time_covariate,
)
from lineage import build_lineage_structure
from models.mirt import build_mirt_model
from persistence import load_trace, save_df, save_json, save_pit, save_summary, save_trace
from ppc import compute_gof, posterior_predictive_mirt
from viz import (
    all_models_forest_fig, capability_timeline_fig, density_overlay_fig,
    forest_fig, hyperparams_fig, pit_ecdf_fig, pit_hist_fig, pred_vs_obs_fig,
    residuals_per_benchmark_fig, save_fig, sota_forest_fig,
)

ROOT = Path(__file__).resolve().parent


class _Heartbeat:
    """Live, file-friendly sampling progress.

    PyMC's rich progress bar renders in-place for a real terminal, so it only
    dumps a final frame to a piped/streamed log — the run looks frozen. This
    callback instead prints a plain line every `every` draws per chain; sampler
    worker processes inherit the parent's stdout, so flush=True reaches the
    stream. Also tallies divergences seen so far per chain.
    """
    def __init__(self, total: int, every: int = 250):
        self.total = total
        self.every = every
        self.seen: dict = {}
        self.div: dict = {}

    def __call__(self, trace, draw):
        c = draw.chain
        self.seen[c] = self.seen.get(c, 0) + 1
        stats = draw.stats[0] if getattr(draw, "stats", None) else {}
        if stats.get("diverging"):
            self.div[c] = self.div.get(c, 0) + 1
        n = self.seen[c]
        if n % self.every == 0:
            phase = "tune" if draw.tuning else "draw"
            print(f"  [chain {c}] {phase} {n}/{self.total}  div={self.div.get(c, 0)}",
                  flush=True)


def convergence(idata) -> dict:
    """Max r-hat, min bulk-ESS, and divergence count across the trace."""
    rh = az.rhat(idata)
    ess = az.ess(idata)
    max_rhat = float(max(float(v.max()) for v in rh.data_vars.values()))
    min_ess = float(min(float(v.min()) for v in ess.data_vars.values()))
    div = int(idata.sample_stats["diverging"].sum()) if "diverging" in idata.sample_stats else -1
    n_draws = int(idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"])
    return {"max_rhat": max_rhat, "min_ess": min_ess, "divergences": div, "n_draws": n_draws}


def sample_mirt(data, K: int, sample_kw: dict, human_order=None, lineage=None,
                lineage_bm=False, variant_offsets=True,
                loading_prior="signed", floor_c=None,
                ceiling_d=None, ceiling_noise=False, known_se=False,
                pooled_noise=False, shared_base_zsn=True, time_t=None,
                theta_t_cells=False, theta_pos=False,
                checkpoint_path=None, stream_path=None):
    """Build + sample one compensatory MIRT; print identified convergence."""
    tag = f"{loading_prior} loadings"
    if human_order:
        tag += ", ordered-human prior"
    if lineage:
        tag += ", Brownian lineage prior" if lineage_bm else ", lineage prior"
    if time_t is not None:
        tag += ", time prior"
    if floor_c is not None:
        tag += ", fixed-c 3PL"
    if ceiling_d is not None:
        tag += ", fixed-d ceiling" if floor_c is None else " + fixed-d ceiling (4PL)"
    if ceiling_noise:
        tag += ", noise-ceiling 4PL"
    if known_se:
        tag += ", known-SE noise"
    if pooled_noise:
        tag += ", pooled noise"
    if not shared_base_zsn:
        tag += ", private structured bases"
    if theta_t_cells:
        tag += ", cell-wise t theta"
    if theta_pos:
        tag += ", positive theta (softplus link)"
    sampler = sample_kw.get("nuts_sampler", "pymc")
    print(f"\n{'='*70}\nFitting MIRT  K = {K}  ({tag}, sampler={sampler})  "
          f"({data.n_obs} obs / {data.n_models} models / {data.n_benchmarks} benchmarks)\n{'='*70}",
          flush=True)
    model = build_mirt_model(data, K, loading_prior=loading_prior,
                             human_order=human_order, lineage=lineage,
                             lineage_bm=lineage_bm,
                             variant_offsets=variant_offsets,
                             floor_c=floor_c, ceiling_d=ceiling_d,
                             ceiling_noise=ceiling_noise, known_se=known_se,
                             pooled_noise=pooled_noise,
                             shared_base_zsn=shared_base_zsn,
                             time_t=time_t, theta_t_cells=theta_t_cells,
                             theta_pos=theta_pos)
    # PyMC sampler uses the heartbeat callback below (so its bar stays off);
    # nutpie/numpyro have no heartbeat, so let their own live progress bar show.
    sample_kw = {**sample_kw, "progressbar": sampler != "pymc"}
    # Low-rank Fisher mass matrix: the diagonal metric rescales each axis
    # independently and misses the dominant correlation directions (theta/A
    # coupling, the tau funnels), so NUTS compensates with deeper trees / more
    # leapfrog steps per draw. The low-rank update captures a few of those
    # directions, cutting steps per effective sample. nutpie-only kwarg.
    if sampler == "nutpie":
        nsk = {**sample_kw.get("nuts_sampler_kwargs", {})}
        nsk.setdefault("low_rank_modified_mass_matrix", True)
        # nutpie stores ALL warmup draws by default: the 12x6000/tune-12000
        # flagship trace carried 12.3 GB of warmup nothing reads, and that dead
        # weight (held in RAM through conversion + log-likelihood) is what OOM-
        # killed two finished 10h+ runs on 2026-08-10. No diagnostic in this
        # repo touches warmup_posterior.
        nsk.setdefault("save_warmup", False)
        if stream_path is not None:
            # Draws stream to disk as they are produced, so a kill, an OOM or a
            # power cut costs the unfinished tail instead of the whole run. The
            # arrays are pre-allocated at full shape and filled in order, which
            # is what makes an in-progress store readable
            # (persistence.load_live_draws trims the NaN tail).
            from nutpie import zarr_store
            stream_path.parent.mkdir(parents=True, exist_ok=True)
            nsk["zarr_store"] = zarr_store.LocalStore(str(stream_path), mkdir=True)
        sample_kw["nuts_sampler_kwargs"] = nsk
    extra = {"callback": _Heartbeat(total=sample_kw["tune"] + sample_kw["draws"])} \
        if sampler == "pymc" else {}
    t0 = time.time()
    if sampler != "pymc":
        print(f"  starting {sampler} sampler at t=0 (no live heartbeat with this backend)",
              flush=True)
    if stream_path is not None:
        print(f"  streaming draws to {stream_path} (read mid-run with "
              f"persistence.load_live_draws)", flush=True)
    with model:
        idata = pm.sample(**sample_kw, **extra)
        # nutpie's zarr store writes the warmup draws whatever save_warmup says,
        # and hands them back in the idata too. That is the RAM the note above is
        # about, so drop them here; the disk copy is inert.
        for group in ("warmup_posterior", "warmup_sample_stats"):
            if group in idata.groups():
                delattr(idata, group)
        if checkpoint_path is not None:
            # Crash insurance: bank the raw draws BEFORE the memory-heavy
            # log-likelihood pass. nutpie holds every draw in RAM, so a kill in
            # post-processing erases the whole run — it cost a finished
            # 10-hour 12x26,000 fit on 2026-08-10 (OOM while a concurrent
            # fit was saving). Overwritten by the full save on success.
            save_trace(idata, checkpoint_path)
            print(f"  raw trace checkpointed to {checkpoint_path}", flush=True)
        print(f"  sampling done in {time.time()-t0:.1f}s, computing log-likelihood...",
              flush=True)
        # pm.compute_log_likelihood can't convert models carrying custom initial
        # values (fgraph limitation). Any initval has served its purpose once
        # sampling is done — clear before the ll pass.
        model.rvs_to_initial_values = dict.fromkeys(model.rvs_to_initial_values)
        pm.compute_log_likelihood(idata)
    conv = convergence(idata)
    ident = mirt_identified_rhat(idata, data)
    # Raw r-hat is inflated by axis-permutation label switching — report it, but
    # the HONEST verdict is r-hat on identified quantities (eta, D, sigma_b).
    print(f"  K={K}: raw max r̂={conv['max_rhat']:.3f} (label-switching, ignore for K>1)  "
          f"divergences={conv['divergences']} / {conv['n_draws']}", flush=True)
    print(f"        identified: eta r̂ max={ident['eta_max_rhat']:.3f} "
          f"mean={ident['eta_mean_rhat']:.3f} (frac>1.01={ident['eta_frac_gt_1.01']:.2f})  "
          f"D r̂={ident.get('D_max_rhat', float('nan')):.3f}  "
          f"sigma_b r̂={ident.get('sigma_b_max_rhat', float('nan')):.3f}", flush=True)
    return idata, conv


# ── Canonical preset: the headline ECI pipeline ─────────────────────────────

def run_canonical(args) -> None:
    results_dir = config.RESULTS_DIR / "canonical"
    plots_dir = config.PLOTS_DIR / "canonical"
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    if args.raw_c:
        config.RAW_C_MODE = True
        print("── --raw-c mode: ECI = C (no affine anchor rescaling) ───────────")
    if args.include_all_benchmarks:
        print("── --include-all-benchmarks: curated exclusions NOT applied ─────")

    # --eci-data-only fits the reference eci_data.csv, whose "pretty names"
    # don't match the config's versioned anchors — remap on the config module
    # so analysis (attribute access) picks them up.
    if args.eci_data_only:
        config.ANCHOR_LOW  = ("Claude 3.5 Sonnet (October 2024)", 130.0)
        config.ANCHOR_HIGH = ("GPT-5", 150.0)
        config.SOTA_MODELS = [
            "GPT-5", "o3-pro", "o3", "Gemini 2.5 Pro (Mar 2025)", "o1",
            "o1-mini", "Gemini 1.5 Pro", "GPT-4o (May 2024)",
            "GPT-4 Turbo (Apr 2024)", "Claude 3 Opus", "GPT-4 (Mar 2023)",
        ]
        config.RELEASE_DATES.update({
            "Claude 3.5 Sonnet (October 2024)": "2024-10-22",
            "GPT-5":                            "2025-08-07",
            "GPT-5 Pro":                        "2025-10-06",
            "o3-pro":                           "2025-06-10",
            "o3":                               "2025-04-16",
            "Gemini 2.5 Pro (Mar 2025)":        "2025-03-25",
            "o1":                               "2024-12-17",
            "o1-mini":                          "2024-09-12",
            "Gemini 1.5 Pro":                   "2024-05-24",
            "GPT-4o (May 2024)":                "2024-05-13",
            "GPT-4 Turbo (Apr 2024)":           "2024-04-09",
            "Claude 3 Opus":                    "2024-02-29",
            "GPT-4 (Mar 2023)":                 "2023-03-14",
        })
        print("── --eci-data-only mode: using the reference ECI dataset ─────")

    print("── Loading data ─────────────────────────────────────────────────")
    data = load_eci_data(eci_data_only=args.eci_data_only,
                         drop_low_obs_models=False,
                         collapse_effort_variants=False,
                         include_all_benchmarks=args.include_all_benchmarks,
                         min_release_date="2024-01-01" if args.post_2023 else None)
    if args.drop_zero_scores:
        before = data.n_obs
        data = drop_zero_scores(data)
        print(f"   --drop-zero-scores: removed {before - data.n_obs} zero-score observations")
    print(f"   n_obs={data.n_obs}  n_models={data.n_models}  "
          f"n_benchmarks={data.n_benchmarks}  zero_scores={data.zero_score_mask.sum()}")

    sample_kw = {
        **SAMPLE_KW,
        "draws": args.draws if args.draws else SAMPLE_KW["draws"],
        "tune": args.tune if args.tune else SAMPLE_KW["tune"],
        "nuts_sampler": args.sampler,
    }
    if args.chains:
        sample_kw["chains"] = args.chains
        sample_kw["cores"] = args.chains

    trace_path = results_dir / "trace.nc"
    if args.skip_sampling:
        print(f"   reusing trace from {trace_path}")
        trace = load_trace(trace_path)
    else:
        trace, _ = sample_mirt(data, 1, sample_kw, loading_prior="normal")
        trace.posterior.attrs["mirt_loading_prior"] = "normal"
        save_trace(trace, trace_path)
        print(f"   saved trace → {trace_path}")

    print("\n── Convergence ──────────────────────────────────────────────────")
    summary = az.summary(trace, round_to=4)
    save_summary(trace, results_dir / "summary.csv")
    n_div = int(trace.sample_stats["diverging"].sum())
    print(f"   max r_hat = {float(np.nanmax(summary['r_hat'].values)):.4f}")
    print(f"   min ess   = {float(np.nanmin(summary[['ess_bulk', 'ess_tail']].values)):.0f}")
    print(f"   divergences = {n_div}")

    # ── SOTA + ECI ──────────────────────────────────────────────────────────
    print("\n── SOTA + ECI ───────────────────────────────────────────────────")
    if args.eci_data_only:
        raw_df = (pd.read_csv(DATA_DIR / "raw" / "eci_data.csv")
                    .rename(columns={"model": "model_version"}))
        raw_df["release_date"] = raw_df["model_version"].map(config.RELEASE_DATES)
    else:
        raw_df = pd.read_csv(DATA_DIR / "processed" / "benchmarks_merged.csv")
    sota = sota_stats_df(trace, data, raw_df)
    save_df(sota, results_dir / "sota.csv")
    print(sota[["model", "release_date", "C_mean", "ECI_mean"]].to_string(index=False))

    # ── Forests for D and the K=1 loading A (the discrimination) ───────────
    print("\n── Forest summaries (D, A) ──────────────────────────────────────")
    benchmarks = data.blookup["benchmark"].tolist()
    D_df = forest_stats_df(trace, "D", benchmarks)
    A_draws = trace.posterior["A"].values[..., 0]
    A_df = forest_stats_from_draws(A_draws.reshape(-1, data.n_benchmarks), benchmarks)
    save_df(D_df, results_dir / "forest_D.csv")
    save_df(A_df, results_dir / "forest_A.csv")

    # ── PPC + GoF ──────────────────────────────────────────────────────────
    print("\n── Posterior predictive + GoF ───────────────────────────────────")
    y_rep_flat = posterior_predictive_mirt(trace, data)
    mu_flat = posterior_predictive_mirt(trace, data, return_mean=True)
    gof = compute_gof(y_rep_flat, data, mu_flat)
    save_json(gof.metrics, results_dir / "gof.json")
    save_pit(gof.pit, data, results_dir / "pit.csv")
    for k, v in gof.metrics.items():
        fmt = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"   {k:34s} {fmt}")

    # Residuals per-benchmark (nonzero scores only — zero-score residuals are
    # dominated by the clip and not meaningful for the per-benchmark box plot)
    nonzero = ~data.zero_score_mask
    resid_df = pd.DataFrame({
        "benchmark": data.blookup["benchmark"].values[data.bench_idx[nonzero]],
        "residual":  data.scores[nonzero] - gof.y_pred_mean[nonzero],
    })

    # ── Plots ──────────────────────────────────────────────────────────────
    print("\n── Plots ────────────────────────────────────────────────────────")
    save_fig(hyperparams_fig(trace), "hyperparams", plots_dir)
    save_fig(forest_fig(D_df, "D", "Per-benchmark difficulty (D)"),
             "forest_D", plots_dir)
    save_fig(forest_fig(A_df, "A", "Per-benchmark discrimination (A, K=1 loading)"),
             "forest_A", plots_dir)

    anchors = [
        (config.ANCHOR_LOW[1],  f"{config.ANCHOR_LOW[0]} = {config.ANCHOR_LOW[1]:.0f}"),
        (config.ANCHOR_HIGH[1], f"{config.ANCHOR_HIGH[0]} = {config.ANCHOR_HIGH[1]:.0f}"),
    ]
    save_fig(sota_forest_fig(sota, x_col="ECI",
                             title="SOTA ECI (anchored: Claude 3.5 Sonnet = 130, GPT-5 = 150)",
                             xaxis_title="ECI (affine-anchored)",
                             anchors=anchors),
             "sota_eci", plots_dir)
    save_fig(sota_forest_fig(sota, x_col="C",
                             title="SOTA capability (C)",
                             xaxis_title="C"),
             "sota_C", plots_dir)

    humans = human_stats_df(trace, data)
    save_df(humans, results_dir / "human_groups.csv")
    print(f"\n   Human groups fitted: {len(humans)}")
    if len(humans):
        print(humans.to_string(index=False))

    all_models = all_models_stats_df(trace, data, metric="C")
    save_df(all_models, results_dir / "all_models_C.csv")
    ai_only = all_models[~all_models["name"].isin(set(humans["name"]) if len(humans) else set())]
    print(f"   AI models in forest: {len(ai_only)}")
    save_fig(all_models_forest_fig(ai_only, metric="C"),
             "all_models_C", plots_dir)

    all_models_eci = all_models_stats_df(trace, data, metric="ECI")
    save_df(all_models_eci, results_dir / "all_models_eci.csv")
    ai_only_eci = all_models_eci[~all_models_eci["name"].isin(
        set(humans["name"]) if len(humans) else set())]
    save_fig(all_models_forest_fig(ai_only_eci, anchors=anchors),
             "all_models_eci", plots_dir)

    tl = timeline_stats_df(trace, data, raw_df)
    if len(humans):
        tl = tl[~tl["name"].isin(set(humans["name"]))]
    save_df(tl, results_dir / "timeline.csv")
    save_fig(capability_timeline_fig(
                tl, human_stats=humans,
                annotate_benchmarks=["GSM8K", "MMLU", "GPQA Diamond",
                                     "Humanity's Last Exam", "GSO-Bench",
                                     "OS World (Screenshot)"],
                annotate_models=[config.ANCHOR_LOW[0], config.ANCHOR_HIGH[0]]),
             "capability_timeline", plots_dir)

    save_fig(pit_hist_fig(gof.pit), "pit_hist", plots_dir)
    save_fig(pit_ecdf_fig(gof.pit), "pit_ecdf", plots_dir)
    save_fig(density_overlay_fig(y_rep_flat, data.scores), "ppc_density", plots_dir)

    hover = [
        f"{data.mlookup['model'].values[data.model_idx[i]]}"
        f" @ {data.blookup['benchmark'].values[data.bench_idx[i]]}"
        for i in range(data.n_obs)
    ]
    save_fig(pred_vs_obs_fig(data.scores, gof.y_pred_mean, hover),
             "pred_vs_obs", plots_dir)
    save_fig(residuals_per_benchmark_fig(resid_df),
             "residuals_per_benchmark", plots_dir)

    print(f"\nDone. Results in {results_dir}, plots in {plots_dir}.")


# ── Exploration: the K-axis MIRT campaign driver ────────────────────────────

def run_exploration(args, parser) -> None:
    # One tag identifies the fit config; reused for BOTH the results/plots folder
    # and the trace filename, so distinct configs never overwrite each other's
    # fixed-name artefacts. The loading prior leads the tag (matching the
    # historical `trace_mirt_k3_signed_...` convention); "normal" stays
    # untagged so its folders reduce to the canonical mirt_humanprior etc.
    tag = "" if args.loading_prior == "normal" else f"_{args.loading_prior}"
    if args.human_merge: tag += "_humanmerge"
    elif args.human_prior: tag += "_humanprior"
    if args.lineage_prior: tag += "_lineageprior"
    if args.lineage_bm: tag += "_lineagebm"
    if args.time_prior: tag += "_timeprior"
    if args.theta_t: tag += "_thetat"
    if args.theta_pos: tag += "_thetapos"
    if args.min_release_date:
        tag += f"_since{args.min_release_date[:4]}"
    if args.no_sg: tag += "_noSG"
    if args.no_sg_gpqa: tag += "_noSGgpqa"
    if args.no_sg_arcagi: tag += "_noSGarcagi"
    if args.apply_exclusions: tag += "_excluded"
    if args.cyber: tag += "_cyber"
    if args.simpleqa_original: tag += "_sqaorig"
    if args.drop_benchmarks:
        tag += "_drop" + "".join(re.sub(r"\W", "", b) for b in args.drop_benchmarks)
    if args.private_bases: tag += "_privbase"
    if args.floors: tag += "_floors"
    if args.ceilings: tag += "_ceilings"
    if args.ceiling_noise: tag += "_ceilnoise"
    if args.known_se: tag += "_knownse"
    if args.item_counts: tag += "_itemcounts"
    if args.pooled_noise: tag += "_poolednoise"

    human_order = (HUMAN_ORDER_MERGED if args.human_merge
                   else HUMAN_ORDER if args.human_prior else None)
    results_dir = ROOT / "results" / (f"mirt{tag}" if tag else "mirt")
    plots_dir = ROOT / "plots" / (f"mirt{tag}" if tag else "mirt")
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    if tag:
        print(f"  per-config folder: {results_dir}", flush=True)

    sample_kw = {
        **SAMPLE_KW,
        "draws": args.draws if args.draws else 2000,
        "tune": args.tune if args.tune else 2000,
        "chains": args.chains if args.chains else SAMPLE_KW["chains"],
        "cores": args.chains if args.chains else SAMPLE_KW["chains"],
        "target_accept": args.target_accept,
        "nuts_sampler": args.sampler,
    }
    if args.seed is not None:
        sample_kw["random_seed"] = args.seed

    data = load_eci_data(
        include_all_benchmarks=not args.apply_exclusions,
        fit_cyber=args.cyber,
        fit_simpleqa_original=args.simpleqa_original,
        drop_benchmarks=args.drop_benchmarks,
        min_release_date=args.min_release_date)
    if args.apply_exclusions:
        print("  --apply-exclusions: curated exclusions applied at fit time "
              "(dropped benchmarks from excluded_benchmarks.txt)", flush=True)
    if sum([args.no_sg, args.no_sg_gpqa, args.no_sg_arcagi]) > 1:
        parser.error("--no-sg / --no-sg-gpqa / --no-sg-arcagi are mutually exclusive "
                     "(one drops all SG obs, the others only its GPQA / ARC-AGI cells).")
    if args.no_sg:
        n_before = data.n_obs
        data = drop_model_observations(data, [SG_MODEL_NAME])
        print(f"  --no-sg: dropped {n_before - data.n_obs} '{SG_MODEL_NAME}' "
              f"observations (tier kept in model index, prior-only theta)", flush=True)
    if args.no_sg_gpqa:
        n_before = data.n_obs
        gpqa = [b for b in data.blookup["benchmark"] if "GPQA" in b]
        data = drop_model_benchmark_cells(data, SG_MODEL_NAME, gpqa)
        print(f"  --no-sg-gpqa: dropped {n_before - data.n_obs} '{SG_MODEL_NAME}' "
              f"GPQA cells {gpqa} (other SG scores kept)", flush=True)
    if args.no_sg_arcagi:
        n_before = data.n_obs
        # ARC-AGI abstraction family only ("ARC-AGI", "ARC-AGI-2") — NOT the easy
        # "ARC (AI2)" science-QA benchmark, which is a different construct.
        arcagi = [b for b in data.blookup["benchmark"] if b.startswith("ARC-AGI")]
        data = drop_model_benchmark_cells(data, SG_MODEL_NAME, arcagi)
        print(f"  --no-sg-arcagi: dropped {n_before - data.n_obs} '{SG_MODEL_NAME}' "
              f"ARC-AGI cells {arcagi} (other SG scores kept)", flush=True)
    floor_c = None
    if args.floors:
        floor_c = load_benchmark_floors(data)
        n_before = int((data.scores < floor_c[data.bench_idx]).sum())
        data = clip_scores_to_floors(data, floor_c)
        print(f"  --floors: fixed-c 3PL; clipped {n_before} below-floor scores up "
              f"to their benchmark chance floor", flush=True)
    ceiling_d = None
    if args.ceilings:
        ceiling_d = load_benchmark_ceilings(data)
        capped = [(b, d) for b, d in
                  zip(data.blookup.sort_values("benchmark_idx")["benchmark"],
                      ceiling_d) if d < 1.0]
        print(f"  --ceilings: fixed-d upper asymptote on {len(capped)} "
              f"benchmark(s): " + ", ".join(f"{b} d={d}" for b, d in capped),
              flush=True)
    if args.known_se:
        measured = np.isfinite(data.n_eff)
        print(f"  --known-se: {int(measured.sum())} of {data.n_obs} cells "
              f"({measured.mean():.1%}) carry a reported stderr, median "
              f"effective test length {np.median(data.n_eff[measured]):.0f} "
              f"tasks; the rest keep the estimated per-benchmark noise",
              flush=True)
        if args.item_counts:
            before = int(measured.sum())
            data = apply_item_count_n_eff(data)
            measured = np.isfinite(data.n_eff)
            print(f"  --item-counts: +{int(measured.sum()) - before} cells get "
                  f"the verified item-count floor; instrument coverage now "
                  f"{int(measured.sum())} of {data.n_obs} ({measured.mean():.1%})",
                  flush=True)
    elif args.item_counts:
        raise SystemExit("--item-counts requires --known-se (it only fills the "
                         "n_eff slot that machinery reads)")
    print(f"\nData: {data.n_obs} obs / {data.n_models} models / "
          f"{data.n_benchmarks} benchmarks", flush=True)
    bench_names = data.blookup.sort_values("benchmark_idx")["benchmark"].tolist()
    model_names = data.mlookup.sort_values("model_idx")["model"].tolist()

    if args.lineage_bm and not args.lineage_prior:
        parser.error("--lineage-bm re-indexes the lineage increments by time; it "
                     "needs --lineage-prior.")
    lineage = (build_lineage_structure(data.mlookup)
               if args.lineage_prior else None)
    if args.lineage_prior:
        if lineage is None:
            parser.error("--lineage-prior set but no usable chains in lineage_map.csv")
        kind = "Brownian (steps scale with the release gap)" if args.lineage_bm \
            else "soft (iid per release)"
        print(f"  --lineage-prior ({kind}): {lineage.n_chains} chains / "
              f"{lineage.n_nodes} nodes / {len(lineage.row_idx)} chained models",
              flush=True)

    # Time covariate needs the lineage structure (chained rows take their chain
    # founder's date), so it is built after it and works with lineage=None too.
    time_t = release_time_covariate(data.mlookup, lineage) if args.time_prior else None
    if time_t is not None:
        dated = int((time_t != 0.0).sum())
        print(f"  --time-prior: per-axis linear trend on the theta prior mean; "
              f"{dated} dated rows spanning {time_t.min():+.2f}..{time_t.max():+.2f} "
              f"years around the mean, {len(time_t) - dated} undated rows at 0",
              flush=True)

    # ── Fit the overcomplete MIRT ─────────────────────────────────────────
    idata_k, conv_k = sample_mirt(data, args.K, sample_kw,
                                  human_order=human_order, lineage=lineage,
                                  lineage_bm=args.lineage_bm, time_t=time_t,
                                  loading_prior=args.loading_prior, floor_c=floor_c,
                                  ceiling_d=ceiling_d,
                                  ceiling_noise=args.ceiling_noise,
                                  known_se=args.known_se,
                                  pooled_noise=args.pooled_noise,
                                  shared_base_zsn=not args.private_bases,
                                  theta_t_cells=args.theta_t,
                                  theta_pos=args.theta_pos,
                                  checkpoint_path=results_dir
                                  / f"trace_mirt_k{args.K}{tag}.nc",
                                  stream_path=(results_dir / "live_draws.zarr"
                                               if args.stream_draws else None))
    # Loading prior travels with the trace so prepare_fit can gate
    # rotation/sign handling without caller flags.
    idata_k.posterior.attrs["mirt_loading_prior"] = args.loading_prior
    if human_order:
        idata_k.posterior.attrs["mirt_human_order"] = json.dumps(human_order)
    if lineage:
        idata_k.posterior.attrs["mirt_lineage_chains"] = json.dumps(lineage.chain_names)
        idata_k.posterior.attrs["mirt_lineage_bm"] = json.dumps(bool(args.lineage_bm))
    if time_t is not None:
        idata_k.posterior.attrs["mirt_time_prior"] = json.dumps(True)
    if args.drop_benchmarks:
        idata_k.posterior.attrs["mirt_drop_benchmarks"] = json.dumps(args.drop_benchmarks)
    if args.cyber:
        idata_k.posterior.attrs["mirt_cyber"] = json.dumps(True)
    if args.simpleqa_original:
        idata_k.posterior.attrs["mirt_simpleqa_original"] = json.dumps(True)
    if args.loading_prior == "bifactor":
        # Axis identity is structural here (axis1 IS the general column), so
        # name the axes on the trace and let every plot label them.
        idata_k.posterior.attrs["mirt_axis_names"] = json.dumps(
            ["g"] + [f"specific{k}" for k in range(1, args.K)])
    if args.private_bases:
        idata_k.posterior.attrs["mirt_shared_base_zsn"] = json.dumps(False)
    if args.theta_t:
        idata_k.posterior.attrs["mirt_theta_t_cells"] = json.dumps(True)
    if args.theta_pos:
        idata_k.posterior.attrs["mirt_theta_pos"] = json.dumps(True)
    if floor_c is not None:
        idata_k.posterior.attrs["mirt_floor_c"] = json.dumps(floor_c.tolist())
    if ceiling_d is not None:
        idata_k.posterior.attrs["mirt_fixed_ceiling_d"] = json.dumps(ceiling_d.tolist())
    if args.known_se:
        idata_k.posterior.attrs["mirt_known_se"] = json.dumps(True)
    if args.item_counts:
        idata_k.posterior.attrs["mirt_item_counts"] = json.dumps(True)
    if args.pooled_noise:
        idata_k.posterior.attrs["mirt_pooled_noise"] = json.dumps(True)
    if args.rotation != "promax":
        # Display rotation travels with the trace so prepare_fit (dashboard,
        # plot_mirt) picks the same interpretation frame the CSVs use here.
        idata_k.posterior.attrs["mirt_display_rotation"] = args.rotation
    save_trace(idata_k, results_dir / f"trace_mirt_k{args.K}{tag}.nc")

    # ── Factors: rank-track then oblique-rotate ────────────────────────────
    A, theta, tau = mirt_factors_from_trace(idata_k)

    # tau spectrum is a PRE-rotation quantity: rotation mixes axes and would
    # scramble the per-axis scales, so dimensionality is read here, first.
    # Bifactor exception: its tau_A is [tau_g, tau_hs, tau_hs, ...] — the
    # specifics share one global scale by construction, so tau cannot tell them
    # apart. Realised per-axis strength lives in the loading column norms (the
    # horseshoe's local scales are per cell), and that is also what shows a
    # specific the data did not need collapsing to an empty column.
    spec = tau_spectrum_df(
        np.linalg.norm(A, axis=1) if args.loading_prior == "bifactor" else tau)
    save_df(spec, results_dir / "mirt_tau_spectrum.csv")
    print("\n── tau_A spectrum (gap ratio flags the live/dead break) ──")
    print(spec.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # Oblique rotation → simple-structure bundles. Three paths:
    #   * signed/signedhs, K>1 — every draw sits in its own orientation, so a
    #     mean-then-rotate would cancel structure; align each draw (rotation +
    #     sign + permutation) with the SAME promax frame prepare_fit uses, so
    #     these CSVs match the dashboard.
    #   * --rotation none — raw frame: the energy rank-tracking in
    #     mirt_factors_from_trace already fixed the cross-chain axis
    #     permutation, and on the non-negative prior the positivity constraint
    #     pins the rotation itself (raw vs rotated axes agree at corr ≥ 0.98
    #     on the K=3 floors fit). Phi is the raw ability correlation.
    #   * 'normal' prior otherwise — no sign symmetry; rotate the
    #     posterior-mean loadings and apply to every draw. Default promax;
    #     --rotation nonneg keeps loadings non-negative (pure positive
    #     bundles, no contrast — only sound on the non-negative prior).
    frame_label = "rotated bundle"
    if args.loading_prior in ("signed", "signedhs") and args.K > 1:
        aligned = align_rotations(A, theta, method="promax")
        A, theta, Phi = aligned.A, aligned.theta, aligned.Phi
    elif args.loading_prior == "bifactor":
        # The bifactor frame IS the axes: the prior's dense/sparse asymmetry
        # already picked the orientation (that is what identifies g), so any
        # rotation here would mix the general column back into the specifics
        # and undo it. Phi is the raw ability correlation; g-vs-specific
        # overlap is a RESULT to read, not something a rotation should set.
        Phi = (np.corrcoef(theta.mean(axis=0).T) if args.K > 1
               else np.array([[1.0]]))
        frame_label = "bifactor frame (axis1 = g, no rotation)"
    elif args.rotation == "none":
        Phi = (np.corrcoef(theta.mean(axis=0).T) if args.K > 1
               else np.array([[1.0]]))
        frame_label = "raw bundle (rank-tracked, no rotation)"
    else:
        rotate = nonneg_rotate if args.rotation == "nonneg" else promax_rotate
        Tload, Ttheta, Phi = rotate(A.mean(axis=0))
        A, theta, Phi = apply_rotation(A, theta, Phi, Tload, Ttheta)

    corr = factor_corr_df(Phi)
    save_df(corr.reset_index().rename(columns={"index": "axis"}),
            results_dir / "mirt_factor_corr.csv")
    print("\n── factor correlation Phi (off-diag = axis-ability overlap) ──")
    print(corr.to_string(float_format=lambda x: f"{x:.3f}"))

    loadings = loadings_table(A, bench_names, data.bench_category)
    save_df(loadings, results_dir / "mirt_loadings.csv")

    scores = factor_scores_df(theta, model_names, is_human=data.is_human)
    save_df(scores, results_dir / "mirt_factor_scores.csv")

    for k in range(min(args.K, 4)):
        sub = loadings[loadings["axis"] == f"axis{k + 1}"]
        print(f"\n── axis{k + 1}  top loadings ({frame_label}) ──")
        print(sub.head(10)[["benchmark", "category", "loading_median"]]
              .to_string(index=False))

    # ── PPC goodness-of-fit + residual analysis ───────────────────────────
    # PSIS-LOO is unreliable here (≈5 obs/model → most points high Pareto-k),
    # so we use the same PPC-based GoF the canonical pipeline uses: Bayesian
    # R², RMSE, MAE on posterior-predictive draws, plus per-observation
    # residuals for the residual-per-benchmark diagnostic.
    n_eff = data.n_eff if args.known_se else None

    def gof_report(idata, label, gtag):
        y_rep = posterior_predictive_mirt(idata, data, floor_c=floor_c,
                                          ceiling_d=ceiling_d, n_eff=n_eff)
        mu = posterior_predictive_mirt(idata, data, floor_c=floor_c,
                                       ceiling_d=ceiling_d, n_eff=n_eff,
                                       return_mean=True)
        gof = compute_gof(y_rep, data, mu)
        save_json(gof.metrics, results_dir / f"mirt_gof_{gtag}.json")
        m = gof.metrics
        print(f"  {label}: R²={m['bayesian_r2']:.3f}  RMSE={m['rmse']:.3f}  "
              f"MAE={m['mae']:.3f}")
        return gof

    print("\n── PPC goodness-of-fit ──")
    gof_k = gof_report(idata_k, f"MIRT (K={args.K})", f"k{args.K}")

    resid_df = pd.DataFrame({
        "model":     [model_names[i] for i in data.model_idx],
        "benchmark": [bench_names[i] for i in data.bench_idx],
        "score":     data.scores,
        "pred_mean": gof_k.y_pred_mean,
        "residual":  data.scores - gof_k.y_pred_mean,
    })
    save_df(resid_df, results_dir / "mirt_residuals.csv")

    if not args.skip_baseline:
        baseline_path = results_dir / "trace_mirt_k1.nc"
        if baseline_path.exists() and not args.refit_baseline:
            # The K=1 MIRT is data-shape-locked: it indexes the same (model,
            # benchmark) coords as the K-axis fit, so a cached trace is safe to
            # reuse only when those dims still match the current data. The K=1
            # GoF (R²/RMSE/MAE) is loaded from the cached JSON instead of
            # recomputed — no extra forward pass.
            idata_1d = az.from_netcdf(baseline_path)
            post = idata_1d.posterior
            if post.sizes.get("model") == data.n_models and \
               post.sizes.get("bench") == data.n_benchmarks:
                print(f"\nReusing cached K=1 baseline → {baseline_path}", flush=True)
                gof_path = results_dir / "mirt_gof_k1.json"
                if gof_path.exists():
                    m = json.loads(gof_path.read_text())
                    print(f"  1D (K=1, cached): R²={m['bayesian_r2']:.3f}  "
                          f"RMSE={m['rmse']:.3f}  MAE={m['mae']:.3f}", flush=True)
            else:
                print(f"\nCached K=1 dims (model={post.sizes.get('model')}, "
                      f"bench={post.sizes.get('bench')}) don't match current data "
                      f"({data.n_models}, {data.n_benchmarks}) — refitting.", flush=True)
                idata_1d, _ = sample_mirt(data, 1, sample_kw, loading_prior="normal",
                                          floor_c=floor_c, ceiling_d=ceiling_d,
                                          ceiling_noise=args.ceiling_noise,
                                          known_se=args.known_se)
                save_trace(idata_1d, baseline_path)
                gof_report(idata_1d, "1D (K=1)", "k1")
        else:
            idata_1d, _ = sample_mirt(data, 1, sample_kw, loading_prior="normal",
                                      floor_c=floor_c, ceiling_d=ceiling_d,
                                      ceiling_noise=args.ceiling_noise,
                                      known_se=args.known_se)
            save_trace(idata_1d, baseline_path)
            gof_report(idata_1d, "1D (K=1)", "k1")

    print(f"\nOutputs → {results_dir}")
    if args.plots:
        import subprocess
        # Forward EVERY flag that feeds plot_mirt's trace tag or data scope —
        # a missing one points it at a different fit's trace (or none at all).
        cmd = [sys.executable, str(ROOT / "diagnostics" / "plot_mirt.py"),
               "--K", str(args.K), "--axes", str(args.K),
               "--loading-prior", args.loading_prior]
        for flag, on in [("--human-prior", args.human_prior),
                         ("--human-merge", args.human_merge),
                         ("--lineage-prior", args.lineage_prior),
                         ("--lineage-bm", args.lineage_bm),
                         ("--time-prior", args.time_prior),
                         ("--no-sg", args.no_sg),
                         ("--no-sg-gpqa", args.no_sg_gpqa),
                         ("--no-sg-arcagi", args.no_sg_arcagi),
                         ("--apply-exclusions", args.apply_exclusions),
                         ("--floors", args.floors),
                         ("--ceilings", args.ceilings),
                         ("--ceiling-noise", args.ceiling_noise),
                         ("--known-se", args.known_se),
                         ("--pooled-noise", args.pooled_noise)]:
            if on:
                cmd.append(flag)
        print(f"\n--plots: rendering figures ({' '.join(cmd[2:])})", flush=True)
        subprocess.run(cmd, check=True)
    else:
        print("Figures: run diagnostics/plot_mirt.py")


def main():
    parser = argparse.ArgumentParser(
        description="Fit the compensatory Beta-MIRT: --preset canonical for the "
                    "headline ECI pipeline (K=1), or the K-axis exploration flags.")
    parser.add_argument("--preset", choices=["canonical"],
                        help="'canonical': K=1, normal prior, curated exclusions, "
                             "humans in, full ECI deliverables → results/canonical/")
    # Shared sampling controls.
    parser.add_argument("--draws", type=int, default=None,
                        help="posterior draws per chain (default: config.SAMPLE_KW "
                             "for canonical, 2000 for exploration)")
    parser.add_argument("--tune", type=int, default=None,
                        help="tuning steps per chain (same defaults as --draws)")
    parser.add_argument("--chains", type=int, default=None,
                        help="override number of chains and cores")
    parser.add_argument("--seed", type=int, default=None,
                        help="override config.SAMPLE_KW random_seed (42). Nutpie is "
                             "deterministic given seed+data+model, so a multi-run "
                             "recipe MUST vary this or the runs are duplicates")
    parser.add_argument("--target-accept", type=float, default=0.95)
    parser.add_argument("--sampler", default="nutpie",
                        choices=["pymc", "nutpie", "numpyro"],
                        help="NUTS backend (default nutpie: Rust, ~2-3x faster on CPU)")
    # Canonical-preset flags.
    parser.add_argument("--skip-sampling", action="store_true",
                        help="[canonical] reuse results/canonical/trace.nc")
    parser.add_argument("--drop-zero-scores", action="store_true",
                        help="[canonical] drop score==0 observations (diagnostic)")
    parser.add_argument("--eci-data-only", action="store_true",
                        help="[canonical] fit data/raw/eci_data.csv (the original "
                             "reference ECI dataset) instead of the processed file")
    parser.add_argument("--raw-c", action="store_true",
                        help="[canonical] report raw C instead of anchored ECI")
    parser.add_argument("--include-all-benchmarks", action="store_true",
                        help="[canonical] keep the curated-excluded benchmarks in the fit")
    parser.add_argument("--min-release-date", metavar="YYYY-MM-DD", default=None,
                        help="[exploration] drop models released before this date. "
                             "Undated models are kept: after the 2026-07-28 date "
                             "backfill the remaining undated ones are genuinely "
                             "recent (Manus, Muse Spark, audio models), and humans "
                             "have no release date by nature.")
    parser.add_argument("--post-2023", action="store_true",
                        help="[canonical] era sensitivity: drop models with a known "
                             "release date before 2024-01-01")
    # Exploration flags.
    parser.add_argument("--K", type=int, default=4,
                        help="[exploration] latent dimension")
    parser.add_argument("--loading-prior", default="signed",
                        choices=["signed", "normal", "signedhs", "bifactor"],
                        help="[exploration] loading-matrix prior. Default 'signed' "
                             "(Normal cells, rotation-invariant); 'normal' is "
                             "non-negative HalfNormal loadings; 'bifactor' is a "
                             "dense non-negative general column (axis1) plus "
                             "non-negative horseshoe specifics (axes 2..K), which "
                             "identifies g without assigning any benchmark to an "
                             "axis. Needs --K >= 2.")
    parser.add_argument("--human-prior", action="store_true",
                        help="[exploration] order human tiers by config.HUMAN_ORDER")
    parser.add_argument("--stream-draws", action="store_true",
                        help="[exploration] write every draw to "
                             "results/<fit>/live_draws.zarr as it is sampled, so a "
                             "killed run keeps what it had. Read a partial store "
                             "with persistence.load_live_draws. nutpie only; costs "
                             "the warmup draws in disk (nutpie's store ignores "
                             "save_warmup)")
    parser.add_argument("--human-merge", action="store_true",
                        help="[exploration] instead use config.HUMAN_ORDER_MERGED: "
                             "the same tiers with the High School branch merged "
                             "into the adult spine (Domain Expert beats both a "
                             "Skilled Generalist and a High School Qualifier, Top "
                             "Performer beats both a Domain Expert and a High "
                             "School Top Performer) via a max over parents. "
                             "Supersedes --human-prior")
    parser.add_argument("--lineage-prior", action="store_true",
                        help="[exploration] soft release-chain prior: each release's "
                             "mean step over its predecessor is positive, but a node "
                             "can regress; effort variants get tight mean-zero offsets")
    parser.add_argument("--lineage-bm", action="store_true",
                        help="[exploration] with --lineage-prior: index the chain by "
                             "TIME — each step's mean and variance scale with the "
                             "release gap in years (Brownian motion with drift), and "
                             "each vendor chain gets its own rate, pooled toward a "
                             "per-axis population rate.")
    parser.add_argument("--time-prior", action="store_true",
                        help="[exploration] add a learned per-axis linear trend in "
                             "release year to the theta prior MEAN, so thinly-"
                             "evaluated models are shrunk toward their era's level "
                             "instead of the whole population's. The slope is signed "
                             "and centered at zero (a flat population reduces to the "
                             "plain prior); chained models use their chain founder's "
                             "date; undated models and humans are unaffected.")
    parser.add_argument("--theta-t", action="store_true",
                        help="[exploration] cell-wise leptokurtic theta: give each "
                             "(model, axis) cell of the exchangeable block a "
                             "Student-t(4) marginal via a per-cell scale mixture "
                             "instead of a Gaussian one. The Gaussian block is "
                             "spherical and leaves the rotation orbit exactly flat; "
                             "independent heavy-tailed columns are the ICA "
                             "identification channel, so the likelihood can prefer "
                             "one orientation. Costs one extra latent per cell, and "
                             "widens the abilities of thinly-measured models")
    parser.add_argument("--theta-pos", action="store_true",
                        help="[exploration] positive likelihood-side ability, the "
                             "semi-compensatory convention: eta reads "
                             "theta_pos = softplus(theta) instead of theta, so with "
                             "non-negative loadings each axis can only add to a "
                             "score, never pull it below the sigmoid(-D) baseline. "
                             "Raw theta keeps every prior block (location pin, "
                             "human/lineage order) and stays the reported ability; "
                             "softplus is monotone, so rankings are unchanged")
    parser.add_argument("--no-sg", action="store_true",
                        help="[exploration] drop the Skilled Generalist tier's observations")
    parser.add_argument("--no-sg-gpqa", action="store_true",
                        help="[exploration] drop only the Skilled Generalist's GPQA cells")
    parser.add_argument("--no-sg-arcagi", action="store_true",
                        help="[exploration] drop only the Skilled Generalist's ARC-AGI cells")
    parser.add_argument("--apply-exclusions", action="store_true",
                        help="[exploration] apply excluded_benchmarks.txt at fit time "
                             "(the canonical scope; exploration default is the full set)")
    parser.add_argument("--cyber", action="store_true",
                        help="[exploration] append the unsaturated benchmarks of "
                             "Epoch's separate cyber ECI (data/curated/cyber_benchmarks.csv; "
                             "regenerate with `python -m diagnostics.fetch_cyber_eci`). "
                             "Adds frontier coverage the pipeline feeds do not carry")
    parser.add_argument("--simpleqa-original", action="store_true",
                        help="[exploration] append the original OpenAI SimpleQA "
                             "(data/curated/simpleqa_original/). A separate column "
                             "from SimpleQA Verified (different set and grader); "
                             "adds 2023-2024 era rows on the unsaturated-QA construct")
    parser.add_argument("--drop-benchmarks", type=lambda s: [b.strip() for b in s.split(",")],
                        default=None, metavar="A,B",
                        help="[exploration] drop the named benchmarks from the fit "
                             "(comma-separated, exact names). For targeted sensitivity "
                             "runs, e.g. the GBAEval+VPCT basin-flip test")
    parser.add_argument("--private-bases", action="store_true",
                        help="[exploration] give each human root and chain founder "
                             "a private Normal(0,1) base and let the ZeroSumNormal "
                             "span only the unstructured rows. Same marginal scale; "
                             "changes how much of the population the location pin "
                             "covers")
    parser.add_argument("--floors", action="store_true",
                        help="[exploration] fixed-c 3PL: clip scores up to each "
                             "benchmark's chance floor and set mu = c + (1-c)*sigmoid "
                             "(floors from data/curated/benchmark_lower_bounds.csv)")
    parser.add_argument("--ceilings", action="store_true",
                        help="[exploration] fixed-d upper asymptote: set "
                             "mu = c + (d-c)*sigmoid with d fixed per benchmark "
                             "from data/curated/benchmark_upper_bounds.csv "
                             "(benchmarks absent from the file keep d=1); with "
                             "--floors this is the fixed 4PL")
    parser.add_argument("--ceiling-noise", action="store_true",
                        help="[exploration] noise 4PL: estimate a per-benchmark "
                             "upper asymptote confined to a noise-sized gap, "
                             "Beta(1,20) (mean 0.048, P(gap>0.10)=0.122) — grading "
                             "noise, not walls. Composes with --ceilings: the gap "
                             "sits on top of each curated wall. Read ceiling_d "
                             "off the trace")
    parser.add_argument("--known-se", action="store_true",
                        help="[exploration] split the Beta noise: fixed per-cell "
                             "instrument precision from reported harness stderr "
                             "(n_eff = p(1-p)/se^2), sigma_b becomes excess-only. "
                             "Cells without stderr are unchanged.")
    parser.add_argument("--pooled-noise", action="store_true",
                        help="[exploration] hierarchical noise: learn the "
                             "population location/spread of the per-benchmark "
                             "noise scale instead of fixing it; thin benchmarks "
                             "shrink toward the shared median")
    parser.add_argument("--item-counts", action="store_true",
                        help="[exploration] with --known-se: cells with no "
                             "reported stderr get the instrument floor "
                             "n_eff = n_items from the VERIFIED rows of "
                             "data/curated/benchmark_n_items.csv (machine rows "
                             "only; conservative — multi-seed runs are treated "
                             "as single-run)")
    parser.add_argument("--rotation", default="promax",
                        choices=["promax", "nonneg", "none"],
                        help="[exploration] display rotation for loadings/timelines. "
                             "promax (default); nonneg keeps loadings non-negative "
                             "(pure positive bundles, no contrast — only sound on the "
                             "'normal' prior); none skips rotation entirely and "
                             "reports the raw rank-tracked axes (the non-negative "
                             "prior already pins the frame). Recorded on the trace "
                             "so the dashboard uses the same frame.")
    parser.add_argument("--skip-baseline", action="store_true",
                        help="[exploration] skip the K=1 baseline fit")
    parser.add_argument("--refit-baseline", action="store_true",
                        help="[exploration] re-fit the K=1 baseline even if cached")
    parser.add_argument("--plots", action="store_true",
                        help="[exploration] render figures via diagnostics/plot_mirt.py")
    args = parser.parse_args()

    if args.preset == "canonical":
        run_canonical(args)
    else:
        run_exploration(args, parser)


if __name__ == "__main__":
    main()
