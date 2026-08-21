"""Build ONE self-contained interactive dashboard of every MIRT / 1D / PPCA fit.

Writes the repo-root `index.html`: a fit selector (nav grouped by
baseline / exploratory / confirmed) + a cross-fit Comparison view, with every
figure a lazily-rendered Plotly plot (only the visible section is live in the
DOM). One command, correct ordering by construction.

Also writes `results/comparisons/{gof_table,loo_waic_table}.csv` and refreshes
`results/comparisons/README.md`. Pass `--png` to also dump static stills
(kaleido/Chrome) to the git-ignored `plots/dashboard/` for slides.

All rotation/identity handling lives in `analysis.prepare_fit`; all figures are
pure builders in `viz/`. The cards span THREE data scopes, set per entry by
`include_all` + `cyber`: curated benchmark exclusions applied at fit time, the
full benchmark set, and the full set plus the cyber-ECI benchmarks. Each fit is
scored on the data it was actually fit on, so GoF is comparable WITHIN a scope;
the global R²/RMSE remain readable across scopes but per-benchmark rows are not
apples-to-apples between them.

Run:  ~/miniforge3/envs/pymc_env/bin/python diagnostics/build_dashboard.py
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis import (  # noqa: E402
    _aligned_reproducibility, crosschain_axis_reproducibility,
    permutation_matched_reproducibility,
    mirt_identified_ess, mirt_identified_rhat, mirt_identified_rhat_nc,
    prepare_fit, trace_loading_prior,
)
from diagnostics.diagnose_chains import modes_path  # noqa: E402
from data import (  # noqa: E402
    clip_scores_to_floors, load_benchmark_ceilings, load_benchmark_floors,
    load_eci_data,
)
from persistence import save_df  # noqa: E402
from viz import (  # noqa: E402
    alignment_methods_fig, assemble_dashboard, build_axis_figures,
    build_comparison, build_fit_figures, ppca_spectrum_fig,
    raw_scores_by_date_fig, signed_display_frames, write_dashboard,
)
from ppc import (  # noqa: E402
    compute_gof, posterior_predictive_mirt,
    posterior_predictive_mirt_nc,
)

INDEX_PATH = ROOT / "index.html"
CMP_DIR = ROOT / "results" / "comparisons"

# Each fit: `type` (baseline/exploratory/confirmed), `kind` ("comp" sum link
# default; "nc" conjunctive product; "1d" original Beta-IRT), `dir` (results
# subfolder, default "mirt"), and `include_all` (data scope: True = full 85
# benchmarks, False = curated exclusions applied at fit time → 77). Each fit is
# rendered against the data scope it was fit on.
# Each fit carries a long `label` (nav + section header, room to be descriptive)
# AND a `short` (used as the axis tick in every cross-fit comparison chart, where
# the long label made the graphs unreadable). "SG" = the Skilled Generalist human
# tier; "both" = human+lineage priors.
FITS = [
    # ── K=4 prior ladder, exploration −2 benchmarks (2026-08-13) ──
    # One rung per theta-prior setting on the flagship config: K=4, `normal`
    # (non-negative) loadings, exploration scope minus FrontierMath v1 +
    # AlgoTune, fixed-c 3PL floors, 12×6000. What the priors and the pooled
    # noise each buy is the point; every rung scores the identical obs set,
    # so ELPD is comparable across rungs. No no-priors rung: decided against
    # (2026-08-13), both rungs carry the priors.
    {"name": "k4_drop2_bothpriors",
     "trace": "trace_mirt_k4_humanprior_lineageprior_lineagebm_dropFrontierMathv1"
              "AlgoTune_floors.nc",
     "label": "K=4 · exploration −2 benchmarks · positive loadings · "
              "human+lineage(BM) priors · 3PL floors · 12×6000 · ladder rung 1 "
              "— rung 1: tree human order, no pooled noise",
     "short": "K=4 −2bench · both priors",
     "type": "exploratory",
     "dir": "mirt_humanprior_lineageprior_lineagebm_dropFrontierMathv1"
            "AlgoTune_floors",
     "include_all": True, "floors": True,
     "drop_benchmarks": ["FrontierMath v1", "AlgoTune"],
     "group": "K=4 prior ladder · exploration −2 benchmarks",
     "nav": "1 · both priors"},
    {"name": "k4_drop2_humanmerge_flagship",
     "trace": "trace_mirt_k4_humanmerge_lineageprior_lineagebm_dropFrontierMathv1"
              "AlgoTune_floors_poolednoise.nc",
     "label": "K=4 · exploration −2 benchmarks · positive loadings · "
              "raw rank-tracked axes (no rotation) · "
              "human-merge+lineage(BM) priors · 3PL floors · pooled noise · "
              "10×20000, tune 7000 · single solution (10/10 chains one basin, "
              "matched corr 0.843), 78/200,000 divergences, D r-hat 1.168 max; "
              "the merged HS→adult edges bind (P(DE>HSQ) 0.64-0.80 under the "
              "tree order → 1.00) at identical GoF (R² 0.9645 vs 0.9647) — "
              "THE forecasting base",
     "short": "K=4 −2bench · pooled · merge · flagship",
     "type": "exploratory",
     "dir": "mirt_humanmerge_lineageprior_lineagebm_dropFrontierMathv1"
            "AlgoTune_floors_poolednoise",
     "include_all": True, "floors": True,
     "drop_benchmarks": ["FrontierMath v1", "AlgoTune"],
     "group": "K=4 prior ladder · exploration −2 benchmarks",
     "nav": "2 · + pooled noise + HS merge (flagship)", "forecast": True},
]

_RETIRED_FITS = [
    # ── retired 2026-08-13: dashboard pruned to the flagship + its K=4
    # prior ladder (no-priors rung pending). Everything below is the pre-prune
    # active set: the cyber prior ladder, the cyber 12×6000 mode pairs, the
    # K=3 SimpleQA-original/MMLU-Pro and −6-benchmark mode reads, the K=3 −3
    # honest-noise card, and the K=4 knownse history cards. Traces and cache
    # pkls left on disk.
    # ── prior ladder, cyber scope, positive loadings (2026-07-30) ──
    # One rung per theta-prior setting on an otherwise fixed config: K=3,
    # `normal` (non-negative) loadings in the RAW frame, --cyber data scope,
    # fixed-c floors + fixed-d ceilings. What each prior buys in identification
    # is the whole point. Rungs 1-3 are read MODE-RESTRICTED to their best basin
    # (chains picked by per-chain logp + loading match, kept subset verified by
    # its own eta r̂), so they compare R²/RMSE/LOO of one solution rather than of
    # a chain average across incompatible ones. Every rung scores the identical
    # 4,654-obs set, so ELPD is comparable across them — but a restricted subset
    # can inflate pareto-k, so read khat alongside ELPD. The 12×6000 pair below
    # is the top rung.
    # Rung 0 is the only one shown WHOLE: with ~8 basins and no converged pair,
    # any single-chain pick is arbitrary, so its R²/RMSE/LOO average over basins.
    {"name": "ladder0_noprior",
     "trace": "trace_mirt_k3_cyber_floors_ceilings.nc",
     "label": "K=3 · cyber · positive loadings · 4PL floors+ceilings · 8×2000 · "
              "no priors · shown whole: ~8 distinct solutions, no converged "
              "chain pair (eta r̂ 1.60) — the ladder's baseline; fit metrics "
              "average over basins",
     "short": "K=3 cy ladder0 · none",
     "type": "exploratory", "dir": "mirt_cyber_floors_ceilings",
     "include_all": True, "cyber": True, "floors": True, "ceilings": True,
     "group": "Prior ladder · cyber · positive", "nav": "0 · no priors"},
    {"name": "ladder1_lineagebm",
     "trace": "trace_mirt_k3_lineageprior_lineagebm_cyber_floors_ceilings.nc",
     "label": "K=3 · cyber · positive loadings · 4PL floors+ceilings · 7×2000 · "
              "lineage+BM only · mode-restricted: majority pair {2,5}, "
              "eta r̂ 1.010",
     "short": "K=3 cy ladder1 · lin+BM · A",
     "type": "exploratory", "dir": "mirt_lineageprior_lineagebm_cyber_floors_ceilings",
     "include_all": True, "cyber": True, "floors": True, "ceilings": True,
     "group": "Prior ladder · cyber · positive", "nav": "1 · lineage+BM · A",
     # Chain 1 has drifted off the pair's solution: adding it takes eta r̂ 1.010 → 1.554.
     "drop_chains": [0, 1, 3, 4, 6]},
    {"name": "ladder1_lineagebm_modeB",
     "trace": "trace_mirt_k3_lineageprior_lineagebm_cyber_floors_ceilings.nc",
     "label": "K=3 · cyber · positive loadings · 4PL floors+ceilings · 7×2000 · "
              "lineage+BM only · mode B (island, −22 nats; chains {3,4}, "
              "eta r̂ 1.008)",
     "short": "K=3 cy ladder1 · lin+BM · B",
     "type": "exploratory", "dir": "mirt_lineageprior_lineagebm_cyber_floors_ceilings",
     "include_all": True, "cyber": True, "floors": True, "ceilings": True,
     "group": "Prior ladder · cyber · positive", "nav": "1 · lineage+BM · B",
     "drop_chains": [0, 1, 2, 5, 6]},
    {"name": "ladder2_human",
     "trace": "trace_mirt_k3_humanprior_cyber_floors_ceilings.nc",
     "label": "K=3 · cyber · positive loadings · 4PL floors+ceilings · 7×2000 · "
              "human prior only · mode-restricted: mode A 5/7 (eta r̂ 1.012), "
              "B island (−47 nats) excluded",
     "short": "K=3 cy ladder2 · human · A",
     "type": "exploratory", "dir": "mirt_humanprior_cyber_floors_ceilings",
     "include_all": True, "cyber": True, "floors": True, "ceilings": True,
     "group": "Prior ladder · cyber · positive", "nav": "2 · human · A",
     "drop_chains": [1, 5]},
    {"name": "ladder2_human_modeB",
     "trace": "trace_mirt_k3_humanprior_cyber_floors_ceilings.nc",
     "label": "K=3 · cyber · positive loadings · 4PL floors+ceilings · 7×2000 · "
              "human prior only · mode B (island, −47 nats; chains {1,5}, "
              "eta r̂ 1.015)",
     "short": "K=3 cy ladder2 · human · B",
     "type": "exploratory", "dir": "mirt_humanprior_cyber_floors_ceilings",
     "include_all": True, "cyber": True, "floors": True, "ceilings": True,
     "group": "Prior ladder · cyber · positive", "nav": "2 · human · B",
     "drop_chains": [0, 2, 3, 4, 6]},
    {"name": "ladder3_human_lineage",
     "trace": "trace_mirt_k3_humanprior_lineageprior_cyber_floors_ceilings_ceilnoise.nc",
     "label": "K=3 · cyber · positive loadings · 4PL floors+ceilings+ceiling "
              "noise · 7×2000 · human+lineage priors · mode-restricted: mode A "
              "3/7 (eta r̂ 1.027), B (−30 nats) excluded",
     "short": "K=3 cy ladder3 · both · A",
     "type": "exploratory",
     "dir": "mirt_humanprior_lineageprior_cyber_floors_ceilings_ceilnoise",
     "include_all": True, "cyber": True, "floors": True, "ceilings": True,
     "group": "Prior ladder · cyber · positive", "nav": "3 · human+lineage · A",
     "drop_chains": [0, 1, 2, 3]},
    {"name": "ladder3_human_lineage_modeB",
     "trace": "trace_mirt_k3_humanprior_lineageprior_cyber_floors_ceilings_ceilnoise.nc",
     "label": "K=3 · cyber · positive loadings · 4PL floors+ceilings+ceiling "
              "noise · 7×2000 · human+lineage priors · mode B (island, −30 nats; "
              "chains {0,1,2,3}, eta r̂ 1.033)",
     "short": "K=3 cy ladder3 · both · B",
     "type": "exploratory",
     "dir": "mirt_humanprior_lineageprior_cyber_floors_ceilings_ceilnoise",
     "include_all": True, "cyber": True, "floors": True, "ceilings": True,
     "group": "Prior ladder · cyber · positive", "nav": "3 · human+lineage · B",
     "drop_chains": [4, 5, 6]},
    # ── cyber-scope 12×6000 pair, one card per posterior mode (2026-07-30) ──
    # Both fits: K=3, --cyber data scope (4,654 obs / 762 models / 108
    # benchmarks), fixed-c floors + fixed-d ceilings + estimated noise gap,
    # human + lineage priors with Brownian (time-indexed) lineage steps. Only
    # the loading prior differs. Neither converges whole (eta r̂ 1.43 / 1.47);
    # the chain-basin split below is the converged read of each basin.
    {"name": "cy5_bignormal_modeA",
     "trace": "trace_mirt_k3_humanprior_lineageprior_lineagebm_cyber_floors_ceilings_ceilnoise.nc",
     "label": "K=3 · cyber · positive loadings · human+lineage+BM priors · 4PL "
              "(floors+ceilings+ceiling noise) · 12×6000 · mode A "
              "(majority 7/12; chains 5,6,9–11 excluded)",
     "short": "K=3 cy · positive · mode A",
     "type": "exploratory", "include_all": True, "cyber": True,
     "floors": True, "ceilings": True,
     "dir": "mirt_humanprior_lineageprior_lineagebm_cyber_floors_ceilings_ceilnoise",
     "group": "K=3 cyber 12×6000 · positive · BM", "nav": "mode A · majority 7/12",
     # Chain 5 sits in mode A's basin at the same logp but with the human tiers
     # slid ~1 logit; dropped so the human rows read one location.
     "drop_chains": [5, 6, 9, 10, 11]},
    {"name": "cy5_bignormal_modeB",
     "trace": "trace_mirt_k3_humanprior_lineageprior_lineagebm_cyber_floors_ceilings_ceilnoise.nc",
     "label": "K=3 · cyber · positive loadings · human+lineage+BM priors · 4PL "
              "(floors+ceilings+ceiling noise) · 12×6000 · mode B "
              "(island, −16 nats; chains 6,9–11 only)",
     "short": "K=3 cy · positive · mode B",
     "type": "exploratory", "include_all": True, "cyber": True,
     "floors": True, "ceilings": True,
     "dir": "mirt_humanprior_lineageprior_lineagebm_cyber_floors_ceilings_ceilnoise",
     "group": "K=3 cyber 12×6000 · positive · BM", "nav": "mode B · island −16 nats",
     "drop_chains": [0, 1, 2, 3, 4, 5, 7, 8]},
    {"name": "cy6_bigsigned_alpha",
     "trace": "trace_mirt_k3_signed_humanprior_lineageprior_lineagebm_cyber_floors_ceilings_ceilnoise.nc",
     "label": "K=3 · cyber · signed loadings · human+lineage+BM priors · 4PL "
              "(floors+ceilings+ceiling noise) · 12×6000 · mode α "
              "(best basin, holds the top-logp chain; chains 1,2,9 only)",
     "short": "K=3 cy · signed · mode α",
     "type": "exploratory", "include_all": True, "cyber": True,
     "floors": True, "ceilings": True,
     "dir": "mirt_signed_humanprior_lineageprior_lineagebm_cyber_floors_ceilings_ceilnoise",
     "group": "K=3 cyber 12×6000 · signed · BM", "nav": "mode α · best 3/12",
     "drop_chains": [0, 3, 4, 5, 6, 7, 8, 10, 11]},
    {"name": "cy6_bigsigned_beta",
     "trace": "trace_mirt_k3_signed_humanprior_lineageprior_lineagebm_cyber_floors_ceilings_ceilnoise.nc",
     "label": "K=3 · cyber · signed loadings · human+lineage+BM priors · 4PL "
              "(floors+ceilings+ceiling noise) · 12×6000 · mode β "
              "(island, −15 nats; the other 9/12 chains)",
     "short": "K=3 cy · signed · mode β",
     "type": "exploratory", "include_all": True, "cyber": True,
     "floors": True, "ceilings": True,
     "dir": "mirt_signed_humanprior_lineageprior_lineagebm_cyber_floors_ceilings_ceilnoise",
     "group": "K=3 cyber 12×6000 · signed · BM", "nav": "mode β · 9/12 −15 nats",
     "drop_chains": [1, 2, 9]},
    # ── newest exploration fit, read MODE-AWARE (2026-08-04) ──
    # K=3, exploration scope + the original SimpleQA + the full MMLU-Pro (4,953
    # obs / 817 models / 103 benchmarks), positive loadings, human + lineage
    # priors, 4PL floors+ceilings, known-SE noise split, pooled sigma_b. The 12
    # chains split three ways (mirt_modes_<trace-stem>.json): A {0,4} wins by 87
    # nats; B {2,3,6,7,8,9,11} sits 87 nats down but carries A's loadings
    # (matched corr 0.862, above the 0.6 same-solution bar) — one solution found
    # at two logp levels, split by the cliff, not by its axes; C {1,5,10} at
    # 0.465 is a genuinely different axis solution. Shown WHOLE — every metric on
    # the card averages the basins, which is the honest reading of a multimodal
    # fit — with a per-mode loading and timeline figure set appended so each
    # solution's axes are readable on their own.
    {"name": "k3_sqaorig_mmlupro_knownse",
     "trace": "trace_mirt_k3_humanprior_lineageprior_sqaorig_mmlupro_floors_"
              "ceilings_knownse_poolednoise.nc",
     "label": "K=3 · +SimpleQA-original +MMLU-Pro · positive loadings · "
              "human+lineage priors · 4PL floors+ceilings · known-SE + pooled "
              "noise · 12×2000 · whole fit, 3 modes shown separately",
     "short": "K=3 sqa+mmlupro · knownse",
     "type": "exploratory",
     "dir": "mirt_humanprior_lineageprior_sqaorig_mmlupro_floors_ceilings_knownse_poolednoise",
     "include_all": True, "sqaorig": True,
     "floors": True, "ceilings": True,
     "group": "K=3 · SimpleQA-original + MMLU-Pro", "nav": "whole fit + 3 modes"},
    # ── newest exploration fit, read MODE-AWARE (2026-08-05) ──
    # K=3, exploration scope minus six benchmarks (FrontierMath v1, GBAEval,
    # BlueprintBench 2, AlgoTune, Video-MME, AudioMultiChallenge → 752 models /
    # 95 benchmarks), positive loadings, human + lineage priors, fixed-c floors.
    # The 12 chains split two ways (mirt_modes_<trace-stem>.json): A {2,4,6}
    # holds the best basin; B {0,1,3,5,7,8,9,10,11} sits 11 nats down at loading
    # corr 0.499 vs A — a different axis-3 solution, not a logp copy of A. Each
    # basin converges alone (eta r̂ 1.009 / 1.003); pooled eta r̂ 1.37 is the
    # split, not bad mixing. Shown WHOLE with a per-mode loading + timeline set
    # appended so each solution's axes read on their own. Axes 1-2 agree across
    # modes (matched corr 0.85 / 0.93); axis 3 is the whole disagreement — a
    # shared easy-knowledge core with the agentic-hard cluster (A) or the
    # factual-recall cluster (B) attached.
    {"name": "k3_dropfmv1_floors",
     "trace": "trace_mirt_k3_humanprior_lineageprior_dropFrontierMathv1GBAEval"
              "BlueprintBench2AlgoTuneVideoMMEAudioMultiChallenge_floors.nc",
     "label": "K=3 · exploration −6 benchmarks · positive loadings · "
              "human+lineage priors · 3PL floors · 12×… · whole fit, 2 modes "
              "shown separately (axis-3 valley: agentic-hard vs factual-recall)",
     "short": "K=3 −6bench · floors",
     "type": "exploratory",
     "dir": "mirt_humanprior_lineageprior_dropFrontierMathv1GBAEvalBlueprintBench2"
            "AlgoTuneVideoMMEAudioMultiChallenge_floors",
     "include_all": True, "floors": True,
     "drop_benchmarks": ["FrontierMath v1", "GBAEval", "BlueprintBench 2",
                         "AlgoTune", "Video-MME", "AudioMultiChallenge"],
     "group": "K=3 · exploration −6 benchmarks", "nav": "whole fit + 2 modes"},

    # ── honest-noise pair, −3 benchmarks (2026-08-07) ──
    # NOTE data-state trap: the k3 predates the 2026-08-07 snapshot refresh
    # known-SE instrument split + pooled excess noise. The K=3 shows the
    # axis-3 identity valley at its fairest measurement (two converged modes,
    # 6v6 chains, 7.3 nats apart); the K=4 dissolves it by giving both rival
    # directions their own axis (one solution, per-axis cross-chain
    # reproducibility 0.92-0.99). K=4's residual eta r-hat max 1.32 localizes
    # to a 5v7 near-degenerate configuration of the human-anchored benchmark
    # block; machine-side quantities differ by <=0.9 points median between the
    # chain groups, and group loading gaps sit within the drawn HDIs.
    {"name": "k3_drop3_pooled",
     "trace": "trace_mirt_k3_humanprior_lineageprior_lineagebm_dropFrontierMathv1"
              "GBAEvalAlgoTune_floors_knownse_poolednoise.nc",
     "label": "K=3 · exploration −3 benchmarks · positive loadings · "
              "human+lineage(BM) priors · 3PL floors · known-SE + pooled noise · "
              "12 chains · 2 converged modes 6v6, 7.3 nats apart (axis-3 "
              "identity: recall/vision vs agentic), shown separately",
     "short": "K=3 −3bench · pooled+knownSE",
     "type": "exploratory",
     "dir": "mirt_humanprior_lineageprior_lineagebm_dropFrontierMathv1GBAEval"
            "AlgoTune_floors_knownse_poolednoise",
     "include_all": True, "floors": True,
     "drop_benchmarks": ["FrontierMath v1", "GBAEval", "AlgoTune"],
     "group": "Honest noise · exploration −3 benchmarks",
     "nav": "K=3 · 2 modes"},
    {"name": "k4_drop2_pooled",
     "trace": "trace_mirt_k4_humanprior_lineageprior_lineagebm_dropFrontierMathv1"
              "AlgoTune_floors_knownse_poolednoise.nc",
     "label": "K=4 · exploration −2 benchmarks (GBAEval readmitted) · positive "
              "loadings · human+lineage(BM) priors · 3PL floors · known-SE + "
              "pooled noise · 12×4000, all chains kept · single solution "
              "(axis reproducibility 0.99/0.99/0.89/0.86); the 7v5 chain-group "
              "ridge is prediction-invariant (per-draw median gap 0.24 pt, "
              "max 12 pt on stale-model legacy cells) — THE forecasting base",
     "short": "K=4 −2bench · pooled+knownSE",
     "type": "exploratory",
     "dir": "mirt_humanprior_lineageprior_lineagebm_dropFrontierMathv1"
            "AlgoTune_floors_knownse_poolednoise",
     "include_all": True, "floors": True,
     "drop_benchmarks": ["FrontierMath v1", "AlgoTune"],
     "group": "Honest noise · exploration −2/−3 benchmarks",
     "nav": "K=4 −2bench · knownSE (history)"},
    {"name": "k4_drop3_pooled",
     "trace": "trace_mirt_k4_humanprior_lineageprior_lineagebm_dropFrontierMathv1"
              "GBAEvalAlgoTune_floors_knownse_poolednoise.nc",
     "label": "K=4 · exploration −3 benchmarks · positive loadings · "
              "human+lineage(BM) priors · 3PL floors · known-SE + pooled noise · "
              "12×4000 · single solution (both K=3 rivals hold their own axis; "
              "axis reproducibility 0.92-0.99); eta r-hat max 1.32 is a "
              "human-anchored-block micro-split, machine-side gap ≤0.9 pt median",
     "short": "K=4 −3bench · pooled+knownSE",
     "type": "exploratory",
     "dir": "mirt_humanprior_lineageprior_lineagebm_dropFrontierMathv1GBAEval"
            "AlgoTune_floors_knownse_poolednoise",
     "include_all": True, "floors": True,
     "drop_benchmarks": ["FrontierMath v1", "GBAEval", "AlgoTune"],
     "group": "Honest noise · exploration −3 benchmarks",
     "nav": "K=4 · single solution"},
    # ── retired 2026-07-30: every pre-cyber-generation card. All fit on
    # superseded data snapshots (702-745 models / 77-85 benchmarks); the
    # 2026-07-30 cyber-generation ladder + 12x6000 mode cards replace them.
    # Traces and cache pkls left on disk.
    # ── K=1 (unidimensional MIRT, curated exclusions applied at fit time) ──
    # Both on the EXCLUDED benchmark set (702 models / 77 benchmarks): directly
    # comparable to each other, NOT to the K=3 no-SG fit (full 85). The priors
    # barely move predictive fit (they constrain theta, not the likelihood):
    # R² 0.937 (no priors) vs 0.935 (human+lineage).
    {"name": "k1_excluded", "trace": "trace_mirt_k1_excluded.nc",
     "label": "K=1 · unidimensional · curated exclusions · no priors", "short": "K=1",
     "type": "baseline", "dir": "mirt_excluded", "include_all": False},
    {"name": "k1_both_excluded", "trace": "trace_mirt_k1_humanprior_lineageprior_excluded.nc",
     "label": "K=1 · unidimensional · curated exclusions · human+lineage priors",
     "short": "K=1 both",
     "type": "exploratory", "dir": "mirt_humanprior_lineageprior_excluded",
     "include_all": False},
    # ── K=3 no-SG (signed, human+lineage priors, full 85-benchmark set) ──
    # Mode-restricted read for accurate axes: 8 chains, chain 4 is the outlier
    # island (leave-one-out identified eta r̂: all-8 = 1.254; drop chain 4 →
    # 1.002; every OTHER single drop ~1.285). Dropping chain 4 reports the
    # 7/8-chain majority mode. SG observations dropped at fit time (scored on the
    # same data). Carries the frontier-forecast figure set.
    {"name": "k3_noSG", "trace": "trace_mirt_k3_signed_humanprior_lineageprior_noSG.nc",
     "label": "K=3 · human+lineage priors · SG scores removed · chain 4 excluded "
              "(majority mode) · frontier forecast",
     "short": "K=3 both · −SG",
     "type": "exploratory", "dir": "mirt_signed_humanprior_lineageprior_noSG",
     "include_all": True, "drop_model_obs": ["Skilled Generalist"],
     "drop_chains": [4], "forecast": True},
    # ── with-SG pair: the chance-floor experiment (2026-07-13) ──
    # Same config/data (full 85-benchmark set, SG kept), only difference = floors.
    # Baseline (8ch×1000): ISLANDS, eta r̂ 1.18, 644 div. Floors (fixed-c 3PL +
    # clip-to-floor, 8ch×4000): a permutation-invariant decomposition shows 7 of
    # 8 chains are ONE solution (eta r̂ 1.001 — the raw-θ "orientations" were just
    # axis-label permutations); chain 1 is the lone genuine minority island (adding
    # it → 1.07, and it is the worst-logp chain). So the floored card is read
    # MODE-RESTRICTED (drop chain 1) = converged with SG kept, eta r̂ 1.001, the
    # drop-SG result without dropping SG. Baseline shown whole (no drop) so the
    # islands it removes are visible.
    {"name": "k3_withSG", "trace": "trace_withSG_refit.nc",
     "label": "K=3 · human+lineage priors · SG kept · no floors (islands baseline)",
     "short": "K=3 both · +SG",
     "type": "exploratory", "dir": "recovery_study", "include_all": True},
    {"name": "k3_withSG_floors",
     "trace": "trace_mirt_k3_signed_humanprior_lineageprior_floors_8x4000.nc",
     "label": "K=3 · human+lineage priors · SG kept · fixed-c 3PL chance floors "
              "+ clip · 8×4000 · chain 1 excluded (majority solution)",
     "short": "K=3 both · +SG · floors",
     "type": "exploratory", "dir": "mirt_signed_humanprior_lineageprior_floors",
     "include_all": True, "floors": True, "drop_chains": [1]},
    # ── positive-loadings companion to the floored signed fit (2026-07-21) ──
    # Same data / scope / floors as k3_withSG_floors, but NON-NEGATIVE loadings
    # (the `normal` prior) rendered in the RAW rank-tracked frame
    # (mirt_display_rotation='none' on the trace): no post-hoc rotation — the
    # positivity constraint pins the frame (all 8 chains carry the same three
    # bundles up to a column permutation, matched corr ≥ 0.9998; the energy
    # rank-track relabel is the only correction applied). Pure positive
    # bundles (agentic/AGI · hard-math/science · easy-knowledge), no contrast
    # axis. Converges WITHOUT mode-restriction (no drop_chains): the HalfNormal
    # loadings remove the sign-flip islands the signed fit needs a chain drop
    # for (8×10000: identified eta r̂ ≈ 1.00, ~1 div). R² ties the signed fit.
    # Named by its full filename, not the bare config name: the unsuffixed
    # trace_mirt_k3_humanprior_lineageprior_floors.nc is whatever the last run of
    # that config wrote, so the card's convergence claim above only holds if the
    # 8×10000 / 85-benchmark file is what loads.
    {"name": "k3_withSG_floors_nonneg",
     "trace": "trace_mirt_k3_humanprior_lineageprior_floors_8x10000_85bm.nc",
     "label": "K=3 · human+lineage priors · SG kept · fixed-c 3PL floors · "
              "positive loadings · raw rank-tracked axes (no rotation)",
     "short": "K=3 both · +SG · floors · positive",
     "type": "exploratory", "dir": "mirt_humanprior_lineageprior_floors",
     "include_all": True, "floors": True, "forecast": True},
    # ── hard-lineage sibling of the fit above (2026-07-23) ──
    # Identical config (positive loadings, raw frame, fixed-c 3PL floors,
    # human+lineage priors, full 85-benchmark set) EXCEPT the lineage chain
    # steps are HARD (HalfNormal, successor ≥ predecessor on every axis) rather
    # than soft (Normal, can regress); effort-variant offsets stay mean-zero and
    # unordered either way. The direct read on what hardening the release order
    # costs: 8×4000 vs the soft twin's numbers — R² 0.956 (soft 0.971) and Φ
    # rises on every pair (axis1↔3 0.35→0.50), stabler chain direction bought
    # with less axis distinctness. Read MODE-RESTRICTED (drop chain 2): all-8
    # eta r̂ 1.089, but leave-one-chain-out isolates chain 2 as the lone
    # minority island (drop it → 1.003; drop any other → 1.099), so the 7/8
    # majority is the converged read. Its two modes are the floor-clipped LOO
    # ladder group (results/comparisons/loo_cv_full.csv): island "wins" ELPD
    # but is the standard LOO trap (1 chain, ess 2, p_loo 1459).
    # Two cards, one per LOO-ladder mode. MAJORITY = drop chain 2 (7/8, the
    # converged read, eta r̂ 1.003). ISLAND = chain 2 alone (the minority basin,
    # eta r̂ undefined on one chain). Same split as the floor-clipped LOO group.
    {"name": "k3_withSG_floors_nonneg_hardlin",
     "trace": "trace_mirt_k3_humanprior_lineageprior_lineagehard_floors.nc",
     "label": "K=3 · human+lineage priors (HARD lineage) · SG kept · fixed-c 3PL "
              "floors · positive loadings · raw rank-tracked axes (no rotation) · "
              "mode 1 (majority, chain 2 excluded)",
     "short": "K=3 both · +SG · floors · positive · hard-lin · majority",
     "type": "exploratory", "dir": "mirt_humanprior_lineageprior_lineagehard_floors",
     "include_all": True, "floors": True, "drop_chains": [2]},
    {"name": "k3_withSG_floors_nonneg_hardlin_island",
     "trace": "trace_mirt_k3_humanprior_lineageprior_lineagehard_floors.nc",
     "label": "K=3 · human+lineage priors (HARD lineage) · SG kept · fixed-c 3PL "
              "floors · positive loadings · raw rank-tracked axes (no rotation) · "
              "mode 2 (island, chain 2 only)",
     "short": "K=3 both · +SG · floors · positive · hard-lin · island",
     "type": "exploratory", "dir": "mirt_humanprior_lineageprior_lineagehard_floors",
     "include_all": True, "floors": True, "drop_chains": [0, 1, 3, 4, 5, 6, 7]},
    # Positive-loadings prior-sensitivity ladder on the 2026-07-22 data
    # (85 benchmarks, no floors), retired 2026-07-30: superseded by the
    # cyber-scope ladder, which asks the same prior question on the current
    # data with floors+ceilings and a mode-restricted read per rung. Traces
    # kept on disk.
    {"name": "k3_pos_none",
     "trace": "trace_mirt_k3.nc",
     "label": "K=3 · positive loadings · no priors · non-negative rotation",
     "short": "K=3 pos · none",
     "type": "baseline", "dir": "mirt", "include_all": True},
    {"name": "k3_pos_human",
     "trace": "trace_mirt_k3_humanprior.nc",
     "label": "K=3 · positive loadings · human prior · non-negative rotation",
     "short": "K=3 pos · human",
     "type": "exploratory", "dir": "mirt_humanprior", "include_all": True},
    # 'both' is multimodal whole (eta r̂ 1.079); leave-one-out isolates chain 5
    # as the lone island (drop it → 1.005). Split one card per LOO-ladder mode:
    # MAJORITY = drop chain 5 (7/8, eta r̂ 1.005); ISLAND = chain 5 alone.
    {"name": "k3_pos_both_majority",
     "trace": "trace_mirt_k3_humanprior_lineageprior.nc",
     "label": "K=3 · positive loadings · human+lineage priors · non-negative "
              "rotation · mode 1 (majority, chain 5 excluded)",
     "short": "K=3 pos · both · majority",
     "type": "exploratory", "dir": "mirt_humanprior_lineageprior", "include_all": True,
     "drop_chains": [5]},
    {"name": "k3_pos_both_island",
     "trace": "trace_mirt_k3_humanprior_lineageprior.nc",
     "label": "K=3 · positive loadings · human+lineage priors · non-negative "
              "rotation · mode 2 (island, chain 5 only)",
     "short": "K=3 pos · both · island",
     "type": "exploratory", "dir": "mirt_humanprior_lineageprior", "include_all": True,
     "drop_chains": [0, 1, 2, 3, 4, 6, 7]},
    # SURGICAL no-SG cut (2026-07-07): exploratory K=3, both theta priors, free
    # rotation (no PLT). Drops ONLY the Skilled Generalist's contradictory GPQA
    # cells at fit time (the taker + its other scores stay), vs the old fit that
    # dropped ALL of SG's scores. 8 chains x 6000 draws on current data (3,764
    # obs / 734 models / 76 benchmarks) for max within-mode precision. R²=0.971.
    # Does NOT fully converge: keeping SG's anomalous ARC-AGI 0.98 (only its GPQA
    # low anchor removed) leaves the straddler island — eta r̂ 1.066 / D r̂ 1.109,
    # 5409 div/48000. Cleaner than a free fit but short of the full-SG cut's one
    # island (r̂ 1.007). See fit_noSGgpqa.py.
    {"name": "k3_noSG_converged", "trace": "trace_mirt_k3_signed_humanprior_lineageprior_noSGgpqa.nc",
     "label": "K=3 · human+lineage priors · free rotation · SG GPQA cells removed", "short": "K=3 both · −SG GPQA",
     "type": "exploratory", "dir": "mirt_signed_humanprior_lineageprior_noSGgpqa",
     "drop_gpqa_cells": "Skilled Generalist"},
    # Full-SG cut (8 chains x 3000 draws, current data): the one-island fit
    # (7/8 chains agree; chain 1 is the outlier mode, excluded here for a
    # mode-restricted read). Carries the frontier-forecast figure set —
    # projected dates the AI frontier crosses each human tier, per axis.
    {"name": "k3_noSG_forecast", "trace": "trace_mirt_k3_signed_humanprior_lineageprior_noSG.nc",
     "label": "K=3 · human+lineage priors · SG scores removed · chain 1 excluded · frontier forecast",
     "short": "K=3 both · −SG · forecast",
     "type": "exploratory", "dir": "mirt_signed_humanprior_lineageprior_noSG",
     "drop_model_obs": ["Skilled Generalist"], "drop_chains": [1], "forecast": True},
    # The controlled-experiment companions of the converged fit:
    {"name": "k3_noSG_nopriors", "trace": "trace_mirt_k3_signed_noSG_nopriors.nc",
     "label": "K=3 · no priors · free rotation · SG scores removed", "short": "K=3 none · −SG",
     "type": "baseline", "dir": "mirt_noSG", "drop_model_obs": ["Skilled Generalist"]},
    {"name": "k3_SGindep", "trace": "trace_mirt_k3_signed_SGindep.nc",
     "label": "K=3 · human+lineage priors · free rotation · SG detached from tier chain", "short": "K=3 both · SG detached",
     "type": "exploratory", "dir": "mirt_noSG"},
    {"name": "k3_noARCAGI", "trace": "trace_mirt_k3_signed_noARCAGI.nc",
     "label": "K=3 · human+lineage priors · free rotation · ARC-AGI benchmark removed", "short": "K=3 both · −ARC-AGI",
     "type": "exploratory", "dir": "mirt_noSG", "drop_bench_obs": ["ARC-AGI"]},
    # Surgical single-cell cut (2026-07-08): drop ONLY the Skilled Generalist's
    # ARC-AGI=0.98 cell (the worst contradictory straddler cell; SG's other
    # scores + the tier stay). 4 chains x 1000/1000 on current data (3,768 obs).
    # R²=0.971. η r̂ 1.43 / D r̂ 1.58, 2 div — best-converged of the SG-cut
    # family (vs full-SG straddler η r̂ ~2.2), but still short of one island.
    {"name": "k3_noSGarcagi", "trace": "trace_mirt_k3_signed_humanprior_lineageprior_noSGarcagi.nc",
     "label": "K=3 · human+lineage priors · free rotation · SG ARC-AGI cell removed", "short": "K=3 both · −SG ARC-AGI",
     "type": "exploratory", "dir": "mirt_signed_humanprior_lineageprior_noSGarcagi",
     "drop_arcagi_cells": "Skilled Generalist"},
    # ── Hypothesis matrix 2026-07-05: what makes K=3 not converge? ──
    # All: signed loadings, cleaned data (3,714 obs / 737 models /
    # 75 benchmarks), ORIGINAL hard human prior, 4 chains x 1000/1000 (quick
    # runs — r-hats wobble ~±0.2 vs 8-chain), founders GPQA/GSO/ARC-AGI-2
    # sliced to K. "baseline" = no theta priors; "exploratory" = with priors.
    {"name": "k3_none",     "trace": "trace_mirt_k3_signed.nc",                              "label": "K=3 · no priors · free rotation",          "short": "K=3 none",         "type": "baseline"},
    {"name": "k3_none_plt", "trace": "trace_mirt_k3_signed_plt.nc",                          "label": "K=3 · no priors · founder anchors",        "short": "K=3 none · PLT",   "type": "baseline",    "dir": "mirt_signed_plt"},
    {"name": "k2_none_plt", "trace": "trace_mirt_k2_signed_plt.nc",                          "label": "K=2 · no priors · founder anchors",        "short": "K=2 none · PLT",   "type": "baseline",    "dir": "mirt_signed_plt"},
    {"name": "k3_lin",      "trace": "trace_mirt_k3_signed_lineageprior.nc",                 "label": "K=3 · lineage prior only · free rotation",  "short": "K=3 lineage",      "type": "exploratory", "dir": "mirt_signed_lineageprior"},
    {"name": "k3_lin_plt",  "trace": "trace_mirt_k3_signed_lineageprior_plt.nc",             "label": "K=3 · lineage prior only · founder anchors","short": "K=3 lineage · PLT","type": "exploratory", "dir": "mirt_signed_lineageprior_plt"},
    {"name": "k3_both",     "trace": "trace_mirt_k3_signed_humanprior_lineageprior.nc",      "label": "K=3 · human+lineage priors · free rotation","short": "K=3 both",         "type": "exploratory", "dir": "mirt_signed_humanprior_lineageprior"},
    {"name": "k3_both_plt", "trace": "trace_mirt_k3_signed_humanprior_lineageprior_plt.nc",  "label": "K=3 · human+lineage priors · founder anchors","short": "K=3 both · PLT", "type": "exploratory", "dir": "mirt_signed_humanprior_lineageprior_plt"},
    {"name": "k2_both",     "trace": "trace_mirt_k2_signed_humanprior_lineageprior.nc",      "label": "K=2 · human+lineage priors · free rotation","short": "K=2 both",         "type": "exploratory", "dir": "mirt_signed_humanprior_lineageprior"},
    {"name": "k2_both_plt", "trace": "trace_mirt_k2_signed_humanprior_lineageprior_plt.nc",  "label": "K=2 · human+lineage priors · founder anchors","short": "K=2 both · PLT", "type": "exploratory", "dir": "mirt_signed_humanprior_lineageprior_plt"},
]

def _trace_path(fit):
    """Trace location for a fit (NON-comp fits live under results/mirt_nc/ etc.)."""
    return ROOT / "results" / fit.get("dir", "mirt") / fit["trace"]


# ── posterior modes ─────────────────────────────────────────────────────────
# A K=3 fit's chains can sit in several likelihood basins, and a figure over ALL
# chains then averages incompatible solutions. The split is detected ONCE, off
# the trace, by `diagnostics/diagnose_chains.py --write-modes`, which writes
# `results/<dir>/mirt_modes_<trace-stem>.json`; the dashboard only READS that
# file, so a build never loads a multi-GB trace to re-detect. Mode views are
# ADDITIONS: the whole-fit figures stay, and no diagnostic number is ever
# mode-restricted (convergence, PPC and PIT describe the whole fit — CLAUDE.md).

def _modes_path(fit):
    """Where this fit's mode split lives — same trace-stem rule as
    `diagnose_chains.modes_path`, so a folder's K=3 and K=1 splits never collide."""
    return modes_path(_trace_path(fit))


def _fit_modes(fit):
    """The persisted mode split for a fit, or None when there is nothing to show.

    None unless the JSON names THIS fit's trace and found more than one mode.
    Also None for a card that already sets `drop_chains`: that card IS one mode,
    so nesting mode views inside it would double-restrict."""
    import json
    p = _modes_path(fit)
    if not p.exists() or fit.get("drop_chains"):
        return None
    doc = json.loads(p.read_text())
    if doc.get("trace") != fit["trace"] or len(doc.get("modes", [])) < 2:
        return None
    return doc


def _mode_tag(m):
    """One-line mode identity for a figure title: which chains, and how far below
    the best basin it sits. ASCII hyphen for the sign, matching the table."""
    chains = ",".join(str(c) for c in m["chains"])
    d = m.get("delta_logp")
    where = ("logp unavailable" if d is None
             else "best logp" if d == 0 else f"{d:.0f} nats")
    return f"mode {m['label']} (chains {chains}; {where})"


def _mode_figures(idata, modes, data, raw, bench):
    """Mode-restricted loading + timeline figures: the same builders as the
    whole-fit card, run on the trace sliced to each basin's chains. Each mode
    gets its own rotation/alignment pass, because a basin's axes are its own."""
    figs = {}
    for m in modes:
        sub = idata.sel(chain=m["chains"])
        v = prepare_fit(sub, data)
        figs.update(build_axis_figures(
            v, data, raw, bench, signed_display_frames(v, sub),
            prefix=f"mode{m['label']}_", suffix=f" · {_mode_tag(m)}"))
    return figs


def _modes_table_html(doc):
    """The card's mode summary: chains, Δlogp, worst matched loading correlation
    inside the mode, and the same against the BEST mode. That last column is what
    separates "same loadings, worse basin" (high corr, big Δlogp — split by the
    logp cliff alone) from "a different axis solution" (low corr). Nothing else
    belongs here — a mode is not a converged fit, and every diagnostic on the
    card is a whole-fit number."""
    df = pd.DataFrame([{"mode": m["label"],
                        "chains": ",".join(str(c) for c in m["chains"]),
                        "n_chains": len(m["chains"]),
                        "Δlogp": m["delta_logp"],
                        "min matched loading corr, within mode":
                            m["min_matched_corr"],
                        "min matched loading corr, vs best mode":
                            ("—" if m.get("matched_corr_to_best") is None
                             else m["matched_corr_to_best"])}
                       for m in doc["modes"]])
    return (f"<h3>posterior modes — {len(df)} solutions across "
            f"{doc['n_chains']} chains; loading and timeline figures repeat per "
            "mode below, every metric above is whole-fit</h3>"
            + df.to_html(classes="cmp", index=False, border=0)
            + f'<div class="statline">basis: {doc["basis"]}</div>')


# ── incremental render cache ─────────────────────────────────────────────────
# Rendering a card costs a multi-GB trace load + PPC; the DATA also moves
# between fitting rounds (e.g. new human tiers), after which old traces can't
# be re-rendered at all. So each rendered card (figures + comparison-row
# metrics) is pickled, keyed by the trace's (size, mtime) + RENDER_REV. On the
# next build: unchanged trace → cached card, no trace load; trace DELETED →
# cached card still served, labelled "trace retired" (delete the .nc freely,
# the plots survive); trace present but fit on a different data generation →
# cached card with a "data superseded" label instead of a crash. Bump
# RENDER_REV (or use --force) when the figure set itself changes.
RENDER_REV = 14  # posterior summaries are median + central interval
CACHE_DIR = ROOT / "results" / "dashboard_cache"


def _fingerprint(trace_path, modes_file=None):
    """Cache key: trace identity + figure-set revision + the MODE SPLIT.

    The mode split has to be in here. A card's mode figures are built at render
    time but its mode table is read from the JSON at assembly, so without this a
    cached card would keep serving a table that promises figures it never
    contains — or, after a re-detection at a different threshold, a table whose
    chain groupings contradict its own figure titles."""
    st = trace_path.stat()
    fp = {"size": st.st_size, "mtime": int(st.st_mtime), "rev": RENDER_REV}
    if modes_file is not None and modes_file.exists():
        import hashlib
        fp["modes"] = hashlib.sha256(modes_file.read_bytes()).hexdigest()[:16]
    return fp


def _drop_model_obs(data, names):
    """Fit-time copy of `data` with the named takers' OBSERVATIONS removed
    (takers stay; their theta becomes prior-only). Mirrors the converged
    no-SG fit, whose card must be scored on the data it was fit to. Thin wrapper
    over the shared helper so the dashboard and fit.py --no-sg never drift."""
    from data import drop_model_observations
    return drop_model_observations(data, names)


def _drop_gpqa_cells(data, model_name):
    """Fit-time copy of `data` with `model_name`'s GPQA cells removed (the
    surgical no-SG-GPQA cut: only the contradictory GPQA scores go, the taker
    and its other scores stay). Mirrors fit_noSGgpqa's data prep so the card is
    scored on exactly the data it was fit to."""
    from data import drop_model_benchmark_cells
    gpqa = [b for b in data.blookup["benchmark"] if "GPQA" in b]
    return drop_model_benchmark_cells(data, model_name, gpqa)


def _drop_arcagi_cells(data, model_name):
    """Fit-time copy of `data` with `model_name`'s ARC-AGI cell(s) removed (the
    surgical no-SG-ARC-AGI cut: only the contradictory ARC-AGI score goes, the
    taker and its other scores stay). Matches fit.py --no-sg-arcagi
    (ARC-AGI abstraction family only, NOT the easy "ARC (AI2)")."""
    from data import drop_model_benchmark_cells
    arcagi = [b for b in data.blookup["benchmark"] if b.startswith("ARC-AGI")]
    return drop_model_benchmark_cells(data, model_name, arcagi)


def _drop_bench_obs(data, names):
    """Fit-time copy of `data` with the named BENCHMARKS' observations removed
    (benchmarks stay indexed; their D becomes prior-only). Mirrors the
    ARC-AGI-removal fit."""
    import numpy as np
    from dataclasses import replace
    from config import LOW_OBS_THRESHOLD
    benches = data.blookup["benchmark"].values
    drop = {list(benches).index(n) for n in names if n in benches}
    keep = ~np.isin(data.bench_idx, list(drop))
    nobs = np.bincount(data.model_idx[keep], minlength=data.n_models)
    return replace(data, scores=data.scores[keep], model_idx=data.model_idx[keep],
                   bench_idx=data.bench_idx[keep],
                   zero_score_mask=data.zero_score_mask[keep],
                   n_eff=None if data.n_eff is None else data.n_eff[keep],
                   n_obs=int(keep.sum()), n_obs_per_model=nobs,
                   is_low_obs=nobs < LOW_OBS_THRESHOLD)


def _trace_n_models(trace_path):
    """Model count of a trace WITHOUT loading it (lazy xarray peek). MIRT
    traces have a named `model` dim; a legacy 1D trace (unnamed dims)
    carries it as C's only dim (`C_dim_0`)."""
    import xarray as xr
    with xr.open_dataset(trace_path, group="posterior") as post:
        n = post.sizes.get("model") or post.sizes.get("C_dim_0")
        return int(n) if n else -1


def _cache_load(name):
    p = CACHE_DIR / f"{name}.pkl"
    if not p.exists():
        return None
    import pickle
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception as e:  # noqa: BLE001 — corrupt/stale cache → just re-render
        print(f"  cache unreadable for {name} ({type(e).__name__}) — re-rendering")
        return None


def _cache_save(name, figures, result, fingerprint):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    import pickle
    with open(CACHE_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump({"fingerprint": fingerprint, "figures": figures,
                     "result": result}, f)


_MODE_EVAL_CACHE = CACHE_DIR / "mode_eval_ess.json"


def _identified_ess_cached(fit, tp, fp, data, allow_load=True):
    """Min/median bulk ESS on the identified eta subset, computed on the card's
    KEPT chains, plus the chain/draw bookkeeping for the mode-evaluation table.
    Cached by trace identity + chain subset — the trace load dominates the cost
    and the table re-assembles on every build.

    The key deliberately drops `rev` (and the mode split) from the fingerprint:
    ESS depends on the posterior, not on which figures the dashboard draws, so a
    RENDER_REV bump must not force every trace to be re-read.

    `allow_load=False` returns None instead of reading the trace — for a card on
    a superseded data generation, where indexing the trace against today's `data`
    would misalign and raise."""
    import json
    drop = sorted(fit.get("drop_chains", []))
    trace_key = {k: v for k, v in fp.items() if k not in ("rev", "modes")}
    cache = (json.loads(_MODE_EVAL_CACHE.read_text())
             if _MODE_EVAL_CACHE.exists() else {})
    hit = cache.get(fit["name"])
    if (hit and hit["drop_chains"] == drop
            and {k: v for k, v in hit["fingerprint"].items()
                 if k not in ("rev", "modes")} == trace_key):
        return hit
    if not allow_load:
        return None
    import xarray as xr
    from types import SimpleNamespace
    ds = xr.open_dataset(tp, group="posterior")
    n_total = int(ds.sizes["chain"])
    keep = [c for c in range(n_total) if c not in set(drop)]
    sub = ds[["A", "theta", "D"]].isel(chain=keep).load()
    ds.close()
    row = {"fingerprint": fp, "drop_chains": drop, "chains_kept": len(keep),
           "chains_total": n_total, "draws_per_chain": int(sub.sizes["draw"]),
           **mirt_identified_ess(SimpleNamespace(posterior=sub), data)}
    cache[fit["name"]] = row
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _MODE_EVAL_CACHE.write_text(json.dumps(cache, indent=1))
    return row


def compute_loo_waic(idata) -> dict:
    """Pointwise PSIS-LOO + WAIC. Returns the
    pointwise log densities so ΔELPD standard errors are computed correctly."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loo = az.loo(idata, pointwise=True)
        waic = az.waic(idata, pointwise=True)
    loo_i = getattr(loo, "loo_i", None)
    if loo_i is None:
        loo_i = loo.elpd_i if hasattr(loo, "elpd_i") else loo["loo_i"]
    waic_i = getattr(waic, "waic_i", None)
    if waic_i is None:
        waic_i = waic.elpd_i if hasattr(waic, "elpd_i") else waic["waic_i"]
    pareto_k = np.asarray(loo.pareto_k.values).ravel()
    loo_i_arr = np.asarray(loo_i.values).ravel()
    waic_i_arr = np.asarray(waic_i.values).ravel()
    return {
        "loo_elpd": float(loo.elpd_loo), "loo_se": float(loo.se),
        "loo_p_eff": float(loo.p_loo), "waic_elpd": float(waic.elpd_waic),
        "waic_se": float(waic.se), "waic_p_eff": float(waic.p_waic),
        "n_obs": int(loo_i_arr.size),
        "pareto_k_good": int((pareto_k < 0.5).sum()),
        "pareto_k_ok": int(((pareto_k >= 0.5) & (pareto_k < 0.7)).sum()),
        "pareto_k_bad": int(((pareto_k >= 0.7) & (pareto_k < 1.0)).sum()),
        "pareto_k_very_bad": int((pareto_k >= 1.0).sum()),
        "pareto_k_max": float(pareto_k.max()), "pareto_k_mean": float(pareto_k.mean()),
        "loo_pointwise": loo_i_arr, "waic_pointwise": waic_i_arr,
    }


def _loo_min_ess_tau(idata) -> tuple:
    """(loo dict | None, min_ess | None, tau_sorted | None) from one loaded trace.

    LOO/WAIC (only if the trace carries a log_likelihood group), min ESS over
    the per-obs log-likelihood, and the descending posterior-mean τ_A spectrum
    (skipped when constant — shared-scale fits)."""
    loo = min_ess = tau = None
    if "log_likelihood" in list(idata.groups()):
        loo = compute_loo_waic(idata)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            min_ess = float(az.ess(idata.log_likelihood)["obs"].min())
    if "tau_A" in idata.posterior:
        t = np.sort(idata.posterior["tau_A"].mean(("chain", "draw")).values)[::-1]
        if float(np.std(t) / np.mean(t)) > 0.02:
            tau = t
    return loo, min_ess, tau


def _per_bench_rmse(scores, y_pred_mean, bench_idx, bench):
    resid = np.asarray(scores) - np.asarray(y_pred_mean)
    pbr = pd.Series(resid ** 2).groupby(bench_idx).mean() ** 0.5
    return pd.Series(pbr.values, index=[bench[i] for i in pbr.index])


def render_fit(fit, data, raw, bench, mod, want_png):
    """Figures + comparison-row metrics for a MIRT fit (comp or non-comp)."""
    idata = az.from_netcdf(_trace_path(fit))
    if fit.get("drop_chains"):
        # Mode-restricted card: drop outlier chain(s) before ANY summary, so the
        # whole card (timeline, forecast, GoF, r̂) reads the majority island.
        keep = [c for c in idata.posterior.chain.values if c not in fit["drop_chains"]]
        idata = idata.sel(chain=keep)
        print(f"  drop_chains {fit['drop_chains']}: kept {len(keep)} chains", flush=True)
    is_nc = fit.get("kind", "comp") == "nc"
    view = prepare_fit(idata, data)
    # Floored/ceilinged fits are scored with the same fixed-c 3PL / fixed-d 4PL
    # link they were fit with. A ceiling-noise fit carries an ESTIMATED
    # `ceiling_d` in its posterior (the fixed wall plus a noise-sized gap);
    # posterior_predictive_mirt picks that up on its own.
    floor_c = load_benchmark_floors(data) if fit.get("floors") else None
    ceiling_d = load_benchmark_ceilings(data) if fit.get("ceilings") else None
    # Same rule for the known-SE noise split, read off the trace so a card needs
    # no registry key: the predictive Beta must carry the per-cell instrument
    # precision the fit used.
    n_eff = data.n_eff if idata.posterior.attrs.get("mirt_known_se") else None
    yrep = (posterior_predictive_mirt_nc(idata, data) if is_nc
            else posterior_predictive_mirt(idata, data, floor_c=floor_c,
                                           ceiling_d=ceiling_d, n_eff=n_eff))
    mu = (posterior_predictive_mirt_nc(idata, data, return_mean=True) if is_nc
          else posterior_predictive_mirt(idata, data, floor_c=floor_c,
                                         ceiling_d=ceiling_d, n_eff=n_eff,
                                         return_mean=True))
    gof = compute_gof(yrep, data, mu)
    figures = build_fit_figures(view, gof, yrep, data, raw, bench, mod, idata,
                                forecast=fit.get("forecast", False))
    modes = _fit_modes(fit)
    if modes:
        print(f"  {len(modes['modes'])} posterior modes: "
              + " | ".join(_mode_tag(m) for m in modes["modes"]), flush=True)
        figures.update(_mode_figures(idata, modes["modes"], data, raw, bench))
    if trace_loading_prior(idata) in ("signed", "signedhs"):
        # rotation-method comparison, from the CSV the fit driver wrote —
        # no recompute (K-tagged first; legacy untagged fallback).
        adir = ROOT / "results" / fit.get("dir", "mirt")
        for cand in ():   # promax-only dashboard (2026-07-05): multi-method figure dropped
            if cand.exists():
                figures["rotation_methods"] = alignment_methods_fig(pd.read_csv(cand))
                break

    pred_rhat = (mirt_identified_rhat_nc(idata, data)["logmu_max_rhat"] if is_nc
                 else mirt_identified_rhat(idata, data)["eta_max_rhat"])
    div = int(idata.sample_stats["diverging"].sum()) if "diverging" in idata.sample_stats else -1
    n_draws = int(idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"])
    # cross-chain axis reproducibility (needs the raw chains, so computed here
    # while the trace is in memory). Householder loadings are signed → align ±;
    # signed-free fits are ALREADY per-draw aligned inside prepare_fit, so
    # their reproducibility is read straight off the aligned view.theta.
    lp = trace_loading_prior(idata)
    if is_nc:
        axis_repro = None
    elif lp in ("signed", "signedhs") and view.K > 1:
        axis_repro = _aligned_reproducibility(view.theta,
                                              int(idata.posterior.sizes["chain"]))
    elif view.A is not None and view.K > 1 and not view.anchored:
        # Free 'normal' loadings: axes are identified only up to PERMUTATION
        # (shared scalar tau_A can't rank them), so crosschain_axis_repro's tau
        # ranking reads ~0 even when every chain finds the same axes. Match
        # chains by loading correlation first.
        axis_repro = permutation_matched_reproducibility(
            view.A, int(idata.posterior.sizes["chain"]))
    else:
        axis_repro = crosschain_axis_reproducibility(idata)
    free_load = int((np.abs(np.median(view.A, axis=0)) > 1e-9).sum()) if view.A is not None else 0
    max_phi = (round(float(np.abs(view.Phi_raw[np.triu_indices(view.K, 1)]).max()), 3)
               if view.K > 1 else "—")
    loo, min_ess, tau = _loo_min_ess_tau(idata)
    if lp in ("signed", "signedhs") and view.A is not None and view.K > 1:
        # Shared scalar tau is flat by construction — the spectrum lives in the
        # ALIGNED per-axis column norms.
        tau = np.sort(np.median(np.linalg.norm(view.A, axis=1), axis=0))[::-1]
    m = gof.metrics
    result = {
        "fit": fit["label"], "name": fit["label"], "type": fit["type"], "K": view.K,
        "free_loadings": free_load, "R2": round(m["bayesian_r2"], 4),
        "RMSE": round(m["rmse"], 4), "MAE": round(m["mae"], 4),
        "PIT_var": round(m["pit_var"], 4), "eta_rhat": round(pred_rhat, 3),
        "divergences": div, "max_phi": max_phi, "pit": gof.pit,
        "per_bench_rmse": _per_bench_rmse(data.scores, gof.y_pred_mean, data.bench_idx, bench),
        "min_ess": min_ess, "tau_sorted": tau,
        "axis_repro": axis_repro, "div_pct": 100.0 * div / n_draws,
        # mean loading vectors + axis names, for the cross-fit axis-match map
        "mean_load": (np.median(view.A, axis=0) if view.A is not None else None),
        "axis_names": view.names,
    }
    if loo:
        result.update(loo)   # keeps n_obs so build_comparison can flag obs-count mismatch
    del idata
    return figures, result


def _k2_diagnosis_row(data):
    """K=2 row for the axis-reproducibility diagnosis ONLY.

    K=2 has no dashboard card (its axes are a difficulty split, see FITS), but
    it is the pivotal contrast for the exploratory diagnosis: drop the
    degenerate weak pair and the same unconstrained prior converges. One extra
    trace load, one row."""
    path = ROOT / "results" / "mirt" / "trace_mirt_k2.nc"
    if not path.exists():
        return None
    if _trace_n_models(path) != data.n_models:
        return None          # old data generation — indexing it would misalign
    idata = az.from_netcdf(path)
    rhat = mirt_identified_rhat(idata, data)["eta_max_rhat"]
    div = int(idata.sample_stats["diverging"].sum()) if "diverging" in idata.sample_stats else 0
    n_draws = int(idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"])
    row = {"fit": "K=2 ARD", "type": "exploratory", "K": 2,
           "eta_rhat": round(float(rhat), 3),
           "axis_repro": crosschain_axis_reproducibility(idata),
           "div_pct": 100.0 * div / n_draws}
    del idata
    return row


def _ppca_fig():
    """PPCA ARD τ spectrum (the EDA that motivated MIRT), if the trace is present."""
    path = ROOT / "diagnostics" / "ppca_all_trace.nc"
    if not path.exists():
        return None
    t = az.from_netcdf(path)
    K = t.posterior.sizes["latent"]
    tau = t.posterior["tau"].values.reshape(-1, K)
    tau_sorted = np.sort(tau, axis=1)[:, ::-1]
    med = np.median(tau_sorted, axis=0)
    lo, hi = np.percentile(tau_sorted, [5, 95], axis=0)
    return ppca_spectrum_fig(med, lo, hi, [f"axis{k+1}" for k in range(K)])


def _stat_line(r):
    return (f"type: {r['type']} · K={r['K']} · free loadings={r['free_loadings']} · "
            f"R²={r['R2']} · RMSE={r['RMSE']} · MAE={r['MAE']} · "
            f"identified r̂={r['eta_rhat']} · {r['divergences']} divergences · "
            f"max |Φ|={r['max_phi']}")


def _static_dump(sections, comparison, fmt="png"):
    """Optional static stills → plots/dashboard/ (git-ignored). Needs kaleido/Chrome.

    `fmt` is any kaleido format: "png" (raster, scale=2) or "pdf" (vector, crisper
    for slides/print — resolution-independent, so no `scale`)."""
    base = ROOT / "plots" / "dashboard"
    for sec in sections + [{"id": "comparison", "figures": comparison["figures"]}]:
        out = base / sec["id"]
        out.mkdir(parents=True, exist_ok=True)
        for name, fig in sec["figures"].items():
            try:
                kw = {} if fmt == "pdf" else {"scale": 2}
                fig.write_image(out / f"{name}.{fmt}", **kw)
            except Exception as e:  # noqa: BLE001
                print(f"  {fmt.upper()} skipped {sec['id']}/{name} ({type(e).__name__}: {e})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", action="store_true",
                    help="also dump static PNG stills to plots/dashboard/ (git-ignored)")
    ap.add_argument("--pdf", action="store_true",
                    help="also dump vector PDF stills to plots/dashboard/ (crisper; git-ignored)")
    ap.add_argument("--force", nargs="*", default=[],
                    help="fit names (registry `name`) to re-render even if cached")
    ap.add_argument("--force-all", action="store_true",
                    help="ignore the render cache entirely and re-render every fit")
    args = ap.parse_args()

    CMP_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(ROOT / "data" / "processed" / "benchmarks_merged.csv")

    # Fits span two data scopes (curated-exclusions K=1 vs full-set K=3). Load
    # each scope once and hand every fit the data it was actually fit on so the
    # n_models sanity-check + per-bench RMSE line up with its trace.
    _scope_cache: dict = {}
    def scoped(include_all, cyber, sqaorig=False, drop=None):
        key = (include_all, cyber, sqaorig, tuple(drop or ()))
        if key not in _scope_cache:
            d = load_eci_data(include_all_benchmarks=include_all, fit_cyber=cyber,
                              fit_simpleqa_original=sqaorig,
                              drop_benchmarks=list(drop) if drop else None)
            _scope_cache[key] = (
                d,
                d.blookup.sort_values("benchmark_idx")["benchmark"].tolist(),
                d.mlookup.sort_values("model_idx")["model"].tolist(),
            )
        return _scope_cache[key]

    sections, results, mode_rows = [], [], []
    for fit in FITS:
        data, bench, mod = scoped(fit.get("include_all", True), fit.get("cyber", False),
                                  fit.get("sqaorig", False),
                                  fit.get("drop_benchmarks"))
        tp = _trace_path(fit)
        cache = None if (args.force_all or fit["name"] in args.force) \
            else _cache_load(fit["name"])
        fp = _fingerprint(tp, _modes_path(fit)) if tp.exists() else None
        note = None
        if cache is not None and (fp is None or cache["fingerprint"] == fp):
            # Serve the cached card: trace unchanged, or trace retired/deleted.
            note = "cached" if fp is not None else "cached · trace retired"
            print(f"{note}: {fit['label']}", flush=True)
            figures, r = cache["figures"], cache["result"]
            r["fit"] = fit["label"]      # registry label wins over render-time label
        elif fp is None:
            print(f"skipping {fit['label']} — no trace, no cache ({tp.name})", flush=True)
            continue
        elif _trace_n_models(tp) != data.n_models:
            # Trace belongs to an older data generation, so it CANNOT be
            # re-rendered — indexing it against today's data would misalign every
            # row. The cached card is the only truthful option, so read it even
            # under --force / --force-all: those flags mean "do not serve a cache
            # you could re-render", not "drop a card you cannot". Without this
            # fallback, --force-all silently deleted 11 of 12 cards.
            if cache is None:
                cache = _cache_load(fit["name"])
            if cache is not None:
                note = "cached · data superseded"
                print(f"{note}: {fit['label']} (trace has "
                      f"{_trace_n_models(tp)} models, data has {data.n_models})",
                      flush=True)
                figures, r = cache["figures"], cache["result"]
                r["fit"] = fit["label"]
            else:
                print(f"skipping {fit['label']} — trace fit on a different data "
                      f"generation ({_trace_n_models(tp)} models vs "
                      f"{data.n_models}) and no cached card", flush=True)
                continue
        else:
            print(f"rendering {fit['label']} ({fit['type']}) …", flush=True)
            render = render_fit
            data_fit = _drop_model_obs(data, fit["drop_model_obs"]) \
                if fit.get("drop_model_obs") else data
            if fit.get("drop_bench_obs"):
                data_fit = _drop_bench_obs(data_fit, fit["drop_bench_obs"])
            if fit.get("drop_gpqa_cells"):
                data_fit = _drop_gpqa_cells(data_fit, fit["drop_gpqa_cells"])
            if fit.get("drop_arcagi_cells"):
                data_fit = _drop_arcagi_cells(data_fit, fit["drop_arcagi_cells"])
            if fit.get("floors"):
                # Floored fits were fit on clip-to-floor scores; score them on
                # the same data (mirrors fit.py --floors).
                data_fit = clip_scores_to_floors(data_fit, load_benchmark_floors(data_fit))
            figures, r = render(fit, data_fit, raw, bench, mod, args.png)
            _cache_save(fit["name"], figures, r, fp)
        # Comparison charts key on r["fit"]/r["name"]: use the SHORT label there
        # (the long label made the grouped-bar / heatmap axes unreadable). The
        # long label stays on the nav + section header below.
        # Factor correlations (Phi) are dropped from the page: on the positive
        # cards the frame is raw and non-negative, so Phi reports ability
        # correlation, not a rotation choice worth a panel. Filtered at
        # assembly rather than at render, so the cached figure survives.
        figures = {k: v for k, v in figures.items() if k != "factor_correlations"}
        short = fit.get("short", fit["label"])
        r["fit"] = r["name"] = short
        stat = _stat_line(r) + (f" · <i>{note}</i>" if note and "cached" != note else "")
        # Mode table read from the JSON at assembly, not from the render: a card
        # served from cache still gets its current mode summary.
        modes = _fit_modes(fit)
        sections.append({"id": fit["name"], "label": fit["label"], "type": fit["type"],
                         "stat_line": stat, "figures": figures,
                         "table_html": _modes_table_html(modes) if modes else None,
                         "group": fit.get("group"), "nav": fit.get("nav")})
        results.append(r)
        if fp is not None:
            # Mode-evaluation row: what each card's kept-chain posterior is
            # worth, in the project's preferred readout (min/median bulk ESS
            # + divergences), next to its fit metrics. A superseded card gets its
            # row from the ESS cache only — its trace cannot be indexed against
            # today's data — and is dropped from the table if there is none.
            e = _identified_ess_cached(fit, tp, fp, data,
                                       allow_load=note != "cached · data superseded")
        if fp is not None and e is not None:
            n_kept = e["chains_kept"] * e["draws_per_chain"]
            # ESS travels with the draw count it came out of: cards keep
            # 4,000-54,000 draws, so an ESS is only readable next to its
            # ceiling.
            r["eta_ess_min"], r["eta_ess_med"] = e["eta_ess_min"], e["eta_ess_med"]
            r["n_draws_kept"] = n_kept
            mode_rows.append({
                "fit": short,
                "chains": f"{e['chains_kept']}/{e['chains_total']}",
                "draws": e["chains_kept"] * e["draws_per_chain"],
                "R2": r["R2"], "RMSE": r["RMSE"],
                "elpd_loo": round(r["loo_elpd"], 1),
                "loo_se": round(r["loo_se"], 1),
                "eta_rhat": r["eta_rhat"],
                "ess_min": int(round(e["eta_ess_min"])),
                "ess_med": int(round(e["eta_ess_med"])),
                "divergences": r["divergences"],
                # LOO reliability travels with the ELPD it qualifies: share of
                # observations whose importance weights failed the k<0.7 check.
                "khat_gt_0.7": round(
                    (r["pareto_k_bad"] + r["pareto_k_very_bad"]) / r["n_obs"], 3),
            })

    # Raw-data section: no fit, no trace, no cache — the processed table itself
    # plus every opt-in curated extra-benchmark file (cyber, SimpleQA original;
    # same schema as the processed file), one legend-toggleable trace per
    # benchmark.
    extra_csvs = [
        ROOT / "data" / "curated" / "cyber_benchmarks.csv",
        ROOT / "data" / "curated" / "simpleqa_original" / "simpleqa_original.csv",
    ]
    raw_all = pd.concat(
        [raw] + [pd.read_csv(f) for f in extra_csvs if f.exists()],
        ignore_index=True)
    sections.append({
        "id": "rawdata", "label": "Raw scores by release date", "type": "data",
        "stat_line": f"{len(raw_all)} observations · "
                     f"{raw_all['benchmark'].nunique()} benchmarks · click legend "
                     "entries to overlay benchmarks",
        "figures": {"scores_by_release_date": raw_scores_by_date_fig(raw_all)}})

    # ── comparison tables + figures (one pass, no ordered multi-script run) ──
    tables, cmp_figs = build_comparison(results)

    cmp_dir = CMP_DIR

    save_df(tables["gof_table"], cmp_dir / "gof_table.csv")
    if "loo_waic_table" in tables:
        save_df(tables["loo_waic_table"], cmp_dir / "loo_waic_table.csv")
    print("\n" + tables["gof_table"].to_string(index=False), flush=True)

    mode_df = pd.DataFrame(mode_rows)
    if len(mode_df):
        save_df(mode_df, cmp_dir / "mode_eval_table.csv")
        print("\n" + mode_df.to_string(index=False), flush=True)

    # One table on the page: the mode evaluation. gof_table / loo_waic_table
    # stay as CSVs (and feed the README) — every column a reader needs is in
    # the mode table, keyed by the same short fit names.
    tables_html = ("<h3>mode evaluation (per-card kept chains)</h3>"
                   + mode_df.to_html(classes="cmp", index=False, border=0)
                   if len(mode_df) else "")
    comparison = {"tables_html": tables_html, "figures": cmp_figs}

    html = assemble_dashboard(sections, comparison)
    write_dashboard(html, INDEX_PATH)
    _readme(tables["gof_table"])
    if args.png:
        _static_dump(sections, comparison, "png")
    if args.pdf:
        _static_dump(sections, comparison, "pdf")
    size_mb = INDEX_PATH.stat().st_size / 1e6
    print(f"\nDashboard → {INDEX_PATH}  ({size_mb:.1f} MB, {len(sections)} fits)", flush=True)


def _df_to_md(df):
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    rows += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def _readme(tab):
    by_type = {t: [f["label"] for f in FITS if f["type"] == t]
               for t in ("baseline", "exploratory", "confirmed")}
    fit_list = "\n".join(
        f"- **{t}**: " + ", ".join(by_type[t]) for t in by_type if by_type[t])
    txt = f"""# Capability-dimensionality fit dashboard

**Open [`index.html`](../../index.html)** at the repo root — one
self-contained interactive page: a fit
selector (grouped baseline / exploratory / confirmed) + a cross-fit Comparison
view. Every figure renders lazily (only the visible fit is live in the DOM).

Cards span two data scopes: the K=1 fits apply curated benchmark exclusions at
fit time (702 models / 77 benchmarks); the K=3 no-SG fit uses the full set
(751 models / 85 benchmarks). GoF is comparable within a scope; global R²/RMSE
stay readable across scopes, per-benchmark rows do not:

{fit_list}

## Comparison table (`gof_table.csv`)

{_df_to_md(tab)}

Columns: `type`; `free_loadings` = free loading cells (complexity); `R2`/`RMSE`/
`MAE` = fit; `PIT_var` = calibration (ideal 0.083, below = under-confident);
`eta_rhat` = convergence (≤ 1.01 = converged; raw r̂ for 1D, log_μ r̂ for non-comp);
`divergences`; `max_phi` = largest off-diagonal axis correlation. PSIS-LOO / WAIC
per fit are in `loo_waic_table.csv` and the dashboard's Comparison view.

## The headline

The **exploratory** fits post high R² but **do not converge** (eta r̂ ≫ 1.01). The
**confirmed** fits converge only because the Q-matrix supplies the rotation
externally. The **non-comp** fits converge but fit worse than 1D. High R² is not
trust — read `cmp_convergence` alongside `cmp_gof`.

## Regenerate

One command: `python diagnostics/build_dashboard.py` (add `--png` for static
stills → git-ignored `plots/dashboard/`). The ordered-human-prior comparison
(`compare_human_prior.py`) remains a separate one-off.
"""
    (CMP_DIR / "README.md").write_text(txt)


if __name__ == "__main__":
    main()
