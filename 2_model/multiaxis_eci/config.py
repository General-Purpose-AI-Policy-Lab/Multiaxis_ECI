"""Configuration constants for the ECI Bayesian recreation."""
from __future__ import annotations

import math
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # 2_model/multiaxis_eci/config.py -> repo root
INPUT_DIR    = PROJECT_ROOT / "0_input"     # tables of benchmark-data-pipeline, copied by `python -m multiaxis_eci sync`
CURATED_DIR  = PROJECT_ROOT / "1_curated"   # inputs specific to this model (exclusions, lineages, SOTA list, clips)
PIPELINE_DIR = PROJECT_ROOT.parent / "benchmark-data-pipeline"
# name here -> folder of the pipeline checkout; `sync` reads the file from that folder's latest
# run, `<folder>/<YYYYMMDD>/` (the date the pipeline ran)
INPUT_FILES = {
    "all_scores_flat.csv": "3_views",      # every score with release date, organization, category, floor, ceiling
    "human_baselines.csv": "3_views",      # one row per published human measurement
    "models.csv":          "2_database",   # identity, organization, country
    "benchmarks.csv":      "2_database",   # category, floor, ceiling, access class
    "build_manifest.json": "2_database",
}
PROVENANCE_FILE = "provenance.json"


def data_tag() -> str:
    """Date the synced pipeline run happened, YYYYMMDD (its `2_database/<date>/` folder), or ''
    before the first sync.

    Names the per-generation output folder (`5_outputs/data<tag>/`), so fits on two
    data generations never overwrite each other, and a later analysis of the same data
    lands in the same folder.
    """
    import json
    p = INPUT_DIR / PROVENANCE_FILE
    if not p.exists():
        return ""
    return json.loads(p.read_text()).get("pipeline_run_date", "")


DATA_TAG = data_tag()

# ── Output layout ──────────────────────────────────────────────────────────
# Everything a fit or a diagnostic writes goes under 5_outputs/<data generation>/:
#   <fit>/                 tables, trace.nc and figures/ (PNG) + figures/html/ per fit
#   comparisons/           cross-fit tables (frontier gap, chain verdicts) + figures/
#   diagnostics/           one-off diagnostic outputs (residual correlations, chain plots)
# 5_outputs/pre_pipeline/ holds the fits published before the data pipeline existed.
OUTPUTS_DIR     = PROJECT_ROOT / "5_outputs"
DATA_DIR_NAME   = f"data{DATA_TAG}" if DATA_TAG else "data_unsynced"
RESULTS_DIR     = OUTPUTS_DIR / DATA_DIR_NAME
COMPARISONS_DIR = RESULTS_DIR / "comparisons"
DIAGNOSTICS_DIR = RESULTS_DIR / "diagnostics"
LEGACY_RESULTS_DIR = OUTPUTS_DIR / "pre_pipeline"
FIGURES_DIRNAME = "figures"
WRITEUPS_DIR    = PROJECT_ROOT / "6_writeups"

# ── Zero-score diagnostic threshold ───────────────────────────────────────
# The level `ppc.py` scores `zero_pred_below_threshold` against: the posterior
# probability that the exact-zero rows replicate at or below 0.5%. Diagnostic
# only — it enters no likelihood.
ZERO_DIAG_THRESHOLD = 5e-3

# ── Epsilon bound ────────────────────────────────────────────────────────
# Beta likelihood has open support on (0, 1). `models/mirt.py` clips boundary
# scores onto [ECI_EPS, 1 - ECI_EPS]; the same epsilon lets 3_fit/fit.py's
# --drop-zero-scores identify rows on or outside (ECI_EPS, 1 - ECI_EPS).
ECI_EPS = 1e-3
# The bound of the continuous benchmarks under the default Beta-binomial likelihood
# (models/mirt.py, binomial_n): item-scored benchmarks need none (a 0 is k = 0), so only the
# few exact zeros of a continuous score are clipped, to 5e-3 rather than 1e-3 so they sit in
# the noise of a near-zero score (logit -5.3 instead of -6.9) instead of pulling a benchmark's
# noise down. Chosen 2026-10-01; the Beta-only likelihood keeps ECI_EPS.
CONTINUOUS_EPS = 5e-3

# ── Low-observation flag ─────────────────────────────────────────────────
# Models with fewer than this many benchmark observations are flagged as
# data-poor (ECIData.is_low_obs): their posterior is dominated by the prior and
# a few extreme points. Every fit keeps them; the MIRT axis timelines and the
# forecast candidates hide them (SOTA families exempt), the K=1 ECI-H timeline
# draws every dated model. Also the "4 benchmarks" rule of the SOTA list.
LOW_OBS_THRESHOLD = 4

# ── Informed-ability cap ────────────────────────────────────────────────────
# The posterior-SD cap of `analysis.timelines.mirt_informed_mask` (history 0.6 -> 0.3
# -> 0.4 -> 0.33 in its docstring). Since 2026-09-11 it no longer gates the K-axis
# figures or the forecast candidates (`MIN_AXIS_COVERAGE` does); the K=1 frontier tools
# and the bimodality scan still pass it explicitly.
INFORMED_SD_CAP = 0.33

# ── Priors (LogNormal mu, sigma on the log scale) ─────────────────────────
PRIOR_SIGMA_B   = dict(mu=math.log(0.05), sigma=0.5)
PRIOR_TAU_CD    = dict(mu=math.log(3.0),  sigma=1.0)
PRIOR_TAU_ALPHA = dict(mu=math.log(0.5),  sigma=0.5)
PRIOR_ALPHA = dict(mu=0.0, sigma=0.5)   # loglog link: per-bench discrimination cells, x tau_alpha (marginal median 0.5)

# Hierarchical sigma_b (models/mirt.py, build_mirt_model(..., pooled_noise=True)).
# The population LOCATION of the log noise scale is learned, centered on the
# fixed prior's median 0.05; HalfNormal(0.5) on the spread covers the 90% range
# ([0.022, 0.114]) the fixed LogNormal(log 0.05, 0.5) implied, so the pooled
# model nests the fixed one instead of tightening it.
PRIOR_SIGMA_B_POOLED = dict(mu_loc=math.log(0.05), mu_sd=0.5, tau_sd=0.5)

# Ordered human prior (models/mirt.py, build_mirt_model(..., human_order=...)).
# Human baseline tiers enter the MIRT as test-takers but their per-axis ability
# is poorly pinned by sparse data and can violate the natural ranking. This map
# encodes the ranking directly in theta-space as a PARTIAL order: tier → parent
# tier (None = root). On EVERY axis a tier's ability = its parent's + a POSITIVE
# increment, so (with A >= 0) score-level ordering is guaranteed on every
# benchmark — but only along parent chains. Tiers on different branches are
# deliberately incomparable: a Top Performer is not assumed better (or worse)
# than a Committee of Domain Experts, committees only dominate their own base
# tier, and the High School pair is ordered internally without any assumption
# about how it compares to the adult tiers. Declared parents-first.
HUMAN_ORDER = {
    "Average Human": None,
    "Skilled Generalist": "Average Human",
    "Domain Expert": "Skilled Generalist",
    "Committee of Domain Experts": "Domain Expert",
    "Top Performer": "Domain Expert",
    "Committee of Average Humans": "Average Human",
    "Committee of Skilled Generalists": "Skilled Generalist",
    "High School Qualifier": None,
    "High School Top Performer": "High School Qualifier",
}
# Same tiers, High School branch MERGED into the adult spine instead of left
# incomparable (--human-merge). A TUPLE of parents means the tier dominates
# EVERY one of them: theta = max(parents) + a positive increment, so a Domain
# Expert beats both a Skilled Generalist and a High School Qualifier, and a Top
# Performer beats both a Domain Expert and a High School Top Performer. The
# reverse pairings stay unstated: nothing here ranks a High School Qualifier
# against a Skilled Generalist. The max is the price of the extra edge — the
# gradient jumps where two parents cross, which the flat partial order avoids.
HUMAN_ORDER_MERGED = {
    **HUMAN_ORDER,
    "Domain Expert": ("Skilled Generalist", "High School Qualifier"),
    "Top Performer": ("Domain Expert", "High School Top Performer"),
}
# The one human tier whose scores are self-contradictory across benchmark
# families (0.98 on ARC-AGI vs 0.22 on GPQA Diamond — the dataset's only
# cross-family straddler). --no-sg drops its OBSERVATIONS at fit time; the tier
# stays in HUMAN_ORDER (prior-only theta). See data.drop_model_observations.
SG_MODEL_NAME = "Skilled Generalist"
# sigma of the HalfNormal increment between adjacent tiers (z-score space, before
# tau scaling). 1.0 = weakly informative — enforces the ordering but lets the
# data drive the gap sizes. (Reverted 2026-07-05: the soft / pooled / Student-t
# variants each relocated the Skilled Generalist instability without removing
# it; the original hard prior is the reference condition for hypothesis tests.)
PRIOR_DELTA_HUMAN = 1.0

# Lineage prior (models/mirt.py, build_mirt_model(..., lineage=...)). Within a
# vendor-tier release chain, a release's per-axis ability is
#   psi[node] = founder + cumulative Normal(mu>0, s) increments
# (SOFT — improvement is the mean step but a node can regress), and each
# test-taker = psi[node] + a tight mean-zero variant offset. Founder and the
# unstructured anchor share the unit theta scale (Normal(0,1)), so only three
# scales are tunable: the drift mean, the regression tolerance, and the offset.
PRIOR_LINEAGE_DRIFT  = 0.3    # scale of the positive per-step drift mean mu[k]
PRIOR_LINEAGE_DELTA  = 1.0    # scale of the per-step regression tolerance s[k]
PRIOR_LINEAGE_OFFSET = 0.25   # tight scale of the per-axis variant-offset sd tau_o[k]

# Brownian-motion lineage variant: step mean and variance scale with the years
# between releases (delta_dt) instead of being iid per step. Same structure as
# above — one shared drift and one tolerance per axis — only the units change
# (per year, not per release). The median live release gap is 0.285 years, so
# these BM scales reduce to the per-step scales above (0.3, 1.0) on a typical
# step.
PRIOR_LINEAGE_DRIFT_BM    = 1.0   # HalfNormal scale of the improvement rate mu[k], logits/year
PRIOR_LINEAGE_DELTA_BM    = 2.0   # HalfNormal scale of the per-step diffusion s[k], logits/sqrt(year)

# Time prior (models/mirt.py, build_mirt_model(..., time_t=...)). The theta prior
# MEAN becomes a per-axis line in centered release year: theta = beta[k]*t + ZSN.
# beta is signed and centered at zero, so a flat population (beta -> 0) reduces
# to the plain exchangeable prior; the data pick the slope. 0.5 logits/year of
# prior sd spans roughly +-1 logits/year at 2 sd, wide next to the measured
# within-chain climb rates (0.23-0.84) which this population slope need not match.
PRIOR_TIME_BETA = 0.5

# Cell-wise leptokurtic theta (models/mirt.py, build_mirt_model(..., theta_t_cells=True)).
# Degrees of freedom of the per-CELL scale mixture: theta[m,k] = lambda[m,k]*z[m,k]
# with lambda^-2 ~ Gamma(nu/2, nu/2), so each cell is marginally Student-t(nu) at
# unit scale. Fixed, not estimated: nu is the one knob that says how far from
# Gaussian the columns are, and it is exactly the quantity the rotation is
# identified BY — estimating it lets the fit buy back Gaussianity (nu -> inf) and
# with it the flat rotation orbit. 4 is heavy enough for a finite variance but no
# finite kurtosis, matching the excess kurtosis ~7 seen on the fitted axis-3
# column, and it keeps the marginal sd at sqrt(nu/(nu-2)) = 1.41, close to the
# unit scale the rest of the theta priors assume.
PRIOR_THETA_T_NU = 4.0

# Regularized-horseshoe scales, used by the "bifactor" loading prior
# (models/mirt.py) and the sparse-gate model's gate horseshoe (models/mirt_sparse.py).
RH_TAU_SCALE  = 0.5   # global-scale prior width
RH_SLAB_DF    = 3.0   # nu in c^2 ~ Inv-Gamma(nu/2, nu*s^2/2); smaller = heavier slab
RH_SLAB_SCALE = 1.0   # s: slab width that soft-caps "on" loadings

# Positive-lower-triangular rotation identification (Geweke & Zhou 1996; Lopes
# & West 2004) — reached via build_mirt_model(plt_founders=...).
# Founder r loads ONLY axes 1..r (hard zeros above the diagonal) with a
# POSITIVE diagonal loading; below-diagonal cells keep the signed prior. Kills
# rotation + sign flips + axis permutation inside the sampler.
#
# Founder ladder, ORDERED so the first K entries serve any K (slice
# PLT_FOUNDERS[:K] when passing plt_founders to build_mirt_model): general
# <- GPQA Diamond (textbook profile +0.83/~0/~0,
# 176 obs) · claudiness <- GSO-Bench (least-bad agentic founder; no benchmark
# measures claudiness without general — audit 2026-07-05) · ARC <- ARC-AGI-2
# (last = least constrained; tests whether an ARC axis exists at all).
# K=2 -> general + claudiness; K=3 -> + the ARC slot.
# Literature caveat: results depend on the founder choice — the list is part
# of the model specification, not a tuning knob.
PLT_FOUNDERS = ["GPQA Diamond", "GSO-Bench", "ARC-AGI-2"]

# Non-comp MIRT: fixed positive baseline for the easiness intercept (logit).
# Without it the conjunctive product collapses toward 0 at average ability.
NC_C_OFFSET = 1.0

# Non-comp MIRT: per-benchmark intercept spread. Tighter than compensatory
# tau_CD (median 1 vs 3) because the product compounds wide spreads into U-shaped scores.
PRIOR_TAU_C = dict(mu=math.log(1.0), sigma=0.5)

# ── Sampling ──────────────────────────────────────────────────────────────
# Every `step`-th draw is written to disk (`persistence.thin_trace`): 2,500 draws x 8
# chains keep 4,000, enough for every median and interval here, at a fifth of the file.
# Convergence and the deliverable tables use the full run before the save.
SAVE_THIN = 5
# The canonical run: 2,500 draws per chain since 2026-10-01 (10,000 before). The 8 x 10,000
# Beta fit of that day had a bulk ESS of 7,200 at least on every ability (median 15,900), 5,100
# in the tails and 2,150 on its worst loading; a quarter of the draws keeps about 1,800, 1,300
# and 540, and the run takes about 2.7 times less.
SAMPLE_KW = dict(
    draws=2500,
    tune=2000,
    chains=8,             # 10 cores here (4 performance, 6 efficiency): 8 chains in parallel
    cores=8,
    target_accept=0.95,
    random_seed=42,
    return_inferencedata=True,
    progressbar=True,
)

# ── ECI affine anchors (For comparability) ───────────────────────────────
# IDs match `model_version` in 0_input/all_scores_flat.csv.
# Where models expose a reasoning-effort suffix (_low/_medium/_high), we pick
# `_medium` as a default-ish operating point and keep the choice consistent
# across the SOTA list — anchors are arbitrary; differences are invariant.
ANCHOR_LOW  = ("claude-3-5-sonnet-20241022", 130.0)
ANCHOR_HIGH = ("gpt-5-2025-08-07_medium",    150.0)

# ── SOTA families (data-driven; newest first) ───────────────────────────────
# NOT hand-maintained. `1_curated/1_compute_sota.py` writes
# 1_curated/sota_families.txt = (every world frontier record on the canonical
# ECI-H, whatever its age, Epoch-style "highest capability accessible at each
# date") ∪ (every family within 10 ECI of the best model of each organisation
# on the recent frontier, records of the last 24 months), candidates with
# >= 4 observations. A family is a release
# (`data.model_family`: base model plus snapshot), every reasoning effort and
# run variant included. Listing families means every effort of a SOTA release
# is shown on the timelines and admitted as a record candidate by the country
# frontier; the SOTA table keeps the best effort of each (`analysis.sota_stats_df`).
# Display and reporting only: the list does not decide what the fit sees, so it
# can be read off the canonical fit without circularity. Re-run the script after
# a canonical fit. Empty when the file is missing.
def _load_sota_families() -> list[str]:
    p = CURATED_DIR / "sota_families.txt"
    if p.exists():
        return [ln.strip() for ln in p.read_text().splitlines()
                if ln.strip() and not ln.startswith("#")]
    return []


SOTA_FAMILIES: list[str] = _load_sota_families()

# The one gate of the K-axis figures and the forecast candidates (`candidate_mask`):
# a dated model is shown on an axis when the axis shares of the benchmarks IT was
# scored on sum to at least this much, one full axis-unit of evidence whatever the
# benchmarks. Below it the position is the prior and the lineage link alone; on the
# 2026-09-08 K=4 fit that is GPT-2 XL, davinci, text-davinci-002 and PaLM 540B on
# the fluid-intelligence axis, whose 5 to 8 scores are all on legacy QA sets. User
# decision 2026-09-11: this rule alone, replacing the posterior-SD cap, the
# low-observation flag and the SOTA exemption that had gated the figures; what
# keeps a thinly-measured release out of the trend fit is the one-effort-per-family
# rule of `analysis.regimes` and the weighting by posterior SD. A K=1 fit has one
# axis with share 1 everywhere, so any scored release passes.
MIN_AXIS_COVERAGE = 1.0

# The ability scale of the figures (analysis.scale): "human" re-expresses every axis so the
# posterior median of the first anchor tier sits at 0 and the second at 1, draw by draw; any
# other value keeps the fitted logit scale. Tables and loadings are never rescaled.
ABILITY_SCALE = "human"
HUMAN_UNIT_ANCHORS = ("Average Human", "Top Performer")

# The frontier-forecast fit shared by the dashboard, the memo and the blog post
# (analysis.regimes, user decision 2026-09-10): the running top-`top_k` frontier of
# posterior MEDIANS of one effort per family (the best on the axis) and one release
# per organization and day, split into
# reasoning models and the others, one weighted straight line per regime (noise =
# each point's posterior SD plus a common dispersion), the reasoning line projected
# from its first release, the others' drawn over their own span; 80% intervals. The
# candidates are `candidate_mask`'s (MIN_AXIS_COVERAGE). The per-draw ENVELOPE and
# the regression bases of `mirt_frontier_forecast` remain available as `fit_basis`
# alternatives (`fit_start` only matters to them).
FORECAST_KW = dict(fit_basis="regimes", top_k=2, fit_start="2024-10-01", hdi_prob=0.8)

# A release is a reasoning model when its family carries a reasoning level, a
# thinking budget or a thinking variant in the models table, or when its name
# matches one of these patterns (case-insensitive; families whose default mode
# reasons without an effort suffix in the data). Edit here when a new family lands.
REASONING_FAMILY_PATTERNS = [
    r"(^|[^a-z0-9])(o1|o3|o4)[-_]", r"DeepSeek-R1", r"QwQ",
    # Gemini 2.5 Flash-Lite ships with thinking off; its thinking runs carry a budget.
    r"gemini-2\.5-(?!flash-lite)", r"gemini-3", r"grok-4", r"grok-3-mini", r"kimi-k2\.[5-9]", r"kimi-k3",
    r"glm-5", r"qwen3", r"MiniMax-M2", r"deepseek-v3\.[12]", r"deepseek-v4",
    r"claude-opus-4", r"claude-sonnet-4", r"claude-mythos", r"claude-fable", r"gpt-5", r"gpt-6",
    r"magistral", r"gpt-oss",
    # Reasoning by default without an effort suffix in the data, per their developers (2026-10-01):
    r"muse-spark", r"grok-build", r"grok-code-fast", r"mimo-v2", r"gemma-4", r"seed-oss",
    r"mercury-2", r"nemotron-3", r"glm-4\.[5-9]", r"laguna", r"-thinking($|[_-])",
    r"step-3\.7", r"step-5", r"hy4", r"Ring-2", r"Composer 2", r"^swe-1\.[6-9]", r"solar-pro-3",
    r"command-a-plus", r"reka-flash-3", r"MiniMax-M1", r"codex-mini", r"cogito-.*-v2",
    r"ChatGPT agent", r"gemini-robotics-er", r"qwen-plus-2025-04-28",
]

# Every pattern above and below must match a current model (tests/test_model_metadata.py), so a
# rule nothing uses any more is dropped rather than left to catch a future name by accident.
# Releases that match a pattern above but do not reason, per their developers (2026-10-01): Qwen's
# separate Instruct / Coder / Omni non-thinking models, the first Qwen3-Max (its Thinking release
# came later), xAI's fast non-reasoning variant, OpenAI's Instant and Chat models. A run with
# reasoning off (`_none`) is a non-reasoning run whatever its family (user, 2026-10-01). Checked
# before the patterns and the models table, so a match here is always a non-reasoning model.
NON_REASONING_PATTERNS = [
    r"-instruct-2507", r"qwen3-coder", r"qwen3-next-.*-instruct", r"qwen3-omni-.*-instruct",
    r"^qwen3-max-2025-09-23", r"non-reasoning", r"-instant($|[_-])", r"_chat$", r"_none$",
]

# Lines hybrid by design (thinking optional), named by their release ids: a release of these that
# thinks on some runs is hybrid even where the data shows no run without thinking, so its bare
# run's regime is undetermined (analysis.regimes.hybrid_families). Releases with a `_none` /
# `_minimal` / chat run or a `_thinking` variant are found hybrid from the data, whatever the line.
HYBRID_LINE_PATTERNS = [r"^claude-", r"^muse-spark"]

# Test-takers whose regime is undetermined (user decision 2026-10-01): left out of both regime
# fits (they stay in the model and on the figures). The bare run of every hybrid release is one
# too (analysis.regimes.regime_ignored): since the pipeline's 2026-10-01 build those runs gather
# the scores of unknown thinking mode only (system cards, agent harnesses), the runs without
# thinking being `_none` and the thinking ones `_thinking` or an effort. Here: Muse Spark of
# April (Instant by default, the mode of its scores is unknown; no other run of the release).
REGIME_IGNORED = frozenset({"muse-spark"})

# Optional backcast clamp per axis (envelope basis): a tier already passed at
# the window start is backcast at the envelope's early rate, and an entry here
# would keep that date from landing before it. Empty on purpose: the figures
# report the RAW extrapolated date (the crossover window still clips the
# drawing at 2015, with the true year printed at the edge).
FORECAST_BACKCAST_FLOOR: dict[str, str] = {}

# Axis titles are never hard-coded: each fit folder carries an axis_names.json
# written as a template by the fit and filled in by hand (analysis.axes).
# Frontier releases pinned into every per-axis forest whatever their posterior SD:
# a frontier model is the headline of the figure, so a wide interval is drawn
# rather than the row dropped. Names of the published fit; a refit's newest
# frontier releases are added here by hand.
FOREST_PINNED_RELEASES = {"claude-mythos-preview-early", "claude-fable-5", "gpt-5.6-sol_max"}
# Release dates of the "pretty" model names of Epoch's reference ECI table, which
# --eci-data-only fits instead of the pipeline's view. The view itself carries a date
# on every row the pipeline could date (release_dates.csv there); this map is only
# for that diagnostic mode.
REFERENCE_ECI_DATES: dict[str, str] = {
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
}


PPC_SEED      = 42
PIT_TIE_SEED  = 0
DENSITY_SEED  = 1

# ── Raw-C mode ─────────────────────────────────────────────────────────────
# When True, ECI = C (identity affine — no anchor rescaling). Set via the
# --raw-c CLI flag in 3_fit/fit.py. Default False keeps the affine anchor transform.
RAW_C_MODE = False
