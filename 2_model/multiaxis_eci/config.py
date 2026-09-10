"""Configuration constants for the ECI Bayesian recreation."""
from __future__ import annotations

import math
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]   # 2_model/multiaxis_eci/config.py -> repo root
INPUT_DIR    = PROJECT_ROOT / "0_input"     # tables of benchmark-data-pipeline, copied by `python -m multiaxis_eci sync`
CURATED_DIR  = PROJECT_ROOT / "1_curated"   # inputs specific to this model (exclusions, lineages, SOTA list, clips)
PIPELINE_DIR = PROJECT_ROOT.parent / "benchmark-data-pipeline"
# name here -> path in the pipeline checkout
INPUT_FILES = {
    "all_scores_flat.csv": "3_views/all_scores_flat.csv",   # every score with release date, organization, category, floor, ceiling
    "human_baselines.csv": "3_views/human_baselines.csv",   # one row per published human measurement
    "models.csv":          "2_database/models.csv",         # identity, organization, country
    "benchmarks.csv":      "2_database/benchmarks.csv",     # category, floor, ceiling, access class
    "build_manifest.json": "2_database/build_manifest.json",
}
PROVENANCE_FILE = "provenance.json"


def data_tag() -> str:
    """Date of the synced pipeline build as YYYYMMDD, or '' before the first sync.

    Names the per-generation output folder (`5_outputs/data<tag>/`), so fits on two
    data generations never overwrite each other.
    """
    import json
    p = INPUT_DIR / PROVENANCE_FILE
    if not p.exists():
        return ""
    built = json.loads(p.read_text()).get("pipeline_built_at", "")
    return built[:10].replace("-", "")


DATA_TAG = data_tag()

# ── Output layout ──────────────────────────────────────────────────────────
# Everything a fit or a diagnostic writes goes under 5_outputs/<data generation>/:
#   <fit>/                 tables, trace.nc and figures/ (PNG) + figures/html/ per fit
#   comparisons/           cross-fit tables (country frontier, chain verdicts) + figures/
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

# ── Low-observation flag ─────────────────────────────────────────────────
# Models with fewer than this many benchmark observations are flagged as
# data-poor (ECIData.is_low_obs): their posterior is dominated by the prior and
# a few extreme points. Every fit keeps them; the MIRT axis timelines and the
# forecast candidates hide them (SOTA families exempt), the K=1 ECI-H timeline
# draws every dated model. Also the "4 benchmarks" rule of the SOTA list.
LOW_OBS_THRESHOLD = 4

# ── Informed-ability cap ────────────────────────────────────────────────────
# A model's ability on an axis counts as measured when its posterior SD is below
# this cap (analysis.timelines.mirt_informed_mask; history 0.6 -> 0.3 -> 0.4 ->
# 0.33 in its docstring). The one value behind the measured timelines, the
# forecast candidates (FORECAST_KW), the country frontier and the bimodality scan.
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
SAMPLE_KW = dict(
    draws=10000,
    tune=2000,
    chains=8,             # 15 physical cores — 8 chains run in parallel
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
# 1_curated/sota_families.txt = (world frontier records of the last 24 months on
# the canonical ECI-H, Epoch-style "highest capability accessible at each date")
# ∪ (every family within 10 ECI of the best model of each organisation on that
# frontier), candidates with >= 4 observations. A family is a release
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

# The SOTA exemption of the informed filter is not unconditional: a SOTA release
# is admitted on an axis with a wide interval only while its ability there is
# at least weakly measured, posterior SD below this cap. Above it the position
# is the lineage prior alone (SD ~1 on this scale): on the published fit's
# Legacy QA axis, whose benchmarks carry no observation on any model released
# after 2025-06, such prior-only frontier releases held the running max and
# suppressed every measured record (the fit collapsed to 2 points 77 days
# apart and the slope flipped sign between the posterior mean and median).
# The rule replaced an axis INDEX (`FORECAST_NO_SOTA_AXES = {3}`) on 2026-09-10:
# the index named the published fit's fourth axis and, applied to a refit whose
# fourth axis was another one, fitted a trend on 2 points under a cloud of 123.
SOTA_EXEMPT_SD_CAP = 0.8

# The frontier-forecast fit shared by the dashboard, the memo and the blog post:
# the per-draw running-max ENVELOPE over the informed cloud (non-decreasing by
# definition, so no draw can carry a negative trend — the record regression it
# replaces left a third of the Agentic axis's draws with negative slopes),
# extended forward at its recent rate; 80% HDIs. `fit_start` only matters to
# the regression bases kept for sensitivity runs (records/frontier/informed).
# Each caller still supplies its own sota_exempt/backcast_floor/horizon_date.
FORECAST_KW = dict(fit_basis="envelope", fit_start="2024-10-01", sd_cap=INFORMED_SD_CAP,
                   hdi_prob=0.8)

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
