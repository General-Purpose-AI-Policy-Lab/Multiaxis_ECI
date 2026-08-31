"""Configuration constants for the ECI Bayesian recreation."""
from __future__ import annotations

import math
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR     = PROJECT_ROOT / "data"
RESULTS_DIR  = PROJECT_ROOT / "results"
PLOTS_DIR    = PROJECT_ROOT / "plots"

# ── Zero-score diagnostic threshold ───────────────────────────────────────
# The level `ppc.py` scores `zero_pred_below_threshold` against: the posterior
# probability that the exact-zero rows replicate at or below 0.5%. Diagnostic
# only — it enters no likelihood.
ZERO_DIAG_THRESHOLD = 5e-3

# ── Epsilon bound ────────────────────────────────────────────────────────
# Beta likelihood has open support on (0, 1). `models/mirt.py` clips boundary
# scores onto [ECI_EPS, 1 - ECI_EPS]; the same epsilon lets fit.py's
# --drop-zero-scores identify rows on or outside (ECI_EPS, 1 - ECI_EPS).
ECI_EPS = 1e-3

# ── Low-observation flag ─────────────────────────────────────────────────
# Models with fewer than this many benchmark observations are flagged as
# data-poor: their posterior C is dominated by prior + a few extreme points
# and shouldn't appear in the headline timeline / forest.
LOW_OBS_THRESHOLD = 4

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
# IDs match `model_version` in data/processed/benchmarks_merged.csv.
# Where models expose a reasoning-effort suffix (_low/_medium/_high), we pick
# `_medium` as a default-ish operating point and keep the choice consistent
# across the SOTA list — anchors are arbitrary; differences are invariant.
ANCHOR_LOW  = ("claude-3-5-sonnet-20241022", 130.0)
ANCHOR_HIGH = ("gpt-5-2025-08-07_medium",    150.0)

# ── SOTA models (data-driven; newest first) ─────────────────────────────────
# NOT hand-maintained. `diagnostics/compute_sota.py` writes
# data/curated/sota_models.txt = (frontier envelope on overall 1D capability C,
# Epoch-style "highest C accessible at each date") ∪ (each flagship lineage's
# current leader), restricted to the recent era. Re-run that script after a
# data/fit refresh. Used for the is_sota plot exemption + drop-filter protection;
# the ECI ANCHORS above are separate and protected independently.
def _load_sota_models() -> list[str]:
    p = DATA_DIR / "curated" / "sota_models.txt"
    if p.exists():
        return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]
    return [ANCHOR_LOW[0], ANCHOR_HIGH[0]]      # fallback: at least protect the anchors


SOTA_MODELS: list[str] = _load_sota_models()

# Axes (0-based) where the frontier forecast does NOT grant SOTA the exemption
# from the SD < 0.4 informed filter. Axis 4 is the legacy knowledge/NLP axis:
# its defining benchmarks (OpenBookQA 0.89 axis share, ARC (AI2) 0.84,
# Adversarial NLI 0.82, BoolQ, CSQA2, BBH, SuperGLUE, HellaSwag, PIQA) have no
# observation on any model released after 2025-06, so every SOTA candidate
# there carries SD ~1.0 and its position is the lineage prior, not a
# measurement. Exempted, those ghosts hold the running max and suppress every
# measured record: the fit collapsed to 2 points 77 days apart and the slope
# flipped sign between the posterior mean and median. The other axes keep the
# exemption because their measured models already outrank the ghosts. Drop
# this entry once the axis has live coverage (BALROG, SimpleQA Verified and
# SimpleBench all load on it and are still being run).
FORECAST_NO_SOTA_AXES: set[int] = {3}

# The frontier-forecast fit shared by the dashboard, the memo and the LW post:
# records only, fit from the reasoning-model cutoff, SD<0.4 cloud, 50% HDIs.
# Each caller still supplies its own sota_exempt/back_start/horizon_date.
FORECAST_KW = dict(fit_basis="records", fit_start="2024-10-01", sd_cap=0.4,
                   hdi_prob=0.5)

# Display strings for the flagship's 4 axes, opt-in per fit (figure dict keys
# stay axis{k} so cache/anchor ids don't churn when a caller passes these).
AXIS_TITLES = {"axis1": "Axis 1 — Fluid Intelligence",
              "axis2": "Axis 2 — Scientific Knowledge and Reasoning",
              "axis3": "Axis 3 — Agentic",
              "axis4": "Axis 4 — Legacy QA"}

RELEASE_DATES: dict[str, str] = {
    # Pre-2023 models the merge leaves undated. Without these the era filter
    # cannot see them: it drops only models with a KNOWN date, so a 2019 T5 or a
    # 2020 GPT-3 slipped through every "post-2023" cut. Dates are the public
    # releases of the paper / API tier, not the eval run.
    "T5-Small":                      "2019-10-23",  # arXiv 1910.10683
    "T5-Base":                       "2019-10-23",
    "T5-Large":                      "2019-10-23",
    "T5-3B":                         "2019-10-23",
    "T5-11B":                        "2019-10-23",
    "ada":                           "2020-06-11",  # GPT-3 API base tiers
    "curie":                         "2020-06-11",
    "davinci":                       "2020-06-11",
    "text-ada-001":                  "2022-01-27",  # InstructGPT text-*-001 tier
    "text-babbage-001":              "2022-01-27",
    "text-curie-001":                "2022-01-27",
    "claude-mythos-preview-early":   "2026-06-09",  # early preview, no public date; placed with Fable 5
    "gpt-5.5-codex":                 "2026-04-23",  # SEAL-only rows carry no date; node_date from lineage_map.csv
    "gemini-3.1-pro-preview (high)": "2026-02-19",  # same release as the dated bare preview
    "claude-fable-5":                "2026-06-09",
    "claude-fable-5_high":           "2026-06-09",
    "claude-fable-5_max":            "2026-06-09",
    "claude-fable-5_xhigh":          "2026-06-09",
    "gemini-3-pro-preview":          "2025-11-18",
    "gpt-5-pro-2025-10-06":          "2025-10-06",
    "gpt-5-2025-08-07_medium":       "2025-08-07",
    "o3-pro-2025-06-10_medium":      "2025-06-10",
    "o3-pro-2025-06-10":             "2025-06-10",  # SEAL-only bare row carries no date

    "o3-2025-04-16_medium":          "2025-04-16",
    "gemini-2.5-pro-preview-03-25":  "2025-03-31",
    "o1-2024-12-17_medium":          "2024-12-17",
    "o1-mini-2024-09-12_medium":     "2024-09-12",
    "gemini-1.5-pro-002":            "2024-09-24",
    "gpt-4o-2024-05-13":             "2024-05-13",
    "gpt-4-turbo-2024-04-09":        "2024-04-09",
    "claude-3-opus-20240229":        "2024-02-29",
    "gpt-4-0314":                    "2023-03-14",

    # Undated-model backfill, 2026-07-30. Reviewed one at a time; every entry
    # has a source. Mostly SEAL-only audio/agent rows, which the scrape leaves
    # undated by design (its `createdAt` is the leaderboard-add date). Matters
    # for the --time-prior covariate, which otherwise parks these at the
    # population-mean era: harmless for the 2025-26 rows, wrong for the old ones.
    # Dates are the public model release, not the eval run.
    "Switch-Base":                   "2021-01-11",  # arXiv 2101.03961 v1 (paper only, no API release)
    "Switch-Large":                  "2021-01-11",
    "davinci-002":                   "2022-03-15",  # Epoch benchmarked_models.csv
    "gemma-2-9b":                    "2024-06-24",  # Epoch benchmarked_models.csv (blog says 06-27)
    "gpt-4o-mini-audio-preview-2024-12-17": "2024-12-17",  # vendor snapshot ID in the name
    "sonar":                         "2025-01-21",  # Epoch benchmarked_models.csv
    "Phi-4-multimodal-instruct":     "2025-02-26",  # Microsoft Azure "next generation of Phi" blog
    "Manus 1.0":                     "2025-03-05",  # public launch 2025-03-05/06
    "Qwen2.5-Omni-7B":               "2025-03-26",  # qwen.ai/blog?id=qwen2.5-omni
    "Kimi-Audio-7B-Instruct":        "2025-04-25",  # MoonshotAI/Kimi-Audio release + tech report
    "nova-premier":                  "2025-04-30",  # AWS GA announcement
    "gpt-4o-audio-preview-2025-06-03": "2025-06-03",  # vendor snapshot ID in the name
    "gemma-3n-E4B-it":               "2025-06-26",  # ai.google.dev/gemma/docs/releases
    "ChatGPT agent":                 "2025-07-17",  # OpenAI "Introducing ChatGPT agent"
    "Voxtral-Small-24B-2507":        "2025-07-15",  # mistral.ai/news/voxtral (2507 = YYMM)
    "gpt-realtime-2025-08-28":       "2025-08-28",  # vendor snapshot ID in the name
    "MiMo-Audio-7B-Instruct":        "2025-09-19",  # XiaomiMiMo/MiMo-Audio open-source release
    "MiMo-Audio-7B-Instruct (Thinking)": "2025-09-19",
    "Qwen3-Omni-30B-A3B-Instruct":   "2025-09-22",  # QwenLM/Qwen3-Omni News section
    "LFM2-Audio-1.5B":               "2025-10-01",  # liquid.ai/blog/lfm2-audio
    "Manus 1.5":                     "2025-10-16",  # manus.im/blog/manus-1.5-release
    "gemini-2.5-flash-native-audio-preview-12-2025 (thinking)": "2025-12-01",      # month-only from the name
    "gemini-2.5-flash-native-audio-preview-12-2025 (non-thinking)": "2025-12-01",
    "mimo-v2-flash":                 "2025-12-16",
    "gpt-realtime-mini-2025-12-15":  "2025-12-15",  # vendor snapshot ID in the name
    "Manus_1.6 (Max)":               "2025-12-15",  # manus.im/blog/manus-max-release
    "gpt-realtime-1.5":              "2026-02-23",
    "mercury-2":                     "2026-02-24",  # Inception Labs launch
    "gemini-3.1-flash-live-preview": "2026-03-26",
    "mimo-v2.5":                     "2026-04-22",
    "tml-interaction-small":         "2026-05-11",
    "gpt-realtime-2":                "2026-05-07",  # OpenAI Realtime API changelog
    "gpt-realtime-2 (xHigh)":        "2026-05-07",
    "Inkling (Thinking)":            "2026-07-15",  # Epoch benchmarked_models.csv
    "Inkling (xHigh)":               "2026-07-15",

    # Second backfill, 2026-08-04. Closes the 14 rows the merge still left
    # undated. Epoch's benchmarked_models.csv is the source unless noted; where
    # its `Version release date` is blank the group's `Publication date` is the
    # model's own first release (verified per row, not assumed).
    #
    # Six are bare SEAL/WeirdML spellings of a model the table already carries
    # under a versioned name (the scrape prints no snapshot ID). Dated rather
    # than aliased, matching o3-pro-2025-06-10 above: an alias would have to
    # guess which effort variant the board ran.
    "gpt-4o":                        "2024-05-13",  # SEAL bare row = the gpt-4o group's first release
    "o1-preview":                    "2024-09-12",  # SEAL bare row of o1-preview-2024-09-12
    "o4-mini":                       "2025-04-16",  # SEAL bare row of o4-mini-2025-04-16
    "o4-mini-2025-04-16":            "2025-04-16",  # versioned, but its only row is SEAL-only
    "glm-5p2":                       "2026-06-16",  # SEAL spelling of GLM-5.2
    "gpt-5.6 (sol)":                 "2026-07-09",  # SEAL spelling of GPT-5.6 Sol
    "gpt-5-chat":                    "2025-08-07",  # WeirdML row; shipped with GPT-5
    "codestral-2405":                "2024-05-29",  # Mistral Codestral launch (2405 = YYMM)
    "mimo-v2-pro":                   "2026-03-18",
    "mimo-v2-pro_high":              "2026-03-18",
    "mimo-v2.5-pro":                 "2026-04-22",  # Epoch says 04-23; matched to the
                                                    # reviewed mimo-v2.5 entry and the
                                                    # lineage_map v2.5-pro node
    "learnlm-1.5-pro-experimental":  "2024-11-19",  # AI Studio preview, first reported 2024-11-19
    "ml-elephant":                   "2024-01-04",  # Zoo (ex-KittyCAD) ML-ephant text-to-CAD launch
    # No public date for the 1.1 point release. Stamped with the Anthropic model
    # card that carries its scores, which bounds it above; the year is what the
    # era filter and the time covariate actually use.
    "claude-instant-1.1":            "2023-07-08",
}


PPC_SEED      = 42
PIT_TIE_SEED  = 0
DENSITY_SEED  = 1

# ── Raw-C mode ─────────────────────────────────────────────────────────────
# When True, ECI = C (identity affine — no anchor rescaling). Set via the
# --raw-c CLI flag in fit.py. Default False keeps the affine anchor transform.
RAW_C_MODE = False
