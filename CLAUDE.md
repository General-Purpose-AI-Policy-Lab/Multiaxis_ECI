# ECI Bayesian Recreation — Beta-MIRT framework

PyMC recreation of the Bayesian Epoch Capabilities Index. One model framework,
the K-axis compensatory 2PL Beta-MIRT (`models/mirt.py`), serves two jobs:

- **Canonical index** (`python fit.py --preset canonical`): K=1 on the curated
  benchmark scope. Produces the anchored ECI scale, SOTA table, forests,
  timeline → `results/canonical/`.
- **Exploration** (`python fit.py --K 3 ...`): K-axis capability decomposition
  on the full benchmark set → `results/mirt{tag}/`.

`data/curated/excluded_benchmarks.txt` (ARC-AGI and other benchmarks that are
easy for humans) is applied at fit time by `data.py`, and only to the canonical
scope. Exploration loads `include_all_benchmarks=True` because the
multidimensional structure is partly defined by those benchmarks. The processed
file always holds every benchmark; exclusion is a per-fit modeling choice.

## Data pipeline

```
data/
├── pipeline/                          # refresh notebook (authoritative)
│   ├── pipeline.ipynb                     # Restart Kernel → Run All
│   ├── canonical/                         # tracked name/alias maps
│   ├── snapshots/                         # gitignored; one folder per fetch date
│   ├── intermediate/                      # gitignored; per-stage CSVs
│   └── output/                            # section 10 swaps these into the tree
│       ├── benchmarks_merged.csv
│       ├── human_baselines.csv
│       └── pipeline_report.md             # row counts, sha256, score diffs
├── processed/benchmarks_merged.csv    # what data.py loads at runtime
├── curated/
│   ├── excluded_benchmarks.txt            # drop list, applied at fit time
│   ├── human_baselines.csv                # 9 human tiers as test-takers
│   ├── sota_models.txt                    # written by diagnostics/compute_sota.py
│   ├── lineage_map.csv                    # vendor release chains — GENERATED,
│   │                                      #   rebuild with
│   │                                      #   diagnostics/build_lineage_map.py
│   │                                      #   after any data refresh; never hand-edit
│   ├── lineage_node_overrides.csv         # reviewed node/parent re-filings, the
│   │                                      #   tracked overlay a rebuild preserves
│   ├── benchmark_lower_bounds.csv         # chance floors c_b   (--floors)
│   ├── benchmark_upper_bounds.csv         # score ceilings d_b  (--ceilings)
│   ├── benchmark_score_clips.csv          # reviewed per-cell score clips
│   ├── benchmark_n_items.csv              # item counts + verification stamps
│   │                                      #   (--item-counts reads VERIFIED rows)
│   ├── cyber_benchmarks.csv               # cyber ECI rows      (--cyber)
│   ├── simpleqa_original/                 # original OpenAI SimpleQA (--simpleqa-original)
│   ├── row_drops_*.csv, row_fixes_*.csv   # curated anti-joins / corrections
│   └── row_adds_*.csv                     # locally measured cells, appended
│                                          #   after the drops; a collision with
│                                          #   a feed raises rather than dedups
└── raw/eci_data.csv                   # reference set (--eci-data-only)
```

Columns of the processed file: `model_version, score, release_date,
organization, benchmark, stderr, source, category`.

**Refresh**: open `pipeline.ipynb`, Restart Kernel → Run All (section 10 swaps
outputs into `data/{processed,curated}/`); read `output/pipeline_report.md`
(snapshot date, sha256, top score deltas, missing map entries, unrecognized
files, NaN-dropped rows); revert with `git checkout data/processed
data/curated`; re-run the canonical fit. Same-UTC-day re-runs reuse today's
snapshot folder, so delete `snapshots/YYYY-MM-DD/` to force a fresh fetch. Full
docs in [`data/pipeline/README.md`](data/pipeline/README.md).

**Five feeds merged into one long table:**

1. **[Epoch AI benchmark ZIP](https://epoch.ai/data/benchmark_data.zip)** —
   CC-BY, ~weekly. Per-benchmark CSVs mapped by `FILE_SPEC` to
   `(benchmark, score column, divisor)`. The bulk of the table. `FILE_SPEC` is
   an **allowlist**, so an unmapped upstream file is silently absent from the
   fit; the report lists unrecognized files for that reason. A file in both
   `FILE_SPEC` and `FILE_SKIP` raises, because skip is checked first and the
   spec line would be dead. `FILE_SCORE_TRANSFORM` rescales non-proportion
   columns (AlgoTune `1 - 1/speedup`). Vending-Bench 2 is in `FILE_SKIP`: only
   three of its takers land in the informative score range, so its taker panel
   cannot identify the loading row, and its dollar-balance score needed a
   transform onto a scale no other benchmark shares.
2. **[Epoch AI live feed](https://epoch.ai/data/benchmarks.csv)** — CC-BY, the
   table the epoch.ai benchmark pages read. All Epoch internal evals come from
   here (section 03d); the ZIP supplies the external benchmarks. The overlap is
   byte-identical, so the routing costs no rows. The feed stamps every run with
   `task` + `task version`, which the ZIP does not carry, so a specific
   problem-set version can be selected. `LIVE_TASK_SKIP` reports genuinely new
   tasks. FrontierMath v1 and v2 are **separate benchmarks, never pooled**: v2
   shifted the level about +0.20 on Tiers 1-3 with the ranking intact (paired
   r = 0.988), so one column would carry a bimodal difficulty.
3. **[RAND RR-A3797-1](https://www.rand.org/pubs/research_reports/RRA3797-1.html)**
   — bio/chem PhD benchmarks digitised from the report (WMDP, LAB-Bench,
   MMLU(Pro) Bio/Chem, GPQA Bio/Chem, BioLP-bench). RAND names map to Epoch's
   versioned IDs via `canonical/model_aliases.csv`; three RAND fine-tunes are
   dropped rather than aliased (pipeline README).
4. **[Scale SEAL leaderboards](https://scale.com/leaderboard)** — fourteen
   boards (MCP Atlas, MultiChallenge, MultiNRC, VISTA, TutorBench, SEAL Tool
   Use, SWE-Bench Pro public/private, SEAL Instruction Following, PRBench
   Finance/Legal, Remote Labor Index, VisualToolBench, AudioMultiChallenge).
   No public API: scraped from each page's Next.js `__next_f` payload, since
   the visible HTML table is a stale snapshot. Per-chunk `json.loads` keeps
   multi-byte UTF-8 intact (plain `unicode_escape` mangles the `†` marker).
   Raw HTML cached in `snapshots/<date>/seal/`. "Thinking"-mode variants are
   distinct test-takers, matching the effort-variant convention. Prefer the ZIP
   for any board Epoch publishes, except where the overlap is partial: `rli`
   stays scraped because Epoch's file covers 10 models against the board's 15
   (union 17).
5. **[Kaggle Open Benchmarks](https://www.kaggle.com/api/v1/benchmarks/<owner>/<slug>/leaderboard)**
   — public JSON, no auth. The first INDEPENDENTLY RE-RUN source: the other
   four feeds are vendor self-reports or leaderboard scrapes, so this is the
   first to supply real per-row stderr at scale (`--known-se` currently has
   stderr on roughly a fifth of cells). Three boards are ingested, merged per
   board by one `kaggle_wins` boolean: `open-benchmarks/mmlu` → MMLU,
   kaggle_wins=True (51 models; no offset against the ZIP column, p = 0.25 on
   7 overlapping models, so Kaggle wins collisions and the ZIP's 136 rows
   backfill the rest for 180 obs); `open-benchmarks/mmlu-pro` → MMLU-Pro,
   kaggle_wins=True (55 models; the sole source for this column, because Kaggle
   agrees with TIGER-Lab's own runs but not with the vendor-submitted rows on
   TIGER-Lab's leaderboard, p = 0.014 on the latter; chance floor 0.10);
   `deepmind/simpleqa-verified` → SimpleQA
   Verified, kaggle_wins=False (46 models; the Epoch-live-feed column has more
   models, one source and the effort tags Kaggle lacks, so it wins every
   collision). Kaggle publishes no effort setting, so its rows are filed
   `_unknown` and an untagged name never collides with a tagged one: 38 of the
   46 enter, 23 on models the column lacks and 15 as an unknown-effort
   test-taker beside an existing tagged row for the same model. That is the
   effort-variant convention applied consistently, and it is why the two can
   sit 14 points apart (`gpt-5.4-2026-03-05_xhigh` 0.448, the untagged row
   0.305). A board's identity is verified from its own confidence
   interval, not its documentation: `n_eff = p(1-p)/(ci/1.96)^2` recovers the
   item count (MMLU 14,043 against a true 14,042; MMLU-Pro 12,032 exact;
   SimpleQA Verified ~938 against 1,000), which also caught Kaggle's MMLU-Pro
   running chain-of-thought (a CoT/direct mismatch reads ~20 points; the
   observed gap against TIGER-Lab is 0.7). The same check rejected
   `bobfraserg/gsm8k` (personal account, CI of 0), `open-benchmarks/aime-2025`
   (CI of 0 on a 30-item exam) and `open-benchmarks/math-500` (CI implies
   8,000 items for 500) — org-owned boards only, never personal accounts.
   SciCode was examined and not ingested: its Subproblem CIs imply all 288
   subproblems are independent, but they nest in 65 problems with forwarded
   solutions (`benchmark_n_items.csv`), so Kaggle's stderr would run ~2.1x too
   tight; SciCode stays sourced from the Epoch ZIP, corroborated by Kaggle's
   39-model overlap (offset -0.027, corr 0.894). Full merge and guard detail
   in the pipeline README.

**Steps**: fetch → parse Epoch CSVs → merge RAND → scrape SEAL → read internal
evals from the live feed → merge Kaggle boards → canonicalise names → dedup
(exact repeats removed; disagreeing
duplicates on `(model, benchmark)` resolved by `DEDUP_POLICY`, default
`"max"`) → attach category metadata → write outputs. The pipeline emits
every benchmark (`APPLY_CURATED_EXCLUSIONS=False`).

**Name review queue**: `output/missing_models.csv` lists models whose RESOLVED
identity is in neither `canonical/model_aliases.csv` nor
`canonical/reviewed_models.txt` — resolved meaning after the trailing `*`/`†`
footnote-marker strip and after alias mapping. Keying it on the raw string
instead re-listed every marker variant on every refresh (37 of 151 entries on
2026-08-04) and buried the genuinely new releases. A name in the queue is a
decision, not a defect: alias it if it is another spelling of a known
test-taker, add it to `reviewed_models.txt` if it is a real new model, or give
it an empty canonical to drop it.

**Score scale**: "normalised" upstream means [0,1] by unit conversion
(percent→proportion, 0-10→0-1), **not** chance-corrected: a 4-choice MC
benchmark still clusters near 0.25. Chance floors live in
`benchmark_lower_bounds.csv`, read only by `--floors`. `stderr` is used only by
`--known-se`.

### Load-time flags (`load_eci_data`)

Defaults give the canonical broad-index configuration.

| flag | default | effect when flipped |
|---|---|---|
| `fit_humans` | `True` | append `human_baselines.csv` as IRT test-takers |
| `include_all_benchmarks` | `False` | keep the excluded benchmarks (exploration default) |
| `drop_low_obs_models` | `False` | drop models with `< LOW_OBS_THRESHOLD` obs (SOTA, anchors, humans protected) |
| `collapse_effort_variants` | `False` | merge `_low`/`_medium`/`_high` into one test-taker |
| `fit_cyber` | `False` | append the cyber ECI benchmarks |
| `fit_simpleqa_original` | `False` | append the original OpenAI SimpleQA (separate column from SimpleQA Verified: different set and grader) |
| `drop_benchmarks` | `None` | drop the named benchmarks |
| `eci_data_only` | `False` | bypass the processed file, fit `data/raw/eci_data.csv` |
| `min_release_date` / `max_release_date` | `None` | era filter on models with a **known** date (undated kept, humans unaffected) |

## Statistical model

Compensatory K-axis 2PL Beta-MIRT (`models/mirt.py`). For observation $n$ on
model $m_n$ and benchmark $b_n$:

$$\text{score}_n \sim \text{Beta}\big(\mu_n\phi_{b_n},\; (1-\mu_n)\phi_{b_n}\big),\quad
  \mu_n = \sigma\Big(\textstyle\sum_k A_{b_n k}\,\theta_{m_n k} - D_{b_n}\Big),\quad
  \phi_b = \tfrac{1}{4\sigma_b^2}-1$$

$\theta$ abilities, $A$ loadings, $D$ difficulty, $\sigma_b$ noise. Variance is
$4\sigma_b^2\mu_n(1-\mu_n)$, so $\sigma_b$ is the SD at the maximum-variance
point $\mu=\tfrac12$. Validity needs $\sigma_b<\tfrac12$; the prior keeps it
well below. K=1 is the canonical index: $\mu = \sigma(A_b\theta_m - D_b)$, the
2PL with $A_b$ as discrimination.

### Priors

Non-centered throughout: NUTS samples a unit-scale `*_z` and a
`pm.Deterministic` multiplies by the relevant `tau`, breaking the funnel
between each scale and its vector.

| Variable | Prior | Derived | Why |
|---|---|---|---|
| `sigma_b` | $\text{LogNormal}(\log 0.05,\,0.5)$ per benchmark | $\phi_b=\tfrac1{4\sigma_b^2}-1$ | median 0.05, 90% interval $\approx[0.022,0.114]$; LogNormal forces positivity |
| `tau_CD` | $\text{LogNormal}(\log 3,\,1)$ | scales $D$ | keeps most $D$ within $\pm 6$ logits |
| `theta` | $\text{ZeroSumNormal}(1)$ per axis over models | — | unit scale; sum-to-zero pins the location inside the geometry, no flat direction and no post-hoc recentering |
| `D_z` | $\mathcal N(0,1)$ per benchmark | $D=D_z\tau_{CD}$ | location already pinned on the theta side |
| `tau_A` | per loading prior, below | scales $A$ | fixes the $A/\theta$ scale trade-off (typical loadings ≈ 0.5) |

**Loading priors** (`--loading-prior`, default `signed`):

- **`signed`** — iid Normal cells × one shared scale. Exactly
  rotation-invariant, so the sampler wanders the orbit freely and axes are
  identified per draw after sampling (`analysis.align_rotations`). The only
  prior that can represent a **contrast** axis; forcing a contrast non-negative
  splits it into two highly correlated positive axes.
- **`normal`** — $A_{bk}=\tilde A_{bk}\tau_A$ with
  $\tilde A_{bk}\sim\text{HalfNormal}(1)$ and one shared
  $\tau_A\sim\text{LogNormal}(\log 0.5,0.5)$. Non-negative, so each axis is a
  "good-at-these" bundle. The canonical fit uses this.
- **`signedhs`** — signed cells × per-cell regularized horseshoe (shared global
  scale, Inv-Gamma slab). Sparsity picks the rotation inside the sampler.
- **`bifactor`** (`--K >= 2`) — axis 1 is a dense non-negative general column
  with its own $\tau_g$; axes 2..K are non-negative specifics under a per-cell
  regularized horseshoe with **one** global scale shared across all of them.
  No benchmark is assigned anywhere: the prior constrains the loading pattern's
  shape, not its content. Identification comes from the escape fee. A horseshoe
  cell is either squeezed to ~0 or escapes roughly unshrunk at near-constant
  cost, so among the configurations the likelihood cannot separate, total cost
  scales with the number of escaped cells. Shared variance parked in a specific
  must buy an escape on nearly every benchmark; in the exempt general column it
  is free. A specific the data do not need collapses to an empty column, which
  is a built-in dimensionality check. Axis identity is structural, so nothing
  downstream rotates; only the cross-chain permutation of specifics is matched
  (`permutation_matched_reproducibility`). Axis names go to `mirt_axis_names`.

**Structural options** (compose with any loading prior):

- `human_order` — hard partial order on human tiers (`config.HUMAN_ORDER`).
  Each tier's ability is its parent's plus a positive increment on every axis,
  so with $A\ge0$ the score ordering holds on every benchmark. Branches are
  deliberately incomparable. A tuple of parents
  (`config.HUMAN_ORDER_MERGED`, `--human-merge`) states dominance over ALL of
  them, $\theta = \max(\text{parents}) + \delta$, which merges the High School
  branch into the adult spine at two points: Domain Expert over
  max(Skilled Generalist, High School Qualifier) and Top Performer over
  max(Domain Expert, High School Top Performer). The DAG is unrolled into
  root→tier PATHS and the max taken over a tier's paths, exact because
  $\max(a,b)+\delta=\max(a+\delta,b+\delta)$, so the increment stays per TIER
  and the merge adds no parameters (6 either way on the live scope: 8 paths →
  12). The cost is the kink: the gradient jumps where two parents of a merged
  tier cross, so read its divergence count against the tree fit.
- `lineage` — soft vendor release-chain prior (`lineage.py`). Within a chain,
  `psi[node] = founder + Σ Normal(mu>0, s)`: improvement is the mean step but a
  node can regress. Effort variants get tight mean-zero offsets sharing one sd
  per axis. A chain is a TREE, not only a line: the map's `parent` column names a
  predecessor explicitly, so a side product line hangs off its branch point
  instead of splicing into the spine and claiming a near-zero gap to whichever
  releases bracket it by calendar. Branching is for a line that continues from
  its branch point (Fable off Opus 4.8, ChatGPT-4o off 4o 2024-11-20). A
  single-node chain survives when its node keeps >=2 (node, variant) groups:
  zero Brownian deltas, variant offsets only, so a first-release vendor's
  effort rungs still tie to one ability (Inkling; Inkling Small with its
  transcribed ARC ladder). Only a lone bare release drops — one row, nothing
  to tie. Worked example, the gpt 5.6 codenames:
  sol / terra / luna are peer model TYPES, not a tier ladder. `CODENAME_CHAIN`
  in `build_lineage_map.py` keeps sol on the flagship spine (+0.027 over 5.5,
  above on 20/28 effort-matched cells); terra and luna branch off the 5.5 node
  as parented off-spine siblings, each taking one Brownian delta from 5.5 with
  its effort variants (including the effort:unknown leaderboard rows) tied
  through the variant offsets. Both sit BELOW 5.5 (−0.067 on 3/27, −0.198 on
  0/27), so the positive-mean delta pays a soft prior cost the data can
  overrule; the branch was chosen because founding them as single-node chains
  left every variant free, which the K=3 mode diagnosis identified as a driver
  of the axis-3 bimodality. A 5.7-terra would parent onto `terra 5.6`,
  continuing the branch. Sol's `_pro<effort>` rows are the effort parser
  swallowing a Pro-MODE tag (upstream "GPT 5.6 Sol (Max + Pro)"), an advanced
  harness rather than a release, and are excluded by `OUT_HARNESS`.
  Live: **48 chains / 202 nodes / 154 deltas**. The map's `in_chain=yes` set
  equals the live set: a rebuild demotes any chain the fit would drop (single
  node, single variant group — lone releases like `mimo-flash`), so the map
  never lists a chain that carries no prior.
- `lineage_bm` — index the chain by TIME instead of by release. Over a gap of
  `dt` years the step is `Normal(drift·dt, s²·dt)`, Brownian motion with drift
  observed at the release dates. Mean grows in `dt`, sd in `√dt`, so the prior
  stops depending on how many intermediate releases the map happens to list
  (two half-steps compose to one whole step) and a rate in logits/year is
  comparable across vendors shipping at different cadences. A per-release drift
  is not: it equals rate/cadence, and with no clock in the model the two are
  not separable. The drift is **one shared positive value per axis**. Live
  median gap 0.238 yr; `PRIOR_LINEAGE_{DRIFT,DELTA}_BM` are set so a typical
  step reduces to the per-release scales.
- `time_t` — the theta prior MEAN becomes a per-axis line in centered release
  year, $\theta_{mk}=\beta_k(t_m-\bar t)+\text{ZSN}$, with
  $\beta_k\sim\mathcal N(0, 0.5)$. A thin model is shrunk toward its ERA's
  level rather than the whole population's. $\beta$ is signed and centered at
  zero, so a flat population reduces to the plain prior and no trend is
  imposed. $t$ is centered so the shifts sum to zero and cannot move the
  location the ZeroSumNormal pins. Chained rows carry their chain FOUNDER's
  date, so the trend shifts a chain's base level only and cancels in every
  within-chain difference. Undated models and humans get 0. This is a shrinkage
  target evaluated at observed dates, not a forecast; `analysis/forecast.py`
  extrapolates fitted trajectories instead. If theta residuals bend against
  release date, replace the covariate construction in place (fixed-knot
  piecewise, or era bins with a random-walk prior) rather than adding a second
  flag.
- `theta_t_cells` — per-CELL leptokurtic theta. iid
  $\theta^{raw}_{mk}\sim t_\nu$ drawn with the closed-form density, $\nu=4$
  fixed (`PRIOR_THETA_T_NU`), then re-centered per axis
  ($\theta_{mk}=\theta^{raw}_{mk}-\overline{\theta^{raw}_{\cdot k}}$), which
  restores the exact sum-to-zero location pin the ZeroSumNormal gives the
  Gaussian block. The third identification channel for the rotation, next to
  loading-side sparsity and theta-side structure: the Gaussian block is
  spherical, so with signed loadings prior AND likelihood are identical along
  the whole rotation orbit, while independent non-Gaussian columns pin the
  mixing matrix up to permutation and sign (ICA; Comon 1994, Bonhomme-Robin
  2009, and the leptokurtic condition varimax implicitly tests, Rohe-Zeng
  arXiv:2004.05387). Direct density, NOT the
  $\lambda_{mk}z_{mk}$/inverse-Gamma scale mixture: the mixture has the same
  marginal but its `n*K` latent scales each form a funnel with their
  coordinate, which is where the theta-t divergences came from. Must be per
  CELL: one scale per model is elliptical, and a rotation of an elliptical
  distribution is the same distribution, so a per-row shape buys nothing.
  $\nu$ is fixed because it is the quantity the rotation is identified BY, and
  an estimated $\nu$ can buy Gaussianity back. Covers the unstructured rows
  plus the human-root / chain-founder slices, never the ordered increments,
  which are already non-Gaussian. Centering is linear, so the columns stay
  independent and leptokurtic. Widens thinly-measured abilities (prior sd 1.41
  vs 1.00), so read it with the coverage limit.
- `theta_pos` — positive likelihood-side ability, the semi-compensatory
  convention (`models/mirt_interaction.py`):
  $\eta = \sum_k A_{bk}\,\mathrm{softplus}(\theta_{mk}) - D_b$. Raw $\theta$
  keeps every prior block (ZeroSumNormal pin, human/lineage order) and stays
  the reported ability; the positive copy is a `theta_pos` Deterministic. With
  $A\ge0$ an axis can only add to a score, never pull it below the
  $\sigma(-D_b)$ baseline. softplus is monotone, so rankings and raw-space
  order constraints survive; the location becomes likelihood-identified (the
  nonlinearity stops $D$ absorbing a constant shift). With the signed family
  the elementwise nonlinearity breaks rotation/reflection invariance, so
  post-hoc alignment degrades to permutation matching (warned at build).
- `shared_base_zsn` (default `True`) — human roots and chain founders are
  sliced out of the *same* per-axis `ZeroSumNormal` as the unstructured models,
  so every starting point is drawn from one population and the sum-to-zero
  spans all of them. That is what pins the otherwise-free overall location,
  since $\theta$ reaches the likelihood only through $A\theta-D$. Marginals are
  unchanged (a ZeroSumNormal entry has sd $\sqrt{1-1/n}\approx1$ at these
  sizes). `--private-bases` gives each base a private $\mathcal N(0,1)$
  instead. The initial-point logp is identical either way, so the golden-logp
  locks cannot see this; `test_structured_bases_share_one_zerosumnormal`
  guards it.
- `plt_founders` — positive-lower-triangular rotation identification for the
  signed family (`config.PLT_FOUNDERS`). Founder $r$ loads only axes $1..r$
  with a positive diagonal. The founder list is part of the model
  specification, not a tuning knob.
- `pin_benchmark` — difficulty sea-level anchor.

### Boundary clipping

Beta has open support on $(0,1)$. The 84 boundary rows (80 exact-0, 4 exact-1
on the full scope) are clipped onto $[\varepsilon,1-\varepsilon]$ with
$\varepsilon=10^{-3}$ (`ECI_EPS`). Interior scores are untouched.

```python
clipped = np.clip(data.scores, ECI_EPS, 1.0 - ECI_EPS)
pm.Beta("obs", alpha=mu*phi, beta=(1-mu)*phi, observed=clipped)
```

Rejected: Beta-CDF left-censoring (`pm.logcdf` / `pm.Censored`), whose gradient
path triggers max-tree-depth and unrecoverable divergences; Smithson-Verkuilen
$y^*=(y(N-1)+0.5)/N$, which nudges *every* score rather than the boundaries.
`ZERO_DIAG_THRESHOLD` and `data.zero_score_mask` feed the `ppc.py` calibration
diagnostic `zero_pred_below_threshold`.

### ECI scale

Per draw, $\text{ECI}=a+b\theta$, pinned by
$\text{ECI}_{\text{Claude 3.5 Sonnet (2024-10-22)}}=130$ and
$\text{ECI}_{\text{GPT-5 (2025-08-07, medium)}}=150$. The choice is arbitrary
and differences are invariant; these match Epoch's public dashboard for direct
comparability. Anchor IDs in `config.py`; both must exist in the fitted data or
`eci_transform` raises.

## Project structure

```
config.py       # paths, priors, sampling, SOTA list, ECI anchors, release dates
data.py         # ECIData dataclass + load_eci_data()
lineage.py      # release-chain structure for the lineage prior
persistence.py  # save/load trace, CSV, JSON, PIT
ppc.py          # posterior predictive per family + PIT + GoF
fit.py          # the fit CLI
models/
├── mirt.py             # build_mirt_model — the compensatory family
├── mirt_nc.py          # conjunctive product link
├── mirt_sparse.py      # sparse-gate conjunctive (horseshoe gates)
├── mirt_interaction.py # semi-compensatory: + pairwise ability interactions
└── qmatrix.py          # category → axis maps
analysis/
├── stats.py            # capability_draws, eci_transform, SOTA/forest/timeline
├── rotation.py         # per-draw alignment, varimax/promax/geomin
├── convergence.py      # identified r-hat and ESS per family
├── factors.py          # trace introspection, loading/score tables
└── timelines.py  forecast.py  fitview.py
viz/            # Plotly builders: core, gof, forecast, mirt, compare, dashboard
fits/           # drivers for the non-compensatory families
diagnostics/    # build_dashboard, plot_mirt, align_mirt, forecast_mirt,
                # forecast_only (re-render one card's forecast figures),
                # compute_sota, loo_compare, residual_corr, notebooks
tests/          # test_pipeline.py + test_safety_net.py (golden logp locks)
results/        # canonical/ · mirt{tag}/ · comparisons/
plots/          # gitignored figure output
index.html      # the all-fits dashboard (tracked)
```

## Fit CLI

```bash
python fit.py --preset canonical                     # the headline K=1 pipeline
python fit.py --K 3 --loading-prior normal --human-prior --lineage-prior --floors
python fit.py --K 3 --loading-prior bifactor --human-prior --lineage-prior --floors
```

Sampling: `--draws --tune --chains --target-accept --sampler` (default
`nutpie`). Canonical defaults come from `config.SAMPLE_KW` (10,000 draws ×
8 chains); exploration defaults to 2,000 draws. On a long run add
`--stream-draws`: nutpie then writes every draw into
`results/<fit>/live_draws.zarr` as it lands, so a kill or an OOM costs the
unfinished tail instead of the run, and `persistence.load_live_draws` reads the
store while sampling continues (it trims the NaN tail to the prefix every chain
has finished). Costs 200 kB per draw of disk — 27 GB for 8 chains x (7,000 tune
+ 10,000 draws) — because nutpie's store writes the warmup draws whatever
`save_warmup` says. Those are dropped from the returned trace, never from disk. Exploration model defaults are
`--K 4 --loading-prior signed`.

| flag | scope | effect |
|---|---|---|
| `--preset canonical` | — | K=1, normal prior, curated exclusions, full ECI deliverables |
| `--skip-sampling` | canonical | reuse `results/canonical/trace.nc` (must match the current data shape) |
| `--raw-c` | canonical | report raw theta instead of anchored ECI |
| `--eci-data-only` | canonical | fit `data/raw/eci_data.csv` |
| `--drop-zero-scores` | canonical | drop `score == 0` rows (diagnostic) |
| `--include-all-benchmarks` | canonical | keep the excluded benchmarks |
| `--post-2023` | canonical | era sensitivity: drop known dates before 2024-01-01 |
| `--K` | exploration | latent dimension |
| `--loading-prior` | exploration | `signed` / `normal` / `signedhs` / `bifactor` |
| `--human-prior` | exploration | order human tiers by `config.HUMAN_ORDER` |
| `--stream-draws` | exploration | write each draw to `results/<fit>/live_draws.zarr` as it is sampled; read a partial store with `persistence.load_live_draws` |
| `--human-merge` | exploration | instead `config.HUMAN_ORDER_MERGED`: the High School branch merged into the adult spine via a max over parents |
| `--lineage-prior` | exploration | soft release-chain prior |
| `--lineage-bm` | exploration | with `--lineage-prior`: Brownian steps scaled by the release gap |
| `--time-prior` | exploration | per-axis linear trend in release year on the theta prior mean |
| `--theta-t` | exploration | per-cell Student-t(4) theta block (ICA-style rotation identification) |
| `--theta-pos` | exploration | eta reads softplus(theta) instead of theta (semi-compensatory convention); raw theta stays the reported ability |
| `--private-bases` | exploration | private base per human root and chain founder |
| `--floors` | exploration | fixed-c 3PL, floors from `benchmark_lower_bounds.csv` |
| `--ceilings` | exploration | fixed-d asymptote from `benchmark_upper_bounds.csv`; with `--floors` this is the fixed 4PL |
| `--ceiling-noise` | exploration | estimate a noise-sized gap `Beta(1,20)` on top of each wall |
| `--known-se` | exploration | split Beta noise into fixed instrument precision + estimated excess |
| `--item-counts` | exploration | with `--known-se`: no-stderr cells get the floor `n_eff = n_items` from the VERIFIED rows of `benchmark_n_items.csv` (machine rows only; multi-seed runs conservatively read as single-run) |
| `--pooled-noise` | exploration | hierarchical `sigma_b`; thin benchmarks shrink to the shared median |
| `--cyber` | exploration | append the cyber ECI benchmarks |
| `--simpleqa-original` | exploration | append the original OpenAI SimpleQA (2023-2024 era rows; own column, never pooled with SimpleQA Verified) |
| `--drop-benchmarks A,B` | exploration | drop named benchmarks (sensitivity runs) |
| `--no-sg` / `--no-sg-gpqa` / `--no-sg-arcagi` | exploration | drop Skilled-Generalist cells |
| `--apply-exclusions` | exploration | fit the canonical scope |
| `--min-release-date` | exploration | drop models with a known earlier date |
| `--rotation` | exploration | display frame: `promax` (default) / `nonneg` / `none` |
| `--skip-baseline` / `--refit-baseline` | exploration | control the K=1 baseline fit |
| `--plots` | exploration | render figures via `diagnostics/plot_mirt.py` |

**`--floors`** clips each observed score up to its benchmark chance floor and
sets `μ = c_b + (1−c_b)·σ(η)`, with `c_b` read from file and never estimated,
so no parameters are added. Below-chance scores then read as uninformative-low
ability instead of point demands. `floor_c=None` is byte-identical to the 2PL.
PPC and GoF are floor-aware. The interaction family keeps no floor (below-chance
is signal there); the product links have no single η to floor.

**`--ceilings`** is the mirror: `μ = c_b + (d_b−c_b)·σ(η)`, `d_b` fixed per
benchmark, benchmarks absent from the file keeping an inert d = 1. A freely
estimated wall is weakly identified on this data, which is why d is fixed and
`--ceiling-noise` is confined to a noise-sized gap. In-sample fit is unchanged;
the effect is on the latent scale, where saturation reads as ceiling rather
than hardness, so frontier score gaps map to larger ability gaps and forecasts
asymptote at d_b instead of drifting toward 1.

**`--known-se`** uses the fact that φ+1 is an effective test length
(Var = μ(1−μ)/(1+φ) is the variance of an average of 1+φ solve/fail items). A
reported harness stderr converts to the same unit as `n_eff = p(1−p)/se²` and
the relative variances add: `1/N_tot = 1/n_eff + 4σ_b²`. σ_b then measures
construct misfit only, not total scatter. Roughly a fifth of cells carry a
stderr; the rest get `n_eff = inf` and are unchanged. On short tests the
instrument dominates the noise budget, on long ones it is negligible.

**Trace tags.** The results folder and trace file are named by the flag set:
`trace_mirt_k{K}{_signed|_signedhs|_bifactor}{_humanprior|_humanmerge}{_lineageprior}{_lineagebm}{_timeprior}{_thetat}{_thetapos}{_since<year>}{_noSG*}{_excluded}{_cyber}{_sqaorig}{_drop<names>}{_privbase}{_floors}{_ceilings}{_ceilnoise}{_knownse}{_itemcounts}{_poolednoise}.nc`.
`--loading-prior normal` contributes no tag. Options not exposed on the CLI
(`plt_founders`, `pin_benchmark`) are reached by calling
`models.mirt.build_mirt_model(...)` directly; the non-compensatory, sparse-gate
and interaction families run via `python -m fits.fit_nc` and siblings.

## Current scope

| scope | obs | test-takers | benchmarks |
|---|---|---|---|
| canonical (curated exclusions) | 4,271 | 777 | 92 |
| exploration (all benchmarks) | 4,998 | 821 | 100 |
| exploration + `--cyber` | 5,086 | 821 | 109 |

`results/canonical/trace.nc` predates the current data snapshot. Re-fit before
quoting any canonical number.

## Known limits

**Coverage / extrapolation (applies to every fit).** An ability on an axis is
trustworthy only if the test-taker was evaluated on benchmarks that *load* on
that axis. Models from 2021-2023 took only easy benchmarks, so their hard-axis
ability is extrapolated rather than measured and can land misleadingly high
with a wide CI. This is confounded with era: well-pinned models (SD<0.3) took
benchmarks of mean difficulty +0.74, poorly-pinned ones +0.09. It is a
missing-data problem and more sampling does not fix it.

- Mitigation is plot-side only: `mirt_informed_mask` (SD<0.4) drops these from
  figures, never from the fit or the diagnostics.
- **SOTA exception**: models in `config.SOTA_MODELS` (data-driven, via
  `diagnostics/compute_sota.py`) are exempt from that drop. A frontier release
  is always shown, with its wide CI drawn honestly rather than hidden.
  `_release_dates` backfills missing dates from `config.RELEASE_DATES` so a
  dateless preview still lands on the x-axis.
- **The forecast takes that exemption back on a stale axis.** Axes listed in
  `config.FORECAST_NO_SOTA_AXES` are fit with `sota_exempt=False`. Axis 4 is
  there because its defining benchmarks (OpenBookQA 0.89 axis share,
  ARC (AI2) 0.84, Adversarial NLI 0.82, BoolQ, CSQA2, BBH, SuperGLUE,
  HellaSwag, PIQA) carry no observation on any model released after 2025-06:
  every SOTA candidate sits at SD ~1.0 on the lineage prior alone, and
  exempted those points hold the running max and suppress every measured
  record (the record set collapsed to 2 points 77 days apart and the slope
  flipped sign between the posterior mean and median). The other axes keep the
  exemption, their measured models already outranking the prior-only points.
- **Records are read off the posterior MEDIAN**, the number the timelines
  plot, so every fitted point is a point the reader can see on top of the
  cloud. The mean disagrees on both posterior shapes this data produces: a
  thinly-evaluated release is right-skewed and its mean sits above every
  plotted point, and a ridge-split ability is bimodal and its mean lands in
  the empty valley between the lumps.
- **Humans, same caveat.** A tier is measured on an axis only if it scores on
  some benchmark loading there (|promax median loading| ≥ 0.5). Humans took
  almost no agentic benchmarks, so on that axis most tiers sit at prior-only
  levels and their forecast-crossover dates are prior-driven.
  `make_memo_figs._measured_tiers` computes the measured set; a faint or open
  marker for unmeasured tiers is the honest rendering.

**Exploratory K=3 is bimodal, and no exchangeable prior fixes it.** The
residual structure past axis 1 splits into two directions of near-equal
strength, which is a flat posterior valley rather than a rotation artifact.
Loading-side priors rearrange the valley; only information on θ (the human and
lineage priors) or more benchmark coverage moves it. Report mode-aware, and
monitor coverage. Do not run another K=3 with a new loading prior expecting the
axis 2-3 boundary to resolve.

**The compensatory link is the one that fits.** Conjunctive product links fit
worse than K=1 on this data: weakness on one axis is bought back by strength on
another.

**Convergence is judged on identified quantities only** — `eta`, `D`, `sigma_b`
via `mirt_identified_rhat`. Raw r-hat on `A`, `theta` and `tau_A` is
permutation-inflated and ignored.

**Diagnostic honesty rule.** Convergence, PPC and PIT reports always cover the
whole fit. Plot-side filters such as `mirt_informed_mask` never touch a
diagnostic number.

## Dashboard

`python diagnostics/build_dashboard.py` renders every registered fit into the
tracked repo-root `index.html`: a fit selector plus a cross-fit comparison
view, figures rendered lazily. Comparison CSVs go to `results/comparisons/`;
the render cache lives in the gitignored `results/dashboard_cache/`. Always
pass `--force-all` when serving results, so no superseded card is served from
cache. `--png` also dumps stills to `plots/dashboard/`.

**Mode-aware cards.** A multimodal fit gets one extra loading + timeline figure
set per posterior mode, labelled with its chains and Δlogp, plus a mode summary
table. The prerequisite is
`python diagnostics/diagnose_chains.py --write-modes --trace <trace>`, which
writes `results/<fit>/mirt_modes_<trace-stem>.json`; the build only reads it, so
no trace is loaded to re-detect the split. Mode restriction is plot-side only:
convergence, PPC, PIT, GoF and LOO on the card always describe the whole fit.

Oblique timelines use **promax deliberately**: its target-chasing tilts axes
toward the dominant factor, so a skill-axis timeline includes the general rise,
which is the wanted reading (the Φ share is stated in each title).
`align_rotations` also offers geomin, kept as the principled-oblique reference
rather than the display frame.

## Environment

- Miniforge (`~/miniforge3`), conda env `pymc_env`.
- `~/miniforge3/envs/pymc_env/bin/python fit.py --preset canonical`, or
  `conda activate pymc_env` first.
- `pymc 5.28.5`, `arviz 0.23.4`, `pytensor 2.38.3`, numpy, pandas, scipy,
  `plotly 6.x`, `kaleido` + Chrome (`plotly_get_chrome -y` once per env, or
  `fig.write_image` raises). `nutpie` (Rust NUTS) is the default sampler, 2-3×
  faster than PyMC NUTS on this CPU. The heartbeat progress callback fires only
  under `--sampler pymc`.
- **In notebooks** set `progressbar=False`: the ipywidgets bar hangs in
  VS Code / Cursor / Jupyter. On the CLI, `True` falls back to text mode.
- Tests: `python -m pytest -m "not slow"` (~20 s) or full `pytest`.
  `tests/test_safety_net.py` pins golden initial-point log-probabilities for
  every model configuration, so an unintended math change fails loudly.

## PyMC conventions

- Explain *why* a prior or parameterisation was chosen: scale, domain
  constraints, sampler geometry, identifiability.
- `pm.Deterministic` for identified quantities derived from raw variables.
- Non-centered parameterisation throughout (`*_z × tau`).
- Comments state what the code *is*, including constraints the code cannot
  show. Never what changed, never where code used to live.

## Style guide

Applies to everything user-facing: chat replies, documents, figure text,
slides, commit messages.

**Voice.** Short declarative sentences. Claim first, evidence immediately
after: "It converges: min/median ESS 104 / 35,289." Aim at ASD-STE100
Simplified Technical English for dense passages. No em dashes, no storytelling
arcs, no rhetorical questions, no "delve / crucially / notably / it is worth
noting / comprehensive", no "not just X but Y", no praise adjectives, no
wrap-up paragraph restating what was just said. Define a technical term once,
in a short parenthetical at first use. If a sentence survives with a word
removed, remove the word.

**Numbers.** Every quantitative claim carries its number; every count carries
its denominator ("136 / 80,000 divergences"). Effect sizes against posterior
SD, never bare deltas. Claim strength matches evidence: "explored" is not
"tested". Preferred convergence readout is min/median bulk ESS plus
divergences; r̂ is secondary.

**Documents.** Final result first, failures and history to an appendix. The
audience is a statistician, not a code reviewer: model math, priors, numbers,
never file names or run bookkeeping. Bold topic lead-ins ("**Result.**").

**Figures.** Self-explanatory from title and legend alone, no text blocks on
the canvas, every line and band in the legend. Short titles. English axis text
always, even in a French document. Dashboard fonts and colors, colorblind-safe
palette. Open the render and inspect it before shipping.

**Slides (beamer).** Support, not a script: 3-8 word bullets, anything sayable
orally stays off the slide. Math carries the content; semantic color macros
(`\pos` / `\con` / `\HL`).

**Process.** When editing text Yassine has hand-revised, propose each change
and ask; never batch-rewrite silently. In diagnosis: facts only, one plot per
claim, pedagogical rather than a number dump.

## References

**Project**
- [Alexander Barry — Epoch Capabilities Index](https://github.com/AlexanderTBarry/Epoch-Capabilities-Index)
  — the original R + Stan implementation this project reproduces in PyMC.
- [*A Rosetta Stone for AI Benchmarks*](https://epoch.ai/publications/a-rosetta-stone-for-ai-benchmarks)
  ([arXiv 2512.00193](https://arxiv.org/abs/2512.00193)) — the methodology
  paper: same 1-capability / per-benchmark-difficulty / per-benchmark-slope
  framework, ECI defined over ≥ 4 benchmarks per model.
- [Epoch Capabilities Index dashboard](https://epoch.ai/eci) ·
  [ECI documentation](https://epoch.ai/data/eci-documentation)

**Data sources**
- [Epoch benchmark ZIP](https://epoch.ai/data/benchmark_data.zip)
  ([landing page](https://epoch.ai/data/ai-benchmarking-dashboard))
- [Epoch live feed](https://epoch.ai/data/benchmarks.csv). Companion files on
  the same host: `random_baselines.json` (chance floors),
  `score_ceilings.json`, `benchmarked_models.csv` (the best bulk source for
  release-date backfills).
- [RAND RR-A3797-1](https://www.rand.org/pubs/research_reports/RRA3797-1.html)
- [Scale SEAL leaderboards](https://scale.com/leaderboard)
- [Kaggle Open Benchmarks](https://www.kaggle.com/api/v1/benchmarks/<owner>/<slug>/leaderboard)

**Non-compensatory MIRT** (background for `models/mirt_nc.py`, IRF
$P=\prod_k\sigma(a_k\theta_k+c_k)^{q_k}$)
- [Bolt & Lall (2003)](https://journals.sagepub.com/doi/10.1177/0146621603258350)
  — Bayesian/MCMC non-comp estimation.
- [PMC5978509](https://pmc.ncbi.nlm.nih.gov/articles/PMC5978509/) — flat
  likelihood, dimension-switching, simple structure as remedy.
- [Tamano et al. (2025), arXiv 2507.15222](https://arxiv.org/abs/2507.15222)
  — fitting non-comp data as compensatory underestimates high skills.

**Internal diagnostics**
- [`diagnostics/residual_corr.py`](diagnostics/residual_corr.py) — pairwise
  benchmark residual correlations on the canonical fit.
- [`diagnostics/multidim_ppca_ard.ipynb`](diagnostics/multidim_ppca_ard.ipynb)
  — PPCA + ARD on logit scores.
- [`diagnostics/plot_mirt.py`](diagnostics/plot_mirt.py) — single-fit deep dive.
- [`data/pipeline/README.md`](data/pipeline/README.md) — full pipeline docs.
