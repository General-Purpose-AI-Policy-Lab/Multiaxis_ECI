# Results inventory for the arXiv preprint: data generation 2026-10-19

Read-only audit, 2026-11-03. Repo at origin/main 2789fef, plus the uncommitted step-0 placement change (`2_model/multiaxis_eci/analysis/human_placement.py`, `4_diagnostics/5_place_humans.py`, `tests/test_human_placement.py`). Paths are relative to the repo root. `F/` stands for `5_outputs/data20261019/mirt_lineageprior_lineagebm/`, `O/` for `5_outputs/data20261019/`. This file replaces the 2026-10-29 inventory (checkout 8a66950, 34 commits behind).

Status codes:
- **C**: citable now, from a file in the tree.
- **T**: citable after the 50-chain flagship trace is on this machine (from Jérémy, or a local R1 refit).
- **N**: not citable.

## 0. Key facts first

- **The flagship exists upstream and its structure-level results are citable.** 50 chains (5 runs x 10, seeds 44-48, 2,000 tune / 2,000 draws), ceilings on, models only, 5,487 obs / 887 models / 107 benchmarks. Five modes: A 21 / B 5 / C 19 / D 4 / E 1 chains (`F/mirt_modes_trace_mirt_k4_lineageprior_lineagebm.json`). Every committed table reads mode A. The 2026-10-29 verdict ("no per-axis K=4 result citable") no longer holds for axes, loadings, modes, integrated LOO and GoF. It still holds for per-model abilities, crossing dates and human tiers.
- **Mode A fails r-hat <= 1.01 within itself.** eta r-hat 1.048, min / median eta bulk ESS 276 / 7,699, 110 divergences in 8,400 saved draws (21 chains x 400, thin 5) (`O/comparisons/mode_eval_table.csv`). The dashboard drops the 29 non-A chains before any summary (`4_diagnostics/4_build_dashboard.py`, `drop_chains`), so these are within-mode numbers, not mode-inflated. No D or sigma_b r-hat is recorded. Divergences over all 42,000 mode-A draws are not recorded.
- **No trace of any fit the repo reads at K=4 is on this machine.** The K=4 traces on data20261019 that are on disk are the failed 5-chain refit, its mode splits, and the step-0 NUTS-from-optima check (section 1).
- **The committed human placement is stale.** `F/human_placement.json` (2026-10-28) predates the step-0 ceiling fix. On the 5-chain proxy the fix moves axis 4 (Top Performer becomes a lower bound, Domain Expert is no longer drawn).
- **K=1 ECI-H, its crossings and the frontier gaps stay citable**, with one new caveat: every canonical run includes the Skilled Generalist SimpleQA Verified 0.944 row withdrawn on 2026-10-21.

## 1. Traces on this machine

| trace | fit | data | use |
|---|---|---|---|
| `deliverables/flagship_refit_20261021/source/5_outputs/data20261019/mirt_lineageprior_lineagebm/trace_mirt_k4_lineageprior_lineagebm.nc` (12.4 GB) | flagship spec, 5 chains x 12,000, tune 5,000, seed 42, code 8a66950 | 20261019, SimpleQA Verified human row still in the ceiling bound | failed: 1,612 / 60,000 divergences, eta r-hat 1.101, D r-hat 1.125 (`deliverables/flagship_refit_20261021/README.md`, `validation/`). Proxy for step-0 checks only. **N** |
| `deliverables/flagship_refit_20261021/exploratory_modes/mode_{A,B,C}/mirt_lineageprior_lineagebm/trace_mirt_k4_lineageprior_lineagebm.nc` | same refit split by mode: A 2 chains, B 1, C 2 | same | proxy (nested audit, placement before/after). **N** as results |
| `deliverables/flagship_refit_20261021/validation/{raw_checkpoint,derived_export_attempt1_mode_A}.nc` | same refit | same | none |
| `6_writeups/preprint/step0/map_search/nuts_from_optima_trace.nc` | flagship spec, 10 PyMC chains x 300 draws started at MAP optima | 20261019 | noise-collapse check only |
| `deliverables/canonical_sciencespo_20261022/source/5_outputs/data20261019/canonical_humanmerge/trace.nc` (1.55 GB) | canonical K=1 ECI-H, 8 x 2,500, code 8a66950 | 20261019, with the SimpleQA row | a second run of `O/canonical_humanmerge/`; tier means agree to 0.4 ECI-H |
| `results/**/*.nc` (22 files) | pre-pipeline fits, 2026-07 to 2026-09 | older data | blog post and history only |

Not on disk: the 50-chain flagship, the 20-chain no-ceiling fit, the joint fit, every bifactor / simple-structure / pinned / semicomp0 fit, the canonical access-class fits and the repo's own 2026-10-20 ECI-H run. The `--orthogonal` fit (option added in 4926945) has no output folder in the tree. The 50-chain pinned fit named in commit 997a517 (14 modes) is not in the tree; the committed pinned folders hold 10 chains.

## 2. Fits on data20261019

| folder (`O/`) | config | chains | modes (chains, Δlogp nats) | chains read | GoF of read chains | status |
|---|---|---|---|---|---|---|
| `mirt_lineageprior_lineagebm/` (flagship) | K=4, normal loadings, lineage BM, skew-normal theta, estimated ceilings, floors, pooled noise, Beta-binomial; models only; data without the SimpleQA Verified human row | 50 | A 21 (0), B 5 (−9.6), C 19 (−17.3), D 4 (−25.8), E 1 (−54.2) | A (best logp; no `selected_chains`) | R² 0.9649, RMSE 0.0429, LOO-PIT var 0.0666, 853 / 5,487 Pareto k > 0.7 (`mirt_gof_k4.json`) | **C** structure, **T** per draw |
| `mirt_lineageprior_lineagebm_noceiling/` | same, no ceilings | 20 (seeds 42 + 43) | A 17 (0), B 3 (−8.7) | A (`selected_chains`, user 2026-10-20) | R² 0.9637, RMSE 0.0441, PIT var 0.0671, 768 high k. eta r-hat 1.165, ESS 67 / 6,044, 59 divergences in 6,800 saved draws (`git show 8a66950:5_outputs/data20261019/comparisons/mode_eval_table.csv`) | **C** as sensitivity; not like-for-like (20 vs 50 chains, other seeds) |
| `mirt_lineageprior_lineagebm_noceiling_humanmerge/` (joint) | no ceilings, tiers inside, merged order | 10 (seed 42) | 1 (min matched corr 0.887) | all | R² 0.9605, RMSE 0.0464, 767 / 5,544 high k | **C** as negative result |
| `canonical_humanmerge/` (ECI-H) | K=1, pt1, merged order, curated exclusions; 4,410 obs, 809 takers (7 tiers), 93 benchmarks | 8 x 2,500 | 1 | all | R² 0.9127, RMSE 0.0745, PIT var 0.0755 (`gof.json`) | **C** |
| `canonical/` | K=1, no human order | 8 x 2,500 | not recorded | all | R² 0.9118 | **C** as sensitivity |
| `canonical_{public,semi_private,private}_humanmerge/` | K=1 per access class | 4 x 2,000 | not recorded | all | R² 0.8974 / 0.9472 / 0.8945 | **C** for frontier gaps (quick fits) |
| exploration fits (3.6, 3.7) | see there | 10 to 50 | 4 to 16 | various | R² 0.9622 to 0.9687 | **N** as results, **C** as negative findings |

K=1 convergence, max r-hat / min bulk ESS over every row of `summary.csv`: humanmerge 1.0081 / 654; canonical 1.0058 / 826; public 1.0039 / 735; semi-private 1.0096 / 159; private 1.0136 / 272. The deliverables ECI-H run: r-hat 1.0055, ESS 1,270 / 18,187, 7 / 20,000 divergences (`deliverables/canonical_sciencespo_20261022/validation/K1_report.md`).

## 3. Results the paper could cite

### 3.1 K=4 flagship structure (mode A, 21 / 50 chains)

| result | value | source | status |
|---|---|---|---|
| axis names | 1 Scientific Knowledge & Reasoning; 2 Agentic Capabilities & Programming; 3 Cyber Capabilities; 4 Commonsense & Factual Knowledge. User, 2026-10-21. Mode A at 50 chains matches the 20-chain mode A at axis corr 0.999 (commit 1f35d7d) | `F/axis_names.json` | C |
| top benchmarks per axis | 1: OTIS Mock AIME, MMLU Chemistry, MATH L5, AudioMultiChallenge, GPQA Main Chemistry. 2: OSWorld, SWE Atlas, DeepSWE, MCP Atlas, APEX-Agents. 3: CyScenarioBench, CyberGym, Furniture Assembly, GBAEval. 4: OpenBookQA, ARC AI2, SimpleQA Verified, SimpleBench, PIQA | `F/axis_names.json`, `F/mirt_loadings.csv` | C |
| axis correlations | max 0.637 (1-2), min 0.229 (3-4) | `F/mirt_factor_corr.csv` | C |
| modes | 5; min matched corr within A 0.889, C 0.883, D 0.804; A to every other mode matched corr <= 0.02 | modes JSON | C |
| mode content | B: visual axis, agentic and cyber together. C: A's agentic and cyber axes, knowledge and maths on separate axes. D: agentic and cyber together, a science axis, a visual axis | `README.md` Limitations; `5_outputs/sampler_study/data_wishlist.md` | C (descriptive) |
| integrated LOO vs A, 3,116 scores reliable in every mode | B −6.5 ± 3.2, C −9.0 ± 4.1, D −14.4 ± 6.9, E −35.7 ± 6.4 | `F/investigation/integrated_loo_5modes_50chains.json` | C |
| integrated LOO vs A, all 5,487 scores | B −26.7 ± 7.6, C −18.4 ± 9.1, D −10.5 ± 15.9, E −66.9 ± 14.1. Integrated k > 0.7: A 1,797, B 1,109, C 1,754, D 928, E 420 | same | C, with the k counts |
| PSIS elpd, mode A | −16,526.5 ± 196.4; 15.2 % of scores k > 0.7 | `O/comparisons/mode_eval_table.csv` | C. Do not mix with the JSON's `elpd_psis` (−16,334.4), a chain-averaged estimate |
| GoF, mode A | R² 0.9649, RMSE 0.0429, MAE 0.0289, LOO-PIT var 0.0666 (ideal 0.083) on 4,528 scores | `F/mirt_gof_k4.json` | C |
| K=1 baseline, same scope | R² 0.9284, RMSE 0.0682, LOO-PIT var 0.0751 | `F/mirt_gof_k1.json` | C in-sample only; no K=1 LOO elpd on this scope |
| convergence, mode A | eta r-hat 1.048; eta ESS 276 / 7,699; 110 / 8,400 saved draws divergent | `O/comparisons/mode_eval_table.csv` | C as stated; fails r-hat <= 1.01 |
| loading scale | one shared tau_A, 0.837 [0.749, 0.931] | `F/mirt_tau_spectrum.csv` | C (no axis ranking from it) |

### 3.2 Ceilings (sensitivity finding)

| result | value | fit and chains | source | status |
|---|---|---|---|---|
| robust mixture LOO, ceilings minus no ceilings | +55.8 ± 12.1 nats on 4,967 / 5,487 scores (ESS >= 50); +57.2 ± 9.5 on 4,502 (ESS >= 200); +46.9 ± 22.6 on all | 4 x 1,000 mixture chains per side. Ceilings: seed-42 10-chain fit, mode B (merged + visual, 2 chains). No ceilings: 20-chain mode A (same structure). Ceiling bound still included the SimpleQA Verified human row | `F/investigation/README.md` | C with caveats: not the current flagship's structure or data; output files `O/diagnostics/robust_loo_*` absent |
| PSIS-LOO at equal structure, reliable scores | +100 ± 10 (merged structure), +80 ± 14 (agentic apart, the current mode-A structure) | same two fits | same | C |
| estimated vs benchmark-forecasting ceilings | median gap 0.054 vs 0.055 on 80 benchmarks; corr 0.69 (Spearman 0.74) | seed-42 10-chain fit | `F/investigation/ceilings_vs_mbf.csv` | C |
| ceilings informed by data | 8 / 85 (posterior SD < 60 % of prior); 70 / 85 prior-dominated (> 85 %) | same | `F/investigation/information_and_loo.csv` | C |
| integrated LOO, ceilings vs no ceilings | +43 ± 9, +88 ± 13 on reliable scores | same | `F/investigation/integrated_loo_compare.csv` | N: founder bug, marked "faussé" |
| skew-normal theta prior | 14 / 30 chains in the best region vs 4 / 30 Gaussian (Fisher p ≈ 0.01); LOO better by 60-100 nats in 3 / 3 paired seeds | 2026-10-13 to 16, older data, no ceilings | `5_outputs/sampler_study/GEOMETRY_REPORT.md` | C as design motivation |

### 3.3 Human tiers on the K=4 axes

Committed placement `F/human_placement.json` (2026-10-28, after the ARC-AGI exclusion of commit f4a3fc0, before the step-0 fix). [q10, median, q90] on the fitted scale (model mean 0):

| axis | scores / benchmarks | tiers drawn / 9 | Skilled Generalist | Domain Expert | placement r-hat, ESS |
|---|---|---|---|---|---|
| 1 Scientific | 25 / 18 | 6 | −0.23 [−0.30, −0.15] | 0.97 [0.89, 1.05] | 1.02, 346 |
| 2 Agentic | 3 / 3 (METR Time Horizons, OSWorld, The Agent Company) | 4 | 1.45 [1.28, 1.59] | 1.59 [1.47, 1.73] | 1.01, 437 |
| 3 Cyber | 1 / 1 (Cybench) | 1 (Committee of Domain Experts) | not drawn | not drawn | 1.00, 1,321 |
| 4 Commonsense | 6 / 5 | 6 | 2.90 [1.94, 4.30] | 4.67 [2.83, 7.08] | 1.00, 1,469 |

- Status **T** for every row. The placement must be re-run on the 50-chain trace with the step-0 fix (`6_writeups/preprint/step0/human_placement/NOTE.md`).
- Step-0 effect on the 5-chain proxy (mode A): 4 / 35 scores become ceiling scores; axis-4 tiers drawn 6 / 9 to 4 / 9; Top Performer becomes a lower bound at 4.32 (median 6.41 to 6.53); Domain Expert is no longer drawn. Axes 1-3 unchanged.
- Axis 3 rests on 1 score and supports no tier claim at any status.
- All 42 human-scored benchmark-axis pairs of the flagship axes are judged (22 / 4 / 7 / 9 per axis; step-0 NOTE). An R1 or R2 refit with new axis titles starts with every pair unjudged.
- Placement rule robustness: placing with the frontier's profile instead of the typical model's moves axis-4 tiers up by at most 0.95 and down by at most 0.03 (`F/investigation/human_profile_sensitivity.csv`, seed-42 mode A, 8 chains). C as a check of the rule, on an older fit.

### 3.4 Crossover dates on K=4

- No crossing table exists for the 50-chain flagship. The forecast figures (`docs/figures/forecast_axis{1,2,3,4}.png`, 2026-10-23) show lines and bands, no dates. They predate the ARC exclusion: axis 1 still draws Average Human and Committee of Average Humans. **N** now, **T** via `4_diagnostics/3_plot_mirt.py --forecast` after the placement re-run.
- The refit's crossings (`deliverables/flagship_refit_20261021/majority_presentation_20261022/conditional_human_crossings.csv`, `all_five_forecasts_20261022/`) come from a non-converged 5-chain run on its lower-logp family. **N**.

### 3.5 Joint fit with the tiers inside

One mode, 10 / 10 chains, 1 / 20,000 divergences. Tiers sit a median −6.4 SD from their axis-by-axis placement on the science axis (range −10.6 to −4.6); Domain Expert −5.9 SD against +1.2 SD placed (`O/mirt_lineageprior_lineagebm_noceiling_humanmerge/README.md`). **C** as evidence that the compensatory sum trades tier axes. The comparison uses the no-ceiling fit's pre-10-28 placement.

### 3.6 Less-compensation study

Source `O/LESS_COMPENSATION.md` (French). Problem on flagship mode A: within-model posterior corr of axes 1-2 is −0.46.

| variant | finding | status |
|---|---|---|
| `--theta-general` alone | learned λ 0.1 to 0.35; no change for well-measured models | C (negative) |
| `semicomp0`, 6 chains, β ≈ 2.0 | anti-corr −0.46 to −0.15, but forces frontier models to balance (GPT-6 astra axis 1 2.2 to 0.8) | C (negative) |
| simple structure s = 2.5, 30 chains | 9 modes, best 2 / 30 chains; axis-1-2 corr +0.04; loglik −60 nats; integrated LOO −43.6 ± 10.1 on 3,588 reliable scores vs flagship A (+2.1 ± 18.9 on all) | C (compensation removed at a predictive cost) |
| simple structure s = 5, 50 chains | best 2 / 50 chains; loglik −104; modes file has 16 modes, 4 unlabelled | C (negative) |
| continuation, marginalized axis | 0 / 10 chains within 10 nats of the best mode | C (negative) |
| geomin rotation of mode A | anti-corr −0.46 to −0.03; repairs axis 4; no clean agentic direction | C |
| pinned `_pingeomin-1p5`, 10 chains | 4 modes; 8 / 10 chains in the pinned orientation; mode A 55 nats below unpinned s = 2.5, 110 below the flagship | C (exploratory) |
| `_pincategory-1p5`, `_pingeomin-1`, 10 chains | 10 and 7 modes, mostly 1 chain each; not read in the report | N |

### 3.7 Bifactor fits (2026-10-27 to 11-02)

| folder (`O/`) | chains | modes | chains read | g title |
|---|---|---|---|---|
| `mirt_bifactor_lineageprior_lineagebm/` | 50 | 8 | mode D, 6 chains, −41.9 nats (best integrated LOO, user 11-02) | Geolocation, Trivia & Commonsense |
| `mirt_bifactor_lineageprior_lineagebm_gsum/` | 50 | 11 | mode D, 12 chains, −25.5 | General Capabilities (Math, Science & Cyber) |
| `..._k5/`, `..._gsum_k5/` | 10 | 9 and 8, mostly 1 chain each | none | unnamed |
| `..._humanmerge/`, `..._gsum_humanmerge/` | 20 | 4 and 6 | none | named |

**N** as results: the integrated LOO that chose mode D is in `diagnostics/integrated_loo.log`, gitignored and absent, and no comparison against flagship mode A is on disk. **C** as one finding: every bifactor fit splits into 4 to 11 modes, and the selected mode is not the best-logp mode.

### 3.8 Step 0 (2026-11-02 to 11-03)

| result | value | source (`6_writeups/preprint/step0/`) | status |
|---|---|---|---|
| noise collapse at MAP | 60 / 60 optima collapse: median 80 / 107 benchmarks with σ_b < 0.01, 1 to 6 at the 0.005 floor; loglik about 3,900 nats above the sampled posterior | `map_search/NOTE.md`, `map_starts_all.csv`, `map_sigma_b_all.csv` | C (methods: optimiser summaries invalid here) |
| NUTS from the 10 best optima | 10 / 10 chains leave during tuning; μ_s −3.41 to −3.46 vs −3.43 sampled; 20 / 3,000 divergences | `map_search/nuts_from_optima.csv` | C (300 draws per chain) |
| nested benchmarks, K=1 | shared-question pairs −0.057 over 30 pairs (baseline −0.057); RAND-RAND +0.044 over 105 pairs (excess about +0.10); FrontierMath T1-3 / T4 0.63 (99th pct, 65-69 models); AISI CTF Practitioner / Expert 0.60 (98th, 10 models) | `nested_audit/NOTE.md` | C (canonical K=1) |
| nested benchmarks, K=4 | RAND-RAND 0.005 to 0.013 per mode; 6 / 10 of mode A's axis-1 top benchmarks are RAND rows | same, on the 5-chain refit | T |
| placement ceiling rule | see 3.3 | `human_placement/NOTE.md` | C as method, numbers T |

The nested-audit NOTE lists `audit.py`, `pairs.csv`, `families.csv`, `axis_top10_families.csv`, `baselines.json`. Only `NOTE.md` is in `step0/nested_audit/`.

### 3.9 K=1 ECI-H

| result | value | source | status |
|---|---|---|---|
| tier ECI-H, mean [HDI] (obs) | Average Human 84.1 [74.4, 92.2] (3); Skilled Generalist 115.0 [110.5, 118.6] (9); HS Qualifier 137.4 [130.5, 144.3] (1); Domain Expert 152.5 [149.4, 156.1] (18); Committee of Domain Experts 155.4 [152.0, 159.4] (3); HS Top Performer 162.1 [154.7, 171.1] (4); Top Performer 245.7 [222.9, 276.4] (2) | `O/canonical_humanmerge/human_groups_eci.csv` | C |
| crossing dates, median [80 %] | Average Human passed (left-censored, 2023-02); Skilled Generalist 2023-03; HS Qualifier 2024-12 [2024-09, 2025-03]; Domain Expert 2025-08 [2025-06, 2025-11]; Committee of DE 2025-11 [2025-11, 2026-02]; HS Top Performer 2026-02 [2025-11, 2026-04]; Top Performer 2029-03 [2028-04, 2030-01], extrapolated | `deliverables/canonical_sciencespo_20261022/figures/eci_human_crossings_18mo.csv` (gitignored; the 10-22 run) | C |
| model vs tier | gpt-4-0314 above Average Human in 20,000 / 20,000 draws (+35.1 [24.5, 48.2]); Claude 3.5 Sonnet (2024-10-22) above Domain Expert in 0 / 20,000 (−22.5 [−28.2, −17.9]); gpt-6-astra_xhigh above Top Performer in 0 / 20,000 (−62.5 [−109.1, −33.1]) | `deliverables/canonical_sciencespo_20261022/validation/K1_report.md` | C |

Caveat for every row: both runs fit the Skilled Generalist SimpleQA Verified 0.944 row, withdrawn on 2026-10-21 as a label-quality check (commit 095dfc0). SimpleQA Verified is in the canonical scope (`1_curated/excluded_benchmarks.txt` does not list it). The effect is unmeasured; a canonical refit removes it.

### 3.10 Frontier gaps (canonical K=1, the 2026-10-20 runs)

Tables `O/comparisons/frontier_gap_{openness,country}_table.{md,csv}`; notes `5_outputs/open_closed_frontier/README.md`, `5_outputs/us_cn_frontier/README.md`. Mean months behind, frontier points of the last 12 months, median [80 %]:

| comparison | all (93) | public (61) | semi-private (10) | private (19) |
|---|---|---|---|---|
| open behind closed | 6.7 [6.0, 7.3] | 5.5 [4.7, 6.3] | 6.7 [6.0, 7.5] | 7.8 [6.3, 8.8] |
| China behind US | 6.5 [5.9, 7.2] | 5.6 [4.8, 6.3] | 6.5 [5.8, 7.2] | 6.6 [5.6, 7.7] |

- Private minus public: open 2.2 [0.7, 3.7] months (4.0 [2.2, 5.5] on an average day); China 1.1 [−0.1, 2.4] (2.5 [1.3, 4.2]).
- Trend-line lag today: 9.2 [8.2, 10.3] (open), 8.9 [7.8, 9.9] (China). Slopes: closed and US 24.9 ECI-H/yr, open 18.3, China 19.0.
- Coverage on private benchmarks: 28 open vs 103 closed candidates; 25 Chinese vs 105 US; 10 frontier points each.
- Status **C**. Check before citing: the row "Lag of the latest frontier point" is identical in both tables in every column (5.5, 4.2, 5.0, 4.2). This is plausible if the latest open record is Chinese and its reference US, but it is not verified.
- The access-class traces and the 10-20 ECI-H trace are not on disk. A re-run needs refits.

### 3.11 Not computed

- ECI-H vs Epoch's published ECI: no script. Inputs: `deliverables/sciencespo_plan_20261021/epoch_eci.csv` (display names) and `O/canonical_humanmerge/all_models_eci.csv`, with a name map. **N**.
- Per-model K=4 abilities: `F/` holds no ability table. The ranks in `LESS_COMPENSATION.md` (GPT-6 astra 1 / 135 / 7 / 294 in mode A) come from session scripts in the gitignored `traces_backup/`. **T**.

## 4. Contradictions and stale items

Against `README.md` "Current results" and "Limitations":

1. README: "40 human scores fall on benchmarks that measure an axis", "30 on scientific knowledge & reasoning". The committed placement has 35 and 25 (`F/human_placement.json`). Commit f4a3fc0 (2026-10-28) took ARC-AGI and ARC-AGI-2 off axis 1; the last README content change is c4a567f (2026-10-26).
2. README Limitations: "the Skilled Generalist at 0.40 on MATH Level 5 but 0.98 on ARC-AGI" as an axis-1 disagreement. ARC-AGI no longer counts on axis 1.
3. README figures `docs/figures/forecast_axis*.png` (2026-10-23) draw Average Human and Committee of Average Humans on axis 1. The committed placement no longer draws them (`drawn` list, axis 1). Axis 4 will change again with the step-0 fix.
4. README: ceilings "+56 ± 12 nats on the 4,967 reliable scores" at equal axis structure. The structure is the merged one (ceilings mode B, 2 chains of the seed-42 10-chain fit), on data whose ceiling bound still held the SimpleQA Verified human row. The current flagship's mode A has the agentic-apart structure; the only unbugged number for it is PSIS +80 ± 14.
5. README: integrated LOO "C −9.0 ± 4.0". JSON: ± 4.1. Rounding only.

Elsewhere:

6. `4_diagnostics/README.md` calls `robust_loo.py` "kept for the record", replaced by `integrated_loo.py`, while the README's ceilings claim rests on robust LOO. The same file says integrated LOO takes k > 0.7 "from about 740 to 290 per mode"; on the 50-chain flagship mode A has 1,797 / 5,487 integrated k > 0.7 against 853 for PSIS.
7. `6_writeups/preprint/PLAN.md` §0: "eta r-hat 1.048 on the whole fit (mode-inflated)". It is computed after dropping the 29 non-A chains, so it is within mode A.
8. `PLAN.md` §0 lists an "orthogonal" fit and "9-16 modes" for every exploration variant. No orthogonal output exists; the free bifactor has 8 modes, gsum 11, the pinned fits 4 to 10.
9. Free bifactor `selected_chains.why`: mode D's "g is not empty, unlike the best-logp mode A". Commit 955f20d: mode D's g is "nearly empty", titled "Geolocation, Trivia & Commonsense". The gsum modes file repeats the same `why` text.
10. `O/mirt_lineageprior_lineagebm_general_simplestruct5/` modes file: 4 / 16 modes unlabelled (written before 997a517).
11. The canonical ECI-H runs and the robust LOO used data with the withdrawn SimpleQA Verified human row; the 50-chain flagship does not.
12. `O/mirt_lineageprior_lineagebm_noceiling/human_placement.json` uses pre-10-28 content judgments; it was not re-placed because its trace is gone (commit 6c25217).

Resolved since 2026-10-29: README now describes the ceiling flagship; the K=4 recipe says 2,000 draws; axis names are no longer marked provisional; `O/comparisons/gof_table.csv` and `mode_eval_table.csv` now describe the flagship; integrated LOO no longer has the founder bug.

## 5. Gaps for the paper

| gap | blocks | needs |
|---|---|---|
| r-hat and ESS on D and sigma_b within mode A (and C); divergences on all draws | convergence table | the trace, minutes |
| eta r-hat <= 1.01 within mode A | a "converged" claim | more chains in A, or report 1.048 as is |
| human placement with the ceiling fix | every tier claim | the trace, `5_place_humans.py` |
| K=4 crossing dates per axis and tier, per mode | forecast section | the trace, `3_plot_mirt.py --forecast` |
| per-model abilities, mode A | any model ranking | the trace, `diagnose_chains.py --write-tables` |
| ceilings comparison at the flagship's structure, like-for-like chains | ceilings claim | R2 at 50 chains, seeds 44-48, then integrated or robust LOO |
| K ladder LOO (K=1 to 5) on one scope | why K=4 | R3, R4 |
| theta bimodality, lineage regressions, PIT by benchmark / year / source | diagnostics | the trace, minutes |
| ECI-H vs Epoch ECI | external validation | name map, about 1 h |
| backtest, symmetric drift, recovery, no-RAND | R5 to R8 | refits |
| ECI-H without the withdrawn SimpleQA row | K=1 tier numbers | canonical refit, hours |
| nested-audit artifacts | reproducibility of step 0.2 | restore `audit.py` and its CSVs |

## 6. What changed since 2026-10-29

- Flagship: 10 chains (seed 42) to 20 chains (2026-10-21, commit 095dfc0) to 50 chains (2026-10-23, commit 1f35d7d). Two modes became five. Mode A keeps the top benchmarks of the 10-chain ceiling fit's mode A (agentic and cyber apart).
- SimpleQA Verified human row withdrawn (095dfc0); its ceiling moved from 0.967 to 0.937.
- Integrated LOO founder bug fixed (211e576); flagship modes rescored (c4a567f); both 2026-10-21 integrated-LOO tables marked "faussé".
- ARC-AGI and ARC-AGI-2 off axis 1 for human placement (f4a3fc0).
- Less-compensation study, axis pins, bifactor and gsum fits (d8ca516 to 955f20d); `--orthogonal` option (4926945), not fitted in the tree.
- Traces now float32, one draw in two, without nutpie's per-parameter sample stats (fd584c4, 936e7fb, bc77fdf).
- Step 0 (2026-11-02 to 11-03): noise collapse, nested audit, placement fix.
