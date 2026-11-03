# Referee red team: K=4 Beta-MIRT preprint (pre-writing)

Date 2026-11-03. Repo read at origin/main 2789fef plus the uncommitted step-0 placement change (data generation 20261019). Read only: no fit run, no code edited. Paths are relative to `/Users/yassineessifi/Desktop/ECI_Bayesian/`. Supersedes the 2026-10-29 version (commit 8a66950, 40 commits behind).

## 0. Verdict

The fitted-score layer is still sound. The decided framing (four bundles, no unique benchmark-to-axis allocation, multimodality as a finding) matches the evidence better than the blog framing did. Three things now threaten it from inside the repo:

1. The deliverables read mode A, but the evidence for A over C is thin: 21 / 50 against 19 / 50 chains, and 9.0 ± 4.1 nats of integrated LOO on 3,116 / 5,487 scores. The two modes differ on the axis content the paper's motivation needs (mathematics and abstract reasoning with or apart from scientific knowledge).
2. The convergence numbers the repo reports for "the flagship" are mode A only (21 / 50 chains), and mode A itself is not converged by the repo's own target: eta r̂ 1.048, min bulk ESS 276.
3. Within mode A, the compensatory sum produces per-axis rankings that contradict the raw scores (GPT-6 astra 294th / 887 on axis 4 with a top SimpleQA Verified score). Every variant that removes this costs fit and fragments into 9 to 14 modes.

Facts that frame the rest:

- Flagship: 50 chains, five runs of 10 (seeds 44 to 48), 2,000 tune + 2,000 draws, saved one draw in five (400 per chain). Five modes: A 21 chains, B 5 (−9.6 nats), C 19 (−17.3), D 4 (−25.8), E 1 (−54.2) (`5_outputs/data20261019/mirt_lineageprior_lineagebm/mirt_modes_trace_mirt_k4_lineageprior_lineagebm.json`). Minimum matched loading correlation of each other mode to A: 0.002 to 0.02.
- Fixed integrated LOO (`investigation/integrated_loo_5modes_50chains.json`): on the 3,116 scores reliable in every mode, B −6.5 ± 3.2, C −9.0 ± 4.1, D −14.4 ± 6.9, E −35.7 ± 6.4 against A. On all 5,487 scores, B −26.7 ± 7.6, C −18.4 ± 9.1, D −10.5 ± 15.9, E −66.9 ± 14.1.
- `5_outputs/data20261019/comparisons/mode_eval_table.csv`: chains 21 / 50, 8,400 draws, eta r̂ 1.048, min / median bulk ESS 276 / 7,699, 110 divergences, Pareto k > 0.7 on 15.2%.
- No K=4 trace is on this machine. `5_outputs/**/diagnostics/` is gitignored and absent. Every trace-level answer below needs the 50-chain trace or a refit.

---

## 1. Objections, ranked by damage

Each item gives the claim at risk, the evidence, what would answer it, and the cost.

### R1. "Mode A is the mode" is not supported against mode C. (new)

- **Claim at risk.** README, "Choosing the mode": the deliverables read A as "the best log-density, the best prediction and the mode most chains reach". Every figure, placement and crossing date is mode A.
- **Evidence.**
  - Chain counts: 21 against 19, per seed A 2/4/4/6/5 and C 5/2/4/4/4. The difference is within one binomial SE (about 3 chains). Basin attraction is not posterior mass in any case.
  - Log density: the A to C gap of mode means is 17.3 nats. Inside A, chain mean log densities span 0 to −47.7 (SD 11.4). 6 / 21 A chains sit below C's best chain (−27.0).
  - Integrated LOO, reliable scores: C −9.0 ± 4.1 nats over 3,116 scores, 0.003 nats per score, 2.2 SE.
  - Integrated LOO, all scores: README says "A also leads on every score", but D is −10.5 ± 15.9 (0.7 SE). That is no lead.
  - The reliable set is selected unevenly. A score counts as unreliable when its largest Pareto k over the mode's chains exceeds 0.7. A has 21 chains and loses 1,797 scores; D has 4 and loses 928; E has 1 and loses 420. The comparison set is therefore shaped most by A's own diagnostics.
  - Each chain is scored on 400 saved draws.
  - The data wishlist (`5_outputs/sampler_study/data_wishlist.md`) separates A from C with only 33 of its 2,000 most informative unobserved cells (A-D 1,027, B-C 607). A and C predict nearly the same even off the observed pattern. They differ in how they carve the axes.
  - Content: C keeps A's agentic and cyber axes but puts factual and scientific knowledge on one axis and mathematics and abstract reasoning on another (README, Limitations). That split is the one the introduction motivates (ARC-AGI and VPCT against GPQA).
- **What would answer it.**
  1. Report A and C side by side for every mode-dependent result: loadings, abilities, placements, crossings.
  2. Recompute the integrated LOO at equal chains: score random 4-chain subsets of A and C, and report the distribution of the A − C difference.
  3. Define reliability per chain, not as the maximum over chains.
  4. Hold-out-cell K-fold CV on A and C chains (refit per fold, warm-started in each mode) as the decisive predictive test.
  5. State that the data cannot rank A and C, and that A is read by convention.
- **Cost.** Items 1-3: hours on the trace. Item 4: about 5 refits of 10 chains (about 3 h each at the logged 2.4 s per iteration).

### R2. The selected mode is not converged, and the reported diagnostics describe mode A only. (revised from old R1)

- **Claim at risk.** Any posterior interval on a per-axis quantity, and the statement in `comparisons/README.md` that `eta_rhat` 1.048 is the flagship card's number.
- **Evidence.**
  - The dashboard card drops every chain outside the best mode before computing r̂, ESS, divergences, GoF and LOO (`4_diagnostics/4_build_dashboard.py`, `_flagship_drop_chains`, `render_fit`). `mode_eval_table.csv` confirms 21 / 50 chains. So eta r̂ 1.048, min bulk ESS 276 and 110 divergences are mode A's, not the fit's. `6_writeups/preprint/PLAN.md` §0 reads 1.048 as "the whole fit (mode-inflated)". That reading is wrong: it is within one mode.
  - AGENTS.md states the opposite rule: "convergence, PPC, PIT, GoF and LOO on the card always describe the whole fit".
  - The 110 divergences are counted on the 8,400 saved draws (1.3%). The count over the 42,000 sampled draws of mode A, and over the whole 100,000-draw fit, is not on disk.
  - Saving one draw in five kept about 41% of a loading's ESS (commit 936e7fb). The 50-chain trace predates the switch to one in two, so its tables rest on the reduced ESS.
  - Inside A, the 47.7-nat spread of chain log densities and the matching threshold (min matched loading correlation > 0.8; A sits at 0.889) leave open sub-modes within A.
  - The old 5-chain refit (1,612 / 60,000 divergences, min eta ESS 30) stays on record.
- **What would answer it.**
  1. Report divergences, min / median bulk ESS and identified r̂ on all 50 chains, and separately per mode with ≥ 4 chains (A, B, C, D).
  2. Report the sensitivity of the mode partition to the 0.8 match threshold and the 20-nat gap rule.
  3. Fix the dashboard to the stated rule, or change the rule and say so.
  4. Rerun at one draw in two, so that per-mode ESS is not capped by thinning.
- **Cost.** Items 1-3: hours on the trace. Item 4: one 50-chain refit (about a day).

### R3. Compensatory explaining-away makes per-axis rankings contradict the scores. (new)

- **Claim at risk.** Per-axis abilities, per-axis frontier records and crossing dates, all read off θ_k in mode A.
- **Evidence** (`5_outputs/data20261019/LESS_COMPENSATION.md`, mode A of the flagship):
  - GPT-6 astra ranks 1st on axis 1, 7th on cyber, and 294th / 887 on axis 4. On SimpleQA Verified, axis 4's main benchmark, it scores 0.76 against 0.75 for the axis-4 leader.
  - glm-5.3-flash ranks 1st on the agentic axis while it scores below recent Claude models and GPT-6 on every shared agentic benchmark (APEX-Agents 0.53 against 0.65 to 0.76; DeepSWE 0.63 against 0.73).
  - Within-model posterior correlation of axes 1 and 2: −0.46, about −0.3 on three other pairs.
  - The variants that remove the effect fail in other ways:

    | variant | result |
    |---|---|
    | `--link semicomp0` | β ≈ 2.0; forces frontier models to balance: astra falls from 2.2 to 0.8 on axis 1, behind models it beats on every shared benchmark. Rejected. |
    | `--theta-general` alone | λ 0.1 to 0.35; nothing moves for well-measured models. The compensation is in the likelihood. |
    | simple structure s = 2.5 | anti-correlation +0.04; best mode reached by 2 / 30 chains, 9 modes; −60 nats log-likelihood; integrated LOO −43.6 ± 10.1 on 3,588 reliable scores |
    | simple structure s = 5 | 2 / 50 chains in the best mode, 12 modes, −104 nats |
    | `--axis-pin` geomin, offset 1.5 | 4 modes in 10 chains; 14 modes at 50 chains (commit 997a517); best chain 110 nats below the flagship |
    | geomin rotation of mode A | fixes axis 4 (astra 5th) but leaves glm-5.3-flash 2nd agentic; 14% of loading cells turn negative |
    | bifactor, `--general-sums`, `--orthogonal` | 8, 11 modes at 50 chains; selected mode D by LOO and by a "g not empty" judgment |

- **What would answer it.**
  1. A dominance audit: for each axis, the share of model pairs that the axis ranking orders against the score ordering on every shared benchmark dominated by that axis. Report it for A, C and the s = 2.5 mode A.
  2. Report per-bundle quantities that the likelihood identifies, for example the predicted score on a fixed reference benchmark per bundle, next to θ_k.
  3. State in the model section that θ_k is a compensatory coordinate, not a skill score, and that rankings on it can invert raw-score dominance.
- **Cost.** Hours on the trace. No refit.

### R4. Per-axis quantities are not identified, and "four recurring bundles" has no stated test. (revised from old R2)

- **Claim at risk.** The paper's thesis: "the data identify a recurring four-bundle structure".
- **Evidence.**
  - The modes differ on at least one axis almost entirely: minimum matched loading correlation to A 0.005 (B), 0.002 (C), 0.002 (D), 0.02 (E).
  - The bundles do not recur unchanged. B and D merge agentic and cyber and add a visual axis; C splits knowledge from mathematics; D's axes are led by CritPt / OTIS, DeepResearch Bench / GSM8K, GDPval / GBAEval, GeoBench / SpireClimb (wishlist header).
  - 45 / 107 benchmarks have no axis carrying half their loading variance in mode A; 7 / 107 have one above 0.8 (`mirt_loadings.csv`).
  - The wishlist is regenerated for the five modes (89,422 unobserved cells). That part of old R2 is answered.
- **What would answer it.**
  1. A pre-stated recurrence criterion: for each bundle, the matched loading correlation per axis and per mode pair, and the set of benchmarks whose dominant axis is the same in every mode with ≥ 4 chains.
  2. Name as "recurring" only the cores that pass it.
  3. One figure of predicted scores on the top-separating unobserved cells, per mode.
- **Cost.** Hours on the trace. The per-mode loading tables exist only for A today.

### R5. Axis names leave out top-loading benchmarks that contradict them, and mode A does not resolve the paper's motivating contrast. (new)

- **Claim at risk.** The four names in README and `axis_names.json`, and the introduction's premise.
- **Evidence** (mode A, `mirt_loadings.csv`, axis share):
  - "Cyber Capabilities": Furniture Assembly 0.77 (3rd), Lech Mazur Writing 0.70 (5th), TriviaQA 0.55 (10th). README lists none of them.
  - "Scientific Knowledge & Reasoning": AudioMultiChallenge 0.75 (5th). ARC-AGI-2 (0.48) and VPCT (0.66) are dominated by this axis.
  - "Commonsense & Factual Knowledge": DeepResearch Bench 0.59, Balrog 0.54.
  - The names were confirmed on 2026-10-21 from an earlier chain set and carried to the 50-chain mode A.
  - The introduction says humans score near-perfectly on ARC-AGI and VPCT and near chance on GPQA, and no single ordering produces both. Mode A puts VPCT and ARC-AGI-2 on the same axis as GPQA and MMLU chemistry. The human placement then removes ARC-AGI from that axis by hand (commit f4a3fc0). Under the placement, the Average Human's VPCT 1.0 is predicted at 0.333, the chance floor (`human_placement.json`, axis 1). Mode C separates the two groups.
- **What would answer it.**
  1. Name axes by a stated rule, and print the full top 10 with shares in the paper.
  2. Flag content-incoherent members in the text.
  3. Show the ARC-AGI / VPCT / GPQA contrast in A and in C.
- **Cost.** Minutes to hours. No refit.

### R6. The human placement still rests on plug-ins, misfits its own scores, and its content filter was edited after seeing placements. (revised from old R3)

- **Claim at risk.** Every tier position and crossing date on the K=4 axes.
- **Evidence.**
  - The committed `human_placement.json` was computed on the 50-chain trace, before the step-0 fix. 24 / 35 scores fall outside the 80% band of the predicted mean (axis 1: 20 / 25; axis 2: 1 / 3; axis 3: 0 / 1; axis 4: 3 / 6). The band omits observation noise, so this overstates misfit.
  - Misses noise cannot explain:

    | Tier | Benchmark | Observed | Predicted |
    |---|---|---|---|
    | Skilled Generalist | GSM8K | 0.968 | 0.719 |
    | Domain Expert | WMDP Chemistry | 0.433 | 0.769 |
    | Domain Expert | WMDP Biology | 0.605 | 0.865 |
    | Domain Expert | LAB-Bench LitQA2 | 0.70 | 0.445 |
    | Domain Expert | DTBench | 0.987 | 0.802 |
    | Average Human | VPCT | 1.0 | 0.333 |

  - Thin axes: the agentic placement rests on 3 scores, the cyber placement on 1 (Cybench).
  - Unchanged method: plug-in medians of A, D, φ and d; argmax axis assignment (45 / 107 benchmarks have no majority axis); off-axis profile from a regression on shrunk model medians; model φ_b reused for human aggregates.
  - Content judgments changed after placements were seen:
    - ARC-AGI excluded from the science axis, so the Average Human and the Committee of Average Humans are no longer drawn there (f4a3fc0);
    - the Average Human's MMLU taken off g, so the g placement went from r̂ 1.74 to 1.00 (2cd9e57);
    - several pairs "judged by Claude, to validate" (3c6f464).
  - Answered since 10-29:
    - 0 unjudged pairs remain on the current titles (42 pairs judged);
    - the ceiling artifact is fixed by the "at least this score" rule. That fix is uncommitted, absent from `docs/model_math.md` §8, and tested only on mode A of the 5-chain refit. There, Top Performer on axis 4 becomes a lower bound (4.32) instead of a point at 6.41 [4.10, 9.33].
- **What would answer it.** The old six checks stand, in this order:
  1. Pseudo-human validation on held-out models.
  2. Leave-one-benchmark-out prediction of human scores.
  3. Placement on 100 to 200 posterior draws, per mode.
  4. A predictive band with noise.
  5. The joint-fit comparison on the ceiling spec.
  6. A sensitivity table (content filter off, default drop).

  Also: freeze the content file before the final placement, and list every edit made after a placement was seen.
- **Cost.** Hours on the trace for items 1-4 and 6. One refit for item 5.

### R7. Axis 1 is partly an evaluation-campaign factor. (new, from step 0)

- **Claim at risk.** The science axis as a capability, and the human tiers placed on it.
- **Evidence.**
  - The 15 RAND rows share a residual-correlation excess of about +0.10 in K=1 (+0.044 against −0.057 over 105 pairs), and K=4 absorbs it (0.005 to 0.013) (`6_writeups/preprint/step0/nested_audit/NOTE.md`).
  - On the 50-chain mode A table, 13 / 15 RAND benchmarks are dominated by axis 1, and 7 of axis 1's top 10 by share are RAND rows (step 0 found 6 on the 5-chain refit).
  - 15 / 25 human scores that place tiers on axis 1 are on RAND benchmarks.
  - FrontierMath T1-3 / T4 v2 (residual correlation 0.43 to 0.63, 96th to 99th percentile) and AISI CTF Practitioner / Expert (0.60 to 0.67, n = 10) are not absorbed and build no axis.
- **What would answer it.** R8 of the run list: the flagship spec without the 15 RAND benchmarks. Report axis-1 content, its frontier slope and the tier placements with and without them. A campaign random effect on η is the alternative.
- **Cost.** One refit: 10 chains for a first read (about 3 h), 50 for the paper (about a day).

### R8. The crossover forecast is a long extrapolation of a short, selected, partly prior-driven series, now read in one mode only. (old R4, unchanged in code)

- **Claim at risk.** Every crossing date.
- **Evidence.**
  - `analysis/regimes.py` and the lineage drift prior are unchanged since 10-29.
  - The line is fitted on top-2 frontier points since 2024-10, about 2 years, and projected about 3.2 years to 2030.
  - The band is the mean line's credible band, not a predictive band. The dispersion grid stops at s = 0.5. Serial dependence is ignored.
  - Points are posterior medians, so mode and placement uncertainty are dropped.
  - Selection on the inputs:
    - winner's curse from `family_best` (438 / 988 versions carry an effort suffix);
    - a HalfNormal(1) positive drift prior;
    - benchmark-coverage selection.
  - The archived backtest overpredicted the axis-1 frontier by 0.33 logits, with 4 / 11 records inside the 50% band.
  - Crossing dates moved by 8 to 24 months between chain groups on the blog-era fit.
  - New: README says "no other mode stands out enough to be drawn beside it", so the forecasts are mode A only while C holds 19 / 50 chains (R1).
- **What would answer it.**
  1. Rolling-origin refit backtest at two cutoffs.
  2. A predictive band with dispersion and AR(1).
  3. Per-draw crossing dates for A and C.
  4. Sensitivity rows: `fit_start`, top_k, break rule, symmetric drift.
  5. No backcast dates; conditional wording.
- **Cost.** Items 2-5: hours on the trace. Backtest: 2 refits. Symmetric drift: 1 refit.

### R9. Specification and analysis choices keep being made after seeing results. (old R6, expanded)

- **Claim at risk.** Every headline that depends on a choice with alternatives.
- **Evidence.**
  - Old items stand: skew-normal θ adopted partly for chain placement; forecast rules dropped the same day; the top-2 rule; outcome-driven canonical exclusions.
  - New since 10-29:
    - **Mode selection rule differs by fit.** The flagship reads the best log-density mode (code default). The free bifactor and `--general-sums` read mode D, chosen on "best integrated LOO, and its g is not empty, unlike the best-logp mode A (user, 2026-11-02)". Earlier, `--general-sums` read chains {0, 2}, with chain 6 dropped for divergences.
    - **Axis titles renamed repeatedly** on the exploration fits within four days (dcd5a7c, 01ee114, e25770a, ee55e97, 955f20d).
    - **Human content judgments edited after placements** (R6).
- **What would answer it.**
  1. One pre-stated mode-selection rule for all fits.
  2. A dated freeze note covering spec, chain rule, axis naming rule and content file, before the final run.
  3. A specification-curve appendix.
- **Cost.** Hours.

### R10. The evidence chain is not reproducible from the repo, and step-0 K=4 checks ran on a different fit. (new)

- **Claim at risk.** Every LOO, convergence and placement number quoted from the 50-chain fit.
- **Evidence.**
  - No K=4 trace is on this machine.
  - `5_outputs/**/diagnostics/` is gitignored and absent, so the per-mode integrated-LOO CSVs and the robust-LOO outputs behind +56 ± 12 cannot be inspected.
  - Deleted traces: the no-ceiling traces, the 20-chain trace and the 10-chain seed-42 ceiling trace that produced the robust LOO (`investigation/README.md`).
  - Scripts outside the repo: the less-compensation scripts sit in a session scratchpad and in `traces_backup/`.
  - Step-0 K=4 checks ran on mode A (2 chains), B (1) and C (2) of the failed 5-chain refit, not on the flagship: the nested audit, the placement before/after and the MAP comparison baseline. Their mode labels are not shown to match the flagship's.
  - Float32 storage is not a concern: quantiles move by 1e-5 of their Monte Carlo error, and the LOO check tolerance of 1e-3 per point bounds the total at about 0.2 nats.
- **What would answer it.**
  1. Archive the 50-chain trace with a hash (Zenodo or OSF).
  2. Un-ignore and commit the small diagnostic CSVs and JSONs.
  3. Move the analysis scripts into the repo.
  4. Rerun the step-0 audits on the flagship trace.
- **Cost.** Hours, plus the trace transfer.

### R11. K=4 has no same-data, same-spec support. (old R5)

- **Claim at risk.** "Four" bundles.
- **Evidence.**
  - Still no K = 1..6 ladder at the current spec. In-sample only: K=1 RMSE 0.068 against K=4 0.043 (`mirt_gof_k1.json`, `mirt_gof_k4.json`).
  - New: the K=5 bifactor fits reach RMSE 0.0395 (free) and 0.0410 (`--general-sums`), but split into 9 and 8 modes over 10 chains, nearly one chain per mode. Their LOO is not computed.
  - Nested families are now audited (R7). Shared questions add no residual correlation (−0.057 over 30 pairs, equal to the baseline). That part of old R5 is answered.
- **What would answer it.** Hold-out-cell K-fold CV for K = 3, 4, 5 at fixed spec and fixed sampler budget, on two splits (random 5% of cells, newest 10% of models). Otherwise state that K is chosen for interpretability.
- **Cost.** Days of refits. A MAP screen is ruled out, because MAP lands in the noise collapse (see section 2).

### R12. Ceilings are mostly prior, and the evidence for them comes from a different fit. (old R11, revised)

- **Claim at risk.** "Estimated ceilings improve held-out prediction by +56 ± 12 nats".
- **Evidence.**
  - 8 / 85 ceilings are data-informed. The MBF agreement is partly circular.
  - The robust LOO compared mode B (2 chains) of the 10-chain seed-42 ceiling fit against mode A of the 20-chain no-ceiling fit. That ceiling fit still carried the SimpleQA Verified human bound (0.944), later withdrawn as mis-measured (095dfc0). SimpleQA Verified was the largest per-benchmark loss (−12 nats). Both traces are deleted.
  - The R2 sensitivity fit (no ceilings) has 20 chains on seeds 42-43; the flagship has 50 on seeds 44-48.
  - Human scores still bound ceilings in the models-only fit (`data.load_observed_best`). A human score is above every model's on 21 benchmarks (LMCA, OpenBookQA, HellaSwag and others).
- **What would answer it.**
  1. Recompute the robust LOO on the 50-chain flagship against a like-for-like 50-chain no-ceiling fit.
  2. Add a sensitivity with ceilings bounded by model scores only.
  3. List the 8 informed ceilings.
- **Cost.** One 50-chain refit (no ceilings) plus robust-LOO runs (hours each).

### R13. Frontier gaps: the lag is weakly identified and the private-vs-public contrast is confounded. (old R7, unchanged)

- **Claim at risk.** The open/closed and US/China lags on ECI-H.
- **Evidence.**
  - Nothing changed since 10-29.
  - Lags rest on 14 to 17 record points; lag draws are discrete; thin US leaders drive the tail.
  - The private-minus-public contrast compares disjoint benchmark sets.
  - `family_best` favours variant-rich vendors.
- **What would answer it.**
  1. Make the time-averaged per-draw lag the headline.
  2. A specification curve over the record rule, top_k and leader cap.
  3. A composition control.
  4. A one-effort-per-family variant.
- **Cost.** Hours.

### R14. Data provenance biases levels in known directions. (old R8)

- **Claim at risk.** Score levels, and so the abilities built on them.
- **Evidence.**
  - Unchanged: 1,430 / 5,588 rows (26%) independently re-run; RAND 565 rows digitised; dedup policy "max"; 438 / 988 test-takers are effort variants.
  - The SimpleQA Verified human bound is withdrawn. The other human bounds remain (R12).
- **What would answer it.**
  1. A provenance table.
  2. A source-restricted refit or a source effect on η (R7 overlaps).
  3. A dedup-mean refit.
- **Cost.** One refit each.

### R15. Beta likelihood, boundaries and N_b. (old R10, unchanged)

- **Claim at risk.** The likelihood on the 23 continuous benchmarks and on the boundary rows.
- **Evidence.**
  - The 23 continuous benchmarks use Beta with exact zeros moved to 0.005.
  - Boundary rows: 87 zeros and 19 ones out of 5,487.
  - `zero_pred_below_threshold` is 0.552. Vendor scores are not item counts.
- **What would answer it.**
  1. A zero-placement sensitivity (0.001 / 0.01, or the 23 dropped).
  2. PIT grouped by benchmark family and source.
- **Cost.** Hours on the trace, or one refit.

### R16. Calibration and standard workflow checks. (old R12, partly answered)

- **Claim at risk.** The paper's calibration and workflow claims.
- **Evidence.**
  - LOO-PIT variance 0.0666 against 0.083, KS p ≈ 0, on 4,528 / 5,487 rows. The 853 rows with k > 0.7 are excluded, so the PIT sample is selected.
  - Missing:
    - a prior predictive check;
    - a recovery study at the current spec and missingness;
    - PIT grouped by benchmark, year and source;
    - a prior-sensitivity table (drift, skew, ceiling prior, σ_b floor, loading scale).
  - Done: the multi-start MAP check (see section 2).
- **What would answer it.** Prior predictive (minutes); grouped PIT (hours); recovery (one refit); sensitivity rows (refits, placement-only rows minutes).
- **Cost.** As listed.

---

## 2. Objections answered or dropped since 2026-10-29

- **Old R9, noise collapse: answered for sampling.** MAP was rerun on 60 starts: the collapse is present at 60 / 60 optima, with 76 to 87 / 107 benchmarks at σ_b < 0.01. But 10 / 10 NUTS chains started at the best optima leave it during tuning (`step0/map_search/NOTE.md`). What remains moves to R16 and the methods text: state that optimiser-based summaries are invalid, and report σ_b ≥ 0.005 as a prior choice.
- **Old R3, 19 unjudged benchmark-axis pairs: answered.** 0 unjudged on the current titles; the rule for new titles is "keep with a warning".
- **Old R3, LMCA ceiling artifact: answered in code**, uncommitted (R6).
- **Old R1, three hand-picked chain subsets in the docs: answered for the flagship.** Deliverables read the best mode by code (`best_mode_chains`), and README states all five modes. Bifactor fits still hand-select (R9).
- **Old R2, wishlist on the old fit: answered.** Regenerated for the five 50-chain modes.
- **README showing `_noceiling` results under the ceiling name: answered.**
- **Old R5, nested families as a source of axes: answered for shared questions.** The RAND campaign part moves to R7.
- **Old R12, "integrated LOO gives founders a spurious prior": answered.** Fixed in 211e576, with tests. All mode comparisons were recomputed.
- **Float32 and one-in-two storage: not an objection.** The cost of the old one-in-five thinning on the existing trace sits in R2.

---

## 3. Minimum work plan before submission

1. **Retrieve and archive the 50-chain trace** (R10). Without it, items 2 to 5 are blocked.
2. **On the trace, hours each:**
   - whole-fit and per-mode convergence (R2);
   - A against C at equal chains (R1);
   - the dominance audit (R3);
   - the per-axis recurrence criterion (R4);
   - the full top-10 tables (R5);
   - the placement rerun with the committed fix, then pseudo-human, LOBO and draw-propagated placement, per mode (R6);
   - per-draw crossings for A and C, and a predictive band (R8);
   - grouped PIT (R16).
3. **Freeze note:** spec, mode rule, naming rule, content file (R9).
4. **Refits, one at a time, each with config stated and a go:**
   - no-RAND (R7);
   - 50-chain no-ceiling, like-for-like (R12);
   - K = 3 / 5 CV (R11);
   - backtest at two cutoffs (R8);
   - symmetric drift (R8);
   - recovery (R16).
5. **Fix the dashboard** to whole-fit diagnostics, or restate the rule (R2).

---

## 4. Claims the paper can make now

1. "On 5,487 benchmark scores of 887 models, a four-axis compensatory model reproduces observed scores with RMSE 0.043, against 0.068 for one axis." The cross-chain agreement figure (median 0.15 posterior SD) comes from the failed 5-chain refit. Recompute it on the 50-chain trace before quoting.
2. "The posterior has at least five modes over 50 chains. Two of them, reached by 21 and 19 chains, cannot be ranked by the data: they differ by 9 ± 4 nats of integrated LOO on 3,116 scores and are separated by few unobserved cells. They differ in whether mathematics and abstract reasoning share an axis with scientific knowledge."
3. "Agentic, cyber and commonsense-factual bundles recur across the main modes; the allocation of mathematics, abstract reasoning and scientific knowledge does not." Condition: the recurrence criterion of R4 passes.
4. "Under the compensatory sum, per-axis abilities can rank a model far below what its own scores on that axis's benchmarks suggest. Variants that remove this cost 44 to 104 nats and fragment into 9 to 14 modes."
5. "Estimated ceilings improved held-out prediction on an earlier 10-chain fit (+56 ± 12 nats on 4,967 scores). They are informed by the data for 8 of 85 benchmarks."
6. Frontier gaps on ECI-H: unchanged from 10-29. About 6 to 7 months over the last year, consistent with a constant lag since late 2024. The private-public difference is not separated from composition.
7. Human placements and crossing dates only as conditional statements, per mode (A and C). Ceiling scores are reported as lower bounds.

**Do not claim:**

- that A is the posterior's mode;
- converged per-axis intervals;
- mode probabilities from chain counts;
- that the axes are named by their content without listing the off-topic members;
- the ARC-AGI / GPQA contrast as resolved by the flagship;
- crossings dated before the data window;
- Top Performer levels on axis 4;
- private-vs-public gaps as contamination evidence.
