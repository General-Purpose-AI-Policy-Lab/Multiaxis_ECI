# Assumption ladder — notes

Scope for every fit: exploration set minus FrontierMath v1 and AlgoTune, floors
(fixed-c 3PL), pooled noise. 5,004 obs / 835 test-takers / 98 benchmarks.
Loading prior `normal` throughout.

## Scope gate

All five traces carry a byte-identical `observed_data["obs"]` vector, equal to
the vector `load_eci_data(include_all_benchmarks=True,
drop_benchmarks=["FrontierMath v1", "AlgoTune"])` + `clip_scores_to_floors`
produces today, clipped to `[ECI_EPS, 1-ECI_EPS]`. `np.array_equal` on all
5,004 entries, per fit. No pair is excluded. See `obs_check.json`.

## Convergence

Judged on identified quantities only (`eta` over a fixed 400-observation
subset, seed 0, plus `D`, `sigma_b`, `tau_CD`) — the `mirt_identified_rhat`
convention. Raw r-hat on `A` / `theta` / `tau_A` is permutation-inflated and is
not reported. Mode verdicts come from `diagnostics/diagnose_chains.py
--write-modes`, which wrote `mirt_modes_<trace-stem>.json` next to each of the
four traces that lacked one (K=1, step 1, step 2, K=3); the flagship K=4
already had one.

| step | chains x draws | divergences | eta min/median ESS | eta max r-hat | modes |
|---|---|---|---|---|---|
| 0 K=1 | 10 x 10,000 | 0 / 100,000 | 2,375 / 141,380 | 1.004 | 1 |
| 1 K=4 no priors | 12 x 3,000 | 778 / 36,000 | 21 / 617 | 1.554 | 2 |
| 2 K=4 + human | 12 x 3,000 | 80 / 36,000 | 67 / 20,952 | 1.114 | 2 |
| 3 K=4 + both | 10 x 20,000 | 78 / 200,000 | 44 / 79,266 | 1.142 | 1 |
| x K=3 + both | 12 x 3,000 | 12 / 36,000 | 27 / 2,673 | 1.352 | 2 |

Step 1's split is 6 chains vs 6 (A = 1,2,3,6,9,11; B = 0,4,5,7,8,10), Δlogp
-14.4, cross-mode matched loading corr 0.39. Step 2's is 5 vs 7 (A = 5,7,8,9,10;
B = 0,1,2,3,4,6,11), Δlogp -10.5, cross-mode corr 0.041; within each mode the
loading corr is 0.99, so the human order sharpens each solution without picking
between them. The flagship (both priors, K=4) is the only K>1 rung with one
solution across all 10 chains.

## The K=3 split

`mirt_modes_trace_mirt_k3_humanmerge_lineageprior_lineagebm_dropFrontierMathv1AlgoTune_floors_poolednoise.json`:
mode A = chains 0,1,2,3,4,5,6,7,10,11 (Δlogp 0.0, within-mode loading corr
1.0), mode B = chains 8,9 (Δlogp -77.1, within-mode corr 0.976, matched corr to
mode A 0.082). Mode A holds 10 of 12 chains and wins by 77 nats.

What trades (`k3_mode_loadings.csv`, `k3_mode_theta.csv`, means taken after
matching every chain onto its mode's first chain, then the two modes onto each
other by the best permutation and sign):

- Axis 1 is the same in both modes — cross-mode loading correlation 0.934.
  Axes 2 and 3 are what move (0.489 and 0.479).
- Three benchmark clusters compete for two residual axes: **easy 2021-era QA**
  (OpenBookQA, ARC (AI2), ScienceQA, BBH, GSM8K, MMLU), **frontier
  agentic / long-horizon** (GBAEval, ProofBench, FrontierCode, GDP.pdf,
  BlueprintBench 2, Cybench, SWE-Bench Verified), and
  **abstraction / visual / competition math** (VPCT, ARC-AGI, MATH Level 5,
  OTIS Mock AIME, SimpleQA Verified).
- Mode A puts easy QA and the agentic cluster on ONE axis (OpenBookQA 0.796
  axis share, GBAEval 0.785, ProofBench 0.544, FrontierCode 0.643) and gives
  the other residual axis to abstraction plus recall (VPCT 0.813, SimpleQA
  Verified 0.778, ARC-AGI 0.632, OTIS AIME 0.500).
- Mode B moves the agentic cluster onto the abstraction axis instead (GBAEval
  0.781, ProofBench 0.729, Cybench 0.727, SWE-Bench 0.557, VPCT 0.524, ARC-AGI
  0.442) and leaves easy QA with recall (OpenBookQA 0.781, ARC (AI2) 0.713,
  SimpleQA Verified 0.486).
- The permutation-invariant version of the same statement: on the cosine
  similarity of each benchmark's axis-share profile, only 1.2% of the 4,753
  benchmark pairs move by more than 0.5, and mean |change| is 0.109. The pairs
  that move are exactly the easy-QA-to-agentic ones: OpenBookQA ~ GBAEval
  0.99 -> 0.20, ARC (AI2) ~ ProofBench 0.96 -> 0.24, and the reverse for
  easy-QA-to-recall: ARC (AI2) ~ SimpleQA Verified 0.25 -> 0.90.
- Ability shifts concentrate where coverage is thin. Per-axis mean |Δtheta| is
  0.171 / 0.555 / 0.606. The largest single shifts are 2021-2023
  easy-benchmark-only models (phi-2 3.80, Phi-3-small 2.87, gpt-4-0314 2.77,
  chatglm2-6b 2.50) and the two one-observation human tiers (High School
  Qualifier 3.03, High School Top Performer 2.90). The nine human tiers move
  0.52 to 3.03.

## LOO

`az.loo(pointwise=True)` on the trace's own `log_likelihood`.

**Thinning.** The flagship log-likelihood group is 10 x 20,000 x 5,004
(8 GB in float64) and the machine has 24 GB, so draws are thinned by a uniform
stride to about 1,500 per chain: step 0 stride 6 (16,670 draws), steps 1/2 and
K=3 stride 2 (18,000), flagship stride 13 (15,390).

**reff.** arviz derives the PSIS `reff` from the posterior group, which is
38 GB on the flagship. Instead `reff` is the relative ESS of the pointwise log
densities (the `r_eff_log_lik` convention of the loo R package), computed the
same way for every fit.

**Sensitivity** (`ladder_loo_sensitivity.csv`). `reff` is inert: forcing 0.05,
0.5 or 1.0 moves elpd by at most 8 nats on any fit. Thinning is not inert and
biases elpd upward as draws shrink — halving the draws again adds +12 (step 0),
+52 (step 1), +41 (step 2), +33 (step 3), +25 (K=3). Absolute elpd here is
therefore an upward-biased estimate at these draw counts. The known
full-draw flagship value, 7711.20 +- 76.46, sits below the thinned 7799.24
reported here, consistent with that direction. The paired deltas are stable:
at twice-coarser thinning (`ladder_pairwise_thin2.csv`) step1-step0 is 1391.81
vs 1352.24, step2-step1 -18.08 vs -7.28, step3-step2 172.03 vs 180.29,
step3-step0 1545.76 vs 1525.25, step3-K3 339.51 vs 331.04 — every move well
inside the pair's own SE.

**PSIS strain.** About 5.4 observations per test-taker, so Pareto-k is bad on a
large minority of rows (k > 0.7 on 332 / 5,004 at K=1 rising to 1,543 / 5,004
at step 1) and every fit raises the arviz `warning` flag. p_loo runs
1,176 to 2,470 against 5,004 observations. Read the k-restricted delta column
next to the full-row one; both are reported.

## Paired comparisons

`ladder_pairwise.csv` (full 5,004 rows) and the k-restricted sensitivity in the
same file. `ladder_compare.csv` holds `az.compare(..., ic="loo")` on the five
ELPDData objects — valid because the scope gate proved identical observations.
az.compare ranks step3 > step1 > step2 > K=3 > step0 and puts 0.68 stacking
weight on step 3, 0.30 on step 1, 0.02 on step 2, 0.00 on the other two.

## Files

| file | content |
|---|---|
| `ladder.py` | the table: scope gate, identified convergence, LOO, paired deltas |
| `sensitivity.py` | thinning and reff sensitivity |
| `k3_modes.py` | per-mode loadings and abilities for the K=3 split |
| `ladder_table.csv` | the headline table |
| `ladder_pairwise.csv` | paired dELPD, full rows and k<0.7-on-both rows |
| `ladder_compare.csv` | `az.compare` output |
| `ladder_loo_sensitivity.csv`, `ladder_pairwise_thin2.csv` | sensitivity |
| `k3_mode_loadings.csv`, `k3_mode_theta.csv` | K=3 mode forensics |
| `loo_pointwise.npz`, `obs_check.json` | pointwise elpd_i / Pareto-k, scope gate record |
