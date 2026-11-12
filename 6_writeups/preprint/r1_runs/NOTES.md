# R1 local refit: K=4 ceiling flagship, models only, lineage prior with Brownian steps

Goal: reproduce the upstream 50-chain fit (1f35d7d, Jérémy, 2026-10-23) as five runs of
10 chains, seeds 44-48, merged with `--merge-chains`.

## Checks before run 1 (2026-11-03)

- Spec parsed by `fit.py`'s own parser, no sampling: tag `_lineageprior_lineagebm`,
  ceiling noise, floors, skew-normal theta, pooled noise, Beta-binomial on item rows.
- Data: 5,487 obs / 887 models / 107 benchmarks (101 single-benchmark test-takers dropped).
- `1_curated/lineage_map.csv` is identical to the upstream `lineage_map_fitted.csv`.
- `mirt.py` changes since 1f35d7d are opt-in options only (`--theta-general`,
  `--general-sums`, `--orthogonal`, ...). Default change since then: `SAVE_THIN` 5 -> 2 and
  float32 traces. Thinning is applied after sampling, so the draws do not depend on it.
- Sampler here: nutpie 0.16.8, pymc 5.28.5. The upstream versions are not recorded on disk.
- Upstream tables backed up to
  `5_outputs/data20261019/traces_backup/mirt_lineageprior_lineagebm_upstream_20261103/`.
  The fit deletes the upstream modes JSON (it has no `selected_chains`) and rewrites the
  tables, so the backup is the only upstream copy on this machine.
- Machine: 15 logical cores, 24 GB RAM, 232 GB free disk.

## Runs

| run | seed | chains after | wall time | divergences | modes (chains) |
|---|---|---|---|---|---|
| 1 | 44 | 10 | 2 h 13 min (K=4 sampling 2 h 03, K=1 baseline 7.4 min) | 204 / 20,000 (unthinned) | A {1,4,8}, B {7} −20.4, C {0,3,5} −26.3, D {2,6,9} −27.9 |
| 2 | 45 | 20 | 2 h 07 min (K=4 sampling 2 h 05, K=1 cached) | 345 / 20,000 this run; 549 / 40,000 in all | A {1,4,12,15,17,18}, B {0,8,10,16} −18.3, C {7,11,19} −21.5, D {2,6,9} −30.9, E {3,5,13,14} −34.4 |
| 3 | 46 | 30 | 4 h 35 min (K=4 sampling 4 h 33; same CPU time as run 2, the laptop slept) | 518 / 20,000 this run; 1,067 / 60,000 in all | A {1,4,8,12,15,16,17,18,22,25,27,28,29}, B {7,11,19,20} −19.1, C {0,3,5,10,13,14,21,23,24,26} −23.7, D {2,6,9} −27.6 |
| 4 | 47 | 40 | 2 h 33 min (K=4 sampling 2 h 30, under `caffeinate -i`) | 355 / 20,000 this run; 1,422 / 80,000 in all | A 17 chains, B {11,15,19,25} −7.6, C {7,20} −18.8, D 14 chains −20.3, E {2,6,9} −25.3 |
| 5 | 48 | 50 | 2 h 06 min (K=4 sampling 2 h 03, under `caffeinate -i`) | 224 / 20,000 this run; 1,646 / 100,000 in all | A 24 chains, B {7,11,19,20} −18.9, C 18 chains −21.3, D {2,6,9,43} −29.0 |

## Run 1 (seed 44), 2026-11-03 12:48-15:01

- Command: `fit.py --K 4 --lineage-prior --lineage-bm --chains 10 --seed 44 --tune 2000 --draws 2000`.
  Log `run1_seed44.log`; modes `modes_after_run1.json` (`run1_modes.log`); comparison `run1_vs_upstream.txt`.
- Peak memory 6.1 GB (streamed draws). Trace 566 MB. K=1 baseline sampled once (0 divergences), cached for runs 2-5.
- Identified r-hat over all 10 chains: eta max 1.156, D 1.247, sigma_b 1.209 (several modes, expected).
- **Not a chain-for-chain reproduction.** Every chain's log density differs from upstream chain 0-9's
  (for example chain 1: −22,675.8 here, −22,702.2 upstream), and the mode partition differs
  (here A is {1,4,8}; among upstream chains 0-9, A was {0,6}).
- **Upstream mode A is reached.** Local mode A's chains match upstream mode A's median loadings at a
  minimum axis correlation of 0.932-0.997 (best axis permutation), and its mean chain log density,
  −22,676.1, matches upstream mode A's −22,677.8. Whether local B-D are upstream's B-D cannot be
  checked from disk: upstream kept tables for mode A only.
- Trace after run 1 copied to `5_outputs/data20261019/traces_backup/r1_local/trace_after_run1_seed44.nc`.
- The fit folder now holds run 1's whole-fit tables beside the upstream `human_placement*` and
  `axis_names.json`, which the fit does not rewrite.

## Run 2 (seed 45, merged onto run 1), 2026-11-03 15:03-17:10

- Command: run 1's plus `--seed 45 --merge-chains .../r1_local/trace_after_run1_seed44.nc`. Log `run2_seed45.log`;
  modes `modes_after_run2.json` (`run2_modes.log`).
- Peak memory 6.3 GB. Merged trace 1.13 GB, copied to `traces_backup/r1_local/trace_after_run2_seed45.nc`.
- The console's divergences and identified r-hat (eta max 1.141, D 1.206, sigma_b 1.217) cover run 2's 10 chains
  only: they are printed before the merge.
- Mode labels are re-derived on every merge (ordered by log density), so run 1's labels do not carry over:
  chain 8, in run 1's A, is in B on 20 chains.

## Run 3 (seed 46, merged onto 20 chains), 2026-11-03 17:18-21:53

- Command: run 1's plus `--seed 46 --merge-chains .../r1_local/trace_after_run2_seed45.nc`. Log `run3_seed46.log`;
  modes `modes_after_run3.json` (`run3_modes.log`).
- Wall time 16,522 s against run 2's 7,648 s, for the same CPU time (52,481 s user against 52,414 s). The power
  log shows the cause: lid-closed sleep from 18:30, "Thermal Emergency Sleep" at 18:48 and 20:23, idle sleeps
  on battery after 21:02. The process was suspended, not altered: draws do not depend on wall time.
- Peak memory 8.3 GB. Merged trace 1.70 GB, copied to `traces_backup/r1_local/trace_after_run3_seed46.nc`.
- Run 3's own 10 chains: 518 divergences, eta r-hat max 1.168, D 1.258, sigma_b 1.229.
- On 30 chains, mode detection re-run on the union regroups the 20-chain split: its B {0,8,10,16} splits
  between A (8, 16) and C (0, 10), its E {3,5,13,14} joins C, its C {7,11,19} becomes B, D is unchanged.
  4 modes: A 13, C 10, B 4, D 3.

## Run 4 (seed 47, merged onto 30 chains), 2026-11-04 11:02-13:35

- Command: run 1's plus `--seed 47 --merge-chains .../r1_local/trace_after_run3_seed46.nc`, prefixed with
  `caffeinate -i` (lid open, AC). Log `run4_seed47.log`; modes `modes_after_run4.json` (`run4_modes.log`).
- CPU time 68,226 s user, against about 52,400 s for runs 2 and 3: this seed's chains cost more per draw.
- Peak memory 10.2 GB. Merged trace 2.26 GB, copied to `traces_backup/r1_local/trace_after_run4_seed47.nc`.
- Run 4's own 10 chains: 355 divergences, eta r-hat max 1.070, D 1.191, sigma_b 1.065.
- 40-chain modes: A {1,4,8,12,16,17,18,22,27,28,29,31,33,34,36,37,38}, B {11,15,19,25},
  C {7,20}, D {0,3,5,10,13,14,21,23,24,26,30,32,35,39}, E {2,6,9}.
- Run 5 (seed 48) launched 2026-11-04 right after, on Yassine's advance go for it.

## Run 5 (seed 48, merged onto 40 chains), 2026-11-04 13:37-15:43

- Command: run 1's plus `--seed 48 --merge-chains .../r1_local/trace_after_run4_seed47.nc`, under
  `caffeinate -i`. Log `run5_seed48.log`; modes `modes_after_run5.json` (`run5_modes.log`).
- Peak memory 12.1 GB. Final trace 2.83 GB in the fit folder (not copied: it is the deliverable).
- Run 5's own 10 chains: 224 divergences, eta r-hat max 1.088, D 1.246, sigma_b 1.128.

## Mode counts as chains were added

| chains | modes | sizes (Δlogp, nats) |
|---|---|---|
| 10 | 4 | A 3, B 1 (−20.4), C 3 (−26.3), D 3 (−27.9) |
| 20 | 5 | A 6, B 4 (−18.3), C 3 (−21.5), D 3 (−30.9), E 4 (−34.4) |
| 30 | 4 | A 13, B 4 (−19.1), C 10 (−23.7), D 3 (−27.6) |
| 40 | 5 | A 17, B 4 (−7.6), C 2 (−18.8), D 14 (−20.3), E 3 (−25.3) |
| 50 | 4 | A 24, B 4 (−18.9), C 18 (−21.3), D 4 (−29.0) |

Labels are re-derived on every count, ordered by mean log density. Upstream 50 chains: A 21, B 5 (−9.6),
C 19 (−17.3), D 4 (−25.8), E 1 (−54.2).

## Did the chain set reproduce the upstream fit?

Per-chain comparison in `final_vs_upstream_per_chain.txt` (same chain index = same seed and chain number).

- **Seeds 45-48 (chains 10-49): yes, chain for chain.** Each chain's mean log density is within 4 nats
  of its upstream twin (median |difference| 1.0-2.0 nats per seed). The residual is Monte Carlo error
  of the mean over different thinnings (1,000 saved draws here, 400 upstream).
- **Seed 44 (chains 0-9): no.** Differences up to 39.9 nats (median 11.6); 6 / 10 chains sit in a
  different mode from their upstream twin. The cause is not established here.
- **Mode A is reproduced.** 24 chains at mean log density −22,676.6 against upstream's 21 at −22,677.8.
  Every local A chain matches upstream mode A's loading table at a minimum axis correlation of
  0.919-0.998 (median 0.991, best axis permutation per chain).
- Labels differ on 4 chains of seeds 45-48 (13, 15, 20, 25) although their log densities match: the
  mode split is re-clustered on a chain set whose seed-44 tenth differs. The upstream single-chain mode E
  (chain 13, −22,732.0) is chain 13 here too (−22,732.1), filed under C.
- Not checked: whether local B-D are upstream's B-D by loadings (upstream kept only mode A's tables).

## State of the fit folder after run 5

- `trace_mirt_k4_lineageprior_lineagebm.nc`: the 50-chain trace. `mirt_modes_*.json`: its 4-mode split.
- `mirt_loadings.csv` and the other tables: written by the fit on all 50 chains, not on mode A.
  The upstream deliverable tables read mode A (`diagnose_chains.py --write-tables` after a chain selection).
- `human_placement*`, `axis_names.json`: still upstream's, untouched by the fit.
- Intermediate traces after runs 1-4 (5.7 GB) in `5_outputs/data20261019/traces_backup/r1_local/`.

## Mode detection check (2026-11-05, on Yassine's question)

- **Axis matching.** `diagnose_chains._match_corr` did permute axes before comparing two chains, but
  greedily, axis by axis. On the 50 chains it understated 207 / 1,225 pairs (by up to 0.66) and was
  asymmetric on 259. Replaced by the exhaustive best permutation (K <= 6; Hungarian above), with a test.
  The splits after 10, 20, 30, 40 and 50 chains are identical under both rules: no pair crossed 0.8.
  The modes file's `matched_corr_to_best` changes: B 0.436, C 0.599, D 0.500 (greedy: 0.001-0.053;
  upstream's 0.002-0.020 are the same greedy artefact). `modes_after_run*.json` keep the greedy values.
- **Within-chain axis swaps: none.** Per chain, 100-draw blocks against the first block: where the
  identity match is poor, the best permutation is no better, so axes do not relabel inside a chain.
- **Chains move between modes.** Each 200-draw segment matched to each mode's reference loadings
  (`segment_assignment.txt`): 18 / 50 chains have a segment nearest another mode. Clear cases: chains 11
  and 19 (mode B) spend their first 400 draws nearest A (corr 0.99-1.00 against 0.75-0.83 to the
  runner-up); chain 10 (mode C) spends its last 400 nearest A. Segment shares: A 121, C 89, B 20, D 20
  of 250. The mode references themselves sit close to the 0.8 cut: A-B 0.77, A-C 0.78, B-C 0.64,
  A-D 0.65, C-D 0.42.

## Red team of the mode reading (Fable 5.1 agent, 2026-11-05, read-only)

Verdict on segment-level modes: weak. Its computed findings:
- The logp-basin rule never fires: largest gap between sorted chain-mean logps 15.6 nats (re-checked
  here), under the 20-nat cut. The split is loadings-only. Chain-mean logp SE is about 6 nats
  (per-draw sd 82, per-chain ESS median 177).
- The moves are real: 15 / 250 segments sit nearest another mode by a margin > 0.1, robust to rank
  correlation, to dropping the 10 largest loadings, and to leave-one-chain-out references. Every pair of
  modes has an observed within-chain transition: one connected posterior, mixing time about the chain length.
- The modes barely differ on identified quantities: eta A vs B median 0.19 posterior SD (A-C 0.13,
  A-D 0.21); chain-mean D correlated 0.99 across all 50 chains. 70-94 % of each inter-mode loading
  difference is a linear remix of the axes, mostly axes 2 and 4.
- Segment logp (SE 12-16 nats) cannot confirm or refute a move.
- The 0.8 cut sits on the noise floor (stable-A segment pairs reach down to 0.779).
Its reasoning-only points: upstream's "distinct axis solutions" rested partly on the greedy artefact
(corr to best 0.00-0.05, now 0.44-0.60). Seed 44 may differ because its upstream run predates
095dfc0 (2026-10-21, SimpleQA Verified baseline withdrawn); not verified.
Its suggestion: report eta, D, abilities and forecasts pooled over the 50 chains; report loadings as
one posterior with a flagged ridge; drop per-mode tables, or run longer chains before keeping them.
Nothing built on this yet.

## Identified convergence and aligned loadings (2026-11-05)

Scripts `aligned_loadings.py`, `eta_rhat_cells.py`; logs `aligned_loadings.log`, `eta_rhat_cells.log`; tables
`convergence_identified.csv`, `chain_alignment.csv`, `aligned_loadings_pooled.csv`, `aligned_loadings_by_chain.csv`,
`aligned_theta_rhat.csv`, `mode_remix.csv`, `eta_rhat_by_obs.csv`.

- eta, all 5,487 obs, 50 chains: r-hat max 1.448, median 1.005; 251 / 5,487 over 1.05 (127 / 887 models,
  45 / 107 benchmarks); bulk ESS min 97, median 16,251. Mode A only (24 chains): max 1.268, 20 / 5,487 over 1.05.
- D: 19 / 107 over 1.05 on 50 chains (max 1.206); 0 / 107 on mode A (max 1.043). sigma_b: 9 / 107 and 0 / 107.
- Divergences on saved draws: 812 / 50,000.
- Axes aligned per chain onto upstream's named axes: mode A chains match at median min |corr| 0.988; B 0.739,
  C 0.772, D 0.580. Aligned loadings over 50 chains: r-hat median 1.116, 306 / 428 cells over 1.05.
  Aligned abilities: 801 / 887 (axis 1) to 674 / 887 (axis 4) over 1.05. Per-axis abilities do NOT converge
  across modes, against the red team's "pool abilities" suggestion; only eta, and D within mode A, come close.
- Remix: each mode's loadings are 67-81 % a linear mix of mode A's axes. Science <-> Agentic trades in all
  three modes; Agentic <-> Cyber in B; Commonsense <-> Science/Agentic in C and D.
- Aligned r-hat per mode (`aligned_rhat_by_mode.log`): loadings over 1.05 in A 90 / 428, B 241, C 103, D 59;
  abilities over 1.05 in mode A: Science 297, Agentic 361, Cyber 314, Commonsense 181 (of 887). Per-axis
  quantities do not converge inside mode A either: chains drift along the ridge within a mode too.

## Red team 2: can the axis split be identified? (Fable 5.1 agent, 2026-11-05, read-only)

Verdict: no, not under exploratory-only plus positive loadings; a post-hoc frame does not fix it.
- Correction to the "exact trade" explanation: positivity already closes the exact A·T, θ·T^-T orbit.
  On mode-A median loadings at most 0.03-0.12 of one axis can be subtracted from another before a loading
  goes negative (re-checked here); the mode-to-mode mixes need 0.3-0.6. The modes differ in eta itself
  (rms 0.30-0.36 against mean cell SD 0.39): a soft likelihood ridge on sparse cells, not a gauge orbit.
- No near-pure benchmark (max axis share 0.68, OpenBookQA, re-checked); the NMF uniqueness conditions
  (separability, sufficiently scattered: 0.82 < 1) fail. The minimum-volume positive frame per draw moved
  abilities over r-hat 1.05 from 2,857 to 2,829 / 3,548 (thinned check): no effect.
- Per-axis r-hat is worse for well-measured models (max 1.64 at > 30 scores, 1.30 at 1-3).
- Positivity: needed for the meaning (a gain on any axis never lowers a score), but it does not pick axes.
- Untried prior with theory behind it: a volume prior inside the model (Arngren, Schmidt & Larsen 2011);
  no guarantee here.
- Its most defensible path: report invariants (per-benchmark predictions, dominance order, eta composites
  over benchmark families as per-capability series); per-axis abilities to a mode-flagged appendix.
  Failure mode: composites on families a model never took are prior-driven (unobserved cells: 11.7 % over
  r-hat 1.05, observed 3.2 %).

## Modes as packings of stable bundles (2026-11-05)

- Dominant axis per benchmark per mode (`dominant_axis_by_mode.txt`): 16 / 107 benchmarks keep the same
  dominant axis in all four modes. In B and D the agentic and cyber benchmarks share one axis; C moves the
  knowledge-recall benchmarks (MMLU, WMDP) from Science to Commonsense.
- Alignment-free bundles (`bundle_cores.txt`, `bundle_coassignment.csv`): two benchmarks are co-assigned in a
  chain when they share a dominant axis. Cores whose every pair is co-assigned in >= 90 % of 50 chains:
  10 groups of 3+, covering 59 / 107 (34 benchmarks alone). The six largest: cyber ranges and CTFs (12),
  math and science reasoning (12), knowledge recall (7), computer-use agents (6), factual QA (5),
  long-horizon agentic SWE (5). At >= 70 %: 12 groups covering 96 / 107.
- Reading: there are more stable bundles (about 6-7) than axes (4); a mode is one way of packing them into
  four axes. Which bundles share an axis is what the data barely decides.
