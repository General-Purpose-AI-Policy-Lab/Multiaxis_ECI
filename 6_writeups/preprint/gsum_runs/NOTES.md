# Gsum local refit: K=4 bifactor, axes = g + deviations, models only, lineage prior with Brownian steps

Goal: refit the upstream 50-chain gsum fit (955f20d, Jérémy, 2026-11-02; trace not on this machine)
as five runs of 10 chains, seeds 44-48, merged with `--merge-chains`, the R1 recipe
(`../r1_runs/NOTES.md`). Decision 10 of the preprint plan (2026-11-08).

## Checks before run 1 (2026-11-08)

- Spec parsed by `fit.py`'s own parser, no sampling (`--K 4 --loading-prior bifactor --general-sums
  --lineage-prior --lineage-bm --chains 10 --tune 2000 --draws 2000 --seed 44`, everything else default):
  tag `_bifactor_lineageprior_lineagebm_gsum`, floors, ceiling noise, skew-normal theta, pooled noise,
  Beta-binomial on item rows, effort means, release nodes, models only (no humans), save thin 2,
  streamed draws.
- Data: 5,487 obs / 887 models / 107 benchmarks (101 single-benchmark test-takers dropped; 72
  below-floor scores clipped to the chance floor). Same counts as R1.
- `1_curated/lineage_map.csv` is byte-identical to the fit folder's `lineage_map_fitted.csv`.
- Code since 955f20d touching `mirt.py`, `fit.py`, `fitspec.py`: 4926945 (`--orthogonal`, opt-in) and
  2789fef (orthogonal review fixes, `--merge-chains` error message). Nothing on the gsum default path.
- Sampler: nutpie 0.16.8, pymc 5.28.5. Upstream versions and seeds are not recorded; the upstream modes
  file names mode D "on the seed-44 frame", so seed 44 is shared at least.
- Upstream tables backed up to
  `5_outputs/data20261019/traces_backup/mirt_bifactor_lineageprior_lineagebm_gsum_upstream_20261108/`
  (20 files, `diff -r` identical): tables, both `mirt_modes_*.json`, `axis_names.json`,
  `human_placement*`, `placement_tests/`. The live modes file carries `selected_chains` for upstream
  mode D (chains 0,4,6,12,14,22,25,26,35,38,46,49, chosen on integrated LOO and a non-empty g); the fit
  renames it with a date stamp rather than deleting it, and the backup is the second copy.
- Machine: 15 logical cores, 24 GB RAM, 185 GB free disk, AC power, run under `caffeinate -i`.

## Runs

| run | seed | chains after | wall time | divergences | modes (chains) |
|---|---|---|---|---|---|
| 1 | 44 | 10 | 2 h 31 min net of a 2 h 26 min pause (K=4 sampling 2 h 21, K=1 baseline 7.1 min) | 89 / 20,000 (unthinned) | A {5}, B {0,2} −14.0, C {4,8} −14.3, D {1,3,7,9} −45.3, E {6} −47.5 |
| 2 | 45 | 20 | 1 h 58 min (K=4 sampling 1 h 56, K=1 cached) | 98 / 20,000 this run; 187 / 40,000 in all | A {5,10,15}, B {18} −15.2, C {0,2,17} −23.1, D {4,6,12,14} −28.3, E {1,3,7,8,9,11,13,16,19} −49.4 |

## Run 1 (seed 44), 2026-11-11 14:32-19:29

- Command: `fit.py --K 4 --loading-prior bifactor --general-sums --lineage-prior --lineage-bm --chains 10 --seed 44 --tune 2000 --draws 2000`,
  under `caffeinate -i` (lid open, AC). Log `run1_seed44.log`; times `run1_seed44.times`; modes
  `modes_after_run1.json` (`run1_modes.log`); comparison `run1_vs_upstream.txt` (script `compare_upstream.py`).
- **Pause.** The sampler was suspended (SIGSTOP) 26 s after its start, on Yassine's request: another
  project was running a local `llama-server`. It was resumed (SIGCONT) at 16:58:31, once that server had
  been absent for 10 min. Pause 14:32:44-16:58:31, 8,747 s. The process was suspended, not altered:
  draws do not depend on wall time. The log's "sampling done in 17,210.8 s" includes the pause;
  net of it, 8,464 s.
- Wall time 4 h 57 min in all, 2 h 31 min net. CPU time 64,094 s user.
- Peak memory 5.3 GB (streamed draws). Trace 579 MB (1,000 saved draws per chain, thin 2). K=1 baseline
  sampled once (0 / 20,000 divergences), cached for runs 2-5.
- Divergences 89 / 20,000 unthinned; 42 / 10,000 on the saved draws.
- Identified r-hat over all 10 chains: eta max 1.244, D 1.302, sigma_b 1.336 (several modes).
- 5 modes on 10 chains: A {5}, B {0,2} −14.0, C {4,8} −14.3, D {1,3,7,9} −45.3, E {6} −47.5
  (Δ in nats of mean chain log density below A's −23,401.0).
- **Not a chain-for-chain reproduction of upstream chains 0-9** (same index read as same seed and chain;
  upstream seeds are not recorded). |local − upstream| chain log density: median 17.9 nats, max 52.2
  (chain 4: −23,422.1 here, −23,474.3 upstream).
- **Upstream mode D's loadings are reached by 2 / 10 chains.** Per-chain median loadings against the
  upstream mode D table, best axis permutation, minimum axis correlation: chain 4 0.989 and chain 6 0.984;
  the other 8 chains 0.647-0.820. Chains 4 and 6 are also upstream mode D chains by index. They sit in
  two different local modes (C and E), 33 nats apart in mean log density (−23,422.1 and −23,448.5).
- g against g: the local g column correlates with upstream mode D's g at 0.992-0.998 in 4 / 10 chains
  (2, 4, 6, 8) and at 0.228-0.454 in 6 / 10 (0, 1, 3, 5, 7, 9), where upstream's g is matched by a local
  specific column instead (correlation 0.953-0.965).
- The raw trace keeps each chain's own order of the specific axes, so `compare_upstream.py` aligns a
  mode's chains onto its best-logp chain before taking the mode's pooled median.
- Trace after run 1 copied to `5_outputs/data20261019/traces_backup/gsum_local/trace_after_run1_seed44.nc`
  (`cmp` identical).
- The fit set the upstream modes file aside as `mirt_modes_*_gsum_2026-11-02.json` (byte-identical to the
  backup copy, `selected_chains` intact) and wrote a new 5-mode file without a selection.
- The fit folder now holds run 1's whole-fit tables (`mirt_loadings.csv`, `mirt_factor_corr.csv`,
  `mirt_tau_spectrum.csv`, `mirt_gof_k4.json`, plus new `mirt_factor_scores.csv`, `mirt_residuals.csv`,
  `mirt_gof_k1.json`, `trace_mirt_k1.nc`) beside the upstream `human_placement*`, `axis_names.json` and
  `placement_tests/`, which the fit does not rewrite.

## Run 2 (seed 45, merged onto run 1), 2026-11-12 10:47-12:46

- Command: run 1's plus `--seed 45 --merge-chains .../gsum_local/trace_after_run1_seed44.nc`. Log `run2_seed45.log`;
  times `run2_seed45.times`; modes `modes_after_run2.json` (`run2_modes.log`). No pause.
- Wall time 1 h 58 min (7,111 s; K=4 sampling 6,958 s). CPU time 56,718 s user.
- Peak memory 4.4 GB. Merged trace 1.16 GB (20 chains x 1,000 saved draws), copied to
  `traces_backup/gsum_local/trace_after_run2_seed45.nc` (`cmp` identical).
- Divergences 98 / 20,000 this run; 187 / 40,000 over both runs (unthinned).
- The console's divergences and identified r-hat (eta max 1.282, D 1.373, sigma_b 1.287) cover run 2's 10 chains
  only: they are printed before the merge.
- 5 modes on 20 chains: A {5,10,15}, B {18} −15.2, C {0,2,17} −23.1, D {4,6,12,14} −28.3,
  E {1,3,7,8,9,11,13,16,19} −49.4 (Δ in nats of mean chain log density below A's −23,395.7).
- Mode labels are re-derived on every merge (ordered by log density), so run 1's labels do not carry over:
  chain 8, in run 1's C with chain 4, is in E on 20 chains; chains 4 and 6, in run 1's C and E, share D.
- The run-1 modes file (no selection) was replaced by the 20-chain one. The upstream modes file
  `mirt_modes_*_gsum_2026-11-02.json` is still byte-identical to the backup copy.
