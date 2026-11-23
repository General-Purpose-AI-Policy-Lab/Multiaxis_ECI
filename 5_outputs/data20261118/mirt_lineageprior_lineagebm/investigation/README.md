# Flagship of the 2026-11-18 data: choosing the mode and naming the axes

K = 4, models only, skew-normal abilities, estimated ceilings, lineage prior + Brownian lineage (the flagship flags). 50 chains: five runs of 10 (seeds 44 to 48), 2,000 draws each after 2,000 tuning steps, every second draw saved. Sampled 2026-11-18 to 20, before five rate-scored benchmarks moved to `items` (commit `fcea240`): the item table it saw is frozen beside the trace as `benchmark_n_items_fitted.csv`, and the frozen table reproduces the trace's data fingerprint (`8be6cd6e7f6b4860`).

## Modes

`diagnose_chains.py --write-modes` (loading match above 0.8, no logp gap above 20 nats):

| Mode | Chains | Δ logp | Within-mode matched corr. |
|---|---|---|---|
| A | 24 | 0 | 0.85 |
| B | 1 | −1.0 | n/a |
| C | 17 | −1.2 | 0.89 |
| D | 8 | −62.0 | 0.94 |

Divergences per run: 72, 279, 206, 157 and 133 of 20,000 draws.

## Axes against the 2026-10-19 flagship

`2_match_axes_to_20261019.py` compares each mode's mean loadings (chains aligned within the mode) with the 2026-10-19 fit, over the 102 benchmarks the two fits share (min / mean matched |corr| over the four axes):

- **C carries the named axes**: 0.90 / 0.93 against the 2026-10-19 mode A, whose axes were named. Against the named display frame, 0.97, 0.90, 0.92 and 0.93 axis by axis, and every signature benchmark is among its axis's top nine by share.
- **A is the 2026-10-19 mode B** (0.91 / 0.93), the solution with agentic and cyber benchmarks together and a visual axis; its axis 2 matches the named agentic axis at 0.76 only.
- B and D share a third solution (0.95 / 0.97 to each other), at best 0.67 from any 2026-10-19 mode.

## Integrated leave-one-out

`4_diagnostics/integrated_loo.py`, each score's latent block integrated out, every chain scored alone at 1,000 draws and each mode the average of its chains; run with `1_integrated_loo_checkpointed.py`, which saves each chain's pointwise result as it goes (the script itself writes nothing until the last chain). Result: `integrated_loo_4modes_50chains.json`.

| Mode | elpd | Monte Carlo SE | vs A, every score | vs A, 3,939 scores with k < 0.7 in every mode |
|---|---|---|---|---|
| A | −17,706.5 | 4.6 | | |
| B | −17,692.0 | (1 chain) | +14.6 ± 25.9 | −44.8 ± 13.8 |
| C | −17,676.9 | 5.6 | **+29.6 ± 16.0** | **+11.3 ± 8.0** |
| D | −17,706.6 | 3.3 | −0.1 ± 23.9 | −31.2 ± 12.8 |

C leads A on both counts, by 1.8 and 1.4 standard errors; on the 2026-10-19 data the same pair of solutions gave the same verdict (the named solution ahead by 26.7 ± 7.6). About a quarter of the scores keep a Pareto k above 0.7 after integration (1,347 for C, 1,564 for A).

## Decision

User, 2026-11-23: mode C is the reference (`selected_chains` of the modes file) and keeps the names of the 2026-10-19 flagship (`axis_names.json`). Eight new (benchmark, axis) pairs of the human placement were judged the same day (`1_curated/human_axis_content.csv`): on the agentic axis, PIQA, SimpleBench and WinoGrande are out; on the cyber axis, METR Time Horizons is in and LMCA out; on the commonsense axis, VPCT is in and MMLU Biology and Chemistry are out.
