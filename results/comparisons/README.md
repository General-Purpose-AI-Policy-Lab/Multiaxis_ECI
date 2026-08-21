# Capability-dimensionality fit dashboard

**Open [`index.html`](../../index.html)** at the repo root — one
self-contained interactive page: a fit
selector (grouped baseline / exploratory / confirmed) + a cross-fit Comparison
view. Every figure renders lazily (only the visible fit is live in the DOM).

Cards span two data scopes: the K=1 fits apply curated benchmark exclusions at
fit time (702 models / 77 benchmarks); the K=3 no-SG fit uses the full set
(751 models / 85 benchmarks). GoF is comparable within a scope; global R²/RMSE
stay readable across scopes, per-benchmark rows do not:

- **exploratory**: K=3 · cyber · positive loadings · 4PL floors+ceilings · 8×2000 · no priors · shown whole: ~8 distinct solutions, no converged chain pair (eta r̂ 1.60) — the ladder's baseline; fit metrics average over basins, K=3 · cyber · positive loadings · 4PL floors+ceilings · 7×2000 · lineage+BM only · mode-restricted: majority pair {2,5}, eta r̂ 1.010, K=3 · cyber · positive loadings · 4PL floors+ceilings · 7×2000 · lineage+BM only · mode B (island, −22 nats; chains {3,4}, eta r̂ 1.008), K=3 · cyber · positive loadings · 4PL floors+ceilings · 7×2000 · human prior only · mode-restricted: mode A 5/7 (eta r̂ 1.012), B island (−47 nats) excluded, K=3 · cyber · positive loadings · 4PL floors+ceilings · 7×2000 · human prior only · mode B (island, −47 nats; chains {1,5}, eta r̂ 1.015), K=3 · cyber · positive loadings · 4PL floors+ceilings+ceiling noise · 7×2000 · human+lineage priors · mode-restricted: mode A 3/7 (eta r̂ 1.027), B (−30 nats) excluded, K=3 · cyber · positive loadings · 4PL floors+ceilings+ceiling noise · 7×2000 · human+lineage priors · mode B (island, −30 nats; chains {0,1,2,3}, eta r̂ 1.033), K=3 · cyber · positive loadings · human+lineage+BM priors · 4PL (floors+ceilings+ceiling noise) · 12×6000 · mode A (majority 7/12; chains 5,6,9–11 excluded), K=3 · cyber · positive loadings · human+lineage+BM priors · 4PL (floors+ceilings+ceiling noise) · 12×6000 · mode B (island, −16 nats; chains 6,9–11 only), K=3 · cyber · signed loadings · human+lineage+BM priors · 4PL (floors+ceilings+ceiling noise) · 12×6000 · mode α (best basin, holds the top-logp chain; chains 1,2,9 only), K=3 · cyber · signed loadings · human+lineage+BM priors · 4PL (floors+ceilings+ceiling noise) · 12×6000 · mode β (island, −15 nats; the other 9/12 chains)

## Comparison table (`gof_table.csv`)

| fit | type | K | free_loadings | R2 | RMSE | MAE | PIT_var | eta_rhat | divergences | max_phi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K=3 cy ladder0 · none | exploratory | 3 | 324 | 0.9566 | 0.0454 | 0.0299 | 0.0462 | 1.6 | 181 | 0.914 |
| K=3 cy ladder1 · lin+BM · A | exploratory | 3 | 324 | 0.9581 | 0.0468 | 0.0315 | 0.0528 | 1.01 | 33 | 0.667 |
| K=3 cy ladder1 · lin+BM · B | exploratory | 3 | 324 | 0.9558 | 0.0483 | 0.0321 | 0.0524 | 1.008 | 9 | 0.907 |
| K=3 cy ladder2 · human · A | exploratory | 3 | 324 | 0.9568 | 0.0466 | 0.0303 | 0.0487 | 1.012 | 27 | 0.495 |
| K=3 cy ladder2 · human · B | exploratory | 3 | 324 | 0.9557 | 0.0472 | 0.0308 | 0.0493 | 1.015 | 3 | 0.623 |
| K=3 cy ladder3 · both · A | exploratory | 3 | 324 | 0.9587 | 0.0463 | 0.0311 | 0.0522 | 1.027 | 359 | 0.586 |
| K=3 cy ladder3 · both · B | exploratory | 3 | 324 | 0.9582 | 0.046 | 0.0311 | 0.0517 | 1.033 | 2 | 0.808 |
| K=3 cy · positive · mode A | exploratory | 3 | 324 | 0.9586 | 0.0464 | 0.0311 | 0.0524 | 1.005 | 36 | 0.63 |
| K=3 cy · positive · mode B | exploratory | 3 | 324 | 0.958 | 0.0462 | 0.0312 | 0.0516 | 1.009 | 24 | 0.697 |
| K=3 cy · signed · mode α | exploratory | 3 | 324 | 0.9608 | 0.0447 | 0.0301 | 0.0512 | 1.007 | 52 | 0.276 |
| K=3 cy · signed · mode β | exploratory | 3 | 324 | 0.9604 | 0.0448 | 0.0299 | 0.0506 | 1.005 | 240 | 0.06 |

Columns: `type`; `free_loadings` = free loading cells (complexity); `R2`/`RMSE`/
`MAE` = fit; `PIT_var` = calibration (ideal 0.083, below = under-confident);
`eta_rhat` = convergence (≤ 1.01 = converged; raw r̂ for 1D, log_μ r̂ for non-comp);
`divergences`; `max_phi` = largest off-diagonal axis correlation. PSIS-LOO / WAIC
per fit are in `loo_waic_table.csv` and the dashboard's Comparison view.

## The headline

The **exploratory** fits post high R² but **do not converge** (eta r̂ ≫ 1.01). The
**confirmed** fits converge only because the Q-matrix supplies the rotation
externally. The **non-comp** fits converge but fit worse than 1D. High R² is not
trust — read `cmp_convergence` alongside `cmp_gof`.

## Regenerate

One command: `python diagnostics/build_dashboard.py` (add `--png` for static
stills → git-ignored `plots/dashboard/`). The ordered-human-prior comparison
(`compare_human_prior.py`) remains a separate one-off.
