# Capability-dimensionality fit dashboard

**Open [`index.html`](../../../index.html)** at the repo root — one
self-contained interactive page: a fit selector + a cross-fit Comparison view.
Every figure renders lazily (only the visible fit is live in the DOM).

Registered cards (2):

- K=4 · full exploration scope · positive loadings · raw rank-tracked axes (no rotation) · human-merge+lineage(BM) priors · 3PL floors · pooled noise; THE forecasting base (the blog post's fit)  ·  *exploratory*
- K=2 demo · positive loadings · 3PL floors (default) · 4x400 (red-team walkthrough)  ·  *exploratory*

Data scope scored on:

- 802 test-takers / 97 benchmarks / 5307 observations

## Comparison table (`gof_table.csv`)

| fit | type | K | free_loadings | R2 | RMSE | MAE | PIT_var | eta_rhat | divergences | max_phi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K=4 · pooled · merge · flagship | exploratory | 4 | 388 | 0.9606 | 0.0457 | 0.0304 | 0.0474 | 1.142 | 44 | 0.91 |

Columns: `type`; `free_loadings` = free loading cells (complexity); `R2`/`RMSE`/
`MAE` = fit; `PIT_var` = calibration (ideal 0.083, below = under-confident);
`eta_rhat` = r̂ on the identified linear predictor (≤ 1.01 = converged);
`divergences`; `max_phi` = largest off-diagonal axis correlation. PSIS-LOO / WAIC
per fit are in `loo_waic_table.csv`. `mode_eval_table.csv` adds chains, kept
draws, ESS and divergences for every card whose trace is still on disk.

## Convergence

`eta_rhat` is 1.142 on the one card, against a ≤ 1.01 target. High R² is not trust: read
`cmp_convergence` beside `cmp_gof`, and the ESS column of
`mode_eval_table.csv`.

## Regenerate

One command: `python 4_diagnostics/4_build_dashboard.py` (`--force <name>` for one
card, `--force-all` for every card, `--png` for static stills → git-ignored
`dashboard_stills/`).
