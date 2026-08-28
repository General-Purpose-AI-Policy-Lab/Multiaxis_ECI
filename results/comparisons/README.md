# Capability-dimensionality fit dashboard

**Open [`index.html`](../../index.html)** at the repo root — one
self-contained interactive page: a fit selector + a cross-fit Comparison view.
Every figure renders lazily (only the visible fit is live in the DOM).

Registered cards (2):

- K=4 · exploration −2 benchmarks · positive loadings · raw rank-tracked axes (no rotation) · human-merge+lineage(BM) priors · 3PL floors · pooled noise · 10×20000, tune 7000 · single solution (10/10 chains one basin, matched corr 0.843), 78/200,000 divergences, D r-hat 1.168 max; the merged HS→adult edges bind (P(DE>HSQ) 0.64-0.80 under the tree order → 1.00) at identical GoF (R² 0.9645 vs 0.9647) — THE forecasting base  ·  *exploratory*
- K=2 demo · positive loadings · 3PL floors · 4x400 (red-team walkthrough)  ·  *exploratory*

Data scope scored on:

- 829 test-takers / 96 benchmarks / 4923 observations

## Comparison table (`gof_table.csv`)

| fit | type | K | free_loadings | R2 | RMSE | MAE | PIT_var | eta_rhat | divergences | max_phi |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| K=4 −2bench · pooled · merge · flagship | exploratory | 4 | 392 | 0.9645 | 0.0413 | 0.0275 | 0.0464 | 1.142 | 78 | 0.947 |
| K=2 demo | exploratory | 2 | 192 | 0.9362 | 0.059 | 0.0388 | 0.0541 | 1.271 | 0 | 0.221 |

Columns: `type`; `free_loadings` = free loading cells (complexity); `R2`/`RMSE`/
`MAE` = fit; `PIT_var` = calibration (ideal 0.083, below = under-confident);
`eta_rhat` = r̂ on the identified linear predictor (≤ 1.01 = converged);
`divergences`; `max_phi` = largest off-diagonal axis correlation. PSIS-LOO / WAIC
per fit are in `loo_waic_table.csv`. `mode_eval_table.csv` adds chains, kept
draws, ESS and divergences for every card whose trace is still on disk.

## Convergence

`eta_rhat` spans 1.142–1.271 across the 2 cards, against a ≤ 1.01 target. High R² is not trust: read
`cmp_convergence` beside `cmp_gof`, and the ESS column of
`mode_eval_table.csv`.

## Regenerate

One command: `python diagnostics/build_dashboard.py` (`--force <name>` for one
card, `--force-all` for every card, `--png` for static stills → git-ignored
`plots/dashboard/`).
