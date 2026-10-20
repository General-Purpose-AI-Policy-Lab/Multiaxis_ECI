# 4. Diagnostics

Post-fit tools, all command-line scripts. They read a fitted trace, usually
`5_outputs/<data generation>/canonical_humanmerge/trace.nc` (`python 3_fit/fit.py --preset
canonical --human-merge`, see [docs/cli.md](../docs/cli.md)); exploration traces sit in the
`mirt*/` folders beside it.

The four numbered scripts are the reproduction path, in order. The rest is a
toolbox with no order to it, reached for when a fit looks wrong. The curated-data
builders that used to live here moved to [`../1_curated/`](../1_curated/README.md),
because they run before the fit, not after it. The exploratory notebooks moved
to [`../archive/notebooks/`](../archive/notebooks/README.md).

Commands and their options: [`../docs/cli.md`](../docs/cli.md).

## The reproduction path

| Script | Purpose |
|---|---|
| `1_frontier_gap.py` | Open-weights vs closed frontier (or US vs CN with `--group country`) on one K=1 canonical trace, in ECI-H: the running top-2 per group (records and runners-up), the months each open point trails the closed second-best (per posterior draw; dot = posterior mean; `--frontier-k 1` for records against the first closed model above), the same lag month by month from the two per-draw frontiers with its twelve-month trailing mean (a row of the table), one trend line per group through its records with the gap and the lag it implies today (gap over the closed slope), human-tier crossings. One benchmark access scope per run (`--access all|public|semi_private|private`); a class scope is linked onto the all-benchmarks fit through its closed OpenAI, Anthropic and Google models |
| `2_plot_frontier_gap.py` | The scopes side by side, from what step 1 wrote: frontier panels, the months-behind figure (its static renders, English and French, in the lab's print style; from 2024, axis cut at 20 months), the table of the three lags per scope (every quantity and the differences between scopes in its CSV, crossing dates as CSV) |
| `3_plot_mirt.py` | Single-fit deep-dive figures for one MIRT trace → the fit's `figures/k{K}/` |
| `5_place_humans.py` | Place the human tiers on a fit without human baselines (the flagship), axis by axis with the benchmark side frozen (`analysis.human_placement`): the benchmarks each axis dominates, filtered by `1_curated/human_axis_content.csv`, the ordered tier prior, the typical-model profile on the other axes. Writes `human_placement.json` (positions, drawn tiers, convergence, each score observed vs predicted and the first model that reached it) and its draws beside the trace; the figures draw those tiers. `--secondary` places them on the secondary mode's chains (the modes file's `secondary_chains`) into `secondary_mode/`, which the secondary mode's figures and the `chain_groups/` comparison read. Rerun after the selected chains, the axis names, the human baselines or the content file change |
| `4_build_dashboard.py` | Build the all-fits interactive dashboard → repo-root `index.html` (+ the data generation's `comparisons/*.csv`). Card registry: `6_writeups/dashboard/dashboard_fits.json` |

Step 4 needs traces (or their render cache) that are gitignored, so on a fresh
clone every registered card is skipped with a warning and the build refuses to
overwrite `index.html` — the tracked dashboard is the published artifact of the
snapshot it was built from. Re-fit (steps in the root README), `--add` your
trace as a card, then build.

## Toolbox: examine a fit

| Script | Purpose |
|---|---|
| `diagnose_chains.py` | Are the chains one converged solution, or do some sit in a separate likelihood basin? No re-sampling. `--write-modes` writes the split the deliverables read, and keeps a `selected_chains` choice already in the file (a refit of the trace sets that file aside, as `mirt_modes_<stem>_<day>.json`, so a selection never outlives its chains) |
| `theta_bimodality.py` | Which test-takers the chain split actually moves, per axis, before and after alignment |
| `integrated_loo.py` | Leave-one-out per chain group (posterior mode) with each score's own latent block (its release level, or its model's theta) integrated out by a Laplace cavity, so PSIS weighs only the global parameters: on the K=4 flagship it takes the scores with Pareto k > 0.7 from about 740 to 290 per mode. Groups are compared at equal draws (chain averages). No re-sampling |
| `robust_loo.py` | Silva-Zanella mixture leave-one-out per mode (plain, or balanced by a first estimate). Kept for the record: on the K=4 flagship its Monte Carlo spread between chains (800 to 1,200 elpd) was too wide to rank the modes; `integrated_loo.py` replaced it |
| `align_mirt.py` | Per-draw rotation-alignment comparison on an existing signed trace |
| `residual_corr.py` | Is 1D capability sufficient? Observed minus model-implied benchmark correlations (see below) |
| `ppca_explained_variance.py` | PPCA scree / explained-variance report on logit scores |
| `compare_human_prior.py` | Ordered-human-prior vs independent-theta comparison on the confirmed Q-matrix fit. Historical: archived traces, flat `HUMAN_ORDER`, not re-runnable and not affected by the 2026-09-14 move to `--human-merge` for analyses |
| `forecast_only.py` | Re-render only the frontier-forecast figures of one dashboard card, on the card's own view (its chains, the display frame of `mirt_loadings.csv`, the axis titles) |
| `plot_lineage.py` | Reference render of the lineage prior's structure, as a multi-page PDF |
| `lineage_regressions.py` | ECI-H along every lineage edge of `1_curated/lineage_map.csv`, flagged drops first; `tests/test_lineage_map.py` fails on a flagged edge not listed in `1_curated/lineage_reviewed_exceptions.csv` |

## Toolbox: audit the inputs

| Script | Purpose |
|---|---|
| `audit_lower_bounds.py` | Checks the pipeline's chance floors (`lower_bound` of the view) against the fit data: coverage and below-floor scores; `--write-clips` refreshes `1_curated/benchmark_score_clips.csv` |

## `residual_corr.py`

For every benchmark pair $(b_1, b_2)$:

- **Observed corr** — pairwise-complete correlation of scores across the
  models that scored both. Record $N_{\text{shared}}$.
- **Implied corr** — correlation of model-predicted means
  $\mu_{m,b} = \sigma(A_b\,\theta_m - D_b)$ from posterior means, on the
  same per-pair model set.
- **Residual** = observed − implied.

Clusters benchmarks hierarchically on $|\text{residual}|$ (Ward) and reports
mean $|\text{residual}|$ within vs between clusters at $k = 2, 3, 4, 5$.

**Reliability filter.** Pairs with $N_{\text{shared}} < 30$ are excluded
from the top-20 list and the within/between ratios — observed correlations
on a handful of shared models are dominated by sampling noise. Heatmap
shows all pairs but annotates unreliable cells with `·`.

This is one diagnostic, not a substitute for proper factor analysis (PCA
scree on the score matrix, loadings inspection, model comparison) before
concluding anything about latent structure.

### Run

```bash
python 4_diagnostics/residual_corr.py
```

Outputs:
- Console: pair coverage, top-20 table, cluster memberships + ratios.
- `diagnostics/residual_corr_heatmap.png` (+ `html/`) — clustered heatmap (gitignored).
- `diagnostics/residual_corr_matrix.csv`, `diagnostics/residual_corr_top20.csv` (gitignored), all under the data generation's folder.
