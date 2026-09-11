# 4. Diagnostics

Post-fit tools, all command-line scripts. They read a fitted trace (the
canonical one at `5_outputs/<data generation>/canonical/trace.nc`, from
`python 3_fit/fit.py --preset canonical`; exploration traces in the `mirt*/` folders beside it).

The four numbered scripts are the reproduction path, in order. The rest is a
toolbox with no order to it, reached for when a fit looks wrong. The curated-data
builders that used to live here moved to [`../1_curated/`](../1_curated/README.md),
because they run before the fit, not after it. The exploratory notebooks moved
to [`../archive/notebooks/`](../archive/notebooks/README.md).

Commands and their options: [`../docs/cli.md`](../docs/cli.md).

## The reproduction path

| Script | Purpose |
|---|---|
| `1_frontier_gap.py` | Open-weights vs closed frontier (or US vs CN with `--group country`) on one K=1 canonical trace, in ECI-H: records per group, the months each open record trails the closed frontier (per posterior draw), one trend line per group with the gap and lag it implies, human-tier crossings. One benchmark access scope per run (`--access all|public|semi_private|private`) |
| `2_plot_frontier_gap.py` | The scopes side by side, from what step 1 wrote: frontier panels, the months-behind figure, the summary table (crossing dates as CSV) with each scope's change against all benchmarks |
| `3_plot_mirt.py` | Single-fit deep-dive figures for one MIRT trace → the fit's `figures/k{K}/` |
| `4_build_dashboard.py` | Build the all-fits interactive dashboard → repo-root `index.html` (+ the data generation's `comparisons/*.csv`). Card registry: `6_writeups/dashboard/dashboard_fits.json` |

Step 4 needs traces (or their render cache) that are gitignored, so on a fresh
clone every registered card is skipped with a warning and the build refuses to
overwrite `index.html` — the tracked dashboard is the published artifact of the
snapshot it was built from. Re-fit (steps in the root README), `--add` your
trace as a card, then build.

## Toolbox: examine a fit

| Script | Purpose |
|---|---|
| `diagnose_chains.py` | Are the chains one converged solution, or do some sit in a separate likelihood basin? No re-sampling |
| `theta_bimodality.py` | Which test-takers the chain split actually moves, per axis, before and after alignment |
| `align_mirt.py` | Per-draw rotation-alignment comparison on an existing signed trace |
| `residual_corr.py` | Is 1D capability sufficient? Observed minus model-implied benchmark correlations (see below) |
| `ppca_explained_variance.py` | PPCA scree / explained-variance report on logit scores |
| `compare_human_prior.py` | Ordered-human-prior vs independent-theta comparison on the confirmed Q-matrix fit |
| `forecast_only.py` | Re-render only the frontier-forecast figures of one dashboard card |
| `plot_lineage.py` | Reference render of the lineage prior's structure, as a multi-page PDF |

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
