# Diagnostics

Post-fit tools, all of them command-line scripts. They read fitted traces (the
canonical one lives at `results/canonical/trace.nc`, from
`python fit.py --preset canonical`; exploration traces live in
`results/mirt*/`). The exploratory notebooks that used to sit here moved to
[`../notebooks/`](../notebooks/README.md).

Commands and their options: [`../docs/cli.md`](../docs/cli.md).

## Examine a fit

| Script | Purpose |
|---|---|
| `diagnose_chains.py` | Are the chains one converged solution, or do some sit in a separate likelihood basin? No re-sampling |
| `theta_bimodality.py` | Which test-takers the chain split actually moves, per axis, before and after alignment |
| `align_mirt.py` | Per-draw rotation-alignment comparison on an existing signed trace |
| `residual_corr.py` | Is 1D capability sufficient? Observed minus model-implied benchmark correlations (see below) |
| `ppca_explained_variance.py` | PPCA scree / explained-variance report on logit scores |
| `compare_human_prior.py` | Ordered-human-prior vs independent-theta comparison on the confirmed Q-matrix fit |

## Render figures

| Script | Purpose |
|---|---|
| `plot_mirt.py` | Single-fit deep-dive figures for one MIRT trace → `plots/<out>/` |
| `plot_crossovers.py` | Human-tier crossover dates as a 2x3 grid: US/CN x all / open-only / closed-only |
| `plot_lineage.py` | Reference render of the lineage prior's structure, as a multi-page PDF |
| `forecast_only.py` | Re-render only the frontier-forecast figures of one dashboard card |

## Build and publish

| Script | Purpose |
|---|---|
| `build_dashboard.py` | Build the all-fits interactive dashboard → repo-root `index.html` (+ `results/comparisons/*.csv`). Card registry: `dashboard_fits.json` |
| `country_frontier.py` | US vs China frontier on a K=1 canonical trace: per-country record-setters, per-draw OLS trends, gap / lag / crossovers |

## Maintain the curated data

| Script | Purpose |
|---|---|
| `compute_sota.py` | Recompute the data-driven SOTA list → `data/curated/sota_models.txt` |
| `build_lineage_map.py` | Draft builder for `data/curated/lineage_map.csv` |
| `build_country_map.py` | Country of origin (US / CN / Other) for every model_version → `data/curated/model_country.csv` |
| `fetch_cyber_eci.py` | Fetch Epoch's separate cyber-ECI benchmark table into a curated additive file |
| `audit_lower_bounds.py` | Read-only check of the curated chance floors in `benchmark_lower_bounds.csv` |
| `audit_model_names.py` | Read-only audit of upstream model identity against its own display name |

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
python diagnostics/residual_corr.py
```

Outputs:
- Console: pair coverage, top-20 table, cluster memberships + ratios.
- `plots/residual_corr_heatmap.html` — clustered heatmap (gitignored).
- `results/residual_corr_matrix.csv`, `results/residual_corr_top20.csv` (gitignored).
