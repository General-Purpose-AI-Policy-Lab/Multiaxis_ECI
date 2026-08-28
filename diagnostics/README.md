# Diagnostics

Post-fit tools. They read fitted traces (the canonical one lives at
`results/canonical/trace.nc`, from `python fit.py --preset canonical`;
exploration traces live in `results/mirt*/`).

| Script | Purpose |
|---|---|
| `build_dashboard.py` | Build the all-fits interactive dashboard → repo-root `index.html` (+ `results/comparisons/*.csv`) |
| `plot_mirt.py` | Single-fit deep-dive figures for one MIRT trace → `plots/<out>/` |
| `align_mirt.py` | Per-draw rotation-alignment comparison on an existing signed trace (no re-sampling) |
| `compare_human_prior.py` | Ordered-human-prior vs independent-theta comparison on the confirmed Q-matrix fit |
| `compute_sota.py` | Recompute the data-driven SOTA list → `data/curated/sota_models.txt` |
| `residual_corr.py` | Is 1D capability sufficient? (see below) |
| `ppca_explained_variance.py` | PPCA scree / explained-variance report on logit scores |
| `build_lineage_map.py` | Draft builder for `data/curated/lineage_map.csv` |
| `*.ipynb` | One-off investigation notebooks (kept for the record; imports may predate the current layout) |

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
~/miniforge3/envs/pymc_env/bin/python diagnostics/residual_corr.py
```

Outputs:
- Console: pair coverage, top-20 table, cluster memberships + ratios.
- `plots/residual_corr_heatmap.html` — clustered heatmap (gitignored).
- `results/residual_corr_matrix.csv`, `results/residual_corr_top20.csv` (gitignored).
