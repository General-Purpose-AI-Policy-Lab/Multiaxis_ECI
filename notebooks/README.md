# Notebooks

One-off investigations, kept for the record rather than maintained. Each one
answered a question at a point in time; none is part of a workflow, nothing
imports them, and their cell outputs are the result they were kept for. Imports
may predate the current layout, so expect to fix a path before re-running one.

The maintained equivalents live in [`../diagnostics/`](../diagnostics/README.md).

**How many axes are there?**

| Notebook | What it found |
|---|---|
| `multidim_ppca_ard.ipynb` | Whether the benchmark suite spans more than one latent axis, by PPCA/ARD on the logit scores. The starting point for everything below. |
| `axis3_instrument_swap.ipynb` | Why the knowledge axis rises then falls, told per vendor lineage with the benchmark scores that drive it. |

**Why the K=3 chains disagreed**

| Notebook | What it found |
|---|---|
| `k3_basin_deepdive.ipynb` | Two basins of six chains each: identified eta r-hat 1.676 pooled but 1.004 / 1.003 inside each basin. The two are a partial rotation of the whole frame, with basin A ahead by 32.2 nats. |
| `recovery_study.ipynb` | Simulating from one known parameter point and refitting, to ask whether the basins are intrinsic to the model and observation pattern or an artifact of the real data. |
| `k3_c_ceilings_axis2.ipynb` | Estimated per-benchmark ceilings turn out imposed rather than learned: only 3 of 98 benchmarks pull the gap off its prior, and axis 2 splits into three basins. |
| `k3_bm_vs_nobm.ipynb` | Brownian lineage steps beat per-release steps on LOO by 99.7 ± 29.2 nats, but do not remove the two basins. |
| `floors_k3_3pl_report.ipynb` | The recorded convergence metrics across the chance-floor campaign: r-hat, divergences and logp spread with floors off and on. |

**Model families and priors that were tried and dropped**

| Notebook | What it found |
|---|---|
| `interaction_gamma_onesided.ipynb` | The semi-compensatory K=3 fit with one-sided interaction terms. |
| `lineage_bm_rate_diagnosis.ipynb` | The per-family rate layer, retired: 36 vendor rates collapse to one, for 5.9x the sampling time and a second posterior mode. |

**Do the forecasts hold up?**

| Notebook | What it found |
|---|---|
| `backtest_frontier_2025.ipynb` | Fits the frontier trend on record-setters released before 2025-01-01 and extrapolates, to check it against what 2025 actually did. |
