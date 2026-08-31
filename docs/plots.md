# Figures

Every figure in this project is built by a `viz/` Plotly builder and read by
one of four callers: the single-fit CLI, the folder sweep, the dashboard, and
the LessWrong post's figure script. This file says where the output lands,
which command produces it, what each figure shows, and the four reading
conventions a figure cannot state on its own.

Model math and priors: [model_math.md](model_math.md). Flags, tags and the
fit CLI: [../README.md](../README.md).

## Where figures go

| destination | written by | tracked? |
|---|---|---|
| `plots/mirt_k{K}{tag}/` | `3_diagnostics/3_plot_mirt.py`, and `2_fit.py --plots` | no |
| `plots/canonical/` | `2_fit.py --preset canonical` | no |
| `plots/dashboard/` | `3_diagnostics/4_build_dashboard.py --png` / `--pdf` | no |
| `index.html` (repo root) | `3_diagnostics/4_build_dashboard.py` | **yes** |
| `lw_post/figures/` | `lw_post/figures/make_all.py`, a local-only folder not in this repository | no |

`plots/` is gitignored entirely: figures are regenerable from the traces, and
one dashboard build writes hundreds of them. `index.html` is the tracked
artifact and the thing you serve. It is self-contained, so a browser opens it
with no server.

One fit writes one HTML and one PNG per figure into its own
`plots/mirt_k{K}{tag}/`. K is in that folder name, so a K=3 and a K=4 run of
one flag set do not overwrite each other.

## The commands

Single trace. A trace path is the only argument: `FitSpec.from_trace` recovers
the flag set, the data scope and the destination folder from it.

```bash
python 3_diagnostics/3_plot_mirt.py --forecast --trace \
  results/mirt_humanmerge_lineageprior_lineagebm/trace_mirt_k4_humanmerge_lineageprior_lineagebm.nc
```

Folder sweep. Renders every MIRT trace under a directory, one child process
each, forecasts on. `--dry-run` prints the per-trace decision and renders
nothing.

```bash
python 3_diagnostics/3_plot_mirt.py --folder results/ --dry-run
python 3_diagnostics/3_plot_mirt.py --folder results/mirt_humanmerge_lineageprior_lineagebm
```

Dashboard. `--force-all` ignores the render cache, which is the only way to be
sure no superseded card is served. `--add TRACE --name NAME --label LABEL`
registers a new card, `--remove NAME` drops one, `--list` prints the registry.

```bash
python 3_diagnostics/4_build_dashboard.py --list
python 3_diagnostics/4_build_dashboard.py --force-all
```

LessWrong post figures. Every one reads the flagship through
`analysis.FLAGSHIP` / `open_flagship`, so the fit identity, the majority-chain
policy and the forecast settings are the repo's. `--cached` never opens the
trace: it reuses the forecast cache pickle and fails if it is missing.

```bash
python lw_post/figures/make_all.py all --cached   # local-only, not shipped here
```

## Figure catalogue

Names below are the figure keys. On disk each becomes
`mirt_<key>.html` / `mirt_<key>.png`; on the dashboard each is one panel.

### Goodness of fit

Computed on the **whole** fit, never on a plot-side subset.

| key | what it shows |
|---|---|
| `gof_pred_vs_observed` | posterior-mean prediction against observed score, one point per observation. Points off the diagonal are misfit; a bend in the cloud is a link problem, a widening is a noise problem |
| `gof_posterior_predictive` | the replicated score density overlaid on the observed one. The two should have the same shape, including the pile-up near 0 and 1 |
| `gof_pit` | histogram of the probability integral transform. Flat means calibrated. A U shape means the predictive is too narrow, a dome means too wide |
| `pit_ecdf` | the same PIT as a cumulative curve against the uniform diagonal. Reads small deviations that the histogram's binning hides. Single-fit CLI only; the dashboard card carries the cross-fit `cmp_pit_ecdf` instead |
| `gof_bench_scores_vs_pred` | per benchmark (dropdown), models ranked by observed score, with the predictive median and its percentile band. Systematic misfit reads as the two marker sets separating; the band width shows how much of the gap the fit calls benchmark noise |
| `gof_bench_icc` | per benchmark (dropdown), the fitted item characteristic curve mu = c + (d−c)·sigma(eta) drawn against the observed scores, with each model placed on that benchmark's logit scale. Dots hugging the curve mean the benchmark is well predicted; vertical bias or wide scatter flags misfit. Compensatory linear-link fits only, since the product and log-logistic links have no single eta |
| `gof_residuals` | residual box per benchmark. One benchmark's box shifted off zero is a difficulty or loading the fit cannot place |

### Abilities over time

| key | what it shows |
|---|---|
| `timeline_{k}_{axis}` | the **measured** timeline for axis k: ability against release date, 50% intervals, models at posterior SD >= 0.4 and low-observation models dropped, human tiers as horizontal bands. The headline per-axis figure |
| `timeline_{k}_{axis}_all` | the all-models companion: every dated model including the sparse pre-2023 ones, drawn with their prior-wide intervals. Use it to see who the measured view drops and why |
| `timeline_difficulty` | benchmark difficulty D against benchmark release date, 50% intervals. The mirror of the ability timeline on the same latent scale |
| `axes_timeline_compare` | every axis's running frontier (cumulative best ability) on one panel. Answers which axis is moving fastest. Single-fit CLI only, K >= 2 |
| `axes_scatter_matrix` | pairwise scatter of model abilities across axes, coloured by organization, informed models only. Single-fit CLI only |

### Loadings and axis structure

| key | what it shows |
|---|---|
| `loadings_{k}_{axis}` | per-axis loading forest: which benchmarks load on axis k, sorted, with intervals and a reference line at 0. This is what names an axis |
| `loadings_heatmap` | benchmark by axis, in **axis share** rather than raw loading: the fraction of a benchmark's squared loading-row norm pointing along the axis, top 20 per axis. Purity defines an axis better than steepness, since on raw loadings a long half-aligned row out-ranks a short pure one |
| `factor_correlations` | correlation heatmap of the axis abilities, K >= 2 (single-fit CLI figure; the dashboard filters it out of its cards and reports `max_phi` in the comparison table instead). When the display frame is promax the title flags it and the raw ability correlation is annotated, so an oblique correlation is never read as the raw one |
| `axis_strength` | forest of per-axis strength, i.e. the loading column norms (or `tau_A` where the fit has a per-axis scale). How much of the fit each axis carries |
| `qmatrix` | the allowed-loading pattern, for conjunctive and anchored fits only |

### Forecast trio

Three figures per axis, added by `--forecast` on the CLI and by
`"forecast": True` on a dashboard entry. Gated on K > 1, the compensatory
family, and human tiers being in the fit.

| key | what it shows |
|---|---|
| `forecast_{k}_{axis}` | the measured timeline with the fitted frontier trend, its 50% band, and a dashed marker at each human tier's projected crossover date |
| `forecast_{k}_{axis}_when` | the crossover dates alone: tier on the y-axis, projected date and interval on the x-axis, coloured passed against future. The readable version of the dashed markers |
| `forecast_{k}_{axis}_prob` | P(frontier > tier) over the forecast grid, one S-curve per tier, with reference lines at 0.5 and 0.975. Read a date off it rather than a point estimate |

### K against K=1

Rendered by the single-fit CLI when a `trace_mirt_k1.nc` baseline sits in the
same results folder.

| key | what it shows |
|---|---|
| `factor1_vs_1d` | axis-1 ability against the 1D capability, with the correlation. A sanity check: the dominant axis should track C |
| `pred_k_vs_k1` | K-axis against K=1 predicted means, coloured by which one erred less on that observation |
| `r2_delta_per_bench` | per-benchmark R² of the K-axis fit minus the K=1 fit. Names the benchmarks the extra axes actually buy |

## Conventions a reader must know

**The informed filter is plot-side only.** `mirt_informed_mask` drops a
model's axis ability from a figure when its posterior SD >= 0.4. It never
touches the fit and never touches a diagnostic: convergence, PPC, PIT, GoF
and LOO always describe the whole fit. The same holds for a mode-restricted
dashboard card, which is an addition to the whole-fit figures, not a
replacement. The 0.4 cap corresponds to a 95% interval width of about 1.57.

**SOTA models are exempt from that drop.** Models in `config.SOTA_MODELS`
stay on every timeline even when sparse and wide, because a frontier release
is the headline of the figure and its uncertainty is better communicated by
the drawn interval than by a silent omission.

**Records are read off the posterior median**, the number the timelines plot,
so every fitted point is a point the reader can see. The mean fails on both
shapes this data produces: a thinly-evaluated release is right-skewed and its
mean sits above every plotted point, and a ridge-split ability is bimodal and
its mean lands in the empty valley between the two lumps.

**The forecast rule lives in one place.** `config.FORECAST_KW` is
`fit_basis="records"`, `fit_start="2024-10-01"` (the reasoning-model cutoff),
`sd_cap=0.4`, `hdi_prob=0.5`. The dashboard card, the memo and the LW post all
read it, so the three cannot drift apart on the basis, the cap or the interval
width. The cloud and the trend fit share the one cap, which is what makes
every fitted record also a plotted point.

`config.FORECAST_NO_SOTA_AXES` takes the SOTA exemption back on a stale axis.
Axis 4 is in that set: its defining benchmarks (OpenBookQA 0.89 axis share,
ARC (AI2) 0.84, Adversarial NLI 0.82, BoolQ, CSQA2, BBH, SuperGLUE,
HellaSwag, PIQA) carry no observation on any model released after 2025-06, so
every SOTA candidate there sits at SD ~1.0 on the lineage prior alone.
Exempted, those points held the running max and suppressed every measured
record: the record set collapsed to 2 points 77 days apart and the slope
flipped sign between the posterior mean and median. The other axes keep the
exemption, their measured models already outranking the prior-only points.

## Memory

**The flagship trace is 38 GB against 26 GB of RAM**, so it cannot be opened
whole. Every figure in this catalogue is a median or an interval, and 20,000
draws pin those as well as 200,000, so thinning costs nothing that a figure
shows.

- `--thin N` on `plot_mirt.py` keeps every n-th draw.
- **Folder mode picks the thin itself**, one kept draw per 2 GB of file:
  `thin = max(1, round(size_bytes / 2e9))`. The flagship gets thin 19, a
  20 GB trace gets 10, anything under 3 GB gets 1. An explicit `--thin` wins.
- **One child process per trace** in folder mode, so the OS reclaims all of
  the memory between fits and one out-of-memory kill costs one fit rather
  than the sweep.
- `analysis.open_flagship()` is the flagship reader every downstream script
  uses. It defaults to `FLAGSHIP_MAJORITY_CHAINS` (0-5, 7, 8) and
  `FLAGSHIP_THIN` (10). Chains 6 and 9 sit in the second solution: they lift
  the whole human block on the legacy-QA axis by about 2.3 logits while the
  machine rows move by 0.25. The majority is the default so the reported fit
  is one solution rather than an average of two, and a caller has to pass
  `chains=None` on purpose to re-admit them.
