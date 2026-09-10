# Figures

Every figure in this project is built by a `viz/` Plotly builder and read by
one of four callers: the single-fit CLI, the folder sweep, the dashboard, and
the blog post's figure script. This file says where the output lands,
which command produces it, what each figure shows, and the four reading
conventions a figure cannot state on its own.

Model math and priors: [model_math.md](model_math.md). Flags, tags and the
fit CLI: [../README.md](../README.md).

## Where figures go

All paths below sit under `5_outputs/<data generation>/` (`config.RESULTS_DIR`), whose first level is the pipeline build date the fit was made on.

| destination | written by | tracked? |
|---|---|---|
| `mirt{tag}/figures/k{K}/` | `4_diagnostics/3_plot_mirt.py`, and `3_fit/fit.py --plots` | no |
| `canonical/figures/` | `3_fit/fit.py --preset canonical` | no |
| `comparisons/figures/` | `4_diagnostics/1_country_frontier.py`, `2_plot_crossovers.py` | no |
| `diagnostics/` | `residual_corr.py`, `diagnose_chains.py --fig`, `align_mirt.py` | no |
| `dashboard_stills/` | `4_diagnostics/4_build_dashboard.py --png` / `--pdf` | no |
| `index.html` (repo root) | `4_diagnostics/4_build_dashboard.py` | **yes** |
| `6_writeups/blogpost/figures/` | `6_writeups/blogpost/figures/make_all.py` | **yes**, except `*.html` |

Every `figures/` folder holds the PNGs, with the interactive Plotly twins under `html/` and the French versions under `fr/` (same layout inside). They are gitignored: regenerable from the traces, and one dashboard build writes hundreds of them. `index.html` is the tracked artifact and the thing you serve. It is self-contained, so a browser opens it with no server. The blog post's figures are tracked because the post is a deliverable rather than a render, but their interactive twins are not: ~4.7 MB each of inlined JS, rebuilt by `make_all.py`.

One fit writes one PNG (and one HTML) per figure into its own `figures/k{K}/`. K is in that folder name, so a K=3 and a K=4 run of one flag set share their tables and traces but never their figures.

When `diagnose_chains.py --write-modes` has found more than one posterior mode for the trace, the folder holds the **majority chains'** figures (file names end in `_majority`, titles name the chains) and a `minority/` subfolder holds the same set on every other chain (`_minority`); both groups are put back on the fit's display frame through `mirt_loadings.csv`, so axis k is the same axis in both. Every figure in a group's folder, the posterior predictive and the PIT included, is computed on that group's draws: a chain group is a posterior mode of its own, and its predictive distribution is that mode's, where the pooled predictive would mix two incompatible solutions. The whole-fit numbers stay in the fit's tables (`gof.json`, `summary.csv`) and on the dashboard card. A `chain_groups/` subfolder puts the two groups side by side to locate the disagreement (`viz/chain_groups.py`): `chain_logp` (each chain's mean log-density below the best mode, coloured by group), `loadings_compare` (per axis, the top benchmarks' loadings in both groups, dumbbell rows), `abilities_compare` (median abilities of every test-taker, group against group, human tiers as squares, with the correlation), `human_tiers_compare` (each tier's ability in both groups, dodged rows) and `crossover_compare` (projected crossing dates of the tiers in both groups on the shared window). The figures a write-up would embed (per-axis timelines and forecasts, `forests_per_axis`, `loadings_per_axis`, `gof_pit`, `axes_timeline_compare`) are also rendered in French under one `fr/` folder for both chain groups (`_majority_fr`, `_minority_fr`) through the shared string table of `viz/i18n.py`, the same table the blog post's French renders use.

**Axis names are given by hand.** Nothing in the code names an axis. The fit and the plotting CLI write an `axis_names.json` template beside the trace when none exists (each axis's top benchmarks by share, empty titles, `confirmed: false`), and every figure calls the axes `Axis 1`, `Axis 2`, ... until a person fills in the titles (and `title_fr` for the French renders) and two or three signature benchmarks and sets `confirmed: true`. A confirmed title is applied only while one of its signature benchmarks stays among the axis's top benchmarks (`analysis.load_axis_titles`); the blog post's scripts refuse to run on unconfirmed or drifted names (`analysis.require_axis_titles`). The published fit's names live in `5_outputs/pre_pipeline/mirt_humanmerge_lineageprior_lineagebm/axis_names.json`.

Every figure is English by default; builders that carry text take `lang="fr"`, and the canonical fit writes that render of its `capability_timeline` under `figures/fr/`. That figure is the post's figure 1 drawn by the same `capability_timeline_fig`: anchored ECI-H scale, 80% intervals, and the interval of the bottom and top human tiers (Average Human, Top Performer) as a faint band, so the human ladder's bounds read as uncertain. The country-frontier figure draws the same two bands.

## The commands

Single trace. A trace path is the only argument: `FitSpec.from_trace` recovers
the flag set, the data scope and the destination folder from it.

```bash
python 4_diagnostics/3_plot_mirt.py --forecast --trace \
  5_outputs/data20260908/mirt_humanmerge_lineageprior_lineagebm/trace_mirt_k4_humanmerge_lineageprior_lineagebm.nc
```

Folder sweep. Renders every MIRT trace under a directory, one child process
each, forecasts on. `--dry-run` prints the per-trace decision and renders
nothing.

```bash
python 4_diagnostics/3_plot_mirt.py --folder 5_outputs/data20260908/ --dry-run
python 4_diagnostics/3_plot_mirt.py --folder 5_outputs/data20260908/mirt_humanmerge_lineageprior_lineagebm
```

Dashboard. `--force-all` ignores the render cache, which is the only way to be
sure no superseded card is served. `--add TRACE --name NAME --label LABEL`
registers a new card, `--remove NAME` drops one, `--list` prints the registry.

```bash
python 4_diagnostics/4_build_dashboard.py --list
python 4_diagnostics/4_build_dashboard.py --force-all
```

Blog-post figures. Every one reads the flagship through
`analysis.FLAGSHIP` / `FLAGSHIP_TRACE` / `open_flagship`, so the fit identity,
the chain policy and the forecast settings are the repo's and no script names a
trace of its own. `--cached` never opens the trace: it reuses the forecast cache
pickle and fails if it is missing.

```bash
python 6_writeups/blogpost/figures/make_all.py all --cached
```

The post's figures are the same builders as the outputs at another scale: each builder takes a `FigureStyle` (`viz/style.py`), `DASHBOARD` for the cards and the per-fit folders, `POST` for a figure shared flat at about 2,000 pixels. The post's scripts own only the flagship trace, the caches beside it, the fixed windows (2023 to 2030 for the trend, 2015 to 2030 for the crossings) and the hand-placed callouts of figure 1; the figure design lives in `viz/` and cannot drift between the post and the outputs.

## Figure catalogue

Names below are the figure keys; on the dashboard each is one panel. On disk the
name is the explicit form of the key (`viz.figure_filename`): `timeline_2_reasoning`
becomes `timeline_axis2_reasoning.png`, `forecast_1_math_when` becomes
`forecast_axis1_math_crossover_dates.png`; other keys are unchanged.

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
| `timeline_{k}_{axis}` | the **measured** timeline for axis k: ability against release date, 50% intervals, models at posterior SD >= 0.33 and low-observation models dropped, human tiers as horizontal bands. The headline per-axis figure |
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
| `forests_per_axis` | the post's forest figure (`forest_grid_fig`): per axis the top models among the timeline candidates, the pinned frontier releases (`config.FOREST_PINNED_RELEASES`, drawn even when wide) and every human tier, 95% intervals, K >= 2 |
| `loadings_per_axis` | the post's loadings figure (`loadings_grid_fig`): per axis the 20 benchmarks with the largest axis share, bar = loading with its 95% interval, colour and the right-hand number = share, K >= 2 |

### Forecast pair

Two figures per axis, added by `--forecast` on the CLI and by
`"forecast": True` on a dashboard entry. Gated on K > 1, the compensatory
family, and human tiers being in the fit.

| key | what it shows |
|---|---|
| `forecast_{k}_{axis}` | the post's trend panel (`frontier_trend_fig`): the measured cloud with 80% whiskers, the per-draw record envelope extended at its recent rate with its 80% band, the human tiers as dashed lines named in the right margin, and a today line |
| `forecast_{k}_{axis}_when` | the post's crossover panel (`crossover_panels_fig`): per tier the median crossing date, a thick 50% bar over a thin 80% bar, both split at today (green behind us, red ahead); whatever runs past the window is clipped at the edge and dated |

### K against K=1

Rendered by the single-fit CLI when a `trace_mirt_k1.nc` baseline sits in the
same results folder.

| key | what it shows |
|---|---|
| `factor1_vs_1d` | axis-1 ability against the 1D capability, with the correlation. A sanity check: the dominant axis should track C |
| `pred_k_vs_k1` | K-axis against K=1 predicted means, coloured by which one erred less on that observation |
| `r2_delta_per_bench` | per-benchmark R² of the K-axis fit minus the K=1 fit. Names the benchmarks the extra axes actually buy |

## Conventions a reader must know

**The informed filter is plot-side only.** `analysis.timelines.candidate_mask` drops a
model's axis ability from a figure when its posterior SD is at or above
`config.INFORMED_SD_CAP` (0.33, a 95% interval width of about 1.3) or when the
model is flagged low-observation, SOTA families excepted. It never touches the
fit and never touches a diagnostic: convergence, PPC, PIT, GoF and LOO are
computed on every draw of the posterior they describe, never on a plot-side
subset of models. On the dashboard that posterior is the whole fit, and a
mode-restricted card is an addition to the whole-fit figures, not a replacement;
in a fit's own figure folder it is the chain group the folder shows (see
"Where figures go"). The K=1 ECI-H timeline of the canonical fit draws every
dated model.

**SOTA models are exempt from that drop.** Models of the `config.SOTA_FAMILIES` releases
stay on every timeline even when sparse and wide, because a frontier release
is the headline of the figure and its uncertainty is better communicated by
the drawn interval than by a silent omission.

**Records are read off the posterior median**, the number the timelines plot,
so every fitted point is a point the reader can see. The mean fails on both
shapes this data produces: a thinly-evaluated release is right-skewed and its
mean sits above every plotted point, and a ridge-split ability is bimodal and
its mean lands in the empty valley between the two lumps.

**The forecast rule lives in one place.** `config.FORECAST_KW` is
`fit_basis="envelope"`, `fit_start="2024-10-01"` (the reasoning-model cutoff,
used only by the regression bases kept for sensitivity runs), `sd_cap=0.33`,
`hdi_prob=0.8`. The dashboard card and the blog post figures both read it, so
the two cannot drift apart on the basis, the cap or the interval width. The
cloud and the trend fit share the one cap, which is what makes every fitted
record also a plotted point.

**The SOTA exemption stops at prior-only positions.** A SOTA release is admitted
with a wide interval only while its ability on the axis is at least weakly
measured (posterior SD below `config.SOTA_EXEMPT_SD_CAP`, 0.8). On the published
fit's Legacy QA axis, whose defining benchmarks carry no observation on any
model released after 2025-06, every SOTA candidate sat at SD ~1.0 on the lineage
prior alone; exempted, those points held the running max and suppressed every
measured record (the record set collapsed to 2 points 77 days apart and the
slope flipped sign between the posterior mean and median). Until 2026-09-10 the
remedy was an axis index (`FORECAST_NO_SOTA_AXES = {3}`), which a refit with
another fourth axis turned into a trend fitted on 2 points under a cloud of 123;
the uncertainty rule needs no axis identity. Titles, too, follow identity:
`analysis.load_axis_titles` applies a hand-confirmed title only to an axis whose
top benchmarks still contain one of its signature benchmarks (`axis_names.json`).

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
  uses, and `analysis.FLAGSHIP_TRACE` is the one path any of them names. It
  defaults to `FLAGSHIP_CHAINS` and `FLAGSHIP_THIN` (10). The flagship has one
  posterior mode over all ten chains, so `FLAGSHIP_CHAINS` is `None` and every
  chain is read. A multimodal flagship would name its majority there instead,
  and a caller would pass `chains=None` on purpose to re-admit the rest.
