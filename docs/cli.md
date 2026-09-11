# CLI reference

Every flag `3_fit/fit.py` accepts, where a fit's output lands, and the commands that
plot, diagnose and publish it. The [README](../README.md) covers the two runs
that matter; this file is the rest of the surface.

`python 3_fit/fit.py --help` is the generated version of the flag tables below.

## Flags

`[canon]` applies to `--preset canonical` only, `[expl]` to exploration only,
unmarked to both.

**Sampling**

| flag | effect |
|---|---|
| `--draws N` | posterior draws per chain. Default 10,000 canonical, 2,000 exploration |
| `--tune N` | tuning steps per chain. Default 2,000 |
| `--chains N` | chains and cores. Default 8 |
| `--sampler {pymc,nutpie,numpyro}` | NUTS backend. Default `nutpie` |
| `--target-accept X` | `[expl]` default 0.95. Raise toward 0.99 if divergences appear |
| `--seed N` | `[expl]` override seed 42. Nutpie is deterministic given seed + data + model, so a multi-run recipe must vary this |
| `--stream-draws` / `--no-stream-draws` | on by default: nutpie writes every draw to `<fit>/live_draws.zarr` as it lands instead of holding the run in RAM (a 10,000 x 8 K=4 run sat at 13 GB in memory on 2026-09-11; streamed, it needs a few hundred MB while sampling and about 8 GB of free disk, warmup included). A killed run keeps what it had (`persistence.load_live_draws` reads a partial store); the store is deleted once the thinned trace is saved. `--no-stream-draws` keeps the old in-RAM behaviour |

**Model**

| flag | effect |
|---|---|
| `--K N` | `[expl]` latent dimension. Default 4 |
| `--loading-prior {normal,signed,pt1,bifactor}` | `[expl]` default `normal`, non-negative. `signed` allows a contrast axis and is rotation-invariant; `pt1` is `normal` under product-to-one loadings per axis (Epoch's identification); `bifactor` is a dense general column plus horseshoe specifics, `--K >= 2` |
| `--link {linear,loglog}` | `[expl]` `linear` is the 2PL; `loglog` is a disjunctive best-axis family |
| `--human-prior` | order human tiers by `multiaxis_eci.config.HUMAN_ORDER`, a tree partial order. Also accepted by `--preset canonical`, which then writes to `canonical_humanprior/` under the data generation's folder |
| `--human-merge` | instead use `multiaxis_eci.config.HUMAN_ORDER_MERGED`, which merges the High School branch into the adult spine via a max over parents. Also accepted by `--preset canonical` (`canonical_humanmerge/`) |
| `--lineage-prior` | `[expl]` soft vendor release-chain prior: each release's mean step over its predecessor is positive, but a node can regress |
| `--lineage-bm` | `[expl]` with `--lineage-prior`: index the chain by time, so each step scales with the release gap in years |
| `--theta-pos` | `[expl]` eta reads softplus(theta), the semi-compensatory convention. Raw theta stays the reported ability |
| `--time-prior` | `[expl]` add a learned per-axis linear trend in release year to the theta prior MEAN, so a thinly-evaluated model is shrunk toward its era's level rather than the whole population's. The slope is signed and centered at zero, so a flat population reduces to the plain prior |
| `--theta-t` | `[expl]` cell-wise leptokurtic theta: each (model, axis) cell of the exchangeable block gets a Student-t(4) marginal (direct closed-form density, no extra latents) instead of a Gaussian one |
| `--private-bases` | `[expl]` give each human root and chain founder a private Normal(0,1) base and let the ZeroSumNormal span only the unstructured rows. Same marginal scale; changes how much of the population the location pin carries |

**Data scope**

| flag | effect |
|---|---|
| `--apply-exclusions` | `[expl]` apply `excluded_benchmarks.txt`, i.e. fit the canonical scope |
| `--include-all-benchmarks` | `[canon]` the mirror: keep the curated-excluded benchmarks |
| `--drop-benchmarks A,B` | `[expl]` drop the named benchmarks (comma-separated, exact names) for a sensitivity run |
| `--access CLASS` | `[canon]` keep one access class of benchmarks only (the `access` column of `0_input/benchmarks.csv`): `public`, `semi_private`, `private`, or `nonpublic` for everything but public; results go to `canonical_<CLASS>/`. The ECI anchors must keep observations in the class (checked before sampling) |
| `--open-only` / `--closed-only` | `[canon]` aliases of `--access public` / `--access nonpublic` |
| `--simpleqa-original` | `[expl]` append OpenAI's original SimpleQA (`1_curated/simpleqa_original/`) as a column separate from SimpleQA Verified (different set and grader); adds 2023-2024 era rows |
| `--no-sg` | `[expl]` drop the Skilled Generalist tier's observations. The tier keeps its slot in the human-order prior, so its theta becomes prior-only |
| `--drop-zero-scores` | `[canon]` drop `score == 0` observations. Diagnostic: tells whether the zero rows drive bad NUTS geometry |
| `--eci-data-only` | `[canon]` fit `1_curated/eci_data.csv`, the original reference ECI dataset, instead of the pipeline's score view |

**Likelihood**

| flag | effect |
|---|---|
| `--no-floors` | `[expl]` drop the chance floors, which are on by default. Emits the `_nofloors` tag token, so the sensitivity run gets its own folder |
| `--no-pooled-noise` | `[expl]` drop the hierarchical `sigma_b`, which is on by default, and give a thin benchmark a free scale. Emits the `_unpooled` tag token |
| `--ceiling-noise` | `[expl]` estimate a per-benchmark upper asymptote confined to a noise-sized gap, Beta(1,20). Grading noise, not walls |
| `--keep-isolated-families` | keep the model families (base model + snapshot, all efforts and variants together) seen on a single benchmark in the fit's scope; dropped by default since 2026-09-08 because they carry no cross-benchmark information and the 2020-2022 tail of them fed a second posterior mode. Both modes; emits `_keepiso` |
| `--censor-bounds` | censored Beta likelihood at the score bounds: a reported 0 (or 1) is the event y ≤ eps_b (y ≥ 1 − eps_b), eps_b = 1 / (2 N_b) from `1_curated/benchmark_n_items.csv` (half an item), instead of a density at a clipped 0.001. Both modes; emits `_censor` |
| `--known-se` | `[expl]` split the Beta noise: fixed per-cell instrument precision from the reported harness stderr (`n_eff = p(1-p)/se^2`), so `sigma_b` becomes excess-only. Cells without stderr are unchanged |

**Run control and output**

| flag | effect |
|---|---|
| `--preset canonical` | K=1, pt1 loading prior, curated exclusions, humans in, full ECI-H deliverables to `5_outputs/<data generation>/canonical/` |
| `--skip-sampling` | `[canon]` reuse that folder's `trace.nc`. Must match the current data shape |
| `--raw-c` | `[canon]` report raw C instead of anchored ECI-H |
| `--skip-baseline` / `--refit-baseline` | `[expl]` skip or force the K=1 baseline fit |
| `--plots` | `[expl]` render the fit's figures in-process |

## Where a fit's output goes

Everything a fit or a diagnostic writes lives under `5_outputs/`, whose first level is the data generation: `data<YYYYMMDD>`, the build date of the pipeline tables in `0_input/provenance.json` (`config.DATA_TAG`). Fits on two data generations therefore never overwrite each other. `5_outputs/pre_pipeline/` holds the fits published with the post, on the pre-pipeline dataset, kept as they were.

Inside a generation, one folder per fit holds its tables, its trace and its figures (`figures/` for the PNGs, `figures/html/` for the interactive twins, `figures/fr/` for the French versions); `comparisons/` holds the cross-fit tables (frontier gap, chain verdicts) with their own `figures/`; `diagnostics/` the one-off diagnostic outputs.

A fit's flags become one tag, and the tag names the results folder and the trace, so the two cannot drift apart. A default contributes no token, so the K=4 command above reduces to:

```
flags    --K 4 --human-merge --lineage-prior --lineage-bm
tag      _humanmerge_lineageprior_lineagebm
results  5_outputs/data20260908/mirt_humanmerge_lineageprior_lineagebm/
trace      ├── trace_mirt_k4_humanmerge_lineageprior_lineagebm.nc
figures    └── figures/k4/            (PNG; html/ beneath it)
```

`FitSpec.from_trace` reads that identity back off a trace, so a trace path is
the only thing a plotting or diagnostic caller has to name.

## Frontier gap: open weights vs closed, by benchmark access (reproduction steps 1-2)

```bash
python 1_curated/3_build_model_openness.py                      # open / closed per model, after a sync
python 4_diagnostics/1_frontier_gap.py --access all              # then public, semi_private, private
python 4_diagnostics/2_plot_frontier_gap.py
```

`1_frontier_gap.py` reads one canonical trace, the fit on all benchmarks
(`canonical/`) or on one access class (`canonical_<CLASS>/`, from `3_fit/fit.py
--preset canonical --access CLASS`), and compares the two groups' frontiers in
ECI-H: records per group, the months every open record trails the closed
frontier of the same posterior draw (a record whose crossing has to be dated
back from the closed field's first measured day in more than two thirds of the
draws is left out: `analysis.frontier_gap.MAX_DATED_BACK_FRAC`), one trend line
per group (frontier points
released since `--fit-start`, default `2024-10-01`, in the running top-`--top-k`,
default 2) with the gap and the lag it implies, and each line's human-tier
crossings. `--group country` compares US and CN instead; `--results-dir DIR`
overrides the trace folder; `--min-obs N` (default 2) is the K=1 reading of the
coverage rule, a candidate needs that many scores in the scope; `--allow-stale`
joins a trace that predates the data snapshot by model name instead of refusing;
`--today DATE` pins the today line.
Outputs land in the data generation's `comparisons/` as
`frontier_gap_<group>_<scope>_*`. `2_plot_frontier_gap.py` draws every scope it
finds side by side (panels, the months-behind figure, the
summary table with each scope's change against all benchmarks; the crossing dates stay in CSV) with no trace
loaded; `--y-range LO,HI` pins the panels' axis. Every figure is also written in French
under `comparisons/figures/fr/` (`<stem>_fr.png`, the vector twin in `fr/svg/`), the
`viz.i18n` string walk over the finished English figure. The write-up's
`5_outputs/open_closed_frontier/make_plots.sh` runs both over the four scopes and copies
the deliverables, French included, into that folder.

## Plot a fit

```bash
python 4_diagnostics/3_plot_mirt.py --trace \
  5_outputs/data20260908/mirt_humanmerge_lineageprior_lineagebm/trace_mirt_k4_humanmerge_lineageprior_lineagebm.nc
```

Figures land in the fit's own `figures/k{K}/`, one PNG each with the HTML twin
under `html/`. `--forecast` adds the frontier-projection set, `--thin N` keeps every
n-th draw, `--axes N` limits the axis count, `--out DIR` overrides the
destination. Full catalogue: [plots.md](plots.md).

### Plot everything

```bash
python 4_diagnostics/3_plot_mirt.py --folder 5_outputs/data20260908/ --dry-run   # decisions only
python 4_diagnostics/3_plot_mirt.py --folder 5_outputs/data20260908/             # render
```

Folder mode globs `DIR/*/*.nc` plus `DIR/*.nc`, so pointing at one fit folder
also works. It auto-thins by file size, one kept draw per 2 GB, and `--thin`
overrides: the 38 GB K=4 trace gets thin 19, which is what fits it in 26 GB of
RAM. One child process per trace, so a single failure does not end the sweep.

## Diagnose a fit

The fit itself prints the verdict that counts. Convergence is judged on
identified quantities only, `eta`, `D` and `sigma_b`, because raw per-axis
r-hat on `A` and `theta` is permutation-inflated: two chains that found the
same fit disagree only on which axis they call axis 1, and r-hat cannot tell
that apart from a real disagreement. The sampling log carries both lines, the
raw r-hat labelled as ignorable and the identified r-hat beside the divergence
count over its denominator.

Two scripts go further when chains disagree.

```bash
python 4_diagnostics/diagnose_chains.py --trace TRACE --name LABEL
python 4_diagnostics/diagnose_chains.py --trace TRACE --write-modes
python 4_diagnostics/theta_bimodality.py --trace TRACE
```

`diagnose_chains.py` asks whether the chains found one solution or several. It
prints a row per chain, giving that chain's log-probability gap to the best
chain, how well its loading columns match the pooled mean, and which basin it
sits in, then a verdict. On the small K=2 demo trace
(`5_outputs/pre_pipeline/mirt/trace_mirt_k2.nc`) it reports `chains=4 divergences=0`, a
98.2-nat logp spread with one island chain, `eta r-hat: all=1.534
majority=1.225`, and `VERDICT: ISLANDS | recommended drop_chains = none`. A
verdict row is appended to the data generation's `comparisons/chain_verdicts.csv`.

Three refinements: `--fig` also renders the per-chain diagnostic figure,
`--match-thresh X` overrides the loading-match threshold behind the basin
assignment, and `--out-csv PATH` redirects the verdict row.

`--write-modes` persists the split to
`<fit>/mirt_modes_<trace-stem>.json` and stops, loading no data, so a
superseded trace still splits. The dashboard only reads that file; when it is
present a multimodal fit gets one extra loading and timeline figure set per
mode, labelled with its chains and Δlogp. On the demo trace it writes
`2 mode(s)` (the majority chains 0/2/3 and the island chain 1). Mode
restriction is plot-side only: convergence, PPC, PIT and GoF on the card
always describe the whole fit.

`theta_bimodality.py` answers which test-takers the split actually moves. It
reports per-axis how many abilities are bimodal across chains, before and after
axis alignment, and writes `<fit>/theta_bimodality.csv` plus a
`bimodality.html` viewer. On the demo trace it reports
`split takers: raw 210/829 -> aligned 1/829`, so axis alignment absorbs almost
all of the apparent splits. It defaults to the flagship K=4 fit when `--trace`
is omitted.

## Build the dashboard

```bash
python 4_diagnostics/4_build_dashboard.py --force-all
```

Renders every registered fit into the tracked repo-root `index.html`: a fit
selector, per-fit figures rendered lazily, and a cross-fit comparison view
(GoF, LOO). Always pass `--force-all` when serving results so no stale card
comes from cache. Open `index.html` in a browser; no server needed.

The tracked `index.html` cannot be rebuilt from a fresh clone: it renders from
the `.nc` traces, which are gitignored (the K=4 trace alone runs to tens of GB) and must
be re-fitted first. Treat the committed dashboard as a published artifact of the
snapshot it was built from, not as something the repo regenerates on demand.
`--png` / `--pdf` also dump stills to the data generation's `dashboard_stills/`.

### Manage the cards

Cards are managed by command, not by editing source.

```bash
python 4_diagnostics/4_build_dashboard.py --list
python 4_diagnostics/4_build_dashboard.py --add TRACE --name NAME --label LABEL
python 4_diagnostics/4_build_dashboard.py --remove NAME
python 4_diagnostics/4_build_dashboard.py --force NAME       # re-render one card
```

`--add` reads the fit's identity off the trace with `FitSpec.from_trace`,
validates it, appends it to the tracked registry
`6_writeups/dashboard/dashboard_fits.json`, and renders nothing. `--name` is the cache
key and the `--force` target, so it must be unique; `--label` is the section
header and nav entry. Four options refine the card: `--type` is one of `data`,
`baseline`, `exploratory` (the default), `confirmed`; `--short` is the axis
tick in cross-fit comparison charts, where the long label makes the graph
unreadable; `--nav` overrides the menu entry; `--forecast` adds the trend and
crossover figures.

`--list` prints every card with an `origin` column, `code` for the entries in
`4_build_dashboard.py` and `json` for the ones `--add` wrote. `--remove` drops a
`json` entry. Validation runs before any trace opens, so a typo in a spec flag
fails in the first second rather than mid-render. A card whose trace and render
cache are both absent (every card, on a fresh clone) is warned about and
skipped, not fatal; the build exits non-zero only when NO card can render.
