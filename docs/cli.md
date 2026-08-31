# CLI reference

Every flag `2_fit.py` accepts, where a fit's output lands, and the commands that
plot, diagnose and publish it. The [README](../README.md) covers the two runs
that matter; this file is the rest of the surface.

`python 2_fit.py --help` is the generated version of the flag tables below.

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
| `--stream-draws` | `[expl]` nutpie writes every draw to `results/<fit>/live_draws.zarr` as it lands, so a killed run keeps what it had. Read a partial store with `multiaxis_eci.persistence.load_live_draws` |

**Model**

| flag | effect |
|---|---|
| `--K N` | `[expl]` latent dimension. Default 4 |
| `--loading-prior {normal,signed,pt1,bifactor}` | `[expl]` default `normal`, non-negative. `signed` allows a contrast axis and is rotation-invariant; `pt1` is `normal` under product-to-one loadings per axis (Epoch's identification); `bifactor` is a dense general column plus horseshoe specifics, `--K >= 2` |
| `--link {linear,loglog}` | `[expl]` `linear` is the 2PL; `loglog` is a disjunctive best-axis family |
| `--human-prior` | `[expl]` order human tiers by `multiaxis_eci.config.HUMAN_ORDER`, a tree partial order |
| `--human-merge` | `[expl]` instead use `multiaxis_eci.config.HUMAN_ORDER_MERGED`, which merges the High School branch into the adult spine via a max over parents |
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
| `--cyber` | `[expl]` append the cyber ECI benchmarks |
| `--open-only` | `[canon]` keep only benchmarks whose items are public; results go to `results/canonical_open/` |
| `--closed-only` | `[canon]` the complement of `--open-only`: only benchmarks NOT public+verified in `1_data/curated/benchmark_access.csv`; results go to `results/canonical_closed/` |
| `--simpleqa-original` | `[expl]` append OpenAI's original SimpleQA (`1_data/curated/simpleqa_original/`) as a column separate from SimpleQA Verified (different set and grader); adds 2023-2024 era rows |
| `--no-sg` | `[expl]` drop the Skilled Generalist tier's observations. The tier keeps its slot in the human-order prior, so its theta becomes prior-only |
| `--drop-zero-scores` | `[canon]` drop `score == 0` observations. Diagnostic: tells whether the zero rows drive bad NUTS geometry |
| `--eci-data-only` | `[canon]` fit `1_data/raw/eci_data.csv`, the original reference ECI dataset, instead of the processed file |

**Likelihood**

| flag | effect |
|---|---|
| `--no-floors` | `[expl]` drop the chance floors, which are on by default. Emits the `_nofloors` tag token, so the sensitivity run gets its own folder |
| `--no-pooled-noise` | `[expl]` drop the hierarchical `sigma_b`, which is on by default, and give a thin benchmark a free scale. Emits the `_unpooled` tag token |
| `--ceiling-noise` | `[expl]` estimate a per-benchmark upper asymptote confined to a noise-sized gap, Beta(1,20). Grading noise, not walls |
| `--known-se` | `[expl]` split the Beta noise: fixed per-cell instrument precision from the reported harness stderr (`n_eff = p(1-p)/se^2`), so `sigma_b` becomes excess-only. Cells without stderr are unchanged |

**Run control and output**

| flag | effect |
|---|---|
| `--preset canonical` | K=1, pt1 loading prior, curated exclusions, humans in, full ECI deliverables to `results/canonical/` |
| `--skip-sampling` | `[canon]` reuse `results/canonical/trace.nc`. Must match the current data shape |
| `--raw-c` | `[canon]` report raw C instead of anchored ECI |
| `--skip-baseline` / `--refit-baseline` | `[expl]` skip or force the K=1 baseline fit |
| `--plots` | `[expl]` render the fit's figures in-process |

## Where a fit's output goes

A fit's flags become one tag, and the tag names the results folder, the trace
and the plots folder, so the three cannot drift apart. A default contributes no
token, so the K=4 command above reduces to:

```
flags   --K 4 --human-merge --lineage-prior --lineage-bm
tag     _humanmerge_lineageprior_lineagebm
results results/mirt_humanmerge_lineageprior_lineagebm/
trace     └── trace_mirt_k4_humanmerge_lineageprior_lineagebm.nc
plots   plots/mirt_k4_humanmerge_lineageprior_lineagebm/
```

`FitSpec.from_trace` reads that identity back off a trace, so a trace path is
the only thing a plotting or diagnostic caller has to name.

## Country frontier and crossovers (reproduction steps 1-2)

```bash
python 3_diagnostics/1_country_frontier.py
python 3_diagnostics/2_plot_crossovers.py
```

`1_country_frontier.py` builds the US/China frontier comparison from the
canonical traces (all three scopes). Flags: `--results-dir DIR` overrides the
default `results/`; `--open-only` / `--closed-only` restrict to one access
scope; `--allow-stale` proceeds when a trace predates the current data snapshot
(otherwise it refuses); `--horizon DATE` sets the forecast horizon;
`--fit-start DATE` (default `2024-10-01`) sets the trend-fit window;
`--y-range LO,HI` pins the y-axis. `2_plot_crossovers.py` renders the
crossover panels from the CSVs step 1 wrote; its "today" line is pinned to the
published snapshot date in the source, not the wall clock.

## Plot a fit

```bash
python 3_diagnostics/3_plot_mirt.py --trace \
  results/mirt_humanmerge_lineageprior_lineagebm_dropFrontierMathv1AlgoTune_floors_poolednoise/trace_mirt_k3_humanmerge_lineageprior_lineagebm_dropFrontierMathv1AlgoTune_floors_poolednoise.nc
```

Figures land in the fit's own `plots/mirt_k{K}{tag}/`, one HTML and one PNG
each. `--forecast` adds the frontier-projection set, `--thin N` keeps every
n-th draw, `--axes N` limits the axis count, `--out DIR` overrides the
destination. Full catalogue: [plots.md](plots.md).

### Plot everything

```bash
python 3_diagnostics/3_plot_mirt.py --folder results/ --dry-run   # decisions only
python 3_diagnostics/3_plot_mirt.py --folder results/             # render
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
python 3_diagnostics/diagnose_chains.py --trace TRACE --name LABEL
python 3_diagnostics/diagnose_chains.py --trace TRACE --write-modes
python 3_diagnostics/theta_bimodality.py --trace TRACE
```

`diagnose_chains.py` asks whether the chains found one solution or several. It
prints a row per chain, giving that chain's log-probability gap to the best
chain, how well its loading columns match the pooled mean, and which basin it
sits in, then a verdict. On the small K=2 demo trace it reports `chains=4
divergences=0`, a 14.1-nat logp spread, `eta r-hat: all=1.271`, and
`VERDICT: ISLANDS | recommended drop_chains = none`. A verdict row is appended
to `results/comparisons/chain_verdicts.csv`.

Three refinements: `--fig` also renders the per-chain diagnostic figure,
`--match-thresh X` overrides the loading-match threshold behind the basin
assignment, and `--out-csv PATH` redirects the verdict row.

`--write-modes` persists the split to
`results/<fit>/mirt_modes_<trace-stem>.json` and stops, loading no data, so a
superseded trace still splits. The dashboard only reads that file; when it is
present a multimodal fit gets one extra loading and timeline figure set per
mode, labelled with its chains and Δlogp. On the demo trace it writes
`1 mode(s)`. Mode restriction is plot-side only: convergence, PPC, PIT and GoF
on the card always describe the whole fit.

`theta_bimodality.py` answers which test-takers the split actually moves. It
reports per-axis how many abilities are bimodal across chains, before and after
axis alignment, and writes `results/<fit>/theta_bimodality.csv` plus a
`bimodality.html` viewer. On the demo trace it reports
`split takers: raw 1/829 -> aligned 0/829`, so alignment absorbs the one
apparent split. It defaults to the flagship K=4 fit when `--trace` is omitted.

## Build the dashboard

```bash
python 3_diagnostics/4_build_dashboard.py --force-all
```

Renders every registered fit into the tracked repo-root `index.html`: a fit
selector, per-fit figures rendered lazily, and a cross-fit comparison view
(GoF, LOO). Always pass `--force-all` when serving results so no stale card
comes from cache. Open `index.html` in a browser; no server needed.

The tracked `index.html` cannot be rebuilt from a fresh clone: it renders from
the `.nc` traces, which are gitignored (the K=4 trace alone is 38 GB) and must
be re-fitted first. Treat the committed dashboard as a published artifact of the
snapshot it was built from, not as something the repo regenerates on demand.
`--png` / `--pdf` also dump stills to `plots/dashboard/`.

### Manage the cards

Cards are managed by command, not by editing source.

```bash
python 3_diagnostics/4_build_dashboard.py --list
python 3_diagnostics/4_build_dashboard.py --add TRACE --name NAME --label LABEL
python 3_diagnostics/4_build_dashboard.py --remove NAME
python 3_diagnostics/4_build_dashboard.py --force NAME       # re-render one card
```

`--add` reads the fit's identity off the trace with `FitSpec.from_trace`,
validates it, appends it to the tracked registry
`3_diagnostics/dashboard_fits.json`, and renders nothing. `--name` is the cache
key and the `--force` target, so it must be unique; `--label` is the section
header and nav entry. Four options refine the card: `--type` is one of `data`,
`baseline`, `exploratory` (the default), `confirmed`; `--short` is the axis
tick in cross-fit comparison charts, where the long label makes the graph
unreadable; `--nav` overrides the menu entry; `--forecast` adds the trend,
crossover and exceedance figures.

`--list` prints every card with an `origin` column, `code` for the entries in
`4_build_dashboard.py` and `json` for the ones `--add` wrote. `--remove` drops a
`json` entry. Validation runs before any trace opens, so a typo in a spec flag
fails in the first second rather than mid-render. A card whose trace and render
cache are both absent (every card, on a fresh clone) is warned about and
skipped, not fatal; the build exits non-zero only when NO card can render.
