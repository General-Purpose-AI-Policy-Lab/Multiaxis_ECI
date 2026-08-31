# ECI Bayesian Recreation

A PyMC recreation of the [Epoch Capabilities Index](https://epoch.ai/eci)
(ECI), a scale that places AI models and human baseline tiers on one ability
axis from their benchmark scores. One framework, a K-axis compensatory 2PL
Beta-MIRT (an item-response model with a Beta likelihood), serves the K=4
capability decomposition and the K=1 anchored index alike.

Scope: 4,923 observations, 829 test-takers (models plus human baseline tiers),
96 benchmarks for K=4; 4,184 / 781 / 88 for the canonical K=1 index, which
also applies the curated exclusions. The data comes from five public feeds
(Epoch AI ZIP, Epoch AI live feed, RAND, Scale SEAL, Kaggle Open Benchmarks)
merged by a tracked pipeline notebook.

Model math, priors and identification: [docs/model_math.md](docs/model_math.md).
Figures and how to read them: [docs/plots.md](docs/plots.md).
Data pipeline: [data/pipeline/README.md](data/pipeline/README.md).
Licensing of the upstream scores: [NOTICE.md](NOTICE.md).

## Setup

Any Python >= 3.11 environment works.

```bash
pip install pymc arviz pytensor nutpie numpy pandas scipy plotly kaleido \
            matplotlib requests pytest
plotly_get_chrome -y   # once per env; figure export raises without it
```

Pinned working versions: pymc 5.28.5, arviz 0.23.4, pytensor 2.38.3,
plotly 6.x. `nutpie` (Rust NUTS) is the default sampler, 2-3x faster than PyMC
NUTS on CPU. We develop in a Miniforge conda env named `pymc_env`.

Sanity check (~4 min):

```bash
python -m pytest -m "not slow"
```

## Run the K=4 model

This is the project's main fit: four ability axes. It is the only card the
dashboard forecasts from, and the base the post figures read.

```bash
python fit.py --K 4 --human-merge --lineage-prior --lineage-bm
```

- `--K 4` fits four ability axes instead of one.
- `--human-merge` orders the human tiers, with the High School branch merged
  into the adult spine by a max over parents.
- `--lineage-prior` ties each vendor's releases into a chain whose mean step is
  positive, so a release starts from its predecessor's level.
- `--lineage-bm` indexes that chain by time, so a step's mean and variance grow
  with the release gap in years instead of with the release count.

Four things that shape this fit are defaults, not flags:

- **Non-negative loadings** (`--loading-prior normal`), so each axis is a
  bundle of benchmarks a model is good at. `--loading-prior signed` opts out.
- **Chance floors** (the fixed-c 3PL): scores are clipped up to each
  benchmark's floor and `mu = c + (1-c)·sigmoid`. Floors are read from file and
  never estimated, so no parameters are added. `--no-floors` opts out.
- **Hierarchical benchmark noise**, so a thin benchmark shrinks toward the
  shared median. `--no-pooled-noise` opts out.
- **Retired benchmarks.** FrontierMath v1, FrontierMath Tier 4 v1, AlgoTune and
  MindCube are dropped at load time for every fit, with no flag over them. Both
  v1 columns are superseded by v2, AlgoTune's score needed a transform no other
  benchmark shares, and MindCube's 5-taker panel cannot identify a loading row.

The non-compensatory, sparse-gate and interaction families have their own
drivers: `python -m fits.fit_nc` and siblings.

## Fit the canonical index

```bash
python fit.py --preset canonical
```

Samples 10,000 draws x 8 chains and writes everything to `results/canonical/`:
`trace.nc`, `summary.csv`, `all_models_eci.csv`, `sota.csv`, `timeline.csv`,
forest CSVs, `gof.json`, `pit.csv`. Figures go to `plots/canonical/`. ECI is
the per-draw affine transform of ability pinned at Claude 3.5 Sonnet
(2024-10-22) = 130 and GPT-5 (2025-08-07, medium) = 150, matching Epoch's
dashboard. `--skip-sampling` reuses the existing trace when only the
downstream tables changed.

`results/canonical/trace.nc` predates the current data snapshot. Re-fit before
quoting any canonical number.

## Flags

`[canon]` applies to `--preset canonical` only, `[expl]` to exploration only,
unmarked to both. Every flag `fit.py` accepts is listed below;
`python fit.py --help` is the generated reference.

**Sampling**

| flag | effect |
|---|---|
| `--draws N` | posterior draws per chain. Default 10,000 canonical, 2,000 exploration |
| `--tune N` | tuning steps per chain. Default 2,000 |
| `--chains N` | chains and cores. Default 8 |
| `--sampler {pymc,nutpie,numpyro}` | NUTS backend. Default `nutpie` |
| `--target-accept X` | `[expl]` default 0.95. Raise toward 0.99 if divergences appear |
| `--seed N` | `[expl]` override seed 42. Nutpie is deterministic given seed + data + model, so a multi-run recipe must vary this |
| `--stream-draws` | `[expl]` nutpie writes every draw to `results/<fit>/live_draws.zarr` as it lands, so a killed run keeps what it had. Read a partial store with `persistence.load_live_draws` |

**Model**

| flag | effect |
|---|---|
| `--K N` | `[expl]` latent dimension. Default 4 |
| `--loading-prior {normal,signed,pt1,bifactor}` | `[expl]` default `normal`, non-negative. `signed` allows a contrast axis and is rotation-invariant; `pt1` is `normal` under product-to-one loadings per axis (Epoch's identification); `bifactor` is a dense general column plus horseshoe specifics, `--K >= 2` |
| `--link {linear,loglog}` | `[expl]` `linear` is the 2PL; `loglog` is a disjunctive best-axis family |
| `--human-prior` | `[expl]` order human tiers by `config.HUMAN_ORDER`, a tree partial order |
| `--human-merge` | `[expl]` instead use `config.HUMAN_ORDER_MERGED`, which merges the High School branch into the adult spine via a max over parents |
| `--lineage-prior` | `[expl]` soft vendor release-chain prior: each release's mean step over its predecessor is positive, but a node can regress |
| `--lineage-bm` | `[expl]` with `--lineage-prior`: index the chain by time, so each step scales with the release gap in years |
| `--theta-pos` | `[expl]` eta reads softplus(theta), the semi-compensatory convention. Raw theta stays the reported ability |
| `--time-prior` | `[expl]` add a learned per-axis linear trend in release year to the theta prior MEAN, so a thinly-evaluated model is shrunk toward its era's level rather than the whole population's. The slope is signed and centered at zero, so a flat population reduces to the plain prior |
| `--theta-t` | `[expl]` cell-wise leptokurtic theta: each (model, axis) cell of the exchangeable block gets a Student-t(4) marginal via a per-cell scale mixture instead of a Gaussian one |
| `--private-bases` | `[expl]` give each human root and chain founder a private Normal(0,1) base and let the ZeroSumNormal span only the unstructured rows. Same marginal scale; changes how much of the population the location pin carries |

**Data scope**

| flag | effect |
|---|---|
| `--apply-exclusions` | `[expl]` apply `excluded_benchmarks.txt`, i.e. fit the canonical scope |
| `--include-all-benchmarks` | `[canon]` the mirror: keep the curated-excluded benchmarks |
| `--drop-benchmarks A,B` | `[expl]` drop the named benchmarks (comma-separated, exact names) for a sensitivity run |
| `--cyber` | `[expl]` append the cyber ECI benchmarks |
| `--open-only` | `[canon]` keep only benchmarks whose items are public; results go to `results/canonical_open/` |
| `--closed-only` | `[canon]` the complement of `--open-only`: only benchmarks NOT public+verified in `data/curated/benchmark_access.csv`; results go to `results/canonical_closed/` |
| `--simpleqa-original` | `[expl]` append OpenAI's original SimpleQA (`data/curated/simpleqa_original/`) as a column separate from SimpleQA Verified (different set and grader); adds 2023-2024 era rows |
| `--no-sg` | `[expl]` drop the Skilled Generalist tier's observations. The tier keeps its slot in the human-order prior, so its theta becomes prior-only |
| `--drop-zero-scores` | `[canon]` drop `score == 0` observations. Diagnostic: tells whether the zero rows drive bad NUTS geometry |
| `--eci-data-only` | `[canon]` fit `data/raw/eci_data.csv`, the original reference ECI dataset, instead of the processed file |

**Likelihood**

| flag | effect |
|---|---|
| `--no-floors` | `[expl]` drop the chance floors, which are on by default |
| `--no-pooled-noise` | `[expl]` drop the hierarchical `sigma_b`, which is on by default, and give a thin benchmark a free scale |
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

## Plot a fit

```bash
python diagnostics/plot_mirt.py --trace \
  results/mirt_humanmerge_lineageprior_lineagebm_dropFrontierMathv1AlgoTune_floors_poolednoise/trace_mirt_k3_humanmerge_lineageprior_lineagebm_dropFrontierMathv1AlgoTune_floors_poolednoise.nc
```

Figures land in the fit's own `plots/mirt_k{K}{tag}/`, one HTML and one PNG
each. `--forecast` adds the frontier-projection set, `--thin N` keeps every
n-th draw, `--axes N` limits the axis count, `--out DIR` overrides the
destination. Full catalogue: [docs/plots.md](docs/plots.md).

### Plot everything

```bash
python diagnostics/plot_mirt.py --folder results/ --dry-run   # decisions only
python diagnostics/plot_mirt.py --folder results/             # render
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
python diagnostics/diagnose_chains.py --trace TRACE --name LABEL
python diagnostics/diagnose_chains.py --trace TRACE --write-modes
python diagnostics/theta_bimodality.py --trace TRACE
```

`diagnose_chains.py` asks whether the chains found one solution or several. It
prints a row per chain, giving that chain's log-probability gap to the best
chain, how well its loading columns match the pooled mean, and which basin it
sits in, then a verdict. On the small K=2 demo trace it reports `chains=4
divergences=0`, a 14.1-nat logp spread, `eta r-hat: all=1.271`, and
`VERDICT: ISLANDS | recommended drop_chains = none`. A verdict row is appended
to `results/comparisons/chain_verdicts.csv`.

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
python diagnostics/build_dashboard.py --force-all
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
python diagnostics/build_dashboard.py --list
python diagnostics/build_dashboard.py --add TRACE --name NAME --label LABEL
python diagnostics/build_dashboard.py --remove NAME
python diagnostics/build_dashboard.py --force NAME       # re-render one card
```

`--add` reads the fit's identity off the trace with `FitSpec.from_trace`,
validates it, appends it to the tracked registry
`diagnostics/dashboard_fits.json`, and renders nothing. `--name` is the cache
key and the `--force` target, so it must be unique; `--label` is the section
header and nav entry. Four options refine the card: `--type` is one of `data`,
`baseline`, `exploratory` (the default), `confirmed`; `--short` is the axis
tick in cross-fit comparison charts, where the long label makes the graph
unreadable; `--nav` overrides the menu entry; `--forecast` adds the trend,
crossover and exceedance figures.

`--list` prints every card with an `origin` column, `code` for the entries in
`build_dashboard.py` and `json` for the ones `--add` wrote. `--remove` drops a
`json` entry. Validation runs before any trace opens, so a typo in a spec flag
fails in the first second rather than mid-render.

## Refresh the data

Open `data/pipeline/pipeline.ipynb`, Restart Kernel -> Run All. The last
section swaps fresh outputs into `data/{processed,curated}/`. Then read
`data/pipeline/output/pipeline_report.md` (row counts, score diffs, name
review queue), rebuild the lineage map with
`python diagnostics/build_lineage_map.py`, and re-run the canonical fit.
Revert a refresh with `git checkout data/processed data/curated`. Full
pipeline docs: [data/pipeline/README.md](data/pipeline/README.md).

## Repo map

```
fit.py            # the fit CLI (canonical preset + exploration)
config.py         # paths, priors, sampling defaults, ECI anchors, forecast rule
data.py           # load_eci_data() -> ECIData
models/           # model builders (mirt.py is the main family)
analysis/         # FitSpec, ECI transform, rotations, convergence, forecasts
diagnostics/      # dashboard build, per-fit plots, chain diagnosis, audits
viz/              # Plotly figure builders
data/             # pipeline notebook, processed table, curated files
fits/             # drivers for the non-compensatory / sparse / interaction families
evals/            # local eval harnesses (LAB-Bench cloning); needs OPENROUTER_API_KEY
deliverables/     # figure + table sets built for a specific write-up
results/          # one folder per fit; canonical/ is the index
plots/            # figure output (gitignored)
tests/            # fast unit tests + golden logp locks
docs/             # model math + figure catalogue
index.html        # the all-fits dashboard (tracked)
```

## Reading the results

**Axes have no fixed names, so they are labelled after sampling.** Swapping two
axes leaves every prediction unchanged, because the swap moves the loadings `A`
and the abilities `theta` together and only their product reaches the
likelihood. Each draw therefore picks its own labels, and they are fixed
afterwards: within every draw axes are sorted strongest first, by `tau_A` where
each axis has its own scale, and by realised loading energy under the default
`normal` prior, which shares one `tau_A` across axes. So "axis 2" means "the
second strongest axis" in every draw.

Whether the chains agree on those axes is a separate question and a separate
number. Each chain's mean loading columns are matched to the pooled mean over
every permutation and sign, all 24 of them at K=4, and the median correlation
is reported per axis. That is a diagnostic, not a relabelling: it says which
axis the chains disagree about, which the identified r-hat cannot.

**An ability is trustworthy only where it was measured.** A test-taker's
ability on an axis rests on benchmarks that load on that axis. Models from
2021-2023 took only easy benchmarks, so their hard-axis ability is
extrapolated, not measured, and can land high with a wide interval. Figures
drop those rows through `mirt_informed_mask` (posterior SD < 0.4); the fit and
the diagnostics keep every row.
