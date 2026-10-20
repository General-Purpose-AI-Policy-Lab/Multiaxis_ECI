# How far behind the closed frontier are open-weights models? ECI-H by benchmark access

The question of [Ihle (2026), *How far behind are open models?*](https://www.lesswrong.com/posts/rJcCrXyEsJKmmDpWG/how-far-behind-are-open-models), asked on our ECI-H scale instead of per-benchmark score thresholds: how many months does the open-weights frontier trail the closed one, and does the answer change with the access class of the benchmarks?

Data generation 2026-10-19 (pipeline commit `2216933`: the feeds refreshed on 2026-10-19, with the seven CAISI cyber suites, CyScenarioBench as run by Anthropic, SpireClimb and the lab's own Claude runs), today pinned at 2026-10-19. Figures and tables are copies of `5_outputs/data20261019/comparisons/frontier_gap_openness_*`; French versions in `fr/`. The months-behind figure, English and French, is drawn in the lab's print style (`viz.frontier_gap_print`, vector twins in `svg/` and `fr/svg/`; the `.html` twin is the interactive one): from 2024, axis cut at 20 months (a few intervals run off the top), overlapping record dots drawn as one disc split by colour.

![Months behind](frontier_gap_openness_lag.png)

## Results

Open-weights records of the last twelve months, median [80% interval]:

| | All (93) | Public (61) | Semi-private (10) | Private (19) |
|---|---|---|---|---|
| Months behind the closed frontier | 7.0 [6.2, 8.0] | 6.2 [5.2, 7.5] | 7.0 [6.1, 8.2] | 8.0 [6.7, 9.6] |

- **About seven months.** On all benchmarks the open records of the last year sit where the closed frontier stood 7.0 months earlier; the trend lines through the records say the same order of magnitude (9.1 [7.7, 10.5] months today, gap over the closed slope). The closed frontier gains 23.0 ECI-H a year against 18.7 for open weights: the gap in points is widening.
- **Private benchmarks show a longer lag**, 1.8 [0.1, 3.6] months more than public benchmarks (two fits on disjoint benchmarks, so an independent comparison). This is Ihle's direction (8-10 months private, 4-6 public), smaller than on the 2026-10-01 data (2.4 [0.6, 4.2]).
- **Semi-private sits in between** (7.0 months, against 6.2 on public and 8.0 on private).
- **The private trend lines no longer diverge.** On private benchmarks the open records rise as fast as the closed ones (25.2 against 24.5 ECI-H a year, 7 open records), so the trend-line lag there (7.5 [5.4, 9.6] months) is shorter than on public benchmarks (9.8); the record lags above remain the headline.

The three lags per scope: [`frontier_gap_openness_table.md`](frontier_gap_openness_table.md); every quantity (slopes, gaps in ECI-H) and the differences between scopes, private − public among them, in [`frontier_gap_openness_table.csv`](frontier_gap_openness_table.csv).

![Frontier panels](frontier_gap_openness_panels.png)

## Method

- **Fits.** The canonical K=1 ECI-H fit on its 93 benchmarks (`--human-merge`, 8 x 2,500 draws) and three quick fits (4 x 2,000) on one access class each. The index leaves out, besides the benchmarks built to be easy for humans, the six mainly visual ones (VISTA, GeoBench, BlueprintBench 2, VisualToolBench, EnigmaEval, Furniture Assembly: humans at or above the best models, a few labs' vision models favoured, open models rarely scored); SpireClimb, played through screenshots, stays in like OSWorld (user, 2026-10-20). Classes come from the pipeline: a scored set anyone can download without a human reviewing the request is public (an automatic gate counts), a held-out scored set with a released sample is semi-private (an illustrative example is not a sample), a set never released or given on reviewed request is private. The classes agree with Ihle's on all 17 benchmarks of his post. List: [`frontier_gap_openness_benchmark_classes.md`](frontier_gap_openness_benchmark_classes.md).
- **One scale.** The all-benchmarks fit is pinned by its two anchors (Claude 3.5 Sonnet Oct 2024 = 130, GPT-5 medium = 150). Each class fit is linked onto it through the closed OpenAI, Anthropic and Google models with at least four scores in the class (mean-sigma, per draw): the anchors alone hold two to four scores per class, other labs' models may be tuned to some benchmarks, and linking through open models would erase the difference being measured.
- **Open or closed** (`1_curated/model_openness.csv`): open when the weights are downloadable at release or within 30 days (GLM-5.3 open, grok-2 closed), and a hosted version of open weights is open (Qwen3.5-Plus).
- **Records.** Candidates are dated releases with at least two scores in the scope; a group's frontier is the running maximum of posterior-median ECI-H, one effort per family and one release per day. In a class, an open record only counts if the release is within 5 ECI-H of the open record of its date on all benchmarks (in 2024 the open "records" on private benchmarks were Llama 3.1 8B and Pixtral 12B, the open models that happened to be scored).
- **Months behind.** Per posterior draw, the months between an open record and the first closed model at or above it. A level the class's closed models already exceeded when first scored is read off the all-benchmarks closed frontier instead; on all and public benchmarks, where nothing precedes davinci (2020), it is dated back at the closed frontier's early rate and a record that needs this in more than two thirds of draws is dropped. Curves are 6-month Gaussian smooths of the per-draw lags.
- **Trend lines.** One straight line per group through its records since 2024-10-01, each weighted by its posterior SD plus a common dispersion; the lag today is the gap over the closed slope (how long ago the closed line stood where the open line stands).

## Caveats

- Class fits are quick fits (max r̂ ≤ 1.014, at most 52 divergences out of 8,000, on private). The private fit's own tables read ECI-H off Claude 3.7 Sonnet and GPT-5 medium (`--anchors`): Claude 3.5 Sonnet has no private score left.
- Open coverage is thin outside public benchmarks: 28 open candidates on private benchmarks against 103 closed, 7 records.
- Gaps in ECI-H points depend on the linking; lags in months do not (any positive affine map per draw leaves them unchanged).
- The human tiers of a class are placed by the few baselines left in it; on private benchmarks they sit implausibly high, and no number here reads them.
- The frontier accelerates over 2024-2026, so a straight line through the records is a summary, not a model; the record lags are the headline.

## Run

```bash
python 1_curated/3_build_model_openness.py --pipeline ../benchmark-data-pipeline   # after a sync, on the checkout of the synced run
python 3_fit/fit.py --preset canonical --human-merge
python 3_fit/fit.py --preset canonical --access public       --human-merge --chains 4 --draws 2000 --tune 2000
python 3_fit/fit.py --preset canonical --access semi_private --human-merge --chains 4 --draws 2000 --tune 2000
python 3_fit/fit.py --preset canonical --access private      --human-merge --chains 4 --draws 2000 --tune 2000 \
    --anchors claude-3-7-sonnet-20250219,gpt-5-2025-08-07_medium   # Claude 3.5 Sonnet has no private score left
TODAY=2026-10-19 5_outputs/open_closed_frontier/make_plots.sh
```

The code is `analysis.frontier_gap`, run per scope by `4_diagnostics/1_frontier_gap.py` and put side by side by `4_diagnostics/2_plot_frontier_gap.py`. The records CSVs (`frontier_gap_openness_<scope>_records.csv`) give every open record's lag, the closed models that first beat it and the share of borrowed or dated-back draws.
