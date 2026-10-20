# How far behind the US frontier are Chinese models? ECI-H by benchmark access

How many months the frontier of Chinese models trails the frontier of US models on our ECI-H scale, and whether the answer changes with the access class of the benchmarks. This complements the open-weights vs closed cut of [`../open_closed_frontier/`](../open_closed_frontier/) with the country cut, on the same fits and the same method.

Data generation 2026-10-19 (pipeline commit `2216933`: the feeds refreshed on 2026-10-19, with the seven CAISI cyber suites, CyScenarioBench as run by Anthropic, SpireClimb and the lab's own Claude runs), today pinned at 2026-10-19. Figures and tables are copies of `5_outputs/data20261019/comparisons/frontier_gap_country_*`; French versions in `fr/`. The months-behind figure, English and French, is drawn in the lab's print style with two curves only, public and private benchmarks (`viz.frontier_gap_print`, vector twins in `svg/` and `fr/svg/`; the `.html` twin is the interactive one): from 2024, axis cut at 20 months (a few intervals run off the top), overlapping record dots drawn as one disc split by colour.

![Months behind](frontier_gap_country_lag.png)

## Results

Chinese records of the last twelve months, median [80% interval]:

| | All (93) | Public (61) | Semi-private (10) | Private (19) |
|---|---|---|---|---|
| Months behind the US frontier | 6.9 [6.1, 7.9] | 6.1 [5.1, 7.3] | 7.4 [6.5, 8.6] | 7.7 [6.5, 9.3] |

- **About seven months.** On all benchmarks the Chinese records of the last year sit where the US frontier stood 6.9 months earlier; the trend lines through the records say the same order of magnitude (9.0 [7.4, 10.4] months today, gap over the US slope). The US frontier gains 23.0 ECI-H a year against 18.9 for Chinese models: the gap in points is widening.
- **Private benchmarks: a longer lag, not a clear one.** 1.6 [−0.1, 3.3] months more than public benchmarks (two fits on disjoint benchmarks), an 80% interval that just includes zero; the private scope has 7 Chinese records.
- **Semi-private and private show the longest lags** (7.4 and 7.7 months, against 6.1 on public).
- **The same answer as the open-weights cut.** Most Chinese records are open-weights models (Kimi, GLM, DeepSeek, Qwen's and Xiaomi's open series), with a few API-only models among them; on the same data the open-weights lag is 7.0 months on all benchmarks.

The three lags per scope: [`frontier_gap_country_table.md`](frontier_gap_country_table.md); every quantity (slopes, gaps in ECI-H) and the differences between scopes in [`frontier_gap_country_table.csv`](frontier_gap_country_table.csv).

![Frontier panels](frontier_gap_country_panels.png)

## Method

- **Fits.** The canonical K=1 ECI-H fit on its 93 benchmarks (`--human-merge`, 8 x 2,500 draws) and three quick fits (4 x 2,000) on one access class each, the same fits as the open/closed note. The index leaves out the benchmarks built to be easy for humans and the six mainly visual ones (VISTA, GeoBench, BlueprintBench 2, VisualToolBench, EnigmaEval, Furniture Assembly); SpireClimb, played through screenshots, stays in like OSWorld (user, 2026-10-20). Access classes come from the pipeline; list: [`frontier_gap_country_benchmark_classes.md`](frontier_gap_country_benchmark_classes.md).
- **Groups.** A model's country is the pipeline's `country` column (its developer's); models of other countries are left out.
- **One scale.** Each class fit is linked onto the all-benchmarks fit through the US models of OpenAI, Anthropic and Google with at least four scores in the class (mean-sigma, per draw).
- **Records, months behind, trend lines.** As in the open/closed note: running maximum of posterior-median ECI-H per group; per draw, the months between a Chinese record and the first US model at or above it; one straight line per group through its records since 2024-10-01, the lag today being the gap over the US slope.

## Caveats

- Class fits are quick fits (max r̂ ≤ 1.014, at most 52 divergences out of 8,000, on private).
- Chinese coverage is thin outside public benchmarks: 25 Chinese candidates on private benchmarks against 105 US ones, 7 records.
- Gaps in ECI-H points depend on the linking; lags in months do not.
- The frontier accelerates over 2024-2026, so a straight line through the records is a summary, not a model; the record lags are the headline.

## Reproduce

```bash
TODAY=2026-10-19 5_outputs/us_cn_frontier/make_plots.sh
```

It runs the open/closed script with `GROUP=country` (`4_diagnostics/1_frontier_gap.py --group country` on the four scopes, then `2_plot_frontier_gap.py --group country --print-scopes public,private`) and copies the deliverables here.
