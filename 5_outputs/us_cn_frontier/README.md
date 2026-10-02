# How far behind the US frontier are Chinese models? ECI-H by benchmark access

How many months the frontier of Chinese models trails the frontier of US models on our ECI-H scale, and whether the answer changes with the access class of the benchmarks. This replaces the open-weights vs closed cut of [`../open_closed_frontier/`](../open_closed_frontier/) with the country cut, on the same fits and the same method.

Data generation 2026-10-01 (pipeline commit `6b68fe1`: the run of 2026-10-01, rebuilt in place on 2026-10-02 without two faulty Kaggle rows), today pinned at 2026-10-01. Figures and tables are copies of `5_outputs/data20261001/comparisons/frontier_gap_country_*`; French versions in `fr/`, vector twins in `fr/svg/`. The French months-behind figure is drawn in the lab's print style (`viz.frontier_gap_print`) with two curves only, all benchmarks and private benchmarks, and its axis stops at 24 months (a few early intervals run off the top).

![Months behind](frontier_gap_country_lag.png)

## Results

Chinese records of the last twelve months, median [80% interval]:

| | All (85) | Public (59) | Semi-private (10) | Private (14) |
|---|---|---|---|---|
| Months behind the US frontier | 7.7 [7.1, 8.6] | 6.8 [5.8, 8.1] | 8.1 [6.9, 9.6] | 9.7 [7.6, 11.5] |

- **About eight months.** On all benchmarks the Chinese records of the last year sit where the US frontier stood 7.7 months earlier; the trend lines through the records say the same order of magnitude (10.6 [9.3, 11.9] months today, gap over the US slope). The US frontier gains 22.2 ECI-H a year against 16.2 for Chinese models: the gap in points is widening.
- **Private benchmarks show a longer lag**, 2.8 [0.2, 5.1] months more than public benchmarks (two fits on disjoint benchmarks).
- **Semi-private sits in between** (8.1 months, against 6.8 on public).
- **The same answer as the open-weights cut.** Most Chinese records are open-weights models (Kimi, GLM, DeepSeek, Qwen's open series), with a few API-only Qwen models among them; on the same data the open-weights lag is also 7.7 months on all benchmarks.

The three lags per scope: [`frontier_gap_country_table.md`](frontier_gap_country_table.md); every quantity (slopes, gaps in ECI-H) and the differences between scopes in [`frontier_gap_country_table.csv`](frontier_gap_country_table.csv).

![Frontier panels](frontier_gap_country_panels.png)

## Method

- **Fits.** The canonical K=1 ECI-H fit on its 85 benchmarks (`--human-merge`, 8 x 2,500 draws) and three quick fits (4 x 2,000) on one access class each, the same fits as the open/closed note. The index leaves out the benchmarks built to be easy for humans and the six mainly visual ones (VISTA, GeoBench, BlueprintBench 2, VisualToolBench, EnigmaEval, Furniture Assembly). Access classes come from the pipeline; list: [`frontier_gap_country_benchmark_classes.md`](frontier_gap_country_benchmark_classes.md).
- **Groups.** A model's country is the pipeline's `country` column (its developer's); models of other countries are left out.
- **One scale.** Each class fit is linked onto the all-benchmarks fit through the US models of OpenAI, Anthropic and Google with at least four scores in the class (mean-sigma, per draw).
- **Records, months behind, trend lines.** As in the open/closed note: running maximum of posterior-median ECI-H per group; per draw, the months between a Chinese record and the first US model at or above it; one straight line per group through its records since 2024-10-01, the lag today being the gap over the US slope.

## Caveats

- Class fits are quick fits (max r̂ ≤ 1.022, at most 16 divergences out of 8,000, on private).
- Chinese coverage is thin outside public benchmarks: 49 Chinese candidates on private benchmarks against 148 US ones, 9 records.
- Gaps in ECI-H points depend on the linking; lags in months do not.
- The frontier accelerates over 2024-2026, so a straight line through the records is a summary, not a model; the record lags are the headline.

## Reproduce

```bash
TODAY=2026-10-01 5_outputs/us_cn_frontier/make_plots.sh
```

It runs the open/closed script with `GROUP=country` (`4_diagnostics/1_frontier_gap.py --group country` on the four scopes, then `2_plot_frontier_gap.py --group country --print-scopes all,private`) and copies the deliverables here.
