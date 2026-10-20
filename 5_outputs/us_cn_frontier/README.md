# How far behind the US frontier are Chinese models? ECI-H by benchmark access

How many months the frontier of Chinese models trails the frontier of US models on our ECI-H scale, and whether the answer changes with the access class of the benchmarks. This complements the open-weights vs closed cut of [`../open_closed_frontier/`](../open_closed_frontier/) with the country cut, on the same fits and the same method.

Data generation 2026-10-01 (pipeline commit `956c91b`: the run of 2026-10-01, rebuilt on 2026-10-05 after the review of the scores with no effort suffix, Chess Puzzles out), today pinned at 2026-10-01. Figures and tables are copies of `5_outputs/data20261001/comparisons/frontier_gap_country_*`; French versions in `fr/`. The months-behind figure, English and French, is drawn in the lab's print style with two curves only, public and private benchmarks (`viz.frontier_gap_print`, vector twins in `svg/` and `fr/svg/`; the `.html` twin is the interactive one): from 2024, axis cut at 20 months (a few intervals run off the top), overlapping record dots drawn as one disc split by colour.

![Months behind](frontier_gap_country_lag.png)

## Results

Months behind the US frontier over the last twelve months, median [80% interval]:

| | All (84) | Public (58) | Semi-private (10) | Private (14) |
|---|---|---|---|---|
| At the Chinese frontier points (records and runners-up) | 6.5 [5.8, 7.2] | 5.1 [4.3, 5.9] | 6.7 [5.9, 7.4] | 6.8 [6.0, 7.6] |
| On an average day (second-best to second-best) | 6.6 [5.7, 7.3] | 5.9 [5.0, 6.8] | 8.0 [7.3, 8.6] | 8.9 [8.4, 9.8] |

- **About six and a half months.** On all benchmarks the Chinese frontier points of the last year (the records and their runners-up, read against the US second-best) sit where the US frontier stood 6.5 months earlier, and the lag on an average day of the year (second row) says the same; the trend lines through those points say the same order of magnitude (9.0 [7.9, 10.0] months today, gap over the US slope). The US frontier gains 25.3 ECI-H a year against 18.5 for Chinese models: the gap in points is widening.
- **Private benchmarks: a longer lag.** 1.7 [0.6, 2.9] months more than public benchmarks at the frontier points (two fits on disjoint benchmarks), 3.0 [1.9, 4.2] on an average day. The second reading is larger because the private scope has few Chinese points (9), months apart: the point lags are the minima of the sawtooth the lag draws between them, and the time average counts the months in between. The private reading stays thin (20 Chinese candidates, two scores each).
- **Semi-private sits between** (6.7 months, against 5.1 on public).
- **The same answer as the open-weights cut.** Most Chinese frontier points are open-weights models (Kimi, GLM, DeepSeek, Qwen's open series), with a few API-only Qwen models among them; on the same data the open-weights lag is 6.7 months on all benchmarks.

The three lags per scope: [`frontier_gap_country_table.md`](frontier_gap_country_table.md); every quantity (slopes, gaps in ECI-H) and the differences between scopes in [`frontier_gap_country_table.csv`](frontier_gap_country_table.csv).

![Frontier panels](frontier_gap_country_panels.png)

## Method

- **Fits.** The canonical K=1 ECI-H fit on its 84 benchmarks (`--human-merge`, 8 x 2,500 draws) and three quick fits (4 x 2,000) on one access class each, the same fits as the open/closed note. The index leaves out the benchmarks built to be easy for humans and the six mainly visual ones (VISTA, GeoBench, BlueprintBench 2, VisualToolBench, EnigmaEval, Furniture Assembly). Access classes come from the pipeline; list: [`frontier_gap_country_benchmark_classes.md`](frontier_gap_country_benchmark_classes.md).
- **Groups.** A model's country is the pipeline's `country` column (its developer's); models of other countries are left out.
- **One scale.** Each class fit is linked onto the all-benchmarks fit through the US models of OpenAI, Anthropic and Google with at least four scores in the class (mean-sigma, per draw).
- **Records, months behind, trend lines.** As in the open/closed note: running maximum of posterior-median ECI-H per group; per draw, the months between a Chinese record and the first US model at or above it (the dot is the posterior mean of that lag, its bar the 50% interval; US candidates with a posterior SD above 6 ECI-H date no crossing); the curve is a 6-month Gaussian kernel through the per-draw record lags; the same lag month by month from the two per-draw frontiers, averaged over the trailing twelve months, is the table's second row; one straight line per group through its records since 2024-10-01, the lag today being the gap over the US slope. The investigation behind the curve and the dots (2026-10-20): [`investigation_2026-10-20/`](investigation_2026-10-20/).

## Caveats

- Class fits are quick fits (max r̂ ≤ 1.034, at most 24 divergences out of 8,000, on semi-private).
- Chinese coverage is thin outside public benchmarks: 20 Chinese candidates on private benchmarks against 99 US ones, 9 frontier points.
- The uncertainty of a record's lag is mostly the US side's: which US model first beat it, and when. Half of the crossing draws of the 2026 records name GPT-5 Pro or GPT-5.2 Pro efforts scored on two or three benchmarks.
- Gaps in ECI-H points depend on the linking; lags in months do not.
- The frontier accelerates over 2024-2026, so a straight line through the records is a summary, not a model; the record lags are the headline.

## Reproduce

```bash
TODAY=2026-10-01 5_outputs/us_cn_frontier/make_plots.sh
```

It runs the open/closed script with `GROUP=country` (`4_diagnostics/1_frontier_gap.py --group country` on the four scopes, then `2_plot_frontier_gap.py --group country --print-scopes public,private`) and copies the deliverables here.
