# US vs China frontier — all / open / closed

## Run

```
python 3_fit/fit.py --preset canonical --chains 4 --draws 2000 --tune 2000
python 3_fit/fit.py --preset canonical --open-only --chains 4 --draws 2000 --tune 2000
python 3_fit/fit.py --preset canonical --closed-only --chains 4 --draws 2000 --tune 2000
./make_plots.sh
```

Rebuild inputs when data changes:

```
python -m multiaxis_eci sync   # 0_input/models.csv carries the country of every model
python 4_diagnostics/1_country_frontier.py [--open-only|--closed-only] --fit-start 2024-10-01 --y-range 50,255
python 4_diagnostics/2_plot_crossovers.py
python 6_writeups/us_cn_frontier/make_tables.py
```

## Inputs

- `0_input/benchmarks.csv` — access class per benchmark (open = `public`), from the pipeline
- `1_curated/model_country.csv` + `model_country_overrides.csv` — US/CN/Other per model

## Outputs (this folder)

- `country_frontier_{canonical,canonical_open,canonical_closed}.{png,html,csv}`
- `country_crossover_{canonical,canonical_open,canonical_closed}.csv`
- `country_crossovers.{png,html}`
- `summary_tables.png`
- `open_benchmarks.txt`
- per-draw arrays: `5_outputs/<data generation>/comparisons/country_frontier_draws_*.npz`
