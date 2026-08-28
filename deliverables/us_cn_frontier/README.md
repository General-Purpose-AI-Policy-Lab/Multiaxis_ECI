# US vs China frontier — all / open / closed

## Run

```
python fit.py --preset canonical --chains 4 --draws 2000 --tune 2000
python fit.py --preset canonical --open-only --chains 4 --draws 2000 --tune 2000
python fit.py --preset canonical --closed-only --chains 4 --draws 2000 --tune 2000
./make_plots.sh
```

Rebuild inputs when data changes:

```
python diagnostics/build_country_map.py
python diagnostics/country_frontier.py [--open-only|--closed-only] --fit-start 2024-10-01 --y-range 50,255
python diagnostics/plot_crossovers.py
python deliverables/us_cn_frontier/make_tables.py
```

## Inputs

- `data/curated/benchmark_access.csv` — access class per benchmark (open = public+verified)
- `data/curated/model_country.csv` + `model_country_overrides.csv` — US/CN/Other per model

## Outputs (this folder)

- `country_frontier_{canonical,canonical_open,canonical_closed}.{png,html,csv}`
- `country_crossover_{canonical,canonical_open,canonical_closed}.csv`
- `country_crossovers.{png,html}`
- `summary_tables.png`
- `open_benchmarks.txt`
- per-draw arrays: `results/comparisons/country_frontier_draws_*.npz`
