#!/bin/zsh
# US-vs-China frontier figures + crossovers + tables, all three benchmark
# scopes (all / open-only / closed-only) on ONE shared ECI scale.
#
# Prerequisites: the three quick fits exist under the current data generation —
#   python 3_fit/fit.py --preset canonical --chains 4 --draws 2000 --tune 2000
#   python 3_fit/fit.py --preset canonical --open-only --chains 4 --draws 2000 --tune 2000
#   python 3_fit/fit.py --preset canonical --closed-only --chains 4 --draws 2000 --tune 2000
set -e
cd "$(dirname "$0")/../.."
PY=python
OUT=6_writeups/us_cn_frontier
# The data generation's output folder (5_outputs/data<YYYYMMDD>/), read from the library.
GEN=$($PY -c "import sys; sys.path.insert(0, '2_model'); from multiaxis_eci import config; print(config.RESULTS_DIR)")

$PY 4_diagnostics/1_country_frontier.py               --fit-start 2024-10-01 --y-range 50,255
$PY 4_diagnostics/1_country_frontier.py --open-only   --fit-start 2024-10-01 --y-range 50,255
$PY 4_diagnostics/1_country_frontier.py --closed-only --fit-start 2024-10-01 --y-range 50,255
$PY 4_diagnostics/2_plot_crossovers.py
$PY $OUT/make_tables.py

for t in canonical canonical_open canonical_closed; do
  cp "$GEN/comparisons/figures/country_frontier_$t.png" "$GEN/comparisons/figures/html/country_frontier_$t.html" \
     "$GEN/comparisons/country_frontier_$t.csv" "$GEN/comparisons/country_crossover_$t.csv" "$OUT/"
done
cp "$GEN/comparisons/figures/country_crossovers.png" "$GEN/comparisons/figures/html/country_crossovers.html" "$OUT/"
echo "done -> $OUT/"
