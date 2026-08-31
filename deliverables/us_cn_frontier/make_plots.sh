#!/bin/zsh
# US-vs-China frontier figures + crossovers + tables, all three benchmark
# scopes (all / open-only / closed-only) on ONE shared ECI scale.
#
# Prerequisites: the three quick fits exist —
#   python 2_fit.py --preset canonical --chains 4 --draws 2000 --tune 2000
#   python 2_fit.py --preset canonical --open-only --chains 4 --draws 2000 --tune 2000
#   python 2_fit.py --preset canonical --closed-only --chains 4 --draws 2000 --tune 2000
set -e
cd "$(dirname "$0")/../.."
PY=python

$PY 3_diagnostics/1_country_frontier.py               --fit-start 2024-10-01 --y-range 50,255
$PY 3_diagnostics/1_country_frontier.py --open-only   --fit-start 2024-10-01 --y-range 50,255
$PY 3_diagnostics/1_country_frontier.py --closed-only --fit-start 2024-10-01 --y-range 50,255
$PY 3_diagnostics/2_plot_crossovers.py
$PY deliverables/us_cn_frontier/make_tables.py

for t in canonical canonical_open canonical_closed; do
  cp plots/country_frontier_$t.png plots/country_frontier_$t.html \
     results/comparisons/country_frontier_$t.csv \
     results/comparisons/country_crossover_$t.csv \
     deliverables/us_cn_frontier/
done
cp plots/country_crossovers.png plots/country_crossovers.html deliverables/us_cn_frontier/
echo "done -> deliverables/us_cn_frontier/"
