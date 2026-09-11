#!/bin/zsh
# Open-weights vs closed frontier on ECI-H, by benchmark access class (all / public / semi-private /
# private): the per-scope step on each canonical trace, the cross-scope figures and tables, then a
# copy of the deliverables into this folder.
#
# Prerequisites: the four K=1 fits of the current data generation,
#   python 3_fit/fit.py --preset canonical                                   # canonical/ (full length)
#   python 3_fit/fit.py --preset canonical --access public       --chains 4 --draws 2000 --tune 2000
#   python 3_fit/fit.py --preset canonical --access semi_private --chains 4 --draws 2000 --tune 2000
#   python 3_fit/fit.py --preset canonical --access private      --chains 4 --draws 2000 --tune 2000
# and the openness labels: python 1_curated/3_build_model_openness.py (after a sync).
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
OUT=5_outputs/open_closed_frontier
GROUP=${GROUP:-openness}          # GROUP=country make_plots.sh for the US vs CN cut
TODAY=${TODAY:-}                  # TODAY=2026-09-11 pins the today line
T=(); [[ -n "$TODAY" ]] && T=(--today "$TODAY")
GEN=$($PY -c "import sys; sys.path.insert(0, '2_model'); from multiaxis_eci import config; print(config.RESULTS_DIR)")

for A in all public semi_private private; do
  $PY 4_diagnostics/1_frontier_gap.py --access $A --group $GROUP $T
done
$PY 4_diagnostics/2_plot_frontier_gap.py --group $GROUP $T

F=$GEN/comparisons/figures
for name in panels lag table; do
  cp "$F/frontier_gap_${GROUP}_$name.png" "$OUT/"
done
for name in panels lag; do
  cp "$F/html/frontier_gap_${GROUP}_$name.html" "$OUT/"
done
cp "$GEN/comparisons/frontier_gap_${GROUP}_table.csv" "$GEN/comparisons/frontier_gap_${GROUP}_table.md" \
   "$GEN/comparisons/frontier_gap_${GROUP}_benchmark_classes.md" "$OUT/"
for A in all public semi_private private; do
  cp "$GEN/comparisons/frontier_gap_${GROUP}_${A}_records.csv" "$GEN/comparisons/frontier_gap_${GROUP}_${A}_summary.csv" "$OUT/"
done
echo "done -> $OUT/"
