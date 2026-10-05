#!/bin/zsh
# Open-weights vs closed frontier on ECI-H, by benchmark access class (all / public / semi-private /
# private): the per-scope step on each canonical trace, the cross-scope figures and tables, then a
# copy of the deliverables into this folder — English at the top level (vector twin of the print
# figure in svg/), French under fr/ with the vector twins the lab's site embeds in fr/svg/.
#
# Prerequisites: the four K=1 fits of the current data generation, with the human tier order every
# analysis passes (--human-merge, decision 2026-09-14),
#   python 3_fit/fit.py --preset canonical --human-merge                     # canonical_humanmerge/ (full length)
#   python 3_fit/fit.py --preset canonical --access public       --human-merge --chains 4 --draws 2000 --tune 2000
#   python 3_fit/fit.py --preset canonical --access semi_private --human-merge --chains 4 --draws 2000 --tune 2000
#   python 3_fit/fit.py --preset canonical --access private      --human-merge --chains 4 --draws 2000 --tune 2000 \
#       --anchors claude-3-7-sonnet-20250219,gpt-5-2025-08-07_medium   # Claude 3.5 Sonnet has no private score left
# and the openness labels: python 1_curated/3_build_model_openness.py (after a sync).
#
# SCOPE_SUFFIX names the fit folders to read. _humanmerge (the default) is the merged tier order,
# the one every analysis reads and the one the committed figures come from; SCOPE_SUFFIX= (empty)
# reads the plain canonical*/ set, which is now only a sensitivity check.
#   SCOPE_SUFFIX= TODAY=2026-09-29 5_outputs/open_closed_frontier/make_plots.sh   # without the order
set -e
cd "$(dirname "$0")/../.."
PY=.venv/bin/python
GROUP=${GROUP:-openness}          # GROUP=country for the US vs CN cut (5_outputs/us_cn_frontier/make_plots.sh)
OUT=5_outputs/$([[ $GROUP == country ]] && echo us_cn_frontier || echo open_closed_frontier)
PRINT_SCOPES=${PRINT_SCOPES:-}    # PRINT_SCOPES=public,private: the curves of the print figures (EN, FR)
P=(); [[ -n "$PRINT_SCOPES" ]] && P=(--print-scopes "$PRINT_SCOPES")
TODAY=${TODAY:-}                  # TODAY=2026-09-29 pins the today line
SCOPE_SUFFIX=${SCOPE_SUFFIX-_humanmerge}   # the merged tier order; SCOPE_SUFFIX= reads plain canonical*/
T=(); [[ -n "$TODAY" ]] && T=(--today "$TODAY")
GEN=$($PY -c "import sys; sys.path.insert(0, '2_model'); from multiaxis_eci import config; print(config.RESULTS_DIR)")

for A in all public semi_private private; do
  D=(); [[ -n "$SCOPE_SUFFIX" ]] && \
    D=(--results-dir "$GEN/canonical$([[ $A == all ]] || echo _$A)$SCOPE_SUFFIX")
  $PY 4_diagnostics/1_frontier_gap.py --access $A --group $GROUP $T $D
done
$PY 4_diagnostics/2_plot_frontier_gap.py --group $GROUP $T $P

F=$GEN/comparisons/figures
mkdir -p "$OUT/fr/svg"
for name in panels lag table; do
  cp "$F/frontier_gap_${GROUP}_$name.png" "$OUT/"
  [[ -f "$F/svg/frontier_gap_${GROUP}_$name.svg" ]] && mkdir -p "$OUT/svg" && cp "$F/svg/frontier_gap_${GROUP}_$name.svg" "$OUT/svg/"
  cp "$F/fr/frontier_gap_${GROUP}_${name}_fr.png" "$OUT/fr/"          # French renders
  cp "$F/fr/svg/frontier_gap_${GROUP}_${name}_fr.svg" "$OUT/fr/svg/"  # ... and their vector twins
done
for name in panels lag; do
  cp "$F/html/frontier_gap_${GROUP}_$name.html" "$OUT/"
  cp "$F/fr/html/frontier_gap_${GROUP}_${name}_fr.html" "$OUT/fr/"
done
cp "$GEN/comparisons/frontier_gap_${GROUP}_table.csv" "$GEN/comparisons/frontier_gap_${GROUP}_table.md" \
   "$GEN/comparisons/frontier_gap_${GROUP}_benchmark_classes.md" "$OUT/"
for A in all public semi_private private; do
  cp "$GEN/comparisons/frontier_gap_${GROUP}_${A}_records.csv" "$GEN/comparisons/frontier_gap_${GROUP}_${A}_summary.csv" "$OUT/"
done
echo "done -> $OUT/"
