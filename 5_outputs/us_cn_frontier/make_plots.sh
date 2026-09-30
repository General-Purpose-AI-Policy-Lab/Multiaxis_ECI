#!/bin/zsh
# US vs Chinese frontier on ECI-H, by benchmark access class: the open/closed script with
# GROUP=country, its deliverables copied here. The French print figure keeps two curves, all
# benchmarks and private benchmarks (user, 2026-09-30); PRINT_SCOPES= (empty) draws every scope.
#   TODAY=2026-09-30 5_outputs/us_cn_frontier/make_plots.sh
cd "$(dirname "$0")/../.." || exit 1
GROUP=country PRINT_SCOPES=${PRINT_SCOPES-all,private} exec 5_outputs/open_closed_frontier/make_plots.sh
