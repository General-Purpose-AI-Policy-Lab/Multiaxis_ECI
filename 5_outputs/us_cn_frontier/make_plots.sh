#!/bin/zsh
# US vs Chinese frontier on ECI-H, by benchmark access class: the open/closed script with
# GROUP=country, its deliverables copied here. The print figures (English and French) keep two
# curves, public and private benchmarks (user, 2026-10-05; all and private before); PRINT_SCOPES=
# (empty) draws every scope.
#   TODAY=2026-10-01 5_outputs/us_cn_frontier/make_plots.sh
cd "$(dirname "$0")/../.." || exit 1
GROUP=country PRINT_SCOPES=${PRINT_SCOPES-public,private} exec 5_outputs/open_closed_frontier/make_plots.sh
