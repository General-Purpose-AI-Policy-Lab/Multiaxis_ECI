#!/bin/zsh
# all four scopes of one group with --frontier-k 2, into the scratch folder
G=$1; OUT=$2; cd $(git rev-parse --show-toplevel)
GEN=5_outputs/data20261001
for A in all public semi_private private; do
  D=$GEN/canonical$([[ $A == all ]] || echo _$A)_humanmerge
  .venv/bin/python 4_diagnostics/1_frontier_gap.py --access $A --group $G --today 2026-10-01 --results-dir $D --frontier-k 2 --out-dir $OUT 2>&1 | grep -v Warning | grep "candidates, \|average day\|Traceback\|Error" 
done
echo "done $G"
