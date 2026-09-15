# K=4 without the lineage prior: a refuted experiment, not a usable fit

Do not read this folder as a result. It is the A/B run on 2026-09-14 to test where the flagship fit's divergences come from, and it answered by failing.

`python 3_fit/fit.py --K 4 --human-merge --draws 2000 --tune 2000`, same data as the flagship (5,307 observations, 802 models, 97 benchmarks) and same everything else, with `--lineage-prior --lineage-bm` dropped.

## What it showed

| | flagship, same length | this fit |
|---|---|---|
| divergences | 241 / 16,000 (1.5 %) | **2,453 / 3,200 (76.7 %)**, three chains at 100 % |
| modes (`diagnose_chains`, thresh 0.8) | 3 | **8, one per chain** |
| logp spread | 20.9 nats | **1,777 nats** |
| eta r-hat / D r-hat | 1.148 / 1.102 | **3.410 / 4.085** |
| majority group | [2,4,5,7], eta r-hat 1.040 | **none** |

The hypothesis under test was that the lineage block caused the divergences, since in the flagship fit the divergence indicator correlates +0.64 with `lin_drift` on axis 1 within chain 1 and +0.50 on axis 3 within chain 4, and the thetas that move are whole Claude release chains. The A/B refutes it. The lineage prior is what makes the K=4 fit identifiable at all: without it the eight chains each settle on their own solution, hundreds of nats apart, and there is nothing left to pool. Its `lin_drift` correlation marks where the residual ridge sits, not what creates it.

The `bayesian_r2` of 0.9638 in `mirt_gof_k4.json`, better than the flagship's 0.9606, is a trap: it is the predictive mean of eight unrelated solutions.

## Consequences

Keep `--lineage-prior --lineage-bm` on every K=4 fit. The open question is the axis non-identification itself, not the lineage block.

No figures were rendered here, and `axis_names.json` is an unconfirmed template, both on purpose: a single-chain majority would have produced meaningless panels.

The verdict row is `k4_nolineage` in `../comparisons/chain_verdicts.csv`. The fit this one is compared against is `../mirt_humanmerge_lineageprior_lineagebm_short2000x8/` (same length) and `../mirt_humanmerge_lineageprior_lineagebm/` (the 10,000 x 8 flagship).
