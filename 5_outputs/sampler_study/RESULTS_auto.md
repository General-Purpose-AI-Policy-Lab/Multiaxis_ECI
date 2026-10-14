# Night of 2026-10-12: automatic results

## Fast model (minimal JAX version): multistart optimisation

| variant | starts | -logpost best | spread of the top 20 | distinct optima among the top 20 (non-negative: loading match < 0.9; signed: largest principal angle > 20°) |
|---|---|---|---|---|
| base K=3 | 60 | 13976.5 | 48.6 | 18 / 20 |
| base K=4 | 98 | 13483.8 | 18.7 | 20 / 20 |
| gauss K=4 | 60 | 1345.9 | 38.9 | 20 / 20 |
| signed K=4 | 60 | 13408.5 | 16.6 | 20 / 20 |

## Full model: multistart MAP (L-BFGS, JAX)

| run | starts | best log posterior (incl. Jacobian) | loglik at the best | nearest NUTS modes of the top 10 |
|---|---|---|---|---|
| map200 | 147 | -16254.9 | -11261.6 | C 0.62, D 0.566, D 0.672, D 0.706, C 0.709, A 0.603, D 0.689, D 0.664, A 0.667, A 0.607 |
| map60 | 45 | -16594.6 | -11585.5 | B 0.66, E 0.68, D 0.582, E 0.625, D 0.64, D 0.704, E 0.67, C 0.568, C 0.714, D 0.607 |
| mapj03 | 100 | -16614.6 | -11721.5 | D 0.693, D 0.622, A 0.621, C 0.647, D 0.592, C 0.562, D 0.576, D 0.641, E 0.642, A 0.614 |

## NUTS started from MAP optima vs plain NUTS (1800 + 300, 10 chains)

| run | chains above -14395 (regular chains) | stuck in noise-collapse spikes | median loglik | best loglik | best logpost | families above |
|---|---|---|---|---|---|---|
| control_nuts2 | 5/10 | 0 | -14411 | -14373 | -21112 | B,D,E |
| control_nuts3 | 2/10 | 0 | -14423 | -14389 | -21140 | D,E |
| control_nuts4 | 7/10 | 0 | -14385 | -14364 | -21105 | B,D,E |
| control_nuts5 | 3/10 | 0 | -14414 | -14379 | -21125 | B,C |
| control_nuts6 | 3/10 | 0 | -14433 | -14363 | -21111 | B,C,D |
| control_nuts7 | 3/10 | 0 | -14421 | -14378 | -21124 | A,B,D |
| map_nuts | 0/6 | 4 | -14423 | -14397 | -21144 |  |
| map_nuts2 | 3/7 | 3 | -14396 | -14367 | -21108 | B,D,E |
| map_nuts3 | 1/3 | 7 | -14399 | -14389 | -21147 | B |

Totals: NUTS from MAP optima 4/16; plain NUTS 23/60.
