# Item counts: what is observed, what is reconstructed

Date 2026-11-03. Data generation 20261019, flagship scope (5,487 obs / 107 benchmarks).

**Observed.** Only the reported score, a share in [0, 1]. No per-question answers.
**Reconstructed.** k = score × N_b. N_b comes from `1_curated/benchmark_n_items.csv`, read from documentation.

- 82 / 107 benchmarks are item-scored (4,539 / 5,487 obs). The other 25 keep the Beta.
- Verification stamps of the 82:
  - 46 primary source;
  - 8 score grid;
  - 4 stderr inversion;
  - 5 estimates;
  - 19 unverified.

**Score-grid check (`grid_check.py`, `grid_check.csv`).** One run on N questions can only score j / N. For each benchmark we measure the share of scores within 0.0005 of the grid.
- 56 / 82 benchmarks are testable, meaning a random score would land on the grid less than half the time. The other 26 have N so large that the grid tells nothing.
- On-grid share among the 56:
  - ≥ 0.9: 15;
  - 0.5 to 0.9: 20;
  - < 0.5: 21.
- For 19 / 21 of the low ones, the item-count note names averaging over runs, pass@k or partial credit. Examples: SimpleBench (mean of 5 runs), ARC-AGI (fractional credit), AISI CTF (mean of 5 or 10 runs). An average of R runs sits on the finer grid j / (R·N). So the check confirms N only where scores come from one run. Elsewhere it neither confirms nor refutes N.

**Averaged scores in the likelihood.** An average is entered as one run of N with a fractional k. Its coin-flip variance is therefore taken as p(1−p)/N. The true value is lower: between p(1−p)/(R·N) and p(1−p)/N. The error is conservative: such a benchmark gets slightly less weight than it deserves, never more.
