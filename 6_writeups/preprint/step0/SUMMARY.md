# Step 0: what changes before R1 and R2

Date 2026-11-03. Data generation 20261019. Code at origin/main 4926945 plus one uncommitted change, listed under item 3.

1. **Noise collapse: no change to the model.** The collapse is still a mode of the density: 60 / 60 MAP optima have 76 to 87 / 107 benchmarks with σ_b < 0.01, and 1 to 6 at the 0.005 floor. But 10 / 10 NUTS chains started at the 10 best optima leave it during tuning and sample where the 5-chain fit samples (μ_s −3.41 to −3.46 against −3.43). Any summary based on an optimiser (MAP, Laplace) would land in the collapse. See `map_search/NOTE.md`.
2. **Nested benchmarks: no data change (Yassine, 2026-11-03).** Shared questions alone show no excess residual correlation (−0.057 over 30 pairs, the same as the baseline). The 15 RAND rows share a same-source excess of about +0.10, which K=4 absorbs: in mode A, 6 of the 10 benchmarks axis 1 owns most are RAND rows. FrontierMath T1-3 / T4 and AISI CTF Practitioner / Expert stand out, but neither builds an axis. To do: report these in the limitations, and add a sensitivity fit without the 15 RAND benchmarks to the run list (its config needs a go). See `nested_audit/NOTE.md`.
3. **Human placement: one code change, applied, not committed.**
   - A human score within 2 noise SDs of the fitted ceiling (or ≥ 0.99) now enters as "at least this score" and draws no tier. On axis 4, Top Performer becomes a lower bound (it was a point at 6.4 [4.0, 9.5]), and Domain Expert is no longer drawn.
   - Still open:
     - unjudged benchmark-axis pairs: kept with a warning, the current rule (Yassine, 2026-11-03). Judge, or map in `RULE_TITLES`, the new axis titles of R1 and R2 before quoting their placements;
     - re-run `5_place_humans.py` on the 50-chain trace;
     - optionally, draw lower bounds on the figures.

   See `human_placement/NOTE.md`.
