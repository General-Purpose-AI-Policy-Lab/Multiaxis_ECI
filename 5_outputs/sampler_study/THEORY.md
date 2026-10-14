# Why the flagship's posterior is multimodal, and what can help (night of 2026-10-12)

## The observation that settles the diagnosis

On the flagship's 20-chain NUTS trace, the five modes A–E predict the observed scores almost
identically: the posterior-mean logits η of the 5113 observed cells differ between modes by 0.045–0.13
(RMS), against a spread of 1.75 within a mode (correlations ≥ 0.997). Yet their loading subspaces
(98 × 4) and ability subspaces (850 × 4) are far apart: principal angles of 1–4° for the first direction
(the shared general factor), then up to 23–45° and 48–90° for the others. The modes are not rotations
of one solution: they are **different rank-4 completions of a 6 %-observed matrix** (850 models × 98
benchmarks, median 4 scores per model, missingness structured by release date) that fit the observed
cells equally well and differ only on the cells never observed — which is exactly what the per-axis
abilities and the forecasts read.

## Consequences

1. **It is an identification problem before it is a sampling problem.** The likelihood cannot tell the
   modes apart (their log-likelihoods differ by ~10–40 nats on 5113 cells, comparable to a chain's own
   ±60 fluctuation); what separates them is the prior (lineage, loadings) and small differences in fit.
   A perfect sampler would put mass on all of them, roughly in proportion to exp(log posterior) times
   their volume; "the best mode" is the most probable completion, not the true one.
2. **Why no sampler crosses.** Going from one completion to another means moving a block of
   benchmarks from one axis to another together with the abilities of every model scored on them; the
   straight paths in (A, θ) space break the fit on the observed cells (the bilinear model is not
   convex, and the non-negative loadings close the easy rotations). The barrier is high-dimensional
   and collective.
3. **Why tempering does not help.** At low T the likelihood is weak and the prior is isotropic: the
   chains wander. Near T ≈ 0.06–0.1 (the jump in log-likelihood seen in every annealing run) the
   completion crystallises, decided by small fluctuations: a first-order transition. Tempered methods
   (annealing, parallel tempering, SMC) are known to fail across first-order transitions — the
   swap/rewighting rates vanish exponentially in the system size, which is what the round-2 runs show.
4. **What should help.**
   - *For finding modes:* cheap local optimisation from many starts (the full model's MAP takes
     ~20 s with JAX gradients, versus ~1 h for a NUTS chain), then NUTS from the best optima
     ("optimise, then sample"); weight the modes by stacking or by Laplace/bridge estimates.
   - *For the model:* more identification — confirmatory or semi-confirmatory loadings (each
     benchmark mostly on the axis its content names), stronger shrinkage of cross-loadings, or fewer
     axes; or a rotation-invariant (signed) loading prior with a post-hoc rotation, if the modes turn
     out to share a subspace in that parameterisation.
   - *For the conclusions:* report how much the per-axis results (names, tier placements, crossing
     dates) move across the top modes; if little, the mode choice is immaterial.

## Tests run tonight (results in RESULTS.md)

- T1: a minimal JAX version of the model (same likelihood and floors, fixed noise, θ ~ N(0, I), no
  lineage prior) optimised from 100 random starts: does it show the same multiplicity of optima?
- T2: the full model optimised from 60 random starts (L-BFGS, JAX): how many distinct optima, at what
  log posterior, and do they include A and the D/E-like families?
- T3: NUTS (10 chains) started from the 10 best full-model optima, against plain NUTS from jittered
  starts (35–40 % of chains above −14395 at the same budget).
