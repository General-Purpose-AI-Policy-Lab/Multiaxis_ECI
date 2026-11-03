# Step 0.1: noise-collapse check on the floored flagship model

Date 2026-11-02. Code at origin/main 4926945. Data generation 20261019.

## Result

**The noise collapse is still there at every optimum. The floor caps it but does not remove it.** At all 60 / 60 optima the benchmark noise population slides down as a block:

- The mean of log σ_b is −5.45 (range −5.71 to −5.26), against −3.43 in the sampled posterior.
- The spread of log σ_b is 3.40 (3.28 to 3.53), against 0.78 sampled.
- 80 / 107 benchmarks (76 to 87) have σ_b < 0.01.
- 1 to 6 benchmarks per optimum sit at the floor 0.005.

This is the signature the 2026-10-12 sampler study reported (log-σ mean about −5, spread 1.6 to 2.9). The difference: individual σ_b now stop at 0.005 instead of falling to 10⁻³.

**NUTS does not stay there.** 10 / 10 chains started exactly at the 10 best optima leave the collapse during tuning and sample the same region as the 5-chain fit: μ_s −3.41 to −3.46, τ_s 0.77 to 0.81, no benchmark's median within 20% of the floor (see "Sampling check"). The collapse is a mode of the density, not a trap for the sampler. The 2026-10-12 study had 14 / 30 such chains trapped, on older code and an older benchmark set.

## Question

The 2026-10-12 sampler study found "spikes": parameter settings where many benchmarks' noise σ_b fell to 10⁻³ or below. There the log density sat 3,000 to 5,000 nats above the region the fits sample. A floor σ_b ≥ 0.005 was added afterwards (commit 91fe355). The search was not rerun. This note reruns it.

## Method

- **Model.** The K=4 flagship as `3_fit/fit.py` builds it (tag `_lineageprior_lineagebm`). It uses non-negative loadings, a Brownian lineage prior with effort means, a skew-normal θ block, fixed chance floors, estimated ceilings, the Beta-binomial likelihood, hierarchical noise and σ_b floor 0.005. Models only: 5,487 obs / 887 models / 107 benchmarks.
- **Noise model.** σ_b = 0.005 + exp(μ_s + τ_s z_b), with μ_s ~ N(log 0.05, 0.5), τ_s ~ HalfNormal(0.5) and z_b ~ N(0, 1).
- **Objective.** The log density that NUTS samples (unconstrained coordinates, Jacobian included).
- **Starts.** 60 valid starts, the count of the 2026-10-12 study. Each start is the jittered point a NUTS chain starts from. 13 / 73 jittered points had zero density, so the optimiser could not move from them. They were replaced with fresh draws, and the table marks them `valid = False`.
- **Optimiser.** L-BFGS-B, capped at 100,000 iterations. 58 / 60 starts converged. 2 / 60 hit the cap.

## Numbers

| quantity, over 60 optima | min | median | max | sampled posterior (5-chain trace, same spec, indication) |
|---|---|---|---|---|
| log density | −16,586 | −16,320 | −16,077 | median −22,700 |
| log-likelihood | −11,576 | −11,341 | −11,099 | median −15,250 |
| noise mean μ_s (log scale) | −5.71 | −5.44 | −5.26 | −3.43 |
| noise spread τ_s | 3.28 | 3.41 | 3.53 | 0.78 |
| benchmarks with σ_b < 0.01 | 76 | 80 | 87 | 0 (posterior medians) |
| benchmarks at the floor (σ_b ≤ 0.00505) | 1 | 3 | 6 | 0 (posterior medians) |
| smallest σ_b | 0.005 | 0.005 | 0.005 | 0.0053 (lowest single draw) |

- Higher optima have more benchmarks at the floor: the correlation between log density and the count at the floor is 0.56. The 10 highest optima have 3 to 6 at the floor.
- The same few benchmarks hit the floor: MMLU at 93% of optima, LMCA 72%, Lech Mazur Writing 70%, APEX-Agents 42%, DeepResearch Bench 30%, FrontierCode 20%. 95 / 107 benchmarks never do.
- The log-likelihood gain over the sampled posterior is about 3,900 nats. The 2026-10-12 optima had the same overfit (log-likelihood −11,262 to −12,200).

## Reading

- **The floor bounds the density but does not change the shape.** Without the floor the density grew without limit as σ_b → 0. With it, each benchmark's noise stops at 0.005 (φ_b ≤ 9,999), and the population prior pays only a few nats for sliding down. The pooled noise prior is weak enough that μ_s moves 4.9 prior SD (from −3.0 to −5.45) and τ_s sits at 6.8 times its prior scale (3.4 against 0.5).
- **This does not by itself show that the fits are wrong.** A posterior mode can sit far from the posterior's mass in 5,280 dimensions. The 5-chain sampled posterior on the same spec stays at μ_s −3.43 and τ_s 0.78, and no posterior median is near the floor. What matters for R1 and R2 is whether a chain can enter the collapsed region and stay there. Only the sampling check answers that.

## Sampling check

`nuts_from_optima.py` runs one PyMC NUTS chain from each of the 10 best optima, started exactly there, with 1,800 tuning and 300 kept draws, target acceptance 0.95 and seed 191000. Run 2026-11-02, 3,236 s.

| quantity, per chain over kept draws | range over 10 chains | 5-chain fit, same spec |
|---|---|---|
| log density, median | −22,727 to −22,665 | about −22,700 |
| log-likelihood, median | −15,278 to −15,215 | about −15,250 |
| μ_s, median | −3.46 to −3.41 | −3.43 |
| τ_s, median | 0.77 to 0.81 | 0.78 |
| benchmarks with median σ_b within 20% of the floor | 0 (all 10 chains) | 0 |
| smallest σ_b in any draw | 0.0055 to 0.0062 | 0.0053 |
| divergences | 0 to 5 (20 / 3,000 in total) | |

- Every chain leaves the optimum during tuning. Its first kept draw already sits at log density −22,949 to −22,553, about 6,500 nats below its start.
- Per benchmark, the pooled median σ_b of the 10 chains is 0.87 to 1.21 times the 5-chain median (median ratio 1.00). Benchmarks cut 20 to 40 times at the optima (GBAEval, Surface Evolver Bench, SpireClimb) are back at their sampled noise.
- The lowest single draws are CyberGym 0.0055, LAB-Bench LitQA2 0.0058 and GDP.pdf 0.0058: brief visits near the floor, as in the 5-chain fit.
- Mixing: 300 draws per chain is short. PyMC reports r-hat > 1.01 and ESS < 100 per chain on some parameters, and chain 7 reached maximum tree depth. These are the raw parameters that the identified-quantity convergence check ignores. The check answers where chains go, not how well they mix.
- A first run the same day (2,820 s, 20 divergences) sampled but lost its draws to a readout crash. A 3-draw dry run showed two chains not moving from the optimum in 2 tuning steps. With full tuning they move.

## Reading of the sampling check

- The collapse is a narrow, high peak that holds little posterior mass. NUTS adapts its step size during tuning, steps off the peak and does not return within 300 draws.
- On the current code and data, the noise collapse does not trap chains in this check (0 / 10, against 14 / 30 in the 2026-10-12 study). Both the code (the floor) and the benchmark set changed since that study, so this check cannot say which change removed the trap. This is 10 chains of 300 draws: it shows chains do not get stuck, not that they never visit.
- Optimisation-based summaries (MAP, Laplace, any fit started by an optimiser) would land in the collapse. Sampling-based fits do not, on this evidence.

## Correction log

A first version of this note (same day) reported "no collapse: 0 / 60 optima at the floor, smallest σ_b 0.007". That was a readout bug. The noise was compiled from the random variables instead of their values, so it returned fresh prior draws, not the optimum. The giveaway: those "optimum" noise means had mean −2.97 and SD 0.48, which is the prior N(−3.0, 0.5). The log densities and log-likelihoods were correct. All σ_b columns were recomputed from the saved optima, and the script is fixed.

## Artifacts

All in `6_writeups/preprint/step0/map_search/`:
- `map_starts_all.csv`: one row per start (73 rows, 60 valid), sorted by log density.
- `map_sigma_b_all.csv`: σ_b of every benchmark at every optimum (rows `seed, start`).
- `map_starts_seed{1..8}_optima.npz`: the optima in NUTS coordinates.
- `map_search.py`: the search. Seeds 1 to 6 ran 10 starts each, before the redraw of zero-density starts existed. Seeds 7 and 8 ran 7 and 6 starts with the redraw.
- `map_sigma_b_by_benchmark.csv`: per benchmark, σ_b median over the optima, the 5-chain sampled median, their ratio, the share of optima at the floor and below 0.01, and the observation count.
- `nuts_from_optima.py`: the sampling check. `nuts_from_optima.csv` (one row per chain), `nuts_from_optima_trace.nc` (the draws), `nuts_log.txt` (the log).
