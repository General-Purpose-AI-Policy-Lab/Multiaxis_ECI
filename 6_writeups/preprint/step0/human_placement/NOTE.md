# Step 0.3: human placement fixes

Date 2026-11-03. Code at origin/main 4926945 plus the uncommitted change below. Data generation 20261019.

## Result

**(a) Confirmed and fixed (Yassine's option 1, "at least this score").** A human score near a benchmark's fitted ceiling now counts as a lower bound. Such a tier is no longer drawn as a point.
- On the 50-chain flagship's axis 4, Top Performer was placed at 6.4 [4.0, 9.5] (80% interval) from one score, SimpleBench 0.954. The fitted ceiling there is 0.970, and the noise SD at the ceiling is 0.017. Near the ceiling the curve is flat: abilities 4 and 9 both predict about 0.97, so the score cannot tell how high the tier is.
- Domain Expert has no axis-4 score of its own. It was drawn because it sits between two scored tiers in the order, and it inherited 4.7 [2.8, 7.1].

**(c) Fixed by the same rule.** The fixed 0.99 cutoff now applies only to ceilings at 1. Elsewhere a score is a ceiling score when it is within 2 noise SDs of the fitted ceiling.

**(b) Nothing to fix on the current flagship, but a rule must be chosen.** All 42 human-scored benchmark-axis pairs are judged (22 / 4 / 7 / 9 per axis). The 19 unjudged pairs in the red-team report came from the checkout before the 2026-11-02 pull, which had other axis names. The rule (keep, drop or stop on an unjudged pair) still matters for R1 and R2: a fit with new axis titles starts with every pair unjudged.

## The change

- **Which scores count.** A score is a ceiling score when it is ≥ 0.99, or when d_b − score < 2 × SD. Here d_b is the fitted ceiling, and SD is the fit's own noise at μ = d_b:
  - Beta-binomial (N items): SD² = d(1−d)(N+φ) / (N(1+φ));
  - Beta: SD² = d(1−d) / (1+φ).
- **Likelihood.** A ceiling score enters the placement as log P(score ≥ observed), not as the point density:
  - item-scored benchmarks: the Beta-binomial upper tail;
  - continuous benchmarks: the Beta survival function.

  This probability only rises with ability, so the score bounds the tier from below and says nothing else.
- **Display.**
  - A ceiling score does not make its tier drawn. A tier is still drawn when it sits between two tiers that each have a point score, because those two bound it on both sides.
  - An undrawn tier whose only scores on the axis are ceiling scores is written to `human_placement.json` under `lower_bounds`, with its 10% quantile.
  - Each `consistency` row has a new field, `at_ceiling`.
- **Unchanged.** A score of exactly 1.0 behaved this way already: at k = N, P(score ≥ observed) equals the point likelihood. No refit is needed.

## Before and after, on mode A of the 5-chain refit

The 50-chain flagship trace is not on this machine. The check therefore runs on mode A of the 5-chain refit, with the same data and spec. Mode A's own axis names borrow the flagship's judgments of the same content.

- 4 / 35 scores used are newly read as ceiling scores:
  - axis 1: High School Top Performer, MMLU Biology 0.898;
  - axis 4: Skilled Generalist, GSM8K 0.968; Top Performer, SimpleBench 0.954; Committee of Average Humans, WinoGrande 0.94.
- Axis 4, tiers drawn: 6 / 9 before, 4 / 9 after.
  - Top Performer: before, drawn at 6.41 [4.10, 9.33]; after, not drawn, lower bound 4.32.
  - Domain Expert: before, drawn at 4.67 [2.93, 7.11] (between Skilled Generalist and Top Performer); after, not drawn (no scored descendant left).
- The medians barely move: Top Performer 6.41 → 6.53. The point likelihood was already flat there. What changes is how the number is read: a lower bound, not a position.
- Axis 1: High School Top Performer stays drawn, 2.04 [1.75, 2.43], held by its other scores. Axes 2 and 3: no change.

## Not done

- The figures do not draw lower bounds yet. They draw drawn tiers only, so a bounded tier disappears from the figure. An upward arrow or an open marker is a plotting change, left for later.
- The flagship placement must be re-run on the 50-chain trace (`4_diagnostics/5_place_humans.py`) before it is quoted.
- (b) decided (Yassine, 2026-11-03): keep with a warning, the current rule. No code change. A fit with new axis titles needs its pairs judged, or mapped in `RULE_TITLES`, before its placement is quoted.

## Tests

- New: `tests/test_human_placement.py::test_a_score_within_noise_of_the_fitted_ceiling_bounds_from_below_only`. It checks three things:
  - the at-ceiling flag (0.954 against 0.970 on 213 items counts; 0.80 does not);
  - the drawn rule;
  - that P(score ≥ observed) rises with μ on both the Beta-binomial and the Beta.
- Fast suite (`pytest -m "not slow"`): 508 passed and 3 failed, in 159 s.
  - All 3 failures are `TestLayoutPaths` checks that scan every file in the checkout. They flag untracked material already on disk: `deliverables/` (including a vendored venv), `AGENTS.md`, `6_writeups/preprint/agent_reports/`, and the step-0 scripts that read `deliverables/`.
  - No failure involves the placement code.
  - `tests/test_benchmark_scope.py` (untracked, already on disk) was skipped: it imports `load_retired_benchmarks`, which `data.py` does not define.

## Artifacts

All in `6_writeups/preprint/step0/human_placement/`:
- `check_mode_a.py`: the before/after run.
- `check_mode_a_log.txt`: its output.
- `before/` and `after/`: the two placements.
- `pytest_log.txt`: the test run.

Code changes, not committed:
- `2_model/multiaxis_eci/analysis/human_placement.py`: `at_ceiling`, `_at_least_logp`, `CEILING_SD`, the drawn rule, `lower_bounds`.
- `4_diagnostics/5_place_humans.py`: prints "at least" rows and marks lower-bound tiers with `>`.
- `tests/test_human_placement.py`.
