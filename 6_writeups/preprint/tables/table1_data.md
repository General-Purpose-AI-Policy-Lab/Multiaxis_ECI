# Table 1. Data of the multi-axis model, by source feed

Observations attributed to a feed: 5487 / 5487 (join on model and benchmark).
Fit scope: 5487 observations, 887 models, 107 benchmarks; processed table: 5588 rows.

| feed | observations | models | benchmarks | with reported SE |
|---|---|---|---|---|
| Epoch AI benchmark hub (ZIP) | 2959 | 762 | 47 | 96 |
| Epoch AI internal evaluations (live feed) | 1146 | 374 | 13 | 1111 |
| RAND RR-A3797-1 | 565 | 40 | 15 | 0 |
| Scale SEAL leaderboards | 336 | 185 | 13 | 0 |
| Epoch AI cyber ECI | 205 | 74 | 21 | 0 |
| Kaggle Open Benchmarks | 169 | 76 | 3 | 169 |
| GPAI Policy Lab local runs | 105 | 16 | 12 | 105 |
| Cybench leaderboard | 2 | 2 | 1 | 0 |
| Total | 5487 | 887 | 107 | 1481 |

Models and benchmarks count those with at least one score from the feed; the feed rows do not sum to the total for these two columns. Reported SE: a positive `stderr` in the processed table.

Feed grouping: the pipeline's own `feed` column of the processed table.

- Cybench leaderboard: 2 source strings, e.g. Cybench leaderboard (cybench.github.io, snapshot 2026-09-07; 35/40 tasks); Cybench leaderboard (cybench.github.io, snapshot 2026-09-07; all 40 tasks)
- Epoch AI benchmark hub (ZIP): 104 source strings, e.g. EpochAI; https://conceptualreasoning.ai/dtbench; https://conceptualreasoning.ai/lmca
- Epoch AI cyber ECI: 1 source strings, e.g. Epoch AI cyber ECI (values partly read off published plots)
- Epoch AI internal evaluations (live feed): 1 source strings, e.g. Epoch AI internal evaluation
- GPAI Policy Lab local runs: 105 source strings, e.g. GPAI Policy Lab run 20261014_claude-haiku-4-5-20251001_none_gpqa-biology (4 epochs x 19 questions); GPAI Policy Lab run 20261015_claude-haiku-5-5_low_rand-family (4 epochs x 19 questions); GPAI Policy Lab run 20261015_claude-haiku-5-5_max_rand-family-no-seqqa (4 epochs x 19 questions)
- Kaggle Open Benchmarks: 3 source strings, e.g. Kaggle Open Benchmarks (open-benchmarks/mmlu-pro); Kaggle Open Benchmarks (deepmind/simpleqa-verified); Kaggle Open Benchmarks (open-benchmarks/mmlu)
- RAND RR-A3797-1: 1 source strings, e.g. RAND RR-A3797-1 (Dev et al. 2025)
- Scale SEAL leaderboards: 14 source strings, e.g. Scale SEAL (visual_language_understanding); Scale SEAL (multinrc); Scale SEAL (tool_use_enterprise)

## Human baselines (0_input/human_baselines.csv)

56 scores, 9 tiers, 42 benchmarks; 56 scores on a benchmark of the multi-axis fit. The 44 scores that place the tiers (README) need the fitted loadings and are not recomputed here.

| tier | scores | on_fit_benchmarks | benchmarks |
|---|---|---|---|
| Domain Expert | 18 | 18 | 18 |
| Skilled Generalist | 11 | 11 | 11 |
| Average Human | 9 | 9 | 9 |
| Committee of Average Humans | 5 | 5 | 5 |
| High School Top Performer | 4 | 4 | 4 |
| Top Performer | 4 | 4 | 4 |
| Committee of Domain Experts | 3 | 3 | 3 |
| Committee of Skilled Generalists | 1 | 1 | 1 |
| High School Qualifier | 1 | 1 | 1 |
| Total | 56 | 56 | 42 |

## Single-axis model scope (5_outputs/data20261019/canonical_humanmerge)

- With human rows: 4410 observations, 93 benchmarks (gof.json n_obs; difficulties in summary.csv). Without: 4370 observations.
- Human rows: 40 scores of 7 tiers on the 93 benchmarks, from the fit's source human file (57 rows; 0_input/human_baselines.csv has 56).
- pit.csv: 4324 rows = 4286 AI + 38 human, 93 benchmarks; it omits the 86 scores of exactly 0 or 1 (2 of them human).
