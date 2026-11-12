# Flagship analysis for the preprint (work item B), 2026-11-12

Fit: K = 4 compensatory, ceilings, lineage prior with Brownian steps, models only. Local
50-chain trace (seeds 44-48), modes A 24 / B 4 / C 18 / D 4. Everything below reuses the
repo's own scripts; the two scripts in this folder only assemble tables.

## 1. Convergence inside each large mode

`convergence_by_mode.py` -> `convergence_by_mode.csv`. Same statistics as
`r1_runs/convergence_identified.csv`.

| mode | chains | eta r-hat max | eta cells over 1.05 | D over 1.05 | sigma_b over 1.05 | eta bulk ESS min / median | divergences (saved draws) |
|---|---|---|---|---|---|---|---|
| A | 24 | 1.268 | 20 / 5,487 | 0 / 107 | 0 / 107 | 65 / 15,943 | 369 / 24,000 |
| C | 18 | 1.052 | 1 / 5,487 | 0 / 107 | 1 / 107 | 241 / 12,788 | 407 / 18,000 |
| all | 50 | 1.448 | 251 / 5,487 | 19 / 107 | 9 / 107 | 97 / 16,251 | 812 / 50,000 |

Reading: inside a mode the predictions (eta), the difficulties and the noise converge on all
but 20 / 5,487 cells (A) and 1 / 5,487 (C). Across modes they do not. Per-axis abilities do not
converge inside a mode either (`r1_runs/aligned_rhat_by_mode.log`): 181 to 361 of 887 over 1.05
in mode A.

## 2. Human tiers placed on mode A

`diagnose_chains.py --write-tables`, then `5_place_humans.py` (both rules: secondary axis,
ceiling lower bound). Log `place_modeA.log`. The fit folder's tables and `human_placement.json`
are now mode A of the local trace. Placement sampler: r-hat 1.00 to 1.01, 0 divergences.

| axis | human scores used | tiers drawn | lower bound only |
|---|---|---|---|
| 1 Scientific Knowledge & Reasoning | 31 | 8 | none |
| 2 Agentic Capabilities & Programming | 6 | 6 | Top Performer |
| 3 Cyber Capabilities | 1 | 1 (Committee of Domain Experts) | none |
| 4 Commonsense & Factual Knowledge | 9 | 4 | Top Performer |

The Domain Expert is drawn on axes 1 and 2 only. It has no score on axes 3 and 4.

## 3. Crossing dates on mode A

`crossings_modeA.py` -> `crossings_modeA.csv` (median, 50 % and 80 % intervals, probability
passed today). The CSV keeps raw dates; the paper prints "passed" for a tier already passed and
no date before 2023.

| axis | result |
|---|---|
| 1 Science | 8 / 8 drawn tiers passed. Domain Expert: median 2024-09. Committee of Domain Experts: 2025-07 (80 %: 2025-04 to 2025-10). High School Top Performer: 2026-03, probability passed 0.89. |
| 2 Agentic | 6 / 6 drawn tiers passed, medians 2025-09 to 2026-02, probability passed at least 0.999. Rests on 6 scores. |
| 3 Cyber | Committee of Domain Experts passed, median 2024-03 (80 %: 2023-04 to 2025-03). Rests on 1 score (Cybench). |
| 4 Commonsense | Not passed. Average Human: probability passed 0.33. Committee of Average Humans: median 2031-01 (80 %: 2024-03 to 2035-03). Skilled Generalist: 2031-03 (80 %: 2027-11 to 2035-05). Committee of Skilled Generalists: 2037-06 (80 %: 2030-09 to 2044-10). |

## 4. Mode C has other axes

Mode C's chains do not match mode A's axes: best match per axis r = 0.89 / 0.76 / 0.92 / 0.81
(`modeC_frame_refused.log`), under the 0.8 cut of the repo's alignment. Mode C is therefore read
on its own frame: `diagnose_chains.py --write-tables --secondary`, tables in the fit folder's
`secondary_mode/`. Its axes are not named yet (`secondary_mode/axis_names.json` is the template).

Top benchmarks per mode C axis, median loading (share of the benchmark's loading on that axis):

| mode C axis | top benchmarks | nearest mode A axis |
|---|---|---|
| 1 | ARC-AGI-2 3.20 (0.76), ARC-AGI 2.41 (0.79), OTIS Mock AIME 2.24 (0.92), FrontierMath Tier 4 v2 1.95 (0.67), MATH Level 5 1.94 (0.87), CritPt 1.63 (0.83) | 1 Science, r 0.89 |
| 2 | GSM8K 1.76 (0.86), Fiction.LiveBench 1.58 (0.58), SimpleQA Verified 1.55 (0.85), WMDP Biology 1.33 (0.61), ARC AI2 1.30 (0.57), MMLU 1.22 (0.67) | 4 Commonsense, r 0.81 |
| 3 | CyberGym 1.76 (0.84), Furniture Assembly 1.75 (0.78), Cybench 1.63 (0.65), CyScenarioBench 1.52 (0.85), AISI CTF Practitioner 1.51 (0.62), GBAEval 1.40 (0.74) | 3 Cyber, r 0.92 |
| 4 | DeepSWE 1.46 (0.41), GSO-Bench 1.36 (0.46), Aider Polyglot 1.24 (0.33), OSWorld 1.22 (0.62), MirrorCode 1.18 (0.54), The Agent Company 1.07 (0.41) | 2 Agentic, r 0.76 |

Dominant axis of each of the 107 benchmarks, mode A (rows) against mode C (columns):

| | C1 | C2 | C3 | C4 |
|---|---|---|---|---|
| A1 Science (38) | 22 | 15 | 1 | 0 |
| A2 Agentic (29) | 7 | 0 | 4 | 18 |
| A3 Cyber (23) | 1 | 1 | 19 | 2 |
| A4 Commonsense (17) | 0 | 13 | 1 | 3 |

Reading: the cyber axis is the same in both modes (19 / 23). Mode A's science axis splits in
mode C: 22 / 38 benchmarks stay on a formal-reasoning axis (ARC-AGI, maths), 15 / 38 (the
knowledge-recall ones: MMLU, WMDP) join commonsense and factual QA on axis 2. Mode A's agentic
axis keeps 18 / 29 on mode C's axis 4 and gives 7 to the reasoning axis. Mean dominant share is
the same, 0.56 (A) and 0.55 (C). No tier is placed on mode C: the placement needs its four axis
names and the content judgments that go with them.

What holds in both modes is already on disk: the benchmark bundles (59 / 107 benchmarks in cores
co-assigned in at least 90 % of the 50 chains, `r1_runs/bundle_cores.txt`) and the fitted scores
(modes differ by 0.13 to 0.21 posterior SD on eta, `r1_runs/NOTES.md`).

## 5. Not done, by choice

- Tier placement and crossings on mode C. They need four axis names from Yassine.
- Pseudo-human and leave-one-benchmark-out validation of the placement. Limitation sentence.
- Stacking weights over modes. Limitation sentence.
