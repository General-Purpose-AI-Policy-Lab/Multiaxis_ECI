# Digest of Jérémy's variants report (read 2026-11-08)

Source: `5_outputs/data20261019/comparisons/variants_report.html` on
`origin/main` (commits 0013061, bc3b5a9, 8edc4f9, written by Claude for
Jérémy; "today" inside the report is 2026-11-06). When the digest was written
the report was not on this machine: local `main` had 4 commits origin lacked,
origin had 13 commits local lacked, and the files below were read with
`git show origin/main:<path>`. Later the same day origin/main was merged into
local `main` (merge commit e50436b; the one conflict, in
`2_model/multiaxis_eci/analysis/human_placement.py`, keeps both the
secondary-axis selection and the ceiling-score lower bound; 115 tests pass),
so every path cited below is now on disk.

"Report" below means a number the HTML asserts (its text, its embedded JSON
data block, or its eight embedded figures). "Repo" means a tracked file in a
fit folder under `5_outputs/data20261019/`. The integrated LOO logs the
report draws on live in each fit's `diagnostics/` folder, which `.gitignore`
excludes, so the LOO numbers exist only in the report. Fit folders on origin
carry the modes file, `axis_names.json`, `mirt_gof_k4.json`,
`mirt_factor_corr.csv`, `mirt_tau_spectrum.csv` and, where placement ran,
`human_placement.json`. No trace of any variant is on this machine.

## 1. What the report compares

Twelve fits: six loading structures at K = 4, each fitted twice. "Models
only + 2b" fits the AI models alone on 50 chains and places the human tiers
afterwards, axis by axis, with the benchmarks frozen (the "2b" placement).
"Humans in" puts the nine human tiers inside the fit on 20 chains
(`--human-merge`). All twelve use the 2026-10-19 data (5,487 scores, 887
test-takers, 107 benchmarks), the lineage prior with Brownian steps, estimated
ceilings and floors (report header; `r1_runs/NOTES.md` for the data counts).

The six structures (report, "The variants in brief"):

| key | structure | CLI |
|---|---|---|
| flagship | compensatory, free non-negative loadings | `--K 4 --lineage-prior --lineage-bm` |
| free | bifactor: a general column g and 3 specifics, displayed separately | `--loading-prior bifactor` |
| gsum | bifactor, displayed axes are g and g + its own deviation | `--loading-prior bifactor --general-sums` |
| orth | bifactor with g and specifics held uncorrelated over the test-takers | `--loading-prior bifactor --orthogonal` |
| ss25 | soft simple structure: a prior pushes each benchmark toward one axis, concentration fixed at 2.5, higher-order general level | `--theta-general --simple-structure --share-conc 2.5` |
| pin | ss25 with each benchmark's axis fixed a priori from the flagship's geomin rotation, offset 1.5 | `... --axis-pin pins/Q_geomin.csv --pin-offset 1.5` |

Not in the report but present on origin: the no-ceiling fit (20 chains, 2
modes, A 17 / 20; `..._noceiling/` modes file), the K = 5 free and gsum
bifactors (10 chains each, 9 and 8 modes, best mode 1 chain in both; `_k5`
modes files), the simple-structure fit at concentration 5 (50 chains, 16
modes, best mode 2 / 50), the two other pins (`pincategory-1p5`, 10 chains,
10 modes, every mode one chain; `pingeomin-1`, 10 chains, 7 modes) and
semicomp0 (no modes file, axes unnamed). `LESS_COMPENSATION.md` (French,
2026-10-23 to 10-27) covers semicomp0, theta-general, simple structure, the
rotation study and the pins. Its own verdict: semicomp0 removes the
compensation but biases the frontier (GPT-6 astra falls from 2.2 to 0.8 on
axis 1), simple structure costs 60 nats of log-likelihood at concentration
2.5 and reaches its best mode in 2 / 30 chains, and "nothing in the data
fixes the orientation of the axes"; the user stopped that line on
2026-10-27 and kept the flagship.

## 2. Scorecard, models only + 2b (report "Scorecard" and section 1)

"AI prediction" is the integrated LOO of the AI scores against the
flagship's mode A, on the observations reliable for both fits (Pareto
k < 0.7). "Sharpness" is the mean share of each benchmark's dominant axis in
its loading variance (1 = one axis per benchmark).

| fit | Δ LOO vs flagship A (nats, ± 1 SE) | reliable obs | selected mode: chains / 50, modes | sharpness | report verdict |
|---|---|---|---|---|---|
| flagship | 0 (reference) | 3,690 | A: 21 / 50, 5 modes | 0.55 | Keep |
| gsum | −5 ± 7 | 3,057 | D: 12 / 50, 11 modes | 0.78 | Alternative |
| free | −12 ± 11 | 3,414 | D: 6 / 50, 8 modes | 0.67 | Drop ("emptied g", 2 dominant benchmarks) |
| orth | −13 ± 11 | 3,658 | A: 1 / 50, 11 modes | 0.60 | Drop ("thin g", mode of 1 chain) |
| pin | −34 ± 9 | 3,254 | A: 12 / 50, 14 modes, 1 stuck chain | 0.86 | Axis check |
| ss25 | −44 ± 10 | 3,588 | A: 2 / 50, 10 modes | 0.80 | Axis check |

Source: report JSON `fits[*].loo`, `primary_n`, `n_modes`, `share`;
verdict chips in the scorecard script.

Repo check of the mode counts and selections (each fit's
`mirt_modes_trace_*.json` on origin):

- Flagship: 5 modes, A 21 chains, B 5 (Δ log density −9.6), C 19 (−17.3),
  D 4 (−25.8), E 1 (−54.2). No `selected_chains` entry; mode A is the best
  log-density mode and the report's reference. Agrees with the report. The
  local refit of 2026-11-04 (seeds 44-48, 50 chains) gives A 24, B 4, C 18,
  D 4 (`r1_runs/NOTES.md`): same best mode, different split, and the report's
  figures read Jérémy's trace, not the local one.
- gsum: 11 modes. The best log-density mode A has 2 chains. The selected mode
  D has 12 chains and sits 25.5 nats below A in mean chain log density. The
  file's reason: "best integrated LOO of the modes, and its g is not empty,
  unlike the best-logp mode A (user, 2026-11-02)". The report states the
  selection by LOO (gsum card) but not that the best-density mode has an
  empty g.
- free: 8 modes, selected D (6 chains, −41.9 nats below A's 3 chains) for the
  same two reasons. Its g in mode D dominates 2 / 107 benchmarks (GeoBench,
  TriviaQA; report JSON `n_dom`; `axis_names.json` titles it "Geolocation,
  Trivia & Commonsense"). The modes file calls that g "not empty" and the
  report calls it "emptied": the same 2 benchmarks, read two ways.
- orth: 11 modes, A 1 chain; the report's scorecard (1 / 50) agrees. Axes
  unnamed (`axis_names.json` confirmed = false).
- ss25: 10 modes, A 2 chains. Agrees.
- pin: 14 modes, A 12 chains, mode N is chain 35 alone at −26,423.8 nats
  (stuck). Agrees.

Per-mode LOO (report JSON `loo_modes`, section 1 figure): gsum's other
large modes F (10 chains) −9.6 ± 7.3 and J (8 chains) not shown above the
cutoff; free's E and F (13 chains each) −6.2 ± 9.0 and −3.3 ± 7.8, so the
free bifactor's LOO ordering of modes is within one SE of flat. Flagship's
C (19 chains) −9.3 ± 4.2, D −18.8 ± 7.8, E −45.2 ± 7.6.

## 3. Humans inside the fit (report section 2, 3, 4)

Same structure, humans in versus models only, scored on the AI scores only:
every structure predicts the AI scores worse with the tiers inside, by 73 to
120 nats (report JSON `fits[*_h].ai`): flagship −85 ± 10, free −85 ± 12,
gsum −87 ± 12, orth −73 ± 10, pin −76 ± 11, ss25 −120 ± 13. In every
humans-in fit the Domain Expert tier lands 3.8 to 7.0 SD of the models below
today's best model on one axis and 10.6 to 19.8 SD above on another (report
NOTES block: flagship_h −7.0 on cyber, +14.4 elsewhere; gsum_h −3.8 on
factual, +10.6), and several expert crossing dates fall before the first
model in the data (2019-11-05): flagship_h cyber "passed in 2011", gsum_h
factual 2021, free_h 2017. The report calls this the signature of
compensation and drops all six humans-in fits. This matches PLAN section 0's
"joint, humans inside" line and the commit messages of c9d0e88, b582e03,
193c90e ("not a reference fit: the tiers compensate across axes").

## 4. Where the experts land and when AI passes them (report sections 3, 4, 4b)

With 2b, the expert sits −1.8 to +4.3 SD of the models from today's best
model across the flagship, free and gsum axes (report section 3 text). The
crossing dates (report table 4b, median and 80 % HDI; a tier with no score of
its own on an axis is not placed there and only a bound is known):

| axis | Domain Expert | Committee of Experts | Top Performer |
|---|---|---|---|
| flagship: Science & reasoning | passed Sep 2024 | passed Jun 2025 | not placed |
| flagship: Agentic & programming | passed Dec 2025 | passed Mar 2026 | not placed |
| flagship: Cyber | not placed (below the committee, so passed) | passed Mar 2024 | not placed |
| flagship: Commonsense & factual | May 2037 (2029–2046), no score of its own | not placed | Jan 2044 (2031–2056) |
| flagship, all axes | ≈ 2037 | ≥ 2037 | ≥ 2044 |
| gsum: g (maths, science & cyber) | passed Jan 2026 | passed on average Aug 2026 | Jan 2031 (2030–2031) |
| gsum: g + desk research, agentic work & commonsense | Jul 2030 (2028–2032) | not placed | Mar 2032 (2029–2034) |
| gsum: g + agentic & cyber | passed Feb 2026 | passed Mar 2026 | not placed |
| gsum: g + commonsense & factual | Sep 2028 (2026–2031), no score of its own | not placed | Feb 2030 (2027–2033) |
| gsum, all axes | ≈ mid-2030 | ≥ mid-2030 | ≥ 2032 |

The report's own caveat: committee-of-experts scores exist on 3 benchmarks
and Top Performer scores on 4, so those tiers are placed on 1 to 3 axes
only, and the commonsense axis is the bottleneck on both fits (slowest
slope, saturated benchmarks, expert without a score of its own).

Repo check (`human_placement.json` on origin, both fits re-placed on
2026-11-06 with the secondary-axis rule of commit 7dcf624): the flagship
draws the expert on axes 1, 2 and 4 and only the committee on axis 3 (one
Cybench score); the per-axis placement uses 31 / 6 / 1 / 9 human scores. gsum
draws tiers on all four axes with 51 / 4 / 5 / 6 scores; on g the Average
Human, Skilled Generalist and Domain Expert medians are 1.54, 1.54 and 1.67
(flagship axis 1: −1.84, −0.23, 1.03), so g barely separates the adult tiers
while the flagship's science axis spreads them over 2.9 units. Placement
convergence is r-hat 1.00-1.01, 0 divergences on every axis of both fits.
The report's figures 5 (eight forecast panels, embedded as images) show the
flagship's mode A at "21 of 50 chains" and gsum's mode D at "12 of 50".

## 5. Do the axes recur across structures? (report section 6)

Share of a flagship axis's dominant benchmarks that are dominant on the
matched axis of another fit (report JSON `overlap`): Science recurs at 71 %
(free), 82 % (ss25), 89 % (pin); Cyber at 84 % (free), 80 % (ss25), 92 %
(pin). Agentic and Commonsense are less stable (44-53 % for ss25 and pin).
For gsum the matched shares are 100 % (g against Science), 3.7 % (g + desk
research against Agentic), 20 % (g + agentic & cyber against Cyber) and 35 %
(g + commonsense against Commonsense): gsum's three deviation axes are not
the flagship's axes 2-4. Pairwise agreement of dominant-axis assignments
over the 107 benchmarks (report JSON `pairs`): flagship-gsum 0.47,
flagship-free 0.60, flagship-ss25 0.67, flagship-pin 0.69, ss25-pin 0.83.

## 6. The report's conclusions ("Verdict", cards, "What is still fragile")

1. Keep the flagship with 2b: best LOO or tied, most reproducible mode
   (21 / 50), four rising nameable axes. Listed cons: blurry axes (0.55),
   committee placed on 3 axes and Top Performer on 1, commonsense expert
   without a score of its own (2037).
2. gsum is the alternative: tied on LOO (−5 ± 7), "g + specifics" reading
   with g dominant on 89 / 107 benchmarks, sharper axes (0.78). Listed cons:
   11 modes with the selected mode (12 / 50) chosen on LOO, committee placed
   on 2 axes and Top Performer on 3, commonsense expert without a score of its
   own (2028).
3. Drop the humans-in fits (compensation, −73 to −120 nats on the AI scores).
4. Keep ss25 and pin as a check on the axes: sharpest (0.80-0.86), the same
   Science and Cyber axes, but −34 to −44 nats and poorly reproducible;
   sharpness partly imposed by the prior.
5. All-axes dates are fragile: expert ≈ 2037 (flagship) against ≈ mid-2030
   (gsum); committee and Top Performer are bounds only; the 2b placement is
   itself an assumption (other axes at the typical-model profile, content
   judgments) but the only tested method without compensation.

Jérémy's message prefers gsum for its g + deviations display. The report he
commissioned ranks the flagship first and gsum as the alternative. The two
are not in conflict on the numbers (LOO tie); they weigh reproducibility
(21 / 50 against 12 / 50) and readability (one general level plus deviations)
differently.

## 7. The bifactor gsum fit

**What the model is.** `--loading-prior bifactor` makes axis 1 a dense
non-negative general column (HalfNormal cells times its own LogNormal scale
`tau_g`, the "normal" prior on one column) and axes 2..K non-negative
specifics under a regularized horseshoe with one global scale `tau_hs`
shared by all specifics (`models/mirt.py` lines 773-794 and 1264-1278;
`config.RH_TAU_SCALE` 0.5, `RH_SLAB_DF` 3, `RH_SLAB_SCALE` 1). No benchmark
is assigned to an axis; a specific cell is either squeezed toward 0 or
escapes at a near-constant prior cost, so shared variance is cheapest in the
exempt general column and a specific the data do not need empties. Axis
identity is structural (g is column 1), so nothing is rotated downstream.

`--general-sums` changes the ability side, not the loadings
(`_assemble_theta`, `mirt.py` lines 214-224 and 247-250). Every ability block
(exchangeable cells, lineage steps, effort offsets, human steps) is built in
(g, z) coordinates and the ability the likelihood reads is

    theta[:, 0] = g,    theta[:, k] = g + sigma_k * z_k,   k = 1..K-1,

with `sigma_k ~ LogNormal(0, 0.5)` per deviation axis (`theta_dev_sd`,
`config.PRIOR_GENERAL_SUMS_SD`; median 1, so two displayed axes correlate at
1 / (1 + sigma²) = 0.5 a priori). A benchmark then reads
eta = a_0 g + Σ_k a_k (g + sigma_k z_k) − D with every a ≥ 0. Written back
in (g, z) this is a bifactor whose general loading is a_0 + Σ a_k, never
below Σ_k B_k / sigma_k where B_k = sigma_k a_k is the loading on z_k: the
bifactor restricted to benchmarks that are positive mixes of the displayed
axes (`build_mirt_model` docstring, lines 869-876). The free bifactor lets a
benchmark rise less with g than its specific does (a contrast against g); gsum
forbids that. The lineage drift is HalfNormal on g and on every z_k, so a
displayed axis climbs along a chain at mu_0 + sigma_k mu_k ≥ mu_0. The option
needs the bifactor loadings and excludes `--theta-general` and
`--orthogonal` (`mirt.py` lines 1082-1090). Tag `_gsum`; the trace attr
`mirt_general_sums` and the axis names `["g", "g+specific1", ...]` are
written at fit time (`fit.py` lines 806-809, 818-819).

**How its axes are displayed.** Axis 1 is g. Axes 2-4 are g plus one
deviation each, so a model's position on "g + agentic & cyber" is its
general level plus its agentic-and-cyber surplus or deficit. Jérémy's
"g + the secondary deviations" is this: one general level and three
displayed axes that each add a deviation to it. The deviations z_k are not
displayed on their own. Axis titles (mode D, `axis_names.json`, confirmed,
g+specific1 renamed 2026-11-06): g "General Capabilities (Math, Science &
Cyber)"; g+specific1 "General Capabilities + Desk Research, Agentic Work &
Commonsense"; g+specific2 "General Capabilities + Agentic & Cyber
Capabilities"; g+specific3 "General Capabilities + Commonsense & Factual
Knowledge". The report's embedded figures carry the new name; its JSON data
block still titles g+specific1 "Desk Research, Trivia & Commonsense" (the
pre-rename title). The Observable Plot script's short labels (`AX.gsum`) use
the older wording too.

**What the fitted gsum looks like (repo, mode D unless stated).**

- Loading energy: `mirt_tau_spectrum.csv` gives tau medians 14.1 for g
  against 3.6, 3.4 and 3.2 for the three deviation axes (gap ratio 3.96
  between g and axis 2). g dominates 89 / 107 benchmarks (report JSON
  `n_dom`), mean dominant share 0.82 on those.
- Axis correlations over the models (`mirt_factor_corr.csv`): g with
  g+specific1 0.86, with g+specific2 0.66, with g+specific3 0.70; the
  deviation axes with each other 0.45 to 0.66. The flagship's four axes
  correlate at 0.23 to 0.64 (`mirt_lineageprior_lineagebm/mirt_factor_corr.csv`).
  The higher correlations are built in: each displayed axis contains g.
- Modes: 11 modes on 50 chains; the selected D holds 12 / 50 chains and sits
  25.5 nats below the best log-density mode A (2 chains), whose g is empty.
  Modes F (10 chains, −30.1), J (8, −53.6) and G (6, −31.9) are the other
  large basins (`mirt_modes_trace_mirt_k4_bifactor_lineageprior_lineagebm_gsum.json`).
- Goodness of fit, whole trace (`mirt_gof_k4.json`): Bayesian R² 0.963,
  RMSE 0.045, PIT variance 0.067 (ideal 0.083), 764 / 4,617 PIT observations
  with high Pareto k. For comparison the flagship folder's file says
  R² 0.928, RMSE 0.068, PIT variance 0.075, 394 / 4,987 high k, while
  `comparisons/README.md` (the dashboard card, mode A only) says R² 0.965,
  RMSE 0.043 for the flagship. The two flagship numbers disagree and neither
  is explained on disk; the report does not use the GoF files.
- Human placement (`human_placement.json`): tiers drawn on all four axes;
  51 human scores inform g. Expert dates in section 4 above. On g+specific1
  the Domain Expert sits at 6.9 [5.9, 8.7] against a frontier near 3.3 from
  4 human scores (report figure 5, gsum panel 2), the widest placement of the
  eight panels.

**What in the report supports Jérémy's preference.** The LOO tie with the
flagship (−5 ± 7 nats on 3,057 reliable observations); sharper axes (0.78
against 0.55); g dominant on 89 / 107 benchmarks, which is the one-number
general index the K = 1 ECI also reports, now with three deviations on top;
human tiers placed on all four axes (the flagship places the expert on three
and only the committee on cyber); the Top Performer placed on 3 axes against
the flagship's 1; and an earlier all-axes expert date (≈ mid-2030 against
≈ 2037) that rests on the same data.

**What in the report and the repo weighs against it.** Eleven modes against
five, and the chosen mode is not the best-density one: 12 / 50 chains, chosen
on LOO and on "g is not empty", 25.5 nats below a 2-chain mode whose g is
empty. Three of its four axes overlap the flagship's at 4 %, 20 % and 35 %
(section 5), so the paper's four capability bundles would be replaced by one
general level and three deviations whose content differs from every other
structure's axes. The displayed axes correlate at 0.45 to 0.86 by
construction. The report's own verdict ranks it second. The commonsense
expert still has no score of its own, so its 2028 date rests on the tier
order exactly as the flagship's 2037 does.

## 8. Where the report and the repo disagree or cannot be checked

1. The LOO numbers (every "Δ nats" above) come from `diagnostics/`
   integrated-LOO logs that are gitignored; the only tracked copy is the
   report's JSON block.
2. The report's flagship mode A is 21 / 50 chains (Jérémy's trace); the
   local refit of the same spec and seeds gives 24 / 50 (`r1_runs/NOTES.md`),
   with no chain-for-chain match on seed 44. PLAN section 0 records that mode
   A alone misses the r-hat 1.01 target (eta r-hat 1.048).
3. Free bifactor's g: "emptied" in the report, "not empty" in the modes file;
   both describe 2 dominant benchmarks.
4. Flagship goodness of fit: fit-folder JSON (R² 0.928) against the dashboard
   README (0.965). Unexplained on disk.
5. gsum's g+specific1 title: renamed on 2026-11-06 in `axis_names.json` and
   the figures; the report's JSON and plot labels keep the earlier title.
6. The report's "12 fits" exclude the no-ceiling, K = 5, concentration-5 and
   two other pin fits that sit on origin; PLAN section 0 lists them.
7. Nothing in the report can be reproduced on this machine: no gsum, free,
   orth, ss25 or pin trace is local. The code it relies on (secondary-axis
   placement rule 7dcf624, LOO and PPC fixes 886746b, 969daf6, 9c14739) is on
   local `main` since the merge of 2026-11-08.
