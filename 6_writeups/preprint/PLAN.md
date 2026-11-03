# Preprint plan (arXiv)

Working title (from the blog post, to confirm): *A Multi-Axis Bayesian Epoch
Capabilities Index with Human Baselines*. Authors: Yassine Essifi, Jérémy
Andréoletti (GPAI Policy Lab). Drafted 2026-10-29 from five agent surveys in
`agent_reports/` (results, methods, writing assets, related work, red team).
Nothing had been run or changed in the repo on that date; section 0 records the state on 2026-11-03.

## 0. State on 2026-11-03 (after the pull to 2789fef and step 0)

Sections 1, 4, 9 and 10 below were written on 2026-10-29 against a checkout
34 commits behind. This section supersedes them where they disagree.

**What origin/main now holds (fitted by Jérémy, tables only on this machine).**

| fit | folder under `5_outputs/data20261019/` | chains | modes | status for the paper |
|---|---|---|---|---|
| ceiling flagship, models only | `mirt_lineageprior_lineagebm/` | 50 (seeds 44-48) | 5: A 21, B 5 (−10 nats), C 19 (−17), D 4 (−26), E 1 (−54) | R1 in spec. Mode A: 8,400 draws, min / median ESS 276 / 7,699, 110 divergences; eta r-hat 1.048 on the whole fit (mode-inflated). Fixed integrated LOO favours A on 3,116 reliable scores (C −9.0 ± 4.0, D −14.4 ± 6.9). |
| no ceilings | `..._noceiling/` | 20 (seeds 42, 43) | 2: A 17, B 3 (−8.7) | R2 in spec, but 20 chains against 50 and other seeds. Robust LOO: ceilings +55.8 ± 12.1 nats on 4,967 / 5,487 scores. |
| joint, humans inside | `..._noceiling_humanmerge/` | 10 | 1 | the unimodal negative result (tiers trade axes) |
| bifactor free / gsum / k5 / orthogonal, axis-pin, semicomp0, simple-structure 2.5 / 5 | own folders, `LESS_COMPENSATION.md` | 10-50 | 4-16 | the "less compensation" study, closed: every variant fragments (9-16 modes) or biases the frontier; feeds §4 ladder and limitations |

**No K=4 trace is on this machine.** Every trace-level item (mode-aware ESS,
placement on draws, pseudo-human validation, per-mode crossings, re-running
the placement with the step-0 ceiling fix) is blocked on either Jérémy's
50-chain trace or a local refit. Traces are now float32, one draw in two
(commits 936e7fb, bc77fdf), so a retrieved trace is about half the old size.

**Step 0 results (`step0/SUMMARY.md`).**

1. Noise collapse: present at 60 / 60 MAP optima (80 / 107 benchmarks with
   σ_b < 0.01) but 10 / 10 NUTS chains started there leave it. No model
   change. Report in the methods: optimiser-based summaries are invalid here.
2. Nested benchmarks: shared questions add no residual correlation (−0.057,
   the baseline). The 15 RAND rows share +0.10, absorbed by K=4 (6 of mode
   A's axis-1 top 10 are RAND). FrontierMath T1-3 / T4 and AISI CTF
   Practitioner / Expert at the 96th-99th percentile. Decision: keep all,
   report in limitations, add **R8** (flagship spec without the 15 RAND
   benchmarks).
3. Human placement: a score within 2 noise SD of the fitted ceiling enters as
   "at least this score" and draws no tier. Code applied, tests pass (508 /
   511, 3 layout failures unrelated), **not committed**. Unjudged pairs stay
   "keep with a warning". Must be re-run on the 50-chain trace.

**Revised run list.** R1 and R2 exist in spec upstream. What remains is a
decision, not a fit: take the 50-chain ceiling fit as the flagship (robust LOO
+56 nats, A leads the integrated LOO) with the 20-chain no-ceiling fit as the
sensitivity, or refit the no-ceiling spec at 50 chains on seeds 44-48 for a
like-for-like mode count. R3 (bare K=4), R4 (K=3, K=5 under the flagship
prior), R5 (backtest), R6 (symmetric drift), R7 (recovery) and R8 (no RAND)
are still to run. None launches without a stated config and a go.

**Decision 7, taken 2026-11-03: refit R1 locally, later, in runs small
enough to leave the laptop usable.** No launch until Yassine says go. The
recipe, to be confirmed at launch:

- Machine: 15 logical cores, 24 GB. The 50-chain flagship was 5 runs of 10
  chains (seeds 44-48) merged with `--merge-chains`. Here: **10 runs of 5
  chains**, seeds 44-53, so 10 cores stay free. The 5-chain refit of
  2026-10-21 ran on this machine, so 5 chains are known to fit in memory.
- Per run: the flagship spec (`--K 4 --lineage-prior --lineage-bm`, ceilings,
  floors, skew-normal theta and models-only are the defaults), 2,000 tune and
  2,000 draws (`config.SAMPLE_KW` tune; draws as the 50-chain fit: 400 kept
  per chain at the old thin 5), saved at `config.SAVE_THIN` 2.
- Run 1 writes `trace_mirt_k4_lineageprior_lineagebm.nc`. Before every later
  run, copy that trace out of the fit folder (the run overwrites it), then
  pass `--merge-chains <copy> --seed <next>`; `fit.py` refuses a merge of a
  different spec or a reused seed. Every table is rebuilt on the union after
  each run, so modes can be read after 10, 25 or 50 chains.
- Order: the ceiling fit first. The no-ceiling fit (R2) follows with the same
  recipe only if decision 2 asks for a like-for-like mode count.
- Each run should be timed; at the simple-structure runs' logged 2.4 s per
  iteration at 10 chains, a 5-chain run of 4,000 iterations is a few hours.

**Next steps, in order.**

1. Commit the step-0 placement fix and the `6_writeups/preprint/` folder
   (Yassine's call; the folder is untracked).
2. R1 as the split local refit above, when Yassine gives the go.
3. Decision 2 on the evidence above.
4. Refresh `agent_reports/results_inventory.md` and `red_team.md` (in
   progress, 2026-11-03).
5. With the trace: re-run `5_place_humans.py`; mode-aware convergence per
   mode (A, C at least, 19-21 chains each); placement validation; per-mode
   crossings; PIT by benchmark / year / source.
6. Documentation fixes (section 7); then R3-R8 one at a time.

## 1. Where things stand

**Data generation.** The paper is on the 2026-10-19 pipeline run: 5,487 obs /
887 models / 107 benchmarks for K=4; 4,410 obs / 93 benchmarks for the K=1
index. Every number from the blog post (4,923 / 829 / 96, humans inside) is
superseded.

**Method today.** K=4 compensatory Beta-MIRT, nonnegative loadings, skew-normal
abilities, fixed chance floors, estimated ceilings, Beta-binomial on
item-scored benchmarks, hierarchical noise with floor 0.005, Brownian lineage
prior with tree chains and effort means, fitted on models only. Nine human
tiers placed on the axes afterward. Forecast: top-2 frontier points, reasoning
and non-reasoning lines, projection to 2030. Frontier gaps (open/closed,
US/China) on the K=1 ECI-H. Full equations with file:line in
`agent_reports/methods_inventory.md`.

**What is citable now.**

| Result | Where | Status |
|---|---|---|
| K=1 ECI-H, humans inside, converged (r-hat max 1.006, min ESS 1,270, 7/20,000 div) | `5_outputs/data20261019/canonical_humanmerge/` (trace in `deliverables/canonical_sciencespo_20261022/`) | citable |
| ECI-H human crossings: Domain Expert 2025-08, Committee 2025-11, Top Performer 2029-03 (extrapolated) | `deliverables/canonical_sciencespo_20261022/figures/eci_human_crossings_18mo.csv` | citable with caveats |
| Open behind closed 6.7 months (80%: 6.0 to 7.3); China behind US 6.5 | `5_outputs/open_closed_frontier/`, `us_cn_frontier/` | citable |
| Ceilings improve robust LOO by +56 ± 12 nats on 4,967/5,487 scores; 8/85 ceilings data-informed | `mirt_lineageprior_lineagebm/investigation/README.md` | citable, one seed |
| Joint fit with humans inside: tiers trade axes (−6.4 model-SD on science) | `mirt_lineageprior_lineagebm_noceiling_humanmerge/` | citable as a negative result |
| K=4 fitted scores: RMSE 0.043 vs 0.068 for K=1; chains agree on fitted cells to 0.15 SD median | ceiling fit + 5-chain refit | citable |

**What is NOT citable.** Any per-axis K=4 quantity: axis names, abilities,
human placements, crossing dates. Reasons:

- The 10-chain ceiling flagship's trace, r-hat, ESS and crossover tables are
  not on this machine (gitignored, fitted by Jérémy). Only GoF and the modes
  file exist here.
- The independent 5-chain refit of the same spec failed: 1,612/60,000
  divergences, min bulk ESS 30 on eta, r-hat 1.10, two loading families
  (DeepResearch and MMLU Chemistry move between axes by 1.6 to 1.8 posterior
  SD).
- The no-ceiling fit the README reports is itself at eta r-hat 1.165 on its 17
  selected chains.
- Three rules rank the modes three ways on the ceiling fit: log-density favours
  A by 23.8 nats, PSIS-LOO favours B by 22 ± 9, integrated LOO is a tie
  (+3 ± 8).
- Axis sets differ by fit: the no-ceiling fit has a science axis, the ceiling
  fit does not (Fluid & Formal Sciences; Agentic & Programming; Cyber; Visual
  & Commonsense, provisional).
- Human placement misfits its own scores: 22/41 human scores outside the 80%
  band on the ceiling fit (band omits observation noise). 19 benchmark-axis
  pairs have no content judgment and are kept by default. A tier's score at a
  fitted ceiling (LMCA) reads as near-unlimited ability.

**Convergence diagnosis** (sampler study, `5_outputs/sampler_study/`). The
modes are different completions of a score matrix that is 6% observed. They
predict observed cells almost identically and differ on unobserved cells. No
sampler tried crosses between them. The fit with humans inside is unimodal
(1/20,000 divergences), which shows ability-side information removes the
multimodality.

## 2. The thesis (decision 1)

The paper's headline must be one that the evidence supports. Three framings:

**A. "Four capability bundles, not uniquely allocated."** Methods paper.
The data identify a recurring four-bundle structure but not a unique
benchmark-to-axis allocation; the modes differ only on untested pairs; the
paper lists the measurements that would separate them (`data_wishlist.md`).
Per-axis results and crossings reported per mode as conditional results. This
is the framing the current evidence supports without new fits. It treats the
multimodality as a finding.

**B. "A converged four-axis index with human crossings."** Results paper, the
blog post's framing. Needs a frozen final run that converges by the paper's
own standard, a resolved human placement, and a K ladder. Not available
today; needs the refits in section 4.

**C. Two papers.** Paper 1: the K=1 ECI-H plus frontier gaps (everything
citable now, converged). Paper 2: the multi-axis model, when B's conditions
are met. Fastest route to a first arXiv submission.

A and B are not exclusive: A can be written now and upgraded toward B as runs
land.

## 3. Proposed structure (framing A, methods first)

1. **Introduction.** One index cannot order humans and models both (ARC-AGI
   vs GPQA). Bayesian: every ability carries its uncertainty. Contributions
   list, each with its number.
2. **Data.** Five feeds, scope table, provenance table (26% of rows
   independently re-run; 771 rows digitised from plots), dedup policy, effort
   variants as test-takers, exclusions, floors, known ceilings, human tiers
   (57 scores).
3. **Model.** 3.1 One-axis Beta-MIRT and ECI-H anchors. 3.2 K-axis extension,
   loadings, identification (positivity, skew, label matching). 3.3
   Likelihood: floors, estimated ceilings, Beta-binomial, noise floor. 3.4
   Ability priors: skew-normal block, Brownian lineage over release trees,
   effort means. 3.5 Human tiers: inside at K=1, placed at K=4, and why
   (the joint fit's trade-off). 3.6 Computation: nutpie, chains, mode
   detection, identified convergence.
4. **Model evidence.** LOO ladder on ONE scope (K=1 → bare K=4 → +lineage →
   +ceilings; humans inside vs placed), Pareto-k beside every elpd.
   Calibration (LOO-PIT), recovery study, prior predictive.
5. **Results, mode-invariant.** ECI-H timeline vs Epoch's ECI; fitted-score
   agreement; the four bundles with their stable cores; what moves between
   modes and why (the 6% fill).
6. **Results, per mode.** Axis loadings, abilities, human placements, frontier
   lines and crossing probabilities, side by side for the modes. "Passed,
   p = X" for tiers already passed; no date before the data.
7. **Frontier gaps** on ECI-H: open/closed, US/China, trailing 12-month lag as
   the headline, private vs public as not separated from composition.
8. **Limitations.** Coverage (old models never took hard benchmarks),
   placement conditional on a model-like profile, extrapolation to 2030 on two
   years of points, vendor self-reports, choices made after seeing results
   (skew-normal, exclusions, top-2), ceilings mostly at prior.
9. **Appendices.** Full priors table, failures (signed loadings, horseshoe,
   non-compensatory, K=3), convergence per fit, human baseline sources
   (blog Annex F, 52 rows), data wishlist, benchmark table.

## 4. Analyses still needed

Every fit needs Yassine's go, with config and data version stated first.

**Refits (hours each; one machine).**

| # | Run | Supports | Decision it informs |
|---|---|---|---|
| R1 | Frozen final K=4 ceiling fit, 20 to 30 chains × 2,000 tune / 2,000 draws, two seeds, full diagnostics saved in the repo | §3.6, §5, §6 | the paper's flagship; replaces the trace not on this machine |
| R2 | Same without ceilings, same chains, same seeds | §4 ladder, sensitivity | ceilings on/off |
| R3 | Bare K=4 (no lineage) on the same scope | §4 ladder | lineage prior's contribution |
| R4 | K=3 and K=5 on the same scope, same chains | §4, referee objection 5 | why four axes |
| R5 | Forecast backtest: refit with release-date cutoffs (e.g. 2025-06, 2025-12), score the next 12 months | §6, §8 | whether a 2030 projection is defensible |
| R6 | Symmetric lineage drift sensitivity (N(0,·) instead of HalfNormal) | §8 | winner's-curse objection |
| R7 | Recovery study on the current spec (simulate from the fit, refit) | §4 | identification claim |
| R8 | Flagship spec without the 15 RAND benchmarks (added 2026-11-03, nested audit) | §8 limitations | whether axis 1 depends on the RAND campaign |

**Hours, no refit (on a trace).**

- Mode-aware convergence: r-hat and ESS within each mode, 3+ chains per mode;
  stacking weights from held-out-cell K-fold CV as a sensitivity. Never chain
  counts as mode weights.
- Human placement validation: place held-out models as pseudo-humans
  (truth known); leave-one-benchmark-out prediction of human scores;
  placement on 100 to 200 posterior draws instead of medians; a predictive
  band with observation noise; fix the ceiling artifact; judge the 19
  unjudged pairs.
- Per-mode and per-draw crossing dates; specification curve for the forecast
  (top-1/2/3, regime break date, effort rule).
- Specification curve for the frontier gap (top-k, record rule, window).
- ECI-H vs Epoch's published ECI on the current data (needs a name map;
  `deliverables/sciencespo_plan_20261021/epoch_eci.csv`). The 08-17 memo had
  Spearman 0.979 on old data.
- PIT grouped by benchmark, year and source.

**Minutes.**

- Rerun the multi-start MAP search on the floored model: DONE 2026-11-03
  (`step0/map_search/NOTE.md`).
- Prior predictive check.
- Residual correlations and PPCA on the canonical trace.
- Provenance table of the 5,588 rows by source and re-run status.
- Nested benchmark series audit: DONE 2026-11-03
  (`step0/nested_audit/NOTE.md`); R8 added.

## 5. Figures and tables (first list)

1. ECI-H timeline with human tiers (builder exists:
   `6_writeups/blogpost/figures/make_timeline`).
2. ECI-H vs Epoch ECI scatter (new).
3. Top-20 loadings per axis, per mode (`docs/make_readme_figures.py`).
4. Mode comparison: matched loading profiles and benchmark allocations
   (refit's `plot_all_five_axes.py`, to port).
5. Fitted-score agreement across chains (new, one panel).
6. LOO ladder table with Pareto-k column (`ladder_notes.md` prose reusable).
7. LOO-PIT histogram (`make_pit` reusable).
8. Per-axis frontier forecast with placed tiers, per mode
   (`4_diagnostics/3_plot_mirt.py --forecast`).
9. Crossing table: tier × axis × mode, p-passed or date with 50/80% intervals.
10. Frontier gap panels and lag table (`2_plot_frontier_gap.py`).
11. Human placement validation (pseudo-humans, LOBO) (new).
12. Prior graph (`docs/make_prior_graph.py`, needs update to the models-only
    model).
13. Appendix: Neal's funnel (`memo/make_divergence_fig.py`), recovery study.

Rules: self-explanatory from title and legend; English; colorblind palette;
"frontier points" not "records"; no date before 2023 on any crossing.

## 6. Writing setup and tools (decision 2)

- Folder `6_writeups/preprint/`: `main.tex`, `sections/`, `figures/`,
  `refs.bib`. Plain `article` class with arXiv-friendly packages; latexmk and
  biber are installed. Build script that pulls figures from `5_outputs/`.
- `refs.bib` seeded from `agent_reports/related_work.md`: 99 entries, 93 fully
  verified via arXiv API and Crossref, 6 flagged. Trim to ~60.
- Related-work corrections to carry: Rosetta Stone is Epoch AI + Google
  DeepMind (Ho, Denain, Atanasov, Albanie, Shah); Barry's code is
  `Alexander-Barry/ECI_replication` (the CLAUDE.md link is dead); the human
  baselines critique is Wei et al., ICML 2025 position paper (2506.13776);
  Epoch's own gap numbers: open 4 months behind on ECI (Edwards & Emberson,
  2026-05), China 7 months (Emberson, 2026-01).
- Skills in this session: `orwell-writing` for prose passes, `beamer` not
  needed, `paper-headline-summary` for abstract drafting, `code-review` for
  the analysis scripts. Catalog plugins worth enabling (card shown in chat):
  Deep Review (referee pass), Research Desk (citation check, related-work
  drafting), research-superpowers (manuscript workflow). None is required.
- Who writes prose: the LessWrong post had a no-ghostwriting rule. arXiv has
  none. To decide.
- Categories: stat.ML primary, cs.LG and cs.AI cross-list, or cs.AI primary.
  To decide.

## 7. Documentation to fix before writing (the paper will cite the repo)

- README "Current results" describes the no-ceiling fit under the ceiling
  flagship's name; says the flagship awaits a second seed (never run); names
  three different selected chain subsets (17; {2,5,15,17}; {2,4,5}).
- CLAUDE.md and AGENTS.md describe the pre-reorganisation layout and the
  blog-era scope; AGENTS.md line 414 reads "Codex 3.5 Sonnet".
- `docs/model_math.md` §5.4 founders N(0,1) vs code; §8 says 56 human scores
  vs README 57. `docs/plots.md` says the forecast needs humans in the fit.
- `config.py:225` says per-axis τ_o; code uses one scalar. `frontier_gap.py`
  docstring says top-1; default is top-2.
- `4_diagnostics/README.md` calls `robust_loo.py` superseded while the
  ceilings decision rests on it.
- `make_crossover.py` still prints clipped pre-2023 dates.

## 8. Public claims the paper must correct (from the blog post)

Humans inside → placed afterward. Four axis names changed; no science axis in
the ceiling fit. "All nine tiers passed on Scientific Knowledge" retracted.
All crossing dates recomputed under the two-regime forecast. Mode statement
unified (the post said both "one solution" and "4 of 10 chains in a second
mode"). The Annex B ladder mixed two scopes (5,004 vs 4,923 obs); rerun on
one scope. Ceilings moved from future work to default. PIT 0.046 was
in-sample; LOO-PIT is 0.067. Beta → Beta-binomial. Gaussian → skew-normal
abilities. 97 → 107 benchmarks. The memo's "axis 2 passed in 2022" never
reappears.

## 9. Open decisions for Yassine

1. Thesis framing: A, decided 2026-10-29.
2. Flagship for the paper: DEFERRED (2026-10-29). R1 (ceilings on) and R2
   (ceilings off) run with the same chains and seeds; the main fit is the one
   that converges better, the other is the sensitivity.
3. Which refits to launch, and in what order (R1 first in every framing).
4. Whether the frontier-gap analyses go in this paper or a separate one.
5. Human tiers at K=4: placement with the validation checks, the joint fit
   as the main line with the trade-off reported, or both.
6. Prose authorship rule and arXiv categories.
7. DECIDED 2026-11-03: refit R1 locally, later, as 10 runs of 5 chains
   merged (section 0). Every trace-level step waits on it.
8. Commit the step-0 placement fix and the preprint folder.

## 10. Order of work

1. Decisions 1 to 7 above.
2. Documentation fixes (section 7), so the repo the paper cites is consistent.
3. Minutes-level checks (section 4), in one session.
4. Launch R1 (and R2 if ceilings are the open question); while it runs, build
   the LaTeX skeleton, bib, Data and Model sections, the ECI-H and frontier-gap
   results (all citable now).
5. On R1: mode-aware diagnostics, placement validation, per-mode figures,
   crossing tables.
6. R3 to R7 as decided; ladder table; limitations.
7. Internal referee pass (red-team list in `agent_reports/red_team.md`),
   then Orwell pass, then submission checklist.
