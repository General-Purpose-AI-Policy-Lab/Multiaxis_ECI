---
name: data-refresh
description: Procedure for every data refresh or addition — refreshing the feeds of benchmark-data-pipeline, syncing them into Multiaxis_ECI, or adding a model, a benchmark, a source or a human baseline by hand. Use it whenever the user asks to refresh the feeds ("rafraîchis les flux"), sync new data, add or exclude a benchmark, add evaluation results, or start a refit on new data. It lists every table a new model or benchmark must reach in both repositories, the automatic audit to run, how to investigate what it flags, and which judgement calls go to the user.
---

# Data refresh: add, check, investigate, escalate

Every new model, benchmark or human baseline must reach every table that reads it, in two
repositories: `../benchmark-data-pipeline` (scores, identities, metadata) and this one (curated
model inputs, fits, analyses). Nothing is ingested silently and nothing is decided silently: what
the data cannot settle is investigated, and what is a judgement call goes to the user.

Four levels, the same as `1_curated/0_audit_sync.py`:

- **BLOCK**: a fit would get it wrong or skip it silently. Fix before any fit.
- **DECIDE**: a judgement call. Research it, then ask the user with a recommendation.
- **CHECK**: unexpected. Investigate, then either explain it in the report or escalate it.
- **INFO**: what changed, for the record.

Standing rules: one sampling at a time on the laptop; commit only when asked, push only when
explicitly asked; never delete with `rm` (use `~/.Trash`); a refit happens in a worktree, not on
main, until the user approves the merge.

## 1. Pipeline: refresh and build (`../benchmark-data-pipeline`)

Work on a branch (`refresh-YYYYMMDD`), in a worktree when another session uses the checkout.

1. `python -m benchmark_data refresh`, then `python -m benchmark_data build`.
2. **Read the whole build log.** Every line below is a finding, never noise:
   - unknown files, tasks or labels of a feed (`*_SPEC` / `*_SKIP` / `FILE_SKIP`): each one is
     either ingested (a `_SPEC` entry naming the benchmark) or set aside (a `_SKIP` entry with the
     reason). Variants of a benchmark (sub-settings, other harnesses, developer-reported series)
     are a DECIDE: one series, separate benchmarks, or skipped.
   - unresolved model names: an alias in `0_input/feed/<feed>/aliases.csv` (dated snapshot when
     the source dates its run; effort in machine-name grammar).
   - a benchmark missing from `0_input/metadata/benchmarks.csv`: the build refuses it.
   - scores below a floor, corrections that stopped applying, release-date conflicts.
3. **Each new benchmark gets a full row** in `0_input/metadata/benchmarks.csv`, appended (never
   re-sort the file; keep LF line endings):
   - `status`, `exclusion_criterion` (e.g. `frontier_coverage` for a developer's report on its own
     models only), `category`;
   - `floor` with `floor_reason` and `floor_source`: the chance level (1/k for k-choice MCQ, 0 when
     the answer cannot be guessed). The source is the page that documents the task format (the
     benchmark's paper or page), never the data file the scores come from;
   - `ceiling` with reason and source only when every item is verified solvable, or a human
     reaches the top of the scale;
   - `access` (public, semi_private, private) with `access_verified` and `access_evidence_url`;
   - a `note`: what the benchmark measures, who ran it, how many items.
   Also `benchmark_release_dates.csv`, and `organizations.csv` for a new organisation (country).
4. **Each new human baseline** (`human_baselines.csv`): only under conditions comparable to the
   models'. A 1.0 obtained by construction (answer key, validators, solution authors) is rejected.
   Every new baseline is a DECIDE.
5. Scores that look wrong: `score_corrections.csv`, `score_drop_rules.csv` (a class of truncated
   runs), `score_variant_fixes.csv` (the effort a run used). Each with its source and a line in
   `docs/decisions.md`.
6. `docs/decisions.md`: one section per refresh with every decision taken and why.
7. `pytest`, `python -m benchmark_data check`; re-bless the expected tables with the command it
   prints once the differences are understood; `python -m benchmark_data parity`.
8. Commit in two steps so the manifest names a commit that exists: first the code and metadata,
   then build and commit the build (`Run YYYYMMDD: ...`). A metadata-only fix rebuilds the same
   run with `build --date YYYYMMDD`. Never amend a commit a build manifest names.

## 2. Multiaxis: sync and audit

In the refit worktree (`git worktree add ../Multiaxis_ECI-<name> -b <branch>`; symlink `.venv`).

1. `python -m multiaxis_eci sync --pipeline <pipeline checkout of the run>`.
2. `python 1_curated/0_audit_sync.py` (it compares with the previous generation and writes
   `5_outputs/data<YYYYMMDD>/sync_audit.md`). The first pass lists everything to do.
3. Run the builders, **all pointed at the same pipeline checkout as the sync**:
   - `python 1_curated/2_build_lineage_map.py`, then `pytest tests/test_lineage_map.py`. Fix
     misplacements in the builder rules or `lineage_node_overrides.csv`; a new chain or a model
     left out of every chain with several scores is a DECIDE when the family is not obvious.
   - `python 1_curated/3_build_model_openness.py --pipeline <same checkout>` (with the wrong
     checkout, Epoch's labels of the new models are missed). Research every `unknown` (weights
     within 30 days of release?) and add `model_openness_overrides.csv` rows with the evidence;
     a model still inside its 30-day window is labelled as of today with a note containing
     "re-check" and the date the window closes.
   - `python 1_curated/4_build_model_regimes.py`, read the diff.
   - `1_curated/benchmark_n_items.csv`: one row per new benchmark (N items checked against the
     score grid, or `continuous`), with the source.
   - `python 4_diagnostics/audit_lower_bounds.py --write-clips`, read the diff (new below-floor
     scores are raised to the floor, same rule as their neighbours).
   - `1_curated/organization_countries.csv` for a new organisation without a country.
   - `1_curated/excluded_benchmarks.txt`: whether each new benchmark enters the canonical index
     (DECIDE: not built to be easy for humans, not mainly visual).
4. Re-run the audit until it reports no BLOCK, then the tests: `pytest` (coverage tests in
   `tests/test_curated_coverage.py`). Golden logp failures after a data change are expected:
   re-pin all of them in one pass from the assertion's ACTUAL values, by the line pytest reports
   (two goldens can share a value), and add a dated entry to the docstring of
   `tests/test_safety_net.py` with the new scope (obs / models / benchmarks).
5. Commit the sync, the curated tables, the goldens and `sync_audit.md` together.

## 3. Investigating CHECK findings

Explain each one in the report to the user or escalate it; never leave one unread.

- **frontier jump**: open the source page; same metric, harness and scoring as before? New
  frontier models, or a changed series (renamed, rescaled, re-run)? Do the same models jump on
  other benchmarks? Our own runs (`gpai_runs`) are a source like any other.
- **score revised / scores dropped**: an upstream revision (METR, AISI and Epoch republish) or an
  identity change (alias, effort relabel). Check the feed diff in the pipeline.
- **effort inversion**: a higher effort far below a lower one is usually a truncated run (token
  limit): a drop rule in the pipeline, with its evidence.
- **lineage drop**: wrong parent node, different harness, or a real regression; look at the
  release notes before moving anything.
- **alias**: one release under two spellings: an alias in the pipeline, never a merge here.
- **close benchmark names**: a duplicate series, a split by who ran it, or a new version (pooling
  versions in one loading row is not allowed; the old version may need retiring).
- **thin benchmark**, **chance level?**, **undated model**, **no country**: complete the metadata
  or state why it stays so.
- **pipeline commit on no remote**: the pipeline branch must be merged and pushed (ask) before the
  refit reaches main.
- **openness to re-check**: re-check after the date in the note; a label that changes moves the
  open/closed analysis.

## 4. Escalating DECIDE findings

Research first (sources, how similar cases were decided: `docs/decisions.md`, the curated files'
notes, the memory), then ask in **one batch** with `AskUserQuestion`: at most four questions, the
recommended option first and marked "(Recommandé)", each option stating its consequence. Typical
calls:

- a new benchmark in or out of the canonical index; a series split, pooled or skipped;
- a human baseline accepted or rejected; a floor or ceiling that is not obvious;
- a new lineage chain or a doubtful node; a model left out of every chain;
- an openness label inside its 30-day window; an access class without clear evidence;
- after a K=4 fit: carrying the axis names over (signature benchmarks still define the axis?),
  and the `human_axis_content.csv` judgements of new benchmarks with human scores.

Record each answer where it lives: the pipeline's `docs/decisions.md` and metadata notes, the
curated files' note columns, comments in `excluded_benchmarks.txt`, the `decided` column of
`human_axis_content.csv` ("YYYY-MM-DD user").

## 5. After the fits

- `python 1_curated/1_compute_sota.py` once the canonical fit exists; read the diff.
- Frontier analyses: `TODAY=YYYY-MM-DD 5_outputs/open_closed_frontier/make_plots.sh` and
  `5_outputs/us_cn_frontier/make_plots.sh`, then update the numbers, counts and caveats of both
  READMEs (scope sizes, divergences of the class fits, candidate and record counts).
- K=4: `diagnose_chains.py --write-modes`, match the axes to the named ones before reusing names
  (ask), `--write-tables`, `5_place_humans.py` (judgements of new benchmarks first), figures,
  dashboard.

## 6. Report to the user

A short report: what entered (models, benchmarks, baselines), the decisions taken (by whom), the
CHECK findings and their explanation, what is still pending (re-checks with dates, decisions
awaiting an answer, branches not pushed).
