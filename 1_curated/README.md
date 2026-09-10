# 1_curated/

Inputs that belong to this model rather than to the data. Everything about the scores themselves (feeds, aliases, release dates, benchmark inclusion, categories, chance floors, known ceilings, access classes, human baselines) lives in [`benchmark-data-pipeline`](https://github.com/General-Purpose-AI-Policy-Lab/benchmark-data-pipeline) and arrives in `0_input/` through `python -m multiaxis_eci sync`.

| File | Kind | Consumed by | Edit by hand? |
|---|---|---|---|
| `excluded_benchmarks.txt` | hand-maintained list | `data.py` at fit time: the canonical preset, and exploration under `--apply-exclusions` | yes, one benchmark name per line (the pipeline's names), `#` comments allowed |
| `lineage_map.csv` | generated draft, then hand-reviewed | `lineage.py` (lineage prior) | yes, it is the reviewed source of truth; `2_build_lineage_map.py` redrafts it from `0_input/` and `lineage_node_overrides.csv` |
| `lineage_node_overrides.csv` | hand-maintained | `2_build_lineage_map.py` | yes |
| `sota_families.txt` | generated | `config.SOTA_FAMILIES` (timelines always show every effort of the release, the country frontier admits them as record candidates, the SOTA table of the canonical fit keeps the best effort; the list never decides what the fit sees) | no, regenerate with `python 1_curated/1_compute_sota.py` after a canonical fit: every world frontier record whatever its age plus every family within 10 ECI of the best model of each organisation on the recent frontier (records of the last 24 months), one family (base model and snapshot) per line, candidates with at least 4 observations |
| `benchmark_score_clips.csv` | generated draft, then reviewed | `data.clip_scores_to_floors` applies these row-level clips at fit time (floors fits, the default); drift against the current data warns loudly | no, refresh with `python 4_diagnostics/audit_lower_bounds.py --write-clips` after a sync, review the diff, commit |
| `simpleqa_original/` | curated extra column | `--simpleqa-original` | see its README |
| `eci_data.csv` | reference | `--eci-data-only` (Epoch's original ECI table) | no |
| `benchmark_n_items.csv` | hand-researched | `data.load_boundary_eps` for `--censor-bounds` (eps_b = 1 / (2 N_b)); a benchmark without a row falls back to 0.001 with a warning | yes, `benchmark, n_items, source_url, note, verification` with the pipeline's names |

The two builders run after a sync: `2_build_lineage_map.py` redrafts the lineage map from the new model list; `1_compute_sota.py` reads the canonical fit of the current data generation (`all_models_eci.csv`, `timeline.csv`), so it runs once that fit exists and samples nothing itself.

## Effort variants

A release appears as several test-takers when the sources record it at several reasoning efforts (`_low`, `_high`, a token budget, `_thinking`); they are fitted separately and grouped into one family by `data.model_family` where a rule needs the release (isolated families, the SOTA list). From its 2026-09-09 build (the next sync here) the pipeline folds Epoch's `_unknown` suffix into the bare name: an unrecorded effort and no suffix are the same state of knowledge, only an explicit `_none` (reasoning off) is a configuration of its own.
