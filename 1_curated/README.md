# 1_curated/

Inputs that belong to this model rather than to the data. Everything about the scores themselves (feeds, aliases, release dates, benchmark inclusion, categories, chance floors, known ceilings, access classes, human baselines) lives in [`benchmark-data-pipeline`](https://github.com/General-Purpose-AI-Policy-Lab/benchmark-data-pipeline) and arrives in `0_input/` through `python -m multiaxis_eci sync`.

| File | Kind | Consumed by | Edit by hand? |
|---|---|---|---|
| `excluded_benchmarks.txt` | hand-maintained list | `data.py` at fit time: the canonical preset, and exploration under `--apply-exclusions` | yes, one benchmark name per line (the pipeline's names), `#` comments allowed |
| `lineage_map.csv` | generated draft, then hand-reviewed | `lineage.py` (lineage prior), `1_compute_sota.py` (flagship chains) | yes, it is the reviewed source of truth; `2_build_lineage_map.py` redrafts it from `0_input/` and `lineage_node_overrides.csv` |
| `lineage_node_overrides.csv` | hand-maintained | `2_build_lineage_map.py` | yes |
| `sota_models.txt` | generated | `config.SOTA_MODELS` (plot exemptions, drop-filter protection) | no, regenerate with `python 1_curated/1_compute_sota.py` after a sync |
| `benchmark_score_clips.csv` | generated draft, then reviewed | `data.clip_scores_to_floors` applies these row-level clips at fit time (floors fits, the default); drift against the current data warns loudly | no, refresh with `python 4_diagnostics/audit_lower_bounds.py --write-clips` after a sync, review the diff, commit |
| `simpleqa_original/` | curated extra column | `--simpleqa-original` | see its README |
| `eci_data.csv` | reference | `--eci-data-only` (Epoch's original ECI table) | no |
| `benchmark_n_items.csv` | reference | nothing reads it; item counts per benchmark, kept for the record | yes |

The two builders are numbered in the order they run after a sync: `1_compute_sota.py` needs the lineage map that `2_build_lineage_map.py` writes only when chains changed, so re-run `2_` first when the model list moved.

## `_unknown` model variants

A model appears as both a bare `X` and an `X_unknown` test-taker when the source data records it in more than one run configuration: `_unknown` collects thinking-mode or unspecified-effort rows, the bare name is the base configuration. They are kept as separate test-takers, like the `_low` / `_high` effort variants; merging them would mix configurations. Identical scores on a handful of cells come from Epoch's own double-logging and are left as the pipeline delivers them.
