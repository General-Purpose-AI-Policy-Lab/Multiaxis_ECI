# 1. Data

Everything the fit reads, and the four steps that produce it. Run them in order
after an upstream refresh; skip straight to `python 2_fit.py` if the tables here
are already current.

| Step | What it does |
|---|---|
| `1_pipeline/pipeline.ipynb` | Restart Kernel → Run All. Pulls a dated snapshot of the upstream feeds and writes `processed/benchmarks_merged.csv`. See [`1_pipeline/README.md`](1_pipeline/README.md) |
| `2_compute_sota.py` | Recompute the data-driven SOTA list → `curated/sota_models.txt` |
| `3_build_lineage_map.py` | Draft the vendor release chains → `curated/lineage_map.csv` |
| `4_build_country_map.py` | Country of origin (US / CN / Other) per model_version → `curated/model_country.csv` |

The last three read `processed/benchmarks_merged.csv`, so they run after the
notebook and before the fit. They write into `curated/`, which is reviewed by
hand: each one drafts, you check the diff, and the overrides files next to their
outputs are where a correction goes so the next run does not undo it.

## Storage

| Folder | Contents |
|---|---|
| `raw/` | The original reference ECI CSV. Only `--eci-data-only` reads it |
| `processed/` | `benchmarks_merged.csv`, the table `multiaxis_eci/data.py` actually fits |
| `curated/` | Hand-maintained inputs: chance floors, benchmark access classes, human baselines, exclusion and retirement lists, lineage and country maps, per-refresh row fixes |
| `1_pipeline/canonical/` | The name and alias dictionaries the notebook resolves against |
| `1_pipeline/output/` | The notebook's reports, including `pipeline_report.md` |

Revert a bad refresh with `git checkout 1_data/processed 1_data/curated`.

## Where it comes from

Five public feeds: Epoch AI (ZIP and live), RAND, Scale SEAL and Kaggle. Their
licensing terms differ and not all of them are open. Read [`../NOTICE.md`](../NOTICE.md)
before republishing anything derived from these tables.
