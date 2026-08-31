# NOTICE: third-party data

The `LICENSE` (CC-BY-4.0) covers what this project produces: the model code,
the analysis and diagnostics layer, the curated files under `1_data/curated/`, and
the fitted results under `results/`.

It does **not** cover the upstream benchmark scores those results are derived
from. Each source carries its own terms. Anyone republishing this analysis must
credit Epoch AI, the RAND report and Scale AI.

Full per-section detail: [`1_data/1_pipeline/README.md`](1_data/1_pipeline/README.md).

## Benchmark score sources

| Source | Terms |
|---|---|
| [`epoch.ai/data/benchmark_data.zip`](https://epoch.ai/data/benchmark_data.zip) and [`epoch.ai/data/benchmarks.csv`](https://epoch.ai/data/benchmarks.csv) | **CC-BY** (Epoch AI), attribution required |
| RAND **RR-A3797-1** (Dev et al. 2025, *Toward Comprehensive Benchmarking of the Biological Knowledge of Frontier LLMs*) | Cite [RR-A3797-1](https://www.rand.org/pubs/research_reports/RRA3797-1.html) |
| [Scale SEAL leaderboards](https://labs.scale.com/leaderboard) | **No open license.** Scale AI Terms of Service applies. Aggregated as research with attribution; confirm terms at <https://scale.com/legal/terms> before redistributing |
| SimpleQA original scores (`1_data/curated/simpleqa_original/`) | MIT, from the upstream benchmark repository |

The Epoch Capabilities Index this project recreates is Epoch AI's:
<https://epoch.ai/eci>.

## Evaluation harness

`evals/lab_bench_cloning/` runs the public CloningScenarios split of
**LAB-Bench** ([futurehouse/lab-bench](https://huggingface.co/datasets/futurehouse/lab-bench),
[arXiv 2407.10362](https://arxiv.org/abs/2407.10362)). The dataset is fetched on
demand and is **not** redistributed here; its own license applies. Run outputs
under `evals/*/out/` are gitignored.

## Attribution when citing

> Beta-MIRT recreation of the Epoch Capabilities Index by the General-Purpose
> AI Policy Lab (CC-BY-4.0), over benchmark scores from Epoch AI (CC-BY),
> RAND RR-A3797-1 (Dev et al. 2025) and Scale AI SEAL.
