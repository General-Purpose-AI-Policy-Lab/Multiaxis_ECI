# Sync audit, data generation 20261019

Reference: 0_input/ at `HEAD`. Produced by `1_curated/0_audit_sync.py`; the procedure is `.claude/skills/data-refresh/SKILL.md`.

| level | findings |
|---|---|
| BLOCK | 0 |
| DECIDE | 1 |
| CHECK | 4 |
| INFO | 2 |

## DECIDE

**human placement**
- `41 benchmarks`: human scores not yet judged against the named axes (mirt_lineageprior_lineagebm), at placement time: ARC-AGI, ARC-AGI-2, BioLP-bench, BlueprintBench 2, Cybench, DTBench, EBR-bench, FrontierMath Erdos, FrontierMath Tiers 1-3 v2, Furniture Assembly, GPQA Diamond, GPQA Diamond Biology, GPQA Diamond Chemistry, GPQA Main Biology, GPQA Main Chemistry, GSM8K, GeoBench, HellaSwag, InterCode-CTF, LAB-Bench Cloning, LAB-Bench LitQA2, LAB-Bench Protocol, LAB-Bench SeqQA, LMCA, MATH Level 5, METR Time Horizons, MMLU, MMLU Biology, MMLU Chemistry, OSWorld, OTIS Mock AIME 2024-2025, PIQA, SimpleBench, SpireClimb, The Agent Company, TriviaQA, VPCT, Visual Task Assessment (VISTA), WMDP Biology, WMDP Chemistry, WinoGrande


## CHECK

**provenance**
- `../benchmark-data-pipeline`: the pipeline checkout's latest run is 20261016, the synced one 20261019: run 3_build_model_openness.py with --pipeline pointing at the checkout of the synced run, or Epoch's labels of the new models are missed (they then come out `unknown`)

**human baseline**
- `SimpleQA Verified / Skilled Generalist`: baseline 0.944 removed upstream

**openness to re-check**
- `mistral-large-4-0`: API public preview on 2026-10-06; open weights announced as coming soon (mistralai/Mistral-Large-4-1T-A52B still private on 2026-10-20, press: end of October). Becomes open if the weights land by 2026-11-05 (30-day window): re-check
- `launa-omni-1`: Salesforce AI Research voice model (2026-10-12), inside Salesforce / Slack products; no API, no weights found on 2026-10-20 (window open until about 2026-11-11)


## INFO

**provenance**
- `20261019`: pipeline commit 786d28664, run 20261019

**scores**
- `5588`: +0 (model, benchmark) cells; 5588 carried over, 0 revised by more than 0.05
