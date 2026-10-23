# Sync audit, data generation 20261019

Reference: 0_input/ at `095dfc0739e1e920a9bfa41f9febe8fc5e87335d~1`. Produced by `1_curated/0_audit_sync.py`; the procedure is `.claude/skills/data-refresh/SKILL.md`.

| level | findings |
|---|---|
| BLOCK | 0 |
| DECIDE | 0 |
| CHECK | 4 |
| INFO | 2 |

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
