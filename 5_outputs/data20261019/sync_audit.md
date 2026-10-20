# Sync audit, data generation 20261019

Reference: 0_input/ at `766474abaeaa2cb856d495f16ef0f2df0e2997ac~1`. Produced by `1_curated/0_audit_sync.py`; the procedure is `.claude/skills/data-refresh/SKILL.md`.

| level | findings |
|---|---|
| BLOCK | 0 |
| DECIDE | 16 |
| CHECK | 13 |
| INFO | 12 |

## DECIDE

**new benchmark**
- `CAISI CTF-Archive`: enters the canonical index (not in excluded_benchmarks.txt): confirm it is neither built to be easy for humans nor mainly visual (the 2026-09-29 rule)
- `CAISI CTF-Archive-Diamond`: enters the canonical index (not in excluded_benchmarks.txt): confirm it is neither built to be easy for humans nor mainly visual (the 2026-09-29 rule)
- `CAISI CVE-Bench`: enters the canonical index (not in excluded_benchmarks.txt): confirm it is neither built to be easy for humans nor mainly visual (the 2026-09-29 rule)
- `CAISI Cybench`: enters the canonical index (not in excluded_benchmarks.txt): confirm it is neither built to be easy for humans nor mainly visual (the 2026-09-29 rule)
- `CAISI ExploitGym Userspace`: enters the canonical index (not in excluded_benchmarks.txt): confirm it is neither built to be easy for humans nor mainly visual (the 2026-09-29 rule)
- `CAISI OSS-Fuzz`: enters the canonical index (not in excluded_benchmarks.txt): confirm it is neither built to be easy for humans nor mainly visual (the 2026-09-29 rule)
- `CAISI SEC-Bench Pro`: enters the canonical index (not in excluded_benchmarks.txt): confirm it is neither built to be easy for humans nor mainly visual (the 2026-09-29 rule)
- `CyScenarioBench (Anthropic)`: enters the canonical index (not in excluded_benchmarks.txt): confirm it is neither built to be easy for humans nor mainly visual (the 2026-09-29 rule)
- `SpireClimb`: enters the canonical index (not in excluded_benchmarks.txt): confirm it is neither built to be easy for humans nor mainly visual (the 2026-09-29 rule)

**human baseline**
- `SpireClimb / Top Performer`: new baseline 0.9: accept? (comparable conditions, not 1.0 by construction)

**lineage**
- `Muse-Spark-Open-816B-A42B`: founds a new chain `muse-spark-open` (node spark open): a new line, or a node of an existing one?
- `Muse-Spark-Open-816B-A42B_xhigh`: founds a new chain `muse-spark-open` (node spark open): a new line, or a node of an existing one?
- `mimo-v2.6-flash`: founds a new chain `mimo-flash` (node v2.6-flash): a new line, or a node of an existing one?
- `muse-wonder-1.0`: founds a new chain `muse-wonder` (node wonder 1.0): a new line, or a node of an existing one?
- `muse-wonder-1.0_max`: founds a new chain `muse-wonder` (node wonder 1.0): a new line, or a node of an existing one?
- `muse-wonder-1.0_xhigh`: founds a new chain `muse-wonder` (node wonder 1.0): a new line, or a node of an existing one?


## CHECK

**provenance**
- `2216933a2`: the synced pipeline commit is on no remote branch: merge and push the pipeline branch before the refit reaches main, or 0_input names a commit nobody else can check out
- `../benchmark-data-pipeline`: the pipeline checkout's latest run is 20261016, the synced one 20261019: run 3_build_model_openness.py with --pipeline pointing at the checkout of the synced run, or Epoch's labels of the new models are missed (they then come out `unknown`)

**new benchmark**
- `CAISI Cybench`: close to existing ['Cybench']: a duplicate series, a split or a new version of one of them? (pooling, versions and retirements are decisions)
- `CAISI ExploitGym Userspace`: only 4 test-takers
- `CAISI OSS-Fuzz`: only 4 test-takers
- `CAISI SEC-Bench Pro`: only 4 test-takers
- `CyScenarioBench (Anthropic)`: close to existing ['CyScenarioBench']: a duplicate series, a split or a new version of one of them? (pooling, versions and retirements are decisions)

**frontier jump**
- `CyScenarioBench`: best score 0.367 -> 0.590 (gpt-6-astra 0.590, gpt-5.6-sol 0.270, gpt-5.5 0.260): same metric and harness as before? genuine new frontier models?
- `GSO-Bench`: best score 0.471 -> 0.882 (claude-fable-5-1_thinking 0.882, gpt-6-astra 0.794, claude-fable-5_thinking 0.784): same metric and harness as before? genuine new frontier models?

**release date**
- `launa-omni-1`: undated (1 scores): left out of the dated views, the timelines and the frontier analyses

**scores dropped**
- `6`: scores of models still present are gone: CyScenarioBench (3), METR Time Horizons (3)

**openness to re-check**
- `mistral-large-4-0`: API public preview on 2026-10-06; open weights announced as coming soon (mistralai/Mistral-Large-4-1T-A52B still private on 2026-10-20, press: end of October). Becomes open if the weights land by 2026-11-05 (30-day window): re-check
- `launa-omni-1`: Salesforce AI Research voice model (2026-10-12), inside Salesforce / Slack products; no API, no weights found on 2026-10-20 (window open until about 2026-11-11)


## INFO

**provenance**
- `20261019`: pipeline commit 2216933a2, run 20261019

**new benchmark**
- `CAISI CTF-Archive`: 7 test-takers, category Cyber, floor 0.0, ceiling 1.0, access private, scores 0.085-0.506
- `CAISI CTF-Archive-Diamond`: 8 test-takers, category Cyber, floor 0.0, ceiling 1.0, access private, scores 0.317-0.705
- `CAISI CVE-Bench`: 8 test-takers, category Cyber, floor 0.0, ceiling nan, access public, scores 0.267-0.667
- `CAISI Cybench`: 8 test-takers, category Cyber, floor 0.0, ceiling 1.0, access public, scores 0.167-0.735
- `CAISI ExploitGym Userspace`: 4 test-takers, category Cyber, floor 0.0, ceiling nan, access private, scores 0.026-0.444
- `CAISI OSS-Fuzz`: 4 test-takers, category Cyber, floor 0.0, ceiling nan, access private, scores 0.024-0.232
- `CAISI SEC-Bench Pro`: 4 test-takers, category Cyber, floor 0.0, ceiling nan, access public, scores 0.273-0.902
- `CyScenarioBench (Anthropic)`: 7 test-takers, category Cyber, floor 0.0, ceiling nan, access private, scores 0.000-0.676
- `SpireClimb`: 7 test-takers, category Agentic Computer Use, floor 0.0, ceiling 1.0, access dynamic, scores 0.000-0.257

**new models**
- `35`: Muse-Spark-Open-816B-A42B, Muse-Spark-Open-816B-A42B_xhigh, claude-haiku-5-5, claude-haiku-5-5_high, claude-haiku-5-5_medium, claude-haiku-5-5_xhigh, claude-mythos-5-1, claude-opus-5-5_high, claude-opus-5-5_low, claude-opus-5-5_thinking, claude-sonnet-5, claude-sonnet-5-5, deepseek-v4.1-flash_low, deepseek-v4.1-flash_none, gemini-4-argon, gemini-4-argon_high, gemini-4-argon_medium, glm-5.3-flash_high, glm-5.3-flash_low, glm-5.3_low, gpt-5.4-mini-2026-03-17, gpt-6.1-sol, gpt-oss-120b, grok-4.7_high, grok-4.7_low, grok-4.7_medium, launa-omni-1, mimo-v2.6-flash, mimo-v2.6-pro, mistral-large-4-0, mistral-large-4-0_high, muse-glimmer_high, muse-wonder-1.0, muse-wonder-1.0_max, muse-wonder-1.0_xhigh

**scores**
- `5588`: +270 (model, benchmark) cells; 5312 carried over, 0 revised by more than 0.05
