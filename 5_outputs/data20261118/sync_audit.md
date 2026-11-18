# Sync audit, data generation 20261118

Reference: 0_input/ at `HEAD`. Produced by `1_curated/0_audit_sync.py`; the procedure is `.claude/skills/data-refresh/SKILL.md`.

| level | findings |
|---|---|
| BLOCK | 0 |
| DECIDE | 10 |
| CHECK | 35 |
| INFO | 4 |

## DECIDE

**new benchmark**
- `GDP.pdf v1.1`: enters the canonical index (not in excluded_benchmarks.txt): confirm it is neither built to be easy for humans nor mainly visual (the 2026-09-29 rule)

**lineage**
- `gemini-4-argon_low`: founds a new chain `gemini-argon` (node argon 4): a new line, or a node of an existing one?
- `gemini-4-carbon`: founds a new chain `gemini-carbon` (node carbon 4): a new line, or a node of an existing one?
- `gemini-4-carbon_high`: founds a new chain `gemini-carbon` (node carbon 4): a new line, or a node of an existing one?
- `gemini-4-carbon_low`: founds a new chain `gemini-carbon` (node carbon 4): a new line, or a node of an existing one?
- `gemini-4-carbon_medium`: founds a new chain `gemini-carbon` (node carbon 4): a new line, or a node of an existing one?
- `gemini-4-hydrogen`: founds a new chain `gemini-hydrogen` (node hydrogen 4): a new line, or a node of an existing one?
- `gemini-4-hydrogen_high`: founds a new chain `gemini-hydrogen` (node hydrogen 4): a new line, or a node of an existing one?
- `qwen3.5-0.8b`: founds a new chain `qwen-open-0.8b` (node 3.5-0.8B): a new line, or a node of an existing one?
- `qwen3.5-0.8b_none`: founds a new chain `qwen-open-0.8b` (node 3.5-0.8B): a new line, or a node of an existing one?


## CHECK

**provenance**
- `ed95c6468`: the synced pipeline commit is on no remote branch: merge and push the pipeline branch before the refit reaches main, or 0_input names a commit nobody else can check out

**benchmark removed**
- `GDP.pdf`: in the previous generation, not in this one: a retirement or a rename upstream? (a rename needs every curated table renamed too)

**new benchmark**
- `GDP.pdf v1.1`: floor 0 but the weakest tenth scores above 0.36: is there a chance level (multiple choice)?
- `GDP.pdf v1.1`: close to existing ['GDP.pdf']: a duplicate series, a split or a new version of one of them? (pooling, versions and retirements are decisions)

**frontier jump**
- `EBR-bench`: best score 0.762 -> 0.924 (gpt-6.2-astra_max 0.924, claude-fable-5-2_max 0.905, gemini-4-carbon_high 0.876): same metric and harness as before? genuine new frontier models?
- `MirrorCode`: best score 0.774 -> 0.978 (claude-sonnet-5-5_max 0.978, claude-opus-5-5_max 0.956, claude-fable-5-1_max 0.933): same metric and harness as before? genuine new frontier models?
- `Remote Labor Index`: best score 0.212 -> 0.412 (claude-fable-5-2 0.412, claude-opus-5-5 0.212, gpt-6-astra 0.208): same metric and harness as before? genuine new frontier models?
- `SimpleQA Verified`: best score 0.775 -> 0.940 (claude-fable-5-2_max 0.940, gpt-6.2-astra_max 0.785, gemini-4-carbon_high 0.777): same metric and harness as before? genuine new frontier models?
- `SpireClimb`: best score 0.257 -> 0.413 (gpt-6.2-astra_max 0.413, gpt-6-astra_max 0.340, gpt-6.1-sol_max 0.238): same metric and harness as before? genuine new frontier models?

**model removed**
- `Mistral-7B-Instruct-v0.2`: had 3 scores, gone: an alias change upstream? (its curated rows - lineage, openness, overrides - may now be orphans)
- `claude-fable-5-1_thinking`: had 4 scores, gone: an alias change upstream? (its curated rows - lineage, openness, overrides - may now be orphans)
- `claude-fable-5_thinking`: had 6 scores, gone: an alias change upstream? (its curated rows - lineage, openness, overrides - may now be orphans)
- `claude-opus-5_thinking`: had 4 scores, gone: an alias change upstream? (its curated rows - lineage, openness, overrides - may now be orphans)
- `claude-sonnet-5_thinking`: had 6 scores, gone: an alias change upstream? (its curated rows - lineage, openness, overrides - may now be orphans)
- `grok-3-mini-beta_high`: had 5 scores, gone: an alias change upstream? (its curated rows - lineage, openness, overrides - may now be orphans)
- `grok-3-mini-beta_low`: had 6 scores, gone: an alias change upstream? (its curated rows - lineage, openness, overrides - may now be orphans)
- `text-davinci-001`: had 7 scores, gone: an alias change upstream? (its curated rows - lineage, openness, overrides - may now be orphans)

**lineage**
- `doubao-seed-2.0-lite-260428_high`: left out of every chain (3 scores): right?
- `doubao-seed-2.0-pro-260215_high`: left out of every chain (3 scores): right?
- `doubao-seed-2.1-pro-260628_high`: left out of every chain (3 scores): right?
- `doubao-seed-2.1-turbo-260628_high`: left out of every chain (3 scores): right?
- `doubao-seed-evolving_high`: left out of every chain (3 scores): right?
- `seed-1.6`: left out of every chain (3 scores): right?

**score revised**
- `The Agent Company / Llama-3.1-70B-Instruct`: 0.128 -> 0.065: an upstream revision (source page, METR/AISI update) or an identity change?
- `PostTrainBench v1.1 / gpt-5.6-sol_max`: 0.362 -> 0.310: an upstream revision (source page, METR/AISI update) or an identity change?
- `SpireClimb / gpt-6-astra_max`: 0.257 -> 0.340: an upstream revision (source page, METR/AISI update) or an identity change?
- `GSM8K / internlm-chat-20b`: 0.157 -> 0.434: an upstream revision (source page, METR/AISI update) or an identity change?
- `APEX-Agents / muse-spark-1.2_xhigh`: 0.364 -> 0.454: an upstream revision (source page, METR/AISI update) or an identity change?
- `APEX-Agents / mimo-v2.6-pro`: 0.595 -> 0.544: an upstream revision (source page, METR/AISI update) or an identity change?
- `MirrorCode / claude-opus-5-5_max`: 0.774 -> 0.956: an upstream revision (source page, METR/AISI update) or an identity change?

**scores dropped**
- `106`: scores of models still present are gone: GDP.pdf (41), Terminal Bench 2.0 (14), APEX-Agents (11), GSO-Bench (6), WeirdML v2 (4), Aider Polyglot (2), PIQA (2), Mystery Game Puzzles (2), SimpleQA Verified (2), Cybench (2), DTBench (2), LMCA (2)

**openness to re-check**
- `mistral-large-4-0`: API public preview on 2026-10-06; the weights (mistralai/Mistral-Large-4-1T-A52B, private on 2026-10-20) are public on Hugging Face on 2026-11-18 and Epoch dates the open release 2026-10-29, inside the 30-day window that closed on 2026-11-05. Re-checked 2026-11-18: open
- `muse-ember-1.0`: Meta AI, released 2026-11-09; Epoch: API access, no weights announced on 2026-11-18. Inside its 30-day window (closes 2026-12-09): re-check
- `muse-ember-1.0_max`: same release as muse-ember-1.0: re-check after 2026-12-09
- `doubao-seed-evolving_high`: ByteDance, released 2026-11-02; Epoch: API access, no weights announced on 2026-11-18. Inside its 30-day window (closes 2026-12-02): re-check


## INFO

**provenance**
- `20261118`: pipeline commit ed95c6468, run 20261118

**new benchmark**
- `GDP.pdf v1.1`: 19 test-takers, category Domain Specific Questions, floor 0.0, ceiling nan, access public, scores 0.328-0.564

**new models**
- `73`: MiniMax-M3.1-Flash_xhigh, Muse-Spark-Open-816B-A42B_high, Muse-Spark-Open-816B-A42B_low, Muse-Spark-Open-816B-A42B_medium, beam-1.0, beam-1.0_high, beam-1.0_low, beam-1.0_max, beam-1.0_medium, beam-1.0_xhigh, claude-fable-5-2, claude-fable-5-2_high, claude-fable-5-2_low, claude-fable-5-2_max, claude-fable-5-2_medium, claude-fable-5-2_xhigh, claude-opus-4-1-20250805_2K, deepseek-v4.1-flash-1017, deepseek-v4.1-flash-1017_high, deepseek-v4.1-flash-1017_low, deepseek-v4.1-flash-1017_max, deepseek-v4.1-flash-1017_none, deepseek-v4.1-flash-1017_xhigh, doubao-seed-2.0-lite-260428_high, doubao-seed-2.0-pro-260215_high, doubao-seed-2.1-pro-260628_high, doubao-seed-2.1-turbo-260628_high, doubao-seed-evolving_high, gemini-2.5-flash_16K, gemini-2.5-pro-preview-05-06_1K, gemini-2.5-pro_1K, gemini-4-argon_low, gemini-4-carbon, gemini-4-carbon_high, gemini-4-carbon_low, gemini-4-carbon_medium, gemini-4-hydrogen, gemini-4-hydrogen_high, gemma-3-270m-it, gemma-4-E2B-it_thinking, gemma-4-E4B-it_thinking, gpt-6.1-terra, gpt-6.1-terra_high, gpt-6.1-terra_low, gpt-6.1-terra_max, gpt-6.1-terra_medium, gpt-6.1-terra_none, gpt-6.1-terra_xhigh, gpt-6.2-astra_max, grok-3-mini_medium, hy4-preview_high, inkling-2-small, inkling-2-small_xhigh, kimi-k2.7-code_high, kimi-k3.1-pro, kimi-k3.1-pro_max, kimi-k3.1-pro_xhigh, ministral-3b-2512, ministral-3b-2512_thinking, ministral-8b-2512_thinking, mistral-large-4-0-oss_high, mistral-large-4-0-oss_medium, muse-ember-1.0, muse-ember-1.0_max, nemotron-3-nano-4b_thinking, nemotron-3-ultra_high, nemotron-3.5-bolt-e4b_thinking, nvidia-nemotron-nano-9b-v2, qwen3.5-0.8b, qwen3.5-0.8b_none, qwen3.5-2b_none, qwen3.5-4b_thinking, seed-1.6

**scores**
- `5975`: +387 (model, benchmark) cells; 5359 carried over, 7 revised by more than 0.05
