# LAB-Bench Cloning — local eval

Runs the 33-item public CloningScenarios split of
[LAB-Bench](https://huggingface.co/datasets/futurehouse/lab-bench)
([arXiv 2407.10362](https://arxiv.org/abs/2407.10362)) against any model through OpenRouter.

```bash
export OPENROUTER_API_KEY=...
python run.py --selftest                                    # offline, no key needed
python run.py --model anthropic/claude-sonnet-4.6 --effort high
```

Resume is automatic: re-running the same `--model`/`--effort` skips items already on disk, so a
run interrupted or left with `ERROR` rows is filled by repeating the command.

## Why this benchmark

It carries the **highest per-cell mode disagreement in the K=3 fit** — on the top cell, mode A
predicts 0.92 and mode B predicts 0.40 for the same model. Our 31 rows come from RAND
RR-A3797-1, a **published report that will never gain a model**, so every 2025-2026 test-taker
is permanently missing from that column. No pipeline refresh can fix it; the only way to fill
those cells is to run the public set.

33 items also makes it the cheapest decisive measurement available — well under $1 per model.

## Protocol

- **Multiple choice, no LLM grader.** `ideal` + `distractors` are shuffled into lettered
  options and the reply is matched on the letter. That removes the grader call (half the cost
  of the SimpleQA harness this is adapted from) and the grader-disagreement failure mode.
- **Option order is seeded on the item id**, so a re-run or a resumed run scores each item
  against the identical layout.
- **An abstention option is appended last** (LAB-Bench's own convention). It is never the
  correct answer; picking it scores `C` (NOT_ATTEMPTED), not `B`. An unparseable reply is also
  `C` — the model failed to answer rather than choosing a wrong answer.
- **The `canary` field is never sent.** It is a training-contamination marker; the selftest
  asserts it stays out of the prompt.
- **Option counts vary**: 5 of them 24 times, up to 9 (including abstention). Real-option
  chance averages **0.226**; `benchmark_lower_bounds.csv` carries 0.20, slightly conservative.

## Effort is part of the test-taker identity

This repo treats effort variants as **distinct test-takers** — `claude-sonnet-5_max` is a
different row from `claude-sonnet-5`. `--effort` is forwarded to OpenRouter **verbatim and
never downgraded**: if an upstream rejects `max` or `xhigh`, that test-taker cannot be
reproduced here, which is a result to record rather than something to paper over. Filing a
no-effort run against an effort-suffixed row would be the same mis-attribution that made
Kaggle's untagged SimpleQA rows unusable.

The SimpleQA harness this replaces sent no reasoning parameter at all, which is why its
calibration set had to exclude every suffixed model.

## Calibration

`calibration_targets.csv` holds all 31 RAND rows with their scores and the stderr implied by
n=33. Reproduce a few before trusting a new number.

**Read the stderr before drawing conclusions.** At n=33 a single model's stderr is ~**0.087**,
so one model agreeing with RAND to within 5pp proves very little. Calibrate across several
models and look for a *systematic* offset — that is what n=33 can detect. It cannot pin one
model's ability precisely, and it does not need to: the modes disagree by 40-52pp on the cells
this eval exists to fill.

Five of the 31 RAND rows carry an effort suffix (`_16K`, `_medium`), so those need a matching
`--effort` to be comparable.

## Files

| Path | Tracked | What |
|---|---|---|
| `run.py` | yes | the harness; `--selftest` covers prompt construction, letter parsing, metrics |
| `calibration_targets.csv` | yes | the 31 RAND rows to calibrate against |
| `data_cache/` | no (gitignored) | the fetched 33-row dataset, sha256 recorded in each meta.json |
| `out/` | no (gitignored) | per-run `<model>__<effort>.csv` + `.meta.json` (metrics, usage, cost) |
