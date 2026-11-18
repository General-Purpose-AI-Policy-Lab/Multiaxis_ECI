# Sync audit, data generation 20261118

Reference: 0_input/ at `main`. Produced by `1_curated/0_audit_sync.py`; the procedure is `.claude/skills/data-refresh/SKILL.md`.

| level | findings |
|---|---|
| BLOCK | 0 |
| DECIDE | 0 |
| CHECK | 4 |
| INFO | 2 |

## CHECK

**openness to re-check**
- `mistral-large-4-0`: API public preview on 2026-10-06; the weights (mistralai/Mistral-Large-4-1T-A52B, private on 2026-10-20) are public on Hugging Face on 2026-11-18 and Epoch dates the open release 2026-10-29, inside the 30-day window that closed on 2026-11-05. Re-checked 2026-11-18: open
- `muse-ember-1.0`: Meta AI, released 2026-11-09; Epoch: API access, no weights announced on 2026-11-18. Inside its 30-day window (closes 2026-12-09): re-check
- `muse-ember-1.0_max`: same release as muse-ember-1.0: re-check after 2026-12-09
- `doubao-seed-evolving_high`: ByteDance, released 2026-11-02; Epoch: API access, no weights announced on 2026-11-18. Inside its 30-day window (closes 2026-12-02): re-check


## INFO

**provenance**
- `20261118`: pipeline commit 3a654fed3, run 20261118

**scores**
- `5975`: +19 (model, benchmark) cells; 5956 carried over, 0 revised by more than 0.05
