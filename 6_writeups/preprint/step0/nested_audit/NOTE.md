# Step 0.2: do nested benchmarks share their luck?

Date 2026-11-03. Data generation 20261019. Code at origin/main 4926945.

## Result

**Shared questions do not, on their own, make two benchmarks move together. Two other sources do: the RAND evaluation campaign, and two same-harness tier pairs.**

- **Shared questions alone show no excess.** A RAND subset and the Epoch/Kaggle column that contains its questions (GPQA Diamond, MMLU, MMLU-Pro) have a mean residual correlation of −0.057 over 30 pairs in K=1. The mean over all pairs is −0.057, so there is no excess. Example: GPQA Diamond contains every question of the RAND GPQA Diamond Chemistry rows, and their residual correlation is 0.17 (77th percentile). The two scores come from separate runs, so luck on a shared question is drawn again in each run; only question difficulty is shared, and the model's difficulty parameter carries it.
- **The RAND rows share an excess among themselves.** In K=1, the 105 RAND–RAND pairs have a mean residual correlation of +0.044, against −0.057 for all pairs: about +0.10. Only 4 / 105 are above the 95th percentile, close to the 5 expected by chance. So this is a small, broad excess, not a few extreme pairs. The likely cause is one evaluation campaign: the same harness, the same runs and the same digitising. Content cannot explain it, because the same content from another source shows no excess.
- **The K=4 axes absorb the RAND excess.** In each K=4 mode, the RAND–RAND mean falls to 0.005 to 0.013. In mode A, 6 of the 10 benchmarks that axis 1 owns most are RAND rows: MMLU Chem, GPQA Main Chem, MMLU Bio, WMDP Chem, MMLU Pro Chem and GPQA Main Bio. In modes B and C, RAND rows are 3 or 4 of 10 on two axes. This is the referee's concern in a weak form: part of axis 1 in mode A may be "RAND campaign" rather than "science knowledge". The data here cannot say how large that part is.
- **Two family pairs stand out in every fit.**
  - FrontierMath Tiers 1-3 v2 and Tier 4 v2: residual correlation 0.43 to 0.63, at the 96th to 99th percentile, over 65 to 69 models.
  - AISI CTF Practitioner and Expert: 0.60 to 0.67, at the 98th to 99.8th percentile, but over only 10 models.

  K=4 does not absorb either pair: no axis is built from them. Each family has at most 1 benchmark among any axis's top 10.
- **5 / 13 shared-question pairs cannot be checked.** They have 0 to 3 models in common: Cybench and CAISI Cybench, the two CyScenarioBench series, the two CAISI CTF-Archive series, and MMLU-Pro and its Bio/Chem subsets. With no common models they cannot share luck across models either, so they cannot build an axis this way.

## What was measured

- **Residual.** A score minus its fitted value (posterior mean over 500 thinned draws).
- **Residual correlation (Yen's Q3).** The Pearson correlation of two benchmarks' residuals over the models that took both. A positive value means the two benchmarks are lucky or unlucky together, beyond what ability explains.
- **Baseline.** Every benchmark pair with at least 10 common models: 570 pairs in K=1, 860 in K=4. The baselines have mean −0.057 (K=1) and −0.047 (K=4), and 95th percentiles 0.47 (K=1) and 0.39 to 0.40 (K=4). A pair counts as dependent only above its fit's 95th percentile. By construction, 5% of ordinary pairs sit there.
- **Fits.**
  - K=1: the canonical K=1 index, 8 chains. Humans are dropped from the residuals. Its scope excludes ARC-AGI and some others.
  - K=4: the three saved modes of the 5-chain flagship refit: A (2 chains), B (1) and C (2). The 5-chain fit did not converge as a whole, so the K=4 numbers are an indication only.
- **Pairs.** 40 listed pairs. 13 are class A (shared questions) and 27 are class B (same family, other questions). 24 or 25 have at least 10 common models.

## Pairs above the 95th percentile (at least 10 common models)

| pair | class | common models | K=1 | K=4 A | K=4 B | K=4 C |
|---|---|---|---|---|---|---|
| FrontierMath T1-3 v2 / T4 v2 | B | 65 to 69 | 0.63 (99) | 0.44 (97) | 0.43 (96) | 0.46 (97) |
| AISI CTF Practitioner / Expert | B | 10 | 0.60 (98) | 0.67 (99.8) | 0.65 (99.5) | 0.63 (99.3) |
| GPQA Main Chem / Diamond Chem (both RAND) | A | 56 | 0.54 (97) | 0.37 (94) | 0.40 (94.8) | 0.44 (97) |
| AISI CTF Technical Non-Expert / Apprentice | B | 17 | 0.53 (97) | 0.27 (88) | 0.25 (87) | 0.30 (90) |
| MMLU / MMLU-Pro | B | 55 | 0.48 (95) | 0.23 (86) | 0.30 (91) | 0.34 (93) |
| MMLU Pro Bio / Chem (RAND) | B | 42 | 0.49 (96) | 0.06 (67) | 0.14 (78) | 0.12 (75) |
| WMDP Bio / Chem (RAND) | B | 38 | 0.48 (95) | −0.08 (44) | −0.14 (36) | −0.16 (31) |

Values are residual correlations, with the percentile in brackets. In K=1, 7 / 24 listed pairs are above the 95th percentile. In K=4 the count is 2 / 25 for modes A and B and 3 / 25 for mode C.

## Decision (Yassine, 2026-11-03)

Keep every benchmark in R1 and R2. Report the RAND same-source effect and the FrontierMath tier pair in the limitations. Add a sensitivity fit to the run list: the flagship spec without the 15 RAND benchmarks, to show whether axis 1 depends on them.

## Options considered

No option was recommended.

1. **Keep every benchmark.**
   - For: no information is lost.
   - Against: the RAND campaign effect stays in the data. In mode A it can be read as part of axis 1.
2. **Drop the RAND rows where an Epoch or Kaggle column holds the same questions** (GPQA Main/Diamond Bio/Chem, MMLU Bio/Chem, MMLU Pro Bio/Chem).
   - For: removes the double count of the same questions and part of the campaign effect.
   - Against: the shared questions themselves showed no excess. Bio/chem coverage thins, and WMDP and LAB-Bench stay.
3. **Drop every RAND row**, 15 benchmarks.
   - For: removes the campaign effect completely.
   - Against: this is the main bio/chem coverage. Axis 1 in mode A would lose most of what it owns. That is a model change, and it needs a refit.
4. **Keep the benchmarks and add a source effect** (one shared random effect per evaluation campaign, a "testlet" effect).
   - For: keeps the data and absorbs the campaign effect explicitly, so an axis cannot take it.
   - Against: a new model component. It needs a refit and a prior choice.
5. **Merge each same-harness tier pair** (FrontierMath T1-3 with T4; the AISI CTF tiers) into one column.
   - For: removes the two strongest pairs.
   - Against: those pairs do not build any axis now. Merging also mixes difficulties.
6. **Report only.** Keep the data, and state in the paper's limitations the RAND effect (+0.10 mean residual correlation, absorbed by K=4) and the two tier pairs.
   - For: no refit needed.
   - Against: a referee may still ask for option 3 or 4 as a sensitivity run.

## Limits

- The K=4 trace is the 5-chain refit (not converged as a whole), not the 50-chain flagship on origin. That trace is not on disk here.
- Many pairs have 10 to 20 common models. At n = 10, the 95% interval of a correlation of 0.6 runs from about 0 to 0.9.
- Residuals use the posterior mean fitted score, not per-draw residuals.
- "Same campaign" is inferred. Content and source are separated only for the bio/chem rows, which have a cross-source comparison.

## Artifacts

All in `6_writeups/preprint/step0/nested_audit/`:
- `audit.py`: the script, which runs in about 20 seconds.
- `pairs.csv`: one row per listed pair. Columns: common models, Spearman score correlation, residual correlation and percentile per fit, and dominant axis per K=4 mode.
- `families.csv`: family-level means, and the RAND against same-content comparison.
- `axis_top10_families.csv`: per K=4 mode and axis, the 10 benchmarks the axis owns most (by axis share) and how many come from each family.
- `baselines.json`: the baseline distribution per fit.
