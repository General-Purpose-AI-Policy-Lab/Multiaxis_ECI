# Skilled Generalist contradiction & human axis-3 lift — tables only

Converged fit: K=3, human+lineage priors, free rotation, Skilled Generalist scores removed.
Promax (oblique) frame. "capability demanded" = logit(score) + difficulty D.

---

## 1 · Skilled Generalist vs Committee of Average Humans

Both score **0.98 on ARC-AGI** (the shared row).

### Skilled Generalist — 10 observations

| benchmark | score | difficulty D | general loading | capability demanded |
|---|---|---|---|---|
| MATH Level 5 | 0.40 | −1.09 | 1.81 | −1.5 |
| ARC-AGI | 0.98 | 2.28 | 1.62 | +6.2 |
| GPQA Diamond | 0.22 | −0.31 | 0.74 | −1.6 |
| GPQA Diamond Chemistry | 0.22 | −0.07 | 0.72 | −1.3 |
| GPQA Main Chemistry | 0.31 | 0.12 | 0.69 | −0.7 |
| GPQA Main Biology | 0.43 | −0.54 | 0.52 | −0.8 |
| GPQA Diamond Biology | 0.22 | −0.41 | 0.36 | −1.7 |
| OS World (Screenshot) | 0.72 | 0.50 | 0.31 | +1.5 |
| VISTA | 0.55 | 0.79 | 0.21 | +1.0 |
| GSM8K | 0.97 | −0.02 | −0.78 | +3.4 |

capability demanded spread: **−1.7 → +6.2 (7.9)**

### Committee of Average Humans — 4 observations

| benchmark | score | difficulty D | general loading | capability demanded |
|---|---|---|---|---|
| ARC-AGI | 0.98 | 2.28 | 1.62 | +6.2 |
| CSQA2 | 0.94 | 0.11 | −0.40 | +2.9 |
| WinoGrande | 0.94 | −1.11 | −0.19 | +1.6 |
| HellaSwag | 0.96 | −1.34 | −0.33 | +1.7 |

capability demanded spread: **+1.6 → +6.2 (4.5)**

---

## 2 · Skilled Generalist provenance — one label, many populations

GPQA chance floor = 0.25 (4-choice).

| benchmark | score | source | population tested | note |
|---|---|---|---|---|
| ARC-AGI | 0.98 | ARC Prize | STEM Graduates | |
| GPQA Diamond | 0.219 | GPQA paper | non-experts pursuing PhDs in **other** domains | **below chance** |
| GPQA Diamond Biology | 0.22 | RAND | not in-domain PhD | **below chance** |
| GPQA Diamond Chemistry | 0.22 | RAND | not in-domain PhD | **below chance** |
| GPQA Main Biology | 0.43 | RAND | not in-domain PhD | RAND Extended-as-Main error |
| GPQA Main Chemistry | 0.31 | RAND | not in-domain PhD | RAND Extended-as-Main error |
| GSM8K | 0.97 | ACL 2024 | vetted annotators, bachelor's+, passed qualification exam | |
| MATH Level 5 | 0.40 | MATH paper | "a CS PhD student who does not especially like mathematics" | n=1 |
| OS World (Screenshot) | 0.72 | arXiv 2404.07972 | individuals not familiar with the software | |
| VISTA | 0.55 | Scale | 16 full-time employees | |

---

## 3 · What benchmarks lift humans on axis 3 (applied/agentic ⟷ formal-math)

lift = (score − 0.5) × axis-3 loading. Axis-3 top loaders GSO/Cybench/Aider/OS World/SWE were **taken by no human**.

### Average Human — net axis-3 lift +1.49

| benchmark | score | axis-3 loading | lift |
|---|---|---|---|
| OpenBookQA | 0.92 | +0.59 | +0.25 |
| CSQA2 | 0.90 | +0.60 | +0.24 |
| Adversarial NLI | 0.85 | +0.55 | +0.19 |
| SimpleBench | 0.84 | +0.47 | +0.16 |
| ScienceQA | 0.88 | +0.35 | +0.14 |
| SuperGLUE | 0.90 | +0.34 | +0.14 |
| BoolQ | 0.90 | +0.28 | +0.11 |
| ARC-AGI | 0.77 | +0.37 | +0.10 |
| TriviaQA | 0.80 | +0.25 | +0.07 |
| ARC-AGI-2 | 0.60 | +0.57 | +0.06 |
| BIG-Bench Hard (BBH) | 0.68 | +0.23 | +0.04 |
| MMLU | 0.34 | +0.06 | −0.01 |

### Committee of Average Humans — net +0.81

| benchmark | score | axis-3 loading | lift |
|---|---|---|---|
| CSQA2 | 0.94 | +0.60 | +0.26 |
| HellaSwag | 0.96 | +0.53 | +0.24 |
| ARC-AGI | 0.98 | +0.37 | +0.18 |
| WinoGrande | 0.94 | +0.29 | +0.13 |

### Domain Expert — net +0.03

| benchmark | score | axis-3 loading | lift |
|---|---|---|---|
| MMLU Biology | 0.90 | +0.09 | +0.04 |
| GPQA Diamond Chemistry | 0.83 | +0.10 | +0.03 |
| LAB-Bench LitQA2 | 0.70 | +0.13 | +0.03 |
| MMLU | 0.90 | +0.06 | +0.03 |
| WMDP Chemistry | 0.43 | −0.27 | +0.02 |
| WMDP Biology | 0.60 | −0.23 | −0.02 |
| MATH Level 5 | 0.90 | −0.37 | −0.15 |

### Top Performer — net +0.01

| benchmark | score | axis-3 loading | lift |
|---|---|---|---|
| BIG-Bench Hard (BBH) | 0.94 | +0.23 | +0.10 |
| MMLU Biology | 0.90 | +0.09 | +0.04 |
| MMLU Chemistry | 0.90 | +0.05 | +0.02 |
| MATH Level 5 | 0.90 | −0.37 | −0.15 |

### Committee of Skilled Generalists — net +0.08

| benchmark | score | axis-3 loading | lift |
|---|---|---|---|
| PIQA | 0.95 | +0.19 | +0.08 |
