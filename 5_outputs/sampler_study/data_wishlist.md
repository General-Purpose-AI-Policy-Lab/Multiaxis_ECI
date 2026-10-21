# Wishlist : mesures pour départager les modes du flagship

Générée par `data_wishlist.py` à partir du flagship du 19 octobre (K = 4, sans humains, capacités skew-normal, plafonds estimés), 20 chaînes (graines 42 et 43) en 4 modes :

- **A** : 9 chaînes (1, 2, 4, 6, 8, 10, 11, 13, 17), la meilleure log-densité ; axes : *Connaissances et raisonnements scientifiques* ; *Capacités agentiques & Programmation* ; *Capacités cyber* ; *Sens commun & connaissances factuelles*
- **B** : 2 chaînes (12, 14), 16,0 nats sous A ; axes : ARC AI2 / OpenBookQA ; GeoBench / SpireClimb ; OSWorld / GDPval ; OTIS Mock AIME 2024-2025 / AudioMultiChallenge
- **C** : 7 chaînes (0, 5, 7, 15, 16, 18, 19), 27,0 nats sous A ; axes : APEX-Agents / DeepResearch Bench ; CyberGym / CyScenarioBench ; OTIS Mock AIME 2024-2025 / MATH Level 5 ; SimpleQA Verified / GSM8K
- **D** : 2 chaînes (3, 9), 33,2 nats sous A ; axes : CritPt / OTIS Mock AIME 2024-2025 ; DeepResearch Bench / GSM8K ; GDPval / Furniture Assembly ; GeoBench / SpireClimb

Les livrables lisent A. PSIS-LOO ne classe pas les modes et la LOO intégrée ne séparait pas A de D sur les 10 premières chaînes (+3 ± 8 nats pour A) : seules de nouvelles mesures peuvent les départager.

**Comment lire.** Pour chaque case (modèle, benchmark) sans score, chaque mode donne une prédiction : sa moyenne et son écart-type, qui ajoute à l'incertitude du mode le bruit propre au benchmark. Pour deux modes, l'utilité *z* est l'écart entre leurs moyennes en écarts-types prédictifs ; une mesure tombée sur la prédiction d'un mode multiplie à peu près ses chances contre l'autre par e^(z²/2) (z = 1,5 : × 3 ; z = 2 : × 7). Une case vaut sa meilleure paire, indiquée à côté (`A-C` : la case départage A et C) ; le csv donne aussi `z_A`, sa meilleure paire avec A. Les benchmarks sont classés par leur meilleure case ; pour chacun, les modèles par ordre d'utilité : si un benchmark n'est pas accessible, prendre le suivant ; si un modèle ne l'est pas, le suivant sur la même ligne. Les axes de B, C et D, qui n'ont pas de noms, sont désignés par leurs deux benchmarks les plus colinéaires. La disponibilité en API n'a pas été vérifiée ; `open` = poids ouverts. 89 422 cases non observées ; `data_wishlist.csv` garde les dix meilleurs modèles de chaque benchmark, parmi tous les modèles (`question = benchmark`), parmi ceux d'Anthropic (`anthropic`) et parmi les poids ouverts (`open`).

**Paires départagées par les 2 000 meilleures cases** : A-D 790, C-D 410, B-D 371, B-C 328, A-C 72, A-B 29.

**Prudence.** B et D ne reposent que sur 2 chaînes chacun.

## En bref (lecture du 21 octobre, écrite à la main)

1. **La plupart des meilleures cases opposent deux modes que les livrables ne lisent pas** (B-D, C-D, B-C : 1 109 des 2 000 meilleures). Pour les livrables, ce qui compte est de départager A de chacun des autres : voir la section « Départager A ».
2. **A contre C**, le rival le plus fréquent (7 chaînes, 27 nats plus bas), est le plus difficile à départager : les deux modes ont les mêmes axes agentique et cyber et ne diffèrent que par la frontière entre connaissances et raisonnement. Les meilleures cases sont à z ≈ 2 : gpt-5.4-nano (sans raisonnement) sur ARC AI2 (A 0,79 contre C 0,91) et OpenBookQA (A 0,72 contre C 0,87) ; Phi-3-small sur Lech Mazur Writing ; falcon-180B sur CyberGym.
3. **A contre D** (l'ancienne structure, 33 nats plus bas) se départage comme avant : petits modèles d'OpenAI et petits modèles ouverts sur les QCM scientifiques (gpt-5.4-nano sur GPQA Main Chimie, A 0,67 contre D 0,46 ; Qwen2.5-Coder 1,5B sur MATH Level 5, A 0,60 contre D 0,16) et sur le sens commun dans l'autre sens (gpt-5-nano sur OpenBookQA, A 0,67 contre D 0,88).
4. **A contre B** est le moins séparable (meilleure case z = 1,8 : gpt-6-sol sans raisonnement sur GPQA Main Chimie, A 0,78 contre B 0,62).
5. **Un petit lot teste A contre les trois** : gpt-5.4-nano à plusieurs efforts sur ARC AI2, OpenBookQA et GPQA Main Chimie. Côté Anthropic, Claude Opus 4.1 thinking sur GPQA Main Chimie (A 0,63 contre B 0,74 et D 0,75) départage A de B et de D ; les Claude récents sur ARC AI2 et OpenBookQA séparent surtout C de D.

## Benchmarks classés

| rang | benchmark | meilleur z (paire) | axe dans A | modèles par ordre d'utilité (z, paire ; prédictions des modes) |
|---|---|---|---|---|
| 1 | OpenBookQA | 3.3 (B-D) | Sens commun & connaissances factuelles | gpt-5-nano-2025-08-07_high 3.3 B-D [A 0.67 B 0.62 C 0.74 D 0.88] ; gpt-4.1-nano-2025-04-14 3.3 B-D [A 0.62 B 0.56 C 0.72 D 0.82] ; gpt-5-nano-2025-08-07_medium 3.0 B-D [A 0.69 B 0.64 C 0.75 D 0.87] ; gpt-5-nano-2025-08-07_low 2.9 B-D [A 0.66 B 0.62 C 0.76 D 0.85] ; gpt-5-nano-2025-08-07 2.8 B-D [A 0.71 B 0.65 C 0.79 D 0.87] |
| 2 | ARC AI2 | 3.2 (B-D) | Sens commun & connaissances factuelles | gpt-4.1-nano-2025-04-14 3.2 B-D [A 0.69 B 0.61 C 0.77 D 0.84] ; gpt-5-nano-2025-08-07_high 3.1 B-D [A 0.76 B 0.68 C 0.80 D 0.90] ; gpt-5-nano-2025-08-07_medium 2.9 B-D [A 0.77 B 0.70 C 0.81 D 0.90] ; gpt-5-nano-2025-08-07_low 2.8 B-D [A 0.74 B 0.68 C 0.81 D 0.87] ; gpt-5-nano-2025-08-07 2.7 B-D [A 0.79 B 0.72 C 0.84 D 0.90] |
| 3 | GeoBench | 2.6 (B-C) | Capacités cyber | Phi-3-small-8k-instruct (open) 2.6 B-C [A 0.31 B 0.18 C 0.58 D 0.51] ; falcon-180B (open) 2.5 B-C [A 0.56 B 0.75 C 0.34 D 0.49] ; phi-2 (open) 2.5 B-C [A 0.20 B 0.07 C 0.47 D 0.22] ; Qwen2.5-72B (open) 2.3 B-C [A 0.47 B 0.31 C 0.67 D 0.56] ; qwen2.5-7b-instruct (open) 2.2 B-C [A 0.33 B 0.20 C 0.56 D 0.46] |
| 4 | GPQA Main Chemistry | 2.4 (A-D) | Connaissances et raisonnements scientifiques | gpt-5.4-nano-2026-03-17_low 2.4 A-D [A 0.67 B 0.49 C 0.64 D 0.46] ; gpt-5.4-nano-2026-03-17_medium 2.3 A-D [A 0.72 B 0.55 C 0.70 D 0.52] ; gpt-5-nano-2025-08-07_high 2.3 A-D [A 0.64 B 0.49 C 0.61 D 0.44] ; gpt-5.4-nano-2026-03-17_high 2.3 A-D [A 0.75 B 0.58 C 0.72 D 0.54] ; gpt-5.4-nano-2026-03-17_none 2.3 A-D [A 0.54 B 0.39 C 0.51 D 0.36] |
| 5 | ARC-AGI | 2.3 (B-C) | Connaissances et raisonnements scientifiques | qwen2.5-72b-instruct (open) 2.3 B-C [A 0.03 B 0.04 C 0.01 D 0.03] ; qwen2-72b-instruct (open) 1.8 B-C [A 0.01 B 0.02 C 0.00 D 0.01] ; gemma-3-27b-it (open) 1.4 A-B [A 0.01 B 0.03 C 0.01 D 0.02] ; DeepSeek-V3 (open) 1.4 B-C [A 0.04 B 0.04 C 0.02 D 0.03] ; phi-4 (open) 1.4 B-C [A 0.03 B 0.05 C 0.02 D 0.04] |
| 6 | DeepResearch Bench | 2.2 (B-C) | Sens commun & connaissances factuelles | falcon-180B (open) 2.2 B-C [A 0.30 B 0.26 C 0.39 D 0.31] ; Phi-3-small-8k-instruct (open) 2.1 B-C [A 0.35 B 0.36 C 0.23 D 0.29] ; Baichuan-2-13B-Base (open) 2.1 B-D [A 0.21 B 0.19 C 0.28 D 0.34] ; DeepSeek-Coder-V2-Base (open) 2.1 B-D [A 0.28 B 0.26 C 0.36 D 0.41] ; Qwen2.5-Coder-14B (open) 2.0 B-D [A 0.26 B 0.25 C 0.31 D 0.38] |
| 7 | MATH Level 5 | 2.2 (B-D) | Connaissances et raisonnements scientifiques | Baichuan-2-13B-Base (open) 2.2 B-D [A 0.79 B 0.83 C 0.75 D 0.32] ; Qwen2.5-Coder-1.5B (open) 2.2 A-D [A 0.60 B 0.61 C 0.53 D 0.16] ; text-davinci-002 2.0 A-D [A 0.26 B 0.31 C 0.43 D 0.72] ; Qwen2.5-Coder-14B (open) 2.0 A-D [A 0.78 B 0.78 C 0.67 D 0.35] ; Qwen2.5-Coder-7B (open) 1.9 B-D [A 0.66 B 0.68 C 0.55 D 0.25] |
| 8 | CyberGym | 2.2 (C-D) | Capacités cyber | Phi-3-small-8k-instruct (open) 2.2 C-D [A 0.05 B 0.08 C 0.40 D 0.01] ; Qwen2.5-72B (open) 1.9 C-D [A 0.06 B 0.03 C 0.26 D 0.01] ; falcon-180B (open) 1.9 C-D [A 0.47 B 0.30 C 0.12 D 0.44] ; DeepSeek-Coder-V2-Base (open) 1.8 A-C [A 0.48 B 0.27 C 0.10 D 0.30] ; qwen2.5-72b-instruct (open) 1.8 C-D [A 0.06 B 0.03 C 0.22 D 0.02] |
| 9 | WMDP Chemistry | 2.1 (A-D) | Connaissances et raisonnements scientifiques | Qwen2.5-Coder-7B (open) 2.1 A-D [A 0.63 B 0.61 C 0.57 D 0.49] ; Baichuan-2-13B-Base (open) 2.0 A-D [A 0.57 B 0.54 C 0.50 D 0.40] ; gpt-4.1-nano-2025-04-14 2.0 A-D [A 0.66 B 0.57 C 0.62 D 0.53] ; Qwen2.5-Coder-14B (open) 2.0 A-D [A 0.67 B 0.65 C 0.62 D 0.54] ; Qwen2.5-Coder-32B (open) 1.9 A-D [A 0.69 B 0.68 C 0.65 D 0.57] |
| 10 | Lech Mazur Writing | 2.1 (A-C) | Capacités cyber | Phi-3-small-8k-instruct (open) 2.1 A-C [A 0.40 B 0.56 C 0.73 D 0.49] ; Llama-2-70b-chat (open) 2.0 C-D [A 0.63 B 0.61 C 0.45 D 0.68] ; Qwen2.5-72B (open) 2.0 A-C [A 0.53 B 0.62 C 0.76 D 0.61] ; qwen2.5-72b-instruct (open) 1.9 A-C [A 0.60 B 0.65 C 0.76 D 0.66] ; Baichuan-2-13B-Base (open) 1.8 C-D [A 0.55 B 0.58 C 0.33 D 0.61] |
| 11 | OTIS Mock AIME 2024-2025 | 2.1 (B-D) | Connaissances et raisonnements scientifiques | Baichuan-2-13B-Base (open) 2.1 B-D [A 0.42 B 0.53 C 0.44 D 0.09] ; Qwen2.5-Coder-14B (open) 1.8 B-D [A 0.39 B 0.43 C 0.32 D 0.09] ; text-davinci-002 1.7 A-D [A 0.05 B 0.07 C 0.12 D 0.40] ; Qwen2.5-Coder-32B (open) 1.6 B-D [A 0.43 B 0.48 C 0.38 D 0.14] ; Baichuan-2-7B-Base (open) 1.6 B-D [A 0.43 B 0.59 C 0.51 D 0.19] |
| 12 | HellaSwag | 2.1 (C-D) | Sens commun & connaissances factuelles | gpt-5-nano-2025-08-07_high 2.1 C-D [A 0.80 B 0.82 C 0.78 D 0.86] ; gpt-4.1-nano-2025-04-14 2.0 C-D [A 0.79 B 0.81 C 0.76 D 0.84] ; gpt-5-nano-2025-08-07_minimal 1.9 C-D [A 0.82 B 0.83 C 0.79 D 0.86] ; gpt-5-nano-2025-08-07_low 1.9 C-D [A 0.81 B 0.82 C 0.79 D 0.86] ; gpt-5-nano-2025-08-07_medium 1.8 C-D [A 0.81 B 0.83 C 0.80 D 0.86] |
| 13 | MMLU Pro Biology | 2.0 (A-D) | Connaissances et raisonnements scientifiques | Qwen2.5-Coder-1.5B (open) 2.0 A-D [A 0.75 B 0.75 C 0.68 D 0.51] ; Baichuan-2-13B-Base (open) 2.0 B-D [A 0.84 B 0.84 C 0.76 D 0.61] ; Qwen2.5-Coder-7B (open) 1.9 A-D [A 0.84 B 0.83 C 0.77 D 0.66] ; Qwen2.5-Coder-14B (open) 1.8 A-D [A 0.87 B 0.87 C 0.82 D 0.74] ; CodeQwen1.5-7B (open) 1.7 B-D [A 0.63 B 0.63 C 0.55 D 0.39] |
| 14 | MMLU Pro Chemistry | 1.9 (A-D) | Connaissances et raisonnements scientifiques | Baichuan-2-13B-Base (open) 1.9 A-D [A 0.69 B 0.67 C 0.59 D 0.40] ; Qwen2.5-Coder-7B (open) 1.9 A-D [A 0.71 B 0.68 C 0.61 D 0.46] ; Qwen2.5-Coder-14B (open) 1.8 A-D [A 0.78 B 0.76 C 0.70 D 0.57] ; Qwen2.5-Coder-1.5B (open) 1.8 A-D [A 0.58 B 0.56 C 0.50 D 0.33] ; gpt-4.1-nano-2025-04-14 1.6 A-D [A 0.75 B 0.65 C 0.70 D 0.60] |
| 15 | TriviaQA | 1.9 (B-C) | Capacités cyber | gpt-4.1-nano-2025-04-14 1.9 B-C [A 0.78 B 0.80 C 0.67 D 0.72] ; gpt-5-nano-2025-08-07_minimal 1.7 B-C [A 0.81 B 0.81 C 0.71 D 0.77] ; gemma-3-27b-it (open) 1.7 A-C [A 0.85 B 0.83 C 0.73 D 0.79] ; Inkling_xhigh (open) 1.6 C-D [A 0.80 B 0.86 C 0.90 D 0.82] ; Inkling_high (open) 1.6 A-C [A 0.79 B 0.85 C 0.89 D 0.81] |

## Départager A de chacun des autres modes

La meilleure case de chaque benchmark pour chaque paire avec A.

### A contre B

| benchmark | modèle | z (A-B) | A | B |
|---|---|---|---|---|
| GPQA Main Chemistry | gpt-6-sol_none | 1.8 | 0.78 | 0.62 |
| ARC-AGI | gemma-3-27b-it | 1.4 | 0.01 | 0.03 |
| WMDP Chemistry | gpt-6-sol_none | 1.4 | 0.80 | 0.74 |
| CyberGym | gemma-3-27b-it | 1.4 | 0.39 | 0.11 |
| Cybench | gemma-3-27b-it | 1.4 | 0.26 | 0.09 |
| GPQA Diamond Chemistry | gpt-5.4-nano-2026-03-17_low | 1.3 | 0.64 | 0.53 |
| Terminal Bench 2.0 | o4-mini-2025-04-16_medium | 1.3 | 0.28 | 0.36 |
| GPQA Main Biology | gpt-5.4-nano-2026-03-17_none | 1.3 | 0.65 | 0.54 |

### A contre C

| benchmark | modèle | z (A-C) | A | C |
|---|---|---|---|---|
| Lech Mazur Writing | Phi-3-small-8k-instruct | 2.1 | 0.40 | 0.73 |
| ARC AI2 | gpt-5.4-nano-2026-03-17_none | 2.0 | 0.79 | 0.91 |
| OpenBookQA | gpt-5.4-nano-2026-03-17_none | 1.9 | 0.72 | 0.87 |
| CyberGym | falcon-180B | 1.8 | 0.47 | 0.12 |
| ARC-AGI | qwen2.5-72b-instruct | 1.7 | 0.03 | 0.01 |
| TriviaQA | gemma-3-27b-it | 1.7 | 0.85 | 0.73 |
| DeepResearch Bench | Phi-3-small-8k-instruct | 1.7 | 0.35 | 0.23 |
| MMLU Pro Biology | Llama-2-70b-chat | 1.5 | 0.62 | 0.52 |

### A contre D

| benchmark | modèle | z (A-D) | A | D |
|---|---|---|---|---|
| OpenBookQA | gpt-5-nano-2025-08-07_high | 2.6 | 0.67 | 0.88 |
| GPQA Main Chemistry | gpt-5.4-nano-2026-03-17_low | 2.4 | 0.67 | 0.46 |
| ARC AI2 | gpt-5-nano-2025-08-07_high | 2.4 | 0.76 | 0.90 |
| MATH Level 5 | Qwen2.5-Coder-1.5B | 2.2 | 0.60 | 0.16 |
| WMDP Chemistry | Qwen2.5-Coder-7B | 2.1 | 0.63 | 0.49 |
| MMLU Pro Biology | Qwen2.5-Coder-1.5B | 2.0 | 0.75 | 0.51 |
| MMLU Pro Chemistry | Baichuan-2-13B-Base | 1.9 | 0.69 | 0.40 |
| DeepResearch Bench | Qwen2.5-Coder-14B | 1.8 | 0.26 | 0.38 |

## Ce que nous pouvons mesurer nous-mêmes

### Modèles Anthropic (comme la campagne de mesures propres, protocole RAND)

| rang | benchmark | meilleur z (paire) | modèles par ordre d'utilité (z, paire ; prédictions des modes) |
|---|---|---|---|
| 1 | ARC AI2 | 2.3 (B-D) | claude-3-5-haiku-20241022 2.3 B-D [A 0.89 B 0.91 C 0.86 D 0.81] ; claude-haiku-5-5_none 2.3 C-D [A 0.89 B 0.93 C 0.94 D 0.82] ; claude-sonnet-5_none 2.0 C-D [A 0.94 B 0.96 C 0.96 D 0.91] ; claude-sonnet-4-5-20250929_none 2.0 B-D [A 0.95 B 0.96 C 0.94 D 0.92] ; claude-sonnet-4-20250514_thinking 2.0 B-C [A 0.95 B 0.96 C 0.90 D 0.94] |
| 2 | OpenBookQA | 2.2 (C-D) | claude-haiku-5-5_none 2.2 C-D [A 0.85 B 0.90 C 0.90 D 0.74] ; claude-3-5-haiku-20241022 2.0 B-D [A 0.84 B 0.87 C 0.82 D 0.75] ; claude-sonnet-5_none 1.7 C-D [A 0.91 B 0.92 C 0.93 D 0.85] ; claude-haiku-4-5-20251001_none 1.6 B-D [A 0.85 B 0.88 C 0.86 D 0.78] ; claude-sonnet-4-5-20250929_none 1.5 B-D [A 0.92 B 0.93 C 0.90 D 0.86] |
| 3 | GPQA Main Chemistry | 2.0 (A-D) | claude-opus-4-1-20250805_thinking 2.0 A-D [A 0.63 B 0.74 C 0.65 D 0.75] ; claude-opus-4-20250514_thinking 1.9 A-D [A 0.62 B 0.71 C 0.63 D 0.72] ; claude-opus-4-1-20250805_none 1.9 A-D [A 0.52 B 0.60 C 0.53 D 0.62] ; claude-opus-4-1-20250805_16K 1.9 A-D [A 0.60 B 0.70 C 0.62 D 0.71] ; claude-opus-4-20250514_16K 1.9 A-D [A 0.60 B 0.68 C 0.60 D 0.69] |
| 4 | TriviaQA | 1.5 (B-C) | claude-3-haiku-20240307 1.5 B-C [A 0.75 B 0.74 C 0.82 D 0.79] ; claude-3-5-haiku-20241022 1.4 B-D [A 0.82 B 0.79 C 0.85 D 0.85] ; claude-haiku-5-5_none 1.3 B-D [A 0.89 B 0.83 C 0.88 D 0.90] ; claude-haiku-4-5-20251001_none 1.3 B-D [A 0.85 B 0.82 C 0.86 D 0.87] ; claude-sonnet-4-20250514_none 1.2 B-C [A 0.86 B 0.84 C 0.89 D 0.88] |
| 5 | HellaSwag | 1.3 (B-C) | claude-sonnet-4-20250514_none 1.3 B-C [A 0.90 B 0.89 C 0.92 D 0.90] ; claude-sonnet-4-5-20250929_none 1.3 B-C [A 0.91 B 0.90 C 0.93 D 0.91] ; claude-3-haiku-20240307 1.3 C-D [A 0.83 B 0.82 C 0.85 D 0.80] ; claude-sonnet-4-20250514_16K 1.3 B-C [A 0.90 B 0.90 C 0.92 D 0.90] ; claude-sonnet-4-20250514_32K 1.3 B-C [A 0.89 B 0.88 C 0.91 D 0.89] |
| 6 | GPQA Diamond Chemistry | 1.3 (A-D) | claude-opus-4-1-20250805_thinking 1.3 A-D [A 0.69 B 0.75 C 0.70 D 0.75] ; claude-opus-4-20250514_thinking 1.2 A-D [A 0.67 B 0.72 C 0.68 D 0.73] ; claude-opus-4-1-20250805_none 1.2 A-D [A 0.57 B 0.62 C 0.58 D 0.63] ; claude-sonnet-4-5-20250929_thinking 1.2 A-D [A 0.71 B 0.76 C 0.71 D 0.76] ; claude-opus-4-20250514_none 1.2 A-D [A 0.56 B 0.60 C 0.56 D 0.61] |
| 7 | WMDP Chemistry | 1.3 (A-D) | claude-opus-4-1-20250805_none 1.3 A-D [A 0.73 B 0.75 C 0.74 D 0.77] ; claude-sonnet-4-5-20250929_none 1.3 A-D [A 0.71 B 0.74 C 0.72 D 0.75] ; claude-opus-4-1-20250805_thinking 1.2 A-D [A 0.77 B 0.80 C 0.78 D 0.80] ; claude-sonnet-4-5-20250929_thinking 1.2 A-D [A 0.76 B 0.79 C 0.77 D 0.79] ; claude-opus-4-20250514_none 1.1 A-D [A 0.73 B 0.76 C 0.74 D 0.77] |
| 8 | GDPval | 1.3 (A-D) | claude-3-5-sonnet-20241022 1.3 A-D [A 0.24 B 0.29 C 0.27 D 0.34] ; claude-3-7-sonnet-20250219_none 1.3 A-D [A 0.27 B 0.33 C 0.29 D 0.36] ; claude-3-5-haiku-20241022 1.0 A-D [A 0.16 B 0.24 C 0.19 D 0.25] ; claude-3-7-sonnet-20250219_1K 0.9 A-D [A 0.25 B 0.30 C 0.27 D 0.32] ; claude-2.0 0.8 B-D [A 0.12 B 0.09 C 0.09 D 0.17] |
| 9 | MMLU Chemistry | 1.3 (A-D) | claude-opus-4-1-20250805_none 1.3 A-D [A 0.70 B 0.73 C 0.71 D 0.74] ; claude-opus-4-1-20250805_thinking 1.3 A-D [A 0.75 B 0.78 C 0.76 D 0.79] ; claude-opus-4-1-20250805 1.2 A-D [A 0.74 B 0.78 C 0.76 D 0.78] ; claude-opus-4-1-20250805_16K 1.2 A-D [A 0.73 B 0.76 C 0.74 D 0.77] ; claude-sonnet-4-5-20250929_none 1.2 A-D [A 0.68 B 0.72 C 0.69 D 0.73] |
| 10 | OTIS Mock AIME 2024-2025 | 1.3 (A-D) | claude-sonnet-4-6_none 1.3 A-D [A 0.66 B 0.60 C 0.64 D 0.56] ; claude-sonnet-4-6_high 1.0 A-D [A 0.94 B 0.93 C 0.94 D 0.93] ; claude-opus-5-5_low 0.9 A-D [A 0.96 B 0.95 C 0.96 D 0.95] ; claude-sonnet-4-6_low 0.9 A-D [A 0.76 B 0.71 C 0.75 D 0.70] ; claude-opus-5-5_medium 0.9 A-D [A 0.98 B 0.98 C 0.98 D 0.98] |

### Modèles à poids ouverts

| rang | benchmark | meilleur z (paire) | modèles par ordre d'utilité (z, paire ; prédictions des modes) |
|---|---|---|---|
| 1 | GeoBench | 2.6 (B-C) | Phi-3-small-8k-instruct (open) 2.6 B-C [A 0.31 B 0.18 C 0.58 D 0.51] ; falcon-180B (open) 2.5 B-C [A 0.56 B 0.75 C 0.34 D 0.49] ; phi-2 (open) 2.5 B-C [A 0.20 B 0.07 C 0.47 D 0.22] ; Qwen2.5-72B (open) 2.3 B-C [A 0.47 B 0.31 C 0.67 D 0.56] ; qwen2.5-7b-instruct (open) 2.2 B-C [A 0.33 B 0.20 C 0.56 D 0.46] |
| 2 | ARC AI2 | 2.5 (B-D) | gpt-oss-20b (open) 2.5 B-D [A 0.78 B 0.76 C 0.82 D 0.90] ; Inkling_xhigh (open) 2.3 C-D [A 0.96 B 0.95 C 0.87 D 0.97] ; gpt-oss-20b_high (open) 2.2 B-D [A 0.80 B 0.77 C 0.83 D 0.91] ; Inkling_high (open) 2.2 C-D [A 0.95 B 0.94 C 0.85 D 0.97] ; Inkling (open) 2.2 C-D [A 0.95 B 0.95 C 0.86 D 0.96] |
| 3 | OpenBookQA | 2.3 (B-D) | gpt-oss-20b (open) 2.3 B-D [A 0.70 B 0.69 C 0.77 D 0.86] ; gpt-oss-20b_high (open) 2.3 B-D [A 0.72 B 0.70 C 0.77 D 0.88] ; Inkling_high (open) 2.3 C-D [A 0.91 B 0.89 C 0.75 D 0.93] ; Inkling_xhigh (open) 2.2 C-D [A 0.91 B 0.89 C 0.77 D 0.94] ; gpt-oss-120b_high (open) 2.2 B-D [A 0.77 B 0.75 C 0.82 D 0.90] |
| 4 | ARC-AGI | 2.3 (B-C) | qwen2.5-72b-instruct (open) 2.3 B-C [A 0.03 B 0.04 C 0.01 D 0.03] ; qwen2-72b-instruct (open) 1.8 B-C [A 0.01 B 0.02 C 0.00 D 0.01] ; gemma-3-27b-it (open) 1.4 A-B [A 0.01 B 0.03 C 0.01 D 0.02] ; DeepSeek-V3 (open) 1.4 B-C [A 0.04 B 0.04 C 0.02 D 0.03] ; phi-4 (open) 1.4 B-C [A 0.03 B 0.05 C 0.02 D 0.04] |
| 5 | DeepResearch Bench | 2.2 (B-C) | falcon-180B (open) 2.2 B-C [A 0.30 B 0.26 C 0.39 D 0.31] ; Phi-3-small-8k-instruct (open) 2.1 B-C [A 0.35 B 0.36 C 0.23 D 0.29] ; Baichuan-2-13B-Base (open) 2.1 B-D [A 0.21 B 0.19 C 0.28 D 0.34] ; DeepSeek-Coder-V2-Base (open) 2.1 B-D [A 0.28 B 0.26 C 0.36 D 0.41] ; Qwen2.5-Coder-14B (open) 2.0 B-D [A 0.26 B 0.25 C 0.31 D 0.38] |
| 6 | MATH Level 5 | 2.2 (B-D) | Baichuan-2-13B-Base (open) 2.2 B-D [A 0.79 B 0.83 C 0.75 D 0.32] ; Qwen2.5-Coder-1.5B (open) 2.2 A-D [A 0.60 B 0.61 C 0.53 D 0.16] ; Qwen2.5-Coder-14B (open) 2.0 A-D [A 0.78 B 0.78 C 0.67 D 0.35] ; Qwen2.5-Coder-7B (open) 1.9 B-D [A 0.66 B 0.68 C 0.55 D 0.25] ; gemma-7b (open) 1.9 A-D [A 0.22 B 0.27 C 0.38 D 0.60] |
| 7 | GPQA Main Chemistry | 2.2 (A-D) | qwen3-30b-a3b-instruct-2507 (open) 2.2 A-D [A 0.58 B 0.43 C 0.55 D 0.40] ; qwen3-30b-a3b-thinking-2507 (open) 2.1 A-D [A 0.66 B 0.52 C 0.63 D 0.47] ; Qwen3-30B-A3B_thinking (open) 2.0 A-D [A 0.59 B 0.46 C 0.55 D 0.42] ; Qwen3-30B-A3B_none (open) 1.9 A-D [A 0.46 B 0.35 C 0.43 D 0.33] ; Qwen3-30B-A3B (open) 1.9 A-D [A 0.57 B 0.44 C 0.54 D 0.40] |
| 8 | CyberGym | 2.2 (C-D) | Phi-3-small-8k-instruct (open) 2.2 C-D [A 0.05 B 0.08 C 0.40 D 0.01] ; Qwen2.5-72B (open) 1.9 C-D [A 0.06 B 0.03 C 0.26 D 0.01] ; falcon-180B (open) 1.9 C-D [A 0.47 B 0.30 C 0.12 D 0.44] ; DeepSeek-Coder-V2-Base (open) 1.8 A-C [A 0.48 B 0.27 C 0.10 D 0.30] ; qwen2.5-72b-instruct (open) 1.8 C-D [A 0.06 B 0.03 C 0.22 D 0.02] |
| 9 | WMDP Chemistry | 2.1 (A-D) | Qwen2.5-Coder-7B (open) 2.1 A-D [A 0.63 B 0.61 C 0.57 D 0.49] ; Baichuan-2-13B-Base (open) 2.0 A-D [A 0.57 B 0.54 C 0.50 D 0.40] ; Qwen2.5-Coder-14B (open) 2.0 A-D [A 0.67 B 0.65 C 0.62 D 0.54] ; Qwen2.5-Coder-32B (open) 1.9 A-D [A 0.69 B 0.68 C 0.65 D 0.57] ; Qwen2.5-Coder-1.5B (open) 1.8 A-D [A 0.55 B 0.52 C 0.51 D 0.41] |
| 10 | Lech Mazur Writing | 2.1 (A-C) | Phi-3-small-8k-instruct (open) 2.1 A-C [A 0.40 B 0.56 C 0.73 D 0.49] ; Llama-2-70b-chat (open) 2.0 C-D [A 0.63 B 0.61 C 0.45 D 0.68] ; Qwen2.5-72B (open) 2.0 A-C [A 0.53 B 0.62 C 0.76 D 0.61] ; qwen2.5-72b-instruct (open) 1.9 A-C [A 0.60 B 0.65 C 0.76 D 0.66] ; Baichuan-2-13B-Base (open) 1.8 C-D [A 0.55 B 0.58 C 0.33 D 0.61] |
