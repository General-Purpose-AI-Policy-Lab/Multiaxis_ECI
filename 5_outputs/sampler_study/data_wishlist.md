# Wishlist : mesures pour départager les modes du flagship

Générée par `data_wishlist.py` à partir du flagship du 19 octobre (K = 4, sans humains, capacités skew-normal, plafonds estimés), 50 chaînes en 5 modes (le mode E, atteint par une seule chaîne, est laissé de côté) :

- **A** : 21 chaînes (0, 6, 12, 16, 17, 18, 22, 27, 28, 29, 31, 33, 34, 36, 37, 38, 41, 44, 47, 48, 49), la meilleure log-densité ; axes : *Connaissances et raisonnements scientifiques* ; *Capacités agentiques & Programmation* ; *Capacités cyber* ; *Sens commun & connaissances factuelles*
- **B** : 5 chaînes (7, 11, 15, 19, 25), 9,6 nats sous A ; axes : GeoBench / Lech Mazur Writing ; OSWorld / SWE Atlas ; OTIS Mock AIME 2024-2025 / AudioMultiChallenge ; OpenBookQA / ARC AI2
- **C** : 19 chaînes (2, 3, 4, 5, 8, 10, 14, 21, 23, 24, 26, 30, 32, 35, 39, 40, 42, 45, 46), 17,3 nats sous A ; axes : CyberGym / CyScenarioBench ; DeepResearch Bench / APEX-Agents ; OTIS Mock AIME 2024-2025 / MATH Level 5 ; SimpleQA Verified / GSM8K
- **D** : 4 chaînes (1, 9, 20, 43), 25,8 nats sous A ; axes : CritPt / OTIS Mock AIME 2024-2025 ; DeepResearch Bench / GSM8K ; GDPval / GBAEval ; GeoBench / SpireClimb

Les livrables lisent A. PSIS-LOO ne classe pas les modes ; sur les 3 116 scores où la LOO intégrée est fiable dans tous les modes, A prédit le mieux, de peu face à B (−6,5 ± 3,2 nats) et C (−9,0 ± 4,1), davantage face à D (−14,4 ± 6,9) ; A est aussi devant sur l'ensemble des scores. Seules de nouvelles mesures peuvent les départager franchement.

**Comment lire.** Pour chaque case (modèle, benchmark) sans score, chaque mode donne une prédiction : sa moyenne et son écart-type, qui ajoute à l'incertitude du mode le bruit propre au benchmark. Pour deux modes, l'utilité *z* est l'écart entre leurs moyennes en écarts-types prédictifs ; une mesure tombée sur la prédiction d'un mode multiplie à peu près ses chances contre l'autre par e^(z²/2) (z = 1,5 : × 3 ; z = 2 : × 7). Une case vaut sa meilleure paire, indiquée à côté (`A-C` : la case départage A et C) ; le csv donne aussi `z_A`, sa meilleure paire avec A. Les benchmarks sont classés par leur meilleure case ; pour chacun, les modèles par ordre d'utilité : si un benchmark n'est pas accessible, prendre le suivant ; si un modèle ne l'est pas, le suivant sur la même ligne. Les axes des modes autres que A, qui n'ont pas de noms, sont désignés par leurs deux benchmarks les plus colinéaires. La disponibilité en API n'a pas été vérifiée ; `open` = poids ouverts. 89 422 cases non observées ; `data_wishlist.csv` garde les dix meilleurs modèles de chaque benchmark, parmi tous les modèles (`question = benchmark`), parmi ceux d'Anthropic (`anthropic`) et parmi les poids ouverts (`open`).

**Paires départagées par les 2 000 meilleures cases** : A-D 1027, B-C 607, C-D 239, B-D 94, A-C 33.

## En bref (lecture du 23 octobre, écrite à la main)

1. **A et B ne se départagent presque plus** : la meilleure case est à z = 0,9 (gpt-6-sol sans raisonnement sur GPQA Main Chimie). Sur 50 chaînes, B prédit à peu près comme A ; le choix entre les deux ne change guère les livrables.
2. **A contre C**, le rival le plus fréquent (19 chaînes, 17 nats plus bas), reste difficile : seules 33 des 2 000 meilleures cases les opposent. Les meilleures sont à z ≈ 1,7-1,8 : gpt-5.4-nano sans raisonnement sur ARC AI2 (A 0,80 contre C 0,91) et sur OpenBookQA (A 0,74 contre C 0,88) ; Qwen2.5-72B sur Lech Mazur Writing.
3. **A contre D** est le plus facile (1 027 des 2 000 meilleures cases) : gpt-5.4-nano (effort low) sur GPQA Main Chimie (A 0,67 contre D 0,46, z = 2,5), gpt-4.1-nano sur WMDP et GSM8K.
4. **Un petit lot teste A contre C et D à la fois** : gpt-5.4-nano sans raisonnement et en effort low sur ARC AI2, OpenBookQA et GPQA Main Chimie.
5. **Côté Anthropic**, Claude Opus 4.1 thinking ou Sonnet 4.5 thinking sur GPQA Main Chimie départagent A de D (A 0,63-0,65 contre D 0,75) ; Claude 3 Haiku et Sonnet 4 sur ARC AI2 et OpenBookQA opposent surtout B à C.

## Benchmarks classés

| rang | benchmark | meilleur z (paire) | axe dans A | modèles par ordre d'utilité (z, paire ; prédictions des modes) |
|---|---|---|---|---|
| 1 | GPQA Main Chemistry | 2.5 (A-D) | Connaissances et raisonnements scientifiques | gpt-5.4-nano-2026-03-17_low 2.5 A-D [A 0.67 B 0.58 C 0.63 D 0.46] ; gpt-5.4-nano-2026-03-17_medium 2.4 A-D [A 0.72 B 0.64 C 0.69 D 0.51] ; gpt-5.4-nano-2026-03-17_high 2.4 A-D [A 0.75 B 0.66 C 0.71 D 0.54] ; gpt-5.4-nano-2026-03-17_none 2.4 A-D [A 0.54 B 0.46 C 0.50 D 0.36] ; gpt-5-nano-2025-08-07_high 2.4 A-D [A 0.64 B 0.56 C 0.60 D 0.44] |
| 2 | ARC AI2 | 2.3 (B-C) | Sens commun & connaissances factuelles | gpt-5.4-nano-2026-03-17_none 2.3 B-C [A 0.80 B 0.76 C 0.91 D 0.84] ; o3-mini-2025-01-31_medium 2.0 C-D [A 0.89 B 0.90 C 0.84 D 0.92] ; Inkling_high (open) 2.0 B-C [A 0.94 B 0.95 C 0.84 D 0.95] ; Inkling (open) 1.9 B-C [A 0.94 B 0.96 C 0.86 D 0.95] ; claude-3-haiku-20240307 1.9 B-C [A 0.86 B 0.88 C 0.77 D 0.81] |
| 3 | OpenBookQA | 2.2 (B-C) | Sens commun & connaissances factuelles | gpt-5.4-nano-2026-03-17_none 2.2 B-C [A 0.74 B 0.70 C 0.88 D 0.81] ; Inkling_high (open) 1.8 B-C [A 0.89 B 0.91 C 0.76 D 0.91] ; gpt-5.4-nano-2026-03-17_low 1.8 B-C [A 0.76 B 0.74 C 0.88 D 0.84] ; Inkling (open) 1.8 B-C [A 0.89 B 0.91 C 0.78 D 0.91] ; gpt-5-nano-2025-08-07_minimal 1.8 B-C [A 0.66 B 0.62 C 0.78 D 0.78] |
| 4 | ARC-AGI | 2.1 (B-C) | Connaissances et raisonnements scientifiques | qwen2.5-72b-instruct (open) 2.1 B-C [A 0.03 B 0.04 C 0.01 D 0.03] ; qwen2-72b-instruct (open) 1.6 B-C [A 0.01 B 0.01 C 0.00 D 0.01] ; DeepSeek-V3 (open) 1.4 B-C [A 0.03 B 0.04 C 0.02 D 0.03] ; qwen3-14b_thinking (open) 1.4 B-C [A 0.25 B 0.27 C 0.13 D 0.19] ; Qwen2.5-72B (open) 1.4 B-C [A 0.04 B 0.05 C 0.01 D 0.03] |
| 5 | GeoBench | 2.1 (B-C) | Capacités cyber | Phi-3-small-8k-instruct (open) 2.1 B-C [A 0.34 B 0.22 C 0.58 D 0.44] ; phi-2 (open) 2.1 B-C [A 0.24 B 0.12 C 0.47 D 0.18] ; falcon-180B (open) 1.9 B-C [A 0.54 B 0.68 C 0.34 D 0.54] ; Qwen2.5-72B (open) 1.9 B-C [A 0.49 B 0.37 C 0.67 D 0.51] ; qwen2.5-7b-instruct (open) 1.8 B-C [A 0.36 B 0.24 C 0.56 D 0.41] |
| 6 | Lech Mazur Writing | 2.0 (B-C) | Capacités cyber | Qwen2.5-72B (open) 2.0 B-C [A 0.57 B 0.57 C 0.77 D 0.64] ; qwen2.5-72b-instruct (open) 1.9 B-C [A 0.62 B 0.62 C 0.77 D 0.67] ; Baichuan-2-13B-Base (open) 1.8 B-C [A 0.53 B 0.60 C 0.33 D 0.57] ; DeepSeek-Coder-V2-Base (open) 1.8 B-C [A 0.73 B 0.75 C 0.54 D 0.73] ; qwen2.5-7b-instruct (open) 1.8 B-C [A 0.47 B 0.46 C 0.72 D 0.57] |
| 7 | DeepResearch Bench | 2.0 (B-C) | Sens commun & connaissances factuelles | Phi-3-small-8k-instruct (open) 2.0 B-C [A 0.34 B 0.36 C 0.23 D 0.30] ; phi-2 (open) 1.9 C-D [A 0.24 B 0.26 C 0.14 D 0.28] ; falcon-180B (open) 1.8 B-C [A 0.32 B 0.28 C 0.39 D 0.29] ; falcon-40b (open) 1.7 C-D [A 0.29 B 0.26 C 0.34 D 0.23] ; opt-13b (open) 1.7 C-D [A 0.22 B 0.18 C 0.24 D 0.12] |
| 8 | HellaSwag | 1.9 (C-D) | Sens commun & connaissances factuelles | gpt-4.1-nano-2025-04-14 1.9 C-D [A 0.79 B 0.80 C 0.76 D 0.83] ; gpt-5-nano-2025-08-07_high 1.8 C-D [A 0.80 B 0.81 C 0.79 D 0.85] ; gpt-5-nano-2025-08-07_low 1.6 C-D [A 0.81 B 0.81 C 0.79 D 0.85] ; gpt-5-nano-2025-08-07_minimal 1.6 C-D [A 0.81 B 0.82 C 0.79 D 0.85] ; gemma-3-4b-it (open) 1.6 C-D [A 0.72 B 0.74 C 0.67 D 0.77] |
| 9 | CyberGym | 1.9 (C-D) | Capacités cyber | Qwen2.5-72B (open) 1.9 C-D [A 0.09 B 0.03 C 0.26 D 0.02] ; Phi-3-small-8k-instruct (open) 1.8 C-D [A 0.11 B 0.05 C 0.39 D 0.04] ; qwen2.5-72b-instruct (open) 1.8 C-D [A 0.09 B 0.04 C 0.21 D 0.03] ; gpt-oss-120b_medium (open) 1.8 A-D [A 0.41 B 0.31 C 0.36 D 0.17] ; gpt-oss-120b_high (open) 1.7 A-D [A 0.40 B 0.29 C 0.36 D 0.16] |
| 10 | WMDP Chemistry | 1.8 (A-D) | Connaissances et raisonnements scientifiques | gpt-4.1-nano-2025-04-14 1.8 A-D [A 0.66 B 0.62 C 0.62 D 0.53] ; gpt-5.4-nano-2026-03-17_none 1.8 A-D [A 0.69 B 0.65 C 0.67 D 0.58] ; gpt-5-nano-2025-08-07_minimal 1.8 A-D [A 0.67 B 0.63 C 0.63 D 0.56] ; gpt-5.4-nano-2026-03-17_low 1.7 A-D [A 0.75 B 0.72 C 0.73 D 0.66] ; gpt-5-nano-2025-08-07_low 1.6 A-D [A 0.70 B 0.67 C 0.67 D 0.60] |
| 11 | TriviaQA | 1.8 (B-C) | Capacités cyber | gemma-3-27b-it (open) 1.8 B-C [A 0.83 B 0.84 C 0.72 D 0.80] ; gpt-4.1-nano-2025-04-14 1.8 B-C [A 0.77 B 0.80 C 0.67 D 0.74] ; gpt-5-nano-2025-08-07_minimal 1.7 B-C [A 0.80 B 0.81 C 0.71 D 0.78] ; gemma-2-27b (open) 1.6 B-C [A 0.79 B 0.81 C 0.69 D 0.75] ; claude-3-haiku-20240307 1.6 B-C [A 0.76 B 0.74 C 0.82 D 0.78] |
| 12 | GPQA Diamond Chemistry | 1.8 (A-D) | Connaissances et raisonnements scientifiques | gpt-5.4-nano-2026-03-17_low 1.8 A-D [A 0.64 B 0.59 C 0.62 D 0.51] ; gpt-5.4-nano-2026-03-17_none 1.7 A-D [A 0.50 B 0.46 C 0.48 D 0.39] ; gpt-5.4-nano-2026-03-17_medium 1.7 A-D [A 0.70 B 0.65 C 0.68 D 0.57] ; gpt-5.4-nano-2026-03-17_high 1.7 A-D [A 0.73 B 0.68 C 0.71 D 0.60] ; gpt-5.4-nano-2026-03-17_xhigh 1.7 A-D [A 0.77 B 0.73 C 0.75 D 0.66] |
| 13 | GSM8K | 1.7 (A-D) | Connaissances et raisonnements scientifiques | gpt-4.1-nano-2025-04-14 1.7 A-D [A 0.79 B 0.71 C 0.78 D 0.58] ; gpt-5-nano-2025-08-07_high 1.6 A-D [A 0.90 B 0.86 C 0.88 D 0.75] ; gpt-5-nano-2025-08-07_minimal 1.5 A-D [A 0.81 B 0.74 C 0.80 D 0.63] ; gpt-5-nano-2025-08-07_low 1.5 A-D [A 0.85 B 0.80 C 0.83 D 0.69] ; gpt-5-nano-2025-08-07_medium 1.5 A-D [A 0.90 B 0.86 C 0.87 D 0.76] |
| 14 | CadEval | 1.7 (B-C) | Capacités agentiques & Programmation | qwen2.5-72b-instruct (open) 1.7 B-C [A 0.28 B 0.32 C 0.17 D 0.28] ; Qwen2.5-72B (open) 1.6 B-C [A 0.27 B 0.32 C 0.15 D 0.27] ; falcon-180B (open) 1.5 B-C [A 0.17 B 0.14 C 0.34 D 0.18] ; qwen2-72b-instruct (open) 1.4 B-C [A 0.19 B 0.21 C 0.12 D 0.19] ; Phi-3-small-8k-instruct (open) 1.4 B-C [A 0.26 B 0.28 C 0.10 D 0.19] |
| 15 | Cybench | 1.7 (B-C) | Capacités cyber | qwen2.5-72b-instruct (open) 1.7 B-C [A 0.08 B 0.05 C 0.19 D 0.06] ; Qwen2.5-72B (open) 1.5 B-C [A 0.07 B 0.04 C 0.22 D 0.05] ; gemini-2.0-flash-thinking-exp-01-21 1.4 A-D [A 0.19 B 0.14 C 0.17 D 0.09] ; claude-3-7-sonnet-20250219_none 1.3 A-D [A 0.27 B 0.32 C 0.27 D 0.38] ; gpt-oss-120b_high (open) 1.3 A-D [A 0.46 B 0.35 C 0.45 D 0.26] |

## Départager A de chacun des autres modes

La meilleure case de chaque benchmark pour chaque paire avec A.

### A contre B

| benchmark | modèle | z (A-B) | A | B |
|---|---|---|---|---|
| GPQA Main Chemistry | gpt-6-sol_none | 0.9 | 0.78 | 0.70 |
| ARC-AGI | gemma-3-27b-it | 0.7 | 0.01 | 0.02 |
| GeoBench | falcon-40b | 0.7 | 0.38 | 0.53 |
| WMDP Chemistry | gpt-6-sol_none | 0.7 | 0.80 | 0.77 |
| SimpleQA Verified | falcon-180B | 0.7 | 0.06 | 0.10 |
| GPQA Main Biology | gpt-5.4-nano-2026-03-17_none | 0.7 | 0.66 | 0.59 |
| Cybench | claude-3-7-sonnet-20250219_none | 0.7 | 0.27 | 0.32 |
| CyberGym | gpt-oss-120b_high | 0.6 | 0.40 | 0.29 |

### A contre C

| benchmark | modèle | z (A-C) | A | C |
|---|---|---|---|---|
| ARC AI2 | gpt-5.4-nano-2026-03-17_none | 1.8 | 0.80 | 0.91 |
| OpenBookQA | gpt-5.4-nano-2026-03-17_none | 1.7 | 0.74 | 0.88 |
| Lech Mazur Writing | Qwen2.5-72B | 1.7 | 0.57 | 0.77 |
| CyberGym | DeepSeek-Coder-V2-Base | 1.6 | 0.43 | 0.09 |
| TriviaQA | gemma-3-27b-it | 1.5 | 0.83 | 0.72 |
| ARC-AGI | qwen2.5-72b-instruct | 1.5 | 0.03 | 0.01 |
| DeepResearch Bench | Phi-3-small-8k-instruct | 1.4 | 0.34 | 0.23 |
| MMLU Pro Biology | Llama-2-70b-chat | 1.3 | 0.61 | 0.52 |

### A contre D

| benchmark | modèle | z (A-D) | A | D |
|---|---|---|---|---|
| GPQA Main Chemistry | gpt-5.4-nano-2026-03-17_low | 2.5 | 0.67 | 0.46 |
| WMDP Chemistry | gpt-4.1-nano-2025-04-14 | 1.8 | 0.66 | 0.53 |
| GPQA Diamond Chemistry | gpt-5.4-nano-2026-03-17_low | 1.8 | 0.64 | 0.51 |
| CyberGym | gpt-oss-120b_medium | 1.8 | 0.41 | 0.17 |
| GSM8K | gpt-4.1-nano-2025-04-14 | 1.7 | 0.79 | 0.58 |
| MATH Level 5 | text-davinci-001 | 1.6 | 0.06 | 0.39 |
| GPQA Main Biology | gpt-4.1-nano-2025-04-14 | 1.6 | 0.60 | 0.45 |
| MMLU Chemistry | gpt-4.1-nano-2025-04-14 | 1.6 | 0.65 | 0.55 |

## Ce que nous pouvons mesurer nous-mêmes

### Modèles Anthropic (comme la campagne de mesures propres, protocole RAND)

| rang | benchmark | meilleur z (paire) | modèles par ordre d'utilité (z, paire ; prédictions des modes) |
|---|---|---|---|
| 1 | GPQA Main Chemistry | 2.1 (A-D) | claude-opus-4-1-20250805_thinking 2.1 A-D [A 0.63 B 0.69 C 0.65 D 0.75] ; claude-sonnet-4-5-20250929_thinking 2.0 A-D [A 0.65 B 0.70 C 0.66 D 0.75] ; claude-opus-4-1-20250805_16K 2.0 A-D [A 0.61 B 0.65 C 0.62 D 0.71] ; claude-opus-4-1-20250805 2.0 A-D [A 0.62 B 0.67 C 0.64 D 0.74] ; claude-opus-4-20250514_thinking 2.0 A-D [A 0.62 B 0.67 C 0.63 D 0.73] |
| 2 | ARC AI2 | 1.9 (B-C) | claude-3-haiku-20240307 1.9 B-C [A 0.86 B 0.88 C 0.77 D 0.81] ; claude-sonnet-4-20250514_32K 1.9 B-C [A 0.94 B 0.95 C 0.90 D 0.93] ; claude-sonnet-4-20250514_thinking 1.9 B-C [A 0.94 B 0.96 C 0.90 D 0.94] ; claude-sonnet-4-20250514_59K 1.8 B-C [A 0.95 B 0.96 C 0.91 D 0.94] ; claude-3-7-sonnet-20250219 1.8 B-C [A 0.95 B 0.96 C 0.91 D 0.94] |
| 3 | OpenBookQA | 1.7 (B-C) | claude-3-haiku-20240307 1.7 B-C [A 0.81 B 0.84 C 0.71 D 0.76] ; claude-sonnet-4-20250514_32K 1.5 B-C [A 0.90 B 0.91 C 0.84 D 0.88] ; claude-sonnet-4-20250514_thinking 1.5 B-C [A 0.90 B 0.92 C 0.85 D 0.89] ; claude-sonnet-4-20250514_59K 1.4 B-C [A 0.90 B 0.92 C 0.85 D 0.89] ; claude-haiku-5-5_none 1.4 C-D [A 0.86 B 0.87 C 0.91 D 0.77] |
| 4 | TriviaQA | 1.6 (B-C) | claude-3-haiku-20240307 1.6 B-C [A 0.76 B 0.74 C 0.82 D 0.78] ; claude-sonnet-4-20250514_32K 1.4 B-C [A 0.85 B 0.83 C 0.89 D 0.86] ; claude-sonnet-4-20250514_thinking 1.3 B-C [A 0.85 B 0.84 C 0.89 D 0.86] ; claude-sonnet-4-20250514_59K 1.3 B-C [A 0.85 B 0.84 C 0.89 D 0.86] ; claude-sonnet-4-20250514_8K 1.3 B-C [A 0.85 B 0.84 C 0.89 D 0.86] |
| 5 | GPQA Diamond Chemistry | 1.4 (A-D) | claude-opus-4-1-20250805_thinking 1.4 A-D [A 0.69 B 0.72 C 0.70 D 0.76] ; claude-sonnet-4-5-20250929_thinking 1.4 A-D [A 0.71 B 0.73 C 0.71 D 0.77] ; claude-opus-4-20250514_thinking 1.4 A-D [A 0.67 B 0.70 C 0.68 D 0.73] ; claude-opus-4-1-20250805_32K 1.3 A-D [A 0.65 B 0.68 C 0.66 D 0.72] ; claude-opus-4-1-20250805 1.3 A-D [A 0.68 B 0.71 C 0.69 D 0.75] |
| 6 | OTIS Mock AIME 2024-2025 | 1.3 (A-D) | claude-sonnet-4-6_none 1.3 A-D [A 0.65 B 0.62 C 0.63 D 0.56] ; claude-sonnet-4-6_high 1.2 A-D [A 0.94 B 0.94 C 0.94 D 0.92] ; claude-sonnet-4-6_low 1.0 A-D [A 0.76 B 0.73 C 0.75 D 0.69] ; claude-opus-5-5_low 1.0 A-D [A 0.96 B 0.96 C 0.96 D 0.95] ; claude-sonnet-4-6_max 0.9 A-D [A 0.94 B 0.94 C 0.94 D 0.93] |
| 7 | MMLU Chemistry | 1.3 (A-D) | claude-opus-4-1-20250805_none 1.3 A-D [A 0.70 B 0.71 C 0.71 D 0.74] ; claude-opus-4-1-20250805_thinking 1.3 A-D [A 0.75 B 0.76 C 0.76 D 0.79] ; claude-sonnet-4-5-20250929_none 1.3 A-D [A 0.68 B 0.70 C 0.69 D 0.73] ; claude-opus-4-1-20250805_32K 1.3 A-D [A 0.73 B 0.75 C 0.75 D 0.78] ; claude-opus-4-1-20250805 1.3 A-D [A 0.74 B 0.76 C 0.76 D 0.79] |
| 8 | HellaSwag | 1.3 (B-C) | claude-sonnet-4-5-20250929_59K 1.3 B-C [A 0.91 B 0.90 C 0.93 D 0.91] ; claude-sonnet-4-20250514_32K 1.3 B-C [A 0.89 B 0.88 C 0.91 D 0.89] ; claude-sonnet-4-20250514_16K 1.3 B-C [A 0.90 B 0.90 C 0.92 D 0.90] ; claude-haiku-4-5-20251001_32K 1.3 B-C [A 0.88 B 0.87 C 0.90 D 0.88] ; claude-sonnet-4-5-20250929_thinking 1.3 B-C [A 0.91 B 0.90 C 0.93 D 0.91] |
| 9 | Cybench | 1.3 (A-D) | claude-3-7-sonnet-20250219_none 1.3 A-D [A 0.27 B 0.32 C 0.27 D 0.38] ; claude-3-5-sonnet-20241022 1.2 A-D [A 0.22 B 0.27 C 0.23 D 0.32] ; claude-3-7-sonnet-20250219_1K 1.0 A-D [A 0.27 B 0.31 C 0.27 D 0.35] ; claude-3-7-sonnet-20250219_8K 0.9 A-D [A 0.29 B 0.32 C 0.29 D 0.36] ; claude-3-7-sonnet-20250219_12K 0.9 C-D [A 0.31 B 0.34 C 0.30 D 0.38] |
| 10 | GDPval | 1.3 (A-D) | claude-3-5-sonnet-20241022 1.3 A-D [A 0.24 B 0.26 C 0.27 D 0.34] ; claude-3-7-sonnet-20250219_none 1.3 A-D [A 0.27 B 0.30 C 0.29 D 0.36] ; claude-3-5-haiku-20241022 1.1 A-D [A 0.16 B 0.20 C 0.19 D 0.26] ; claude-3-7-sonnet-20250219_1K 0.9 A-D [A 0.25 B 0.28 C 0.27 D 0.32] ; claude-3-haiku-20240307 0.8 A-D [A 0.04 B 0.05 C 0.05 D 0.07] |

### Modèles à poids ouverts

| rang | benchmark | meilleur z (paire) | modèles par ordre d'utilité (z, paire ; prédictions des modes) |
|---|---|---|---|
| 1 | GPQA Main Chemistry | 2.2 (A-D) | qwen3-30b-a3b-instruct-2507 (open) 2.2 A-D [A 0.58 B 0.51 C 0.54 D 0.40] ; qwen3-30b-a3b-thinking-2507 (open) 2.1 A-D [A 0.66 B 0.59 C 0.62 D 0.47] ; Qwen3-30B-A3B_thinking (open) 2.0 A-D [A 0.58 B 0.52 C 0.54 D 0.42] ; Qwen3-30B-A3B_none (open) 1.9 A-D [A 0.46 B 0.41 C 0.42 D 0.33] ; Qwen3-30B-A3B (open) 1.9 A-D [A 0.57 B 0.51 C 0.53 D 0.41] |
| 2 | ARC-AGI | 2.1 (B-C) | qwen2.5-72b-instruct (open) 2.1 B-C [A 0.03 B 0.04 C 0.01 D 0.03] ; qwen2-72b-instruct (open) 1.6 B-C [A 0.01 B 0.01 C 0.00 D 0.01] ; DeepSeek-V3 (open) 1.4 B-C [A 0.03 B 0.04 C 0.02 D 0.03] ; qwen3-14b_thinking (open) 1.4 B-C [A 0.25 B 0.27 C 0.13 D 0.19] ; Qwen2.5-72B (open) 1.4 B-C [A 0.04 B 0.05 C 0.01 D 0.03] |
| 3 | GeoBench | 2.1 (B-C) | Phi-3-small-8k-instruct (open) 2.1 B-C [A 0.34 B 0.22 C 0.58 D 0.44] ; phi-2 (open) 2.1 B-C [A 0.24 B 0.12 C 0.47 D 0.18] ; falcon-180B (open) 1.9 B-C [A 0.54 B 0.68 C 0.34 D 0.54] ; Qwen2.5-72B (open) 1.9 B-C [A 0.49 B 0.37 C 0.67 D 0.51] ; qwen2.5-7b-instruct (open) 1.8 B-C [A 0.36 B 0.24 C 0.56 D 0.41] |
| 4 | Lech Mazur Writing | 2.0 (B-C) | Qwen2.5-72B (open) 2.0 B-C [A 0.57 B 0.57 C 0.77 D 0.64] ; qwen2.5-72b-instruct (open) 1.9 B-C [A 0.62 B 0.62 C 0.77 D 0.67] ; Baichuan-2-13B-Base (open) 1.8 B-C [A 0.53 B 0.60 C 0.33 D 0.57] ; DeepSeek-Coder-V2-Base (open) 1.8 B-C [A 0.73 B 0.75 C 0.54 D 0.73] ; qwen2.5-7b-instruct (open) 1.8 B-C [A 0.47 B 0.46 C 0.72 D 0.57] |
| 5 | DeepResearch Bench | 2.0 (B-C) | Phi-3-small-8k-instruct (open) 2.0 B-C [A 0.34 B 0.36 C 0.23 D 0.30] ; phi-2 (open) 1.9 C-D [A 0.24 B 0.26 C 0.14 D 0.28] ; falcon-180B (open) 1.8 B-C [A 0.32 B 0.28 C 0.39 D 0.29] ; falcon-40b (open) 1.7 C-D [A 0.29 B 0.26 C 0.34 D 0.23] ; opt-13b (open) 1.7 C-D [A 0.22 B 0.18 C 0.24 D 0.12] |
| 6 | ARC AI2 | 2.0 (B-C) | Inkling_high (open) 2.0 B-C [A 0.94 B 0.95 C 0.84 D 0.95] ; Inkling (open) 1.9 B-C [A 0.94 B 0.96 C 0.86 D 0.95] ; Inkling_xhigh (open) 1.9 B-C [A 0.95 B 0.96 C 0.86 D 0.96] ; Inkling_thinking (open) 1.8 B-C [A 0.94 B 0.96 C 0.86 D 0.95] ; qwen3.6-flash_thinking (open) 1.7 B-C [A 0.91 B 0.92 C 0.83 D 0.92] |
| 7 | CyberGym | 1.9 (C-D) | Qwen2.5-72B (open) 1.9 C-D [A 0.09 B 0.03 C 0.26 D 0.02] ; Phi-3-small-8k-instruct (open) 1.8 C-D [A 0.11 B 0.05 C 0.39 D 0.04] ; qwen2.5-72b-instruct (open) 1.8 C-D [A 0.09 B 0.04 C 0.21 D 0.03] ; gpt-oss-120b_medium (open) 1.8 A-D [A 0.41 B 0.31 C 0.36 D 0.17] ; gpt-oss-120b_high (open) 1.7 A-D [A 0.40 B 0.29 C 0.36 D 0.16] |
| 8 | OpenBookQA | 1.8 (B-C) | Inkling_high (open) 1.8 B-C [A 0.89 B 0.91 C 0.76 D 0.91] ; Inkling (open) 1.8 B-C [A 0.89 B 0.91 C 0.78 D 0.91] ; Inkling_xhigh (open) 1.8 B-C [A 0.89 B 0.91 C 0.77 D 0.92] ; Inkling_thinking (open) 1.7 B-C [A 0.89 B 0.91 C 0.78 D 0.91] ; qwen3-30b-a3b-instruct-2507 (open) 1.7 B-C [A 0.76 B 0.73 C 0.87 D 0.83] |
| 9 | TriviaQA | 1.8 (B-C) | gemma-3-27b-it (open) 1.8 B-C [A 0.83 B 0.84 C 0.72 D 0.80] ; gemma-2-27b (open) 1.6 B-C [A 0.79 B 0.81 C 0.69 D 0.75] ; gemma-2-27b-it (open) 1.5 B-C [A 0.80 B 0.81 C 0.70 D 0.77] ; gemma-4-31b-it (open) 1.4 B-C [A 0.82 B 0.80 C 0.88 D 0.84] ; Qwen3-30B-A3B_none (open) 1.4 B-C [A 0.82 B 0.83 C 0.74 D 0.81] |
| 10 | CadEval | 1.7 (B-C) | qwen2.5-72b-instruct (open) 1.7 B-C [A 0.28 B 0.32 C 0.17 D 0.28] ; Qwen2.5-72B (open) 1.6 B-C [A 0.27 B 0.32 C 0.15 D 0.27] ; falcon-180B (open) 1.5 B-C [A 0.17 B 0.14 C 0.34 D 0.18] ; qwen2-72b-instruct (open) 1.4 B-C [A 0.19 B 0.21 C 0.12 D 0.19] ; Phi-3-small-8k-instruct (open) 1.4 B-C [A 0.26 B 0.28 C 0.10 D 0.19] |
