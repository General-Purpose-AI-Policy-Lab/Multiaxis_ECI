# Wishlist : mesures pour départager les modes du flagship (alternatives classées)

Générée le 2026-10-12 à partir des prédictions des cinq modes NUTS (A–E) du flagship (K = 4, sans humains).
Pour chaque question (quels modes départager), les benchmarks sont classés par ordre d'utilité, et pour chacun les
modèles par ordre d'utilité : si un benchmark n'est pas accessible, prendre le suivant ; si un modèle ne l'est pas, le suivant
sur la même ligne. L'utilité d'une case est l'écart entre les scores prédits par les modes concernés (pour « tous » :
l'écart-type entre les cinq modes divisé par l'incertitude de prédiction dans un mode). La disponibilité en API n'a pas pu être
vérifiée ; `open` = poids ouverts, exécutables sans API. Fichier complet : 62 234 cases non observées.

## les cinq modes à la fois (séparation globale)

| rang | benchmark | meilleur séparation | modèles par ordre d'utilité (valeur ; prédictions des modes) |
|---|---|---|---|
| 1 | GPQA Main Chemistry | 1.20 | gpt-6-luna_none 1.20 [0.37/0.41/0.68/0.87/0.38] ; gpt-5.6-luna_none 1.18 [0.35/0.39/0.64/0.84/0.36] ; gpt-5.6-terra_none 1.17 [0.41/0.45/0.69/0.89/0.41] ; claude-sonnet-5_none 1.16 [0.40/0.43/0.67/0.86/0.40] ; gpt-5.4-nano-2026-03-17_none 1.15 [0.35/0.39/0.61/0.77/0.34] |
| 2 | OTIS Mock AIME 2024-2025 | 1.06 | inkling-small_none (open) 1.06 [0.24/0.24/0.35/0.25/0.62] ; gpt-6-astra_none 0.90 [0.98/0.97/0.95/0.97/0.87] ; o3-pro-2025-06-10_medium 0.85 [0.85/0.86/0.81/0.86/0.67] ; o3-pro-2025-06-10_high 0.85 [0.88/0.89/0.84/0.89/0.72] ; inkling-small_minimal (open) 0.83 [0.55/0.56/0.64/0.56/0.84] |
| 3 | VPCT | 1.04 | gemini-3.6-flash_minimal 1.04 [0.50/0.50/0.50/0.49/0.77] ; glm-5.2 (open) 1.03 [0.50/0.52/0.54/0.50/0.79] ; grok-4.5 1.02 [0.59/0.61/0.68/0.61/0.90] ; grok-4.6_low 1.01 [0.57/0.59/0.65/0.57/0.89] ; Inkling (open) 1.01 [0.40/0.41/0.52/0.43/0.77] |
| 4 | MCP Atlas | 0.99 | gpt-5.4-nano-2026-03-17_high 0.99 [0.22/0.22/0.34/0.23/0.60] ; gpt-5.4-nano-2026-03-17_medium 0.97 [0.22/0.22/0.34/0.23/0.58] ; gpt-5.4-nano-2026-03-17_xhigh 0.96 [0.26/0.26/0.38/0.27/0.63] ; gpt-5-nano-2025-08-07_high 0.96 [0.09/0.10/0.21/0.10/0.42] ; gpt-5.4-nano-2026-03-17_low 0.95 [0.21/0.21/0.32/0.23/0.54] |
| 5 | GPQA Main Biology | 0.95 | gpt-5.4-nano-2026-03-17_none 0.95 [0.56/0.61/0.77/0.89/0.60] ; gemma-3-27b-it (open) 0.95 [0.52/0.58/0.71/0.83/0.49] ; gpt-5-nano-2025-08-07_minimal 0.94 [0.52/0.56/0.71/0.81/0.53] ; gpt-5.4-mini-2026-03-17_none 0.94 [0.60/0.62/0.79/0.88/0.66] ; gpt-5.6-luna_none 0.94 [0.60/0.63/0.82/0.93/0.67] |
| 6 | APEX-Agents | 0.94 | gpt-5.4-nano-2026-03-17_high 0.94 [0.10/0.10/0.16/0.10/0.31] ; gpt-5.4-nano-2026-03-17_medium 0.93 [0.10/0.10/0.15/0.10/0.30] ; gpt-5.4-nano-2026-03-17_low 0.91 [0.10/0.10/0.15/0.10/0.29] ; gpt-5.4-nano-2026-03-17_xhigh 0.90 [0.13/0.12/0.18/0.12/0.32] ; gpt-5.4-nano-2026-03-17_none 0.85 [0.08/0.08/0.13/0.08/0.27] |
| 7 | GSO-Bench | 0.92 | grok-4.6_xhigh 0.92 [0.47/0.46/0.38/0.46/0.23] ; grok-4.6_high 0.88 [0.47/0.47/0.38/0.46/0.23] ; grok-4.5_high 0.86 [0.38/0.37/0.30/0.37/0.18] ; grok-4.6_medium 0.85 [0.49/0.49/0.40/0.48/0.25] ; muse-spark-1.1 0.84 [0.41/0.39/0.33/0.40/0.18] |
| 8 | WMDP Chemistry | 0.92 | gpt-5.4-nano-2026-03-17_none 0.92 [0.61/0.65/0.74/0.81/0.56] ; gpt-5.6-luna_none 0.92 [0.62/0.64/0.76/0.84/0.60] ; gpt-6-luna_none 0.91 [0.64/0.67/0.79/0.86/0.61] ; gpt-5-nano-2025-08-07_minimal 0.90 [0.57/0.60/0.70/0.74/0.52] ; gpt-6-sol_none 0.87 [0.79/0.81/0.87/0.93/0.70] |
| 9 | GPQA Diamond Chemistry | 0.91 | gpt-6-luna_none 0.91 [0.47/0.53/0.71/0.84/0.53] ; gpt-5.6-luna_none 0.89 [0.47/0.52/0.69/0.82/0.51] ; gpt-5.4-nano-2026-03-17_none 0.85 [0.40/0.45/0.58/0.70/0.43] ; gpt-5.6-terra_none 0.85 [0.55/0.60/0.76/0.87/0.59] ; gpt-5.4-mini-2026-03-17_none 0.83 [0.43/0.47/0.60/0.71/0.47] |
| 10 | MMLU Pro Chemistry | 0.91 | gpt-5-nano-2025-08-07_minimal 0.91 [0.62/0.66/0.78/0.83/0.53] ; gpt-5.4-nano-2026-03-17_none 0.87 [0.71/0.74/0.85/0.91/0.62] ; gpt-4.1-nano-2025-04-14 0.86 [0.56/0.61/0.71/0.78/0.48] ; gpt-5.6-luna_none 0.81 [0.77/0.78/0.89/0.94/0.74] ; gemma-3-27b-it (open) 0.80 [0.65/0.72/0.80/0.88/0.61] |
| 11 | Aider Polyglot | 0.89 | gpt-5.4-nano-2026-03-17_none 0.89 [0.29/0.28/0.25/0.27/0.51] ; gpt-5.4-nano-2026-03-17_low 0.75 [0.44/0.44/0.41/0.42/0.64] ; gpt-5.4-nano-2026-03-17_high 0.70 [0.52/0.52/0.49/0.51/0.70] ; gpt-5.4-nano-2026-03-17_medium 0.69 [0.50/0.50/0.47/0.48/0.68] ; gpt-5.1-2025-11-13_none 0.61 [0.58/0.59/0.57/0.57/0.71] |
| 12 | VisualToolBench | 0.89 | gpt-6-astra_max 0.89 [0.89/0.87/0.79/0.83/0.57] ; gpt-6-astra_xhigh 0.88 [0.88/0.87/0.78/0.83/0.56] ; gpt-6-astra_high 0.86 [0.86/0.85/0.76/0.81/0.54] ; gpt-6-astra_medium 0.83 [0.87/0.86/0.77/0.81/0.56] ; grok-4.5 0.82 [0.23/0.24/0.27/0.23/0.42] |

## A contre D

| rang | benchmark | meilleur écart | modèles par ordre d'utilité (valeur ; prédictions des modes) |
|---|---|---|---|
| 1 | GPQA Main Chemistry | 0.50 | gpt-6-luna_none 0.50 [A 0.37 vs D 0.87] ; gpt-5.6-luna_none 0.49 [A 0.35 vs D 0.84] ; gpt-5.6-terra_none 0.48 [A 0.41 vs D 0.89] ; gpt-6-luna_low 0.47 [A 0.43 vs D 0.90] ; claude-sonnet-5_none 0.46 [A 0.40 vs D 0.86] |
| 2 | GPQA Diamond Chemistry | 0.37 | gpt-6-luna_none 0.37 [A 0.47 vs D 0.84] ; gpt-5.6-luna_none 0.36 [A 0.47 vs D 0.82] ; gpt-5.6-terra_none 0.32 [A 0.55 vs D 0.87] ; claude-sonnet-5_none 0.32 [A 0.53 vs D 0.85] ; gpt-5.4-nano-2026-03-17_none 0.30 [A 0.40 vs D 0.70] |
| 3 | GPQA Main Biology | 0.33 | gpt-5.6-luna_none 0.33 [A 0.60 vs D 0.93] ; gpt-6-luna_none 0.33 [A 0.61 vs D 0.94] ; gpt-5.4-nano-2026-03-17_none 0.33 [A 0.56 vs D 0.89] ; gemma-3-27b-it (open) 0.31 [A 0.52 vs D 0.83] ; gpt-5.6-terra_none 0.29 [A 0.66 vs D 0.95] |
| 4 | Lech Mazur Writing | 0.30 | Mistral-7B-Instruct-v0.2 (open) 0.30 [A 0.67 vs D 0.37] ; open-mistral-7b (open) 0.29 [A 0.61 vs D 0.32] ; Mistral-7B-Instruct-v0.1 (open) 0.29 [A 0.61 vs D 0.32] ; Mistral-7B-Instruct-v0.3 (open) 0.28 [A 0.71 vs D 0.43] ; Mistral-7B-v0.1 (open) 0.28 [A 0.63 vs D 0.36] |
| 5 | Cybench | 0.29 | DeepSeek-Coder-V2-Base (open) 0.29 [A 0.23 vs D 0.53] ; Qwen2.5-Coder-32B (open) 0.26 [A 0.16 vs D 0.42] ; Qwen2.5-Coder-14B (open) 0.25 [A 0.10 vs D 0.35] ; Baichuan-2-13B-Base (open) 0.22 [A 0.08 vs D 0.30] ; Yi-9B (open) 0.20 [A 0.09 vs D 0.29] |
| 6 | MMLU Pro Chemistry | 0.24 | DeepSeek-R1-Distill-Qwen-14B (open) 0.24 [A 0.38 vs D 0.61] ; gemma-3-27b-it (open) 0.23 [A 0.65 vs D 0.88] ; glm-4.7-flash_none (open) 0.22 [A 0.40 vs D 0.63] ; gpt-4.1-nano-2025-04-14 0.22 [A 0.56 vs D 0.78] ; gpt-5-nano-2025-08-07_minimal 0.21 [A 0.62 vs D 0.83] |
| 7 | CyberGym | 0.23 | DeepSeek-Coder-V2-Base (open) 0.23 [A 0.18 vs D 0.41] ; gpt-4-0314 0.22 [A 0.29 vs D 0.07] ; Qwen2.5-Coder-32B (open) 0.21 [A 0.12 vs D 0.33] ; Qwen2.5-Coder-14B (open) 0.20 [A 0.09 vs D 0.28] ; Baichuan-2-13B-Base (open) 0.18 [A 0.07 vs D 0.25] |
| 8 | WMDP Chemistry | 0.22 | gpt-5.6-luna_none 0.22 [A 0.62 vs D 0.84] ; gpt-6-luna_none 0.22 [A 0.64 vs D 0.86] ; gpt-5.6-terra_none 0.20 [A 0.66 vs D 0.86] ; gpt-6-luna_low 0.20 [A 0.66 vs D 0.86] ; gpt-6-luna 0.20 [A 0.67 vs D 0.87] |
| 9 | OpenBookQA | 0.22 | Inkling_high (open) 0.22 [A 0.61 vs D 0.83] ; Inkling_xhigh (open) 0.22 [A 0.62 vs D 0.84] ; Inkling (open) 0.21 [A 0.63 vs D 0.84] ; Inkling_thinking (open) 0.21 [A 0.63 vs D 0.84] ; gpt-5-nano-2025-08-07_minimal 0.15 [A 0.92 vs D 0.77] |
| 10 | WMDP Biology | 0.22 | deepseek-r1-0528-qwen3-8b (open) 0.22 [A 0.43 vs D 0.65] ; mpt-7b (open) 0.20 [A 0.44 vs D 0.63] ; opt-66b (open) 0.19 [A 0.43 vs D 0.62] ; text-curie-001 0.19 [A 0.40 vs D 0.59] ; curie 0.18 [A 0.43 vs D 0.61] |
| 11 | MATH Level 5 | 0.22 | text-davinci-002 0.22 [A 0.67 vs D 0.46] ; text-davinci-001 0.19 [A 0.50 vs D 0.30] ; gemma-2b (open) 0.19 [A 0.49 vs D 0.30] ; Qwen2.5-Coder-3B (open) 0.18 [A 0.38 vs D 0.55] ; Qwen2.5-Coder-1.5B (open) 0.17 [A 0.30 vs D 0.47] |
| 12 | MCP Atlas | 0.21 | Qwen2.5-Coder-32B (open) 0.21 [A 0.51 vs D 0.29] ; Baichuan-2-13B-Base (open) 0.19 [A 0.36 vs D 0.17] ; Qwen2.5-Coder-14B (open) 0.18 [A 0.41 vs D 0.23] ; DeepSeek-Coder-V2-Base (open) 0.16 [A 0.56 vs D 0.41] ; Qwen2.5-Coder-3B (open) 0.14 [A 0.25 vs D 0.11] |

## A contre E

| rang | benchmark | meilleur écart | modèles par ordre d'utilité (valeur ; prédictions des modes) |
|---|---|---|---|
| 1 | MCP Atlas | 0.38 | gpt-5.4-nano-2026-03-17_high 0.38 [A 0.22 vs E 0.60] ; gpt-5.4-nano-2026-03-17_xhigh 0.37 [A 0.26 vs E 0.63] ; gpt-5.4-nano-2026-03-17_medium 0.36 [A 0.22 vs E 0.58] ; gpt-5.4-nano-2026-03-17_low 0.33 [A 0.21 vs E 0.54] ; gpt-5-nano-2025-08-07_high 0.32 [A 0.09 vs E 0.42] |
| 2 | OTIS Mock AIME 2024-2025 | 0.38 | inkling-small_none (open) 0.38 [A 0.24 vs E 0.62] ; inkling-small_minimal (open) 0.29 [A 0.55 vs E 0.84] ; nemotron-3-ultra_thinking (open) 0.28 [A 0.40 vs E 0.68] ; inkling-small_low (open) 0.24 [A 0.64 vs E 0.88] ; o3-pro-2025-06-10_low 0.21 [A 0.80 vs E 0.59] |
| 3 | VPCT | 0.37 | Inkling (open) 0.37 [A 0.40 vs E 0.77] ; Inkling_thinking (open) 0.34 [A 0.42 vs E 0.76] ; Inkling_high (open) 0.33 [A 0.45 vs E 0.78] ; inkling-small_xhigh (open) 0.33 [A 0.42 vs E 0.75] ; inkling-small_high (open) 0.33 [A 0.42 vs E 0.74] |
| 4 | GPQA Main Chemistry | 0.37 | Inkling (open) 0.37 [A 0.51 vs E 0.88] ; Inkling_thinking (open) 0.33 [A 0.55 vs E 0.88] ; Inkling_high (open) 0.33 [A 0.56 vs E 0.89] ; inkling-small_xhigh (open) 0.30 [A 0.44 vs E 0.74] ; Inkling_xhigh (open) 0.30 [A 0.61 vs E 0.91] |
| 5 | GeoBench | 0.36 | Inkling_high (open) 0.36 [A 0.38 vs E 0.74] ; Inkling (open) 0.36 [A 0.39 vs E 0.74] ; Inkling_xhigh (open) 0.35 [A 0.40 vs E 0.75] ; Inkling_thinking (open) 0.34 [A 0.40 vs E 0.74] ; Baichuan-2-13B-Base (open) 0.27 [A 0.16 vs E 0.43] |
| 6 | ARC-AGI-2 | 0.34 | glm-5.3-flash_max (open) 0.34 [A 0.64 vs E 0.30] ; kimi-k2.7-code (open) 0.17 [A 0.49 vs E 0.32] ; qwen3.7-plus 0.16 [A 0.24 vs E 0.41] ; qwen3.7-plus_thinking 0.16 [A 0.32 vs E 0.48] ; deepseek-v4-pro_high (open) 0.15 [A 0.47 vs E 0.32] |
| 7 | ARC-AGI | 0.33 | glm-4.7-flash_thinking (open) 0.33 [A 0.45 vs E 0.12] ; gemma-4-26b-a4b_minimal (open) 0.27 [A 0.57 vs E 0.30] ; gpt-oss-120b_high (open) 0.20 [A 0.41 vs E 0.61] ; glm-5.3-flash_max (open) 0.20 [A 0.89 vs E 0.69] ; gpt-oss-120b_medium (open) 0.18 [A 0.30 vs E 0.48] |
| 8 | VisualToolBench | 0.33 | gpt-6-astra_high 0.33 [A 0.86 vs E 0.54] ; gpt-6-astra_xhigh 0.32 [A 0.88 vs E 0.56] ; gpt-6-astra_max 0.31 [A 0.89 vs E 0.57] ; gpt-6-astra_medium 0.31 [A 0.87 vs E 0.56] ; gpt-6-astra_low 0.31 [A 0.85 vs E 0.55] |
| 9 | GSO-Bench | 0.29 | gpt-6-astra_none 0.29 [A 0.54 vs E 0.83] ; gpt-6-astra 0.28 [A 0.56 vs E 0.84] ; gpt-6-astra_xhigh 0.28 [A 0.56 vs E 0.85] ; gpt-6-astra_low 0.28 [A 0.55 vs E 0.83] ; gpt-6-astra_high 0.28 [A 0.55 vs E 0.83] |
| 10 | InterCode-CTF | 0.27 | glm-4.7-flash_none (open) 0.27 [A 0.61 vs E 0.35] ; glm-4.7-flash_thinking (open) 0.25 [A 0.70 vs E 0.45] ; Baichuan-2-13B-Base (open) 0.24 [A 0.44 vs E 0.20] ; DeepSeek-Coder-V2-Base (open) 0.22 [A 0.77 vs E 0.54] ; Qwen2.5-Coder-32B (open) 0.22 [A 0.74 vs E 0.53] |
| 11 | Cybench | 0.25 | gpt-oss-120b_high (open) 0.25 [A 0.45 vs E 0.20] ; gpt-oss-120b_medium (open) 0.24 [A 0.44 vs E 0.19] ; QwQ-32B_16K (open) 0.23 [A 0.40 vs E 0.17] ; QwQ-32B (open) 0.22 [A 0.38 vs E 0.16] ; gemma-3-27b-it (open) 0.22 [A 0.29 vs E 0.07] |
| 12 | Fiction.LiveBench | 0.25 | Inkling (open) 0.25 [A 0.61 vs E 0.86] ; Inkling_thinking (open) 0.20 [A 0.66 vs E 0.86] ; Inkling_high (open) 0.19 [A 0.68 vs E 0.87] ; inkling-small_low (open) 0.18 [A 0.50 vs E 0.69] ; inkling-small_none (open) 0.18 [A 0.31 vs E 0.49] |

## A contre B

| rang | benchmark | meilleur écart | modèles par ordre d'utilité (valeur ; prédictions des modes) |
|---|---|---|---|
| 1 | MCP Atlas | 0.36 | DeepSeek-Coder-V2-Base (open) 0.36 [A 0.56 vs B 0.20] ; Qwen2.5-Coder-32B (open) 0.32 [A 0.51 vs B 0.19] ; Qwen2.5-Coder-14B (open) 0.25 [A 0.41 vs B 0.16] ; Baichuan-2-13B-Base (open) 0.25 [A 0.36 vs B 0.11] ; falcon-180B (open) 0.22 [A 0.44 vs B 0.22] |
| 2 | OpenBookQA | 0.29 | Inkling_high (open) 0.29 [A 0.61 vs B 0.90] ; Inkling_xhigh (open) 0.29 [A 0.62 vs B 0.91] ; Inkling (open) 0.28 [A 0.63 vs B 0.91] ; Inkling_thinking (open) 0.28 [A 0.63 vs B 0.91] ; gpt-4.1-nano-2025-04-14 0.27 [A 0.91 vs B 0.63] |
| 3 | AISI CTF Suites - Technical Non-Expert | 0.23 | Baichuan-2-13B-Base (open) 0.23 [A 0.54 vs B 0.31] ; DeepSeek-Coder-V2-Base (open) 0.22 [A 0.74 vs B 0.52] ; Gopher (280B) 0.22 [A 0.21 vs B 0.42] ; Qwen2.5-Coder-32B (open) 0.21 [A 0.69 vs B 0.48] ; Baichuan-2-7B-Base (open) 0.20 [A 0.49 vs B 0.29] |
| 4 | Aider Polyglot | 0.23 | DeepSeek-Coder-V2-Base (open) 0.23 [A 0.49 vs B 0.27] ; Qwen2.5-Coder-32B (open) 0.21 [A 0.44 vs B 0.24] ; Qwen2.5-Coder-14B (open) 0.17 [A 0.34 vs B 0.17] ; falcon-180B (open) 0.13 [A 0.37 vs B 0.24] ; Baichuan-2-13B-Base (open) 0.13 [A 0.22 vs B 0.09] |
| 5 | ARC AI2 | 0.22 | gpt-4.1-nano-2025-04-14 0.22 [A 0.92 vs B 0.70] ; Inkling_high (open) 0.22 [A 0.70 vs B 0.92] ; Inkling (open) 0.21 [A 0.72 vs B 0.93] ; Inkling_xhigh (open) 0.21 [A 0.72 vs B 0.93] ; Inkling_thinking (open) 0.20 [A 0.73 vs B 0.93] |
| 6 | GeoBench | 0.21 | DeepSeek-Coder-V2-Base (open) 0.21 [A 0.32 vs B 0.53] ; Qwen2.5-Coder-32B (open) 0.16 [A 0.35 vs B 0.51] ; Phi-3-small-8k-instruct (open) 0.14 [A 0.49 vs B 0.35] ; Qwen2.5-Coder-14B (open) 0.14 [A 0.31 vs B 0.45] ; Baichuan-2-13B-Base (open) 0.12 [A 0.16 vs B 0.28] |
| 7 | MATH Level 5 | 0.19 | text-davinci-002 0.19 [A 0.67 vs B 0.48] ; Qwen2.5-Coder-7B (open) 0.15 [A 0.42 vs B 0.58] ; Baichuan-2-7B-Base (open) 0.15 [A 0.45 vs B 0.60] ; Qwen2.5-Coder-3B (open) 0.15 [A 0.38 vs B 0.52] ; DeepSeek-Coder-V2-Base (open) 0.14 [A 0.63 vs B 0.77] |
| 8 | InterCode-CTF | 0.19 | Baichuan-2-13B-Base (open) 0.19 [A 0.44 vs B 0.25] ; Qwen2.5-Coder-32B (open) 0.17 [A 0.74 vs B 0.57] ; falcon-180B (open) 0.17 [A 0.67 vs B 0.50] ; DeepSeek-Coder-V2-Base (open) 0.16 [A 0.77 vs B 0.60] ; Qwen2.5-Coder-14B (open) 0.16 [A 0.64 vs B 0.48] |
| 9 | NL2Bash | 0.17 | gpt-4.1-nano-2025-04-14 0.17 [A 0.63 vs B 0.47] ; gemma-3-12b-it (open) 0.15 [A 0.43 vs B 0.29] ; gpt-5-nano-2025-08-07_high 0.15 [A 0.75 vs B 0.60] ; gpt-5-nano-2025-08-07_low 0.14 [A 0.73 vs B 0.58] ; gpt-5-nano-2025-08-07_medium 0.14 [A 0.76 vs B 0.62] |
| 10 | OTIS Mock AIME 2024-2025 | 0.16 | text-davinci-002 0.16 [A 0.34 vs B 0.18] ; DeepSeek-Coder-V2-Base (open) 0.12 [A 0.33 vs B 0.46] ; Phi-3-small-8k-instruct (open) 0.12 [A 0.38 vs B 0.25] ; Baichuan-2-13B-Base (open) 0.11 [A 0.21 vs B 0.32] ; Qwen2.5-Coder-32B (open) 0.11 [A 0.29 vs B 0.40] |
| 11 | APEX-Agents | 0.14 | DeepSeek-Coder-V2-Base (open) 0.14 [A 0.24 vs B 0.10] ; Qwen2.5-Coder-32B (open) 0.10 [A 0.18 vs B 0.08] ; falcon-180B (open) 0.10 [A 0.21 vs B 0.12] ; Baichuan-2-13B-Base (open) 0.08 [A 0.13 vs B 0.05] ; Qwen2.5-Coder-14B (open) 0.07 [A 0.14 vs B 0.06] |
| 12 | CyberGym | 0.14 | DeepSeek-Coder-V2-Base (open) 0.14 [A 0.18 vs B 0.32] ; Llama-2-34b 0.13 [A 0.16 vs B 0.29] ; tiny-recursion-model (open) 0.10 [A 0.30 vs B 0.41] ; falcon-180B (open) 0.10 [A 0.33 vs B 0.43] ; Nemotron-4 15B 0.10 [A 0.14 vs B 0.24] |

## A contre C

| rang | benchmark | meilleur écart | modèles par ordre d'utilité (valeur ; prédictions des modes) |
|---|---|---|---|
| 1 | OpenBookQA | 0.32 | Inkling_high (open) 0.32 [A 0.61 vs C 0.93] ; Inkling_xhigh (open) 0.31 [A 0.62 vs C 0.93] ; Inkling (open) 0.30 [A 0.63 vs C 0.93] ; Inkling_thinking (open) 0.30 [A 0.63 vs C 0.93] ; gpt-4.1-nano-2025-04-14 0.23 [A 0.91 vs C 0.68] |
| 2 | GPQA Main Chemistry | 0.31 | gpt-6-luna_none 0.31 [A 0.37 vs C 0.68] ; gpt-5.6-luna_none 0.29 [A 0.35 vs C 0.64] ; gpt-6-luna_low 0.28 [A 0.43 vs C 0.71] ; gpt-5.6-terra_none 0.28 [A 0.41 vs C 0.69] ; gpt-6-luna 0.27 [A 0.46 vs C 0.74] |
| 3 | CyberGym | 0.30 | DeepSeek-Coder-V2-Base (open) 0.30 [A 0.18 vs C 0.48] ; Qwen2.5-Coder-32B (open) 0.27 [A 0.12 vs C 0.40] ; Qwen2.5-Coder-14B (open) 0.24 [A 0.09 vs C 0.32] ; Baichuan-2-13B-Base (open) 0.23 [A 0.07 vs C 0.30] ; Qwen2.5-Coder-7B (open) 0.19 [A 0.06 vs C 0.24] |
| 4 | Cybench | 0.27 | DeepSeek-Coder-V2-Base (open) 0.27 [A 0.23 vs C 0.51] ; Qwen2.5-Coder-32B (open) 0.26 [A 0.16 vs C 0.41] ; Qwen2.5-Coder-14B (open) 0.22 [A 0.10 vs C 0.32] ; Qwen2.5-Coder-7B (open) 0.18 [A 0.06 vs C 0.24] ; Baichuan-2-13B-Base (open) 0.17 [A 0.08 vs C 0.25] |
| 5 | MCP Atlas | 0.27 | Qwen2.5-Coder-32B (open) 0.27 [A 0.51 vs C 0.23] ; DeepSeek-Coder-V2-Base (open) 0.27 [A 0.56 vs C 0.30] ; Qwen2.5-Coder-14B (open) 0.22 [A 0.41 vs C 0.18] ; Baichuan-2-13B-Base (open) 0.22 [A 0.36 vs C 0.14] ; text-davinci-002 0.17 [A 0.09 vs C 0.26] |
| 6 | Aider Polyglot | 0.26 | Qwen2.5-Coder-32B (open) 0.26 [A 0.44 vs C 0.18] ; DeepSeek-Coder-V2-Base (open) 0.26 [A 0.49 vs C 0.23] ; Qwen2.5-Coder-14B (open) 0.22 [A 0.34 vs C 0.12] ; Baichuan-2-13B-Base (open) 0.17 [A 0.22 vs C 0.06] ; Yi-9B (open) 0.16 [A 0.24 vs C 0.08] |
| 7 | ARC AI2 | 0.24 | Inkling_high (open) 0.24 [A 0.70 vs C 0.94] ; Inkling (open) 0.23 [A 0.72 vs C 0.95] ; Inkling_xhigh (open) 0.23 [A 0.72 vs C 0.95] ; Inkling_thinking (open) 0.22 [A 0.73 vs C 0.95] ; gpt-4.1-nano-2025-04-14 0.18 [A 0.92 vs C 0.74] |
| 8 | GPQA Diamond Chemistry | 0.24 | gpt-6-luna_none 0.24 [A 0.47 vs C 0.71] ; gpt-5.6-luna_none 0.22 [A 0.47 vs C 0.69] ; gpt-5.6-terra_none 0.21 [A 0.55 vs C 0.76] ; gpt-6-luna_low 0.20 [A 0.59 vs C 0.79] ; claude-sonnet-5_none 0.20 [A 0.53 vs C 0.73] |
| 9 | InterCode-CTF | 0.24 | gemma-3-27b-it (open) 0.24 [A 0.42 vs C 0.66] ; gpt-5-nano-2025-08-07_minimal 0.23 [A 0.37 vs C 0.60] ; gpt-4.1-nano-2025-04-14 0.20 [A 0.26 vs C 0.46] ; gpt-5-nano-2025-08-07_low 0.20 [A 0.41 vs C 0.61] ; Chinchilla (70B) 0.18 [A 0.21 vs C 0.39] |
| 10 | NL2Bash | 0.24 | gemma-3-27b-it (open) 0.24 [A 0.77 vs C 0.54] ; gpt-5-nano-2025-08-07_minimal 0.22 [A 0.72 vs C 0.51] ; gpt-4.1-nano-2025-04-14 0.21 [A 0.63 vs C 0.42] ; gemma-2-27b-it (open) 0.20 [A 0.54 vs C 0.34] ; gpt-5.4-nano-2026-03-17_none 0.20 [A 0.86 vs C 0.66] |
| 11 | GPQA Main Biology | 0.22 | gpt-6-luna_none 0.22 [A 0.61 vs C 0.83] ; gpt-5.6-luna_none 0.22 [A 0.60 vs C 0.82] ; gpt-5.4-nano-2026-03-17_none 0.21 [A 0.56 vs C 0.77] ; gpt-5.4-mini-2026-03-17_none 0.20 [A 0.60 vs C 0.79] ; gemma-3-27b-it (open) 0.19 [A 0.52 vs C 0.71] |
| 12 | AISI CTF Suites - Technical Non-Expert | 0.19 | Baichuan-2-13B-Base (open) 0.19 [A 0.54 vs C 0.35] ; Baichuan-2-7B-Base (open) 0.16 [A 0.49 vs C 0.32] ; Qwen2.5-Coder-32B (open) 0.16 [A 0.69 vs C 0.53] ; DeepSeek-Coder-V2-Base (open) 0.16 [A 0.74 vs C 0.59] ; Qwen2.5-Coder-14B (open) 0.15 [A 0.61 vs C 0.45] |

## D contre E

| rang | benchmark | meilleur écart | modèles par ordre d'utilité (valeur ; prédictions des modes) |
|---|---|---|---|
| 1 | GPQA Main Chemistry | 0.50 | Inkling_xhigh (open) 0.50 [D 0.41 vs E 0.91] ; Inkling_high (open) 0.50 [D 0.40 vs E 0.89] ; gpt-6-luna_none 0.49 [D 0.87 vs E 0.38] ; Inkling (open) 0.49 [D 0.39 vs E 0.88] ; Inkling_thinking (open) 0.48 [D 0.40 vs E 0.88] |
| 2 | OTIS Mock AIME 2024-2025 | 0.37 | inkling-small_none (open) 0.37 [D 0.25 vs E 0.62] ; inkling-small_minimal (open) 0.28 [D 0.56 vs E 0.84] ; nemotron-3-ultra_thinking (open) 0.27 [D 0.41 vs E 0.68] ; inkling-small_low (open) 0.24 [D 0.64 vs E 0.88] ; o3-pro-2025-06-10_low 0.21 [D 0.80 vs E 0.59] |
| 3 | MCP Atlas | 0.37 | gpt-5.4-nano-2026-03-17_high 0.37 [D 0.23 vs E 0.60] ; gpt-5.4-nano-2026-03-17_xhigh 0.35 [D 0.27 vs E 0.63] ; gpt-5.4-nano-2026-03-17_medium 0.35 [D 0.23 vs E 0.58] ; gpt-5.4-nano-2026-03-17_low 0.32 [D 0.23 vs E 0.54] ; gpt-5-nano-2025-08-07_high 0.31 [D 0.10 vs E 0.42] |
| 4 | ARC-AGI-2 | 0.36 | glm-5.3-flash_max (open) 0.36 [D 0.65 vs E 0.30] ; qwen3.7-plus 0.20 [D 0.20 vs E 0.41] ; qwen3.7-plus_thinking 0.19 [D 0.29 vs E 0.48] ; kimi-k2.7-code (open) 0.18 [D 0.51 vs E 0.32] ; qwen3.6-plus 0.16 [D 0.13 vs E 0.29] |
| 5 | ARC-AGI | 0.35 | glm-4.7-flash_thinking (open) 0.35 [D 0.46 vs E 0.12] ; gemma-4-26b-a4b_minimal (open) 0.26 [D 0.56 vs E 0.30] ; gpt-oss-120b_high (open) 0.21 [D 0.40 vs E 0.61] ; glm-5.3-flash_max (open) 0.21 [D 0.90 vs E 0.69] ; qwen3.6-35b-a3b (open) 0.20 [D 0.40 vs E 0.60] |
| 6 | GPQA Main Biology | 0.34 | gemma-3-27b-it (open) 0.34 [D 0.83 vs E 0.49] ; gpt-5.4-nano-2026-03-17_none 0.28 [D 0.89 vs E 0.60] ; gpt-5-nano-2025-08-07_minimal 0.28 [D 0.81 vs E 0.53] ; QwQ-32B_16K (open) 0.27 [D 0.84 vs E 0.57] ; gpt-4.1-nano-2025-04-14 0.27 [D 0.75 vs E 0.48] |
| 7 | VPCT | 0.34 | Inkling (open) 0.34 [D 0.43 vs E 0.77] ; grok-4.6_low 0.32 [D 0.57 vs E 0.89] ; inkling-small_high (open) 0.32 [D 0.43 vs E 0.74] ; inkling-small (open) 0.31 [D 0.40 vs E 0.71] ; inkling-small_xhigh (open) 0.31 [D 0.45 vs E 0.75] |
| 8 | GeoBench | 0.33 | Inkling_high (open) 0.33 [D 0.41 vs E 0.74] ; Inkling_xhigh (open) 0.32 [D 0.43 vs E 0.75] ; Inkling (open) 0.32 [D 0.42 vs E 0.74] ; Inkling_thinking (open) 0.31 [D 0.43 vs E 0.74] ; glm-5.3-flash_max (open) 0.26 [D 0.38 vs E 0.64] |
| 9 | Cybench | 0.33 | DeepSeek-Coder-V2-Base (open) 0.33 [D 0.53 vs E 0.20] ; QwQ-32B_16K (open) 0.30 [D 0.47 vs E 0.17] ; gemma-3-27b-it (open) 0.30 [D 0.37 vs E 0.07] ; QwQ-32B (open) 0.30 [D 0.46 vs E 0.16] ; Qwen2.5-Coder-32B (open) 0.29 [D 0.42 vs E 0.13] |
| 10 | GPQA Diamond Chemistry | 0.32 | gpt-5.6-luna_none 0.32 [D 0.82 vs E 0.51] ; gpt-6-luna_none 0.31 [D 0.84 vs E 0.53] ; gpt-5.6-terra_none 0.28 [D 0.87 vs E 0.59] ; gpt-5.4-nano-2026-03-17_none 0.27 [D 0.70 vs E 0.43] ; claude-sonnet-5_none 0.27 [D 0.85 vs E 0.57] |
| 11 | InterCode-CTF | 0.30 | glm-4.7-flash_none (open) 0.30 [D 0.65 vs E 0.35] ; glm-4.7-flash_thinking (open) 0.27 [D 0.72 vs E 0.45] ; Qwen2.5-Coder-14B (open) 0.23 [D 0.65 vs E 0.42] ; DeepSeek-Coder-V2-Base (open) 0.22 [D 0.77 vs E 0.54] ; Qwen2.5-Coder-7B (open) 0.22 [D 0.50 vs E 0.28] |
| 12 | MMLU Pro Chemistry | 0.30 | gpt-4.1-nano-2025-04-14 0.30 [D 0.78 vs E 0.48] ; gpt-5-nano-2025-08-07_minimal 0.30 [D 0.83 vs E 0.53] ; gpt-5.4-nano-2026-03-17_none 0.29 [D 0.91 vs E 0.62] ; gemma-3-27b-it (open) 0.27 [D 0.88 vs E 0.61] ; DeepSeek-R1-Distill-Qwen-14B (open) 0.25 [D 0.61 vs E 0.36] |

Les prédictions entre crochets sont dans l'ordre A/B/C/D/E (question « tous ») ou pour les deux modes comparés.
