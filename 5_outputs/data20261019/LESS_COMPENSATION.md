# Une variante du flagship avec moins de compensation (23 octobre 2026)

Code sur `main` depuis le 26 octobre (options d'exploration `--theta-general`, `--simple-structure`, `--share-conc` ; commits d8ca516 et suivants). Les fits sont dans ce dossier, sous `mirt_semicomp0_lineageprior_lineagebm/` et `mirt_lineageprior_lineagebm_general_simplestruct{5,2p5}/`. Scripts d'analyse dans le scratchpad de la session (`comp/` : `diag_frontier.py`, `frozen.py`, `compare.py`, `joint_theta.py`), copiés dans `../traces_backup/less_comp_scripts/` avec le comparatif semicomp0 (`compare_sc6.txt`) ; driver du week-end : `../traces_backup/run_simplestruct_weekend.sh`, journal `../traces_backup/weekend_20261023.log`.

## Le problème, chiffré sur le mode A du flagship

- GPT-6 astra est 1ᵉʳ sur l'axe 1 et 7ᵉ en cyber, mais 294ᵉ sur 887 sur l'axe 4. Pourtant, sur SimpleQA Verified, le benchmark principal de cet axe, il fait 0,76 contre 0,75 pour gemini-3.8-flash, qui est 1ᵉʳ.
- glm-5.3-flash est 1ᵉʳ sur l'axe agentique, alors qu'il est en dessous des Claude récents et de GPT-6 sur tous les benchmarks agentiques communs (APEX-Agents 0,53 contre 0,65–0,76, DeepSWE 0,63 contre 0,73). Son 0,09 à Mystery Game Puzzles, chargé à 32 % sur l'axe 4, fait plonger θ4, et la somme compense par θ2.
- Au sein d'un même modèle, les axes sont anti-corrélés a posteriori : −0,46 entre les axes 1 et 2, environ −0,3 sur trois autres paires. C'est la signature de l'*explaining away* de la somme compensatoire, dont le canal est formé par les chargements croisés.

## Ce qui a été essayé

| variante | idée | résultat |
|---|---|---|
| `--theta-general` seul (HO-IRT, de la Torre & Song 2009) | Les abilities sont corrélées entre axes via un niveau général : λ_k·g + √(1−λ_k²)·u_k sur le bloc de base, les pas de lignée et les offsets d'effort. | Testé avec le côté benchmarks figé aux médianes du mode A. Le λ appris est faible (0,1 à 0,35), car la dérive de lignée porte déjà la co-progression. Même forcé à 0,7, rien ne bouge pour les modèles bien mesurés. **La compensation vient de la vraisemblance, pas du prior.** |
| `--link semicomp0` (soft-min sans décalage, déjà dans le dépôt) | Pénalité de déséquilibre entre axes chargés, β appris. | 6 chaînes, β ≈ 2,0. L'anti-corrélation tombe à environ −0,15, astra remonte à environ la 26ᵉ place sur l'axe 4 et glm-5.3-flash descend vers la 250ᵉ en agentique. **Mais** la pénalité ne mord que sur les modèles évalués sur des benchmarks multi-axes, c'est-à-dire la frontière. Elle les force à l'équilibre : astra tombe de 2,2 à 0,8 sur l'axe 1, derrière deepseek-v4.1-flash et muse-spark, qu'il bat pourtant sur chaque benchmark commun. C'est un biais structurel du lien, rejeté. |
| `--simple-structure` (concentration apprise) | Structure *between-item* douce (Adams, Wilson & Wang 1997) : A_b = τ·r_b·softmax(s·u_b). | Smoke test : s appris vers 0,9, part maximale médiane 0,40, exactement l'étalement du flagship. Laissée libre, la donnée garde les chargements croisés. |

## Ce qui tourne ce week-end

La structure between-item est une hypothèse de construction (un benchmark mesure d'abord une capacité). Les données éparses ne l'imposent pas d'elles-mêmes, et le confirmatoire strict à Q fixé coûtait environ 1400 nats dans l'étude géométrie. Le compromis retenu fixe la concentration et laisse apprendre l'affectation de chaque benchmark à un axe, avec le facteur général pour porter l'information entre axes. C'est le HO-IRT *between-item* de de la Torre & Song.

1. `--theta-general --simple-structure --share-conc 5`, soit une part maximale médiane a priori de 0,94 : 30 chaînes (graines 44 à 46, 2000 itérations de rodage + 1000 tirages).
2. La même avec `--share-conc 2.5` (environ 0,75) : 20 chaînes (graines 44 et 45).

Ensuite, `../traces_backup/run_simplestruct_extra.sh` fusionne des chaînes supplémentaires : graines 47 et 48 pour s = 5 (50 chaînes au total), graine 46 pour s = 2,5 (30 chaînes). Il refait ensuite les modes et la comparaison. Le rythme mesuré au démarrage est d'environ 2,4 s par itération avec 10 chaînes.

Chaque variante produit son trace fusionné, ses modes (`mirt_modes_*.json`), les abilities alignées de chaque mode et `diagnostics/frontier_compare.txt`, qui compare au mode A du flagship : rangs des modèles suivis, points de frontière par axe depuis juin 2025, anti-corrélation au sein d'un modèle.

## À regarder lundi

- `frontier_compare.txt` de chaque variante :
  - astra et sol sur l'axe 4, glm-5.3-flash et deep-think en agentique, `_none` contre `_high` en cyber, falcon-180B ;
  - et surtout que la frontière de l'axe 1 garde GPT-6, Gemini 3.1 Pro et GPT-5.x pro, le contre-exemple de semicomp0.
- Ce que coûte l'ajustement : log-vraisemblance et LOO-PIT dans le dossier du fit, comparés au flagship.
- La part maximale réelle des lignes et le nombre de modes.
- L'intégration exacte en LOO n'est pas encore faite : `integrated_loo.py` suppose le prior skew-normal indépendant, que `--theta-general` remplace.

## Résultats du week-end (lus le 26 octobre)

| | flagship, mode A | s = 2,5, mode A | s = 5, mode A |
|---|---|---|---|
| chaînes dans le meilleur mode | 21/50 | **2/30** | **2/50** |
| nombre de modes | 5 | 9 | 12 |
| log-vraisemblance (moyenne par chaîne) | ≈ −15 226 | ≈ −15 285 (−60) | ≈ −15 330 (−104) |
| part maximale médiane des lignes (L1) | 0,40 | 0,63 | 0,82 |
| anti-corrélation entre axes, paire 1-2 | −0,46 | +0,04 | +0,07 |

Rangs des modèles suivis (axes 1/2/3/4) dans le mode A à s = 2,5, flagship entre parenthèses :
- GPT-6 astra : 2/32/1/40 (1/135/7/294) ;
- Claude Opus 5.5 : 15/2/8/105 (67/16/23/141) ;
- glm-5.3-flash en agentique : 69ᵉ (1ᵉʳ) ;
- gemini-3-deep-think en agentique : 80ᵉ (3ᵉ) ;
- en cyber, `_high` repasse devant `_none` pour GPT-5.1 (259ᵉ contre 337ᵉ) ;
- falcon-180B en cyber : 396ᵉ (97ᵉ).

Les frontières :
- **Axe 1** : garde GPT-6 astra, les Gemini 3 Pro, les GPT-5.x pro, o3-pro et qwen3.8. C'est le contre-exemple qui faisait échouer semicomp0.
- **Axe 2 (agentique)** : n'a plus glm-5.3-flash, deep-think ni gpt-5.6-terra. Il ne reste que des Claude, plus gpt-5.4/5.5 et kimi-k3.
- **Axe 4** : perd les Gemini Flash 3.6 à 3.8 au profit des Gemini Pro, de GPT-5 et d'o3-pro.

**En bref.** La structure quasi between-item supprime la compensation sans le biais de semicomp0. Les axes 1 à 3 gardent leur sens (correspondance avec le flagship de 0,8 à 0,9 pour les axes 1 et 3, mais seulement 0,5 pour l'axe 2) ; l'axe 4 en garde le cœur (OpenBookQA, ARC AI2, SimpleQA, WinoGrande). Le coût est de 60 nats en log-vraisemblance à s = 2,5, contre environ 1400 pour le confirmatoire strict.

**Deux problèmes restent.**
1. Le meilleur mode n'est atteint que par 2 chaînes, parce que l'affectation des benchmarks aux axes crée beaucoup de modes discrets.
2. Les benchmarks très minces (CAISI ExploitGym, OSS-Fuzz et SEC-Bench Pro, 4 scores chacun ; SpireClimb, 7) sont affectés à l'axe 4 sans raison de contenu. Dans le flagship aussi, leurs chargements sont étalés.

## LOO intégrée face au flagship (26 octobre)

`integrated_loo.py` couvre désormais le facteur général. Les blocs d'un fondateur ou d'un modèle hors lignée y sont intégrés en (s, g). La revue du script a aussi révélé le bug du fondateur sous le prior skew-normal ; il est corrigé et testé (voir `mirt_lineageprior_lineagebm/investigation/README.md`, section du 26 octobre).

Mode A de la variante s = 2,5 (2 chaînes × 200 tirages) contre le mode A du flagship (21 chaînes × 400) :

| | elpd intégrée | scores à k > 0,7 |
|---|---|---|
| flagship, mode A | −16 718,8 | 1 797 |
| s = 2,5, mode A | −16 716,7 | 671 |
| écart, tous les scores | +2,1 ± 18,9 | |
| écart, 3 588 scores fiables des deux côtés | **−43,6 ± 10,1** | |

**Lecture.**
- Sur les scores où l'estimation est fiable, la variante prédit moins bien, de 44 nats, soit environ 0,012 nat par score. La perte se concentre sur l'axe scientifique : OTIS AIME, CritPt, MATH Level 5, MMLU, ARC-AGI.
- Sur tous les scores, c'est l'égalité. Mais ce total mêle les 1 797 scores non fiables du flagship, contre 671 seulement pour la variante, dont la structure plus simple rend la LOO bien mieux estimée.
- Supprimer la compensation a donc un coût prédictif réel, mais modeste. C'est un choix d'interprétation des axes, pas un gain d'ajustement.

## Explorer les postérieurs multimodaux (26–27 octobre)

Dans les variantes à structure simple, le meilleur mode n'est atteint que par 2 chaînes sur 30 (s = 2,5) ou sur 50 (s = 5). Les modes diffèrent par l'axe dominant de 33 à 40 benchmarks sur 107 : ce sont des organisations entières des axes. Deux pistes justifiées, absentes de l'étude d'octobre, ont été testées. Les scripts sont dans `../traces_backup/less_comp_scripts/` (`cont*.py`, `mix*.py`).

| piste | idée | résultat |
|---|---|---|
| Continuation sur la concentration, s = 0,9 → 1,2 → 1,5 → 1,9 → 2,5 (10 chaînes, environ 2 700 itérations) | Homotopie dans l'espace des modèles : on part du postérieur mou, proche du flagship, et la vraisemblance reste entière à chaque palier. | 0/10 chaînes à moins de 10 nats du meilleur mode, contre 2/30 pour NUTS seul ; meilleure chaîne à −71. Les chaînes réorganisent leurs axes à chaque palier (31 à 54 benchmarks changent d'axe) : c'est une transition de phase, comme pour le recuit. |
| Axe de chaque benchmark marginalisé (option expérimentale `axis_mixture`, 10 chaînes, 1 500 + 300 itérations) | Between-item strict à matrice Q inconnue, l'affectation sommée dans la vraisemblance (Rao-Blackwell). | Log-densités étalées sur 330 nats ; 21 à 49 affectations diffèrent entre chaînes, chacune sûre des siennes (probabilité médiane de l'axe le plus probable : 0,92 à 1,00). L'ajustement est bien moins bon que s = 2,5. L'affectation suit la configuration des abilities de chaque chaîne : la barrière est dans l'orientation des axes. |

Conclusion : comme les méthodes de l'étude d'octobre, ni l'une ni l'autre ne bat NUTS seul. Rien dans les données ne fixe l'orientation des axes. La piste standard qui reste serait de l'ancrer par le contenu : BSEM de Muthén & Asparouhov (2012), avec un axe principal par benchmark fixé par jugement et des chargements croisés à prior de petite variance.

**Décision (27 octobre, utilisateur) :** la piste est arrêtée et le flagship actuel est conservé. L'option expérimentale `axis_mixture` n'a pas été gardée dans le code ; `--theta-general`, `--simple-structure` et `--share-conc` restent comme options d'exploration.

## Réouverture : fixer l'orientation au lieu de la laisser au sampler (27 octobre)

Rouverte le jour même, avec la consigne de changer plus fondamentalement la géométrie ou la procédure d'exploration. Le diagnostic précédent dit où est la barrière : l'orientation des axes, que rien dans les données ne fixe. Deux façons standard de la fixer sans la demander au sampler, testées dans l'ordre du coût :

1. **Rotation a posteriori des tirages du flagship** (la logique de l'analyse factorielle exploratoire : on estime dans une base quelconque, on identifie après coup par un critère de structure simple ; Poworoznek, Ferrari & Dunson 2021, Papastamoulis & Ntzoufras 2022). Une rotation (A, θ) → (A T, θ T⁻ᵀ) laisse η = Aθᵀ, donc l'ajustement et la LOO, strictement inchangés : zéro coût, zéro problème de convergence. L'outillage existait déjà dans `analysis/rotation.py` (`geomin_rotate`, `promax_rotate`, `_varimax`), écrit pour le prior `signed`.
2. **Fit à structure simple avec l'orientation donnée a priori** : l'axe principal de chaque benchmark est fourni au modèle, qui penche ses parts vers cet axe (décalage sur le logit avant le softmax) tout en laissant chaque chargement croisé libre. C'est la cible douce du BSEM de Muthén & Asparouhov (2012). L'affectation vient de l'extérieur du fit échantillonné : soit de la rotation geomin du flagship (EFA puis ESEM, Asparouhov & Muthén 2009), soit du contenu (les catégories des benchmarks). Nouvelle option `--axis-pin FICHIER --pin-offset O` (tag `_pin<label>-<offset>`, tests dans `tests/test_axis_pin.py`), fichiers dans `pins/`.

### 1. Rotation du mode A du flagship

Une seule transformation globale par critère, calculée sur les chargements moyens du mode A (21 chaînes) et appliquée à chaque tirage ; scripts `rot.py`, `compare.py`, `qcheck.py` dans `../traces_backup/pin_scripts/`.

| | flagship | varimax (orthogonale) | geomin (oblique) | promax (oblique) | fit s = 2,5, mode A |
|---|---|---|---|---|---|
| part maximale médiane des lignes | 0,39 | 0,48 | 0,50 | 0,53 | 0,63 |
| cellules < −0,05 | 0 | 11 % | 14 % | 27 % | 0 |
| anti-corrélation intra-modèle, paire 1-2 | −0,46 | +0,18 | −0,03 | −0,06 | +0,04 |
| astra, rangs axes 1/2/3/4 | 1/135/7/294 | 2/794/17/430 | 1/238/3/**5** | 1/572/1/1 | 2/32/1/40 |
| glm-5.3-flash, agentique | 1ᵉʳ | 4ᵉ | **2ᵉ** | 4ᵉ | 69ᵉ |
| accord de l'argmax des lignes avec le mode A à s = 2,5 | 65 % | | 75 % | | 100 % |

Geomin est la bonne rotation (promax rend trop de chargements négatifs, varimax vide l'axe 2). Elle supprime les anti-corrélations intra-modèle et répare l'axe 4 (SimpleQA Verified devient pur, 1,51 sur l'axe 4 et moins de 0,3 ailleurs ; astra 5ᵉ), et ses axes restent ceux du flagship (cosinus 0,83 à 0,95). Mais elle ne répare pas l'axe agentique : OSWorld reste chargé 1,23 sur l'axe 2 et 0,92 sur l'axe cyber (contre 1,94 / 0,24 dans le fit s = 2,5), et ARC-AGI / ARC-AGI-2 gardent 1,0 à 1,5 sur l'axe 2, ce qui maintient glm-5.3-flash et deep-think en tête. Le sous-espace de chargements du flagship (de dimension 4) ne contient pas de direction agentique propre ; le fit s = 2,5 change ce sous-espace, et c'est ce que coûtent ses 60 nats de vraisemblance. La rotation est donc la moitié bon marché de la réponse : elle ne peut pas remplacer un fit à structure simple, mais elle en fournit l'orientation (75 % des affectations du meilleur mode, les désaccords portant sur les lignes minces ou étalées).

### 2. Fits épinglés (lancés le 27 octobre à 12 h 34, `../traces_backup/run_pin.sh`, journal `pin_20261027.log`)

Sur `--theta-general --simple-structure --share-conc 2.5`, graine 44, 10 chaînes, 2000 + 1000 itérations, environ 3 h 30 chacun :

1. `_pingeomin-1p5` : affectation = argmax des chargements du flagship après geomin (107 benchmarks), décalage 1,5 (part maximale 0,93 a priori) ;
2. `_pincategory-1p5` : affectation par le contenu, 72 benchmarks sur 107 (Mathematics, Chemistry, Biology → 1 ; Autonomous SWE, Agentic Computer Use → 2 ; Cyber → 3 ; Trivia & Commonsense QA → 4 ; les catégories General Reasoning, Multimodal, Domain Specific, Core AGI et Writing restent libres) ;
3. `_pingeomin-1` : décalage 1 (0,80 a priori), pour mesurer ce que le prior impose.

À lire dans `diagnostics/` de chaque dossier : `modes.log` et `mode_chains.txt` (une seule organisation ?), `chain_loglik.txt` (log-vraisemblance par chaîne, à comparer aux ≈ −15 285 du mode A à s = 2,5 et aux ≈ −15 226 du flagship), `frontier_compare.txt` (rangs, frontières, anti-corrélations, face au flagship et au mode A à s = 2,5). Le critère de succès : les chaînes d'un fit épinglé se retrouvent dans un seul mode, au niveau de vraisemblance du mode A que NUTS seul n'atteignait que 2 fois sur 30, avec les mêmes frontières ; et les deux épinglages, données et contenu, s'accordent.
