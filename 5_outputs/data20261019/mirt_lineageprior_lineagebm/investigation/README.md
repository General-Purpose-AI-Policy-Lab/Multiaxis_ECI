# Sensibilité aux plafonds (`--ceiling-noise`), nuit du 20 au 21 octobre 2026

> **Depuis le 21 octobre**, ce fit est le flagship : les plafonds sont le défaut des fits K axes (`--no-ceiling` pour s'en passer) et ce dossier, d'abord `mirt_lineageprior_lineagebm_ceilnoise/`, porte le nom du flagship. Ce que ce rapport appelle « le flagship » est le fit sans plafonds, désormais `mirt_lineageprior_lineagebm_noceiling/`, et le fit joint `mirt_lineageprior_lineagebm_noceiling_humanmerge/`. Les scripts lisent les nouveaux chemins ; le texte ci-dessous garde les noms de la nuit.

Fit de sensibilité du flagship avec un plafond estimé par benchmark, d_b = 1 − e_b :
- prior : e_b ~ 0,25 · Beta(1,43 ; 4,67), la population des 1 − L de benchmark-forecasting (MBF) ;
- borne basse : le plafond ne descend jamais sous le meilleur score observé (modèles ou baselines humaines), et d_b = 1 là où quelqu'un a atteint 1,0 ;
- les plafonds connus du pipeline restent fixes.

Fit : `3_fit/fit.py --K 4 --lineage-prior --lineage-bm --ceiling-noise --chains 10 --seed 42` (skew-normal par défaut), données du 19 octobre, une seule graine.

Comparaisons :
- avec le flagship, même spécification sans plafond (20 chaînes) ;
- avec les asymptotes L du fit principal de MBF (`harvey_joint_skew_Lobs_Lceil_Lb1`, données arrêtées au 1ᵉʳ octobre).

Scripts dans ce dossier (`1_…` à `3_…`), lancés depuis la racine du dépôt.

## En bref

1. **Les plafonds améliorent nettement l'ajustement, quelle que soit la structure des axes.**
   - **LOO robuste par mélange** (le test décisif, qui couvre aussi les plafonds), à structure du flagship actuel : **+56 ± 12 nats** sur les 4 967 scores bien couverts, +47 ± 23 sur tous.
   - PSIS-LOO, à structure d'axes égale, sur les scores dont l'estimation est fiable des deux côtés (k de Pareto < 0,7) : +100 nats (erreur type 10) pour la structure du flagship actuel, +80 nats (erreur type 14) pour celle où l'agentique est séparé.
   - Sur tous les scores, le fit avec plafonds, mode principal, gagne +123 nats (erreur type 23) sur le flagship, mode principal.
   - Le gain vient surtout des QCM de connaissances : MMLU +44, Lech Mazur Writing +26, WMDP Biologie +15, CyberGym +10, MMLU-Pro Biologie +9.
   - Il y a quelques pertes, dont SimpleQA Verified −17, VisualToolBench −9 et GPQA Main Chemistry −8.
2. **Ils rejoignent ceux de MBF.**
   - Sur 80 benchmarks estimés des deux côtés, l'écart médian au plafond est le même (0,054 contre 0,055) ; corrélation des écarts 0,69 (Spearman 0,74).
   - L'accord est frappant là où les données parlent : WMDP Biologie 0,879 contre 0,878, MMLU-Pro Biologie 0,937 contre 0,934, MMLU 0,954 contre 0,949, GPQA Diamond 0,966 contre 0,962.
   - Les deux modèles partagent le même prior et la même règle de borne, donc l'accord sur les benchmarks muets (le nuage autour de 0,94) ne prouve rien.
3. **La plupart des plafonds restent ceux du prior.**
   - Seuls 8 sur 85 sont nettement informés par les données (écart type a posteriori sous 60 % de celui du prior) : MMLU, WMDP Biologie, MMLU-Pro Biologie, GPQA Diamond, MMLU-Pro, LMCA, MMLU-Pro Chimie, WeirdML v2.
   - 70 restent dominés par le prior (au-dessus de 85 %).
   - Les plafonds informés sont ceux de benchmarks dont la frontière plafonne : leur posterior se colle au meilleur score observé, qui est aussi la borne. Pour MMLU, l'écart estimé (4,6 %) est du même ordre que les erreurs d'étiquetage recensées dans la littérature, qui les estime à quelques pour cent.
4. **Le plafond inverse le classement des deux structures d'axes, sans les départager nettement.** Les deux structures existent comme modes dans les deux fits :

   | structure | flagship (sans plafond) | avec plafonds |
   |---|---|---|
   | agentique + cyber fusionnés, axe visuel (le flagship actuel) | mode A, 17 chaînes, meilleure log-densité | mode B, 2 chaînes, −23,8 nats |
   | agentique séparé, cyber séparé (l'ancien flagship) | mode B, 3 chaînes, −8,7 nats | mode A, 8 chaînes, meilleure log-densité |

   La LOO ne suit pas la log-densité. Avec plafonds, elle préfère la structure fusionnée de 22 ± 9 nats sur les scores fiables ; sans plafond, les deux sont à égalité (6 ± 7). Les modes à 2 ou 3 chaînes rendent ces écarts fragiles.
   **Meilleure combinaison en LOO : la structure du flagship actuel, avec plafonds** (elpd −16 410, contre −16 614 pour le flagship).
5. **Un axe lent tenait lieu de plafond.**
   - Dans le flagship, l'axe « connaissances scientifiques » est celui dont la frontière progresse le moins depuis 2024 : +0,8 écart-type, contre +1,9 à +2,3 pour les autres. Il porte les QCM saturés sous 1 (MMLU, WMDP, MMLU-Pro Biologie et Chimie).
   - Dans le mode principal du fit avec plafonds, ces QCM rejoignent l'axe « fluide & sciences formelles », qui progresse vite (+1,7). Leur saturation s'explique par le plafond.
   - L'axe lent qui reste (+0,4) ne porte plus que des tâches de sens commun faciles pour les humains : OpenBookQA, ARC AI2, HellaSwag, PIQA, SimpleQA, WinoGrande.

## LOO intégrée (nuit du 21 octobre)

`4_diagnostics/integrated_loo.py` intègre le bloc latent de chaque score (le niveau de sa sortie, ou la capacité d'un modèle hors lignée) au lieu de le pondérer. Je l'ai étendu au prior skew-normal et aux plafonds :
- les blocs à prior skew-normal sont intégrés par échantillonnage d'importance autour de l'ajustement de Laplace : contre une intégration exacte sur grille, l'erreur reste sous 0,01 nat en moyenne et sous 0,07 par tirage ;
- la carte des lignées est désormais figée à côté de chaque trace (`lineage_map_fitted.csv`).

Script `4_integrated_loo_compare.py` ; tableaux `diagnostics/integrated_loo_*.csv` des deux fits.

| comparaison | scores fiables (k < 0,7 des deux côtés) | tous les scores |
|---|---|---|
| plafonds B − flagship A (structure du flagship) | **+43 ± 9** | −364 ± 112 |
| plafonds A − flagship B (agentique séparé) | **+88 ± 13** | −40 ± 117 |
| flagship A − flagship B | +33 ± 6 | +146 ± 78 |
| plafonds A − plafonds B | +3 ± 8 | +178 ± 105 |

- **Sur les scores fiables, les plafonds gagnent à structure égale**, comme en PSIS.
- **Sur tous les scores, le signe s'inverse.** L'inversion vient des ~2 150 scores où la LOO intégrée n'est pas fiable (k jusqu'à 12), concentrés sur les benchmarks dont les plafonds sont appris des données (MMLU, Lech Mazur Writing, WMDP Biologie, MMLU-Pro Biologie, LMCA). La LOO intégrée intègre la capacité du modèle, pas le plafond du benchmark : quand un score fixe les deux, l'estimation n'est pas fiable, dans un sens comme dans l'autre.
- **Sans plafond, le mode A du flagship est meilleur que son mode B** : +33 ± 6 nats sur 3 567 scores fiables. Cela conforte le mode principal retenu.

### LOO robuste par mélange : la comparaison décisive

`4_diagnostics/robust_loo.py` (Silva & Zanella 2024, mélange équilibré) échantillonne un mélange des postérieurs leave-one-out et couvre tous les paramètres, plafonds compris. Il a été lancé avec 4 chaînes × 1 000 tirages, sur le mode A du flagship (départ de la chaîne 5) et sur le mode B des plafonds, de même structure d'axes (départ de la chaîne 3). Fichiers : `5_outputs/data20261019/diagnostics/robust_loo_flagA.*`, `robust_loo_ceilB.*`.

| scores comparés | plafonds − flagship |
|---|---|
| tous (5 487) | **+46,9 ± 22,6** |
| au moins 50 tirages effectifs des deux côtés (4 967) | **+55,8 ± 12,1** |
| au moins 200 (4 502) | **+57,2 ± 9,5** |

- Les 4 chaînes de mélange avec plafonds font toutes mieux que les 4 du flagship : de −16 638 à −16 657, contre −16 696 à −16 729. Chaque mélange est resté dans son mode (corrélation minimale des axes 0,91 avec plafonds, 0,997 sans).
- **Gains** : MMLU +28, Lech Mazur Writing +7, WMDP Biologie +6, GPQA Diamond +5, MMLU-Pro Biologie +4,5, CyberGym +4.
- **Pertes** : SimpleQA Verified −12, GPQA Main Chemistry −7,5, MMLU Biologie −3,5.

## Placement des humains avec plafonds

`5_place_humans.py` lit désormais le plafond d_b d'un fit `--ceiling-noise` : μ = c + (d − c)·σ(η). Comparaison à structure égale (flagship mode A contre plafonds mode B, mêmes noms d'axes, posés provisoirement), en écarts-types des modèles :

| axe | frontière moins Domain Expert, sans plafond | avec plafonds |
|---|---|---|
| Intelligence fluide & sciences formelles | −0,82 | −0,54 |
| Agentique & cyber | +1,09 | +1,06 |
| Connaissances scientifiques | **+0,63** | **−0,51** |
| Visuel & sens commun | −3,33 | −2,69 |

Avec plafonds, la frontière n'a pas encore atteint le Domain Expert sur l'axe sciences.

**Mais un artefact y contribue.** Le plafond est borné par le meilleur score observé, humains compris. Sur LMCA, ce meilleur score est celui du Domain Expert (0,87), et le plafond s'y colle (0,88). Le placement traite un score comme une simple borne basse au-dessus de 0,99 ; avec des plafonds, il faudrait le comparer au plafond d_b et non à 1. Un score humain posé sur le plafond se lit sinon comme une capacité presque illimitée. **À corriger avant toute conclusion sur les paliers.**

## Évaluation externe des plafonds (recherche en ligne, 21 octobre)

Audits publiés du taux d'erreur, statements des mainteneurs, et scores publiés hors de nos données (une borne basse du plafond). MX = ce fit, MBF = benchmark-forecasting.

| Benchmark | MX | MBF | Indice externe | Plus proche |
|---|---|---|---|---|
| ARC AI2 | 0,989 | 0,962 | Llama 3.1 405B 96,9 % (vérifié) ; o3 98,1 % (non vérifié) | MX (MBF sous des scores publiés) |
| TriviaQA | 0,931 | 0,902 | Llama 3.1 405B 91,8 % en correspondance exacte ; 3 % de réponses périmées ou fausses (EVOUNA) | MX |
| OpenBookQA | 0,964 | 0,941 | 0,952 au classement officiel ; ~0,96 en voyant toutes les options | MX |
| Lech Mazur Writing | 0,899 | 0,867 | Plafond structurel : la meilleure histoire sur 13 000 obtient 9,26/10 en moyenne des juges, soit ~0,93 | MX |
| MMLU-Pro Chimie, GPQA Main Biologie | 0,940 / 0,925 | 0,905 / 0,865 | Nos propres scores récents dépassent MBF | MX |
| MMLU Biologie (sous-ensemble RAND) | 0,927 | 0,943 | Virologie : 57 % d'items fautifs (MMLU-Redux) ; plafond du sous-ensemble ~0,86 si les modèles ratent les clés fausses | MX |
| CyberGym | 0,867 | 0,936 | 95 % au niveau 1 (NSFOCUS avec GLM-5.3, rapport du fournisseur) ; MiMo-V2.6 95 % (non vérifié) | MBF, et les deux trop bas, si ces scores sont comparables au réglage d'Epoch |
| METR Time Horizons | 0,922 | 0,950 | Version 1.1 : 15 tâches retirées, 53 corrigées ; METR : peu de tâches restent hors de portée | MBF |
| Terminal Bench 2.0 | 0,964 | 0,929 | Audit (ICML 2026) : 28 tâches défectueuses sur 89, corrigées en 2.1 (+1 à 12 points) | MBF |
| HellaSwag | 0,974 | 0,967 | 3,7 % de labels faux (annotation par LLM) : ~0,96 | légèrement MBF |
| WMDP Biologie | 0,879 | 0,878 | 90,1 % pour Gemini 3.1 Pro (rapport d'un labo, refus filtrés, modèle sans refus) | les deux trop bas, si comparable |
| LMCA | 0,880 | 0,944 | 0,85 d'Epoch = accord entre experts ; un prédicteur sans bruit irait à ~0,92–0,93 | selon la lecture : MX (plafond = niveau humain) ou MBF |
| MMLU, GPQA Diamond, MCP Atlas, SEAL Tool Use, CritPt, SimpleQA Verified | | | Audits trop larges (MMLU entre 0,935 et 0,98), réfutés par nos scores (GPQA Diamond : Epoch 0,92) ou absents | indécidable |

**Bilan.** Aucun des deux modèles ne domine. MX est meilleur sur les benchmarks qui ne sont plus évalués et partout où des scores récents dépassent MBF (7 cas). MBF est meilleur sur les benchmarks agentiques et cyber en montée rapide (3 cas), où le plafond MX est tiré par la pente (voir `ceiling_vs_slope.png`). Les deux sont trop bas là où des scores publiés hors de nos données les dépassent (CyberGym, peut-être WMDP Biologie).

**Ce que cela apprend sur la borne.** Trois des « meilleurs scores observés » qui bornent nos plafonds sont des lignes humaines mesurées dans d'autres conditions :
- SimpleQA Verified (0,944, mesuré sur le SimpleQA *original* et avec vérification des sources ; le meilleur modèle est à 0,775) ;
- OpenBookQA (0,92, une borne basse de crowdworkers) ;
- HellaSwag (0,956, items validés par des humains à la construction).

Comme LMCA, elles tiennent le plafond en l'air pour de mauvaises raisons. Cela plaide pour ne borner les plafonds que par les scores des modèles, ou par des baselines humaines mesurées dans des conditions comparables.

## Tableaux

- `ceilings_vs_mbf.csv`, `ceilings_vs_mbf.png` (script 1) : plafond, borne et L de MBF, par benchmark.
- `information_and_loo.csv` (script 2) : information apportée par les données sur chaque plafond, et gain de LOO par benchmark.
- Script 3 : progression de la frontière par axe dans chaque solution, axe porteur des benchmarks saturés, et LOO de chaque combinaison de structure et de plafond.

### LOO par combinaison (PSIS, toutes observations ; mode de chaque fit)

| fit, mode | structure | chaînes | elpd | k > 0,7 |
|---|---|---|---|---|
| flagship A | fusionnée + visuel | 17 | −16 614 | 782 |
| flagship B | agentique séparé | 3 | −16 557 | 833 |
| plafonds A | agentique séparé | 8 | −16 491 | 897 |
| plafonds B | fusionnée + visuel | 2 | −16 410 | 826 |

## Limites

- **Une seule graine et 10 chaînes** pour le fit avec plafonds ; son mode B ne repose que sur 2 chaînes (800 tirages).
- **LOO approximative.** 15 à 17 % des scores ont un k de Pareto au-dessus de 0,7, et la LOO intégrée ne couvre ni le prior skew-normal ni les plafonds. Les écarts sur les scores fiables sont les plus solides.
- **Données différentes.** MBF ajuste la trajectoire de la frontière dans le temps, sur les données du 1ᵉʳ octobre ; ce fit ajuste chaque score, sur celles du 19.
- **Divergences.** 133 sur 20 000 tirages, contre 92 pour le flagship.

## À décider

- **Placement des humains avec plafonds** : censurer les scores humains collés au plafond (les lire comme « au moins ce niveau »), ou ne borner les plafonds que par les scores des modèles. Sans l'un des deux, LMCA place le Domain Expert près de la saturation.

- **Adopter les plafonds dans le flagship.** La LOO robuste confirme le gain (+56 ± 12 nats sur les scores bien couverts), qui tient aussi à structure d'axes égale ; les plafonds informés concordent avec MBF. À faire au prochain refit, avec 20 chaînes (deux graines) pour stabiliser les modes, et après extension de la LOO intégrée au skew-normal et aux plafonds si l'on veut départager les modes proprement.
- **Le choix de structure reste ouvert.** La structure retenue le 20 octobre (agentique & cyber fusionnés, axe visuel) est aussi la meilleure en LOO avec plafonds. Mais l'écart avec la structure à agentique séparé est petit, et s'inverse selon le critère (log-densité ou LOO).

## Plafonds tenus par un score humain (21 octobre, `7_human_bound_ceilings.py`)

Le plafond ne descend jamais sous le meilleur score observé, modèle ou humain. Sur 21 benchmarks une ligne humaine dépasse tous les modèles (`human_bound_ceilings.csv`). La borne ne change le plafond que là où elle dépasse nettement ce que le prior donnerait seul (médiane 0,95, d ≥ 0,75 toujours) :

- **LMCA** : humain 0,87 (corrélation entre deux experts, sur un sous-ensemble plus facile ; ~0,84 corrigé), meilleur modèle 0,68, plafond 0,88 [0,87 ; 0,91], collé à la borne ;
- **MMLU Biology / Chemistry** (sous-ensembles RAND) : 0,898 est l'estimation du papier MMLU pour tout MMLU, pas une mesure sur ces items. Mais la figure `mmlu_subsets_saturation.png` (`8_figure_mmlu_subsets.py`) montre que les données s'arrêtent aux modèles de mi-2025 (o1, o3, o4-mini, R1 vers 0,77) et que la courbe monte encore : le plafond n'y est pas identifié, il reste près du prior (0,93 et 0,94, contre 0,944 pour MBF) et la borne à 0,898 pèse peu ;
- ailleurs (SimpleQA Verified, SimpleBench, PIQA, WinoGrande, GSM8K, OpenBookQA, HellaSwag, GeoBench), la borne tombe près de la médiane du prior et pèse peu ; sous 0,75 (The Agent Company, LitQA2, OSWorld, VISTA, BlueprintBench) elle ne pèse rien.

**Décision (utilisateur, 21 octobre)** : la borne par les scores humains reste telle quelle, LMCA compris ; elle ne pèse vraiment que sur LMCA.
