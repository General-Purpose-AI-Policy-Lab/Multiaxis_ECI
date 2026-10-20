# Géométrie du postérieur et identification des axes (13–16 octobre 2026)

Synthèse des explorations sur le modèle rapide (banc) et sur le flagship. Les scripts et les sorties
brutes sont dans `~/sampler_study/night/`. Le flagship a été testé sur une copie du dépôt
(`~/sampler_study/mx`), avec un interrupteur d'étude (`MX_THETA_SKEW`) : **le dépôt n'est pas modifié**.

## En bref

1. **Un prior skew-normal indépendant par axe sur les capacités est la seule modification qui améliore à la fois la convergence et l'ajustement.**
   - Sur le flagship, avec le pipeline actuel (nutpie, matrice de masse low-rank, 1 800 + 300 itérations, trois graines appariées), il met **14 chaînes sur 30 au-dessus de −14 395, contre 4 sur 30** pour le prior actuel (test exact de Fisher, p ≈ 0,01).
   - Les chaînes se ressemblent davantage : correspondance médiane entre paires de 0,82, contre 0,42.
   - Il y a moins de divergences, les runs sont environ 20 % plus rapides, et la meilleure chaîne atteint une log-vraisemblance de −14 341 (mode A : −14 385).
2. **Ce n'est pas un artefact du prior.**
   - Sur cases retenues (5 % des scores observés, deux découpages), le skew-normal prédit mieux que le prior actuel (+7,2 et +2,5 de densité prédictive).
   - Sur le flagship, sa LOO est meilleure pour chacune des trois graines : elpd −15 434 / −15 465 / −15 446, contre −15 537 / −15 512 / −15 506 pour les témoins low-rank appariés, soit un écart de 60 à 100.
   - La structure apprise vient des données : le prior sur α est symétrique, et pourtant 24 chaînes sur 24 du banc aboutissent au même motif, 3 axes asymétriques à gauche et 1 axe symétrique.
3. **Il ne règle pas tout.**
   - Il fixe surtout l'**orientation des axes**. La multiplicité des **complétions** de la matrice (désaccord des chaînes sur les cases jamais observées) n'est pas clairement réduite sur le banc : RMS de 0,12 avec une graine, 0,27 avec l'autre, contre 0,14 pour le témoin.
   - Sur le flagship, les asymétries apprises varient davantage d'une chaîne à l'autre, car la lignée produit déjà une part de l'asymétrie.
4. **La matrice de masse low-rank du pipeline est préférable à la diagonale sur le flagship.**
   - Elle donne 127 pas par itération contre 1 023 (saturation) et 36 divergences contre 61.
   - Le banc, qui n'a ni lignée ni hyperparamètres en entonnoir, disait l'inverse.

## Pourquoi les modes existent

- Le modèle ne voit que θAᵀ.
- Avec des capacités N(0, I) et des chargements HalfNormal, **toute rotation laisse la vraisemblance et le prior invariants**. Seule la positivité des chargements, plus, sur le flagship, la dérive et la diffusion de lignée propres à chaque axe, fixe l'orientation.
- Les chaînes s'arrêtent alors dans des « coins » différents du cône positif. Pour passer d'un coin à l'autre, certains chargements doivent descendre vers zéro, ce qui forme un long entonnoir en coordonnées logarithmiques.
- Environ deux tiers des « familles » de chaînes partagent le même sous-espace de chargements. Le dernier tiers diffère réellement par la 4ᵉ direction, la moins bien déterminée : ce sont les complétions multiples de la matrice, observée à 6 %.
- Intégrer les capacités (modèle collapsé) rend les optimums discrets et reproductibles, environ une douzaine. La plupart des optimums du modèle joint ne sont que des réajustements de θ.

## Ce qui a été essayé

| piste | type | résultat |
|---|---|---|
| CNC (θ = θ̂(A) + L(A)·z, exact) | échantillonnage | meilleur bassin d'attraction, mais pas de passage d'un mode à l'autre ; coût ×40 sur le flagship |
| sauts de mode par translation | échantillonnage | log-acceptation d'environ −10⁴ |
| mouvements de groupe ou de rotation | échantillonnage | rotations réalisables seulement sous 0,01 rad (chargements quasi nuls) |
| réalignement a posteriori | post-traitement | 0 rotation réalisable sur 1 000 tirages du flagship ; avec tolérance, 8 → 7 familles |
| matrice de masse low-rank / diagonale | échantillonnage | low-rank meilleure sur le flagship |
| chargements repliés A = \|z\| | paramétrisation | léger mieux seul, rien en combinaison |
| prior par famille de benchmarks, ancres, confirmatoire | modèle | n'unifie pas la marginale ; le confirmatoire strict coûte de la vraisemblance |
| parcimonie Gamma(α < 1) | modèle | entonnoirs (jusqu'à 1 365 divergences), désaccord accru |
| capacités corrélées (R libre) | modèle | rouvre la liberté de rotation (12 directions au lieu de 6) ; seule ou combinée, plus de familles |
| Student-t(4) indépendant | modèle | meilleur mélange, mais chaînes en désaccord sur les prédictions |
| **skew-normal indépendant** | **modèle** | **meilleur sur tous les critères** (voir plus haut) |
| skew-normal + corrélation positive à un facteur | modèle | banc : comparable au skew-normal seul (2–3 familles), sans gain net ; flagship : en cours |

## Données empiriques sur l'asymétrie

Figure : `figures/ability_distribution_modeA.png`.
- Les capacités des 850 modèles ont une **queue à gauche** : beaucoup de modèles anciens ou faibles loin sous la frontière, et un plafond à droite. L'asymétrie est la plus forte sur l'axe agentique (−1,5) ; l'axe 1, intelligence fluide, est quasi symétrique.
- Les incréments de lignée et les offsets sont gaussiens (kurtosis 3,03–3,11) : rien ne justifie des sauts à queue lourde.
- Le raisonnement n'apparaît pas comme un saut dans les lignées. Il passe par la direction d'effort, presque entièrement sur l'axe 1, et par de nouvelles lignées (`o-*`, `deepseek-r`, `magistral`, `qwq`).

## Tableaux

### Flagship (nutpie low-rank, 1 800 + 300, 10 chaînes par run)

| run | graine | chaînes au-dessus de −14 395 | log-vraisemblance médiane | meilleure | elpd LOO |
|---|---|---|---|---|---|
| témoin diagonal | 201000 | 4/10 | −14 411 | −14 374 | −15 491 |
| témoin low-rank | 201000 | 1/10 | −14 439 | −14 376 | −15 537 |
| skew-normal | 201000 | 5/10 | −14 393 | −14 377 | −15 434 |
| témoin low-rank | 202000 | 0/10 | −14 425 | −14 408 | −15 512 |
| skew-normal | 202000 | 4/10 | −14 405 | −14 341 | −15 465 |
| témoin low-rank | 203000 | 3/10 | −14 412 | −14 354 | −15 506 |
| skew-normal | 203000 | 5/10 | −14 399 | −14 339 | −15 446 |
| **total low-rank** | | **témoin 4/30, skew-normal 14/30** | | | |

Divergences par run, témoin contre skew-normal : 36 contre 24 (graine 201000), 32 contre 25 (202000) et 83 contre 23 (203000). Le skew-normal reste à 127 pas par itération sur toutes ses chaînes ; le témoin sature parfois à 511 ou 1 023.

La LOO est approximative : environ 15 % des cellules ont un k de Pareto au-dessus de 0,7, car il n'y a que 500 tirages répartis sur plusieurs modes. Seules les comparaisons relatives sont à lire.

### Banc, cases retenues (5 % des scores observés, 12 chaînes)

| découpage | témoin | skew-normal | skew + un facteur (g gaussien) | skew + un facteur (g asymétrique) |
|---|---|---|---|---|
| 0 | −870,9 | −863,7 | −861,3 | −865,7 |
| 1 | −811,3 | −808,8 | −810,4 | −813,4 |

## Nuit du 15 au 16 octobre

Sur batterie à partir de minuit, la file a été réordonnée : troisième paire flagship (graine 203000) d'abord, puis skew-normal + un facteur sur le flagship. Journal : `~/sampler_study/night/logs/night_1015.txt`.

- La paire de la graine 203000 a pu se terminer. Le témoin a été interrompu par la veille de 3h47 à 9h47 (batterie vide) puis a repris. Résultat : skew-normal 5/10 contre témoin 3/10, LOO −15 446 contre −15 506.
- Le skew-normal avec corrélation à un facteur sur le flagship n'a pas pu passer pendant la nuit. Il tourne à la suite le 16 au matin.

## Suite donnée

- **Option `--theta-skew` ajoutée au dépôt le 16 octobre**, désactivée par défaut. Elle passe par `build_mirt_model(..., theta_skew=True)`, `config.PRIOR_THETA_SKEW_ALPHA`, et apparaît dans `docs/model_math.md` §5.4 et dans `docs/cli.md`. C'est la version testée ici : skew-normal standardisée par axe sur le bloc partagé, sans recentrage exact. À évaluer au refit en LOO face au fit phare, au même titre que `--ceiling-noise`.
- **Le skew-normal est devenu le défaut des fits K-axes le 20 octobre** (`FitSpec.theta_skew`, `--normal-theta` pour revenir au bloc gaussien), après le refit sur les données du 19 octobre : 2 modes au lieu de 7, dont le meilleur atteint par 17 chaînes sur 20. L'indice canonique K=1 garde le bloc gaussien.
- Garder la matrice de masse low-rank.
- Ne pas poursuivre CNC, les mouvements de rotation, la parcimonie Gamma ni la corrélation libre.
- Skew-normal + corrélation positive à un facteur (θ_k = λ_k·g + √(1 − λ_k²)·s_k) : pas de gain par rapport au skew-normal seul.
  - Sur le banc, les corrélations apprises sont faibles (0,02 à 0,15). La variante avec un niveau général g lui-même asymétrique est moins bonne.
  - Sur le flagship, le modèle se sert du niveau général (λ de 0,53 à 0,80), mais :

    | graines 201000 / 202000 | skew-normal seul | skew + un facteur |
    |---|---|---|
    | chaînes au-dessus de −14 395 | 5/10 et 4/10 (9/20) | 2/10 et 5/10 (7/20) |
    | meilleure log-vraisemblance | −14 377 / −14 341 | −14 390 / −14 384 |
    | correspondance médiane entre paires de chaînes | 0,82 / 0,64 | 0,46 / 0,73 |
    | elpd LOO | −15 434 / −15 465 | −15 481 / −15 450 |

    Les résultats sont mitigés ou un peu moins bons. Le skew-normal seul reste la recommandation.

La figure citée plus haut (`figures/`) n'est pas suivie par git (règle `5_outputs/**/figures/`). Elle se régénère avec le script de la session, dans `~/sampler_study/night/`.
