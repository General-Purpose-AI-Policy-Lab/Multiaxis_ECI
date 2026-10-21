# Étude des échantillonneurs : rapport final (8 → 12 octobre 2026)

## La question

Le flagship de Multiaxis_ECI (MIRT compensatoire à K = 4, ajusté sans les baselines humaines, prior de lignée brownien) a un postérieur multimodal. Son fit long, 20 chaînes nutpie, s'est réparti en cinq modes, A à E. A a la meilleure log-densité : log-vraisemblance −14 385, log-postérieur −21 124 en moyenne (−21 108 pour une chaîne typique). La question posée : quel échantillonneur trouve le ou les meilleurs modes de façon fiable ?

Critère fixé par l'utilisateur : une chaîne réussit si sa log-vraisemblance moyenne à T = 1 dépasse −14 395. Atteindre le niveau de A (−14 385) ou mieux est l'objectif. Chaque chaîne est notée sur ses 150 derniers tirages, et sa famille est donnée par la correspondance de ses loadings avec les modes NUTS (minimum des |corr| après permutation des axes).

## Réponse courte

1. **Aucune méthode d'exploration ne bat NUTS seul.**
   - Recuit, recuit avec sélection, recuit « population », parallel tempering et SMC tempéré ont tous été essayés.
   - À budget égal (environ 2 100 itérations par chaîne), NUTS seul met environ 35–40 % des chaînes au-dessus du seuil.
   - Le meilleur recuit en met 23 %, la recette « zone critique » de base 13 %.
2. **A n'est pas un optimum isolé.** Plusieurs solutions d'axes, proches des modes D, E, B et parfois C, atteignent la vraisemblance de A :
   - log-vraisemblance de −14 363 à −14 385 ;
   - log-postérieur de −21 105 à −21 125 pour les meilleures chaînes, au niveau de A.
   - Les tests de stabilité (10 chaînes relancées depuis un état) confirment au moins deux de ces familles (proches de D et de E) : médiane −14 389/−14 390, 6 chaînes sur 10 au-dessus du seuil.
3. **Le bassin de A lui-même n'est presque jamais retrouvé en run court.**
   - Sur environ 300 chaînes de 2 000 à 4 000 itérations, une seule (nutpie) correspond aux loadings de A à plus de 0,8.
   - Le fit long nutpie en avait mis 4 sur 20 dans A.
4. **Recommandation pour le flagship :**
   - garder NUTS (nutpie) avec beaucoup de chaînes plutôt qu'un échantillonneur d'exploration ;
   - surtout, traiter le choix du mode comme une question de modèle : comparer A et les familles D/E/B par LOO intégré, et voir si les noms d'axes, les forecasts et les tiers humains en dépendent.

## Méthodes et résultats

### Premier tour (8–9 octobre, dans `/tmp`, effacé par le redémarrage du 9)

| Méthode | Résultat | Enseignement |
|---|---|---|
| Pathfinder (pymc-extras) | abandonné | les chemins L-BFGS s'arrêtent après environ 20 itérations ou plantent |
| Recuit géométrique (16–30 paliers, T₀ = 10⁻³–10⁻⁴) | 0 à 2 chaînes sur 10 | les premiers « gains » venaient d'un défaut : le bruit ajouté par PyMC à chaque palier ; puis d'états pas encore équilibrés à T = 1 |
| Recuit « zone critique » (paliers longs à T = 0,03–0,4, 3 pires chaînes remplacées par les 3 meilleures sur la moyenne du palier) | 3, 2, 2 sur 10 | le meilleur du premier tour |
| keep-best 5 / rééchantillonnage amorti / population | 1/10, 0/10, (2ᵉ tour) 0/10 | une sélection plus forte réduit la diversité |
| Parallel tempering (48 répliques de T = 0,1 à 1, HMC 64 sauts) | aucun aller-retour en 4 500 itérations | échanges à 45–50 % vers T = 0,1–0,2 mais 7 % près de T = 1 ; il faudrait beaucoup plus de répliques |
| SMC tempéré, essais 1 et 2 | échec | mutation trop faible ; puis particules piégées par φ_b < 0 (défaut du modèle, voir plus bas) |

### Deuxième tour (9–12 octobre, `~/sampler_study`, une tâche à la fois)

Taux de réussite par méthode, toutes graines confondues. Les familles listées sont celles des chaînes au-dessus du seuil.

| Méthode (budget par chaîne) | Runs | Chaînes au-dessus de −14 395 | Par run | Meilleure log-vraisemblance | Meilleur log-postérieur | Familles au-dessus du seuil |
|---|---|---|---|---|---|---|
| NUTS PyMC, 1 800 réglages + 300 tirages | 5 (+1 au tour 1) | **20/50** (+1/10) ≈ 35 % | 5, 2, 7, 3, 3 (+1) | −14 363 | −21 105 | E 7, D 6, B 5, C 2 |
| NUTS PyMC, 4 000 + 300 | 1 | 5/10 | 5 | −14 354 | −21 109 | D 4, B 1 |
| nutpie, 1 800 + 300 | 3 | **11/30** ≈ 37 % | 4, 4, 3 | −14 370 | −21 111 | B 5, D 4, E 1, A 1 |
| nutpie, 4 000 + 300 | 1 | 4/10 | 4 | −14 370 | −21 119 | D 2, A 1, C 1 |
| Recuit zone critique, base (T = 0,03–0,4) | 4 (+3 au tour 1) | 2/40 (+7/30) ≈ 13 % | 0, 0, 0, 2 (+3, 2, 2) | −14 363 | −21 118 | B, E |
| Recuit zone resserrée (T = 0,04–0,2) | 3 | 7/30 ≈ 23 % | 4, 2, 1 | −14 366 | −21 123 | B 3, C 2, D 1, E 1 |
| Recuit keep-best 2 | 1 | 2/10 | 2 | −14 352 | **−21 083** | D, E |
| Recuit paliers critiques doublés | 1 | 0/10 | 0 | −14 396 | −21 153 | — |
| Recuit rééchantillonnage amorti | 1 | 0/10 | 0 | −14 399 | −21 146 | — |
| Recuit zone élargie (T = 0,02–0,6) | 1 | 1/10 | 1 | −14 395 | −21 143 | E |
| Recuit population (30 chaînes arrêtées à T = 0,4, les 10 meilleures finies) | 1 | 0/10 | 0 | −14 397 | −21 138 | — |
| SMC tempéré corrigé (256 particules, mutation 50 × 20) | 1 | 26/256 → 36/256 après rajeunissement | — | −14 302 → −14 257 | — | particules non équilibrées (médiane −14 491 → −14 449) |

Les temps par run sont similaires : environ 1 h 05 pour NUTS PyMC ou nutpie à 1 800 réglages, 1 h 30–1 h 40 pour un recuit zone critique, 2 h 05–2 h 25 pour les témoins à 4 000 réglages. Le SMC a demandé 11 h 30 jusqu'à λ = 1 et 45 minutes de rajeunissement.

### Tests de stabilité : 10 chaînes NUTS relancées depuis un état prometteur

| Depuis | Avant | Après (10 chaînes, 1 000 + 300 à 1 000 + 400) | Lecture |
|---|---|---|---|
| hot_slow2, chaîne 5 (tour 1) | −14 365 | −14 392 à −14 458, famille D | état transitoire, retombe dans D |
| crit_select, chaîne 5 (tour 1) | −14 357 | 6/10 au-dessus du seuil, médiane −14 389, log-postérieur ≈ −21 140, famille E (0,78–0,81) | **vrai mode, au niveau de A** |
| crit_kb2, chaîne 4 | −14 352 / −21 083 | 6/10 au-dessus du seuil, médiane −14 390, −21 128 à −21 170, famille D (0,70–0,76) | **vrai mode, au niveau de A** ; le pic était transitoire (≈ −40) |
| les 10 meilleures particules du SMC | −14 272 à −14 346 | −14 423 à −14 465, proches de A mais correspondance ≤ 0,61 | pics non équilibrés |

Ce que montrent ces tests : une chaîne qui sort d'un recuit, d'un SMC ou d'une phase courte affiche souvent une vraisemblance de 30 à 100 trop haute. Elle n'a pas encore exploré tout le volume de son mode. Une comparaison n'a donc de sens qu'après une phase longue à T = 1, moyennée sur beaucoup de tirages (la log-vraisemblance varie de ±60 d'un tirage à l'autre).

## Ce que ça dit des modes

- **Plusieurs solutions d'axes se valent presque.** Les meilleures chaînes des contrôles NUTS ont une log-vraisemblance de −14 363 à −14 374 et un log-postérieur de −21 105 à −21 120 (control_nuts4, control_nuts6, nutpie). C'est égal ou supérieur à la moyenne de A (−21 124), et dans l'étendue de ses propres chaînes (−21 107 à −21 171). Elles appartiennent aux familles E, D et B.
- **Les familles D et E stabilisées** restent 20 à 40 sous la chaîne typique de A en log-postérieur. A reste donc légèrement en tête, sans être le seul candidat crédible.
- **Le bassin de A est rarement atteint** en run court. Il demande l'adaptation longue d'un fit complet, ou il est simplement étroit.

## Défauts trouvés et corrigés en route

1. **Domaine de la précision du bruit** (`2_model/multiaxis_eci/models/mirt.py`, **non commité dans le dépôt**).
   - φ_b = 1/(4σ_b²) − 1 devient négatif dès que σ_b dépasse 0,5. La bêta-binomiale écrite à la main renvoyait alors environ +37 par observation au lieu de −∞.
   - La traduction JAX de PyMC désactive en plus toutes ses vérifications de paramètres, et la Beta censurée plafonnait −∞ à −700.
   - La correction ajoute une vérification a, b > 0 dans la bêta-binomiale, plus un terme `noise_domain` qui vaut −∞ si φ ≤ 0, sur tous les chemins de vraisemblance.
   - Aucun fit valide ne change : les 471 tests passent, log-densités de référence comprises.
2. **Lecture des loadings dans les scripts SMC et PT.** `model.compile_fn(model["A"])` retirait au hasard les variables dont A dépend, si bien que la correspondance aux modes calculée pour les particules était du bruit. Corrigé avec `pytensor.function(..., replace_rvs_by_values(...))`.
3. **Bruit de PyMC sur les états hérités.** PyMC ajoute ±1 aux valeurs de départ qu'on lui fournit, ce qui brouillait chaque palier de recuit. Corrigé avec `init="adapt_diag"`.

## Recommandations pour le flagship

1. **Garder NUTS.** Pour mieux couvrir les modes, augmenter le nombre de chaînes et la longueur de l'adaptation plutôt que changer d'échantillonneur. nutpie vaut PyMC NUTS par chaîne (37 % contre 35–40 %) et adapte plus vite.
2. **Comparer explicitement les meilleures familles.** A et les variantes de D, E et B au niveau de A :
   - LOO intégré, comme pour le choix A contre B ;
   - effet sur les noms d'axes, les forecasts et le placement 2b des tiers humains ;
   - si les conclusions de fond sont stables, le choix du mode compte peu ; sinon, il faut le documenter, voire moyenner les modes (empilement).
3. **Si l'on veut quand même le SMC :** il atteint λ = 1 une fois le modèle corrigé, mais ses particules ne sont pas équilibrées. Il faudrait une mutation bien plus lourde près de λ = 1, ou un rajeunissement beaucoup plus long, pour un coût de plusieurs dizaines d'heures. Je ne le recommande pas.

## Déroulé et incidents

- **9 octobre au soir** : redémarrage de la machine (mise à jour de macOS 27), qui a vidé `/tmp`. L'étude a été reconstruite dans `~/sampler_study`, en dehors de ~/Documents (macOS bloque l'accès des tâches launchd à ce dossier), avec un lanceur qui reprend où il s'est arrêté et une tâche launchd.
- **Points de contrôle programmés** : le premier est resté bloqué sur une autorisation ; après validation, ils n'ont tourné que le 10 à 12 h 27 et le 12 à 22 h 25. La machine dormait sans doute une fois le lanceur arrêté.
- **Suivi depuis la session principale** : interrompu le 11 vers 10 h 45. La file s'est vidée le 11 à 22 h 58 sans nouvelles tâches, soit environ 24 h de machine inutilisée.
- **Aucune tâche n'a échoué.** Les résultats bruts sont dans `REPORT_auto.md`, `NOTES.md`, `runs/*/chains.csv` et `logs/`.

## Nettoyage

- Tâche launchd : `launchctl unload ~/Library/LaunchAgents/org.gpaipolicylab.sampler-study.plist`, puis supprimer ce fichier.
- `~/sampler_study` (environ 2,5 Go, dont l'instantané `mx/` et les états `final.nc`) peut être supprimé une fois ce rapport conservé, copié dans `Multiaxis_ECI/5_outputs/sampler_study/`.
- Routines « Étude échantillonneurs » (point de contrôle, rapport final) : à supprimer dans la section Routines.
- À commiter dans le dépôt si on la garde : la correction du domaine de φ dans `models/mirt.py`.

---

## Nuit du 12 au 13 octobre : réflexion théorique, modèle rapide, « optimiser puis échantillonner »

### Diagnostic (night/THEORY.md)

Sur la trace du flagship, les modes A–E prédisent **presque les mêmes scores observés** : écart RMS de 0,045 à 0,13 sur les
logits des 5 113 cellules observées, pour une dispersion de 1,75, et des corrélations ≥ 0,997. Leurs sous-espaces de loadings
(98 × 4) et de capacités (850 × 4) sont pourtant très différents : une direction commune (le facteur général, angle de 1 à 4°),
puis des angles principaux allant jusqu'à 23–45° et 48–90° pour les autres. Ce ne sont pas des rotations d'une même solution. Ce
sont des **complétions différentes d'une matrice observée à 6 %** (médiane de 4 scores par modèle, données manquantes
structurées par l'époque). Elles ajustent également bien les cases observées et ne diffèrent que sur les cases jamais
observées, celles que lisent les capacités par axe.

Conséquences :
- c'est un problème d'**identification** avant d'être un problème d'échantillonnage ;
- passer d'une complétion à une autre demande de déplacer ensemble des blocs de benchmarks et les capacités de tous les
  modèles concernés, ce qu'aucun échantillonneur local ne fait ;
- les méthodes tempérées butent sur une transition du premier ordre (le saut de vraisemblance vers T ≈ 0,06–0,1).

### Modèle rapide (JAX, même vraisemblance, bruit fixé, sans prior de lignée) : optimisation depuis de nombreux départs

| variante | départs | écart de −log-postérieur entre les 20 meilleurs optimums | optimums distincts parmi les 20 meilleurs |
|---|---|---|---|
| base (K = 4, loadings positifs) | 98 | 18,7 | 20/20 |
| loadings signés (rotation libre ; distincts si angle principal > 20°) | 60 | 16,6 | 20/20 |
| K = 3 | 60 | 48,6 | 18/20 |
| vraisemblance gaussienne sur les logits | 60 | 38,9 | 20/20 |

Le paysage est **vitreux** : une multitude d'optimums locaux distincts de qualité presque égale, et non quelques modes. Ni la
positivité des loadings, ni K = 4 plutôt que 3, ni la forme de la vraisemblance n'en sont la cause, ce qui conforte le
diagnostic : c'est la rareté des cases observées. Les optimums du modèle rapide ne ressemblent pas aux modes du flagship
(correspondance de 0,1 à 0,47) : le prior de lignée façonne les solutions.

### Modèle complet : MAP depuis de nombreux départs, puis NUTS depuis les 10 meilleurs optimums

- Le MAP du modèle complet est très rapide : 60 départs en 102 s, 200 en 340 s (L-BFGS, gradients JAX).
- Il **surajuste** : log-vraisemblance de −11 262 à −12 200 aux optimums, contre −14 385 dans le postérieur. Aucun optimum ne
  correspond à un mode NUTS à plus de 0,72.
- NUTS lancé depuis les 10 meilleurs optimums (3 runs, même budget que les témoins) :
  - **14 chaînes sur 30 restent piégées** dans des pics d'effondrement du bruit : log-vraisemblance de −11 078 à −13 434,
    log-prior d'environ −4 900 (contre −6 750 dans le postérieur ordinaire), soit 3 000 à 5 000 points de log-postérieur
    au-dessus des modes ordinaires ;
  - parmi les chaînes ordinaires, seulement **4/16 (25 %) dépassent −14 395**, contre **23/60 (38 %) pour NUTS seul**
    (control_nuts2 à 7 : 5, 2, 7, 3, 3, 3).
  - « Optimiser puis échantillonner » ne bat donc pas NUTS seul.

### Deuxième défaut du modèle : effondrement du bruit par benchmark

Dans les chaînes piégées, σ_b tend vers 0 (≤ 10⁻³) pour de nombreux benchmarks, y compris notés par items (GPQA Diamond,
HellaSwag, ARC-AGI). Le prior hiérarchique sur σ_b dérive avec : moyenne log ≈ −5, dispersion de 1,6 à 2,9, contre environ
−3,4 et 0,7 dans les chaînes ordinaires. Le postérieur tel que spécifié contient donc des **pics de densité bien plus hauts
que les modes que NUTS échantillonne**. Partie d'un point quelconque, une chaîne NUTS ne les trouve pas ; l'optimisation,
si (le code garde la trace d'un épisode semblable : Chess Puzzles, σ_b = 4·10⁻⁴). Ce n'est pas un défaut numérique comme
φ_b < 0 : c'est une propriété du modèle. Il faudrait borner σ_b par en dessous (par exemple σ_b ≥ 0,01, ou un prior plus serré
sur la moyenne et la dispersion de log σ_b), puis vérifier que les résultats du flagship n'en dépendent pas.

### Bilan

1. **Aucune méthode testée ne bat NUTS seul.** C'est vrai pour le recuit, le parallel tempering, le SMC et « optimiser puis
   échantillonner ».
2. **La multimodalité vient de l'identification** : une matrice observée à 6 %, avec une multitude de complétions de rang 4
   aussi bonnes les unes que les autres sur les données. Les remèdes sont du côté du modèle et des données :
   - une structure plus confirmatoire des loadings ;
   - un plancher sur σ_b ;
   - des mesures ciblées : voir `data_wishlist.md`, alternatives classées par question (refaite le 21 octobre pour les modes A et B du flagship avec plafonds, par `data_wishlist.py` ; la version du 12 octobre est dans l’historique git) ;
   - et, en attendant, rapporter la sensibilité des conclusions au mode choisi.
3. **Le modèle a des pics d'effondrement du bruit** : à corriger avant tout nouveau fit de référence.
