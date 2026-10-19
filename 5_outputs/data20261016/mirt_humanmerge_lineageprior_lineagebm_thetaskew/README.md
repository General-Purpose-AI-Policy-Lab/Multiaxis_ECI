# Fit joint skew-normal avec les paliers humains (analyse secondaire, 16–19 octobre 2026)

```bash
python 3_fit/fit.py --K 4 --human-merge --lineage-prior --lineage-bm --theta-skew --chains 10 --seed 42
```

Les données sont celles du pipeline au 16 octobre, qui comprennent les évaluations Claude faites à
la main (`gpai_runs`, protocole RAND).

**Question posée.** Avec le prior skew-normal sur les capacités (`--theta-skew`), les paliers humains
se placent-ils correctement lorsqu'ils sont ajustés avec les modèles dès le départ ?

**Réponse : non.** La compensation de l'ancien fit joint demeure, sous une autre forme. La référence
reste un fit sans humains, suivi du placement des paliers axe par axe (`5_place_humans.py`).

## Résultats

- **Modes** : A (chaînes 0, 1, 2, 3, 5, 9) et B (chaînes 4, 6, 7, 8), à −9 nats.
  Vérification prédictive a posteriori : R² = 0,961.
- **Les axes changent de contenu** (benchmarks principaux de la meilleure chaîne) :
  - ARC-AGI, ARC-AGI-2, OTIS, MATH ;
  - WMDP, MMLU-Pro ;
  - SimpleQA, OpenBookQA, Balrog, SimpleBench ;
  - CyScenarioBench, GBAEval, Furniture Assembly, GDPval.

  Il n'y a plus d'axe agentique distinct, et l'agentique se mélange au cyber. La part de chargement
  portée par les benchmarks signatures du flagship tombe à 0,06–0,10, contre 0,36–0,66 dans l'ancien
  fit joint.
- **Écart des paliers à leur placement de référence**, en écarts-types de la population des modèles,
  axes appariés par benchmarks signatures. Médiane sur la famille principale de chaînes, comparée à
  l'ancien fit joint (données du 1ᵉʳ octobre, sans skew-normal) :

  | axe de référence | ancien fit joint | ce fit |
  |---|---|---|
  | Intelligence fluide & sciences formelles | −2,2 à −3,4 | −1,4 à −2,5 |
  | Connaissances et raisonnements scientifiques | −1 à −3 | −4 à −7 |
  | Agentique & programmation | +9 à +11 | +3 à +7 (axe de contenu différent) |
  | Cyber | −0,3 à −5 | +2 à +8 (axe de contenu différent) |

  Dans la même famille de chargements, la position des paliers varie fortement d'une chaîne à
  l'autre (de +3 à +8 pour Average sur un même axe).
- Script de comparaison : `~/sampler_study/night/joint_humans.py` (hors dépôt).

## Ce qui n'a pas été fait

Le témoin sans skew-normal sur les mêmes données a été arrêté à la demande de l'utilisateur. On ne
peut donc pas séparer ici l'effet du skew-normal de celui des nouvelles données.
