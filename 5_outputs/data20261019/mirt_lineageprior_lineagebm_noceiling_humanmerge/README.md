# Fit joint avec les paliers humains (analyse secondaire, données du 19 octobre 2026)

```bash
python 3_fit/fit.py --K 4 --human-merge --lineage-prior --lineage-bm --chains 10 --seed 42
```

Prior skew-normal sur les capacités (le défaut depuis le 20 octobre ; le fit a été lancé avec `--theta-skew`, l'option d'alors). Les paliers humains sont ajustés avec les modèles, sous l'ordre fusionné.

**Question posée.** Avec le prior skew-normal et les données du 19 octobre, les paliers humains se placent-ils correctement lorsqu'ils sont ajustés avec les modèles dès le départ ?

**Réponse : non.** La compensation demeure. La référence reste le flagship ajusté sans humains, suivi du placement axe par axe (`5_place_humans.py`).

## Résultats

- **Convergence.** Un seul mode : les 10 chaînes concordent (corrélation intra-mode 0,89). Une divergence sur 20 000 tirages ; R² = 0,961.
  Le fit joint du 16 octobre en avait deux.
- **Les axes sont ceux du flagship actuel.** La part portée par les benchmarks signatures de ses quatre axes vaut 0,68 (fluide), 0,64 (agentique & cyber), 0,58 (sciences) et 0,43 (visuel & sens commun), en médiane sur les chaînes.
- **Écart des paliers à leur placement axe par axe**, en écarts-types de la population des modèles sur chaque axe : médiane sur les chaînes moins placement (script `compare_tiers.py`).

  | axe | écart (médiane des paliers) | étendue sur les paliers |
  |---|---|---|
  | Intelligence fluide & sciences formelles | +1,1 | +0,3 à +3,7 |
  | Capacités agentiques & cyber | +0,7 | −3,1 à +1,7 |
  | Connaissances et raisonnements scientifiques | **−6,4** | −10,6 à −4,6 |
  | Visuel & sens commun | +0,5 | −1,9 à +4,4 |

  Comme dans tous les fits joints depuis le billet, l'ajustement place les paliers loin sous les modèles sur un axe et les compense sur les autres. Ici, l'axe sacrifié est celui des connaissances scientifiques : le Domain Expert y est à −5,9 écarts-types, contre +1,2 dans le placement axe par axe.

## Historique

- **16 octobre** (données du 16, mêmes options) : deux modes ; même conclusion, avec des axes qui se réorganisaient (pas d'axe agentique distinct).
- **1ᵉʳ octobre et avant** (prior gaussien) : la compensation plaçait les paliers environ 5 écarts-types sous les modèles sur le cyber et 8 au-dessus sur l'agentique.
