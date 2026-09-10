"""French renders of finished figures: a string table walked over the figure's JSON.

One table for the blog post and the per-fit `fr/` folders. Entries are applied in order, so a
longer phrase must come before any phrase it contains ("probability" before "ability"); axis
titles come from `axis_names.json` in English, and their French forms are listed here.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

FR_TABLE: list[tuple[str, str]] = [
    # Dashboard and per-fit figure titles
    ("Forecast:", "Prévision :"),
    ("non-reasoning models", "modèles sans raisonnement"),
    ("ability (Average Human = 0, Top Performer = 1)", "capacité (Humain moyen = 0, Meilleur performeur = 1)"),
    ("(80% intervals)", "(intervalles à 80 %)"),
    ("(50% intervals)", "(intervalles à 50 %)"),
    ("(crossing dates)", "(dates de croisement)"),
    (": measured", " : modèles mesurés"),
    (": all models", " : tous les modèles"),
    ("Top releases (best effort), frontier releases and human tiers per axis",
     "Meilleures sorties (meilleur effort), sorties frontière et niveaux humains par axe"),
    ("Top models, frontier releases and human tiers per axis",
     "Meilleurs modèles, sorties frontière et niveaux humains par axe"),
    ("frontier releases (shown even when wide)", "sorties frontière (même larges)"),
    ("PIT histogram", "Histogramme PIT"),
    ("95% band under H0", "bande à 95 % sous H0"),
    ("(uniform 1/12", "(uniforme 1/12"),
    ("variance", "variance"),
    ("axis share", "part de l'axe"),
    ("loading", "saturation"),
    ("Benchmark difficulty", "Difficulté des benchmarks"),
    ("difficulty (D)", "difficulté (D)"),

    # Figure titles and long captions
    ("AI capability, human baselines and benchmark difficulty on the ECI-H scale",
     "Capacités des IA, références humaines et difficulté des benchmarks sur l'échelle ECI-H"),
    ("continues past the window (date shown)",
     "dépasse la fenêtre (date indiquée)"),
    ("ability on the agentic axis (median, 95% interval)",
     "capacité sur l'axe agentique (médiane, intervalle à 95 %)"),
    ("theta (median, 95% interval)", "theta (médiane, intervalle à 95 %)"),
    ("ability (median, 95% interval)", "capacité (médiane, intervalle à 95 %)"),
    ("(median, 95% interval)", "(médiane, intervalle à 95 %)"),
    ("Per-axis abilities over time", "Capacités par axe au fil du temps"),
    ("(measured models, 50% intervals)", "(modèles mesurés, intervalles à 50 %)"),
    # Axis titles
    ("Axis 1: Fluid Intelligence", "Axe 1 : Intelligence fluide"),
    ("Axis 2: Scientific Knowledge and Reasoning",
     "Axe 2 : Connaissances et raisonnement scientifiques"),
    ("Axis 3: Agentic Capabilities", "Axe 3 : Capacités agentiques"),
    ("Axis 4: Legacy QA", "Axe 4 : Questions-Réponses (obsolète)"),
    # Human tiers, in the post's capitalized style (the dashboard's
    # HUMAN_LEVEL_LABELS_FR keeps its own lowercase labels)
    ("Committee of Average Humans", "Comité d'Humains Moyens"),
    ("Committee of Skilled Generalists", "Comité de Généralistes Qualifiés"),
    ("Committee of Domain Experts", "Comité d'Experts du Domaine"),
    ("High School Top Performer", "Lycéen Meilleur Performeur"),
    ("High School Qualifier", "Lycéen Qualifié"),
    ("Average Human", "Humain Moyen"),
    ("Skilled Generalist", "Généraliste Qualifié"),
    ("Domain Expert", "Expert du Domaine"),
    ("Top Performer", "Meilleur Performeur"),
    # Crossover legend
    ("already behind us", "déjà derrière nous"),
    ("still ahead", "encore à venir"),
    ("50% interval (thick)", "intervalle 50 % (épais)"),
    ("80% interval (thin)", "intervalle 80 % (fin)"),
    ("95% interval", "intervalle 95 %"),
    ("50% interval", "intervalle 50 %"),
    # Chain-mode figures
    ("majority chains", "chaînes majoritaires"),
    ("minority chains", "chaînes minoritaires"),
    # PIT
    ("calibrated (uniform)", "calibré (uniforme)"),
    ("density", "densité"),
    # Axis captions, legend entries, small words — last
    ("Release date", "Date de sortie"),
    ("Crossing date", "Date de croisement"),
    ("Human tiers", "Niveaux humains"),
    ("human tiers", "niveaux humains"),
    ("AI models", "Modèles d'IA"),
    ("Benchmark difficulty", "Difficulté des benchmarks"),
    ("Benchmarks", "Benchmarks"),
    ("frontier trend", "tendance de la frontière"),
    ("median", "médiane"),
    ("models", "modèles"),
    ("probability", "probabilité"),   # BEFORE "ability", its substring
    ("ability", "capacité"),
    ("today", "aujourd'hui"),
    ("Axis ", "Axe "),
]



def translate_text(s: str, extra: list[tuple[str, str]] | None = None) -> str:
    for en, fr in FR_TABLE:
        s = s.replace(en, fr)
    for a, b in extra or ():
        s = s.replace(a, b)
    return s


def _walk(node, extra):
    if isinstance(node, str):
        return translate_text(node, extra)
    if isinstance(node, dict):
        return {k: _walk(v, extra) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [_walk(v, extra) for v in node]
    if isinstance(node, np.ndarray):
        # String data arrays (the forests' y categories) must translate WITH the layout's
        # categoryarray, or the categories split and every row loses its label.
        if node.dtype.kind in ("U", "S", "O"):
            return [_walk(v, extra) for v in node.tolist()]
        return node
    return node


def translate_fig(fig: go.Figure, extra: list[tuple[str, str]] | None = None) -> go.Figure:
    """A COPY of `fig` with every string field passed through `FR_TABLE`, then through
    `extra` (figure-specific replacements applied after the table)."""
    return go.Figure(_walk(fig.to_plotly_json(), extra))
