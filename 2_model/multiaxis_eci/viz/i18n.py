"""French renders of finished figures: a string table walked over the figure's JSON.

One table for the blog post and the per-fit `fr/` folders. Entries are applied in order, so a
longer phrase must come before any phrase it contains ("probability" before "ability"); axis
titles come from `axis_names.json` in English, and their French forms are listed here.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

# French forms of the frontier-gap groups (analysis.frontier_gap.GROUP_TITLES and GROUP_NOUNS): the
# caption, then the plural and the singular inside a sentence. A new group needs only its line here.
FR_GROUPS: dict[str, tuple[str, str, str]] = {
    "closed": ("Modèles fermés", "modèles fermés", "modèle fermé"),
    "open": ("Modèles à poids ouverts", "modèles à poids ouverts", "modèle à poids ouverts"),
    "US": ("Modèles américains", "modèles américains", "modèle américain"),
    "CN": ("Modèles chinois", "modèles chinois", "modèle chinois"),
}


def _frontier_gap_entries() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """The French of every frontier-gap caption that names a group, for every group pair of
    analysis.frontier_gap: the phrases (titles, axis captions, table rows, hover), then the legend
    entries and the bare group names, which must come after every phrase containing them."""
    from multiaxis_eci.analysis.frontier_gap import GROUP_NOUNS, GROUP_PAIRS, GROUP_TITLES
    missing = sorted({g for pair in GROUP_PAIRS.values() for g in pair} - set(FR_GROUPS))
    if missing:
        raise KeyError(f"viz.i18n.FR_GROUPS has no French for the frontier-gap groups {missing}")
    phrases, legend, bare = [], [], []
    for kind, (lead, fol) in GROUP_PAIRS.items():
        titles, nouns = GROUP_TITLES[kind], GROUP_NOUNS[kind]
        lt, ft = titles[lead], titles[fol]
        (ln, ln1), (fn, _) = nouns[lead], nouns[fol]
        _, lp, ls = FR_GROUPS[lead]
        _, fp, _ = FR_GROUPS[fol]
        phrases += [
            (f"{lt} vs {ft} (ECI-H), by benchmark access",
             f"{FR_GROUPS[lead][0]} contre {fp} (ECI-H), par accès aux benchmarks"),
            (f"Months behind the {ln}<br>frontier (ECI-H)",
             f"Mois de retard sur la frontière<br>des {lp} (ECI-H)"),
            (f"Months behind the {ln} frontier, median [80% interval]",
             f"Mois de retard sur la frontière des {lp}, médiane [intervalle à 80 %]"),
            (f"Release date of the {fn} frontier point", f"Date de sortie du point de frontière des {fp}"),
            # The summary table's rows
            (f"Mean lag over the last 12 months, {fn} frontier to {ln} frontier",
             f"Retard moyen des 12 derniers mois, de la frontière des {fp} à celle des {lp}"),
            (f"Mean lag of the last 12 months' {fn} frontier points",
             f"Retard moyen des points de frontière des {fp} des 12 derniers mois"),
            (f"Lag of the latest {fn} frontier point", f"Retard du dernier point de frontière des {fp}"),
            (f"Gap today ({lt} - {ft} trend lines)",
             f"Écart aujourd'hui (droites de tendance : {lp} − {fp})"),
            (f"Slope difference ({lt} - {ft})", f"Différence de pente ({lp} − {fp})"),
            (f"Lag today on the trend lines (gap / {ln} slope)",
             f"Retard aujourd'hui sur les droites de tendance (écart / pente des {lp})"),
            (f"Lag today (gap / {ln} slope)", f"Retard aujourd'hui (écart / pente des {lp})"),
            (f"{ft} frontier slope", f"Pente de la frontière des {fp}"),
            (f"{lt} frontier slope", f"Pente de la frontière des {lp}"),
            (f"first {ln1} above:", f"premier {ls} au-dessus :"),
        ]
        for g in (fol, lead):
            cap, plural, _ = FR_GROUPS[g]
            legend += [(f"{titles[g]}: frontier point", f"{cap} : point de frontière"),
                       (f"{titles[g]}: trend (80% band)", f"{cap} : tendance (bande à 80 %)"),
                       (f"{titles[g]} trend", f"tendance des {plural}")]
            bare.append((titles[g], cap))
    return phrases, legend + bare


_FG_PHRASES, _FG_NAMES = _frontier_gap_entries()

FR_TABLE: list[tuple[str, str]] = [
    # Dashboard and per-fit figure titles
    ("Forecast:", "Prévision :"),
    ("non-reasoning models", "modèles sans raisonnement"),
    ("ability (Average Human = 0, Top Performer = 1)", "capacité (Humain médian = 0, Meilleur performeur = 1)"),
    ("(80% intervals)", "(intervalles à 80 %)"),
    ("(50% intervals)", "(intervalles à 50 %)"),
    (" · forecast crossing dates", " · dates de croisement prévues"),
    (" · measured models", " · modèles mesurés"),
    (" · all models", " · tous les modèles"),
    ("Top releases (best effort), frontier releases and human tiers per axis",
     "Meilleures sorties (meilleur effort), sorties frontière et niveaux humains par axe"),
    ("Top models, frontier releases and human tiers per axis",
     "Meilleurs modèles, sorties frontière et niveaux humains par axe"),
    ("frontier releases (shown even when wide)", "sorties frontière (même larges)"),
    # Before "PIT histogram", which it contains.
    ("LOO-PIT histogram", "Histogramme LOO-PIT"),
    ("PIT histogram (in-sample)", "Histogramme PIT (sur l'échantillon d'apprentissage)"),
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
    # Human tiers, capitalized as French titles are: first word only
    # (the same labels as the dashboard's HUMAN_LEVEL_LABELS_FR)
    ("Committee of Average Humans", "Comité d'humains médians"),
    ("Committee of Skilled Generalists", "Comité de généralistes qualifiés"),
    ("Committee of Domain Experts", "Comité d'experts du domaine"),
    ("High School Top Performer", "Lycéen, meilleur performeur"),
    ("High School Qualifier", "Lycéen qualifié"),
    ("Average Human", "Humain médian"),
    ("Skilled Generalist", "Généraliste qualifié"),
    ("Domain Expert", "Expert du domaine"),
    ("Top Performer", "Meilleur performeur"),
    # Trend legend and subtitle (the per-fit panels), before "models" and "AI models"
    ("Frontier forecast against human tiers",
     "Prévision de la frontière face aux niveaux humains"),
    ("Non-reasoning models (frontier)", "Modèles sans raisonnement (frontière)"),
    ("Reasoning models (frontier)", "Modèles de raisonnement (frontière)"),
    ("Non-frontier AI models", "Modèles d'IA non-frontière"),
    ("Projected trend", "Tendance projetée"),
    ("Projected frontier", "Frontière projetée"),
    # Crossover legend
    ("already behind us", "déjà derrière nous"),
    ("still ahead", "encore à venir"),
    ("50% interval (thick)", "intervalle 50 % (épais)"),
    ("80% interval (thin)", "intervalle 80 % (fin)"),
    ("95% interval", "intervalle 95 %"),
    ("50% interval", "intervalle 50 %"),
    # Chain-mode figures
    ("primary mode chains", "chaînes du mode principal"),
    ("secondary mode chains", "chaînes du mode secondaire"),
    ("other chains", "autres chaînes"),
    # PIT
    ("calibrated (uniform)", "calibré (uniforme)"),
    ("density", "densité"),
    # Frontier gap, by benchmark access: every caption naming a group is written from
    # FR_GROUPS for every group pair (_frontier_gap_entries). Before the small words below:
    # "Closed models" has to land before "models", "Gap today" before "today".
    *_FG_PHRASES,
    ("How far behind: ", "De combien de mois de retard : "),
    ("Δ semi-private - all", "Δ semi-privés − tous"),
    ("Δ public - all", "Δ publics − tous"),
    ("Δ private - public", "Δ privés − publics"),
    ("Δ private - all", "Δ privés − tous"),
    ("ECI-H / yr", "ECI-H / an"),
    ("quantity", "quantité"),
    ("unit", "unité"),
    # Legend entries and hover labels of both frontier figures, then the bare group names
    *_FG_NAMES,
    ("Semi-private benchmarks", "Benchmarks semi-privés"),
    ("Private benchmarks", "Benchmarks privés"),
    ("Public benchmarks", "Benchmarks publics"),
    ("All benchmarks", "Tous les benchmarks"),
    # The same four lower-cased, as the single-scope figures' captions carry them
    ("semi-private benchmarks", "benchmarks semi-privés"),
    ("private benchmarks", "benchmarks privés"),
    ("public benchmarks", "benchmarks publics"),
    ("all benchmarks", "tous les benchmarks"),
    ("crossing dated back", "croisement rétrodaté"),
    ("crossing measured", "croisement mesuré"),
    ("lower bound", "borne inférieure"),
    # Hover fragments: the longest form of each first
    ("at least %{y:.1f} months behind", "au moins %{y:.1f} mois de retard"),
    (" months behind (backcast in ", " mois de retard (rétrodaté dans "),
    (" months behind", " mois de retard"),
    ("(bound in ", "(borné dans "),
    ("% of draws)", " % des tirages)"),
    (" months<extra>", " mois<extra>"),
    ("months", "mois"),          # the table's unit column, after every phrase containing it

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
    ("Ability: ", "Capacité : "),     # BEFORE "Ability", with the French colon spacing
    ("Ability", "Capacité"),
    ("ability", "capacité"),
    ("today", "aujourd'hui"),
    ("Axis ", "Axe "),
]



def translate_text(s: str, extra: list[tuple[str, str]] | None = None,
                   first: list[tuple[str, str]] | None = None) -> str:
    for a, b in first or ():
        s = s.replace(a, b)
    for en, fr in FR_TABLE:
        s = s.replace(en, fr)
    for a, b in extra or ():
        s = s.replace(a, b)
    return s


def _walk(node, extra, first):
    if isinstance(node, str):
        return translate_text(node, extra, first)
    if isinstance(node, dict):
        return {k: _walk(v, extra, first) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [_walk(v, extra, first) for v in node]
    if isinstance(node, np.ndarray):
        # String data arrays (the forests' y categories) must translate WITH the layout's
        # categoryarray, or the categories split and every row loses its label.
        if node.dtype.kind in ("U", "S", "O"):
            return [_walk(v, extra, first) for v in node.tolist()]
        return node
    return node


def translate_fig(fig: go.Figure, extra: list[tuple[str, str]] | None = None, *,
                  first: list[tuple[str, str]] | None = None) -> go.Figure:
    """A COPY of `fig` with every string field passed through `first` (a fit's own axis
    titles, which the table's generic axis names would otherwise translate half-way:
    "Axis 1: Fluid Intelligence & Mathematics"), then `FR_TABLE`, then `extra`
    (figure-specific replacements applied after the table)."""
    return go.Figure(_walk(fig.to_plotly_json(), extra, first))
