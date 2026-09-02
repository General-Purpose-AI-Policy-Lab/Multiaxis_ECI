"""French renders of the post's Plotly figures, in `figures/fr/`.

The blog post ships on the lab's (French) site first and is crossposted to
LessWrong in English, so every figure exists in both languages. Rather than
threading a language flag through eight scripts, this one re-runs their
`main()` with `save_print`/`save_html` intercepted: the finished figure is
deep-walked and every string field (titles, axis captions, legend names,
annotations, hovertemplates — and data-level category labels like the
crossover tier rows) is passed through the EN→FR table below, then written
under `fr/` (the HTML twins land in `fr/html/` through `save_html` as usual).

The matplotlib diagrams (human_arrangement_lw, model_family_example_lw, the
prior graphs) draw their text at plot time and are NOT covered here.

Usage:
    python blogpost/figures/make_french.py [--cached] [--only NAME[,NAME...]]

`--cached` skips every figure that would need to open the 13 GB trace
(keeps: crossovers, the pooled trend, the 1D timeline). `--only` names a
subset: timeline, trend, trend_majority, trend_minority, crossover,
crossover_majority, crossover_minority, crossover_majority95,
crossover_minority95, forests, loadings, human_modes, split_takers, pit.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from multiaxis_eci.viz.core import save_html, save_print  # noqa: E402

FR_DIR = HERE / "fr"

# Ordered substring pairs: longest / most specific FIRST — "High School Top
# Performer" must translate before "Top Performer", full captions before the
# words inside them. Applied to every string in the figure, including data
# arrays (the crossover y categories ARE the tier names); the tokens are
# alphabetic phrases, so date strings and numbers pass through untouched.
FR = [
    # Figure titles and long captions
    ("AI capability, human capability and benchmark difficulty on the ECI-H scale",
     "Capacités des IA, niveaux humains et difficulté des benchmarks sur l'échelle ECI-H"),
    ("(pre-data dates: backward extrapolation of the early record trend)",
     "(dates pré-données : extrapolation rétrograde)"),
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
    ("Axis 1 — Fluid Intelligence", "Axe 1 : Intelligence fluide"),
    ("Axis 2 — Scientific Knowledge and Reasoning",
     "Axe 2 : Connaissances et raisonnement scientifiques"),
    ("Axis 3 — Agentic Capabilities", "Axe 3 : Capacités agentiques"),
    ("Axis 4 — Legacy QA", "Axe 4 : Questions-Réponses (saturées)"),
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
    ("Benchmarks", "Benchmarks"),
    ("frontier trend", "tendance de la frontière"),
    ("median", "médiane"),
    ("models", "modèles"),
    ("probability", "probabilité"),   # BEFORE "ability", its substring
    ("ability", "capacité"),
    ("today", "aujourd'hui"),
]


def _tr(s: str) -> str:
    for en, fr in FR:
        s = s.replace(en, fr)
    return s


def _walk(node):
    if isinstance(node, str):
        return _tr(node)
    if isinstance(node, dict):
        return {k: _walk(v) for k, v in node.items()}
    if isinstance(node, (list, tuple)):
        return [_walk(v) for v in node]
    if isinstance(node, np.ndarray):
        # String/object data arrays (e.g. the forest figures' y categories)
        # must translate WITH the layout's categoryarray, or the categories
        # split in two and every row loses its label. Numeric arrays pass.
        if node.dtype.kind in ("U", "S", "O"):
            return [_walk(v) for v in node.tolist()]
        return node
    return node


def translate(fig: go.Figure) -> go.Figure:
    """A COPY of `fig` with every string field passed through the FR table."""
    return go.Figure(_walk(fig.to_plotly_json()))


def _fr_save_print(fig, path, **kw):
    return save_print(translate(fig), path, **kw)


def _fr_save_html(fig, path):
    return save_html(translate(fig), path)


def _patched(module):
    module.save_print = _fr_save_print
    module.save_html = _fr_save_html
    return module


def main(cached_only: bool = False, only: set[str] | None = None) -> None:
    FR_DIR.mkdir(exist_ok=True)

    def want(name: str, needs_trace: bool = False) -> bool:
        if only is not None and name not in only:
            return False
        return not (cached_only and needs_trace)

    if want("timeline"):
        import make_timeline_plotly
        _patched(make_timeline_plotly).main(REPO / "results/canonical", "_draft",
                                            out_dir=FR_DIR)
    if want("trend"):
        import make_trend_plotly
        _patched(make_trend_plotly).main(tag="", out_dir=FR_DIR, cached=True)
    for tag, chains in (("majority", [2, 4, 5, 6, 7, 9]),
                        ("minority", [0, 1, 3, 8])):
        if want(f"trend_{tag}", needs_trace=True):
            import make_trend_plotly
            _patched(make_trend_plotly).main(tag=f"_{tag}", out_dir=FR_DIR,
                                             chains=chains)
    if want("crossover"):
        import make_crossover_plotly
        _patched(make_crossover_plotly).main(out_dir=FR_DIR, cached=True)
    for tag, chains, probs in (("majority", [2, 4, 5, 6, 7, 9], (0.5, 0.8)),
                               ("minority", [0, 1, 3, 8], (0.5, 0.8)),
                               ("majority95", [2, 4, 5, 6, 7, 9], (0.95,)),
                               ("minority95", [0, 1, 3, 8], (0.95,))):
        if want(f"crossover_{tag}"):
            import make_crossover_plotly
            _patched(make_crossover_plotly).main(tag=f"_{tag}", out_dir=FR_DIR,
                                                 cached=True, chains=chains,
                                                 probs=probs)
    if want("forests", needs_trace=True):
        import make_forests_plotly
        _patched(make_forests_plotly).main(tag="_draft", out_dir=FR_DIR)
    if want("loadings", needs_trace=True):
        import make_loadings_plotly
        _patched(make_loadings_plotly).main(tag="_draft", out_dir=FR_DIR)
    if want("human_modes", needs_trace=True):
        import make_human_modes_plotly
        _patched(make_human_modes_plotly).main(out_dir=FR_DIR)
    if want("split_takers", needs_trace=True):
        import make_split_takers_plotly
        _patched(make_split_takers_plotly).main(out_dir=FR_DIR)
    if want("pit", needs_trace=True):
        import make_pit_plotly
        _patched(make_pit_plotly).main(out_dir=FR_DIR)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--cached", action="store_true",
                   help="skip every figure that needs the trace")
    p.add_argument("--only", default=None,
                   help="comma-separated subset of figure names")
    args = p.parse_args()
    main(cached_only=args.cached,
         only=None if args.only is None else set(args.only.split(",")))
