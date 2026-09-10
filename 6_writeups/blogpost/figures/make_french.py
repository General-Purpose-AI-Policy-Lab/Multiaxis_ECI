"""French renders of the post's Plotly figures, in `figures/fr/`, named `<stem>_fr`.

The blog post ships on the lab's (French) site first and is crossposted to
LessWrong in English, so every figure exists in both languages. Rather than
threading a language flag through eight scripts, this one re-runs their
`main()` with `save_print`/`save_html` intercepted: the finished figure is
deep-walked and every string field (titles, axis captions, legend names,
annotations, hovertemplates — and data-level category labels like the
crossover tier rows) is passed through the EN→FR table below, then written
under `fr/` (the HTML twins land in `fr/html/` through `save_html` as usual,
and every print export also gets a vector SVG twin in `fr/svg/`, for the
site's figure embeds).

The matplotlib diagrams (human_arrangement_lw, model_family_example_lw, the
prior graphs) draw their text at plot time and are NOT covered here.

Usage:
    python 6_writeups/blogpost/figures/make_french.py [--cached] [--only NAME[,NAME...]]

`--cached` skips every figure that would need to open the 13 GB trace
(keeps: crossovers, the pooled trend, the 1D timelines). `--only` names a
subset: timeline, timeline_humanmerge, trend, trend_majority, trend_minority,
crossover, crossover_majority, crossover_minority, crossover_majority95,
crossover_minority95, forests, forests_minority, loadings, human_modes,
split_takers, pit. `--post` is the subset the French post embeds (what
`fr/` tracks): the two timelines, the majority trend, the four chain-mode
crossovers, both forests, loadings, human_modes, split_takers, pit.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import plotly.graph_objects as go

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "2_model"))
sys.path.insert(0, str(HERE))

from multiaxis_eci.viz.core import save_html, save_print, save_svg  # noqa: E402
from multiaxis_eci.viz.i18n import FR_TABLE, translate_fig  # noqa: E402

FR_DIR = HERE / "fr"

# The figures the French post embeds, i.e. the set `fr/` tracks (the pooled
# trend/crossover and the minority trend are English-post appendix renders).
POST = {"timeline", "timeline_humanmerge", "trend_majority",
        "crossover_majority", "crossover_minority", "crossover_majority95",
        "crossover_minority95", "forests", "forests_minority", "loadings",
        "human_modes", "split_takers", "pit"}

# Ordered substring pairs: longest / most specific FIRST — "High School Top
# Performer" must translate before "Top Performer", full captions before the
# words inside them. Applied to every string in the figure, including data
# arrays (the crossover y categories ARE the tier names); the tokens are
# alphabetic phrases, so date strings and numbers pass through untouched.
FR = FR_TABLE          # the library's table, shared with the per-fit fr/ folders


def translate(fig: go.Figure, extra: list[tuple[str, str]] | None = None) -> go.Figure:
    return translate_fig(fig, extra)


def _patched(module, extra: list[tuple[str, str]] | None = None):
    """Route the module's savers through the FR translation; `extra` adds
    figure-specific replacements (e.g. the loadings grid drops " (obsolète)"
    from the axis-4 title so it clears the colorbar). Every print export also
    writes its SVG twin (`fr/svg/<stem>_fr.svg`, same print layout)."""
    def _print(fig, path, **kw):
        fr, out = translate(fig, extra), _fr_path(path)
        png = save_print(fr, out, **kw)
        save_svg(fr, out, width=kw.get("width"), height=kw.get("height"))
        return png
    module.save_print = _print
    module.save_html = lambda fig, path: save_html(translate(fig, extra), _fr_path(path))
    return module


def _fr_path(path) -> Path:
    """`<stem>_fr<suffix>`: every French export carries the language in its
    name, so a French file dropped next to its English twin cannot be mistaken
    for it (the HTML twin in `fr/html/` gets the same stem)."""
    path = Path(path)
    return path.with_name(path.stem + "_fr" + path.suffix)


def main(cached_only: bool = False, only: set[str] | None = None) -> None:
    FR_DIR.mkdir(exist_ok=True)

    def want(name: str, needs_trace: bool = False) -> bool:
        if only is not None and name not in only:
            return False
        return not (cached_only and needs_trace)

    if want("timeline"):
        import make_timeline_plotly
        _patched(make_timeline_plotly).main(make_timeline_plotly.DEFAULT_RESULTS, "_draft",
                                            out_dir=FR_DIR)
    if want("timeline_humanmerge"):
        # The --human-merge variant of Figure 1 (canonical_humanmerge/ of the data generation).
        import make_timeline_plotly
        _patched(make_timeline_plotly).main(make_timeline_plotly.DEFAULT_RESULTS.parent / "canonical_humanmerge",
                                            "_humanmerge", out_dir=FR_DIR)
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
    if want("forests_minority", needs_trace=True):
        import make_forests_plotly
        _patched(make_forests_plotly).main(
            tag="_draft", out_dir=FR_DIR,
            chains=make_forests_plotly.MINORITY_CHAINS)
    if want("loadings", needs_trace=True):
        import make_loadings_plotly
        _patched(make_loadings_plotly,
                 extra=[(" (obsolète)", "")]).main(tag="_draft", out_dir=FR_DIR)
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
    p.add_argument("--post", action="store_true",
                   help="only the figures the French post embeds (the fr/ set)")
    args = p.parse_args()
    only = None if args.only is None else set(args.only.split(","))
    if args.post:
        only = POST if only is None else only & POST
    main(cached_only=args.cached, only=only)
