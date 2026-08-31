"""One entry point for the LessWrong post's Plotly figures.

Every figure here reads the flagship fit through `analysis.FLAGSHIP` /
`open_flagship`, so the fit identity, the majority-chain policy and the forecast
settings are the ones the rest of the repo uses. Nothing is hand-copied.

    ~/miniforge3/envs/pymc_env/bin/python lw_post/figures/make_all.py all
    ...                                                    crossover
    ...                                                    trend
    ...                                                    timeline
    ...                                                    forests
    ...                                                    loadings
    ...                                                    axis-timelines
    ...   --cached      never open the trace: reuse the forecast cache pickle,
    ...                 fail if it is missing; the three per-axis figures are
    ...                 skipped, they have no cache
    ...   --out DIR     write the figures somewhere other than this folder

The three per-axis figures have a matplotlib twin in `make_results_figs.py` /
`memo/make_memo_figs.py`; the Plotly ones here write `*_lw_plotly.png`, so the
two sets can be compared side by side.

The forecast cache is `results/mirt.../lw_forecast_cache_50.pkl`, next to the
trace it was computed from, not in the temp dir: it is derived from one specific
fit and it takes a 38 GB trace read to rebuild.
"""
from __future__ import annotations

import argparse
import pickle
import sys
from functools import lru_cache
from pathlib import Path
from textwrap import shorten

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from analysis import (FLAGSHIP, FLAGSHIP_TRACE, open_flagship,  # noqa: E402
                      prepare_fit)
from config import AXIS_TITLES, FORECAST_KW, FORECAST_NO_SOTA_AXES  # noqa: E402
from data import PROCESSED_FILE  # noqa: E402

AXES = ["axis1", "axis2", "axis3"]      # Legacy QA is out of the forecast scope
HDI = 0.5                               # every interval on both forecast figures
END = pd.Timestamp("2030-01-01")        # right edge of the trend figure
# The cache sits in the trace's own folder, which the tag grammar does not
# derive, so it is read off `FLAGSHIP_TRACE` and never off `results_dir`.
CACHE = FLAGSHIP_TRACE.parent / "lw_forecast_cache_50.pkl"

# Defining benchmarks per axis. The prose titles in AXIS_TITLES are keyed
# `axis1..4` by position, so they are valid only while the axes keep these
# identities; a re-fit that reorders them must fail loudly rather than mislabel.
EXPECTED_TOPS = {
    "axis1": {"ARC-AGI-2", "VPCT", "ARC-AGI"},
    "axis2": {"WMDP Chemistry", "WMDP Biology"},
    "axis3": {"GBAEval", "Remote Labor Index", "ProofBench"},
    "axis4": {"OpenBookQA", "ARC (AI2)", "Adversarial NLI"},
}


def check_axis_identity(view, data, top_n: int = 5) -> None:
    """Raise if an axis's highest-share benchmarks are not the expected ones."""
    A = view.require_A()
    bench = data.blookup.sort_values("benchmark_idx")["benchmark"].tolist()
    med = np.median(A, axis=0)
    share = med ** 2 / np.maximum((med ** 2).sum(axis=1, keepdims=True), 1e-12)
    for k, name in enumerate(view.names):
        tops = [bench[b] for b in np.argsort(-share[:, k])[:top_n]]
        print(f"  {name} ({AXIS_TITLES[name]}): {tops}")
        want = EXPECTED_TOPS[name]
        if not want & set(tops):
            raise SystemExit(
                f"axis identity check failed for {name}: top-{top_n} by share "
                f"{tops} contains none of {sorted(want)}. Refusing to label.")


@lru_cache(maxsize=1)
def load_flagship():
    """The flagship fit once: (FitView, data, raw scores table), axes checked.

    `prepare_fit` reads the trace's display-rotation tag, so the view is already
    in the frame the fit's own CSVs are written in.
    """
    idata = open_flagship(keep=["A", "theta", "tau_A"])
    data, *_ = FLAGSHIP.load_data(idata)
    view = prepare_fit(idata, data)
    check_axis_identity(view, data)
    missing = [a for a in AXES if a not in view.names]
    if missing:
        raise SystemExit(f"axes {missing} not in the fit: {view.names}")
    return view, data, pd.read_csv(PROCESSED_FILE)


def compute(view, data, raw) -> dict:
    """Per axis: the forecast plus its timeline/human tables, for the trend figure.

    The crossover figure reads only the ForecastResult (slope/intercept draws)
    out of this, through `make_trend_plotly.forecast`'s cache.
    """
    from analysis import (mirt_frontier_forecast, mirt_human_axis_stats,
                          mirt_model_timeline_df)

    out = {}
    for name in AXES:
        k = view.names.index(name)
        fc = mirt_frontier_forecast(view.theta, k, data, raw,
                                    **dict(FORECAST_KW, horizon_date=END,
                                           sota_exempt=k not in FORECAST_NO_SOTA_AXES))
        out[name] = {
            "fc": fc,
            "tl": mirt_model_timeline_df(view.theta, k, data, raw, sd_cap=0.4,
                                         hdi_prob=HDI),
            "hs": mirt_human_axis_stats(view.theta, k, data, hdi_prob=HDI),
        }
    return out


def forecast_cache(cached: bool = False) -> dict:
    """The per-axis trend objects, from the cache beside the trace or recomputed."""
    if CACHE.exists():
        print(f"  reused {CACHE}")
        return pickle.loads(CACHE.read_bytes())
    if cached:
        raise SystemExit(f"--cached but {CACHE} is missing — run without --cached "
                         "once to rebuild it from the trace.")
    per_axis = compute(*load_flagship())
    CACHE.write_bytes(pickle.dumps(per_axis))
    print(f"  wrote {CACHE}")
    return per_axis


def two_column_layout(fig, col_dom, titles, k: int = 4) -> None:
    """Set a 2-column grid's x domains directly and recentre each panel title.

    At post type scale the right column's y tick labels live in the inter-column
    gutter and need roughly 40% of the plot area, which is more than
    `make_subplots`' fractional spacing gives, so the domains cannot be left to
    it. `titles` is the axis-title lookup whose values identify a panel-title
    annotation among the figure's other annotations.
    """
    # Plotly numbers the first axis bare: xaxis, xaxis2, xaxis3, ...
    for i, ax in enumerate(["xaxis"] + [f"xaxis{n}" for n in range(2, k + 1)]):
        fig.layout[ax].domain = col_dom[i % 2]
    panels = [a for a in fig.layout.annotations if a.text in titles.values()]
    for i, ann in enumerate(panels):
        ann.x = sum(col_dom[i % 2]) / 2


# ── per-axis result figures ─────────────────────────────────────────────────
# `_pretty` (readable model labels) and `_FOREST_FRONTIER` (the frontier
# releases pinned into every forest) are imported from the memo module rather
# than copied, so the post and the memo cannot name a model differently.

def _memo():
    sys.path.insert(0, str(REPO / "memo"))
    import make_memo_figs
    return make_memo_figs


def forest_frames(view, data, n_top: int = 11, sd_cap: float = 0.5) -> list:
    """Per axis, the forest rows: top models by mean ability at posterior
    SD < `sd_cap`, the pinned frontier releases, and every human tier.

    Ascending by mean, so the strongest row lands at the top of the panel.
    """
    mmf = _memo()
    names = data.mlookup.sort_values("model_idx")["model"].tolist()
    is_h = np.asarray(data.is_human, dtype=bool)
    frames = []
    for k in range(view.K):
        th = view.theta[:, :, k]                        # (S, M)
        mean, sd = th.mean(0), th.std(0)
        lo, hi = np.percentile(th, [2.5, 97.5], axis=0)
        top = [i for i in np.argsort(-mean) if not is_h[i] and sd[i] < sd_cap][:n_top]
        rows = ([(i, "model") for i in top]
                + [(i, "frontier") for i in range(len(names))
                   if names[i] in mmf._FOREST_FRONTIER and not is_h[i] and i not in top]
                + [(i, "human") for i in np.where(is_h)[0]])
        rows.sort(key=lambda r: mean[r[0]])
        full = [names[i] if is_h[i] else mmf._pretty(names[i]) for i, _ in rows]
        # Word-boundary shortening (a mid-token cut makes two token-budget
        # variants of one model read identically). Plotly MERGES duplicate
        # categories into one row, so a label that collides after shortening
        # keeps its full text instead of stacking two test-takers on one line.
        short = [shorten(f, width=34, placeholder=" …") for f in full]
        dup = {s for s in short if short.count(s) > 1}
        frames.append(pd.DataFrame({
            "name": [f if s in dup else s for s, f in zip(short, full)],
            "kind": [kind for _, kind in rows],
            "mean": [mean[i] for i, _ in rows],
            "hdi_low": [lo[i] for i, _ in rows],
            "hdi_high": [hi[i] for i, _ in rows]}))
    return frames


def forests(out_dir: Path) -> Path:
    """Top models, frontier releases and human tiers, one panel per axis."""
    from viz import forest_grid_fig
    from viz.core import save_print

    view, data, _raw = load_flagship()
    fig = forest_grid_fig(forest_frames(view, data),
                          [AXIS_TITLES[n] for n in view.names])
    return save_print(fig, out_dir / "forests_axes_lw_plotly")


def loadings(out_dir: Path, top_n: int = 20) -> Path:
    """The benchmarks that define each axis, ranked by axis share."""
    from analysis import loadings_table
    from viz import loadings_grid_fig
    from viz.core import save_print

    view, data, _raw = load_flagship()
    bench = data.blookup.sort_values("benchmark_idx")["benchmark"].tolist()
    # 2.5/97.5 to match the interval every other flagship loading figure draws.
    ldf = loadings_table(view.require_A(), bench, hdi=(2.5, 97.5))
    fig = loadings_grid_fig(ldf, AXIS_TITLES, top_n=top_n)
    return save_print(fig, out_dir / "loadings_axes_lw_plotly")


# The dashboard timeline builder labels its two data series in French.
_EN_SERIES = {"Capacité des modèles IA": "AI models",
              "Difficulté des benchmarks": "benchmark difficulty"}


def axis_timelines(out_dir: Path) -> Path:
    """Per-axis ability over release date, the measured (SD < 0.4) cloud.

    The panels come from `viz.dashboard.build_axis_figures`, the same builder the
    dashboard cards use, so the post cannot drift from the card.

    Its human reference lines are dropped here. They are shaded by rank WITHIN a
    panel, and the tier order is not the same on every axis, so one shared legend
    would give a tier a shade three of the four panels do not use. The tiers are
    on the forest figure, where each one is its own row.
    """
    from viz import build_axis_figures, subplot_grid
    from viz.core import save_print

    view, data, raw = load_flagship()
    bench = data.blookup.sort_values("benchmark_idx")["benchmark"].tolist()
    figs = build_axis_figures(view, data, raw, bench, axis_titles=AXIS_TITLES,
                              human_labels={})
    panels = [figs[f"timeline_{k + 1}_{n}"] for k, n in enumerate(view.names)]
    for f in panels:
        f.data = tuple(t for t in f.data if t.legendgroup != "humans")
        f.layout.shapes = ()
        for tr in f.data:
            tr.name = _EN_SERIES.get(tr.name, tr.name)
            if tr.legendgrouptitle.text:
                tr.legendgrouptitle.text = "Human tiers"
    grid = subplot_grid(panels, [AXIS_TITLES[n] for n in view.names],
                        width=1250, height=950, vertical_spacing=0.12,
                        title=dict(text="Per-axis abilities over time "
                                        "(measured models, 50% intervals)", x=0.5),
                        margin=dict(l=70, r=120, t=90, b=60))
    # type="date" must be explicit: the legend-only human traces carry x=[None]
    # and Plotly otherwise reads the axis as numeric and drops the date strings.
    grid.update_xaxes(type="date", title_text="Release date")
    grid.update_yaxes(title_text="ability")
    return save_print(grid, out_dir / "timelines_axes_lw_plotly")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("what", nargs="?", default="all",
                   choices=["all", "crossover", "trend", "timeline",
                            "forests", "loadings", "axis-timelines"])
    p.add_argument("--cached", action="store_true",
                   help="never open the trace; fail if a cache is missing")
    p.add_argument("--out", type=Path, default=HERE)
    args = p.parse_args()

    if args.what in ("all", "crossover"):
        import make_crossover_plotly
        print("crossover:")
        # Reads lw_crossover_50_80.csv beside the trace when it exists;
        # --cached refuses the trace-opening rebuild.
        make_crossover_plotly.main(out_dir=args.out, cached=args.cached)
    if args.what in ("all", "trend"):
        import make_trend_plotly
        print("trend:")
        make_trend_plotly.main(out_dir=args.out, cached=args.cached)
    if args.what in ("all", "timeline"):
        import make_timeline_plotly
        print("timeline:")
        make_timeline_plotly.main(REPO / "results/canonical", "_draft",
                                  out_dir=args.out)
    # The three per-axis figures read the trace (one load, shared by all three).
    for what, fn in (("forests", forests), ("loadings", loadings),
                     ("axis-timelines", axis_timelines)):
        if args.what in ("all", what):
            if args.cached:
                # No cache exists for these: they are posterior summaries, not a
                # forecast. Explicitly asking for one under --cached is an error.
                if args.what != "all":
                    raise SystemExit(f"--cached but {what} needs the trace")
                print(f"{what}: skipped (--cached, and it needs the trace)")
                continue
            print(f"{what}:")
            print(f"  wrote {fn(args.out)}")


if __name__ == "__main__":
    main()
