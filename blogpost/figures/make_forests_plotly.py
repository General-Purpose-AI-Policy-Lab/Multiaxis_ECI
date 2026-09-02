"""K=4 forest figure: top models and human tiers per axis, at the post's scale.

Same rows as `make_all.forests()` — `forest_frames`'s top models by posterior
mean under the trend figure's gate (SD < FORECAST_KW["sd_cap"] and not
low-obs, or SOTA), the pinned frontier releases, and every human tier — drawn by
the dashboard's own `viz.forest_grid_fig`, so markers, whiskers and the legend
are the definitions the fit's CSVs use. The type constants match
`make_timeline_plotly.py` / `make_loadings_plotly.py`. Reads the flagship
trace over the MAJORITY chains by default (`MAJORITY_CHAINS`, the mode the
post's trend and crossover figures report), with `--chains` for another
subset — `0,1,3,8` is the minority-mode appendix variant — or `--chains all`
for the whole posterior. A chain subset's axes are permuted back onto the
fit-level display frame first, exactly as in `make_crossover_plotly`. Axis
identity is checked against `make_all.EXPECTED_TOPS` before any label is
applied.

The chain subset names the output: `forests_axes_plotly_draft_majority`,
`_minority`, `_c<chains>`, or no suffix for the whole posterior.

Usage:
    python blogpost/figures/make_forests_plotly.py [--trace FILE] [--tag _draft]
                                                   [--chains 0,1,3,8 | all]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from multiaxis_eci.analysis import (FLAGSHIP, FLAGSHIP_THIN,  # noqa: E402
                      FLAGSHIP_TRACE as TRACE, prepare_fit)
from multiaxis_eci.config import AXIS_TITLES  # noqa: E402
from make_all import (check_axis_identity, forest_frames,  # noqa: E402
                      two_column_layout)
from multiaxis_eci.viz import forest_grid_fig  # noqa: E402
from multiaxis_eci.viz.core import save_html, save_print  # noqa: E402


# None drops the in-figure title: the post's caption carries the
# description. Set a string to draw it on the canvas again, e.g.
# "Top models, frontier releases and human tiers per axis".
TITLE = None

# The majority mode of the flagship fit (6 of 10 chains), the same subset the
# trend and crossover figures draw; the minority appendix variant is 0,1,3,8.
MAJORITY_CHAINS = [2, 4, 5, 6, 7, 9]
MINORITY_CHAINS = [0, 1, 3, 8]


def chain_suffix(chains: list[int] | None) -> str:
    """File-name suffix naming the chain subset a render came from, so the
    majority render is never mistaken for the whole-posterior one."""
    if chains is None:
        return ""
    if sorted(chains) == MAJORITY_CHAINS:
        return "_majority"
    if sorted(chains) == MINORITY_CHAINS:
        return "_minority"
    return "_c" + "".join(str(c) for c in sorted(chains))

# Post-scale sizing, shared with the timeline and loadings figures: the figure
# is shared flat, so type must read without zooming. ~20 rows per panel at this
# tick size need the taller canvas; save_print's scale=2 doubles the pixels.
FONT_TITLE = 42       # figure title
FONT_AXIS = 36        # panel titles
FONT_TICK = 30        # tick labels (row names) and x-axis captions
FONT_LEGEND = 30      # the single bottom legend
MARKER = 13           # row markers, matching the timeline's model points
ERRBAR_W = 2.6        # whisker line width (caps stay 0, the forest convention)
# Wider than the other figures' 1900: the gutter must hold 34-character row
# names at tick size (~520 px) plus clearance, or the right column's labels
# run into the left column's whiskers.
WIDTH, HEIGHT = 2100, 2500


def main(trace: Path = TRACE, tag: str = "_draft", out_dir: Path = HERE,
         chains: list[int] | None = MAJORITY_CHAINS) -> None:
    idata = FLAGSHIP.open_posterior(keep=["A", "theta", "tau_A"],
                                    thin=FLAGSHIP_THIN, chains=chains, path=trace)
    data, *_ = FLAGSHIP.load_data(idata)
    view = prepare_fit(idata, data)
    if chains is not None:
        # A chain subset ranks the axes in its own order; put panel k back on
        # the axis panel k carries everywhere else before any label is applied.
        from make_crossover_plotly import _display_frame
        view = _display_frame(view, data, trace)
    check_axis_identity(view, data)     # SystemExit before any mislabeled axis

    frames = forest_frames(view, data)  # n_top / gate: make_all's defaults
    # The post draws frontier releases as ordinary model rows: one marker class
    # for machines, no "shown even when wide" legend entry. The rows themselves
    # stay — only the kind (marker symbol + legend label) collapses.
    for df in frames:
        df.loc[df["kind"] == "frontier", "kind"] = "model"
    # ncols=2 lays the axes row-major: axis1 top-left ... axis4 bottom-right.
    fig = forest_grid_fig(frames, [AXIS_TITLES[n] for n in view.names],
                          title=TITLE or " ", width=WIDTH)
    if TITLE is None:
        fig.layout.title = None

    # r keeps the long axis-2 panel title on the paper; b holds the legend row
    # below the bottom panels' x captions (legend top pinned at y=-0.06).
    fig.update_layout(height=HEIGHT, title_font_size=FONT_TITLE,
                      margin=dict(l=80, r=280, t=120, b=280),
                      legend=dict(font_size=FONT_LEGEND,
                                  yanchor="top", y=-0.06))
    fig.update_annotations(font_size=FONT_AXIS)          # panel titles
    # The gutter here holds 34-character row names.
    two_column_layout(fig, [(0.0, 0.27), (0.73, 0.96)], AXIS_TITLES)
    fig.update_yaxes(tickfont_size=FONT_TICK)
    # The x caption repeats on all four panels, so it stays at tick size.
    fig.update_xaxes(tickfont_size=FONT_TICK, title_font_size=FONT_TICK)
    fig.update_traces(selector=dict(type="scatter"), marker_size=MARKER,
                      error_x=dict(thickness=ERRBAR_W))

    out = out_dir / f"forests_axes_plotly{tag}{chain_suffix(chains)}"
    save_html(fig, out)
    save_print(fig, out)
    print(f"  wrote {out.with_suffix('.png')}")
    for name, df in zip(view.names, frames):
        mach = df[df["kind"] != "human"]
        hum = df[df["kind"] == "human"]
        print(f"  {name}: {len(df)} rows "
              f"({len(mach)} machine, {len(hum)} human); "
              f"top model {mach.loc[mach['mean'].idxmax(), 'name']!r}, "
              f"top tier {hum.loc[hum['mean'].idxmax(), 'name']!r}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trace", type=Path, default=TRACE)
    p.add_argument("--tag", default="_draft")
    p.add_argument("--chains", default=",".join(map(str, MAJORITY_CHAINS)),
                   help="comma-separated chain subset (e.g. 0,1,3,8), or "
                        "'all' for the whole posterior; default: the "
                        "majority chains")
    args = p.parse_args()
    main(args.trace, args.tag,
         chains=None if args.chains == "all"
         else [int(c) for c in args.chains.split(",")])
