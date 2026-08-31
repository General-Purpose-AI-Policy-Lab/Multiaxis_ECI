"""K=4 forest figure: top models and human tiers per axis, at the post's scale.

Same rows as `make_all.forests()` — `forest_frames`'s top models by posterior
mean at SD < 0.5, the pinned frontier releases, and every human tier — drawn by
the dashboard's own `viz.forest_grid_fig`, so markers, whiskers and the legend
are the definitions the fit's CSVs use. The type constants match
`make_timeline_plotly.py` / `make_loadings_plotly.py`. Reads the flagship
trace over ALL chains: the post's figures are whole-posterior, never
mode-restricted. Axis identity is checked against `make_all.EXPECTED_TOPS`
before any label is applied.

Usage:
    python blogpost/figures/make_forests_plotly.py [--trace FILE] [--tag _draft]
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
from multiaxis_eci.viz.core import save_print  # noqa: E402


# None drops the in-figure title: the post's caption carries the
# description. Set a string to draw it on the canvas again, e.g.
# "Top models, frontier releases and human tiers per axis".
TITLE = None

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


def main(trace: Path = TRACE, tag: str = "_draft", out_dir: Path = HERE) -> None:
    idata = FLAGSHIP.open_posterior(keep=["A", "theta", "tau_A"],
                                    thin=FLAGSHIP_THIN, chains=None, path=trace)
    data, *_ = FLAGSHIP.load_data(idata)
    view = prepare_fit(idata, data)
    check_axis_identity(view, data)     # SystemExit before any mislabeled axis

    frames = forest_frames(view, data)  # n_top / sd_cap: make_all's defaults
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

    out = out_dir / f"forests_axes_plotly{tag}"
    fig.write_html(out.with_suffix(".html"))
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
    args = p.parse_args()
    main(args.trace, args.tag)
