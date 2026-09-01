"""K=4 loadings figure: the top-20 benchmarks per axis, ranked by axis share.

Same content as `make_all.loadings()` but drawn as a 2x2 grid at the post's
type scale (the constants match `make_timeline_plotly.py`), reading the
flagship trace in `results/mirt_humanmerge_lineageprior_lineagebm`.
The panels come from the dashboard's own `viz.loadings_grid_fig`, so the bar,
whisker and share definitions are the ones the fit's CSVs use. Axis identity
is checked against `make_all.EXPECTED_TOPS` before any label is applied.

Usage:
    python blogpost/figures/make_loadings_plotly.py [--trace FILE] [--tag _draft]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from multiaxis_eci.analysis import (FLAGSHIP, FLAGSHIP_THIN,  # noqa: E402
                      FLAGSHIP_TRACE as TRACE, prepare_fit)
from multiaxis_eci.analysis import loadings_table  # noqa: E402
from multiaxis_eci.config import AXIS_TITLES  # noqa: E402
from make_all import check_axis_identity, two_column_layout  # noqa: E402
from multiaxis_eci.viz import loadings_grid_fig  # noqa: E402
from multiaxis_eci.viz.core import save_html, save_print  # noqa: E402

TOP_N = 20

# None drops the in-figure title: the post's caption carries the
# description. Set a string to draw it on the canvas again, e.g.
# f"The {TOP_N} benchmarks that define each axis, ranked by axis share".
TITLE = None

# Post-scale sizing, shared with the timeline figure: the figure is shared
# flat, so type must read without zooming. 20 rows per panel at this tick size
# need the taller canvas; save_print's scale=2 doubles the pixel dimensions.
FONT_TITLE = 42       # figure title
FONT_AXIS = 36        # panel titles, axis titles, colorbar title
FONT_TICK = 30        # tick labels (benchmark name + share)
ERRBAR_W = 2.0        # whisker line width
ERRBAR_CAP = 5        # whisker cap width
WIDTH, HEIGHT = 1900, 2200

# Each panel's x range runs this factor past its longest whisker, reserving a
# clear column at the right edge for the axis-share numbers ("0.99" at tick
# size is ~20% of a panel's width, so the whisker must stop short of ~75%).
X_HEADROOM = 1.45


def main(trace: Path = TRACE, tag: str = "_draft", out_dir: Path = HERE) -> None:
    # The modes file for this trace reports one mode over all 10 chains, so no
    # majority restriction applies; medians are pinned fine at thin=10.
    idata = FLAGSHIP.open_posterior(keep=["A", "theta", "tau_A"],
                                    thin=FLAGSHIP_THIN, chains=None, path=trace)
    data, *_ = FLAGSHIP.load_data(idata)
    view = prepare_fit(idata, data)
    check_axis_identity(view, data)     # SystemExit before any mislabeled axis

    bench = data.blookup.sort_values("benchmark_idx")["benchmark"].tolist()
    # 2.5/97.5 to match the interval every other flagship loading figure draws.
    ldf = loadings_table(view.require_A(), bench, hdi=(2.5, 97.5))
    # ncols=2 lays the axes row-major: axis1 top-left ... axis4 bottom-right.
    fig = loadings_grid_fig(ldf, AXIS_TITLES, ncols=2, top_n=TOP_N,
                            title=TITLE or " ", width=WIDTH)
    if TITLE is None:
        fig.layout.title = None

    fig.update_layout(height=HEIGHT, title_font_size=FONT_TITLE,
                      margin=dict(l=80, r=240, t=170, b=110),
                      coloraxis_colorbar=dict(title_font_size=FONT_AXIS,
                                              tickfont_size=FONT_TICK, x=1.04))
    fig.update_annotations(font_size=FONT_AXIS)          # panel titles
    # The gutter also clears the LEFT panel's share-number column: the right
    # column's longest tick label must end short of the left panel's edge.
    two_column_layout(fig, [(0.0, 0.30), (0.70, 0.96)], AXIS_TITLES)
    fig.update_yaxes(tickfont_size=FONT_TICK)
    # The x caption repeats on all four panels and sits beside the colorbar,
    # so it stays at tick size rather than axis-title size.
    fig.update_xaxes(tickfont_size=FONT_TICK, title_font_size=FONT_TICK)
    fig.update_traces(selector=dict(type="bar"),
                      error_x=dict(thickness=ERRBAR_W, width=ERRBAR_CAP))

    # The axis share is a right-aligned column at each panel's right edge (the
    # post's caption reads "the number on the right"), not part of the tick
    # label the builder writes. Ticks carry the bare benchmark name, and each x
    # range takes X_HEADROOM past the panel's longest whisker so no bar or
    # whisker reaches the number column. This runs after update_annotations, so
    # the numbers keep tick size while panel titles keep axis size.
    for i, bar in enumerate(t for t in fig.data if t.type == "bar"):
        n = "" if i == 0 else str(i + 1)
        names = list(bar.y)
        fig.layout[f"yaxis{n}"].ticktext = names
        hi = float(np.max(np.asarray(bar.x) + np.asarray(bar.error_x["array"])))
        fig.layout[f"xaxis{n}"].range = (0.0, hi * X_HEADROOM)
        for name, share in zip(names, bar.customdata):
            fig.add_annotation(xref=f"x{n} domain", x=0.98, xanchor="right",
                               yref=f"y{n}", y=name, text=f"{float(share):.2f}",
                               showarrow=False, font=dict(size=FONT_TICK))

    out = out_dir / f"loadings_axes_plotly{tag}"
    save_html(fig, out)
    save_print(fig, out)
    print(f"  wrote {out.with_suffix('.png')}")
    for axis in dict.fromkeys(ldf["axis"]):
        n = min(TOP_N, (ldf["axis"] == axis).sum())
        print(f"  {axis}: {n} rows")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trace", type=Path, default=TRACE)
    p.add_argument("--tag", default="_draft")
    args = p.parse_args()
    main(args.trace, args.tag)
