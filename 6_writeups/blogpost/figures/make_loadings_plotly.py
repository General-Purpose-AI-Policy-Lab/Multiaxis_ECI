"""K=4 loadings figure: the top-20 benchmarks per axis, ranked by axis share.

Same content as `make_all.loadings()` but drawn as a 2x2 grid at the post's
type scale (the constants match `make_timeline_plotly.py`), reading the
flagship trace (`analysis.FLAGSHIP_TRACE`).
The panels come from the dashboard's own `viz.loadings_grid_fig`, so the bar,
whisker and share definitions are the ones the fit's CSVs use. Axis identity
is checked against `analysis.check_axis_identity` (config.AXIS_SIGNATURES) before any label is applied.

Usage:
    python 6_writeups/blogpost/figures/make_loadings_plotly.py [--trace FILE] [--tag _draft]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "2_model"))
sys.path.insert(0, str(HERE))


from multiaxis_eci.analysis import (  # noqa: E402
    FLAGSHIP,
    FLAGSHIP_THIN,
    loadings_table,
    prepare_fit,
    require_axis_titles,
)
from multiaxis_eci.analysis import FLAGSHIP_TRACE as TRACE
from multiaxis_eci.viz import POST, loadings_grid_fig  # noqa: E402
from multiaxis_eci.viz.core import save_html, save_print  # noqa: E402

TOP_N = 20

# None drops the in-figure title: the post's caption carries the
# description. Set a string to draw it on the canvas again, e.g.
# f"The {TOP_N} benchmarks that define each axis, ranked by axis share".
TITLE = None

# The gutter also clears the LEFT panel's share-number column: the right
# column's longest tick label must end short of the left panel's edge.
COL_DOMAINS = [(0.0, 0.30), (0.70, 0.96)]


def main(trace: Path = TRACE, tag: str = "_draft", out_dir: Path = HERE) -> None:
    # The modes file for this trace reports one mode over all 10 chains, so no
    # majority restriction applies; medians are pinned fine at thin=10.
    idata = FLAGSHIP.open_posterior(keep=["A", "theta", "tau_A"],
                                    thin=FLAGSHIP_THIN, chains=None, path=trace)
    data, *_ = FLAGSHIP.load_data(idata)
    view = prepare_fit(idata, data)
    titles = require_axis_titles(trace.parent, view, data)   # SystemExit before any mislabel

    bench = data.blookup.sort_values("benchmark_idx")["benchmark"].tolist()
    # 2.5/97.5 to match the interval every other flagship loading figure draws.
    ldf = loadings_table(view.require_A(), bench, hdi=(2.5, 97.5))
    # ncols=2 lays the axes row-major: axis1 top-left ... axis4 bottom-right.
    # The x caption repeats on all four panels beside the colorbar, so it drops
    # the interval note: the post's caption carries "median, 95% interval", and
    # the long French form ran into the colorbar.
    fig = loadings_grid_fig(ldf, titles, ncols=2, top_n=TOP_N, title=TITLE,
                            x_title="loading", style=POST, col_domains=COL_DOMAINS)

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
