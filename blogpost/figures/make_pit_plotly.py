"""PIT histogram of the flagship fit, one wide panel, in the post's Plotly style.

The calibration view: u_n = P(Y_rep <= y_n) under the fitted posterior
predictive. A calibrated fit puts the PIT uniform on [0, 1], so the histogram
sits flat on the dotted density-1 line and the PIT variance sits at the uniform
1/12 ~ 0.083. Variance BELOW 1/12 means the predictive intervals are wider than
the data needs.

Statistics are the production PPC, so the numbers match the fit's own GoF:
`FitSpec.load_data` runs the same loads, floor read and floor clip `fit.py`
runs before sampling (and checks the trace's model/bench dims against today's
data), `ppc.posterior_predictive_mirt` draws the floor-aware predictive, and
`ppc.pit_values` scores it with the boundary rows excluded — PIT is degenerate
at an exact 0 or 1.

All ten chains are pooled: PIT is a whole-fit diagnostic, and the majority /
minority split is an ability-side statement that no calibration number is read
through.

Usage:
    python blogpost/figures/make_pit_plotly.py [--trace FILE] [--tag ""]
                                              [--max-draws 2000]
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

from multiaxis_eci.analysis import (FLAGSHIP, FLAGSHIP_THIN,  # noqa: E402
                      FLAGSHIP_TRACE as TRACE)
from multiaxis_eci.ppc import boundary_mask, pit_values, posterior_predictive_mirt  # noqa: E402
from multiaxis_eci.viz.core import AI_COLOR, save_print  # noqa: E402


# None drops the in-figure title: the post's caption carries the
# description. Set a string to draw it on the canvas again.
TITLE = None

BINS = 20             # bin edges on [0, 1]
MAX_DRAWS = 2000      # posterior draws entering the predictive
UNIFORM = 1.0         # the calibrated density, and the reference line
REF_COLOR = "#444"

# Post-scale sizing, shared with the trend / forests / crossover figures.
FONT_TITLE = 42       # figure title (drawn only when TITLE is set)
FONT_AXIS = 36        # axis titles
FONT_NOTE = 28        # the n / variance line under the x title
FONT_TICK = 30
FONT_LEGEND = 30
REF_W = 6             # the dotted reference line, thick enough to read at scale
WIDTH, HEIGHT = 2200, 1000


def main(trace: Path = TRACE, tag: str = "", out_dir: Path = HERE,
         max_draws: int = MAX_DRAWS) -> None:
    idata = FLAGSHIP.open_posterior(keep=["A", "theta", "D", "phi_b"],
                                    thin=FLAGSHIP_THIN, chains=None, path=trace)
    data, floor_c, n_eff = FLAGSHIP.load_data(idata)
    print(f"  data scope: {data.n_obs} obs, {data.n_models} test-takers, "
          f"{data.n_benchmarks} benchmarks")
    y_rep = posterior_predictive_mirt(idata, data, floor_c=floor_c,
                                      n_eff=n_eff, max_draws=max_draws)
    pit = pit_values(y_rep, data.scores, boundary_mask(data))
    print(f"  n {pit.size} (of {data.n_obs} obs; "
          f"{int(boundary_mask(data).sum())} boundary rows excluded)")
    print(f"  PIT mean {pit.mean():.4f}, variance {pit.var():.4f} "
          f"(uniform 0.5 / {1 / 12:.4f})")

    dens, edges = np.histogram(pit, bins=BINS, range=(0, 1), density=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=(edges[:-1] + edges[1:]) / 2, y=dens, width=np.diff(edges),
        marker=dict(color=AI_COLOR, line=dict(color="white", width=3)),
        opacity=0.85, showlegend=False,
        hovertemplate="PIT %{x:.3f}<br>density %{y:.2f}<extra></extra>"))
    # A real trace, not add_hline: the reference belongs in the legend, and it
    # is drawn last so it stays visible over the bars.
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[UNIFORM, UNIFORM], mode="lines",
        line=dict(color=REF_COLOR, width=REF_W, dash="dot"),
        name="calibrated (uniform)", hoverinfo="skip"))

    # The n / variance line rides under the x title: the figure title is off,
    # and the reader needs the sample size and the variance next to the shape.
    xtitle = ("PIT  u<sub>n</sub> = P(Y<sub>rep</sub> ≤ y<sub>n</sub>)"
              f'<br><span style="font-size:{FONT_NOTE}px">n = {pit.size}, '
              f"variance {pit.var():.3f} (uniform 1/12 ≈ 0.083)</span>")
    fig.update_xaxes(title=dict(text=xtitle, font=dict(size=FONT_AXIS)),
                     range=[0, 1], dtick=0.2, tickfont=dict(size=FONT_TICK),
                     gridcolor="#e9e9e9")
    fig.update_yaxes(title=dict(text="density", font=dict(size=FONT_AXIS)),
                     rangemode="tozero", tickfont=dict(size=FONT_TICK),
                     gridcolor="#e9e9e9")
    fig.update_layout(
        template="plotly_white", width=WIDTH, height=HEIGHT, bargap=0,
        margin=dict(l=190, r=90, t=90, b=250),
        legend=dict(orientation="h", yanchor="top", y=-0.42, xanchor="center",
                    x=0.5, font=dict(size=FONT_LEGEND),
                    bgcolor="rgba(0,0,0,0)", borderwidth=0))
    if TITLE is not None:
        fig.update_layout(title=dict(text=TITLE, x=0.5,
                                     font=dict(size=FONT_TITLE)), margin_t=190)

    out = out_dir / f"pit_plotly{tag}"
    fig.write_html(out.with_suffix(".html"))
    save_print(fig, out)
    print(f"  wrote {out.with_suffix('.png')} and {out.with_suffix('.html')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trace", type=Path, default=TRACE)
    p.add_argument("--tag", default="")
    p.add_argument("--max-draws", type=int, default=MAX_DRAWS)
    args = p.parse_args()
    main(args.trace, args.tag, max_draws=args.max_draws)
