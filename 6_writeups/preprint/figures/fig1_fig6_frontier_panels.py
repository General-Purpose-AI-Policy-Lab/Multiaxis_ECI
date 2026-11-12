"""Figures 1 and 6: the frontier trend and the placed human baselines per axis, modes A and C.

Each panel is built by the repo's own builder (viz.dashboard.forecast_figures: the dashboard's and
the per-fit folders' panel, same frontier rule, same placement), on the mode's own frame and axis
names, then restyled to ../PLOTLY_STYLE_GUIDE.md: no title, the "clean" template, the palette, the
final size. Writes fig1_axis{k}_modeA and fig6_axis{k}_modeC (.pdf and .png) beside this file; the
last panel of each figure carries the x title and the legend.

    python 6_writeups/preprint/figures/fig1_fig6_frontier_panels.py
"""
import re
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT / "2_model")]

from multiaxis_eci.analysis import (  # noqa: E402
    FLAGSHIP_TRACE, FitSpec, align_to_reference_loadings, attach_placement, best_mode_chains,
    load_axis_titles, prepare_fit, secondary_frame_dir, secondary_mode_chains,
)
from multiaxis_eci.data import PROCESSED_FILE  # noqa: E402
from multiaxis_eci.viz.dashboard import forecast_figures  # noqa: E402
from multiaxis_eci.viz.forecast import _spread_labels  # noqa: E402
from style import BG, INK, LIGHT, MID, PALETTE, TINTS, ZOOM, save  # noqa: E402

# The builder's colours, by role: both regimes' frontier points, trend and band take one palette
# colour each (a trend takes the colour of the series it fits); the other models are a third series.
ROLE = {(192, 80, 77): 0, (255, 149, 0): 0, (128, 100, 162): 1, (32, 163, 158): 2}
PANEL_H, LAST_EXTRA, LABEL_PX = round(290 * ZOOM), 70, 10   # canvas px; a panel prints 290 px tall


def role(color):
    """Palette index of a builder colour ('#rrggbb', 'rgb(...)' or 'rgba(...)'), None if unknown."""
    if not isinstance(color, str):
        return None
    if color.startswith("#"):
        rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
    else:
        rgb = tuple(int(float(v)) for v in re.findall(r"[\d.]+", color)[:3])
    return ROLE.get(rgb)


# ponytail: the figure is recoloured after the build. If a second document needs this look, move
# the palette into viz.style.FigureStyle and let the builder read it.
def restyle(fig, label, last, panel_h=PANEL_H, cloud_px=5):
    """The builder's standalone panel in the guide's style. `label` names the axis above the plot."""
    legend_lines = [t for t in fig.data if t.x is not None and len(t.x) == 1 and t.x[0] is None
                    and t.mode == "lines"]
    for n, t in enumerate(legend_lines):                      # one legend entry for both trends
        t.update(name="Frontier trend (band: 80 % interval)", showlegend=n == 0)
    for t in fig.data:
        if t in legend_lines:
            t.line.update(color=INK, width=1.2)
        elif t.fill == "tonexty":                               # a trend's 80 % band
            t.fillcolor = TINTS[role(t.fillcolor)]
        elif t.mode == "lines" and t.line.color:              # a trend, thinner than the data
            t.line.update(color=PALETTE[role(t.line.color)], width=1.2 if t.opacity is None else 0.8)
            t.opacity = None
        elif t.mode == "markers":
            i = role(t.marker.color)
            cloud = i == 2
            t.marker.update(color=TINTS[i] if cloud else PALETTE[i], size=cloud_px if cloud else 7,
                            opacity=1, line=dict(width=0.5 if cloud else 1, color=BG))
            if t.error_y.array is not None:
                t.error_y.update(color=PALETTE[i], thickness=0.8, width=0)
    tiers = [a for a in fig.layout.annotations if a.xref == "paper"]
    lines = [s for s in fig.layout.shapes if s.xref == "x domain"]          # the tier lines
    fig.layout.shapes = [s for s in fig.layout.shapes if s.xref != "paper"]  # drop the old leaders
    for s in fig.layout.shapes:
        s.update(opacity=1, layer="below", line=dict(color=MID, width=0.7, dash=s.line.dash))
    for a in fig.layout.annotations:
        a.font = dict(size=LABEL_PX if a.xref == "paper" else 9.5, color=MID)
    height = panel_h + (LAST_EXTRA if last else 0)
    if tiers:
        # Names respread for this panel's height and type size (the builder spread them for its own).
        assert len(tiers) == len(lines), "tier names and tier lines do not pair up"
        lo, hi = fig.layout.yaxis.range
        gap = 1.3 * LABEL_PX / (panel_h - 50) * (hi - lo)
        levels = np.array([s.y0 for s in lines], dtype=float)
        for a, level, y in zip(tiers, levels, _spread_labels(levels, gap, lo + 0.02 * (hi - lo),
                                                             hi - 0.02 * (hi - lo))):
            a.update(y=y, x=1.012)
            if abs(y - level) > 0.25 * gap:
                fig.add_shape(type="line", xref="paper", yref="y", x0=1.0, y0=level, x1=1.01, y1=y,
                              line=dict(color=MID, width=0.5))
    fig.add_annotation(text=f"<b>{label}</b>", xref="paper", yref="paper", x=0, y=1, xanchor="left",
                       yanchor="bottom", yshift=4, font=dict(size=12, color=INK))
    longest = max((len(a.text) for a in tiers), default=0)
    fig.update_layout(
        template="clean", title=None, font=dict(family="Arial, sans-serif", size=9.5, color=INK),
        width=None, height=None, showlegend=last,
        legend=dict(orientation="h", x=0, xanchor="left", y=-62 / (panel_h - 50), yanchor="top", font=dict(size=10, color=INK),
                    bgcolor="rgba(0,0,0,0)", borderwidth=0, traceorder="normal"),
        margin=dict(l=4, r=14 + int(0.56 * LABEL_PX * longest), t=24, b=4))
    fig.update_xaxes(showline=True, linecolor=INK, linewidth=1, ticks="outside", tickcolor=INK, ticklen=4,
                     showgrid=False, tickfont=dict(size=12, color=INK),
                     title=dict(text="Release date" if last else None, font=dict(size=12, weight=500)))
    fig.update_yaxes(showline=False, ticks="", showgrid=True, gridcolor=LIGHT, gridwidth=0.7,
                     tickfont=dict(size=12, color=INK),
                     title=dict(text="Ability (fitted scale)", font=dict(size=12, weight=500)))
    return fig, height


def build(mode):
    """The builder's own trend panels of a mode ('A' or 'C'), in axis order, with the axis titles."""
    trace, fit = FLAGSHIP_TRACE, FLAGSHIP_TRACE.parent
    chains, frame = ((best_mode_chains(trace), fit) if mode == "A"
                     else (secondary_mode_chains(trace), secondary_frame_dir(trace)))
    with xr.open_dataset(trace, group="posterior") as post:
        spec = FitSpec.from_trace(az.InferenceData(posterior=post.isel(draw=slice(0, 1))), trace)
    idata = spec.open_posterior(keep=("A", "theta", "tau_A", "D", "phi_b", "alpha", "ceiling_d"),
                                path=trace, chains=chains)
    data, _, _ = spec.load_data(idata)
    view = align_to_reference_loadings(prepare_fit(idata, data), data, frame / "mirt_loadings.csv")
    view = attach_placement(view, frame)
    titles = load_axis_titles(frame, view, data)
    figs = {k: f for k, f in forecast_figures(view, data, pd.read_csv(PROCESSED_FILE), view.names,
                                              view.theta, axis_titles=titles).items()
            if not k.endswith("_when")}
    assert len(figs) == view.K, f"mode {mode}: {len(figs)} trend panels for {view.K} axes"
    return [figs[k] for k in sorted(figs)], [titles.get(n, n) for n in view.names], len(chains)


if __name__ == "__main__":
    for fig_no, mode in ((1, "A"), (6, "C")):
        panels, labels, n_chains = build(mode)
        for k, (fig, label) in enumerate(zip(panels, labels), start=1):
            fig, height = restyle(fig, label, last=k == len(panels))
            save(fig, HERE / f"fig{fig_no}_axis{k}_mode{mode}", height / ZOOM)
            print(f"mode {mode} ({n_chains} chains), axis {k}: prints {height / ZOOM:.0f} px tall")
