"""Figures 1 and 6: the frontier trend and the placed human baselines per axis, modes A and C.

Each panel is built by the repo's own builder (viz.dashboard.forecast_figures: the dashboard's and
the per-fit folders' panel, same frontier rule, same placement), on the mode's own frame and axis
names, then restyled to ../PLOTLY_STYLE_GUIDE.md: no title, the "clean" template, the palette, the
final size. Writes fig1_axis{k}_modeA and fig6_axis{k}_modeC (.pdf and .png) beside this file. Every
panel carries its own x title and legend, so it reads alone. The legend sits inside the plot, against
an edge, where it covers no data (free_spot), and under the plot only when no such place exists.
Each human baseline's name sits on its line inside the plot (name_spots), so the plot takes the full
width, and the baseline lines are drawn over the bands and the other models.

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
from style import BG, INK, LIGHT, MID, PALETTE, TINTS, W, ZOOM, save  # noqa: E402

# The builder's colours, by role: both regimes' frontier points, trend and band take one palette
# colour each (a trend takes the colour of the series it fits); the other models are a third series.
ROLE = {(192, 80, 77): 0, (255, 149, 0): 0, (128, 100, 162): 1, (32, 163, 158): 2}
PANEL_H, LEGEND_H, LABEL_PX = round(290 * ZOOM), 92, 10   # canvas px; a panel prints 290 px tall
LEGEND_NAMES = {0: "Reasoning frontier", 1: "Non-reasoning frontier", 2: "Other models"}
LEGEND_BOX = (160, 19)                                    # canvas px: legend width, height of one entry
FAN = 16                                                  # canvas px from a line's end to its name, in a column of names
RIGHT = 18                                                # canvas px right of the plot: half of the last tick label


def role(color):
    """Palette index of a builder colour ('#rrggbb', 'rgb(...)' or 'rgba(...)'), None if unknown."""
    if not isinstance(color, str):
        return None
    if color.startswith("#"):
        rgb = tuple(int(color[i:i + 2], 16) for i in (1, 3, 5))
    else:
        rgb = tuple(int(float(v)) for v in re.findall(r"[\d.]+", color)[:3])
    return ROLE.get(rgb)


def footprint(fig):
    """What a legend or a name must not cover, as (x, y) in axis fractions: every point, whisker,
    trend, band interior, and the 'today' line. The baseline lines are left out: they may pass
    behind the legend."""
    x0, x1 = pd.to_datetime(list(fig.layout.xaxis.range))
    y0, y1 = fig.layout.yaxis.range
    steps = np.linspace(0, 1, 11)[:, None]
    fx, fy, below = [], [], None
    for t in fig.data:
        if t.x is None or len(t.x) == 0 or t.x[0] is None:
            continue
        x = np.asarray((pd.to_datetime(np.asarray(t.x)) - x0) / (x1 - x0), dtype=float)
        y = np.asarray(t.y, dtype=float)
        lo, hi = y, y
        if t.error_y.array is not None:                              # a whisker, end to end
            lo, hi = y - np.asarray(t.error_y.arrayminus, dtype=float), y + np.asarray(t.error_y.array, dtype=float)
        if t.fill == "tonexty" and below is not None:                # the inside of a band
            lo = below
        below = y
        fx.append(np.tile(x, len(steps)))
        fy.append(((lo + steps * (hi - lo) - y0) / (y1 - y0)).ravel())
    for s in fig.layout.shapes:                                      # the 'today' line
        if s.xref == "x" and s.x0 == s.x1:
            fy.append(np.linspace(0, 1, 80))
            fx.append(np.full(80, float((pd.Timestamp(s.x0) - x0) / (x1 - x0))))
    return np.concatenate(fx), np.concatenate(fy)


def covers(mark, xa, xb, ya, yb):
    """Whether the rectangle [xa, xb] x [ya, yb] (axis fractions) holds a point of the footprint."""
    return bool(((mark[0] > xa) & (mark[0] < xb) & (mark[1] > ya) & (mark[1] < yb)).any())


def free_spot(mark, box, plot):
    """Where a legend of `box` px fits inside a `plot` px plotting area without covering the
    footprint `mark`: the axis fractions of its top-left corner, or None. Tried in this order:
    along the top edge from the left, along the bottom edge from the right, down the left edge, up
    the right edge. The first free place wins, so the legend stays against an edge."""
    w, h, air = box[0] / plot[0], box[1] / plot[1], 0.025
    # Flush with the plot's left or right edge at the ends, so the opaque ground leaves no stub of a
    # grid line beside it.
    xs, ys = np.append(np.arange(0.0, 1.0 - w, 0.02), 1.0 - w), np.arange(0.99, 0.01 + h, -0.02)
    for cx, cy in ([(cx, 0.99) for cx in xs] + [(cx, 0.01 + h) for cx in xs[::-1]]
                   + [(0.0, cy) for cy in ys] + [(1.0 - w, cy) for cy in ys[::-1]]):
        if not covers(mark, cx - air, cx + w + air, cy - h - air, cy + air):
            return float(cx), float(cy)
    return None


def overlap(a, b):
    """Whether two rectangles (x0, x1, y0, y1) overlap."""
    return a[0] < b[1] and b[0] < a[1] and a[2] < b[3] and b[2] < a[3]


def name_spots(mark, levels, widths, plot, taken, only):
    """Where a baseline's name sits on its own line: {baseline: (x of the text's right end, side)},
    x in axis fractions, side 'above' or 'below' the line, for the baselines in `only`. `levels` are
    all the lines' heights in axis fractions, `widths` the names' widths in px, `taken` the
    rectangles already used. A name goes as far right as it can, above its line before below it,
    where it covers no data, no other baseline's line and no other name. A baseline with no such
    place is left out."""
    h, gap, air = 1.25 * LABEL_PX / plot[1], 2 / plot[1], 0.006
    taken, spots = list(taken), {}
    for n in sorted(only, key=lambda n: -levels[n]):               # from the top line down
        w, others = widths[n] / plot[0], np.delete(levels, n)
        for xb, (side, ya) in ((xb, s) for xb in np.arange(0.996, w, -0.01)
                               for s in (("above", levels[n] + gap), ("below", levels[n] - gap - h))):
            box = (xb - w, xb, ya, ya + h)
            if (ya > 0 and ya + h < 1 and not covers(mark, box[0] - air, xb + air, ya - air, ya + h + air)
                    and not ((others > ya - gap) & (others < ya + h + gap)).any()
                    and not any(overlap(box, t) for t in taken)):
                spots[n] = (float(xb), side)
                taken.append(box)
                break
    return spots


def name_stack(mark, levels, widths, plot, taken):
    """For baselines too close together to be named each on its line: their lines end at one x and
    the names follow in a column, spread apart, each tied to its line by a short leader. Returns
    (x where the lines end, the names' heights, the column's rectangle), in axis fractions. The
    column goes as far right as it fits, and moves down before it moves left when something is in
    its way (in these panels that is the trend's band, above it). With no free place it takes the
    first one tried."""
    order = np.argsort(levels)[::-1]
    step, w, air = 1.3 * LABEL_PX / plot[1], (max(widths) + FAN) / plot[0], 0.006
    first = None
    for x in np.arange(0.996 - w, 0.0, -0.01):
        for top in np.arange(0.98, levels.max() - 2 * step, -0.02):
            ys = _spread_labels(levels[order], step, 0.02, top)
            box = (float(x), float(x + w), ys[-1] - step / 2, ys[0] + step / 2)
            out = (float(x), ys[np.argsort(order)], box)
            first = first or out
            if (not covers(mark, x - air, x + w + air, box[2] - air, box[3] + air)
                    and not any(overlap(box, t) for t in taken)):
                return out
    return first


# ponytail: the figure is recoloured after the build. If a second document needs this look, move
# the palette into viz.style.FigureStyle and let the builder read it.
def restyle(fig, label, panel_h=PANEL_H, cloud_px=5):
    """The builder's standalone panel in the guide's style. `label` names the axis above the plot."""
    # The baseline lines go over the bands and the other models, under the frontier and its trend.
    under = max((n for n, t in enumerate(fig.data) if t.fill == "tonexty" or (
        t.mode == "markers" and role(t.marker.color) == 2 and t.x is not None and len(t.x) > 1)), default=-1) + 1
    legend_lines = [t for t in fig.data if t.x is not None and len(t.x) == 1 and t.x[0] is None
                    and t.mode == "lines"]
    for n, t in enumerate(legend_lines):                      # one legend entry for both trends
        t.update(name="Trend, 80 % band", showlegend=n == 0, legendrank=4)
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
            if t.x is not None and len(t.x) == 1 and t.x[0] is None:        # its legend entry
                t.update(name=LEGEND_NAMES[i], legendrank=i + 1)
    tiers = [a for a in fig.layout.annotations if a.xref == "paper"]
    lines = [s for s in fig.layout.shapes if s.xref == "x domain"]          # the tier lines
    assert len(tiers) == len(lines), "tier names and tier lines do not pair up"
    fig.layout.shapes = [s for s in fig.layout.shapes if s.xref == "x"]     # the 'today' line stays
    for s in fig.layout.shapes:
        s.update(opacity=1, layer="below", line=dict(color=MID, width=0.7, dash=s.line.dash))
    for a in fig.layout.annotations:
        a.font = dict(size=LABEL_PX if a.xref == "paper" else 9.5, color=MID)
    for a in tiers:                                            # a grid line stops around a name
        a.update(bgcolor=BG, borderpad=1)
    plot = (W * ZOOM - 62 - RIGHT, panel_h - 70)               # the plotting area, canvas px
    mark = footprint(fig)
    box = (LEGEND_BOX[0], LEGEND_BOX[1] * (5 if tiers else 4) + 10)
    spot = free_spot(mark, box, plot)
    legend = dict(orientation="h", x=0, xanchor="left", y=-62 / (panel_h - 50), yanchor="top")
    if spot:                                                   # anchored on the plot edge it sits against
        at_right = spot[0] + box[0] / plot[0] > 0.999
        legend = dict(x=1 if at_right else spot[0], xanchor="right" if at_right else "left", y=spot[1], yanchor="top")
    height = panel_h + (0 if spot else LEGEND_H)

    # Each baseline's name sits on its line, inside the plot, where nothing else is drawn. Lines too
    # close together for that end early, and their names follow in a column with leaders.
    x0, x1 = pd.to_datetime(list(fig.layout.xaxis.range))
    lo, hi = fig.layout.yaxis.range
    levels = np.array([(s.y0 - lo) / (hi - lo) for s in lines])
    widths = np.array([0.48 * LABEL_PX * len(a.text) for a in tiers])   # Arial, measured on a render
    taken = [(spot[0], spot[0] + box[0] / plot[0], spot[1] - box[1] / plot[1], spot[1])] if spot else []
    spots, ends, stacked = name_spots(mark, levels, widths, plot, taken, range(len(lines))), np.ones(len(lines)), {}
    crowded = np.flatnonzero((np.abs(levels[:, None] - levels[None, :]) < 1.5 * LABEL_PX / plot[1]).sum(1) > 1)
    if len(spots) < len(lines) and len(crowded):
        x_end, ys, column = name_stack(mark, levels[crowded], widths[crowded], plot, taken)
        ends[crowded], stacked = x_end, dict(zip(crowded, ys))
        spots = name_spots(mark, levels, widths, plot, taken + [column], set(range(len(lines))) - set(crowded))
    for n, (a, s) in enumerate(zip(tiers, lines)):
        if n in stacked:
            a.update(xref="x domain", x=ends[n] + (FAN - 3) / plot[0], xanchor="left", y=lo + stacked[n] * (hi - lo),
                     yanchor="middle")
            fig.add_shape(type="line", xref="x domain", yref="y", x0=ends[n], y0=s.y0, x1=ends[n] + (FAN - 5) / plot[0],
                          y1=a.y, line=dict(color=MID, width=0.5))
        else:
            x, side = spots.get(n, (0.996, "above"))           # no free place: at the right edge
            a.update(xref="x domain", x=x, xanchor="right", y=s.y0, yanchor="bottom" if side == "above" else "top",
                     yshift=2 if side == "above" else -2)
    kept = list(fig.data)
    for s, end in zip(lines, ends):                            # a line is a trace, to sit between traces
        fig.add_scatter(x=[x0.isoformat(), (x0 + end * (x1 - x0)).isoformat()], y=[s.y0, s.y0], mode="lines",
                        line=dict(color=MID, width=0.7, dash=s.line.dash), showlegend=False, hoverinfo="skip")
    fig.data = kept[:under] + list(fig.data[len(kept):]) + kept[under:]
    if tiers:                                                  # the grey lines, in the legend too
        fig.add_scatter(x=[None], y=[None], mode="lines", line=dict(color=MID, width=0.7),
                        name="Human baseline", legendrank=5, hoverinfo="skip")
    fig.add_annotation(text=f"<b>{label}</b>", xref="paper", yref="paper", x=0, y=1, xanchor="left",
                       yanchor="bottom", yshift=4, font=dict(size=12, color=INK))
    fig.update_layout(
        template="clean", title=None, font=dict(family="Arial, sans-serif", size=9.5, color=INK),
        width=None, height=None, showlegend=True,
        # An opaque ground under the entries, no frame: a baseline line stops at the legend.
        legend=dict(font=dict(size=10, color=INK), bgcolor=BG, borderwidth=0, **legend),
        margin=dict(l=4, r=RIGHT, t=24, b=4))
    fig.update_xaxes(showline=True, linecolor=INK, linewidth=1, ticks="outside", tickcolor=INK, ticklen=4,
                     showgrid=False, tickfont=dict(size=12, color=INK),
                     title=dict(text="Release date", font=dict(size=12, weight=500)))
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
            fig, height = restyle(fig, label)
            save(fig, HERE / f"fig{fig_no}_axis{k}_mode{mode}", height / ZOOM)
            print(f"mode {mode} ({n_chains} chains), axis {k}: prints {height / ZOOM:.0f} px tall")
