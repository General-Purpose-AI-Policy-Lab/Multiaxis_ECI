"""Type and marker scales shared by every figure builder.

Two scales, one look. `DASHBOARD` is the on-screen scale of the dashboard cards and the
per-fit figure folders; `POST` is the print scale of the blog post, where a figure is shared
flat at about 2,000 pixels and must read without zooming (type 2.3x, markers 1.9x). A builder
takes a `FigureStyle` and reads every size from it, so the post's scripts differ from the
outputs only by the style they pass, never by post-processing the figure.
"""
from __future__ import annotations

from dataclasses import dataclass

import plotly.graph_objects as go


@dataclass(frozen=True)
class FigureStyle:
    """Sizes in pixels (fonts, markers, line widths) and the figure's outer dimensions."""

    font_title: int = 18
    font_axis: int = 14
    font_tick: int = 12
    font_legend: int = 12
    font_tier: int = 12       # human tier names written at the right of their lines
    font_note: int = 12       # secondary notes (sample sizes, clipped dates)
    marker: int = 7           # model points
    marker_median: int = 10   # a median dot on an interval bar
    errbar: float = 1.4       # interval bar line width
    bar_thick: float = 6.0    # the wide interval bar (50 %)
    bar_thin: float = 2.0     # the narrow interval bar (80 %)
    trend: float = 2.2        # trend median dash
    refline: float = 1.3      # tier lines and the today line
    width: int = 1100
    height_per_row: int = 460
    name: str = "dashboard"


DASHBOARD = FigureStyle()
POST = FigureStyle(font_title=42, font_axis=36, font_tick=30, font_legend=30, font_tier=26,
                   font_note=19, marker=13, marker_median=17, errbar=2.6, bar_thick=11.0,
                   bar_thin=4.0, trend=5.0, refline=2.2, width=2200, height_per_row=800,
                   name="post")


def apply_fonts(fig: go.Figure, style: FigureStyle) -> go.Figure:
    """Set the title, axis, tick and legend font sizes of a finished figure from `style`.

    Panel titles (subplot annotations) follow the axis size. Used by builders after the layout
    is assembled, so every builder sizes its text the same way.
    """
    fig.update_layout(title_font_size=style.font_title,
                      legend_font_size=style.font_legend)
    for ax in list(fig.select_xaxes()) + list(fig.select_yaxes()):
        ax.tickfont.size = style.font_tick
        if ax.title.font.size is None:          # an explicit caption size wins
            ax.title.font.size = style.font_axis
    for ann in fig.layout.annotations or ():
        if ann.font.size is None:
            ann.font.size = style.font_axis
    if fig.layout.legend and fig.layout.legend.grouptitlefont is not None:
        fig.layout.legend.grouptitlefont.size = style.font_legend
    return fig
