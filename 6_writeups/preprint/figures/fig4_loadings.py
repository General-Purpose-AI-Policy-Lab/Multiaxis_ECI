"""Fig 4: the benchmarks that load most on each axis, one figure per posterior mode (A and C).

Reads mirt_loadings.csv (axis, benchmark, loading_median, hdi_low, hdi_high, axis_share) and
axis_names.json of the multi-axis fit, mode A in 5_outputs/data20261019/mirt_lineageprior_lineagebm/
and mode C in its secondary_mode/. Per axis: the 10 benchmarks with the largest axis_share, sorted by
share. Writes fig4_loadings_mode{A,C}.{pdf,png} next to this file, at the paper's text width.
"""
import json
from pathlib import Path

import pandas as pd
from plotly.subplots import make_subplots
from style import INK, MID, PALETTE, save

HERE = Path(__file__).resolve().parent
FIT = HERE.parents[2] / "5_outputs/data20261019/mirt_lineageprior_lineagebm"
AXES = ["axis1", "axis2", "axis3", "axis4"]

for mode, folder in [("A", FIT), ("C", FIT / "secondary_mode")]:
    df = pd.read_csv(folder / "mirt_loadings.csv")
    names = json.loads((folder / "axis_names.json").read_text())["axes"]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.055)
    for k, ax in enumerate(AXES, start=1):
        top = df[df.axis == ax].nlargest(10, "axis_share")
        print(f"mode {mode} {ax}: first {top.benchmark.iloc[0]}")
        fig.add_bar(x=top.loading_median, y=top.benchmark, orientation="h", row=k, col=1,
                    marker_color=PALETTE[0], showlegend=False, width=0.7,
                    error_x=dict(type="data", array=top.hdi_high - top.loading_median,
                                 arrayminus=top.loading_median - top.hdi_low,
                                 color=INK, thickness=1, width=2))
        yref = "y" if k == 1 else f"y{k}"
        for b, s in zip(top.benchmark, top.axis_share):       # share column in the right margin
            fig.add_annotation(xref="paper", x=1, xanchor="left", xshift=6, yref=yref, y=b,
                               text=f"{s:.2f}", font_color=MID)
        fig.add_annotation(xref="paper", x=0, xanchor="left", yref=f"{yref} domain", y=1,
                           yanchor="bottom", yshift=2, text=f"Axis {k}: {names[ax]['title']}",
                           font=dict(size=12, weight=500, color=INK))
    fig.add_annotation(xref="paper", x=1, xanchor="left", xshift=6, yref="y domain", y=1,
                       yanchor="bottom", yshift=2, text="share", font_color=MID)
    # Horizontal bars: the visible axis is y, at the bars' base; x carries the grid.
    fig.update_yaxes(autorange="reversed", showline=True, linecolor=INK, showgrid=False, ticks="")
    fig.update_xaxes(showline=False, ticks="", showgrid=True, gridcolor="#dcdcdc", rangemode="tozero")
    fig.update_xaxes(title_text="Loading (median, 95 % interval)", row=4, col=1)
    fig.update_layout(bargap=0.3, margin_r=40, margin_t=24)
    save(fig, HERE / f"fig4_loadings_mode{mode}", height=780)
