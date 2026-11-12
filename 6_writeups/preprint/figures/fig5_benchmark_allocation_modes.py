"""Fig 5: each benchmark's dominant axis in posterior mode A against posterior mode C.

Reads the multi-axis fit's loading tables, mode A (5_outputs/data20261019/mirt_lineageprior_lineagebm/
mirt_loadings.csv) and mode C (the same folder's secondary_mode/mirt_loadings.csv), and each folder's
axis_names.json. A benchmark's dominant axis is the axis with its largest axis_share.
Writes fig5_benchmark_allocation_modes.{pdf,png} next to this file, at the paper's text width.
"""
import json
import textwrap
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from style import BG, INK, PALETTE, TINTS, save

HERE = Path(__file__).resolve().parent
FIT = HERE.parents[2] / "5_outputs/data20261019/mirt_lineageprior_lineagebm"
AXES = ["axis1", "axis2", "axis3", "axis4"]


def dominant(folder):
    df = pd.read_csv(folder / "mirt_loadings.csv")
    names = json.loads((folder / "axis_names.json").read_text())["axes"]
    return df.loc[df.groupby("benchmark").axis_share.idxmax()].set_index("benchmark").axis, names


def wrap(s, n):
    return "<br>".join(textwrap.wrap(s, n))


a, names_a = dominant(FIT)
c, names_c = dominant(FIT / "secondary_mode")
tab = pd.crosstab(a, c.reindex(a.index)).reindex(index=AXES, columns=AXES, fill_value=0)
print(tab, f"\n{tab.values.sum()} benchmarks")
assert tab.values.tolist() == [[22, 15, 1, 0], [7, 0, 4, 18], [1, 1, 19, 2], [0, 13, 1, 3]], "differs from NOTE.md s4"

rows = [wrap(f"A{i + 1} {names_a[k]['title']} ({tab.loc[k].sum()})", 20) for i, k in enumerate(AXES)]
cols = [wrap(f"C{i + 1} {names_c[k]['title']}", 14) for i, k in enumerate(AXES)]
zmax = tab.values.max()
fig = go.Figure(go.Heatmap(
    z=tab.values, x=cols, y=rows, colorscale=[[0, TINTS[0]], [1, PALETTE[0]]], zmin=0, zmax=zmax,
    xgap=2, ygap=2, hoverinfo="skip",
    colorbar=dict(title=dict(text="Benchmarks", side="right", font=dict(size=12, weight=500)),
                  tickfont_size=12, thickness=12, outlinewidth=0, ticks="")))
for i, r in enumerate(rows):          # cell counts: INK on light cells, BG on dark ones
    for j, col in enumerate(cols):
        v = tab.values[i, j]
        fig.add_annotation(x=col, y=r, text=str(v), font=dict(size=12, color=BG if v > 0.7 * zmax else INK))
fig.update_xaxes(title_text="Dominant axis in mode C", showline=False, ticks="", tickangle=0)
fig.update_yaxes(title_text="Dominant axis in mode A", autorange="reversed", showgrid=False)
save(fig, HERE / "fig5_benchmark_allocation_modes", height=420)
