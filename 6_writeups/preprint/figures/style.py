"""The figure template of the preprint: section 1 of ../PLOTLY_STYLE_GUIDE.md, copied as is.

Importing this module registers the Plotly template "clean" and makes it the default. A figure
script imports PALETTE, TINTS, INK, MID, LIGHT, BG, ramp and label_line_ends from here, builds at
its final size (W, the paper's text width, 160 mm) and writes a vector PDF plus a PNG preview with
`save`. No title goes inside a figure: the paper's caption carries it.
"""
import plotly.graph_objects as go
import plotly.io as pio
from plotly.colors import sample_colorscale

# Categorical palette, in order of use. Colour-blind-safe in this order.
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
#           blue       orange     aqua       yellow     magenta    green      violet     red
# Light tint of each colour, same order: secondary data, uncertainty bands.
TINTS = ["#b8d2f1", "#f7c5b1", "#9dddc6", "#f5ca6f", "#f4c2d5", "#b2dab2", "#d1cbf3", "#f6c3c3"]

INK = "#262626"    # text, visible axis, ticks
MID = "#6e6e6e"    # secondary text, reference lines
LIGHT = "#dcdcdc"  # grid, hatching
BG = "white"       # "rgba(0,0,0,0)" when the figure sits on a tinted page
FONT = "Arial, sans-serif"  # one family for everything; must be installed where you export

# Sizes are in px. Exported at final size and inserted at 100 %, 1 px = 0.75 pt.
_grid_axis = dict(
    showline=False, ticks="", showgrid=True, gridcolor=LIGHT, gridwidth=0.7, zeroline=False,
    tickfont=dict(size=12), title=dict(font=dict(size=12, weight=500), standoff=6),
    automargin=True,
)
_line_axis = dict(
    _grid_axis, showgrid=False, showline=True, linecolor=INK, linewidth=1,
    ticks="outside", tickcolor=INK, ticklen=4,
)

pio.templates["clean"] = go.layout.Template(
    layout=dict(
        colorway=PALETTE,
        font=dict(family=FONT, size=9.5, color=INK),
        paper_bgcolor=BG, plot_bgcolor=BG,
        margin=dict(l=4, r=4, t=4, b=4),  # automargin adds the room the axes need
        xaxis=_line_axis,                 # one visible axis...
        yaxis=_grid_axis,                 # ...the other one carried by the grid
        legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)", borderwidth=0),
        annotationdefaults=dict(font=dict(size=9.5), showarrow=False),
        shapedefaults=dict(line=dict(color=MID, width=1), layer="below"),
        hoverlabel=dict(font=dict(family=FONT, size=12)),
    ),
    data=dict(
        scatter=[go.Scatter(
            line=dict(width=2),
            marker=dict(size=7, opacity=1, line=dict(width=1, color=BG)),  # gap ring
        )],
        bar=[go.Bar(marker=dict(line=dict(width=0)))],
    ),
)
pio.templates.default = "clean"


def ramp(i, n):
    """n shades of palette colour i, from its tint to the full colour (ordered series)."""
    # Interpolated in sRGB. Switch to OKLab if the steps look uneven.
    return sample_colorscale([[0, TINTS[i]], [1, PALETTE[i]]], [k / (n - 1) for k in range(n)])


def label_line_ends(fig):
    """Replace the legend by a label at the end of each named line, in the series colour."""
    named = [(i, t) for i, t in enumerate(fig.data) if t.name and t.showlegend is not False]
    for i, t in named:
        fig.add_annotation(  # in the right margin, so the grid does not cross the text
            xref="paper", x=1, y=t.y[-1], text=t.name, xanchor="left", xshift=4,
            font=dict(size=10, color=t.line.color or PALETTE[i % len(PALETTE)]),
        )
    # Rough room for the labels: about 6 px per character at 10 px.
    fig.update_layout(showlegend=False, margin_r=10 + 6 * max(len(t.name) for _, t in named))


W = round(160 * 96 / 25.4)   # full text width in px: 160 mm at 96 px per inch


def save(fig, path, height, width=W):
    """Write `path`.pdf (vector, final size) and `path`.png (preview, twice the pixels)."""
    fig.write_image(f"{path}.pdf", width=width, height=height)
    fig.write_image(f"{path}.png", width=width, height=height, scale=2)
