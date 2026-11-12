# Plotly figure style guide

A brand-neutral set of design rules for data plots, with a Plotly template that applies the ones a template can apply. The goal is figures that read easily, look consistent with each other and survive print.

Section 1 is the template: paste it once, at the top of a script or in a shared module. Sections 2 to 8 are the rules, including those a template cannot enforce. Section 9 is a working example.

Requires Plotly 5.23 or later (font weight), and `kaleido` for static export.

## 1. Template

```python
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
```

## 2. Format and size

- Build each figure at its **final size** and insert it at 100 %. A figure resized afterwards has text of a different size from its neighbours.
- Set the size at export: `fig.write_image("fig.svg", width=W, height=H)`, with `W = mm × 96 / 25.4`.
- Deliver **vector** output (SVG or PDF). PNG is for previews only.
- **No title inside the figure** when the document gives it a caption. Keep the labels and annotations needed to read it.
- Use a transparent background (`BG`) when the page is not white, so that the paper shows through. Then check the figure on that page colour.

## 3. Type

- **One font family** for everything.
- **One size per role.** Every text of the same role has the same size in every figure.

| Role | Size | Weight |
| --- | --- | --- |
| Tick values | 12 px (9 pt) | Regular |
| Axis titles | 12 px (9 pt) | Medium |
| Legend, and direct labels that replace it | 10 px (7.5 pt) | Regular, series colour for direct labels |
| Annotations, point names | 9.5 px (7 pt) | Regular |

- These sizes are for print-size figures. For a screen-only figure, multiply all of them by the same factor.
- Format numbers for the language of the document: decimal and thousands separators with `layout.separators`, a non-breaking space before a unit where the language asks for one. Use `×` for factors and `−` for minus, not the letters `x` and `-`.

## 4. Colour

- Use the palette **in order**. One series takes the first colour, two take the first two.
- A colour has **one meaning per figure**, and an entity keeps its colour from one figure to the next.
- Three neutrals do all the rest: `INK` for text and the visible axis, `MID` for secondary text and reference lines not tied to a series, `LIGHT` for the grid and background hatching.
- Each colour has three uses:
  - **Full**: main data.
  - **Tint**: secondary data of the same series, for example the points that are not on the frontier. Not the mid grey.
  - **Ramp**: `ramp(i, n)`, one shade per item from tint to full colour, for ordered items of one family. Never darker than the full colour.
- Distinct categories take different colours. An ordered progression takes a ramp of one colour.
- A fitted model or trend takes **the colour of the series it fits**. So does an annotation attached to a series (an arrow, a `+150 %`).
- More than eight series: group the small ones into "Other" or split into small multiples. Do not invent a ninth colour.
- How the tints are made: each colour is mixed with white until its contrast with the background is 1.55, so all tints are equally visible. Recompute them if the background changes.
- Aqua, yellow and magenta have less than 3:1 contrast on white. Avoid them for full/tint pairs, and if a label in one of them is hard to read, write it in `INK` next to a coloured marker.

## 5. Axes and grid

- **One visible axis**, in `INK`: usually x; for bars, the axis at their base. The other axis has no line and no tick marks, only grid lines and tick values.
- No grid on the visible axis.
- Tick marks for dates and quantities, **not for categories**: `fig.update_xaxes(ticks="")`.
- Horizontal bars: swap the roles.

  ```python
  fig.update_yaxes(showline=True, linecolor=INK, showgrid=False)
  fig.update_xaxes(showline=False, ticks="", showgrid=True, gridcolor=LIGHT)
  ```

- Bars with the value written on each bar: remove the tick values and the grid of the value axis, they repeat the labels.
- **One y scale per plot.** Two measures with different scales take two plots.

## 6. Lines and markers

| Element | Width |
| --- | --- |
| Data line | 2 px (1.5 pt) |
| Trend or fitted line | 1.2 px (0.9 pt), set per trace |
| Grid | 0.7 px (0.5 pt) |

- A trend line is clearly thinner than the data it fits.
- Markers are **opaque**, with a thin ring in the background colour that separates them from what lies under them. On a transparent background the ring cannot be drawn in Plotly: reduce the overlap instead.
- An annotated point keeps its colour and gets a **larger marker** (`size=10`).
- A darker or lighter colour separates categories (frontier or not). It does not mean "this point is annotated".
- Decide line styles (solid, dashed, dotted) by role and keep them the same in all figures.

## 7. Layer order

Plotly draws shapes with `layer="below"` first, then traces in the order they are added, then annotations. Add things from the bottom up:

1. Background zones (`add_vrect`, `add_hrect`).
2. Reference lines ("today", a threshold).
3. Secondary data, in tint.
4. Uncertainty of the model (`fill="tonexty"` in the tint, line width 0).
5. Model or trend.
6. Raw data, with error bars.
7. Annotated points.
8. Annotations: arrow, then text.

Bar traces are always drawn under scatter traces, whatever the order.

## 8. Annotations and legend

- Put an annotation next to what it describes, above or beside it, with no leader line if possible.
- No overlap between annotations, or between an annotation and data.
- **No box** around annotations. Where a text has to cross a line or the grid, give it `bgcolor=BG` and `borderpad=1` so the line stops around the letters. Only there.
- Keep **arrows for an explanation**, for example `+150 %` between two dates. A point name needs no arrow.
- With few series, prefer **direct labels** at the end of the lines to a legend: `label_line_ends(fig)`.
- The legend has **no frame**.
- Keep credits and source mentions.

## 9. Example

```python
import math

years = list(range(2018, 2026))
a = [12, 15, 21, 26, 35, 41, 52, 60]
b = [30, 31, 33, 32, 36, 37, 39, 40]
trend = [10 + 7 * (y - 2018) for y in years]
noise = [v + 6 * math.sin(3 * k) for k, v in enumerate(a)]

fig = go.Figure()
fig.add_vline(x=2023, line_dash="dot")                                   # reference line
fig.add_scatter(x=years, y=noise, mode="markers", showlegend=False,     # secondary data, tint
                marker_color=TINTS[0])
fig.add_scatter(x=years, y=trend, mode="lines", showlegend=False,       # trend, series colour
                line=dict(width=1.2, color=PALETTE[0]))
fig.add_scatter(x=years, y=a, mode="lines+markers", name="Series A", line_color=PALETTE[0])
fig.add_scatter(x=years, y=b, mode="lines+markers", name="Series B", line_color=PALETTE[1])
fig.add_annotation(x=2023, y=62, text="Policy change", xanchor="right", xshift=-4,
                   font_color=MID)
fig.update_yaxes(title_text="Share of users", ticksuffix="%")
label_line_ends(fig)

fig.write_image("example.svg", width=520, height=300)       # 137.6 mm wide
fig.write_image("example.png", width=520, height=300, scale=2)  # preview only
```

## Checklist

- Exported at final size, vector, no title in the image.
- One font, one size per role.
- Colours in palette order, one meaning each, model in the colour of its series.
- One visible axis, grid on the other, no ticks on categories.
- Trend thinner than data, markers opaque.
- No overlapping text, no boxes, arrows only for explanations.
- Looked at the rendered figure on the real page colour.
