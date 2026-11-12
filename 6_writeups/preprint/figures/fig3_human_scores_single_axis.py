"""Fig 3: where each human score falls in the single-axis model's leave-one-out prediction.

Reads 6_writeups/preprint/single_axis/human_pit.csv (one row per human score: model, benchmark,
score, pit, pareto_k). Rows are grouped under their human baseline, sorted by pit inside a group.
Writes fig3_human_scores_single_axis.{pdf,png} next to this file, at the paper's text width.
"""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from style import BG, INK, LIGHT, PALETTE, save

HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE.parent / "single_axis/human_pit.csv")
extreme = (df.pit < 0.05) | (df.pit > 0.95)
print(f"{extreme.sum()} / {len(df)} scores in the bottom or top 5 %")

# One header row per baseline (largest group first), then its benchmarks.
order = df.model.value_counts().index
y, text, rows = [], [], []
for tier in order:
    text.append(f"<b>{tier}</b>"); y.append(len(y))
    for _, r in df[df.model == tier].sort_values("pit").iterrows():
        text.append(r.benchmark); y.append(len(y)); rows.append((y[-1], r.pit, r.pareto_k > 0.7))
rows = pd.DataFrame(rows, columns=["y", "pit", "bad_k"])

fig = go.Figure()
for x0, x1 in [(0, 0.05), (0.95, 1)]:
    fig.add_vrect(x0=x0, x1=x1, fillcolor=LIGHT, opacity=0.6, line_width=0, layer="below")
fig.add_scatter(x=[None], y=[None], mode="markers", name="bottom or top 5 % of the prediction",
                marker=dict(symbol="square", size=10, color=LIGHT, line_width=0))
for bad, name in [(False, "human score"),
                  (True, "human score, leave-one-out estimate unreliable (Pareto k > 0.7)")]:
    g = rows[rows.bad_k == bad]
    fig.add_scatter(x=g.pit, y=g.y, mode="markers", name=name,
                    marker=dict(color=BG if bad else PALETTE[0],
                                line=dict(color=PALETTE[0] if bad else BG, width=1.5 if bad else 1)))

fig.update_xaxes(title_text="Position of the observed score in the model's prediction (LOO-PIT)",
                 range=[-0.025, 1.025], dtick=0.1)
fig.update_yaxes(tickvals=y, ticktext=text, range=[len(y) - 0.5, -0.7], ticksuffix=" ")
fig.update_layout(legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom", xanchor="left",
                              font_color=INK))
save(fig, HERE / "fig3_human_scores_single_axis", height=760)
