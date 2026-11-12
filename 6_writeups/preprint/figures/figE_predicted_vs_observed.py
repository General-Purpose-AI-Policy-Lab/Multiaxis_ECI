"""Appendix E figure: observed score against the multi-axis model's posterior-mean prediction.

Reads 5_outputs/data20261019/mirt_lineageprior_lineagebm/mirt_residuals.csv (model, benchmark,
score, pred_mean, residual; one row per observation). Prints RMSE and the correlation.
Writes figE_predicted_vs_observed.{pdf,png} next to this file, at the paper's text width.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from style import BG, MID, PALETTE, save

HERE = Path(__file__).resolve().parent
df = pd.read_csv(HERE.parents[2] / "5_outputs/data20261019/mirt_lineageprior_lineagebm/mirt_residuals.csv")
rmse = np.sqrt(np.mean((df.score - df.pred_mean) ** 2))
print(f"{len(df)} obs, RMSE {rmse:.4f}, correlation {np.corrcoef(df.score, df.pred_mean)[0, 1]:.4f}")

fig = go.Figure()
fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="observed = predicted",
                line=dict(color=MID, width=1.2))
fig.add_scatter(x=df.pred_mean, y=df.score, mode="markers", name="score",
                marker=dict(size=4, color=PALETTE[0], line=dict(width=0.5, color=BG)))
fig.update_xaxes(title_text="Predicted score (posterior mean)", range=[-0.02, 1.02], dtick=0.2,
                 constrain="domain")
fig.update_yaxes(title_text="Observed score", range=[-0.02, 1.02], dtick=0.2,
                 scaleanchor="x", scaleratio=1, constrain="domain")
fig.update_layout(legend=dict(x=0.02, y=0.98))
save(fig, HERE / "figE_predicted_vs_observed", height=520)
