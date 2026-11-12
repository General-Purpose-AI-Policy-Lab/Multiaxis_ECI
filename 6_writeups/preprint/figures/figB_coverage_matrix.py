"""Appendix B figure: which model was scored on which benchmark, in the multi-axis fit's data.

Reads the fit's own data object (built by fit.py's parser through r1_runs/aligned_loadings.load_data,
no trace) and the release date of each model (multiaxis_eci.data._known_release_date_by_model).
Models are ordered by release date (undated last), benchmarks by the median release date of the
models scored on them. Prints the fill; writes figB_coverage_matrix.{pdf,png} next to this file, at
the paper's text width.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "2_model"))
sys.path.insert(0, str(HERE.parent / "r1_runs"))
import aligned_loadings as al  # noqa: E402
from multiaxis_eci.data import _known_release_date_by_model  # noqa: E402
from style import INK, PALETTE, save  # noqa: E402

data = al.load_data()
names = data.mlookup.set_index("model_idx").model.sort_index().values   # lookup is 1-based
dates = _known_release_date_by_model().reindex(names).values

obs = pd.DataFrame({"m": data.model_idx, "b": data.bench_idx, "date": dates[data.model_idx]})
m_order = pd.Series(dates).sort_values(na_position="last", kind="stable").index.values
b_order = obs.groupby("b").date.median().sort_values(na_position="last").index.values
m_pos = np.empty(data.n_models, int); m_pos[m_order] = np.arange(data.n_models)
b_pos = np.empty(data.n_benchmarks, int); b_pos[b_order] = np.arange(data.n_benchmarks)

z = np.full((data.n_models, data.n_benchmarks), np.nan)
z[m_pos[data.model_idx], b_pos[data.bench_idx]] = 1
cells = data.n_models * data.n_benchmarks
print(f"{data.n_obs} obs, {data.n_models} models, {data.n_benchmarks} benchmarks, "
      f"{pd.isna(dates).sum()} undated models")
print(f"{data.n_obs:,} of {cells:,} cells observed ({100 * data.n_obs / cells:.1f} %)")

years = pd.DatetimeIndex(dates[m_order]).year
first = pd.Series(np.arange(data.n_models), index=years).groupby(level=0).min()
keep = [first.index[-1]]                                 # walk down, drop ticks crowding the last kept
for y in first.index[::-1][1:]:
    if first[keep[-1]] - first[y] >= 25:
        keep.append(y)
first = first[sorted(keep)]
fig = go.Figure(go.Heatmap(z=z, colorscale=[[0, PALETTE[0]], [1, PALETTE[0]]], showscale=False,
                           hoverinfo="skip"))
# Categories on x carry no ticks and no line; the date axis (y) is the visible one, in INK.
fig.update_xaxes(title_text="Benchmarks, ordered by the median release date of the models scored on them",
                 showticklabels=False, showline=False, ticks="", range=[-0.5, data.n_benchmarks - 0.5])
fig.update_yaxes(title_text="Models, ordered by release date; a filled cell is an observed score",
                 tickvals=first.values, ticktext=[str(y) for y in first.index], showgrid=False,
                 showline=True, linecolor=INK, ticks="outside", tickcolor=INK, ticklen=4,
                 range=[-0.5, data.n_models - 0.5])
save(fig, HERE / "figB_coverage_matrix", height=720)
