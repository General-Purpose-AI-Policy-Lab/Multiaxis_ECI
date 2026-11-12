"""Figure 9: the frontier of two groups of models on the single-axis index, all benchmarks.

Top panel closed against open-weights models, bottom panel US against Chinese models. Reads what
4_diagnostics/1_frontier_gap.py wrote for the all-benchmarks scope (candidates, frontier points and
the trend-line draws in 5_outputs/data20261019/comparisons/, through the repo's loader); no trace.
Each group: its models (tint), its frontier points with their interval, its trend line and the
trend's 80 % band. Prints the two 12-month lags of the summary tables.
Writes fig9_frontier_gaps.pdf and .png beside this file.

    python 6_writeups/preprint/figures/fig9_frontier_gaps.py
"""
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path[:0] = [str(HERE), str(ROOT / "2_model")]

from multiaxis_eci.analysis.frontier_gap import GROUP_TITLES, load_scope  # noqa: E402
from multiaxis_eci.viz.forecast import _today  # noqa: E402
from style import BG, INK, MID, PALETTE, TINTS, save  # noqa: E402

DATA = ROOT / "5_outputs/data20261019/comparisons"
WINDOW = (pd.Timestamp("2024-01-01"), pd.Timestamp("2027-07-01"))
PANELS = (("openness", "Closed and open-weights models"), ("country", "US and Chinese models"))

fig = make_subplots(rows=len(PANELS), cols=1, shared_xaxes=True, vertical_spacing=0.09)
for row, (kind, label) in enumerate(PANELS, start=1):
    gap = load_scope(DATA, kind, "all")
    yref = "y" if row == 1 else f"y{row}"
    shown = []
    for i, g in enumerate((gap.leader, gap.follower)):            # leader first: palette order
        tl = g.tl[(g.tl["release_date"] >= WINDOW[0]) & (g.tl["release_date"] <= WINDOW[1])]
        cloud, front = tl[~tl["is_record"]], tl[tl["is_record"]]
        shown += [tl["mean"], front["hdi_low"], front["hdi_high"]]
        fc = g.forecast(horizon=WINDOW[1])
        gx = pd.to_datetime(fc.grid_dates)
        fig.add_scatter(x=cloud["release_date"], y=cloud["mean"], mode="markers", showlegend=False,
                        marker=dict(color=TINTS[i], size=4, line=dict(width=0.5, color=BG)), row=row, col=1)
        fig.add_scatter(x=gx, y=fc.lo, mode="lines", line=dict(width=0), showlegend=False, row=row, col=1)
        fig.add_scatter(x=gx, y=fc.hi, mode="lines", line=dict(width=0), fill="tonexty",
                        fillcolor=TINTS[i], showlegend=False, row=row, col=1)
        fig.add_scatter(x=gx, y=fc.median, mode="lines", showlegend=False,
                        line=dict(color=PALETTE[i], width=1.2, dash="6px,4px"), row=row, col=1)
        fig.add_scatter(x=front["release_date"], y=front["mean"], mode="markers", showlegend=False,
                        marker=dict(color=PALETTE[i], size=7, line=dict(width=1, color=BG)),
                        error_y=dict(type="data", symmetric=False, array=front["hdi_high"] - front["mean"],
                                     arrayminus=front["mean"] - front["hdi_low"], color=PALETTE[i],
                                     thickness=0.8, width=0), row=row, col=1)
        fig.add_annotation(xref="paper", x=1, yref=yref, y=float(fc.median[-1]), xanchor="left", xshift=5,
                           text=GROUP_TITLES[kind][g.label], font=dict(size=10, color=PALETTE[i]))
    lo, hi = min(s.min() for s in shown), max(s.max() for s in shown)
    fig.update_yaxes(title_text="Single-axis index", range=[lo - 0.05 * (hi - lo), hi + 0.12 * (hi - lo)],
                     row=row, col=1)
    fig.add_vline(x=_today(None).strftime("%Y-%m-%d"), line=dict(color=MID, width=0.7, dash="2px,3px"),
                  row=row, col=1)
    if row == 1:
        fig.add_annotation(x=_today(None).strftime("%Y-%m-%d"), xref="x", yref=f"{yref} domain", y=1, text="today",
                           xanchor="right", yanchor="top", xshift=-4, font=dict(size=9.5, color=MID))
    fig.add_annotation(text=f"<b>{label}</b>", xref="paper", x=0, yref=f"{yref} domain", y=1, xanchor="left",
                       yanchor="bottom", yshift=4, font=dict(size=12, color=INK))
    summary = pd.read_csv(DATA / f"frontier_gap_{kind}_all_summary.csv").set_index("quantity")
    for quantity, meaning in (("envelope_lag_last_12m_mean", "mean lag of the last 12 months' frontier points"),
                              ("lag_12m_mean", "mean lag over the last 12 months, frontier to frontier")):
        q = summary.loc[quantity]
        print(f"{label}: {GROUP_TITLES[kind][gap.follower.label]} behind by {q['median']:.1f} months "
              f"(80 %: {q['hdi80_low']:.1f} to {q['hdi80_high']:.1f}), {meaning}")
for name, kw in (("Frontier point, with its interval", dict(mode="markers", marker=dict(color=INK, size=7))),
                 ("Other models", dict(mode="markers", marker=dict(color=MID, size=4, opacity=0.5))),
                 ("Frontier trend (band: 80 % interval)",
                  dict(mode="lines", line=dict(color=INK, width=1.2, dash="6px,4px")))):
    fig.add_scatter(x=[None], y=[None], name=name, **kw)
fig.update_xaxes(range=[w.strftime("%Y-%m-%d") for w in WINDOW], dtick="M12", tickformat="%Y")
fig.update_xaxes(title_text="Release date", row=len(PANELS), col=1)
fig.update_layout(margin=dict(t=24, r=118), legend=dict(orientation="h", x=0, xanchor="left", y=-0.075,
                                                        yanchor="top"))
save(fig, HERE / "fig9_frontier_gaps", height=620)
