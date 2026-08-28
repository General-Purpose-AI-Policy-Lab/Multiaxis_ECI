"""Human-tier crossover dates as a 2x3 panel grid: US/CN x scope, where the
scopes are all benchmarks / open-only / closed-only.

Reads the per-scope CSVs diagnostics/country_frontier.py writes — no trace is
loaded, so this re-renders instantly after a country_frontier run.

  python diagnostics/plot_crossovers.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from viz.core import save_fig  # noqa: E402

US_COLOR, CN_COLOR = "#1A9641", "#D7191C"   # green US / red CN; the darkened
                                            # shades keep the pair separable
                                            # for deuteranopes
TODAY = pd.Timestamp("2026-08-27")

# Panel layout: rows = country, cols = scope.
PANELS = [
    ("US", "canonical",        1, 1, US_COLOR, "US · all benchmarks"),
    ("US", "canonical_open",   1, 2, US_COLOR, "US · open only"),
    ("US", "canonical_closed", 1, 3, US_COLOR, "US · closed only"),
    ("CN", "canonical",        2, 1, CN_COLOR, "CN · all benchmarks"),
    ("CN", "canonical_open",   2, 2, CN_COLOR, "CN · open only"),
    ("CN", "canonical_closed", 2, 3, CN_COLOR, "CN · closed only"),
]
DATE_COLS = ["crossover_date_median", "crossover_hdi_low", "crossover_hdi_high",
             "crossover_hdi95_low", "crossover_hdi95_high"]


def main():
    cmp_dir = config.RESULTS_DIR / "comparisons"
    dfs = {tag: pd.read_csv(cmp_dir / f"country_crossover_{tag}.csv",
                            parse_dates=DATE_COLS)
           for tag in ("canonical", "canonical_open", "canonical_closed")}

    # One y slot per tier, ordered weakest at bottom by the all-scope ECI so
    # every panel reads upward like the frontier figures.
    order = (dfs["canonical"].drop_duplicates("tier")
             .sort_values("human_eci_median")["tier"].tolist())

    # Shared x-range over everything drawn, so every panel is comparable.
    all_dates = pd.concat([dfs[t][c].dropna() for t in dfs for c in DATE_COLS])
    x0 = (all_dates.min() - pd.DateOffset(months=3)).strftime("%Y-%m-%d")
    x1 = (all_dates.max() + pd.DateOffset(months=3)).strftime("%Y-%m-%d")

    fig = make_subplots(rows=2, cols=3, shared_xaxes=True, shared_yaxes=True,
                        subplot_titles=[p[5] for p in PANELS],
                        horizontal_spacing=0.04, vertical_spacing=0.09)

    for country, tag, row, col, color, _label in PANELS:
        d = dfs[tag]
        d = d[(d.country == country) & d.crossover_date_median.notna()]
        y = [order.index(t) for t in d.tier]
        # One forest segment per tier, spanning the 95% HDI. ISO strings, not
        # Timestamps — kaleido's JSON encoder rejects a bare pd.Timestamp
        # inside a plain list.
        seg_x, seg_y = [], []
        for lo, hi, yy in zip(d.crossover_hdi95_low, d.crossover_hdi95_high, y):
            seg_x += [lo.strftime("%Y-%m-%d"), hi.strftime("%Y-%m-%d"), None]
            seg_y += [yy, yy, None]
        fig.add_trace(go.Scatter(
            x=seg_x, y=seg_y, mode="lines", showlegend=False,
            line=dict(color=color, width=2.4), opacity=0.9,
            hoverinfo="skip"), row=row, col=col)
        fig.add_trace(go.Scatter(
            x=d.crossover_date_median, y=y, mode="markers", showlegend=False,
            marker=dict(size=10, color=color, symbol="diamond",
                       line=dict(width=1, color="white")),
            text=[f"P(passed now)={p:.2f}" for p in d.p_passed_now],
            hovertemplate="%{x|%Y-%m}<br>%{text}<extra></extra>"),
            row=row, col=col)
        fig.add_vline(x=TODAY.strftime("%Y-%m-%d"),
                      line=dict(color="#666666", dash="dot", width=1.2),
                      row=row, col=col)

    fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                             line=dict(color="#666666", dash="dot", width=1.2),
                             name=f"today ({TODAY:%Y-%m-%d})", hoverinfo="skip"))

    fig.update_xaxes(type="date", range=[x0, x1])
    fig.update_yaxes(tickmode="array", tickvals=list(range(len(order))),
                     ticktext=order, range=[-0.6, len(order) - 0.4])
    fig.update_layout(
        template="plotly_white",
        title=dict(text="Frontier crossover of human tiers — "
                       "median, 95% HDI", x=0.5),
        legend=dict(orientation="h", yanchor="bottom", y=-0.12,
                   xanchor="center", x=0.5),
        margin=dict(l=200, r=40, t=90, b=80), height=820, width=1650)

    save_fig(fig, "country_crossovers", config.PLOTS_DIR)
    print(f"wrote {config.PLOTS_DIR / 'country_crossovers.html'}")


if __name__ == "__main__":
    main()
