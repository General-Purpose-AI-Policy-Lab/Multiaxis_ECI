"""Summary-table image for the 3-scope US/CN frontier comparison.

Reads the per-scope frontier CSVs plus the per-draw npz dumps
3_diagnostics/1_country_frontier.py writes, computes deltas vs the all-benchmarks
scope per draw (cross-scope draws are independent posteriors), and renders
one PNG of the two tables.

  python deliverables/us_cn_frontier/make_tables.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from multiaxis_eci import config  # noqa: E402

TAGS = {"all": "canonical", "open": "canonical_open", "closed": "canonical_closed"}
OUT = Path(__file__).parent / "summary_tables.png"


def fmt(med, lo, hi, signed=True):
    s = "+" if signed else ""
    return f"{med:{s}.1f}  [{lo:{s}.1f}, {hi:{s}.1f}]"


def med95(x):
    x = x[np.isfinite(x)]
    lo, hi = az.hdi(x, hdi_prob=0.95)
    return float(np.median(x)), float(lo), float(hi)


def main():
    cmp_dir = config.RESULTS_DIR / "comparisons"
    csv = {k: pd.read_csv(cmp_dir / f"country_frontier_{t}.csv")
           for k, t in TAGS.items()}
    npz = {k: np.load(cmp_dir / f"country_frontier_draws_{t}.npz")
           for k, t in TAGS.items()}

    def row(k, country):
        d = csv[k]
        return d[(d.country == country) & (d.variant == "informed")].iloc[0]

    def slope_cell(k, c):
        r = row(k, c)
        return fmt(r.eci_slope_median, r.eci_slope_hdi95_low, r.eci_slope_hdi95_high)

    def dslope_cell(k):
        m, lo, hi = med95(npz[k]["slope_us_eci"] - npz[k]["slope_cn_eci"])
        return fmt(m, lo, hi)

    def delta_cell(key, k):
        n = min(len(npz[k][key]), len(npz["all"][key]))
        m, lo, hi = med95(npz[k][key][:n] - npz["all"][key][:n])
        return fmt(m, lo, hi)

    slopes = [
        ["US", "CN", "Δ US−CN"],
        [slope_cell("all", "US"), slope_cell("all", "CN"), dslope_cell("all")],
        [slope_cell("open", "US"), slope_cell("open", "CN"), dslope_cell("open")],
        [slope_cell("closed", "US"), slope_cell("closed", "CN"), dslope_cell("closed")],
        [delta_cell("slope_us_eci", "open"), delta_cell("slope_cn_eci", "open"), "—"],
        [delta_cell("slope_us_eci", "closed"), delta_cell("slope_cn_eci", "closed"), "—"],
    ]

    def gap_cell(k, which):
        r = row(k, "CN")
        if which == "gap":
            return fmt(r.gap_at_last_common_eci_median, r.gap_hdi95_low,
                       r.gap_hdi95_high, signed=False)
        return fmt(r.lag_months_median, r.lag_hdi95_low, r.lag_hdi95_high,
                   signed=False)

    gaps = [
        ["US−CN gap (ECI)", "CN lag (months)"],
        [gap_cell("all", "gap"), gap_cell("all", "lag")],
        [gap_cell("open", "gap"), gap_cell("open", "lag")],
        [gap_cell("closed", "gap"), gap_cell("closed", "lag")],
        [delta_cell("gap_eci", "open"), delta_cell("lag_months", "open")],
        [delta_cell("gap_eci", "closed"), delta_cell("lag_months", "closed")],
    ]

    HDR = dict(fill_color="#1f3a5f", font=dict(color="white", size=14),
               align="left", height=30)
    CEL = dict(fill_color=[["#f7f9fc", "white"] * 4], font=dict(size=13.5),
               align="left", height=28)
    cols = ["", "all", "open only", "closed only",
            "Δ open−all", "Δ closed−all"]
    fig = make_subplots(rows=2, cols=1,
                        specs=[[{"type": "table"}], [{"type": "table"}]],
                        subplot_titles=("Frontier slopes (ECI/yr) — median [95% HDI]",
                                        "Gap and lag — median [95% HDI]"),
                        row_heights=[0.55, 0.45], vertical_spacing=0.16)
    fig.add_trace(go.Table(header=dict(values=cols, **HDR),
                           cells=dict(values=slopes, **CEL)), row=1, col=1)
    fig.add_trace(go.Table(header=dict(values=cols, **HDR),
                           cells=dict(values=gaps, **CEL)), row=2, col=1)
    fig.update_layout(width=1400, height=400, margin=dict(l=20, r=20, t=60, b=5),
                      title=dict(text="US vs China frontier — 3 scopes, quick fits "
                                     "(4×2,000), 2026-08-27", x=0.5,
                                 font=dict(size=15)))
    fig.write_image(str(OUT), scale=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
