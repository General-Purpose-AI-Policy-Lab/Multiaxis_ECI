"""1D ECI timeline with benchmark difficulties, in the dashboard's Plotly style.

Same content as the matplotlib draft (`make_timeline_difficulties.py`) but drawn
by the dashboard's own `viz.core.capability_timeline_fig`, so the post and the
dashboard cannot drift apart. Differences from the dashboard call:

  * everything is mapped onto the anchored ECI scale, not raw theta
  * intervals are 80%, tight enough to read at post scale
  * labels are English and the type is scaled up for a shared figure

Usage:
    python blogpost/figures/make_timeline_plotly.py [--results DIR] [--tag _draft]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import multiaxis_eci.config as config  # noqa: E402
import multiaxis_eci.viz.core as vc  # noqa: E402
from multiaxis_eci.analysis.stats import eci_affine  # noqa: E402
from multiaxis_eci.viz.core import save_print  # noqa: E402

HERE = Path(__file__).resolve().parent
DATES_CSV = HERE / "benchmark_release_dates.csv"
HDI_PROB = 0.80

# The figure starts at the modern era: models and benchmark diamonds released
# before this date are out of scope (and so are their callouts).
X_MIN = "2022-01-01"

# Post-scale sizing: the figure is shared flat, so type and markers must read
# without zooming. Markers and error bars run ~2x the dashboard's, keeping
# pace with the type.
FONT_TITLE = 42       # figure title
FONT_AXIS = 36        # axis titles
FONT_TICK = 26        # tick labels
FONT_CALLOUT = 28     # named-point leader labels
FONT_TIER = 26        # right-margin human tier names
MARKER_BENCH = 16     # benchmark points
MARKER_MODEL = 13     # model points
ERRBAR_BENCH = 2.6    # benchmark error-bar line width
ERRBAR_MODEL = 2.0    # model error-bar line width

# Points to name on the canvas. The dashboard draws these as plain labels under
# the marker, so only well-separated points are worth naming.
# Label anchors in DATA coordinates: the point keeps its marker and a thin
# leader runs to the text, which sits in empty canvas. Plotly draws labels at
# the marker by default, which is unreadable inside the 2024-2026 cloud.
# Low / left points, each label hand-placed in nearby empty canvas.
LABEL_LOW = {
    "GPQA Diamond": ("2023-05-01", 140),
    # Own anchor rather than the shared column: its column slot (y ~157 at
    # 2024-03) sat right on the 2024 benchmark points it names.
    "FrontierMath": ("2023-10-01", 165),
}
# High points (hard benchmarks and top models) share ONE column, ordered by the
# value they point at, so no two leader lines cross. The column sits in the
# still-sparse 2023 band: close to the 2024-2026 cloud the leaders point into,
# without touching it.
LABEL_HIGH_BENCH = ["Remote Labor Index", "Humanity's Last Exam"]
HIGH_COL_X = "2024-03-01"
# Two slots only (RLI, HLE): the model labels sit RIGHT of their own points
# instead, where the top-right corner above the tier names is empty. The
# second slot stays near HLE's own height so its leader stays short.
HIGH_COL_Y = [216, 172]
N_TOP_MODELS = 3
_EFFORTS = {"unknown", "max", "xhigh", "high", "medium", "low", "minimal",
            "promax", "proxhigh", "prohigh", "promedium", "prolow"}


def _base_name(name: str) -> str:
    stem, _, suffix = name.rpartition("_")
    return stem if stem and suffix in _EFFORTS else name


def _interval(x: np.ndarray):
    q = [(1 - HDI_PROB) / 2, (1 + HDI_PROB) / 2]
    lo, hi = np.quantile(x, q, axis=0)
    return np.median(x, axis=0), lo, hi


def main(results: Path, tag: str, out_dir: Path = HERE) -> None:
    post = xr.open_dataset(results / "trace.nc", group="posterior")
    models = [str(m) for m in post["model"].values]
    benches = [str(b) for b in post["bench"].values]

    C = post["theta"].values[..., 0].reshape(-1, len(models))
    A = post["A"].values[..., 0].reshape(-1, len(benches))
    D = post["D"].values.reshape(-1, len(benches))

    # Per-draw anchor transform, so the ECI scale is the reported one.
    lo_i = models.index(config.ANCHOR_LOW[0])
    hi_i = models.index(config.ANCHOR_HIGH[0])
    a, b = eci_affine(C, lo_i, hi_i)

    m_med, m_lo, m_hi = _interval(a[:, None] + b[:, None] * C)
    d_med, d_lo, d_hi = _interval(a[:, None] + b[:, None] * (D / A))

    hg = pd.read_csv(results / "human_groups.csv")
    human_names = set(hg["name"])

    raw = pd.read_csv(REPO / "1_data/processed/benchmarks_merged.csv").dropna(
        subset=["release_date"])
    dates = raw.groupby("model_version")["release_date"].min()
    dates = pd.concat([dates, pd.Series({m: d for m, d in config.RELEASE_DATES.items()
                                         if m not in dates.index})])

    ai = pd.DataFrame({"name": models, "mean": m_med,
                       "hdi_low": m_lo, "hdi_high": m_hi})
    ai["release_date"] = pd.to_datetime(ai["name"].map(dates))
    humans = ai[ai["name"].isin(human_names)].merge(hg[["name", "n_obs"]], on="name")
    ai = ai[~ai["name"].isin(human_names)].dropna(subset=["release_date"])
    ai["kind"] = "model"

    bd = pd.read_csv(DATES_CSV)
    dif = pd.DataFrame({"name": benches, "mean": d_med,
                        "hdi_low": d_lo, "hdi_high": d_hi})
    dif = dif.merge(bd[["benchmark", "release_date"]], left_on="name",
                    right_on="benchmark", how="left")
    dif["release_date"] = pd.to_datetime(dif["release_date"], errors="coerce")
    undated = int(dif["release_date"].isna().sum())
    dif = dif.dropna(subset=["release_date"])
    dif["kind"] = "benchmark"

    # Era cut: pre-2022 models and diamonds stay off the canvas entirely.
    cut = pd.Timestamp(X_MIN)
    ai = ai[ai["release_date"] >= cut]
    dif = dif[dif["release_date"] >= cut]

    tl = pd.concat([ai, dif], ignore_index=True)[
        ["name", "kind", "release_date", "mean", "hdi_low", "hdi_high"]]

    # The dashboard function labels human tiers through a French lookup by
    # default; pass {} so the tiers keep their English names.
    fig = vc.capability_timeline_fig(tl, human_stats=humans, human_labels={})

    # No legend: the two series are named in the caption, and each human tier
    # gets its name at the right end of its own line (the matplotlib draft's
    # arrangement). Crowded tiers are nudged apart vertically, which is why the
    # label can sit a little off its line.
    fig.update_layout(showlegend=False)
    import plotly.colors as pc
    rows = humans.sort_values("mean", ascending=False).reset_index(drop=True)
    fracs = np.linspace(0.92, 0.35, len(rows)) if len(rows) > 1 else [0.7]
    tier_colors = [pc.sample_colorscale("Blues", float(f))[0] for f in fracs]
    x_right = ai["release_date"].max()
    x_left = ai["release_date"].min()      # first dated model
    gap = 0.038 * (220 - 30)
    prev, ys = np.inf, []
    for lvl in rows["mean"]:                       # top-down: keep them apart
        y = min(float(lvl), prev - gap)
        ys.append(y)
        prev = y
    for (_, r), y, col in zip(rows.iterrows(), ys, tier_colors):
        fig.add_annotation(
            x=1.005, y=y, xref="paper", yref="y",
            text=r["name"], showarrow=False, xanchor="left",
            font=dict(size=FONT_TIER, color=col))

    # English text and post-scale type: the figure is shared flat, so it has to
    # be readable without zooming.
    fig.update_layout(
        title=dict(text="AI capability and benchmark difficulty on the ECI scale",
                   x=0.5, font=dict(size=FONT_TITLE)),
        xaxis=dict(title=dict(text="Release date", font=dict(size=FONT_AXIS)),
                   tickfont=dict(size=FONT_TICK),
                   dtick="M12", tickformat="%Y", tickangle=0,
                   range=[(x_left - pd.Timedelta(days=90)).strftime("%Y-%m-%d"),
                          (x_right + pd.Timedelta(days=75)).strftime("%Y-%m-%d")]),
        yaxis=dict(title=dict(text="ECI", font=dict(size=FONT_AXIS)),
                   tickfont=dict(size=FONT_TICK), range=[30, 220]),
        legend=dict(font=dict(size=15)),
        height=1250, width=1900, margin=dict(l=130, r=430, t=120, b=110),
    )
    for tr in fig.data:
        if tr.name == "Difficulté des benchmarks":
            tr.name = f"Benchmark difficulty ({HDI_PROB:.0%} interval)"
            tr.marker.size = MARKER_BENCH
            tr.error_y.thickness = ERRBAR_BENCH
            tr.error_y.width = 4
        elif tr.name == "Capacité des modèles IA":
            tr.name = f"AI models ({HDI_PROB:.0%} interval)"
            tr.marker.size = MARKER_MODEL
            tr.error_y.thickness = ERRBAR_MODEL
    for tr in fig.data:
        if getattr(tr, "legendgrouptitle", None) and tr.legendgrouptitle.text:
            tr.legendgrouptitle.text = "Human tiers"
    # Named points, each with a leader to a label in empty canvas.
    def _label(row, anchor, color):
        ax_, ay_ = anchor
        fig.add_annotation(
            x=row["release_date"].strftime("%Y-%m-%d"), y=float(row["mean"]),
            ax=ax_, ay=ay_, axref="x", ayref="y", xref="x", yref="y",
            text=row["name"], showarrow=True, arrowhead=0, arrowwidth=1.8,
            arrowcolor=color, opacity=0.95,
            font=dict(size=FONT_CALLOUT, color=color), xanchor="left")

    for name, anchor in LABEL_LOW.items():
        hit = dif[dif["name"] == name]
        if len(hit):
            _label(hit.iloc[0], anchor, "#d63384")

    high = [(dif[dif["name"] == n].iloc[0], "#d63384")
            for n in LABEL_HIGH_BENCH if len(dif[dif["name"] == n])]
    # Sorted by target height, then paired with the column top-down: leaders
    # then fan out without crossing.
    high.sort(key=lambda t: -float(t[0]["mean"]))
    for (r, color), y in zip(high, HIGH_COL_Y):
        _label(r, (HIGH_COL_X, y), color)

    # Top models: label just RIGHT of each point, running into the empty
    # top-right corner (the tier names start lower, at ~161). Slots are the
    # points' own heights, nudged apart top-down so the texts never touch.
    model_rows, seen = [], set()
    for _, r in ai.sort_values("mean", ascending=False).iterrows():
        base = _base_name(r["name"])
        if base in seen:
            continue
        seen.add(base)
        r = r.copy()
        r["name"] = base
        model_rows.append(r)
        if len(seen) == N_TOP_MODELS:
            break
    prev = np.inf
    for r in model_rows:                        # already strongest-first
        y = min(float(r["mean"]), prev - 6.0)   # ~one text height apart
        prev = y
        anchor_x = (r["release_date"] + pd.Timedelta(days=45)).strftime("%Y-%m-%d")
        _label(r, (anchor_x, y), "#20a39e")
    print(f"  labelled high: {[r['name'] for r, _ in high]} "
          f"+ models right of their points: {[r['name'] for r in model_rows]}")
    if fig.layout.legend.grouptitlefont is not None:
        fig.layout.legend.grouptitlefont.size = 16

    out = out_dir / f"eci_1d_timeline_plotly{tag}"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out.with_suffix(".html"))
    try:
        save_print(fig, out)
        print(f"  wrote {out.with_suffix('.png')}")
    except Exception as e:
        print(f"  PNG export failed ({type(e).__name__}: {e}); HTML written")
    print(f"  {len(ai)} models, {len(humans)} tiers, {len(dif)} benchmarks "
          f"({undated} undated, skipped)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=REPO / "results/canonical")
    p.add_argument("--tag", default="_draft")
    args = p.parse_args()
    main(args.results, args.tag)
