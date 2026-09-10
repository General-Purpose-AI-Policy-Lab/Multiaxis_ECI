"""1D ECI timeline with benchmark difficulties, in the dashboard's Plotly style.

The post's figure 1, drawn
by the library's own `viz.core.capability_timeline_fig`, the builder behind the
canonical fit's `capability_timeline` figure and the dashboard's timelines, so
the post cannot drift from them. What this script adds on top of the builder's
options (ECI scale, 80% intervals, human bands, tier names at the right): the
curated benchmark release dates, the post-scale type, and the hand-placed
callouts.

Usage:
    python 6_writeups/blogpost/figures/make_timeline_plotly.py [--results DIR] [--tag _draft]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "2_model"))
import multiaxis_eci.config as config  # noqa: E402
import multiaxis_eci.viz.core as vc  # noqa: E402
from multiaxis_eci.analysis.stats import eci_affine  # noqa: E402

# The canonical K=1 fit of the current data generation; the blog post's figure 1.
DEFAULT_RESULTS = config.RESULTS_DIR / "canonical"
from multiaxis_eci.viz import POST  # noqa: E402
from multiaxis_eci.viz.core import save_html, save_print  # noqa: E402

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
# Human tiers: every tier's HDI_PROB interval as a faint band in the tier's own
# color, under every point (layer below), so a prior-driven tier reads as
# uncertain. The alpha is low because the one-observation tiers span 40+ ECI
# points and their bands overlap the well-measured ones: at this value two
# stacked bands still stay lighter than a single error bar.
HUMAN_BAND_ALPHA = 0.06

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
# Display override: wrapped onto two lines and lifted, so the text block ends
# before the 2025 benchmark whiskers instead of running into them.
LABEL_WRAP = {"Humanity's Last Exam": "Humanity's<br>Last Exam"}
HIGH_COL_X = "2024-03-01"
# Two slots only (RLI, HLE): the model labels sit RIGHT of their own points
# instead, where the top-right corner above the tier names is empty.
HIGH_COL_Y = [216, 179]
N_TOP_MODELS = 3

def _base_name(name: str) -> str:
    """The release a test-taker belongs to, by the pipeline's identity (data.model_family)."""
    from multiaxis_eci.data import model_family
    return model_family(name)


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

    raw = pd.read_csv(REPO / "0_input/all_scores_flat.csv").dropna(
        subset=["release_date"])
    dates = raw.groupby("model_version")["release_date"].min()

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

    # Every tier banded and named at the right end of its line (no legend: the
    # two series are named in the caption); the builder nudges crowded tiers apart.
    fig = vc.capability_timeline_fig(tl, human_stats=humans, lang="en",
                                     hdi_prob=HDI_PROB, y_label="ECI-H",
                                     human_bands=True,
                                     human_band_alpha=HUMAN_BAND_ALPHA,
                                     tier_names_at_right=True, style=POST)
    fig.update_layout(showlegend=False)
    x_right = ai["release_date"].max()
    x_left = ai["release_date"].min()      # first dated model

    # Post-scale type: the figure is shared flat, so it has to be readable
    # without zooming.
    fig.update_layout(
        title=dict(text="AI capability, human baselines and benchmark difficulty on the ECI-H scale",
                   x=0.5, font=dict(size=FONT_TITLE)),
        xaxis=dict(title=dict(text="Release date", font=dict(size=FONT_AXIS)),
                   tickfont=dict(size=FONT_TICK),
                   dtick="M12", tickformat="%Y", tickangle=0,
                   range=[(x_left - pd.Timedelta(days=90)).strftime("%Y-%m-%d"),
                          (x_right + pd.Timedelta(days=75)).strftime("%Y-%m-%d")]),
        yaxis=dict(title=dict(text="ECI-H", font=dict(size=FONT_AXIS)),
                   tickfont=dict(size=FONT_TICK), range=[30, 220]),
        legend=dict(font=dict(size=15)),
        height=1250, width=1900, margin=dict(l=130, r=430, t=120, b=110),
    )
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
        if r["name"] in LABEL_WRAP:
            r = r.copy()
            r["name"] = LABEL_WRAP[r["name"]]
        _label(r, (HIGH_COL_X, y), color)

    # Top models: one column in the TOP-RIGHT corner, past the last point and
    # above the tier names (which start at ~161), text running into the right
    # margin. Strongest model takes the top slot, so the three leaders fan
    # down-left to their points without crossing.
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
    model_col_x = (x_right + pd.Timedelta(days=20)).strftime("%Y-%m-%d")
    MODEL_COL_Y = [209, 198, 187]
    for r, y in zip(model_rows, MODEL_COL_Y):   # already strongest-first
        _label(r, (model_col_x, y), "#20a39e")
    print(f"  labelled high: {[r['name'] for r, _ in high]} "
          f"+ models in the top-right column: {[r['name'] for r in model_rows]}")
    if fig.layout.legend.grouptitlefont is not None:
        fig.layout.legend.grouptitlefont.size = 16

    out = out_dir / f"eci_1d_timeline_plotly{tag}"
    out.parent.mkdir(parents=True, exist_ok=True)
    save_html(fig, out)
    try:
        save_print(fig, out)
        print(f"  wrote {out.with_suffix('.png')}")
    except Exception as e:
        print(f"  PNG export failed ({type(e).__name__}: {e}); HTML written")
    print(f"  {len(ai)} models, {len(humans)} tiers, {len(dif)} benchmarks "
          f"({undated} undated, skipped)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--tag", default="_draft")
    args = p.parse_args()
    main(args.results, args.tag)
