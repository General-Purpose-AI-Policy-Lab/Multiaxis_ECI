"""1D ECI timeline with benchmark difficulties, for the blog post.

Three series on one pair of axes:
  * every dated AI test-taker, x = release date, y = anchored ECI (80% interval)
  * every dated benchmark, x = benchmark release date, y = D_b / A_b mapped
    through the SAME per-draw anchor transform (80% interval)
  * the human tiers as horizontal reference lines, labelled on the right

D_b / A_b is the ability at which the expected score is 0.5, because
mu = sigmoid(A_b * theta - D_b) in the canonical K=1 fit (no floors). The ratio
is formed per posterior draw, so the interval is real posterior uncertainty and
not a propagated point estimate.

Usage:
    python blogpost/figures/make_timeline_difficulties.py [--results DIR] [--tag _draft]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import multiaxis_eci.config as config  # noqa: E402  (anchors + dateless-model backfill)
from multiaxis_eci.analysis.stats import eci_affine  # noqa: E402
from multiaxis_eci.viz.core import AI_COLOR, HUMAN_COLOR  # noqa: E402

HERE = Path(__file__).resolve().parent
DATES_CSV = HERE / "benchmark_release_dates.csv"

# AI blue and human orange come from the shared colorblind-safe palette.
# Difficulties take a warm magenta, which no other figure in the post uses.
_DIFF_COLOR = "#d6427e"

HDI_PROB = 0.80

# Difficulty points to name, with hand-placed label anchors: the 2024-2026
# region is too dense for any offset rule, so the label sits in nearby empty
# canvas and a thin leader points at the marker.
_LABEL_POS = {
    "TriviaQA":               ("2017-07-01", 52),
    "MMLU":                   ("2020-10-01", 85),
    "GSM8K":                  ("2021-11-01", 94),
    "GPQA Diamond":           ("2021-04-01", 140),
    "FrontierMath":           ("2021-04-01", 164),
    "Humanity's Last Exam":   ("2021-04-01", 174),
    "Remote Labor Index":     ("2023-05-01", 205),
}
_YLIM = (30, 218)

# Effort suffixes stripped when labelling top models; variants of one model
# collapse to a single label.
_EFFORTS = {"unknown", "max", "xhigh", "high", "medium", "low", "minimal",
            "promax", "proxhigh", "prohigh", "promedium", "prolow"}
_N_TOP_MODELS = 4


def _base_name(name: str) -> str:
    stem, _, suffix = name.rpartition("_")
    return stem if stem and suffix in _EFFORTS else name


def _interval(x: np.ndarray, axis: int = 0):
    """Median and equal-tailed HDI_PROB interval, matching analysis.stats.post_stats."""
    q = [(1 - HDI_PROB) / 2, (1 + HDI_PROB) / 2]
    lo, hi = np.quantile(x, q, axis=axis)
    return np.median(x, axis=axis), lo, hi


def main(results: Path, tag: str) -> None:
    post = xr.open_dataset(results / "trace.nc", group="posterior")
    models = [str(m) for m in post["model"].values]
    benches = [str(b) for b in post["bench"].values]

    # (draws, model) capability and (draws, bench) discrimination / difficulty.
    C = post["theta"].values[..., 0].reshape(-1, len(models))
    A = post["A"].values[..., 0].reshape(-1, len(benches))
    D = post["D"].values.reshape(-1, len(benches))

    # Per-draw anchor transform: ECI = a + b * C, pinned at the two anchors.
    lo_i, hi_i = models.index(config.ANCHOR_LOW[0]), models.index(config.ANCHOR_HIGH[0])
    a, b = eci_affine(C, lo_i, hi_i)

    eci = a[:, None] + b[:, None] * C
    m_med, m_lo, m_hi = _interval(eci)

    # Same transform on the half-score ability of each benchmark.
    diff_eci = a[:, None] + b[:, None] * (D / A)
    d_med, d_lo, d_hi = _interval(diff_eci)

    # Humans are the test-takers with no release date in the score table; take
    # the tier list from the fit's own human table so the split is not guessed.
    human_names = set(pd.read_csv(results / "human_groups.csv")["name"])

    raw = pd.read_csv(REPO / "1_data/processed/benchmarks_merged.csv").dropna(
        subset=["release_date"])
    model_dates = raw.groupby("model_version")["release_date"].min()
    curated = {m: d for m, d in config.RELEASE_DATES.items()
               if m not in model_dates.index}
    model_dates = pd.concat([model_dates, pd.Series(curated)])

    ai = pd.DataFrame({"name": models, "med": m_med, "lo": m_lo, "hi": m_hi})
    ai["date"] = pd.to_datetime(ai["name"].map(model_dates))
    humans = ai[ai["name"].isin(human_names)].sort_values("med")
    ai = ai[~ai["name"].isin(human_names)].dropna(subset=["date"])

    bd = pd.read_csv(DATES_CSV)
    bd = bd[bd["benchmark"].isin(benches)]
    dif = pd.DataFrame({"name": benches, "med": d_med, "lo": d_lo, "hi": d_hi})
    dif = dif.merge(bd[["benchmark", "release_date"]], left_on="name",
                    right_on="benchmark", how="left")
    dif["date"] = pd.to_datetime(dif["release_date"], errors="coerce")
    undated = dif["date"].isna().sum()
    dif = dif.dropna(subset=["date"])

    # ---- figure ---------------------------------------------------------
    plt.rcParams.update({"font.size": 13, "axes.labelsize": 14,
                         "xtick.labelsize": 12, "ytick.labelsize": 12})
    fig, ax = plt.subplots(figsize=(15, 10.5))

    ax.errorbar(ai["date"], ai["med"],
                yerr=[ai["med"] - ai["lo"], ai["hi"] - ai["med"]],
                fmt="o", ms=5.5, color=AI_COLOR, ecolor=AI_COLOR,
                elinewidth=1.0, alpha=0.65, capsize=0,
                label=f"AI models ({HDI_PROB:.0%} interval)")

    ax.errorbar(dif["date"], dif["med"],
                yerr=[dif["med"] - dif["lo"], dif["hi"] - dif["med"]],
                fmt="D", ms=7, color=_DIFF_COLOR, ecolor=_DIFF_COLOR,
                elinewidth=1.4, alpha=0.9, capsize=2, zorder=4,
                label=f"benchmark difficulty D/A ({HDI_PROB:.0%} interval)")

    y_lo, y_hi = _YLIM
    ax.set_ylim(*_YLIM)

    # The scale is anchored by the per-draw a, b transform above; the anchor
    # lines are deliberately NOT drawn — the caption states them.
    xmax = ai["date"].max()

    # Human tiers: dashed line at the level, labels nudged apart on the right
    # margin so crowded tiers stay legible without moving the lines.
    span = y_hi - y_lo
    prev, label_y = -np.inf, []
    for lvl in humans["med"]:
        y = max(lvl, prev + 0.035 * span)
        label_y.append(y)
        prev = y
    for (_, r), ly in zip(humans.iterrows(), label_y):
        ax.axhline(r["med"], color=HUMAN_COLOR, ls="--", lw=1.0, alpha=0.85,
                   zorder=1)
        ax.text(xmax + pd.Timedelta(days=40), ly, r["name"],
                color=HUMAN_COLOR, fontsize=12, va="center")
    ax.plot([], [], color=HUMAN_COLOR, ls="--", lw=1.0,
            label="human tiers (posterior median)")

    for _, r in dif[dif["name"].isin(_LABEL_POS)].iterrows():
        lx, ly = _LABEL_POS[r["name"]]
        ax.annotate(r["name"], xy=(r["date"], r["med"]),
                    xytext=(pd.Timestamp(lx), ly), fontsize=11.5,
                    color=_DIFF_COLOR, ha="left", va="center", zorder=6,
                    arrowprops=dict(arrowstyle="-", color=_DIFF_COLOR,
                                    lw=0.9, alpha=0.6,
                                    shrinkA=2, shrinkB=4))

    # Top models, one label per model (effort variants collapsed), stacked in
    # the empty upper-left canvas with leaders to the markers.
    picks, seen = [], set()
    for _, r in ai.sort_values("med", ascending=False).iterrows():
        base = _base_name(r["name"])
        if base in seen:
            continue
        seen.add(base)
        picks.append((base, r))
        if len(picks) == _N_TOP_MODELS:
            break
    label_top = 210.0
    for base, r in picks:
        ax.annotate(base, xy=(r["date"], r["med"]),
                    xytext=(pd.Timestamp("2021-01-01"), label_top),
                    fontsize=11.5, color=AI_COLOR, ha="left", va="center",
                    zorder=6,
                    arrowprops=dict(arrowstyle="-", color=AI_COLOR,
                                    lw=0.9, alpha=0.6, shrinkA=2, shrinkB=3))
        label_top -= 9

    ax.set_ylabel("ECI")
    ax.set_xlabel("Release date")
    ax.set_title("ECI timeline with benchmark difficulties (1D)", fontsize=15)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="lower right", frameon=False, fontsize=12)
    ax.margins(x=0.02)
    fig.tight_layout()

    out = HERE / f"eci_1d_timeline_difficulties{tag}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    off = dif[(dif["med"] < y_lo) | (dif["med"] > y_hi)]
    print(f"wrote {out}")
    print(f"  {len(ai)} dated AI models, {len(humans)} human tiers, "
          f"{len(dif)} dated benchmarks ({undated} skipped for no date)")
    print(f"  labelled top models: {[b for b, _ in picks]}")
    print(f"  ECI y-range {y_lo} to {y_hi}; "
          f"{len(off)} difficulty medians off-scale: {list(off['name'])}")
    for n in ("MMLU", "GSM8K", "GPQA Diamond", "FrontierMath",
              "Humanity's Last Exam"):
        r = dif[dif["name"] == n]
        if len(r):
            r = r.iloc[0]
            print(f"  check {n:24s} {r['med']:7.1f}  "
                  f"[{r['lo']:.1f}, {r['hi']:.1f}]")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, default=REPO / "results/canonical")
    p.add_argument("--tag", default="_draft", help="filename suffix")
    args = p.parse_args()
    main(args.results, args.tag)
