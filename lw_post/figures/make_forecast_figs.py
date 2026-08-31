"""The two forecast figures for the LessWrong post, in matplotlib.

The flagship K=4 fit, records basis from Oct 2024, SD cap 0.4. Three axes only
(axis 4, Legacy QA, is out of the post's forecast scope). Every interval drawn
on either figure is a 50% HDI.

  forecast_trend_lw.png     frontier trend per axis, 1x3, window 2020-2030,
                            50% band.
  forecast_crossover_lw.png crossover dates per axis, three stacked panels on
                            ONE shared time window, so the panels are
                            comparable. Median dot and a 50% whisker from the
                            flagship's stored crossover table; the 80% bounds
                            in the same table are printed, not drawn. An
                            interval straddling today reads as an undecided
                            crossing. Axis 2's dates carry the
                            backward-extrapolation caveat on the panel.

  ~/miniforge3/envs/pymc_env/bin/python lw_post/figures/make_forecast_figs.py

The forecast itself (`compute`, the cache path) lives in `make_all.py`, which
owns the flagship fit identity; the panel drawers and labels live in `figbase`.
This file is the layout only. The trend-object pickle beside the trace is
reused whenever it exists, so a layout edit never reloads the trace.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import figbase  # noqa: E402
from analysis import FLAGSHIP_TRACE  # noqa: E402
from config import AXIS_TITLES as LW_TITLES  # noqa: E402
from make_all import AXES, END, forecast_cache  # noqa: E402

TREND_START = pd.Timestamp("2020-01-01")

# Right edge each panel may not pass; None = the content sets it. At the 50%
# widths the stored CSV holds, no panel's natural window reaches these caps:
# they are a safety clip, not an active bound.
PANEL_HARD_MAX = {"axis1": pd.Timestamp("2031-07-01"),
                  "axis3": pd.Timestamp("2040-01-01")}
# Panels whose dates are extrapolated backwards out of the fit window.
BACKCAST_NOTE = {
    "axis2": "dates are the post-2024 record trend extrapolated backwards, "
             "not observed crossings",
}


def trend_fig(per_axis: dict) -> Path:
    """`figbase.timeline_forecast_panel` per axis, 1x3, window capped at 2030."""
    today = pd.Timestamp.today().normalize()
    fig, axs = plt.subplots(len(AXES), 1, figsize=(11, 12.5))
    for ax, name in zip(np.atleast_1d(axs), AXES):
        d = per_axis[name]
        fc = d["fc"]
        figbase.timeline_forecast_panel(ax, d["tl"], d["hs"], fc,
                                        LW_TITLES[name], today,
                                        show_ylabel=True)
        ax.set_xlim(TREND_START, END)
        ax.xaxis.set_major_locator(mdates.YearLocator(base=1))
        if name == AXES[-1]:
            ax.set_xlabel("release date (window: 2020–2030)")
        print(f"  {name}: slope median {np.median(fc.slope):+.3f}/yr, "
              f"P(slope>0)={float((fc.slope > 0).mean()):.3f}, "
              f"trend at 2030 = {fc.median[-1]:.2f} "
              f"[{fc.lo[-1]:.2f}, {fc.hi[-1]:.2f}] (50% HDI)")
    handles = [
        plt.Line2D([], [], marker="o", color=figbase.AI_COLOR, ls="none",
                   alpha=0.5, label="AI models (dated)"),
        plt.Line2D([], [], color=figbase.TREND_COLOR, ls="--",
                   label="frontier trend (50% band)"),
        plt.Line2D([], [], color=figbase.HUMAN_COLOR, ls="--", alpha=0.6,
                   label="human tiers"),
        plt.Line2D([], [], color="#444", ls=":", label="today"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Frontier trend per axis", fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    out = HERE / "forecast_trend_lw.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")
    return out


# The memo's scheme: green where the median crossing is behind us, red where it
# is ahead. The color encodes the MEDIAN only — on the Agentic axis the draws
# are far from unanimous, which the text gives as a probability.
FUTURE_COLOR = figbase.FUTURE_COLOR
PAST_COLOR = figbase.PASSED_COLOR


def _window(rows: pd.DataFrame, hard_max: pd.Timestamp | None):
    """The panel's own time window: the span its estimates occupy, plus 5%."""
    lo = pd.to_datetime(rows["crossover_hdi_low"]).min()
    hi = pd.to_datetime(rows["crossover_hdi_high"]).max()
    if hard_max is not None:
        hi = min(hi, hard_max)
    pad = pd.Timedelta(days=max(60, int(0.05 * (hi - lo).days)))
    return lo - pad, hi + pad


def _crossover_row(ax, y, r, start, end, span_days, color) -> str:
    """Draw one tier's median and 50% whisker; return its print line."""
    med = pd.to_datetime(r["crossover_date_median"])
    lo = pd.to_datetime(r["crossover_hdi_low"])
    hi = pd.to_datetime(r["crossover_hdi_high"])
    lo80 = pd.to_datetime(r["hdi80_low"])
    hi80 = pd.to_datetime(r["hdi80_high"])
    pad = pd.Timedelta(days=int(0.012 * span_days))

    if pd.isna(med):
        ax.plot(end - pad, y, marker=">", color="#999", ms=7)
        return "no crossing (slope not positive in most draws)"

    # 50% whiskers only on the canvas: a wider layer runs off both window ends
    # on two of three panels and reads as a full-width rule. The 80% bounds go
    # to this function's print line instead.
    if not (pd.isna(lo) or lo > end or hi < start):
        a, b = max(lo, start), min(hi, end)
        ax.plot([a, b], [y, y], color=color, lw=2.0, alpha=0.85)
        if lo < start:                      # clipped: arrow tip, not a bound
            ax.plot(a, y, marker="<", color=color, ms=6, alpha=0.85)
        if hi > end:
            ax.plot(b, y, marker=">", color=color, ms=6, alpha=0.85)
    if start <= med <= end:
        ax.plot(med, y, marker="o", color=color, ms=6)
    return (f"median {med.date()}, 50% [{lo.date()}, {hi.date()}], "
            f"80% [{lo80.date()}, {hi80.date()}], "
            f"p(passed)={r['p_passed_now']:.2f}")


def _crossover_panel(ax, name: str, rows: pd.DataFrame,
                     today: pd.Timestamp,
                     window: tuple | None = None) -> list[str]:
    start, end = window if window else _window(rows, PANEL_HARD_MAX.get(name))
    span = (end - start).days
    labels, notes = [], []
    for y, (_, r) in enumerate(rows.iterrows()):
        labels.append(r["tier"])
        med = pd.to_datetime(r["crossover_date_median"])
        color = PAST_COLOR if (pd.notna(med) and med <= today) else FUTURE_COLOR
        notes.append(f"{r['tier']}: "
                     f"{_crossover_row(ax, y, r, start, end, span, color)}")

    if start <= today <= end:
        ax.axvline(today, color="#444", ls=":", lw=1.4)
        ax.annotate("today", (today, len(labels) - 0.35), fontsize=11,
                    color="#444", ha="center", va="bottom")

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=12)
    ax.set_ylim(-0.7, len(labels) - 0.3)
    ax.set_xlim(start, end)
    title = LW_TITLES[name]
    if name in BACKCAST_NOTE:
        title += "  (backward extrapolation of the post-2024 record trend)"
        # Under the panel, not inside it: every row's 50% interval spans most
        # of this window, so there is no empty patch in the panel to sit in.
        ax.set_xlabel(BACKCAST_NOTE[name], fontsize=11, color="#444",
                      style="italic", loc="left")
    ax.set_title(title, fontsize=14, loc="left")

    # Tick density follows the panel's own span: month minors only where a
    # month is resolvable at all.
    yrs = span / 365.25
    ax.xaxis.set_major_locator(mdates.YearLocator(
        base=1 if yrs < 9 else 2 if yrs < 20 else 5))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(
        mdates.MonthLocator(bymonth=range(1, 13)) if yrs < 3 else
        mdates.MonthLocator(bymonth=(1, 4, 7, 10)) if yrs < 9 else
        mdates.YearLocator())
    # Labelled months on short panels so a reader can place a median without
    # measuring; January stays blank because the year major already sits there.
    if yrs < 6:
        from matplotlib.ticker import FuncFormatter

        def _month(x, _pos):
            d = mdates.num2date(x)
            return "" if d.month == 1 else d.strftime("%b")
        ax.xaxis.set_minor_formatter(FuncFormatter(_month))
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="x", which="minor", length=3.5, color="#666",
                   labelsize=9, labelcolor="#666")
    ax.grid(axis="x", color="0.85", lw=0.7)
    ax.grid(axis="x", which="minor", color="0.93", lw=0.5)
    return notes


def crossover_fig(cx: pd.DataFrame) -> Path:
    """Reads the crossover intervals straight off disk; no trace load once cached."""
    today = pd.Timestamp.today().normalize()
    # One shared x window over all three axes, so the panels are comparable. It
    # has to span axis 2's backcast and axis 1's late 2020s.
    lo = pd.to_datetime(cx["crossover_hdi_low"]).min()
    hi = pd.to_datetime(cx["crossover_hdi_high"]).max()
    pad = pd.Timedelta(days=int(0.02 * (hi - lo).days))
    window = (lo - pad, hi + pad)
    fig, axs = plt.subplots(len(AXES), 1, figsize=(15, 12.5))
    for ax, name in zip(np.atleast_1d(axs), AXES):
        rows = cx[cx.axis == name].sort_values("human_mean")
        print(f"  {LW_TITLES[name]}")
        for n in _crossover_panel(ax, name, rows, today, window):
            print(f"    {n}")
    handles = [
        plt.Line2D([], [], marker="o", color=PAST_COLOR, ls="-",
                   label="median crossing already behind us"),
        plt.Line2D([], [], marker="o", color=FUTURE_COLOR, ls="-",
                   label="median crossing still ahead (50% interval)"),
        plt.Line2D([], [], color="#444", ls=":", label="today"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=13, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("When the frontier trend reaches each human tier",
                 fontsize=17)
    fig.tight_layout(rect=(0, 0.02, 1, 0.965))
    out = HERE / "forecast_crossover_lw.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")
    return out


def main() -> None:
    # Crossover figure: the flagship's stored 50%/80% table.
    cx = pd.read_csv(FLAGSHIP_TRACE.parent / "lw_crossover_50_80.csv")
    crossover_fig(cx)

    # Trend figure: the per-draw lines at hdi_prob=0.5, from the cache beside
    # the trace.
    trend_fig(forecast_cache())


if __name__ == "__main__":
    main()
