"""The frontier gap across benchmark access scopes: reads what 1_frontier_gap.py wrote for every
scope it finds (all / public / semi_private / private) and draws them side by side, no trace
loaded.

  frontier_gap_<group>_panels      one frontier panel per scope, same ECI-H range
  frontier_gap_<group>_lag         months behind the leader's frontier per follower record and scope
                                   (the figure of Ihle 2026, on ECI-H)
  frontier_gap_<group>_crossovers  human-tier crossing dates of each group's line, per scope (CSV only)
  frontier_gap_<group>_table       the summary table (PNG, CSV and Markdown), with the change of
                                   every quantity against the all-benchmarks scope

  python 4_diagnostics/2_plot_frontier_gap.py [--group openness|country] [--today YYYY-MM-DD]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2_model"))

from multiaxis_eci import config  # noqa: E402
from multiaxis_eci.analysis.frontier_gap import (  # noqa: E402
    GROUP_PAIRS,
    GROUP_TITLES,
    SCOPES,
    load_scope,
    summarize,
)
from multiaxis_eci.viz.core import save_fig, save_svg  # noqa: E402
from multiaxis_eci.viz.frontier_gap import (  # noqa: E402
    SCOPE_TITLES,
    frontier_panels_fig,
    lag_fig,
    summary_table_fig,
)
from multiaxis_eci.viz.i18n import translate_fig  # noqa: E402

# (quantity, group role, label, unit, signed)
TABLE_ROWS = [
    ("slope", "leader", "{leader} frontier slope", "ECI-H / yr", True),
    ("slope", "follower", "{follower} frontier slope", "ECI-H / yr", True),
    ("slope_diff", None, "Slope difference ({leader} - {follower})", "ECI-H / yr", True),
    ("gap_today", None, "Gap today ({leader} - {follower} trend lines)", "ECI-H", True),
    ("lag_today", None, "Lag today (gap / {follower} slope)", "months", False),
    ("envelope_lag_latest_record", None, "Lag of the latest {follower} record", "months", False),
    ("envelope_lag_last_12m_mean", None, "Mean lag of the last 12 months' {follower} records",
     "months", False),
]
# Per-draw arrays behind each table row, for the differences against the all-benchmarks scope.
DRAWS = {"slope_leader": lambda r: r.leader.fit.b if r.leader.fit else None,
         "slope_follower": lambda r: r.follower.fit.b if r.follower.fit else None,
         "slope_diff": lambda r: r.line.get("slope_diff"),
         "gap_today": lambda r: r.line.get("gap_eci"),
         "lag_today": lambda r: r.line.get("lag_months"),
         "envelope_lag_latest_record": lambda r: r.lag_draws[:, -1] if r.lag_draws.shape[1] else None,
         "envelope_lag_last_12m_mean": lambda r: _last_12m(r)}


def _last_12m(r):
    if not r.lag_draws.shape[1]:
        return None
    recent = ((r.lag_df["release_date"] >= r.lag_df["release_date"].max() - pd.Timedelta(days=365))
              & ~r.lag_df["censored"].astype(bool))
    return np.nanmean(r.lag_draws[:, recent.to_numpy()], axis=1) if recent.any() else None


def fmt(s: dict, signed: bool, digits: int = 1) -> str:
    if not np.isfinite(s["median"]):
        return "n/a"
    f = f"{{:{'+' if signed else ''}.{digits}f}}"
    return f"{f.format(s['median'])} [{f.format(s['hdi80_low'])}, {f.format(s['hdi80_high'])}]"


def benchmark_classes_md() -> str:
    """The benchmarks of the canonical scope by access class, as the fits saw them."""
    from multiaxis_eci.data import BENCHMARKS_FILE, load_excluded_benchmarks, read_scores
    table = pd.read_csv(BENCHMARKS_FILE)
    in_scope = set(read_scores()["benchmark"]) - load_excluded_benchmarks()
    lines = ["# Benchmarks by access class (canonical scope, curated exclusions applied)", ""]
    for cls in ("public", "semi_private", "private", "open_problems"):
        names = sorted(n for n in table.loc[table["access"] == cls, "name"] if n in in_scope)
        lines += [f"## {cls} ({len(names)})", ""] + [f"- {n}" for n in names] + [""]
    return "\n".join(lines)


def markdown_table(df: pd.DataFrame) -> str:
    """A GitHub-flavoured Markdown table of a frame of strings (no tabulate dependency)."""
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.tolist()) + " |")
    return "\n".join(lines) + "\n"


def build_table(results: dict, scopes: list[str], kind: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cells (strings, median [80% interval]) and the long numeric frame behind them."""
    leader, follower = GROUP_PAIRS[kind]
    titles = GROUP_TITLES[kind]
    names = {"leader": titles[leader], "follower": titles[follower]}
    cells, long = [], []
    for q, role, label, unit, signed in TABLE_ROWS:
        key = f"{q}_{role}" if role else q
        row = {"quantity": label.format(**names), "unit": unit}
        base = DRAWS[key](results["all"]) if "all" in results else None
        for s in scopes:
            x = DRAWS[key](results[s])
            summ = summarize(x) if x is not None else {"median": np.nan}
            row[SCOPE_TITLES[s]] = fmt(summ, signed)
            long.append({"scope": s, "quantity": key, "unit": unit, **summ})
            if s != "all" and base is not None and x is not None:
                n = min(len(x), len(base))
                d = summarize(x[:n] - base[:n])
                row[f"Δ {SCOPE_TITLES[s].split()[0].lower()} - all"] = fmt(d, True)
                long.append({"scope": s, "quantity": f"{key}_minus_all", "unit": unit, **d})
        cells.append(row)
    return pd.DataFrame(cells), pd.DataFrame(long)


def save_en_fr(fig, name: str, figures_dir: Path) -> None:
    """The English render, then its French twin under `figures/fr/`.

    Both get the PNG and the interactive twin (`html/` beside each); the French one also gets
    the vector SVG the lab's site embeds, in `fr/svg/`, the blog post's convention. The French
    figure is `viz.i18n`'s string walk over the finished English one, so the two cannot drift:
    the record names the months-behind figure places are not translated, and its layout, worked
    out in pixels, carries over untouched.
    """
    save_fig(fig, name, figures_dir)
    fr_dir = figures_dir / "fr"
    fr = translate_fig(fig)
    save_fig(fr, f"{name}_fr", fr_dir)
    save_svg(fr, fr_dir / f"{name}_fr.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", default="openness", choices=list(GROUP_TITLES))
    ap.add_argument("--today", default=None, help="pin the today line (default: the wall clock)")
    ap.add_argument("--y-range", default=None, metavar="LO,HI",
                    help="fix the panels' ECI-H range (default: read off the candidates)")
    args = ap.parse_args()
    kind = args.group
    titles = GROUP_TITLES[kind]
    leader, follower = GROUP_PAIRS[kind]

    cmp_dir = config.COMPARISONS_DIR
    results = {}
    for s in SCOPES:
        try:
            results[s] = load_scope(cmp_dir, kind, s)
        except FileNotFoundError:
            print(f"  {s}: no 1_frontier_gap.py output yet (fit canonical_{s}/ then run it); skipped")
    if not results:
        raise SystemExit("nothing to draw")
    scopes = list(results)
    figures_dir = cmp_dir / config.FIGURES_DIRNAME
    stem = f"frontier_gap_{kind}"
    y_range = tuple(float(v) for v in args.y_range.split(",")) if args.y_range else None

    fig = frontier_panels_fig(results, scopes, today=args.today, group_titles=titles, y_range=y_range,
                              title=f"{titles[leader]} vs {titles[follower]} (ECI-H), by benchmark access")
    save_en_fr(fig, f"{stem}_panels", figures_dir)

    fig = lag_fig(results, scopes, today=args.today, label_scope="all",
                  follower_title=titles[follower].lower(), leader_title=titles[leader].lower(),
                  title=f"How far behind the {titles[leader].lower()} frontier are {titles[follower].lower()}?")
    save_en_fr(fig, f"{stem}_lag", figures_dir)

    cx_all = []
    for s in scopes:
        cx = results[s].crossovers.copy()
        cx["scope"] = s
        cx["axis"] = cx["axis"].map(lambda g, s=s: f"{titles[g]} · {SCOPE_TITLES[s].lower()}")
        cx_all.append(cx)
    cx_all = pd.concat(cx_all, ignore_index=True)
    cx_all.to_csv(cmp_dir / f"{stem}_crossovers.csv", index=False)

    cells, long = build_table(results, scopes, kind)
    long.to_csv(cmp_dir / f"{stem}_table.csv", index=False)
    (cmp_dir / f"{stem}_table.md").write_text(markdown_table(cells))
    fig = summary_table_fig(cells, f"{titles[leader]} vs {titles[follower]}: median [80% interval]",
                            width=1700)
    # A table is not a plot: it has no HTML twin worth keeping, so it skips save_en_fr.
    fig.write_image(str(figures_dir / f"{stem}_table.png"), scale=2)
    fr = translate_fig(fig)
    (figures_dir / "fr").mkdir(parents=True, exist_ok=True)
    fr.write_image(str(figures_dir / "fr" / f"{stem}_table_fr.png"), scale=2)
    save_svg(fr, figures_dir / "fr" / f"{stem}_table_fr.png")
    print(f"wrote {figures_dir / stem}_{{panels,lag,table}}.png (+ fr/, fr/svg/), "
          f"{cmp_dir / stem}_{{table.csv,table.md,crossovers.csv}}")
    (cmp_dir / f"{stem}_benchmark_classes.md").write_text(benchmark_classes_md())
    pd.set_option("display.width", 250)
    print(cells.to_string(index=False))


if __name__ == "__main__":
    main()
