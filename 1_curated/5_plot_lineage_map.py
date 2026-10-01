"""One large figure of 1_curated/lineage_map.csv, for checking the map at a glance after a sync.

One block per organization, one row per chain. The nodes of a chain sit in date order on its
row: the spine (nodes without an explicit parent, each continuing the previous one) is the solid
line, a node with an explicit `parent` hangs from it by an arc above the row. Under each node,
its date, then every raw model_version filed under it in small type with its variant and its
number of scores. The last block lists the models kept out of every chain (`in_chain` = no).

Reads the map file itself, not a fitted structure (that is 4_diagnostics/plot_lineage.py), so it
shows what review has to check: which run went to which node, and in what order.

  python 1_curated/5_plot_lineage_map.py      # also run at the end of 2_build_lineage_map.py
"""
from __future__ import annotations

from pathlib import Path as FilePath

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch
from matplotlib.path import Path

ROOT = FilePath(__file__).resolve().parent.parent
MAP = ROOT / "1_curated" / "lineage_map.csv"
OUT = ROOT / "1_curated" / "lineage_map_overview"   # .pdf (vector, zoomable) and .png

COL_W = 1.7          # inches per node column
LEFT = 1.6           # inches for the chain names
LINE = 0.075         # inches per line of small type
SPINE, BRANCH, INK, GREY, BAND = "#2a6fb0", "#d0631b", "#222222", "#6b6b6b", "#eef1f5"


def chain_nodes(rows: pd.DataFrame) -> pd.DataFrame:
    """One row per node of a chain, in date order, with its parent (explicit or the spine's)."""
    nd = (rows.groupby("node", sort=False)
              .agg(date=("node_date", "min"), parent=("parent", "first"))
              .reset_index().sort_values(["date", "node"]).reset_index(drop=True))
    nd["explicit"] = nd["parent"].notna()
    prev = None
    for i, r in nd.iterrows():
        if not r.explicit:
            nd.loc[i, "parent"] = prev
            prev = r.node
    return nd


def variant_lines(rows: pd.DataFrame, node: str) -> list[str]:
    sub = rows[rows["node"] == node].sort_values(["variant", "raw_string"])
    return [f"{r.raw_string}  ·  {r.variant}  ·  {int(r.n_obs)}" for r in sub.itertuples()]


def layout(m: pd.DataFrame):
    """[(vendor, [(chain, nodes, rows, n_lines)])] in a stable order, and the widest chain."""
    blocks, widest = [], 1
    for vendor, vrows in sorted(m.groupby("vendor"), key=lambda kv: kv[0].lower()):
        chains = []
        for chain, crows in sorted(vrows.groupby("chain"), key=lambda kv: kv[1]["node_date"].min()):
            nodes = chain_nodes(crows)
            n_lines = max(len(crows[crows["node"] == n]) for n in nodes["node"])
            chains.append((chain, nodes, crows, n_lines))
            widest = max(widest, len(nodes))
        blocks.append((vendor, chains))
    return blocks, widest


def main() -> None:
    m = pd.read_csv(MAP, dtype={"n_obs": float})
    inside = m[m["in_chain"] == "yes"].copy()
    outside = m[m["in_chain"] != "yes"].sort_values(["variant", "raw_string"])
    blocks, widest = layout(inside)

    # Heights in inches: per block a header, per chain a row (arc room, name, dot, date, variants).
    head_h, arc_h, name_h = 0.32, 0.30, 0.16
    chain_h = [arc_h + name_h + 0.12 + LINE * (n + 1.5) for _, ch in blocks for *_, n in ch]
    out_cols = 6
    out_h = 0.4 + LINE * 1.25 * (len(outside) // out_cols + 1)
    height = 0.6 + head_h * len(blocks) + sum(chain_h) + out_h
    width = LEFT + COL_W * widest + 0.3
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)          # y grows downward, in inches
    ax.axis("off")

    n_nodes = sum(len(nodes) for _, ch in blocks for _, nodes, *_ in ch)
    ax.text(0.2, 0.32, f"Lineage map: {len(inside)} runs in {n_nodes} nodes of "
            f"{sum(len(ch) for _, ch in blocks)} chains, {len(outside)} runs out of every chain "
            f"({MAP.relative_to(ROOT)})", fontsize=11, weight="bold", color=INK, va="center")
    ax.text(width - 0.2, 0.32, "blue line: spine   orange arc: explicit parent   under a node: model_version · "
            "variant · scores", fontsize=7, color=GREY, ha="right", va="center")
    y = 0.6
    for vendor, chains in blocks:
        ax.add_patch(plt.Rectangle((0, y), width, head_h, color=BAND, lw=0))
        ax.text(0.2, y + head_h / 2, vendor, fontsize=10, weight="bold", color=INK, va="center")
        y += head_h
        for chain, nodes, rows, n_lines in chains:
            row_h = arc_h + name_h + 0.12 + LINE * (n_lines + 1.5)
            yd = y + arc_h + name_h            # the dot line
            ax.text(0.2, yd, chain, fontsize=7.5, weight="bold", color=INK, va="center")
            xs = {r.node: LEFT + COL_W * (i + 0.5) for i, r in nodes.iterrows()}
            for r in nodes.itertuples():
                x = xs[r.node]
                if isinstance(r.parent, str) and r.parent in xs:
                    if r.explicit:
                        # A quadratic curve whose control point sits above the row (y grows
                        # downward here), so every branch arc bulges upward, whatever its span.
                        x0, x1 = xs[r.parent], x
                        lift = min(arc_h + name_h * 0.5, 0.12 + 0.05 * abs(x1 - x0) / COL_W)
                        curve = Path([(x0, yd - 0.05), ((x0 + x1) / 2, yd - 0.05 - 2 * lift),
                                      (x1, yd - 0.05)], [Path.MOVETO, Path.CURVE3, Path.CURVE3])
                        ax.add_patch(FancyArrowPatch(path=curve, arrowstyle="-|>",
                                     mutation_scale=6, color=BRANCH, lw=0.8))
                    else:
                        ax.plot([xs[r.parent], x], [yd, yd], color=SPINE, lw=1.2, zorder=1)
                col = BRANCH if r.explicit else SPINE
                ax.plot(x, yd, "o", ms=5, color=col, zorder=2)
                ax.text(x, yd - 0.09, r.node, fontsize=6.5, weight="bold", color=col,
                        ha="center", va="bottom")
                ax.text(x, yd + 0.09, str(r.date), fontsize=5, color=GREY, ha="center", va="top")
                for j, line in enumerate(variant_lines(rows, r.node)):
                    ax.text(x - COL_W / 2 + 0.06, yd + 0.2 + LINE * j, line, fontsize=3.6,
                            color=INK, va="top", family="DejaVu Sans Mono")
            y += row_h
            ax.plot([0.15, width - 0.15], [y - 0.03, y - 0.03], color="#dddddd", lw=0.4)

    ax.add_patch(plt.Rectangle((0, y), width, head_h, color=BAND, lw=0))
    ax.text(0.2, y + head_h / 2, f"Out of every chain ({len(outside)}): model_version · variant",
            fontsize=10, weight="bold", color=INK, va="center")
    y += head_h + 0.08
    per_col = len(outside) // out_cols + 1
    cw = (width - 0.4) / out_cols
    for k, r in enumerate(outside.itertuples()):
        c, j = divmod(k, per_col)
        ax.text(0.2 + c * cw, y + LINE * 1.25 * j, f"{r.raw_string}  ·  {r.variant}",
                fontsize=3.8, color=GREY, va="top", family="DejaVu Sans Mono")

    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"), dpi=110)
    plt.close(fig)
    print(f"wrote {OUT.with_suffix('.pdf').relative_to(ROOT)} and .png "
          f"({width:.0f} x {height:.0f} in)")


if __name__ == "__main__":
    main()
