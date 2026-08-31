"""The three per-axis result figures for the blog post, in matplotlib.

The Plotly twins of these live in `make_all.py`; both read the flagship through
`make_all.load_flagship`, so the two sets cannot describe different posteriors.
The panel drawers and labels come from `figbase`, which is tracked, so these
figures regenerate from a clone of this repository alone.

  python blogpost/figures/make_results_figs.py
  ...  --forests-only   only the forest figure
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from textwrap import shorten  # noqa: E402

import figbase  # noqa: E402
from figbase import AI_COLOR, AXIS_TITLES, HUMAN_COLOR  # noqa: E402
# The fit comes from `make_all`, which axis-checks it, so this file and the
# Plotly set cannot read a different posterior or accept a different labelling.
from make_all import load_flagship  # noqa: E402


def forests_axes_2x2(view, data, out: Path, n_top: int = 11,
                     sd_cap: float = 0.5) -> Path:
    """Top models, pinned frontier releases and human tiers, one panel per axis.

    2 rows x 2 cols with a single figure-level legend, rather than the memo's
    1xK strip: the post is always K=4 and a 2x2 reads at page width.
    """
    names = data.mlookup.sort_values("model_idx")["model"].tolist()
    fig, axes = plt.subplots(2, 2, figsize=(13, 13.6))
    for k, ax in enumerate(axes.flat):
        th = view.theta[:, :, k]                  # (S, M)
        mean, sd = th.mean(0), th.std(0)
        lo, hi = np.percentile(th, [2.5, 97.5], axis=0)
        is_h = np.asarray(data.is_human, dtype=bool)
        top = [i for i in np.argsort(-mean)
               if not is_h[i] and sd[i] < sd_cap][:n_top]
        frontier = [i for i in range(len(names))
                    if names[i] in figbase.FOREST_FRONTIER
                    and not is_h[i] and i not in top]
        tiers = list(np.where(is_h)[0])
        rows = ([(i, AI_COLOR, "o") for i in top]
                + [(i, AI_COLOR, "o") for i in frontier]
                + [(i, HUMAN_COLOR, "s") for i in tiers])
        rows.sort(key=lambda r: mean[r[0]])
        for y, (i, col, mark) in enumerate(rows):
            ax.plot([lo[i], hi[i]], [y, y], color=col, lw=1.6, alpha=0.9)
            ax.plot(mean[i], y, marker=mark, color=col, ms=6)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(
            [shorten(names[i] if is_h[i] else figbase.pretty(names[i]),
                     width=34, placeholder=" …")
             for i, _, _ in rows], fontsize=8)
        ax.set_title(AXIS_TITLES.get(view.names[k], view.names[k]), fontsize=10)
        ax.set_xlabel("ability (mean, 95% HDI)")
        ax.grid(axis="x", color="0.92", lw=0.6)
    handles = [
        plt.Line2D([], [], marker="o", color=AI_COLOR, ls="none",
                   label="models"),
        plt.Line2D([], [], marker="s", color=HUMAN_COLOR, ls="none",
                   label="human tiers"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Top models and human tiers per axis", fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96), h_pad=3.0, w_pad=4.0)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")
    return out


def main(forests_only: bool = False) -> None:
    view, data, raw = load_flagship()    # SystemExit if an axis is mislabelled
    forests_axes_2x2(view, data, HERE / "forests_axes_lw.png")
    if forests_only:
        return
    figbase.loadings_axes(view, data, HERE / "loadings_axes_lw.png")
    # per-axis abilities over time, the one the post carries as a PDF
    figbase.axis_timelines(view, data, raw, HERE / "timelines_axes_lw.pdf")


if __name__ == "__main__":
    main(forests_only="--forests-only" in sys.argv)
