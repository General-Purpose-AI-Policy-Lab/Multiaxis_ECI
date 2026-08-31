"""The two per-axis result figures for the LessWrong post.

Same code path as the memo (`memo/make_memo_figs.py`): the flagship K=4 fit,
majority chains only, raw display frame. Only the axis display strings and the
output location change, so the post and the memo cannot drift apart.

  python lw_post/figures/make_results_figs.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT / "memo"))
sys.path.insert(0, str(HERE))

import make_memo_figs as mmf  # noqa: E402
# The fit comes from `make_all`, which axis-checks it, so this file and the
# Plotly set cannot read a different posterior or accept a different labelling.
from make_all import load_flagship  # noqa: E402


def forests_axes_2x2(view, data, n_top: int = 11, sd_cap: float = 0.5) -> None:
    """Same as memo's forests_axes (data prep, top-model selection, frontier
    releases, human tiers, colors, markers) but 2 rows x 2 cols instead of
    1xK, with a single figure-level legend. K=4 assumed (post is always K=4)."""
    plt, np, shorten = mmf.plt, mmf.np, mmf.shorten
    _AI_COLOR, _HUMAN_COLOR = mmf._AI_COLOR, mmf._HUMAN_COLOR
    _FOREST_FRONTIER, _pretty = mmf._FOREST_FRONTIER, mmf._pretty
    _AXIS_TITLES, ASSETS = mmf._AXIS_TITLES, mmf.ASSETS

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
                    if names[i] in _FOREST_FRONTIER
                    and not is_h[i] and i not in top]
        tiers = list(np.where(is_h)[0])
        rows = ([(i, _AI_COLOR, "o") for i in top]
                + [(i, _AI_COLOR, "o") for i in frontier]
                + [(i, _HUMAN_COLOR, "s") for i in tiers])
        rows.sort(key=lambda r: mean[r[0]])
        for y, (i, col, mark) in enumerate(rows):
            ax.plot([lo[i], hi[i]], [y, y], color=col, lw=1.6, alpha=0.9)
            ax.plot(mean[i], y, marker=mark, color=col, ms=6)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(
            [shorten(names[i] if is_h[i] else _pretty(names[i]),
                     width=34, placeholder=" …")
             for i, _, _ in rows], fontsize=8)
        ax.set_title(_AXIS_TITLES.get(view.names[k], view.names[k]), fontsize=10)
        ax.set_xlabel("ability (mean, 95% HDI)")
        ax.grid(axis="x", color="0.92", lw=0.6)
    handles = [
        plt.Line2D([], [], marker="o", color=_AI_COLOR, ls="none", label="models"),
        plt.Line2D([], [], marker="s", color=_HUMAN_COLOR, ls="none",
                   label="human tiers"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Top models and human tiers per axis",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.03, 1, 0.96), h_pad=3.0, w_pad=4.0)
    out = ASSETS / "forests_axes.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main(forests_only: bool = False) -> None:
    mmf.ASSETS = HERE                    # figure functions write here
    view, data, _raw = load_flagship()   # SystemExit if an axis is mislabelled
    fig_fns = ((forests_axes_2x2, "forests_axes"),) if forests_only else (
        (mmf.loadings_axes, "loadings_axes"),
        (forests_axes_2x2, "forests_axes"),
    )
    for fn, stem in fig_fns:
        fn(view, data)
        (HERE / f"{stem}.png").replace(HERE / f"{stem}_lw.png")
        print(f"  -> {stem}_lw.png")
    if forests_only:
        return
    # per-axis abilities over time; the memo writes this one as a PDF
    mmf.axis_timelines(view, data, _raw)
    (HERE / "timelines_axes.pdf").replace(HERE / "timelines_axes_lw.pdf")
    print("  -> timelines_axes_lw.pdf")


if __name__ == "__main__":
    main(forests_only="--forests-only" in sys.argv)
