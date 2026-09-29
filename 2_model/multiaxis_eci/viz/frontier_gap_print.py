"""The months-behind figure for print, in matplotlib: the French render the lab's reports use, in
the lab's house style (navy ink, 11 pt body, a solid date axis with outward ticks and no label, no
vertical axis, horizontal grid only, text written on the curves with a white halo, a handful of
well-known models named, the lab's marks in the corners). Same data as `frontier_gap.lag_fig`:
one dot per open-weights record with its 50% interval, one smoothed curve per access scope with
its 80% band. The interactive twin stays the Plotly figure.

What one adjusts (texts, colours, named models and their offsets, sizes) is a named constant below.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patheffects import withStroke  # noqa: E402

from multiaxis_eci.viz.frontier_gap import _lag_curve, _today  # noqa: E402

# ── Style ─────────────────────────────────────────────────────────────────────────────────────
ENCRE = "#14243d"            # navy ink in place of black: texts, axis, ticks
GRILLE = "#e0e0e0"
GRIS = "#555555"             # secondary texts: the lab's marks, the today label
COULEURS = {"all": "#24467a", "public": "#3b7a3b", "semi_private": "#b8912a", "private": "#7a1f2b"}
BANDE_ALPHA = 0.10           # the 80% band under each curve
COURBE_LW = 1.8
POINT_MS = 4.2               # record dots, white edge
BARRE_LW = 0.8               # their 50% interval
BARRE_ALPHA = 0.35
HALO = [withStroke(linewidth=3, foreground="white")]
FIGSIZE = (6.2, 3.7)         # inches: wider than a page column, reduced at layout
CORPS = 11                   # body size, as the report's boxes (Times 11 on a 170 x 237 mm page)
Y_PAS = 4                    # months between horizontal grid lines
# The house style as rcParams, applied while the figure is drawn and while it is saved (tick labels
# are made at draw time, so a style applied only while building would miss them).
STYLE = {"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
         "font.size": CORPS, "axes.labelsize": "medium", "svg.fonttype": "none",
         "text.color": ENCRE, "axes.labelcolor": ENCRE, "xtick.color": ENCRE, "ytick.color": ENCRE,
         "axes.edgecolor": ENCRE, "axes.spines.top": False, "axes.spines.right": False,
         "axes.spines.left": False, "xtick.direction": "out", "xtick.major.size": 5,
         "ytick.left": False, "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRILLE,
         "grid.linewidth": 0.8, "axes.axisbelow": True, "legend.frameon": False,
         "axes.unicode_minus": True}

# ── Texts (French) ────────────────────────────────────────────────────────────────────────────
Y_LABEL = "Mois de retard sur la frontière\ndes modèles fermés"
SCOPES_FR = {"all": "Tous les benchmarks", "public": "Benchmarks publics",
             "semi_private": "Benchmarks semi-privés", "private": "Benchmarks privés"}
AUJOURDHUI = "aujourd’hui"
# A few records everyone knows, named beside their dot on all benchmarks: name -> (text, offset in
# points from the dot, horizontal and vertical alignment). Positions chosen by eye.
MODELES = {"LLaMA-65B": ("LLaMA 65B", (0, 7), "center", "bottom"),
           "Llama-3.1-405B": ("Llama 3.1 405B", (-6, 0), "right", "center"),
           "DeepSeek-R1": ("DeepSeek R1", (0, -7), "center", "top"),
           "kimi-k3": ("Kimi K3", (0, -8), "center", "top")}
MARQUE_LOGO = ("GPAI", "Policy Lab")  # top right, after a bracket
MARQUE_SITE = "gpaipolicylab.org"     # bottom left
MARQUE_COPYRIGHT = "©"                # bottom right
MARQUE_LOGO_DY = 3                    # top of the logo, in points above the axes' top right corner
MARQUE_DY = -22                       # site and copyright, in points under the date axis
LIBELLE_DX = 3                        # points between a curve's end and its name
LIBELLE_ECART = 1.1                   # line heights: curve names closer than this are pushed apart


def _spread(ys: list[float], gap: float) -> list[float]:
    """Move sorted positions apart until neighbours are `gap` apart, keeping their mean."""
    order = np.argsort(ys)
    y = np.array(ys, float)[order]
    for _ in range(50):
        moved = False
        for i in range(1, len(y)):
            if y[i] - y[i - 1] < gap:
                push = (gap - (y[i] - y[i - 1])) / 2
                y[i - 1] -= push
                y[i] += push
                moved = True
        if not moved:
            break
    out = np.empty_like(y)
    out[order] = y
    return list(out)


def lag_fig_print(results: dict, scopes: list[str], *, today=None, counts: dict | None = None,
                  smooth_months: float = 6.0, label_scope: str = "all", brand: bool = True):
    """The months-behind figure in the house style, French texts. `counts` gives each scope's
    number of benchmarks for its name ("Benchmarks privés (13)"). Returns the matplotlib figure."""
    today_d = _today(today)
    curves = {s: _lag_curve(results[s], smooth_months, today_d) for s in scopes}
    drawn = [s for s in scopes if curves[s] is not None]
    first = pd.Timestamp(results[label_scope].lag_df["release_date"].min())
    w0, w1 = first - pd.Timedelta(days=120), pd.Timestamp(f"{today_d.year + 1}-01-01")

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
        lows, highs, ends = [0.0], [0.0], []
        for s in drawn:
            col = COULEURS.get(s, ENCRE)
            grid_d, med, lo, hi = curves[s]
            ax.fill_between(grid_d, lo, hi, color=col, alpha=BANDE_ALPHA, lw=0, zorder=1)
            ax.plot(grid_d, med, color=col, lw=COURBE_LW, zorder=3, solid_capstyle="round")
            df = results[s].lag_df
            df = df[df["lag_months_median"].notna() & (df["release_date"] >= w0)]
            ax.vlines(df["release_date"], df["lag_hdi50_low"], df["lag_hdi50_high"], color=col,
                      lw=BARRE_LW, alpha=BARRE_ALPHA, zorder=2)
            ax.plot(df["release_date"], df["lag_months_median"], "o", ms=POINT_MS, color=col,
                    mec="white", mew=0.8, zorder=4)
            inside = (pd.DatetimeIndex(grid_d) >= w0)
            lows += [np.nanmin(lo[inside]), float(df["lag_hdi50_low"].min())]
            highs += [np.nanmax(hi[inside]), float(df["lag_hdi50_high"].max())]
            if np.isfinite(med[-1]):
                name = SCOPES_FR.get(s, s) + (f" ({counts[s]})" if counts and s in counts else "")
                ends.append((float(med[-1]), name, col, pd.Timestamp(grid_d[-1])))

        # The range hugs what is drawn (a month of air), the grid keeps its round steps inside it
        y_lo, y_hi = min(lows) - 1, max(highs) + 2
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlim(w0, w1)
        ax.set_yticks(np.arange(np.ceil(y_lo / Y_PAS) * Y_PAS, y_hi, Y_PAS))
        ax.axhline(0, color="#b0b0b0", lw=0.8, zorder=1)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_ylabel(Y_LABEL)

        # Today: a dotted rule, named at its foot (the lab's marks hold the top right corner)
        ax.axvline(today_d, color=GRIS, lw=0.8, ls=(0, (1, 2)), zorder=1)
        ax.annotate(AUJOURDHUI, (today_d, 0), xycoords=("data", "axes fraction"), xytext=(-3, 3),
                    textcoords="offset points", ha="right", va="bottom", fontsize="small", color=GRIS)

        # Each curve named just past its right end, in its colour; names that would touch are pushed
        # apart by the least that separates them, measured in points once the layout is known
        fig.canvas.draw()
        pt_par_mois = ax.bbox.height * 72 / fig.dpi / (y_hi - y_lo)
        ligne = LIBELLE_ECART * plt.rcParams["font.size"] * 0.833  # "small"
        y_pt = [e[0] * pt_par_mois for e in ends]
        names = []
        for (y_end, name, col, x_end), y0, y1 in zip(ends, y_pt, _spread(y_pt, ligne)):
            names.append(ax.annotate(name, (x_end, y_end), xytext=(LIBELLE_DX, y1 - y0),
                        textcoords="offset points", ha="left", va="center", fontsize="small",
                        color=col, path_effects=HALO, zorder=6, annotation_clip=False))

        # A few well-known records named beside their dot
        df = results[label_scope].lag_df.set_index("name")
        for key, (text, (dx, dy), ha, va) in MODELES.items():
            if key in df.index and np.isfinite(df.at[key, "lag_months_median"]):
                ax.annotate(text, (df.at[key, "release_date"], df.at[key, "lag_months_median"]),
                            xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                            fontsize="small", color=ENCRE, path_effects=HALO, zorder=7)

        if brand:
            fig.canvas.draw()
            rendu = fig.canvas.get_renderer()
            # The right-hand marks end where the curve names do, clear of the today rule
            droite = max([1.0] + [ax.transAxes.inverted().transform(n.get_window_extent(rendu))[1, 0]
                                  for n in names])
            logo = [ax.annotate(m, (droite, 1), xycoords="axes fraction", xytext=(0, MARQUE_LOGO_DY - 9 * i),
                                textcoords="offset points", ha="left", va="top", fontsize=7.5,
                                fontweight="bold" if i == 0 else "normal", color=ENCRE,
                                annotation_clip=False)
                    for i, m in enumerate(MARQUE_LOGO)]
            largeur = max(m.get_window_extent(rendu).width for m in logo) * 72 / fig.dpi
            for m in logo:
                m.xyann = (-largeur, m.xyann[1])
            ax.annotate("[", (droite, 1), xycoords="axes fraction", xytext=(-largeur - 3, MARQUE_LOGO_DY + 2),
                        textcoords="offset points", ha="right", va="top", fontsize=17, color=ENCRE,
                        annotation_clip=False)
            # The site starts where the figure does, at the left edge of the y caption
            gauche = min(0.0, ax.transAxes.inverted().transform(
                ax.yaxis.label.get_window_extent(rendu))[0, 0])
            ax.annotate(MARQUE_SITE, (gauche, 0), xycoords="axes fraction", xytext=(0, MARQUE_DY),
                        textcoords="offset points", ha="left", va="top", fontsize=6, color=GRIS,
                        annotation_clip=False)
            ax.annotate(MARQUE_COPYRIGHT, (droite, 0), xycoords="axes fraction", xytext=(0, MARQUE_DY),
                        textcoords="offset points", ha="right", va="top", fontsize=6, color=GRIS,
                        annotation_clip=False)
    return fig


def save_print(fig, png_path, svg_path) -> None:
    """The PNG (300 dpi) and the SVG with its text kept as text, cropped to what is drawn."""
    from pathlib import Path
    for p in (png_path, svg_path):
        Path(p).parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(STYLE):
        fig.savefig(png_path, dpi=300, bbox_inches="tight", facecolor="white")
        fig.savefig(svg_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
