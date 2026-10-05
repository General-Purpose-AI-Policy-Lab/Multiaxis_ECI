"""The months-behind figure for print, in matplotlib: the static render the lab's reports use, in
French and in English (`lang`), in the lab's house style (navy ink, 11 pt body, a solid date axis with outward ticks and no label, no
vertical axis, horizontal grid only, text written on the curves with a white halo, a handful of
well-known models named, the lab's marks in the corners). Same data as `frontier_gap.lag_fig`:
one dot per follower record (open-weights models, or Chinese models with `group="country"`) with its
50% interval, one smoothed curve per access scope with its 80% band, from 2024 (`GAP_START`) on an
axis cut at 20 months (`LAG_Y_MAX`). Dots that overlap share one disc split into sectors, one per
colour. The interactive twin stays the Plotly figure.

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

from multiaxis_eci.analysis.frontier_gap import GROUP_NOUNS, GROUP_PAIRS  # noqa: E402
from multiaxis_eci.viz.frontier_gap import GAP_START, LAG_Y_MAX, _lag_curve, _today  # noqa: E402
from multiaxis_eci.viz.i18n import FR_GROUPS  # noqa: E402

# ── Style ─────────────────────────────────────────────────────────────────────────────────────
ENCRE = "#14243d"            # navy ink in place of black: texts, axis, ticks
GRILLE = "#e0e0e0"
GRIS = "#555555"             # secondary texts: the lab's marks, the today label
COULEURS = {"all": "#24467a", "public": "#3b7a3b", "semi_private": "#b8912a", "private": "#7a1f2b"}
BANDE_ALPHA = 0.10           # the 80% band under each curve
COURBE_LW = 1.8
POINT_MS = 3.4               # record dots, white edge
POINT_MEW = 0.6
# Dots that overlap (the same record once per scope, or two records of one date and lag) share one
# disc, each showing only its sector in its own colour (user, 2026-10-05): dots whose centres are
# closer than this share of a diameter are drawn as one split disc.
CHEVAUCHEMENT = 0.7
BARRE_LW = 0.8               # their 50% interval
BARRE_ALPHA = 0.35
HALO = [withStroke(linewidth=3, foreground="white")]
FIGSIZE = (6.2, 3.7)         # inches: wider than a page column, reduced at layout
CORPS = 11                   # body size, as the report's boxes (Times 11 on a 170 x 237 mm page)
Y_PAS = 4                    # months between horizontal grid lines
# Top of the vertical axis per group, None to follow the data: both stop at 20 months for
# legibility, the few intervals above it run off the top (user, 2026-10-05; the country figure
# stopped at 24 since 2026-09-30).
Y_MAX = {"openness": LAG_Y_MAX, "country": LAG_Y_MAX}
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
# The vertical caption names the leader group whose frontier the lag is read off, in the French
# of viz.i18n.FR_GROUPS: "... des modèles fermés", "... des modèles américains".
Y_LABEL = {kind: f"Mois de retard sur la frontière\ndes {FR_GROUPS[leader][1]}"
           for kind, (leader, _) in GROUP_PAIRS.items()}
SCOPES_FR = {"all": "Tous les benchmarks", "public": "Benchmarks publics",
             "semi_private": "Benchmarks semi-privés", "private": "Benchmarks privés"}
AUJOURDHUI = "aujourd’hui"
# The same texts in English, for the English render of the same figure (lang="en").
TEXTS = {"fr": {"y": Y_LABEL, "scopes": SCOPES_FR, "today": AUJOURDHUI},
         "en": {"y": {kind: f"Months behind the frontier\nof {GROUP_NOUNS[kind][leader][0]}"
                      for kind, (leader, _) in GROUP_PAIRS.items()},
                "scopes": {"all": "All benchmarks", "public": "Public benchmarks",
                           "semi_private": "Semi-private benchmarks",
                           "private": "Private benchmarks"},
                "today": "today"}}
# A few records everyone knows, named beside their dot on all benchmarks, per group: name -> (text,
# offset in points from the dot, horizontal and vertical alignment). Positions chosen by eye; a
# group without an entry names none.
MODELES = {"openness": {"LLaMA-65B": ("LLaMA 65B", (0, 7), "center", "bottom"),
                        "Llama-3.1-405B": ("Llama 3.1 405B", (-6, 0), "right", "center"),
                        "DeepSeek-R1": ("DeepSeek R1", (0, -7), "center", "top"),
                        "kimi-k3": ("Kimi K3", (0, -8), "center", "top")},
           "country": {"DeepSeek-R1": ("DeepSeek R1", (6, 0), "left", "center"),
                       "kimi-k3": ("Kimi K3", (-6, 0), "right", "center")}}
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


def _sector(a0: float, a1: float, n: int = 24) -> list[tuple[float, float]]:
    """The vertices of a disc sector from angle a0 to a1 (radians), centred on the marker."""
    t = np.linspace(a0, a1, n)
    return [(0.0, 0.0), *zip(np.cos(t), np.sin(t)), (0.0, 0.0)]


def _draw_dots(ax, dots: list[tuple], fig) -> None:
    """Draw `dots` ((x, y, colour) in data units), a group of overlapping ones as one disc split
    into equal sectors, one per colour (scope) in the group, in the order the scopes were drawn."""
    if not dots:
        return
    fig.canvas.draw()
    xy = ax.transData.transform([(mdates.date2num(x), y) for x, y, _ in dots])
    diam = POINT_MS * fig.dpi / 72                      # marker diameter in pixels
    groups, seen = [], set()
    for i in range(len(dots)):
        if i in seen:
            continue
        g, todo = [i], [i]
        seen.add(i)
        while todo:                                     # chains of overlaps form one group
            j = todo.pop()
            near = np.where(np.hypot(*(xy - xy[j]).T) < CHEVAUCHEMENT * diam)[0]
            for k in near:
                if k not in seen:
                    seen.add(k)
                    g.append(k)
                    todo.append(k)
        groups.append(sorted(g))
    for g in groups:
        x, y = np.mean([xy[i] for i in g], axis=0)
        cx, cy = ax.transData.inverted().transform((x, y))
        cols = list(dict.fromkeys(dots[i][2] for i in g))   # one sector per colour, drawing order
        if len(cols) == 1:
            ax.plot(mdates.num2date(cx), cy, "o", ms=POINT_MS, color=cols[0],
                    mec="white", mew=POINT_MEW, zorder=4)
            continue
        step = 2 * np.pi / len(cols)
        for n, col in enumerate(cols):
            a0 = np.pi / 2 - n * step                   # first sector from the top, clockwise
            ax.plot(mdates.num2date(cx), cy, marker=_sector(a0 - step, a0), ms=POINT_MS * 1.1,
                    ls="none", color=col, mec="white", mew=POINT_MEW * 0.6, zorder=4)


def lag_fig_print(results: dict, scopes: list[str], *, today=None, counts: dict | None = None,
                  smooth_months: float = 6.0, label_scope: str = "all", brand: bool = True,
                  group: str = "openness", lang: str = "fr"):
    """The months-behind figure in the house style, French texts (`lang="en"`: English). `counts`
    gives each scope's number of benchmarks for its name ("Benchmarks privés (13)"); `group` picks
    the vertical caption (`openness` or `country`). Returns the matplotlib figure."""
    txt = TEXTS[lang]
    today_d = _today(today)
    curves = {s: _lag_curve(results[s], smooth_months, today_d) for s in scopes}
    drawn = [s for s in scopes if curves[s] is not None]
    first = pd.Timestamp(results[label_scope].lag_df["release_date"].min())
    w0 = max(first - pd.Timedelta(days=120), GAP_START)
    w1 = pd.Timestamp(f"{today_d.year + 1}-01-01")

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
        lows, highs, ends, dots = [0.0], [0.0], [], []
        for s in drawn:
            col = COULEURS.get(s, ENCRE)
            grid_d, med, lo, hi = curves[s]
            ax.fill_between(grid_d, lo, hi, color=col, alpha=BANDE_ALPHA, lw=0, zorder=1)
            ax.plot(grid_d, med, color=col, lw=COURBE_LW, zorder=3, solid_capstyle="round")
            df = results[s].lag_df
            df = df[df["lag_months_median"].notna() & (df["release_date"] >= w0)]
            ax.vlines(df["release_date"], df["lag_hdi50_low"], df["lag_hdi50_high"], color=col,
                      lw=BARRE_LW, alpha=BARRE_ALPHA, zorder=2)
            dots += [(pd.Timestamp(d), float(v), col)
                     for d, v in zip(df["release_date"], df["lag_months_median"])]
            inside = (pd.DatetimeIndex(grid_d) >= w0)
            lows += [np.nanmin(lo[inside]), float(df["lag_hdi50_low"].min())]
            highs += [np.nanmax(hi[inside]), float(df["lag_hdi50_high"].max())]
            if np.isfinite(med[-1]):
                name = txt["scopes"].get(s, s) + (f" ({counts[s]})" if counts and s in counts else "")
                ends.append((float(med[-1]), name, col, pd.Timestamp(grid_d[-1])))

        # The range hugs what is drawn (a month of air), the grid keeps its round steps inside it
        y_lo, y_hi = min(lows) - 1, max(highs) + 2
        if Y_MAX.get(group) is not None:
            y_hi = min(y_hi, Y_MAX[group])
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlim(w0, w1)
        ax.set_yticks(np.arange(np.ceil(y_lo / Y_PAS) * Y_PAS, y_hi + 1e-9, Y_PAS))  # a capped top keeps its tick
        ax.axhline(0, color="#b0b0b0", lw=0.8, zorder=1)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_ylabel(txt["y"][group])
        _draw_dots(ax, dots, fig)   # once the limits are set: overlap is read in points

        # Today: a dotted rule, named at its foot (the lab's marks hold the top right corner)
        ax.axvline(today_d, color=GRIS, lw=0.8, ls=(0, (1, 2)), zorder=1)
        ax.annotate(txt["today"], (today_d, 0), xycoords=("data", "axes fraction"), xytext=(-3, 3),
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
        for key, (text, (dx, dy), ha, va) in MODELES.get(group, {}).items():
            if key in df.index and np.isfinite(df.at[key, "lag_months_median"]):
                ax.annotate(text, (df.at[key, "release_date"], df.at[key, "lag_months_median"]),
                            xytext=(dx, dy), textcoords="offset points", ha=ha, va=va,
                            fontsize="x-small", color=ENCRE, path_effects=HALO, zorder=7)

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
