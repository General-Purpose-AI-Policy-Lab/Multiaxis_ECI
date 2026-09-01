"""Two standalone diagrams for the LW post, in the docs/make_prior_graph.py
style (light-gray rounded panel, green rounded-rect nodes, green "+delta"
arrows). Style constants and helper functions (plate/edge/tbox/fedge/canvas)
are copied verbatim from docs/make_prior_graph.py so the look matches exactly.

  human_arrangement_lw.png  -> the tier-forest panel of the prior graph, the
                                bottom panel only, re-rendered at high res.
  model_family_example_lw.png -> a new illustrative one-family release chain
                                with effort variants hanging off a release,
                                in the same visual language.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

INK = "#26324B"
MUT = "#5a6478"
PLF = "#f2f4f8"   # plate fill
PLE = "#d8dde7"   # plate edge
HUM = "#0B8A66"   # green
HUMF = "#e9f6f1"  # green fill

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "text.color": INK,
})

OUT_DIR = Path(__file__).resolve().parent
DPI = 240

# --lang fr: the finished figure's Text objects are passed through this table
# (ordered, most specific first) and the render lands in fr/ instead. Math
# strings pass through untouched; only these phrases translate.
LANG = "en"
FR = [
    ("Committee of Average Humans", "Comité d'humains moyens"),
    ("Committee of Skilled Generalists", "Comité de généralistes qualifiés"),
    ("Committee of Domain Experts", "Comité d'experts du domaine"),
    ("High School Top Performer", "Lycéen, meilleur performeur"),
    ("High School Qualifier", "Lycéen qualifié"),
    ("Average Human", "Humain moyen"),
    ("Skilled Generalist", "Généraliste qualifié"),
    ("Domain Expert", "Expert du domaine"),
    ("Top Performer", "Meilleur performeur"),
    ("dashed: second parent —", "tirets = second parent"),
    ("one illustrative family", "une famille illustrative"),
    ("expected gain grows with the gap", "le gain attendu croît avec l'écart"),
    ("small offsets, unordered", "petits écarts, non ordonnés"),
    ("Model 1.0\n(Jan 2025)", "Modèle 1.0\n(janv. 2025)"),
    ("Model 1.5\n(Apr 2025)", "Modèle 1.5\n(avr. 2025)"),
    ("Model 2.0\n(Dec 2025)", "Modèle 2.0\n(déc. 2025)"),
    ("root", "racine"),
]


def _translate_fig(fig):
    import matplotlib.text
    for t in fig.findobj(matplotlib.text.Text):
        s = t.get_text()
        for en, fr in FR:
            s = s.replace(en, fr)
        t.set_text(s)


def plate(ax, x0, y0, x1, y1, label="", fill=PLF, pos="br"):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=0.14",
                                facecolor=fill, edgecolor=PLE, lw=1.1, zorder=0.5))
    if label:
        lx, ha = (x1 - 0.14, "right") if pos == "br" else (x0 + 0.14, "left")
        ax.text(lx, y0 + 0.1, label, ha=ha, va="bottom", fontsize=9.2,
                color=MUT, style="italic", zorder=2)


def edge(ax, p, q, color=INK, rA=0.36, rB=0.36, ls="-", lw=1.2, rad=0.0):
    p, q = np.asarray(p, float), np.asarray(q, float)
    u = (q - p) / np.linalg.norm(q - p)
    a, b = p + u * rA, q - u * (rB + 0.05)
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11,
                                 color=color, lw=lw, linestyle=ls, zorder=2,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0))


def tbox(ax, x, y, text, color, fill, fs=9, h=0.52, w=None):
    if w is None:
        w = 0.076 * len(text) + 0.46
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=0.12",
                                facecolor=fill, edgecolor=color, lw=1.3, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, zorder=4)
    return w, h


def fedge(ax, p, q, color, label=None, lab_dx=0.17, ls="-"):
    edge(ax, p, q, color=color, rA=0.0, rB=0.06, ls=ls, lw=1.25)
    if label:
        mx, my = (p[0] + q[0]) / 2 + lab_dx, (p[1] + q[1]) / 2
        ax.text(mx, my, label, ha="left", va="center", fontsize=9.4, color=color,
                zorder=4)


def canvas(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def save(fig, name):
    out_dir = OUT_DIR / "fr" if LANG == "fr" else OUT_DIR
    out_dir.mkdir(exist_ok=True)
    if LANG == "fr":
        _translate_fig(fig)
    out = out_dir / name
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# ================= Figure 1: human tier arrangement =================
# Verbatim re-render of the bottom ("tier forest") panel of
# docs/make_prior_graph.py page 3.

def make_human_arrangement():
    fig, ax = canvas(11.6, 7.4, (0.2, 11.5), (0.3, 7.75))

    plate(ax, 0.4, 0.55, 11.3, 7.5, "axes  k = 1…K")
    T = {}
    tiers = [
        ("Average Human",                    5.3, 1.35),
        ("Skilled Generalist",               4.2, 2.85),
        ("Committee of Average Humans",      7.5, 2.85),
        ("Domain Expert",                    4.2, 4.35),
        ("Committee of Skilled Generalists", 7.5, 4.35),
        ("Top Performer",                    3.0, 5.85),
        ("Committee of Domain Experts",      6.2, 5.85),
        ("High School Qualifier",            1.7, 1.35),
        ("High School Top Performer",        1.7, 2.85),
    ]
    for name, x, y in tiers:
        T[name] = (x, y) + tbox(ax, x, y, name, HUM, HUMF, fs=9.3, h=0.5)

    def tpt(name, side):
        x, y, w, h = T[name]
        return (x, y + h / 2) if side == "t" else (x, y - h / 2)

    forest = [
        ("Average Human", "Skilled Generalist"),
        ("Average Human", "Committee of Average Humans"),
        ("Skilled Generalist", "Domain Expert"),
        ("Skilled Generalist", "Committee of Skilled Generalists"),
        ("Domain Expert", "Top Performer"),
        ("Domain Expert", "Committee of Domain Experts"),
        ("High School Qualifier", "High School Top Performer"),
    ]
    for a, b in forest:
        fedge(ax, tpt(a, "t"), tpt(b, "b"), HUM, label=r"$+\delta$")

    def rpt(name):
        x, y, w, h = T[name]
        return (x + w / 2, y)

    merges = [("High School Qualifier", "Domain Expert"),
              ("High School Top Performer", "Top Performer")]
    for a, b in merges:
        edge(ax, rpt(a), tpt(b, "b"), color=HUM, rA=0.0, rB=0.06,
             ls=(0, (3, 2)), lw=1.1)
    ax.text(8.2, 1.35, "dashed: second parent —\n"
            r"$\theta_t=\max(\mathrm{parents})+\delta$",
            ha="left", va="center", fontsize=9.2, color=HUM)

    for rx in (5.3, 1.7):
        ax.text(rx, 0.92, "root", ha="center", va="center", fontsize=9,
                color=MUT)

    save(fig, "human_arrangement_lw.png")


# ================= Figure 2: model-family release chain example =================

def make_model_family_example():
    fig, ax = canvas(12.6, 7.0, (0.2, 12.6), (0.2, 6.6))

    plate(ax, 0.4, 0.4, 12.2, 6.3, "one illustrative family")

    releases = [
        ("Model 1.0\n(Jan 2025)", 2.1, 2.4),
        ("Model 1.5\n(Apr 2025)", 5.7, 3.5),
        ("Model 2.0\n(Dec 2025)", 9.7, 5.0),
    ]
    R = {}
    for lab, x, y in releases:
        R[lab] = (x, y) + tbox(ax, x, y, lab, HUM, HUMF, fs=10, h=0.95, w=2.15)

    labs = [r[0] for r in releases]
    for a, b in zip(labs[:-1], labs[1:]):
        xa, ya, wa, ha_ = R[a]
        xb, yb, wb, hb_ = R[b]
        p = (xa + wa / 2 * 0.55, ya + ha_ / 2 * 0.55)
        q = (xb - wb / 2 * 0.55, yb - hb_ / 2 * 0.55)
        edge(ax, p, q, color=HUM, rA=0.0, rB=0.06, lw=1.4)

    ax.text((R[labs[0]][0] + R[labs[2]][0]) / 2, 1.55,
            "expected gain grows with the gap", ha="center", va="center",
            fontsize=9.4, color=HUM, style="italic")

    # thin dotted time axis under the releases
    ax.plot([1.1, 11.3], [1.05, 1.05], color=MUT, lw=1.0, zorder=1)
    for lab, x, y in releases:
        ax.plot([x, x], [0.95, 1.15], color=MUT, lw=1.0, zorder=1)
        _, _, w, h = R[lab]
        ax.plot([x, x], [1.15, y - h / 2], color=PLE, lw=0.9,
                ls=(0, (2, 2)), zorder=1)

    # effort variants hanging off the newest release
    parent_lab = labs[-1]
    px, py, pw, ph = R[parent_lab]
    variants = [("low", -1.4), ("high", 0.0), ("max", 1.4)]
    vy = py - ph / 2 - 1.35
    for vlab, dx in variants:
        vx = px + dx
        tbox(ax, vx, vy, vlab, HUM, HUMF, fs=9.5, h=0.55, w=1.05)
        ax.plot([vx, px], [vy + 0.28, py - ph / 2], color=MUT, lw=0.9,
                zorder=1)
    ax.text(px + 2.1, vy, "small offsets, unordered", ha="left", va="center",
            fontsize=9.2, color=MUT, style="italic")

    save(fig, "model_family_example_lw.png")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--lang", choices=["en", "fr"], default="en")
    LANG = p.parse_args().lang
    make_human_arrangement()
    make_model_family_example()
