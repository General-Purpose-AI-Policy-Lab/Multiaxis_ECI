"""Beta-MIRT prior graph, centered parametrization. 4-page vector PDF.

Represents the flagship fit: K=4, non-negative loadings, fixed guessing
floors (no ceilings), pooled per-benchmark noise, merged human order,
Brownian-motion lineage steps over release gaps.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, FancyArrowPatch
from matplotlib.backends.backend_pdf import PdfPages

INK = "#26324B"
MUT = "#5a6478"
PLF = "#f2f4f8"   # plate fill
PLE = "#d8dde7"   # plate edge
OBS = "#dfe8f3"   # observed fill
FLR = "#C4568C"   # floor constant
HUM = "#0B8A66"   # human prior
FAM = "#C77800"   # model-family prior
HUMF = "#e9f6f1"
FAMF = "#fbf1de"
FLRF = "#faeef5"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "text.color": INK,
})

R = 0.36


def stoch(ax, x, y, math, dist=None, r=R, fs=13, below=False, dist_bg=None):
    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=INK, lw=1.5, zorder=3))
    ax.text(x, y, math, ha="center", va="center", fontsize=fs, zorder=4)
    if dist:
        dy, va = (-r - 0.14, "top") if below else (r + 0.14, "bottom")
        bbox = (dict(boxstyle="square,pad=0.1", facecolor=dist_bg, edgecolor="none")
                if dist_bg else None)
        ax.text(x, y + dy, dist, ha="center", va=va, fontsize=9.2,
                color=MUT, zorder=4, bbox=bbox)


def det(ax, x, y, math, r=R, fs=13, color=INK):
    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=color, lw=1.5,
                        linestyle=(0, (4, 2.4)), zorder=3))
    ax.text(x, y, math, ha="center", va="center", fontsize=fs, zorder=4)


def obsn(ax, x, y, math, r=R, fs=13):
    ax.add_patch(Circle((x, y), r, facecolor=OBS, edgecolor=INK, lw=1.5, zorder=3))
    ax.text(x, y, math, ha="center", va="center", fontsize=fs, zorder=4)


def plate(ax, x0, y0, x1, y1, label="", fill=PLF, pos="br"):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=0.14",
                                facecolor=fill, edgecolor=PLE, lw=1.1, zorder=0.5))
    if label:
        lx, ha = (x1 - 0.14, "right") if pos == "br" else (x0 + 0.14, "left")
        ax.text(lx, y0 + 0.1, label, ha=ha, va="bottom", fontsize=9.2,
                color=MUT, style="italic", zorder=2)


def edge(ax, p, q, color=INK, rA=R, rB=R, ls="-", lw=1.2, rad=0.0):
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


def mbox(ax, x0, y0, x1, y1, title_s, sub=None, note=None, edgec=PLE, fill=PLF,
         tc=INK):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=0.16",
                                facecolor=fill, edgecolor=edgec, lw=1.3, zorder=2))
    cx, top = (x0 + x1) / 2, y1
    ax.text(cx, top - 0.42, title_s, ha="center", va="center", fontsize=11.5,
            fontweight="bold", color=tc, zorder=3)
    if sub:
        ax.text(cx, top - 0.82, sub, ha="center", va="center", fontsize=9.5,
                color=MUT, zorder=3)
    if note:
        ax.text(cx, top - 1.14, note, ha="center", va="center", fontsize=9,
                color=MUT, zorder=3)


def zsn_node(ax, x, y):
    """The single per-axis ZeroSumNormal every base level is sliced from."""
    stoch(ax, x, y, r"$\theta^{\mathrm{zsn}}_{\cdot k}$",
          r"$\mathrm{ZeroSumNormal}(1)$")


def canvas(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


out = "/Users/yassineessifi/Desktop/ECI_Bayesian/docs/prior_graph.pdf"

with PdfPages(out) as pdf:
    # ================= Page 1 : overview =================
    fig, ax = canvas(12.8, 8.8, (0, 11.2), (0, 7.75))
    ax.text(0.05, 7.65, "Overview", fontsize=16, fontweight="bold",
            ha="left", va="top")

    mbox(ax, 0.6, 5.6, 4.4, 6.95, "Test-taker structure",
         "human tiers · model families · effort variants", "priors: pages 3–4")
    mbox(ax, 5.6, 5.6, 9.4, 6.95, "Benchmark structure",
         "loading scales · difficulty · noise", "page 2")

    stoch(ax, 2.5, 4.35, r"$\theta_{mk}$", r=0.4)
    ax.text(2.5, 3.7, "abilities", ha="center", va="center", fontsize=9.5,
            color=MUT)
    stoch(ax, 6.35, 4.35, r"$A_{bk}$")
    stoch(ax, 7.5, 4.35, r"$D_b$")
    stoch(ax, 8.65, 4.35, r"$\phi_b$")

    mbox(ax, 3.3, 2.2, 7.2, 3.35, "Beta-MIRT likelihood")
    ax.text(5.25, 2.55, r"$\eta_n \;\rightarrow\; \mu_n \;\rightarrow\;"
            r" \mathrm{Beta}$", ha="center", va="center", fontsize=9.5,
            color=MUT, zorder=3)
    mbox(ax, 7.5, 2.35, 10.7, 3.3, "Fixed: floors",
         r"$c_b$  chance floor, read from file",
         edgec=FLR, fill=FLRF, tc=FLR)

    obsn(ax, 5.25, 1.0, r"$y_n$", r=0.4)

    edge(ax, (2.5, 5.6), (2.5, 4.35), rA=0.0, rB=0.4)
    edge(ax, (6.35, 5.6), (6.35, 4.35), rA=0.0)
    edge(ax, (7.5, 5.6), (7.5, 4.35), rA=0.0)
    edge(ax, (8.65, 5.6), (8.65, 4.35), rA=0.0)
    edge(ax, (2.5, 3.95), (4.1, 3.35), rA=0.4, rB=0.0)
    edge(ax, (6.35, 3.99), (5.8, 3.35), rB=0.0)
    edge(ax, (7.5, 3.99), (6.5, 3.35), rB=0.0)
    edge(ax, (8.65, 3.99), (7.1, 3.4), rB=0.0)
    edge(ax, (7.5, 2.83), (7.22, 2.83), rA=0.0, rB=0.0, color=FLR)
    edge(ax, (5.25, 2.2), (5.25, 1.0), rA=0.0, rB=0.4)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ================= Page 2 : the model =================
    fig, ax = canvas(14.7, 9.9, (0, 13.0), (-2.35, 7.45))
    ax.text(0.05, 7.35, "Beta-MIRT model  (K = 4)", fontsize=16,
            fontweight="bold", ha="left", va="top")

    stoch(ax, 5.3, 6.35, r"$\tau_A$", r"$\mathrm{LogNormal}(\log 0.5,\;0.5)$")
    stoch(ax, 7.3, 6.35, r"$\tau_{CD}$", r"$\mathrm{LogNormal}(\log 3,\;1)$")
    # pooled noise hyperpriors: the sigma_b population is learned, not fixed
    stoch(ax, 9.8, 6.35, r"$\mu_\sigma$", r"$\mathcal{N}(\log 0.05,\;0.5)$")
    stoch(ax, 11.6, 6.35, r"$\tau_\sigma$", r"$\mathrm{HalfNormal}(0.5)$")

    # model block (left)
    plate(ax, 0.6, 2.5, 3.5, 5.5)
    ax.text(3.36, 5.38, "models  m = 1…M  ·  axes  k = 1…K", ha="right", va="top",
            fontsize=8.2, color=MUT, style="italic", zorder=2)
    ax.text(0.85, 5.02, r"rows of $\theta$:", ha="left", va="center", fontsize=9.2,
            color=MUT)
    bands = [(r"$\mathrm{ZeroSumNormal}(1)$",
              "white", PLE, INK),
             ("+  human prior", HUMF, HUM, HUM),
             ("+  model-family prior", FAMF, FAM, FAM)]
    for i, (lab, fc, ec, tc) in enumerate(bands):
        yb = 4.62 - i * 0.5
        ax.add_patch(FancyBboxPatch((0.85, yb - 0.2), 2.5, 0.42,
                                    boxstyle="round,pad=0,rounding_size=0.08",
                                    facecolor=fc, edgecolor=ec, lw=1.1, zorder=2))
        ax.text(2.1, yb + 0.01, lab, ha="center", va="center", fontsize=9,
                color=tc, zorder=3)
    stoch(ax, 2.1, 2.95, r"$\theta_{mk}$")

    # benchmark plate (right)
    plate(ax, 3.9, 2.5, 12.6, 5.5, "benchmarks  b = 1…B", pos="bl")
    plate(ax, 4.2, 2.95, 6.65, 5.2, "axes  k = 1…K", fill="white")
    stoch(ax, 5.3, 4.35, r"$A_{bk}$", r"$\mathrm{HalfNormal}(\tau_A)$",
          dist_bg="white")
    stoch(ax, 7.3, 4.35, r"$D_b$", r"$\mathcal{N}(0,\;\tau_{CD}^2)$",
          dist_bg=PLF)
    stoch(ax, 10.7, 4.55, r"$\sigma_b$",
          "$\\log\\sigma_b\\sim$\n$\\mathcal{N}(\\mu_\\sigma,\\;\\tau_\\sigma^2)$",
          dist_bg="white")
    det(ax, 10.7, 3.25, r"$\phi_b$")
    ax.text(11.15, 3.25, r"$\phi_b=\frac{1}{4\sigma_b^2}-1$", ha="left",
            va="center", fontsize=10,
            bbox=dict(boxstyle="square,pad=0.1", facecolor=PLF, edgecolor="none"))

    plate(ax, 1.45, 0.1, 8.7, 2.1, "observations  n = 1…N")
    det(ax, 2.7, 1.2, r"$\eta_n$")
    det(ax, 4.9, 1.2, r"$\mu_n$")
    obsn(ax, 7.3, 1.2, r"$y_n$")

    ax.text(4.9, -0.35, r"$\eta_n=\sum_k A_{b_n k}\,\theta_{m_n k}-D_{b_n}$",
            ha="center", va="center", fontsize=11)
    ax.text(4.9, -0.87, r"$\mu_n=c_{b_n}+(1-c_{b_n})\,\sigma(\eta_n)$",
            ha="center", va="center", fontsize=11)
    ax.text(6.55, -0.87, r"$c_b$: chance floor, fixed", ha="left", va="center",
            fontsize=9, color=FLR)
    ax.text(4.9, -1.36,
            r"$y_n\sim\mathrm{Beta}\!\left(\mu_n\phi_{b_n},\;(1-\mu_n)\phi_{b_n}\right)$",
            ha="center", va="center", fontsize=11)

    edge(ax, (5.3, 6.35), (5.3, 4.35))            # tau_A -> A
    edge(ax, (7.3, 6.35), (7.3, 4.35))            # tau_CD -> D
    edge(ax, (9.8, 6.35), (10.7, 4.55))           # mu_sigma -> sigma
    edge(ax, (11.6, 6.35), (10.7, 4.55))          # tau_sigma -> sigma
    edge(ax, (10.7, 4.55), (10.7, 3.25))          # sigma -> phi
    edge(ax, (2.1, 2.95), (2.7, 1.2))             # theta -> eta (short, no crossing)
    edge(ax, (5.3, 4.35), (2.7, 1.2))             # A -> eta
    edge(ax, (7.3, 4.35), (2.7, 1.2))             # D -> eta
    edge(ax, (2.7, 1.2), (4.9, 1.2))              # eta -> mu
    edge(ax, (4.9, 1.2), (7.3, 1.2))              # mu -> y
    edge(ax, (10.7, 3.25), (7.3, 1.2))            # phi -> y

    ly, lx = -1.75, 0.35
    ax.add_patch(Circle((lx, ly), 0.13, facecolor="white", edgecolor=INK, lw=1.3))
    ax.text(lx + 0.26, ly, "stochastic", fontsize=9.4, va="center")
    ax.add_patch(Circle((lx + 1.65, ly), 0.13, facecolor="white", edgecolor=INK,
                        lw=1.3, linestyle=(0, (3, 1.8))))
    ax.text(lx + 1.91, ly, "deterministic", fontsize=9.4, va="center")
    ax.add_patch(Circle((lx + 3.6, ly), 0.13, facecolor=OBS, edgecolor=INK, lw=1.3))
    ax.text(lx + 3.86, ly, "observed", fontsize=9.4, va="center")
    ax.add_patch(Rectangle((lx + 5.1 - 0.09, ly - 0.09), 0.18, 0.18, facecolor=FLR))
    ax.text(lx + 5.36, ly, "fixed constant", fontsize=9.4, va="center")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ================= Page 3 : human prior =================
    fig, ax = canvas(13.0, 13.2, (0, 11.6), (0.3, 12.1))
    ax.text(0.05, 12.0, "Human prior", fontsize=16,
            fontweight="bold", ha="left", va="top")

    # generative rule as a graph
    plate(ax, 0.5, 8.3, 11.3, 11.5, "axes  k = 1…K")
    zsn_node(ax, 1.7, 10.15)
    plate(ax, 3.15, 8.7, 5.45, 11.05, "root tiers  r", fill="white")
    det(ax, 4.3, 10.15, r"$\theta_{rk}$", color=HUM)
    plate(ax, 5.95, 8.7, 8.25, 11.05, "tier increments  t", fill="white")
    stoch(ax, 7.1, 9.7, r"$\delta_{tk}$", r"$\mathrm{HalfNormal}(1)$",
          below=True)
    det(ax, 10.0, 10.0, r"$\theta_{tk}$", color=HUM)
    ax.text(10.0, 10.55, "non-root tiers  t", ha="center", va="bottom",
            fontsize=9.2, color=MUT)
    ax.text(10.0, 9.3,
            r"$\theta_{tk}=\max_{p\,\in\,\mathrm{par}(t)}\theta_{pk}+\delta_{tk}$",
            ha="center", va="center", fontsize=10.5)
    edge(ax, (1.7, 10.15), (4.3, 10.15))
    edge(ax, (4.3, 10.15), (10.0, 10.0), rad=-0.08)  # bows over delta_ek
    edge(ax, (7.1, 9.7), (10.0, 10.0))

    # the tier forest (instantiation)
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

    # HUMAN_ORDER_MERGED: the two merge points where the High School branch
    # joins the adult spine. The child dominates ALL its parents:
    # theta = max(parents) + delta, so these carry no second delta.
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

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

    # ============ Page 4 : model-family prior, Brownian steps ============
    fig, ax = canvas(13.6, 12.2, (0, 13.4), (0.75, 12.6))
    ax.text(0.05, 12.5, "Model-family prior · time-based steps", fontsize=16,
            fontweight="bold", ha="left", va="top")

    # generative rule as a graph. tau_o is ONE scalar for the whole model, so it
    # sits outside the axes plate.
    stoch(ax, 11.4, 11.75, r"$\tau_o$", r"$\mathrm{HalfNormal}(0.25)$")
    plate(ax, 0.4, 6.05, 13.0, 11.05, "axes  k = 1…K")
    stoch(ax, 3.2, 10.35, r"$r_k$", r"$\mathrm{HalfNormal}(1)$")
    stoch(ax, 7.4, 10.35, r"$s_k$", r"$\mathrm{HalfNormal}(2)$")

    zsn_node(ax, 1.5, 7.9)

    plate(ax, 2.65, 6.3, 9.4, 9.0, "families  c", fill="white")
    det(ax, 3.35, 7.1, r"$\psi_{1k}$", color=FAM)
    ax.text(3.35, 6.65, "founder", ha="center", va="top", fontsize=9,
            color=MUT)

    plate(ax, 4.25, 6.55, 9.25, 8.75, "releases  j ≥ 2", fill="white")
    stoch(ax, 7.5, 8.15, r"$\Delta_{jk}$",
          r"$\mathcal{N}(r_k\Delta t_j,\;s_k^2\,\Delta t_j)$", below=True)
    det(ax, 5.5, 7.1, r"$\psi_{jk}$", color=FAM)
    ax.text(6.8, 6.78, r"$\psi_{jk}=\psi_{j-1,k}+\Delta_{jk}$", ha="center",
            va="center", fontsize=10)

    plate(ax, 9.7, 6.55, 11.5, 9.4, "variant groups  g", fill="white")
    stoch(ax, 10.6, 8.2, r"$o_{gk}$", r"$\mathcal{N}(0,\;\tau_o^2)$",
          dist_bg="white")

    det(ax, 12.3, 7.1, r"$\theta_{mk}$", color=FAM, fs=12)
    ax.text(12.25, 6.5, r"$=\psi_{jk}+o_{gk}$", ha="center", va="center",
            fontsize=9.5)

    edge(ax, (3.2, 10.35), (7.5, 8.15))            # r -> Delta
    edge(ax, (7.4, 10.35), (7.5, 8.15))            # s -> Delta
    edge(ax, (1.5, 7.9), (3.35, 7.1))              # ZSN -> founder
    edge(ax, (3.35, 7.1), (5.5, 7.1))              # founder -> psi_j
    edge(ax, (7.5, 8.15), (5.5, 7.1))              # Delta -> psi_j
    edge(ax, (11.4, 11.75), (10.6, 8.2))           # tau_o -> o
    edge(ax, (5.5, 7.1), (12.3, 7.1), color=FAM)   # psi -> theta
    edge(ax, (10.6, 8.2), (12.3, 7.1))             # o -> theta

    # the release chain, drawn on a time axis (instantiation)
    plate(ax, 0.4, 0.95, 13.0, 5.3, "axes  k = 1…K")
    YR = 7.0            # x units per year
    RATE = 0.8          # y units per year (the drift slope r_ck)
    nodes = [(r"$\psi_1$", 0.0), (r"$\psi_2$", 0.25), (r"$\psi_3$", 1.45)]
    X0, Y0 = 1.3, 3.0
    NB = {}
    for lab, t in nodes:
        x, y = X0 + YR * t, Y0 + RATE * t
        NB[lab] = (x, y) + tbox(ax, x, y, lab, FAM, FAMF, fs=11, h=0.56, w=0.9)

    # step sd grows as sqrt(dt): drawn as a band through the arriving release
    for lab, dt in [(r"$\psi_2$", 0.25), (r"$\psi_3$", 1.20)]:
        x, y, w, h = NB[lab]
        hh = 0.8 * np.sqrt(dt)
        ax.add_patch(Rectangle((x - 0.17, y - hh), 0.34, 2 * hh,
                               facecolor=FAM, alpha=0.18, edgecolor="none",
                               zorder=1))
    x3, y3 = NB[r"$\psi_3$"][:2]
    ax.text(x3 + 0.4, y3 + 0.75, r"$s_k\sqrt{\Delta t_j}$", ha="left",
            va="center", fontsize=9.4, color=FAM)

    steps = [(r"$\psi_1$", r"$\psi_2$", r"$+\,r_k\Delta t_2$", "right"),
             (r"$\psi_2$", r"$\psi_3$", r"$+\,r_k\Delta t_3$", "center")]
    for a, b, dlab, ha in steps:
        xa, ya, wa, _ = NB[a]
        xb, yb, wb, _ = NB[b]
        p, q = (xa + wa / 2, ya), (xb - wb / 2, yb)
        edge(ax, p, q, color=FAM, rA=0.0, rB=0.06, lw=1.25)
        # the short step has no room over its own arrow: label it left of the
        # arriving box instead of over the midpoint
        if ha == "right":
            lx, ly = q[0] - 0.05, max(p[1], q[1]) + 0.16
        else:
            lx, ly = (p[0] + q[0]) / 2, (p[1] + q[1]) / 2 + 0.14
        ax.text(lx, ly, dlab, ha=ha, va="bottom", fontsize=9.4, color=FAM)

    # time axis
    ax.plot([0.9, 12.7], [2.0, 2.0], color=MUT, lw=1.0, zorder=1)
    for lab, t in nodes:
        x, y, w, h = NB[lab]
        ax.plot([x, x], [1.9, 2.1], color=MUT, lw=1.0, zorder=1)
        ax.plot([x, x], [2.1, y - h / 2], color=PLE, lw=0.9, ls=(0, (2, 2)),
                zorder=1)
        ax.text(x, 1.72, f"{t:.2f} yr" if t else "0 yr", ha="center", va="top",
                fontsize=8.8, color=MUT)
    for xm, dlab in [(2.18, r"$\Delta t_2=0.25$ yr"),
                     (7.25, r"$\Delta t_3=1.20$ yr")]:
        ax.text(xm, 1.32, dlab, ha="center", va="top", fontsize=9, color=MUT)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

print("wrote", out)
