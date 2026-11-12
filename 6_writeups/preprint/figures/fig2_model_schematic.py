"""Figure 2: plate diagram of the multi-axis model (the flagship fit).

Flagship flags (trace attr `mirt_spec`): K = 4, non-negative "normal" loadings,
fixed chance floors, estimated ceilings, Beta-binomial on item-scored benchmarks,
skew-normal ability block, pooled noise with a floor, lineage prior with Brownian
steps and effort means, models only. Every prior label is read from
2_model/multiaxis_eci/models/mirt.py (build_mirt_model) and config.py.

Visual language of docs/make_prior_graph.py page 2. Its helpers are copied, not
imported: that script parses argv and draws all four pages at import.
"""
import sys
from pathlib import Path

import matplotlib
import matplotlib.patches
import matplotlib.text
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "2_model"))
from multiaxis_eci.config import (  # noqa: E402
    PRIOR_CEILING_GAP, PRIOR_EFFORT_MEAN, PRIOR_LINEAGE_DELTA_BM,
    PRIOR_LINEAGE_DRIFT_BM, PRIOR_LINEAGE_OFFSET, PRIOR_SIGMA_B_POOLED,
    PRIOR_TAU_ALPHA, PRIOR_TAU_CD, PRIOR_THETA_SKEW_ALPHA, SIGMA_B_FLOOR)

INK = "#262626"   # the preprint style guide's INK
MUT = "#5a6478"
PLF = "#f2f4f8"   # plate fill
PLE = "#d8dde7"   # plate edge
OBS = "#dfe8f3"   # observed fill
FLR = "#C4568C"   # fixed constant
FLRF = "#faeef5"
FAM = "#C77800"   # lineage prior
FAMF = "#fbf1de"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.cal": "Arial",          # the normal family prints as a plain N
    "mathtext.fallback": "stixsans",  # sum and norm glyphs Arial lacks
    "text.color": INK,
})

R = 0.36
FS_NODE = 12.5
FS_LAB = 9.6
FS_PLATE = 9.0
FS_EQ = 11.5


def _lnum(x):
    """log-scale constant as printed: log 0.05, log 3."""
    return f"\\log {np.exp(x):g}"


def stoch(ax, x, y, math, dist=None, r=R, below=False, bg=None, color=INK):
    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=color, lw=1.5, zorder=3))
    ax.text(x, y, math, ha="center", va="center", fontsize=FS_NODE, zorder=4)
    if dist:
        dy, va = (-r - 0.1, "top") if below else (r + 0.1, "bottom")
        bbox = (dict(boxstyle="square,pad=0.08", facecolor=bg, edgecolor="none")
                if bg else None)
        ax.text(x, y + dy, dist, ha="center", va=va, fontsize=FS_LAB, color=MUT,
                zorder=4, bbox=bbox, linespacing=1.25)


def det(ax, x, y, math, r=R, color=INK):
    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor=color, lw=1.5,
                        linestyle=(0, (4, 2.4)), zorder=3))
    ax.text(x, y, math, ha="center", va="center", fontsize=FS_NODE, zorder=4)


def obsn(ax, x, y, math, r=R):
    ax.add_patch(Circle((x, y), r, facecolor=OBS, edgecolor=INK, lw=1.5, zorder=3))
    ax.text(x, y, math, ha="center", va="center", fontsize=FS_NODE, zorder=4)


def const(ax, x, y, math, h=0.56):
    """Fixed constant read from file: a rounded square in the floor colour."""
    ax.add_patch(FancyBboxPatch((x - h / 2, y - h / 2), h, h,
                                boxstyle="round,pad=0,rounding_size=0.08",
                                facecolor=FLRF, edgecolor=FLR, lw=1.5, zorder=3))
    ax.text(x, y, math, ha="center", va="center", fontsize=FS_NODE, zorder=4)


def plate(ax, x0, y0, x1, y1, label="", fill=PLF, pos="br", edgec=PLE, tc=MUT):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=0.14",
                                facecolor=fill, edgecolor=edgec, lw=1.1, zorder=0.5))
    if label:
        lx, ha = (x1 - 0.12, "right") if pos[1] == "r" else (x0 + 0.12, "left")
        ly, va = (y0 + 0.08, "bottom") if pos[0] == "b" else (y1 - 0.08, "top")
        ax.text(lx, ly, label, ha=ha, va=va, fontsize=FS_PLATE,
                color=tc, style="italic", zorder=2)


def edge(ax, p, q, color=INK, rA=R, rB=R, ls="-", lw=1.2, rad=0.0):
    p, q = np.asarray(p, float), np.asarray(q, float)
    u = (q - p) / np.linalg.norm(q - p)
    a, b = p + u * rA, q - u * (rB + 0.05)
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11,
                                 color=color, lw=lw, linestyle=ls, zorder=2,
                                 connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0))


def note(ax, x, y, s, ha="center", va="center", color=INK, fs=FS_LAB, bg=None):
    bbox = dict(boxstyle="square,pad=0.08", facecolor=bg, edgecolor="none") if bg else None
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=color, zorder=4, bbox=bbox,
            linespacing=1.3)


def two(name, args):
    """A prior label on two lines, the family over its arguments."""
    return f"${name}$\n$({args})$"


# 1 unit = 0.6 in: printed at a 6.5 in text width the labels read at ~6.5 pt.
X0, X1, Y0, Y1 = 0.0, 16.8, -2.75, 10.9
U = 0.6
fig, ax = plt.subplots(figsize=((X1 - X0) * U, (Y1 - Y0) * U))
ax.set_xlim(X0, X1)
ax.set_ylim(Y0, Y1)
ax.set_aspect("equal")
ax.axis("off")

# ---------------- ability side (left) ----------------
# effort-offset hyperparameters: not per-axis cells, so above the axes plate
plate(ax, 3.2, 8.62, 5.92, 10.3, r"effort levels  $\ell$", fill="white", pos="bl")
stoch(ax, 5.2, 9.45, r"$m_\ell$", rf"$\mathcal{{N}}(0,\,{PRIOR_EFFORT_MEAN:g}^2)$")
stoch(ax, 6.9, 9.3, r"$\tau_o$", two(r"\mathrm{HalfNormal}", f"{PRIOR_LINEAGE_OFFSET:g}"))

plate(ax, 0.2, 2.15, 8.55, 8.5, "axes  k = 1…K", pos="bl")
stoch(ax, 7.85, 7.3, r"$v_k$", "$\\mathcal{N}(0,\\,1)$\n$u=v/\\Vert v\\Vert$")
stoch(ax, 1.5, 7.3, r"$\alpha_k$", rf"$\mathcal{{N}}(0,\,{PRIOR_THETA_SKEW_ALPHA:g}^2)$",
      bg=PLF)
stoch(ax, 3.0, 7.3, r"$r_k$", two(r"\mathrm{HalfNormal}", f"{PRIOR_LINEAGE_DRIFT_BM:g}"),
      bg=PLF)
stoch(ax, 4.85, 7.3, r"$s_k$", two(r"\mathrm{HalfNormal}", f"{PRIOR_LINEAGE_DELTA_BM:g}"),
      bg=PLF)

# exchangeable skew-normal block: models outside the chains and the chain founders
plate(ax, 0.35, 3.75, 2.65, 6.8, "", fill="white")
stoch(ax, 1.5, 6.2, r"$\theta^{\mathrm{sn}}_{ik}$",
      "$\\mathrm{SkewNormal}$\n$(\\xi_k,\\,\\omega_k,\\,\\alpha_k)$", below=True, bg="white")
note(ax, 1.5, 4.85, "mean 0, sd 1", fs=FS_PLATE, color=MUT)
note(ax, 1.5, 4.22, "unchained\nmodels, chain\nfounders", fs=FS_PLATE, color=MUT)

# lineage prior: founder level, Brownian step over the release gap, variant offset
plate(ax, 2.8, 3.75, 8.4, 6.8, "lineage prior", fill=FAMF, edgec=FAM, tc=FAM, pos="bl")
det(ax, 3.4, 5.45, r"$\psi_{1k}$", color=FAM)
note(ax, 3.4, 4.88, "founder", fs=FS_PLATE, color=MUT)
plate(ax, 3.95, 3.9, 6.55, 6.65, "releases  j", fill="white", pos="tr")
stoch(ax, 4.75, 6.0, r"$\Delta_{jk}$")
note(ax, 5.17, 5.38, "$\\mathcal{N}(r_k\\Delta t_j,$\n$s_k^2\\,\\Delta t_j)$", ha="left",
     color=MUT)
det(ax, 4.75, 4.75, r"$\psi_{jk}$", color=FAM)
plate(ax, 6.7, 3.9, 8.3, 6.65, "groups  g", fill="white")
stoch(ax, 7.42, 5.95, r"$o_{gk}$")
note(ax, 6.98, 5.0, "$\\mathcal{N}(m_\\ell u_k,$\n$\\tau_o^2)$", ha="left", color=MUT)

XT = 4.9
det(ax, XT, 3.05, r"$\theta_{mk}$")
note(ax, 0.45, 3.2, r"$\theta_{mk}=\theta^{\mathrm{sn}}_{mk}$  unchained", ha="left", color=MUT)
note(ax, 0.45, 2.74, r"$\theta_{mk}=\psi_{j(m)k}+o_{g(m)k}$  chained", ha="left", color=MUT)

# ---------------- benchmark side (right) ----------------
XA, XD, XS, XN, XC, XF = 9.95, 12.05, 15.0, 13.75, 12.4, 11.0
stoch(ax, XA, 9.3, r"$\tau_A$",
      two(r"\mathrm{LogNormal}",
          f"{_lnum(PRIOR_TAU_ALPHA['mu'])},\\,{PRIOR_TAU_ALPHA['sigma']:g}"))
stoch(ax, XD, 9.3, r"$\tau_{CD}$",
      two(r"\mathrm{LogNormal}", f"{_lnum(PRIOR_TAU_CD['mu'])},\\,{PRIOR_TAU_CD['sigma']:g}"))
P = PRIOR_SIGMA_B_POOLED
stoch(ax, 13.95, 9.3, r"$\mu_\sigma$",
      two(r"\mathcal{N}", f"{_lnum(P['mu_loc'])},\\,{P['mu_sd']:g}^2"))
stoch(ax, 15.9, 9.3, r"$\tau_\sigma$", two(r"\mathrm{HalfNormal}", f"{P['tau_sd']:g}"))

plate(ax, 8.75, 2.15, 16.6, 8.5, "benchmarks  b")
plate(ax, 8.9, 5.55, 11.05, 8.3, "axes  k", fill="white", pos="tr")
stoch(ax, XA, 7.35, r"$A_{bk}$", two(r"\mathrm{HalfNormal}", r"\tau_A"), below=True,
      bg="white")
stoch(ax, XD, 7.35, r"$D_b$", r"$\mathcal{N}(0,\;\tau_{CD}^2)$", below=True, bg=PLF)
stoch(ax, XS, 7.35, r"$\sigma_b$",
      f"$\\log(\\sigma_b-{SIGMA_B_FLOOR:g})$\n$\\sim\\mathcal{{N}}(\\mu_\\sigma,\\,\\tau_\\sigma^2)$",
      below=True, bg=PLF)

const(ax, XF, 4.3, r"$c_b$")
note(ax, XF, 3.68, "chance floor", color=FLR, fs=FS_PLATE)
stoch(ax, XC, 4.3, r"$d_b$",
      rf"$1-d_b\sim{PRIOR_CEILING_GAP['scale']:g}\,\times$" "\n"
      rf"$\mathrm{{Beta}}({PRIOR_CEILING_GAP['alpha']:g},\,{PRIOR_CEILING_GAP['beta']:g})$"
      "\n" r"$d_b\geq$ best score", bg=PLF)
const(ax, XN, 4.3, r"$N_b$")
note(ax, XN, 3.68, "item count", color=FLR, fs=FS_PLATE)
det(ax, XS, 4.3, r"$\phi_b$")
note(ax, XS + 0.48, 4.3, r"$=\frac{1}{4\sigma_b^2}-1$", ha="left", color=MUT)

# ---------------- observations ----------------
XE, XM, XY = 8.6, 11.7, 13.95
plate(ax, 7.6, 0.2, 15.6, 1.95, "observations  n = 1…N")
det(ax, XE, 1.2, r"$\eta_n$")
det(ax, XM, 1.2, r"$\mu_n$")
obsn(ax, XY, 1.2, r"$y_n$")

# ---------------- edges ----------------
edge(ax, (5.2, 9.45), (7.42, 5.95))                 # m_l -> o
edge(ax, (7.85, 7.3), (7.42, 5.95))                 # v -> o (through u)
edge(ax, (6.9, 9.3), (7.42, 5.95))                 # tau_o -> o
edge(ax, (1.5, 7.3), (1.5, 6.2))                  # alpha -> skew block
edge(ax, (3.0, 7.3), (4.75, 6.0))                  # r -> Delta
edge(ax, (4.85, 7.3), (4.75, 6.0))                  # s -> Delta
edge(ax, (1.5, 6.2), (3.4, 5.45), color=FAM)       # founder rows
edge(ax, (3.4, 5.45), (4.75, 4.75), color=FAM)      # psi_1 -> psi_j
edge(ax, (4.75, 6.0), (4.75, 4.75))                 # Delta -> psi_j
edge(ax, (2.45, 3.75), (XT, 3.05), rA=0.0)           # unchained rows -> theta
edge(ax, (4.75, 4.75), (XT, 3.05), color=FAM)       # psi -> theta
edge(ax, (7.42, 5.95), (XT, 3.05), color=FAM)       # o -> theta
edge(ax, (XT, 3.05), (XE, 1.2))                     # theta -> eta

edge(ax, (XA, 9.3), (XA, 7.35))                     # tau_A -> A
edge(ax, (XD, 9.3), (XD, 7.35))                     # tau_CD -> D
edge(ax, (13.95, 9.3), (XS, 7.35))                  # mu_s -> sigma
edge(ax, (15.9, 9.3), (XS, 7.35))                   # tau_s -> sigma
edge(ax, (XS, 7.35), (XS, 4.3), rA=R + 0.75)        # sigma -> phi (below its label)
edge(ax, (XA, 7.35), (XE, 1.2), rA=R + 0.45)        # A -> eta
edge(ax, (XD, 7.35), (XE, 1.2), rA=R + 0.35)        # D -> eta
edge(ax, (XF, 4.3), (XM, 1.2), color=FLR, rA=0.95)  # c -> mu
edge(ax, (XC, 4.3), (XM, 1.2))                      # d -> mu
edge(ax, (XN, 4.3), (XY, 1.2), color=FLR, rA=0.95)  # N -> y
edge(ax, (XS, 4.3), (XY, 1.2))             # phi -> y
edge(ax, (XE, 1.2), (XM, 1.2))                      # eta -> mu
edge(ax, (XM, 1.2), (XY, 1.2))                      # mu -> y

# ---------------- display equations ----------------
ex, nx = 1.0, 9.3
note(ax, ex, -0.25, r"$\eta_n=\sum_k A_{b_n k}\,\theta_{m_n k}-D_{b_n}$", ha="left", fs=FS_EQ)
note(ax, ex, -0.85, r"$\mu_n=c_{b_n}+(d_{b_n}-c_{b_n})\,\sigma(\eta_n)$", ha="left", fs=FS_EQ)
note(ax, ex, -1.45,
     r"$y_n\sim\mathrm{Beta}\left(\mu_n\phi_{b_n},\;(1-\mu_n)\,\phi_{b_n}\right)$",
     ha="left", fs=FS_EQ)
note(ax, nx, -1.45, "continuous score", ha="left", color=MUT)
note(ax, ex, -2.05,
     r"$N_{b_n}y_n\sim\mathrm{BetaBinomial}\left(N_{b_n},\;\mu_n\phi_{b_n},\;(1-\mu_n)\,\phi_{b_n}\right)$",
     ha="left", fs=FS_EQ)
note(ax, nx, -2.05, "score on $N_b$ items", ha="left", color=MUT)

# ---------------- legend ----------------
lx, ly = 1.0, 1.45
items = [("stochastic", dict(facecolor="white", edgecolor=INK)),
         ("deterministic", dict(facecolor="white", edgecolor=INK, linestyle=(0, (3, 1.8)))),
         ("observed", dict(facecolor=OBS, edgecolor=INK))]
for i, (lab, kw) in enumerate(items):
    x = lx + (i % 2) * 2.4
    y = ly - (i // 2) * 0.6
    ax.add_patch(Circle((x, y), 0.13, lw=1.3, **kw))
    ax.text(x + 0.26, y, lab, fontsize=FS_LAB, va="center")
x, y = lx + 2.4, ly - 0.6
ax.add_patch(FancyBboxPatch((x - 0.13, y - 0.13), 0.26, 0.26,
                            boxstyle="round,pad=0,rounding_size=0.05",
                            facecolor=FLRF, edgecolor=FLR, lw=1.3))
ax.text(x + 0.26, y, "fixed constant", fontsize=FS_LAB, va="center")

# Final size: the tight bounding box is scaled to 160 mm wide, every font size,
# line width and arrow head with it, so the PDF is inserted at 100 %.
WIDTH_IN = 160 / 25.4
bb = fig.get_tightbbox(fig.canvas.get_renderer())
f = WIDTH_IN / bb.width
for t in fig.findobj(matplotlib.text.Text):
    t.set_fontsize(t.get_fontsize() * f)
for p in fig.findobj(matplotlib.patches.Patch):
    p.set_linewidth(p.get_linewidth() * f)
    if isinstance(p, FancyArrowPatch):
        p.set_mutation_scale(p.get_mutation_scale() * f)
w, h = fig.get_size_inches()
fig.set_size_inches(w * f, h * f)

out = Path(__file__).resolve().parent / "fig2_model_schematic"
fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0)
fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight", pad_inches=0)
print("wrote", out.with_suffix(".pdf"), out.with_suffix(".png"))
