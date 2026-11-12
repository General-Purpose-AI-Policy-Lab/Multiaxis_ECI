"""The flagship's likelihood page as it stands on 2026-11-03, in the style of docs/prior_graph.pdf page 2.

Page 1: the model graph, the nodes changed since docs/prior_graph.pdf drawn in red.
Page 2: every symbol, defined in words.
Page 3: what the beta-binomial does to the spread of an observed score.

    python 6_writeups/preprint/step0/prior_graph_now/make_pages.py
"""
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle
from scipy import stats

INK = "#26324B"
MUT = "#5a6478"
PLF = "#f2f4f8"
PLE = "#d8dde7"
OBS = "#dfe8f3"
FLR = "#C4568C"
FAM = "#C77800"
NEW = "#D85A30"   # changed since docs/prior_graph.pdf
NEWF = "#fbeee8"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "text.color": INK,
})
R = 0.36
HERE = Path(__file__).resolve().parent


def stoch(ax, x, y, math, dist=None, color=INK, below=False):
    ax.add_patch(Circle((x, y), R, facecolor="white", edgecolor=color, lw=1.6, zorder=3))
    ax.text(x, y, math, ha="center", va="center", fontsize=13, zorder=4)
    if dist:
        dy, va = (-R - 0.14, "top") if below else (R + 0.14, "bottom")
        ax.text(x, y + dy, dist, ha="center", va=va, fontsize=9.2, color=MUT, zorder=4,
                bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none"))


def det(ax, x, y, math, color=INK):
    ax.add_patch(Circle((x, y), R, facecolor="white", edgecolor=color, lw=1.6,
                        linestyle=(0, (4, 2.4)), zorder=3))
    ax.text(x, y, math, ha="center", va="center", fontsize=13, zorder=4)


def obsn(ax, x, y, math, color=INK):
    ax.add_patch(Circle((x, y), R, facecolor=OBS, edgecolor=color, lw=1.6, zorder=3))
    ax.text(x, y, math, ha="center", va="center", fontsize=13, zorder=4)


def const(ax, x, y, math, color=FLR):
    s = 0.6
    ax.add_patch(Rectangle((x - s / 2, y - s / 2), s, s, facecolor="white", edgecolor=color,
                           lw=1.6, zorder=3))
    ax.text(x, y, math, ha="center", va="center", fontsize=13, zorder=4)


def plate(ax, x0, y0, x1, y1, label="", fill=PLF, pos="br"):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle="round,pad=0,rounding_size=0.14",
                                facecolor=fill, edgecolor=PLE, lw=1.1, zorder=0.5))
    if label:
        lx, ha = (x1 - 0.14, "right") if pos == "br" else (x0 + 0.14, "left")
        ax.text(lx, y0 + 0.1, label, ha=ha, va="bottom", fontsize=9.2, color=MUT, style="italic",
                zorder=2)


def edge(ax, p, q, color=INK, rA=R, rB=R):
    p, q = np.asarray(p, float), np.asarray(q, float)
    u = (q - p) / np.linalg.norm(q - p)
    a, b = p + u * rA, q - u * (rB + 0.05)
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, color=color, lw=1.2,
                                 zorder=2, shrinkA=0, shrinkB=0))


def canvas(w, h, xlim, ylim):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def page_graph():
    fig, ax = canvas(14.4, 11.6, (0, 14.4), (-3.9, 7.7))
    ax.text(0.05, 7.6, "Beta-MIRT model (K = 4), flagship on 2026-11-03", fontsize=16,
            fontweight="bold", ha="left", va="top")

    # hyperpriors
    stoch(ax, 2.05, 6.35, r"$\alpha_k$", r"$\mathcal{N}(0,\;4^2)$", color=NEW)
    stoch(ax, 5.3, 6.35, r"$\tau_A$", r"$\mathrm{LogNormal}(\log 0.5,\;0.5)$")
    stoch(ax, 7.3, 6.35, r"$\tau_{CD}$", r"$\mathrm{LogNormal}(\log 3,\;1)$")
    stoch(ax, 10.6, 6.35, r"$\mu_\sigma$", r"$\mathcal{N}(\log 0.05,\;0.5^2)$")
    stoch(ax, 12.4, 6.35, r"$\tau_\sigma$", r"$\mathrm{HalfNormal}(0.5)$")

    # model plate
    plate(ax, 0.6, 2.5, 3.5, 5.5)
    ax.text(0.75, 5.38, "models  m · axes  k", ha="left", va="top", fontsize=8.6, color=MUT,
            style="italic", zorder=2)
    for i, (lab, ec, fc) in enumerate([("SkewNormal, mean 0, sd 1", NEW, NEWF),
                                       ("+  model-family prior", FAM, "#fbf1de")]):
        yb = 4.75 - i * 0.5
        ax.add_patch(FancyBboxPatch((0.8, yb - 0.2), 2.6, 0.42, boxstyle="round,pad=0,rounding_size=0.08",
                                    facecolor=fc, edgecolor=ec, lw=1.1, zorder=2))
        ax.text(2.1, yb + 0.01, lab, ha="center", va="center", fontsize=8.8, color=ec, zorder=3)
    stoch(ax, 2.1, 3.05, r"$\theta_{mk}$")
    ax.text(2.1, 3.85, "humans: not in the fit", ha="center", va="center", fontsize=8.4, color=NEW)

    # benchmark plate
    plate(ax, 3.9, 2.5, 13.6, 5.5, "benchmarks  b", pos="br")
    plate(ax, 4.2, 3.2, 6.4, 5.25, "axes  k", fill="white")
    stoch(ax, 5.3, 4.45, r"$A_{bk}$", r"$\tau_A\cdot\mathrm{HalfNormal}(1)$", below=True)
    stoch(ax, 7.3, 4.45, r"$D_b$", r"$\tau_{CD}\cdot\mathcal{N}(0,1)$", below=True)
    const(ax, 7.3, 3.0, r"$c_b$")
    stoch(ax, 9.0, 4.45, r"$g_b$", "Beta(1.43, 4.67)\n" + r"$g_b\leq(1-\mathrm{best}_b)/0.25$",
          color=NEW)
    det(ax, 9.0, 3.0, r"$d_b$", color=NEW)
    stoch(ax, 11.0, 4.45, r"$\sigma_b$", r"$0.005+\exp(\mu_\sigma+\tau_\sigma z_b)$", color=NEW)
    det(ax, 11.0, 3.0, r"$\phi_b$")
    const(ax, 12.9, 3.3, r"$N_b$", color=NEW)

    # observation plate
    plate(ax, 1.45, 0.1, 12.0, 2.1, "observations  n = (model, benchmark)")
    det(ax, 2.9, 1.2, r"$\eta_n$")
    det(ax, 5.3, 1.2, r"$\mu_n$")
    stoch(ax, 8.0, 1.2, r"$p_n$", color=NEW)
    obsn(ax, 10.6, 1.2, r"$k_n$", color=NEW)

    for p, q in [((2.05, 6.35), (2.1, 4.97)), ((5.3, 6.35), (5.3, 4.45)), ((7.3, 6.35), (7.3, 4.45)),
                 ((10.6, 6.35), (11.0, 4.45)), ((12.4, 6.35), (11.0, 4.45)),
                 ((9.0, 4.45), (9.0, 3.0)), ((11.0, 4.45), (11.0, 3.0))]:
        edge(ax, p, q, rA=R if p[1] > 5.9 and p[0] != 2.05 else R, rB=R if q[1] != 4.97 else 0.0)
    edge(ax, (2.1, 3.05), (2.9, 1.2))
    edge(ax, (5.3, 4.45), (2.9, 1.2))
    edge(ax, (7.3, 4.45), (2.9, 1.2))
    edge(ax, (7.3, 3.0), (5.3, 1.2), rA=0.3)
    edge(ax, (9.0, 3.0), (5.3, 1.2), color=NEW)
    edge(ax, (2.9, 1.2), (5.3, 1.2))
    edge(ax, (5.3, 1.2), (8.0, 1.2), color=NEW)
    edge(ax, (11.0, 3.0), (8.0, 1.2), color=NEW)
    edge(ax, (8.0, 1.2), (10.6, 1.2), color=NEW)
    edge(ax, (12.9, 3.3), (10.6, 1.2), rA=0.3, color=NEW)

    eq = [
        (r"$\eta_n=\sum_k A_{b_n k}\,\theta_{m_n k}-D_{b_n}$", INK),
        (r"$\mu_n=c_{b_n}+(d_{b_n}-c_{b_n})\,\mathrm{logistic}(\eta_n),\qquad d_b=1-0.25\,g_b$", NEW),
        (r"$p_n\sim\mathrm{Beta}\left(\mu_n\phi_{b_n},\;(1-\mu_n)\phi_{b_n}\right),\qquad"
         r"\phi_b=\frac{1}{4\sigma_b^2}-1$", NEW),
        (r"$k_n\mid p_n\sim\mathrm{Binomial}(N_{b_n},\;p_n)$" + "    so  " +
         r"$k_n\sim\mathrm{BetaBinomial}(N_{b_n},\;\mu_n\phi_{b_n},\;(1-\mu_n)\phi_{b_n})$", NEW),
    ]
    for i, (s, c) in enumerate(eq):
        ax.text(6.7, -0.45 - 0.55 * i, s, ha="center", va="center", fontsize=11.5, color=c)
    ax.text(6.7, -2.75, "82 / 107 benchmarks.  The other 25 have no question count: there the score "
            r"$y_n$ itself is drawn,  $y_n\sim\mathrm{Beta}(\mu_n\phi_{b_n},\,(1-\mu_n)\phi_{b_n})$,"
            "  zeros set to 0.005.", ha="center", va="center", fontsize=9.6, color=MUT)

    lx, ly = 1.3, -3.5
    ax.add_patch(Circle((lx, ly), 0.13, facecolor="white", edgecolor=INK, lw=1.3))
    ax.text(lx + 0.26, ly, "random", fontsize=9.4, va="center")
    ax.add_patch(Circle((lx + 1.6, ly), 0.13, facecolor="white", edgecolor=INK, lw=1.3,
                        linestyle=(0, (3, 2))))
    ax.text(lx + 1.86, ly, "computed from its parents", fontsize=9.4, va="center")
    ax.add_patch(Circle((lx + 4.6, ly), 0.13, facecolor=OBS, edgecolor=INK, lw=1.3))
    ax.text(lx + 4.86, ly, "observed", fontsize=9.4, va="center")
    ax.add_patch(Rectangle((lx + 6.2 - 0.12, ly - 0.12), 0.24, 0.24, facecolor="white",
                           edgecolor=FLR, lw=1.3))
    ax.text(lx + 6.46, ly, "fixed, read from a file", fontsize=9.4, va="center")
    ax.add_patch(Circle((lx + 9.2, ly), 0.13, facecolor="white", edgecolor=NEW, lw=1.6))
    ax.text(lx + 9.46, ly, "new or changed since docs/prior_graph.pdf", fontsize=9.4, va="center",
            color=NEW)
    return fig


GLOSSARY = [
    ("Indices", None),
    ("m", "a model (a test-taker), e.g. GPT-5 at medium effort"),
    ("b", "a benchmark, e.g. SimpleBench"),
    ("k", "an axis of ability, k = 1 ... 4"),
    ("n", "one observation: the score of model $m_n$ on benchmark $b_n$"),
    ("Data", None),
    ("$N_b$", "number of questions on benchmark b (from the item-count file); NEW"),
    ("$k_n$", "number of questions model $m_n$ answered right on $b_n$: score x $N_b$; NEW"),
    ("$y_n$", "the score itself, in [0, 1]; used only where $N_b$ is unknown (25 / 107)"),
    ("$c_b$", "chance floor: the score of random guessing (0.25 on 4-choice questions)"),
    ("$\\mathrm{best}_b$", "highest score anyone, model or human, reached on b"),
    ("Abilities and benchmarks", None),
    ("$\\theta_{mk}$", "ability of model m on axis k"),
    ("$\\alpha_k$", "skewness of axis k's abilities: 0 is a symmetric bell, < 0 a long left tail; NEW"),
    ("$A_{bk}$", "loading: how strongly benchmark b's score rises with axis k (>= 0)"),
    ("$D_b$", "difficulty of benchmark b"),
    ("$\\tau_A,\\ \\tau_{CD}$", "typical size of the loadings, and of the difficulties"),
    ("From ability to expected score", None),
    ("$\\eta_n$", "ability minus difficulty for this pair, on the logit scale"),
    ("$\\mathrm{logistic}(x)$", "$1/(1+e^{-x})$: turns a logit into a share in (0, 1)"),
    ("$g_b$", "how far the ceiling sits below 1, as a share of the largest gap allowed (0.25); NEW"),
    ("$d_b$", "ceiling: the best expected score a perfect model gets, $1 - 0.25\\,g_b$; NEW"),
    ("$\\mu_n$", "expected score of model $m_n$ on $b_n$: between $c_b$ and $d_b$"),
    ("Noise", None),
    ("$\\sigma_b$", "noise of benchmark b: the score's SD when $\\mu = 0.5$ and $N_b$ is large; floor 0.005 NEW"),
    ("$z_b$", "a standard normal draw, one per benchmark"),
    ("$\\mu_\\sigma,\\ \\tau_\\sigma$", "typical log-noise across benchmarks, and its spread"),
    ("$\\phi_b$", "precision: $1/(4\\sigma_b^2)-1$; large $\\phi$ means little scatter"),
    ("$p_n$", "the success rate model $m_n$ really has on $b_n$'s questions: $\\mu_n$ plus misfit; NEW"),
]


def page_glossary():
    fig = plt.figure(figsize=(11.5, 11.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.text(0.04, 0.975, "Symbols", fontsize=16, fontweight="bold", va="top")
    y = 0.93
    for sym, txt in GLOSSARY:
        if txt is None:
            y -= 0.01
            ax.text(0.04, y, sym, fontsize=11.5, fontweight="bold", color=MUT, va="center")
            y -= 0.033
            continue
        new = txt.endswith("NEW")
        body = txt[:-3].rstrip("; ") if new else txt
        ax.text(0.06, y, sym, fontsize=12, va="center", color=NEW if new else INK)
        ax.text(0.2, y, body, fontsize=11, va="center")
        if new:
            ax.text(0.97, y, "new", fontsize=9.5, va="center", ha="right", color=NEW)
        y -= 0.03
    return fig


def page_spread():
    mu, sigma = 0.80, 0.05
    phi = 1 / (4 * sigma ** 2) - 1
    a, b = mu * phi, (1 - mu) * phi
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for ax, N in zip(axes, (30, 3000)):
        k = np.arange(N + 1)
        y = k / N
        binom = stats.binom.pmf(k, N, mu) * N
        bb = stats.betabinom.pmf(k, N, a, b) * N
        grid = np.linspace(0.001, 0.999, 2000)
        if N == 30:
            w = 0.4 / N
            ax.bar(y - w / 2, binom, width=w, color="#9aa3b5", label="Binomial only (no misfit)")
            ax.bar(y + w / 2, bb, width=w, color="#2b6cb0", label="Beta-binomial (used)")
        else:
            ax.plot(y, binom, color="#9aa3b5", lw=2, label="Binomial only (no misfit)")
            ax.plot(y, bb, color="#2b6cb0", lw=2, label="Beta-binomial (used)")
        ax.plot(grid, stats.beta.pdf(grid, a, b), color="#d08a18", lw=1.6, ls="--",
                label="Beta (continuous benchmarks)")
        sd_bin = np.sqrt(mu * (1 - mu) / N)
        sd_bb = np.sqrt(mu * (1 - mu) * (N + phi) / (N * (1 + phi)))
        ax.set_title(f"{N} questions: SD {sd_bin:.3f} binomial, {sd_bb:.3f} beta-binomial",
                     fontsize=11)
        ax.set_xlim(0.55, 1.0)
        ax.set_ylim(bottom=0)
        ax.set_xlabel("observed score  k / N")
        ax.set_yticks([])
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0].set_ylabel("probability density")
    axes[0].legend(frameon=False, fontsize=9.5, loc="upper left")
    fig.suptitle(rf"Spread of one observed score, expected score $\mu$ = {mu}, noise $\sigma_b$ = {sigma}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    out = HERE / "prior_graph_now.pdf"
    with PdfPages(out) as pdf:
        for i, make in enumerate((page_graph, page_glossary, page_spread), start=1):
            fig = make()
            pdf.savefig(fig, bbox_inches="tight")
            fig.savefig(HERE / f"page{i}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
    print("wrote", out)
