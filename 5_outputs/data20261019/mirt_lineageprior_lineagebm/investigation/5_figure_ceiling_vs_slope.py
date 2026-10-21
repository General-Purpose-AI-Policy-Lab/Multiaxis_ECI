"""Why a ceiling comes out low: a plateau of the top scores, or the slope the mid-range scores ask for.

For two benchmarks, CyberGym (a low ceiling the slope pulls down) and WMDP Biology (a ceiling the top
scores pin), three panels each:
  1. the flagship without ceilings: each model's score against its ability along the benchmark's
     loading direction, u = A_b . theta_m / |A_b| (posterior median), and the fitted curve
     c + (1 - c) sigmoid(|A_b| u - D_b) (median and 80% band over the draws);
  2. the same with the --ceiling-noise fit, whose curve stops at d_b;
  3. the ceiling fit's draws of d_b against the slope |A_b|: the trade-off between them, the best
     observed score as a dashed line.
Points coloured by release date. Writes ceiling_vs_slope.png beside this script.
"""
import sys
from pathlib import Path

import arviz as az
import matplotlib
import numpy as np
import pandas as pd
import xarray as xr
from scipy.special import expit

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
GEN = HERE.parents[1]
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "2_model"))
from multiaxis_eci.analysis.fitspec import FitSpec, best_mode_chains  # noqa: E402

FITS = {"no ceilings (_noceiling)": GEN / "mirt_lineageprior_lineagebm_noceiling" / "trace_mirt_k4_lineageprior_lineagebm_noceiling.nc",
        "with ceilings (flagship)": GEN / "mirt_lineageprior_lineagebm"
        / "trace_mirt_k4_lineageprior_lineagebm.nc"}
BENCH = ["CyberGym", "WMDP Biology"]
THIN = 10

dates = pd.read_csv(ROOT / "0_input" / "all_scores_flat.csv", parse_dates=["release_date"]) \
    .groupby("model_version")["release_date"].min()


def load(trace):
    post = xr.open_dataset(trace, group="posterior")
    spec = FitSpec.from_trace(az.InferenceData(posterior=post.isel(draw=slice(0, 1))), trace)
    data, floor_c, _ = spec.load_data()
    ch = best_mode_chains(trace)
    p = post.isel(chain=ch, draw=slice(None, None, THIN))
    flat = lambda v: p[v].values.reshape(-1, *p[v].shape[2:])          # noqa: E731
    d = flat("ceiling_d") if "ceiling_d" in p else None
    return data, floor_c, flat("A"), flat("theta"), flat("D"), d


fits = {k: load(v) for k, v in FITS.items()}
fig, axes = plt.subplots(len(BENCH), 3, figsize=(16, 5.2 * len(BENCH)),
                         gridspec_kw=dict(hspace=0.45, wspace=0.28))
for row, bname in enumerate(BENCH):
    for col, (label, (data, floor_c, A, th, D, dceil)) in enumerate(fits.items()):
        bench = data.blookup.sort_values("benchmark_idx")["benchmark"].tolist()
        names = data.mlookup.sort_values("model_idx")["model"].tolist()
        b = bench.index(bname)
        c = floor_c[b] if floor_c is not None else 0.0
        a = A[:, b, :]                                              # (S, K)
        norm = np.linalg.norm(a, axis=1)                           # (S,)
        sel = np.where(data.bench_idx == b)[0]
        m = data.model_idx[sel]
        u = np.einsum("smk,sk->sm", th[:, m, :], a) / norm[:, None]  # (S, n)
        u_med = np.median(u, 0)
        grid = np.linspace(u_med.min() - 0.5, u_med.max() + 1.5, 200)
        d = dceil[:, b] if dceil is not None else np.ones(len(norm))
        curve = c + (d - c)[:, None] * expit(norm[:, None] * grid[None, :] - D[:, b][:, None])
        ax = axes[row, col]
        ax.fill_between(grid, *np.quantile(curve, [0.1, 0.9], 0), color="#9db8e0", alpha=0.45, lw=0)
        ax.plot(grid, np.median(curve, 0), color="#1f4e9c", lw=2)
        when = pd.to_datetime([dates.get(names[i]) for i in m])
        yr = np.array([t.year + t.dayofyear / 365 if pd.notna(t) else np.nan for t in when])
        sc = ax.scatter(u_med, data.scores[sel], c=yr, cmap="viridis", s=26, edgecolor="white", lw=0.5,
                        vmin=2023, vmax=2026.9, zorder=3)
        best = data.scores[sel].max()
        ax.axhline(best, color="grey", ls="--", lw=0.8)
        if dceil is not None:
            ax.axhline(np.median(d), color="#c0392b", lw=1.2, label=f"ceiling d = {np.median(d):.3f}")
            ax.legend(loc="lower right", fontsize=8)
        ax.set(ylim=(0, 1.02), xlabel="ability along the benchmark's loading direction (u)",
               ylabel="score", title=f"{bname}, {label}\nslope |A| = {np.median(norm):.2f}", )
        ax.title.set_fontsize(10)
    data, floor_c, A, th, D, dceil = fits["with ceilings (flagship)"]
    bench = data.blookup.sort_values("benchmark_idx")["benchmark"].tolist()
    b = bench.index(bname)
    norm = np.linalg.norm(A[:, b, :], axis=1)
    ax = axes[row, 2]
    ax.scatter(norm, dceil[:, b], s=6, alpha=0.4, color="#1f4e9c")
    best = data.scores[data.bench_idx == b].max()
    ax.axhline(best, color="grey", ls="--", lw=0.8, label=f"best observed {best:.3f}")
    r = np.corrcoef(norm, dceil[:, b])[0, 1]
    ax.set(xlabel="slope |A_b| (posterior draws)", ylabel="ceiling d_b",
           title=f"{bname}, ceiling fit: ceiling vs slope draws\ncorrelation {r:+.2f}")
    ax.title.set_fontsize(10)
    ax.legend(loc="upper right", fontsize=8)
cb = fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.015, pad=0.02)
cb.set_label("release date")
fig.savefig(HERE / "ceiling_vs_slope.png", dpi=140, bbox_inches="tight")
print(f"-> {HERE / 'ceiling_vs_slope.png'}")
