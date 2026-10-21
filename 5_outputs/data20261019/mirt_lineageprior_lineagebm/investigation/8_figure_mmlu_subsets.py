"""Saturation on MMLU Biology and MMLU Chemistry (RAND subsets), with MMLU itself for reference.

Per benchmark: each model's score against its ability along the benchmark's loading direction in the
--ceiling-noise fit, u = A_b . theta_m / |A_b| (posterior median); the ceiling fit's curve
c + (d - c) sigmoid(|A| u - D) (median, 80% band); the flagship's curve without ceilings, its own u
mapped onto this axis by a linear fit over the benchmark's models (dashed); the ceiling d (red), MBF's
L (orange, dotted) and the best observed score (thin grey). Points coloured by release date.
Adds the High School Top Performer's 0.898 (an estimate for all of MMLU, not measured on these items), which
bounds the ceiling from below, and names the top models. Writes mmlu_subsets_saturation.png beside this script.
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

FLAG = GEN / "mirt_lineageprior_lineagebm_noceiling" / "trace_mirt_k4_lineageprior_lineagebm_noceiling.nc"
CEIL = GEN / "mirt_lineageprior_lineagebm" / "trace_mirt_k4_lineageprior_lineagebm.nc"
BENCH = ["MMLU Biology", "MMLU Chemistry", "MMLU"]
HUMAN = 0.898
THIN = 10
mbf = pd.read_csv(HERE / "ceilings_vs_mbf.csv").set_index("benchmark")["L_median"]
dates = pd.read_csv(ROOT / "0_input" / "all_scores_flat.csv", parse_dates=["release_date"]) \
    .groupby("model_version")["release_date"].min()


def load(trace):
    post = xr.open_dataset(trace, group="posterior")
    spec = FitSpec.from_trace(az.InferenceData(posterior=post.isel(draw=slice(0, 1))), trace)
    data, floor_c, _ = spec.load_data()
    p = post.isel(chain=best_mode_chains(trace), draw=slice(None, None, THIN))
    flat = lambda v: p[v].values.reshape(-1, *p[v].shape[2:])          # noqa: E731
    return data, floor_c, flat("A"), flat("theta"), flat("D"), \
        (flat("ceiling_d") if "ceiling_d" in p else None)


def projected(fit, b, sel):
    data, floor_c, A, th, D, d = fit
    a = A[:, b, :]
    norm = np.linalg.norm(a, axis=1)
    u = np.einsum("smk,sk->sm", th[:, data.model_idx[sel], :], a) / norm[:, None]
    return np.median(u, 0), norm, D[:, b], (d[:, b] if d is not None else np.ones(len(norm))), \
        (floor_c[b] if floor_c is not None else 0.0)


flag, ceil = load(FLAG), load(CEIL)
data = ceil[0]
bench = data.blookup.sort_values("benchmark_idx")["benchmark"].tolist()
names = data.mlookup.sort_values("model_idx")["model"].tolist()
fig, axes = plt.subplots(1, 3, figsize=(19, 6.2), gridspec_kw=dict(wspace=0.18))
for ax, bname in zip(axes.ravel(), BENCH):
    b = bench.index(bname)
    sel = np.where(data.bench_idx == b)[0]
    y = data.scores[sel]
    u_c, n_c, D_c, d_c, c = projected(ceil, b, sel)
    u_f, n_f, D_f, d_f, _ = projected(flag, b, sel)            # same observations, same order
    slope, icpt = np.polyfit(u_f, u_c, 1)                      # flagship axis -> ceiling-fit axis
    grid = np.linspace(u_c.min() - 0.4, u_c.max() + 1.2, 200)
    cur_c = c + (d_c - c)[:, None] * expit(n_c[:, None] * grid[None, :] - D_c[:, None])
    gf = (grid - icpt) / slope                                  # the same points on the flagship's axis
    cur_f = c + (1 - c) * expit(n_f[:, None] * gf[None, :] - D_f[:, None])
    ax.fill_between(grid, *np.quantile(cur_c, [0.1, 0.9], 0), color="#9db8e0", alpha=0.45, lw=0)
    ax.plot(grid, np.median(cur_c, 0), color="#1f4e9c", lw=2, label="with ceilings")
    ax.plot(grid, np.median(cur_f, 0), color="#555555", lw=1.4, ls="--", label="no ceilings")
    when = pd.to_datetime([dates.get(names[i]) for i in data.model_idx[sel]])
    yr = np.array([t.year + t.dayofyear / 365 if pd.notna(t) else np.nan for t in when])
    sc = ax.scatter(u_c, y, c=yr, cmap="viridis", s=20, edgecolor="white", lw=0.4, vmin=2023, vmax=2026.9,
                    zorder=3)
    ax.axhline(np.median(d_c), color="#c0392b", lw=1.1, label=f"MX ceiling {np.median(d_c):.3f}")
    if bname in mbf.index and np.isfinite(mbf[bname]):
        ax.axhline(mbf[bname], color="#e67e22", lw=1.3, ls=":", label=f"MBF L {mbf[bname]:.3f}")
    ax.axhline(y.max(), color="grey", lw=0.6)
    ax.axhline(HUMAN, color="#8e44ad", lw=1.2, ls="-.", label=f"HS Top Performer {HUMAN} (bound)")
    for j in np.argsort(-y)[:4]:
        ax.annotate(names[data.model_idx[sel][j]].split('_')[0], (u_c[j], y[j]), fontsize=6.5,
                    xytext=(4, -9 - 3 * (j % 2)), textcoords="offset points", color="#333")
    lo = max(0.0, min(y.min(), np.median(cur_c, 0).min()) - 0.05)
    ax.set_ylim(lo, 1.01)
    ax.set_title(f"{bname}  (n = {len(sel)})", fontsize=10)
    ax.set_xlabel("ability along the loading direction (u)", fontsize=8)
    ax.legend(loc="lower right", fontsize=7)
    ax.tick_params(labelsize=8)
cb = fig.colorbar(sc, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
cb.set_label("release date")
fig.suptitle("Saturation on the MMLU subsets: fitted curves with and without ceilings", fontsize=13, y=1.0)
fig.savefig(HERE / "mmlu_subsets_saturation.png", dpi=120, bbox_inches="tight")
print(f"-> {HERE / 'mmlu_subsets_saturation.png'}")
