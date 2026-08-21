"""Residual correlation analysis: is 1D capability sufficient?

For each benchmark pair (b1, b2):
  observed[b1,b2]  = corr(score_obs[m,b1], score_obs[m,b2]) over models m scoring both
  implied[b1,b2]   = corr(mu[m,b1],         mu[m,b2])         using posterior-mean theta,D,A
  residual         = observed - implied

Reads the canonical K=1 trace (results/canonical/trace.nc, from
`fit.py --preset canonical`). If 1D is enough, residuals are tiny and
unstructured. Clustered structure (e.g. all-math residuals coordinated
positive, math-vs-agentic negative) is the signature of a second latent factor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.cluster.hierarchy import fcluster, leaves_list, linkage
from scipy.spatial.distance import squareform

# Repo root = parent of this file's directory
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis import capability_draws  # noqa: E402
from data import load_eci_data  # noqa: E402
from persistence import save_df  # noqa: E402
from viz import save_fig  # noqa: E402

# ── Configuration ─────────────────────────────────────────────────────────
# Pairs with fewer than RELIABLE_N shared models are excluded from the top-20
# list, the within/between cluster ratio, and contribute only a filled value
# to the clustering distance (so they don't drive cluster assignment).
RELIABLE_N = 30

PLOTS_DIR   = ROOT / "plots"
RESULTS_DIR = ROOT / "results"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Load trace + data ─────────────────────────────────────────────────────
trace = az.from_netcdf(ROOT / "results/canonical/trace.nc")
data = load_eci_data()

C_mean = capability_draws(trace).mean(axis=0)                        # (M,)
D_mean = trace.posterior["D"].mean(("chain", "draw")).values         # (B,)
A_mean = trace.posterior["A"].mean(("chain", "draw")).values[:, 0]   # (B,)

M = data.n_models
B = data.n_benchmarks

# Sanity check: trace and the freshly-loaded data must have matching shapes.
# A mismatch typically means the trace was produced under a different
# data-loading config (e.g. --eci-data-only) and the posterior entries don't
# align with this run's model/benchmark indices.
if (C_mean.shape[0], D_mean.shape[0]) != (M, B):
    raise RuntimeError(
        f"Trace shape (M={C_mean.shape[0]}, B={D_mean.shape[0]}) does not match "
        f"data shape (M={M}, B={B}). Re-run `python fit.py --preset canonical` "
        f"with the same flags that produced the data you want to analyze."
    )

bench_names = data.blookup.sort_values("benchmark_idx")["benchmark"].tolist()

# ── Build (M × B) score matrix and predicted-mu matrix ────────────────────
score_mat = np.full((M, B), np.nan)
score_mat[data.model_idx, data.bench_idx] = data.scores

# mu_{m,b} = sigmoid(A_b * theta_m - D_b) for every (m, b) — the K=1 link.
# Restrict to the observed cells so observed and implied correlations are
# computed over the same per-pair model set.
logit = A_mean[None, :] * C_mean[:, None] - D_mean[None, :]  # (M, B)
mu_mat = 1.0 / (1.0 + np.exp(-logit))
mu_mat[np.isnan(score_mat)] = np.nan


# ── Pairwise correlations (vectorized, pairwise-complete) ────────────────
def pairwise_corr(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (corr_matrix, n_shared_matrix). NaN entries in X = missing.

    Returned `corr` is NaN where fewer than 3 shared non-NaN observations
    exist (pandas' min_periods semantics) or where either column is constant
    on the shared subset.
    """
    C = pd.DataFrame(X).corr(min_periods=3).values
    obs_mask = (~np.isnan(X)).astype(float)
    N = (obs_mask.T @ obs_mask).astype(int)
    return C, N


obs_corr, n_shared = pairwise_corr(score_mat)
imp_corr, _        = pairwise_corr(mu_mat)
residual = obs_corr - imp_corr

# Single source of truth for "reliable": both sites and reads downstream
# (top-20 filter, cluster ratio, heatmap annotation) derive from this mask.
upper        = np.triu_indices(B, k=1)
not_diag     = ~np.eye(B, dtype=bool)
reliable     = (n_shared >= RELIABLE_N) & not_diag
has_residual = ~np.isnan(residual)

# ── Quick pair-coverage diagnostic ────────────────────────────────────────
n_pairs_total    = len(upper[0])
n_pairs_residual = int(has_residual[upper].sum())
n_pairs_reliable = int((reliable & has_residual)[upper].sum())
print("\n── Pair coverage ──")
print(f"   total pairs:        {n_pairs_total}")
print(f"   pairs with any data: {n_pairs_residual}")
print(f"   pairs with n≥{RELIABLE_N}: {n_pairs_reliable}  "
      f"({100*n_pairs_reliable/n_pairs_total:.0f}%)")
print(f"   median n_shared:    {int(np.median(n_shared[upper]))}")

# ── Cluster the residual matrix (Ward linkage on 1 - |residual|) ─────────
# We want benchmarks that share a missing latent factor — i.e. pairs with
# LARGE coordinated |residual| — to be CLOSE in the clustering distance.
# So distance = 1 - |residual|: high residual → small distance.
#
# |residual| can theoretically exceed 1 (it's the difference of two
# correlations in [-1, 1]), but in practice this only happens on unreliable
# pairs (n_shared < 5) where observed corr swings to ±1 by chance. We
# replace those pairs with the median |residual| of reliable pairs anyway,
# so the clip on the rare remaining out-of-range values is defensive only.
#
# Linkage method: average linkage is the textbook choice for arbitrary
# dissimilarity matrices, but in practice on this dataset it isolates
# singleton outlier benchmarks (e.g. WeirdML) and produces less
# thematically coherent clusters than Ward. Ward's Euclidean assumption is
# imperfect here but yields more interpretable groupings. The top-20
# |residual| ranking below is the robust headline; cluster memberships
# are best read as a visualization aid.
absR = np.abs(residual)
fill_value = np.nanmedian(absR[reliable])
absR_filled = np.where(reliable | np.eye(B, dtype=bool), absR, fill_value)
np.fill_diagonal(absR_filled, 0.0)

dist_mat = np.clip(1.0 - absR_filled, 0.0, None)
np.fill_diagonal(dist_mat, 0.0)

dist = squareform(dist_mat, checks=False)
Z = linkage(dist, method="ward")
leaf_order = leaves_list(Z)

# ── Top 20 |residual| pairs (reliable pairs only) ─────────────────────────
i_up, j_up = upper
mask_up = reliable[i_up, j_up] & has_residual[i_up, j_up]
pair_df = pd.DataFrame({
    "benchmark_1":   [bench_names[i] for i in i_up[mask_up]],
    "benchmark_2":   [bench_names[j] for j in j_up[mask_up]],
    "residual":      residual[i_up[mask_up], j_up[mask_up]],
    "observed_corr": obs_corr[i_up[mask_up], j_up[mask_up]],
    "implied_corr":  imp_corr[i_up[mask_up], j_up[mask_up]],
    "n_shared":      n_shared[i_up[mask_up], j_up[mask_up]],
})
pair_df["abs_residual"] = pair_df["residual"].abs()
top20 = (pair_df.sort_values("abs_residual", ascending=False)
                .head(20)
                .drop(columns="abs_residual")
                .round({"residual": 3, "observed_corr": 3, "implied_corr": 3}))

print(f"\n── Top 20 |residual correlations| (n_shared ≥ {RELIABLE_N}) ──")
print(top20.to_string(index=False))
print(f"\n   (Pairs with n_shared < {RELIABLE_N} excluded — they are dominated by "
      f"sampling noise in observed corr, not real structure.)")
print(f"   Eligible pairs: {n_pairs_reliable} of {n_pairs_total} total")

# ── Cluster at k = 2, 3, 4, 5; within/between |residual| ratio ────────────
print(f"\n── Residual-structure clustering "
      f"(Ward linkage, n_shared ≥ {RELIABLE_N} for ratio) ──")

# Precompute the reliable upper-triangle residual vector once.
ru_i, ru_j = i_up[mask_up], j_up[mask_up]
ru_absr    = np.abs(residual[ru_i, ru_j])

for k in [2, 3, 4, 5]:
    labels = fcluster(Z, t=k, criterion="maxclust")
    same = labels[ru_i] == labels[ru_j]
    w = ru_absr[same].mean() if same.any() else np.nan
    b = ru_absr[~same].mean() if (~same).any() else np.nan
    ratio = w / b if b else np.nan
    print(f"\nk = {k}: within={w:.4f}  between={b:.4f}  ratio={ratio:.3f}")
    for cid in sorted(set(labels)):
        members = [bench_names[i] for i in range(B) if labels[i] == cid]
        print(f"  C{cid} ({len(members)}): {', '.join(members)}")

# ── Persist CSVs FIRST, then the heatmap ──────────────────────────────────
# CSVs are cheap and order matters — if the figure write fails, the
# analysis tables are already on disk.
pd.DataFrame(residual, index=bench_names, columns=bench_names).to_csv(
    RESULTS_DIR / "residual_corr_matrix.csv")
save_df(top20.reset_index(drop=True),
        RESULTS_DIR / "residual_corr_top20.csv")
print(f"\nresidual_corr_matrix.csv, residual_corr_top20.csv → {RESULTS_DIR}")

# ── Heatmap (clustered order) ─────────────────────────────────────────────
ordered_names = [bench_names[i] for i in leaf_order]
R_ordered = residual[np.ix_(leaf_order, leaf_order)]
N_ordered = n_shared[np.ix_(leaf_order, leaf_order)]
mask_unreliable = N_ordered < RELIABLE_N
text = np.where(mask_unreliable, "·", "")

fig = go.Figure(data=go.Heatmap(
    z=R_ordered,
    x=ordered_names, y=ordered_names,
    colorscale="RdBu_r", zmid=0, zmin=-0.6, zmax=0.6,
    colorbar=dict(title="residual<br>(obs − implied)"),
    text=text, texttemplate="%{text}",
    textfont=dict(size=10, color="black"),
    hovertemplate=("<b>%{y}</b> vs <b>%{x}</b><br>"
                   "residual = %{z:.3f}<br>"
                   "n_shared = %{customdata}<extra></extra>"),
    customdata=N_ordered,
))
fig.update_layout(
    title=("Residual correlation matrix (observed − 1D-implied), "
           "Ward-clustered<br>"
           f"<sub>· marks pairs with n_shared < {RELIABLE_N} (unreliable)</sub>"),
    xaxis=dict(tickangle=-90, side="bottom"),
    height=900, width=1000,
    margin=dict(l=200, b=200),
)
save_fig(fig, "residual_corr_heatmap", PLOTS_DIR)
print(f"residual_corr_heatmap → {PLOTS_DIR}")
