"""PPCA explained-variance scree report on the CURRENT data.

Epoch reports dimensionality as *explained variance* per principal component
("PC1 explains X% of score variance"), not as the raw ARD axis-strength (tau)
spectrum that notebooks/multidim_ppca_ard.ipynb prints. This script fits the
same probabilistic-PCA-with-ARD model on the column-centered logit score matrix
(handling the missing cells the sparse benchmark table has), then converts the
posterior loadings into the explained-variance framing:

    for each posterior draw:
        W (B x K) --SVD--> singular values s_k        (orthogonal axes)
        signal var_k          = s_k^2
        total var             = sum_k s_k^2  +  B * sigma^2   (+ noise floor)
        explained-var ratio_k = var_k / total var

Reporting it per draw gives a Bayesian scree plot (median + 90% interval) rather
than a single point estimate. Output: a PDF scree report and a CSV.

    python diagnostics/ppca_explained_variance.py            # all data, k>=8, no humans
    python diagnostics/ppca_explained_variance.py --k 6 --draws 2000
    python diagnostics/ppca_explained_variance.py --post2023
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import ECI_EPS                       # noqa: E402
from data import load_eci_data                   # noqa: E402

MIN_BENCH_OBS = 3
RESULTS_DIR = ROOT / "results"


@dataclass
class PreppedData:
    x: np.ndarray
    model_idx: np.ndarray
    bench_idx: np.ndarray
    M: int
    B: int
    model_names: np.ndarray
    bench_names: np.ndarray
    n_obs: int
    k: int


def prepare_matrix(data, k, exclude_humans=True, min_bench_obs=MIN_BENCH_OBS):
    """Filter (rows >=k obs, drop humans; cols >=min_bench_obs) -> clip -> logit ->
    column-center. Same recipe as the multidim_ppca_ard notebook."""
    model_names_all = data.mlookup["model"].to_numpy()
    bench_names_all = data.blookup["benchmark"].to_numpy()

    keep_model = data.n_obs_per_model >= k
    if exclude_humans:
        keep_model &= ~data.is_human
    obs_keep = keep_model[data.model_idx]
    m_old, b_old, y = data.model_idx[obs_keep], data.bench_idx[obs_keep], data.scores[obs_keep]

    bench_counts = np.bincount(b_old, minlength=data.n_benchmarks)
    col_ok = (bench_counts >= min_bench_obs)[b_old]
    m_old, b_old, y = m_old[col_ok], b_old[col_ok], y[col_ok]

    kept_models = np.sort(np.unique(m_old))
    kept_bench = np.sort(np.unique(b_old))
    m_remap = {o: n for n, o in enumerate(kept_models)}
    b_remap = {o: n for n, o in enumerate(kept_bench)}
    model_idx = np.array([m_remap[i] for i in m_old])
    bench_idx = np.array([b_remap[j] for j in b_old])
    M, B = len(kept_models), len(kept_bench)

    z = logit(np.clip(y, ECI_EPS, 1.0 - ECI_EPS))
    col_mean = np.array([z[bench_idx == j].mean() for j in range(B)])
    x = z - col_mean[bench_idx]

    return PreppedData(x, model_idx, bench_idx, M, B,
                       model_names_all[kept_models], bench_names_all[kept_bench],
                       len(y), k)


def build_ppca(prep, K):
    import pymc as pm
    coords = {"model": prep.model_names, "bench": prep.bench_names, "latent": np.arange(K)}
    with pm.Model(coords=coords) as model:
        mi = pm.Data("model_idx", prep.model_idx, dims="obs_id")
        bi = pm.Data("bench_idx", prep.bench_idx, dims="obs_id")
        Z = pm.Normal("Z", 0.0, 1.0, dims=("model", "latent"))
        tau = pm.LogNormal("tau", mu=np.log(0.3), sigma=1.0, dims="latent")   # ARD per axis
        W_z = pm.Normal("W_z", 0.0, 1.0, dims=("bench", "latent"))
        W = pm.Deterministic("W", W_z * tau, dims=("bench", "latent"))
        mu = (Z[mi] * W[bi]).sum(axis=-1)
        sigma = pm.HalfNormal("sigma", 2.0)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=prep.x, dims="obs_id")
    return model


def explained_variance(idata, prep, K):
    """Per-draw SVD of the loadings -> orthogonal per-axis variance & ratios.

    Returns a dict of arrays shaped (n_draws, K) for signal_var, ratio_total
    (incl. the B*sigma^2 noise floor) and ratio_signal (latent structure only),
    plus the per-draw noise-floor variance.
    """
    post = idata.posterior
    W = post["W"].values                          # (chain, draw, B, K)
    C, S, B, _ = W.shape
    W = W.reshape(C * S, B, K)
    sigma = post["sigma"].values.reshape(C * S)   # (n_draws,)

    sv = np.linalg.svd(W, compute_uv=False)       # (n_draws, K) descending
    signal_var = sv ** 2                           # variance along orthogonal axis k
    noise_var = B * sigma ** 2                      # total noise-floor variance
    total = signal_var.sum(axis=1) + noise_var     # (n_draws,)
    signal_total = signal_var.sum(axis=1)

    ratio_total = signal_var / total[:, None]
    ratio_signal = signal_var / signal_total[:, None]
    return dict(signal_var=signal_var, noise_var=noise_var,
                ratio_total=ratio_total, ratio_signal=ratio_signal)


def svd_loadings(idata, prep, K):
    """Per-draw benchmark loadings on each orthogonal PC (= SVD left singular
    vectors, scaled by the singular value), sign-canonicalized across draws.

    Returns (median loadings B x K, sign-agreement B x K in [0,1]).
    """
    W = idata.posterior["W"].values                # (chain, draw, B, K)
    C, S, B, _ = W.shape
    W = W.reshape(C * S, B, K)
    U, sv, _ = np.linalg.svd(W, full_matrices=False)  # U (N,B,K), sv (N,K) descending
    L = U * sv[:, None, :]                          # loadings carry the axis scale

    for k in range(K):                             # sign-canonicalize each axis
        col = L[:, :, k]
        ref = col[np.argmax(col.var(axis=1))].copy()
        for _ in range(3):
            signs = np.where(col @ ref < 0, -1.0, 1.0)
            col = col * signs[:, None]
            ref = col.mean(axis=0)
        L[:, :, k] = col

    med = np.median(L, axis=0)                      # (B, K)
    agree = np.mean(np.sign(L) == np.sign(med)[None, :, :], axis=0)
    return med, agree


def summarize(ev, K):
    """Median + 5/95 percentile per axis for each ratio, returned as a DataFrame."""
    rows = []
    for r in range(K):
        rt = ev["ratio_total"][:, r]
        rs = ev["ratio_signal"][:, r]
        sv = ev["signal_var"][:, r]
        rows.append(dict(
            axis=r + 1,
            singular_value=np.median(np.sqrt(sv)),
            expl_var_pct_total=100 * np.median(rt),
            expl_var_pct_total_lo=100 * np.percentile(rt, 5),
            expl_var_pct_total_hi=100 * np.percentile(rt, 95),
            expl_var_pct_signal=100 * np.median(rs),
            expl_var_pct_signal_lo=100 * np.percentile(rs, 5),
            expl_var_pct_signal_hi=100 * np.percentile(rs, 95),
        ))
    df = pd.DataFrame(rows)
    # cumulative on the median (draw-wise cumulative would double-count CI)
    df["cum_pct_total"] = df["expl_var_pct_total"].cumsum()
    df["cum_pct_signal"] = df["expl_var_pct_signal"].cumsum()
    return df


def make_variance_pdf(df, ev, K, out_pdf):
    """Standalone PDF: explained variance per component, incl. noise floor."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    noise_pct = 100 * np.median(ev["noise_var"] /
                                (ev["signal_var"].sum(axis=1) + ev["noise_var"]))
    axes = df["axis"].to_numpy()

    with PdfPages(out_pdf) as pdf:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        med = df["expl_var_pct_total"].to_numpy()
        lo = med - df["expl_var_pct_total_lo"].to_numpy()
        hi = df["expl_var_pct_total_hi"].to_numpy() - med
        bars = list(med) + [noise_pct]
        xall = list(axes) + [K + 1]
        ax.bar(xall[:-1], med, color="#55A868", width=0.7, zorder=3,
               yerr=[lo, hi], capsize=3, ecolor="#2f5d3f", error_kw=dict(lw=1.2))
        ax.bar([K + 1], [noise_pct], color="#999999", width=0.7, zorder=3)
        for x, v in zip(xall, bars):
            ax.text(x, v + 1.0, f"{v:.0f}%", ha="center", va="bottom", fontsize=9)
        ax.set_xlabel("Latent axis  (last bar = residual noise floor)")
        ax.set_ylabel("Share of total score variance  (%)")
        ax.set_xticks(xall)
        ax.set_xticklabels([str(a) for a in axes] + ["noise"])
        ax.set_ylim(0, max(bars) * 1.25 + 6)
        ax.set_title("Explained variance per principal component", fontsize=12)
        ax.grid(axis="y", ls=":", alpha=0.5, zorder=0)
        fig.tight_layout()
        pdf.savefig(fig); plt.close(fig)

    return noise_pct


def make_loadings_pdf(loadings, agree, df, prep, out_pdf, n_load_axes=3, top=12):
    """Standalone PDF: one page per leading axis showing what it loads on."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    names = prep.bench_names

    with PdfPages(out_pdf) as pdf:
        for r in range(n_load_axes):
            med_l = loadings[:, r]
            order = np.argsort(-np.abs(med_l))[:top]
            order = order[np.argsort(med_l[order])]      # ascending for barh
            vals = med_l[order]
            labs = [names[b] for b in order]
            agr = agree[order, r]
            colors = ["#C44E52" if v < 0 else "#4C72B0" for v in vals]

            fig, ax = plt.subplots(figsize=(9.0, 5.6))
            y = np.arange(len(vals))
            ax.barh(y, vals, color=colors, zorder=3)
            ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=8)
            span = abs(vals).max()
            ax.set_xlim(min(0, vals.min()) - 0.28 * span,
                        max(0, vals.max()) + 0.28 * span)
            for yi, v, a in zip(y, vals, agr):
                ax.text(v + (0.02 * span if v >= 0 else -0.02 * span),
                        yi, f"{a*100:.0f}%", va="center",
                        ha="left" if v >= 0 else "right", fontsize=7, color="#555")
            ax.axvline(0, color="k", lw=0.8)
            pct = df["expl_var_pct_total"].iloc[r]
            ax.set_xlabel("Loading   (sign of benchmark's contribution to the axis)")
            ax.set_title(f"Axis {r+1} — {pct:.0f}% of total variance\n"
                         f"top {top} benchmarks by |loading|   "
                         f"(% = sign agreement across posterior draws)", fontsize=10)
            ax.grid(axis="x", ls=":", alpha=0.5, zorder=0)
            fig.tight_layout()
            pdf.savefig(fig); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8, help="min benchmark scores per model")
    ap.add_argument("--K", type=int, default=8, help="latent axes (overcomplete for ARD)")
    ap.add_argument("--include-humans", action="store_true")
    ap.add_argument("--post2023", action="store_true",
                    help="drop models with a known release date before 2024-01-01")
    ap.add_argument("--draws", type=int, default=1500)
    ap.add_argument("--tune", type=int, default=1000)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--reuse", action="store_true",
                    help="reuse the cached .nc trace instead of re-sampling")
    args = ap.parse_args()

    data = load_eci_data(
        include_all_benchmarks=True,
        min_release_date="2024-01-01" if args.post2023 else None)
    prep = prepare_matrix(data, k=args.k, exclude_humans=not args.include_humans)
    scope = "post2023" if args.post2023 else "all"
    tag = f"{scope}_k{args.k}_K{args.K}" + ("_humans" if args.include_humans else "")
    print(f"\n[{tag}]  {prep.M} models x {prep.B} benchmarks, {prep.n_obs} cells "
          f"({100*prep.n_obs/(prep.M*prep.B):.1f}% dense)")

    import arviz as az
    import pymc as pm
    RESULTS_DIR.mkdir(exist_ok=True)
    trace_path = RESULTS_DIR / f"ppca_explained_variance_{tag}.nc"
    if args.reuse and trace_path.exists():
        idata = az.from_netcdf(trace_path)
        print(f"reusing cached trace -> {trace_path}")
    else:
        with build_ppca(prep, K=args.K):
            idata = pm.sample(draws=args.draws, tune=args.tune, chains=args.chains,
                              target_accept=0.9, random_seed=42, progressbar=False)
        idata.to_netcdf(trace_path)
    ndiv = int(idata.sample_stats["diverging"].sum())
    print(f"divergences: {ndiv} / {args.chains*args.draws}")

    ev = explained_variance(idata, prep, K=args.K)
    df = summarize(ev, K=args.K)
    loadings, agree = svd_loadings(idata, prep, K=args.K)

    out_csv = RESULTS_DIR / f"ppca_explained_variance_{tag}.csv"
    out_pdf_var = RESULTS_DIR / "explained_variance.pdf"
    out_pdf_load = RESULTS_DIR / "axis_loadings.pdf"
    df.to_csv(out_csv, index=False)
    noise_pct = make_variance_pdf(df, ev, args.K, out_pdf_var)
    make_loadings_pdf(loadings, agree, df, prep, out_pdf_load)

    print("\nexplained variance per component:")
    print(f"  (residual noise floor = {noise_pct:.1f}% of total score variance)")
    for _, r in df.iterrows():
        print(f"  PC{int(r.axis)}: {r.expl_var_pct_signal:5.1f}% of signal "
              f"[{r.expl_var_pct_signal_lo:.1f}, {r.expl_var_pct_signal_hi:.1f}]   "
              f"cum {r.cum_pct_signal:5.1f}%   "
              f"({r.expl_var_pct_total:4.1f}% of total)")
    print(f"\ncsv -> {out_csv}\nexplained variance pdf -> {out_pdf_var}"
          f"\naxis loadings pdf -> {out_pdf_load}")


if __name__ == "__main__":
    main()
