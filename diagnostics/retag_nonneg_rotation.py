"""Re-tag a fitted 'normal'-prior MIRT trace to a different display rotation,
and regenerate its rotation-dependent CSVs — WITHOUT resampling.

Rotation is a post-sampling step and GoF/R²/residuals are rotation-invariant,
so a fit can be switched between display frames in place: set
posterior.attrs['mirt_display_rotation'] (so prepare_fit / the dashboard pick
it up) and rewrite mirt_loadings.csv / mirt_factor_corr.csv /
mirt_factor_scores.csv in that frame. Mirrors the post-sampling block of
fit.run_exploration.

Frames:
  nonneg — non-negativity-constrained varimax (pure positive bundles).
  none   — raw rank-tracked axes, no rotation at all (the non-negative prior
           pins the frame; Phi = raw ability correlation).
  promax — the default frame; clears the attr instead of setting it.

Usage:
    python -m diagnostics.retag_nonneg_rotation <trace.nc> \
        [--rotation nonneg|none|promax] [--dry-run]
"""
import argparse
import gc
import json
from pathlib import Path

import h5py
import numpy as np
from xarray.backends.file_manager import FILE_CACHE

from analysis import (
    apply_rotation, factor_corr_df, factor_scores_df, loadings_table,
    mirt_factors_from_trace, nonneg_rotate, promax_rotate, tau_spectrum_df,
)
from data import load_eci_data, load_benchmark_floors, clip_scores_to_floors
from persistence import load_trace, save_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", type=Path)
    ap.add_argument("--rotation", default="nonneg",
                    choices=["nonneg", "none", "promax"],
                    help="display frame to re-tag to (default nonneg; promax "
                         "clears the attr — it is the pipeline default)")
    ap.add_argument("--dry-run", action="store_true",
                    help="report the rotation but don't rewrite trace/CSVs")
    args = ap.parse_args()

    results_dir = args.trace.parent
    idata = load_trace(args.trace)

    # data must match the fit scope (exploration = all benchmarks; floors clip
    # only shifts observed scores, never the loading geometry, but we load it
    # the same way the fit did for the benchmark/model lookups). The scope
    # flags are read off the trace so the rewritten CSVs index the same
    # benchmarks/models the fit did — a cyber fit has 9 extra benchmarks, a
    # --drop-benchmarks fit has fewer, and loading the wrong scope would
    # silently mislabel every loading row.
    attrs = idata.posterior.attrs
    drop = (json.loads(attrs["mirt_drop_benchmarks"])
            if "mirt_drop_benchmarks" in attrs else None)
    data = load_eci_data(include_all_benchmarks=True,
                         fit_cyber="mirt_cyber" in attrs,
                         fit_simpleqa_original="mirt_simpleqa_original" in attrs,
                         drop_benchmarks=drop)
    if "mirt_floor_c" in attrs:
        data = clip_scores_to_floors(data, load_benchmark_floors(data))
    bench_names = data.blookup.sort_values("benchmark_idx")["benchmark"].tolist()
    model_names = data.mlookup.sort_values("model_idx")["model"].tolist()

    A, theta, tau = mirt_factors_from_trace(idata)
    if A.shape[1] != len(bench_names):
        raise SystemExit(
            f"scope mismatch: trace has {A.shape[1]} benchmarks, reloaded data "
            f"has {len(bench_names)} — the trace carries a scope flag this "
            f"script does not reconstruct")
    K = A.shape[2]
    if args.rotation == "none":
        # Raw frame: rank-tracking (inside mirt_factors_from_trace) already
        # fixed the cross-chain axis permutation; nothing else to apply.
        Phi = np.corrcoef(theta.mean(axis=0).T) if K > 1 else np.array([[1.0]])
    else:
        rotate = nonneg_rotate if args.rotation == "nonneg" else promax_rotate
        Tload, Ttheta, Phi = rotate(A.mean(axis=0))
        A, theta, Phi = apply_rotation(A, theta, Phi, Tload, Ttheta)

    L = A.mean(axis=0)
    neg = int((L < -0.05).sum())
    print(f"frame '{args.rotation}': {A.shape[0]} draws, {L.shape[0]} benchmarks, "
          f"K={K}")
    print(f"  negative loading cells (<-0.05) at mean: {neg}/{L.size}  "
          f"worst = {L.min():+.3f}")
    print("  Phi off-diag: " +
          ", ".join(f"{Phi[i, j]:+.2f}" for i in range(Phi.shape[0])
                    for j in range(i + 1, Phi.shape[1])))
    for k in range(K):
        order = np.argsort(-L[:, k])[:8]
        print(f"  axis{k+1}: " +
              ", ".join(f"{bench_names[i]}={L[i, k]:+.2f}" for i in order))

    if args.dry_run:
        print("\n[dry-run] no files written")
        return

    # CSVs first — these read only the in-memory rotated arrays, no file handle.
    save_df(tau_spectrum_df(tau), results_dir / "mirt_tau_spectrum.csv")
    save_df(factor_corr_df(Phi).reset_index().rename(columns={"index": "axis"}),
            results_dir / "mirt_factor_corr.csv")
    save_df(loadings_table(A, bench_names, data.bench_category),
            results_dir / "mirt_loadings.csv")
    save_df(factor_scores_df(theta, model_names, is_human=data.is_human),
            results_dir / "mirt_factor_scores.csv")

    # The trace only needs one string attr changed. Edit it in place (the .nc
    # is HDF5; h5py can update a group attribute without touching the data)
    # instead of re-serialising the file — a full re-save rewrites the whole
    # trace, 19 GB on the flagship, to store one string.
    # close() alone is not enough: xarray's CachingFileManager keeps the HDF5
    # handle alive in a module-level cache, and HDF5 refuses a read-write open
    # while any read handle on the same file exists in the process.
    idata.close()
    del idata
    gc.collect()
    FILE_CACHE.clear()
    with h5py.File(args.trace, "r+") as f:
        post = f["posterior"]
        if args.rotation == "promax":
            post.attrs.pop("mirt_display_rotation", None)
        else:
            post.attrs["mirt_display_rotation"] = args.rotation
    print(f"\nRe-tagged trace + rewrote loadings/corr/scores → {results_dir}")


if __name__ == "__main__":
    main()
