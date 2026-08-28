"""Re-render ONLY the frontier-forecast figures of one dashboard card.

`build_dashboard.py --force <fit>` re-renders the whole card: posterior
predictive over every draw, LOO, ESS on the log-likelihood. On the 10x20000
flagship that is ~25 minutes, and none of it is touched by a change to the
forecast rule. This path reads the abilities alone, rebuilds the 3 figures per
axis through `viz.dashboard.forecast_figures` (the SAME function the card uses,
so the two cannot drift), patches them into the cached card and re-emits
index.html from the caches.

    python diagnostics/forecast_only.py                    # flagship
    python diagnostics/forecast_only.py --fit <name> --no-html

The ability array is cached to `theta_view_<trace stem>.<key>.f32.npy` beside
the trace, keyed on the trace's (size, mtime) and the dropped chains. Building it reads A, theta and tau_A out of the trace and applies the
same per-draw canonicalisation `prepare_fit` does; every later run memory-maps
the cache. float32 costs ~1e-7 on a scale whose posterior SDs are ~0.3.

Only valid for a compensatory fit whose FitView frame is the raw/rank-tracked
one (normal loading prior). A signed fit forecasts in the promax frame, which
needs the loadings, so this path refuses it and sends you back to
build_dashboard.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis.factors import canonicalize_factors  # noqa: E402
from analysis.fitview import FitView  # noqa: E402
from config import PROJECT_ROOT  # noqa: E402
from diagnostics.build_dashboard import (CACHE_DIR, _cache_load,  # noqa: E402
                                         _trace_path, all_fits)
from viz.dashboard import forecast_figures  # noqa: E402

DEFAULT_FIT = "k4_drop2_humanmerge_flagship"


def _theta_cache_path(trace: Path, drop_chains=None) -> Path:
    """Cache file for one (trace revision, kept chains) pair.

    The key carries the trace's (size, mtime) — the identity
    `build_dashboard._fingerprint` uses — so a re-fit at the same tag never
    serves the old abilities, and the dropped chains, because they are dropped
    before the array is built."""
    st = trace.stat()
    key = json.dumps({"size": st.st_size, "mtime": int(st.st_mtime),
                      "drop_chains": sorted(drop_chains or [])}, sort_keys=True)
    h = hashlib.sha256(key.encode()).hexdigest()[:12]
    return trace.with_name(f"theta_view_{trace.stem}.{h}.f32.npy")


def _load_theta(trace: Path, drop_chains=None) -> np.ndarray:
    """(S, M, K) abilities in the FitView frame, from cache or from the trace.

    Reads only A / theta / tau_A out of the posterior group — a few GB against
    the trace's tens, because the log-likelihood and the deterministic eta are
    the bulk and neither is needed here.
    """
    cache = _theta_cache_path(trace, drop_chains)
    if cache.exists():
        print(f"  theta cache: {cache.name}", flush=True)
        return np.load(cache, mmap_mode="r")

    print(f"  no theta cache for this trace revision — rebuilding from "
          f"{trace.name} …", flush=True)
    ds = xr.open_dataset(trace, group="posterior", engine="h5netcdf")
    if drop_chains:
        ds = ds.sel(chain=[c for c in ds.chain.values if c not in drop_chains])
    A = ds["A"].values
    theta = ds["theta"].values
    tau = ds["tau_A"].values
    ds.close()
    S = A.shape[0] * A.shape[1]
    A, theta, tau = (A.reshape(S, *A.shape[2:]), theta.reshape(S, *theta.shape[2:]),
                     tau.reshape(S, tau.shape[2]))
    # rank_track=True is what prepare_fit picks for a normal-prior, unanchored
    # fit: axes are permuted per draw by tau energy before anything is summarised.
    _, theta, _ = canonicalize_factors(A, theta, tau, rank_track=True)
    theta = theta.astype(np.float32)
    np.save(cache, theta)
    print(f"  wrote {cache.name} ({theta.nbytes / 1e9:.1f} GB)", flush=True)
    return theta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fit", default=DEFAULT_FIT, help="registry `name`")
    ap.add_argument("--no-html", action="store_true",
                    help="patch the card cache but skip the index.html re-emit")
    args = ap.parse_args()

    fits = all_fits()
    fit = next((f for f in fits if f["name"] == args.fit), None)
    if fit is None:
        sys.exit(f"unknown fit {args.fit!r} — registry names: "
                 + ", ".join(f["name"] for f in fits))
    card = _cache_load(fit["name"])
    if card is None:
        sys.exit(f"no cached card for {args.fit!r} — run build_dashboard.py "
                 f"--force {args.fit} once first")

    if (fit.get("kind", "comp") != "comp"
            or fit["spec"].loading_prior == "signed"):
        sys.exit(f"{args.fit!r} is not a raw-frame compensatory fit — its forecast "
                 f"frame needs the loadings, so use build_dashboard.py --force")

    data = fit["spec"].load_data()[0]
    raw = pd.read_csv(PROJECT_ROOT / "data" / "processed" / "benchmarks_merged.csv")
    trace = _trace_path(fit)
    theta = np.asarray(_load_theta(trace, fit.get("drop_chains")), dtype=np.float64)

    names = card["result"]["axis_names"]
    view = FitView(theta=theta, Phi=None, Phi_raw=None, names=names,
                   K=theta.shape[2], is_nc=False, anchored=False, rotated=False)
    figs = forecast_figures(view, data, raw, names, theta)
    if not figs:
        sys.exit("no forecast figures produced — is this a K=1 or human-free fit?")

    stale = [k for k in card["figures"] if k.startswith("forecast_")]
    for k in stale:
        card["figures"].pop(k)
    card["figures"].update(figs)
    with open(CACHE_DIR / f"{fit['name']}.pkl", "wb") as fh:
        pickle.dump(card, fh)
    print(f"patched {len(figs)} forecast figures into {fit['name']} "
          f"(replaced {len(stale)})", flush=True)

    if not args.no_html:
        subprocess.run([sys.executable, str(ROOT / "diagnostics" / "build_dashboard.py")],
                       check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
