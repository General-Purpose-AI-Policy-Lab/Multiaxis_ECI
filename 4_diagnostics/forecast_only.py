"""Re-render ONLY the frontier-forecast figures of one dashboard card.

`4_build_dashboard.py --force <fit>` re-renders the whole card: posterior
predictive over every draw, LOO, ESS on the log-likelihood. None of it is
touched by a change to the forecast rule. This path reads the posterior
variables the abilities need (no log-likelihood), builds the card's view the
way the card does, rebuilds the forecast figures through
`viz.dashboard.forecast_figures` (the SAME function the card uses, so the two
cannot drift), patches them into the cached card and re-emits index.html from
the caches.

    python 4_diagnostics/forecast_only.py                    # flagship
    python 4_diagnostics/forecast_only.py --fit <name> --no-html

The view is the card's own (`card_view`): `prepare_fit` on the card's chains,
put back on the fit's display frame (`mirt_loadings.csv`) when the card keeps a
chain subset, and titled from `axis_names.json`. Until 2026-10-06 this script
rank-tracked the axes by itself and left the titles out, so its panels could
sit in another axis order than the rest of the card and read "Axis k".

Only valid for a compensatory fit whose FitView frame is the raw/rank-tracked
one (normal loading prior). A signed fit forecasts in the promax frame, which
needs the oblique alignment the full card computes, so this path refuses it and
sends you back to build_dashboard.
"""
from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
from pathlib import Path

import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "2_model"))

from multiaxis_eci.analysis import (  # noqa: E402
    align_to_reference_loadings,
    load_axis_titles,
    prepare_fit,
)
from multiaxis_eci.scripts import load as _load_script  # noqa: E402

_bd = _load_script("4_diagnostics/4_build_dashboard.py")
CACHE_DIR, _cache_load = _bd.CACHE_DIR, _bd._cache_load
_trace_path, all_fits = _bd._trace_path, _bd.all_fits
from multiaxis_eci.viz.dashboard import forecast_figures  # noqa: E402

DEFAULT_FIT = "k4_humanmerge_flagship"
# What prepare_fit and the human-tier placement read; the log-likelihood, the bulk of
# the trace, is left on disk.
VIEW_VARS = ("A", "theta", "tau_A", "D", "phi_b")


def card_view(fit, data):
    """The card's view and axis titles, built as `4_build_dashboard.render_fit` builds them:
    the card's chains (its `drop_chains` left out), `prepare_fit`, the display frame of
    `mirt_loadings.csv` for a chain subset, the titles of `axis_names.json`."""
    trace = _trace_path(fit)
    drop = set(fit.get("drop_chains") or [])
    with xr.open_dataset(trace, group="posterior") as post:
        n_chains = int(post.sizes["chain"])
    chains = [c for c in range(n_chains) if c not in drop] if drop else None
    idata = fit["spec"].open_posterior(keep=VIEW_VARS, chains=chains, path=trace)
    view = prepare_fit(idata, data)
    ref = trace.parent / "mirt_loadings.csv"
    if drop and view.A is not None and view.K >= 2 and ref.exists():
        view = align_to_reference_loadings(view, data, ref)
    titles = (load_axis_titles(trace.parent, view, data)
              if view.A is not None and view.K >= 2 else None)
    return view, titles


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
        sys.exit(f"no cached card for {args.fit!r} — run 4_build_dashboard.py "
                 f"--force {args.fit} once first")

    if (fit.get("kind", "comp") != "comp"
            or fit["spec"].loading_prior == "signed"):
        sys.exit(f"{args.fit!r} is not a raw-frame compensatory fit — its forecast "
                 f"frame needs the loadings, so use 4_build_dashboard.py --force")

    data = fit["spec"].load_data()[0]
    from multiaxis_eci.data import PROCESSED_FILE, SCORE_COLUMNS
    raw = pd.read_csv(PROCESSED_FILE)[SCORE_COLUMNS]
    view, titles = card_view(fit, data)
    figs = forecast_figures(view, data, raw, view.names, view.theta, axis_titles=titles)
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
        subprocess.run([sys.executable, str(ROOT / "4_diagnostics" / "4_build_dashboard.py")],
                       check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
