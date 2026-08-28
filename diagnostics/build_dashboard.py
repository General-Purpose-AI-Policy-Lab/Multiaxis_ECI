"""Build ONE self-contained interactive dashboard of every MIRT / 1D / PPCA fit.

Writes the repo-root `index.html`: a fit selector (nav grouped by
baseline / exploratory / confirmed) + a cross-fit Comparison view, with every
figure a lazily-rendered Plotly plot (only the visible section is live in the
DOM).

Also writes `results/comparisons/{gof_table,loo_waic_table}.csv` and refreshes
`results/comparisons/README.md`. Pass `--png` to also dump static stills
(kaleido/Chrome) to the git-ignored `plots/dashboard/` for slides.

All rotation/identity handling lives in `analysis.prepare_fit`; all figures are
pure builders in `viz/`. Each entry's `spec` (an `analysis.FitSpec`) owns its
trace path and its data scope, and every fit is scored on the data it was
actually fit on, so GoF is comparable WITHIN a scope; the global R²/RMSE remain
readable across scopes but per-benchmark rows are not apples-to-apples between
them.

Run:  ~/miniforge3/envs/pymc_env/bin/python diagnostics/build_dashboard.py
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import warnings
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from analysis import (  # noqa: E402
    _aligned_reproducibility, crosschain_axis_reproducibility,
    permutation_matched_reproducibility,
    mirt_identified_ess, mirt_identified_rhat, mirt_identified_rhat_nc,
    prepare_fit, trace_loading_prior, FLAGSHIP, FLAGSHIP_TRACE, FitSpec,
)
from diagnostics.diagnose_chains import modes_path  # noqa: E402
from data import load_benchmark_floors  # noqa: E402
from persistence import save_df  # noqa: E402
from viz import (  # noqa: E402
    assemble_dashboard, build_axis_figures,
    build_comparison, build_fit_figures,
    raw_scores_by_date_fig, signed_display_frames, write_dashboard,
)
from ppc import (  # noqa: E402
    compute_gof, posterior_predictive_mirt,
    posterior_predictive_mirt_nc,
)

INDEX_PATH = ROOT / "index.html"
CMP_DIR = ROOT / "results" / "comparisons"

# One entry per card, rendered against the data scope its `spec` names.
# Required: spec (a FitSpec: it owns the trace path and the data scope), name
# (cache key and --force target), label (nav entry and section header, room to
# be descriptive), type (baseline / exploratory / confirmed).
# Optional: short (the axis tick in every cross-fit comparison chart, where the
# long label makes the graph unreadable), group (cards sharing a string are
# grouped in the nav), nav (menu entry), forecast (adds the trend / crossover /
# exceedance figures), drop_chains (a mode-restricted card: those chains go
# before every summary), kind ("comp" sum link default; "nc" conjunctive
# product; "1d" original Beta-IRT), trace_path (an explicit file, for a trace
# whose on-disk name predates the current tag grammar).
# `_validate_fits` checks the required keys and the type before any trace opens.
#
#   {"name": "k4_something",                  # cache key and --force target
#    "spec": FitSpec(K=4, loading_prior="normal", floors=True),
#    "label": "K=4 · <config> — <what it shows>",
#    "short": "K=4 short",
#    "type": "exploratory",
#    "forecast": True},
FITS = [
    {"name": "k4_drop2_humanmerge_flagship",
     "spec": FLAGSHIP,
     # Fitted before FrontierMath v1 and AlgoTune moved to the retirement list,
     # so its folder still carries the `_drop` and `_poolednoise` tokens the
     # current grammar drops. The data scope is the same either way (5,004 obs /
     # 835 takers / 98 benchmarks), so the spec still scores this trace.
     "trace_path": FLAGSHIP_TRACE,
     "label": "K=4 · exploration −2 benchmarks · positive loadings · "
              "raw rank-tracked axes (no rotation) · "
              "human-merge+lineage(BM) priors · 3PL floors · pooled noise · "
              "10×20000, tune 7000 · single solution (10/10 chains one basin, "
              "matched corr 0.843), 78/200,000 divergences, D r-hat 1.168 max; "
              "the merged HS→adult edges bind (P(DE>HSQ) 0.64-0.80 under the "
              "tree order → 1.00) at identical GoF (R² 0.9645 vs 0.9647) — "
              "THE forecasting base",
     "short": "K=4 −2bench · pooled · merge · flagship",
     "type": "exploratory",
     "nav": "K=4 flagship · pooled noise + HS merge", "forecast": True},
]

# Cards added by `--add` instead of by editing FITS above. Tracked, so a
# collaborator gets the same dashboard; the file is a plain list of entries with
# the spec serialised by `dataclasses.asdict` and the trace path stored
# explicitly, which is what lets a legacy folder name keep resolving.
FITS_JSON = ROOT / "diagnostics" / "dashboard_fits.json"


def _json_fits() -> list[dict]:
    """The `--add` registry as registry entries. Empty when the file is absent."""
    if not FITS_JSON.exists():
        return []
    out = []
    for e in json.loads(FITS_JSON.read_text()):
        e = dict(e)
        d = dict(e.pop("spec"))
        d["drop_benchmarks"] = tuple(d.get("drop_benchmarks") or ())
        # The stored spec outlives FitSpec's field set: an entry written before
        # a flag was removed still carries that field, and a missing key falls
        # back to the dataclass default. Unknown keys are reported, not fatal —
        # the alternative is one stale entry breaking every dashboard build.
        known = {f.name for f in dataclasses.fields(FitSpec)}
        stale = sorted(set(d) - known)
        if stale:
            print(f"  dashboard_fits.json {e.get('name')!r}: ignoring fields "
                  f"FitSpec no longer has: {stale}", flush=True)
        e["spec"] = FitSpec(**{k: v for k, v in d.items() if k in known})
        # Stored repo-relative when the trace is inside the tree, so the
        # tracked file is not one machine's home directory.
        e["trace_path"] = ROOT / e["trace_path"]
        e["origin"] = "json"
        out.append(e)
    return out


def all_fits() -> list[dict]:
    """Every card: the code registry first, then the `--add` one."""
    return FITS + _json_fits()


def _trace_path(fit):
    """Trace location for a fit. Its `spec` owns the folder and the filename,
    unless the entry carries an explicit `trace_path`: a trace fitted under an
    earlier tag grammar sits under a name its spec no longer derives, and the
    card must keep pointing at the file rather than at the name."""
    return fit.get("trace_path") or fit["spec"].trace_path


# ── posterior modes ─────────────────────────────────────────────────────────
# A K=3 fit's chains can sit in several likelihood basins, and a figure over ALL
# chains then averages incompatible solutions. The split is detected ONCE, off
# the trace, by `diagnostics/diagnose_chains.py --write-modes`, which writes
# `results/<dir>/mirt_modes_<trace-stem>.json`; the dashboard only READS that
# file, so a build never loads a multi-GB trace to re-detect. Mode views are
# ADDITIONS: the whole-fit figures stay, and no diagnostic number is ever
# mode-restricted (convergence, PPC and PIT describe the whole fit — CLAUDE.md).

def _modes_path(fit):
    """Where this fit's mode split lives — same trace-stem rule as
    `diagnose_chains.modes_path`, so a folder's K=3 and K=1 splits never collide."""
    return modes_path(_trace_path(fit))


def _fit_modes(fit):
    """The persisted mode split for a fit, or None when there is nothing to show.

    None unless the JSON names THIS fit's trace and found more than one mode.
    Also None for a card that already sets `drop_chains`: that card IS one mode,
    so nesting mode views inside it would double-restrict."""
    p = _modes_path(fit)
    if not p.exists() or fit.get("drop_chains"):
        return None
    doc = json.loads(p.read_text())
    if doc.get("trace") != _trace_path(fit).name or len(doc.get("modes", [])) < 2:
        return None
    return doc


def _mode_tag(m):
    """One-line mode identity for a figure title: which chains, and how far below
    the best basin it sits. ASCII hyphen for the sign, matching the table."""
    chains = ",".join(str(c) for c in m["chains"])
    d = m.get("delta_logp")
    where = ("logp unavailable" if d is None
             else "best logp" if d == 0 else f"{d:.0f} nats")
    return f"mode {m['label']} (chains {chains}; {where})"


def _mode_figures(idata, modes, data, raw, bench):
    """Mode-restricted loading + timeline figures: the same builders as the
    whole-fit card, run on the trace sliced to each basin's chains. Each mode
    gets its own rotation/alignment pass, because a basin's axes are its own."""
    figs = {}
    for m in modes:
        sub = idata.sel(chain=m["chains"])
        v = prepare_fit(sub, data)
        figs.update(build_axis_figures(
            v, data, raw, bench, signed_display_frames(v, sub),
            prefix=f"mode{m['label']}_", suffix=f" · {_mode_tag(m)}"))
    return figs


def _modes_table_html(doc):
    """The card's mode summary: chains, Δlogp, worst matched loading correlation
    inside the mode, and the same against the BEST mode. That last column is what
    separates "same loadings, worse basin" (high corr, big Δlogp — split by the
    logp cliff alone) from "a different axis solution" (low corr). Nothing else
    belongs here — a mode is not a converged fit, and every diagnostic on the
    card is a whole-fit number."""
    df = pd.DataFrame([{"mode": m["label"],
                        "chains": ",".join(str(c) for c in m["chains"]),
                        "n_chains": len(m["chains"]),
                        "Δlogp": m["delta_logp"],
                        "min matched loading corr, within mode":
                            m["min_matched_corr"],
                        "min matched loading corr, vs best mode":
                            ("—" if m.get("matched_corr_to_best") is None
                             else m["matched_corr_to_best"])}
                       for m in doc["modes"]])
    return (f"<h3>posterior modes — {len(df)} solutions across "
            f"{doc['n_chains']} chains; loading and timeline figures repeat per "
            "mode below, every metric above is whole-fit</h3>"
            + df.to_html(classes="cmp", index=False, border=0)
            + f'<div class="statline">basis: {doc["basis"]}</div>')


# ── incremental render cache ─────────────────────────────────────────────────
# Rendering a card costs a multi-GB trace load + PPC; the DATA also moves
# between fitting rounds (e.g. new human tiers), after which old traces can't
# be re-rendered at all. So each rendered card (figures + comparison-row
# metrics) is pickled, keyed by the trace's (size, mtime) + RENDER_REV. On the
# next build: unchanged trace → cached card, no trace load; trace DELETED →
# cached card still served, labelled "trace retired" (delete the .nc freely,
# the plots survive); trace present but fit on a different data generation →
# cached card with a "data superseded" label instead of a crash. Bump
# RENDER_REV (or use --force) when the figure set itself changes.
RENDER_REV = 14  # posterior summaries are median + central interval
CACHE_DIR = ROOT / "results" / "dashboard_cache"


def _fingerprint(trace_path, modes_file=None):
    """Cache key: trace identity + figure-set revision + the MODE SPLIT.

    The mode split has to be in here. A card's mode figures are built at render
    time but its mode table is read from the JSON at assembly, so without this a
    cached card would keep serving a table that promises figures it never
    contains — or, after a re-detection at a different threshold, a table whose
    chain groupings contradict its own figure titles."""
    st = trace_path.stat()
    fp = {"size": st.st_size, "mtime": int(st.st_mtime), "rev": RENDER_REV}
    if modes_file is not None and modes_file.exists():
        import hashlib
        fp["modes"] = hashlib.sha256(modes_file.read_bytes()).hexdigest()[:16]
    return fp


def _trace_n_models(trace_path):
    """Model count of a trace WITHOUT loading it (lazy xarray peek). MIRT
    traces have a named `model` dim; a legacy 1D trace (unnamed dims)
    carries it as C's only dim (`C_dim_0`)."""
    import xarray as xr
    with xr.open_dataset(trace_path, group="posterior") as post:
        n = post.sizes.get("model") or post.sizes.get("C_dim_0")
        return int(n) if n else -1


def _cache_load(name):
    p = CACHE_DIR / f"{name}.pkl"
    if not p.exists():
        return None
    import pickle
    try:
        with open(p, "rb") as f:
            return pickle.load(f)
    except Exception as e:  # noqa: BLE001 — corrupt/stale cache → just re-render
        print(f"  cache unreadable for {name} ({type(e).__name__}) — re-rendering")
        return None


def _cache_save(name, figures, result, fingerprint):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    import pickle
    with open(CACHE_DIR / f"{name}.pkl", "wb") as f:
        pickle.dump({"fingerprint": fingerprint, "figures": figures,
                     "result": result}, f)


_MODE_EVAL_CACHE = CACHE_DIR / "mode_eval_ess.json"


def _identified_ess_cached(fit, tp, fp, data, allow_load=True):
    """Min/median bulk ESS on the identified eta subset, computed on the card's
    KEPT chains, plus the chain/draw bookkeeping for the mode-evaluation table.
    Cached by trace identity + chain subset — the trace load dominates the cost
    and the table re-assembles on every build.

    The key deliberately drops `rev` (and the mode split) from the fingerprint:
    ESS depends on the posterior, not on which figures the dashboard draws, so a
    RENDER_REV bump must not force every trace to be re-read.

    `allow_load=False` returns None instead of reading the trace — for a card on
    a superseded data generation, where indexing the trace against today's `data`
    would misalign and raise."""
    import json
    drop = sorted(fit.get("drop_chains", []))
    trace_key = {k: v for k, v in fp.items() if k not in ("rev", "modes")}
    cache = (json.loads(_MODE_EVAL_CACHE.read_text())
             if _MODE_EVAL_CACHE.exists() else {})
    hit = cache.get(fit["name"])
    if (hit and hit["drop_chains"] == drop
            and {k: v for k, v in hit["fingerprint"].items()
                 if k not in ("rev", "modes")} == trace_key):
        return hit
    if not allow_load:
        return None
    import xarray as xr
    with xr.open_dataset(tp, group="posterior") as ds:
        n_total = int(ds.sizes["chain"])       # lazy open: metadata only
    keep = [c for c in range(n_total) if c not in set(drop)]
    sub = fit["spec"].open_posterior(
        keep=("A", "theta", "theta_pos", "alpha", "D"),
        chains=keep, path=tp).posterior
    row = {"fingerprint": fp, "drop_chains": drop, "chains_kept": len(keep),
           "chains_total": n_total, "draws_per_chain": int(sub.sizes["draw"]),
           **mirt_identified_ess(az.InferenceData(posterior=sub), data)}
    cache[fit["name"]] = row
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    _MODE_EVAL_CACHE.write_text(json.dumps(cache, indent=1))
    return row


def compute_loo_waic(idata) -> dict:
    """Pointwise PSIS-LOO + WAIC. Returns the
    pointwise log densities so ΔELPD standard errors are computed correctly."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loo = az.loo(idata, pointwise=True)
        waic = az.waic(idata, pointwise=True)
    loo_i = getattr(loo, "loo_i", None)
    if loo_i is None:
        loo_i = loo.elpd_i if hasattr(loo, "elpd_i") else loo["loo_i"]
    waic_i = getattr(waic, "waic_i", None)
    if waic_i is None:
        waic_i = waic.elpd_i if hasattr(waic, "elpd_i") else waic["waic_i"]
    pareto_k = np.asarray(loo.pareto_k.values).ravel()
    loo_i_arr = np.asarray(loo_i.values).ravel()
    waic_i_arr = np.asarray(waic_i.values).ravel()
    return {
        "loo_elpd": float(loo.elpd_loo), "loo_se": float(loo.se),
        "loo_p_eff": float(loo.p_loo), "waic_elpd": float(waic.elpd_waic),
        "waic_se": float(waic.se), "waic_p_eff": float(waic.p_waic),
        "n_obs": int(loo_i_arr.size),
        "pareto_k_good": int((pareto_k < 0.5).sum()),
        "pareto_k_ok": int(((pareto_k >= 0.5) & (pareto_k < 0.7)).sum()),
        "pareto_k_bad": int(((pareto_k >= 0.7) & (pareto_k < 1.0)).sum()),
        "pareto_k_very_bad": int((pareto_k >= 1.0).sum()),
        "pareto_k_max": float(pareto_k.max()), "pareto_k_mean": float(pareto_k.mean()),
        "loo_pointwise": loo_i_arr, "waic_pointwise": waic_i_arr,
    }


def _loo_min_ess_tau(idata) -> tuple:
    """(loo dict | None, min_ess | None, tau_sorted | None) from one loaded trace.

    LOO/WAIC (only if the trace carries a log_likelihood group), min ESS over
    the per-obs log-likelihood, and the descending posterior-mean τ_A spectrum
    (skipped when constant — shared-scale fits)."""
    loo = min_ess = tau = None
    if "log_likelihood" in list(idata.groups()):
        loo = compute_loo_waic(idata)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            min_ess = float(az.ess(idata.log_likelihood)["obs"].min())
    if "tau_A" in idata.posterior:
        t = np.sort(idata.posterior["tau_A"].mean(("chain", "draw")).values)[::-1]
        if float(np.std(t) / np.mean(t)) > 0.02:
            tau = t
    return loo, min_ess, tau


def _per_bench_rmse(scores, y_pred_mean, bench_idx, bench):
    resid = np.asarray(scores) - np.asarray(y_pred_mean)
    pbr = pd.Series(resid ** 2).groupby(bench_idx).mean() ** 0.5
    return pd.Series(pbr.values, index=[bench[i] for i in pbr.index])


def render_fit(fit, data, raw, bench, mod):
    """Figures + comparison-row metrics for a MIRT fit (comp or non-comp)."""
    idata = az.from_netcdf(_trace_path(fit))
    if fit.get("drop_chains"):
        # Mode-restricted card: drop outlier chain(s) before ANY summary, so the
        # whole card (timeline, forecast, GoF, r̂) reads the majority island.
        keep = [c for c in idata.posterior.chain.values if c not in fit["drop_chains"]]
        idata = idata.sel(chain=keep)
        print(f"  drop_chains {fit['drop_chains']}: kept {len(keep)} chains", flush=True)
    is_nc = fit.get("kind", "comp") == "nc"
    view = prepare_fit(idata, data)
    # Floored fits are scored with the same fixed-c 3PL link they were fit
    # with. A ceiling-noise fit carries its ESTIMATED `ceiling_d` in the
    # posterior; posterior_predictive_mirt picks that up on its own.
    floor_c = load_benchmark_floors(data) if fit["spec"].floors else None
    # Same rule for the known-SE noise split: the predictive Beta must carry the
    # per-cell instrument precision the fit used.
    n_eff = data.n_eff if fit["spec"].known_se else None
    yrep = (posterior_predictive_mirt_nc(idata, data) if is_nc
            else posterior_predictive_mirt(idata, data, floor_c=floor_c,
                                           n_eff=n_eff))
    mu = (posterior_predictive_mirt_nc(idata, data, return_mean=True) if is_nc
          else posterior_predictive_mirt(idata, data, floor_c=floor_c,
                                         n_eff=n_eff, return_mean=True))
    gof = compute_gof(yrep, data, mu)
    figures = build_fit_figures(view, gof, yrep, data, raw, bench, mod, idata,
                                forecast=fit.get("forecast", False))
    modes = _fit_modes(fit)
    if modes:
        print(f"  {len(modes['modes'])} posterior modes: "
              + " | ".join(_mode_tag(m) for m in modes["modes"]), flush=True)
        figures.update(_mode_figures(idata, modes["modes"], data, raw, bench))
    pred_rhat = (mirt_identified_rhat_nc(idata, data)["logmu_max_rhat"] if is_nc
                 else mirt_identified_rhat(idata, data)["eta_max_rhat"])
    div = int(idata.sample_stats["diverging"].sum()) if "diverging" in idata.sample_stats else -1
    n_draws = int(idata.posterior.sizes["chain"] * idata.posterior.sizes["draw"])
    # cross-chain axis reproducibility (needs the raw chains, so computed here
    # while the trace is in memory). Householder loadings are signed → align ±;
    # signed-free fits are ALREADY per-draw aligned inside prepare_fit, so
    # their reproducibility is read straight off the aligned view.theta.
    lp = trace_loading_prior(idata)
    if is_nc:
        axis_repro = None
    elif lp == "signed" and view.K > 1:
        axis_repro = _aligned_reproducibility(view.theta,
                                              int(idata.posterior.sizes["chain"]))
    elif view.A is not None and view.K > 1 and not view.anchored:
        # Free 'normal' loadings: axes are identified only up to PERMUTATION
        # (shared scalar tau_A can't rank them), so crosschain_axis_repro's tau
        # ranking reads ~0 even when every chain finds the same axes. Match
        # chains by loading correlation first.
        axis_repro = permutation_matched_reproducibility(
            view.A, int(idata.posterior.sizes["chain"]))
    else:
        axis_repro = crosschain_axis_reproducibility(idata)
    free_load = int((np.abs(np.median(view.A, axis=0)) > 1e-9).sum()) if view.A is not None else 0
    max_phi = (round(float(np.abs(view.Phi_raw[np.triu_indices(view.K, 1)]).max()), 3)
               if view.K > 1 else "—")
    loo, min_ess, tau = _loo_min_ess_tau(idata)
    if lp == "signed" and view.A is not None and view.K > 1:
        # Shared scalar tau is flat by construction — the spectrum lives in the
        # ALIGNED per-axis column norms.
        tau = np.sort(np.median(np.linalg.norm(view.A, axis=1), axis=0))[::-1]
    m = gof.metrics
    result = {
        "fit": fit["label"], "name": fit["label"], "type": fit["type"], "K": view.K,
        "free_loadings": free_load, "R2": round(m["bayesian_r2"], 4),
        "RMSE": round(m["rmse"], 4), "MAE": round(m["mae"], 4),
        "PIT_var": round(m["pit_var"], 4), "eta_rhat": round(pred_rhat, 3),
        "divergences": div, "max_phi": max_phi, "pit": gof.pit,
        "per_bench_rmse": _per_bench_rmse(data.scores, gof.y_pred_mean, data.bench_idx, bench),
        "min_ess": min_ess, "tau_sorted": tau,
        "axis_repro": axis_repro, "div_pct": 100.0 * div / n_draws,
        # mean loading vectors + axis names, for the cross-fit axis-match map
        "mean_load": (np.median(view.A, axis=0) if view.A is not None else None),
        "axis_names": view.names,
    }
    if loo:
        result.update(loo)   # keeps n_obs so build_comparison can flag obs-count mismatch
    del idata
    return figures, result


def _stat_line(r):
    return (f"type: {r['type']} · K={r['K']} · free loadings={r['free_loadings']} · "
            f"R²={r['R2']} · RMSE={r['RMSE']} · MAE={r['MAE']} · "
            f"identified r̂={r['eta_rhat']} · {r['divergences']} divergences · "
            f"max |Φ|={r['max_phi']}")


def _static_dump(sections, comparison, fmt="png"):
    """Optional static stills → plots/dashboard/ (git-ignored). Needs kaleido/Chrome.

    `fmt` is any kaleido format: "png" (raster, scale=2) or "pdf" (vector, crisper
    for slides/print — resolution-independent, so no `scale`)."""
    base = ROOT / "plots" / "dashboard"
    for sec in sections + [{"id": "comparison", "figures": comparison["figures"]}]:
        out = base / sec["id"]
        out.mkdir(parents=True, exist_ok=True)
        for name, fig in sec["figures"].items():
            try:
                kw = {} if fmt == "pdf" else {"scale": 2}
                fig.write_image(out / f"{name}.{fmt}", **kw)
            except Exception as e:  # noqa: BLE001
                print(f"  {fmt.upper()} skipped {sec['id']}/{name} ({type(e).__name__}: {e})")


# The four types `assemble_dashboard` emits, in nav order. A card whose type is
# outside this set reaches the tables but never the HTML, so the set is checked
# up front rather than discovered after a render.
FIT_TYPES = ("data", "baseline", "exploratory", "confirmed")


def _validate_fits(fits=None):
    """Refuse a malformed registry before any trace is opened.

    A render costs minutes per card and `index.html` is written only at the end,
    so a missing key found mid-loop discards the whole build. A card with
    neither a trace nor a cache can never appear, and an unknown `type` reaches
    the CSVs but not the HTML: both are errors here, not silent skips.
    """
    problems, seen = [], set()
    for i, fit in enumerate(fits if fits is not None else all_fits()):
        # Index within the registry the entry came from, so the message points
        # at the right list.
        code = fit.get("origin", "code") == "code"
        where = (f"{'FITS' if code else FITS_JSON.name}"
                 f"[{i if code else i - len(FITS)}] "
                 f"{fit.get('name', '<no name>')!r}")
        for key in ("spec", "name", "label", "type"):
            if key not in fit:
                problems.append(f"{where}: missing required key {key!r}")
        if fit.get("type") not in FIT_TYPES:
            problems.append(f"{where}: type {fit.get('type')!r} is not one of "
                            f"{FIT_TYPES} — the card would never render")
        name = fit.get("name")
        if name in seen:
            problems.append(f"{where}: duplicate name — both cards share one cache")
        seen.add(name)
        if "spec" in fit and "name" in fit:
            if not _trace_path(fit).exists() and not (CACHE_DIR / f"{name}.pkl").exists():
                problems.append(f"{where}: no trace at {_trace_path(fit)} and no "
                                f"cached card — check the spec's flags")
    if problems:
        sys.exit("registry errors:\n  " + "\n  ".join(problems))


def _list_fits(fits=None):
    """One line per registered card: what it is and whether it can render."""
    fits = all_fits() if fits is None else fits
    print(f"{len(fits)} registered fit(s):\n")
    for fit in fits:
        _print_fit(fit)


def _print_fit(fit):
    tp, name = _trace_path(fit), fit["name"]
    trace = f"{tp.stat().st_size / 1e9:.1f} GB" if tp.exists() else "MISSING"
    cached = "cached" if (CACHE_DIR / f"{name}.pkl").exists() else "no cache"
    print(f"  {name}")
    print(f"    origin={fit.get('origin', 'code')}  K={fit['spec'].K}  "
          f"type={fit['type']}  forecast={fit.get('forecast', False)}  "
          f"trace={trace}  {cached}")
    print(f"    tag: {fit['spec'].tag or '(none)'}")


def _add_fit(args):
    """Register one already-fitted trace as a card, in `FITS_JSON`.

    The spec is recovered from the trace itself (attrs and dims only, no data
    read), so the card cannot disagree with the fit about K, its flags or its
    data scope. The trace path is stored explicitly (repo-relative when it is
    inside the tree): a trace fitted under an earlier tag grammar sits under a
    name today's spec no longer derives.
    Adding does not render — the caller runs the build.
    """
    from diagnostics.plot_mirt import _spec_of

    trace = Path(args.add).resolve()
    if not trace.exists():
        sys.exit(f"no trace at {trace}")
    if any(f["name"] == args.name for f in all_fits()):
        sys.exit(f"name {args.name!r} is already registered — pick another, or "
                 f"--remove it first")
    spec = _spec_of(trace)
    rel = trace.relative_to(ROOT) if trace.is_relative_to(ROOT) else trace
    entry = {"name": args.name, "label": args.label, "type": args.type,
             "spec": dataclasses.asdict(spec), "trace_path": str(rel)}
    for key in ("short", "nav"):
        if getattr(args, key):
            entry[key] = getattr(args, key)
    if args.forecast:
        entry["forecast"] = True
    existing = json.loads(FITS_JSON.read_text()) if FITS_JSON.exists() else []
    FITS_JSON.write_text(json.dumps(existing + [entry], indent=2) + "\n")
    _validate_fits()
    print(f"added to {FITS_JSON.name}:")
    _print_fit(_json_fits()[-1])


def _remove_fit(name):
    """Drop one `--add`ed card. A code entry is not removable from here."""
    if any(f["name"] == name for f in FITS):
        sys.exit(f"{name!r} is a code entry in FITS — remove it by editing "
                 f"{Path(__file__).name}")
    entries = json.loads(FITS_JSON.read_text()) if FITS_JSON.exists() else []
    keep = [e for e in entries if e["name"] != name]
    if len(keep) == len(entries):
        sys.exit(f"no registered fit named {name!r}")
    FITS_JSON.write_text(json.dumps(keep, indent=2) + "\n")
    print(f"removed {name!r} from {FITS_JSON.name} ({len(keep)} left)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png", action="store_true",
                    help="also dump static PNG stills to plots/dashboard/ (git-ignored)")
    ap.add_argument("--pdf", action="store_true",
                    help="also dump vector PDF stills to plots/dashboard/ (crisper; git-ignored)")
    ap.add_argument("--force", nargs="*", default=[],
                    help="fit names (registry `name`) to re-render even if cached")
    ap.add_argument("--force-all", action="store_true",
                    help="ignore the render cache entirely and re-render every fit")
    ap.add_argument("--list", action="store_true",
                    help="print the registry (name, K, tag, trace, cache) and exit")
    ap.add_argument("--add", metavar="TRACE",
                    help="register an already-fitted trace as a card in "
                         "dashboard_fits.json (needs --name and --label); "
                         "renders nothing")
    ap.add_argument("--remove", metavar="NAME",
                    help="drop a card added by --add")
    ap.add_argument("--name", help="with --add: the cache key / --force target")
    ap.add_argument("--label", help="with --add: the section header")
    ap.add_argument("--type", default="exploratory", choices=FIT_TYPES,
                    help="with --add: card type (default exploratory)")
    ap.add_argument("--short", help="with --add: short tick for comparison charts")
    ap.add_argument("--nav", help="with --add: nav menu entry")
    ap.add_argument("--forecast", action="store_true",
                    help="with --add: include the forecast figures")
    args = ap.parse_args()

    if args.remove:
        _remove_fit(args.remove)
        return
    if args.add:
        if not (args.name and args.label):
            ap.error("--add needs --name and --label")
        _add_fit(args)
        return

    fits = all_fits()
    _validate_fits(fits)
    if args.list:
        _list_fits(fits)
        return

    CMP_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(ROOT / "data" / "processed" / "benchmarks_merged.csv")

    # Fits span several data scopes. Each spec loads its own scope once — the
    # same loads, drops and clips its fit ran — so the n_models sanity-check and
    # the per-bench RMSE line up with its trace. Specs are hashable, so the
    # cache keys on the spec itself and two fits sharing a scope share the load.
    _scope_cache: dict = {}
    def scoped(spec):
        if spec not in _scope_cache:
            d = spec.load_data()[0]
            _scope_cache[spec] = (
                d,
                d.blookup.sort_values("benchmark_idx")["benchmark"].tolist(),
                d.mlookup.sort_values("model_idx")["model"].tolist(),
            )
        return _scope_cache[spec]

    sections, results, mode_rows = [], [], []
    for fit in fits:
        data, bench, mod = scoped(fit["spec"])
        tp = _trace_path(fit)
        cache = None if (args.force_all or fit["name"] in args.force) \
            else _cache_load(fit["name"])
        fp = _fingerprint(tp, _modes_path(fit)) if tp.exists() else None
        note = None
        if cache is not None and (fp is None or cache["fingerprint"] == fp):
            # Serve the cached card: trace unchanged, or trace retired/deleted.
            note = "cached" if fp is not None else "cached · trace retired"
            print(f"{note}: {fit['label']}", flush=True)
            figures, r = cache["figures"], cache["result"]
            r["fit"] = fit["label"]      # registry label wins over render-time label
        elif fp is None:
            print(f"skipping {fit['label']} — no trace, no cache ({tp.name})", flush=True)
            continue
        elif _trace_n_models(tp) != data.n_models:
            # Trace belongs to an older data generation, so it CANNOT be
            # re-rendered — indexing it against today's data would misalign every
            # row. The cached card is the only truthful option, so read it even
            # under --force / --force-all: those flags mean "do not serve a cache
            # you could re-render", not "drop a card you cannot". Without this
            # fallback, --force-all silently deleted 11 of 12 cards.
            if cache is None:
                cache = _cache_load(fit["name"])
            if cache is not None:
                note = "cached · data superseded"
                print(f"{note}: {fit['label']} (trace has "
                      f"{_trace_n_models(tp)} models, data has {data.n_models})",
                      flush=True)
                figures, r = cache["figures"], cache["result"]
                r["fit"] = fit["label"]
            else:
                print(f"skipping {fit['label']} — trace fit on a different data "
                      f"generation ({_trace_n_models(tp)} models vs "
                      f"{data.n_models}) and no cached card", flush=True)
                continue
        else:
            print(f"rendering {fit['label']} ({fit['type']}) …", flush=True)
            # Floors are already clipped: the spec's load_data mirrors fit.py.
            figures, r = render_fit(fit, data, raw, bench, mod)
            _cache_save(fit["name"], figures, r, fp)
        # Comparison charts key on r["fit"]/r["name"]: use the SHORT label there
        # (the long label made the grouped-bar / heatmap axes unreadable). The
        # long label stays on the nav + section header below.
        # Factor correlations (Phi) are dropped from the page: on the positive
        # cards the frame is raw and non-negative, so Phi reports ability
        # correlation, not a rotation choice worth a panel. Filtered at
        # assembly rather than at render, so the cached figure survives.
        figures = {k: v for k, v in figures.items() if k != "factor_correlations"}
        short = fit.get("short", fit["label"])
        r["fit"] = r["name"] = short
        stat = _stat_line(r) + (f" · <i>{note}</i>" if note and "cached" != note else "")
        # Mode table read from the JSON at assembly, not from the render: a card
        # served from cache still gets its current mode summary.
        modes = _fit_modes(fit)
        sections.append({"id": fit["name"], "label": fit["label"], "type": fit["type"],
                         "stat_line": stat, "figures": figures,
                         "table_html": _modes_table_html(modes) if modes else None,
                         "group": fit.get("group"), "nav": fit.get("nav")})
        results.append(r)
        if fp is not None:
            # Mode-evaluation row: what each card's kept-chain posterior is
            # worth, in the project's preferred readout (min/median bulk ESS
            # + divergences), next to its fit metrics. A superseded card gets its
            # row from the ESS cache only — its trace cannot be indexed against
            # today's data — and is dropped from the table if there is none.
            e = _identified_ess_cached(fit, tp, fp, data,
                                       allow_load=note != "cached · data superseded")
        if fp is not None and e is not None:
            n_kept = e["chains_kept"] * e["draws_per_chain"]
            # ESS travels with the draw count it came out of: cards keep
            # 4,000-54,000 draws, so an ESS is only readable next to its
            # ceiling.
            r["eta_ess_min"], r["eta_ess_med"] = e["eta_ess_min"], e["eta_ess_med"]
            r["n_draws_kept"] = n_kept
            mode_rows.append({
                "fit": short,
                "chains": f"{e['chains_kept']}/{e['chains_total']}",
                "draws": e["chains_kept"] * e["draws_per_chain"],
                "R2": r["R2"], "RMSE": r["RMSE"],
                "elpd_loo": round(r["loo_elpd"], 1),
                "loo_se": round(r["loo_se"], 1),
                "eta_rhat": r["eta_rhat"],
                "ess_min": int(round(e["eta_ess_min"])),
                "ess_med": int(round(e["eta_ess_med"])),
                "divergences": r["divergences"],
                # LOO reliability travels with the ELPD it qualifies: share of
                # observations whose importance weights failed the k<0.7 check.
                "khat_gt_0.7": round(
                    (r["pareto_k_bad"] + r["pareto_k_very_bad"]) / r["n_obs"], 3),
            })

    # Every registry entry skipped: a stale registry or a missing results tree,
    # never a dashboard worth writing. Exits non-zero so a caller cannot read
    # "Dashboard → index.html" as a fresh build.
    if not results:
        sys.exit(f"no card rendered: all {len(fits)} registry entries skipped "
                 "(no trace and no cached card) — index.html left untouched")

    # Raw-data section: no fit, no trace, no cache — the processed table itself
    # plus every opt-in curated extra-benchmark file (cyber, SimpleQA original;
    # same schema as the processed file), one legend-toggleable trace per
    # benchmark.
    extra_csvs = [
        ROOT / "data" / "curated" / "cyber_benchmarks.csv",
        ROOT / "data" / "curated" / "simpleqa_original" / "simpleqa_original.csv",
    ]
    raw_all = pd.concat(
        [raw] + [pd.read_csv(f) for f in extra_csvs if f.exists()],
        ignore_index=True)
    sections.append({
        "id": "rawdata", "label": "Raw scores by release date", "type": "data",
        "stat_line": f"{len(raw_all)} observations · "
                     f"{raw_all['benchmark'].nunique()} benchmarks · click legend "
                     "entries to overlay benchmarks",
        "figures": {"scores_by_release_date": raw_scores_by_date_fig(raw_all)}})

    # ── comparison tables + figures (one pass, no ordered multi-script run) ──
    tables, cmp_figs = build_comparison(results)

    cmp_dir = CMP_DIR

    save_df(tables["gof_table"], cmp_dir / "gof_table.csv")
    if "loo_waic_table" in tables:
        save_df(tables["loo_waic_table"], cmp_dir / "loo_waic_table.csv")
    print("\n" + tables["gof_table"].to_string(index=False), flush=True)

    mode_df = pd.DataFrame(mode_rows)
    if len(mode_df):
        save_df(mode_df, cmp_dir / "mode_eval_table.csv")
        print("\n" + mode_df.to_string(index=False), flush=True)

    # One table on the page: the mode evaluation. gof_table / loo_waic_table
    # stay as CSVs (and feed the README) — every column a reader needs is in
    # the mode table, keyed by the same short fit names.
    tables_html = ("<h3>mode evaluation (per-card kept chains)</h3>"
                   + mode_df.to_html(classes="cmp", index=False, border=0)
                   if len(mode_df) else "")
    comparison = {"tables_html": tables_html, "figures": cmp_figs}

    html = assemble_dashboard(sections, comparison)
    write_dashboard(html, INDEX_PATH)
    _readme(fits, tables["gof_table"],
            sorted({(d.n_models, d.n_benchmarks, d.n_obs)
                    for d, _, _ in _scope_cache.values()}))
    if args.png:
        _static_dump(sections, comparison, "png")
    if args.pdf:
        _static_dump(sections, comparison, "pdf")
    size_mb = INDEX_PATH.stat().st_size / 1e6
    print(f"\nDashboard → {INDEX_PATH}  ({size_mb:.1f} MB, {len(sections)} fits)", flush=True)


def _df_to_md(df):
    cols = list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    rows += ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join(rows)


def _readme(fits, tab, scopes):
    """Rewrite `results/comparisons/README.md` from the registry and the table
    this build just produced. `scopes` is the (n_models, n_benchmarks, n_obs) of
    every data scope the cards were scored on."""
    fit_list = "\n".join(f"- {f['label']}  ·  *{f['type']}*" for f in fits)
    scope_list = "\n".join(f"- {m} test-takers / {b} benchmarks / {n} observations"
                           for m, b, n in scopes)
    # The across-scope caveat is only a caveat when there is more than one scope.
    scope_note = ("\nGoF is comparable within a scope; global R²/RMSE stay "
                  "readable across scopes, per-benchmark rows do not.\n"
                  if len(scopes) > 1 else "")
    scope_head = "Data scope" if len(scopes) == 1 else "Data scopes"
    # A range reads wrong with one card, so state the single value instead.
    rhat_line = (f"`eta_rhat` is {tab['eta_rhat'].iloc[0]} on the one card"
                 if len(tab) == 1 else
                 f"`eta_rhat` spans {tab['eta_rhat'].min()}–"
                 f"{tab['eta_rhat'].max()} across the {len(tab)} cards")
    txt = f"""# Capability-dimensionality fit dashboard

**Open [`index.html`](../../index.html)** at the repo root — one
self-contained interactive page: a fit selector + a cross-fit Comparison view.
Every figure renders lazily (only the visible fit is live in the DOM).

Registered cards ({len(fits)}):

{fit_list}

{scope_head} scored on:

{scope_list}
{scope_note}
## Comparison table (`gof_table.csv`)

{_df_to_md(tab)}

Columns: `type`; `free_loadings` = free loading cells (complexity); `R2`/`RMSE`/
`MAE` = fit; `PIT_var` = calibration (ideal 0.083, below = under-confident);
`eta_rhat` = r̂ on the identified linear predictor (≤ 1.01 = converged);
`divergences`; `max_phi` = largest off-diagonal axis correlation. PSIS-LOO / WAIC
per fit are in `loo_waic_table.csv`. `mode_eval_table.csv` adds chains, kept
draws, ESS and divergences for every card whose trace is still on disk.

## Convergence

{rhat_line}, against a ≤ 1.01 target. High R² is not trust: read
`cmp_convergence` beside `cmp_gof`, and the ESS column of
`mode_eval_table.csv`.

## Regenerate

One command: `python diagnostics/build_dashboard.py` (`--force <name>` for one
card, `--force-all` for every card, `--png` for static stills → git-ignored
`plots/dashboard/`).
"""
    (CMP_DIR / "README.md").write_text(txt)


if __name__ == "__main__":
    main()
