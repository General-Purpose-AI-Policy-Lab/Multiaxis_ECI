"""Human tiers placed axis by axis on a fit WITHOUT human baselines (the "2b" placement).

The flagship is fitted on the models alone (user, 2026-10-08): fitted with the human tiers, the
compensatory MIRT put them about 5 sd below the models on one axis and 8 above on another, one
error paying for the other, and no prior, extra axis or link fixed it (docs/model_math.md). The
tiers are placed afterwards, one axis at a time, with the benchmark side frozen:

* the benchmarks of axis k are those whose largest share of loading variance is on k
  (`dominant_axis`), minus those whose content has nothing to do with the axis title
  (`1_curated/human_axis_content.csv`, judged pair by pair; a plausible link keeps a pair);
  a row of that file naming a tier takes that tier's score alone off the axis, the last resort
  when it splits the placement in two (`tier_exclusions`);
* on those benchmarks a tier's score follows the fit's own likelihood at the medians of the
  loadings, difficulties and noise, with its other axes set to the profile of a typical model at
  the same ability on k (a linear regression of the other axes on axis k over the models);
* the tiers keep their hard partial order (config.HUMAN_ORDER_MERGED, the committee chain
  included) with the flagship's human priors, so one tier's scores inform its neighbours.

A tier is drawn when it has a score of its own below the ceiling on one of those benchmarks, or
sits between two such tiers in the order (any ancestor and any descendant); a ceiling score stays
in the likelihood but only bounds a tier from below. `consistency` checks each placement against
the scores it rests on and against the first model that beat each score.

    python 4_diagnostics/5_place_humans.py [--trace T]      # writes the placement beside the trace
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from multiaxis_eci.config import (
    CONTINUOUS_EPS,
    CURATED_DIR,
    PROJECT_ROOT,
    HUMAN_ORDER_MERGED,
    PRIOR_DELTA_HUMAN,
    PRIOR_HUMAN_ROOT,
)

CONTENT_FILE = CURATED_DIR / "human_axis_content.csv"
# A score at or above this only bounds a tier from below (the benchmark's own ceiling, 1 for
# every benchmark with a human score at the ceiling today).
CEILING_SCORE = 0.99
PLACEMENT_JSON = "human_placement.json"
PLACEMENT_DRAWS = "human_placement_draws.npz"
# The secondary mode's placement (the modes file's `secondary_chains`), in its own folder beside
# the trace so the primary one stays where every deliverable reads it.
SECONDARY_DIR = "secondary_mode"


# ── inputs ───────────────────────────────────────────────────────────────────────────────────

def _content_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False)
    df["keep"] = df["keep"].str.strip().str.lower() == "yes"
    return df


def content_rules(path: Path = CONTENT_FILE) -> dict[str, dict[str, bool]]:
    """{axis title: {benchmark: keep}} from the curated content file's whole-pair rows (no tier)."""
    out: dict[str, dict[str, bool]] = {}
    for r in _content_table(path).query("tier == ''").itertuples():
        out.setdefault(r.axis_title, {})[r.benchmark] = r.keep
    return out


def tier_exclusions(path: Path = CONTENT_FILE) -> dict[str, set[tuple[str, str]]]:
    """{axis title: {(benchmark, tier)}}: the scores of one tier the content file takes off an axis
    whose placement they stopped converging (a last resort, the pair stays judged as a whole)."""
    out: dict[str, set[tuple[str, str]]] = {}
    for r in _content_table(path).query("tier != '' and not keep").itertuples():
        out.setdefault(r.axis_title, set()).add((r.benchmark, r.tier))
    return out


def dominant_axis(A_med: np.ndarray) -> np.ndarray:
    """(B,) index of the axis carrying the largest share of each benchmark's loading variance."""
    return np.argmax(A_med ** 2, axis=1)


def human_scores(spec) -> pd.DataFrame:
    """The human baselines of the fit's data scope, read with the humans in: tier, benchmark,
    score, n_items (0 for a Beta-scored benchmark)."""
    spec_h = dataclasses.replace(spec, no_humans=False)
    data_h = spec_h.load_data()[0]
    names = data_h.mlookup.sort_values("model_idx")["model"].tolist()
    bench = data_h.blookup.sort_values("benchmark_idx")["benchmark"].tolist()
    i = np.where(np.asarray(data_h.is_human)[data_h.model_idx])[0]
    bn = spec_h.binomial_n(data_h)
    n = (np.where(np.isfinite(bn[i]) & (bn[i] > 0), bn[i], 0.0) if bn is not None
         else np.zeros(len(i)))
    return pd.DataFrame({"tier": [names[data_h.model_idx[j]] for j in i],
                         "benchmark": [bench[data_h.bench_idx[j]] for j in i],
                         "score": data_h.scores[i], "n_items": n})


def _ancestors(tier: str, order=HUMAN_ORDER_MERGED) -> set[str]:
    p = order.get(tier)
    ps = () if p is None else (p,) if isinstance(p, str) else tuple(p)
    out = set(ps)
    for q in ps:
        out |= _ancestors(q, order)
    return out


def drawn_tiers(scores: pd.DataFrame, tiers: list[str]) -> list[str]:
    """The display rule: an own score below the ceiling, or between two tiers that have one."""
    own = set(scores.loc[scores.score < CEILING_SCORE, "tier"])
    return [t for t in tiers if t in own
            or (_ancestors(t) & own and any(t in _ancestors(d) for d in own))]


# ── placement ────────────────────────────────────────────────────────────────────────────────

def _profile(theta_models: np.ndarray, k: int):
    """Regression of the other axes on axis k over the models: (intercepts, slopes), (K-1,) each."""
    others = [j for j in range(theta_models.shape[1]) if j != k]
    X = np.column_stack([np.ones(len(theta_models)), theta_models[:, k]])
    coef = np.linalg.lstsq(X, theta_models[:, others], rcond=None)[0]
    return others, coef


def _tier_theta(th, k, K, profile, xp):
    """(n, K) abilities of tiers at axis-k ability `th` (n,), the other axes at the typical-model
    profile; `xp` is numpy or pytensor.tensor."""
    others, coef = profile
    cols = {k: th}
    for j, o in enumerate(others):
        cols[o] = coef[0][j] + coef[1][j] * th
    return xp.stack([cols[i] for i in range(K)], axis=-1)


def place_axis(k, scores, bench, A, D, phi, floor_c, theta_models, tiers, seed=81, beta=None,
               ceiling_d=None):
    """Posterior draws (S, n_tiers) of the tiers on axis k from `scores` (rows of
    `human_scores` restricted to the axis's benchmarks), and the sampler's convergence. `beta`
    (the softmin links' sharpness, fixed at the fit's median) swaps the compensatory eta for the
    softmin one (`models.mirt.softmin_eta`, no offset: semicomp0). `ceiling_d` (per benchmark, the
    fit's median d_b under --ceiling-noise) caps the curve as the fit does: mu = c + (d - c) sigmoid."""
    import arviz as az
    import pymc as pm
    import pytensor.tensor as pt

    from multiaxis_eci.models.mirt import softmin_eta

    from multiaxis_eci.models.mirt import _beta_binomial_logp, _human_block, _human_structure

    struct = _human_structure(HUMAN_ORDER_MERGED, tiers)
    order = [tiers[r] for r in struct[0]]
    pos = {t: i for i, t in enumerate(order)}
    others, coef = _profile(theta_models, k)
    b = np.array([bench.index(x) for x in scores.benchmark])
    t = np.array([pos[x] for x in scores.tier])
    y = scores.score.values
    yc = np.clip(y, CONTINUOUS_EPS, 1 - CONTINUOUS_EPS)
    n = scores.n_items.values
    kk = np.clip(y, 0, 1) * n
    c = floor_c[b] if floor_c is not None else np.zeros(len(b))
    d = ceiling_d[b] if ceiling_d is not None else np.ones(len(b))
    slope = A[b, k] + A[b][:, others] @ coef[1]
    icpt = A[b][:, others] @ coef[0] - D[b]
    with pm.Model():
        base = pm.Normal("base", 0.0, PRIOR_HUMAN_ROOT, shape=(struct[3], 1))
        _, th = _human_block(struct, 1, base, PRIOR_DELTA_HUMAN)
        th = pm.Deterministic("th", th[:, 0])
        if beta is None:
            eta = slope * th[t] + icpt
        else:
            eta = softmin_eta(A[b], _tier_theta(th[t], k, A.shape[1], (others, coef), pt), D[b],
                              beta, lse=pt.logsumexp, log=pt.log)
        mu = c + (d - c) * pm.math.sigmoid(eta)
        pm.Potential("lik", _beta_binomial_logp(yc, mu * phi[b], (1 - mu) * phi[b], n, kk).sum())
        tr = pm.sample(1000, tune=1000, chains=4, nuts_sampler="nutpie", random_seed=seed,
                       progressbar=False, target_accept=0.95)
    s = az.summary(tr, var_names=["th"])
    conv = dict(rhat=float(s.r_hat.max()), ess_bulk=float(s.ess_bulk.min()),
                ess_tail=float(s.ess_tail.min()),
                divergences=int(tr.sample_stats["diverging"].sum()), n_scores=int(len(scores)))
    return order, tr.posterior["th"].values.reshape(-1, len(order)), conv, (others, coef)


def consistency(k, scores, bench, A, D, floor_c, order, draws, profile, data, dates, beta=None,
                ceiling_d=None):
    """Per score a placement used: observed, predicted (median and 80% interval at the tier's
    draws), and the release date of the first model that reached the score."""
    from scipy.special import expit
    others, coef = profile
    rows = []
    for r in scores.itertuples():
        b = bench.index(r.benchmark)
        th = draws[:, order.index(r.tier)]
        if beta is None:
            eta = (A[b, k] + A[b, others] @ coef[1]) * th + A[b, others] @ coef[0] - D[b]
        else:
            from multiaxis_eci.models.mirt import softmin_eta
            eta = softmin_eta(A[b], _tier_theta(th, k, A.shape[1], profile, np), D[b], beta)
        c = floor_c[b] if floor_c is not None else 0.0
        d = ceiling_d[b] if ceiling_d is not None else 1.0
        pred = c + (d - c) * expit(eta)
        sel = data.bench_idx == b
        hit = dates.iloc[data.model_idx[sel][data.scores[sel] >= r.score]].dropna()
        rows.append({"tier": r.tier, "benchmark": r.benchmark, "observed": round(float(r.score), 3),
                     "predicted": round(float(np.median(pred)), 3),
                     "predicted_q10": round(float(np.quantile(pred, 0.1)), 3),
                     "predicted_q90": round(float(np.quantile(pred, 0.9)), 3),
                     "first_model_at_score": hit.min().strftime("%Y-%m") if len(hit) else None})
    return rows


def run_placement(trace: Path, view, data, spec, titles: dict, floor_c,
                  out_dir: Path | None = None, beta: float | None = None,
                  rule_titles: dict | None = None, ceiling_d: np.ndarray | None = None) -> dict:
    """Place every axis's tiers; write `human_placement.json` and its draws in `out_dir` (default:
    beside the trace). `view` must be the aligned view the figures read (same axis order and
    names). `beta`: a semicomp0 fit's median sharpness (softmin eta). `rule_titles` maps an axis
    title to the title whose content judgments it borrows (an exploration fit naming its axes
    otherwise). `ceiling_d`: a --ceiling-noise fit's median ceilings, per benchmark."""
    if np.asarray(data.is_human).any():
        raise ValueError("the 2b placement reads a fit without human baselines (--no-humans)")
    rules, dropped = content_rules(), tier_exclusions()
    hs = human_scores(spec)
    bench = data.blookup.sort_values("benchmark_idx")["benchmark"].tolist()
    missing = sorted(set(hs.benchmark) - set(bench))
    if missing:
        raise ValueError(f"human-scored benchmarks missing from the fit: {missing}")
    A = np.median(view.require_A(), 0)
    D, phi = view.D_med, view.phi_med
    theta_models = np.median(view.theta, 0)
    dom = dominant_axis(A)
    tiers = [t for t in HUMAN_ORDER_MERGED if t in set(hs.tier)]
    s = pd.read_csv(PROJECT_ROOT / "0_input" / "all_scores_flat.csv")
    first = s.groupby("model_version").release_date.min()
    names_m = data.mlookup.sort_values("model_idx")["model"].tolist()
    dates = pd.to_datetime(pd.Series([first.get(n) for n in names_m]))
    out = {"trace": trace.name, "rule": "dominant axis (largest loading-variance share), "
           "content filter 1_curated/human_axis_content.csv, ordered tiers (HUMAN_ORDER_MERGED), "
           "other axes at the typical-model profile; drawn: own score below the ceiling or "
           "between two such tiers", "softmin_beta": beta,
           "ceiling_d": ceiling_d is not None, "axes": []}
    draws_out = {}
    for k, name in enumerate(view.names):
        title = titles.get(name, name).partition(": ")[2] or None
        on_axis = {bench[b] for b in np.where(dom == k)[0]}
        # an axis's own judgments over those it borrows
        judged = ({**rules.get((rule_titles or {}).get(title, title), {}), **rules.get(title, {})}
                  if title else {})
        used = sorted(b for b in on_axis if judged.get(b, True)) if title else []
        unjudged = sorted(b for b in on_axis if b not in judged and b in set(hs.benchmark)) \
            if title else []
        off = ({*dropped.get((rule_titles or {}).get(title, title), ()), *dropped.get(title, ())}
               if title else set())
        sc = hs[hs.benchmark.isin(used) & ~pd.Series(list(zip(hs.benchmark, hs.tier)),
                                                      index=hs.index).isin(off)]
        ax = {"axis": name, "title": title, "benchmarks_with_human_scores": sorted(set(sc.benchmark)),
              "unjudged_benchmarks": unjudged, "drawn": [], "tiers": {}, "convergence": None,
              "consistency": []}
        if title and len(sc):
            order, draws, conv, prof = place_axis(k, sc, bench, A, D, phi, floor_c, theta_models,
                                                  tiers, beta=beta, ceiling_d=ceiling_d)
            ax["drawn"] = drawn_tiers(sc, order)
            ax["tiers"] = {t: [round(float(q), 3) for q in np.quantile(draws[:, i], [0.1, 0.5, 0.9])]
                           for i, t in enumerate(order)}
            ax["convergence"] = conv
            ax["consistency"] = consistency(k, sc, bench, A, D, floor_c, order, draws, prof,
                                            data, dates, beta=beta, ceiling_d=ceiling_d)
            ax["order"] = order
            draws_out[name] = draws
        out["axes"].append(ax)
    out_dir = Path(out_dir) if out_dir is not None else trace.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / PLACEMENT_JSON).write_text(json.dumps(out, indent=1))
    np.savez(out_dir / PLACEMENT_DRAWS, _A_med=A, **draws_out)
    return out


# ── reading it back ─────────────────────────────────────────────────────────────────────────

def load_placement(fit_dir: Path) -> dict | None:
    """The placement written beside a fit, its draws attached per axis, or None."""
    j, z = fit_dir / PLACEMENT_JSON, fit_dir / PLACEMENT_DRAWS
    if not (j.exists() and z.exists()):
        return None
    doc = json.loads(j.read_text())
    with np.load(z) as npz:
        for ax in doc["axes"]:
            ax["draws"] = npz[ax["axis"]] if ax["axis"] in npz.files else None
        doc["A_med"] = npz["_A_med"]
    return doc


def attach_placement(view, fit_dir: Path):
    """The view with the fit's placement attached when the fit has one (figures read it). The
    placement was computed on one view of the fit (its selected chains, on its display frame);
    a view whose axes differ (another chain group, another order) gets none, with a warning."""
    doc = load_placement(fit_dir)
    if doc is None or view.A is None:
        return view
    A = np.median(view.A, 0)
    same = (A.shape == doc["A_med"].shape and
            min(np.corrcoef(A[:, k], doc["A_med"][:, k])[0, 1] for k in range(A.shape[1])) > 0.95)
    if [a["axis"] for a in doc["axes"]] != list(view.names) or not same:
        import warnings
        warnings.warn(f"{fit_dir.name}: the human placement was made on other axes than this "
                      "view's; no tier drawn (rerun 4_diagnostics/5_place_humans.py)")
        return view
    return dataclasses.replace(view, human_placement=doc)


def placement_axis(view, k):
    """(drawn tier names, their draws (S, n)) on axis k, or ([], None)."""
    doc = getattr(view, "human_placement", None)
    if not doc:
        return [], None
    ax = doc["axes"][k]
    if ax.get("draws") is None or not ax["drawn"]:
        return [], None
    idx = [ax["order"].index(t) for t in ax["drawn"]]
    return list(ax["drawn"]), ax["draws"][:, idx]


def placement_frame(view, k, hdi_prob: float = 0.8) -> pd.DataFrame:
    """The drawn tiers on axis k as the figures' human-tier frame (name, mean, hdi_low,
    hdi_high, n_obs)."""
    import arviz as az
    tiers, draws = placement_axis(view, k)
    rows = []
    for i, t in enumerate(tiers):
        lo, hi = az.hdi(draws[:, i], hdi_prob=hdi_prob)
        rows.append({"name": t, "mean": float(np.mean(draws[:, i])), "hdi_low": float(lo),
                     "hdi_high": float(hi), "n_obs": 0})
    return pd.DataFrame(rows, columns=["name", "mean", "hdi_low", "hdi_high", "n_obs"]) \
        .sort_values("mean").reset_index(drop=True)


def placement_crossings(view, k, fc, axis_name, probs=(0.5, 0.8)) -> pd.DataFrame:
    """The trend crossing of each drawn tier on axis k (`crossover_table` on a stand-in data
    holding the tiers alone)."""
    from multiaxis_eci.analysis.forecast import crossover_table
    tiers, draws = placement_axis(view, k)
    if not tiers:
        return pd.DataFrame()
    stub = SimpleNamespace(mlookup=pd.DataFrame({"model": tiers, "model_idx": range(len(tiers))}),
                           is_human=np.ones(len(tiers), bool))
    return crossover_table(fc, draws[:, :, None], 0, stub, axis_name, probs=probs)
