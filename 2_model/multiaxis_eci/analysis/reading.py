"""Reading an axis on predicted scores instead of on the fitted ability.

A fitted ability theta_k is one coordinate of a sum, A_b . theta - D_b, and the data pin the sum
far better than its parts: a model strong on three axes and scored once on a benchmark that
loads on the fourth can be handed a low fourth ability and a slightly higher other one at no
cost to the fit (compensation). Read on theta_k alone, the frontier of an axis then holds
models that are not the best on any of its benchmarks and misses those that are.

The predicted-score reading asks the question the axis is named after: what is this test-taker
predicted to score on the benchmarks this axis dominates? For axis k, over the benchmarks whose
largest share of loading variance is on k (the placement's own rule, `dominant_axis`),

    score_k(m) = sum_b w_b * (A_b . theta_m - D_b - mu_b) / sigma_b  /  sum_b w_b,

with w_b the benchmark's share on k and (mu_b, sigma_b) the mean and spread of its predictor
over the models, so every benchmark counts on a common scale whatever its slope. The result is
then centred and scaled over the models (mean 0, sd 1), the scale the abilities are on. It is
linear in theta, score_k = theta . v_k - c_k: the ability along the direction the axis's
benchmarks point to, not along the axis alone. Only products A . theta enter, so it does not
change when abilities move between axes while the predicted scores stay put.

A human tier is carried to the same scale by the same map: a tier fitted with the models is a
row of theta like any other; a tier placed on a fit without humans (`analysis.human_placement`)
is, by that placement's own assumption, a typical model at its ability on the axis, so its
predicted scores are those of that profile, with the benchmark side at its medians.

`predicted_view` hands back a `FitView` whose `theta` (and placed tiers) hold this reading, so
every ability-side figure (timelines, forecasts, crossings, forests) reads it unchanged. The
loadings, the item curves and the tables keep the fitted abilities.
"""
from __future__ import annotations

import copy
import dataclasses

import numpy as np

from multiaxis_eci.config import AXIS_READING

READINGS = ("predicted", "theta")
# Short enough for a per-axis caption; the scale (logit, standard deviations of the models,
# mean 0) is said in docs/plots.md, as the fitted scale's is.
PREDICTED_LABEL = {
    "en": "predicted score on the axis's benchmarks",
    "fr": "score prédit sur les benchmarks de l'axe",
}


def dominant_benchmarks(A_med: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(B,) dominant axis of each benchmark and (B, K) axis shares, from median loadings: the
    rule of `human_placement.dominant_axis`, on the axes a benchmark rises with."""
    pos = np.maximum(A_med, 0) ** 2
    share = pos / np.maximum(pos.sum(axis=1, keepdims=True), 1e-12)
    return np.argmax(pos, axis=1), share


def axis_score_maps(A_med: np.ndarray, theta_med: np.ndarray, D_med: np.ndarray,
                    is_human: np.ndarray | None = None) -> list[dict | None]:
    """Per axis, what `axis_scores` needs: `bench` (indices of the benchmarks the axis
    dominates), `w` (their weights, share / sigma_b, summing to 1 before the division by
    sigma_b), `mu` (their mean predictor over the models), and the medians' own direction and
    offset, `v` (K,) and `c`, with which a tier placed outside the fit is read. None for an
    axis that dominates no benchmark. Human rows are left out of every mean and spread."""
    models = (np.ones(len(theta_med), bool) if is_human is None
              else ~np.asarray(is_human, dtype=bool))
    dom, share = dominant_benchmarks(A_med)
    maps = []
    for k in range(A_med.shape[1]):
        bs = np.where(dom == k)[0]
        if not len(bs):
            maps.append(None)
            continue
        eta = theta_med[models] @ A_med[bs].T - D_med[bs]            # (models, nb)
        mu, sd = eta.mean(0), np.maximum(eta.std(0), 1e-9)
        w = share[bs, k] / share[bs, k].sum() / sd
        maps.append({"bench": bs, "w": w, "mu": mu, "v": w @ A_med[bs],
                     "c": float((D_med[bs] + mu) @ w)})
    return maps


def _raw_scores(theta: np.ndarray, A: np.ndarray, D: np.ndarray, m: dict) -> np.ndarray:
    """(S, M) weighted mean of the standardised predictors of axis map `m`, draw by draw."""
    v = np.einsum("b,sbk->sk", m["w"], A[:, m["bench"]])              # (S, K)
    c = (D[:, m["bench"]] + m["mu"]) @ m["w"]                         # (S,)
    return np.einsum("smk,sk->sm", theta, v) - c[:, None]


def axis_scores(view, is_human: np.ndarray | None = None):
    """((S, M, K) predicted-score reading of `view`, its per-axis maps with the centring `m`
    and scale `s` added). Human rows are read by the same map and left out of the centring."""
    A, th, D = view.A, view.theta, view.D       # in the trace's precision: (S, M, K) is large
    models = (np.ones(th.shape[1], bool) if is_human is None
              else ~np.asarray(is_human, dtype=bool))
    maps = axis_score_maps(*(np.median(x, 0).astype(float) for x in (A, th, D)), is_human)
    out = np.array(th, copy=True)
    for k, m in enumerate(maps):
        if m is None:           # an axis no benchmark leans on most keeps its ability
            continue
        raw = _raw_scores(th, A, D, m)
        med = np.median(raw, 0)[models]
        m["m"], m["s"] = float(med.mean()), float(max(med.std(), 1e-9))
        out[:, :, k] = (raw - m["m"]) / m["s"]
    return out, maps


def _placed_tiers(view, maps) -> dict:
    """The placement of `view` with every tier's draws carried to the predicted-score scale:
    a tier at ability t on axis k is the typical model at t (`human_placement._profile`), read
    by the medians' map of that axis."""
    from multiaxis_eci.analysis.human_placement import _profile, _tier_theta
    theta_med = np.median(view.theta, 0)
    doc = copy.copy(view.human_placement)
    doc["axes"] = [dict(ax) for ax in doc["axes"]]
    for k, ax in enumerate(doc["axes"]):
        m = maps[k] if k < len(maps) else None
        if ax.get("draws") is None or m is None:
            continue
        prof = _profile(theta_med, k)
        th = np.asarray(ax["draws"], dtype=float)
        ax["draws"] = np.column_stack([
            ((_tier_theta(th[:, i], k, view.K, prof, np) @ m["v"] - m["c"]) - m["m"]) / m["s"]
            for i in range(th.shape[1])])
    return doc


def has_predicted_reading(view) -> bool:
    """Whether `view` can be read on predicted scores: a compensatory fit with a linear
    predictor (its difficulty draws are on the view), at least two axes, in its raw frame
    (a signed fit's display frame is a per-draw rotation its difficulties do not follow)."""
    return (not view.is_nc and view.A is not None and view.K >= 2 and view.D is not None
            and not view.phi_is_promax and view.D.shape[0] == view.theta.shape[0])


def predicted_view(view, data):
    """`view` with `theta` and the placed human tiers on the predicted-score reading, marked
    `reading="predicted"`; `view` itself where that reading does not exist
    (`has_predicted_reading`) or is already applied."""
    if view.reading == "predicted" or not has_predicted_reading(view):
        return view
    scores, maps = axis_scores(view, np.asarray(data.is_human, dtype=bool))
    changes = {"theta": scores, "reading": "predicted", "theta_fit": view.theta}
    if view.human_placement is not None:
        changes["human_placement"] = _placed_tiers(view, maps)
    return dataclasses.replace(view, **changes)


def reading_view(view, data, reading: str | None = None):
    """`view` under `reading` ("predicted" or "theta"; default `config.AXIS_READING`)."""
    reading = reading or AXIS_READING
    if reading not in READINGS:
        raise ValueError(f"reading must be one of {READINGS}, got {reading!r}")
    if reading == "theta":
        if view.reading != "theta":
            raise ValueError("a view on the predicted-score reading cannot be read back on "
                             "theta: keep the fitted view")
        return view
    return predicted_view(view, data)


PREDICTED_HUMAN_LABEL = {
    "en": "predicted score on the axis's benchmarks ({0} = 0, {1} = 1)",
    "fr": "score prédit sur les benchmarks de l'axe (Humain médian = 0, Meilleur performeur = 1)",
}


def value_label(view, fallback: str | None = None, lang: str = "en",
                human_units: bool = False) -> str | None:
    """The caption of an ability-side axis under the view's reading: the predicted-score one
    (in human units where the caller re-expressed the axis, `analysis.scale`), or `fallback`,
    the caller's ability caption, on the fitted abilities."""
    if view.reading != "predicted":
        return fallback
    if human_units:
        from multiaxis_eci.config import HUMAN_UNIT_ANCHORS
        return PREDICTED_HUMAN_LABEL[lang].format(*HUMAN_UNIT_ANCHORS)
    return PREDICTED_LABEL[lang]
