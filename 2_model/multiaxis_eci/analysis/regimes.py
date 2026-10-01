"""Two-regime frontier forecast: reasoning models against the others, fitted on posterior medians.

The candidates of an axis (`timelines.candidate_mask`) are first reduced to one effort per
family, the one with the highest posterior median on the axis (`family_best`; a family is a
base model and snapshot across its reasoning efforts and run variants), then to one release
per organization and day (`one_per_org_day`: o1-mini and o1-preview, both OpenAI on
2024-09-12, count as one frontier point, the better one, rather than filling a top-2 between
them). They split into two
regimes, reasoning models and the others (`reasoning_mask`, decided per run by `run_mode`: a
run without thinking (`_none`, `_minimal`, a chat or Instant endpoint) is never a reasoning
model; a run with a reasoning level, a thinking budget or a thinking variant always is; a run
with no mode of its own (the bare release) reads its family: reasoning when another run of the
release thinks or its name matches `config.REASONING_FAMILY_PATTERNS`). The bare run of a
HYBRID release, one that runs both with and without thinking, has no determined regime and
sits out of both fits (`regime_ignored`). The reasoning fit set is the running top-k of
posterior-median abilities by release date AMONG reasoning families (`frontier_topk`, k = 2 by
default: a release enters when it is at least the second best of its regime seen so far), so
the reasoning line starts at the first reasoning releases even where non-reasoning models
still stood above them; the others' fit set is the non-reasoning part of the top-k frontier
over every family, so their line stops where reasoning models took the frontier over (user
decision 2026-09-11). Each regime gets its own straight line fitted on the posterior MEDIANS,
each point weighted by its posterior SD plus a common dispersion term integrated out over a
grid (`weighted_line_fit`, analytic, no sampling). The reasoning line is projected forward
from the first reasoning release; the others' line is drawn only over its own points.
Crossing dates read the piecewise frontier: the reasoning line after the switch, the others'
line before it (`regime_crossover_df`).

Fitting medians rather than draws conditions on the fitted axis: the posterior correlation
between points (the axis's own orientation) is not propagated into the trend bands, which is
consistent with drawing the cloud at its medians (user decision 2026-09-10).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

import arviz as az
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from multiaxis_eci.analysis.forecast import ForecastResult, _to_date, _to_year
from multiaxis_eci.analysis.stats import post_stats
from multiaxis_eci.config import (
    HYBRID_LINE_PATTERNS,
    NON_REASONING_PATTERNS,
    REASONING_FAMILY_PATTERNS,
    REGIME_IGNORED,
)
from multiaxis_eci.data import (
    ECIData,
    is_bare,
    model_family,
    model_organization,
    model_product,
    models_table,
)

# Reasoning levels that turn thinking off, and run variants that never think.
OFF_LEVELS = frozenset({"none", "minimal"})
OFF_VARIANTS = frozenset({"chat"})

_PATTERN = re.compile("|".join(REASONING_FAMILY_PATTERNS), re.I)
_NOT = re.compile("|".join(NON_REASONING_PATTERNS), re.I)
_HYBRID = re.compile("|".join(HYBRID_LINE_PATTERNS), re.I)


@lru_cache(maxsize=None)
def run_mode(name: str) -> str:
    """The thinking mode a run states by itself: "off" (`config.NON_REASONING_PATTERNS`, a
    reasoning level of OFF_LEVELS, a variant of OFF_VARIANTS), "on" (any other reasoning level,
    a thinking budget, the thinking variant) or "default" (none stated: the bare release, a
    pro mode without effort, a name the models table does not know)."""
    if _NOT.search(name):
        return "off"
    m = models_table()
    row = m[m["model_version"] == name]
    if row.empty:
        return "default"
    level, tokens, variant = row.iloc[0][["reasoning_level", "reasoning_tokens", "variant"]]
    if level in OFF_LEVELS or variant in OFF_VARIANTS:
        return "off"
    if level or tokens or variant == "thinking":
        return "on"
    return "default"


@lru_cache(maxsize=1)
def _family_modes() -> dict:
    """Release (`data.model_family`) -> the set of run modes its runs state, plus "thinking"
    when one of them is the thinking variant."""
    out: dict[str, set] = {}
    for name, variant in zip(models_table()["model_version"], models_table()["variant"]):
        modes = out.setdefault(model_family(name), set())
        modes.add(run_mode(name))
        if variant == "thinking":
            modes.add("thinking")
    return out


def _reasoning_families() -> frozenset:
    """Releases with a run that thinks (a reasoning level, a budget, the thinking variant)."""
    return frozenset(f for f, modes in _family_modes().items() if "on" in modes)


@lru_cache(maxsize=1)
def hybrid_families() -> frozenset:
    """Releases that run both with and without thinking, so their bare run's mode is unknown:
    a run of theirs turns thinking off, or a separate thinking variant exists (the bare run is
    then the other mode, or a mix), or the release thinks on some runs and belongs to a line
    hybrid by design (`config.HYBRID_LINE_PATTERNS`: Claude's extended thinking, Muse Spark's
    Instant / Thinking modes), whose data need not show an off run."""
    return frozenset(f for f, modes in _family_modes().items()
                     if "off" in modes or "thinking" in modes
                     or ("on" in modes and _HYBRID.search(f)))


def is_reasoning(name: str) -> bool:
    """Whether a test-taker is a reasoning model, decided per run (`run_mode`): off runs never
    are, on runs always are, a run with no stated mode is when its release has a run that
    thinks or its name matches `config.REASONING_FAMILY_PATTERNS`."""
    mode = run_mode(name)
    if mode != "default":
        return mode == "on"
    return model_family(name) in _reasoning_families() or bool(_PATTERN.search(name))


def regime_ignored(name: str) -> bool:
    """Whether a test-taker is left out of both regime fits: `config.REGIME_IGNORED`, plus the
    bare run of every hybrid release (`hybrid_families`), whose scores mix or hide the thinking
    mode (the pipeline files the runs without thinking as `_none` and the thinking ones with
    their effort or `_thinking`, the bare run keeping the scores of unknown mode)."""
    if name in REGIME_IGNORED:
        return True
    return run_mode(name) == "default" and is_bare(name) and model_family(name) in hybrid_families()


def regime_table() -> pd.DataFrame:
    """One row per model of the models table: its regime ("reasoning", "non-reasoning" or
    "undetermined", the last in neither regime fit), its run mode and the evidence that decided.
    Written to 1_curated/model_regimes.csv by 1_curated/4_build_model_regimes.py, the reviewed
    snapshot tests/test_model_metadata.py holds the code to."""
    rows = []
    for name in models_table()["model_version"]:
        mode = run_mode(name)
        if regime_ignored(name):
            regime = "undetermined"
            why = "config.REGIME_IGNORED" if name in REGIME_IGNORED else "bare run of a hybrid release"
        elif mode != "default":
            regime = "reasoning" if mode == "on" else "non-reasoning"
            why = "non-reasoning pattern" if _NOT.search(name) else f"run mode {mode}"
        elif model_family(name) in _reasoning_families():
            regime, why = "reasoning", "a run of the release thinks"
        elif _PATTERN.search(name):
            regime, why = "reasoning", f"pattern {_PATTERN.search(name).group(0)!r}"
        else:
            regime, why = "non-reasoning", "no reasoning evidence"
        rows.append({"model_version": name, "regime": regime, "mode": mode, "evidence": why})
    return pd.DataFrame(rows).sort_values("model_version").reset_index(drop=True)


def reasoning_mask(data: ECIData) -> np.ndarray:
    names = data.mlookup.sort_values("model_idx")["model"].tolist()
    return np.array([is_reasoning(m) for m in names], dtype=bool)


def family_best(tl: pd.DataFrame) -> pd.DataFrame:
    """One row per family (`data.model_product`: a release across its efforts, a pro mode or
    access tier apart): the effort with the highest posterior median (`mean`) on the axis,
    keeping its own name and release date. Ties keep the first."""
    d = tl.assign(_fam=tl["name"].map(model_product))
    d = d.sort_values(["_fam", "mean"], ascending=[True, False], kind="stable")
    return d.drop_duplicates("_fam", keep="first").drop(columns="_fam").reset_index(drop=True)


def one_per_org_day(tl: pd.DataFrame) -> pd.DataFrame:
    """One row per organization and release date: the highest posterior median (`mean`) among
    the releases an organization put out the same day, so simultaneous siblings cannot fill a
    top-k frontier between themselves. Names absent from the models table (no organization)
    are never pooled. Ties keep the first."""
    org = tl["name"].map(model_organization)
    key = np.where(org.to_numpy() == "", tl["name"].to_numpy(),
                   org.to_numpy() + "@" + pd.to_datetime(tl["release_date"]).dt.strftime("%Y-%m-%d").to_numpy())
    d = tl.assign(_key=key).sort_values(["_key", "mean"], ascending=[True, False], kind="stable")
    return d.drop_duplicates("_key", keep="first").drop(columns="_key").reset_index(drop=True)


def frontier_topk(tl: pd.DataFrame, k: int = 2) -> list[str]:
    """Releases in the running top-`k` of posterior medians at their date: the k-th largest
    median among the models released so far (this one included) is not above theirs. k = 1 is
    the record-setter rule (running max, ties included)."""
    d = tl.sort_values("release_date", kind="stable").reset_index(drop=True)
    means = d["mean"].to_numpy()
    return [d.loc[i, "name"] for i in range(len(d))
            if means[i] >= np.sort(means[: i + 1])[-min(k, i + 1)] - 1e-9]


@dataclass
class LineFit:
    """A straight line fitted on medians: posterior samples of intercept `a` (at `t0`) and
    slope `b` per year, the points it used and their time span."""
    a: np.ndarray
    b: np.ndarray
    t0: float
    names: list
    t_min: float
    t_max: float
    dispersion: float = field(default=0.0)

    def at(self, t: np.ndarray) -> np.ndarray:
        """(n_samples, len(t)) line values at absolute years `t`."""
        return self.a[:, None] + self.b[:, None] * (np.asarray(t, float) - self.t0)[None, :]


def weighted_line_fit(t: np.ndarray, y: np.ndarray, sd: np.ndarray, names, *,
                      n_samples: int = 4000, prior_sd=(10.0, 10.0),
                      s_grid=(0.0, 0.05, 0.1, 0.2, 0.35, 0.5), seed: int = 0) -> LineFit:
    """Bayesian straight line through medians `y` at years `t`, noise variance sd_i² + s² with
    the common dispersion s integrated over `s_grid` by marginal likelihood. Weak Gaussian
    priors on the intercept (at the first point) and the slope. Analytic: the posterior is a
    mixture of Gaussians, sampled `n_samples` times."""
    t, y, sd = np.asarray(t, float), np.asarray(y, float), np.asarray(sd, float)
    order = np.argsort(t)
    t, y, sd = t[order], y[order], sd[order]
    t0 = float(t[0])
    X = np.column_stack([np.ones_like(t), t - t0])
    V0 = np.diag(np.asarray(prior_sd, float) ** 2)
    parts = []
    for s in s_grid:
        S = np.diag(sd ** 2 + s ** 2)
        C = X @ V0 @ X.T + S
        lml = multivariate_normal(mean=np.zeros(len(t)), cov=C, allow_singular=True).logpdf(y)
        Si = np.linalg.inv(S)
        Vn = np.linalg.inv(np.linalg.inv(V0) + X.T @ Si @ X)
        bn = Vn @ X.T @ Si @ y
        parts.append((lml, s, bn, Vn))
    lp = np.array([p[0] for p in parts])
    w = np.exp(lp - lp.max())
    w /= w.sum()
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(parts), size=n_samples, p=w)
    ab = np.empty((n_samples, 2))
    for j in np.unique(pick):
        rows = pick == j
        ab[rows] = rng.multivariate_normal(parts[j][2], parts[j][3], size=int(rows.sum()))
    return LineFit(a=ab[:, 0], b=ab[:, 1], t0=t0, names=list(np.asarray(names)[order]),
                   t_min=float(t[0]), t_max=float(t[-1]),
                   dispersion=float(np.sum(w * np.array([p[1] for p in parts]))))


def _band(fit: LineFit, grid: pd.DatetimeIndex, hdi_prob: float):
    F = fit.at(_to_year(grid))
    lo, hi = np.percentile(F, [(1 - hdi_prob) / 2 * 100, (1 + hdi_prob) / 2 * 100], axis=0)
    return np.median(F, axis=0), lo, hi


def two_regime_forecast(theta_draws: np.ndarray, k: int, data: ECIData, tl: pd.DataFrame, *,
                        top_k: int = 2, hdi_prob: float = 0.8,
                        horizon_date="2030-01-01") -> ForecastResult:
    """The two-regime forecast of axis k from the candidates' timeline frame `tl` (name,
    release_date, mean): one effort per family (`family_best`), one release per organization
    and day (`one_per_org_day`), the reasoning fit set as the top-`top_k` frontier among
    reasoning families, the others' as the non-reasoning part of the frontier over every
    family. Returns a ForecastResult on the reasoning line (`fit_basis`
    "regimes"), carrying the others' line as `other`, the switch date (first reasoning
    release) as `switch_year` and both fit sets' union as `frontier_names`. Raises ValueError
    when fewer than three reasoning families qualify."""
    names = data.mlookup.sort_values("model_idx")["model"].tolist()
    idx = {m: i for i, m in enumerate(names)}
    tl = tl[~tl["name"].map(regime_ignored)]   # undetermined regime: in neither fit
    tl = tl.assign(release_date=pd.to_datetime(tl["release_date"]))
    best = one_per_org_day(family_best(tl))
    if best.empty:
        raise ValueError("no frontier candidate")
    # A family's best reasoning run, picked among its reasoning runs: its best run on the axis may
    # be one with reasoning off (`_none`), which joins the others' pool instead.
    reason = one_per_org_day(family_best(tl[tl["name"].map(is_reasoning)]))
    reason = reason[reason["name"].isin(frontier_topk(reason, top_k))].sort_values("release_date")
    front = best[best["name"].isin(frontier_topk(best, top_k))].sort_values("release_date")
    other = front[~front["name"].map(is_reasoning)]
    if len(reason) < 3:
        raise ValueError(f"only {len(reason)} reasoning families on the frontier")

    def fit(sub):
        t = _to_year(pd.DatetimeIndex(sub["release_date"]))
        sd = np.array([theta_draws[:, idx[m], k].std() for m in sub["name"]])
        return weighted_line_fit(t, sub["mean"].to_numpy(), sd, sub["name"].tolist())

    fit_r = fit(reason)
    grid = pd.date_range(reason["release_date"].min(), pd.Timestamp(horizon_date), freq="MS")
    med, lo, hi = _band(fit_r, grid, hdi_prob)
    other_fc = None
    if len(other) >= 3:
        fit_o = fit(other)
        grid_o = pd.date_range(other["release_date"].min(), other["release_date"].max(),
                               freq="MS")
        if len(grid_o) >= 2:
            med_o, lo_o, hi_o = _band(fit_o, grid_o, hdi_prob)
            other_fc = ForecastResult(
                grid_dates=grid_o.values, median=med_o, lo=lo_o, hi=hi_o, slope=fit_o.b,
                intercept=fit_o.a - fit_o.b * fit_o.t0, frontier_names=fit_o.names,
                last_obs_date=other["release_date"].max(), fit_names=fit_o.names,
                fit_basis="regimes", kind="line")
    return ForecastResult(
        grid_dates=grid.values, median=med, lo=lo, hi=hi, slope=fit_r.b,
        intercept=fit_r.a - fit_r.b * fit_r.t0,
        frontier_names=sorted(set(reason["name"]) | set(other["name"])),
        last_obs_date=reason["release_date"].max(), fit_names=fit_r.names, fit_basis="regimes",
        kind="line", other=other_fc, switch_year=fit_r.t_min)


def regime_crossover_df(fc: ForecastResult, theta_draws: np.ndarray, k: int, data: ECIData, *,
                        axis_name: str = "", today=None, hdi_prob: float = 0.5,
                        seed: int = 0) -> pd.DataFrame:
    """Per human tier, when the piecewise frontier reaches it, on the schema of
    `mirt_crossover_df`.

    The frontier is the others' line up to its last fitted release, the reasoning line from
    the switch (the first reasoning release) onwards, and where the two overlap, the higher of
    the two (user decision 2026-09-11: the non-reasoning frontier often stays above the first
    reasoning releases for a while). Per sample of the reasoning line paired with a posterior
    draw of the tier and a paired (intercept, slope) sample of the others' line, the crossing
    is the earliest of: the reasoning line reaching the tier at or after the switch (the switch
    itself when it is already above there; none when its slope is not positive), and the
    others' line reaching the tier at or before its last release (backcast before its first
    point when it was above there all along; none when it never reached the tier before its
    end). `p_passed_now` is the probability the reasoning line exceeds the tier today.
    """
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.today().normalize()
    t_now = float(_to_year(pd.DatetimeIndex([today]))[0])
    names = data.mlookup.sort_values("model_idx")["model"].tolist()
    rng = np.random.default_rng(seed)
    n = len(fc.slope)
    a_r, b_r, t_sw = fc.intercept, fc.slope, float(fc.switch_year)
    other = fc.other
    if other is not None:
        t_end_o = float(_to_year(pd.DatetimeIndex([pd.Timestamp(other.last_obs_date)]))[0])
    rows = []
    for i, m in enumerate(names):
        if not data.is_human[i]:
            continue
        th_all = theta_draws[:, i, k]
        h_mean, h_lo, h_hi = post_stats(th_all, hdi_prob=hdi_prob)
        h = rng.choice(th_all, size=n)
        level_sw = a_r + b_r * t_sw
        f_now = a_r + b_r * t_now
        p_now = float((f_now > h).mean())
        # The reasoning line, from the switch on: the switch itself when already above.
        with np.errstate(divide="ignore", invalid="ignore"):
            on_reason = np.where(b_r > 0, (h - a_r) / b_r, np.nan)
        on_reason = np.where(level_sw >= h, t_sw, on_reason)
        if other is not None:
            # The others' line, up to its last release: a line above the tier at its end
            # reached it at (h - a) / b, or has been above all along when it is not rising.
            j = rng.choice(len(other.slope), size=n)          # paired (intercept, slope) samples
            a_o, b_o = other.intercept[j], other.slope[j]
            with np.errstate(divide="ignore", invalid="ignore"):
                on_other = np.where(b_o > 0, (h - a_o) / b_o, -np.inf)
            on_other = np.where(a_o + b_o * t_end_o >= h, on_other, np.nan)
            cross = np.fmin(on_reason, on_other)               # the earliest, nan-tolerant
        else:
            # One line only: a tier already passed at the switch is backcast on that line.
            with np.errstate(divide="ignore", invalid="ignore"):
                cross = np.where(level_sw >= h, np.where(b_r > 0, (h - a_r) / b_r, -np.inf),
                                 on_reason)
        # A near-flat line sends a crossing centuries away; the calendar range keeps the
        # summaries convertible (such dates read as 'before the data' / 'not this century').
        star = np.clip(cross[~np.isnan(cross)], 1990.0, 2099.0)    # -inf reads as 'before 1990'
        defined = star.size / n
        frac_pos = float((b_r > 0).mean())
        if defined < 0.5 or star.size == 0:
            date_med = date_lo = date_hi = pd.NaT
            status = "no_crossing"
        else:
            lo_yr, hi_yr = az.hdi(np.asarray(star), hdi_prob=hdi_prob)
            date_med = _to_date([np.median(star)])[0]
            date_lo, date_hi = _to_date([lo_yr])[0], _to_date([hi_yr])[0]
            status = ("passed_ci" if p_now >= 0.975 else
                      "passed_mean" if date_med < today else "future")
        rows.append({"axis": axis_name, "tier": m, "human_mean": h_mean,
                     "human_hdi_low": h_lo, "human_hdi_high": h_hi,
                     "crossover_date_median": date_med, "crossover_hdi_low": date_lo,
                     "crossover_hdi_high": date_hi, "p_passed_now": p_now,
                     "frac_positive_slope": frac_pos, "status": status})
    return pd.DataFrame(rows).sort_values("human_mean").reset_index(drop=True)
