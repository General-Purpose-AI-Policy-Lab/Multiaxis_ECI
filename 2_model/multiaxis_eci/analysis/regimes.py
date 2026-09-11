"""Two-regime frontier forecast: reasoning models against the others, fitted on posterior medians.

The candidates of an axis (`timelines.candidate_mask`) are first reduced to one effort per
family, the one with the highest posterior median on the axis (`family_best`; a family is a
base model and snapshot across its reasoning efforts and run variants), then to one release
per organization and day (`one_per_org_day`: o1-mini and o1-preview, both OpenAI on
2024-09-12, count as one frontier point, the better one, rather than filling a top-2 between
them). They split into two
regimes, reasoning models and the others (`reasoning_mask`: a family carries a reasoning
level, a thinking budget or a thinking variant in the models table, or matches
`config.REASONING_FAMILY_PATTERNS`). The reasoning fit set is the running top-k of
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
from multiaxis_eci.config import REASONING_FAMILY_PATTERNS
from multiaxis_eci.data import MODELS_FILE, ECIData, model_family, model_organization

_OFF_LEVELS = {"", "none", "minimal"}


@lru_cache(maxsize=1)
def _reasoning_families() -> frozenset:
    """Families with a reasoning level, a thinking budget or a thinking variant in the table."""
    m = pd.read_csv(MODELS_FILE, dtype=str).fillna("")
    on = m[(~m["reasoning_level"].isin(_OFF_LEVELS)) | (m["reasoning_tokens"] != "")
           | (m["variant"] == "thinking")]
    return frozenset(model_family(x) for x in on["model_version"])


_PATTERN = re.compile("|".join(REASONING_FAMILY_PATTERNS), re.I)


def is_reasoning(name: str) -> bool:
    """Whether a test-taker is a reasoning model: its family shows reasoning evidence in the
    models table, or its name matches one of the reasoning family patterns."""
    return model_family(name) in _reasoning_families() or bool(_PATTERN.search(name))


def reasoning_mask(data: ECIData) -> np.ndarray:
    names = data.mlookup.sort_values("model_idx")["model"].tolist()
    return np.array([is_reasoning(m) for m in names], dtype=bool)


def family_best(tl: pd.DataFrame) -> pd.DataFrame:
    """One row per family (`data.model_family`): the effort with the highest posterior median
    (`mean`) on the axis, keeping its own name and release date. Ties keep the first."""
    d = tl.assign(_fam=tl["name"].map(model_family))
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
    best = one_per_org_day(family_best(tl.assign(release_date=pd.to_datetime(tl["release_date"]))))
    if best.empty:
        raise ValueError("no frontier candidate")
    is_r = best["name"].map(is_reasoning)
    reason = best[is_r]
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
        star = np.clip(cross[np.isfinite(cross)], 1990.0, 2099.0)
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
