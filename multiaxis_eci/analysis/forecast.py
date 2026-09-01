"""Frontier forecasting: when the AI frontier crosses each human tier."""
from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
from dataclasses import dataclass

from multiaxis_eci.analysis.stats import _release_dates, post_stats
from multiaxis_eci.analysis.timelines import mirt_informed_mask
from multiaxis_eci.data import ECIData

# ── Frontier forecasting (when does the AI frontier outpace human tiers) ─────

_YEAR0 = pd.Timestamp("2000-01-01")


def _to_year(dates) -> np.ndarray:
    """Datetime(s) → decimal years (a smooth x-axis for the linear trend)."""
    days = (pd.to_datetime(dates) - _YEAR0) / pd.Timedelta(days=1)
    return 2000.0 + np.asarray(days, dtype=float) / 365.25


def _to_date(years) -> pd.Series:
    """Inverse of _to_year: decimal years → Timestamps."""
    return _YEAR0 + pd.to_timedelta((np.asarray(years) - 2000.0) * 365.25, unit="D")


@dataclass
class ForecastResult:
    """Per-draw linear extrapolation of a single axis's frontier.

    `slope`/`intercept` are (S,) — one latent-scale line per posterior draw, so
    both the model-ability posterior AND (downstream) the human posterior feed
    the crossover uncertainty. `grid_dates` + `median`/`lo`/`hi` are the forecast
    band, `lo`/`hi` being the hdi_prob HDI per grid point (default 50%) (same summary as the human
    bands, via `az.hdi`). `frontier_names` are the frontier-envelope models
    (record-setters ∪ SOTA) — always the highlight set; `fit_names` are the
    models the OLS was actually regressed on (= frontier for `fit_basis`
    "frontier", = every informed candidate ∪ SOTA for "informed", = the
    caller's list for "frozen").
    """
    grid_dates: np.ndarray
    median: np.ndarray
    lo: np.ndarray
    hi: np.ndarray
    slope: np.ndarray
    intercept: np.ndarray
    frontier_names: list
    last_obs_date: pd.Timestamp
    fit_names: list | None = None
    fit_basis: str = "frontier"
    # Envelope extras (kind == "envelope"): the per-draw running-max frontier.
    # `env_x` (F,) are the candidate years sorted ascending, `env_E` (S, F) the
    # cumulative max of theta over them, `slope_early` (S,) the envelope's
    # growth rate over its FIRST `rate_window` years (used to backcast), and
    # `backcast_floor` the earliest year a backcast crossing may name (None =
    # no backcast: a tier already passed at window start is censored there).
    # `slope`/`intercept` hold the FORWARD extension (recent rate anchored at
    # the last record), so every linear consumer works unchanged for t >= end.
    kind: str = "linear"
    env_x: np.ndarray | None = None
    env_E: np.ndarray | None = None
    slope_early: np.ndarray | None = None
    backcast_floor: float | None = None


def _weighted_median_rows(v: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Row-wise weighted median of `v` (S, P) under shared weights `w` (P,):
    the smallest value whose cumulative weight reaches half the total."""
    order = np.argsort(v, axis=1)
    cw = np.cumsum(w[order], axis=1)
    pick = (cw >= 0.5 * w.sum()).argmax(axis=1)
    return np.take_along_axis(np.take_along_axis(v, order, 1),
                              pick[:, None], 1)[:, 0]


def _theilsen_slopes(x: np.ndarray, y: np.ndarray,
                     w: np.ndarray | None) -> np.ndarray:
    """Per-draw Theil-Sen slope: the (weighted) median over all pairwise
    slopes (y_j - y_i)/(x_j - x_i). `x` (F,), `y` (S, F); pair weights are
    w_i * w_j. Ties in release date are excluded (dx = 0)."""
    i, j = np.triu_indices(len(x), 1)
    dx = x[j] - x[i]
    keep = np.abs(dx) > 1e-9
    i, j, dx = i[keep], j[keep], dx[keep]
    sp = (y[:, j] - y[:, i]) / dx                          # (S, P)
    if w is None:
        return np.median(sp, axis=1)
    return _weighted_median_rows(sp, w[i] * w[j])


def mirt_frontier_forecast(theta_draws: np.ndarray, k: int, data: ECIData,
                           raw_df: pd.DataFrame, *,
                           horizon_date: str | pd.Timestamp | None = None,
                           fit_start: str = "2023-01-01",
                           back_start: str | pd.Timestamp | None = None,
                           sd_cap: float | None = 0.4,
                           drop_low_obs: bool = True,
                           sota_exempt: bool = True,
                           fit_basis: str = "frontier",
                           fit_names: list[str] | None = None,
                           weights: str | None = None,
                           estimator: str = "ols",
                           backcast_floor=None,
                           rate_window: float = 1.5,
                           hdi_prob: float = 0.5) -> ForecastResult:
    """Extrapolate axis-k's capability trend linearly in time.

    Candidate models use the same guard as `mirt_model_timeline_df` (dated,
    non-human, informed/low-obs filters with the SOTA exemption). The frontier
    ENVELOPE is the record-setters of the posterior-MEDIAN trajectory (kept
    stable across draws) UNION the SOTA models.

    `fit_basis` picks what the per-draw OLS is regressed on:
      - "frontier" (default): the frontier envelope only — a running-best
        trajectory, so the projection tracks the record, not the average.
      - "records": the running-max record-setters only, WITHOUT the SOTA
        union — the pure per-axis frontier. Use when a saturated axis makes
        the SOTA union misleading (recent general SOTA models sit below the
        old records there and tilt the line downward). With `sota_exempt`
        a SOTA release is a CANDIDATE however thin its evaluation, but it
        joins the OLS only by breaking the median record like any other model.
      - "informed": EVERY candidate (all informed models ∪ SOTA), a denser,
        lower-variance fit that tracks the typical model rather than the record.
        The frontier envelope is still returned (for highlighting) but does not
        alone drive the line.
      - "envelope": no regression at all. Per draw, the frontier is the
        RUNNING MAX of theta over every candidate — non-decreasing by
        definition, so a draw can never carry a negative trend. Crossings
        inside the window are the observed record-step dates; ahead of it the
        envelope extends linearly at its growth rate over the last
        `rate_window` years; behind it (a tier already passed at window
        start) it backcasts at the rate over its FIRST `rate_window` years,
        clamped at `backcast_floor` — or is censored at the window start when
        no floor is given. The `weights`/`estimator` knobs do not apply.
    Either way the OLS is per-draw, so ability uncertainty propagates into the
    slope/intercept, hence into the forecast band and every crossover date.

    Pass `fit_names` to freeze the fit set: the OLS runs on exactly those models
    and `fit_basis` reads "frozen". Use it when the same set must be compared
    across subsets of the posterior, which the SD mask would otherwise re-pick
    for each one.

    `weights="precision"` turns the per-draw OLS into a WLS with FIXED weights
    1/SD²(θ_m,k) (posterior SD on this axis): an early record measured wide
    keeps its place in the fit but loses the leverage that lets it flip the
    slope draw by draw — the pathology behind negative-slope draws and
    decade-long crossover tails. None (default) keeps the unweighted OLS.

    `estimator="theilsen"` replaces the per-draw least-squares line with the
    per-draw Theil-Sen line: the slope is the median of all pairwise slopes
    (weighted median with pair weights w_i*w_j under `weights="precision"`),
    the intercept the (weighted) median of y - slope*x. A single aberrant
    point changes at most half the pairs, so it cannot flip the slope's sign
    the way it moves a mean-based fit. Composes with `weights`.
    """
    if fit_basis not in ("frontier", "records", "informed", "envelope"):
        raise ValueError(f"fit_basis must be 'frontier', 'records', 'informed' "
                         f"or 'envelope', got {fit_basis!r}")
    model_dates, _ = _release_dates(raw_df)
    names = data.mlookup.sort_values("model_idx")["model"].tolist()
    if fit_names is not None:
        # Frozen fit set: the caller already chose the models, so no candidate
        # walk and no selection filters. Everything downstream is unchanged.
        pos = {m: i for i, m in enumerate(names)}
        for m in fit_names:
            if m not in pos:
                raise ValueError(f"fit_names: {m!r} is not in the fitted data")
            if m not in model_dates.index:
                raise ValueError(f"fit_names: {m!r} has no release date")
        idx = np.array([pos[m] for m in fit_names])
        dates = pd.to_datetime(pd.Series([model_dates[m] for m in fit_names]).values)
        front = frontier = np.ones(len(idx), dtype=bool)
        fit_basis = "frozen"
    else:
        informed = (mirt_informed_mask(theta_draws, sd_cap)[:, k] if sd_cap is not None
                    else np.ones(data.n_models, dtype=bool))
        sota = (data.is_sota if data.is_sota is not None
                else np.zeros(data.n_models, dtype=bool))

        idx, dates = [], []
        for i, m in enumerate(names):
            if data.is_human[i] or m not in model_dates.index:
                continue
            # SOTA models are exempt from the informed/low-obs filter (a frontier
            # release is always shown even when sparsely evaluated) UNLESS
            # sota_exempt=False, which requires every model — SOTA included — to be
            # well-measured (SD < sd_cap, enough obs) on THIS axis. Off is the
            # informed-only frontier: the trend is anchored only by models actually
            # measured on the axis, never by a sparse SOTA point (e.g. an n=1 release
            # with SD > 0.5). Such releases still appear on the timeline plot; they
            # just do not bend the fitted slope.
            fails_filter = (drop_low_obs and data.is_low_obs[i]) or not informed[i]
            if fails_filter and not (sota_exempt and sota[i]):
                continue
            idx.append(i)
            dates.append(model_dates[m])
        if len(idx) < 2:
            raise ValueError(f"axis {k}: too few dated models to fit a frontier trend")

        idx = np.array(idx)
        order = np.argsort(_to_year(dates))
        idx, dates = idx[order], pd.to_datetime(pd.Series(dates).values)[order]
        # Records are read off the posterior MEDIAN, which is the number
        # `post_stats` plots, so a model is in the fit set exactly when the
        # reader can see it sitting on top of the cloud. The mean disagrees on
        # the two posterior shapes this data produces: a thinly-evaluated
        # release is right-skewed (its mean is pulled above every plotted
        # point) and a ridge-split ability is bimodal (its mean lands in the
        # empty valley between the two lumps and matches neither).
        level = np.median(theta_draws[:, idx, k], axis=0)      # (n_candidates,)

        # Frontier = record-setters of the median trajectory (the envelope)
        # UNION the SOTA models, so every flagged frontier release is fit on
        # even when it sits below the running max on this particular axis (e.g.
        # a strong general model that is weak on the agentic axis).
        is_record = level >= np.maximum.accumulate(level) - 1e-9
        frontier = is_record | sota[idx]                       # record-setters ∪ SOTA
        if fit_basis in ("informed", "envelope"):
            # Regress on every candidate (all informed models ∪ SOTA), not just the
            # envelope — denser, lower-variance, tracks the typical model.
            front = np.ones(len(idx), dtype=bool)
        else:
            base = is_record if fit_basis == "records" else frontier
            recent = base & (dates >= pd.Timestamp(fit_start))
            front = recent if recent.sum() >= 2 else base      # relax if too sparse

    x = _to_year(dates[front])                             # (F,)
    y = theta_draws[:, idx[front], k]                      # (S, F)
    env_x = env_E = slope_early = None
    floor_y = (None if backcast_floor is None
               else float(_to_year(pd.DatetimeIndex([pd.Timestamp(backcast_floor)]))[0]))
    if fit_basis == "envelope":
        srt = np.argsort(x, kind="stable")                 # frozen sets may be unsorted
        env_x, env_E = x[srt], np.maximum.accumulate(y[:, srt], axis=1)
        # Several releases can share day one; the envelope AT the start date is
        # the day's running max (last same-day index), not the first candidate.
        j0 = int(np.searchsorted(env_x, env_x[0], side="right")) - 1
        i0 = min(np.searchsorted(env_x, env_x[-1] - rate_window), len(env_x) - 2)
        i1 = max(np.searchsorted(env_x, env_x[0] + rate_window), j0 + 1)
        slope = (env_E[:, -1] - env_E[:, i0]) / (env_x[-1] - env_x[i0])
        slope_early = (env_E[:, i1] - env_E[:, j0]) / (env_x[i1] - env_x[0])
        # Forward extension as a line anchored at the last record, so every
        # linear consumer (f_now, future crossings) works unchanged for t >= end.
        intercept = env_E[:, -1] - slope * env_x[-1]
    else:
        if weights is None:
            w = np.ones_like(x)
        elif weights == "precision":
            # Fixed per-model weights (not per draw): the estimator stays
            # linear in y, so the per-draw slope distribution still propagates
            # the posterior; the weights only re-apportion leverage within the
            # fit set.
            sd = y.std(axis=0)
            w = 1.0 / np.maximum(sd, 1e-3) ** 2
        else:
            raise ValueError(f"weights must be None or 'precision', got {weights!r}")
        if estimator == "ols":
            xw = (w * x).sum() / w.sum()
            xc = x - xw
            slope = (y * (w * xc)).sum(1) / (w * xc**2).sum()   # (S,)
            intercept = (y * w).sum(1) / w.sum() - slope * xw   # (S,)
        elif estimator == "theilsen":
            slope = _theilsen_slopes(x, y, None if weights is None else w)
            resid = y - slope[:, None] * x[None, :]             # (S, F)
            intercept = (np.median(resid, axis=1) if weights is None
                         else _weighted_median_rows(resid, w))
        else:
            raise ValueError(f"estimator must be 'ols' or 'theilsen', "
                             f"got {estimator!r}")

    last_obs = pd.to_datetime(dates.max())
    horizon = (pd.Timestamp(horizon_date) if horizon_date is not None
               else min(last_obs + pd.DateOffset(years=5), pd.Timestamp("2032-01-01")))
    # Start the grid at the first point the trend was fit on, so the projected
    # line (and band) is drawn back over the historical window too — a visual
    # check of the linear fit against its own fit set — not only forward. Pass
    # back_start to extend the drawn line further back than the fit set (e.g.
    # over the full data cloud when the fit window is short).
    # The envelope has no line to draw before its first record: ignore
    # back_start there and start the grid at the window start.
    grid_start = (pd.to_datetime(back_start)
                  if back_start is not None and fit_basis != "envelope"
                  else pd.to_datetime(dates[front].min()))
    grid = pd.date_range(grid_start, horizon, freq="MS")
    xg = _to_year(grid)                                    # (G,)
    if fit_basis == "envelope":
        # Observed record steps inside the window, forward line beyond it.
        pos = np.clip(np.searchsorted(env_x, xg, side="right") - 1, 0, len(env_x) - 1)
        f = env_E[:, pos]
        beyond = xg > env_x[-1]
        f[:, beyond] = (env_E[:, -1][:, None]
                        + slope[:, None] * (xg[beyond] - env_x[-1])[None, :])
    else:
        f = intercept[:, None] + slope[:, None] * xg[None, :]  # (S, G)
    # hdi_prob HDI per grid point (narrowest interval holding the mass), the same
    # summary the human reference bands use — more intuitive than equal-tailed
    # quantiles and consistent across the figure.
    band = np.array([az.hdi(f[:, g], hdi_prob=hdi_prob) for g in range(f.shape[1])])
    return ForecastResult(
        grid_dates=grid.values,
        median=np.median(f, axis=0),
        lo=band[:, 0],
        hi=band[:, 1],
        slope=slope, intercept=intercept,
        frontier_names=[names[i] for i in idx[frontier]],   # envelope → highlight
        fit_names=[names[i] for i in idx[front]],           # what OLS fit on
        last_obs_date=last_obs, fit_basis=fit_basis,
        kind="envelope" if fit_basis == "envelope" else "linear",
        env_x=env_x, env_E=env_E, slope_early=slope_early,
        backcast_floor=floor_y,
    )


def mirt_crossover_df(fc: ForecastResult, theta_draws: np.ndarray, k: int,
                      data: ECIData, *, axis_name: str = "",
                      today: str | pd.Timestamp | None = None,
                      hdi_prob: float = 0.5) -> pd.DataFrame:
    """Per human tier: when the frontier line reaches that tier, with HDIs.

    Crossover year is solved per draw (t* = (θ_h − intercept)/slope, slope>0),
    giving a median date + hdi_prob HDI (same summary as the human bands, via
    `az.hdi`). `p_passed_now` is the posterior probability the frontier already
    exceeds the tier today; `status` reports whether that is decisive
    (`passed_ci`), only true on the mean (`passed_mean`), still `future`, or
    `no_crossing` when the trend is flat/declining. (HDI assumes the crossover
    distribution is roughly unimodal; a slope posterior straddling 0 gives a
    heavy right tail — read `frac_positive_slope` alongside it.)
    """
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.today().normalize()
    t_now = _to_year(today)
    f_now = fc.intercept + fc.slope * t_now                # (S,)
    names = data.mlookup.sort_values("model_idx")["model"].tolist()

    rows = []
    for i, m in enumerate(names):
        if not data.is_human[i]:
            continue
        th = theta_draws[:, i, k]                          # (S,)
        h_mean, h_lo, h_hi = post_stats(th, hdi_prob=hdi_prob)
        p_now = float((f_now > th).mean())
        pos = fc.slope > 0
        frac_pos = float(pos.mean())
        if getattr(fc, "kind", "linear") == "envelope":
            # Three regimes per draw: crossed INSIDE the window (the observed
            # record-step date), already above at the window START (backcast at
            # the early rate, clamped at the floor — censored at the window
            # start when no floor is given), or still AHEAD (forward line).
            E, xs = fc.env_E, fc.env_x
            reached = E >= th[:, None]
            ever = reached.any(axis=1)
            first = np.argmax(reached, axis=1)
            # Same DATE as the window start counts as "already above": several
            # releases can share day one (xs[0] == xs[1] == ...), and a tier
            # first beaten by the second same-day candidate is no more an
            # observed crossing than one beaten by the first.
            at_start = ever & (xs[first] == xs[0])
            inside = ever & (xs[first] > xs[0])
            # The envelope at the start DATE is the day-one running max (last
            # same-day index), the value the backcast extrapolates down from.
            j0 = int(np.searchsorted(xs, xs[0], side="right")) - 1
            with np.errstate(divide="ignore", invalid="ignore"):
                fwd = np.where(pos, xs[-1] + (th - E[:, -1]) / fc.slope, np.nan)
                back = np.where(fc.slope_early > 1e-9,
                                xs[0] - (E[:, j0] - th) / fc.slope_early, np.nan)
            back = (np.full_like(back, xs[0]) if fc.backcast_floor is None
                    else np.maximum(np.nan_to_num(back, nan=fc.backcast_floor),
                                    fc.backcast_floor))
            cross = np.where(inside, xs[first],
                             np.where(at_start, back, fwd))   # (S,) NaN = never
            star = cross[np.isfinite(cross)]
            defined = star.size / cross.size
        else:
            star = (th[pos] - fc.intercept[pos]) / fc.slope[pos]   # crossover years
            defined = frac_pos
        if defined < 0.5 or star.size == 0:
            date_med = date_lo = date_hi = pd.NaT
            status = "no_crossing"
        else:
            lo_yr, hi_yr = az.hdi(np.asarray(star), hdi_prob=hdi_prob)
            date_med = _to_date([np.median(star)])[0]
            date_lo = _to_date([lo_yr])[0]
            date_hi = _to_date([hi_yr])[0]
            if p_now >= 0.975:
                status = "passed_ci"
            elif date_med < today:
                status = "passed_mean"
            else:
                status = "future"
        rows.append({
            "axis": axis_name, "tier": m,
            "human_mean": h_mean, "human_hdi_low": h_lo, "human_hdi_high": h_hi,
            "crossover_date_median": date_med,
            "crossover_hdi_low": date_lo, "crossover_hdi_high": date_hi,
            "p_passed_now": p_now, "frac_positive_slope": frac_pos, "status": status,
        })
    return pd.DataFrame(rows).sort_values("human_mean").reset_index(drop=True)


