"""The gap between two groups' frontiers on a K=1 trace, in ECI-H: open-weights models against
closed ones by default (`--group openness`), US against CN as the older cut (`--group country`).

Every quantity is computed on one benchmark access scope at a time (the trace of a fit on all
benchmarks, or on the public / semi-private / private class only), so the scopes can be compared
afterwards: the same code runs on each trace and the ECI-H scale is pinned by the same two
anchors in every fit.

Frontier of a group: the running maximum of posterior-median ECI-H over the group's candidates
by release date (`regimes.frontier_topk` with k = 1), one effort per family, the best on the
axis (`regimes.family_best`), one release per organization and day (`regimes.one_per_org_day`)
and one record at most per day, the candidates being those of the timelines
(`timelines.candidate_mask`: dated, not a human tier, evaluated on the axis), with at least two
scores in the scope, the K=1 reading of the K-axis coverage rule (the CLI's `--min-obs`). Three
readings of the gap:

1. `frontier_lag`: for every record of the follower group (open), the months between its
   release and the release of the first leader (closed) model to have beaten it: per posterior
   draw, the earliest leader candidate whose ECI-H draw is at or above the record's, records
   or not, so the lag carries the abilities' uncertainty and the first model above is named
   with its share of draws. This
   is the "backward-looking" lag of Ihle (2026,
   https://www.lesswrong.com/posts/rJcCrXyEsJKmmDpWG/how-far-behind-are-open-models), with
   ECI-H in place of per-benchmark score thresholds. A record the leader has never matched
   (open ahead) has no lag. A level the frontier already exceeded on its first measured day
   (the closed models scored before it were never run in the scope) is dated back at the
   frontier's early growth rate (`BACKCAST_WINDOW_YEARS`), the envelope forecast's convention;
   the fraction of such draws is reported per record, and a record that needs the assumption in
   more than `MAX_DATED_BACK_FRAC` of draws is dropped from the series.
2. `line_gap`: one straight line per group fitted on the posterior medians of its frontier
   (top-`top_k` points released since `fit_start`, `regimes.weighted_line_fit`, each point
   weighted by its posterior SD), giving the gap in ECI-H points and the lag in months (gap over
   the follower's slope) at a date, today by default.
3. `crossovers`: when each group's line reaches every human tier (`regimes.regime_crossover_df`
   on the single line).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from multiaxis_eci import config
from multiaxis_eci.analysis.forecast import ForecastResult, _to_year
from multiaxis_eci.analysis.regimes import (
    LineFit,
    family_best,
    frontier_topk,
    one_per_org_day,
    regime_crossover_df,
    weighted_line_fit,
)
from multiaxis_eci.analysis.stats import post_stats
from multiaxis_eci.data import MODEL_OPENNESS_FILE, MODELS_FILE, load_model_openness

# Smallest posterior SD (ECI-H points) a frontier point carries in the line fits.
SD_FLOOR = 0.5
# (leader, follower): the gap is leader minus follower, the lag is the follower's.
GROUP_PAIRS = {"openness": ("closed", "open"), "country": ("US", "CN")}
GROUP_TITLES = {"openness": {"closed": "Closed models", "open": "Open-weights models"},
                "country": {"US": "US models", "CN": "Chinese models"}}
# Benchmark access scopes: the fit folder and the caption of each.
SCOPES = {"all": ("canonical", "all benchmarks"),
          "public": ("canonical_public", "public benchmarks"),
          "semi_private": ("canonical_semi_private", "semi-private benchmarks"),
          "private": ("canonical_private", "private benchmarks")}


def group_labels(kind: str, names: list[str]) -> dict[str, str]:
    """Test-taker -> group label. `openness` reads 1_curated/model_openness.csv (open / closed;
    `unknown` is left out, so an unlabelled candidate is caught downstream), `country` the
    pipeline's models table (US / CN; other countries left out)."""
    if kind == "openness":
        lab = load_model_openness()
        return {m: lab[m] for m in names if lab.get(m) in ("open", "closed")}
    if kind == "country":
        df = pd.read_csv(MODELS_FILE, dtype=str).fillna("")
        cmap = dict(zip(df["model_version"], df["country"]))
        return {m: cmap[m] for m in names if cmap.get(m) in ("US", "CN")}
    raise ValueError(f"unknown group kind {kind!r}")


@dataclass
class GroupFrontier:
    label: str
    tl: pd.DataFrame            # candidates: name, release_date, mean, hdi_low, hdi_high, sd, is_record, in_fit
    records: list[str]          # the frontier: running max of posterior-median ECI-H
    fit: LineFit | None         # the trend line through the frontier (ECI-H per year)
    env_years: np.ndarray       # (F,) candidate release years, ascending
    env_E: np.ndarray           # (S, F) running max of per-draw ECI-H over the candidates so far

    def forecast(self, horizon="2030-01-01", hdi_prob: float = 0.8) -> ForecastResult | None:
        """The line as a ForecastResult (fit_basis 'regimes', no other regime), for the crossover
        code and the trend figures."""
        if self.fit is None:
            return None
        grid = pd.date_range(pd.Timestamp(self.tl.loc[self.tl["in_fit"], "release_date"].min()),
                             pd.Timestamp(horizon), freq="MS")
        F = self.fit.at(_to_year(grid))
        lo, hi = np.percentile(F, [(1 - hdi_prob) / 2 * 100, (1 + hdi_prob) / 2 * 100], axis=0)
        return ForecastResult(
            grid_dates=grid.values, median=np.median(F, axis=0), lo=lo, hi=hi, slope=self.fit.b,
            intercept=self.fit.a - self.fit.b * self.fit.t0, frontier_names=self.records,
            last_obs_date=pd.Timestamp(self.tl.loc[self.tl["in_fit"], "release_date"].max()),
            fit_names=self.fit.names, fit_basis="regimes", kind="line", switch_year=self.fit.t_min)


def build_group_frontier(E: np.ndarray, names: list[str], model_dates: pd.Series,
                         members: np.ndarray, label: str, *, fit_start: str,
                         top_k: int = 2, hdi_prob: float = 0.8) -> GroupFrontier:
    """One group's candidates (`members`, (n,) bool over `names`, already restricted to the
    timelines' candidates), its records, its trend line and its per-draw envelope, from the
    per-draw ECI-H array `E` (S, n)."""
    idx = np.flatnonzero(members)
    rows = []
    for i in idx:
        med, lo, hi = post_stats(E[:, i], hdi_prob=hdi_prob)
        rows.append({"name": names[i], "release_date": pd.Timestamp(model_dates[names[i]]),
                     "mean": med, "hdi_low": lo, "hdi_high": hi, "sd": float(E[:, i].std()),
                     "idx": int(i)})
    tl = pd.DataFrame(rows, columns=["name", "release_date", "mean", "hdi_low", "hdi_high",
                                     "sd", "idx"])
    tl = tl.sort_values("release_date", kind="stable").reset_index(drop=True)
    # Records and fit points are read on one effort per family, the best on the axis, and one
    # release per organization and day (`regimes.family_best`, `one_per_org_day`: the trend
    # panels' rule), so a record is never its own runner-up and a thinly-run effort never
    # stands in for its release. A record must also be the best release of its day, whatever
    # the organization: two records on one day would be one record and an instant runner-up.
    # The cloud keeps every effort.
    best = one_per_org_day(family_best(tl)) if len(tl) else tl
    per_day = (best.sort_values("mean", ascending=False).drop_duplicates("release_date")
                   .sort_values("release_date", kind="stable")) if len(tl) else tl
    records = frontier_topk(per_day, 1) if len(tl) else []
    tl["is_record"] = tl["name"].isin(records)
    fit_pts = (best[best["name"].isin(frontier_topk(best, top_k))
                    & (best["release_date"] >= pd.Timestamp(fit_start))] if len(tl) else tl)
    tl["in_fit"] = tl["name"].isin(fit_pts["name"])
    fit = None
    if len(fit_pts) >= 3:
        # The two anchors have no posterior spread in ECI-H by construction (130 and 150 in every
        # draw); a zero SD would make the line's noise matrix singular, so every point keeps at
        # least SD_FLOOR of spread.
        fit = weighted_line_fit(_to_year(pd.DatetimeIndex(fit_pts["release_date"])),
                                fit_pts["mean"].to_numpy(),
                                np.maximum(fit_pts["sd"].to_numpy(), SD_FLOOR),
                                fit_pts["name"].tolist())
    if len(tl):
        order = tl["idx"].to_numpy()
        env_E = np.maximum.accumulate(E[:, order], axis=1)
        env_years = _to_year(pd.DatetimeIndex(tl["release_date"]))
    else:
        env_E, env_years = np.zeros((E.shape[0], 0)), np.zeros(0)
    return GroupFrontier(label=label, tl=tl, records=records, fit=fit,
                         env_years=env_years, env_E=env_E)


# The closed frontier before its first measurement is unobserved. A level it already exceeded on
# that first day is dated back at the rate the frontier grew over its first BACKCAST_WINDOW_YEARS
# of observation, the convention of `forecast.mirt_frontier_forecast`'s envelope for a tier
# passed before the window.
BACKCAST_WINDOW_YEARS = 1.5
# A record whose crossing has to be dated back in more than this share of draws is dropped from the
# lag series: its lag would be a reading of the assumed early rate, not of a measured crossing.
# "Dated back" is `backcast` plus `censored`, the same first-day case with a non-positive rate.
MAX_DATED_BACK_FRAC = 2.0 / 3.0


def frontier_lag(follower: GroupFrontier, leader: GroupFrontier, E: np.ndarray,
                 probs: tuple[float, float] = (0.5, 0.8)) -> tuple[pd.DataFrame, np.ndarray]:
    """Per follower record, the months since the leader's frontier reached its level, per draw.

    In each draw, the crossing is the release date of the FIRST leader candidate (by release
    date, records or not) whose ECI-H draw is at or above the record's draw: the first closed
    model to have beaten it, Ihle's reading. Lag = record release minus that date, negative when
    the record led. Which candidate that is varies with the draw; `first_leader_above` names the
    most frequent ones with their share of draws, so a wide interval can be read (DeepSeek V3
    0324 on private benchmarks is GPT-4 0613's equal: in half the draws GPT-4 0613 beat it in
    June 2023, in the other half the first closed model above it came a year later).

    `backcast`: the frontier already
    exceeded the level on its first measured day (the closed models that carried it there were
    never scored in the scope), so the crossing is dated back from that day at the rate the
    frontier grew over its first `BACKCAST_WINDOW_YEARS` of observation in that draw, an
    estimate under a stated assumption. `censored`: the same case when the early rate is not
    positive; the lag then counts at its lower bound (record minus first day). `undefined`: the
    frontier never reached the level (the follower is ahead of everything the leader released).
    Fractions of each case per record; a record backcast (censored) in most draws is flagged
    and drawn hollow, and one dated back in more than `MAX_DATED_BACK_FRAC` of draws is left out
    of the returned frame and draws altogether."""
    recs = follower.tl[follower.tl["is_record"]].reset_index(drop=True)
    S = E.shape[0]
    lags = np.full((S, len(recs)), np.nan)
    rows = []
    has_leader = leader.env_E.shape[1] > 0
    if has_leader:
        years = leader.env_years.astype(float)                             # (F,) ascending
        M = leader.env_E                                                     # (S, F) running max
        d0 = float(years[0])
        i0 = int(np.flatnonzero(years <= d0 + 1e-9)[-1])                   # last first-day row
        iw = int(np.flatnonzero(years <= d0 + BACKCAST_WINDOW_YEARS + 1e-9)[-1])
        F0 = M[:, i0]
        span = float(years[iw] - d0)
        with np.errstate(divide="ignore", invalid="ignore"):
            early_rate = (M[:, iw] - F0) / span if span > 0 else np.zeros(S)
        leader_names = leader.tl["name"].to_numpy()                        # (F,) by release date
    for r, rec in recs.iterrows():
        level = E[:, int(rec["idx"])]                                        # (S,)
        t_r = float(_to_year(pd.DatetimeIndex([rec["release_date"]]))[0])
        backcast = censored = np.zeros(S, dtype=bool)
        if has_leader:
            reached = M >= level[:, None]                                    # (S, F)
            any_ = reached.any(axis=1)
            first = np.argmax(reached, axis=1)                               # first row at/above
            cross = years[first]
            who = pd.Series(leader_names[first[any_]]).value_counts(normalize=True)
            first_above = "; ".join(f"{m} ({f:.0%})" for m, f in who.head(3).items())
            on_day_one = any_ & (first <= i0) & (level < F0)
            with np.errstate(divide="ignore", invalid="ignore"):
                t_back = d0 - (F0 - level) / early_rate
            backcast = on_day_one & (early_rate > 0)
            censored = on_day_one & ~(early_rate > 0)
            cross = np.where(backcast, t_back, np.where(censored, d0, cross))
            lag = np.where(any_, (t_r - cross) * 12.0, np.nan)
        else:
            lag = np.full(S, np.nan)
            first_above = ""
        lags[:, r] = lag
        finite = lag[np.isfinite(lag)]
        row = {"name": rec["name"], "release_date": rec["release_date"],
               "level_eci_median": rec["mean"], "frac_undefined": float(np.mean(~np.isfinite(lag))),
               "frac_backcast": float(backcast.mean()), "backcast": bool(backcast.mean() > 0.5),
               "frac_censored": float(censored.mean()), "censored": bool(censored.mean() > 0.5),
               "first_leader_above": first_above,
               "lag_months_median": float(np.median(finite)) if finite.size else np.nan}
        for p in probs:
            tag = f"{int(round(p * 100))}"
            if finite.size:
                lo, hi = np.quantile(finite, [(1 - p) / 2, (1 + p) / 2])
            else:
                lo = hi = np.nan
            row[f"lag_hdi{tag}_low"], row[f"lag_hdi{tag}_high"] = float(lo), float(hi)
        rows.append(row)
    cols = ["name", "release_date", "level_eci_median", "lag_months_median", "frac_undefined",
            "frac_backcast", "backcast", "frac_censored", "censored", "first_leader_above"] \
        + [f"lag_hdi{int(round(p * 100))}_{s}" for p in probs for s in ("low", "high")]
    df = pd.DataFrame(rows, columns=cols)
    # Records that need the dating-back assumption in most draws are dropped: their lag measures
    # the assumed early rate of the leader's frontier rather than a crossing anyone observed.
    keep = ((df["frac_backcast"] + df["frac_censored"]) <= MAX_DATED_BACK_FRAC).to_numpy()
    return df[keep].reset_index(drop=True), lags[:, keep]


def line_gap(leader: GroupFrontier, follower: GroupFrontier, at) -> dict[str, np.ndarray]:
    """Samples of the leader-minus-follower gap (ECI-H) at date `at`, the follower's lag in
    months (gap over its slope, NaN when the slope is not positive) and the slope difference.
    The two lines are independent posteriors, paired sample by sample."""
    t = float(_to_year(pd.DatetimeIndex([pd.Timestamp(at)]))[0])
    n = min(len(leader.fit.a), len(follower.fit.a))
    gl, gf = leader.fit.at([t])[:n, 0], follower.fit.at([t])[:n, 0]
    gap = gl - gf
    bf = follower.fit.b[:n]
    with np.errstate(divide="ignore", invalid="ignore"):
        lag = np.where(bf > 0, 12.0 * gap / bf, np.nan)
    return {"gap_eci": gap, "lag_months": lag, "slope_diff": leader.fit.b[:n] - bf,
            "level_leader": gl, "level_follower": gf}


def summarize(x: np.ndarray, probs: tuple[float, ...] = (0.5, 0.8)) -> dict[str, float]:
    finite = np.asarray(x)[np.isfinite(x)]
    out = {"median": float(np.median(finite)) if finite.size else np.nan,
           "frac_undefined": float(1 - finite.size / max(len(x), 1))}
    for p in probs:
        tag = int(round(p * 100))
        if finite.size:
            lo, hi = np.quantile(finite, [(1 - p) / 2, (1 + p) / 2])
        else:
            lo = hi = np.nan
        out[f"hdi{tag}_low"], out[f"hdi{tag}_high"] = float(lo), float(hi)
    return out


def human_tiers(E: np.ndarray, names: list[str], is_human: np.ndarray,
                hdi_prob: float = 0.8) -> pd.DataFrame:
    rows = []
    for i in np.flatnonzero(is_human):
        med, lo, hi = post_stats(E[:, i], hdi_prob=hdi_prob)
        rows.append({"name": names[i], "mean": med, "hdi_low": lo, "hdi_high": hi})
    return pd.DataFrame(rows, columns=["name", "mean", "hdi_low", "hdi_high"]) \
        .sort_values("mean").reset_index(drop=True)


def group_crossovers(gf: GroupFrontier, E: np.ndarray, data, *, today, probs=(0.5, 0.8)) -> pd.DataFrame:
    """Crossing dates of one group's line with every human tier, on `regime_crossover_df`'s
    schema (axis = the group label), the first mass in the crossover_hdi columns and the second in
    hdi80_low / hdi80_high."""
    fc = gf.forecast()
    if fc is None:
        return pd.DataFrame()
    E3 = E[:, :, None]
    out = regime_crossover_df(fc, E3, 0, data, axis_name=gf.label, today=today, hdi_prob=probs[0])
    if len(probs) > 1:
        wide = regime_crossover_df(fc, E3, 0, data, axis_name=gf.label, today=today,
                                   hdi_prob=probs[1]).set_index("tier")
        tag = int(round(probs[1] * 100))
        out[f"hdi{tag}_low"] = out["tier"].map(wide["crossover_hdi_low"])
        out[f"hdi{tag}_high"] = out["tier"].map(wide["crossover_hdi_high"])
    return out


@dataclass
class ScopeGap:
    scope: str
    kind: str
    leader: GroupFrontier
    follower: GroupFrontier
    humans: pd.DataFrame
    lag_df: pd.DataFrame        # envelope lag per follower record
    lag_draws: np.ndarray       # (S, R) months behind per record, per draw
    line: dict                  # line_gap samples at `today` (empty when a line is missing)
    line_last: dict             # line_gap samples at the follower's last record
    crossovers: pd.DataFrame
    summary: pd.DataFrame       # one row per quantity

    def summary_rows(self, probs=(0.5, 0.8)) -> pd.DataFrame:
        rows = []

        def add(q, group, x, unit):
            s = summarize(x, probs)
            rows.append({"scope": self.scope, "quantity": q, "group": group, "unit": unit, **s})

        for g in (self.leader, self.follower):
            if g.fit is not None:
                add("slope", g.label, g.fit.b, "ECI-H per year")
        if self.line:
            add("gap_today", f"{self.leader.label}-{self.follower.label}", self.line["gap_eci"], "ECI-H")
            add("lag_today", self.follower.label, self.line["lag_months"], "months")
            add("slope_diff", f"{self.leader.label}-{self.follower.label}", self.line["slope_diff"],
                "ECI-H per year")
        if self.line_last:
            add("gap_last_record", f"{self.leader.label}-{self.follower.label}",
                self.line_last["gap_eci"], "ECI-H")
            add("lag_last_record", self.follower.label, self.line_last["lag_months"], "months")
        if self.lag_draws.shape[1]:
            add("envelope_lag_latest_record", self.follower.label, self.lag_draws[:, -1], "months")
            recent = ((self.lag_df["release_date"] >= self.lag_df["release_date"].max() - pd.Timedelta(days=365))
                      & ~self.lag_df["censored"].astype(bool))
            if recent.any():
                add("envelope_lag_last_12m_mean", self.follower.label,
                    np.nanmean(self.lag_draws[:, recent.to_numpy()], axis=1), "months")
        return pd.DataFrame(rows)


def scope_gap(E: np.ndarray, data, model_dates: pd.Series, keep: np.ndarray, *, scope: str,
              kind: str = "openness", fit_start: str = config.FORECAST_KW["fit_start"],
              top_k: int = config.FORECAST_KW["top_k"], today=None, hdi_prob: float = 0.8) -> ScopeGap:
    """Everything one scope yields. `E` (S, n) is per-draw ECI-H, `keep` (n,) the timelines'
    candidate mask; a kept candidate without a group label raises (fix the curated file)."""
    names = data.mlookup.sort_values("model_idx")["model"].tolist()
    labels = group_labels(kind, names)
    leader_lab, follower_lab = GROUP_PAIRS[kind]
    if kind == "openness":
        missing = [m for i, m in enumerate(names) if keep[i] and m not in labels]
        if missing:
            raise ValueError(f"{len(missing)} candidate(s) without an open/closed label in "
                             f"{MODEL_OPENNESS_FILE.name}: {missing[:10]}{'...' if len(missing) > 10 else ''}"
                             " (add them to model_openness_overrides.csv and rebuild)")
    today = pd.Timestamp(today) if today is not None else pd.Timestamp.today().normalize()
    groups = {}
    for lab in (leader_lab, follower_lab):
        members = np.array([keep[i] and labels.get(m) == lab for i, m in enumerate(names)])
        groups[lab] = build_group_frontier(E, names, model_dates, members, lab,
                                           fit_start=fit_start, top_k=top_k, hdi_prob=hdi_prob)
    leader, follower = groups[leader_lab], groups[follower_lab]
    lag_df, lag_draws = frontier_lag(follower, leader, E)
    line = line_last = {}
    if leader.fit is not None and follower.fit is not None:
        line = line_gap(leader, follower, today)
        line_last = line_gap(leader, follower, follower.tl.loc[follower.tl["is_record"], "release_date"].max())
    cx = pd.concat([group_crossovers(g, E, data, today=today) for g in (leader, follower)],
                   ignore_index=True)
    out = ScopeGap(scope=scope, kind=kind, leader=leader, follower=follower,
                   humans=human_tiers(E, names, data.is_human, hdi_prob), lag_df=lag_df,
                   lag_draws=lag_draws, line=line, line_last=line_last, crossovers=cx,
                   summary=pd.DataFrame())
    out.summary = out.summary_rows()
    return out


# ── On disk: what the cross-scope step (4_diagnostics/2_plot_frontier_gap.py) reads back ──────

def _stem(kind: str, scope: str) -> str:
    return f"frontier_gap_{kind}_{scope}"


def save_scope(res: ScopeGap, out_dir) -> list:
    """The scope's tables (summary, records with their lag, candidates of both groups, human
    tiers, crossovers) as CSV and the draws (lag matrix, line samples, gap and lag samples) as
    one npz, all under `out_dir` / frontier_gap_<kind>_<scope>_*."""
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _stem(res.kind, res.scope)
    paths = []

    def write(df, suffix):
        p = out_dir / f"{stem}_{suffix}.csv"
        df.to_csv(p, index=False)
        paths.append(p)

    write(res.summary, "summary")
    write(res.lag_df, "records")
    cands = pd.concat([g.tl.assign(group=g.label) for g in (res.leader, res.follower)],
                      ignore_index=True).drop(columns=["idx"])
    write(cands, "candidates")
    write(res.humans, "humans")
    write(res.crossovers, "crossovers")
    arrays = {"lag_draws": res.lag_draws}
    for g in (res.leader, res.follower):
        if g.fit is not None:
            arrays[f"fit_a_{g.label}"] = g.fit.a
            arrays[f"fit_b_{g.label}"] = g.fit.b
            arrays[f"fit_meta_{g.label}"] = np.array([g.fit.t0, g.fit.t_min, g.fit.t_max,
                                                      g.fit.dispersion])
    for tag, d in (("today", res.line), ("last", res.line_last)):
        for k, v in d.items():
            arrays[f"{k}_{tag}"] = v
    p = out_dir / f"{stem}.npz"
    # Compressed: the lag matrix is (draws x records) of crossing dates off a short list of
    # releases, so it takes few distinct values — 11 MB of raw float64 for the all-benchmarks
    # scope, 1.2 MB deflated, and `np.load` reads either the same way.
    np.savez_compressed(p, **arrays)
    paths.append(p)
    return paths


def load_scope(out_dir, kind: str, scope: str) -> ScopeGap:
    """Rebuild a ScopeGap from `save_scope`'s files (the per-draw envelopes are not kept; the
    lag matrix and the line samples are)."""
    from pathlib import Path
    out_dir = Path(out_dir)
    stem = _stem(kind, scope)
    read = lambda suffix, **kw: pd.read_csv(out_dir / f"{stem}_{suffix}.csv", **kw)  # noqa: E731
    cands = read("candidates", parse_dates=["release_date"])
    npz = np.load(out_dir / f"{stem}.npz")
    groups = {}
    for lab in GROUP_PAIRS[kind]:
        tl = cands[cands["group"] == lab].drop(columns=["group"]).reset_index(drop=True)
        fit = None
        if f"fit_a_{lab}" in npz:
            t0, t_min, t_max, disp = npz[f"fit_meta_{lab}"]
            fit = LineFit(a=npz[f"fit_a_{lab}"], b=npz[f"fit_b_{lab}"], t0=float(t0),
                          names=tl.loc[tl["in_fit"], "name"].tolist(), t_min=float(t_min),
                          t_max=float(t_max), dispersion=float(disp))
        groups[lab] = GroupFrontier(label=lab, tl=tl, records=tl.loc[tl["is_record"], "name"].tolist(),
                                    fit=fit, env_years=np.zeros(0), env_E=np.zeros((0, 0)))
    leader_lab, follower_lab = GROUP_PAIRS[kind]
    line = {k[:-6]: npz[k] for k in npz.files if k.endswith("_today")}
    line_last = {k[:-5]: npz[k] for k in npz.files if k.endswith("_last")}
    cx_cols = ["crossover_date_median", "crossover_hdi_low", "crossover_hdi_high", "hdi80_low", "hdi80_high"]
    cx = read("crossovers")
    for c in cx_cols:
        if c in cx:
            cx[c] = pd.to_datetime(cx[c])
    return ScopeGap(scope=scope, kind=kind, leader=groups[leader_lab], follower=groups[follower_lab],
                    humans=read("humans"), lag_df=read("records", parse_dates=["release_date"]),
                    lag_draws=npz["lag_draws"], line=line, line_last=line_last, crossovers=cx,
                    summary=read("summary"))
