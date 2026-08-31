"""US vs China frontier on a K=1 canonical trace: per-country record-setters,
per-draw OLS trends, gap/lag/crossovers in ECI. See CALC 1-3 in main().

Country map: 1_data/curated/model_country.csv. Scope flags mirror fit.py's
--open-only / --closed-only. Both selection variants (informed-only and
SOTA-admitted) land in the CSV; figures show informed-only.

Run: python 3_3_diagnostics/1_country_frontier.py [--open-only|--closed-only]
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from multiaxis_eci import config  # noqa: E402
from multiaxis_eci.analysis.forecast import (_to_date, _to_year, mirt_crossover_df,  # noqa: E402
                               mirt_frontier_forecast)
from multiaxis_eci.analysis.stats import _release_dates, capability_draws, eci_transform  # noqa: E402
from multiaxis_eci.analysis.timelines import mirt_informed_mask  # noqa: E402
from multiaxis_eci.data import PROCESSED_FILE, load_eci_data  # noqa: E402
from multiaxis_eci.data import open_only_drop_list  # noqa: E402
from multiaxis_eci.persistence import load_trace, save_df  # noqa: E402
from multiaxis_eci.viz.core import save_fig  # noqa: E402

SD_CAP = 0.4                              # informed-model cutoff
DEFAULT_HORIZON = "2029-01-01"
US_COLOR, CN_COLOR = "#0072B2", "#D55E00" # Okabe-Ito blue / vermillion
VARIANTS = [("informed", False), ("sota_exempt", True)]   # (label, sota_bypass)


@dataclass
class _MiniData:
    """Duck-typed ECIData stand-in with only the fields this script reads."""
    mlookup: pd.DataFrame
    n_models: int
    is_human: np.ndarray
    is_sota: np.ndarray


def load_scope(scope: str):
    """Rebuild the data for scope 'all'/'open'/'closed' (drop lists from fit.py)."""
    drop = (None if scope == "all"
            else open_only_drop_list(include_all_benchmarks=False,
                                     keep_open=(scope == "open")))
    return load_eci_data(drop_benchmarks=drop)


def load_matched(trace_path: Path, scope: str, allow_stale: bool):
    """Trace + rebuilt data on one common model list; theta0 is (S, n_common).
    Trace/data mismatch raises unless --allow-stale (name-intersection join)."""
    trace = load_trace(trace_path)
    data = load_scope(scope)
    theta_all = capability_draws(trace)                      # (S, n_trace)
    trace_names = trace.posterior["theta"].coords["model"].values.tolist()
    data_names = data.mlookup.sort_values("model_idx")["model"].tolist()

    if trace_names != data_names:
        msg = (f"trace has {len(trace_names)} models, rebuilt data scope has "
               f"{len(data_names)} models — they don't match "
               "(results/canonical/trace.nc predates the current data "
               "snapshot; re-fit before quoting numbers)")
        if not allow_stale:
            raise AssertionError(msg + ". Pass --allow-stale for a quick look "
                                 "that joins by model-name intersection.")
        print(f"WARNING: {msg}.\n"
              "  --allow-stale: joining by model-name intersection. "
              "Every number below is a STALE-TRACE number, not for quoting.")

    trace_pos = {m: i for i, m in enumerate(trace_names)}
    data_pos = {m: i for i, m in enumerate(data_names)}
    common = [m for m in trace_names if m in data_pos]
    if len(common) < 2:
        raise ValueError("fewer than 2 models in common between the trace "
                         "and the rebuilt data — nothing to compare")
    theta0 = theta_all[:, [trace_pos[m] for m in common]]                    # (S, n_common)
    rows = [data_pos[m] for m in common]
    mini = _MiniData(
        mlookup=pd.DataFrame({"model": common, "model_idx": np.arange(1, len(common) + 1)}),
        n_models=len(common),
        is_human=data.is_human[rows],
        is_sota=data.is_sota[rows] if data.is_sota is not None else np.zeros(len(common), bool))
    return trace, mini, theta0


def load_country_map() -> dict:
    """model_version -> 'US'/'CN'/'Other' from the built (override-merged) map."""
    df = pd.read_csv(ROOT / "data" / "curated" / "model_country.csv")
    return dict(zip(df["model_version"], df["country"]))


def country_records(theta0: np.ndarray, mini: _MiniData, model_dates: pd.Series,
                    country_map: dict, country: str, *, sota_bypass: bool):
    """(candidate_names_by_date, is_record, n_candidates) for one country:
    dated non-human informed models (sota_bypass admits is_sota regardless);
    records = running max of posterior-median capability."""
    names_all = mini.mlookup.sort_values("model_idx")["model"].tolist()
    informed = mirt_informed_mask(theta0[:, :, None], SD_CAP)[:, 0]

    idx, dates = [], []
    for i, m in enumerate(names_all):
        if mini.is_human[i] or m not in model_dates.index:
            continue
        if country_map.get(m) != country:
            continue
        if not (informed[i] or (sota_bypass and mini.is_sota[i])):
            continue
        idx.append(i)
        dates.append(model_dates[m])
    if not idx:
        return [], np.zeros(0, bool), 0

    idx = np.array(idx)[np.argsort(_to_year(dates))]
    level = np.median(theta0[:, idx], axis=0)                 # (n_candidates,)
    is_record = level >= np.maximum.accumulate(level) - 1e-9
    return [names_all[i] for i in idx], is_record, len(idx)


def eci_points(theta0: np.ndarray, transform) -> np.ndarray:
    """Per-model posterior-median ECI (per-draw affine applied before the median)."""
    return np.median(transform.a[:, None] + transform.b[:, None] * theta0, axis=0)


def eci_forecast_band(fc, transform, hdi_prob: float = 0.5):
    """Forecast line + HDI band in ECI: the per-draw affine applied to each
    draw's line before summarizing, so anchor uncertainty is in the band."""
    xg = _to_year(fc.grid_dates)
    f_latent = fc.intercept[:, None] + fc.slope[:, None] * xg[None, :]        # (S, G)
    f_eci = transform.a[:, None] + transform.b[:, None] * f_latent           # (S, G)
    band = np.array([az.hdi(f_eci[:, g], hdi_prob=hdi_prob) for g in range(f_eci.shape[1])])
    return np.median(f_eci, axis=0), band[:, 0], band[:, 1]


def summarize(draws: np.ndarray, hdi_prob: float = 0.5):
    """median + HDI, NaN draws dropped (lag is NaN on non-positive CN slopes)."""
    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return float("nan"), float("nan"), float("nan")
    lo, hi = az.hdi(finite, hdi_prob=hdi_prob)
    return float(np.median(finite)), float(lo), float(hi)


def build_figure(theta0, mini, transform, model_dates, country_map, records, forecasts,
                 horizon: str, title: str, fit_label: str,
                 y_range: tuple[float, float] | None = None) -> go.Figure:
    """Scatter of every informed US/CN model (record-setters emphasized) plus
    the two forecast median lines + 50% HDI bands, in ECI. Shows the
    informed-only variant only (the honest default; see module docstring)."""
    names = mini.mlookup.sort_values("model_idx")["model"].tolist()
    eci_med = eci_points(theta0, transform)
    informed = mirt_informed_mask(theta0[:, :, None], SD_CAP)[:, 0]

    fig = go.Figure()
    y_all = []
    for country, color in (("US", US_COLOR), ("CN", CN_COLOR)):
        rec_names = set(records[(country, "informed")][0])
        idx = [i for i, m in enumerate(names)
              if not mini.is_human[i] and m in model_dates.index
              and country_map.get(m) == country and informed[i]]
        if not idx:
            continue
        dates = [model_dates[names[i]] for i in idx]
        ys = [eci_med[i] for i in idx]
        y_all += ys
        is_rec = [names[i] in rec_names for i in idx]
        fig.add_trace(go.Scatter(
            x=[d for d, r in zip(dates, is_rec) if not r],
            y=[y for y, r in zip(ys, is_rec) if not r],
            mode="markers", name=f"{country} models (informed)",
            marker=dict(size=6, color=color, opacity=0.45),
            text=[names[i] for i, r in zip(idx, is_rec) if not r],
            hovertemplate="%{text}<br>%{x}<br>ECI %{y:.1f}<extra></extra>"))
        fig.add_trace(go.Scatter(
            x=[d for d, r in zip(dates, is_rec) if r],
            y=[y for y, r in zip(ys, is_rec) if r],
            mode="markers", name=f"{country} frontier record",
            marker=dict(size=11, color=color, symbol="diamond",
                       line=dict(width=1, color="white")),
            text=[names[i] for i, r in zip(idx, is_rec) if r],
            hovertemplate="%{text}<br>%{x}<br>ECI %{y:.1f}<extra></extra>"))

        fc = forecasts[(country, "informed")]
        if fc is None:
            continue
        gx = pd.to_datetime(fc.grid_dates).strftime("%Y-%m-%d")
        med, lo, hi = eci_forecast_band(fc, transform)
        fig.add_trace(go.Scatter(
            x=list(gx) + list(gx[::-1]), y=list(hi) + list(lo[::-1]),
            fill="toself", fillcolor=_rgba(color, 0.12),
            line=dict(width=0), hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            x=gx, y=med, mode="lines", name=f"{country} trend ({fit_label})",
            line=dict(color=color, dash="dash", width=2.2),
            hovertemplate="ECI ≈ %{y:.1f}<br>%{x}<extra></extra>"))
        y_all += list(hi) + list(lo)

    # Human tier reference lines at posterior-median ECI, strongest first so
    # the legend reads top-down in plot order. hlines carry no legend entry of
    # their own, hence the invisible traces (same convention as viz/core.py).
    tiers = sorted(((names[i], eci_med[i]) for i in range(mini.n_models)
                    if mini.is_human[i]), key=lambda t: -t[1])
    fracs = np.linspace(0.9, 0.4, len(tiers)) if len(tiers) > 1 else [0.7]
    for k, ((tname, ty), f) in enumerate(zip(tiers, fracs)):
        col = pc.sample_colorscale("Greys", float(f))[0]
        fig.add_hline(y=ty, line=dict(color=col, dash="dot", width=1.2),
                      layer="below")
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            line=dict(color=col, dash="dot", width=1.8), name=tname,
            legendgroup="humans",
            legendgrouptitle_text="Human tiers (median)" if k == 0 else None,
            hoverinfo="skip"))
        y_all.append(ty)

    # A fixed y_range makes two scopes' figures directly comparable side by
    # side; without it the range pins to this figure's own data.
    if y_range is not None:
        ylo, yhi = y_range
        pad = 0.0
    else:
        ylo, yhi = (min(y_all), max(y_all)) if y_all else (0, 1)
        pad = 0.06 * (yhi - ylo)
    fig.update_layout(
        template="plotly_white",
        title=dict(text=title, x=0.5),
        xaxis=dict(title="Release date", type="date",
                  range=[pd.to_datetime(model_dates.min()).strftime("%Y-%m-%d"),
                         pd.Timestamp(horizon).strftime("%Y-%m-%d")]),
        yaxis=dict(title="ECI (anchored)", range=[ylo - pad, yhi + pad]),
        legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=1.01),
        margin=dict(l=70, r=220, t=70, b=55), height=560, width=1050)
    return fig


def _rgba(hexcolor: str, alpha: float) -> str:
    r, g, b = (int(hexcolor[i:i + 2], 16) for i in (1, 3, 5))
    return f"rgba({r},{g},{b},{alpha})"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default=None,
                    help="dir holding trace.nc (default: results/canonical or "
                         "results/canonical_open with --open-only)")
    ap.add_argument("--open-only", action="store_true",
                    help="use the --open-only data scope (needs "
                         "1_data/curated/benchmark_access.csv)")
    ap.add_argument("--closed-only", action="store_true",
                    help="use the --closed-only data scope (complement of "
                         "--open-only)")
    ap.add_argument("--allow-stale", action="store_true",
                    help="downgrade the trace/data n_models mismatch to a "
                         "warning and join by model-name intersection, "
                         "instead of raising")
    ap.add_argument("--horizon", default=DEFAULT_HORIZON,
                    help=f"common forecast horizon date (default {DEFAULT_HORIZON})")
    ap.add_argument("--fit-start", default="2024-10-01",
                    help="OLS window: only records released on/after this date "
                         "bend the trend (both countries; earlier records still "
                         "plot as diamonds). Without it the US slope is diluted "
                         "by its 2019-2022 records while CN's start mid-2023.")
    ap.add_argument("--y-range", default=None, metavar="LO,HI",
                    help="fix the figure's ECI y-range (e.g. 50,255) so two "
                         "scopes render on one comparable scale; default pins "
                         "to this figure's own data")
    args = ap.parse_args()

    if args.open_only and args.closed_only:
        raise ValueError("--open-only and --closed-only cannot compose")
    scope = "open" if args.open_only else "closed" if args.closed_only else "all"
    tag = {"all": "canonical", "open": "canonical_open",
           "closed": "canonical_closed"}[scope]
    results_dir = Path(args.results_dir) if args.results_dir else config.RESULTS_DIR / tag
    trace_path = results_dir / "trace.nc"

    print(f"Loading trace {trace_path} ...", flush=True)
    trace, mini, theta0 = load_matched(trace_path, scope, args.allow_stale)
    print(f"  {mini.n_models} models in common "
          f"({int(mini.is_human.sum())} human, {int(mini.is_sota.sum())} SOTA)")

    raw_df = pd.read_csv(PROCESSED_FILE)
    country_map = load_country_map()
    transform = eci_transform(theta0, mini)

    model_dates, _ = _release_dates(raw_df)

    # ── CALC 1: records + per-country trend lines ───────────────────────────
    # Record-setters = running max of posterior-median ECI; the trend is a
    # per-draw OLS on records released >= --fit-start (falls back to all
    # records when the window holds < 2), via mirt_frontier_forecast's frozen
    # fit_names path.
    records, forecasts, rows = {}, {}, []
    for country in ("US", "CN"):
        for variant, sota_bypass in VARIANTS:
            names, is_record, n_cand = country_records(
                theta0, mini, model_dates, country_map, country,
                sota_bypass=sota_bypass)
            rec_names = [n for n, r in zip(names, is_record) if r]
            records[(country, variant)] = (rec_names, n_cand)
            fit_set = [n for n in rec_names
                       if pd.Timestamp(model_dates[n]) >= pd.Timestamp(args.fit_start)]
            if len(fit_set) < 2:
                fit_set = rec_names
            fc = None
            if len(fit_set) >= 2:
                fc = mirt_frontier_forecast(
                    theta0[:, :, None], 0, mini, raw_df, fit_names=fit_set,
                    hdi_prob=0.5, horizon_date=args.horizon,
                    back_start=min(model_dates[n] for n in fit_set))
            else:
                print(f"  WARNING: {country}/{variant} has only "
                      f"{len(rec_names)} record-setter(s), skipping its forecast")
            forecasts[(country, variant)] = fc

    # ── CALC 2: gap and lag, per draw ───────────────────────────────────────
    # Both trend lines evaluated at the last common record date; gap = US - CN
    # there (mapped to ECI by the per-draw anchor affine), lag = gap / CN
    # slope in months. One value per draw, summarized as median + 50/95% HDI.
    for variant, _ in VARIANTS:
        fc_us, fc_cn = forecasts[("US", variant)], forecasts[("CN", variant)]
        if fc_us is None or fc_cn is None:
            gap_med = gap_lo = gap_hi = lag_med = lag_lo = lag_hi = float("nan")
            gap_lo95 = gap_hi95 = lag_lo95 = lag_hi95 = float("nan")
            lag_nan_frac = float("nan")
            last_common = pd.NaT
        else:
            last_common = min(fc_us.last_obs_date, fc_cn.last_obs_date)
            t = _to_year(last_common)
            c_us = fc_us.intercept + fc_us.slope * t               # (S,)
            c_cn = fc_cn.intercept + fc_cn.slope * t               # (S,)
            gap_c = c_us - c_cn
            gap_eci = transform.b * gap_c
            # lag uses latent units — the affine b cancels out of the ratio
            lag_months = 12.0 * np.where(fc_cn.slope > 0, gap_c / fc_cn.slope, np.nan)
            lag_nan_frac = float(np.mean(~np.isfinite(lag_months)))
            gap_med, gap_lo, gap_hi = summarize(gap_eci)
            lag_med, lag_lo, lag_hi = summarize(lag_months)
            _, gap_lo95, gap_hi95 = summarize(gap_eci, hdi_prob=0.95)
            _, lag_lo95, lag_hi95 = summarize(lag_months, hdi_prob=0.95)
            if variant == "informed":
                # per-draw arrays for downstream delta computations
                np.savez(config.RESULTS_DIR / "comparisons"
                         / f"country_frontier_draws_{tag}.npz",
                         slope_us_eci=transform.b * fc_us.slope,
                         slope_cn_eci=transform.b * fc_cn.slope,
                         gap_eci=gap_eci, lag_months=lag_months)

        for country in ("US", "CN"):
            fc = forecasts[(country, variant)]
            rec_names, n_cand = records[(country, variant)]
            if fc is not None:
                eci_slope = transform.b * fc.slope
                slope_med, slope_lo, slope_hi = summarize(eci_slope)
                _, slope_lo95, slope_hi95 = summarize(eci_slope, hdi_prob=0.95)
            else:
                slope_med = slope_lo = slope_hi = float("nan")
                slope_lo95 = slope_hi95 = float("nan")
            rows.append({
                "country": country, "variant": variant,
                "n_candidates": n_cand, "n_record_setters": len(rec_names),
                "record_setters": ";".join(rec_names),
                "eci_slope_median": slope_med, "eci_slope_hdi_low": slope_lo,
                "eci_slope_hdi_high": slope_hi,
                "eci_slope_hdi95_low": slope_lo95,
                "eci_slope_hdi95_high": slope_hi95,
                # gap/lag describe the pair; lag reads off the CN row
                "last_common_obs_date": last_common,
                "gap_at_last_common_eci_median": gap_med,
                "gap_hdi_low": gap_lo, "gap_hdi_high": gap_hi,
                "gap_hdi95_low": gap_lo95, "gap_hdi95_high": gap_hi95,
                "lag_months_median": lag_med if country == "CN" else float("nan"),
                "lag_hdi_low": lag_lo if country == "CN" else float("nan"),
                "lag_hdi_high": lag_hi if country == "CN" else float("nan"),
                "lag_hdi95_low": lag_lo95 if country == "CN" else float("nan"),
                "lag_hdi95_high": lag_hi95 if country == "CN" else float("nan"),
                "lag_nan_fraction": lag_nan_frac if country == "CN" else float("nan"),
            })

    out = pd.DataFrame(rows)
    out_path = config.RESULTS_DIR / "comparisons" / f"country_frontier_{tag}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_df(out, out_path)
    print(f"\nwrote {out_path}")

    scope_label = {"all": "all benchmarks", "open": "open benchmarks only",
                   "closed": "closed benchmarks only"}[scope]
    fit_label = f"records since {pd.Timestamp(args.fit_start):%b %Y}"
    fig = build_figure(theta0, mini, transform, model_dates, country_map,
                       records, forecasts, args.horizon,
                       title=f"US vs China frontier (ECI) — {scope_label}",
                       fit_label=fit_label,
                       y_range=(tuple(float(v) for v in args.y_range.split(","))
                                if args.y_range else None))
    save_fig(fig, f"country_frontier_{tag}", config.PLOTS_DIR)
    print(f"wrote {config.PLOTS_DIR / f'country_frontier_{tag}.html'}")

    # ── CALC 3: crossover dates (informed variant, matching the figure) ────
    # Per country x human tier via the shared mirt_crossover_df — dates are
    # unit-invariant, solved on the latent scale per draw. Plus the US-CN line
    # crossing, solved the same per-draw way from the two frozen OLS lines.
    today = pd.Timestamp.today().normalize()
    eci_med_all = eci_points(theta0, transform)
    names_all = mini.mlookup.sort_values("model_idx")["model"].tolist()
    tier_eci = {m: eci_med_all[i] for i, m in enumerate(names_all)
                if mini.is_human[i]}
    xdfs = []
    for country in ("US", "CN"):
        fc = forecasts[(country, "informed")]
        if fc is None:
            continue
        xdf = mirt_crossover_df(fc, theta0[:, :, None], 0, mini,
                                axis_name=country, today=today)
        xdf = xdf.rename(columns={"axis": "country"})
        # Second pass at 95% only for the wider interval columns; merged on
        # tier so row order never matters.
        x95 = mirt_crossover_df(fc, theta0[:, :, None], 0, mini,
                                axis_name=country, today=today,
                                hdi_prob=0.95).set_index("tier")
        xdf["crossover_hdi95_low"] = xdf["tier"].map(x95["crossover_hdi_low"])
        xdf["crossover_hdi95_high"] = xdf["tier"].map(x95["crossover_hdi_high"])
        xdf["human_eci_median"] = xdf["tier"].map(tier_eci)
        xdfs.append(xdf)
    xover = pd.concat(xdfs, ignore_index=True) if xdfs else pd.DataFrame()
    xover_path = config.RESULTS_DIR / "comparisons" / f"country_crossover_{tag}.csv"
    save_df(xover, xover_path)
    print(f"wrote {xover_path}")

    fc_us, fc_cn = forecasts[("US", "informed")], forecasts[("CN", "informed")]
    uscn = None
    if fc_us is not None and fc_cn is not None:
        ds = fc_cn.slope - fc_us.slope
        ok = np.abs(ds) > 1e-12
        tstar = (fc_us.intercept[ok] - fc_cn.intercept[ok]) / ds[ok]
        lo, hi = az.hdi(np.asarray(tstar), hdi_prob=0.5)
        uscn = {"med": _to_date([np.median(tstar)])[0],
                "lo": _to_date([lo])[0], "hi": _to_date([hi])[0],
                "p_cn_faster": float((fc_cn.slope > fc_us.slope).mean())}

    print("\n── Summary ─────────────────────────────────────────────────────")
    for variant, _ in VARIANTS:
        print(f"\nvariant={variant}")
        for country in ("US", "CN"):
            r = out[(out.country == country) & (out.variant == variant)].iloc[0]
            print(f"  {country}: {r.n_candidates} candidates, "
                  f"{r.n_record_setters} record-setters, "
                  f"slope {r.eci_slope_median:+.1f} "
                  f"[{r.eci_slope_hdi_low:+.1f}, {r.eci_slope_hdi_high:+.1f}] ECI/yr")
        r = out[(out.country == "CN") & (out.variant == variant)].iloc[0]
        print(f"  gap @ {r.last_common_obs_date}: {r.gap_at_last_common_eci_median:.1f} "
              f"[{r.gap_hdi_low:.1f}, {r.gap_hdi_high:.1f}]50 "
              f"[{r.gap_hdi95_low:.1f}, {r.gap_hdi95_high:.1f}]95 ECI (US - CN)")
        print(f"  CN lag: {r.lag_months_median:.1f} "
              f"[{r.lag_hdi_low:.1f}, {r.lag_hdi_high:.1f}]50 "
              f"[{r.lag_hdi95_low:.1f}, {r.lag_hdi95_high:.1f}]95 months "
              f"(NaN fraction {r.lag_nan_fraction:.2f})")

    print("\n── Crossovers (informed variant) ───────────────────────────────")
    if uscn is not None:
        print(f"  US/CN trend lines cross: {uscn['med']:%Y-%m} "
              f"[{uscn['lo']:%Y-%m}, {uscn['hi']:%Y-%m}]  "
              f"P(CN slope > US) = {uscn['p_cn_faster']:.2f}")
    for _, r in xover.iterrows():
        dm = r.crossover_date_median
        dtxt = (f"{dm:%Y-%m} [{r.crossover_hdi_low:%Y-%m}, "
                f"{r.crossover_hdi_high:%Y-%m}]" if pd.notna(dm) else "—")
        print(f"  {r.country} x {r.tier:<32s} ECI {r.human_eci_median:6.1f}  "
              f"{r.status:<11s} {dtxt}  P(passed now)={r.p_passed_now:.2f}")


if __name__ == "__main__":
    main()
