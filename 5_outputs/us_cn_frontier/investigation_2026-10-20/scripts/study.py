"""Variant study of the US-CN months-behind lag on the saved per-draw ECI-H."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[4]   # the repository
sys.path.insert(0, str(ROOT / '2_model'))
from multiaxis_eci.analysis.regimes import family_best, one_per_org_day, frontier_topk
from multiaxis_eci.analysis.forecast import _to_year
S = Path(sys.argv[1]); scope = sys.argv[2]
z = np.load(S / f'E_{scope}.npz', allow_pickle=True)
E, names, keep, lab, n_obs = z['E'].astype(float), z['names'], z['keep'], z['label'], z['n_obs']
dates = pd.to_datetime(pd.Series(z['date']).replace('', pd.NaT))
yr = _to_year(pd.DatetimeIndex(dates.fillna(pd.Timestamp('2000-01-01'))))
med = np.median(E, axis=0); sd = E.std(axis=0)
US = keep & (lab == 'US'); CN = keep & (lab == 'CN')
print(f'scope {scope}: {E.shape[0]} draws; US {US.sum()}  CN {CN.sum()}')

def tl(mask):
    return pd.DataFrame({'name': names[mask], 'release_date': dates[mask].values, 'mean': med[mask],
                         'sd': sd[mask], 'idx': np.flatnonzero(mask)}).sort_values('release_date', kind='stable').reset_index(drop=True)

def records(mask):
    t = tl(mask)
    best = one_per_org_day(family_best(t))
    per_day = best.sort_values('mean', ascending=False).drop_duplicates('release_date').sort_values('release_date', kind='stable')
    return [m for m in frontier_topk(per_day, 1)]

cn_tl, us_tl = tl(CN), tl(US)
cn_rec = records(CN); us_rec = records(US)
idx_of = {m: i for i, m in enumerate(names)}
rec_idx = np.array([idx_of[m] for m in cn_rec]); rec_dates = dates[rec_idx].values; rec_yr = yr[rec_idx]
us_idx = us_tl['idx'].to_numpy(); us_yr = yr[us_idx]; us_names = names[us_idx]
order = np.argsort(us_yr, kind='stable'); us_idx, us_yr, us_names = us_idx[order], us_yr[order], us_names[order]

def first_cross_lag(level, M, years, t_r):
    """level (S,), M (S,F) running max by date, years (F,) -> lag months per draw (nan when never reached)."""
    reached = M >= level[:, None]
    any_ = reached.any(axis=1); first = np.argmax(reached, axis=1)
    return np.where(any_, (t_r - years[first]) * 12, np.nan), first, any_

def lag_table(E_us, E_cn_level, label, interp=False):
    """Baseline reading: per CN record, first US candidate at/above it, per draw. E_us (S,F) draws of
    the US candidates by date; E_cn_level (S,R) draws of the record levels."""
    M = np.maximum.accumulate(E_us, axis=1)
    rows = []; L = np.full((E.shape[0], len(cn_rec)), np.nan)
    for r in range(len(cn_rec)):
        if interp:
            lag = interp_lag(E_cn_level[:, r], M, us_yr, rec_yr[r])
            first = None
        else:
            lag, first, any_ = first_cross_lag(E_cn_level[:, r], M, us_yr, rec_yr[r])
        L[:, r] = lag
        f = lag[np.isfinite(lag)]
        row = dict(name=cn_rec[r], date=pd.Timestamp(rec_dates[r]).date(), med=np.median(f), sd=f.std(),
                   q10=np.quantile(f, .1), q90=np.quantile(f, .9), undefined=1 - f.size / lag.size)
        if first is not None:
            who = pd.Series(us_names[first[any_]]).value_counts(normalize=True)
            thin = np.isin(us_names[first[any_]], us_names[n_obs[us_idx] <= 3]).mean()
            row['thin_share'] = thin; row['first'] = '; '.join(f'{m} ({p:.0%})' for m, p in who.head(2).items())
        rows.append(row)
    return pd.DataFrame(rows), L

def interp_lag(level, M, years, t_r):
    """Crossing time read on the US frontier interpolated linearly between its per-draw records
    (the step function joined by straight lines): the date at which continuous progress at the
    observed pace would have reached `level`."""
    S_, F = M.shape
    out = np.full(S_, np.nan)
    for s in range(S_):
        m = M[s]
        # per-draw records: strict increases
        inc = np.r_[True, m[1:] > m[:-1]]
        xs, ys = years[inc], m[inc]
        if level[s] > ys[-1]:
            continue
        if level[s] <= ys[0]:
            out[s] = (t_r - xs[0]) * 12; continue
        j = np.searchsorted(ys, level[s])   # ys[j-1] < level <= ys[j]
        t = xs[j-1] + (level[s] - ys[j-1]) / (ys[j] - ys[j-1]) * (xs[j] - xs[j-1])
        out[s] = (t_r - t) * 12
    return out

E_us = E[:, us_idx]
E_rec = E[:, rec_idx]
base, L0 = lag_table(E_us, E_rec, 'baseline')
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 60)
print('\n== baseline (per-draw first US candidate at/above the record draw) ==')
print(base[base.date >= pd.Timestamp('2024-06-01').date()].round(2).to_string(index=False))
print('CN records n_obs:', dict(zip(cn_rec, n_obs[rec_idx])))

# ── Variance decomposition: freeze one side at its posterior median ──
since = np.array([pd.Timestamp(d) >= pd.Timestamp('2024-10-01') for d in rec_dates])
Em_us = np.broadcast_to(med[us_idx], E_us.shape).copy()
Em_rec = np.broadcast_to(med[rec_idx], E_rec.shape).copy()
cn_only, _ = lag_table(Em_us, E_rec, 'cn-only')
us_only, _ = lag_table(E_us, Em_rec, 'us-only')
print('\n== per-record posterior SD of the lag (months), records since 2024-10 ==')
dec = pd.DataFrame({'name': base.name, 'both': base.sd, 'CN level only': cn_only.sd, 'US side only': us_only.sd,
                    'thin_share': base.thin_share})[since]
print(dec.round(2).to_string(index=False)); print(dec.drop(columns='name').mean().round(2).to_dict())

# ── Dispersion test: are the record medians consistent with one constant lag? ──
m, s = base.med.to_numpy()[since], base.sd.to_numpy()[since]
w = 1 / s**2; mu = (w * m).sum() / w.sum()
chi2 = ((m - mu)**2 * w).sum(); n = since.sum()
print(f'\n== records since 2024-10: n={n}, precision-weighted mean lag {mu:.2f} months, chi2={chi2:.1f} on {n-1} df '
      f'(ratio {chi2/(n-1):.2f}); plain SD of medians {m.std(ddof=1):.2f}, mean posterior SD {s.mean():.2f}')
# posterior of the mean of the record lags (pairing draws): the write-up's "last 12 months mean"
# vs the precision-weighted pooled constant
pooled = np.nanmean(L0[:, since], axis=1)
print(f'   draw-wise mean of record lags: {np.median(pooled):.2f} [{np.quantile(pooled,.1):.2f}, {np.quantile(pooled,.9):.2f}]')
# linear trend in record lags vs date
t = rec_yr[since]
A = np.column_stack([np.ones(n), t - t.mean()])
beta = np.linalg.lstsq(A * np.sqrt(w)[:, None], m * np.sqrt(w), rcond=None)[0]
cov = np.linalg.inv(A.T @ (A * w[:, None]))
print(f'   weighted linear trend of the lag: {beta[1]:+.2f} ± {np.sqrt(cov[1,1]):.2f} months per year')

# ── V2: linear interpolation of the US frontier between its per-draw records ──
interp, L2 = lag_table(E_us, E_rec, 'interp', interp=True)
print('\n== V2: crossing on the US frontier interpolated between records ==')
cmp = pd.DataFrame({'name': base.name, 'date': base.date, 'base med': base.med, 'base sd': base.sd,
                    'interp med': interp.med, 'interp sd': interp.sd})[since]
print(cmp.round(2).to_string(index=False))
mi, si = interp.med.to_numpy()[since], interp.sd.to_numpy()[since]
wi = 1/si**2; mui = (wi*mi).sum()/wi.sum(); chi2i = ((mi-mui)**2*wi).sum()
print(f'   pooled {mui:.2f}; chi2 ratio {chi2i/(n-1):.2f}; SD of medians {mi.std(ddof=1):.2f}; mean post SD {si.mean():.2f}')

# ── V4: leader side restricted to models with >= k scores, or SD cap ──
for k in (3, 4):
    ok = n_obs[us_idx] >= k
    v, _ = lag_table(E_us[:, ok][:, :], E_rec, f'nobs>={k}') if False else (None, None)
# need a version of lag_table with a different US set: quick inline
def lag_with_us(sel, interp=False):
    global us_yr, us_names, us_idx
    keep_ = (us_yr, us_names, us_idx)
    us_yr, us_names, us_idx = us_yr[sel], us_names[sel], us_idx[sel]
    out = lag_table(E[:, us_idx], E_rec, 'x', interp=interp)
    us_yr, us_names, us_idx = keep_
    return out
print('\n== V4: leader candidates restricted (median lag / posterior SD per record since 2024-10; mean over records) ==')
res = {'baseline': base}
for k in (3, 4):
    res[f'n_obs>={k} ({(n_obs[us_idx]>=k).sum()} US)'] = lag_with_us(n_obs[us_idx] >= k)[0]
for cap in (4.0, 3.0):
    res[f'sd<{cap} ({(sd[us_idx]<cap).sum()} US)'] = lag_with_us(sd[us_idx] < cap)[0]
tab = pd.DataFrame({k: v.med.round(1).astype(str) + ' ±' + v.sd.round(1).astype(str) for k, v in res.items()})
tab.insert(0, 'name', base.name); print(tab[since].to_string(index=False))
for k, v in res.items():
    mm, ss = v.med.to_numpy()[since], v.sd.to_numpy()[since]; ww = 1/ss**2
    print(f'   {k:<28s} pooled {(ww*mm).sum()/ww.sum():.2f}  mean post SD {ss.mean():.2f}  SD of medians {mm.std(ddof=1):.2f}  mean q90-q10 {(v.q90-v.q10).to_numpy()[since].mean():.2f}')

# ── V3: probabilistic records: P(model holds the CN frontier at its release) ──
cn_idx = cn_tl['idx'].to_numpy(); cn_yr = yr[cn_idx]
o = np.argsort(cn_yr, kind='stable'); cn_idx, cn_yr = cn_idx[o], cn_yr[o]
E_cn = E[:, cn_idx]
prev_max = np.maximum.accumulate(np.c_[np.full((E.shape[0], 1), -np.inf), E_cn[:, :-1]], axis=1)
p_rec = (E_cn >= prev_max).mean(axis=0)
pr = pd.DataFrame({'name': names[cn_idx], 'date': dates[cn_idx].values, 'p_record': p_rec, 'med': med[cn_idx], 'n_obs': n_obs[cn_idx],
                   'is_record': np.isin(names[cn_idx], cn_rec)})
pr = pr[pr.date >= '2024-10-01']
print('\n== V3: P(record) per CN candidate since 2024-10 (per-draw running max) ==')
print(f'   candidates {len(pr)}; median-records {pr.is_record.sum()}; P>=0.5: {(pr.p_record>=.5).sum()}; P>=0.2: {(pr.p_record>=.2).sum()}; P>=0.1: {(pr.p_record>=.1).sum()}')
print(pr[(pr.p_record >= 0.1)].round(2).to_string(index=False))
np.savez(S / f'study_{scope}.npz', L0=L0, L2=L2, rec=np.array(cn_rec), rec_dates=rec_dates, since=since)

# ── V1: the lag as a function of calendar time, per draw (both frontiers per draw) ──
grid = pd.date_range('2024-01-01', '2026-10-01', freq='MS'); gy = _to_year(grid)
M_us = np.maximum.accumulate(E_us, axis=1)
lagf = np.full((E.shape[0], len(gy)), np.nan)
Mcn = np.maximum.accumulate(E_cn, axis=1)
for g, t in enumerate(gy):
    jc = np.searchsorted(cn_yr, t, side='right') - 1
    if jc < 0: continue
    level = Mcn[:, jc]                                      # CN frontier level at t, per draw
    reached = M_us >= level[:, None]; any_ = reached.any(axis=1); first = np.argmax(reached, axis=1)
    lagf[:, g] = np.where(any_, (t - us_yr[first]) * 12, np.nan)
roll = pd.DataFrame(lagf.T, index=grid).rolling(12, min_periods=12).mean().T.to_numpy()
qs = lambda a: np.nanpercentile(a, [10, 50, 90], axis=0)
print('\n== V1: months behind as a function of calendar time (per-draw frontiers), monthly ==')
lo, mdn, hi = qs(lagf); rlo, rmd, rhi = qs(roll)
for g in range(0, len(grid), 3):
    print(f'   {grid[g].date()}  instantaneous {mdn[g]:5.1f} [{lo[g]:4.1f}, {hi[g]:4.1f}]   12-month mean {rmd[g]:5.1f} [{rlo[g]:4.1f}, {rhi[g]:4.1f}]' if np.isfinite(rmd[g]) else f'   {grid[g].date()}  instantaneous {mdn[g]:5.1f} [{lo[g]:4.1f}, {hi[g]:4.1f}]')
np.savez(S / f'lagf_{scope}.npz', lagf=lagf, grid=grid.values)
