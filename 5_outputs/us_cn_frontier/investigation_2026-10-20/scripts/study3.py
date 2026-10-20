"""Further proposals: (e) records need 3 scores; (a) causal kernel at the end; (j) stability across data generations."""
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path.cwd(); sys.path.insert(0, str(ROOT / '2_model'))
from multiaxis_eci.analysis.regimes import family_best, one_per_org_day, frontier_topk
from multiaxis_eci.analysis.forecast import _to_year
S = Path(sys.argv[1])
def kernel(years, L, grid_y, bw=0.5, causal=False):
    w = np.exp(-0.5 * ((grid_y[:, None] - years[None, :]) / bw) ** 2)
    if causal: w = w * (years[None, :] <= grid_y[:, None])
    fin = np.isfinite(L); num = np.where(fin, L, 0) @ w.T; den = fin.astype(float) @ w.T
    with np.errstate(invalid='ignore', divide='ignore'): return np.where(den > 1.5, num / den, np.nan)
for scope in ('all', 'public'):
    z = np.load(S / f'E_{scope}.npz', allow_pickle=True)
    E, names, keep, lab, n_obs = z['E'].astype(float), z['names'], z['keep'], z['label'], z['n_obs']
    dates = pd.to_datetime(pd.Series(z['date']).replace('', pd.NaT)); yr = _to_year(pd.DatetimeIndex(dates.fillna(pd.Timestamp('2000-01-01'))))
    med, sd = np.median(E, 0), E.std(0)
    US = keep & (lab == 'US') & (sd <= 6); CN = keep & (lab == 'CN')
    us_idx = np.flatnonzero(US); o = np.argsort(yr[us_idx], kind='stable'); us_idx = us_idx[o]; us_yr = yr[us_idx]
    M = np.maximum.accumulate(E[:, us_idx], 1)
    def records(mask):
        t = pd.DataFrame({'name': names[mask], 'release_date': dates[mask].values, 'mean': med[mask], 'sd': sd[mask], 'idx': np.flatnonzero(mask)}).sort_values('release_date', kind='stable')
        best = one_per_org_day(family_best(t)); per_day = best.sort_values('mean', ascending=False).drop_duplicates('release_date').sort_values('release_date', kind='stable')
        return frontier_topk(per_day, 1)
    idx_of = {m: i for i, m in enumerate(names)}
    def lags(recs):
        out = []
        for m in recs:
            i = idx_of[m]; reached = M >= E[:, i][:, None]; any_ = reached.any(1); first = np.argmax(reached, 1)
            out.append(np.where(any_, (yr[i] - us_yr[first]) * 12, np.nan))
        return np.array(out).T, np.array([yr[idx_of[m]] for m in recs])
    grid = pd.date_range('2024-10-01', '2026-10-01', freq='3MS'); gy = _to_year(grid)
    base_recs = records(CN); L0, y0 = lags(base_recs)
    r3 = records(CN & (n_obs >= 3)); L3, y3 = lags(r3)
    print(f'== {scope}: records {len(base_recs)} -> {len(r3)} when a record needs 3 scores; gone: {sorted(set(base_recs) - set(r3))}; new: {sorted(set(r3) - set(base_recs))}')
    k0, k3, kc = kernel(y0, L0, gy), kernel(y3, L3, gy), kernel(y0, L0, gy, causal=True)
    print('   date       kernel as now     records>=3 scores   causal kernel (past records only)')
    for g, d in enumerate(grid):
        f = lambda K: f'{np.nanmedian(K[:, g]):4.1f} [{np.nanpercentile(K[:, g], 10):4.1f}, {np.nanpercentile(K[:, g], 90):4.1f}]'
        print(f'   {d:%Y-%m}    {f(k0)}   {f(k3)}   {f(kc)}')
    since = y0 >= _to_year(pd.DatetimeIndex(['2024-10-01']))[0]; since3 = y3 >= since.min() if False else (y3 >= _to_year(pd.DatetimeIndex(['2024-10-01']))[0])
    p0, p3 = np.nanmean(L0[:, since], 1), np.nanmean(L3[:, since3], 1)
    print(f'   mean record lag since 2024-10: {np.median(p0):.2f} [{np.percentile(p0,10):.2f}, {np.percentile(p0,90):.2f}] -> {np.median(p3):.2f} [{np.percentile(p3,10):.2f}, {np.percentile(p3,90):.2f}]')
# (j) stability across data generations
for s in ('all', 'public'):
    a = pd.read_csv(f'5_outputs/data20260930/comparisons/frontier_gap_country_{s}_records.csv', parse_dates=['release_date'])
    b = pd.read_csv(f'5_outputs/data20261001/comparisons/frontier_gap_country_{s}_records.csv', parse_dates=['release_date'])
    m = a.merge(b, on='name', suffixes=('_0930', '_1001')); m = m[m.release_date_1001 >= '2024-10-01']
    only_a = sorted(set(a[a.release_date >= '2024-10-01'].name) - set(b.name)); only_b = sorted(set(b[b.release_date >= '2024-10-01'].name) - set(a.name))
    d = (m.lag_months_median_1001 - m.lag_months_median_0930)
    print(f'== {s}: records since 2024-10 in both generations {len(m)}; only 0930: {only_a}; only 1001: {only_b}')
    print(f'   median lag change 0930 -> 1001: mean {d.mean():+.2f}, mean |change| {d.abs().mean():.2f}, max |change| {d.abs().max():.2f} ({m.loc[d.abs().idxmax(), "name"]})')
