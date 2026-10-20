"""Uniform alternatives to a score-count rule for records, and the top-2 frontier."""
import sys, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path.cwd(); sys.path.insert(0, str(ROOT / '2_model'))
from multiaxis_eci.analysis.regimes import family_best, one_per_org_day, frontier_topk
from multiaxis_eci.analysis.forecast import _to_year
S = Path(sys.argv[1]); T0 = _to_year(pd.DatetimeIndex(['2024-10-01']))[0]
for scope in ('all', 'public', 'private'):
    z = np.load(S / f'E_{scope}.npz', allow_pickle=True)
    E, names, keep, lab, n_obs = z['E'].astype(float), z['names'], z['keep'], z['label'], z['n_obs']
    dates = pd.to_datetime(pd.Series(z['date']).replace('', pd.NaT)); yr = _to_year(pd.DatetimeIndex(dates.fillna(pd.Timestamp('2000-01-01'))))
    med, sd = np.median(E, 0), E.std(0); Sn = E.shape[0]
    US = keep & (lab == 'US') & (sd <= 6); CN = keep & (lab == 'CN')
    us_idx = np.flatnonzero(US); o = np.argsort(yr[us_idx], kind='stable'); us_idx = us_idx[o]; us_yr = yr[us_idx]
    Eu = E[:, us_idx]; M1 = np.maximum.accumulate(Eu, 1)
    M2 = np.full_like(Eu, -np.inf); m1 = np.full(Sn, -np.inf); m2 = np.full(Sn, -np.inf)   # running second max
    for j in range(Eu.shape[1]):
        x = Eu[:, j]; new1 = x > m1; m2 = np.where(new1, m1, np.maximum(m2, x)); m1 = np.where(new1, x, m1); M2[:, j] = m2
    idx_of = {m: i for i, m in enumerate(names)}
    def lag(i, M):
        reached = M >= E[:, i][:, None]; any_ = reached.any(1); first = np.argmax(reached, 1)
        return np.where(any_, (yr[i] - us_yr[first]) * 12, np.nan)
    def table(mask):
        return pd.DataFrame({'name': names[mask], 'release_date': dates[mask].values, 'mean': med[mask], 'sd': sd[mask], 'idx': np.flatnonzero(mask)}).sort_values('release_date', kind='stable')
    cn = table(CN); best = one_per_org_day(family_best(cn)); per_day = best.sort_values('mean', ascending=False).drop_duplicates('release_date').sort_values('release_date', kind='stable')
    rec1 = frontier_topk(per_day, 1); top2 = frontier_topk(best, 2)
    def headline(recs, M=M1, weights=None):
        L = np.array([lag(idx_of[m], M) for m in recs]).T; t = np.array([yr[idx_of[m]] for m in recs])
        win = t > t.max() - 1.0
        if weights == 'precision':
            w = 1 / np.nanvar(L[:, win], axis=0); v = np.nansum(L[:, win] * w, 1) / w.sum()
        else: v = np.nanmean(L[:, win], 1)
        return f'{np.median(v):4.1f} [{np.percentile(v,10):4.1f}, {np.percentile(v,90):4.1f}]', int(win.sum()), L, t
    print(f'\n===== {scope}')
    h, n, L1, t1 = headline(rec1); print(f'median-record rule (as now): {len([m for m in rec1 if yr[idx_of[m]] >= T0])} records since 2024-10, last-12-month mean {h} on {n} records')
    hp, _, _, _ = headline(rec1, weights='precision'); print(f'   same records, precision-weighted mean: {hp}')
    # A. probabilistic record rule: beat the standing record in a share p of draws
    for p in (0.5, 0.6, 0.75):
        recs, cur = [], None
        for _, r in per_day.iterrows():
            i = int(r.idx)
            if cur is None or (E[:, i] > E[:, cur]).mean() >= p:
                recs.append(r['name']); cur = i
        h, n, _, _ = headline(recs); since = [m for m in recs if yr[idx_of[m]] >= T0]
        two = [m for m in since if n_obs[idx_of[m]] <= 2]
        print(f'   P(beats standing record) >= {p:.2f}: {len(since)} records since 2024-10 ({len(two)} with 2 scores: {two[:4]}), last-12-month mean {h} on {n}')
    # C. top-2 frontier
    t2_since = [m for m in top2 if yr[idx_of[m]] >= T0]; extra = [m for m in t2_since if m not in rec1]
    h_t2, n2, L2, t2 = headline(top2); h_sym, _, L2s, _ = headline(top2, M=M2)
    h_sym1, _, _, _ = headline(rec1, M=M2)
    print(f'   top-2 points since 2024-10: {len(t2_since)} (records {len([m for m in rec1 if yr[idx_of[m]] >= T0])} + runners-up {len(extra)}); lag vs US top-1: {h_t2} on {n2}; '
          f'symmetric top-2 vs US top-2: {h_sym}; records vs US top-2: {h_sym1}')
    mr = np.nanmean(L1[:, t1 >= T0], 0); mx = np.nanmean(np.array([lag(idx_of[m], M1) for m in extra]).T, 0) if extra else np.array([])
    print(f'   mean lag of the records {mr.mean():.1f} (SD between points {mr.std(ddof=1):.1f}); of the runners-up {mx.mean() if mx.size else float("nan"):.1f} (SD {mx.std(ddof=1) if mx.size > 1 else float("nan"):.1f}); runners-up: {[(m, round(float(v),1)) for m, v in zip(extra, mx)][:8]}')
