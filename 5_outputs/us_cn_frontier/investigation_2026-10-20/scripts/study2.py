"""Second batch: who carries the tails, median-path crossing, leader effort rules, kernel bandwidth by LOO, conditional lags of probable records."""
import sys
from pathlib import Path
import numpy as np, pandas as pd
ROOT = Path(__file__).resolve().parents[4]   # the repository
sys.path.insert(0, str(ROOT / '2_model'))
from multiaxis_eci.analysis.regimes import family_best, one_per_org_day, frontier_topk
from multiaxis_eci.analysis.forecast import _to_year
from multiaxis_eci.data import model_product
S = Path(sys.argv[1]); scope = sys.argv[2]
z = np.load(S / f'E_{scope}.npz', allow_pickle=True)
E, names, keep, lab, n_obs = z['E'].astype(float), z['names'], z['keep'], z['label'], z['n_obs']
dates = pd.to_datetime(pd.Series(z['date']).replace('', pd.NaT))
yr = _to_year(pd.DatetimeIndex(dates.fillna(pd.Timestamp('2000-01-01'))))
med = np.median(E, axis=0); sd = E.std(axis=0)
US = keep & (lab == 'US'); CN = keep & (lab == 'CN')
Sdraws = E.shape[0]
def tl(mask):
    return pd.DataFrame({'name': names[mask], 'release_date': dates[mask].values, 'mean': med[mask], 'sd': sd[mask],
                         'idx': np.flatnonzero(mask)}).sort_values('release_date', kind='stable').reset_index(drop=True)
def records(t):
    best = one_per_org_day(family_best(t))
    per_day = best.sort_values('mean', ascending=False).drop_duplicates('release_date').sort_values('release_date', kind='stable')
    return frontier_topk(per_day, 1)
cn_tl, us_tl = tl(CN), tl(US)
cn_rec = records(cn_tl); us_rec = records(us_tl)
idx_of = {m: i for i, m in enumerate(names)}
rec_idx = np.array([idx_of[m] for m in cn_rec]); rec_yr = yr[rec_idx]; rec_dates = dates[rec_idx].values
since = np.array([pd.Timestamp(d) >= pd.Timestamp('2024-10-01') for d in rec_dates])
us_idx = us_tl['idx'].to_numpy(); us_yr = yr[us_idx]; us_names = names[us_idx]

def lags_for(us_sel_idx, level_draws, t_r):
    yrs = yr[us_sel_idx]; o = np.argsort(yrs, kind='stable'); sel = us_sel_idx[o]; yrs = yrs[o]
    M = np.maximum.accumulate(E[:, sel], axis=1)
    reached = M >= level_draws[:, None]; any_ = reached.any(axis=1); first = np.argmax(reached, axis=1)
    return np.where(any_, (t_r - yrs[first]) * 12, np.nan), names[sel][first], any_

print(f'scope {scope}')
# (b) US median-records: n_obs and sd — what a coverage / SD rule on the leader side would drop
ur = us_tl[us_tl.name.isin(us_rec)]
ur = ur.assign(n_obs=n_obs[ur.idx])
print('\n(b) US records (median frontier) since 2024 with few scores or wide posteriors:')
print(ur[(ur.release_date >= '2024-01-01') & ((ur.n_obs <= 3) | (ur.sd >= 4))][['name', 'release_date', 'mean', 'sd', 'n_obs']].round(2).to_string(index=False))

# (a) who carries the crossing mass per record, with n_obs and sd
print('\n(a) first US model above each record (share, n_obs, posterior SD, lag it implies), records since 2024-10:')
for r in np.flatnonzero(since):
    lag, who, any_ = lags_for(us_idx, E[:, rec_idx[r]], rec_yr[r])
    df = pd.DataFrame({'who': who[any_], 'lag': lag[any_]})
    g = df.groupby('who').agg(share=('lag', 'size'), lag=('lag', 'first')).sort_values('share', ascending=False)
    g['share'] /= any_.sum(); g['n_obs'] = [n_obs[idx_of[m]] for m in g.index]; g['sd'] = [sd[idx_of[m]] for m in g.index]
    thin_mass = g.loc[g.n_obs <= 3, 'share'].sum()
    print(f'  {cn_rec[r]:<32s} {pd.Timestamp(rec_dates[r]).date()}  mass on US models with <=3 scores: {thin_mass:.0%}')
    print('     ' + ' | '.join(f'{m[:28]} {s.share:.0%} n={s.n_obs} sd={s.sd:.1f} lag={s.lag:.1f}' for m, s in g.head(5).iterrows()))

# (c) median-path crossing: T(x) = first date the posterior median of the US running max is >= x
o = np.argsort(us_yr, kind='stable'); sel = us_idx[o]; yrs = us_yr[o]
M = np.maximum.accumulate(E[:, sel], axis=1); mpath = np.median(M, axis=0); q10 = np.percentile(M, 10, axis=0); q90 = np.percentile(M, 90, axis=0)
def T_of(path, x):
    reached = path[None, :] >= x[:, None]; any_ = reached.any(axis=1); first = np.argmax(reached, axis=1)
    return np.where(any_, yrs[first], np.nan)
print('\n(c) crossing read on the posterior-median path of the US running max (CN uncertainty propagated; q10/q90 paths give the US-side range):')
rows = []
for r in range(len(cn_rec)):
    lv = E[:, rec_idx[r]]
    l_med = (rec_yr[r] - T_of(mpath, lv)) * 12; l_lo = (rec_yr[r] - T_of(q90, lv)) * 12; l_hi = (rec_yr[r] - T_of(q10, lv)) * 12
    rows.append(dict(name=cn_rec[r], med=np.nanmedian(l_med), sd=np.nanstd(l_med), us_lo=np.nanmedian(l_lo), us_hi=np.nanmedian(l_hi)))
c = pd.DataFrame(rows)
print(c[since].round(2).to_string(index=False))
m_, s_ = c.med.to_numpy()[since], np.maximum(c.sd.to_numpy()[since], 0.3); w_ = 1/s_**2
print(f'   pooled {(w_*m_).sum()/w_.sum():.2f}; SD of medians {m_.std(ddof=1):.2f}; mean CN-side SD {s_.mean():.2f}; chi2 ratio {((m_-(w_*m_).sum()/w_.sum())**2*w_).sum()/(since.sum()-1):.2f}')

# (d) leader effort rules: one effort per family (best median) ; most-scored effort per family
fam = pd.Series([model_product(m) for m in us_names], index=us_names)
best_eff = us_tl.assign(fam=fam.values).sort_values(['fam', 'mean'], ascending=[True, False]).drop_duplicates('fam')
most_eff = us_tl.assign(fam=fam.values, n=n_obs[us_idx]).sort_values(['fam', 'n', 'mean'], ascending=[True, False, False]).drop_duplicates('fam')
print(f'\n(d) leader side: one effort per family — best median ({len(best_eff)} US) / most scores ({len(most_eff)} US):')
for label, sub in (('best-median effort', best_eff), ('most-scored effort', most_eff)):
    meds, sds = [], []
    for r in range(len(cn_rec)):
        lag, _, any_ = lags_for(sub.idx.to_numpy(), E[:, rec_idx[r]], rec_yr[r]); f = lag[any_]
        meds.append(np.median(f)); sds.append(f.std())
    meds, sds = np.array(meds)[since], np.array(sds)[since]; w = 1/sds**2
    print(f'   {label:<20s} pooled {(w*meds).sum()/w.sum():.2f}  mean post SD {sds.mean():.2f}  SD of medians {meds.std(ddof=1):.2f}  medians: {np.round(meds,1).tolist()}')

# (e) kernel bandwidth by leave-one-out on the records since 2024-10 (per-draw lags, weighted by posterior precision)
st = np.load(S / f'study_{scope}.npz', allow_pickle=True); L0 = st['L0']
lm = np.nanmedian(L0, axis=0); ls = np.nanstd(L0, axis=0)
t_all = rec_yr; use = since & np.isfinite(lm)
print('\n(e) leave-one-out predictive error of the kernel smoother through the record lags (records since 2024-10):')
for h in (3, 6, 9, 12, 18, 24, 36, 1000):
    hb = h / 12; err = []
    for i in np.flatnonzero(use):
        others = use.copy(); others[i] = False
        w = np.exp(-0.5 * ((t_all[others] - t_all[i]) / hb) ** 2) / ls[others]**2
        pred = (w * lm[others]).sum() / w.sum()
        err.append(((lm[i] - pred) / ls[i]) ** 2)
    print(f'   bandwidth {h:>4d} months: mean standardized LOO error {np.mean(err):.3f}')

# (f) probable records: lag conditional on the draws where the model holds the CN frontier
cn_idx = cn_tl['idx'].to_numpy(); cn_yr = yr[cn_idx]; o = np.argsort(cn_yr, kind='stable'); cn_idx, cn_yr = cn_idx[o], cn_yr[o]
E_cn = E[:, cn_idx]; prev = np.maximum.accumulate(np.c_[np.full((Sdraws, 1), -np.inf), E_cn[:, :-1]], axis=1)
is_rec = E_cn >= prev; p_rec = is_rec.mean(axis=0)
print('\n(f) CN candidates since 2024-10 with P(record) >= 0.2: lag in the draws where they hold the frontier')
for j in np.flatnonzero((p_rec >= 0.2) & (cn_yr >= _to_year(pd.DatetimeIndex(['2024-10-01']))[0])):
    lag, _, any_ = lags_for(us_idx, E_cn[:, j], cn_yr[j]); f = lag[is_rec[:, j] & any_]; fu = lag[any_]
    print(f'   {names[cn_idx[j]]:<32s} {dates[cn_idx[j]].date()} P={p_rec[j]:.2f} n_obs={n_obs[cn_idx[j]]:2d}  lag|record {np.median(f):5.1f} ±{f.std():.1f}   unconditional {np.median(fu):5.1f}  {"REC" if names[cn_idx[j]] in cn_rec else ""}')
