import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.dates as mdates
from pathlib import Path
S = Path(sys.argv[1]); out = Path(sys.argv[2])
sys.path.insert(0, str(Path(__file__).resolve().parents[4] / '2_model'))
from multiaxis_eci.analysis.forecast import _to_year
today = pd.Timestamp('2026-10-01')
fig, axes = plt.subplots(3, 1, figsize=(9, 11), sharex=True, constrained_layout=True)
for ax, scope, title in zip(axes, ['all', 'public', 'private'], ['All benchmarks (84)', 'Public benchmarks (58)', 'Private benchmarks (14)']):
    st = np.load(S / f'study_{scope}.npz', allow_pickle=True); L0, rec, rd = st['L0'], st['rec'], pd.to_datetime(st['rec_dates'])
    lf = np.load(S / f'lagf_{scope}.npz', allow_pickle=True); lagf, grid = lf['lagf'], pd.to_datetime(lf['grid'])
    w = rd >= pd.Timestamp('2024-01-01')
    med = np.nanmedian(L0, axis=0); mean = np.nanmean(L0, axis=0); q25, q75 = np.nanpercentile(L0, [25, 75], axis=0)
    # current reading: record dots + 6-month Gaussian kernel of the per-draw lags
    yrs = _to_year(pd.DatetimeIndex(rd)); gy = _to_year(grid)
    k = np.exp(-0.5 * ((gy[:, None] - yrs[None, :]) / 0.5) ** 2); fin = np.isfinite(L0)
    num = np.where(fin, L0, 0) @ k.T; den = fin.astype(float) @ k.T
    with np.errstate(invalid='ignore', divide='ignore'): curves = np.where(den > 1.5, num / den, np.nan)
    ax.plot(grid, np.nanmedian(curves, axis=0), color='#888', lw=1.4, ls='--', label='current: 6-month kernel through the records (median)')
    ax.vlines(rd[w], q25[w], q75[w], color='#555', lw=0.8, alpha=0.5)
    ax.plot(rd[w], med[w], 'o', color='#333', ms=5, label='current: record lag, posterior median [50%]')
    ax.plot(rd[w], mean[w], 'o', mfc='none', mec='#333', ms=5, label='record lag, posterior mean')
    # proposal: the lag as a function of calendar time
    lo, mdn, hi = np.nanpercentile(lagf, [10, 50, 90], axis=0)
    ax.step(grid, mdn, where='post', color='#3b7a3b', lw=0.9, alpha=0.6, label='months behind at each date (per-draw frontiers, median)')
    roll = pd.DataFrame(lagf.T, index=grid).rolling(12, min_periods=12).mean().T.to_numpy()
    rlo, rmd, rhi = np.nanpercentile(roll, [10, 50, 90], axis=0)
    ax.fill_between(grid, rlo, rhi, color='#24467a', alpha=0.15, lw=0)
    ax.plot(grid, rmd, color='#24467a', lw=2.2, label='12-month trailing mean of the above, median [80%]')
    # constant lag over the records since 2024-10
    since = st['since']; pooled = np.nanmean(L0[:, since], axis=1)
    plo, pmd, phi = np.percentile(pooled, [10, 50, 90])
    x0 = pd.Timestamp('2024-10-01')
    ax.fill_between([x0, today], [plo, plo], [phi, phi], color='#b8912a', alpha=0.18, lw=0)
    ax.hlines(pmd, x0, today, color='#b8912a', lw=1.6, ls=':', label=f'mean of the record lags since 2024-10: {pmd:.1f} [{plo:.1f}, {phi:.1f}]')
    ax.axvline(today, color='#999', lw=0.8, ls=(0, (1, 2)))
    ax.set_ylim(-1, 20); ax.set_ylabel('Months behind the US frontier'); ax.set_title(title, loc='left', fontsize=11)
    ax.grid(axis='y', color='#e5e5e5'); ax.set_xlim(pd.Timestamp('2024-01-01'), pd.Timestamp('2027-01-01'))
    ax.text(today, 19, f'12-month mean today {rmd[-1]:.1f} [{rlo[-1]:.1f}, {rhi[-1]:.1f}]', ha='right', va='top', fontsize=9, color='#24467a')
axes[0].legend(fontsize=8, loc='upper right', frameon=False)
axes[-1].xaxis.set_major_locator(mdates.YearLocator()); axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
fig.suptitle('US-CN months-behind: current reading vs. the lag as a function of calendar time (data 2026-10-01)', fontsize=11)
fig.savefig(out, dpi=160); print('wrote', out)
