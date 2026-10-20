import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.dates as mdates
from pathlib import Path
d = Path("5_outputs/data20261001/comparisons/")
c = pd.read_csv(d / "frontier_gap_country_all_candidates.csv", parse_dates=["release_date"])
r = pd.read_csv(d / "frontier_gap_country_all_records.csv", parse_dates=["release_date"])
z = np.load(d / "frontier_gap_country_all.npz"); grid = pd.DatetimeIndex(z["lag_grid"]); fn, fm = z["lag_fn"], z["lag_fn_mean"]
w0, w1 = pd.Timestamp("2024-07-01"), pd.Timestamp("2026-10-01")
ENCRE, US, CN = "#14243d", "#24467a", "#b03a2e"
def front(g):
    t = c[(c.group == g) & (c.sd <= (6 if g == "US" else 99))].sort_values("release_date")
    t = t.assign(f=t["mean"].cummax()); return t[t.f.diff().fillna(1) > 0]
fu, fc = front("US"), front("CN")
# the US crossing date of a level, on the median frontier
def cross(level):
    ok = fu[fu.f >= level]; return ok.release_date.iloc[0] if len(ok) else pd.NaT
fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 8.5), sharex=True, constrained_layout=True,
                             gridspec_kw={"height_ratios": [1.15, 1]})
# ── top: the two frontiers (posterior medians) ──
for t, col, lab in ((fu, US, "frontière US (max courant des médianes)"), (fc, CN, "frontière chinoise")):
    a1.step(list(t.release_date) + [w1], list(t.f) + [t.f.iloc[-1]], where="post", color=col, lw=2.2, label=lab)
    a1.plot(t.release_date, t.f, "o", color=col, ms=4)
for k, (_, rr) in enumerate(r[r.release_date >= w0].iterrows()):
    below = k % 2 == 0
    a1.annotate(rr["name"].replace("_thinking", "").replace("_max", "").replace("-A22B-Thinking-2507", "-Thinking"),
                (rr.release_date, rr.level_eci_median), xytext=(4, -10 if below else 7), textcoords="offset points",
                ha="left", va="top" if below else "bottom", fontsize=7, color=CN)
# three example months: the lag is the horizontal distance back to the US frontier
for m, dy in (("2025-03-01", 0), ("2025-10-01", 0), ("2026-03-01", 0)):
    t = pd.Timestamp(m); lvl = fc[fc.release_date <= t].f.iloc[-1]; x0 = cross(lvl)
    a1.annotate("", xy=(x0, lvl), xytext=(t, lvl), arrowprops=dict(arrowstyle="<-", color=ENCRE, lw=1.4, shrinkA=0, shrinkB=0))
    a1.plot([t, t], [lvl - 4, lvl], color=ENCRE, lw=0.8, ls=":")
    a1.text(x0 + (t - x0) / 2, lvl + 1.2, f"{(t - x0).days / 30.44:.1f} mois", ha="center", va="bottom", fontsize=8.5, color=ENCRE,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))
    a1.text(t, lvl - 4.5, f"{t:%Y-%m}", ha="center", va="top", fontsize=7.5, color="#666")
a1.set_ylabel("ECI-H (médiane postérieure)"); a1.set_ylim(118, 190); a1.legend(loc="upper left", frameon=False, fontsize=9)
a1.set_title("Étape 5 — à chaque mois t : le niveau de la frontière chinoise en t, et la date où la frontière US l'avait atteint",
             loc="left", fontsize=10.5, color=ENCRE)
a1.grid(axis="y", color="#eee")
# ── bottom: the sawtooth per month, the record dots, the 12-month mean ──
inside = (grid >= w0) & (grid <= w1)
med = np.nanmedian(fn, axis=0); lo, hi = np.nanpercentile(fn, [10, 90], axis=0)
a2.step(grid[inside], med[inside], where="post", color="#3b7a3b", lw=1.3, label="retard du mois (médiane entre tirages) : la dent de scie")
a2.fill_between(grid[inside], lo[inside], hi[inside], step="post", color="#3b7a3b", alpha=0.08, lw=0)
rr = r[(r.release_date >= w0)]
a2.vlines(rr.release_date, rr.lag_hdi50_low, rr.lag_hdi50_high, color="#333", lw=0.8, alpha=0.5)
a2.plot(rr.release_date, rr.lag_months_mean, "o", color="#333", ms=5, label="retard aux records (moyenne postérieure, barre 50 %) : les minima")
mm = np.nanmedian(fm, axis=0); mlo, mhi = np.nanpercentile(fm, [10, 90], axis=0)
a2.fill_between(grid[inside], mlo[inside], mhi[inside], color=US, alpha=0.15, lw=0)
a2.plot(grid[inside], mm[inside], color=US, lw=2.4, label="moyenne glissante 12 mois (médiane, bande 10-90 %) : la courbe de la figure")
for m in ("2025-03-01", "2025-10-01", "2026-03-01"):
    t = pd.Timestamp(m); g = grid.get_loc(t); a2.plot(t, med[g], "s", color=ENCRE, ms=6, mfc="white", zorder=5)
a2.text(grid[inside][-1], mm[inside][-1], f"  {mm[inside][-1]:.1f}", color=US, va="center", fontsize=9, fontweight="bold")
a2.set_ylabel("Mois de retard sur la frontière US"); a2.set_ylim(0, 12); a2.grid(axis="y", color="#eee")
a2.legend(loc="upper right", frameon=False, fontsize=8.5)
a2.xaxis.set_major_locator(mdates.MonthLocator(bymonth=(1, 7))); a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
a2.set_xlim(w0, w1)
a2.set_xlabel("Scope « tous les benchmarks », données 2026-10-01. Les carrés blancs sont les trois mois fléchés en haut.\n"
              "Par tirage postérieur, la même lecture est faite sur les tirages et non sur les médianes ; les bandes en donnent l'étendue (10-90 %).",
              fontsize=8, color="#555", loc="left")
out = Path("5_outputs/us_cn_frontier/investigation_2026-10-20/etape5_lag_function.png")
fig.savefig(out, dpi=160); print(out)
