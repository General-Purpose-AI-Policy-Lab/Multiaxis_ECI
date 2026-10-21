"""Where do the human tiers sit in the joint fit (tiers fitted with the models) against the flagship's
per-axis placement (tiers placed after a fit on the models alone)?

The joint fit's chain-mean loadings are matched to the flagship's primary-mode axes by their
signature benchmarks (axis_names.json); every ability is put in units of the MODELS' spread on that
axis ((theta - mean over models) / sd over models), the flagship's placement in the same units over
its own models. Per axis and tier: the joint fit's median over its chains, its range, the flagship's
placement, and the difference.

    python 5_outputs/data20261019/mirt_lineageprior_lineagebm_noceiling_humanmerge/compare_tiers.py
"""
import json
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
FLAG = HERE.parent / "mirt_lineageprior_lineagebm_noceiling"
JOINT = HERE / "trace_mirt_k4_lineageprior_lineagebm_noceiling_humanmerge.nc"
TIERS = ["High School Qualifier", "High School Top Performer", "Average Human", "Skilled Generalist",
         "Domain Expert", "Top Performer", "Committee of Average Humans",
         "Committee of Skilled Generalists", "Committee of Domain Experts"]

names = json.loads((FLAG / "axis_names.json").read_text())["axes"]
place = json.loads((FLAG / "human_placement.json").read_text())["axes"]
modes = json.loads((FLAG / "mirt_modes_trace_mirt_k4_lineageprior_lineagebm_noceiling.json").read_text())
fchains = modes["selected_chains"]["chains"]
with xr.open_dataset(FLAG / "trace_mirt_k4_lineageprior_lineagebm_noceiling.nc", group="posterior") as p:
    th_f = p["theta"].isel(chain=fchains).values.mean((0, 1))
mu_f, sd_f = th_f.mean(0), th_f.std(0)

p = xr.open_dataset(JOINT, group="posterior")
bench = [str(b) for b in p["bench"].values]
models = [str(m) for m in p["model"].values]
hum = [models.index(t) for t in TIERS if t in models]
mod = [i for i, m in enumerate(models) if m not in TIERS]
SIG = [[bench.index(b) for b in names[f"axis{k + 1}"]["signature"] if b in bench] for k in range(4)]
A = p["A"].values.mean(1)                                    # chain x bench x K
TH = p["theta"].values.mean(1)                               # chain x model x K
rows = {}
for c in range(p.sizes["chain"]):
    share = A[c] ** 2 / (A[c] ** 2).sum(1, keepdims=True)
    S = np.array([[share[idx, k].mean() for k in range(4)] for idx in SIG])
    r, col = linear_sum_assignment(-S)
    order = np.empty(4, int)
    order[r] = col
    th = TH[c][:, order]
    z = (th - th[mod].mean(0)) / th[mod].std(0)
    rows[c] = dict(z=z, sig=S[r, col], front=z[mod].max(0))
print(f"{len(rows)} chains; signature share per axis (median over chains): "
      f"{np.round(np.median([r['sig'] for r in rows.values()], 0), 2).tolist()}")
for k in range(4):
    title = names[f"axis{k + 1}"]["title"]
    share = A.mean(0) ** 2 / (A.mean(0) ** 2).sum(1, keepdims=True)
    print(f"\n### {title}  (frontier model: joint {np.median([r['front'][k] for r in rows.values()]):+.1f} sd, "
          f"placement fit {((th_f[:, k].max() - mu_f[k]) / sd_f[k]):+.1f} sd)")
    print("| tier | joint (median) | range over chains | per-axis placement | joint - placement |")
    print("|---|---|---|---|---|")
    ax = place[k]
    for t in TIERS:
        if t not in models:
            continue
        v = np.array([r["z"][models.index(t), k] for r in rows.values()])
        q = ax["tiers"].get(t)
        ref = (q[1] - mu_f[k]) / sd_f[k] if q else np.nan
        drawn = "" if t in ax.get("drawn", []) else " (not drawn)"
        print(f"| {t}{drawn} | {np.median(v):+.2f} | {v.min():+.2f} … {v.max():+.2f} | {ref:+.2f} | "
              f"{np.median(v) - ref:+.2f} |")
