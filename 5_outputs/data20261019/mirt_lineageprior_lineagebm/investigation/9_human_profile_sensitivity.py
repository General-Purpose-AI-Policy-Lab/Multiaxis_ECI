"""How much the human placement depends on the profile its other axes are set to.

The placement (analysis.human_placement) sets a tier's other axes to the profile of a typical
model at the same ability on axis k: a regression of the other axes on axis k over all models.
The frontier models sit above that profile on the axes that correlate with k (claude-fable-5_max:
theta 1.9 on axes 1 and 2 at theta 1.0 on axis 4), so a tier read at the typical profile must
earn on axis k what the frontier models earn elsewhere. Three profiles, same scores, same order:

* typical  — the deliverables' rule (regression over every model);
* frontier — the same regression over the axis's frontier models only (the figures' red and
             purple dots, `axis_forecast_inputs`);
* today    — the other axes held constant at the median of the frontier models released in
             the last 12 months.

Writes human_profile_sensitivity.csv (tier medians and 80% intervals per axis and profile, and
the other axes' profile at ability 1 on the axis); the flagship's own placement is not touched.
"""
import sys
import tempfile
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
sys.path.insert(0, str(ROOT / "2_model"))
import multiaxis_eci.analysis.human_placement as hp  # noqa: E402
from multiaxis_eci.analysis import (  # noqa: E402
    FitSpec,
    align_to_reference_loadings,
    best_mode_chains,
    load_axis_titles,
    prepare_fit,
)
from multiaxis_eci.analysis.forecast import _release_dates, axis_forecast_inputs  # noqa: E402
from multiaxis_eci.data import PROCESSED_FILE  # noqa: E402

TRACE = HERE.parent / "trace_mirt_k4_lineageprior_lineagebm.nc"
TODAY = pd.Timestamp("2026-10-21")

with xr.open_dataset(TRACE, group="posterior") as post:
    spec = FitSpec.from_trace(az.InferenceData(posterior=post.isel(draw=slice(0, 1))), TRACE)
chains = best_mode_chains(TRACE)
idata = spec.open_posterior(keep=("A", "theta", "tau_A", "D", "phi_b", "comp_beta", "ceiling_d"),
                            path=TRACE, chains=chains)
ceiling_d = idata.posterior["ceiling_d"].median(("chain", "draw")).values
data, floor_c, _ = spec.load_data(idata)
view = align_to_reference_loadings(prepare_fit(idata, data), data, HERE.parent / "mirt_loadings.csv")
titles = load_axis_titles(HERE.parent, view, data)
raw = pd.read_csv(PROCESSED_FILE)
names = data.mlookup.sort_values("model_idx")["model"].tolist()
dates = pd.to_datetime(_release_dates(raw)[0], errors="coerce")
theta_med = np.median(view.theta, 0)

# Each axis's frontier models, as the forecast figures draw them
frontier = {}
for k, name in enumerate(view.names):
    fc = axis_forecast_inputs(view.theta, k, data, raw, titles.get(name, name), A_draws=view.A)["fc"]
    frontier[k] = np.array([names.index(m) for m in fc.frontier_names])
    recent = [i for i in frontier[k] if dates.get(names[i], pd.NaT) >= TODAY - pd.DateOffset(years=1)]
    print(f"{name}: {len(frontier[k])} frontier models, {len(recent)} in the last 12 months")

typical_profile = hp._profile


def frontier_profile(theta_models, k):
    return typical_profile(theta_models[frontier[k]], k)


def today_profile(theta_models, k):
    others = [j for j in range(theta_models.shape[1]) if j != k]
    recent = [i for i in frontier[k] if dates.get(names[i], pd.NaT) >= TODAY - pd.DateOffset(years=1)]
    level = np.median(theta_models[recent][:, others], 0)
    return others, np.vstack([level, np.zeros(len(others))])


rows = []
for label, prof in [("typical", typical_profile), ("frontier", frontier_profile),
                    ("today", today_profile)]:
    hp._profile = prof
    with tempfile.TemporaryDirectory() as tmp:
        out = hp.run_placement(TRACE, view, data, spec, titles, floor_c, out_dir=tmp,
                               ceiling_d=ceiling_d)
    for k, ax in enumerate(out["axes"]):
        others, coef = prof(theta_med, k)
        for t in ax["drawn"]:
            lo, med, hi = ax["tiers"][t]
            rows.append(dict(axis=ax["title"], profile=label, tier=t, median=med, q10=lo, q90=hi,
                             profile_at_1=" ".join(f"{coef[0][j] + coef[1][j]:.2f}"
                                                   for j in range(len(others)))))
hp._profile = typical_profile
res = pd.DataFrame(rows)
res.to_csv(HERE / "human_profile_sensitivity.csv", index=False)
wide = res.pivot_table(index=["axis", "tier"], columns="profile", values="median", sort=False)
pd.set_option("display.width", 200)
print(wide[["typical", "frontier", "today"]].round(2).to_string())
