"""Crossing dates of the placed human tiers on the flagship's mode A (24 / 50 chains).

The tiers are the ones 5_place_humans.py placed on mode A. The frontier trend and the crossing
rule are the dashboard's (axis_forecast_inputs, placement_crossings). Mode C is left out: its
chains do not match mode A's axes (best match r 0.89 / 0.76 / 0.92 / 0.81, under the 0.8 cut of
align_to_reference_loadings; modeC_frame_refused.log). Writes crossings_modeA.csv beside this file.

    python 6_writeups/preprint/flagship_analysis/crossings_modeA.py
"""
import sys
from pathlib import Path

import arviz as az
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "2_model"))

from multiaxis_eci.analysis import (  # noqa: E402
    FLAGSHIP_TRACE, FitSpec, align_to_reference_loadings, attach_placement,
    axis_forecast_inputs, best_mode_chains, load_axis_titles, prepare_fit,
)
from multiaxis_eci.analysis.human_placement import placement_crossings  # noqa: E402
from multiaxis_eci.data import PROCESSED_FILE  # noqa: E402

trace, fit = FLAGSHIP_TRACE, FLAGSHIP_TRACE.parent
with xr.open_dataset(trace, group="posterior") as post:
    spec = FitSpec.from_trace(az.InferenceData(posterior=post.isel(draw=slice(0, 1))), trace)
raw = pd.read_csv(PROCESSED_FILE)
chains = best_mode_chains(trace)
idata = spec.open_posterior(keep=("A", "theta", "tau_A", "D"), path=trace, chains=chains)
data, _, _ = spec.load_data(idata)
view = align_to_reference_loadings(prepare_fit(idata, data), data, fit / "mirt_loadings.csv")
view = attach_placement(view, fit)
titles = load_axis_titles(fit, view, data)

parts = []
for k, name in enumerate(view.names):
    try:
        fc = axis_forecast_inputs(view.theta, k, data, raw, name, A_draws=view.A)["fc"]
    except ValueError as err:          # too few frontier candidates on this axis
        print(f"{name}: no forecast ({err})")
        continue
    cx = placement_crossings(view, k, fc, name)
    if len(cx):
        cx.insert(1, "axis_title", titles.get(name, name))
        parts.append(cx)

out = pd.concat(parts, ignore_index=True)
out.insert(0, "n_chains", len(chains))
out.to_csv(Path(__file__).with_name("crossings_modeA.csv"), index=False)
pd.set_option("display.width", 250, "display.max_columns", 30)
print(out.to_string())
