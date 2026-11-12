"""Crossing dates of the placed human tiers on the flagship, mode A (24 / 50 chains) and mode C (18 / 50).

Each mode is read on its own frame and under its own axis names: mode A on the fit folder's
mirt_loadings.csv, mode C on secondary_mode/ (its chains do not match mode A's axes: best match
r 0.89 / 0.76 / 0.92 / 0.81, modeC_frame_refused.log). The tiers are the ones 5_place_humans.py
placed on that mode. The frontier trend and the crossing rule are the dashboard's
(axis_forecast_inputs, placement_crossings). Writes crossings_modeA.csv and crossings_modeC.csv.

    python 6_writeups/preprint/flagship_analysis/crossings_by_mode.py
"""
import sys
from pathlib import Path

import arviz as az
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "2_model"))

from multiaxis_eci.analysis import (  # noqa: E402
    FLAGSHIP_TRACE, FitSpec, align_to_reference_loadings, attach_placement, axis_forecast_inputs,
    best_mode_chains, load_axis_titles, prepare_fit, secondary_frame_dir, secondary_mode_chains,
)
from multiaxis_eci.analysis.human_placement import placement_crossings  # noqa: E402
from multiaxis_eci.data import PROCESSED_FILE  # noqa: E402

trace, fit = FLAGSHIP_TRACE, FLAGSHIP_TRACE.parent
with xr.open_dataset(trace, group="posterior") as post:
    spec = FitSpec.from_trace(az.InferenceData(posterior=post.isel(draw=slice(0, 1))), trace)
raw = pd.read_csv(PROCESSED_FILE)
pd.set_option("display.width", 250, "display.max_columns", 30)

for mode, chains, frame in (("A", best_mode_chains(trace), fit),
                            ("C", secondary_mode_chains(trace), secondary_frame_dir(trace))):
    idata = spec.open_posterior(keep=("A", "theta", "tau_A", "D"), path=trace, chains=chains)
    data, _, _ = spec.load_data(idata)
    view = align_to_reference_loadings(prepare_fit(idata, data), data, frame / "mirt_loadings.csv")
    view = attach_placement(view, frame)
    titles = load_axis_titles(frame, view, data)
    parts = []
    for k, name in enumerate(view.names):
        try:
            fc = axis_forecast_inputs(view.theta, k, data, raw, name, A_draws=view.A)["fc"]
        except ValueError as err:          # too few frontier candidates on this axis
            print(f"mode {mode}, {name}: no forecast ({err})")
            continue
        cx = placement_crossings(view, k, fc, name)
        if len(cx):
            cx.insert(1, "axis_title", titles.get(name, name))
            parts.append(cx)
    out = pd.concat(parts, ignore_index=True)
    out.insert(0, "n_chains", len(chains))
    out.to_csv(Path(__file__).with_name(f"crossings_mode{mode}.csv"), index=False)
    print(f"\nmode {mode}\n{out.to_string()}")
