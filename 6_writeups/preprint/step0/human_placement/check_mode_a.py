"""Human placement before and after the 'at least this score' reading of ceiling scores.

Runs analysis.human_placement.run_placement twice on the mode A trace of the 5-chain flagship
refit (data 20261019): CEILING_SD = 0 (only scores >= 0.99 are ceiling scores, the old rule) and
CEILING_SD = 2 (also scores within 2 noise SDs of the fitted ceiling). Writes both placements
under before/ and after/ here, never beside the trace.

    python 6_writeups/preprint/step0/human_placement/check_mode_a.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "2_model"))
sys.path.insert(0, str(ROOT / "4_diagnostics"))

import arviz as az  # noqa: E402
import xarray as xr  # noqa: E402

from multiaxis_eci.analysis import (FitSpec, align_to_reference_loadings, load_axis_titles,  # noqa: E402
                                    prepare_fit, run_placement)
from multiaxis_eci.analysis import human_placement as hp  # noqa: E402

OUT = Path(__file__).resolve().parent
TRACE = (ROOT / "deliverables/flagship_refit_20261021/exploratory_modes/mode_A/"
         "mirt_lineageprior_lineagebm/trace_mirt_k4_lineageprior_lineagebm.nc")
# mode A's own axis names borrow the flagship's judgments of the same content
RULES = {"Scientific Knowledge & Mathematics": "Scientific Knowledge & Reasoning",
         "General Knowledge & Commonsense": "Commonsense & Factual Knowledge"}

with xr.open_dataset(TRACE, group="posterior") as post:
    spec = FitSpec.from_trace(az.InferenceData(posterior=post.isel(draw=slice(0, 1))), TRACE)
idata = spec.open_posterior(keep=("A", "theta", "tau_A", "D", "phi_b", "ceiling_d"), path=TRACE)
ceiling_d = idata.posterior["ceiling_d"].median(("chain", "draw")).values
data, floor_c, _ = spec.load_data(idata)
view = align_to_reference_loadings(prepare_fit(idata, data), data, TRACE.parent / "mirt_loadings.csv")
titles = load_axis_titles(TRACE.parent, view, data)

for label, sd in (("before", 0.0), ("after", 2.0)):
    hp.CEILING_SD = sd
    out = run_placement(TRACE, view, data, spec, titles, floor_c, out_dir=OUT / label,
                        rule_titles=RULES, ceiling_d=ceiling_d)
    print(f"\n== {label} (CEILING_SD {sd}) ==")
    for ax in out["axes"]:
        print(f"{ax['title']}: drawn {ax['drawn']}; lower bounds {ax.get('lower_bounds', {})}")
        for t, q in ax["tiers"].items():
            print(f"   {'*' if t in ax['drawn'] else ' '} {t:34s} {q}")
        print("   at ceiling:", [(r["tier"], r["benchmark"], r["observed"])
                                  for r in ax["consistency"] if r["at_ceiling"]])
