"""Place the human tiers on a fit without human baselines, axis by axis (analysis.human_placement).

Reads the fit on the chains its deliverables read (`best_mode_chains`) and on its display frame
(`mirt_loadings.csv`, its axis names), places the tiers, and writes `human_placement.json` and
`human_placement_draws.npz` beside the trace. The figures (3_plot_mirt.py, the dashboard, the
README's) draw those tiers on that fit. Run it after `diagnose_chains.py --write-tables`, and
again whenever the selected chains, the axis names, the human baselines or
`1_curated/human_axis_content.csv` change.

    python 4_diagnostics/5_place_humans.py [--trace T]     # default: the flagship
    python 4_diagnostics/5_place_humans.py --secondary     # the secondary mode's chains

`--secondary` places the tiers on the chains the modes file names under `secondary_chains` and
writes into `secondary_mode/` beside the trace, which 3_plot_mirt.py reads for the secondary
mode's figures. A secondary mode with axes of its own (`diagnose_chains.py --write-tables
--secondary`) is placed on its own frame and under its own names, `secondary_mode/`'s
`mirt_loadings.csv` and `axis_names.json`.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "2_model"))

import arviz as az  # noqa: E402
import xarray as xr  # noqa: E402

from multiaxis_eci.analysis import (  # noqa: E402
    FLAGSHIP_TRACE,
    FitSpec,
    align_to_reference_loadings,
    best_mode_chains,
    secondary_frame_dir,
    secondary_mode_chains,
    load_axis_titles,
    prepare_fit,
    run_placement,
)
from multiaxis_eci.analysis.human_placement import SECONDARY_DIR  # noqa: E402


# Axis titles of exploration fits that borrow the flagship axes' content judgments
# (1_curated/human_axis_content.csv): semicomp0 mode A, named by the user on 2026-10-08.
RULE_TITLES = {
    # The geomin-rotated / pinned frame's axis 2 (named by the user on 2026-10-27) is the
    # flagship's agentic axis under another name; its axis 3 has judgments of its own.
    "Agentic Programming & Computer Use": "Agentic Capabilities & Programming",
    "Cyber Capabilities & Reasoning": "Cyber Capabilities",
    "Fluid Intelligence & Mathematics": "Fluid Intelligence & Formal Sciences",
    "Scientific Knowledge": "Scientific Knowledge & Reasoning",
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--trace", type=Path, default=FLAGSHIP_TRACE)
    ap.add_argument("--secondary", action="store_true",
                    help="place on the secondary mode's chains, into secondary_mode/")
    args = ap.parse_args()
    trace = args.trace
    with xr.open_dataset(trace, group="posterior") as post:
        spec = FitSpec.from_trace(az.InferenceData(posterior=post.isel(draw=slice(0, 1))), trace)
    chains = secondary_mode_chains(trace) if args.secondary else best_mode_chains(trace)
    if args.secondary and chains is None:
        sys.exit(f"{trace.name}: its modes file names no secondary_chains")
    out_dir = trace.parent / SECONDARY_DIR if args.secondary else trace.parent
    if spec.link not in ("linear", "semicomp0"):
        sys.exit(f"{trace.name}: the placement handles the compensatory and semicomp0 links, "
                 f"not {spec.link}")
    idata = spec.open_posterior(keep=("A", "theta", "tau_A", "D", "phi_b", "comp_beta", "ceiling_d"),
                                path=trace, chains=chains)
    # semicomp0: the softmin eta at the chains' median sharpness
    beta = (float(idata.posterior["comp_beta"].median()) if spec.link == "semicomp0" else None)
    # --ceiling-noise: the curve the tiers are read on stops at the fit's own ceilings
    ceiling_d = (idata.posterior["ceiling_d"].median(("chain", "draw")).values
                 if "ceiling_d" in idata.posterior else None)
    data, floor_c, _ = spec.load_data(idata)
    view = prepare_fit(idata, data)
    # The frame and the names: the fit's, or a secondary mode's own (secondary_frame_dir)
    frame = (secondary_frame_dir(trace) if args.secondary else None) or trace.parent
    ref = frame / "mirt_loadings.csv"
    if ref.exists():
        view = align_to_reference_loadings(view, data, ref)
    titles = load_axis_titles(frame, view, data)
    print(f"{trace.name}: chains {chains}; axes {titles}"
          + (f"; softmin beta {beta:.2f}" if beta is not None else ""), flush=True)
    out = run_placement(trace, view, data, spec, titles, floor_c, out_dir=out_dir, beta=beta,
                        rule_titles=RULE_TITLES, ceiling_d=ceiling_d)
    for ax in out["axes"]:
        print(f"\n{ax['axis']}: {ax['title'] or '(no confirmed name: no tier)'}")
        if ax["unjudged_benchmarks"]:
            print(f"  NOT YET JUDGED for this axis (kept): {ax['unjudged_benchmarks']} "
                  "-> add them to 1_curated/human_axis_content.csv")
        if ax["convergence"]:
            print(f"  convergence {ax['convergence']}")
        for t in ax.get("order", []):
            flag = "*" if t in ax["drawn"] else " "
            print(f"   {flag} {t:34s} {ax['tiers'][t]}")
        for r in ax["consistency"]:
            print(f"     {r['tier']:30s} {r['benchmark']:28s} observed {r['observed']:.2f} "
                  f"predicted {r['predicted']:.2f} [{r['predicted_q10']:.2f}, {r['predicted_q90']:.2f}]"
                  f"  first model at the score: {r['first_model_at_score'] or 'none yet'}")
    print(f"\nwrote {out_dir.relative_to(trace.parent.parent)}/human_placement.json and "
          "human_placement_draws.npz")


if __name__ == "__main__":
    main()
