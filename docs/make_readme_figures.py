"""The README's headline figures, rendered from the current flagship fit into docs/figures/.

The forecast panel of each axis (`frontier_trend_fig` in its standalone design, the one
the per-fit folders carry) and the loadings grid (`loadings_grid_fig`, the top 20
benchmarks per axis), in English. When `diagnose_chains.py --write-modes` found more than
one posterior mode, they are drawn on the majority chains, the per-fit folders' `_majority`
set, but without the chain-group note those renders carry in their title. Same builders,
same forecast rule (`viz.dashboard.forecast_figures`), so the README cannot drift from the
outputs; only the trace and the destination are this script's.

    python docs/make_readme_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "2_model"))

from multiaxis_eci.analysis import (  # noqa: E402
    FLAGSHIP,
    FLAGSHIP_TRACE,
    align_to_reference_loadings,
    load_axis_titles,
    loadings_table,
    prepare_fit,
)
from multiaxis_eci.data import PROCESSED_FILE  # noqa: E402
from multiaxis_eci.scripts import load as _load_script  # noqa: E402
from multiaxis_eci.viz import figure_filename, loadings_grid_fig  # noqa: E402
from multiaxis_eci.viz.dashboard import forecast_figures  # noqa: E402

OUT = ROOT / "docs" / "figures"
_plot = _load_script("4_diagnostics/3_plot_mirt.py")        # PLOT_VARS, chain_split


def main():
    trace = FLAGSHIP_TRACE
    idata = FLAGSHIP.open_posterior(keep=_plot.PLOT_VARS, thin=1, path=trace)
    data = FLAGSHIP.load_data(idata)[0]
    split = _plot.chain_split(trace, int(idata.posterior.sizes["chain"]))
    if split is not None:
        idata = idata.sel(chain=split["majority"])
        print(f"majority chains {split['majority']} / {split['n_chains']}")
    view = prepare_fit(idata, data)
    ref = trace.parent / "mirt_loadings.csv"
    if ref.exists():
        view = align_to_reference_loadings(view, data, ref)
    titles = load_axis_titles(trace.parent, view, data)
    raw = pd.read_csv(PROCESSED_FILE)
    bench = data.blookup.sort_values("benchmark_idx")["benchmark"].tolist()

    figs = {k: f for k, f in forecast_figures(view, data, raw, view.names, view.theta,
                                              axis_titles=titles).items()
            if not k.endswith("_when")}
    ldf = loadings_table(view.require_A(), bench, hdi=(2.5, 97.5))
    figs["loadings_per_axis"] = loadings_grid_fig(ldf, titles, top_n=20)

    OUT.mkdir(parents=True, exist_ok=True)
    for key, fig in figs.items():
        path = OUT / f"{figure_filename(key)}.png"
        fig.write_image(path, scale=2)
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
