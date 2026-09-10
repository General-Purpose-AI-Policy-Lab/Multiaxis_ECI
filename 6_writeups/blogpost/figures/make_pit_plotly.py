"""PIT histogram of the flagship fit, one wide panel, in the post's Plotly style.

The calibration view: u_n = P(Y_rep <= y_n) under the fitted posterior
predictive. A calibrated fit puts the PIT uniform on [0, 1], so the histogram
sits flat on the dotted density-1 line and the PIT variance sits at the uniform
1/12 ~ 0.083. Variance BELOW 1/12 means the predictive intervals are wider than
the data needs.

Statistics are the production PPC, so the numbers match the fit's own GoF:
`FitSpec.load_data` runs the same loads, floor read and floor clip `fit.py`
runs before sampling (and checks the trace's model/bench dims against today's
data), `ppc.posterior_predictive_mirt` draws the floor-aware predictive, and
`ppc.pit_values` scores it with the boundary rows excluded — PIT is degenerate
at an exact 0 or 1.

All ten chains are pooled: PIT is a whole-fit diagnostic, and the majority /
minority split is an ability-side statement that no calibration number is read
through.

Usage:
    python 6_writeups/blogpost/figures/make_pit_plotly.py [--trace FILE] [--tag ""]
                                              [--max-draws 2000]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "2_model"))

from multiaxis_eci.analysis import FLAGSHIP, FLAGSHIP_THIN  # noqa: E402
from multiaxis_eci.analysis import FLAGSHIP_TRACE as TRACE
from multiaxis_eci.ppc import boundary_mask, pit_values, posterior_predictive_mirt  # noqa: E402
from multiaxis_eci.viz import POST, pit_hist_fig  # noqa: E402
from multiaxis_eci.viz.core import save_html, save_print  # noqa: E402

# None drops the in-figure title: the post's caption carries the
# description. Set a string to draw it on the canvas again.
TITLE = None

BINS = 20             # bin edges on [0, 1]
MAX_DRAWS = 2000      # posterior draws entering the predictive


def main(trace: Path = TRACE, tag: str = "", out_dir: Path = HERE,
         max_draws: int = MAX_DRAWS) -> None:
    idata = FLAGSHIP.open_posterior(keep=["A", "theta", "D", "phi_b"],
                                    thin=FLAGSHIP_THIN, chains=None, path=trace)
    data, floor_c, n_eff = FLAGSHIP.load_data(idata)
    print(f"  data scope: {data.n_obs} obs, {data.n_models} test-takers, "
          f"{data.n_benchmarks} benchmarks")
    y_rep = posterior_predictive_mirt(idata, data, floor_c=floor_c,
                                      n_eff=n_eff, max_draws=max_draws)
    pit = pit_values(y_rep, data.scores, boundary_mask(data))
    print(f"  n {pit.size} (of {data.n_obs} obs; "
          f"{int(boundary_mask(data).sum())} boundary rows excluded)")
    print(f"  PIT mean {pit.mean():.4f}, variance {pit.var():.4f} "
          f"(uniform 0.5 / {1 / 12:.4f})")

    # The post's panel has no H0 band: the caption carries the calibration
    # reading, and the n / variance note under the x title gives the numbers.
    fig = pit_hist_fig(pit, BINS, style=POST, title=TITLE, h0_band=False)

    out = out_dir / f"pit_plotly{tag}"
    save_html(fig, out)
    save_print(fig, out)
    print(f"  wrote {out.with_suffix('.png')} and html/{out.stem}.html")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--trace", type=Path, default=TRACE)
    p.add_argument("--tag", default="")
    p.add_argument("--max-draws", type=int, default=MAX_DRAWS)
    args = p.parse_args()
    main(args.trace, args.tag, max_draws=args.max_draws)
