"""Per-draw alignment comparison for a fitted MIRT trace — no re-sampling.

Runs analysis.alignment_report (varimax / wop / matchalign / promax — four
independent post-hoc identifications of the SAME posterior) on an existing
trace and writes the comparison artefacts. This is the only writer of those
files; `3_3_diagnostics/3_plot_mirt.py` reads them back.

The fit's flag set and its data scope come from `analysis.FitSpec.from_trace`,
so K and the scope are the trace's own, never a CLI guess.

Outputs (beside the trace, in its own folder):
  mirt_alignment_loadings.csv   — per (method, axis, benchmark): median, HDI,
                                  sign_confident (HDI excludes 0).
  mirt_alignment_agreement.csv  — per method-pair × axis: |corr| of mean
                                  aligned loadings (columns matched first).
  mirt_alignment_summary.json   — per method: aligned max r-hat, per-axis
                                  chain reproducibility, sign-confidence
                                  counts; plus per-chain divergence fractions.
  plots/mirt/diag_k{K}_signed_alignment.png — loading medians ± HDI per axis
                                  (sign-confident highlighted) + agreement map.

Run:
  ~/miniforge3/envs/pymc_env/bin/python 3_diagnostics/align_mirt.py \
      --trace results/mirt_signed_humanprior_lineageprior/trace_mirt_k3_signed_humanprior_lineageprior.nc
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from multiaxis_eci.analysis import FitSpec, alignment_report  # noqa: E402
from multiaxis_eci.persistence import save_df, save_json  # noqa: E402


def _diag_figure(rep: dict, out_path: Path):
    """One figure: aligned loading medians ± HDI per axis for the varimax
    method (sign-confident loadings solid, straddling-zero ones faded), plus
    the method-agreement matrix. Matches the diag_k*_ ad-hoc figure family."""
    ref_method = "varimax" if "varimax" in rep["methods"] else next(iter(rep["methods"]))
    df = rep["methods"][ref_method]["loadings"]
    axes_names = sorted(df["axis"].unique())
    fig, axs = plt.subplots(1, len(axes_names) + 1,
                            figsize=(4.6 * (len(axes_names) + 1), 6.5))
    for i, ax_name in enumerate(axes_names):
        ax = axs[i]
        sub = (df[df["axis"] == ax_name]
               .reindex(df[df["axis"] == ax_name]["loading_median"].abs()
                        .sort_values(ascending=True).index)
               .tail(20))                                   # 20 largest |median|
        y = np.arange(len(sub))
        conf = sub["sign_confident"].values
        colors = np.where(sub["loading_median"] >= 0, "#2980b9", "#c0392b")
        ax.barh(y, sub["loading_median"], xerr=[
            sub["loading_median"] - sub["hdi_low"],
            sub["hdi_high"] - sub["loading_median"]],
            color=[c if k else "#cccccc" for c, k in zip(colors, conf)],
            error_kw={"elinewidth": 0.8, "alpha": 0.6})
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["benchmark"], fontsize=6)
        n_conf = int(conf.sum())
        ax.set_title(f"{ax_name} ({ref_method})\n"
                     f"{n_conf}/{len(sub)} shown sign-confident", fontsize=9)
    ax = axs[-1]
    ag = rep["agreement"].pivot_table(index=["method_a", "method_b"],
                                      columns="axis", values="abs_corr")
    im = ax.imshow(ag.values, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(ag.shape[1])); ax.set_xticklabels(ag.columns, fontsize=8)
    ax.set_yticks(range(ag.shape[0]))
    ax.set_yticklabels([f"{a}~{b}" for a, b in ag.index], fontsize=8)
    for r in range(ag.shape[0]):
        for c in range(ag.shape[1]):
            ax.text(c, r, f"{ag.values[r, c]:.2f}", ha="center", va="center",
                    fontsize=8)
    ax.set_title("method agreement\n(|corr| of mean aligned loadings)", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True,
                    help="path to the fitted trace. Outputs land in the fit's "
                         "results dir. Alignment assumes the sampled orbit was free — on a "
                         "constrained trace (normal, non-negative loadings) it is "
                         "exploratory only.")
    ap.add_argument("--methods", nargs="+",
                    default=["varimax", "wop", "matchalign", "promax"],
                    choices=["varimax", "wop", "matchalign", "promax", "geomin"])
    ap.add_argument("--drop-chains", type=int, nargs="*", default=None,
                    help="0-indexed chains to EXCLUDE (e.g. for a mode-restricted "
                         "readout). The summary records the subset honestly.")
    args = ap.parse_args()

    trace_path = Path(args.trace).resolve()
    print(f"Loading {trace_path} ...", flush=True)
    idata = az.from_netcdf(trace_path)
    spec = FitSpec.from_trace(idata, trace_path)
    # Outputs land beside the trace, not under `spec.results_dir`: a trace
    # fitted under an earlier tag grammar sits in a folder the spec no longer
    # derives, and plot_mirt reads these files back from the trace's own folder.
    K, results_dir = spec.K, trace_path.parent
    if args.drop_chains:
        keep = [c for c in range(idata.posterior.sizes["chain"])
                if c not in set(args.drop_chains)]
        idata = idata.isel(chain=keep)
        print(f"  dropped chains {args.drop_chains}; using {keep}", flush=True)

    data = spec.load_data(idata)[0]

    rep = alignment_report(idata, data, methods=tuple(args.methods))
    all_load = pd.concat(
        [e["loadings"].assign(method=m) for m, e in rep["methods"].items()],
        ignore_index=True)
    # K-tagged filenames: K=2 and K=3 fits share a results dir.
    save_df(all_load, results_dir / f"mirt_alignment_loadings_k{K}.csv")
    save_df(rep["agreement"], results_dir / f"mirt_alignment_agreement_k{K}.csv")
    summary = {m: {"aligned_max_rhat_A": e["aligned_max_rhat_A"],
                   "reproducibility": e["reproducibility"],
                   "sign_counts": e["sign_counts"],
                   "meta": e["meta"]}
               for m, e in rep["methods"].items()}
    summary["per_chain_divergence_frac"] = rep.get("per_chain_divergence_frac")
    summary["dropped_chains"] = args.drop_chains or []
    save_json(summary, results_dir / f"mirt_alignment_summary_k{K}.json")

    for m, e in rep["methods"].items():
        sc = "  ".join(f"{ax} +{v['n_pos_confident']}/-{v['n_neg_confident']}"
                       for ax, v in e["sign_counts"].items())
        rp = ("/".join(f"{r:.2f}" for r in e["reproducibility"])
              if e["reproducibility"] else "n/a")
        print(f"  {m:10s} sign-confident [{sc}]  chain-reprod {rp}  "
              f"aligned-rhat {e['aligned_max_rhat_A']:.3f}")
    print(rep["agreement"].groupby("axis")["abs_corr"].agg(["min", "median"])
          .to_string(float_format=lambda x: f"{x:.3f}"))

    fig_path = ROOT / "plots" / "mirt" / f"diag_k{K}{spec.tag}_alignment.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    _diag_figure(rep, fig_path)
    print(f"\nOutputs → {results_dir}\nFigure  → {fig_path}")


if __name__ == "__main__":
    main()
