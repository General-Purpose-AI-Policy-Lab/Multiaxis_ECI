"""LOO-CV comparison across the story fits — with data-comparability guards.

One elpd table cannot rank all five campaign fits, because they are not all
fit to the same observations:
  * bare / +human / +both share the identical 4051-row dataset -> a standard
    az.compare is valid there;
  * the noSG fit drops the Skilled Generalist's rows (different n);
  * the floors fit clips 36 reviewed below-floor scores up to the benchmark
    chance floor (36 observed VALUES differ).
For the last two, the honest comparison is a paired pointwise ΔELPD restricted
to the rows both fits saw identically ("shared rows"): noSG vs both on the
non-SG rows, floors vs both on the unclipped rows, floors vs noSG on the
intersection. Same observable, same dominating measure (the 3PL floors the
mean, not the support), so pointwise log densities compare directly.

Row matching is positional: log_likelihood carries a bare obs dim, and the
data.py transforms (drop_model_observations, clip_scores_to_floors) preserve
row order, so masks built on the full-data row order index every trace. The
script ASSERTS each trace's observed_data against the expected score vector
before any arithmetic — silent misalignment is impossible.

PSIS-LOO is strained on this data (~5.4 obs/model -> many high Pareto-k), so
each pairwise delta also reports a sensitivity recomputed on the rows where
both sides have k < 0.7, and the summary table carries the k composition next
to the PPC-based GoF. Read them together.

Outputs (results/comparisons/):
  loo_story_compare.csv   — az.compare on the identical-data fits.
  loo_story_pairwise.csv  — shared-row paired ΔELPD ± SE per pair.
  loo_story_pareto_k.csv  — per-fit elpd + Pareto-k composition + GoF.
  plots/comparisons/loo_story.{pdf,png} — ΔELPD ladder vs 'both' + k stack.

Run (after the campaign fits):
  ~/miniforge3/envs/pymc_env/bin/python diagnostics/loo_compare.py
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import warnings
from pathlib import Path

import arviz as az
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from config import ECI_EPS  # noqa: E402
from data import (clip_scores_to_floors, load_benchmark_floors,  # noqa: E402
                  load_eci_data)

# story registry: fit -> (trace path, data variant). Order = the story.
FITS = {
    "bare":   ("results/mirt_signed/trace_mirt_k3_signed.nc", "base"),
    "human":  ("results/mirt_signed_humanprior/"
               "trace_mirt_k3_signed_humanprior.nc", "base"),
    "both":   ("results/mirt_signed_humanprior_lineageprior/"
               "trace_mirt_k3_signed_humanprior_lineageprior.nc", "base"),
    "nosg":   ("results/mirt_signed_humanprior_lineageprior_noSG/"
               "trace_mirt_k3_signed_humanprior_lineageprior_noSG.nc", "nosg"),
    "floors": ("results/mirt_signed_humanprior_lineageprior_floors/"
               "trace_mirt_k3_signed_humanprior_lineageprior_floors.nc",
               "floors"),
    # K-justification: K=4 on the SAME clipped data as the K=3 floors fit,
    # so the {floors, floors_k4} pair is a fully valid az.compare
    "floors_k4": ("results/mirt_signed_humanprior_lineageprior_floors/"
                  "trace_mirt_k4_signed_humanprior_lineageprior_floors.nc",
                  "floors"),
    # Honest-noise pair (2026-08-08): K=3 vs K=4 on IDENTICAL data — same
    # folder, same drop-3 scope, same known-SE + pooled-noise flags — so the
    # pair is a fully valid az.compare. This is the K-justification on the
    # current data state; the drop-2 flagship has no same-data K=3 twin.
    "k3_pooled3": ("results/mirt_humanprior_lineageprior_lineagebm_"
                   "dropFrontierMathv1GBAEvalAlgoTune_floors_knownse_poolednoise/"
                   "trace_mirt_k3_humanprior_lineageprior_lineagebm_"
                   "dropFrontierMathv1GBAEvalAlgoTune_floors_knownse_poolednoise.nc",
                   "pooled3"),
    "k4_pooled3": ("results/mirt_humanprior_lineageprior_lineagebm_"
                   "dropFrontierMathv1GBAEvalAlgoTune_floors_knownse_poolednoise/"
                   "trace_mirt_k4_humanprior_lineageprior_lineagebm_"
                   "dropFrontierMathv1GBAEvalAlgoTune_floors_knownse_poolednoise.nc",
                   "pooled3"),
}
REFERENCE = "both"          # every ladder rung is measured against this fit


def _loo_pointwise(idata):
    """az.loo with pointwise densities; tolerant to arviz attr renames."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        loo = az.loo(idata, pointwise=True)
    loo_i = getattr(loo, "loo_i", None)
    if loo_i is None:
        loo_i = loo.elpd_i if hasattr(loo, "elpd_i") else loo["loo_i"]
    return loo, np.asarray(loo_i.values).ravel(), \
        np.asarray(loo.pareto_k.values).ravel()


def _expected_obs(data, variant, keep_sg, clipped_scores, extra=None):
    scores = {"base": data.scores,
              "nosg": data.scores[keep_sg],
              "floors": clipped_scores, **(extra or {})}[variant]
    return np.clip(scores, ECI_EPS, 1.0 - ECI_EPS)


def _gof_fields(trace_path: Path) -> dict:
    k = "k4" if "_k4_" in trace_path.name else "k3"
    p = trace_path.parent / f"mirt_gof_{k}.json"
    if not p.exists():
        return {}
    g = json.loads(p.read_text())
    return {"R2": round(g.get("bayesian_r2", float("nan")), 4),
            "RMSE": round(g.get("rmse", float("nan")), 4),
            "MAE": round(g.get("mae", float("nan")), 4),
            "PIT_var": round(g.get("pit_var", float("nan")), 4)}


def _pair_delta(name_a, li_a, k_a, name_b, li_b, k_b, n_label):
    """Paired pointwise ΔELPD (a - b) on already-aligned shared rows."""
    diff = li_a - li_b
    delta, se = float(diff.sum()), float(np.sqrt(diff.size * np.var(diff)))
    ok = (k_a < 0.7) & (k_b < 0.7)                       # Pareto-k sensitivity
    d_ok = diff[ok]
    return {"pair": f"{name_a} - {name_b}", "rows": n_label,
            "n_shared": int(diff.size), "delta_elpd": round(delta, 2),
            "se": round(se, 2), "z": round(delta / se, 2) if se else np.nan,
            "n_k_bad_either": int((~ok).sum()),
            "delta_elpd_k_ok_rows": round(float(d_ok.sum()), 2),
            "n_k_ok_rows": int(ok.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fits", nargs="+", default=list(FITS),
                    choices=list(FITS), help="subset to compare")
    ap.add_argument("--use-verdicts", action="store_true",
                    help="also report each fit on its majority-chain subset "
                         "(drop_chains from chain_verdicts.csv) in the "
                         "summary table")
    args = ap.parse_args()

    data = load_eci_data(include_all_benchmarks=True)
    sg_id0 = int(data.mlookup.loc[data.mlookup["model"] == "Skilled Generalist",
                                  "model_idx"].iloc[0]) - 1
    keep_sg = data.model_idx != sg_id0                    # (n_obs,) bool
    clipped_data = clip_scores_to_floors(data, load_benchmark_floors(data))
    clipped_mask = clipped_data.scores != data.scores     # 36 reviewed rows
    print(f"data: {data.n_obs} obs | SG rows {int((~keep_sg).sum())} | "
          f"floor-clipped rows {int(clipped_mask.sum())}")

    verdicts = {}
    vpath = ROOT / "results/comparisons/chain_verdicts.csv"
    if args.use_verdicts and vpath.exists():
        for _, r in pd.read_csv(vpath).iterrows():
            raw = r.get("recommended_drop_chains", "")
            drops = "" if pd.isna(raw) else str(raw).strip()
            if drops:
                verdicts[r["fit"]] = [int(c) for c in drops.split()]

    # extra score vectors for variants whose fits load a REDUCED scope: built
    # exactly the way the fit built them (drop first, then clip), so the
    # observed_data assert stays byte-exact.
    extra = {}
    if any(FITS[n][1] == "pooled3" for n in args.fits if n in FITS):
        d3 = load_eci_data(include_all_benchmarks=True,
                           drop_benchmarks=["FrontierMath v1", "GBAEval", "AlgoTune"])
        d3 = clip_scores_to_floors(d3, load_benchmark_floors(d3))
        extra["pooled3"] = d3.scores
        print(f"pooled3 scope: {d3.n_obs} obs")

    # one trace in memory at a time: keep only the small loo artefacts
    loos, li, kk, summary_rows, hist_obs = {}, {}, {}, [], {}
    for name in args.fits:
        path, variant = FITS[name]
        tp = ROOT / path
        if not tp.exists():
            print(f"  [skip] {name}: {tp} not found")
            continue
        print(f"  loading {name} ({variant}) ...", flush=True)
        idata = az.from_netcdf(tp)
        expected = _expected_obs(data, variant, keep_sg, clipped_data.scores, extra)
        got = np.asarray(idata.observed_data["obs"].values).ravel()
        if got.size != expected.size:
            # Historical trace: fit on a data state the current file has since
            # replaced. Its per-fit LOO is still exact, and a within-group
            # az.compare stays valid iff every member carries the IDENTICAL
            # observed vector — enforced below at compare time. Cross-variant
            # pairwise deltas are impossible (no row keys survive), so the
            # fit is excluded from section (b) by never being 'both'-aligned.
            print(f"    {name}: HISTORICAL data state ({got.size} obs vs "
                  f"current {expected.size}) — within-group compare only")
            hist_obs[name] = got
        else:
            assert np.allclose(got, expected, atol=1e-9), \
                f"{name}: observed_data does not match the {variant} score vector"
        loo, loo_i, k = _loo_pointwise(idata)
        loos[name], li[name], kk[name] = loo, loo_i, k
        summary_rows.append({
            "fit": name, "variant": variant, "chains": "all",
            "n_obs": int(loo_i.size), "elpd_loo": round(float(loo.elpd_loo), 2),
            "se": round(float(loo.se), 2), "p_loo": round(float(loo.p_loo), 1),
            "k_good": int((k < 0.5).sum()),
            "k_ok": int(((k >= 0.5) & (k < 0.7)).sum()),
            "k_bad": int(((k >= 0.7) & (k < 1.0)).sum()),
            "k_very_bad": int((k >= 1.0).sum()),
            "k_max": round(float(k.max()), 2), **_gof_fields(tp)})
        if name in verdicts:
            keep = [c for c in range(idata.posterior.sizes["chain"])
                    if c not in verdicts[name]]
            loo_m, loo_i_m, k_m = _loo_pointwise(idata.sel(chain=keep))
            summary_rows.append({
                "fit": name, "variant": variant,
                "chains": f"majority (drop {verdicts[name]})",
                "n_obs": int(loo_i_m.size),
                "elpd_loo": round(float(loo_m.elpd_loo), 2),
                "se": round(float(loo_m.se), 2),
                "p_loo": round(float(loo_m.p_loo), 1),
                "k_good": int((k_m < 0.5).sum()),
                "k_ok": int(((k_m >= 0.5) & (k_m < 0.7)).sum()),
                "k_bad": int(((k_m >= 0.7) & (k_m < 1.0)).sum()),
                "k_very_bad": int((k_m >= 1.0).sum()),
                "k_max": round(float(k_m.max()), 2), **_gof_fields(tp)})
        del idata
        gc.collect()

    out_dir = ROOT / "results/comparisons"
    out_dir.mkdir(parents=True, exist_ok=True)

    # (a) valid az.compare: identical-data groups only
    for group, label, csv_name in [
            (("bare", "human", "both"), "identical raw data",
             "loo_story_compare.csv"),
            (("floors", "floors_k4"), "identical clipped data (K choice)",
             "loo_story_compare_K.csv"),
            (("k3_pooled3", "k4_pooled3"),
             "identical pooled drop-3 data (K choice, honest noise)",
             "loo_story_compare_K_pooled.csv")]:
        grp = {n: loos[n] for n in group if n in loos}
        if len(grp) < 2:
            continue
        hist = [n for n in grp if n in hist_obs]
        if hist:
            # all-or-nothing, and byte-identical observed vectors required
            if len(hist) != len(grp) or any(
                    hist_obs[n].size != hist_obs[hist[0]].size
                    or not np.allclose(hist_obs[n], hist_obs[hist[0]], atol=1e-9)
                    for n in hist):
                print(f"  [skip] {csv_name}: group mixes data states")
                continue
            label += f" — HISTORICAL state, n={hist_obs[hist[0]].size}"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cmp_df = az.compare(grp, ic="loo", method="stacking")
        cmp_df.to_csv(out_dir / csv_name)
        print(f"\n── az.compare ({label}: {'/'.join(grp)}) ──")
        print(cmp_df[["rank", "elpd_loo", "elpd_diff", "dse", "weight"]]
              .to_string(float_format=lambda x: f"{x:.2f}"))

    # (b) shared-row paired deltas — the only honest cross-variant comparison
    pairs = []
    if "both" in li:
        if "bare" in li:
            pairs.append(_pair_delta("bare", li["bare"], kk["bare"],
                                     "both", li["both"], kk["both"],
                                     "all 4051 (identical data)"))
        if "human" in li:
            pairs.append(_pair_delta("human", li["human"], kk["human"],
                                     "both", li["both"], kk["both"],
                                     "all 4051 (identical data)"))
        if "nosg" in li:
            pairs.append(_pair_delta(
                "nosg", li["nosg"], kk["nosg"],
                "both", li["both"][keep_sg], kk["both"][keep_sg],
                "non-SG rows"))
        if "floors" in li:
            m = ~clipped_mask
            pairs.append(_pair_delta(
                "floors", li["floors"][m], kk["floors"][m],
                "both", li["both"][m], kk["both"][m], "unclipped rows"))
    if "floors" in li and "nosg" in li:
        m_full = keep_sg & ~clipped_mask                  # full-data space
        m_nosg = (~clipped_mask)[keep_sg]                 # nosg row space
        pairs.append(_pair_delta(
            "floors", li["floors"][m_full], kk["floors"][m_full],
            "nosg", li["nosg"][m_nosg], kk["nosg"][m_nosg],
            "non-SG ∩ unclipped rows"))
    if pairs:
        pw = pd.DataFrame(pairs)
        pw.to_csv(out_dir / "loo_story_pairwise.csv", index=False)
        print("\n── shared-row paired ΔELPD ──")
        print(pw.to_string(index=False))

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "loo_story_pareto_k.csv", index=False)
    print("\n── per-fit summary (elpd + Pareto-k + GoF) ──")
    print(summary.to_string(index=False))

    # figure: ΔELPD ladder vs REFERENCE + Pareto-k composition
    if pairs and len(li) >= 2:
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
        rungs = [p for p in pairs if p["pair"].endswith(f"- {REFERENCE}")]
        y = np.arange(len(rungs))
        for i, p in enumerate(rungs):
            shared = "identical" not in p["rows"]
            ax[0].barh(i, p["delta_elpd"], xerr=p["se"],
                       color="#2980b9", alpha=0.55 if shared else 0.95,
                       hatch="//" if shared else None,
                       error_kw={"elinewidth": 1.2})
            ax[0].annotate(f" n={p['n_shared']}", (p["delta_elpd"], i),
                           fontsize=7, va="center")
        ax[0].axvline(0, color="black", lw=0.8)
        ax[0].set_yticks(y)
        ax[0].set_yticklabels([p["pair"].split(" - ")[0] for p in rungs])
        ax[0].set(title=f"paired ΔELPD vs '{REFERENCE}' "
                        "(hatched = shared-row subset, not identical data)",
                  xlabel="Δ elpd (± SE)")
        order = [n for n in FITS if n in li]
        bottom = np.zeros(len(order))
        for band, col in [("k_good", "#27ae60"), ("k_ok", "#f1c40f"),
                          ("k_bad", "#e67e22"), ("k_very_bad", "#c0392b")]:
            vals = np.array([summary[(summary["fit"] == n)
                                     & (summary["chains"] == "all")][band].iloc[0]
                             for n in order], float)
            ax[1].bar(order, vals, bottom=bottom, color=col,
                      label=band.replace("k_", "k "))
            bottom += vals
        ax[1].legend(fontsize=8)
        ax[1].set(title="Pareto-k composition per fit (~5.4 obs/model "
                        "strains PSIS-LOO)", ylabel="observations")
        fig.tight_layout()
        fig_dir = ROOT / "plots" / "comparisons"
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(fig_dir / "loo_story.pdf")
        fig.savefig(fig_dir / "loo_story.png", dpi=160)
        plt.close(fig)
        print(f"\nfigure → {fig_dir / 'loo_story.pdf'} (+ .png)")


if __name__ == "__main__":
    main()
