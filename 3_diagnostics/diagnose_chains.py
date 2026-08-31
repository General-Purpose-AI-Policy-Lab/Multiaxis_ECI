"""Chain-island diagnosis for a fitted MIRT trace — no re-sampling.

Answers one question per trace: are the chains one converged solution, or do
some sit in a separate likelihood basin ("island")? Ports the ad-hoc cells
from floors_k3_3pl_report.ipynb (per-chain logp majority, mean-A
permutation/sign matching) and recovery_study.ipynb (D-spread, Ward basins,
pre-registered verdict thresholds) into one reusable script, so every fresh
campaign trace gets the same traceable judgment.

Three independent signals, then a verdict:
  1. per-chain mean logp — a chain more than BASIN_NATS below the best chain
     is in a worse basin (logp is rotation-invariant, so this cannot be
     label-switching);
  2. cross-chain D-spread + 2-cluster Ward on z-scored chain-mean D —
     difficulty is also rotation-invariant, so spread means different fits;
  3. per-chain mean-A permutation/sign match to the best-logp chain — chains
     whose matched loadings still disagree found a different solution, not a
     relabeled copy of the same one (this is the signal that separates benign
     label-switching from a genuine island).

Verdict thresholds are the recovery study's pre-registered ones. The
recommended drop_chains is the union of the three islands signals, endorsed
only when the remaining majority is internally converged (eta r-hat <=
conv threshold) and the drop is a minority of chains.

`--write-modes` is the other entry point: instead of a verdict it writes the
chain→mode split to `results/<fit>/mirt_modes_<trace-stem>.json`, which is what
the dashboard reads to render one loading/timeline figure set per mode. Detection
happens ONCE, here, so a dashboard build never loads a multi-GB trace to
re-detect it. That path needs no data scope, so it also works on a trace whose
data generation has since moved.

Outputs:
  results/comparisons/chain_verdicts.csv  — one row per --name (idempotent:
                                            re-running replaces the row).
  plots/mirt/chains_<name>.pdf            — with --fig: Δlogp bars, top
                                            D-spread, chain-distance map.
  results/<fit>/mirt_modes_<stem>.json    — with --write-modes: chains, Δlogp
                                            and matched loading corr per mode.

Run:
  python 3_diagnostics/diagnose_chains.py \
      --trace results/mirt/trace_mirt_k2.nc --name k2_demo --fig
  python 3_diagnostics/diagnose_chains.py \
      --trace results/mirt/trace_mirt_k2.nc --write-modes
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
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist, squareform

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from multiaxis_eci.analysis import mirt_identified_rhat  # noqa: E402
from multiaxis_eci.data import load_eci_data  # noqa: E402

# a chain within this many nats of the best chain-mean logp is the same basin
BASIN_NATS = 20.0

# pre-registered verdict thresholds (recovery_study.ipynb; calibrated on the
# campaign history: converged fits had D-spread 0.19-0.47 / logp spread < 10,
# island fits had D-spread >= 1.7 / logp spread 26-173)
THRESHOLDS = {
    "island_dspread": 1.0, "island_logp": 20.0, "island_eta_rhat": 1.1,
    "conv_rhat": 1.05, "conv_dspread": 0.5, "conv_logp": 10.0,
}

# min matched-loading correlation for "same solution as the reference chain".
# Same-solution chains reach ~0.7 even on a weak axis; a genuine island sits
# well below (0.49 in the floors report) — 0.6 splits the two clusters.
MATCH_THRESH = 0.6


def _load_matching_data(idata):
    """Return the ECIData whose model AND benchmark coords match the trace
    (exploration scope first, canonical excluded scope as fallback). The
    trace's own benchmark-drop list and --cyber flag are replayed from its
    attrs. Matching on coordinate NAMES, not counts: benchmark indices come
    from sorted() over the scope, so a scope that merely has the right count
    would still misalign A/D rows against the obs indices."""
    post = idata.posterior
    trace_models = list(post["model"].values)
    trace_benches = list(post["bench"].values)
    drop = json.loads(post.attrs.get("mirt_drop_benchmarks", "null"))
    cyber = json.loads(post.attrs.get("mirt_cyber", "false"))
    sqa = json.loads(post.attrs.get("mirt_simpleqa_original", "false"))
    for include_all in (True, False):
        try:
            data = load_eci_data(include_all_benchmarks=include_all,
                                 drop_benchmarks=drop, fit_cyber=cyber,
                                 fit_simpleqa_original=sqa)
        except ValueError:
            continue        # a dropped name the curated list already removed
        if (list(data.mlookup.sort_values("model_idx")["model"]) == trace_models
                and list(data.blookup.sort_values("benchmark_idx")["benchmark"])
                == trace_benches):
            return data
    raise RuntimeError(
        f"trace coords ({len(trace_models)} models / {len(trace_benches)} "
        f"benchmarks, cyber={cyber}) match neither the exploration nor the "
        f"canonical data scope — the data under the fit has changed")


def _chain_logp(idata):
    """Per-chain mean log-probability, or None when it is not usable.

    Not usable means either no logp/lp variable at all, or a variable whose
    per-chain mean is all-NaN — an empty sample_stats group (some K=1 traces on
    disk carry zero draws there). Returning None instead of NaN keeps every
    caller on ONE "logp unavailable" path: NaN would silently disable the basin
    comparison, sort modes in arbitrary order, print "nan nats", and make
    json.dumps emit bare NaN, which is not valid JSON."""
    for name in ("logp", "lp"):
        if name in idata.sample_stats:
            v = idata.sample_stats[name].mean("draw").values
            return None if np.all(np.isnan(v)) else v
    return None


def _match_corr(ref, other):
    """Min over axes of the greedily permutation/sign-matched |corr| between two
    chains' mean loading matrices, each (bench, K) — the 'same solution?' score.
    1.0 for K=1, where there is nothing to match."""
    K = ref.shape[1]
    if K == 1:
        return 1.0
    used, corrs = set(), []
    for ja in range(K):
        best = (-1.0, None)
        for jb in range(K):
            if jb in used:
                continue
            r = abs(np.corrcoef(ref[:, ja], other[:, jb])[0, 1])
            if r > best[0]:
                best = (r, jb)
        used.add(best[1])
        corrs.append(best[0])
    return min(corrs)


def _match_to_ref(post, ref_chain: int):
    """Per-chain min matched |corr| against the reference chain's mean loadings."""
    A = post["A"].mean("draw").values                     # (chain, bench, K)
    return np.array([_match_corr(A[ref_chain], A[c]) for c in range(A.shape[0])])


def detect_modes(post, logp, match_thresh: float = MATCH_THRESH) -> list:
    """Group the chains into posterior modes. Two chains share a mode when their
    mean loadings match (min matched |corr| > match_thresh) AND they sit in the
    same logp basin. Both criteria are rotation-invariant, so a mode is a
    distinct SOLUTION, not a label permutation of one.

    A basin is a run of chain logps with no gap wider than BASIN_NATS between
    consecutive SORTED values — a cliff separates basins, a continuum does not.
    Read on the sorted values rather than pairwise, because pairwise is not
    transitive: one family here spans 22 nats internally at loading corr 1.000,
    and a pairwise rule fragments it. The cost of chaining is the mirror case: a
    LADDER of sub-BASIN_NATS steps reads as one basin no matter how far it
    reaches end to end (verified: 15-nat steps spanning 60 nats → one mode). The
    loading criterion is what catches that, so a ladder collapses into one mode
    only while every rung carries the same loadings. Left as is deliberately —
    any cliff test on a continuum has to put its threshold somewhere.

    Then complete linkage on distance = (1 − matched corr) + 1 per basin
    mismatch, cut at 1 − match_thresh. The +1 exceeds the cut, so a mode never
    spans two basins; complete linkage makes every pair inside a mode pass the
    loading test rather than only chaining through neighbours. Modes come back
    best-logp first, labelled A, B, C…

    With logp unavailable the split is loadings-only; with K=1 it is basins-only
    (`_match_corr` is 1.0 there, no axes to match). `basis_note` reports which."""
    A = post["A"].mean("draw").values                     # (chain, bench, K)
    n = A.shape[0]
    basin = np.zeros(n, int)
    if logp is not None:
        order = np.argsort(logp)[::-1]
        for prev, c in zip(order, order[1:]):
            basin[c] = basin[prev] + (logp[prev] - logp[c] > BASIN_NATS)
    corr = np.ones((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            corr[i, j] = corr[j, i] = _match_corr(A[i], A[j])
    dist = (1.0 - corr) + (basin[:, None] != basin[None, :])
    lab = (fcluster(linkage(squareform(dist), "complete"),
                    t=1.0 - match_thresh, criterion="distance")
           if n > 1 else np.ones(1, int))
    modes = []
    for g in np.unique(lab):
        ch = [int(c) for c in np.where(lab == g)[0]]
        modes.append({
            "chains": ch,
            "mean_logp": float(logp[ch].mean()) if logp is not None else None,
            # worst pair inside the mode — how tightly its own chains agree
            "min_matched_corr": (round(float(corr[np.ix_(ch, ch)].min()), 3)
                                 if len(ch) > 1 else 1.0)})
    modes.sort(key=lambda m: -(m["mean_logp"] if m["mean_logp"] is not None else 0.0))
    best = modes[0]
    for label, m in zip("ABCDEFGHIJKL", modes):
        m["label"] = label
        m["delta_logp"] = (None if best["mean_logp"] is None
                           else round(m["mean_logp"] - best["mean_logp"], 1))
        # Worst matched |corr| against the best mode: this is what separates
        # "same loadings, worse basin" from "a different axis solution". None on
        # the best mode itself, which has nothing to compare against.
        m["matched_corr_to_best"] = (
            None if m is best
            else round(float(corr[np.ix_(m["chains"], best["chains"])].min()), 3))
    return modes


def modes_path(trace_path) -> Path:
    """Where a trace's mode split lives. Keyed by the TRACE stem, not the folder:
    a results folder holds the K=3 fit AND its K=1 baseline, and one shared
    filename would let a run on either overwrite the other's split."""
    p = Path(trace_path)
    return p.parent / f"mirt_modes_{p.stem}.json"


def write_modes(idata, trace_path, match_thresh: float) -> tuple:
    """Persist the mode split next to the trace, so the dashboard never has to
    load a multi-GB trace to learn which chains belong together. The `basis`
    string states which criteria were actually EVALUATED — a K=1 fit has no
    loading match and a trace without usable logp has no basin test, and the
    note must not advertise either one it never ran."""
    post = idata.posterior
    logp = _chain_logp(idata)
    K = int(post.sizes["latent"])
    modes = detect_modes(post, logp, match_thresh)
    basis = []
    basis.append(f"mean-loading match (min permutation/sign-matched |corr| > "
                 f"{match_thresh})" if K > 1 else
                 "K=1: one axis, so no loading criterion applies")
    basis.append(f"logp basins (no gap wider than {BASIN_NATS:g} nats between "
                 f"sorted chain logps); Δlogp = mode mean chain logp minus the "
                 f"best mode's" if logp is not None else
                 "logp unavailable in sample_stats, so no basin test was applied "
                 "and Δlogp is null")
    out = {
        "trace": Path(trace_path).name,
        "n_chains": int(post.sizes["chain"]),
        "K": K,
        "basis": " · ".join(basis),
        "chain_delta_logp": (None if logp is None
                             else [round(float(v - logp.max()), 1) for v in logp]),
        "modes": modes,
    }
    out_path = modes_path(trace_path)
    out_path.write_text(json.dumps(out, indent=1))
    return out, out_path


def _eta_rhat_subset(idata, data, chains):
    """Identified eta max r-hat on a chain subset (NaN when < 2 chains)."""
    if len(chains) < 2:
        return float("nan")
    return mirt_identified_rhat(idata.sel(chain=list(chains)), data)["eta_max_rhat"]


def _verdict(eta_rhat, d_rhat, dspread_max, logp_spread):
    t = THRESHOLDS
    if (eta_rhat <= t["conv_rhat"] and d_rhat <= t["conv_rhat"]
            and dspread_max < t["conv_dspread"] and logp_spread < t["conv_logp"]):
        return "CONVERGED"
    if (dspread_max > t["island_dspread"] or logp_spread > t["island_logp"]
            or eta_rhat > t["island_eta_rhat"]):
        return "ISLANDS"
    return "AMBIGUOUS"


def _figure(name, below, majority_mask, dspread, benches, chain_dist, out_path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    colors = ["tab:blue" if m else "tab:red" for m in majority_mask]
    ax[0].bar(range(len(below)), below, color=colors)
    ax[0].axhline(-BASIN_NATS, ls="--", c="k", lw=1)
    ax[0].set(title="chain mean logp (nats below best) — red = island basin",
              xlabel="chain", ylabel="nats below best")
    order = np.argsort(dspread)[::-1][:8]
    ax[1].barh([str(benches[i])[:26] for i in order][::-1],
               dspread[order][::-1], color="tab:blue")
    ax[1].axvline(THRESHOLDS["island_dspread"], ls="--", c="r", lw=1)
    ax[1].set(title="top cross-chain D-spread (rotation-invariant)")
    im = ax[2].imshow(chain_dist, cmap="viridis")
    ax[2].set(title="pairwise chain distance (z-scored mean D)",
              xlabel="chain", ylabel="chain")
    fig.colorbar(im, ax=ax[2], fraction=0.046)
    fig.suptitle(name)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True, help="path to the trace .nc")
    ap.add_argument("--name",
                    help="fit label for the verdict row and figure filename "
                         "(required unless --write-modes)")
    ap.add_argument("--fig", action="store_true", help="save the 3-panel figure")
    ap.add_argument("--match-thresh", type=float, default=MATCH_THRESH)
    ap.add_argument("--write-modes", action="store_true",
                    help="write mirt_modes.json next to the trace and stop "
                         "(no data load, so a superseded trace still splits)")
    ap.add_argument("--out-csv",
                    default=str(ROOT / "results/comparisons/chain_verdicts.csv"))
    args = ap.parse_args()

    print(f"Loading {args.trace} ...", flush=True)
    idata = az.from_netcdf(args.trace)
    post = idata.posterior
    n_chains = int(post.sizes["chain"])

    if args.write_modes:
        out, out_path = write_modes(idata, args.trace, args.match_thresh)
        print(f"  basis: {out['basis']}")
        for m in out["modes"]:
            print(f"  mode {m['label']}: chains {m['chains']}  "
                  f"Δlogp {m['delta_logp']}  within-mode corr "
                  f"{m['min_matched_corr']}  corr to best "
                  f"{m['matched_corr_to_best']}")
        print(f"  {len(out['modes'])} mode(s) → {out_path}")
        return
    if not args.name:
        ap.error("--name is required unless --write-modes")

    data = _load_matching_data(idata)
    divergences = (int(idata.sample_stats["diverging"].sum())
                   if "diverging" in idata.sample_stats else -1)

    # signal 1: logp basins
    logp = _chain_logp(idata)
    if logp is None:
        print("  WARNING: no per-draw logp in sample_stats — basin check skipped")
        below = np.zeros(n_chains)
        logp_majority = np.ones(n_chains, bool)
        ref_chain = 0
    else:
        below = logp - logp.max()
        logp_majority = below > -BASIN_NATS
        ref_chain = int(np.argmax(logp))
    logp_spread = float(-below.min())

    # signal 2: D-spread + Ward basins (rotation-invariant)
    cmean = post["D"].mean("draw").values                 # (chain, bench)
    dspread = cmean.max(0) - cmean.min(0)
    z = (cmean - cmean.mean(0)) / (cmean.std(0) + 1e-9)
    chain_dist = squareform(pdist(z))
    ward = fcluster(linkage(z, "ward"), t=2, criterion="maxclust")

    # signal 3: same-solution matching against the best-logp chain
    min_corr = _match_to_ref(post, ref_chain)
    match_ok = min_corr > args.match_thresh

    island_mask = ~(logp_majority & match_ok)
    majority = [c for c in range(n_chains) if not island_mask[c]]
    islands = [c for c in range(n_chains) if island_mask[c]]

    # identified r-hat: all chains vs the majority subset
    eta_all = _eta_rhat_subset(idata, data, list(range(n_chains)))
    ident_all = mirt_identified_rhat(idata, data)
    d_rhat_all = ident_all.get("D_max_rhat", float("nan"))
    eta_majority = (_eta_rhat_subset(idata, data, majority)
                    if islands else eta_all)

    verdict = _verdict(eta_all, d_rhat_all, float(dspread.max()), logp_spread)
    drop_ok = (islands and len(islands) < n_chains / 2
               and eta_majority <= THRESHOLDS["conv_rhat"])
    recommended_drop = islands if drop_ok else []

    print(f"\n  chains={n_chains}  divergences={divergences}")
    print(f"  logp spread {logp_spread:.1f} nats | D-spread max "
          f"{dspread.max():.2f} | ward clusters {list(ward)}")
    print("  chain   Δlogp    min-match-corr   basin")
    for c in range(n_chains):
        tag = "ISLAND" if island_mask[c] else "majority"
        print(f"    c{c}   {below[c]:8.1f}      {min_corr[c]:.3f}         {tag}")
    print(f"  eta r-hat: all={eta_all:.3f}  majority({majority})={eta_majority:.3f}"
          f"  | D r-hat all={d_rhat_all:.3f}")
    print(f"  VERDICT: {verdict}  |  recommended drop_chains = "
          f"{recommended_drop if recommended_drop else 'none'}")

    row = {"fit": args.name, "trace": str(args.trace), "n_chains": n_chains,
           "divergences": divergences,
           "logp_spread_nats": round(logp_spread, 2),
           "dspread_max": round(float(dspread.max()), 3),
           "eta_rhat_all": round(eta_all, 4),
           "D_rhat_all": round(d_rhat_all, 4),
           "island_chains": " ".join(map(str, islands)),
           "eta_rhat_majority": round(eta_majority, 4),
           "verdict": verdict,
           "recommended_drop_chains": " ".join(map(str, recommended_drop))}
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = (pd.read_csv(out_csv) if out_csv.exists()
          else pd.DataFrame(columns=list(row)))
    df = pd.concat([df[df["fit"] != args.name], pd.DataFrame([row])],
                   ignore_index=True)
    df.to_csv(out_csv, index=False)
    print(f"  verdict row → {out_csv}")

    if args.fig:
        fig_path = ROOT / "plots" / "mirt" / f"chains_{args.name}.pdf"
        _figure(f"{args.name} — {verdict} (logp spread {logp_spread:.0f} nats, "
                f"div {divergences})",
                below, ~island_mask, dspread,
                list(post["bench"].values), chain_dist, fig_path)
        print(f"  figure → {fig_path}")


if __name__ == "__main__":
    main()
