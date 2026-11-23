"""Integrated LOO with a per-chain checkpoint, several chains at a time.

4_diagnostics/integrated_loo.py keeps every chain's pointwise result in memory and writes its
tables at the very end: the run of 2026-11-20 lost 38 scored chains of 50 when its session ended.
This driver scores each chain in its own process, saves the pointwise result beside the trace
(diagnostics/integrated_loo_chains/chain_XX.npz), skips the chains already saved, then calls the
script's own main() with score_chain reading the saved files, so the tables and the JSON are the
script's, unchanged.

    python 5_outputs/data20261118/mirt_lineageprior_lineagebm/investigation/1_integrated_loo_checkpointed.py \
        --worktree . --trace 5_outputs/data20261118/mirt_lineageprior_lineagebm/trace_mirt_k4_lineageprior_lineagebm.nc

(`--worktree`: the checkout whose 4_diagnostics/integrated_loo.py and 2_model/ it runs; put its
2_model/ on PYTHONPATH.) Run on 2026-11-20 with 4 workers: 50 chains, 1,000 draws each; its JSON
is integrated_loo_4modes_50chains.json beside this script.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path

KEYS = ("e_int", "k_int", "e_cond", "k_cond", "done")


def load_module(worktree: Path):
    sys.path.insert(0, str(worktree / "4_diagnostics"))
    import integrated_loo
    return integrated_loo


def score_one(job) -> tuple[int, float]:
    worktree, trace, chain, cache = job
    import numpy as np
    import xarray as xr
    il = load_module(Path(worktree))
    stored = xr.open_dataset(trace, group="log_likelihood")["obs"]
    r = il.score_chain(Path(trace), chain, stored, None)
    tmp = Path(cache) / f"chain_{chain:02d}.tmp.npz"
    np.savez(tmp, fallback=r["fallback"], **{k: r[k] for k in KEYS})
    tmp.rename(Path(cache) / f"chain_{chain:02d}.npz")
    return chain, float(r["e_int"].sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worktree", type=Path, required=True)
    ap.add_argument("--trace", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    try:
        os.setsid()          # its own session: the launching shell's end does not take it down
    except OSError:
        pass
    cache = args.trace.parent / "diagnostics" / "integrated_loo_chains"
    cache.mkdir(parents=True, exist_ok=True)
    modes = json.loads((args.trace.parent / f"mirt_modes_{args.trace.stem}.json").read_text())
    chains = sorted({c for m in modes["modes"] for c in m["chains"]})
    todo = [c for c in chains if not (cache / f"chain_{c:02d}.npz").exists()]
    print(f"{len(chains)} chains, {len(chains) - len(todo)} already saved, {len(todo)} to score, "
          f"{args.workers} at a time", flush=True)
    if todo:
        jobs = [(str(args.worktree), str(args.trace), c, str(cache)) for c in todo]
        with get_context("spawn").Pool(args.workers, maxtasksperchild=1) as pool:
            for chain, total in pool.imap_unordered(score_one, jobs):
                print(f"saved chain {chain}: elpd_int {total:.1f}", flush=True)

    import numpy as np
    il = load_module(args.worktree)
    first = {"f": None}

    def cached(trace_path, chain, stored, lineage_map=None):
        z = np.load(cache / f"chain_{chain:02d}.npz")
        if first["f"] is None:    # main() reads the data and the draw count from one chain's fit
            first["f"] = il.load_fit(trace_path, [chain], lineage_map)
        return dict(f=first["f"], fallback=int(z["fallback"]), **{k: z[k] for k in KEYS})

    il.score_chain = cached
    sys.argv = ["integrated_loo.py", "--trace", str(args.trace)]
    il.main()


if __name__ == "__main__":
    main()
