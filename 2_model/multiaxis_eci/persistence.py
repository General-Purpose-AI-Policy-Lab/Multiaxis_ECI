"""Persistence helpers: trace, summary tables, scalar metrics."""
from __future__ import annotations

import json
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd

from multiaxis_eci.data import ECIData


def save_trace(trace, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # netCDF attributes hold flat scalars and arrays only. nutpie's zarr-store
    # path stamps a NESTED sampler_settings dict on the trace, which raises
    # here, so flatten any dict attribute to JSON instead of dropping the run's
    # provenance (it records target_accept, max tree depth and the seed).
    for key, value in list(trace.attrs.items()):
        if isinstance(value, dict):
            trace.attrs[key] = json.dumps(value)
    trace.to_netcdf(str(path))


def thin_trace(trace, step: int, drop_transformed: bool = True):
    """The trace with every `step`-th draw kept, for saving.

    Every figure and table in this repository is a median or an interval, which 16,000 draws
    pin as well as 80,000, while the trace file scales with the draw count (a 10,000 x 8 K=4
    fit writes about 11 GB unthinned). Thinning happens at save time only: convergence
    (r-hat, ESS, divergences) and the deliverable tables are computed on the full run first.
    `drop_transformed` also removes PyMC's unconstrained duplicates (`*__` variables such as
    `A_z_log__`), which nothing downstream reads. The step is stamped as `mirt_save_thin` so
    a reader knows the file's draws are a subsample. `step` 1 returns the trace unchanged.
    """
    if step is None or step <= 1:
        return trace
    groups = {}
    for name in trace.groups():
        ds = getattr(trace, name)
        if "draw" in ds.dims:
            ds = ds.isel(draw=slice(None, None, step))
        if name == "posterior" and drop_transformed:
            ds = ds.drop_vars([v for v in ds.data_vars if v.endswith("__")])
        groups[name] = ds
    out = az.InferenceData(**groups)
    out.posterior.attrs.update(trace.posterior.attrs)
    out.posterior.attrs["mirt_save_thin"] = int(step)
    return out


def load_trace(path: Path):
    return az.from_netcdf(str(path))


def merge_chains(earlier, later):
    """`earlier`'s chains followed by `later`'s, as one trace: how chains are added to a fit.

    Both must be the same fit (identical `mirt_spec` and data fingerprint, same draws per chain and the same save
    thinning, every other dimension equal); `later`'s chains are renumbered after `earlier`'s.
    Only the groups both traces carry are kept, with `earlier`'s attributes. Raises ValueError
    when the two cannot be one fit.
    """
    pa, pb = earlier.posterior.attrs, later.posterior.attrs
    for key in ("mirt_spec", "mirt_save_thin", "mirt_data_fingerprint"):
        if pa.get(key) != pb.get(key):
            raise ValueError(f"merge_chains: {key} differs ({pa.get(key)!r} vs {pb.get(key)!r})")
    if earlier.posterior.sizes["draw"] != later.posterior.sizes["draw"]:
        raise ValueError("merge_chains: different draws per chain "
                         f"({earlier.posterior.sizes['draw']} vs {later.posterior.sizes['draw']})")
    import xarray as xr  # only needed here
    n0 = earlier.posterior.sizes["chain"]
    groups = {}
    for name in earlier.groups():
        if name not in later.groups():
            continue
        a, b = getattr(earlier, name), getattr(later, name)
        if "chain" not in a.dims:
            groups[name] = a
            continue
        b = b.assign_coords(chain=np.arange(n0, n0 + b.sizes["chain"]))
        a = a.assign_coords(chain=np.arange(n0))
        common = [v for v in a.data_vars if v in b.data_vars]
        merged = xr.concat([a[common], b[common]], dim="chain", join="exact")
        merged.attrs = dict(a.attrs)
        groups[name] = merged
    return az.InferenceData(**groups)


def load_live_draws(path: Path):
    """Read a `--stream-draws` zarr store, INCLUDING one still being written.

    nutpie pre-allocates the full (chain, draw, ...) arrays and fills them as
    the draws land, so an unfinished store reads back at final shape with NaN
    where no draw exists yet. Draws arrive in order, so the trace is trimmed to
    the prefix every chain has finished — enough for r-hat / ESS / a timeline on
    a run still in progress, or for salvaging one that died.

    Warmup groups are dropped: nutpie's store writes them whatever save_warmup
    says, and nothing in this repo reads them.

    xarray's datatree does the reading because arviz cannot open a zarr 3 store.
    """
    import xarray as xr  # only needed here
    idata = az.InferenceData.from_datatree(
        xr.open_datatree(str(path), engine="zarr", consolidated=False))
    for group in ("warmup_posterior", "warmup_sample_stats"):
        if group in idata.groups():
            delattr(idata, group)
    done = np.isfinite(idata.sample_stats["energy"].values).all(axis=0)
    n = int(done.argmin()) if not done.all() else done.size
    if n == 0:
        raise ValueError(f"{path}: no chain has finished a post-warmup draw yet.")
    idata = idata.isel(draw=slice(0, n))
    # The stream holds the draws as sampled: the effort sign is fixed here as in fit.py.
    from multiaxis_eci.models.mirt import align_effort_sign
    align_effort_sign(idata.posterior)
    return idata


def save_summary(trace, path: Path) -> None:
    az.summary(trace, round_to=6).to_csv(path)


def save_df(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def save_json(payload: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def save_pit(pit: np.ndarray, data: ECIData, path: Path,
             pareto_k: np.ndarray | None = None) -> None:
    """PIT skips the boundary-clipped rows (exact 0s and scores ≥ 1 - ECI_EPS)
    — emit only the rows it was computed on, via the same mask ppc uses. A LOO-PIT
    (`GoFResults.pit_all`) comes with its PSIS `pareto_k` per row, written beside it."""
    from multiaxis_eci.ppc import boundary_mask
    nonzero = ~boundary_mask(data)
    df = pd.DataFrame({
        "obs_idx":   np.where(nonzero)[0],
        "model":     data.mlookup["model"].values[data.model_idx[nonzero]],
        "benchmark": data.blookup["benchmark"].values[data.bench_idx[nonzero]],
        "score":     data.scores[nonzero],
        "pit":       pit,
    })
    if pareto_k is not None:
        df["pareto_k"] = pareto_k
    df.to_csv(path, index=False)


def _json_default(o):
    """Coerce numpy scalars/arrays to JSON-serialisable types."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    raise TypeError(f"Object of type {type(o)} is not JSON serializable")
