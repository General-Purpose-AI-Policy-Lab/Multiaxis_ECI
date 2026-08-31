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


def load_trace(path: Path):
    return az.from_netcdf(str(path))


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
    import xarray as xr                                  # only needed here
    idata = az.InferenceData.from_datatree(
        xr.open_datatree(str(path), engine="zarr", consolidated=False))
    for group in ("warmup_posterior", "warmup_sample_stats"):
        if group in idata.groups():
            delattr(idata, group)
    done = np.isfinite(idata.sample_stats["energy"].values).all(axis=0)
    n = int(done.argmin()) if not done.all() else done.size
    if n == 0:
        raise ValueError(f"{path}: no chain has finished a post-warmup draw yet.")
    return idata.isel(draw=slice(0, n))


def save_summary(trace, path: Path) -> None:
    az.summary(trace, round_to=6).to_csv(path)


def save_df(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def save_json(payload: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=_json_default)


def save_pit(pit: np.ndarray, data: ECIData, path: Path) -> None:
    """PIT skips the boundary-clipped rows (exact 0s and scores ≥ 1 - ECI_EPS)
    — emit only the rows it was computed on, via the same mask ppc uses."""
    from multiaxis_eci.ppc import boundary_mask
    nonzero = ~boundary_mask(data)
    df = pd.DataFrame({
        "obs_idx":   np.where(nonzero)[0],
        "model":     data.mlookup["model"].values[data.model_idx[nonzero]],
        "benchmark": data.blookup["benchmark"].values[data.bench_idx[nonzero]],
        "score":     data.scores[nonzero],
        "pit":       pit,
    })
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
