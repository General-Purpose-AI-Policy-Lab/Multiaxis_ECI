"""Saving a trace keeps every n-th draw and drops PyMC's unconstrained duplicates; chains of
two runs of one fit merge into one trace."""
from __future__ import annotations

import arviz as az
import numpy as np
import pytest

from multiaxis_eci.persistence import merge_chains, thin_trace


def _trace(n_draws=100):
    rng = np.random.default_rng(0)
    post = {"theta": rng.normal(size=(2, n_draws, 3)), "A_z_log__": rng.normal(size=(2, n_draws, 3)),
            "A": rng.normal(size=(2, n_draws, 3))}
    idata = az.from_dict(posterior=post,
                         log_likelihood={"obs": rng.normal(size=(2, n_draws, 5))},
                         sample_stats={"diverging": np.zeros((2, n_draws), bool)},
                         observed_data={"y": np.arange(5.0)})
    idata.posterior.attrs["mirt_spec"] = "{}"
    return idata


def test_thin_trace_keeps_every_nth_draw_and_stamps_the_step():
    idata = _trace()
    out = thin_trace(idata, 5)
    assert out.posterior.sizes["draw"] == 20
    assert out.log_likelihood.sizes["draw"] == 20
    assert out.sample_stats.sizes["draw"] == 20
    np.testing.assert_array_equal(out.posterior["theta"].values, idata.posterior["theta"].values[:, ::5])
    assert "A_z_log__" not in out.posterior and "A" in out.posterior
    assert out.posterior.attrs["mirt_save_thin"] == 5
    assert out.posterior.attrs["mirt_spec"] == "{}"          # provenance survives
    assert out.observed_data["y"].size == 5                    # draw-free groups untouched
    assert idata.posterior.sizes["draw"] == 100                # the original is not modified
    assert thin_trace(idata, 1) is idata


def test_merge_chains_appends_the_later_runs_chains():
    """Two thinned runs of one fit become one trace whose later chains follow the earlier ones;
    the data groups are kept once."""
    a, b = thin_trace(_trace(), 5), thin_trace(_trace(), 5)
    b.posterior["theta"] = b.posterior["theta"] + 10.0
    out = merge_chains(a, b)
    assert out.posterior.sizes["chain"] == 4 and out.posterior.sizes["draw"] == 20
    assert list(out.posterior.chain.values) == [0, 1, 2, 3]
    assert float(out.posterior["theta"].isel(chain=slice(2, None)).mean()) > 5.0
    assert out.log_likelihood.sizes["chain"] == 4 and out.sample_stats.sizes["chain"] == 4
    assert np.array_equal(out.observed_data["y"].values, np.arange(5.0))
    assert out.posterior.attrs["mirt_save_thin"] == 5


def test_merge_chains_refuses_two_different_fits():
    a = thin_trace(_trace(), 5)
    b = thin_trace(_trace(), 5)
    b.posterior.attrs["mirt_spec"] = '{"K": 3}'
    with pytest.raises(ValueError, match="mirt_spec"):
        merge_chains(a, b)
    with pytest.raises(ValueError, match="mirt_save_thin"):
        merge_chains(a, thin_trace(_trace(), 4))
    with pytest.raises(ValueError, match="draws per chain"):
        merge_chains(a, thin_trace(_trace(200), 5))
