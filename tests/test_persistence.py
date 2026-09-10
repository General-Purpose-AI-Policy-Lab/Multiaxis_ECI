"""Saving a trace keeps every n-th draw and drops PyMC's unconstrained duplicates."""
from __future__ import annotations

import arviz as az
import numpy as np

from multiaxis_eci.persistence import thin_trace


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
