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


def test_save_trace_writes_draws_in_float32_and_keeps_the_rest(tmp_path):
    from multiaxis_eci.persistence import load_trace, save_trace
    idata = _trace()
    idata.sample_stats["logp"] = (("chain", "draw"), np.full((2, 100), -15000.123456))
    idata.sample_stats.attrs["sampler_settings"] = {"target_accept": 0.95}
    save_trace(idata, tmp_path / "t.nc")
    back = load_trace(tmp_path / "t.nc")
    assert back.posterior["theta"].dtype == np.float32
    assert back.log_likelihood["obs"].dtype == np.float32
    assert back.sample_stats["logp"].dtype == np.float64          # logp sums stay exact
    np.testing.assert_allclose(back.posterior["theta"].values, idata.posterior["theta"].values,
                               rtol=1e-6)
    assert back.posterior.attrs["mirt_spec"] == "{}"
    assert idata.posterior["theta"].dtype == np.float64           # the caller's trace untouched
    assert set(back.groups()) == set(idata.groups())


def test_save_trace_drops_nutpies_per_parameter_stats_whatever_the_thinning(tmp_path):
    """--save-thin 1 and a merge of chains (saved without thin_trace) get the same format."""
    from multiaxis_eci.persistence import load_trace, save_trace
    idata = _trace()
    rng = np.random.default_rng(1)
    idata.sample_stats["gradient"] = (("chain", "draw", "unconstrained_parameter"),
                                      rng.normal(size=(2, 100, 4)))
    idata.sample_stats["logp"] = (("chain", "draw"), rng.normal(size=(2, 100)))
    save_trace(thin_trace(idata, 1), tmp_path / "t.nc")
    back = load_trace(tmp_path / "t.nc")
    assert "gradient" not in back.sample_stats
    assert "unconstrained_parameter" not in back.sample_stats.dims
    assert {"logp", "diverging"} <= set(back.sample_stats.data_vars)
    assert back.posterior.sizes["draw"] == 100
    assert "gradient" in idata.sample_stats                    # the caller's trace untouched


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


def test_merge_chains_reads_an_older_spec_as_the_same_fit():
    """A trace whose mirt_spec predates an option (no key for it) merges with today's run of the
    same fit, as fit.py's own check does (FitSpec.from_spec_json); a real change still refuses."""
    import dataclasses
    import json
    from multiaxis_eci.analysis.fitspec import FitSpec
    today = FitSpec(K=4, lineage_prior=True, lineage_bm=True, theta_skew=True, no_humans=True)
    new = {k: (list(v) if isinstance(v, tuple) else v) for k, v in dataclasses.asdict(today).items()}
    old = {k: v for k, v in new.items() if k not in ("general_sums", "axis_pin", "pin_offset")}
    a, b = thin_trace(_trace(), 5), thin_trace(_trace(), 5)
    a.posterior.attrs["mirt_spec"], b.posterior.attrs["mirt_spec"] = json.dumps(old), json.dumps(new)
    assert merge_chains(a, b).posterior.sizes["chain"] == 4
    b.posterior.attrs["mirt_spec"] = json.dumps({**new, "lineage_bm": False})
    with pytest.raises(ValueError, match="mirt_spec"):
        merge_chains(a, b)
