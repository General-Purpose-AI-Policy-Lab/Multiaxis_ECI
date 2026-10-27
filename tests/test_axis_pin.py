"""--axis-pin: the pinned main axis of a benchmark leans its simple-structure shares a priori,
and the option round-trips through the fit tag and the trace attrs."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2_model"))


def _toy_data(n_models=6, n_bench=3):
    from multiaxis_eci.data import ECIData
    rng = np.random.default_rng(0)
    model_idx = np.repeat(np.arange(n_models), n_bench)
    bench_idx = np.tile(np.arange(n_bench), n_models)
    n = len(model_idx)
    return ECIData(
        scores=rng.uniform(0.1, 0.9, n), zero_score_mask=np.zeros(n, bool), model_idx=model_idx,
        bench_idx=bench_idx,
        mlookup=pd.DataFrame({"model": [f"m{i}" for i in range(n_models)],
                              "model_idx": np.arange(1, n_models + 1)}),
        blookup=pd.DataFrame({"benchmark": [f"b{j}" for j in range(n_bench)],
                              "benchmark_idx": np.arange(1, n_bench + 1)}),
        n_models=n_models, n_benchmarks=n_bench, n_obs=n, zero_diag_threshold=0.01,
        n_obs_per_model=np.full(n_models, n_bench), is_low_obs=np.zeros(n_models, bool),
        excluded_benchmarks=set(), is_human=np.zeros(n_models, bool), bench_category=None,
        is_sota=np.zeros(n_models, bool))


def test_pin_leans_the_prior_shares_of_the_pinned_rows_only():
    """Under the prior, a pinned row's share on its axis sits near the offset's design value
    (0.93 at concentration 2.5, offset 1.5) while an unpinned row keeps the symmetric shares."""
    import pymc as pm
    from multiaxis_eci.models.mirt import build_mirt_model
    data = _toy_data()
    model = build_mirt_model(data, 3, simple_structure=True, share_conc=2.5,
                             axis_pin={"b0": 2, "b1": 0}, pin_offset=1.5)
    with model:
        prior = pm.sample_prior_predictive(400, var_names=["A_share"], random_seed=1).prior
    share = prior["A_share"].values.reshape(-1, 3, 3)          # (draw, bench, axis)
    assert abs(np.median(share[:, 0, 2]) - 0.93) < 0.04
    assert abs(np.median(share[:, 1, 0]) - 0.93) < 0.04
    # unpinned: no preferred axis, and a less peaked row than the pinned ones
    assert abs(share[:, 2].mean(0) - 1 / 3).max() < 0.05
    assert np.median(share[:, 2].max(1)) < np.median(share[:, 0, 2]) - 0.05


def test_pin_is_refused_off_simple_structure_and_on_unknown_input():
    from multiaxis_eci.models.mirt import build_mirt_model
    data = _toy_data()
    with pytest.raises(ValueError, match="simple_structure"):
        build_mirt_model(data, 3, axis_pin={"b0": 0})
    with pytest.raises(ValueError, match="not in data"):
        build_mirt_model(data, 3, simple_structure=True, axis_pin={"nope": 0})
    with pytest.raises(ValueError, match="out of range"):
        build_mirt_model(data, 3, simple_structure=True, axis_pin={"b0": 3})


def test_pin_round_trips_through_tag_and_attrs():
    from multiaxis_eci.analysis.fitspec import FitSpec, _parse_tag, pin_label
    assert pin_label("/x/Q_geomin.csv") == "geomin"
    assert pin_label("q-Cat_2.csv") == "cat2"
    assert pin_label(None) is None
    spec = FitSpec(K=4, lineage_prior=True, lineage_bm=True, theta_general=True,
                   simple_structure=True, share_conc=2.5, axis_pin="geomin", pin_offset=1.5)
    assert spec.tag == "_lineageprior_lineagebm_general_simplestruct2p5_pingeomin-1p5"
    parsed = _parse_tag(spec.tag)
    assert parsed["axis_pin"] == "geomin" and parsed["pin_offset"] == 1.5
    assert parsed["share_conc"] == 2.5 and parsed["simple_structure"] is True
    # the offset defaults with a pin and is dropped without simple structure
    assert FitSpec(K=4, simple_structure=True, axis_pin="geomin").pin_offset == 1.5
    off = FitSpec(K=4, axis_pin="geomin", pin_offset=2.0)
    assert off.axis_pin is None and off.pin_offset is None and "_pin" not in off.tag
