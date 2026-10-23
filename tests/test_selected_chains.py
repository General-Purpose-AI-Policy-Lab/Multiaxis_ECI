"""The modes file's `selected_chains` decides the chains every deliverable reads, and a rewrite of
the mode split keeps it."""
from __future__ import annotations

import json

import arviz as az
import numpy as np

from multiaxis_eci.analysis import best_mode_chains
from multiaxis_eci.scripts import load


def _modes(path, trace, selected=None):
    doc = {"trace": trace.name, "n_chains": 6, "modes": [
        {"label": "A", "chains": [0, 1, 2], "delta_logp": 0.0},
        {"label": "B", "chains": [3, 4, 5], "delta_logp": -30.0}]}
    if selected:
        doc["selected_chains"] = {"chains": selected, "why": "test"}
    path.write_text(json.dumps(doc))


def test_selected_chains_override_the_best_mode(tmp_path):
    trace = tmp_path / "trace_mirt_k4_x.nc"
    modes = tmp_path / "mirt_modes_trace_mirt_k4_x.json"
    _modes(modes, trace)
    assert best_mode_chains(trace) == [0, 1, 2]
    _modes(modes, trace, selected=[2, 0])
    assert best_mode_chains(trace) == [0, 2]
    split = load("4_diagnostics/3_plot_mirt.py").chain_split(trace, 6)
    assert split["primary"] == [0, 2] and split["secondary"] == [1, 3, 4, 5]


def test_rewriting_the_modes_keeps_the_selection(tmp_path, monkeypatch):
    dc = load("4_diagnostics/diagnose_chains.py")
    trace = tmp_path / "trace_mirt_k4_x.nc"
    modes = tmp_path / "mirt_modes_trace_mirt_k4_x.json"
    _modes(modes, trace, selected=[0, 2])
    monkeypatch.setattr(dc, "_chain_logp", lambda idata: np.zeros(6))
    monkeypatch.setattr(dc, "detect_modes", lambda post, logp, t: [
        {"label": "A", "chains": [0, 1, 2, 3, 4, 5], "delta_logp": 0.0}])
    idata = az.from_dict(posterior={"A": np.zeros((6, 2, 3, 2))}, dims={"A": ["bench", "latent"]})
    dc.write_modes(idata, trace, 0.8)
    assert json.loads(modes.read_text())["selected_chains"]["chains"] == [0, 2]


def test_a_refit_sets_the_old_selection_aside(tmp_path):
    """A refit writes the same trace name, so the old selection would carry over to chains it
    was never made on."""
    fit = load("3_fit/fit.py")
    trace = tmp_path / "trace_mirt_k4_x.nc"
    modes = tmp_path / "mirt_modes_trace_mirt_k4_x.json"
    _modes(modes, trace, selected=[0, 2])
    fit.retire_chain_selection(trace)
    assert not modes.exists() and best_mode_chains(trace) is None
    assert len(list(tmp_path.glob("mirt_modes_trace_mirt_k4_x_*.json"))) == 1
    # A computed split carries nothing by hand: removed, not kept.
    _modes(modes, trace)
    fit.retire_chain_selection(trace)
    assert not modes.exists() and len(list(tmp_path.glob("*.json"))) == 1


def test_the_data_fingerprint_sees_what_the_dimensions_miss():
    """A rebuild in place keeps the dimension sizes; a changed score or item count must still
    stop the K=1 baseline cache and --merge-chains."""
    from types import SimpleNamespace

    import pandas as pd
    fit = load("3_fit/fit.py")
    data = SimpleNamespace(
        mlookup=pd.DataFrame({"model": ["a", "b"], "model_idx": [0, 1]}),
        blookup=pd.DataFrame({"benchmark": ["x", "y"], "benchmark_idx": [0, 1]}),
        model_idx=np.array([0, 1, 1]), bench_idx=np.array([0, 0, 1]),
        scores=np.array([0.2, 0.5, 0.9]))
    n = np.array([40.0, 40.0, 25.0])
    ref = fit.data_fingerprint(data, None, n)
    assert fit.data_fingerprint(data, None, n.copy()) == ref
    assert fit.data_fingerprint(data, None, np.array([40.0, 40.0, 50.0])) != ref
    data.scores = np.array([0.2, 0.5, 1.0])
    assert fit.data_fingerprint(data, None, n) != ref
    # under the ceiling model, a bound moved by a human baseline outside the fit (2026-10-21:
    # SimpleQA Verified's 0.944 withdrawn) must stop a merge too
    known, best = np.array([np.nan, 1.0]), np.array([0.775, 0.9])
    with_c = fit.data_fingerprint(data, None, n, (known, best))
    assert with_c != fit.data_fingerprint(data, None, n)
    assert fit.data_fingerprint(data, None, n, (known, np.array([0.944, 0.9]))) != with_c


def test_a_subset_is_put_on_the_display_frame_only_when_its_axes_match(tmp_path):
    """Columns are paired by the best assignment, and a pair below AXIS_MATCH_MIN means the
    subset carries another axis: refused, never silently relabelled (2026-10-06)."""
    from types import SimpleNamespace

    import pandas as pd
    import pytest

    from multiaxis_eci.analysis import align_to_reference_loadings
    from multiaxis_eci.analysis.fitview import FitView
    rng = np.random.default_rng(0)
    B, K = 12, 3
    ref = rng.uniform(0, 1, (B, K))
    bench = [f"b{i}" for i in range(B)]
    csv = tmp_path / "mirt_loadings.csv"
    pd.DataFrame([{"benchmark": b, "axis": f"axis{k + 1}", "loading_median": ref[i, k]}
                  for i, b in enumerate(bench) for k in range(K)]).to_csv(csv, index=False)
    data = SimpleNamespace(blookup=pd.DataFrame({"benchmark": bench, "benchmark_idx": range(B)}))

    def view(A):
        return FitView(theta=np.zeros((1, 2, K)), Phi=np.eye(K), Phi_raw=np.eye(K),
                       names=[f"axis{k + 1}" for k in range(K)], K=K, is_nc=False,
                       anchored=False, rotated=False, A=A[None])

    out = align_to_reference_loadings(view(ref[:, [2, 0, 1]]), data, csv)
    np.testing.assert_allclose(out.A[0], ref)
    other = ref.copy()
    other[:, 1] = rng.uniform(0, 1, B)
    with pytest.raises(SystemExit):
        align_to_reference_loadings(view(other), data, csv)


def test_the_dashboard_cache_key_sees_the_axis_names_and_the_frame(tmp_path):
    """A card is re-rendered when its axis names or display frame change, not only its trace;
    the ESS cache, which depends on the posterior alone, ignores those inputs."""
    bd = load("4_diagnostics/4_build_dashboard.py")
    trace = tmp_path / "trace_mirt_k4_x.nc"
    trace.write_bytes(b"trace")
    names = tmp_path / "axis_names.json"
    names.write_text('{"confirmed": false}')
    before = bd._fingerprint(trace)
    names.write_text('{"confirmed": true}')
    after = bd._fingerprint(trace)
    assert before != after and before["inputs"] != after["inputs"]
    assert {k: v for k, v in before.items() if k != "inputs"} == \
        {k: v for k, v in after.items() if k != "inputs"}
    (tmp_path / "mirt_loadings.csv").write_text("benchmark,axis\n")
    assert bd._fingerprint(trace)["inputs"] != after["inputs"]


def test_the_forecast_only_view_is_the_cards(tmp_path, monkeypatch):
    """forecast_only builds its view as the card does: the card's chains, then the fit's
    display frame and the axis titles, so its panels cannot sit on other axes."""
    from types import SimpleNamespace

    import xarray as xr
    fo = load("4_diagnostics/forecast_only.py")
    trace = tmp_path / "trace_mirt_k4_x.nc"
    xr.Dataset({"A": (("chain", "draw"), np.zeros((6, 2)))}).to_netcdf(trace, group="posterior")
    (tmp_path / "mirt_loadings.csv").write_text("benchmark,axis,loading_median\n")
    calls = []
    spec = SimpleNamespace(open_posterior=lambda keep, chains, path: calls.append(("open", chains)))
    view = SimpleNamespace(A=np.zeros((1, 2, 4)), K=4)
    monkeypatch.setattr(fo, "_trace_path", lambda fit: trace)
    monkeypatch.setattr(fo, "prepare_fit", lambda idata, data: view)
    monkeypatch.setattr(fo, "align_to_reference_loadings",
                        lambda v, data, ref: calls.append(("align", ref.name)) or v)
    monkeypatch.setattr(fo, "load_axis_titles",
                        lambda folder, v, data: calls.append(("titles",)) or {"axis1": "T"})
    out, titles = fo.card_view({"spec": spec, "drop_chains": [0, 3]}, None)
    assert calls == [("open", [1, 2, 4, 5]), ("align", "mirt_loadings.csv"), ("titles",)]
    assert out is view and titles == {"axis1": "T"}
