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
    assert split["majority"] == [0, 2] and split["minority"] == [1, 3, 4, 5]


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
