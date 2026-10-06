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
