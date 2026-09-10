"""Hand-named axes: the template step, the titles it yields, and the French renders."""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from multiaxis_eci.analysis import (
    axis_names_path,
    bare_axis_title,
    load_axis_titles,
    propose_axis_names,
    require_axis_titles,
)
from multiaxis_eci.viz import translate_fig


@dataclass
class _View:
    A: np.ndarray
    names: list

    def require_A(self):
        return self.A


@dataclass
class _Data:
    blookup: pd.DataFrame


def _fit():
    # Axis 1 is carried by b0/b1, axis 2 by b2/b3 (median loadings).
    A = np.zeros((10, 4, 2))
    A[:, 0, 0] = A[:, 1, 0] = 1.0
    A[:, 2, 1] = A[:, 3, 1] = 1.0
    A[:, 1, 1] = 0.2
    data = _Data(pd.DataFrame({"benchmark": ["b0", "b1", "b2", "b3"],
                               "benchmark_idx": [1, 2, 3, 4]}))
    return _View(A, ["axis1", "axis2"]), data


def test_template_then_titles(tmp_path, capsys):
    view, data = _fit()
    assert bare_axis_title("axis3") == "Axis 3" and bare_axis_title("other") == "other"
    # No file: bare titles everywhere, and the post's strict variant refuses.
    assert load_axis_titles(tmp_path, view, data) == {"axis1": "Axis 1", "axis2": "Axis 2"}
    with pytest.raises(SystemExit):
        require_axis_titles(tmp_path, view, data)
    # The step writes a template listing each axis's top benchmarks, unconfirmed.
    path = propose_axis_names(view, data, tmp_path, top_n=2)
    doc = json.loads(path.read_text())
    assert path == axis_names_path(tmp_path) and doc["confirmed"] is False
    assert doc["axes"]["axis1"]["top_benchmarks"] == ["b0", "b1"]
    assert doc["axes"]["axis2"]["title"] is None
    assert load_axis_titles(tmp_path, view, data)["axis1"] == "Axis 1"
    # A second call never overwrites a file a person may have edited.
    path.write_text(path.read_text().replace('"title": null', '"title": "X"', 1))
    propose_axis_names(view, data, tmp_path)
    assert '"X"' in path.read_text()
    # Confirmed names with a holding signature are applied as `Axis k: title`.
    doc["axes"]["axis1"].update(title="Fluid", signature=["b1"])
    doc["axes"]["axis2"].update(title="Science", signature=["b3"])
    doc["confirmed"] = True
    path.write_text(json.dumps(doc))
    assert load_axis_titles(tmp_path, view, data) == {"axis1": "Axis 1: Fluid",
                                                      "axis2": "Axis 2: Science"}
    assert require_axis_titles(tmp_path) == {"axis1": "Axis 1: Fluid", "axis2": "Axis 2: Science"}
    # A signature that no longer holds: the lax reader falls back and warns, the
    # strict one refuses, so a title cannot migrate to another axis unnoticed.
    doc["axes"]["axis2"]["signature"] = ["b0"]
    path.write_text(json.dumps(doc))
    titles = load_axis_titles(tmp_path, view, data, top_n=2)
    assert titles["axis2"] == "Axis 2" and titles["axis1"] == "Axis 1: Fluid"
    assert "identity check failed" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        require_axis_titles(tmp_path, view, data, top_n=2)


def test_translate_fig_keeps_categories_together():
    fig = go.Figure(go.Scatter(x=[1, 2], y=np.array(["Average Human", "Top Performer"]),
                               mode="markers", name="human tiers"))
    fig.update_layout(title="Forecast: Axis 1: Fluid Intelligence (80% intervals)",
                      yaxis=dict(categoryarray=["Average Human", "Top Performer"]),
                      xaxis_title="Release date")
    fr = translate_fig(fig)
    assert fr.layout.title.text == "Prévision : Axe 1 : Intelligence fluide (intervalles à 80 %)"
    assert list(fr.data[0].y) == ["Humain Moyen", "Meilleur Performeur"]
    assert list(fr.layout.yaxis.categoryarray) == ["Humain Moyen", "Meilleur Performeur"]
    assert fr.layout.xaxis.title.text == "Date de sortie"
    assert fig.layout.title.text.startswith("Forecast")        # the original is untouched
