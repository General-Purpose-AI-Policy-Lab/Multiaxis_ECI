"""The predicted-score reading of an axis (analysis.reading): the test-taker's predicted score on
the benchmarks the axis dominates, in place of its fitted ability on the axis alone."""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from multiaxis_eci.analysis import FitView, predicted_view, reading_view
from multiaxis_eci.analysis.reading import (
    PREDICTED_LABEL,
    _raw_scores,
    axis_score_maps,
    axis_scores,
    has_predicted_reading,
    value_label,
)


def _view(A, theta, D, **kw) -> FitView:
    """A one-draw view on loadings (B, K), abilities (M, K) and difficulties (B,)."""
    K = A.shape[1]
    return FitView(theta=theta[None], Phi=np.eye(K), Phi_raw=np.eye(K),
                   names=[f"axis{k + 1}" for k in range(K)], K=K, is_nc=False, anchored=False,
                   rotated=False, A=A[None], D=D[None], D_med=D, **kw)


def _data(n_models, human=()):
    is_human = np.zeros(n_models, bool)
    is_human[list(human)] = True
    return SimpleNamespace(is_human=is_human)


def test_simple_structure_gives_back_the_ability():
    """Where every benchmark loads on one axis only, the predicted score is the ability itself,
    centred and scaled over the models."""
    rng = np.random.default_rng(0)
    theta = rng.normal(0, 1, (40, 2))
    A = np.array([[1.2, 0.0], [0.7, 0.0], [0.0, 2.0], [0.0, 0.4]])
    view = _view(A, theta, rng.normal(0, 1, 4))
    scores, _ = axis_scores(view)
    for k in range(2):
        z = (theta[:, k] - theta[:, k].mean()) / theta[:, k].std()
        np.testing.assert_allclose(scores[0, :, k], z, atol=1e-9)


def test_compensation_leaves_the_score_unchanged():
    """Two abilities moved against each other so that the predicted scores on an axis's
    benchmarks stay put give the same score there, although the ability on the axis differs."""
    rng = np.random.default_rng(1)
    theta = rng.normal(0, 1, (30, 2))
    A = np.array([[1.0, 0.5], [2.0, 1.0], [0.2, 1.5]])     # axis 1 dominates the first two
    D = np.array([0.3, -0.2, 0.1])
    moved = theta.copy()
    moved[0] += 3.0 * np.array([-0.5, 1.0])                # orthogonal to (1, 0.5) and (2, 1)
    maps = axis_score_maps(A, theta, D)
    assert list(maps[0]["bench"]) == [0, 1]
    before = _raw_scores(theta[None], A[None], D[None], maps[0])
    after = _raw_scores(moved[None], A[None], D[None], maps[0])
    np.testing.assert_allclose(after, before, atol=1e-9)
    assert abs(moved[0, 0] - theta[0, 0]) > 1.0


def test_same_predicted_scores_same_reading_whatever_the_frame():
    """Only the products A . theta enter: an invertible change of frame of (A, theta) that
    keeps them leaves the scores of a given set of benchmarks unchanged."""
    rng = np.random.default_rng(2)
    theta = rng.normal(0, 1, (25, 3))
    A = np.abs(rng.normal(0, 1, (6, 3)))
    D = rng.normal(0, 1, 6)
    T = np.array([[1.0, 0.3, 0.0], [0.0, 1.0, 0.2], [0.1, 0.0, 1.0]])
    m = axis_score_maps(A, theta, D)[0]
    a = _raw_scores(theta[None], A[None], D[None], m)
    b = _raw_scores((theta @ np.linalg.inv(T).T)[None], (A @ T)[None], D[None], m)
    np.testing.assert_allclose(a, b, atol=1e-9)


def test_human_rows_are_read_but_not_scaled_on():
    rng = np.random.default_rng(3)
    theta = rng.normal(0, 1, (20, 2))
    A = np.array([[1.0, 0.2], [0.8, 0.1], [0.1, 1.0], [0.3, 0.9]])
    D = np.zeros(4)
    alone, _ = axis_scores(_view(A, theta, D))
    with_human = np.vstack([theta, [[9.0, -9.0]]])
    scores, _ = axis_scores(_view(A, with_human, D), is_human=_data(21, human=[20]).is_human)
    np.testing.assert_allclose(scores[0, :20], alone[0], atol=1e-9)
    assert scores[0, 20, 0] > scores[0, :20, 0].max()       # the tier is read by the same map


def test_predicted_view_marks_and_keeps_the_fit():
    rng = np.random.default_rng(4)
    theta = rng.normal(0, 1, (20, 2))
    A = np.array([[1.0, 0.4], [0.9, 0.2], [0.3, 1.0]])
    view = _view(A, theta, np.zeros(3))
    pv = predicted_view(view, _data(20))
    assert pv.reading == "predicted" and view.reading == "theta"
    assert pv.theta_fit is view.theta and pv.A is view.A
    assert predicted_view(pv, _data(20)) is pv               # applied once
    assert reading_view(view, _data(20), "theta") is view
    assert reading_view(view, _data(20), "predicted").reading == "predicted"
    with pytest.raises(ValueError):
        reading_view(view, _data(20), "eta")
    with pytest.raises(ValueError):
        reading_view(pv, _data(20), "theta")
    assert value_label(pv, "ability") == PREDICTED_LABEL["en"]
    assert value_label(view, "ability") == "ability"


def test_no_reading_without_a_linear_predictor():
    rng = np.random.default_rng(5)
    view = _view(np.abs(rng.normal(0, 1, (4, 2))), rng.normal(0, 1, (10, 2)), np.zeros(4))
    assert has_predicted_reading(view)
    for other in (dataclasses.replace(view, D=None),             # softmin / log-logistic link
                  dataclasses.replace(view, phi_is_promax=True),  # a signed fit's display frame
                  dataclasses.replace(view, is_nc=True)):
        assert not has_predicted_reading(other)
        assert predicted_view(other, _data(10)) is other


def test_placed_tier_reads_as_the_typical_model_at_its_ability():
    """A tier placed on a fit without humans is the typical model at its ability on the axis:
    where the models lie exactly on that profile, a tier placed at a model's ability gets that
    model's score."""
    t = np.linspace(-2, 2, 15)
    theta = np.column_stack([t, 0.5 + 0.8 * t])              # axis 2 a line of axis 1
    A = np.array([[1.0, 0.3], [0.7, 0.2], [0.2, 1.1]])
    D = np.array([0.1, -0.3, 0.2])
    draws = np.array([[t[3], t[11]]])                        # two tiers on axis 1
    placement = {"axes": [{"order": ["a", "b"], "drawn": ["a", "b"], "draws": draws},
                          {"order": [], "drawn": [], "draws": None}]}
    pv = predicted_view(_view(A, theta, D, human_placement=placement), _data(15))
    got = pv.human_placement["axes"][0]["draws"]
    np.testing.assert_allclose(got[0], pv.theta[0, [3, 11], 0], atol=1e-9)
    assert placement["axes"][0]["draws"] is draws            # the fit's placement is untouched
    assert pv.human_placement["axes"][1]["draws"] is None
