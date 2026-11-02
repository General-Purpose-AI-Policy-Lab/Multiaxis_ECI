"""--orthogonal: on the bifactor loadings g and the specifics are held orthogonal by a Potential on
their in-sample correlations, and the axes are displayed as g and g + sigma_k s_k with every eta
unchanged."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2_model"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_axis_pin import _toy_data  # noqa: E402


def test_the_potential_is_the_spread_of_independent_correlations():
    """theta_orthogonal = -N/2 sum_{j<k} r_jk^2 at scale 1, r the in-sample correlation."""
    from multiaxis_eci.models.mirt import build_mirt_model
    data = _toy_data(n_models=8)
    model = build_mirt_model(data, 3, loading_prior="bifactor", theta_skew=True, orthogonal=True)
    assert "theta_orthogonal" in [p.name for p in model.potentials]
    rng = np.random.default_rng(3)
    point = {k: rng.normal(size=np.shape(v)) for k, v in model.initial_point().items()}
    theta, pot = model.compile_fn([model["theta"], model.potentials[0]], inputs=model.value_vars,
                                  on_unused_input="ignore")(point)
    r = np.corrcoef(theta.T)[np.triu_indices(3, 1)]
    np.testing.assert_allclose(pot, -0.5 * 8 * (r ** 2).sum(), rtol=1e-6)


def test_orthogonal_is_refused_off_the_free_bifactor():
    from multiaxis_eci.analysis.fitspec import FitSpec
    from multiaxis_eci.models.mirt import build_mirt_model
    data = _toy_data()
    with pytest.raises(ValueError, match="bifactor"):
        build_mirt_model(data, 3, orthogonal=True)
    with pytest.raises(ValueError, match="bifactor"):
        build_mirt_model(data, 3, loading_prior="bifactor", general_sums=True, orthogonal=True)
    with pytest.raises(ValueError, match="bifactor"):
        FitSpec(K=4, orthogonal=True)
    spec = FitSpec(K=4, loading_prior="bifactor", orthogonal=True, no_humans=True)
    assert spec.tag.startswith("_bifactor") and "_orth" in spec.tag
    assert FitSpec.from_spec_json('{"K": 4, "loading_prior": "bifactor"}').orthogonal is False


def test_the_composite_keeps_every_eta_and_reads_sigma_off_each_domain():
    """eta = theta A^T is the same on the displayed axes; sigma_k is the ratio of the mean
    specific to the mean general loading over the benchmarks dominated by s_k."""
    from multiaxis_eci.analysis import orthogonal_composite
    rng = np.random.default_rng(1)
    S, N = 5, 20
    # benchmarks 0-1 on g, 2-3 on s_1, 4-5 on s_2
    base = np.array([[1.5, 0.1, 0.0], [1.2, 0.0, 0.2], [0.5, 1.0, 0.0],
                     [0.3, 1.4, 0.1], [0.4, 0.0, 0.8], [0.6, 0.1, 1.2]])
    A = base[None] * rng.uniform(0.9, 1.1, (S, 6, 3))
    theta = rng.normal(size=(S, N, 3))
    A_c, theta_c, sigma = orthogonal_composite(A, theta)
    np.testing.assert_allclose(np.einsum("snk,sbk->snb", theta_c, A_c),
                               np.einsum("snk,sbk->snb", theta, A), atol=1e-10)
    np.testing.assert_allclose(sigma[:, 0], A[:, 2:4, 1].mean(1) / A[:, 2:4, 0].mean(1))
    np.testing.assert_allclose(sigma[:, 1], A[:, 4:6, 2].mean(1) / A[:, 4:6, 0].mean(1))
    np.testing.assert_allclose(theta_c[:, :, 0], theta[:, :, 0])
    np.testing.assert_allclose(theta_c[:, :, 1], theta[:, :, 0] + sigma[:, :1] * theta[:, :, 1])
