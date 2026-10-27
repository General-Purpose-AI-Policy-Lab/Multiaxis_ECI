"""--general-sums: on the bifactor loadings the reported axes are g and g + sigma_k z_k, every
theta block being built in (g, z), and the option round-trips through the fit tag and the attrs."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "2_model"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_axis_pin import _toy_data  # noqa: E402


def test_axes_are_g_plus_scaled_deviations():
    """theta[:, 0] is the exchangeable block's first column and theta[:, k] = theta[:, 0] +
    sigma_k * cell[:, k], draw by draw under the prior."""
    import pymc as pm
    from multiaxis_eci.models.mirt import build_mirt_model
    model = build_mirt_model(_toy_data(), 3, loading_prior="bifactor", theta_skew=True,
                             general_sums=True)
    with model:
        prior = pm.sample_prior_predictive(
            200, var_names=["theta", "theta_skew", "theta_dev_sd"], random_seed=2).prior
    theta = prior["theta"].values.reshape(-1, 6, 3)
    cells = prior["theta_skew"].values.reshape(-1, 6, 3)
    sd = prior["theta_dev_sd"].values.reshape(-1, 2)
    np.testing.assert_allclose(theta[:, :, 0], cells[:, :, 0], atol=1e-10)
    np.testing.assert_allclose(theta[:, :, 1:] - theta[:, :, :1], cells[:, :, 1:] * sd[:, None, :],
                               atol=1e-10)
    # prior correlation of two axes at sigma = 1: 1 / (1 + sigma^2) = 0.5, here a wider mix
    r = np.corrcoef(theta[:, :, 1].ravel(), theta[:, :, 2].ravel())[0, 1]
    assert 0.25 < r < 0.7


def test_general_sums_is_refused_off_bifactor_and_with_theta_general():
    from multiaxis_eci.analysis.fitspec import FitSpec
    from multiaxis_eci.models.mirt import build_mirt_model
    data = _toy_data()
    with pytest.raises(ValueError, match="bifactor"):
        build_mirt_model(data, 3, general_sums=True)
    with pytest.raises(ValueError, match="pick one"):
        build_mirt_model(data, 3, loading_prior="bifactor", theta_general=True,
                         general_sums=True)
    with pytest.raises(ValueError, match="bifactor"):
        FitSpec(K=5, general_sums=True)


def test_general_sums_round_trips_through_tag_and_attrs():
    from multiaxis_eci.analysis.fitspec import FitSpec, _parse_tag
    spec = FitSpec(K=5, loading_prior="bifactor", lineage_prior=True, lineage_bm=True,
                   general_sums=True)
    assert spec.tag == "_bifactor_lineageprior_lineagebm_gsum"
    parsed = _parse_tag(spec.tag)
    assert parsed["general_sums"] is True and parsed["loading_prior"] == "bifactor"
    assert "general_sums" not in _parse_tag(FitSpec(K=5, loading_prior="bifactor").tag)
