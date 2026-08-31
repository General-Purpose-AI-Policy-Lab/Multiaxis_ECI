"""Sparse-gate non-compensatory Beta-MIRT.

Conjunctive product link with a per-benchmark, per-axis gate:

    mu = prod_k sigmoid(theta[m,k] + c[b,k]) ^ g[b,k]

`g[b,k]` is the sigmoid EXPONENT, not a slope: g=0 drops axis k from the product
(sigmoid^0 = 1), g=1 is a full conjunctive requirement, g in (0,1) is partial. The
gate is on the exponent because in a product a zero slope leaves a residual
sigmoid(c) factor and so does not remove the axis.

Gates come in two kinds:
  * anchor benchmarks load exactly one axis, fixed — an identity block that names
    the axes;
  * every other benchmark's gates are free under a regularized ("Finnish")
    horseshoe that shrinks most of them toward 0.

Discrimination is fixed at a=1. `theta` carries the human/lineage ordering priors
imported from models.mirt. The Beta likelihood and boundary clipping match
models/mirt_nc.py. The full gate matrix is the Deterministic `g`, read by
ppc.posterior_predictive_mirt_sparse and analysis.mirt_identified_rhat_sparse.
"""
from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from multiaxis_eci.config import (
    ECI_EPS,
    NC_C_OFFSET,
    PRIOR_SIGMA_B,
    PRIOR_TAU_C,
    RH_SLAB_DF,
    RH_SLAB_SCALE,
    RH_TAU_SCALE,
)
from multiaxis_eci.data import ECIData
from multiaxis_eci.lineage import LineageStructure  # noqa: F401  (type hint only)
from multiaxis_eci.models.mirt import _assemble_theta, _human_structure


def _anchor_arrays(data: ECIData, anchors: dict, K: int):
    """Build (anchor_gate, free_cell_mask) from {benchmark_name: axis_idx}.

    anchor_gate : (B, K) float, a fixed 1.0 at each anchor's (row, axis) and 0
        everywhere else.
    free_cell_mask : (B, K) float, 1.0 on EVERY cell of a non-anchor row (its
        gates are estimated) and 0.0 on every cell of an anchor row (fully fixed).

    Anchors must cover each of axes 0..K-1 exactly once (one pure marker per axis).
    Benchmark positions are 0-indexed into the model's `bench` coord.
    """
    bench_names = data.blookup["benchmark"].tolist()
    pos = {b: i for i, b in enumerate(bench_names)}
    anchor_gate = np.zeros((data.n_benchmarks, K))
    free_row = np.ones(data.n_benchmarks)
    axes_used = []
    for name, axis in anchors.items():
        if name not in pos:
            raise ValueError(f"anchor benchmark not in data: {name!r}")
        if not (0 <= axis < K):
            raise ValueError(f"anchor axis {axis} out of range for K={K}")
        bi = pos[name]
        anchor_gate[bi, axis] = 1.0
        free_row[bi] = 0.0
        axes_used.append(axis)
    if sorted(axes_used) != list(range(K)):
        raise ValueError(
            f"anchors must pin each axis 0..{K - 1} exactly once (identity block "
            f"for identification); got axes {sorted(axes_used)}")
    if free_row.sum() == 0:
        raise ValueError("every benchmark is an anchor; no gates left to estimate")
    free_cell_mask = np.repeat(free_row[:, None], K, axis=1)
    return anchor_gate, free_cell_mask


def build_mirt_sparse_model(data: ECIData, anchors: dict, K: int = 3,
                            human_order=None,
                            lineage: "LineageStructure | None" = None,
                            gate_tau0: float = RH_TAU_SCALE) -> pm.Model:
    """Build the sparse-gate non-compensatory Beta-MIRT.

    Parameters
    ----------
    data : ECIData
        Fitted dataset (scores, model_idx, bench_idx, lookups).
    anchors : dict[str, int]
        Pure-anchor identity block, benchmark name -> axis index; must pin each
        axis 0..K-1 exactly once (e.g. {"ARC-AGI-2": 0, "MMLU": 1,
        "OS World (Screenshot)": 2}).
    K : int
        Number of skill axes (default 3).
    human_order, lineage :
        Passed straight to the imported theta-prior blocks (model_mirt). None
        gives the plain ZeroSumNormal anchor.
    gate_tau0 : float
        Global scale of the horseshoe on the gates. Smaller -> stronger sparsity
        (more gates shrink to 0).

    Trace variables: `theta` (model, latent), `c` (bench, latent) easiness
    intercept, `g` (bench, latent) the full gate matrix (fixed anchors + estimated
    horseshoe gates), `sigma_b`, `phi_b`, plus the horseshoe scales.
    """
    anchor_gate, free_cell_mask = _anchor_arrays(data, anchors, K)

    coords = {
        "model":  data.mlookup["model"].tolist(),
        "bench":  data.blookup["benchmark"].tolist(),
        "latent": [f"axis{k + 1}" for k in range(K)],
    }
    with pm.Model(coords=coords) as model:
        ag = pm.Data("anchor_gate", anchor_gate, dims=("bench", "latent"))
        fm = pm.Data("free_cell_mask", free_cell_mask, dims=("bench", "latent"))

        sigma_b = pm.LogNormal("sigma_b", dims="bench", **PRIOR_SIGMA_B)
        phi = pm.Deterministic("phi_b", 1.0 / (4.0 * sigma_b**2) - 1.0, dims="bench")

        # Per-(bench, axis) easiness intercept; b = -c is the difficulty at a=1.
        tau_c = pm.LogNormal("tau_c", **PRIOR_TAU_C)
        c_z = pm.Normal("c_z", 0.0, 1.0, dims=("bench", "latent"))
        c = pm.Deterministic("c", NC_C_OFFSET + c_z * tau_c, dims=("bench", "latent"))

        # theta: human (hard-ordered) + lineage (soft-ordered) blocks, their
        # base levels inside the shared ZeroSumNormal.
        theta = _assemble_theta(                               # (model, latent)
            data.n_models, K,
            human_struct=_human_structure(human_order, coords["model"]),
            lin=lineage)

        # Gates: regularized ("Finnish") horseshoe on the non-anchor cells (>=0,
        # half form so the exponent stays positive); anchor cells keep their fixed
        # 0/1 pattern. The half-Cauchy local and global scales are sampled via the
        # tangent transform — x ~ HalfCauchy(0, s) is x = s*tan(u), u ~ Uniform(0,
        # pi/2) — so NUTS traverses a bounded uniform instead of the heavy-tailed
        # funnel that otherwise pins the step size tiny. The slab (lam_t2) bounds
        # the effective scale, so a large tan() stays finite in the coefficient.
        lam_u = pm.Uniform("gate_lam_u", 0.0, np.pi / 2.0, dims=("bench", "latent"))
        tau_u = pm.Uniform("gate_tau_u", 0.0, np.pi / 2.0)
        lam = pt.tan(lam_u)                                    # ~ HalfCauchy(0, 1)
        tau_g = pm.Deterministic("gate_tau", gate_tau0 * pt.tan(tau_u))  # ~ HalfCauchy(0, tau0)
        c2 = pm.InverseGamma("gate_c2", alpha=RH_SLAB_DF / 2.0,
                             beta=RH_SLAB_DF * RH_SLAB_SCALE**2 / 2.0)
        lam_t2 = c2 * lam**2 / (c2 + tau_g**2 * lam**2)        # slab-regularized local
        g_z = pm.HalfNormal("gate_z", 1.0, dims=("bench", "latent"))
        g_free = g_z * tau_g * pt.sqrt(lam_t2)                 # (B, K) >= 0, sparse
        g = pm.Deterministic("g", ag + fm * g_free, dims=("bench", "latent"))

        # Conjunctive product with a = 1: mu = prod_k sigmoid(theta+c)^g.
        z = theta[data.model_idx] + c[data.bench_idx]          # (n_obs, K)
        log_terms = -pt.softplus(-z)                           # log sigmoid(z), stable
        log_mu = (g[data.bench_idx] * log_terms).sum(axis=-1)  # (n_obs,)
        mu_n = pt.clip(pt.exp(log_mu), ECI_EPS, 1.0 - ECI_EPS)

        phi_n = phi[data.bench_idx]
        clipped = np.clip(data.scores, ECI_EPS, 1.0 - ECI_EPS)
        pm.Beta("obs", alpha=mu_n * phi_n, beta=(1.0 - mu_n) * phi_n, observed=clipped)

    return model
