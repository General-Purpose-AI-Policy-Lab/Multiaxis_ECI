"""FitView + prepare_fit: the single rotation/identity contract every
figure builder consumes."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from analysis.factors import (
    mirt_factors_from_trace, trace_anchors, trace_axis_names,
    trace_display_rotation, trace_loading_prior,
)
from analysis.rotation import (
    align_rotations, apply_rotation, nonneg_rotate, promax_rotate,
)
from data import ECIData

# ── Unified fit view-model (the single rotation/identity contract) ──────────

@dataclass(frozen=True)
class FitView:
    """Normalized post-processing view of a fitted MIRT trace.

    One structure for every MIRT fit family — compensatory (exploratory,
    anchored) and non-compensatory — so figure builders don't each re-derive
    the rotation/identity decision.

    Always present: `theta` (S, M, K) abilities; `Phi` (K, K) axis correlation to
    DISPLAY (promax-oblique when `rotated`, else the raw ability correlation);
    `Phi_raw` (K, K) the raw ability correlation; `names`, `K`, and the family
    flags. Family-specific fields are None when they don't apply:
      * `A`   — (S, B, K) loadings. None for non-comp (slope fixed = 1).
      * `tau` — (S, K) per-axis scale (tau_A). None for non-comp.

    Consumers MUST branch on the flags (`is_nc` / `anchored`) before touching a
    family-specific field. `require_A()` raises instead of returning None so a
    path that forgets to check a non-comp view fails loudly rather than
    silently mis-indexing.
    """
    theta: np.ndarray
    Phi: np.ndarray
    Phi_raw: np.ndarray
    names: list
    K: int
    is_nc: bool
    anchored: bool
    rotated: bool
    A: np.ndarray | None = None
    tau: np.ndarray | None = None

    def require_A(self) -> np.ndarray:
        if self.A is None:
            raise ValueError(
                "FitView has no loadings A — this is a non-compensatory fit "
                "(slope fixed = 1); its benchmark→axis structure is the Q-matrix.")
        return self.A


def prepare_fit(idata, data: ECIData) -> FitView:
    """Turn any MIRT trace into a normalized `FitView`, applying the correct
    rotation-vs-pinned decision — the single code path every figure consumer
    (dashboard, plot_mirt) goes through.

    Gating:
      * non-comp (`constant_data` carries a `Q`): read `theta` directly, no `A`,
        no rotation; `Phi` = ability correlation.
      * anchored **or** signed-family **or** bifactor **or** K == 1: axes
        already identified (anchors), aligned per draw (signed), or fixed by
        the prior's dense/sparse asymmetry (bifactor) → NO promax on top;
        `Phi = Phi_raw` (ability correlation, or the alignment's own Phi).
      * else (exploratory "normal", K ≥ 2): promax-rotate loadings + abilities.
    """
    is_nc = ("Q" in idata.constant_data) if hasattr(idata, "constant_data") else False
    if is_nc:
        K = int(idata.posterior.sizes["latent"])
        names = trace_axis_names(idata, K)
        theta = idata.posterior["theta"].values.reshape(-1, data.n_models, K)
        Phi = np.corrcoef(theta.mean(0).T) if K > 1 else np.array([[1.0]])
        return FitView(theta=theta, Phi=Phi, Phi_raw=Phi, names=names, K=K,
                       is_nc=True, anchored=False, rotated=False)

    anchored = bool(trace_anchors(idata))
    is_signed = trace_loading_prior(idata) in ("signed", "signedhs")
    # Bifactor: the prior's dense-g / sparse-specifics asymmetry already fixed
    # the orientation, so rotating would mix g back into the specifics and undo
    # the identification. Not `anchored` — no benchmark is pinned to an axis, so
    # the dashboard must not draw a Q-matrix for it.
    is_bifactor = trace_loading_prior(idata) == "bifactor"
    A, theta, tau = mirt_factors_from_trace(idata)
    K = theta.shape[2]
    signed_res = None
    if is_signed and K > 1:
        # Signed-free fit: draws sit in arbitrary orientations — align every
        # draw (rotation + sign + permutation) before ANY summary. PROMAX is
        # the FitView frame (dashboard decision 2026-07-05: one frame only;
        # its target-chasing tilts axes toward the dominant factor, so skill
        # timelines INCLUDE the general rise — the wanted reading);
        # align_mirt.py still compares all four methods offline.
        signed_res = align_rotations(A, theta, method="promax")
        A, theta = signed_res.A, signed_res.theta
    names = trace_axis_names(idata, K)
    Phi_raw = np.corrcoef(theta.mean(0).T) if K > 1 else np.array([[1.0]])
    if signed_res is not None and signed_res.Phi is not None:
        Phi_raw = signed_res.Phi        # promax factor correlation
    if anchored or is_signed or is_bifactor or K == 1:
        Phi, rotated = Phi_raw, False
    else:
        # Exploratory 'normal'-prior fit: promax by default. Trace-recorded
        # overrides: mirt_display_rotation='nonneg' opts into the non-negative
        # frame (varimax criterion + hard non-negativity) so its axes read as
        # pure positive bundles with no contrast; 'none' skips rotation
        # entirely — the raw rank-tracked axes ARE the frame (the positivity
        # constraint pins the rotation; Phi = raw ability correlation).
        disp = trace_display_rotation(idata)
        if disp == "none":
            Phi, rotated = Phi_raw, False
        else:
            rotate = nonneg_rotate if disp == "nonneg" else promax_rotate
            Tl, Tt, Phi = rotate(A.mean(0))
            A, theta, Phi = apply_rotation(A, theta, Phi, Tl, Tt)
            rotated = True
    return FitView(theta=theta, Phi=Phi, Phi_raw=Phi_raw, names=names, K=K,
                   is_nc=False, anchored=anchored, rotated=rotated,
                   A=A, tau=tau)
