"""Compensatory K-axis 2PL Beta-IRT (MIRT).

eta = sum_k A[b,k]*theta[m,k] - D[b], mu = sigmoid(eta), score ~ Beta(mu*phi, (1-mu)*phi).
Loading prior: "normal" (non-negative A, single shared scale, composes with
hard anchors / Q-matrix), "signed" (signed-free, rotation resolved per draw in
post-processing), or "bifactor" (dense non-negative general column +
non-negative horseshoe specifics). Convergence judged on identified quantities only (eta, D, sigma_b).
"""
from __future__ import annotations

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from multiaxis_eci.config import (
    ECI_EPS,
    PRIOR_ALPHA,
    PRIOR_DELTA_HUMAN,
    PRIOR_LINEAGE_DELTA,
    PRIOR_LINEAGE_DELTA_BM,
    PRIOR_LINEAGE_DRIFT,
    PRIOR_LINEAGE_DRIFT_BM,
    PRIOR_LINEAGE_OFFSET,
    PRIOR_SIGMA_B,
    PRIOR_SIGMA_B_POOLED,
    PRIOR_TAU_ALPHA,
    PRIOR_TAU_CD,
    PRIOR_THETA_T_NU,
    PRIOR_TIME_BETA,
    RH_SLAB_DF,
    RH_SLAB_SCALE,
    RH_TAU_SCALE,
)
from multiaxis_eci.data import ECIData
from multiaxis_eci.lineage import LineageStructure


def _theta_t_cells(n_rows: int, K: int, nu: float = PRIOR_THETA_T_NU):
    """Per-CELL Student-t(nu) exchangeable theta block, drawn directly and
    re-centered per axis. Every (model, axis) cell is marginally t(nu) at unit
    scale, and each axis's column sums to exactly zero.

    DIRECT t density, not a scale mixture. The mixture form (theta = lambda*z,
    lambda^-2 ~ Gamma(nu/2, nu/2)) has the same marginal but adds n_rows*K
    latent scales, each forming a funnel with its coordinate — the geometry
    NUTS diverges on. The closed-form t density has no latent to funnel
    against; the heavy tail is the only geometric cost left.

    Per CELL, never per ROW. A single scale per model leaves the block
    ELLIPTICAL, and a rotation of an elliptical distribution is the same
    distribution — the rotation orbit stays exactly as flat as under the
    Gaussian. Independent non-Gaussian cells instead pin the mixing matrix up
    to column permutation and sign (Comon 1994), which is the identification
    channel this flag exists for.

    RE-CENTERED per axis because theta reaches the likelihood only through
    A theta - D: a constant shift c of column k is absorbed exactly by
    D_b += A_bk c, a flat direction the ZeroSumNormal removes for the Gaussian
    block. Subtracting the column mean restores the exact pin here. The raw
    cells' common-shift direction is invisible to the likelihood but the t
    prior tames it, so nothing is improper. Centering is linear, so the
    columns stay independent and leptokurtic (the mean of n_rows t-draws is
    tiny against their own spread) — the identification channel survives.
    """
    t = pm.StudentT("theta_t_z", nu=nu, mu=0.0, sigma=1.0, shape=(n_rows, K))
    return t - t.mean(axis=0, keepdims=True)


def _assemble_theta(n_models: int, K: int, human_struct=None,
                    lin: LineageStructure | None = None,
                    lineage_bm: bool = False,
                    variant_offsets: bool = True,
                    shared_base_zsn: bool = True,
                    time_t=None,
                    theta_t_cells: bool = False):
    """Stitch the structured theta blocks and the unstructured rows into ONE
    ZeroSumNormal.

    Human tiers and lineage chains are each a base level plus increments along
    a path. Those bases (human roots, chain founders) sit INSIDE the same
    per-axis ZeroSumNormal as the unstructured models: every starting point is
    drawn from one population, and the sum-to-zero pins the overall location,
    which is otherwise free (theta reaches the likelihood only through
    A·theta − D). A ZeroSumNormal entry has sd sqrt(1 − 1/n) ~ 1 at these
    sizes, i.e. the unit scale a private Normal(0, 1) base would give, minus
    the free level. With no structured blocks this is the plain
    single-ZeroSumNormal path over every model.

    shared_base_zsn=False: the ZeroSumNormal spans only the unstructured rows
    and each base is a private Normal(0, 1). Same marginal scale per base; the
    sum-to-zero then constrains a strictly smaller set, so the single location
    pin covers less of the population, and that share moves with the data scope.
    Both settings give the same free-parameter count and the same initial-point
    log-probability, so no golden-logp lock distinguishes them.

    time_t: per-model centered release year (data.release_time_covariate). Adds
    a per-axis linear trend to the theta prior MEAN,

        theta[m, k] = time_beta[k] * time_t[m] + (the blocks above)

    so an ability is shrunk toward its ERA's level instead of the whole
    population's. That matters only where the data are thin: a well-measured
    model's likelihood swamps its prior mean, while a 2021 model evaluated only
    on easy benchmarks currently has its hard-axis ability extrapolated toward
    the mid-era average. The slope is learned and centered at zero, so a flat
    population reduces this to the plain exchangeable prior — no trend is
    imposed. Signed, because the knowledge axis falls over time on this data;
    a positivity constraint would fight it. Kept separate from the lineage
    drift on purpose: that one is within-chain climb, this one is the
    cross-population level, and tying them would bias both.

    Linear is the restriction here, not the trend itself. It enters the prior
    MEAN only, so a curved truth costs a small prior penalty on measured models
    and misplaces only the thin ones; diagnose it by plotting theta residuals
    against release date, and upgrade the covariate/slope construction in place
    (e.g. piecewise with a fixed knot) rather than adding a second flag.

    theta_t_cells: give the exchangeable block per-cell Student-t tails instead
    of Gaussian ones (see _theta_t_cells; it replaces the ZeroSumNormal for
    this block, with the zero-sum pin restored by re-centering). It covers the
    unstructured rows AND the human-root / chain-founder slices, i.e. exactly
    the block whose Gaussian isotropy makes the rotation orbit exactly flat.
    The structured INCREMENTS (HalfNormal human steps, positive-mean lineage
    steps) are left alone: they are already non-Gaussian and already
    orientation-bearing, so heavier tails there add nothing to identification.
    """
    human_rows = human_struct[0] if human_struct is not None else np.empty(0, dtype=int)
    lin_rows = lin.row_idx if lin is not None else np.empty(0, dtype=int)
    claimed = np.concatenate([human_rows, lin_rows]).astype(int)
    if len(np.unique(claimed)) != len(claimed):
        raise ValueError("structured-theta blocks overlap: a model is claimed by "
                         "two priors (human tier and lineage chain must be disjoint).")
    if time_t is None:
        shift = None
    else:
        time_t = np.asarray(time_t, dtype=float)
        if time_t.shape != (n_models,):
            raise ValueError(f"time_t must be one centered year per model row, "
                             f"got shape {time_t.shape} for {n_models} models.")
        shift = time_t[:, None] * pm.Normal("time_beta", 0.0, PRIOR_TIME_BETA,
                                            shape=(K,))

    def finish(theta_mk):
        """Both assembly paths end here so the trend is added exactly once."""
        return pm.Deterministic("theta",
                                theta_mk if shift is None else theta_mk + shift,
                                dims=("model", "latent"))

    if claimed.size == 0:
        if theta_t_cells:
            return finish(_theta_t_cells(n_models, K))
        theta_t = pm.ZeroSumNormal("theta_t", sigma=1.0, dims=("latent", "model"))
        return finish(theta_t.T)

    rest = np.setdiff1d(np.arange(n_models), claimed)
    n_roots = human_struct[3] if human_struct is not None else 0
    n_chains = lin.n_chains if lin is not None else 0
    if not shared_base_zsn and rest.size == 0:
        raise ValueError("shared_base_zsn=False needs unstructured rows to carry "
                         "the ZeroSumNormal location pin; every model is structured.")
    n_shared = rest.size + (n_roots + n_chains if shared_base_zsn else 0)
    zsn = (_theta_t_cells(n_shared, K) if theta_t_cells
           else pm.ZeroSumNormal("theta_zsn_t", sigma=1.0, shape=(K, n_shared)).T)
    theta_full = pt.zeros((n_models, K))
    theta_full = pt.set_subtensor(theta_full[rest, :], zsn[:rest.size])
    cut = rest.size
    if human_struct is not None:
        base_h = (pm.Deterministic("theta_h_base", zsn[cut:cut + n_roots])
                  if shared_base_zsn else
                  pm.Normal("theta_h_base", 0.0, 1.0, shape=(n_roots, K)))
        cut += n_roots
        rows_h, theta_h = _human_block(human_struct, K, base_h)
        theta_full = pt.set_subtensor(theta_full[rows_h, :], theta_h)
    if lin is not None:
        base_l = (pm.Deterministic("lin_base", zsn[cut:cut + n_chains])
                  if shared_base_zsn else
                  pm.Normal("lin_base", 0.0, 1.0, shape=(n_chains, K)))
        rows_l, theta_l = _lineage_block(lin, K, base_l, bm=lineage_bm,
                                         variant_offsets=variant_offsets)
        theta_full = pt.set_subtensor(theta_full[rows_l, :], theta_l)
    return finish(theta_full)


def _human_structure(human_order, model_names):
    """Resolve the human partial order into 0/1 path matrices, or None if <2
    tiers are in the data.

    `human_order` maps tier → parent tier (None = root; a TUPLE of parents
    means the tier dominates every one of them; a plain list is accepted as the
    legacy single chain, weakest → strongest). Tiers absent from the data are
    skipped and their children re-attach to the nearest present ancestor(s).
    Parents may be declared in any order.

    Returns (rows, R, P, n_roots, groups) with one matrix ROW per directed
    root→tier PATH. Along a path theta is base[its root] plus the increment of
    every tier on it; a tier's theta is the MAX over its own paths, i.e. over
    the rows groups[i]. Unrolling the DAG into paths is exact because
    max(a, b) + delta = max(a + delta, b + delta), so one increment per TIER
    covers a multi-parent tier too and no per-edge parameter is needed. A
    single-parent tree gives exactly one path per tier and the max drops out.
    """
    if not human_order:
        return None
    if not isinstance(human_order, dict):                # legacy chain
        human_order = {t: (None if i == 0 else human_order[i - 1])
                       for i, t in enumerate(human_order)}
    declared = {t: () if p is None else (p,) if isinstance(p, str) else tuple(p)
                for t, p in human_order.items()}
    present = [g for g in human_order if g in model_names]
    if len(present) < 2:
        return None

    def nearest_present(tier, hops=0):
        """the tier's parents, each replaced by its nearest present ancestor"""
        if hops > len(declared):
            raise ValueError("cycle in human_order parent map")
        out = []
        for p in declared.get(tier, ()):
            for a in ([p] if p in present else nearest_present(p, hops + 1)):
                if a not in out:
                    out.append(a)
        return out

    parents = {g: nearest_present(g) for g in present}
    roots = [g for g in present if not parents[g]]
    children = [g for g in present if parents[g]]      # one increment each
    paths: dict[str, list] = {}

    def enumerate_paths(tier, hops=0):
        """[(root, the tiers above that root on the path)], memoized"""
        if tier not in paths:
            if hops > len(present):
                raise ValueError("cycle in human_order parent map")
            paths[tier] = [(tier, ())] if not parents[tier] else [
                (root, above + (tier,)) for p in parents[tier]
                for root, above in enumerate_paths(p, hops + 1)]
        return paths[tier]

    name_to_idx = {m: i for i, m in enumerate(model_names)}
    rows = np.array([name_to_idx[g] for g in present])

    # 0/1 path matrices. R[j,r]=1 iff path j starts at root r;
    # P[j,c]=1 iff children[c]'s increment lies on path j.
    flat = [(i, root, above) for i, g in enumerate(present)
            for root, above in enumerate_paths(g)]
    R = np.zeros((len(flat), len(roots)))
    P = np.zeros((len(flat), len(children)))
    for j, (_, root, above) in enumerate(flat):
        R[j, roots.index(root)] = 1.0
        for tier in above:
            P[j, children.index(tier)] = 1.0
    groups = [np.array([j for j, (i, _, _) in enumerate(flat) if i == tier])
              for tier in range(len(present))]
    return rows, R, P, len(roots), groups


def _human_block(struct, K, base):
    """Partially ordered human tiers: each tier = its root's `base` level plus
    a HalfNormal increment for every tier on the root→tier path (HARD — child
    >= parent on every axis). Where a tier has SEVERAL parents its theta is the
    MAX over its paths, so it dominates all of them. Tiers on different
    branches share no path, so the prior says nothing about their relative
    strength. `base` is the per-root level, sliced out of the shared
    ZeroSumNormal. Returns (row_idx, theta).

    The max costs smoothness: the log-posterior stays continuous but its
    gradient jumps across the surface where two parents of a merged tier are
    equal, which the leapfrog integrator reads as energy error. A merged order
    can therefore diverge where the single-parent tree does not, so read the
    divergence count against the tree fit rather than on its own.

    (Reverted 2026-07-05 to the ORIGINAL hard prior for the hypothesis-testing
    matrix: the soft/pooled/Student-t variants each moved the Skilled
    Generalist instability around without removing it — see the arc-mode
    procedural log. The original is the reference condition.)"""
    rows, R, P, _, groups = struct
    delta = pm.HalfNormal("delta_h", sigma=PRIOR_DELTA_HUMAN,
                          shape=(P.shape[1], K))
    by_path = pt.dot(pt.as_tensor(R), base) + pt.dot(pt.as_tensor(P), delta)
    if all(len(g) == 1 for g in groups):     # single-parent tree: no max at all
        return rows, by_path
    return rows, pt.stack([by_path[g[0]] if len(g) == 1
                           else pt.max(by_path[g], axis=0) for g in groups])


def _lineage_block(lin: LineageStructure, K, base, bm: bool = False,
                   variant_offsets: bool = True):
    """Release chains: psi[node] = founder + cumulative increments; each row's
    theta = psi[node] + a tight mean-zero variant offset. Vectorized (no
    per-chain loop) via the precomputed incidence matrices. `base` is the
    per-chain founder level, sliced out of the shared ZeroSumNormal.

    Increments are SOFT: Normal(mu > 0, s), so improvement is the mean step but
    a node can regress. Steps are indexed by RELEASE, so `mu` is a gain per
    release and one shared value per axis is all the data can carry (median
    live chain has 2 steps).

    bm=True indexes the chain by TIME instead: over a gap of dt years the step
    is Normal(drift*dt, s^2*dt) — a Brownian motion with drift sampled at the
    release dates. Mean grows in dt and sd in sqrt(dt) (variances add over
    time), so a long gap licenses a proportionally larger climb with
    proportionally more uncertainty, and the prior stops depending on how many
    intermediate releases the map happens to list (two half-steps compose to
    one whole step). With dt in years the drift is logits/year, which IS
    comparable across vendors shipping at different cadences — a per-release
    gain is not, since it equals rate/cadence and no clock separates the two.
    The rate is ONE shared drift per axis, same structure as the iid branch:
    a per-family pooled rate was fitted 2026-07-27 and shrank all 36 vendors to
    a single value (sd across vendors 0.015-0.02) at 5.9x the runtime plus a
    stray chain in an inferior basin. Per-vendor realized rates stay readable
    post-hoc off the trace as (psi_last - psi_founder)/span.

    Either way the variant offsets stay mean-zero: effort variants are never
    ordered, and each (node, variant) offset is its own draw, sharing ONE scalar
    sd tau_o across axes. Per-axis tau_o[k] is not identified here: offsets
    reach the likelihood as sum_k A[b,k]*offset[g,k], and two loading columns of
    this fit correlate at ~0.48, so mass slides between those axes almost
    unobservably. A single scale forbids the slide (tau_o ESS 4747, r-hat 1.001);
    per-axis scales license it (ESS 12, r-hat 1.86, and 336 divergences / 16000
    without the ZeroSumNormal damping the compensating founder movement). Only a
    Q-matrix, whose loading columns have disjoint support, leaves nothing to
    slide along. Returns (row_idx, theta)."""
    delta_z = pm.Normal("lin_delta_z", 0.0, 1.0, shape=(lin.n_deltas, K))
    # Same two scales either way; bm only changes their units (per year rather
    # than per release) and lets dt enter deterministically.
    drift_s, spread_s = ((PRIOR_LINEAGE_DRIFT_BM, PRIOR_LINEAGE_DELTA_BM) if bm
                         else (PRIOR_LINEAGE_DRIFT, PRIOR_LINEAGE_DELTA))
    drift = pm.HalfNormal("lin_drift", sigma=drift_s, shape=K)
    spread = pm.HalfNormal("lin_spread", sigma=spread_s, shape=K)
    if bm:
        dt = pt.as_tensor(lin.delta_dt)[:, None]                        # (n_deltas, 1)
        delta = (drift[None, :] * dt
                 + spread[None, :] * pt.sqrt(dt) * delta_z)             # (n_deltas, K)
    else:
        delta = drift[None, :] + spread[None, :] * delta_z              # (n_deltas, K)
    psi = pt.dot(pt.as_tensor(lin.B), base) + pt.dot(pt.as_tensor(lin.C), delta)
    if not variant_offsets:
        # Evaluation settings were merged upstream, so every node carries one
        # row and offset_group is a relabelling of node_idx: psi[node] and
        # offset[group] would be the same quantity twice, identified only in
        # their sum. Drop the offset rather than sample a redundant funnel.
        return lin.row_idx, psi[lin.node_idx]                           # (R, K)
    tau_o = pm.HalfNormal("lin_offset_sd", sigma=PRIOR_LINEAGE_OFFSET)
    offset = tau_o * pm.Normal("lin_offset_z", 0.0, 1.0, shape=(lin.n_groups, K))
    theta_lin = psi[lin.node_idx] + offset[lin.offset_group]            # (R, K)
    return lin.row_idx, theta_lin


def _apply_plt(A, plt_idx, K):
    """Impose the positive-lower-triangular pattern on the founder rows.

    plt_idx is the ordered list of founder row indices (founder r ↔ axis r);
    None is a no-op so callers can apply this unconditionally. Zeros above the
    diagonal remove rotation freedom; |.| on the diagonal removes sign flips
    (a signed Normal cell folded by |.| IS a HalfNormal — no new variables).
    The zeroed cells' underlying z's become prior-only unit Gaussians: benign.
    """
    if plt_idx is None:
        return A
    for r, fi in enumerate(plt_idx):
        if r + 1 < K:
            A = pt.set_subtensor(A[fi, r + 1:], 0.0)
        A = pt.set_subtensor(A[fi, r], pt.abs(A[fi, r]))
    return A


def build_mirt_model(data: ECIData, K: int,
                      anchors: dict | None = None,
                      loading_prior: str = "normal",
                      human_order: dict[str, str | None] | list[str] | None = None,
                      lineage: LineageStructure | None = None,
                      lineage_bm: bool = False,
                      variant_offsets: bool = True,
                      pin_benchmark: str | None = None,
                      plt_founders: list[str] | None = None,
                      floor_c: np.ndarray | None = None,
                      ceiling_noise: bool = False,
                      known_se: bool = False,
                      pooled_noise: bool = False,
                      shared_base_zsn: bool = True,
                      time_t: np.ndarray | None = None,
                      theta_t_cells: bool = False,
                      theta_pos: bool = False,
                      link: str = "linear") -> pm.Model:
    """Build the K-factor compensatory Beta-MIRT model.

    loading_prior: "normal" (non-negative loadings, single shared tau, composes
    with hard anchors — the confirmatory choice), "pt1" (non-negative loadings
    under the product-to-one constraint — see below), "signed" (signed-free
    loadings, identified post-hoc — see below), or "bifactor" (one dense
    general column plus sparse specifics — see below).
    K=1 reproduces the 1D Beta IRT.

    "normal": A[b,k] = HalfNormal(1) * tau_A, tau_A ~ LogNormal(log 0.5, 0.5)
    — non-negative cells, ONE shared scalar scale across axes (no per-axis
    selection), non-centered (a unit-scale z multiplied by tau_A). Loadings
    therefore cannot go negative; each axis is a "good-at-these" bundle.
    Median tau_A 0.5 with HalfNormal(1) cells gives typical loadings ~0.4.

    "pt1": log A[.,k] is a ZeroSumNormal over benchmarks, so each axis's
    loadings have product exactly 1. This is the identification Epoch's public
    ECI uses (sum-to-zero on log alpha), generalized to K axes.

    It removes a real ridge that "normal" leaves open. eta has K exact
    multiplicative degeneracies, one per axis: A[:,k] -> c*A[:,k] with
    theta[:,k] -> theta[:,k]/c leaves eta unchanged AND leaves theta's
    sum-to-zero intact, so theta's zero-sum constraint does not touch them —
    under "normal" they are broken only softly, by theta's prior sitting at
    unit scale against a free tau_A. pt1 breaks all K hard, and therefore has
    NO free loading scale: a tau_A would reinstate one ridge per axis. pt1 sits
    exactly K free dimensions below "normal".

    The gauge value 1 is arbitrary and means nothing; realized axis strength
    lives in theta's per-axis spread, which stays free (its ZeroSumNormal is a
    prior, not a hard constraint). sigma_A is the log-space spread of the
    loadings, the only scale the block needs.

    Costs: loadings are lognormal rather than half-normal, so no benchmark can
    sit at zero loading on an axis it does not measure, and hard anchors are
    unavailable (exp is never exactly 0).

    "signed": A[b,k] = Normal(0,1) * shared scalar tau — no sign constraint,
    no ordering, no orthonormality. The prior is exactly rotation-invariant,
    so the sampler wanders the rotation/sign/permutation orbit freely (raw
    r-hat on A/theta is meaningless by construction); the axes are identified
    AFTER sampling, one draw at a time (analysis.align_rotations — per-draw
    varimax/Procrustes plus sign-permutation matching). This is the only
    exploratory prior that can represent CONTRAST axes (some benchmarks +,
    others −): a contrast forced into non-negative coordinates splits into
    two highly correlated positive axes. Mutually exclusive with anchors:
    hard zeros would break the exact rotation symmetry the post-hoc
    alignment relies on.

    "bifactor": axis 1 is a GENERAL column every benchmark may load, axes 2..K
    are SPECIFICS. Both blocks are non-negative; what separates them is the
    prior's shape, not any benchmark-to-axis assignment. The general column is
    dense (HalfNormal cells × its own LogNormal scale, exactly the "normal"
    block on one column); each specific cell carries a regularized-horseshoe
    local scale, so a cell is either squeezed to ~0 or escapes to roughly
    unshrunk, at a near-constant prior cost per escape. Among the loading
    configurations the likelihood cannot tell apart, total cost is then
    proportional to the NUMBER of escaped cells: shared variance parked in a
    specific column would have to buy an escape on nearly every benchmark,
    while the same variance costs nothing in the exempt general column. So the
    posterior concentrates on "common part in g, few large loadings per
    specific" — the bi-quartimin simplicity criterion of exploratory bifactor
    analysis, moved from a post-hoc rotation into the prior where the sampler
    feels it. WHICH cells escape is left entirely to the likelihood, and a
    specific the data does not need collapses to an empty column (the built-in
    dimensionality check). Axis identity is structural (g is column 1 by
    construction), so no rotation is applied downstream; only the cross-chain
    permutation of the specifics remains. Mutually exclusive with anchors, and
    needs K >= 2 (K=1 is the general column alone, i.e. "normal").

    human_order: optional PARTIAL order on the human-tier model names — a map
    tier → parent tier (None = root), e.g. config.HUMAN_ORDER. On every axis a
    tier's theta = its parent's + a HalfNormal increment (HARD monotone along
    every parent chain; branch tiers stay incomparable) instead of a
    ZeroSumNormal draw. A plain list is accepted as a single chain (legacy). A
    TUPLE of parents (config.HUMAN_ORDER_MERGED) makes the tier dominate all of
    them, theta = max(parents) + the increment; see _human_block for the kink
    that buys.

    lineage: optional release-chain structure (see lineage.py). Chained models get
    psi[node] + variant offset, where psi follows founder + cumulative SOFT
    (Normal, can regress) increments along the chain. Disjoint from human_order.
    lineage_bm=True scales each increment by the release gap dt (in years) —
    Normal(drift*dt, s^2*dt), a Brownian motion with drift observed at the
    release dates — under ONE shared per-axis drift rate (a per-vendor pooled
    rate was fitted 2026-07-27 and removed: roughly one release per year per
    vendor cannot pin a rate; see _lineage_block). The mean-zero effort-variant
    offsets are unaffected.

    Both are priors on theta only, and both compose with any loading_prior. The
    human roots and chain founders join the unstructured models inside one
    ZeroSumNormal, which pins the overall location.

    time_t: optional per-model centered release year
    (data.release_time_covariate) — adds a learned per-axis linear trend in time
    to the theta prior MEAN, so thinly-evaluated models are shrunk toward their
    era's level rather than the whole population's. Prior-mean information only;
    see _assemble_theta for the centering and chain-founder rules.

    theta_t_cells: replace the Gaussian exchangeable theta block with per-CELL
    iid Student-t(nu = config.PRIOR_THETA_T_NU) draws, re-centered per axis so
    each column sums to exactly zero (the location pin the ZeroSumNormal gives
    the Gaussian block). This is the third identification channel for the
    rotation, next to loading-side sparsity and theta-side structure. The
    Gaussian block is spherical, so with a signed loading prior the whole
    rotation orbit has identical prior AND likelihood; independent leptokurtic
    columns do not, and the likelihood can then prefer the orientation in which
    the ability columns are independent (ICA: Comon 1994; Bonhomme & Robin
    2009; the leptokurtic condition varimax implicitly tests, Rohe & Zeng
    arXiv:2004.05387). The t density is closed-form — no latent per-cell
    scales, no scale-coordinate funnels. Applies to the unstructured rows plus
    the human-root/chain-founder slices only, never to the ordered increments —
    see _theta_t_cells for why a per-ROW scale would buy nothing.

    Read the coverage caveat with it: a heavy-tailed prior lets a thinly-measured
    model's ability wander further than a Gaussian one does, so the extrapolated
    abilities the coverage mask already hides get wider, not narrower.

    theta_pos: the likelihood reads a POSITIVE ability, exactly the
    semi-compensatory convention (models/mirt_interaction.py): raw theta keeps
    every prior block — the ZeroSumNormal location pin, the human/lineage order
    structure — and stays the reported ability in the trace, while
    theta_pos = softplus(theta) > 0 is what enters eta = sum_k A * theta_pos - D.
    With A >= 0 every axis then CONTRIBUTES rather than compensates: an ability
    can raise a score above the sigmoid(-D_b) baseline but never drag it below,
    and theta -> -inf compresses to "adds nothing" instead of demanding failure.
    softplus is monotone, so every raw-space order constraint (human tiers,
    lineage steps) survives on the likelihood scale, and rankings read off raw
    theta unchanged. The location becomes likelihood-identified (a constant
    theta shift is no longer absorbed by D through the nonlinearity); the
    sum-to-zero pin stays as a prior centering. With the signed family the
    elementwise nonlinearity breaks the exact rotation/reflection invariance
    that per-draw post-hoc alignment quotients out, so alignment degrades to
    permutation matching — the same caveat theta_t_cells carries, warned below.

    link : str
        "linear" (default) is the compensatory 2PL: eta = A.theta - D.
        "loglog" is the log-logistic IRF mu = 1/(1+(theta.A)^(-alpha_b)),
        i.e. eta = alpha_b * log(sum_k A_bk exp(theta_mk)) with theta, A and
        alpha_b positive. Raw theta keeps every prior block (the ZeroSumNormal
        pin becomes geometric mean 1 per axis; human/lineage/time structure
        survives because exp is monotone). log(sum) is a smooth max
        (within log K of the best axis), so the family is disjunctive - the
        complement of the conjunctive product link in mirt_nc.py. At K=1 it
        is the current 2PL reparameterized with alpha_b as discrimination.
        Difficulty is A's row scale, stored as the Deterministic
        D = -alpha * (row mean of log A); no D is sampled. Requires
        loading_prior="normal"; rejects theta_pos, anchors, plt_founders,
        pin_benchmark. Floors, known_se and pooled_noise compose
        unchanged (they act on mu and phi after the link).

    shared_base_zsn: whether those bases join that ZeroSumNormal (default) or
    each get a private Normal(0, 1) with the sum-to-zero spanning only the
    unstructured rows. Same marginal scale either way; the difference is how
    much of the population the single location pin covers, and that share moves
    with the data scope.

    pin_benchmark: optional anchor-item identification — fix this benchmark's
    difficulty D to exactly 0 ("sea level"). Needed because the human/lineage
    theta priors let structured abilities float, which un-pins the frontier
    benchmarks' D as a rigid block; pinning one well-covered block benchmark
    (e.g. "GPQA Diamond") re-anchors the difficulty scale. Invariant to
    differences/rankings/axes — it only removes the sliding-zero DoF.

    plt_founders: optional positive-lower-triangular identification (Geweke &
    Zhou 1996; Lopes & West 2004) — the standard literature fix for rotation
    indeterminacy. An ORDERED list of exactly K benchmark names; founder r
    (0-based) gets its loading row constrained to the triangular pattern:
      A[founder_r, k] = 0        for k > r   (hard zeros above the diagonal),
      A[founder_r, r] > 0                    (positive diagonal, via |.|),
      A[founder_r, k] free       for k < r   (signed, prior unchanged).
    Together the K rows kill rotation, sign flips, and axis permutation in one
    move — axes are identified IN the sampler, no post-hoc alignment needed.
    Implemented by folding the sign of the existing cell (|A| on the diagonal
    ⇒ HalfNormal-shaped prior) and zeroing above-diagonal cells (their z's
    become benign prior-only Gaussians) — no new random variables. Only for
    the signed family ("signed"). Known literature caveat: the
    result depends on the founder choice — the list is part of the model
    specification, pick rows whose unanchored profiles already match the
    pattern (near-zero where the constraint puts zeros).

    floor_c: optional fixed-c 3PL lower asymptote — a per-benchmark chance floor
    (shape (n_benchmarks,), blookup order, from data.load_benchmark_floors). The
    mean becomes mu = c_b + (1 - c_b) * sigmoid(eta) instead of sigmoid(eta), so
    a random guesser lands at chance (c_b) rather than 0. c is FIXED from the
    known chance rate, not estimated — this sidesteps the notorious weak
    identification of the estimated-c 3PL and adds no new parameters. Intended
    with clip-to-floor observed scores (data.clip_scores_to_floors): together an
    at-chance score reads as uninformative-low ability rather than a point
    demand. Applies only to this compensatory family; the interaction model
    deliberately keeps no floor (below-chance is signal there), and the
    nc/sparse product links have no single eta to floor.

    ceiling_noise: estimate a per-benchmark upper asymptote confined to a
    noise-sized gap (label errors, judge disagreement). The mean becomes
    mu = c_b + (d_b - c_b) * sigmoid(eta) with d_b = 1 - delta_b * (1 - c_b)
    and delta_b ~ Beta(1, 20) (mean 0.048, mode 0, P(gap > 0.10) = 0.122):
    "no ceiling unless the data insists". This is the smooth relaxation of a
    spike-slab at d = 1 — a point-mass mixture would give NUTS a two-story
    posterior per benchmark (the islands failure mode), the Beta dial does not.
    d_b is identified only where observations press the ceiling; elsewhere the
    posterior returns the prior, the honest "not measurable yet" answer.
    Diagnostic use: read `ceiling_d` off the trace to see which benchmarks
    demand d < 1. This is the only ceiling the model carries — a FIXED wall
    read from a curated file was the retired --ceilings flag, and a freely
    estimated one (the retired soft_ceiling flag) was weakly identified /
    multimodal on this data, which is why the gap is confined to noise size.

    known_se: split the Beta noise into a KNOWN per-cell part and an estimated
    per-benchmark remainder, using data.n_eff (the effective test length the
    reported harness stderr implies — see data.load_eci_data). The Beta's
    precision is already a test length: Var = mu(1-mu)/(1+phi) is the variance
    of an average of (1+phi) solve/fail tasks, so phi+1 IS "how many tasks'
    worth of stability this score has", and a reported se converts to the same
    unit as n_eff = p(1-p)/se^2. Two independent noise sources add on the
    RELATIVE-variance (1/length) scale,

        1/N_total = 1/n_eff + 4*sigma_b^2,   phi_n = 1/(1/N_total) - 1

    like resistors in parallel: the weaker source dominates the total. A cell
    with no reported stderr has n_eff = inf, contributes 0 to the sum, and is
    left exactly as it is today. What changes under the flag is the MEANING of
    sigma_b: it becomes the EXCESS scatter beyond the instrument (construct
    misfit — the benchmark measuring something the axes do not), not the total
    scatter. Orthogonal to floor_c / ceiling_noise, which move mu: the same
    phi_n is used whichever mean transform is in force.

    pooled_noise: learn the POPULATION the per-benchmark noise scales are drawn
    from instead of fixing it. sigma_b = exp(mu_s + tau_s * z_b) with
    mu_s ~ Normal(log 0.05, 0.5), tau_s ~ HalfNormal(0.5), z_b ~ Normal(0, 1)
    per benchmark. By default each sigma_b is an independent draw from a FIXED
    population, so a benchmark with few observations can park its noise scale
    at whatever value flatters a posterior mode; under partial pooling it has to
    borrow from the population instead, and the shared median pulls it back. The
    hyperprior on mu_s is centered on the fixed prior's median, so the pooled
    model nests the current one — what is added is that the LOCATION is learned
    rather than asserted. Non-centered (z * tau) per the repo convention: it
    breaks the funnel between tau_s and the per-benchmark z. Everything
    downstream reads the "sigma_b" deterministic, so this composes with
    known_se (which then pools the EXCESS noise), floor_c and ceiling_noise.
    """
    if K < 1:
        raise ValueError(f"K must be >= 1, got {K}")

    if loading_prior not in ("normal", "pt1", "signed", "bifactor"):
        raise ValueError(
            f"loading_prior must be 'normal', 'pt1', 'signed', or "
            f"'bifactor', got {loading_prior!r}")
    if loading_prior == "pt1" and anchors:
        raise ValueError(
            "loading_prior='pt1' builds A as exp(...), which is never exactly "
            "0, so a hard-zero anchor cannot be imposed. Use "
            "loading_prior='normal' for anchored fits.")
    if loading_prior == "bifactor":
        if anchors:
            raise ValueError(
                "loading_prior='bifactor' LEARNS which benchmarks form each "
                "specific (the horseshoe decides) and is mutually exclusive "
                "with hard `anchors` (which IMPOSE it). Pass one or the other, "
                "not both.")
        if K < 2:
            raise ValueError(
                f"loading_prior='bifactor' needs K >= 2 (one general column "
                f"plus at least one specific); got K={K}. For a single general "
                f"column use loading_prior='normal'.")
    if loading_prior == "signed" and anchors:
        raise ValueError(
            "loading_prior='signed' relies on the loadings being EXACTLY "
            "rotation-invariant so post-hoc alignment can resolve the axes; "
            "hard-zero anchors break that symmetry. Pass one or the other, "
            "not both.")
    # PLT: validate the founder list and resolve names -> row indices here, so
    # the A-construction below only has to apply the triangular transform.
    plt_idx = None
    if plt_founders is not None:
        if loading_prior != "signed":
            raise ValueError(
                f"plt_founders identifies the rotation of the signed "
                f"exploratory family ('signed'); got "
                f"loading_prior={loading_prior!r}. For hard structure use "
                f"`anchors` instead.")
        if anchors:
            raise ValueError("plt_founders and anchors are mutually exclusive "
                             "— both fix the rotation.")
        if len(plt_founders) != K:
            raise ValueError(f"plt_founders needs exactly K={K} benchmarks "
                             f"(one per axis, ordered), got {len(plt_founders)}")
        if len(set(plt_founders)) != len(plt_founders):
            raise ValueError("plt_founders must be distinct benchmarks")
        bench_names = data.blookup["benchmark"].tolist()
        unknown = sorted(set(plt_founders) - set(bench_names))
        if unknown:
            raise ValueError(f"plt_founders not in data: {unknown}")
        plt_idx = [bench_names.index(b) for b in plt_founders]

    if floor_c is not None:
        floor_c = np.asarray(floor_c, dtype=np.float64)
        if floor_c.shape != (data.n_benchmarks,):
            raise ValueError(
                f"floor_c must have shape ({data.n_benchmarks},), "
                f"got {floor_c.shape}")
        if not np.all(np.isfinite(floor_c)) or not np.all((floor_c >= 0.0) & (floor_c < 1.0)):
            raise ValueError("floor_c values must be finite and in [0, 1)")

    n_eff = None
    if known_se:
        if getattr(data, "n_eff", None) is None:
            raise ValueError(
                "known_se=True needs data.n_eff (one effective test length per "
                "observation, np.inf where the feed reports no stderr); this "
                "ECIData carries none. Build it with data.load_eci_data().")
        n_eff = np.asarray(data.n_eff, dtype=np.float64)
        if n_eff.shape != (data.n_obs,):
            raise ValueError(f"n_eff must have shape ({data.n_obs},), one entry "
                             f"per observation, got {n_eff.shape}")
        if np.any(n_eff <= 0.0):
            raise ValueError("n_eff values must be positive (inf = unknown)")

    if lineage_bm and lineage is None:
        raise ValueError("lineage_bm=True requires a lineage structure")

    if loading_prior == "signed" and (human_order or lineage):
        import warnings
        warnings.warn(
            "loading_prior='signed' with the ordered human/lineage theta blocks: "
            "those blocks are NOT rotation-invariant across axes (their per-axis "
            "increments softly prefer certain orientations), so the rotation "
            "symmetry is only APPROXIMATE and per-draw alignment is an "
            "approximation rather than an exact quotient. Flip side: the "
            "structured priors act as a weak, substantive orientation anchor. "
            "Interpret aligned axes with that caveat.", UserWarning)

    if theta_pos and loading_prior == "signed":
        import warnings
        warnings.warn(
            "theta_pos with a signed loading prior: softplus in the likelihood "
            "is elementwise, so the rotation/reflection invariance per-draw "
            "post-hoc alignment relies on no longer holds — the likelihood "
            "itself orients the axes (up to permutation). Alignment reduces to "
            "permutation matching; interpret aligned axes with that caveat.",
            UserWarning)

    if link not in ("linear", "loglog"):
        raise ValueError(f"unknown link: {link!r} (use 'linear' or 'loglog')")
    if link == "loglog":
        # The loglog A block replaces the loading prior (log needs A > 0 and
        # structure-free cells); positivity is built in (exp of theta); D is
        # derived from A's row scale, so there is nothing to pin or anchor.
        if loading_prior != "normal":
            raise ValueError("link='loglog' requires loading_prior='normal'")
        if theta_pos:
            raise ValueError("link='loglog' already reads exp(theta); drop theta_pos")
        if anchors:
            raise ValueError("anchors force a loading to exactly 0 (log A = -inf); incompatible with link='loglog'")
        # plt_founders needs no check here: it already requires a signed
        # loading prior (rejected above via loading_prior != "normal").
        if pin_benchmark is not None:
            raise ValueError("link='loglog' derives D from A's row scale; there is no sampled D to pin")

    coords = {
        "model":  data.mlookup["model"].tolist(),
        "bench":  data.blookup["benchmark"].tolist(),
        "latent": [f"axis{k + 1}" for k in range(K)],
    }
    if loading_prior == "bifactor":
        # The sparsity block covers the SPECIFICS only; axis1 (general) is
        # exempt, which is what makes the decomposition a bifactor.
        coords["latent_s"] = [f"axis{k + 1}" for k in range(1, K)]
    with pm.Model(coords=coords) as model:
        if pooled_noise:
            # Partially pooled log noise scale: the population location and
            # spread are learned, so a thin benchmark's sigma_b is pulled toward
            # the shared median instead of being free. Non-centered, and the
            # name/dims stay "sigma_b"/"bench" because every post-fit reader
            # (identified r-hat, PPC, dashboard) indexes the posterior by them.
            mu_s = pm.Normal("sigma_b_mu", PRIOR_SIGMA_B_POOLED["mu_loc"],
                             PRIOR_SIGMA_B_POOLED["mu_sd"])
            tau_s = pm.HalfNormal("sigma_b_tau", PRIOR_SIGMA_B_POOLED["tau_sd"])
            z_s = pm.Normal("sigma_b_z", 0.0, 1.0, dims="bench")
            sigma_b = pm.Deterministic("sigma_b", pt.exp(mu_s + tau_s * z_s),
                                       dims="bench")
        else:
            sigma_b = pm.LogNormal("sigma_b", dims="bench", **PRIOR_SIGMA_B)
        tau_CD = pm.LogNormal("tau_CD", **PRIOR_TAU_CD)

        if link == "loglog":
            # One shared scale for the row-centered log-loading mix
            # (log-ratio units: PRIOR_TAU_ALPHA's median 0.5 reads as typical
            # axis-weight ratios of ~e^+-0.5). Exposed as the (K,) tau_A
            # every downstream consumer expects.
            tau_A_scalar = pm.LogNormal("tau_A_loglog", **PRIOR_TAU_ALPHA)
            tau_A = pm.Deterministic("tau_A", pt.ones(K) * tau_A_scalar, dims="latent")
        elif loading_prior == "normal":
            # Single shared scale for all axes (no per-axis selection).
            tau_A_scalar = pm.LogNormal("tau_A_normal", **PRIOR_TAU_ALPHA)
            tau_A = pm.Deterministic("tau_A", pt.ones(K) * tau_A_scalar, dims="latent")
        elif loading_prior == "pt1":
            # Product-to-one identification (Epoch's public ECI convention,
            # generalized per axis). eta = sum_k A[b,k] theta[m,k] - D[b] has K
            # EXACT multiplicative degeneracies, one per axis: A[:,k] -> c*A[:,k]
            # with theta[:,k] -> theta[:,k]/c leaves eta untouched and leaves
            # theta's sum-to-zero intact, so the theta prior only breaks them
            # softly. Pinning each column's geometric mean to exactly 1 breaks
            # all K hard, which is why there is NO free loading scale here: a
            # tau_A would reinstate one ridge. The gauge value 1 is arbitrary
            # and carries no meaning; realized axis strength lives in theta's
            # per-axis spread, which stays free (its ZeroSumNormal is a prior,
            # not a constraint).
            #
            # sigma_A is the log-space spread of the loadings, the only scale
            # the block needs, and it is SAMPLED -- Barry's tau_alpha is free
            # too (alpha_b ~ LogNormal(0, tau_alpha^2)), so fixing it would
            # depart from the identification this prior exists to reproduce.
            sigma_A = pm.LogNormal("sigma_A", **PRIOR_TAU_ALPHA)
            # Flat unit tau_A so every downstream reader (tau spectrum, axis
            # strength, rank tracking) gets the (K,) vector it expects; the
            # flat-tau path already reads strength off the loading columns.
            tau_A = pm.Deterministic("tau_A", pt.ones(K), dims="latent")
        elif loading_prior == "signed":
            # Shared scalar scale, like "normal" — deliberately NO per-axis tau
            # and NO ordering: either would break the exact rotation invariance
            # that per-draw post-hoc alignment needs (ordering also builds
            # likelihood walls at scale ties).
            tau_A_scalar = pm.LogNormal("tau_A_signed", **PRIOR_TAU_ALPHA)
            tau_A = pm.Deterministic("tau_A", pt.ones(K) * tau_A_scalar, dims="latent")
        elif loading_prior == "bifactor":
            # Two scales, one per block: the general column keeps the "normal"
            # prior's scale, the specifics share ONE horseshoe global scale
            # across all of them (never per-axis — per-axis scales tie, and a
            # tie is a likelihood wall the sampler freezes against; the
            # specifics here are near-tied by construction). tau_A exposes them
            # in axis order so every downstream consumer (rank-tracking, the
            # tau-spectrum CSV, cross-chain reproducibility) reads the same
            # (K,) vector it reads for every other prior.
            tau_g = pm.LogNormal("tau_g", **PRIOR_TAU_ALPHA)
            tau_hs = pm.HalfNormal("tau_hs_bifactor", RH_TAU_SCALE)
            tau_A = pm.Deterministic(
                "tau_A",
                pt.concatenate([pt.stack([tau_g]), pt.ones(K - 1) * tau_hs]),
                dims="latent")
        else:
            # Every prior on the whitelist above has a branch; a new one must
            # add its own rather than inherit whichever branch sits last.
            raise AssertionError(
                f"no tau_A branch for loading_prior={loading_prior!r}")

        # theta: human tiers (hard-ordered) and lineage chains (soft-ordered)
        # are structured blocks whose base levels join the unstructured models
        # in one ZeroSumNormal. _assemble_theta stitches them by row index.
        theta = _assemble_theta(
            data.n_models, K,
            human_struct=_human_structure(human_order, coords["model"]),
            lin=lineage, lineage_bm=lineage_bm,
            variant_offsets=variant_offsets,
            shared_base_zsn=shared_base_zsn, time_t=time_t,
            theta_t_cells=theta_t_cells)
        # Likelihood-side ability. Raw theta stays the reported ability (it
        # carries the location pin and the order structure); the positive copy
        # is what eta reads, so with A >= 0 an axis can only add to a score,
        # never pull it below the sigmoid(-D) baseline. softplus is monotone,
        # so raw-space order constraints survive on the likelihood scale.
        if link == "loglog":
            # exp, not softplus: theta = exp(z) has lognormal marginals and
            # the ZeroSumNormal pin becomes geometric mean 1 per axis. Stored
            # for downstream eta reconstruction (PPC, identified r-hat); the
            # predictor itself reads raw theta inside a logsumexp.
            theta_lik = pm.Deterministic("theta_pos", pt.exp(theta),
                                         dims=("model", "latent"))
        elif theta_pos:
            theta_lik = pm.Deterministic("theta_pos", pt.softplus(theta),
                                         dims=("model", "latent"))
        else:
            theta_lik = theta

        free_mask = np.ones((data.n_benchmarks, K), dtype=float)
        if anchors:
            # CRITICAL: free_mask is indexed by POSITION in the model's `bench`
            # coord, which uses 0-indexed positions. data.blookup["benchmark_idx"]
            # is 1-indexed (starts at 1), so we must NOT use it as an array index.
            # Use enumerate on the same list the model uses for its `bench` coord.
            bench_names = data.blookup["benchmark"].tolist()
            bench_to_pos = {b: i for i, b in enumerate(bench_names)}
            for bench_name, axes in anchors.items():
                if bench_name not in bench_to_pos:
                    raise ValueError(f"anchor benchmark not in data: '{bench_name}'")
                axis_list = [axes] if isinstance(axes, (int, np.integer)) else list(axes)
                for axis_k in axis_list:
                    if not (0 <= axis_k < K):
                        raise ValueError(f"anchor axis {axis_k} out of range for K={K}")
                bi = bench_to_pos[bench_name]
                free_mask[bi, :] = 0.0
                free_mask[bi, axis_list] = 1.0

        if link == "loglog":
            # Free lognormal loadings via an exact log-space split:
            # log A_bk = row mean + row-centered mix. Same K free cells per
            # row as unsplit lognormal cells (1 mean + K-1 centered), so this
            # is a reparameterization, not a constraint. The split exists
            # because the row mean is the difficulty scale (tau_CD units,
            # +-6 logits) while the mix is axis-weight ratios (tau_A units,
            # ~e^+-1); one shared scale would squash one or blow up the other.
            logA_row_z = pm.Normal("logA_row_z", 0.0, 1.0, dims="bench")
            logA_mix_z = pm.ZeroSumNormal("logA_mix_z", dims=("bench", "latent"))
            logA = (logA_row_z * tau_CD)[:, None] + tau_A_scalar * logA_mix_z
            A = pm.Deterministic("A", pt.exp(logA), dims=("bench", "latent"))
            # Discrimination: the slope on log-ability. At K=1 this link is
            # the 2PL reparameterized with alpha in the discrimination seat
            # (typical fitted discriminations ~0.5).
            alpha_z = pm.LogNormal("alpha_z", dims="bench", **PRIOR_ALPHA)
            tau_alpha = pm.LogNormal("tau_alpha", **PRIOR_TAU_ALPHA)
            alpha = pm.Deterministic("alpha", alpha_z * tau_alpha, dims="bench")
        elif loading_prior == "signed":
            # Signed-free: iid Normal cells × shared scale. Spherical in the
            # K-dim row space, so prior and likelihood are both invariant under
            # rotating (A, theta) together — the sampler explores that orbit
            # freely and analysis.align_rotations resolves it per draw. No
            # free_mask (anchors are excluded above), no initval needed.
            A_z = pm.Normal("A_z", 0.0, 1.0, dims=("bench", "latent"))
            A = pm.Deterministic("A", _apply_plt(A_z * tau_A, plt_idx, K),
                                 dims=("bench", "latent"))
        elif loading_prior == "bifactor":
            # Dense general column: the "normal" block on one column, no
            # sparsity, so shared variance is free to sit here.
            g_z = pm.HalfNormal("g_z", sigma=1.0, dims="bench")
            # Specifics: the regularized horseshoe on NON-NEGATIVE cells. The
            # half-Cauchy local scale is what makes each cell two-regime
            # (squeezed to ~0, or escaped to roughly unshrunk); the Inv-Gamma
            # slab soft-caps an escaped loading so no single benchmark can buy
            # an axis with a runaway slope.
            lam = pm.HalfCauchy("lam_hs", 1.0, dims=("bench", "latent_s"))
            c2 = pm.InverseGamma("c2_hs", alpha=RH_SLAB_DF / 2.0,
                                 beta=RH_SLAB_DF * RH_SLAB_SCALE**2 / 2.0)
            lam_t2 = c2 * lam**2 / (c2 + tau_hs**2 * lam**2)
            A_s_z = pm.HalfNormal("A_s_z", sigma=1.0, dims=("bench", "latent_s"))
            A = pm.Deterministic(
                "A",
                pt.concatenate([pt.shape_padright(g_z * tau_g),
                                A_s_z * tau_hs * pt.sqrt(lam_t2)], axis=1),
                dims=("bench", "latent"))
        elif loading_prior == "pt1":
            # Zero-sum runs over the LAST dim, so log A is sampled transposed:
            # each ROW of logA_z is one axis, summing to zero across benchmarks.
            # Non-negativity is automatic (exp), so no free_mask (anchors are
            # rejected above) and no positive-family choice to make.
            logA_z = pm.ZeroSumNormal("logA_z", dims=("latent", "bench"))
            A = pm.Deterministic("A", pt.exp(sigma_A * logA_z).T,
                                 dims=("bench", "latent"))
        else:
            A_z = pm.HalfNormal("A_z", sigma=1.0, dims=("bench", "latent"))
            A_masked = A_z * pt.as_tensor(free_mask)
            A = pm.Deterministic("A", A_masked * tau_A, dims=("bench", "latent"))

        # Difficulty scale. Default: free per-benchmark D_z (location pinned on
        # the theta side by ZeroSumNormal). ANCHOR-ITEM option (pin_benchmark):
        # fix D[pin] == 0 exactly — declares that benchmark the "sea level" of
        # the difficulty scale. This is the standard alternative identification
        # (Reckase 2009; Epoch's WinoGrande pin) used when theta carries
        # substantive structure (lineage/human priors) that lets a subgroup of
        # abilities float — which un-pins the frontier benchmarks' D as a block.
        # Sampling B-1 free difficulties and splicing a fixed 0 at the pin (not
        # recentering a full D_z) avoids leaving a redundant, freely-wandering
        # D_z[pin] in the geometry.
        if link == "loglog":
            # Difficulty read back in logits from A's row scale:
            # eta = alpha*log(theta_pos . A) = alpha*log(theta_pos . u) - D
            # with u = exp(row-centered mix), D = -alpha * (row mean of
            # log A). Exact - a per-row constant cannot enter a zero-sum
            # vector - and it keeps every posterior["D"] reader meaningful.
            D = pm.Deterministic("D", -alpha * logA_row_z * tau_CD, dims="bench")
        else:
            if pin_benchmark is None:
                D_z = pm.Normal("D_z", 0.0, 1.0, dims="bench")
            else:
                bench_names = data.blookup["benchmark"].tolist()   # same list as the `bench` coord
                if pin_benchmark not in bench_names:
                    raise ValueError(f"pin_benchmark not in data: '{pin_benchmark}'")
                pin = bench_names.index(pin_benchmark)              # 0-indexed position (NOT benchmark_idx)
                keep = np.array([i for i in range(data.n_benchmarks) if i != pin])
                D_z_free = pm.Normal("D_z_free", 0.0, 1.0, shape=data.n_benchmarks - 1)
                D_z = pt.set_subtensor(pt.zeros(data.n_benchmarks)[keep], D_z_free)
            D = pm.Deterministic("D", D_z * tau_CD, dims="bench")

        if link == "loglog":
            # Stable predictor: log(sum_k A_bk exp(theta_k)) =
            # logsumexp_k(theta_k + log A_bk), never materializing A or
            # exp(theta). logsumexp is a smooth max (within log K of the
            # best-loaded axis), so the family is disjunctive.
            eta = alpha[data.bench_idx] * pt.logsumexp(
                theta[data.model_idx] + logA[data.bench_idx], axis=-1)
        else:
            A_obs     = A[data.bench_idx]          # (n_obs, K)
            theta_obs = theta_lik[data.model_idx]  # (n_obs, K)
            eta = (A_obs * theta_obs).sum(axis=-1) - D[data.bench_idx]
        mu_n = pm.math.sigmoid(eta)
        if ceiling_noise:
            # Estimated 4PL: the mean saturates at d_b instead of 1. delta is
            # the FRACTION of the (c_b, 1] range left unreachable, so
            # d_b > c_b holds by construction — a free absolute gap could dip
            # below the floor and flip the Beta mean negative.
            #
            # Beta(1, 20): mean 0.048, P(delta > 0.10) = 0.122. Beta(1, beta)
            # is an Exponential folded onto [0, 1]: mode at 0 and log-density
            # near-linear in delta over the small-gap region, so every further
            # point of ceiling costs the same. Constant marginal cost shrinks
            # the many no-ceiling benchmarks toward 0 without fighting the few
            # that genuinely saturate; a quadratic penalty (HalfNormal) does
            # the opposite. Smooth in delta, so no discrete ceiling/no-ceiling
            # mode split.
            #
            # The scale decides identifiability. d reaches the likelihood
            # through log(d_b - c_b), a constant shift of eta that D_b absorbs
            # exactly, so d rides a flat ridge with difficulty and is pinned
            # only by the one signature a ceiling has: frontier scores piling
            # at a common sub-1 level with no ability trend among them, which
            # the 2PL cannot fit because it forces
            # Var = 4 sigma_b^2 mu(1-mu) -> 0 as mu -> 1. The gap is kept
            # small so the mid-scale data still pin A and D as in the 2PL.
            # (The retired soft_ceiling flag was the same asymptote at
            # Beta(1, 19).)
            delta = pm.Beta("ceiling_gap", alpha=1.0, beta=20.0, dims="bench")
            c_full = floor_c if floor_c is not None else np.zeros(data.n_benchmarks)
            d_b = pm.Deterministic("ceiling_d", 1.0 - delta * (1.0 - c_full),
                                   dims="bench")
            c_obs = c_full[data.bench_idx]
            mu_n = c_obs + (d_b[data.bench_idx] - c_obs) * mu_n
        elif floor_c is not None:
            # fixed 3PL: a random guesser lands at chance c_b, not 0. c is a
            # numpy constant, so leaving it at its default folds the link back
            # to the 2PL exactly.
            c_obs = floor_c[data.bench_idx]
            mu_n = c_obs + (1.0 - c_obs) * mu_n

        phi   = pm.Deterministic("phi_b", 1.0 / (4.0 * sigma_b**2) - 1.0, dims="bench")
        if n_eff is None:
            phi_n = phi[data.bench_idx]
        else:
            # Relative variances (1/length) of the two noise sources add. The
            # per-cell instrument part is known, sigma_b carries only the excess.
            # No Deterministic: one entry per observation would dominate the
            # trace. 1/inf == 0 exactly, so an unreported cell reduces to
            # phi_b[b] bit for bit.
            r_n = 1.0 / n_eff + 4.0 * sigma_b[data.bench_idx] ** 2
            phi_n = 1.0 / r_n - 1.0

        a = mu_n * phi_n
        b = (1.0 - mu_n) * phi_n

        clipped = np.clip(data.scores, ECI_EPS, 1.0 - ECI_EPS)
        pm.Beta("obs", alpha=a, beta=b, observed=clipped)

    return model
