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
    CONTINUOUS_EPS,
    ECI_EPS,
    NONCOMP_BETA,
    PRIOR_COMP_BETA,
    PRIOR_COMP_OFFSET_SD,
    EFFORT_TOKEN_LEVEL,
    PRIOR_ALPHA,
    PRIOR_CEILING_GAP,
    PRIOR_DELTA_HUMAN,
    PRIOR_EFFORT_MEAN,
    PRIOR_EFFORT_TOKEN_SLOPE,
    PRIOR_HUMAN_ROOT,
    PRIOR_LINEAGE_DELTA,
    PRIOR_LINEAGE_DELTA_BM,
    PRIOR_LINEAGE_DRIFT,
    PRIOR_LINEAGE_DRIFT_BM,
    PRIOR_LINEAGE_OFFSET,
    PRIOR_LINEAGE_OFFSET_NONE,
    PRIOR_SIGMA_B,
    PRIOR_SIGMA_B_POOLED,
    PRIOR_TAU_ALPHA,
    PRIOR_TAU_CD,
    DEFAULT_PIN_OFFSET,
    PRIOR_SHARE_CONC,
    PRIOR_SHARE_SIZE_SHAPE,
    PRIOR_THETA_GENERAL,
    PRIOR_THETA_SKEW_ALPHA,
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


def _theta_skew_cells(n_rows: int, K: int, alpha_sd: float = PRIOR_THETA_SKEW_ALPHA):
    """Independent skew-normal exchangeable theta block, one learned shape per axis.

    theta[m, k] ~ SkewNormal(xi_k, omega_k, alpha_k), alpha_k ~ Normal(0, alpha_sd), with xi_k and
    omega_k set so each column has mean 0 and sd 1 whatever its shape: the location and scale stay
    those of the Gaussian block, only the asymmetry moves. Per CELL and independent across axes,
    like the t block: an independent non-Gaussian column breaks the rotation invariance of the
    spherical Gaussian block (ICA). Skewed rather than heavy-tailed because that is the abilities'
    shape (a left tail of old and weak models, a ceiling on the right; the sampler study's
    figures/ability_distribution_modeA.png).

    NOT re-centered, unlike the t block: the column mean is pinned softly by the zero-mean prior
    (sd 1/sqrt(n_rows) for the shift D absorbs), which is the version the sampler study tested on the
    flagship (14 of 30 chains above -14395 against 4 of 30, better LOO on every seed)."""
    alpha = pm.Normal("theta_skew_alpha", 0.0, alpha_sd, shape=K)
    delta = alpha / pt.sqrt(1 + alpha ** 2)
    omega = 1.0 / pt.sqrt(1 - 2 * delta ** 2 / np.pi)                 # unit variance
    xi = -omega * delta * np.sqrt(2 / np.pi)                           # zero mean
    return pm.SkewNormal("theta_skew", mu=xi, sigma=omega, alpha=alpha, shape=(n_rows, K))


def _general_loadings(K: int):
    """The higher-order prior's loadings lambda_k in (0, 1), one per axis: every per-axis
    innovation of a test-taker is lambda_k * g + sqrt(1 - lambda_k^2) * u_k, g shared by its axes,
    so two axes correlate a priori at lambda_j * lambda_k (de la Torre & Song 2009). Learned."""
    a, b = PRIOR_THETA_GENERAL
    return pm.Beta("theta_general_lambda", alpha=a, beta=b, dims="latent")


def _general_mix(z, lam):
    """Standard-normal cells z (n, K), iid, mixed to the one-factor correlation
    R = lam lam^T + diag(1 - lam^2): z L^T with L = chol(R), the same distribution as
    lam * g + sqrt(1 - lam^2) * u without a latent g per row."""
    R = pt.outer(lam, lam) + pt.diag(1.0 - lam ** 2)
    return pt.dot(z, pt.linalg.cholesky(R).T)


def _assemble_theta(n_models: int, K: int, human_struct=None,
                    human_delta_sd: float = PRIOR_DELTA_HUMAN,
                    human_root_sd: float = PRIOR_HUMAN_ROOT,
                    lin: LineageStructure | None = None,
                    lineage_bm: bool = False,
                    variant_offsets: bool = True,
                    shared_base_zsn: bool = True,
                    time_t=None,
                    theta_t_cells: bool = False,
                    theta_skew: bool = False,
                    lineage_links: bool = True,
                    effort_means: bool = False,
                    theta_general: bool = False):
    """Stitch the structured theta blocks and the unstructured rows into ONE
    ZeroSumNormal.

    lineage_links=False keeps the lineage map's release nodes and their variant offsets but
    drops the links between releases: every node gets its own base in the ZeroSumNormal
    instead of its chain founder's level plus increments (see _release_block).

    effort_means=True gives those variant offsets a mean per effort level, along one shared
    effort direction (see _variant_offsets), in the chains and the release nodes alike.

    Human tiers and lineage chains are each a base level plus increments along
    a path. The chain founders sit INSIDE the same per-axis ZeroSumNormal as the
    unstructured models: every model starting point is drawn from one population,
    and the sum-to-zero pins the overall location, which is otherwise free (theta
    reaches the likelihood only through A·theta − D). A ZeroSumNormal entry has
    sd sqrt(1 − 1/n) ~ 1 at these sizes, i.e. the unit scale a private Normal(0, 1)
    base would give, minus the free level. With no structured blocks this is the
    plain single-ZeroSumNormal path over every model.

    The human roots are not drawn from the model population: each is a private
    Normal(0, human_root_sd) base (config.PRIOR_HUMAN_ROOT), outside the zero-sum (user,
    2026-10-05). Inside it, Average Human sat 4 to 5 sd from the models on its
    extreme axis of the 2026-10-03 flagship and the unit prior pulled it a third
    of the way back. Humans stay on the models' scale and axes all the same:
    their scores on the shared benchmarks place them.

    shared_base_zsn=False: the ZeroSumNormal spans only the unstructured rows
    and each chain founder is a private Normal(0, 1). Same marginal scale per
    base; the sum-to-zero then constrains a strictly smaller set, so the single
    location pin covers less of the population, and that share moves with the
    data scope.

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

    theta_skew: the same block, independent skew-normal per axis with a learned
    shape instead (see _theta_skew_cells). Exclusive with theta_t_cells.

    theta_general: a higher-order general level under the axes (de la Torre & Song 2009):
    every per-axis innovation, the exchangeable block's cells, the lineage steps and the
    variant offsets alike, is lambda_k * g + sqrt(1 - lambda_k^2) * u_k with one g per
    innovation shared by its K axes, so the axes correlate a priori at lambda_j * lambda_k
    (`_general_loadings`, learned). Under independent axes the compensatory sum explains a
    mixed benchmark's score by whichever axis is already high, and a model strong on one axis
    is pulled down on another (explaining away); the shared level makes a strong axis raise
    the expectation of the others instead. In the skew-normal block g is Gaussian and u the
    skew-normal cells; in the Gaussian blocks the mix is a Cholesky factor (`_general_mix`).
    The marginals keep unit variance.
    """
    if theta_t_cells and theta_skew:
        raise ValueError("theta_t_cells and theta_skew both replace the exchangeable "
                         "theta block; pick one.")
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
        if theta_skew:
            cells = _theta_skew_cells(n_models, K)
            if theta_general:
                lam = _general_loadings(K)
                g = pm.Normal("theta_general_g", 0.0, 1.0, shape=(n_models, 1))
                cells = g * lam[None, :] + pt.sqrt(1.0 - lam ** 2)[None, :] * cells
            return finish(cells)
        theta_t = pm.ZeroSumNormal("theta_t", sigma=1.0, dims=("latent", "model"))
        return finish(theta_t.T)

    rest = np.setdiff1d(np.arange(n_models), claimed)
    n_roots = human_struct[3] if human_struct is not None else 0
    # One base per chain founder, or one per release node when the links are dropped.
    n_chains = (0 if lin is None else lin.n_chains if lineage_links else lin.n_nodes)
    if not shared_base_zsn and rest.size == 0:
        raise ValueError("shared_base_zsn=False needs unstructured rows to carry "
                         "the ZeroSumNormal location pin; every model is structured.")
    n_shared = rest.size + (n_chains if shared_base_zsn else 0)
    lam = _general_loadings(K) if theta_general else None
    if lam is not None and not theta_skew:
        raise ValueError("theta_general is built on the skew-normal block; drop --normal-theta "
                         "and --theta-t with it.")
    zsn = (_theta_t_cells(n_shared, K) if theta_t_cells
           else _theta_skew_cells(n_shared, K) if theta_skew
           else pm.ZeroSumNormal("theta_zsn_t", sigma=1.0, shape=(K, n_shared)).T)
    if lam is not None:
        g = pm.Normal("theta_general_g", 0.0, 1.0, shape=(n_shared, 1))
        zsn = g * lam[None, :] + pt.sqrt(1.0 - lam ** 2)[None, :] * zsn
    theta_full = pt.zeros((n_models, K))
    theta_full = pt.set_subtensor(theta_full[rest, :], zsn[:rest.size])
    cut = rest.size
    if human_struct is not None:
        base_h = pm.Normal("theta_h_base", 0.0, human_root_sd, shape=(n_roots, K))
        rows_h, theta_h = _human_block(human_struct, K, base_h, human_delta_sd)
        theta_full = pt.set_subtensor(theta_full[rows_h, :], theta_h)
    if lin is not None:
        base_l = (pm.Deterministic("lin_base", zsn[cut:cut + n_chains])
                  if shared_base_zsn else
                  pm.Normal("lin_base", 0.0, 1.0, shape=(n_chains, K)))
        rows_l, theta_l = (_lineage_block(lin, K, base_l, bm=lineage_bm,
                                          variant_offsets=variant_offsets,
                                          effort_means=effort_means, general=lam)
                           if lineage_links else
                           _release_block(lin, K, base_l, effort_means=effort_means,
                                          general=lam))
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


def _human_block(struct, K, base, delta_sd: float = PRIOR_DELTA_HUMAN):
    """Partially ordered human tiers: each tier = its root's `base` level plus
    a HalfNormal increment for every tier on the root→tier path (HARD — child
    >= parent on every axis). Where a tier has SEVERAL parents its theta is the
    MAX over its paths, so it dominates all of them. Tiers on different
    branches share no path, so the prior says nothing about their relative
    strength. `base` is the per-root level, a Normal(0, human_root_sd) of its
    own outside the models' ZeroSumNormal. Returns (row_idx, theta).

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
    delta = pm.HalfNormal("delta_h", sigma=delta_sd,
                          shape=(P.shape[1], K))
    by_path = pt.dot(pt.as_tensor(R), base) + pt.dot(pt.as_tensor(P), delta)
    if all(len(g) == 1 for g in groups):     # single-parent tree: no max at all
        return rows, by_path
    return rows, pt.stack([by_path[g[0]] if len(g) == 1
                           else pt.max(by_path[g], axis=0) for g in groups])


def _lineage_block(lin: LineageStructure, K, base, bm: bool = False,
                   variant_offsets: bool = True, effort_means: bool = False, general=None):
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

    Either way the variant offsets stay mean-zero unless effort_means=True (a mean per effort
    level, see _variant_offsets): effort variants are never
    ordered, and each (node, variant) offset is its own draw, sharing ONE scalar
    sd tau_o across axes. Per-axis tau_o[k] is not identified here: offsets
    reach the likelihood as sum_k A[b,k]*offset[g,k], and two loading columns of
    this fit correlate at ~0.48, so mass slides between those axes almost
    unobservably. A single scale forbids the slide (tau_o ESS 4747, r-hat 1.001);
    per-axis scales license it (ESS 12, r-hat 1.86, and 336 divergences / 16000
    without the ZeroSumNormal damping the compensating founder movement). Only a
    Q-matrix, whose loading columns have disjoint support, leaves nothing to
    slide along. `general` (the higher-order loadings, see _assemble_theta) correlates a step's
    axes and the offsets' axes. Returns (row_idx, theta)."""
    delta_z = pm.Normal("lin_delta_z", 0.0, 1.0, shape=(lin.n_deltas, K))
    if general is not None:
        delta_z = _general_mix(delta_z, general)
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
    offset = _variant_offsets(lin, K, effort_means, general)
    theta_lin = psi[lin.node_idx] + offset[lin.offset_group]            # (R, K)
    return lin.row_idx, theta_lin


def _variant_offsets(lin: LineageStructure, K, effort_means: bool = False, general=None):
    """The (node, variant) offsets, (n_groups, K). Non-centered.

    Default: mean-zero offsets, one scale tau_o for every effort level and a scale of their
    own for the reasoning-off variants (`lin.offset_none`, effort:none): with the shared tau_o
    (learned at 0.22) a reasoning-off run scored on one benchmark was pulled toward its
    release, where switching reasoning off can cost far more than a lower effort does
    (2026-10-02).

    effort_means=True gives the offsets a mean per effort level instead, and drops tau_none:

        offset[g, k] = (m[level(g)] + slope * log2(budget_g / 16k)) * u[k] + tau_o * z[g, k]

    one scalar m per level (lin.effort_levels, unordered, Normal(0, PRIOR_EFFORT_MEAN)) along
    ONE effort direction u shared by every level, a unit vector in the K-axis space, and a
    per-doubling slope for the token-budget variants along the same direction. The
    reference groups (`lin.offset_level` == -1: reasoning off, see lineage.effort_level) keep
    a mean of zero, so psi is the release's level with reasoning off, the one setting every
    model has; a release that always reasons has its psi extrapolated through these means.
    tau_o is then the spread of a release's offset around its level's mean, alone for every
    group.

    Rank one rather than a free mean per (level, axis): offsets reach the likelihood as
    sum_k A[b, k] * offset[g, k], and two loading columns of the 2026-10-01 flagship
    correlate at 0.94, so a free per-axis mean would slide between them as a per-axis tau_o
    did (ESS 12, r-hat 1.86). Here only u carries axis detail, pinned by every effort level
    at once. u = v / |v| with v ~ Normal(0, I): the radius |v| is prior-only.

    (m, u) and (-m, -u) give the same means, the same likelihood and the same prior density:
    every m is a plain Normal and the sampler is left with the two mirror copies. The sign is
    fixed after sampling (`align_effort_sign`, draw by draw, so u points from reasoning off
    toward more effort). A HalfNormal on `high` was the gauge until 2026-10-05: its boundary
    trapped one chain of the 2026-10-03 flagship in the mirror half with m_high at 0 and the
    other levels squeezed (5 reasoning-off offsets back at -3.5 sd). No chain of that fit
    approached the plane between the copies (u never turned, |v| > 0.8), so a draw-by-draw
    flip restores the folded prior exactly. At K=1 there is no direction (u = 1) and no
    mirror.

    A release's pretrained checkpoint (`lin.offset_base`, its post-trained model being the
    reference) gets a free mean vector of its own, Normal(0, PRIOR_EFFORT_MEAN) per axis, off
    the effort direction: pretraining against post-training trades knowledge for reasoning.
    """
    tau_o = pm.HalfNormal("lin_offset_sd", sigma=PRIOR_LINEAGE_OFFSET)
    z = pm.Normal("lin_offset_z", 0.0, 1.0, shape=(lin.n_groups, K))
    if general is not None:
        z = _general_mix(z, general)
    if effort_means:
        if lin.offset_level is None:
            return tau_o * z
        return _effort_means(lin, K) + tau_o * z
    none = lin.offset_none
    if none is None or not np.any(none):
        return tau_o * z
    tau_n = pm.HalfNormal("lin_offset_none_sd", sigma=PRIOR_LINEAGE_OFFSET_NONE)
    return pt.switch(pt.as_tensor(none)[:, None], tau_n, tau_o) * z


def _effort_means(lin: LineageStructure, K):
    """The effort means of `_variant_offsets(effort_means=True)`, (n_groups, K): rank one along
    the effort direction, plus the base checkpoints' own mean vector."""
    means = pt.zeros((lin.n_groups, K))
    if lin.offset_base is not None and lin.offset_base.any():
        base_mu = pm.Normal("variant_base_mu", 0.0, PRIOR_EFFORT_MEAN, dims="latent")
        means = means + pt.as_tensor(lin.offset_base.astype(float))[:, None] * base_mu[None, :]
    levels = list(lin.effort_levels)
    if not levels:
        return means
    L = len(levels)
    model = pm.modelcontext(None)
    if "effort_level" not in model.coords:
        model.add_coord("effort_level", levels)
    m = pm.Normal("effort_m", 0.0, PRIOR_EFFORT_MEAN, shape=L)
    if K == 1:
        u = pt.ones(1)
    else:
        # Started off the origin, where the direction is undefined (0 / 0).
        v = pm.Normal("effort_dir_z", 0.0, 1.0, shape=K, initval=np.ones(K))
        u = pm.Deterministic("effort_dir", v / pt.sqrt(pt.sum(v ** 2)), dims="latent")
    pm.Deterministic("effort_mu", m[:, None] * u[None, :], dims=("effort_level", "latent"))
    # One extra slot at the end holds the reference's zero, so level -1 gathers it.
    scale = pt.concatenate([m, pt.zeros(1)])[pt.as_tensor(lin.offset_level)]
    if EFFORT_TOKEN_LEVEL in levels:
        slope = pm.Normal("effort_token_slope", 0.0, PRIOR_EFFORT_TOKEN_SLOPE)
        scale = scale + slope * pt.as_tensor(lin.offset_log2_tokens)
    return means + scale[:, None] * u[None, :]


# The variables (m, u) flips to (-m, -u): the direction's raw draw goes with it, the token
# slope too (it scales u like a level does). effort_mu and every offset are unchanged.
EFFORT_SIGN_VARS = ("effort_m", "effort_dir_z", "effort_dir", "effort_token_slope")


def align_effort_sign(posterior):
    """Fix the sign of the effort means draw by draw, in place: (m, u) -> (-m, -u) wherever
    the mean of m over the levels is negative, so u points from reasoning off toward more
    effort in every draw (`_effort_means`). The flip leaves effort_mu, the offsets and the
    likelihood unchanged. A no-op without an effort direction (K=1, or no effort means).
    Returns the number of flipped draws per chain (None when there is nothing to align)."""
    if "effort_dir" not in posterior or "effort_m" not in posterior:
        return None
    m = posterior["effort_m"]
    flip = m.mean([d for d in m.dims if d not in ("chain", "draw")]) < 0
    for name in EFFORT_SIGN_VARS:
        if name in posterior:
            posterior[name] = posterior[name].where(~flip, -posterior[name])
    return flip.sum("draw").values


def _release_block(lin: LineageStructure, K, base, effort_means: bool = False, general=None):
    """Release nodes without the chains: theta = base[node] + the variant offset.

    The canonical K=1 index's grouping of a release's effort variants, built like the
    lineage prior of the K-axis fit (`_lineage_block`) minus the links between releases: the
    same nodes of 1_curated/lineage_map.csv, the same (node, variant) offset groups and the
    same offset scales (_variant_offsets), but each node's level
    is its own draw from the shared ZeroSumNormal (`base`, one row per node) instead of its
    chain founder's plus increments, so no release is ordered after another. A variant seen on
    one benchmark only (a CritPt score per effort) then sits near its release's level, which
    every run of the release informs, instead of being a free test-taker. Returns
    (row_idx, theta)."""
    offset = _variant_offsets(lin, K, effort_means, general)
    return lin.row_idx, base[lin.node_idx] + offset[lin.offset_group]


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



CENSOR_LOG_FLOOR = -700.0  # log of the smallest censored probability kept finite


def _censored_beta_logp(value, a, b, lo, hi):
    """Beta log-density with left censoring at ``lo`` and right censoring at ``hi``.

    Elementwise: a value at ``lo`` contributes log F(lo), one at ``hi`` log(1 - F(hi)), the
    rest the density; every term is floored at CENSOR_LOG_FLOOR so an underflowing tail
    cannot return -inf. Used by build_mirt_model(censor_eps=...).
    """
    import pytensor.tensor as pt
    dist = pm.Beta.dist(alpha=a, beta=b)
    log_cdf_lo = pm.logcdf(dist, lo)
    log_sf_hi = pt.log1mexp(pt.minimum(pm.logcdf(dist, hi), -1e-12))
    term = pt.switch(pt.le(value, lo), log_cdf_lo,
                     pt.switch(pt.ge(value, hi), log_sf_hi, pm.logp(dist, value)))
    return pt.maximum(term, CENSOR_LOG_FLOOR)


def _beta_binomial_logp(value, a, b, n_items, k):
    """Beta-binomial log-probability of k successes out of n_items where n_items > 0, the plain
    Beta density of `value` elsewhere (continuous benchmarks, n_items = 0).

    k = score * n_items may be fractional (a mean over several runs of the same items): the pmf
    is written with log-gamma functions, which extend it smoothly. Each branch gets safe inputs
    on the other branch's rows (n = 1, k = 0.5), so both stay finite and so do the gradients.
    Used by build_mirt_model(binomial_n=...).
    """
    is_b = pt.gt(n_items, 0.0)
    n = pt.switch(is_b, n_items, 1.0)
    kk = pt.switch(is_b, k, 0.5)
    lgam = pt.gammaln
    log_choose = lgam(n + 1.0) - lgam(kk + 1.0) - lgam(n - kk + 1.0)
    log_beta_ratio = (lgam(kk + a) + lgam(n - kk + b) - lgam(n + a + b)) - (lgam(a) + lgam(b) - lgam(a + b))
    # a, b <= 0 (phi_b = 1/(4 sigma_b^2) - 1 < 0 once sigma_b > 0.5) is outside the model, as
    # pm.Beta's own check says; gammaln of a negative argument stays finite, so without this
    # the item rows score about +37 each there, a spike tempered samplers fell into
    # (2026-10-09). Inside the domain nothing changes.
    valid = pt.and_(pt.gt(a, 0.0), pt.gt(b, 0.0))
    logp = pt.switch(is_b, log_choose + log_beta_ratio, pm.logp(pm.Beta.dist(alpha=a, beta=b), value))
    return pt.switch(valid, logp, -np.inf)


def softmin_eta(A, theta, D, beta, lse=None, log=None):
    """eta of the softmin links over the last axis: s * softmin_beta(theta; A / s) - D with
    s = sum_k A_k (config.PRIOR_COMP_BETA). Broadcasts; numpy by default, pytensor when
    `lse`/`log` are pytensor's (the model), so the fit and every post-fit reader share one
    formula. A cell at exactly 0 weighs nothing (the 1e-12 keeps its log finite). `beta` is a
    scalar or broadcasts against eta (e.g. (S, 1) per draw), not against theta."""
    if lse is None:
        from scipy.special import logsumexp as lse
        log = np.log
    s = A.sum(axis=-1)
    logw = log(A + 1e-12) - log(s)[..., None]
    b_k = beta[..., None] if getattr(beta, "ndim", 0) else beta
    return -s * lse(logw - b_k * theta, axis=-1) / beta - D


def trace_softmin(post):
    """(beta, offset) of a softmin-link posterior as numpy, or None for any other link.
    beta is (chain, draw) for "semicomp" and the config constant for "noncomp" (not sampled);
    offset is (chain, draw, K). The link is read off the trace's `mirt_link` attr."""
    link = post.attrs.get("mirt_link")
    if link not in ("semicomp", "noncomp", "semicomp0"):
        return None
    beta = NONCOMP_BETA if link == "noncomp" else post["comp_beta"].values
    if link == "semicomp0":                     # no offset: zeros of the (chain, draw, K) shape
        return beta, np.zeros(post["A"].shape[:2] + post["A"].shape[-1:])
    return beta, post["comp_offset"].values


def build_mirt_model(data: ECIData, K: int,
                      anchors: dict | None = None,
                      loading_prior: str = "normal",
                      human_order: dict[str, str | None] | list[str] | None = None,
                      human_delta_sd: float = PRIOR_DELTA_HUMAN,
                      human_root_sd: float = PRIOR_HUMAN_ROOT,
                      lineage: LineageStructure | None = None,
                      lineage_bm: bool = False,
                      lineage_links: bool = True,
                      variant_offsets: bool = True,
                      effort_means: bool = False,
                      pin_benchmark: str | None = None,
                      plt_founders: list[str] | None = None,
                      floor_c: np.ndarray | None = None,
                      ceiling_noise: bool = False,
                      known_ceiling: np.ndarray | None = None,
                      ceiling_lower: np.ndarray | None = None,
                      known_se: bool = False,
                      censor_eps: np.ndarray | None = None,
                      binomial_n: np.ndarray | None = None,
                      pooled_noise: bool = False,
                      sigma_b_floor: float = 0.0,
                      shared_base_zsn: bool = True,
                      time_t: np.ndarray | None = None,
                      theta_t_cells: bool = False,
                      theta_skew: bool = False,
                      theta_general: bool = False,
                      simple_structure: bool = False,
                      share_conc: float | None = None,
                      axis_pin: dict | None = None,
                      pin_offset: float = DEFAULT_PIN_OFFSET,
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

    human_delta_sd, human_root_sd: sigma of the HalfNormal increment between ordered tiers
    and sd of the roots' private Normal base. The fit's values travel in FitSpec (and so in
    the trace's mirt_spec); a caller rebuilding a model from a trace passes the spec's,
    never the config's.
    lineage: optional release-chain structure (see lineage.py). Chained models get
    psi[node] + variant offset, where psi follows founder + cumulative SOFT
    (Normal, can regress) increments along the chain. Disjoint from human_order.
    lineage_bm=True scales each increment by the release gap dt (in years) —
    Normal(drift*dt, s^2*dt), a Brownian motion with drift observed at the
    release dates — under ONE shared per-axis drift rate (a per-vendor pooled
    rate was fitted 2026-07-27 and removed: roughly one release per year per
    vendor cannot pin a rate; see _lineage_block). The mean-zero effort-variant
    offsets are unaffected.
    lineage_links=False keeps the release nodes and their variant offsets and drops the
    chains: each node is its own draw from the shared ZeroSumNormal (_release_block). The
    canonical K=1 index uses it so a release's effort variants share its level, as in the
    K-axis fit, without ordering any release after another.
    effort_means=True (with a lineage structure and its variant offsets): the offsets get a
    mean per effort level, m[level] * u along one shared effort direction u, and one spread
    tau_o for every group in place of tau_o and tau_none (_variant_offsets).

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

    simple_structure: with the "normal" loading prior, each loading row is a size times
    simplex shares with a learned concentration (soft between-item structure, see the branch);
    share_conc fixes that concentration instead (left free, the 2026-10-19 data keep the rows
    as spread as the flagship's, median top share 0.40).
    axis_pin: with simple_structure, {benchmark: axis index} naming each benchmark's main
    axis a priori: its share logit gets +pin_offset before the softmax, so the prior leans
    the row toward that axis (offset 1.5 at concentration 2.5: top share 0.93 a priori) while
    the data keep every cross-loading free, a soft target structure in the sense of BSEM
    (Muthén & Asparouhov 2012). The point is the orientation: left free, the assignment of
    benchmarks to axes is a discrete choice the sampler makes per chain (the 2026-10-23
    weekend fits reached their best organisation in 2 chains of 30), and nothing in the data
    picks one. The pin is taken from outside the sampled fit, e.g. the argmax of the flagship's
    loadings after an oblique rotation to simple structure (analysis.rotation.geomin_rotate).
    Benchmarks missing from the dict are left unpinned.
    theta_general: a higher-order general level under the axes, which correlates every
    per-axis innovation of a test-taker (see _assemble_theta, config.PRIOR_THETA_GENERAL).
    theta_skew: the same block (unstructured rows plus human-root/chain-founder
    slices) independent skew-normal per axis, standardised, with a learned shape
    alpha_k ~ Normal(0, config.PRIOR_THETA_SKEW_ALPHA) (see _theta_skew_cells).
    The other non-Gaussian identification channel, fitted to the abilities'
    actual shape (a left tail). In the sampler study (2026-10-13/16) it was the
    one change that made chains agree more AND fit better: on the flagship,
    14 of 30 chains above -14395 against 4 of 30, better LOO on each of three
    seeds, fewer divergences. Exclusive with theta_t_cells.

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
        "semicomp" / "noncomp" are the softmin links (config.PRIOR_COMP_BETA,
        `softmin_eta`): eta = s_b * softmin_beta(theta + o; w_b) - D_b with
        A_b = s_b w_b, compensatory at beta -> 0 and conjunctive (the weakest
        loaded axis caps the score) as beta grows; beta is learned
        ("comp_beta") or fixed at config.NONCOMP_BETA, and o ("comp_offset")
        frees the axes' relative origin ("semicomp0": learned beta, no offset).
        Requires loading_prior="normal"
        (A >= 0, so w_b is a weight).

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

    ceiling_noise: estimate a per-benchmark upper asymptote d_b = 1 - e_b, the
    gap e_b being the share of items a perfect model still fails (wrong keys,
    ambiguous items, judge disagreement). With a share e of wrong keys the mean
    is c + (1 - c - e) * sigmoid(eta), so the gap is absolute, independent of
    the chance floor: e_b = 0.25 * Beta(1.43, 4.67) (config.PRIOR_CEILING_GAP,
    benchmark-forecasting's posterior population of 1 - L). d_b is identified
    only where observations press the ceiling; elsewhere the posterior returns
    the prior. Diagnostic use: read `ceiling_d` off the trace. A FIXED wall read
    from a curated file was the retired --ceilings flag, and a freely estimated
    one (the retired soft_ceiling flag) was weakly identified / multimodal on
    this data, which is why the gap is confined to a noise-sized prior.

    known_ceiling: per-benchmark ceiling the pipeline records (NaN elsewhere,
    data.load_benchmark_ceilings); under ceiling_noise those benchmarks keep it
    instead of estimating one. Ignored without ceiling_noise.

    ceiling_lower: per-benchmark best performance observed outside `data` (NaN where none),
    data.load_observed_best: under ceiling_noise the ceiling never sits below the best
    performance observed on the benchmark, the highest score in `data` (always applied) or in
    this array (the human baselines, fitted or not). The rule is benchmark-forecasting's
    L_floor_observed, under which PRIOR_CEILING_GAP was estimated: the gap's Beta is truncated
    to e_b <= 1 - best (density not renormalised, a constant per benchmark with fixed
    hyperparameters), and a benchmark where someone scored 1.0 gets d_b = 1. Ignored without
    ceiling_noise.

    binomial_n: per-observation item count N of the benchmark (NaN for a benchmark whose
    score is not a share of items): the score enters as k = score * N successes of a
    Beta-binomial instead of a Beta density (see the likelihood block).

    censor_eps: per-observation bound (data.load_boundary_eps) turning the Beta
    likelihood into a censored one: a score at or beyond eps_b contributes
    P(y <= eps_b) (or P(y >= 1 - eps_b)) instead of the density at a clipped
    point. None (default) keeps the clipped-density likelihood.

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

    if simple_structure and loading_prior != "normal":
        raise ValueError("simple_structure reshapes the 'normal' loading prior's rows; "
                         f"not {loading_prior!r}")
    if axis_pin and not simple_structure:
        raise ValueError("axis_pin leans the simple-structure shares; it needs simple_structure")
    pin_onehot = np.zeros((data.n_benchmarks, K))
    if axis_pin:
        bench_pos = {b: i for i, b in enumerate(data.blookup["benchmark"].tolist())}
        for bench_name, axis_k in axis_pin.items():
            if bench_name not in bench_pos:
                raise ValueError(f"axis_pin benchmark not in data: '{bench_name}'")
            if not (0 <= int(axis_k) < K):
                raise ValueError(f"axis_pin axis {axis_k} out of range for K={K}")
            pin_onehot[bench_pos[bench_name], int(axis_k)] = 1.0
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
    if lineage_bm and not lineage_links:
        raise ValueError("lineage_bm scales the links between releases; lineage_links=False "
                         "has none")
    if effort_means and (lineage is None or not variant_offsets):
        raise ValueError("effort_means puts a mean on the variant offsets; it needs a lineage "
                         "structure with its variant offsets")

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

    if link not in ("linear", "loglog", "semicomp", "noncomp", "semicomp0"):
        raise ValueError(f"unknown link: {link!r} (use 'linear', 'loglog', 'semicomp', "
                         "'noncomp' or 'semicomp0')")
    if link in ("semicomp", "noncomp", "semicomp0") and loading_prior != "normal":
        raise ValueError(f"link={link!r} reads each loading row as weights: it needs the "
                         "non-negative loading_prior='normal'")
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
            sigma_b = pm.Deterministic("sigma_b", sigma_b_floor + pt.exp(mu_s + tau_s * z_s),
                                       dims="bench")
        elif sigma_b_floor:
            sigma_b = pm.Deterministic("sigma_b", sigma_b_floor + pm.LogNormal(
                "sigma_b_free", dims="bench", **PRIOR_SIGMA_B), dims="bench")
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
            human_delta_sd=human_delta_sd, human_root_sd=human_root_sd,
            lin=lineage, lineage_bm=lineage_bm,
            variant_offsets=variant_offsets,
            shared_base_zsn=shared_base_zsn, time_t=time_t,
            theta_t_cells=theta_t_cells, theta_skew=theta_skew, lineage_links=lineage_links,
            effort_means=effort_means, theta_general=theta_general)
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
        elif simple_structure:
            # Soft between-item structure (Adams, Wilson & Wang 1997): each benchmark's loading
            # row is a size times shares on the simplex, A_b = tau_A * r_b * softmax(s * u_b),
            # u_b ~ N(0, I). The concentration s is learned (config.PRIOR_SHARE_CONC): large s puts
            # most rows near one axis, s -> 0 spreads them evenly, so the data set how close to
            # simple structure the loadings sit. Cross-loadings are the channel of compensation
            # (a mixed benchmark's score explained by whichever axis is already high), and fewer
            # of them leave fewer models trading one axis for another. The size r_b keeps the
            # scale of the HalfNormal rows: Gamma with their L1 norm's mean, K sqrt(2/pi).
            share_conc = (pm.LogNormal("A_share_conc", **PRIOR_SHARE_CONC) if share_conc is None
                          else pt.as_tensor(float(share_conc)))
            u = pm.Normal("A_share_z", 0.0, 1.0, dims=("bench", "latent"))
            # axis_pin: a fixed offset on the pinned axis's logit (zero rows elsewhere), so
            # the prior leans the row that way and the data can still move it.
            logits = share_conc * (u + pt.as_tensor(float(pin_offset) * pin_onehot))
            shares = pm.Deterministic("A_share", pt.special.softmax(logits, axis=1),
                                      dims=("bench", "latent"))
            size_mean = K * np.sqrt(2 / np.pi)
            r = pm.Gamma("A_size", alpha=PRIOR_SHARE_SIZE_SHAPE,
                         beta=PRIOR_SHARE_SIZE_SHAPE / size_mean, dims="bench")
            A = pm.Deterministic("A", r[:, None] * shares * pt.as_tensor(free_mask) * tau_A,
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
        elif link in ("semicomp", "noncomp", "semicomp0"):
            beta_c = (NONCOMP_BETA if link == "noncomp"
                      else pm.LogNormal("comp_beta", **PRIOR_COMP_BETA))
            offset = (0.0 if link == "semicomp0" else
                      pm.ZeroSumNormal("comp_offset", sigma=PRIOR_COMP_OFFSET_SD, dims="latent"))
            eta = softmin_eta(A[data.bench_idx], theta_lik[data.model_idx] + offset,
                              D[data.bench_idx], beta_c, lse=pt.logsumexp, log=pt.log)
        else:
            A_obs     = A[data.bench_idx]          # (n_obs, K)
            theta_obs = theta_lik[data.model_idx]  # (n_obs, K)
            eta = (A_obs * theta_obs).sum(axis=-1) - D[data.bench_idx]
        mu_n = pm.math.sigmoid(eta)
        if ceiling_noise:
            # Estimated 4PL: the mean saturates at d_b = 1 - e_b instead of 1.
            # e_b is an ABSOLUTE gap, the share of items a perfect model still
            # fails: with a share e of wrong keys a random guesser still lands
            # at c, a perfect model at 1 - e, and mu = c + (1 - c - e) sigmoid.
            # A gap relative to (1 - c) would ask a 4-choice benchmark for a
            # larger gap than a free-answer one for the same error rate.
            # Prior: benchmark-forecasting's posterior population of 1 - L
            # (config.PRIOR_CEILING_GAP); its 0.25 cap keeps d_b above every
            # floor. Benchmarks with a known ceiling keep it.
            #
            # The scale decides identifiability. d reaches the likelihood
            # through log(d_b - c_b), a constant shift of eta that D_b absorbs
            # exactly, so d rides a flat ridge with difficulty and is pinned
            # only by the one signature a ceiling has: frontier scores piling
            # at a common sub-1 level with no ability trend among them, which
            # the 2PL cannot fit because it forces
            # Var = 4 sigma_b^2 mu(1-mu) -> 0 as mu -> 1. The gap is kept
            # small so the mid-scale data still pin A and D as in the 2PL.
            c_full = floor_c if floor_c is not None else np.zeros(data.n_benchmarks)
            if np.any(c_full >= 1.0 - PRIOR_CEILING_GAP["scale"]):
                raise ValueError("a chance floor reaches the lowest possible ceiling "
                                 f"(1 - {PRIOR_CEILING_GAP['scale']})")
            # The ceiling is at least the best performance observed on the benchmark
            # (benchmark-forecasting's L_floor_observed, the rule PRIOR_CEILING_GAP was
            # estimated under): the gap's Beta is truncated to e_b <= 1 - best, as a Uniform
            # on the allowed interval carrying the Beta density in a Potential. Where the
            # bound does not bite (best <= 1 - scale) this is the plain Beta, logp included;
            # where someone scored 1.0 the gap is 0. Forward draws (pm.draw) ignore the
            # Potential, so they are uniform on the interval: read the prior off config.
            scale = PRIOR_CEILING_GAP["scale"]
            best = np.full(data.n_benchmarks, -np.inf)
            np.maximum.at(best, np.asarray(data.bench_idx), np.asarray(data.scores, float))
            if ceiling_lower is not None:
                best = np.fmax(best, np.asarray(ceiling_lower, dtype=np.float64))
            top = np.clip((1.0 - best) / scale, 0.0, 1.0)          # allowed share of the scale
            reached = top <= 1e-9                                   # someone scored 1.0
            gap_raw = pm.Uniform("ceiling_gap_raw", lower=0.0, upper=np.maximum(top, 1e-6),
                                 dims="bench")
            gap_lp = pm.logp(pm.Beta.dist(alpha=PRIOR_CEILING_GAP["alpha"],
                                          beta=PRIOR_CEILING_GAP["beta"]), gap_raw)
            gap = pt.switch(reached, 0.0, scale * gap_raw)
            if known_ceiling is not None:
                known = np.asarray(known_ceiling, dtype=np.float64)
                pinned = np.isfinite(known)
                if np.any(pinned & (best > known + 1e-9)):
                    import warnings
                    warnings.warn("ceiling_noise: a recorded ceiling sits below the best score "
                                  "observed on its benchmark; the recorded ceiling is kept",
                                  UserWarning)
                gap = pt.switch(pinned, 1.0 - np.where(pinned, known, 1.0), gap)
                reached = reached | pinned
            # the Beta density of the estimated gaps only (a pinned one's draw is unused)
            pm.Potential("ceiling_gap_prior", pt.sum(pt.switch(reached, 0.0, gap_lp)))
            gap = pm.Deterministic("ceiling_gap", gap, dims="bench")
            d_b = pm.Deterministic("ceiling_d", 1.0 - gap, dims="bench")
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
        # The precision must stay positive (sigma_b < 0.5; with n_eff, 1/n_eff + 4 sigma_b^2 < 1).
        # pm.Beta's own parameter check is dropped by PyMC's JAX backend (and by the hand-written
        # Beta-binomial), so outside that domain the density is garbage, and tempered samplers
        # were drawn to it (2026-10-09). 0 inside the domain: no fit within it changes.
        pm.Potential("noise_domain", pt.switch(pt.all(pt.gt(phi_n, 0.0)), 0.0, -np.inf))

        if binomial_n is not None:
            # Beta-binomial on the benchmarks whose score is a share of N items: the score is
            # k = score * N successes, k ~ Binomial(N, p), p ~ Beta(mu, phi). The likelihood is a
            # probability (<= 1, so an exact fit earns nothing more), and even with no excess noise
            # (phi -> inf) a score keeps the binomial spread of item-level success, so no benchmark's
            # noise can collapse: on 2026-10-01 one K=4 axis captured Chess Puzzles, its sigma_b fell
            # to 4e-4 and six of eight chains froze in that spike. Scores of 0 and 1 enter as k = 0
            # and k = N, no clip needed. Benchmarks whose score is not a share of items
            # (binomial_n = NaN: correlations, areas, rubric means) keep the Beta density,
            # their exact zeros clipped to config.CONTINUOUS_EPS.
            if censor_eps is not None:
                raise ValueError("binomial_n and censor_eps are not combined")
            n_items = np.asarray(binomial_n, dtype=np.float64)
            if n_items.shape != (data.n_obs,):
                raise ValueError("binomial_n must have one value per observation (NaN = Beta)")
            is_b = np.isfinite(n_items) & (n_items > 0)
            n_items = np.where(is_b, n_items, 0.0)
            k = np.where(is_b, np.clip(data.scores, 0.0, 1.0) * n_items, 0.0)
            # Read by the Beta rows only (the item rows read k and n_items).
            clipped = np.clip(data.scores, CONTINUOUS_EPS, 1.0 - CONTINUOUS_EPS)
            pm.CustomDist("obs", a, b, n_items, k, logp=_beta_binomial_logp, observed=clipped)
        elif censor_eps is None:
            clipped = np.clip(data.scores, ECI_EPS, 1.0 - ECI_EPS)
            pm.Beta("obs", alpha=a, beta=b, observed=clipped)
        else:
            # Censored bounds (data.load_boundary_eps): a score at or beyond eps_b is
            # the event "y <= eps_b" (Beta CDF), not a point at a clipped value. The
            # density at 0.001 is enormous and extremely sensitive to mu and phi, so
            # every exact zero pulled hard on its model's ability or its benchmark's
            # noise; the CDF term is bounded and says only what is known. Interior
            # scores keep the plain Beta density. Written as a CustomDist rather than
            # pm.Censored so the tail probabilities can be floored: far from the data
            # (a jittered start, a nutpie initial point) the CDF underflows to 0 and
            # pm.Censored returns -inf, which no sampler can start from. A censored
            # probability below exp(-700) reads as "impossible", not as an error, and
            # the observed variable keeps its pointwise log-likelihood for LOO.
            lo = np.asarray(censor_eps, dtype=np.float64)
            if lo.shape != (data.n_obs,) or np.any(lo <= 0.0) or np.any(lo >= 0.5):
                raise ValueError("censor_eps must have one value per observation in (0, 0.5)")
            hi = 1.0 - lo
            clipped = np.clip(data.scores, lo, hi)
            pm.CustomDist("obs", a, b, lo, hi, logp=_censored_beta_logp, observed=clipped)
    return model
