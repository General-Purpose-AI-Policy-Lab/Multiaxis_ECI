"""Semi-compensatory Beta-MIRT: compensatory sum plus pairwise ability
interactions (DeMars 2015, Eq. 3).

    tp  = softplus(theta) > 0
    eta = sum_k A[b,k] theta[m,k]  +  sum_{j<k} gamma[b,jk] tp[m,j] tp[m,k]  -  D[b]
    mu  = sigmoid(eta);  score ~ Beta(mu*phi, (1-mu)*phi), clipped to (EPS, 1-EPS)

`gamma` is DeMars's interaction coefficient a_3, constrained NON-NEGATIVE:
gamma > 0 rewards having BOTH abilities on an item ("needs both"), gamma = 0
recovers the compensatory model. One-sided because "having both skills hurts"
is not a hypothesis this project holds. HalfNormal keeps its mode at 0, so no
conjunction stays reachable — it reads as posterior mass piled against the
boundary, never as an interval containing 0, which is why the readout is
`p_above_prior` (multiaxis_eci/fits/fit_interaction.py) and not an HDI-vs-zero test.

Abilities enter the PRODUCT through softplus and the linear term through raw
theta. On raw zero-centred theta the bilinear product pays the same bonus to a
both-weak pair as to a both-strong one, so a positive gamma there is not
conjunction; softplus puts the product in one quadrant, where theta -> -inf
compresses to 0 and the bonus grows only with joint strength. softplus is
monotone, so every order constraint the human/lineage blocks impose in raw space
survives on the likelihood scale, and leaving the linear term raw means
gamma -> 0 recovers the compensatory model exactly. The raw `theta` stays the
reported ability; the positive one entering the product is `theta_pos`. Being
non-linear in theta, the interaction also helps pin the orientation the linear
part leaves loose.

Two loading priors (`loading_prior`), matching the conventions of models.mirt:

- "signed" (default): signed free cells x one shared scale, rotation fixed by K
  PLT founders — one benchmark per axis, founder r loading only axes 0..r with a
  positive diagonal (reused from models.mirt._apply_plt). Because the interaction
  term is bilinear across DIFFERENT axes, the model family is closed under axis
  permutations/reflections but NOT under continuous rotations — post-hoc rotation
  alignment is invalid here, so identification must happen at fit time and any
  cross-chain gamma comparison needs sign/permutation matching first.
- "normal": non-negative HalfNormal cells x the same shared scale, no founders
  (pass plt_founders=None) — non-negativity itself pins reflections, the same
  frame convention as the canonical/confirmatory compensatory fits. Known cost
  on this data (procedural log): at exploratory K=3 a contrast axis forced
  non-negative splits into two near-duplicate positive axes.

`floor_c` (optional, per-benchmark chance floors from data.load_benchmark_floors)
switches the link to the fixed-c 3PL, mu = c_b + (1-c_b)*sigmoid(eta), exactly as
in models.mirt — c is fixed ground truth, never estimated, and the driver clips
observed scores up to the floor so below-chance rows read as uninformative-low.

`gamma_pooling` controls how many gammas are estimated:

- "benchmark" (default): one gamma per (benchmark, axis-pair) — all pairs for
  non-founders, the within-triangle pairs for founders — under a tight Normal prior.
  The richest form, but with a sparse response matrix most cells stay prior-dominated.
- "pooled": ONE shared gamma per axis-pair (K*(K-1)/2 numbers), informed by every
  observation. Answers the identifiable stage-1 question — do benchmarks need both
  skills *on average*, per axis-pair — instead of per benchmark.
- "none": gamma identically zero. This is exactly the compensatory model, kept in
  this file as the matched LOO baseline for the pooled fit (same D-prior, founders,
  and theta blocks, so the comparison isolates gamma).

In every mode a `gamma` Deterministic of shape (benchmark, pair) is written to the
trace so the shared post-fit tools (ppc, convergence) read it uniformly; "pooled"
additionally exposes the compact `gamma_pooled` (pair,) as the readout.

The prior scale is fixed, never learned: pooled gamma is K*(K-1)/2 = 3 numbers, and
a hierarchical scale over three groups is dominated by its own hyperprior while
adding a tau*|z| funnel. INTERACTION_SCALE is the sensitivity knob instead.

Without floor_c there is no lower asymptote: below-chance scores are real signal
and sigmoid(eta) represents them directly.
"""
from __future__ import annotations

import itertools

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from multiaxis_eci.config import ECI_EPS, PRIOR_SIGMA_B, PRIOR_TAU_ALPHA, PRIOR_TAU_CD
from multiaxis_eci.data import ECIData
from multiaxis_eci.lineage import LineageStructure  # noqa: F401  (type hint only)
from multiaxis_eci.models.mirt import _apply_plt, _assemble_theta, _human_structure

# Prior scale on the interaction coefficient. DeMars (2015) used a_3 ~ 1/3 of the
# main-effect discrimination, keeping the response-surface turning point in the tail.
INTERACTION_SCALE = 0.15


def build_mirt_interaction_model(data: ECIData, plt_founders, K: int = 3,
                                 human_order=None,
                                 lineage: "LineageStructure | None" = None,
                                 interaction_scale: float = INTERACTION_SCALE,
                                 gamma_pooling: str = "benchmark",
                                 loading_prior: str = "signed",
                                 floor_c: np.ndarray | None = None) -> pm.Model:
    """Build the semi-compensatory Beta-MIRT with pairwise ability interactions.

    Parameters
    ----------
    data : ECIData
        Fitted dataset (scores, model_idx, bench_idx, lookups).
    plt_founders : list[str] | None
        With loading_prior="signed": exactly K benchmark names, one per axis in
        order (founder r -> axis r). Fixes the rotation of the signed loadings;
        the founder's loadings above the diagonal are zeroed and its diagonal is
        made positive. With loading_prior="normal": must be None — the
        non-negativity constraint pins the frame instead.
    K : int
        Number of axes (default 3).
    human_order, lineage :
        theta-prior blocks (model_mirt); None gives the plain ZeroSumNormal anchor.
    interaction_scale : float
        Prior scale of each interaction coefficient, gamma = |z| * scale with
        z ~ HalfNormal(1) (DeMars a_3 ~ 1/3 of the loading scale).
    gamma_pooling : {"benchmark", "pooled", "none"}
        How many interaction coefficients to estimate (see module docstring).
    loading_prior : {"signed", "normal"}
        "signed" = free signed cells + PLT founders; "normal" = non-negative
        HalfNormal cells, no founders (see module docstring).
    floor_c : optional (n_benchmarks,) array
        Fixed per-benchmark chance floors -> fixed-c 3PL link. Pair with
        data.clip_scores_to_floors in the driver, as in fit.py --floors.

    Trace variables: `A` (loadings), `D` (difficulty), `gamma` (interaction,
    bench x pair — always present), `theta` (reported ability), `theta_pos`
    (softplus, the ability entering the product), `sigma_b`, `phi_b`. When
    gamma_pooling="pooled", also `gamma_pooled` (pair,), the shared readout.
    """
    if gamma_pooling not in ("benchmark", "pooled", "none"):
        raise ValueError(f"gamma_pooling must be benchmark/pooled/none, got {gamma_pooling!r}")
    if loading_prior not in ("signed", "normal"):
        raise ValueError(f"loading_prior must be 'signed' or 'normal', got {loading_prior!r}")
    bench_names = data.blookup["benchmark"].tolist()
    if loading_prior == "normal":
        if plt_founders is not None:
            raise ValueError(
                "loading_prior='normal' pins reflections through non-negativity; "
                "PLT founders identify the signed family. Pass plt_founders=None.")
        plt_idx = None
    else:
        if plt_founders is None:
            raise ValueError("loading_prior='signed' needs K PLT founders to fix "
                             "the rotation (the interaction model has no post-hoc "
                             "alignment fallback)")
        if len(plt_founders) != K:
            raise ValueError(f"need exactly K={K} founders (one per axis), got {len(plt_founders)}")
        if len(set(plt_founders)) != len(plt_founders):
            raise ValueError("plt_founders must be distinct benchmarks")
        unknown = [b for b in plt_founders if b not in bench_names]
        if unknown:
            raise ValueError(f"founders not in data: {unknown}")
        plt_idx = [bench_names.index(b) for b in plt_founders]
    if floor_c is not None:
        floor_c = np.asarray(floor_c, dtype=np.float64)
        if floor_c.shape != (data.n_benchmarks,):
            raise ValueError(f"floor_c must have shape ({data.n_benchmarks},), "
                             f"got {floor_c.shape}")
        if not np.all(np.isfinite(floor_c)) or not np.all((floor_c >= 0.0) & (floor_c < 1.0)):
            raise ValueError("floor_c values must be finite and in [0, 1)")

    pairs = list(itertools.combinations(range(K), 2))          # [(0,1),(0,2),(1,2)] at K=3
    # Interactions: all pairs for non-founders; a founder (loads axes 0..r) keeps
    # only pairs fully inside its triangle, so it stays a clean single/low-axis
    # anchor. Without founders (normal prior) every benchmark keeps every pair.
    inter_mask = np.ones((data.n_benchmarks, len(pairs)))
    if plt_idx is not None:
        for r, fi in enumerate(plt_idx):
            for p, (j, k) in enumerate(pairs):
                if not (j <= r and k <= r):
                    inter_mask[fi, p] = 0.0

    coords = {
        "model":  data.mlookup["model"].tolist(),
        "bench":  data.blookup["benchmark"].tolist(),
        "latent": [f"axis{k + 1}" for k in range(K)],
        "pair":   [f"axis{j + 1}xaxis{k + 1}" for (j, k) in pairs],
    }
    with pm.Model(coords=coords) as model:
        imask = pm.Data("inter_mask", inter_mask, dims=("bench", "pair"))

        sigma_b = pm.LogNormal("sigma_b", dims="bench", **PRIOR_SIGMA_B)
        phi = pm.Deterministic("phi_b", 1.0 / (4.0 * sigma_b**2) - 1.0, dims="bench")

        # Difficulty: free per-benchmark D_z scaled by tau_CD. The scale's location
        # is pinned once, on the theta side (ZeroSumNormal), so D needs no second
        # constraint — a zero-sum D would forbid a nonzero mean difficulty (the data
        # want ~0.57 logits), pushing that level into gamma (whose theta_j*theta_k
        # has nonzero cross-model mean, so it would act as a global intercept).
        tau_CD = pm.LogNormal("tau_CD", **PRIOR_TAU_CD)
        D_z = pm.Normal("D_z", 0.0, 1.0, dims="bench")
        D = pm.Deterministic("D", D_z * tau_CD, dims="bench")

        # Loadings, one shared scale either way. Signed: free cells, PLT founders
        # fix the rotation. Normal: non-negative cells, the constraint itself
        # pins reflections (same frame convention as the compensatory fits).
        tau_A = pm.LogNormal("tau_A", **PRIOR_TAU_ALPHA)
        if loading_prior == "signed":
            A_z = pm.Normal("A_z", 0.0, 1.0, dims=("bench", "latent"))
            A = pm.Deterministic("A", _apply_plt(A_z * tau_A, plt_idx, K),
                                 dims=("bench", "latent"))
        else:
            A_z = pm.HalfNormal("A_z", sigma=1.0, dims=("bench", "latent"))
            A = pm.Deterministic("A", A_z * tau_A, dims=("bench", "latent"))

        # theta: human/lineage blocks, base levels inside the shared ZeroSumNormal.
        theta = _assemble_theta(                               # (model, latent)
            data.n_models, K,
            human_struct=_human_structure(human_order, coords["model"]),
            lin=lineage)
        # Product-scale ability: strictly positive, order-preserving. Raw theta
        # keeps the ZeroSumNormal location pin and the human/lineage order
        # structure and stays the ability the linear term (and every readout)
        # uses; softplus only re-expresses it for the interaction quadrant.
        theta_pos = pm.Deterministic("theta_pos", pt.softplus(theta),
                                     dims=("model", "latent"))

        # Interaction coefficients gamma = a_3, non-negative under a tight fixed
        # scale: |z| * scale with z ~ HalfNormal(1). gamma is always stored at
        # (bench, pair) so ppc/convergence read one shape; the three pooling modes
        # differ only in how many free numbers feed it.
        n_pairs = len(pairs)
        if gamma_pooling == "benchmark":
            g_z = pm.HalfNormal("gamma_z", 1.0, dims=("bench", "pair"))
            gamma = pm.Deterministic("gamma", g_z * interaction_scale * imask,
                                     dims=("bench", "pair"))
        elif gamma_pooling == "pooled":
            g_z = pm.HalfNormal("gamma_z", 1.0, dims="pair")
            gamma_pooled = pm.Deterministic("gamma_pooled", g_z * interaction_scale,
                                            dims="pair")
            # broadcast the shared per-pair value across benchmarks, then mask founders
            gamma = pm.Deterministic("gamma", gamma_pooled[None, :] * imask,
                                     dims=("bench", "pair"))
        else:  # "none" — compensatory baseline, gamma identically zero
            gamma = pm.Deterministic("gamma",
                                     pt.zeros((data.n_benchmarks, n_pairs)),
                                     dims=("bench", "pair"))

        A_obs = A[data.bench_idx]                               # (n_obs, K)
        th_obs = theta[data.model_idx]                          # (n_obs, K)
        eta = (A_obs * th_obs).sum(axis=-1) - D[data.bench_idx]
        if gamma_pooling != "none":                             # gamma == 0 otherwise
            tp_obs = theta_pos[data.model_idx]                  # (n_obs, K), all > 0
            prod = pt.stack([tp_obs[:, j] * tp_obs[:, k] for (j, k) in pairs], axis=1)
            eta = eta + (gamma[data.bench_idx] * prod).sum(axis=-1)

        mu = pt.sigmoid(eta)
        if floor_c is not None:
            # fixed 3PL: a random guesser lands at chance c_b, not 0. c is a
            # numpy constant, so floor_c=None folds back to the plain link exactly.
            c_obs = floor_c[data.bench_idx]
            mu = c_obs + (1.0 - c_obs) * mu
        mu = pt.clip(mu, ECI_EPS, 1.0 - ECI_EPS)
        phi_n = phi[data.bench_idx]
        clipped = np.clip(data.scores, ECI_EPS, 1.0 - ECI_EPS)
        pm.Beta("obs", alpha=mu * phi_n, beta=(1.0 - mu) * phi_n, observed=clipped)

    return model
