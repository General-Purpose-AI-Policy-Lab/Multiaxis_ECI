"""Factor identification after sampling: canonicalization, per-draw
rotation alignment for signed fits, varimax/promax/geomin rotations."""
from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr
from dataclasses import dataclass
from scipy.optimize import linear_sum_assignment

from data import ECIData

# ── MIRT factor post-processing ─────────────────────────────────────────────

def canonicalize_factors(A_draws: np.ndarray,
                          theta_draws: np.ndarray,
                          tau_draws: np.ndarray,
                          rank_track: bool = True):
    """Rank-track a non-negative MIRT factor posterior (permutation fix).

    Parameters
    ----------
    A_draws     : (S, B, K) loadings per posterior draw.
    theta_draws : (S, M, K) abilities per posterior draw.
    tau_draws   : (S, K) per-axis scale (tau_A) per draw.
    rank_track  : if True (default), sort axes by tau_A (descending) within
        each draw so "axis r" is "the r-th strongest axis" everywhere. Set
        False for ANCHORED fits, where axis identity is already pinned by the
        anchors — rank-tracking would scramble that fixed identity across draws.

    Returns (A, theta, tau) with the same shapes. No sign step is needed:
    non-negative loadings have no sign-flip symmetry to resolve.
    """
    A = np.asarray(A_draws, dtype=float).copy()
    theta = np.asarray(theta_draws, dtype=float).copy()
    tau = np.asarray(tau_draws, dtype=float).copy()

    if not rank_track:
        return A, theta, tau

    # Rank-tracking: by default, sort axes by tau (descending) within each draw.
    # If tau is constant across axes per draw (non-ARD model with shared scalar
    # tau_A broadcast to (K,)), tau carries no within-draw ordering signal, so
    # we fall back to loading energy (sum_b A[b,k]^2) per draw — the realised
    # axis magnitudes. Detection: max per-draw range of tau across axes ≈ 0.
    tau_range = tau.max(axis=1) - tau.min(axis=1)
    if tau_range.max() < 1e-6:
        key = (A ** 2).sum(axis=1)                          # (S, K) loading energy
    else:
        key = tau
    order = np.argsort(-key, axis=1)                        # (S, K)
    A = np.take_along_axis(A, order[:, None, :], axis=2)
    theta = np.take_along_axis(theta, order[:, None, :], axis=2)
    tau = np.take_along_axis(tau, order, axis=1)
    return A, theta, tau


def align_factor_signs(A_draws: np.ndarray, theta_draws: np.ndarray):
    """Resolve the per-draw ± sign of each axis for SIGNED loading fits.

    A signed factor model is identified only up
    to a per-column sign flip: (A[:,k], theta[:,k]) -> (-A[:,k], -theta[:,k])
    leaves every prediction unchanged, so draws (and chains) land on either
    sign at random and a naive mean cancels toward 0. Unlike the continuous
    rotation, this is a DISCRETE exact symmetry, so post-hoc alignment resolves
    it exactly (the Nirwan & Bertschinger post-processing step).

    Convention: per axis k, pick the reference benchmark with the largest
    median |loading| (abs BEFORE averaging — a signed mean would cancel), then
    flip each draw so that reference loads positive. eta = A·theta^T is
    invariant to the paired flip, so diagnostics are untouched.

    Returns (A, theta, ref_idx). Non-negative loading fits never need this.
    """
    A = np.asarray(A_draws, dtype=float).copy()
    theta = np.asarray(theta_draws, dtype=float).copy()
    K = A.shape[2]
    ref = np.argmax(np.median(np.abs(A), axis=0), axis=0)      # (K,)
    for k in range(K):
        flip = np.sign(A[:, ref[k], k])
        flip[flip == 0] = 1.0
        A[:, :, k] *= flip[:, None]
        theta[:, :, k] *= flip[:, None]
    return A, theta, ref


# ── per-draw rotation alignment (signed-free fits) ──────────────────────────
#
# A loading_prior="signed" fit samples the rotation orbit freely, so every draw
# lands in its own arbitrary orientation. These helpers resolve the EXACT
# symmetries (rotation, per-axis sign, axis permutation) one draw at a time —
# the order of operations matters: rotating the posterior MEAN (the old promax
# path) averages across orientations first, which cancels structure the same
# way unaligned signs do. Every transform here is applied to (A, theta) as a
# pair, so eta = A·theta^T — and with it every diagnostic — is untouched.

def _signed_perm_matrix(perm: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """(K,K) signed permutation P with P[perm[k], k] = signs[k], so that
    (L @ P)[:, k] = signs[k] * L[:, perm[k]]. P is orthogonal, hence applying
    it to A and theta together preserves eta."""
    K = len(perm)
    P = np.zeros((K, K))
    P[perm, np.arange(K)] = signs
    return P


def _match_columns(L: np.ndarray, ref: np.ndarray):
    """Best signed matching of L's columns onto ref's columns (EXACT, via the
    Hungarian assignment): minimizes sum_k ||s_k·L[:,perm[k]] − ref[:,k]||².

    Because column norms don't depend on the assignment, this is equivalent to
    maximizing sum_k |L[:,perm[k]] · ref[:,k]|. Returns (perm, signs) with
    signs from the matched dot products. This is the sign/permutation step of
    Papastamoulis & Ntzoufras (2022)."""
    dots = L.T @ ref                                     # (K, K) source × target
    row, col = linear_sum_assignment(-np.abs(dots))      # maximize |dot|
    K = L.shape[1]
    perm = np.empty(K, dtype=int)
    signs = np.empty(K)
    for j, k in zip(row, col):                           # source j → target k
        perm[k] = j
        s = np.sign(dots[j, k])
        signs[k] = s if s != 0 else 1.0
    return perm, signs


def _greedy_match_columns(L: np.ndarray, ref: np.ndarray):
    """Greedy variant of _match_columns (MatchAlign, Poworoznek–Ferrari–Dunson):
    repeatedly take the not-yet-used (source, target) pair with the largest
    |dot| and lock it in. Cheaper than Hungarian and usually identical; kept
    separate so the method comparison can show WHERE they differ."""
    dots = L.T @ ref
    absd = np.abs(dots).copy()
    K = L.shape[1]
    perm = np.empty(K, dtype=int)
    signs = np.empty(K)
    for _ in range(K):
        j, k = np.unravel_index(np.argmax(absd), absd.shape)
        perm[k] = j
        s = np.sign(dots[j, k])
        signs[k] = s if s != 0 else 1.0
        absd[j, :] = -np.inf
        absd[:, k] = -np.inf
    return perm, signs


@dataclass
class AlignResult:
    """Per-draw-aligned factor draws. A (S,B,K), theta (S,M,K); `ref` is the
    (B,K) reference orientation every draw was matched to; `Phi` is the mean
    aligned factor-correlation (promax method only, None otherwise); `meta`
    carries method internals (e.g. WOP iterations, promax fallback count)."""
    A: np.ndarray
    theta: np.ndarray
    ref: np.ndarray
    method: str
    Phi: np.ndarray | None = None
    meta: dict | None = None


def _default_align_ref(A_draws: np.ndarray, theta_draws: np.ndarray) -> np.ndarray:
    """Deterministic starting reference: sign-align the draws (the discrete
    part only), take the mean loading matrix, varimax it. Crude — mean-before-
    rotation smears across orientations — but it only has to be a consistent
    TARGET; each draw is then aligned to it exactly."""
    A_s, _, _ = align_factor_signs(A_draws, theta_draws)
    ref, _ = _varimax(A_s.mean(axis=0))
    return ref


def align_rotations(A_draws: np.ndarray, theta_draws: np.ndarray,
                    method: str = "varimax", ref: np.ndarray | None = None,
                    wop_iters: int = 10, wop_tol: float = 1e-6) -> AlignResult:
    """Align every posterior draw of a signed factor fit to a common
    orientation. The four methods differ in WHAT picks each draw's rotation:

    * "varimax"    — each draw rotates to ITS OWN simple structure, then its
                     axes are sign/permutation-matched to the reference
                     (exact Hungarian). P&N (2022) RSP-style.
    * "wop"        — each draw is Procrustes-rotated straight onto the
                     reference (SVD, full orthogonal group — handles signs and
                     permutations implicitly); the reference is re-estimated
                     from the aligned mean and the pass repeats to tolerance.
                     P&N (2022) weighted-orthogonal-Procrustes.
    * "matchalign" — varimax per draw + GREEDY matching (Poworoznek, Ferrari
                     & Dunson).
    * "promax"     — per-draw OBLIQUE rotation (promax: chase an exaggerated
                     varimax target by an unconstrained map) + exact matching;
                     also returns the mean aligned factor correlation Phi.
    * "geomin"     — OBLIQUE geomin criterion (directly optimized simple
                     structure, Browne 2001 — the modern default oblique;
                     preferred over promax). Implementation: per-draw varimax
                     alignment first (resolves the exact symmetry), then ONE
                     global geomin transform of the co-oriented draws — the
                     per-draw GPA optimization would be 16k× an iterative
                     solver for the same answer. Returns Phi.

    * "raw"        — NO simple-structure rotation: orient draws to the
                     variance-ordered PCA frame (per-draw orthogonal
                     Procrustes). Bipolar contrast axes survive as single axes
                     — the frame a raw PCA component / Epoch's "claudiness"
                     lives in, and the honest view for a contrast question.

    All methods transform (A, theta) as a pair, so eta is preserved draw-by-
    draw (asserted on a sample). Agreement ACROSS methods is evidence the
    axes are data-driven rather than criterion-driven."""
    if method not in ("varimax", "wop", "matchalign", "promax", "geomin", "raw"):
        raise ValueError(f"unknown alignment method {method!r}")
    A = np.asarray(A_draws, dtype=float)
    theta = np.asarray(theta_draws, dtype=float)
    S, B, K = A.shape
    if K == 1:                                # nothing to rotate; fix signs only
        A_o, th_o, _ = align_factor_signs(A, theta)
        return AlignResult(A=A_o, theta=th_o, ref=A_o.mean(0), method=method)
    if ref is None:
        ref = _default_align_ref(A, theta)

    A_out = np.empty_like(A)
    th_out = np.empty_like(theta)
    meta: dict = {}
    Phi = None

    if method == "raw":
        # "No-transform" PCA frame: orient every draw to the variance-ordered
        # orthogonal axes of the (sign-aligned) mean, via per-draw orthogonal
        # Procrustes — NO simple-structure (varimax) step. This is the frame a
        # raw PCA component lives in, so a bipolar CONTRAST axis (e.g. Epoch's
        # "claudiness": agentic + vs video/hard-math −) survives as ONE axis
        # instead of being split into positive bundles by varimax. Orthogonal,
        # so Phi stays None (axes uncorrelated in loading space by construction).
        A_s, th_s, _ = align_factor_signs(A, theta)
        Abar = A_s.mean(axis=0)
        _, V = np.linalg.eigh(Abar.T @ Abar)          # ascending eigenvalue
        ref = Abar @ V[:, ::-1]                        # variance-ordered PCA frame
        for s in range(S):
            u, _, vt = np.linalg.svd(A[s].T @ ref)
            R = u @ vt
            A_out[s] = A[s] @ R
            th_out[s] = theta[s] @ R
        meta["raw"] = "PCA-frame Procrustes (no simple-structure rotation)"
    elif method == "geomin":
        # Exact-symmetry resolution per draw (cheap varimax path), then one
        # global oblique transform: after alignment every draw sits in the
        # same orientation, so a single (Tload, Ttheta) pair applied to all
        # draws is coherent — and avoids running the iterative GPA solver S
        # times for what is one rotation decision.
        base = align_rotations(A, theta, method="varimax", ref=ref)
        Tl, Tt, Phi = geomin_rotate(base.A.mean(axis=0))
        A_out = base.A @ Tl
        th_out = base.theta @ Tt
        meta["geomin"] = "global oblique transform after per-draw varimax alignment"
    elif method == "wop":
        for it in range(wop_iters):
            for s in range(S):
                # orthogonal Procrustes: R = argmin ||A_s R - ref||_F
                u, _, vt = np.linalg.svd(A[s].T @ ref)
                R = u @ vt
                A_out[s] = A[s] @ R
                th_out[s] = theta[s] @ R
            new_ref = A_out.mean(axis=0)
            shift = float(np.abs(new_ref - ref).max())
            ref = new_ref
            if shift < wop_tol:
                break
        meta["wop_iterations"] = it + 1
        meta["wop_final_shift"] = shift
    else:
        matcher = _greedy_match_columns if method == "matchalign" else _match_columns
        oblique_fn = promax_rotate if method == "promax" else None
        fallbacks = 0
        Phi_sum = np.zeros((K, K)) if oblique_fn else None
        for s in range(S):
            if oblique_fn:
                try:
                    Tl, Tt, Phi_s = oblique_fn(A[s])
                    L, th_s = A[s] @ Tl, theta[s] @ Tt
                except Exception:
                    # Degenerate draw (singular solve/lstsq); fall back to the
                    # orthogonal rotation for this draw and count it.
                    fallbacks += 1
                    L, R = _varimax(A[s])
                    th_s, Phi_s = theta[s] @ R, np.eye(K)
            else:
                L, R = _varimax(A[s])
                th_s = theta[s] @ R
            perm, signs = matcher(L, ref)
            P = _signed_perm_matrix(perm, signs)
            A_out[s] = L @ P
            th_out[s] = th_s @ P
            if Phi_sum is not None:
                Phi_sum += P.T @ Phi_s @ P
        if oblique_fn:
            meta[f"{method}_fallbacks"] = fallbacks

    if method == "promax":
        Phi = Phi_sum / S

    # Presentation convention (one GLOBAL signed permutation, same for every
    # draw, so it can't re-introduce per-draw wobble): axis1 = largest aligned
    # loading energy, and each axis oriented so its strongest benchmark loads
    # positive — matching the rank/orientation conventions of the other fits.
    mean_load = A_out.mean(axis=0)                        # (B, K)
    order = np.argsort(-(mean_load ** 2).sum(axis=0))
    A_out, th_out, ref = A_out[:, :, order], th_out[:, :, order], ref[:, order]
    mean_load = mean_load[:, order]
    signs = np.array([np.sign(mean_load[np.argmax(np.abs(mean_load[:, k])), k]) or 1.0
                      for k in range(K)])
    A_out *= signs[None, None, :]
    th_out *= signs[None, None, :]
    ref = ref * signs[None, :]
    if Phi is not None:
        Phi = Phi[np.ix_(order, order)] * np.outer(signs, signs)

    # eta invariance spot-check (same spirit as promax_rotate's assert).
    for s in (0, S // 2, S - 1):
        assert np.allclose(A_out[s] @ th_out[s].T, A[s] @ theta[s].T, atol=1e-6), \
            f"alignment ({method}) broke eta invariance at draw {s}"

    return AlignResult(A=A_out, theta=th_out, ref=ref, method=method,
                       Phi=Phi, meta=meta or None)


def _aligned_reproducibility(theta_aligned: np.ndarray, C: int) -> list[float]:
    """crosschain_axis_reproducibility on PRE-ALIGNED draws: axes are already
    comparable across chains (alignment did the ranking/matching), so we just
    correlate each chain-pair's mean ability per axis. Median |corr| per axis."""
    S, M, K = theta_aligned.shape
    th = theta_aligned.reshape(C, S // C, M, K)
    per_chain = [th[c].mean(axis=0) for c in range(C)]          # (M, K) each
    out = []
    for k in range(K):
        pairs = [abs(np.corrcoef(per_chain[i][:, k], per_chain[j][:, k])[0, 1])
                 for i in range(C) for j in range(i + 1, C)]
        out.append(float(np.median(pairs)))
    return out


def permutation_matched_reproducibility(A_rotated: np.ndarray, C: int) -> list[float]:
    """Per-axis cross-chain reproducibility for a FREE non-negative ('normal')
    fit, whose axes are identified only up to PERMUTATION.

    crosschain_axis_reproducibility ranks axes by tau_A before correlating, but
    the 'normal' prior uses ONE shared scalar tau_A (all axes equal), so that
    ranking is arbitrary and the metric reads ~0 even when every chain found the
    same axes. Here each chain's mean loadings (already in the pooled rotation
    frame, so a single global rotation is applied, but per-chain label-switching
    remains) are permutation+sign matched to the pooled mean, then correlated
    across chain pairs. Median |corr| per axis (display order). Loadings, not
    abilities: the non-negative bundle is defined in loading space."""
    from itertools import permutations
    S, B, K = A_rotated.shape
    A = A_rotated.reshape(C, S // C, B, K)
    ref = A_rotated.mean(axis=0)                                # (B, K) pooled mean
    per_chain = []
    for c in range(C):
        Lc = A[c].mean(axis=0)                                  # (B, K)
        best = None
        for p in permutations(range(K)):
            cols, score = [], 0.0
            for a in range(K):
                r = np.corrcoef(ref[:, a], Lc[:, p[a]])[0, 1]
                cols.append(np.sign(r) * Lc[:, p[a]])
                score += abs(r)
            if best is None or score > best[0]:
                best = (score, np.column_stack(cols))
        per_chain.append(best[1])                              # (B, K) ref order
    out = []
    for k in range(K):
        pairs = [abs(np.corrcoef(per_chain[i][:, k], per_chain[j][:, k])[0, 1])
                 for i in range(C) for j in range(i + 1, C)]
        out.append(float(np.median(pairs)))
    return out


def alignment_report(idata, data: ECIData,
                     methods: tuple = ("varimax", "wop", "matchalign", "promax"),
                     hdi: tuple = (3, 97)) -> dict:
    """Run the alignment-method comparison on a signed MIRT trace.

    Per method: aligned draws → per-loading median/HDI + SIGN-CONFIDENCE counts
    per axis (n loadings whose HDI excludes 0, split by sign — the decision
    metric for 'are contrast axes real?'), cross-chain reproducibility on the
    aligned abilities, and r-hat on the aligned loadings. The aligned r-hat is
    a diagnostic, not a convergence proof — the reference is shared across
    chains (mild double-dipping); identified r-hat (eta/D/sigma_b) stays the
    honest sampling verdict and is untouched by any of this.

    Also: the method-agreement matrix — per axis, |corr| between each method
    pair's mean aligned loading vectors (columns matched first, so a pure
    relabeling doesn't read as disagreement) — and per-chain divergence
    fractions (the stuck-chain check).

    Returns {"methods": {name: {...}}, "agreement": DataFrame,
             "per_chain_divergence_frac": list}.
    """
    # Function-level import: factors.mirt_factors_from_trace depends on
    # rotation.canonicalize_factors, so a module-level import here would
    # be circular.
    from analysis.factors import loadings_table, mirt_factors_from_trace

    post = idata.posterior
    C = int(post.sizes.get("chain", 1))
    A, theta, _ = mirt_factors_from_trace(idata, rank_track=False)
    S, B, K = A.shape
    bench_names = data.blookup["benchmark"].tolist()
    cats = (data.blookup["category"].tolist()
            if "category" in data.blookup.columns else None)

    shared_ref = _default_align_ref(A, theta) if K > 1 else None
    out: dict = {"methods": {}}
    mean_loads: dict[str, np.ndarray] = {}
    for m in methods:
        res = align_rotations(A, theta, method=m, ref=shared_ref)
        lo, hi = np.percentile(res.A, hdi, axis=0)             # (B, K) each
        med = np.median(res.A, axis=0)
        df = loadings_table(res.A, bench_names, cats, hdi=hdi)
        df["sign_confident"] = ((df["hdi_low"] > 0) | (df["hdi_high"] < 0))
        sign_counts = {
            f"axis{k + 1}": {"n_pos_confident": int((lo[:, k] > 0).sum()),
                             "n_neg_confident": int((hi[:, k] < 0).sum())}
            for k in range(K)}
        A_cd = res.A.reshape(C, S // C, B, K)
        rhat = az.rhat(xr.DataArray(A_cd, dims=("chain", "draw", "bench", "latent")))
        entry = {
            "loadings": df,
            "mean_loadings": med,
            "sign_counts": sign_counts,
            "aligned_max_rhat_A": float(rhat.to_array().max()),
            "reproducibility": (_aligned_reproducibility(res.theta, C)
                                if C >= 2 else None),
            "meta": res.meta,
        }
        if res.Phi is not None:
            entry["Phi"] = res.Phi
        out["methods"][m] = entry
        mean_loads[m] = res.A.mean(axis=0)

    rows = []
    names = list(methods)
    for i, m1 in enumerate(names):
        for m2 in names[i + 1:]:
            L2 = mean_loads[m2]
            if K > 1:
                perm, signs = _match_columns(L2, mean_loads[m1])
                L2 = L2 @ _signed_perm_matrix(perm, signs)
            for k in range(K):
                r = abs(np.corrcoef(mean_loads[m1][:, k], L2[:, k])[0, 1])
                rows.append({"method_a": m1, "method_b": m2,
                             "axis": f"axis{k + 1}", "abs_corr": float(r)})
    out["agreement"] = pd.DataFrame(rows)

    if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats:
        div = idata.sample_stats["diverging"].values
        out["per_chain_divergence_frac"] = [float(f) for f in div.mean(axis=1)]
    return out


def crosschain_axis_reproducibility(idata, signed: bool = False):
    """Per-axis agreement between chains: do independent chains find the SAME
    axes, or does each run invent its own?

    For each chain, rank its axes strong→weak by mean tau_A, take the per-chain
    mean ability on each ranked axis, then correlate those across every chain
    pair. Median |corr| ≈ 1 means every chain found the same axis; ≈ 0 means
    the axis direction is run-specific noise. This is the diagnosis the
    identified r-hat can't give: r-hat says THAT chains disagree, this says
    WHICH axis they disagree about.

    `signed`: resolve the per-draw ± first via align_factor_signs —
    unaligned signed draws would decorrelate trivially.

    Returns a list of median |corr| per ranked axis (strongest first), or None
    for single-chain traces.
    """
    post = idata.posterior
    if post.sizes.get("chain", 1) < 2 or "tau_A" not in post:
        return None
    tau = post["tau_A"].values                       # (C, S, K)
    th = post["theta"].values                        # (C, S, M, K)
    C, S, M, K = th.shape
    if signed:
        A = post["A"].values
        A_f, th_f, _ = align_factor_signs(A.reshape(C * S, -1, K),
                                          th.reshape(C * S, M, K))
        th = th_f.reshape(C, S, M, K)
    per_chain = []
    for c in range(C):
        order = np.argsort(-tau[c].mean(axis=0))
        per_chain.append(th[c].mean(axis=0)[:, order])   # (M, K), strong→weak
    out = []
    for k in range(K):
        pairs = [abs(np.corrcoef(per_chain[i][:, k], per_chain[j][:, k])[0, 1])
                 for i in range(C) for j in range(i + 1, C)]
        out.append(float(np.median(pairs)))
    return out


def _varimax(L: np.ndarray, gamma: float = 1.0,
             max_iter: int = 200, tol: float = 1e-9):
    """Orthogonal varimax rotation of a (B, K) loading matrix.

    Returns (rotated loadings, K×K orthogonal rotation R) with L_rot = L @ R.
    Used only as the warm start for promax — the final rotation is oblique.
    """
    B, K = L.shape
    R = np.eye(K)
    d = 0.0
    for _ in range(max_iter):
        Lam = L @ R
        grad = L.T @ (Lam**3 - (gamma / B) * Lam @ np.diag((Lam**2).sum(axis=0)))
        u, s, vt = np.linalg.svd(grad)
        R = u @ vt
        d_new = float(s.sum())
        if d != 0.0 and abs(d_new - d) < tol:
            break
        d = d_new
    return L @ R, R


def promax_rotate(L: np.ndarray, kappa: int = 4):
    """Oblique (promax) rotation of a (B, K) loading matrix.

    Promax = varimax warm start, then chase an exaggerated 'power' target
    (each loading raised to `kappa`, sign-preserved) by an UNCONSTRAINED linear
    map — dropping varimax's 90°-between-axes rule so the axes may tilt to line
    up with correlated bundles. Higher kappa = more aggressive simple structure
    (more correlated axes); 4 is the conventional default.

    Returns (Tload, Ttheta, Phi):
      * Tload (K, K): loading transform.  L_rot = L @ Tload.
      * Ttheta (K, K): ability transform = inv(Tload).T, so that
        L_rot @ theta_rot.T == L @ theta.T for theta_rot = theta @ Ttheta
        (predictions invariant — the rotation only relabels axes).
      * Phi (K, K): factor correlation matrix (unit diagonal).
    """
    Lv, R = _varimax(L)
    target = np.sign(Lv) * np.abs(Lv) ** kappa           # power target
    Q = np.linalg.lstsq(Lv, target, rcond=None)[0]       # Lv @ Q ≈ target
    scale = np.sqrt(np.diag(np.linalg.inv(Q.T @ Q)))
    Q = Q * scale[None, :]
    Tload = R @ Q
    Phi = np.linalg.inv(Q.T @ Q)
    Ttheta = np.linalg.inv(Tload).T
    rng = np.random.default_rng(0)
    th = rng.standard_normal((5, L.shape[1]))
    assert np.allclose((L @ Tload) @ (th @ Ttheta).T, L @ th.T, atol=1e-8), \
        "promax rotation broke prediction invariance"
    return Tload, Ttheta, Phi


def _geomin_vgq(L: np.ndarray, eps: float):
    """Geomin criterion value + gradient (Browne 2001). f = sum over benchmarks
    of the geometric mean of the squared loadings (+eps): smallest when every
    benchmark loads few axes — a per-ROW simple-structure penalty, unlike
    varimax's per-column one. eps keeps the log finite at exact zeros."""
    K = L.shape[1]
    L2 = L ** 2 + eps
    pro = np.exp(np.log(L2).sum(axis=1) / K)          # (B,) row geometric means
    f = float(pro.sum())
    Gq = (2.0 / K) * (L / L2) * pro[:, None]
    return f, Gq


def geomin_rotate(L: np.ndarray, eps: float = 0.01,
                  max_iter: int = 500, tol: float = 1e-8):
    """Oblique geomin rotation of a (B, K) loading matrix — the modern default
    oblique criterion (Mplus/ESEM), replacing promax's chase-a-power-target
    heuristic with a directly optimized objective.

    Gradient-projection algorithm (Bernaards & Jennrich 2005, GPFoblq),
    warm-started from varimax (geomin is prone to local minima from cold
    starts; the varimax start makes the per-draw use deterministic).
    Returns (Tload, Ttheta, Phi) with the same contract as promax_rotate:
    L_rot = L @ Tload, theta_rot = theta @ Ttheta, predictions invariant,
    Phi = factor correlation (how much the tilted axes overlap)."""
    Lv, R = _varimax(L)
    K = L.shape[1]
    T = np.eye(K)
    Ti = np.eye(K)
    Lr = Lv
    f, Gq = _geomin_vgq(Lr, eps)
    G = -(Lr.T @ Gq @ Ti).T
    al = 1.0
    for _ in range(max_iter):
        Gp = G - T * (T * G).sum(axis=0, keepdims=True)   # project onto constraint
        s = float(np.sqrt((Gp ** 2).sum()))
        if s < tol:
            break
        al *= 2.0
        for _ls in range(20):
            Tt = T - al * Gp
            Tt = Tt / np.sqrt((Tt ** 2).sum(axis=0, keepdims=True))  # unit columns
            Ti_t = np.linalg.inv(Tt)
            Lt = Lv @ Ti_t.T
            ft, Gq_t = _geomin_vgq(Lt, eps)
            if ft < f - 0.5 * s ** 2 * al:
                break
            al /= 2.0
        T, Ti, Lr, f, Gq = Tt, Ti_t, Lt, ft, Gq_t
        G = -(Lr.T @ Gq @ Ti).T
    Phi = T.T @ T
    Tload = R @ np.linalg.inv(T).T
    Ttheta = np.linalg.inv(Tload).T
    rng = np.random.default_rng(0)
    th = rng.standard_normal((5, K))
    assert np.allclose((L @ Tload) @ (th @ Ttheta).T, L @ th.T, atol=1e-8), \
        "geomin rotation broke prediction invariance"
    return Tload, Ttheta, Phi


def apply_rotation(A: np.ndarray, theta: np.ndarray, Phi: np.ndarray,
                   Tload: np.ndarray, Ttheta: np.ndarray):
    """Rotate every draw, then orient and order the axes for presentation.

    A (S,B,K), theta (S,M,K) are the rank-tracked draws; (Tload, Ttheta, Phi)
    come from promax_rotate on the posterior-mean loadings. After rotating we:
      * orient each axis so its loadings sum positive (rotation can flip an
        axis; we want bundles to read as positive 'good-at-these'), and
      * order axes by total loading energy (sum of squared mean loadings),
        descending, so axis1 is the largest bundle.
    Phi is permuted/sign-flipped to match. Returns (A_rot, theta_rot, Phi).
    """
    A_rot = A @ Tload                                    # (S,B,K)
    theta_rot = theta @ Ttheta                           # (S,M,K)
    mean_load = A_rot.mean(axis=0)                       # (B,K)

    signs = np.where(mean_load.sum(axis=0) < 0.0, -1.0, 1.0)   # (K,)
    A_rot *= signs[None, None, :]
    theta_rot *= signs[None, None, :]
    Phi = Phi * np.outer(signs, signs)

    energy = (A_rot.mean(axis=0) ** 2).sum(axis=0)       # (K,)
    order = np.argsort(-energy)
    A_rot = A_rot[:, :, order]
    theta_rot = theta_rot[:, :, order]
    Phi = Phi[np.ix_(order, order)]
    return A_rot, theta_rot, Phi


def factor_corr_df(Phi: np.ndarray) -> pd.DataFrame:
    """Factor-correlation matrix Phi as a labelled K×K DataFrame.

    Off-diagonals are how strongly the (oblique) bundles co-vary across models —
    e.g. 0.6 between the capability and agentic axes. Orthogonal rotations force
    these to 0; the oblique rotation measures them instead.
    """
    K = Phi.shape[0]
    axes = [f"axis{k + 1}" for k in range(K)]
    return pd.DataFrame(Phi, index=axes, columns=axes).round(4)

