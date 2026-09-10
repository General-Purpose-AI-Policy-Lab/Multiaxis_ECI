"""Axis identity: which fitted axis carries which meaning.

The display titles of `config.AXIS_TITLES` are keyed by position (`axis1`..`axis4`) and describe
the published flagship fit. A refit ranks its axes by loading energy in its own order, so a
title is applied only when the axis's highest-share benchmarks contain one of the signature
benchmarks of `config.AXIS_SIGNATURES`; a figure that would otherwise mislabel an axis falls
back to the bare name (`axis_titles_for`) or refuses to draw (`check_axis_identity`).
"""
from __future__ import annotations

import numpy as np

from multiaxis_eci.config import AXIS_SIGNATURES, AXIS_TITLES


def axis_top_benchmarks(view, data, top_n: int = 5) -> dict[str, list[str]]:
    """Per axis name, the `top_n` benchmarks by axis share (median loadings)."""
    A = view.require_A()
    bench = data.blookup.sort_values("benchmark_idx")["benchmark"].tolist()
    med = np.median(A, axis=0)
    share = med ** 2 / np.maximum((med ** 2).sum(axis=1, keepdims=True), 1e-12)
    return {name: [bench[b] for b in np.argsort(-share[:, k])[:top_n]]
            for k, name in enumerate(view.names)}


def axis_titles_for(view, data, top_n: int = 5) -> dict[str, str]:
    """`config.AXIS_TITLES` for the axes whose identity matches, the bare name otherwise."""
    tops = axis_top_benchmarks(view, data, top_n)
    out = {}
    for name, top in tops.items():
        matches = bool(AXIS_SIGNATURES.get(name, set()) & set(top))
        out[name] = AXIS_TITLES[name] if matches and name in AXIS_TITLES else name
    return out


def check_axis_identity(view, data, top_n: int = 5) -> dict[str, list[str]]:
    """Print each axis's top benchmarks and raise SystemExit if one axis does not carry the
    identity its title claims. Returns the top-benchmark table.

    Raises:
        SystemExit: an axis's top-`top_n` benchmarks contain none of its signature benchmarks.
    """
    tops = axis_top_benchmarks(view, data, top_n)
    for name, top in tops.items():
        print(f"  {name} ({AXIS_TITLES.get(name, name)}): {top}")
        want = AXIS_SIGNATURES.get(name, set())
        if want and not want & set(top):
            raise SystemExit(
                f"axis identity check failed for {name}: top-{top_n} by share {top} contains "
                f"none of {sorted(want)}. Refusing to label.")
    return tops
