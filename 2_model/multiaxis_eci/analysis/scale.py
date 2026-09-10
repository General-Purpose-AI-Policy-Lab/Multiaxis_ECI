"""The ability scale the figures show.

A fitted ability is a latent logit with an arbitrary origin and unit (the prior's, shifted by
the lineage links), so a value of 2.3 on an axis means nothing on its own. For display, every
axis is re-expressed in human units: the posterior median of the Average Human tier sits at 0
and that of the Top Performer tier at 1 (`config.HUMAN_UNIT_ANCHORS`), so a model at 0.5 is
halfway between the two on that axis and a slope of 0.3 per year means the frontier gains
three tenths of the human span a year.

The map is one affine transform per axis, fixed by the two anchor medians, applied to finished
summaries (timeline frames, forecast bands, forest rows): quantiles are equivariant under it,
so intervals, records and crossing dates keep their meaning, and the candidate filters
(`INFORMED_SD_CAP`, the SOTA rules) keep running on the fitted scale they were set for. It
touches the ability-side figures only, never the loadings, the item curves or the tables.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd

from multiaxis_eci.config import ABILITY_SCALE, HUMAN_UNIT_ANCHORS
from multiaxis_eci.data import ECIData

ABILITY_LABEL = {
    "en": f"ability ({HUMAN_UNIT_ANCHORS[0]} = 0, {HUMAN_UNIT_ANCHORS[1]} = 1)",
    "fr": "capacité (Humain moyen = 0, Meilleur performeur = 1)",
}
_FRAME_COLS = ("mean", "hdi_low", "hdi_high")


def human_unit_affine(theta: np.ndarray, data: ECIData):
    """Per-axis (origin, unit) arrays of shape (K,) such that `(x - origin) / unit` puts the
    Average Human median at 0 and the Top Performer median at 1; None when the scale is off,
    an anchor tier is not a test-taker of the fit, or the two medians are not ordered on some
    axis (the unit would flip)."""
    if ABILITY_SCALE != "human":
        return None
    names = data.mlookup.sort_values("model_idx")["model"].tolist()
    lo_name, hi_name = HUMAN_UNIT_ANCHORS
    if lo_name not in names or hi_name not in names:
        return None
    origin = np.median(theta[:, names.index(lo_name), :], axis=0)
    unit = np.median(theta[:, names.index(hi_name), :], axis=0) - origin
    if not (unit > 0).all():
        return None
    return origin, unit


def rescale_frame(df: pd.DataFrame, k: int, affine) -> pd.DataFrame:
    """A timeline / human / forest frame (mean, hdi_low, hdi_high) in human units on axis k."""
    if affine is None or df is None or len(df) == 0:
        return df
    origin, unit = affine
    out = df.copy()
    for c in _FRAME_COLS:
        if c in out:
            out[c] = (out[c] - origin[k]) / unit[k]
    return out


def rescale_forecast(fc, k: int, affine):
    """A ForecastResult in human units on axis k: band and median shifted and scaled, slopes
    divided by the unit; dates and names untouched."""
    if affine is None:
        return fc
    origin, unit = affine
    f = lambda x: None if x is None else (np.asarray(x) - origin[k]) / unit[k]   # noqa: E731
    changes = {"lo": f(fc.lo), "median": f(fc.median), "hi": f(fc.hi),
               "slope": np.asarray(fc.slope) / unit[k],
               "intercept": f(fc.intercept)}
    for extra in ("slope_early",):
        if getattr(fc, extra, None) is not None:
            changes[extra] = np.asarray(getattr(fc, extra)) / unit[k]
    return dataclasses.replace(fc, **changes)


def rescale_theta(theta: np.ndarray, affine) -> np.ndarray:
    """Draws (S, M, K) in human units; for figures that take draws rather than summaries."""
    if affine is None:
        return theta
    origin, unit = affine
    return (theta - origin[None, None, :]) / unit[None, None, :]


def ability_label(affine, lang: str = "en", base: str | None = None) -> str:
    """The ability axis caption: the human-unit label when `affine` is set, else `base`."""
    if affine is not None:
        return ABILITY_LABEL[lang]
    return base or ("ability" if lang == "en" else "capacité")
