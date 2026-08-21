"""Category → skill-axis Q-matrices for compensatory and non-comp MIRT fits.

Axis order: 0 reasoning/math · 1 agentic/coding · 2 knowledge/applied.
"""

AXIS_LABELS_K3 = ["reasoning/math", "agentic/coding", "knowledge/applied"]

# qmatrix3 — strict simple structure: each category loads exactly ONE axis.
# Values are axis indices (int), matching the compensatory anchor convention.
QMATRIX3_CAT_TO_AXIS = {
    "High End Math Reasoning": 0, "General Reasoning": 0, "Core AGI Progress": 0,
    "Autonomous SWE": 1, "Agentic Computer Use": 1,
    "Domain Specific Questions": 2, "Biology": 2, "Chemistry": 2,
    "Multimodal Understanding": 2, "Advanced Language and Writing": 2,
}

# qmatrix3x — cross-loaded: coding & PhD-science ALSO tap reasoning (axis 0), so
# those benchmarks are allowed to load two skills (a list of axis indices).
QMATRIX3X_CAT_TO_AXIS = {
    **QMATRIX3_CAT_TO_AXIS,
    "Autonomous SWE": [1, 0], "Agentic Computer Use": [1, 0],
    "Biology": [2, 0], "Chemistry": [2, 0],
}

QMATRIX_VARIANTS = {"qmatrix3": QMATRIX3_CAT_TO_AXIS, "qmatrix3x": QMATRIX3X_CAT_TO_AXIS}

# ── K=4 confirmatory Q-matrix (compensatory driver only) ────────────────────
# qmatrix3 with Multimodal Understanding pulled OUT of the knowledge axis into
# its own 4th axis (Biology/Chemistry stay in knowledge). Motivated by coverage,
# not benchmark count: Multimodal has only 6 benchmarks but ~174/750 models took
# at least one (23%), vs the Bio/Chem "science" split's 39/750 (5%) — so a
# multimodal axis is far better measured (fewer models extrapolated) than a
# science axis would be. Strict simple structure (one axis per category).
#
# Kept SEPARATE from QMATRIX_VARIANTS on purpose: the non-compensatory driver
# (fits/fit_nc.py) maps every entry of QMATRIX_VARIANTS onto its THREE axis
# names (AXES[3]); a 4th axis index (3) here would make that mapping raise. The
# compensatory driver merges this in locally instead.
AXIS_LABELS_K4 = ["reasoning/math", "agentic/coding", "knowledge/science", "multimodal"]

QMATRIX4_CAT_TO_AXIS = {
    "High End Math Reasoning": 0, "General Reasoning": 0, "Core AGI Progress": 0,
    "Autonomous SWE": 1, "Agentic Computer Use": 1,
    "Domain Specific Questions": 2, "Advanced Language and Writing": 2,
    "Biology": 2, "Chemistry": 2,
    "Multimodal Understanding": 3,
}

# qmatrix4x — cross-loaded K=4: qmatrix4 PLUS the same cross-loads qmatrix3x uses
# (coding & PhD-science ALSO tap reasoning, axis 0). Multimodal stays single
# (axis 3) — qmatrix3x never cross-loaded it, and it's now its own axis. Also kept
# OUT of QMATRIX_VARIANTS (its 4th axis index would break the non-comp 3-name map).
QMATRIX4X_CAT_TO_AXIS = {
    **QMATRIX4_CAT_TO_AXIS,
    "Autonomous SWE": [1, 0], "Agentic Computer Use": [1, 0],
    "Biology": [2, 0], "Chemistry": [2, 0],
}


def axes_as_list(v) -> list[int]:
    """Normalise an axis assignment (int or list of ints) to a list of ints."""
    return [v] if isinstance(v, int) else list(v)
