"""Model builders: one module per structural family.

- mirt: compensatory K-axis Beta-MIRT (loading priors: normal / pt1 / signed / bifactor)
- mirt_nc: non-compensatory (conjunctive product link)
- mirt_sparse: sparse-gate conjunctive (horseshoe on the gates)
- mirt_interaction: compensatory + pairwise ability interactions
- qmatrix: category -> skill-axis maps for the Q-matrix fits
"""
from multiaxis_eci.models.mirt import build_mirt_model
from multiaxis_eci.models.mirt_interaction import INTERACTION_SCALE, build_mirt_interaction_model
from multiaxis_eci.models.mirt_nc import build_mirt_nc_model
from multiaxis_eci.models.mirt_sparse import build_mirt_sparse_model
from multiaxis_eci.models.qmatrix import QMATRIX_VARIANTS, axes_as_list

__all__ = [
    "build_mirt_model",
    "build_mirt_nc_model",
    "build_mirt_sparse_model",
    "build_mirt_interaction_model",
    "INTERACTION_SCALE",
    "QMATRIX_VARIANTS",
    "axes_as_list",
]
