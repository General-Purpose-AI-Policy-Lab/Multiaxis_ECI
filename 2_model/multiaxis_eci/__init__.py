"""The ECI Beta-MIRT library: model builders, data loading, analysis and figures.

Entry points live outside this package. `3_fit/fit.py` at the repository root runs a
fit; `4_diagnostics/` holds the post-fit command-line tools.
"""

from multiaxis_eci import pytensor_compat as _pytensor_compat

# Xcode 27's linker rejects the `-ld64` flag PyTensor adds on macOS; strip it when refused.
_pytensor_compat.apply()
