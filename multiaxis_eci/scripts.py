"""Load a numbered entry-point script as a module.

The reproduction-path scripts carry an order prefix (`2_fit.py`,
`3_diagnostics/4_build_dashboard.py`) so the sequence is visible in a directory
listing. A Python module name cannot start with a digit and neither can a
package directory, so `import` cannot reach them: the prefix buys readability at
the cost of importability.

Library code belongs in this package, where it stays importable. The handful of
places where one entry point genuinely needs another's helper go through here.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from multiaxis_eci.config import PROJECT_ROOT

_cache: dict[str, object] = {}


def load(relpath: str):
    """Import `relpath` (relative to the repository root) and return the module.

    Cached, so two callers loading the same script share one module object and
    its module-level state, the way a normal import would.
    """
    if relpath in _cache:
        return _cache[relpath]
    path = Path(PROJECT_ROOT) / relpath
    if not path.exists():
        raise FileNotFoundError(f"no entry-point script at {path}")
    name = "_eci_script_" + path.stem.lstrip("0123456789_")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # so dataclasses/pickle inside it resolve
    spec.loader.exec_module(mod)
    _cache[relpath] = mod
    return mod
