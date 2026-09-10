"""Axis identity: which fitted axis carries which meaning, and what it is called.

A fit ranks its axes by loading energy in its own order and a refit on new data can reorder,
merge or split them, so no axis title is ever hard-coded. Each fit folder carries an
`axis_names.json` written by `propose_axis_names` (a template listing every axis's top
benchmarks, no titles, `confirmed: false`); a person reads the loadings, fills in the titles
and the signature benchmarks, and sets `confirmed: true`. Until then every figure calls the
axes `Axis 1`, `Axis 2`, ... (`load_axis_titles`), and the post's scripts refuse to run
(`require_axis_titles`). A confirmed title is applied only while the axis still carries its
signature (one of the signature benchmarks among its highest-share benchmarks), so a title
can never migrate to another axis unnoticed.
"""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

AXIS_NAMES_FILE = "axis_names.json"


def axis_top_benchmarks(view, data, top_n: int = 5) -> dict[str, list[str]]:
    """Per axis name, the `top_n` benchmarks by axis share (median loadings)."""
    A = view.require_A()
    bench = data.blookup.sort_values("benchmark_idx")["benchmark"].tolist()
    med = np.median(A, axis=0)
    share = med ** 2 / np.maximum((med ** 2).sum(axis=1, keepdims=True), 1e-12)
    return {name: [bench[b] for b in np.argsort(-share[:, k])[:top_n]]
            for k, name in enumerate(view.names)}


def bare_axis_title(name: str) -> str:
    """`axis3` -> `Axis 3`; anything else is returned unchanged."""
    return f"Axis {name[4:]}" if name.startswith("axis") and name[4:].isdigit() else name


def axis_names_path(results_dir: Path) -> Path:
    return Path(results_dir) / AXIS_NAMES_FILE


def propose_axis_names(view, data, results_dir: Path, top_n: int = 8) -> Path:
    """Write the naming template beside a fit's trace when none exists yet, and return its path.

    The template lists each axis's `top_n` benchmarks by share, with empty `title` and
    `signature` fields and `confirmed: false`. An existing file is never touched: it may hold
    hand-written names. The step runs at the end of every fit and plot so a new fit always
    has a file to fill in.
    """
    path = axis_names_path(results_dir)
    if path.exists():
        return path
    tops = axis_top_benchmarks(view, data, top_n)
    doc = {
        "confirmed": False,
        "how": ("Read the loadings (loadings_per_axis figure, mirt_loadings.csv), give each "
                "axis a short title (title_fr for the French renders) and 2-3 signature benchmarks "
                "among its top benchmarks, then set confirmed to true. Figures title the axes "
                "'Axis k' until then."),
        "axes": {name: {"title": None, "title_fr": None, "signature": [], "top_benchmarks": top}
                 for name, top in tops.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print(f"  axis names: template written to {path}; name the axes by hand to title figures")
    return path


def read_axis_names(results_dir: Path) -> dict | None:
    path = axis_names_path(results_dir)
    return json.loads(path.read_text()) if path.exists() else None


def _titles_from(doc: dict | None, names: list[str], tops: dict[str, list[str]] | None,
                 strict: bool) -> dict[str, str]:
    out = {n: bare_axis_title(n) for n in names}
    if not doc or not doc.get("confirmed"):
        if strict:
            raise SystemExit("axis names are not confirmed: fill in axis_names.json beside the "
                             "trace (titles, signatures, confirmed: true) before labelling")
        return out
    for name in names:
        entry = doc.get("axes", {}).get(name) or {}
        title, sig = entry.get("title"), set(entry.get("signature") or [])
        if not title:
            continue
        if tops is not None and sig and not sig & set(tops.get(name, [])):
            msg = (f"axis identity check failed for {name} ({title!r}): top benchmarks "
                   f"{tops.get(name)} contain none of {sorted(sig)}")
            if strict:
                raise SystemExit(msg + ". Refusing to label.")
            print(f"  WARNING: {msg}; titled {bare_axis_title(name)!r}")
            continue
        out[name] = f"{bare_axis_title(name)}: {title}"
    return out


def load_axis_titles(results_dir: Path, view=None, data=None, top_n: int = 5) -> dict[str, str]:
    """Display titles for a fit's axes: `Axis k: <title>` for every confirmed, hand-named axis
    whose signature still holds, `Axis k` otherwise. With `view` and `data` the signature is
    checked against the current loadings; without them the file is trusted as written."""
    names = list(view.names) if view is not None else [
        n for n in (read_axis_names(results_dir) or {}).get("axes", {})]
    tops = axis_top_benchmarks(view, data, top_n) if view is not None and data is not None \
        else None
    return _titles_from(read_axis_names(results_dir), names, tops, strict=False)


def require_axis_titles(results_dir: Path, view=None, data=None,
                        top_n: int = 5) -> dict[str, str]:
    """`load_axis_titles` for publication: SystemExit unless the names are confirmed and, when
    `view` and `data` are given, every titled axis still carries its signature."""
    names = list(view.names) if view is not None else [
        n for n in (read_axis_names(results_dir) or {}).get("axes", {})]
    tops = axis_top_benchmarks(view, data, top_n) if view is not None and data is not None \
        else None
    if tops is not None:
        for name, top in tops.items():
            print(f"  {name}: {top}")
    return _titles_from(read_axis_names(results_dir), names, tops, strict=True)


def align_to_reference_loadings(view, data, loadings_csv: Path):
    """Permute a chain subset's axes onto the whole fit's display frame.

    `prepare_fit` ranks the axes by loading energy WITHIN whatever draws it is given, and a
    chain group ranks them in its own order. Matching each column to the fit-level
    `mirt_loadings.csv` medians by correlation puts panel k back on the axis panel k carries
    everywhere else (the axis names file included). Raises SystemExit when the match is not a
    permutation: two subset axes closest to the same reference axis means the subset does not
    carry the fit's solution, and no relabelling would make its figures comparable.
    """
    pooled = (pd.read_csv(loadings_csv)
              .pivot(index="benchmark", columns="axis", values="loading_median"))
    bench = list(data.blookup.sort_values("benchmark_idx")["benchmark"])
    P = pooled.loc[bench, sorted(pooled.columns)].values
    M = np.median(view.require_A(), axis=0)
    corr = np.corrcoef(P.T, M.T)[:P.shape[1], P.shape[1]:]
    perm = corr.argmax(axis=1)
    if sorted(perm) != list(range(P.shape[1])):
        raise SystemExit(f"axis match is not a permutation: {perm}\n{corr.round(3)}")
    print("  display axis -> subset column: "
          + ", ".join(f"{k + 1}->{s + 1} (r {corr[k, s]:+.2f})" for k, s in enumerate(perm)))
    changes = {"theta": view.theta[:, :, perm], "A": view.A[:, :, perm]}
    for field in ("Phi", "Phi_raw"):
        m = getattr(view, field, None)
        if isinstance(m, np.ndarray) and m.shape == (len(perm), len(perm)):
            changes[field] = m[np.ix_(perm, perm)]
    tau = getattr(view, "tau", None)
    if isinstance(tau, np.ndarray) and tau.ndim >= 1 and tau.shape[-1] == len(perm):
        changes["tau"] = tau[..., perm]
    return dataclasses.replace(view, **changes)


def axis_title_translations(results_dir: Path) -> list[tuple[str, str]]:
    """(English title, French title) pairs for the French renders, from the optional
    `title_fr` field of `axis_names.json`: `Axis 2: Domain Knowledge` -> `Axe 2 : Connaissances
    de domaine`. Empty when the file is missing, unconfirmed or carries no French titles."""
    doc = read_axis_names(results_dir)
    if not doc or not doc.get("confirmed"):
        return []
    out = []
    for name, entry in doc.get("axes", {}).items():
        title, fr = entry.get("title"), entry.get("title_fr")
        if title and fr:
            k = name[4:] if name.startswith("axis") else name
            out.append((f"{bare_axis_title(name)}: {title}", f"Axe {k} : {fr}"))
    return out
