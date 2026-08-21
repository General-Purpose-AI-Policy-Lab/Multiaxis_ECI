# `results/mirt/` — MIRT scratch outputs (last run wins)

The loose `mirt_*.{csv,json}` and `fit_mirt.log` in this folder are
**transient**: every `diagnostics/fit_mirt.py` run overwrites them with *that*
run's artefacts, regardless of `--K` / `--anchors`. The `_k3` suffix means
"K = 3", **not** any particular K=3 fit — so these files reflect **whatever was
run last** (currently the cross-loaded `--anchors qmatrix3x` fit, R² ≈ 0.954,
not the confirmed strict Q-matrix fit).

Do **not** cite numbers from here as a specific result. Use the stable, named
snapshots instead:

- **`results/mirt_confirmed_k3/`** — the confirmed K=3 skill Q-matrix fit
  (R² 0.955, Φ 0.37 / 0.59 / 0.19). The canonical multi-axis result.
- **`results/mirt_nc/`** — the non-compensatory (conjunctive) fits, tagged per
  variant (`_k3`, `_k3_qm3x`, …).
- **`gallery/`** — every fit rendered from its own named trace;
  `gallery/comparisons/gof_table.csv` is the comparable cross-fit table.

Traces (`trace_mirt_k*.nc`) are gitignored and tagged per fit
(`trace_mirt_k3_aqmatrix3.nc` = confirmed, `_aqmatrix3x` = cross-loaded, …).
