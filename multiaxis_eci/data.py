"""Data loading and preprocessing for the Beta IRT model.

Runtime source: 1_data/processed/benchmarks_merged.csv. This module does only the modeling-stage
transforms (humans, effort collapse, low-obs drop, indices).

"Normalized" in upstream filenames means scores are on the [0, 1] scale via
unit conversion (percentages → proportions, 0-10 → 0-1). It does NOT mean
chance-corrected: a 4-choice multiple-choice benchmark still has scores
clustered around 0.25. Per-benchmark chance floors are NOT carried in this
schema — their reviewed ground truth lives in
`1_data/curated/benchmark_lower_bounds.csv`, read directly by the `--floors`
3PL fit (`load_benchmark_floors`).

Model & benchmark integer indices are derived in-memory by sorting unique
`model_version` / `benchmark` values alphabetically.

The processed file holds every benchmark. Two curated lists filter it here:
  • 1_data/curated/retired_benchmarks.txt is dropped for every fit, with no flag
    over it, before any other scope logic.
  • 1_data/curated/excluded_benchmarks.txt is the "easy for humans / hard for
    machines" drop, a per-fit scope choice (include_all_benchmarks).
  • drop_low_obs_models / fit_humans / collapse_effort_variants — see
    load_eci_data.

Per-model observation counts are surfaced as `n_obs_per_model` and an
`is_low_obs` mask, so downstream plots can split well-observed models from
data-poor ones (default threshold LOW_OBS_THRESHOLD in config).
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from multiaxis_eci.config import DATA_DIR, ECI_EPS, LOW_OBS_THRESHOLD, ZERO_DIAG_THRESHOLD

# Subdirectory layout:
#   1_data/raw/        — reference ECI CSV (only used by --eci-data-only mode)
#   1_data/curated/    — user-maintained inputs (excluded list, human baselines)
#   1_data/processed/  — output of 1_data/1_pipeline/pipeline.ipynb
RAW_DIR       = DATA_DIR / "raw"
CURATED_DIR   = DATA_DIR / "curated"
PROCESSED_DIR = DATA_DIR / "processed"

# Re-run 1_data/1_pipeline/pipeline.ipynb to refresh this file.
PROCESSED_FILE = PROCESSED_DIR / "benchmarks_merged.csv"
# Shared canonical alias map (variant → canonical); applied by the pipeline
# notebook at data-prep time.
ALIASES_FILE = DATA_DIR / "pipeline" / "canonical" / "model_aliases.csv"


def _subset_obs(data: "ECIData", keep: np.ndarray) -> "ECIData":
    """New ECIData with only the observations where `keep` is True.

    Lookups and n_models are unchanged — a model whose every row is dropped
    stays in the index with a prior-only theta. is_low_obs keeps the
    load_eci_data definition (humans exempt from the flag: a tier whose rows
    were dropped stays a prior-only anchor, not a data-poor model to hide).
    Shared tail of the three observation-drop helpers below.
    """
    from dataclasses import replace
    new_n_per_model = np.bincount(data.model_idx[keep], minlength=data.n_models)
    return replace(
        data,
        scores=data.scores[keep],
        zero_score_mask=data.zero_score_mask[keep],
        model_idx=data.model_idx[keep],
        bench_idx=data.bench_idx[keep],
        n_eff=None if data.n_eff is None else data.n_eff[keep],
        n_obs=int(keep.sum()),
        n_obs_per_model=new_n_per_model,
        is_low_obs=(new_n_per_model < LOW_OBS_THRESHOLD) & ~data.is_human,
    )


def drop_zero_scores(data: "ECIData") -> "ECIData":
    """Return a new ECIData with all score==0 observations removed.

    Diagnostic / experimental use: lets us tell whether the zero-score rows
    are responsible for bad NUTS geometry (max tree depth + divergences +
    high r_hat). If r_hat drops dramatically without these points, the
    boundary clipping is the suspect; if it stays bad, the zeros are innocent.
    """
    return _subset_obs(data, ~data.zero_score_mask)


def drop_model_observations(data: "ECIData", names) -> "ECIData":
    """Return a new ECIData with the named test-takers' OBSERVATIONS removed.

    The takers STAY in the model index (n_models unchanged) — only their score
    rows are dropped, so their theta becomes prior-only. Used for the --no-sg
    fit: the Skilled Generalist human tier keeps its slot in the human-order /
    lineage priors, but its self-contradictory scores (0.98 ARC-AGI vs 0.22
    GPQA) no longer drive the likelihood. Names not present are ignored.
    """
    lookup = list(data.mlookup.sort_values("model_idx")["model"])
    drop_ids = {lookup.index(n) for n in names if n in lookup}
    keep = ~np.isin(data.model_idx, list(drop_ids))
    return _subset_obs(data, keep)


def drop_model_benchmark_cells(data: "ECIData", model_name, bench_names) -> "ECIData":
    """Return a new ECIData with specific (model, benchmark) CELLS removed.

    Finer-grained than drop_model_observations: only the named benchmarks are
    dropped for `model_name`; its other scores stay. Used for --no-sg-gpqa,
    which removes the Skilled Generalist's GPQA cells (the LOW side of its
    ARC-AGI-vs-GPQA straddle) while keeping ARC-AGI/GSM8K. The taker keeps its
    slot in the model index (prior-only on those axes it no longer measures).
    Names not present are ignored.
    """
    mlookup = list(data.mlookup.sort_values("model_idx")["model"])
    blookup = list(data.blookup.sort_values("benchmark_idx")["benchmark"])
    if model_name not in mlookup:
        return data
    m = mlookup.index(model_name)
    b_ids = {blookup.index(b) for b in bench_names if b in blookup}
    drop = (data.model_idx == m) & np.isin(data.bench_idx, list(b_ids))
    return _subset_obs(data, ~drop)


def _curated_name_list(filename: str) -> set[str]:
    """One benchmark name per line, '#' comments and blanks skipped. An absent
    file is an empty list, so a curated list is optional."""
    path = CURATED_DIR / filename
    if not path.exists():
        return set()
    names = (pd.read_csv(path, comment="#", header=None)[0].str.strip())
    return set(names[names != ""].tolist())


def load_excluded_benchmarks() -> set[str]:
    """The curated "easy for humans" exclusion list. A per-fit scope choice:
    `load_eci_data` drops it by default and `include_all_benchmarks=True` keeps
    it, so the list also travels on `ECIData` as metadata."""
    return _curated_name_list("excluded_benchmarks.txt")


def load_retired_benchmarks() -> set[str]:
    """Benchmarks no fit may use, dropped by `load_eci_data` before any other
    scope logic. Not a flag: a retired column cannot carry a comparable loading
    row at all (a superseded version pooled into one difficulty, a transformed
    score on a scale no other benchmark shares), so there is no configuration
    in which admitting it is valid. The processed file still holds every
    benchmark; the file states the per-benchmark reason."""
    return _curated_name_list("retired_benchmarks.txt")


BENCHMARK_FLOORS_FILE = CURATED_DIR / "benchmark_lower_bounds.csv"


def load_benchmark_floors(data: "ECIData") -> np.ndarray:
    """Per-benchmark chance floor c_b in blookup order, for the fixed-c 3PL fit.

    Reads the curated `benchmark_lower_bounds.csv` (benchmark, lower_bound,
    reason, source_url) and maps each floor onto the benchmark index order used
    by the model. A benchmark absent from the file gets 0.0 (an inert floor:
    the 3PL link collapses to the plain 2PL there) and is listed in a warning so
    coverage gaps are visible. Values must lie in [0, 1).
    """
    benchmarks = list(data.blookup.sort_values("benchmark_idx")["benchmark"])
    table = pd.read_csv(BENCHMARK_FLOORS_FILE)
    floor_by_bench = dict(zip(table["benchmark"], table["lower_bound"].astype(float)))
    missing = [b for b in benchmarks if b not in floor_by_bench]
    if missing:
        warnings.warn(
            f"{len(missing)} benchmark(s) have no curated chance floor "
            f"(using 0.0): {missing}", UserWarning)
    floors = np.array([floor_by_bench.get(b, 0.0) for b in benchmarks], dtype=np.float64)
    if not np.all((floors >= 0.0) & (floors < 1.0)):
        bad = [(b, f) for b, f in zip(benchmarks, floors) if not 0.0 <= f < 1.0]
        raise ValueError(f"benchmark floors must be in [0, 1): {bad}")
    return floors


BENCHMARK_CLIPS_FILE = CURATED_DIR / "benchmark_score_clips.csv"


def clip_scores_to_floors(data: "ECIData", floors: np.ndarray) -> "ECIData":
    """Return a new ECIData with the reviewed row-level clips applied.

    Companion to the fixed-c 3PL: a score below the chance floor c_b is
    measurement noise around chance, not evidence of sub-random ability, so it
    is raised to c_b. Without this the raw below-chance outliers (e.g. a
    0.054 on a 0.25-floor MCQ) sit far below the model's minimum possible mean
    and distort the per-benchmark noise sigma_b.

    The clips are APPLIED FROM `1_data/curated/benchmark_score_clips.csv` — the
    reviewed, explicit list of every (model, benchmark) score the fit modifies —
    so the record and the behaviour cannot diverge. Draft/refresh the file with
    `3_diagnostics/audit_lower_bounds.py --write-clips` after a data or floor
    change, review the diff, then commit it. The `floors` argument is used only
    to cross-check the file against the current data; two drifts warn loudly:
      * a below-floor observation with no row in the file (stale file — a data
        refresh surfaced a new below-chance score that was never reviewed);
      * a file row that matches no observation, or whose recorded
        original_score no longer matches the data.
    zero_score_mask is refreshed: a zero on a 0-floor benchmark stays a zero,
    so the boundary diagnostics are unchanged there.
    """
    from dataclasses import replace
    table = pd.read_csv(BENCHMARK_CLIPS_FILE)

    model_of = dict(zip(data.mlookup["model_idx"] - 1, data.mlookup["model"]))
    bench_of = dict(zip(data.blookup["benchmark_idx"] - 1, data.blookup["benchmark"]))
    obs_key = [(bench_of[b], model_of[m])
               for b, m in zip(data.bench_idx, data.model_idx)]
    row_by_key = {(r.benchmark, r.model_version): r for r in table.itertuples()}

    new_scores = data.scores.copy()
    matched: set[tuple[str, str]] = set()
    for i, key in enumerate(obs_key):
        r = row_by_key.get(key)
        if r is None:
            continue
        matched.add(key)
        if abs(data.scores[i] - r.original_score) > 1e-6:
            warnings.warn(
                f"benchmark_score_clips.csv: original_score for {key} is "
                f"{r.original_score} but the data has {data.scores[i]:.4f} — "
                f"the data changed since the clip was reviewed; row NOT applied. "
                f"Refresh with 3_diagnostics/audit_lower_bounds.py --write-clips.",
                UserWarning)
            continue
        new_scores[i] = r.clipped_score

    stale = set(row_by_key) - matched
    if stale:
        warnings.warn(
            f"benchmark_score_clips.csv: {len(stale)} row(s) match no current "
            f"observation (e.g. {sorted(stale)[:3]}) — refresh with "
            f"3_diagnostics/audit_lower_bounds.py --write-clips.", UserWarning)

    below = new_scores < floors[data.bench_idx] - 1e-9
    if below.any():
        missing = sorted({obs_key[i] for i in np.flatnonzero(below)})
        warnings.warn(
            f"{len(missing)} below-floor observation(s) have no reviewed clip "
            f"row and stay UNCLIPPED (e.g. {missing[:3]}) — draft rows with "
            f"3_diagnostics/audit_lower_bounds.py --write-clips and review.",
            UserWarning)

    return replace(
        data,
        scores=new_scores,
        zero_score_mask=new_scores == 0.0,
    )


# Human groups dropped before fitting (empty — all groups kept).
HUMAN_GROUPS_DROP: set[str] = set()


# Regex for effort-variant suffixes — strip everything after the last
# underscore that matches the known suffix vocabulary. `none` is a real setting
# upstream ("GPT-5.1 (Thinking, None)" -> gpt-5.1-2025-11-13_none), so it
# belongs here: 15 test-takers carry it, and leaving it out made every one of
# them its own BASE model, which silently defeated collapse_effort_variants and
# the lineage's variant collapse for those rows.
_EFFORT_SUFFIX_RE = re.compile(
    r"_(?:low|medium|high|minimal|max|xhigh|none|unknown|web-?app|"
    r"\d+K|reasoning-(?:low|medium|high))$"
)


def _effort_base(model_version: str) -> str:
    """Strip a trailing effort/context-length suffix to get the base model id."""
    return _EFFORT_SUFFIX_RE.sub("", str(model_version))


def _collapse_effort_variants(df: pd.DataFrame,
                                protected: set[str] | None = None) -> pd.DataFrame:
    """For each base model (e.g. 'gpt-5-2025-08-07'), keep one effort variant.
    Selection order:
       1. any variant in `protected` (SOTA / anchor list) wins.
       2. most-obs wins.
       3. tie-breaks among equally-observed variants:
            `_high` > bare base > `_medium` > alphabetical.
    """
    protected = protected or set()
    counts = df.groupby("model_version").size()
    df = df.copy()
    df["__base__"] = df["model_version"].map(_effort_base)

    def pick_winner(group: pd.Series) -> str:
        variants = group.unique()
        if len(variants) == 1:
            return variants[0]
        # Protected variants always win, regardless of obs count
        prot_in_group = [v for v in variants if v in protected]
        if prot_in_group:
            return sorted(prot_in_group)[0]
        # Most-obs wins; equally-tested variants tie-broken by suffix preference.
        def key(v):
            return (
                -counts[v],
                0 if v.endswith("_high") else 1,            # _high preferred on tie
                0 if v == group.name else 1,                # bare base next
                0 if v.endswith("_medium") else 1,
                v,
            )
        return sorted(variants, key=key)[0]

    winners = df.groupby("__base__")["model_version"].agg(pick_winner)
    keep = set(winners.values)
    return df[df["model_version"].isin(keep)].drop(columns="__base__").reset_index(drop=True)


def _known_release_date_by_model() -> pd.Series:
    """Best-known release date per model, from the FULL processed file plus the
    config.RELEASE_DATES fallback. Used by the era filters, which must see a
    model's date even when its dated rows sit on benchmarks the current fit
    excludes."""
    from multiaxis_eci.config import RELEASE_DATES
    full = pd.read_csv(PROCESSED_FILE)
    dates = pd.to_datetime(full["release_date"], errors="coerce")
    dates = dates.fillna(pd.to_datetime(full["model_version"].map(RELEASE_DATES),
                                        errors="coerce"))
    return dates.groupby(full["model_version"]).max()


def release_time_covariate(mlookup: pd.DataFrame, lineage=None) -> np.ndarray:
    """Centered release year per theta row: the covariate for the time prior
    (models/mirt.py, build_mirt_model(..., time_t=...)).

    Dates come from `_known_release_date_by_model` (processed file, then the
    config.RELEASE_DATES fill), so a model dated only on benchmarks this fit
    excludes still gets its date.

    Two rules make the covariate safe to add to the theta prior mean:

    - **Centered** over the dated rows. The shifts then sum to zero, so the trend
      cannot move the overall level, which is what the ZeroSumNormal exists to
      pin (theta reaches the likelihood only through A*theta - D). Undated rows,
      and humans (undated by nature), get exactly 0: their prior center stays at
      the era-average level it has today.
    - **Chained rows carry their chain FOUNDER's date**, descendants included, so
      the trend shifts a chain's base level only. The lineage increments keep
      measuring within-chain climb rather than re-charging for time the drift
      already prices, and being constant per chain it cancels in every
      within-chain difference.

    Returns years (float), one per model row, in `mlookup` row order.
    """
    if not np.array_equal(mlookup["model_idx"].to_numpy(),
                          np.arange(1, len(mlookup) + 1)):
        raise ValueError("release_time_covariate needs mlookup in theta-row order "
                         "(model_idx 1..n); got a reordered frame, which would "
                         "silently attach dates to the wrong models.")
    epoch = np.datetime64("2000-01-01")   # arbitrary origin; centering removes it
    dates = pd.to_datetime(mlookup["model"].map(_known_release_date_by_model()))
    years = ((dates - epoch) / np.timedelta64(1, "D") / 365.25).to_numpy(float, copy=True)
    if lineage is not None:
        chain_of_row = lineage.B[lineage.node_idx].argmax(1)
        founder_years = ((lineage.founder_date - epoch)
                         / np.timedelta64(1, "D") / 365.25)
        years[lineage.row_idx] = founder_years[chain_of_row]
    dated = ~np.isnan(years)
    if not dated.any():
        raise ValueError("time prior needs at least one dated model; none found.")
    years[dated] -= years[dated].mean()
    years[~dated] = 0.0
    return years


def _load_human_baselines_as_models() -> pd.DataFrame:
    """Read human_baselines.csv and shape it into dataset-compatible rows.

    Each (group, benchmark, score) becomes a row where `model_version` is the
    group name (e.g. "Average Human"). Fitted alongside AI models, the IRT
    naturally estimates each group's capability with full posterior + HDI —
    replaces the ad-hoc post-hoc median-of-implied-C aggregation.

    Boundary scores (exactly 0 or 1) are dropped to avoid Beta-likelihood
    pathologies — same convention as the current human_baseline_levels filter.
    """
    src = CURATED_DIR / "human_baselines.csv"
    if not src.exists():
        return pd.DataFrame()
    hb = pd.read_csv(src)
    # Guard against accidental row duplication. Human rows bypass the pipeline's
    # dedup step (which runs on the AI rows in the notebook), so a re-pasted block
    # in the CSV would otherwise be fit as repeated Beta observations — inflating
    # a group's apparent obs count and over-tightening its posterior. Dedup on the
    # full (benchmark, group, score) triple: exact repeats are dropped, but an
    # intentional second measurement at a *different* score (e.g. the two GPQA
    # Diamond expert baselines, 0.697 vs 0.812) is preserved.
    # Trim the tier and benchmark keys. A stray leading space forks a tier into a
    # second test-taker that HUMAN_ORDER does not name, so it silently loses the
    # ordered-human prior and takes its observations with it — invisible in every
    # diagnostic, since both names look identical when printed. Dedup and the
    # HUMAN_GROUPS_DROP anti-join both key off these strings, so trim first.
    hb["group"] = hb["group"].astype(str).str.strip()
    hb["benchmark"] = hb["benchmark"].astype(str).str.strip()
    hb = hb.drop_duplicates(subset=["benchmark", "group", "score"])
    hb = hb[~hb["group"].isin(HUMAN_GROUPS_DROP)]
    out = pd.DataFrame({
        "model_version": hb["group"].astype(str),
        "score":         hb["score"].astype(float),
        "release_date":  pd.NaT,
        "organization":  "Human baseline",
        "benchmark":     hb["benchmark"].astype(str),
        "stderr":        pd.NA,
        "source":        "human_baselines.csv",
        "category":      "Human baseline",
    })
    # Clamp into the open interval (ECI_EPS, 1 - ECI_EPS) rather than dropping.
    # Beta has open support, so a boundary value cannot be evaluated — but AI rows
    # at the boundary are CLIPPED and kept by the model builders, so dropping the
    # human equivalent is the inconsistent treatment, not the consistent one.
    # Two rows are affected (ARC-AGI-2 / Committee of Average Humans and VPCT /
    # Average Human, both at exactly 0.999); both survive as clipped rows now.
    out["score"] = out["score"].clip(ECI_EPS, 1.0 - ECI_EPS)
    return out[out["score"].between(0.0, 1.0)]


def _load_extra_benchmarks(machine_rows: pd.DataFrame, src: Path,
                           label: str) -> pd.DataFrame:
    """Read a curated extra-benchmark CSV (cyber ECI, SimpleQA original)
    as extra machine rows. The file carries the processed-file
    schema; each fit flag points at its own file.

    Rows survive only on models already in `machine_rows` and only on
    (model, benchmark) cells `machine_rows` does not already carry; the counts
    dropped by each of those two filters are printed. `release_date` and
    `organization` are copied from `machine_rows` to fill the concat schema and
    the file's own values are discarded. Nothing reads that date column: the era
    filter resolves dates from the full processed file and the lineage prior
    from `lineage_map.csv`.
    """
    if not src.exists():
        return pd.DataFrame()
    cb = pd.read_csv(src).drop_duplicates(subset=["model_version", "benchmark"])
    n_file = len(cb)
    # Models already fit. A test-taker seen only here carries no
    # cross-benchmark information.
    known = machine_rows.drop_duplicates("model_version").set_index("model_version")
    cb = cb[cb["model_version"].isin(known.index)]
    n_off_scope = n_file - len(cb)
    # Cells the pipeline already supplies stay with the pipeline.
    have = set(zip(machine_rows["model_version"], machine_rows["benchmark"]))
    cb = cb[[p not in have for p in zip(cb["model_version"], cb["benchmark"])]]
    n_shadowed = n_file - n_off_scope - len(cb)
    if n_off_scope or n_shadowed:
        print(f"   {label}: {n_file} rows in file, {n_off_scope} on models outside "
              f"the fit scope, {n_shadowed} already supplied by the pipeline")
    if cb.empty:
        return pd.DataFrame()
    out = pd.DataFrame({
        "model_version": cb["model_version"].astype(str),
        "score":         cb["score"].astype(float).clip(ECI_EPS, 1.0 - ECI_EPS),
        "release_date":  known.loc[cb["model_version"], "release_date"].values,
        "organization":  known.loc[cb["model_version"], "organization"].values,
        "benchmark":     cb["benchmark"].astype(str),
        "stderr":        pd.NA,
        "source":        cb["source"].astype(str),
        "category":      cb["category"].astype(str),
    })
    return out


@dataclass
class ECIData:
    scores: np.ndarray            # (n_obs,) raw observed scores (zeros preserved)
    zero_score_mask: np.ndarray   # (n_obs,) bool — True where score is exactly 0
    model_idx: np.ndarray         # (n_obs,) 0-indexed model id per observation
    bench_idx: np.ndarray         # (n_obs,) 0-indexed benchmark id per observation
    mlookup: pd.DataFrame         # columns: model, model_idx (1-indexed)
    blookup: pd.DataFrame         # columns: benchmark, benchmark_idx (1-indexed)
    n_models: int
    n_benchmarks: int
    n_obs: int
    zero_diag_threshold: float    # threshold used by ppc.py's zero-score calibration check
    n_obs_per_model: np.ndarray   # (n_models,) integer count of benchmark observations
    is_low_obs: np.ndarray        # (n_models,) bool — True if n_obs_per_model < LOW_OBS_THRESHOLD
    excluded_benchmarks: set[str] # benchmarks dropped from the source file
    is_human: np.ndarray          # (n_models,) bool — True for fitted human-group rows
    # Benchmark category per benchmark index (same order as blookup). Surfaced
    # for downstream metadata only — e.g. labelling MIRT factor loadings by
    # category (fit.py). The model never reads it. Defaults to
    # None so existing constructors (tests, replace()) don't have to supply it.
    bench_category: np.ndarray | None = None
    # (n_obs,) float — effective test length implied by the reported harness
    # stderr, np.inf where none is reported. Read only by the known_se
    # likelihood split (models/mirt.py); see load_eci_data for the conversion.
    # None on a hand-built ECIData, which is what known_se=True rejects.
    n_eff: np.ndarray | None = None
    # (n_models,) bool — True for models in config.SOTA_MODELS. Parallels
    # is_low_obs / is_human: lets plots ALWAYS show SOTA models (e.g. sparse new
    # frontier releases like Fable 5 / Mythos) even when their posterior is wide,
    # rather than dropping them as un-informed. Defaults to None so existing
    # constructors (tests, replace()) don't have to supply it.
    is_sota: np.ndarray | None = None


def load_eci_data(drop_low_obs_models: bool = False,
                   fit_humans: bool = True,
                   eci_data_only: bool = False,
                   collapse_effort_variants: bool = False,
                   include_all_benchmarks: bool = False,
                   fit_cyber: bool = False,
                   fit_simpleqa_original: bool = False,
                   drop_benchmarks: list[str] | None = None,
                   min_release_date: str | None = None,
                   max_release_date: str | None = None) -> ECIData:
    """Load the processed benchmark file and apply modeling-stage transforms.

    Data prep (Epoch ZIP fetch, RAND + Scale SEAL merge, dedup) is performed by
    `1_data/1_pipeline/pipeline.ipynb`, which writes the complete
    `1_data/processed/benchmarks_merged.csv` (every benchmark, no exclusions). Re-run
    that notebook after editing `1_data/1_pipeline/canonical/`.

    `retired_benchmarks.txt` is dropped first and unconditionally, with no flag
    over it: those columns cannot carry a comparable loading row in any
    configuration (see `load_retired_benchmarks`).

    The flags here are all *modeling* choices, not data prep. All default to
    the canonical broad-index configuration (keep everything except the curated
    `excluded_benchmarks.txt` list, which is filtered out here at fit time):
      • drop_low_obs_models (default False) — when True, models with
        <LOW_OBS_THRESHOLD obs are deleted (SOTA + anchors + humans protected).
        Paused: every model is fit, sparse ones simply get wide posteriors.
      • collapse_effort_variants (default False) — when True, for each base
        model (e.g. `gpt-5-2025-08-07`) keep one effort variant. Paused:
        `_low` / `_medium` / `_high` are fit as distinct models.
      • fit_humans (default True) — concatenate `1_data/curated/human_baselines.csv`
        rows as IRT test-takers.
      • eci_data_only (default False) — bypass the processed file and use
        the raw reference ECI CSV.
      • include_all_benchmarks (default False) — diagnostic only. The processed
        file contains every benchmark (the pipeline emits all; exclusion happens here);
        by default the curated `excluded_benchmarks.txt` list is filtered out here
        at fit time. Set True to keep those benchmarks for dimensionality /
        no-exclusions sensitivity analyses. Ignored when eci_data_only is True.
      • fit_cyber (default False) — append `1_data/curated/cyber_benchmarks.csv`,
        the unsaturated benchmarks of Epoch's separate cyber ECI (a different
        data path than the pipeline's feeds, so the refresh never sees them).
        Rows land only on models we already fit and never shadow a pipeline row.
        Off by default: several upstream values were read off published plots,
        so this is a scope choice to make per fit rather than the default
        index scope.
      • fit_simpleqa_original (default False) — append
        `1_data/curated/simpleqa_original/simpleqa_original.csv`, the original
        OpenAI SimpleQA (4,326 questions). A separate column from SimpleQA
        Verified: different question set and grader, scores not comparable.
        Adds 2023-2024 era coverage on the unsaturated-QA construct.
      • drop_benchmarks (default None) — drop the named benchmarks from the
        fit. For targeted sensitivity runs (e.g. the GBAEval+VPCT basin-flip
        test); unlike the curated exclusion list this is per-fit and never the
        default scope. Applied before humans are appended, so their rows on
        the dropped benchmarks go too. A name already absent is a no-op, so
        naming a retired benchmark here changes nothing; any other absent name
        warns, because a typo would otherwise silently fit the full scope.
        Ignored when eci_data_only.
      • min_release_date (default None) — era filter. Drop every model whose
        known release date is BEFORE this ISO date (e.g. "2024-01-01" = the
        post-2023 fit). Models with no release date anywhere
        (mostly recent SEAL/RAND-only entries) are KEPT — the filter removes
        known-old models, it doesn't demand a date. No SOTA/anchor protection:
        a pre-cutoff SOTA model (gpt-4-0314) is dropped like any other. Humans
        are appended after the filter and unaffected. Ignored when
        eci_data_only is True.
      • max_release_date (default None) — the mirror filter. Drop every model
        whose known release date is ON OR AFTER this ISO date. Undated models
        are KEPT (same rule as min_release_date: the filter removes known-new
        models, it doesn't demand a date). Built for temporal holdout tests:
        fit on the past, grade predictions on the post-cutoff models. Same
        no-protection semantics as min_release_date.
    """
    if eci_data_only:
        # Diagnostic mode: use the raw reference ECI data only. Different schema —
        # promote it to the same 8 columns as the processed file.
        df = pd.read_csv(RAW_DIR / "eci_data.csv")
        df = df.rename(columns={"model": "model_version"})
        df["release_date"] = pd.NaT
        df["organization"] = pd.NA
        df["stderr"]       = pd.NA
        df["source"]       = "eci_data.csv"
        df["category"]     = "ECI-only"
        df = df[["model_version", "score", "release_date", "organization",
                  "benchmark", "stderr", "source", "category"]]
    else:
        if not PROCESSED_FILE.exists():
            raise FileNotFoundError(
                f"{PROCESSED_FILE} not found. Run "
                "`1_data/1_pipeline/pipeline.ipynb` (Restart Kernel → Run All) "
                "and copy `output/benchmarks_merged.csv` into `1_data/processed/`."
            )
        df = pd.read_csv(PROCESSED_FILE)

        # Retired benchmarks leave first, before any scope flag is read: no
        # configuration may admit them, so this is the one benchmark drop with
        # no flag governing it. Doing it here also makes an explicit
        # drop_benchmarks of the same names a no-op rather than a second drop.
        retired = load_retired_benchmarks()
        n_retired = int(df["benchmark"].isin(retired).sum())
        df = df[~df["benchmark"].isin(retired)].reset_index(drop=True)
        print(f"   retired benchmarks: dropped {sorted(retired)} "
              f"({n_retired} obs); {df['benchmark'].nunique()} benchmarks remain")

        # Curated "easy-for-humans" exclusions are a MODELING choice, applied here
        # rather than during data generation — the pipeline emits every
        # benchmark. Drop the curated list for the canonical fit; keep everything
        # for the no-exclusions / dimensionality sensitivity run. A model whose
        # every observation sits on excluded benchmarks leaves the fit entirely —
        # say so, because that loss is otherwise invisible in the fit outputs.
        if not include_all_benchmarks:
            models_before = set(df["model_version"].unique())
            df = df[~df["benchmark"].isin(load_excluded_benchmarks())]
            fully_dropped = models_before - set(df["model_version"].unique())
            if fully_dropped:
                preview = ", ".join(sorted(fully_dropped)[:4])
                print(f"   curated exclusions: {len(fully_dropped)} models have "
                      f"no observations left and leave the fit entirely "
                      f"({preview}, ...)")

        if collapse_effort_variants:
            from multiaxis_eci.config import ANCHOR_HIGH, ANCHOR_LOW, SOTA_MODELS
            protected = set(SOTA_MODELS) | {ANCHOR_LOW[0], ANCHOR_HIGH[0]}
            df = _collapse_effort_variants(df, protected=protected)

    excluded = load_excluded_benchmarks()
    df = df.reset_index(drop=True)

    # Era filter (opt-in). Drops models with a KNOWN release date before the
    # cutoff; undated models are kept (they are overwhelmingly recent SEAL/RAND
    # entries — a missing date is not evidence of age). config.RELEASE_DATES
    # fills dates the file lacks (e.g. dateless SOTA previews).
    if (min_release_date or max_release_date) and not eci_data_only:
        # Dates come from the FULL processed file, not the already-filtered df:
        # a model whose only dated rows sit on excluded benchmarks would
        # otherwise look undated and escape the era filter (found with
        # gpt-5.2-pro-2025-12-11, whose dated rows are all ARC-AGI).
        model_date = df["model_version"].map(_known_release_date_by_model())
        for bound, cutoff_str in (("min", min_release_date), ("max", max_release_date)):
            if cutoff_str is None:
                continue
            cutoff = pd.Timestamp(cutoff_str)
            # NaT compares False on both sides → undated models are kept
            drop = (model_date < cutoff) if bound == "min" else (model_date >= cutoff)
            n_dropped = df.loc[drop, "model_version"].nunique()
            df = df[~drop].reset_index(drop=True)
            model_date = model_date[~drop.values].reset_index(drop=True)
            side = "pre-cutoff" if bound == "min" else "post-cutoff"
            print(f"   {bound}_release_date={cutoff_str}: dropped {n_dropped} "
                  f"{side} models ({int(drop.sum())} obs); "
                  f"{df['model_version'].nunique()} models remain")

    # Curated extra-benchmark rows (opt-in). Appended after the era filter so
    # the filter's model set is respected.
    extra_files = [
        (fit_cyber, CURATED_DIR / "cyber_benchmarks.csv", "--cyber"),
        (fit_simpleqa_original,
         CURATED_DIR / "simpleqa_original" / "simpleqa_original.csv",
         "--simpleqa-original"),
    ]
    for enabled, src, label in extra_files:
        if not enabled or eci_data_only:
            continue
        extra = _load_extra_benchmarks(df, src, label)
        if len(extra):
            df = pd.concat([df, extra], ignore_index=True)
            print(f"   {label}: added {len(extra)} obs on "
                  f"{extra['benchmark'].nunique()} benchmarks "
                  f"({extra['model_version'].nunique()} models already in scope); "
                  f"{df['benchmark'].nunique()} benchmarks total")
        elif not src.exists():
            print(f"   {label}: {src.relative_to(DATA_DIR.parent)} absent; "
                  "nothing added")
        else:
            print(f"   {label}: no rows added; every row sits on a model outside "
                  "the fit scope or on a benchmark the pipeline already supplies")

    # Targeted per-fit drops (sensitivity runs). Before humans, so their rows
    # on the dropped benchmarks go too.
    if drop_benchmarks and not eci_data_only:
        # A name already absent is a no-op: the retirement list removes some
        # names a caller may still pass, and a retired name asks for exactly
        # what already happened. Any OTHER absent name is a typo, which would
        # silently fit the full scope, so it warns rather than passing unseen.
        unknown = (set(drop_benchmarks) - set(df["benchmark"])
                   - load_retired_benchmarks())
        if unknown:
            warnings.warn(f"drop_benchmarks not in the table: {sorted(unknown)}; "
                          "nothing dropped for those names")
        n_rows = int(df["benchmark"].isin(drop_benchmarks).sum())
        df = df[~df["benchmark"].isin(drop_benchmarks)].reset_index(drop=True)
        print(f"   drop_benchmarks: removed {sorted(drop_benchmarks)} "
              f"({n_rows} obs); {df['benchmark'].nunique()} benchmarks remain")

    # Humans as IRT rows — must be added after benchmark filtering (so excluded
    # benchmarks like ARC-AGI / HellaSwag drop their human rows too) but
    # before the low_obs filter (so we can opt humans out of that drop).
    human_groups: set[str] = set()
    if fit_humans:
        humans = _load_human_baselines_as_models()
        if len(humans):
            fitted_benchmarks = set(df["benchmark"].unique())
            # A human benchmark absent from the data is dropped. Distinguish an
            # intended drop (benchmark on the curated exclusion list, e.g. ARC-AGI)
            # from a true orphan (a human benchmark never present on the model side
            # — a name mismatch like "GPQA Extended *" vs the data's Diamond/Main,
            # or a benchmark not ingested like PubMedQA). Warn only for orphans so
            # the drop is not silent (finding C2, 2026-07-06); the rows are kept in
            # the curated file as reference and simply don't enter this fit.
            orphans = (set(humans["benchmark"]) - fitted_benchmarks
                       - load_excluded_benchmarks())
            if orphans:
                warnings.warn(
                    f"human_baselines.csv: {len(orphans)} benchmark(s) have no "
                    f"model-side data and are dropped from the fit: {sorted(orphans)}. "
                    "Reconcile the human benchmark name with the data or ingest it.")
            groups_before = set(humans["model_version"].unique())
            humans = humans[humans["benchmark"].isin(fitted_benchmarks)]
            # A tier whose every baseline sits on excluded benchmarks vanishes
            # from the fit — that is a substantive scope change (one fewer human
            # reference level), not routine row filtering, so warn loudly.
            lost_tiers = groups_before - set(humans["model_version"].unique())
            if lost_tiers:
                warnings.warn(
                    f"human_baselines.csv: {len(lost_tiers)} human tier(s) lost "
                    f"ALL their benchmarks to the curated exclusions and leave "
                    f"the fit: {sorted(lost_tiers)}. Add a baseline on a "
                    f"non-excluded benchmark to keep the tier.")
            human_groups = set(humans["model_version"].unique())
            df = pd.concat([df, humans], ignore_index=True)

    if drop_low_obs_models:
        # Drop AI models with fewer than LOW_OBS_THRESHOLD observations —
        # they cause posterior multimodality. Two exemptions kept regardless:
        #   • Humans — sparse anchors are still informative for level lines
        #   • SOTA + anchor models — non-negotiable for the headline plots
        from multiaxis_eci.config import ANCHOR_HIGH, ANCHOR_LOW, SOTA_MODELS
        protected = human_groups | set(SOTA_MODELS) | {ANCHOR_LOW[0], ANCHOR_HIGH[0]}
        counts = df.groupby("model_version").size()
        keep_models = set(counts[counts >= LOW_OBS_THRESHOLD].index) | protected
        df = df[df["model_version"].isin(keep_models)].reset_index(drop=True)

    models = sorted(df["model_version"].unique())
    benchmarks = sorted(df["benchmark"].unique())
    model_to_idx = {m: i for i, m in enumerate(models)}
    bench_to_idx = {b: i for i, b in enumerate(benchmarks)}

    mlookup = pd.DataFrame({"model": models, "model_idx": np.arange(1, len(models) + 1)})
    blookup = pd.DataFrame({"benchmark": benchmarks, "benchmark_idx": np.arange(1, len(benchmarks) + 1)})

    # Per-benchmark category (blookup order). Derived from the real benchmark
    # rows, ignoring the synthetic "Human baseline" category that human rows
    # carry, so a benchmark keeps its true category even after humans are added.
    real_cat = df[df["category"] != "Human baseline"]
    cat_by_bench = (real_cat.groupby("benchmark")["category"]
                            .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iat[0]))
    bench_category = np.array([cat_by_bench.get(b, "Unknown") for b in benchmarks], dtype=object)

    raw = df["score"].to_numpy(dtype=np.float64)
    # Effective test length per observation, from the harness stderr the feeds
    # report. The Beta's precision already IS a test length: with
    # Var = mu(1-mu)/(1+phi), a score behaves like the average of (1+phi)
    # solve/fail tasks, so a reported se on an observed score p converts into the
    # same unit, n_eff = p(1-p)/se^2 — "the length of a simple test that would be
    # exactly this precise". inf means no usable stderr, and the model then
    # estimates that cell's whole noise budget as it always has
    # (models/mirt.py, known_se). Built from the final row order, so it stays
    # aligned with scores / model_idx / bench_idx through every filter above.
    # Near-boundary scores stay at inf: p(1-p) collapses there, so the implied
    # length is dominated by how the score was rounded. The clip stops a single
    # tiny-se row from claiming more precision than any test in this data has.
    se = pd.to_numeric(df["stderr"], errors="coerce").to_numpy(dtype=np.float64)
    n_eff = np.full(len(df), np.inf)
    has_se = np.isfinite(se) & (se > 0.0) & (raw > 0.02) & (raw < 0.98)
    n_eff[has_se] = np.clip(raw[has_se] * (1.0 - raw[has_se]) / se[has_se] ** 2,
                            2.0, 1e6)
    model_idx = df["model_version"].map(model_to_idx).to_numpy()
    n_obs_per_model = np.bincount(model_idx, minlength=len(models))
    is_human = np.array([m in human_groups for m in models], dtype=bool)
    # Humans are exempt from the low-obs flag — sparse human anchors are
    # still informative for the level lines on the timeline plot, and the
    # `drop_low_obs_models` filter above already protects them from being
    # dropped from the dataset.
    is_low_obs = (n_obs_per_model < LOW_OBS_THRESHOLD) & ~is_human
    # SOTA flag (independent of obs count) — drives the always-show-on-plots rule.
    from multiaxis_eci.config import SOTA_MODELS
    sota_set = set(SOTA_MODELS)
    is_sota = np.array([m in sota_set for m in models], dtype=bool)

    return ECIData(
        scores              = raw,
        zero_score_mask     = raw == 0.0,
        model_idx           = model_idx,
        bench_idx           = df["benchmark"].map(bench_to_idx).to_numpy(),
        mlookup             = mlookup,
        blookup             = blookup,
        n_models            = len(models),
        n_benchmarks        = len(benchmarks),
        n_obs               = len(df),
        zero_diag_threshold = ZERO_DIAG_THRESHOLD,
        n_obs_per_model     = n_obs_per_model,
        is_low_obs          = is_low_obs,
        excluded_benchmarks = excluded,
        is_human            = is_human,
        n_eff               = n_eff,
        bench_category      = bench_category,
        is_sota             = is_sota,
    )


def find_model_idx(mlookup: pd.DataFrame, model_name: str) -> int:
    """0-indexed position of `model_name` in the lookup."""
    matches = mlookup.loc[mlookup["model"] == model_name, "model_idx"].values
    if len(matches) == 0:
        raise KeyError(f"Model not found: {model_name!r}")
    return int(matches[0]) - 1


def open_only_drop_list(include_all_benchmarks: bool, keep_open: bool = True) -> list[str]:
    """Benchmarks to drop for --open-only (keep_open=True) or --closed-only
    (keep_open=False), filtered down to whatever load_eci_data would
    otherwise have loaded (curated-exclusion scope unless
    include_all_benchmarks). Lives here rather than in the fit CLI so the CLI and
    the country-frontier diagnostic both build the exact same open-only data scope from one place.
    """
    access_path = DATA_DIR / "curated" / "benchmark_access.csv"
    if not access_path.exists():
        raise FileNotFoundError(
            f"--open-only needs {access_path}, which does not exist yet "
            "(columns: benchmark,access,verified,evidence_url,notes).")
    access = pd.read_csv(access_path)
    open_ok = set(access.loc[(access["access"] == "public") &
                              (access["verified"] == "yes"), "benchmark"])
    # The access file covers the full exploration scope (98 benchmarks);
    # canonical only ever sees the curated-exclusion scope (90). The drop list
    # is filtered down to what load_eci_data will actually have loaded, so the
    # printed kept/dropped counts describe the fit's own scope.
    all_benchmarks = (set(pd.read_csv(DATA_DIR / "processed" / "benchmarks_merged.csv")["benchmark"])
                      - load_retired_benchmarks())
    in_scope = all_benchmarks if include_all_benchmarks \
        else all_benchmarks - load_excluded_benchmarks()
    drop = sorted(in_scope - open_ok) if keep_open else sorted(in_scope & open_ok)
    label = "--open-only" if keep_open else "--closed-only"
    print(f"── {label}: {len(in_scope) - len(drop)} "
          f"{'public/verified' if keep_open else 'closed'} "
          f"benchmarks kept, {len(drop)} dropped ─────")
    return drop
