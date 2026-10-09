"""FitSpec: one object owning a fit's identity.

A fit is its flag set. From that follow the tag, the results and plots folders,
the trace filename, and the data scope the trace must be scored against. This
module is the single owner of all four, so a folder name, a trace name and a
plotted data set cannot drift apart.

`from_trace` recovers the spec of an already-fitted trace. Three sources, in
order: the `mirt_spec` JSON attr, which 3_fit/fit.py writes, the individual `mirt_*`
attrs, and the tokens of the results-folder name for the flags no attr carries
(`--apply-exclusions`, `--no-sg`, `--ceiling-noise`). A flag a tag cannot carry
losslessly is REFUSED, not guessed: `_dropFrontierMathv1AlgoTune` cannot be
split back into benchmark names, so a trace with that token and no attr raises.
An unrecognized token raises too — a partial parse would load the wrong data
scope and mis-index every row.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import arviz as az
import numpy as np

from multiaxis_eci.config import (
    FIGURES_DIRNAME,
    HUMAN_ORDER,
    HUMAN_ORDER_MERGED,
    HUMAN_ORDER_MERGED_PRE_COMMITTEES,
    PRIOR_DELTA_HUMAN,
    PRIOR_HUMAN_ROOT,
    RESULTS_DIR,
    SG_MODEL_NAME,
)

# Loading priors, longest first: "_signed" is a prefix of "_signedhs".
# "signedhs" is parse-only — the prior is gone, its historical folder names are not.
_PRIOR_TOKENS = ("signedhs", "signed", "pt1", "bifactor")


# The K whose results folder carries no K token: every K-axis fit published so far is K=4.
# Another K of the same flag set gets a `_k<K>` folder (2026-10-06; until then only the
# figures were split by K, and a K=3 test run wrote over the K=4 fit's tables).
UNTOKENED_K = 4

# Spec fields since removed, with the value that meant "off". A trace whose mirt_spec carries one
# at that value reads as today's spec; at any other value it was fit with a model no builder
# reproduces any more, and reading it raises. The "easy for humans" fifth axis
# (`--target-axis`, `--target-root-sd`), tested and dropped on 2026-10-07 (docs/model_math.md).
_RETIRED_FIELDS = {"target_axis": False, "target_root_sd": None}

# The IRF links models/mirt.build_mirt_model knows, "linear" (the compensatory 2PL, untagged)
# first; any other emits its name as a tag token.
LINKS = ("linear", "loglog", "semicomp", "noncomp", "semicomp0")

# The human prior of every fit saved before FitSpec carried it (2026-10-06), the config values
# until that day: a trace whose mirt_spec lacks a field was sampled with these. The root sd holds
# from 2026-10-05; a trace from before then (no `theta_h_base`) had its roots inside the models'
# ZeroSumNormal, a model no current builder reproduces.
LEGACY_HUMAN_DELTA_SD = 1.0
LEGACY_HUMAN_ROOT_SD = 3.0


@dataclass(frozen=True)
class FitSpec:
    """The flag set of one compensatory MIRT fit.

    Field names match 3_fit/fit.py's CLI flags. Hashable, so a caller can memoize a
    loaded data scope on the spec itself.
    """
    K: int
    loading_prior: str = "normal"
    link: str = "linear"
    human_prior: bool = False
    human_merge: bool = False
    lineage_prior: bool = False
    lineage_bm: bool = False
    # A mean per effort level on the lineage prior's variant offsets, along one effort direction,
    # with one offset spread (models/mirt._variant_offsets). The default since 2026-10-03 (user):
    # ON carries no token, the opt-out emits `_noeffortmeans`. It needs the lineage prior and is
    # off without it. A trace from before that day was fit without it (`from_trace`).
    effort_means: bool = True
    time_prior: bool = False
    theta_t: bool = False
    theta_pos: bool = False
    no_sg: bool = False
    apply_exclusions: bool = False
    # Isolated model families (one benchmark across all variants) are dropped by default
    # (data.load_eci_data); the opt-out emits `_keepiso`.
    keep_isolated: bool = False
    # Kept so folders of pre-September-2026 fits (`_cyber`) still parse; the cyber
    # benchmarks are part of the pipeline's view now and this flag changes nothing.
    cyber: bool = False
    simpleqa_original: bool = False
    drop_benchmarks: tuple[str, ...] = ()
    private_bases: bool = False
    # Fixed-c 3PL floors are the basic likelihood, not an option: a below-chance
    # score reads as uninformative-low ability instead of a point demand. The
    # floors come from file and add no parameters, so there is nothing to tune.
    # The default (ON) carries no token; the opt-out emits `_nofloors`, so a
    # sensitivity run gets its own folder instead of overwriting the default's.
    floors: bool = True
    ceiling_noise: bool = False
    known_se: bool = False
    # Censored Beta likelihood at the score bounds (data.load_boundary_eps); off by default,
    # emits `_censor`.
    censor_bounds: bool = False
    # Hierarchical sigma_b is the basic noise model, not an option: thin
    # benchmarks shrink to the shared median instead of keeping a free scale
    # the panel cannot pin. The default (ON) carries no token; the opt-out
    # emits `_unpooled` for the same folder-separation reason as `floors`.
    pooled_noise: bool = True
    # Beta-binomial likelihood on the benchmarks whose score is a share of items
    # (data.load_binomial_n), the Beta density on the others: the basic likelihood since
    # 2026-10-01, when a Beta-only K=4 fit froze six of eight chains in a noise spike on
    # Chess Puzzles. The default (ON) carries no token; the opt-out emits `_betalik`. A
    # censored fit (`censor_bounds`) is a Beta variant and turns it off.
    beta_binomial: bool = True
    # sigma of the HalfNormal increment between ordered human tiers (models.mirt._human_block).
    # Not a CLI flag and not in the tag: the config's value is the default, and the spec
    # carries it into the trace (mirt_spec) so a rebuilt model or a figure reads the value the
    # trace was sampled with. A spec without it predates the field (LEGACY_HUMAN_DELTA_SD).
    human_delta_sd: float = PRIOR_DELTA_HUMAN
    # sd of the human roots' private Normal base (models.mirt._assemble_theta), carried the same
    # way; a spec without it predates the field (LEGACY_HUMAN_ROOT_SD).
    human_root_sd: float = PRIOR_HUMAN_ROOT
    # The human baselines left out of the data (load_eci_data(fit_humans=False)): a pure
    # model-vs-model scale, the humans placed afterwards on the frozen benchmark side.
    # Emits `_nohumans`. Exploration, 2026-10-07.
    no_humans: bool = False

    def __post_init__(self):
        # The cross-flag constraint the model cannot express on its own. Checked
        # here so every construction path pays it, not just the CLI.
        if self.no_humans and (self.human_merge or self.human_prior):
            raise ValueError("--no-humans and the ordered-human priors cannot compose: there "
                             "are no human test-takers left to order.")
        if self.link not in LINKS:
            raise ValueError(f"unknown link {self.link!r} (one of {', '.join(LINKS)})")
        if self.lineage_bm and not self.lineage_prior:
            raise ValueError(
                "--lineage-bm re-indexes the lineage increments by time; it "
                "needs --lineage-prior.")
        if self.effort_means and not self.lineage_prior:
            object.__setattr__(self, "effort_means", False)
        if self.censor_bounds and self.beta_binomial:
            object.__setattr__(self, "beta_binomial", False)

    # ── identity ────────────────────────────────────────────────────────────
    @property
    def tag(self) -> str:
        """One tag identifying the fit config; reused for BOTH the results
        folder and the trace filename, so distinct configs never overwrite each
        other's fixed-name artefacts. The loading prior leads the tag (matching
        the historical `trace_mirt_k3_signed_...` convention); "normal" stays
        untagged so its folders reduce to the canonical mirt_humanprior etc."""
        tag = "" if self.loading_prior == "normal" else f"_{self.loading_prior}"
        if self.link != "linear": tag += f"_{self.link}"
        if self.human_merge: tag += "_humanmerge"
        elif self.human_prior: tag += "_humanprior"
        if self.lineage_prior: tag += "_lineageprior"
        if self.lineage_bm: tag += "_lineagebm"
        if self.lineage_prior and not self.effort_means: tag += "_noeffortmeans"
        if self.time_prior: tag += "_timeprior"
        if self.theta_t: tag += "_thetat"
        if self.theta_pos: tag += "_thetapos"
        if self.no_sg: tag += "_noSG"
        if self.apply_exclusions: tag += "_excluded"
        if self.keep_isolated: tag += "_keepiso"
        if self.cyber: tag += "_cyber"
        if self.simpleqa_original: tag += "_sqaorig"
        if self.drop_benchmarks:
            tag += "_drop" + "".join(re.sub(r"\W", "", b)
                                     for b in self.drop_benchmarks)
        if self.private_bases: tag += "_privbase"
        if not self.floors: tag += "_nofloors"
        if self.ceiling_noise: tag += "_ceilnoise"
        if self.known_se: tag += "_knownse"
        if not self.pooled_noise: tag += "_unpooled"
        if self.censor_bounds: tag += "_censor"
        elif not self.beta_binomial: tag += "_betalik"
        if self.no_humans: tag += "_nohumans"
        return tag

    @property
    def results_dir(self) -> Path:
        """`5_outputs/<data generation>/mirt{tag}/`: the data generation is the
        parent folder, so the tag carries only the flag set. A K other than
        `UNTOKENED_K` adds `_k<K>`, so a K=3 run of the flagship's flags never
        writes its fixed-name tables (`mirt_loadings.csv`, the display frame every
        K=4 figure is aligned on), axis names or live store over the K=4 fit's."""
        name = f"mirt{self.tag}" if self.tag else "mirt"
        return RESULTS_DIR / (name if self.K == UNTOKENED_K else f"{name}_k{self.K}")

    @property
    def figures_dir(self) -> Path:
        """`<results_dir>/figures/k{K}/` (PNG) with `html/` beneath it."""
        return self.results_dir / FIGURES_DIRNAME / f"k{self.K}"

    @property
    def trace_path(self) -> Path:
        return self.results_dir / f"trace_mirt_k{self.K}{self.tag}.nc"

    @property
    def baseline_trace_path(self) -> Path:
        """The K=1 baseline 3_fit/fit.py fits beside a K-axis fit. It carries no tag
        in its filename; current saves stamp its spec as the `mirt_spec` attr,
        and an old attr-less baseline is reconstructed from the folder plus
        3_fit/fit.py's baseline rule (see `_baseline_spec`)."""
        return self.results_dir / "trace_mirt_k1.nc"

    @property
    def human_order(self):
        """The human tier order this fit imposes, or None."""
        return (HUMAN_ORDER_MERGED if self.human_merge
                else HUMAN_ORDER if self.human_prior else None)

    def baseline_spec(self) -> FitSpec:
        """The spec of the K=1 baseline 3_fit/fit.py fits beside this fit: same data
        scope, every model-side flag off (see `_BASELINE_OFF`). 3_fit/fit.py stamps
        this as the baseline trace's `mirt_spec` attr so the trace stays
        self-describing — its folder tokens describe the K-axis fit, not it."""
        return _baseline_spec(self)

    # ── construction ────────────────────────────────────────────────────────
    @classmethod
    def from_args(cls, args, parser=None) -> FitSpec:
        """Spec from 3_fit/fit.py's parsed exploration flags. `parser` turns a flag
        conflict into an argparse error instead of a traceback."""
        try:
            return cls(
                K=args.K, loading_prior=args.loading_prior, link=args.link,
                human_prior=args.human_prior, human_merge=args.human_merge,
                lineage_prior=args.lineage_prior, lineage_bm=args.lineage_bm,
                effort_means=args.effort_means,
                time_prior=args.time_prior, theta_t=args.theta_t,
                theta_pos=args.theta_pos,
                no_sg=args.no_sg,
                apply_exclusions=args.apply_exclusions,
                keep_isolated=args.keep_isolated,
                simpleqa_original=args.simpleqa_original,
                drop_benchmarks=tuple(args.drop_benchmarks or ()),
                private_bases=args.private_bases,
                floors=args.floors,
                ceiling_noise=args.ceiling_noise, known_se=args.known_se,
                pooled_noise=args.pooled_noise,
                censor_bounds=args.censor_bounds,
                beta_binomial=args.beta_binomial,
                no_humans=getattr(args, "no_humans", False))
        except ValueError as e:
            if parser is None:
                raise
            parser.error(str(e))

    @staticmethod
    def from_trace(idata, trace_path) -> FitSpec:
        """The spec of an existing trace, from its attrs and its folder name.

        Raises rather than returning a partial spec: a wrong data scope
        mis-indexes every observation, so an unreadable tag is an error, never a
        default.
        """
        trace_path = Path(trace_path)
        post = idata.posterior
        m = re.match(r"trace_mirt_k(\d+)", trace_path.name)
        if m is None:
            raise ValueError(
                f"{trace_path.name!r} is not a MIRT trace filename "
                "(expected trace_mirt_k<K>{tag}.nc)")
        K = int(m.group(1))
        if "latent" in post.sizes and int(post.sizes["latent"]) != K:
            raise ValueError(f"{trace_path.name} names K={K} but its posterior "
                             f"has {int(post.sizes['latent'])} axes")

        # A K=1 baseline is named `trace_mirt_k1.nc` inside the K-axis fit's own
        # folder, so its filename can never round-trip against its (different)
        # spec. The folder it was parsed from IS the check there.
        is_baseline = trace_path.name == "trace_mirt_k1.nc"
        if post.attrs.get("mirt_spec"):
            d = json.loads(post.attrs["mirt_spec"])
            d["drop_benchmarks"] = tuple(d.get("drop_benchmarks") or ())
            # A spec written before the Beta-binomial existed was fit with the Beta, one written
            # before the effort means became the default (2026-10-03) without them.
            d.setdefault("beta_binomial", False)
            d.setdefault("effort_means", False)
            d.setdefault("human_delta_sd", LEGACY_HUMAN_DELTA_SD)
            d.setdefault("human_root_sd", LEGACY_HUMAN_ROOT_SD)
            for field, off in _RETIRED_FIELDS.items():
                if field in d:
                    if d[field] != off:
                        raise ValueError(f"{trace_path.name} was fit with {field}={d[field]!r}, "
                                         "an option since removed (see _RETIRED_FIELDS)")
                    del d[field]
            spec = FitSpec(**d)
        else:
            attr = _attr_flags(post)
            if not any(k.startswith("mirt_") for k in post.attrs):
                import warnings
                warnings.warn(
                    f"{trace_path.name}: trace carries no mirt_* attrs; spec "
                    "recovered from the folder tokens alone. Under that legacy "
                    "grammar an absent floors token reads floors OFF — wrong for "
                    "a crash checkpoint or an old baseline of a floors-ON fit. "
                    "Verify the recovered spec before trusting derived numbers.")
            tag = _folder_tag(trace_path)
            # Without mirt_spec the trace predates the Beta-binomial: the Beta, unless an attr says so.
            spec = FitSpec(K=K, **{**_parse_tag(tag, attr.get("drop_benchmarks", ())),
                                   "beta_binomial": False, "effort_means": False,
                                   "human_delta_sd": LEGACY_HUMAN_DELTA_SD,
                                   "human_root_sd": LEGACY_HUMAN_ROOT_SD, **attr})
            if is_baseline:
                spec = _baseline_spec(spec)
        if not is_baseline:
            _check_round_trip(spec, trace_path)
        return spec

    # ── the data the trace was fit on ───────────────────────────────────────
    def load_data(self, idata=None):
        """The fit's data scope: `(data, floor_c, n_eff)`.

        Same loads, drops, clips and prints 3_fit/fit.py runs before sampling, so a
        plot or a GoF number is scored on exactly the observations the trace saw.
        Pass `idata` to have the trace's own `model` / `bench` dims checked
        against it — a trace from an older data generation cannot be indexed
        against today's data at all, and the benchmark-count mismatch is the one
        that fails silently.
        """
        from multiaxis_eci.data import (
            clip_scores_to_floors,
            drop_model_observations,
            load_benchmark_floors,
            load_eci_data,
        )
        data = load_eci_data(
            include_all_benchmarks=not self.apply_exclusions,
            fit_humans=not self.no_humans,
            drop_isolated_families=not self.keep_isolated,
            fit_simpleqa_original=self.simpleqa_original,
            drop_benchmarks=list(self.drop_benchmarks) or None)
        if self.apply_exclusions:
            print("  --apply-exclusions: curated exclusions applied at fit time "
                  "(dropped benchmarks from excluded_benchmarks.txt)", flush=True)
        if idata is not None:
            _check_dims(idata, data)
        if self.no_sg:
            n_before = data.n_obs
            data = drop_model_observations(data, [SG_MODEL_NAME])
            print(f"  --no-sg: dropped {n_before - data.n_obs} '{SG_MODEL_NAME}' "
                  f"observations (tier kept in model index, prior-only theta)", flush=True)
        floor_c = None
        if self.floors:
            floor_c = load_benchmark_floors(data)
            n_before = int((data.scores < floor_c[data.bench_idx]).sum())
            data = clip_scores_to_floors(data, floor_c)
            print(f"  floors (default; --no-floors opts out): fixed-c 3PL; "
                  f"clipped {n_before} below-floor scores up to their benchmark "
                  f"chance floor", flush=True)
        if self.known_se:
            measured = np.isfinite(data.n_eff)
            print(f"  --known-se: {int(measured.sum())} of {data.n_obs} cells "
                  f"({measured.mean():.1%}) carry a reported stderr, median "
                  f"effective test length {np.median(data.n_eff[measured]):.0f} "
                  f"tasks; the rest keep the estimated per-benchmark noise",
                  flush=True)
        return data, floor_c, (data.n_eff if self.known_se else None)

    def binomial_n(self, data):
        """Per-observation item count for `build_mirt_model(binomial_n=)`, or None for a Beta fit."""
        from multiaxis_eci.data import load_binomial_n
        return load_binomial_n(data) if self.beta_binomial else None

    # ── lazy posterior access ───────────────────────────────────────────────
    def open_posterior(self, keep=None, thin: int = 1, chains=None, path=None,
                       log_lik: bool = False):
        """The trace's posterior as an InferenceData holding only `keep`, thinned.

        The flagship trace is 38 GB against 26 GB of RAM, so a consumer names the
        variables it needs instead of loading the file. Names absent from this
        trace are skipped, so one `keep` list covers every link variant.
        Attributes survive the subset, so `prepare_fit` still reads the display
        rotation, the axis names and the loading prior off the posterior.
        `log_lik` also reads the pointwise log-likelihood, thinned and subset the same way,
        for the LOO-PIT (ppc.pointwise_log_lik); a trace without that group returns without.
        """
        import xarray as xr

        def _subset(ds):
            if thin > 1:
                ds = ds.isel(draw=slice(None, None, thin))
            if chains is not None:
                ds = ds.isel(chain=chains)
            return ds.load()

        trace = path or self.trace_path
        with xr.open_dataset(trace, group="posterior") as post:
            sub = _subset(post if keep is None else post[[v for v in keep if v in post]])
        groups = {"posterior": sub}
        if log_lik:
            try:
                with xr.open_dataset(trace, group="log_likelihood") as ll:
                    groups["log_likelihood"] = _subset(ll)
            except (OSError, KeyError, ValueError):
                pass
        return az.InferenceData(**groups)


# ── the flagship fit ────────────────────────────────────────────────────────
# K=4, positive loadings, Brownian lineage prior, 3PL floors, pooled noise, the exploration
# scope, fitted on the MODELS ALONE (user, 2026-10-08): the human tiers are placed afterwards,
# axis by axis, with the benchmark side frozen (analysis.human_placement,
# 4_diagnostics/5_place_humans.py). Fitted with the human tiers (`human_merge`, the flagship
# until that day, still fitted under `mirt_humanmerge_lineageprior_lineagebm/`), the
# compensatory model put them about 5 sd below the models on one axis and 8 above on another.
# The blog post reports the 2026-08 run of that joint fit (10 chains x 12,000 draws on the
# pre-pipeline snapshot, archived under 5_outputs/pre_pipeline/). `loading_prior` and `floors`
# are named explicitly although both hold the default: the flagship is a statement of one fit,
# and a later default change must not silently redefine it.
FLAGSHIP = FitSpec(K=4, loading_prior="normal", lineage_prior=True, lineage_bm=True,
                   floors=True, no_humans=True)
# The former flagship, the human tiers fitted inside under the merged order: kept as a secondary
# fit in its own folder (`mirt_humanmerge_lineageprior_lineagebm/`), refitted on new data only on
# request (`3_fit/fit.py --K 4 --human-merge --lineage-prior --lineage-bm`), out of the default
# run and of the dashboard (user, 2026-10-08).
JOINT_FIT = FitSpec(K=4, loading_prior="normal", human_merge=True, lineage_prior=True,
                    lineage_bm=True, floors=True)
# The one path every consumer reads, so the figures, the dashboard card and the
# post cannot name different traces: the flagship fit of the CURRENT data
# generation (`config.RESULTS_DIR`). It exists once `3_fit/fit.py --K 4 --lineage-prior
# --lineage-bm --no-humans` has run on that generation (and 5_place_humans.py has placed the
# tiers); `open_flagship` says so when it has not. `FLAGSHIP_TRACE.parent` is the fit's
# results folder and holds its CSVs and caches.
FLAGSHIP_TRACE = FLAGSHIP.trace_path
# Which chains the flagship's summaries read: "best" is the modes file's `selected_chains`
# when it names a subset, else the best-log-density mode (`best_mode_chains`), every chain
# when that file is missing or finds one mode; the rule `3_plot_mirt.chain_split` applies
# to the figures.
FLAGSHIP_CHAINS = "best"
# No further thinning on read: traces are saved every config.SAVE_THIN-th draw already
# (5), and at 2,000 draws a chain a further 10 left 120 draws on three selected chains.
FLAGSHIP_THIN = 1


def best_mode_chains(trace_path) -> list[int] | None:
    """The chains every deliverable of a trace reads, from the modes file beside it: its
    `selected_chains` when the file carries that choice (a subset picked on evidence the mode
    split does not hold, with its reason under `why`), else the best-log-density mode (ties to
    the larger). None when the file is missing, names another trace or reports a single mode
    and no selection."""
    p = Path(trace_path)
    f = p.parent / f"mirt_modes_{p.stem}.json"
    if not f.exists():
        return None
    doc = json.loads(f.read_text())
    if doc.get("trace") != p.name:
        return None
    selected = doc.get("selected_chains")
    if selected:
        return sorted(selected["chains"])
    modes = doc.get("modes", [])
    if len(modes) < 2:
        return None
    best = max(modes, key=lambda m: (m["delta_logp"] if m.get("delta_logp") is not None
                                     else float("-inf"), len(m["chains"])))
    return sorted(best["chains"])


def secondary_mode_chains(trace_path) -> list[int] | None:
    """The chains of the secondary mode the figures render beside the primary one: the modes
    file's `secondary_chains` (a mode the user kept for comparison, e.g. one integrated LOO
    cannot tell from the primary), None when the file names none."""
    p = Path(trace_path)
    f = p.parent / f"mirt_modes_{p.stem}.json"
    if not f.exists():
        return None
    doc = json.loads(f.read_text())
    sec = doc.get("secondary_chains") if doc.get("trace") == p.name else None
    return sorted(sec["chains"]) if sec else None


def open_flagship(keep=None, thin: int = FLAGSHIP_THIN, chains=FLAGSHIP_CHAINS):
    """The flagship posterior, thinned. `chains=None` reads every chain, "best" the
    best-log-density mode (`best_mode_chains`), every chain when there is one mode.

    Raises:
        FileNotFoundError: the flagship has not been fitted on the current data generation.
    """
    if not FLAGSHIP_TRACE.exists():
        raise FileNotFoundError(
            f"{FLAGSHIP_TRACE}: the flagship K=4 fit does not exist for this data generation; "
            "run `python 3_fit/fit.py --K 4 --lineage-prior --lineage-bm --no-humans` first "
            "(the published run sits under 5_outputs/pre_pipeline/).")
    if chains == "best":
        chains = best_mode_chains(FLAGSHIP_TRACE)
    return FLAGSHIP.open_posterior(keep=keep, thin=thin, chains=chains,
                                   path=FLAGSHIP_TRACE)


# ── spec recovery helpers ───────────────────────────────────────────────────

def _attr_flags(post) -> dict:
    """Flags recoverable from the individual `mirt_*` attrs a fit driver writes.

    An attr is written only when its flag is on, so a MISSING attr means
    "unknown", never False — the caller keeps the folder-tag value there.
    """
    a = post.attrs
    out: dict = {}
    if a.get("mirt_loading_prior"):
        out["loading_prior"] = str(a["mirt_loading_prior"])
    if a.get("mirt_link"):
        out["link"] = str(a["mirt_link"])
    if a.get("mirt_human_order"):
        # Tuple parents survive JSON as lists, so compare both sides round-tripped.
        order = json.loads(a["mirt_human_order"])
        if any(order == json.loads(json.dumps(o))
               for o in (HUMAN_ORDER_MERGED, HUMAN_ORDER_MERGED_PRE_COMMITTEES)):
            out["human_merge"] = True
        elif order == json.loads(json.dumps(HUMAN_ORDER)):
            out["human_prior"] = True
        # Neither order: the folder tag decides which flag it was, since
        # guessing here would silently change the data scope.
    if a.get("mirt_lineage_chains"):
        out["lineage_prior"] = True
    if json.loads(a.get("mirt_lineage_bm", "false")):
        out["lineage_bm"] = True
    if "mirt_effort_means" in a:
        out["effort_means"] = bool(json.loads(a["mirt_effort_means"]))
    if a.get("mirt_time_prior"):
        out["time_prior"] = True
    if a.get("mirt_theta_t_cells"):
        out["theta_t"] = True
    if a.get("mirt_theta_pos"):
        out["theta_pos"] = True
    if "mirt_shared_base_zsn" in a and not json.loads(a["mirt_shared_base_zsn"]):
        out["private_bases"] = True
    if a.get("mirt_cyber"):
        out["cyber"] = True
    if a.get("mirt_simpleqa_original"):
        out["simpleqa_original"] = True
    if json.loads(a.get("mirt_censor_bounds", "false")):
        out["censor_bounds"] = True
    if json.loads(a.get("mirt_keep_isolated", "false")):
        out["keep_isolated"] = True
    if "mirt_drop_benchmarks" in a:
        out["drop_benchmarks"] = tuple(json.loads(a["mirt_drop_benchmarks"]))
    if "mirt_floor_c" in a:
        out["floors"] = True
    if a.get("mirt_known_se"):
        out["known_se"] = True
    if a.get("mirt_pooled_noise"):
        out["pooled_noise"] = True
    if "mirt_beta_binomial" in a:
        out["beta_binomial"] = bool(json.loads(a["mirt_beta_binomial"]))
    return out


def _folder_tag(trace_path: Path) -> str:
    """The tag carried by the trace's PARENT DIRECTORY (`5_outputs/<data>/mirt{tag}`).

    Read from the folder, not the filename, because a K=1 baseline trace is
    named `trace_mirt_k1.nc` with no tag at all.
    """
    name = trace_path.parent.name
    if name != "mirt" and not name.startswith("mirt_"):
        raise ValueError(f"{trace_path.parent.name!r} is not a MIRT results "
                         "folder (expected mirt or mirt_<tag>)")
    # Legacy folders carried the data generation as a `_data<YYYYMMDD>` suffix;
    # it named the folder, never the spec. A `_short<draws>x<chains>` suffix marks a
    # smaller run of the same flag set kept beside the full one (the 2000x8 K=4
    # test fit); it names the folder too.
    # A `_k<K>` suffix puts a K other than UNTOKENED_K in a folder of its own.
    return re.sub(r"(_k\d+)?(_data\d{8}|_short\d+x\d+)?$", "", name[4:])


def _parse_tag(tag: str, drop_benchmarks: tuple = ()) -> dict:
    """The flags a results-folder tag encodes, consumed in the order 3_fit/fit.py
    writes them. Raises on a token this parser does not know, and on the two
    lossy `_drop` token when no attr supplied its value."""
    rest, out = tag, {}

    def take(tok: str) -> bool:
        nonlocal rest
        if rest.startswith(tok):
            rest = rest[len(tok):]
            return True
        return False

    for p in _PRIOR_TOKENS:
        if take(f"_{p}"):
            out["loading_prior"] = p
            break
    else:
        out["loading_prior"] = "normal"
    for lk in sorted(LINKS[1:], key=len, reverse=True):     # semicomp0 before semicomp
        if take(f"_{lk}"):
            out["link"] = lk
            break
    if take("_humanmerge"):
        out["human_merge"] = True
    elif take("_humanprior"):
        out["human_prior"] = True
    for tok, fl in (("_lineageprior", "lineage_prior"), ("_lineagebm", "lineage_bm")):
        if take(tok):
            out[fl] = True
    if take("_noeffortmeans"):
        out["effort_means"] = False
    for tok, fl in (("_timeprior", "time_prior"),
                    ("_thetat", "theta_t"), ("_thetapos", "theta_pos")):
        if take(tok):
            out[fl] = True
    for tok, fl in (("_noSG", "no_sg"), ("_excluded", "apply_exclusions"),
                    ("_keepiso", "keep_isolated"),
                    ("_cyber", "cyber"), ("_sqaorig", "simpleqa_original")):
        if take(tok):
            out[fl] = True
    if rest.startswith("_drop"):
        expect = "_drop" + "".join(re.sub(r"\W", "", b) for b in drop_benchmarks)
        if not (drop_benchmarks and take(expect)):
            raise ValueError(
                f"tag {tag!r} carries `_drop` — benchmark names lose their "
                "non-word characters in the tag, so --drop-benchmarks cannot be "
                "recovered from it. The trace must carry mirt_drop_benchmarks or "
                "mirt_spec.")
        out["drop_benchmarks"] = tuple(drop_benchmarks)
    if take("_privbase"):
        out["private_bases"] = True
    # Floors, three grammars. Current: ON is the untagged default, OFF emits
    # `_nofloors`. Legacy: ON emitted `_floors`. Legacy-attrless: three traces
    # (mirt_humanprior, mirt_loglog) sit in token-less folders and were fit
    # WITHOUT floors, so an absent token still reads floors OFF here — a trace
    # fit floors-ON under the current grammar carries the state in its
    # `mirt_floor_c` / `mirt_spec` attrs, which override this. A token zero
    # traces carry (`_ceilings`) raises as unrecognized instead.
    if take("_nofloors"):
        out["floors"] = False
    else:
        out["floors"] = take("_floors")
    for tok, fl in (("_ceilnoise", "ceiling_noise"), ("_knownse", "known_se")):
        if take(tok):
            out[fl] = True
    # Legacy token: ON used to be tagged `_poolednoise`; it names the value the
    # default already holds. The current grammar tags only the opt-out.
    if take("_poolednoise"):
        out["pooled_noise"] = True
    if take("_unpooled"):
        out["pooled_noise"] = False
    if take("_censor"):
        out["censor_bounds"] = True
    if take("_betalik"):
        out["beta_binomial"] = False
    if take("_nohumans"):
        out["no_humans"] = True
    if rest:
        raise ValueError(f"unrecognized tag token {rest!r} in folder tag {tag!r} "
                         "— refusing to guess the fit's data scope")
    return out


# The model-side flags 3_fit/fit.py does NOT forward to the K=1 baseline (3_fit/fit.py's
# baseline call passes floors / ceiling_noise / known_se / censor_eps only). Every
# other model-side flag is off there whatever the folder tag says; the data-scope
# flags are the folder's.
_BASELINE_OFF = dict(loading_prior="normal", link="linear",
                     human_prior=False, human_merge=False,
                     lineage_prior=False, lineage_bm=False, effort_means=False,
                     time_prior=False,
                     theta_t=False, theta_pos=False, private_bases=False,
                     pooled_noise=False)


def _baseline_spec(spec: FitSpec) -> FitSpec:
    return replace(spec, K=1, **_BASELINE_OFF)


def _check_round_trip(spec: FitSpec, trace_path: Path) -> None:
    """The recovered spec must describe the flag set the trace's own name carries.

    Both names are read through `_parse_tag` and the resulting SPECS are
    compared, not the path strings. A flag the tag grammar does not encode is
    then equal on both sides by construction, while every flag it does encode is
    checked, so a misfiled or hand-renamed trace still fails. Only the last two
    path components are read, so a trace in a copy of the tree still validates.
    """
    folder_tag = _folder_tag(trace_path)
    file_K, file_tag = re.match(r"trace_mirt_k(\d+)(.*)",
                                trace_path.stem).groups()
    if (int(file_K), file_tag) != (spec.K, folder_tag):
        raise ValueError(f"spec round-trip failed: trace {trace_path.name!r} names "
                         f"K={file_K} in results folder {trace_path.parent.name!r}, "
                         f"against a recovered spec of K={spec.K}")
    want = FitSpec(K=spec.K, **_parse_tag(spec.tag, spec.drop_benchmarks))
    got = FitSpec(K=spec.K, **_parse_tag(folder_tag, spec.drop_benchmarks))
    # `floors` and `pooled_noise` are compared loosely: their tag grammar
    # changed twice (ON used to emit `_floors`/`_poolednoise`; today ON is the
    # untagged default and OFF emits `_nofloors`/`_unpooled`), so a legacy
    # folder can disagree with the recovered spec's own tag on those two fields
    # by construction — e.g. a --no-pooled-noise fit from before the `_unpooled`
    # token, whose attrs say False while its folder carries nothing. New fits
    # still round-trip exactly (both sides emit the same opt-out token). Every
    # other flag is compared strictly.
    # `beta_binomial` too: ON is untagged, and every trace from before 2026-10-01 sits in an
    # untagged folder with a Beta likelihood. And `effort_means`, untagged when ON since
    # 2026-10-03, every lineage trace from before sitting in an untagged folder without them.
    if replace(got, floors=want.floors, pooled_noise=want.pooled_noise,
               beta_binomial=want.beta_binomial, effort_means=want.effort_means) != want:
        raise ValueError(f"spec round-trip failed: folder tag {folder_tag!r} names "
                         f"a different flag set than the recovered spec, whose own "
                         f"tag is {spec.tag!r}")


def _check_dims(idata, data) -> None:
    """Trace dims against the loaded scope, on models AND benchmarks."""
    sizes = idata.posterior.sizes
    for dim, n, what in (("model", data.n_models, "models"),
                         ("bench", data.n_benchmarks, "benchmarks")):
        if dim in sizes and int(sizes[dim]) != n:
            raise RuntimeError(
                f"trace has {int(sizes[dim])} {what} but this scope has {n} — the "
                "trace was fit on a different data generation and cannot be "
                "indexed against it.")


def _trace_human_prior(post, field: str, legacy: float) -> float:
    """One human-prior value a posterior was sampled with: its mirt_spec's `field`, else the
    `mirt_<field>` attr the spec-less canonical K=1 fit writes, else `legacy` for a trace saved
    before either existed. Reads attrs only, so it serves a FitView or a bare posterior group
    as well as `FitSpec.from_trace`."""
    a = post.attrs
    spec = json.loads(a["mirt_spec"]) if a.get("mirt_spec") else {}
    if field in spec:
        return float(spec[field])
    if a.get(f"mirt_{field}"):
        return float(json.loads(a[f"mirt_{field}"]))
    return legacy


def trace_human_delta_sd(post) -> float:
    """The tier increment sigma a posterior was sampled with (`_trace_human_prior`)."""
    return _trace_human_prior(post, "human_delta_sd", LEGACY_HUMAN_DELTA_SD)


def trace_human_root_sd(post) -> float:
    """The human roots' base sd a posterior was sampled with (`_trace_human_prior`)."""
    return _trace_human_prior(post, "human_root_sd", LEGACY_HUMAN_ROOT_SD)


def sampled_human_order(post, spec: FitSpec):
    """The tier order a posterior was sampled with: its `mirt_human_order` attr, else the
    spec's (a trace saved before the attr existed). A rebuild of the model from a trace goes
    through here, not through `spec.human_order`, which follows config and so moves when
    config's order does (the committee chain, 2026-10-07)."""
    raw = post.attrs.get("mirt_human_order")
    return json.loads(raw) if raw else spec.human_order


def spec_json(spec: FitSpec) -> str:
    """The spec as the `mirt_spec` posterior attr."""
    return json.dumps(asdict(spec))
