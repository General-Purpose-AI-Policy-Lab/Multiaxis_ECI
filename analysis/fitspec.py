"""FitSpec: one object owning a fit's identity.

A fit is its flag set. From that follow the tag, the results and plots folders,
the trace filename, and the data scope the trace must be scored against. This
module is the single owner of all four, so a folder name, a trace name and a
plotted data set cannot drift apart.

`from_trace` recovers the spec of an already-fitted trace. Three sources, in
order: the `mirt_spec` JSON attr, which fit.py writes, the individual `mirt_*`
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

from config import (
    HUMAN_ORDER, HUMAN_ORDER_MERGED, PLOTS_DIR, RESULTS_DIR, SG_MODEL_NAME,
)

# Loading priors, longest first: "_signed" is a prefix of "_signedhs".
# "signedhs" is parse-only — the prior is gone, its historical folder names are not.
_PRIOR_TOKENS = ("signedhs", "signed", "pt1", "bifactor")


@dataclass(frozen=True)
class FitSpec:
    """The flag set of one compensatory MIRT fit.

    Field names match fit.py's CLI flags. Hashable, so a caller can memoize a
    loaded data scope on the spec itself.
    """
    K: int
    loading_prior: str = "normal"
    link: str = "linear"
    human_prior: bool = False
    human_merge: bool = False
    lineage_prior: bool = False
    lineage_bm: bool = False
    time_prior: bool = False
    theta_t: bool = False
    theta_pos: bool = False
    no_sg: bool = False
    apply_exclusions: bool = False
    cyber: bool = False
    simpleqa_original: bool = False
    drop_benchmarks: tuple[str, ...] = ()
    private_bases: bool = False
    # Fixed-c 3PL floors are the basic likelihood, not an option: a below-chance
    # score reads as uninformative-low ability instead of a point demand. The
    # floors come from file and add no parameters, so there is nothing to tune.
    # The tag carries no token for it, the same as `pooled_noise`.
    floors: bool = True
    ceiling_noise: bool = False
    known_se: bool = False
    # Hierarchical sigma_b is the basic noise model, not an option: thin
    # benchmarks shrink to the shared median instead of keeping a free scale
    # the panel cannot pin. The tag carries no token for it, so a fit that
    # turns it off is identified by its `mirt_spec` attr, not by its folder.
    pooled_noise: bool = True

    def __post_init__(self):
        # The three cross-flag constraints the model cannot express on its own.
        # Checked here so every construction path pays them, not just the CLI.
        if self.lineage_bm and not self.lineage_prior:
            raise ValueError(
                "--lineage-bm re-indexes the lineage increments by time; it "
                "needs --lineage-prior.")

    # ── identity ────────────────────────────────────────────────────────────
    @property
    def tag(self) -> str:
        """One tag identifying the fit config; reused for BOTH the results/plots
        folder and the trace filename, so distinct configs never overwrite each
        other's fixed-name artefacts. The loading prior leads the tag (matching
        the historical `trace_mirt_k3_signed_...` convention); "normal" stays
        untagged so its folders reduce to the canonical mirt_humanprior etc."""
        tag = "" if self.loading_prior == "normal" else f"_{self.loading_prior}"
        if self.link == "loglog": tag += "_loglog"
        if self.human_merge: tag += "_humanmerge"
        elif self.human_prior: tag += "_humanprior"
        if self.lineage_prior: tag += "_lineageprior"
        if self.lineage_bm: tag += "_lineagebm"
        if self.time_prior: tag += "_timeprior"
        if self.theta_t: tag += "_thetat"
        if self.theta_pos: tag += "_thetapos"
        if self.no_sg: tag += "_noSG"
        if self.apply_exclusions: tag += "_excluded"
        if self.cyber: tag += "_cyber"
        if self.simpleqa_original: tag += "_sqaorig"
        if self.drop_benchmarks:
            tag += "_drop" + "".join(re.sub(r"\W", "", b)
                                     for b in self.drop_benchmarks)
        if self.private_bases: tag += "_privbase"
        if self.ceiling_noise: tag += "_ceilnoise"
        if self.known_se: tag += "_knownse"
        return tag

    @property
    def results_dir(self) -> Path:
        return RESULTS_DIR / (f"mirt{self.tag}" if self.tag else "mirt")

    @property
    def plots_dir(self) -> Path:
        """Figures land here. K is in the name, unlike `results_dir`: two fits of
        one flag set differing only in K share a results folder but must not
        overwrite each other's figures."""
        return PLOTS_DIR / f"mirt_k{self.K}{self.tag}"

    @property
    def trace_path(self) -> Path:
        return self.results_dir / f"trace_mirt_k{self.K}{self.tag}.nc"

    @property
    def baseline_trace_path(self) -> Path:
        """The K=1 baseline fit.py fits beside a K-axis fit. It carries no tag in
        its filename and no attrs at all, so its spec is reconstructed from the
        folder plus fit.py's baseline rule (see `_baseline_spec`)."""
        return self.results_dir / "trace_mirt_k1.nc"

    @property
    def human_order(self):
        """The human tier order this fit imposes, or None."""
        return (HUMAN_ORDER_MERGED if self.human_merge
                else HUMAN_ORDER if self.human_prior else None)

    # ── construction ────────────────────────────────────────────────────────
    @classmethod
    def from_args(cls, args, parser=None) -> "FitSpec":
        """Spec from fit.py's parsed exploration flags. `parser` turns a flag
        conflict into an argparse error instead of a traceback."""
        try:
            return cls(
                K=args.K, loading_prior=args.loading_prior, link=args.link,
                human_prior=args.human_prior, human_merge=args.human_merge,
                lineage_prior=args.lineage_prior, lineage_bm=args.lineage_bm,
                time_prior=args.time_prior, theta_t=args.theta_t,
                theta_pos=args.theta_pos,
                no_sg=args.no_sg,
                apply_exclusions=args.apply_exclusions, cyber=args.cyber,
                simpleqa_original=args.simpleqa_original,
                drop_benchmarks=tuple(args.drop_benchmarks or ()),
                private_bases=args.private_bases,
                floors=args.floors,
                ceiling_noise=args.ceiling_noise, known_se=args.known_se,
                pooled_noise=args.pooled_noise)
        except ValueError as e:
            if parser is None:
                raise
            parser.error(str(e))

    @staticmethod
    def from_trace(idata, trace_path) -> "FitSpec":
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
            spec = FitSpec(**d)
        else:
            attr = _attr_flags(post)
            tag = _folder_tag(trace_path)
            spec = FitSpec(K=K, **{**_parse_tag(tag, attr.get("drop_benchmarks", ())),
                                   **attr})
            if is_baseline:
                spec = _baseline_spec(spec)
        if not is_baseline:
            _check_round_trip(spec, trace_path)
        return spec

    # ── the data the trace was fit on ───────────────────────────────────────
    def load_data(self, idata=None):
        """The fit's data scope: `(data, floor_c, n_eff)`.

        Same loads, drops, clips and prints fit.py runs before sampling, so a
        plot or a GoF number is scored on exactly the observations the trace saw.
        Pass `idata` to have the trace's own `model` / `bench` dims checked
        against it — a trace from an older data generation cannot be indexed
        against today's data at all, and the benchmark-count mismatch is the one
        that fails silently.
        """
        from data import (
            clip_scores_to_floors, drop_model_observations,
            load_benchmark_floors, load_eci_data,
        )
        data = load_eci_data(
            include_all_benchmarks=not self.apply_exclusions,
            fit_cyber=self.cyber,
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
            print(f"  --floors: fixed-c 3PL; clipped {n_before} below-floor scores up "
                  f"to their benchmark chance floor", flush=True)
        if self.known_se:
            measured = np.isfinite(data.n_eff)
            print(f"  --known-se: {int(measured.sum())} of {data.n_obs} cells "
                  f"({measured.mean():.1%}) carry a reported stderr, median "
                  f"effective test length {np.median(data.n_eff[measured]):.0f} "
                  f"tasks; the rest keep the estimated per-benchmark noise",
                  flush=True)
        return data, floor_c, (data.n_eff if self.known_se else None)

    # ── lazy posterior access ───────────────────────────────────────────────
    def open_posterior(self, keep=None, thin: int = 1, chains=None, path=None):
        """The trace's posterior as an InferenceData holding only `keep`, thinned.

        The flagship trace is 38 GB against 26 GB of RAM, so a consumer names the
        variables it needs instead of loading the file. Names absent from this
        trace are skipped, so one `keep` list covers every link variant.
        Attributes survive the subset, so `prepare_fit` still reads the display
        rotation, the axis names and the loading prior off the posterior.
        """
        import xarray as xr
        with xr.open_dataset(path or self.trace_path, group="posterior") as post:
            sub = post if keep is None else post[[v for v in keep if v in post]]
            if thin > 1:
                sub = sub.isel(draw=slice(None, None, thin))
            if chains is not None:
                sub = sub.isel(chain=chains)
            sub = sub.load()
        return az.InferenceData(posterior=sub)


# ── the flagship fit ────────────────────────────────────────────────────────
# K=4, positive loadings, merged human order + Brownian lineage prior, 3PL
# floors, pooled noise, the exploration scope. THE forecasting base.
# FrontierMath v1 and AlgoTune are out of this scope through the retirement
# list, so no drop flag names them and the tag carries no `_drop`.
# `loading_prior` and `floors` are named explicitly although both hold the
# default: the flagship is a statement of one fit, and a later default change
# must not silently redefine it.
FLAGSHIP = FitSpec(K=4, loading_prior="normal", human_merge=True,
                   lineage_prior=True, lineage_bm=True, floors=True)
# The one path every consumer reads, so the figures, the dashboard card and the
# post cannot name different traces. `FLAGSHIP_TRACE.parent` is the fit's
# results folder and holds its CSVs and caches. 10 chains x 12,000 draws over
# 829 test-takers and 96 benchmarks, which is the current exploration scope.
FLAGSHIP_TRACE = (RESULTS_DIR
                  / "mirt_humanmerge_lineageprior_lineagebm"
                  / "trace_mirt_k4_humanmerge_lineageprior_lineagebm.nc")
# One posterior mode over all ten chains (`mirt_modes_<trace-stem>.json`), so
# every chain is read. A multimodal flagship would name its majority here and
# `open_flagship` would default to it.
FLAGSHIP_CHAINS = None
# Every flagship summary is a median or an interval, which 20,000 draws pin as
# well as 200,000.
FLAGSHIP_THIN = 10


def open_flagship(keep=None, thin: int = FLAGSHIP_THIN, chains=FLAGSHIP_CHAINS):
    """The flagship posterior, thinned. `chains=None` reads every chain."""
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
        if order == json.loads(json.dumps(HUMAN_ORDER_MERGED)):
            out["human_merge"] = True
        elif order == json.loads(json.dumps(HUMAN_ORDER)):
            out["human_prior"] = True
        # Neither order: the folder tag decides which flag it was, since
        # guessing here would silently change the data scope.
    if a.get("mirt_lineage_chains"):
        out["lineage_prior"] = True
    if json.loads(a.get("mirt_lineage_bm", "false")):
        out["lineage_bm"] = True
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
    if "mirt_drop_benchmarks" in a:
        out["drop_benchmarks"] = tuple(json.loads(a["mirt_drop_benchmarks"]))
    if "mirt_floor_c" in a:
        out["floors"] = True
    if a.get("mirt_known_se"):
        out["known_se"] = True
    if a.get("mirt_pooled_noise"):
        out["pooled_noise"] = True
    return out


def _folder_tag(trace_path: Path) -> str:
    """The tag carried by the trace's PARENT DIRECTORY (`results/mirt{tag}`).

    Read from the folder, not the filename, because a K=1 baseline trace is
    named `trace_mirt_k1.nc` with no tag at all.
    """
    name = trace_path.parent.name
    if name != "mirt" and not name.startswith("mirt_"):
        raise ValueError(f"{trace_path.parent.name!r} is not a MIRT results "
                         "folder (expected mirt or mirt_<tag>)")
    return name[4:]


def _parse_tag(tag: str, drop_benchmarks: tuple = ()) -> dict:
    """The flags a results-folder tag encodes, consumed in the order fit.py
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
    if take("_loglog"):
        out["link"] = "loglog"
    if take("_humanmerge"):
        out["human_merge"] = True
    elif take("_humanprior"):
        out["human_prior"] = True
    for tok, fl in (("_lineageprior", "lineage_prior"), ("_lineagebm", "lineage_bm"),
                    ("_timeprior", "time_prior"), ("_thetat", "theta_t"),
                    ("_thetapos", "theta_pos")):
        if take(tok):
            out[fl] = True
    for tok, fl in (("_noSG", "no_sg"), ("_excluded", "apply_exclusions"),
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
    # `tag` no longer emits `_floors`, but folders written while it did must
    # still parse, and there an ABSENT token means floors OFF: three attrless
    # traces sit in such folders (mirt_humanprior, mirt_loglog) and were fit
    # without floors, so falling through to the now-True default would score
    # them against clipped data. The token is parsed because live traces carry
    # it; a token zero traces carry (`_ceilings`) raises as unrecognized instead.
    out["floors"] = take("_floors")
    for tok, fl in (("_ceilnoise", "ceiling_noise"), ("_knownse", "known_se")):
        if take(tok):
            out[fl] = True
    # `tag` no longer emits this token; folders written while it did must still
    # parse, and they name the value the default already holds.
    if take("_poolednoise"):
        out["pooled_noise"] = True
    if rest:
        raise ValueError(f"unrecognized tag token {rest!r} in folder tag {tag!r} "
                         "— refusing to guess the fit's data scope")
    return out


# The model-side flags fit.py does NOT forward to the K=1 baseline (fit.py's
# baseline call passes floors / ceiling_noise / known_se only). Every
# other model-side flag is off there whatever the folder tag says; the data-scope
# flags are the folder's.
_BASELINE_OFF = dict(loading_prior="normal", link="linear",
                     human_prior=False, human_merge=False,
                     lineage_prior=False, lineage_bm=False, time_prior=False,
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
    # `floors` is no longer a folder-identity flag: a legacy folder carries
    # `_floors` and the spec's own tag cannot emit it, so the two sides disagree
    # on that field by construction. Every other flag is still compared.
    if replace(got, floors=want.floors) != want:
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


def spec_json(spec: FitSpec) -> str:
    """The spec as the `mirt_spec` posterior attr."""
    return json.dumps(asdict(spec))
