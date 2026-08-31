"""Behavior locks for the repo reorganization.

Golden values for the three `data_all` (include_all_benchmarks=True)
configurations were re-pinned on 2026-07-27: `_load_human_baselines_as_models`
now CLIPS human scores into the open interval instead of dropping the two rows
that sit exactly on it (ARC-AGI-2 / Committee of Average Humans and VPCT /
Average Human, both 0.999), matching how AI boundary rows are already treated.
Those two benchmarks are on the curated exclusion list, so only include_all
scopes see them: each affected golden moved by exactly -325.632421.

All goldens were re-pinned on 2026-07-27 when the pipeline began auto-swapping
its output into 1_data/processed (97 -> 96 benchmarks after BTF3 was parked for
lacking a sourceable chance floor), and again when FrontierMath v1 was restored
as its own pair of items (96 -> 98 benchmarks, 4447 obs / 765 models), and a
third time when GBAEval's two harness-failure rows were dropped (4447 -> 4445
obs; Grading failed=143 of ~145 tasks, so the 0.000 measured the runner).

The four --floors configurations were also re-pinned after a
floor correction: MindCube's chance baseline is its published 32.35%, not 1/4
(one old/weak model, LongVA-7B at 0.295, now clips up), and SpatialViz-Bench
was set to the paper's own 25.08% Random row. Only floors-aware fits move.

Every lineage-dependent golden moved again the same day when the gpt 5.6
codename releases were re-filed. sol / terra / luna are peer model TYPES, not a
flagship/pro/mini ladder: sol continues the flagship spine (+0.027 over 5.5,
above on 20/28 effort-matched cells) while terra and luna sit 0.067 and 0.198
BELOW it (above on 3/27 and 0/27), so each founds its own chain rather than
branching off a step prior with a positive mean. Sol's "Max + Pro" rows are an
advanced harness, not a release, and leave the chain entirely. This moves the
`signed` goldens too, unlike a pure theta-side change: dropping two singleton
chains takes the lineage block from 148 to 146 deltas, and the parameter count
enters the prior term at the initial point.

All goldens were re-pinned on 2026-08-04 after a data refresh plus three
curation fixes: the 2026-08-04 Epoch snapshot (+291 cells, 34 gone upstream), the
OSWorld original board ingested as two new benchmarks (section 03e), and
`gemini-2.0-flash-02-05` aliased onto `gemini-2.0-flash-001` (a SEAL date-style
name for the same 2025-02-05 release, which had been splitting one test-taker).
Scope: 4,748 rows / 808 models / 101 benchmarks. Data only; no model math moved.

All goldens were re-pinned again on 2026-07-30 on the refreshed data
(3,860 obs / 718 models / 91 benchmarks curated; 4,567 / 762 / 99 on the full
set). Two causes, both data: the refresh itself, and a fix to
human_baselines.csv, where ' Skilled Generalist' carried a leading space and so
forked the tier into a second, prior-less test-taker (762 -> 763 models). Human
rows enter as IRT test-takers, so either moves every configuration. The loader
now strips the tier/benchmark keys, which is what stops that recurring.

All goldens were re-pinned again on 2026-08-04 on the refreshed data (4,080 obs
/ 770 models / 93 benchmarks curated; 4,804 / 818 / 101 on the full set). Two
causes. The refresh moves everything, ~9,200 nats on the plain configurations.
On top of that the eight lineage configurations carry the GPT restructure in the
release map: gpt-3.5-turbo left the flagship chain for its own founder level (it
is the cheap tier, and a lineage step has a positive prior mean, so it cannot be
a step off GPT-4), the GPT-4 Turbo previews were re-filed from the GPT-4 node
onto the Turbo tier they belong to, and the ChatGPT-tuned 4o snapshots became a
branch off 4o 2024-11-20 rather than two rungs of the API spine. 174 -> 179
nodes, so those eight moved a further 28 to 474 nats.

The nine lineage-dependent goldens were re-pinned on 2026-08-06 when terra and
luna were re-filed from their own single-node chains (which
`build_lineage_structure` drops, leaving all 14 effort variants as free
test-takers) onto the flagship chain as parented off-spine branches of the 5.5
node. Each now takes one Brownian delta from 5.5 and its variants — including
the effort:unknown leaderboard rows — tie to the node through the variant
offsets. 184 -> 186 nodes, 147 -> 149 deltas. The signed configurations move
~5.5 nats (parameter-count prior terms only); the normal/bifactor ones move
3.4-4.6k nats because the initial-point BM deltas (drift*dt) now shift eta on
the 92 terra/luna observations.

All goldens were re-pinned later on 2026-08-06 after the Inkling alias merge
(same-day pipeline rerun on the same snapshot: 4,941 rows both sides, zero
score deltas). Seven Inkling spellings collapse to five test-takers —
`Inkling (xHigh)` onto `Inkling_xhigh`, `Inkling (Thinking)` into the
`_unknown` bucket, `Inkling-small` onto `inkling-small` — so the full set goes
818 -> 817 models. Likelihood terms are unchanged; every value moves by
exactly (net -2 test-takers) x K x log(1/sqrt(2*pi)) in theta prior terms
(+1.84 nats at K=1, +5.51 at K=3, +7.35 at K=4).

All goldens were re-pinned a third time on 2026-08-06 after the
missing-models queue was cleared (same snapshot again): five Kaggle spellings
aliased onto known test-takers (deepseek-v3 / v3.1 onto DeepSeek-V3 / V3.1,
grok-4.5-0708 onto grok-4.5_unknown, open-mixtral-8x22b-2404 onto
open-mixtral-8x22b, claude-sonnet-4-5-thinking-20250929 into the _unknown
bucket) and seven real new models added to reviewed_models.txt. 817 -> 812
models, 4,941 -> 4,940 rows (one collision deduped), and one score moved
(DeepSeek-V3 on MMLU 0.872 -> 0.895, the kaggle-wins policy applying to the
merged row), so unlike the Inkling pass the shifts are heterogeneous
(-40 to +51 nats).

All goldens were re-pinned a fourth time on 2026-08-06 for three hand-added
Domain Expert baselines in human_baselines.csv — GDPval 0.5 (strict-win-rate
parity by construction), EBR-bench 0.95 (n=1 experienced player, provisional)
and BlueprintBench 2 0.59 (n=1, provisional) — the human-side tie-breaker
cells for the K=3 axis-3 mode competition. Human rows enter every fit, so all
24 values move. `_none` also joined the effort-suffix vocabulary in
data._EFFORT_SUFFIX_RE the same day; verified inert here (no default-path
consumer — only collapse_effort_variants and the lineage collapse read it).

21 goldens were re-pinned a fifth time on 2026-08-06 after the name-audit
corrections (3_diagnostics/audit_model_names.py; same snapshot, zero score
deltas). The pipeline's Name-config parser now consumes an optional
"Thinking," prefix, so Opus 4.5's "(Thinking, None)" ARC pair moved off the
bare id onto claude-opus-4-5-20251101_none; gpt-5.2's AlgoTune row was
refiled _medium -> _high via a curated drop+add (the board's own label wins
over Epoch's suffix); and five alias rows normalised paren-style ids
(claude-opus-5 (xhigh), the (16K/24K thinking) Geminis, QwQ-32B) onto
standard suffixes, including the 23K -> 24K budget typo. Test-taker count is
unchanged at 812 (splits balanced merges). Three configurations did not move:
the two signed data_all floors fits and nc_k3_full (signed A is 0 at the
initial point, so eta = -D there and a pure test-taker re-labelling with
unchanged scores cannot reach their initial logp).

The full-scope goldens were re-pinned a sixth time on 2026-08-06 for the
Inkling Small ARC ladder. A brief reattribution of Epoch's base-Inkling ARC
pair onto inkling-small_high was REVERTED the same hour: the board's own
table puts Small (High) at 0.78/0.331, matching neither 0.795 nor 0.365278,
and base Inkling is a separate entry at exactly those values, so Epoch was
right. What stays is the transcribed Small effort ladder as curated adds -
12 cells, none/minimal/low/medium/high/xhigh on both ARC benchmarks
(ARC-AGI-2 none is an exact zero, clipped like every boundary row). 812 ->
818 models. ARC-AGI and ARC-AGI-2 are curated-excluded, so only include_all
configurations move.

All goldens were re-pinned on 2026-08-07 on the fresh snapshot (a new UTC day,
so the pipeline fetched rather than reusing 2026-08-06): 5,064 rows / 835
models / 100 benchmarks, 13 score deltas upstream, plus two hand-added
Artificial Analysis cells for Inkling_xhigh (SimpleQA Verified 0.439, GPQA
Diamond 0.872; effort filing pinned by AA's MCP Atlas 76.0% matching our SEAL
row exactly). The 22-name review queue was all effort rungs of known models
(half of them our own 2026-08-06 normalization products under their new
canonical names) and went to reviewed_models.txt wholesale.

The lineage-dependent goldens were re-pinned later on 2026-08-07 for the
single-node-chain rule: build_lineage_structure now keeps a chain whose one
node retains >=2 (node, variant) groups — zero Brownian deltas, variant
offsets only — so a first-release vendor's effort rungs tie to one ability
(Inkling: base + high/xhigh/unknown now share a node, which is what lets the
AA cells inform the base straddler; Inkling Small likewise with its 6-rung ARC
ladder). Also filed: grok-3-mini as a real two-node chain (beta 2025-04-09 ->
GA 2025-06-24), and DeepSeek-R1_high onto the R1 node — the HOST regex strips
'deepseek-' so the row arrived as 'r1_high', where r1 fails ('_' is a
word char); the r1 pattern now accepts r1[-_]. 40 chains / 190 nodes /
150 deltas / 467 offset groups.

The lineage goldens were re-pinned once more on 2026-08-07 after the
unchained-tail sweep: eight families with real structure were filed as chains
— gpt-oss-120b / gpt-oss-20b (4 variant groups each), gemma-4-31b-it (3),
nova-2.0-pro-preview (3), muse-spark as a real 2-node chain (spark 1
2026-04-08 -> 1.1 2026-07-09), qwq (Preview 2024-11-28 -> 32B 2025-03-05),
qwen-coder (480B 2025-07-31 -> next 2026-02-02) and deepseek-flash (V4-flash
2026-04-24 -> 0731). Every gemma id now founds its own single-node chain
(lone ones drop at fit time as before). 48 chains / 202 nodes / 154 deltas /
496 offset groups.

One golden was ADDED on 2026-08-19 (no re-pin): --theta-pos, the positive
likelihood-side ability of the semi-compensatory convention — eta reads
theta_pos = softplus(theta), raw theta keeps every prior block and stays the
reported ability. Opt-in and off everywhere else, so no existing value moves.

Two goldens were ADDED on 2026-08-20 (no re-pin): link="loglog", the
log-logistic IRF eta = alpha_b * logsumexp_k(theta_k + log A_bk). At the
initial point theta = 0 and the row-centered log-loading mix is exactly 0
(ZeroSumNormal), so log A collapses to 0 and the aggregate is logsumexp of K
equal (zero) terms, i.e. alpha_b * log(K). loading_prior="normal" is the only
prior link="loglog" accepts, so each new golden's flag-off twin is the
existing normal-prior golden at the same K. Opt-in and off everywhere else, so
no existing value moves.

The two theta_t goldens were re-pinned on 2026-08-18 when the per-cell block
became a direct re-centered Student-t: pm.StudentT cells replace the
ZeroSumNormal x Gamma-precision scale mixture, whose n*K latent scales each
formed a funnel with their coordinate (the divergence source observed in the
theta-t runs). Same marginal t(4), same coverage, same exact zero-sum pin;
at the initial point the gap to each no-flag twin is now the t(4) densities
at 0 minus the replaced ZeroSumNormal terms. No other golden moves.

All 29 goldens were re-pinned on 2026-08-27 for the retirement of two
benchmarks. FrontierMath v1 and AlgoTune moved to
1_data/curated/retired_benchmarks.txt, which load_eci_data drops for every fit
before any scope flag is read, so no configuration sees them: v1 is superseded
by v2 (about +0.20 level shift on Tiers 1-3, ranking intact at paired
r = 0.988) and AlgoTune's 1 - 1/speedup score shares its scale with no other
benchmark. Scope: 4,265 obs / 787 models / 90 benchmarks curated;
5,004 / 835 / 98 on the full set. Data only; no model math moved. Every value
rises (fewer likelihood terms), by +4,655 to +4,693 nats on the plain
configurations. The floors+ceilings golden moves only +1,710 because
FrontierMath v1 also carried one of the two curated ceilings, so its fixed-d
term leaves with it.

All 29 goldens were re-pinned a second time later on 2026-08-27, for a
human-baselines curation pass: 4 rows left human_baselines.csv (two duplicate
RAND Domain Expert rows on MMLU Biology/Chemistry, whose pipeline-side cause
in sections 03b/08 of pipeline.ipynb was fixed the same day; the MATH Level 5
Top Performer row; and the GDPval Domain Expert 0.5 parity anchor). Pinned
against human_baselines.csv blob 95d4952 (commit 1f464e6). Scope:
4,265 -> 4,261 obs curated, 5,004 -> 5,000 on the full set; models and
benchmarks unchanged. Data only; no model math moved. Values rise by +18 to
+106 nats depending on how many of the 4 rows a configuration's scope carried.

All 29 goldens were re-pinned a third time on 2026-08-27 for the retirement of
FrontierMath Tier 4 v1, which joins FrontierMath v1 and AlgoTune in
1_data/curated/retired_benchmarks.txt on the same rule: both v1 columns are
superseded by the v2 problem set, and Tier 4 v1 additionally caps at 0.60
answerable items where v2 does not, so no configuration may pool the two
versions in one loading row. Scope: 4,261 -> 4,189 obs / 787 -> 781 models /
90 -> 89 benchmarks curated; 5,000 -> 4,928 / 835 -> 829 / 98 -> 97 on the
full set. Data only; no model math moved. Values rise by +2,299 to
+12,250 nats (72 fewer likelihood terms, and every prior block that indexes
models or benchmarks loses its rows). The floors+ceilings golden moved least
and landed exactly ON the floors-only golden, Tier 4 v1 having carried the
last curated ceiling in scope; the fixed-ceiling apparatus was removed
outright in the next entry, so that golden is gone.

The remaining 28 goldens were re-pinned a fourth time on 2026-08-27, for the
retirement of MindCube: 5 test-takers, four of them 2024-era open 7B models
clustered within 0.03 of the 0.3235 chance floor, a panel too thin and too
flat to identify a difficulty and a loading row. Scope: 4,189 -> 4,184 obs /
781 models / 89 -> 88 benchmarks curated; 4,928 -> 4,923 / 829 / 97 -> 96 on
the full set. Values rise by +0.6 to +95.6 nats.

The 29th golden, floors + fixed-d ceilings, was DELETED the same day with the
fixed-ceiling apparatus itself: --ceilings, 1_data/curated/benchmark_upper_bounds.csv
and the ceiling_d argument of build_mirt_model are gone, both curated walls
having been retired with their benchmarks. That the golden had already
collapsed onto the floors-only value is the evidence it locked nothing. The
--ceiling-noise flag STAYS: its Beta(1, 20) gap is estimated, not read from a
file, and it now sits under d_hi = 1 rather than under a curated wall. No
value moves for it — the composition was inert wherever no wall was in scope,
which since the retirements is everywhere.

Three guards:
- Golden initial-point log-probabilities per model configuration — any
  unintended change to the model math fails these.
- Public-API surface — every cross-module name the scripts import must stay
  importable through the package splits.
- Builder coverage for the sparse-gate and interaction models.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pymc as pm
import pytest

from multiaxis_eci.config import HUMAN_ORDER, HUMAN_ORDER_MERGED
from multiaxis_eci.data import load_eci_data
from multiaxis_eci.lineage import build_lineage_structure
from multiaxis_eci.models.mirt import _human_structure, build_mirt_model
from multiaxis_eci.models.mirt_interaction import build_mirt_interaction_model
from multiaxis_eci.models.mirt_nc import build_mirt_nc_model
from multiaxis_eci.models.mirt_sparse import build_mirt_sparse_model

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def data():
    return load_eci_data()


@pytest.fixture(scope="module")
def data_all():
    return load_eci_data(include_all_benchmarks=True)


def total_logp(model) -> float:
    return float(model.compile_logp()(model.initial_point()))


class TestGoldenLogp:
    """Initial-point total logp per configuration, pinned on the current data.

    The initial point is deterministic (transformed-space zeros), so these
    values are exact up to floating-point noise. rtol=1e-6 ≈ 0.06 on the
    ~6e4 magnitudes here — tight enough to catch any real math change.

    Pinned on the 2026-07-30 data state (4,567 obs / 762 models / 99 benchmarks
    on the full set). A legitimate data refresh moves every value here:
    recompute all of them in one pass and re-pin together, never one at a time.
    """
    RTOL = 1e-6

    def test_mirt_k1_normal(self, data):
        np.testing.assert_allclose(
            total_logp(build_mirt_model(data, K=1, loading_prior="normal")),
            -74656.684524, rtol=self.RTOL)

    def test_mirt_k3_normal(self, data):
        np.testing.assert_allclose(
            total_logp(build_mirt_model(data, K=3, loading_prior="normal")),
            -76217.967914, rtol=self.RTOL)

    def test_mirt_k1_pt1(self, data):
        np.testing.assert_allclose(
            total_logp(build_mirt_model(data, K=1, loading_prior="pt1")),
            -74672.762537, rtol=self.RTOL)

    def test_mirt_k3_pt1(self, data):
        np.testing.assert_allclose(
            total_logp(build_mirt_model(data, K=3, loading_prior="pt1")),
            -76266.201954, rtol=self.RTOL)

    def test_mirt_k3_signed(self, data):
        np.testing.assert_allclose(
            total_logp(build_mirt_model(data, K=3, loading_prior="signed")),
            -76268.958770, rtol=self.RTOL)

    def test_mirt_k3_signed_full_options(self, data):
        bench = data.blookup["benchmark"].tolist()
        model = build_mirt_model(
            data, K=3, loading_prior="signed", human_order=HUMAN_ORDER,
            lineage=build_lineage_structure(data.mlookup),
            plt_founders=[bench[5], bench[10], bench[20]])
        np.testing.assert_allclose(
            total_logp(model), -76750.248574, rtol=self.RTOL)

    def test_mirt_k3_normal_human_merged(self, data):
        model = build_mirt_model(data, K=3, loading_prior="normal",
                                 human_order=HUMAN_ORDER_MERGED)
        np.testing.assert_allclose(
            total_logp(model), -76241.102891, rtol=self.RTOL)

    def test_mirt_k3_signed_lineage_bm(self, data):
        model = build_mirt_model(
            data, K=3, loading_prior="signed",
            lineage=build_lineage_structure(data.mlookup), lineage_bm=True)
        np.testing.assert_allclose(
            total_logp(model), -76753.725223, rtol=self.RTOL)

    def test_mirt_k3_normal_lineage_bm(self, data):
        # The signed BM goldens are blind to the delta formula (signed A is 0
        # at the initial point, so eta = -D regardless of theta). With normal
        # loadings A != 0 and the initial-point deltas equal drift*dt, so this
        # value moves if the BM step math changes (iid sibling differs by >13k nats).
        model = build_mirt_model(
            data, K=3, loading_prior="normal",
            lineage=build_lineage_structure(data.mlookup), lineage_bm=True)
        np.testing.assert_allclose(
            total_logp(model), -164610.744003, rtol=self.RTOL)

    def test_mirt_k3_signed_bm_full_options(self, data):
        bench = data.blookup["benchmark"].tolist()
        model = build_mirt_model(
            data, K=3, loading_prior="signed", human_order=HUMAN_ORDER,
            lineage=build_lineage_structure(data.mlookup), lineage_bm=True,
            plt_founders=[bench[5], bench[10], bench[20]])
        np.testing.assert_allclose(
            total_logp(model), -76750.248574, rtol=self.RTOL)

    def test_mirt_k3_normal_time_prior(self, data):
        # Config lock only: at the initial point time_beta = 0, so the trend
        # contributes nothing to theta and this value differs from its no-trend
        # twin by exactly the three Normal(0, PRIOR_TIME_BETA) prior terms
        # (-0.677). The shift ARITHMETIC is locked by
        # TestMIRT::test_time_prior_shifts_theta_by_beta_times_year instead.
        from multiaxis_eci.data import release_time_covariate
        lin = build_lineage_structure(data.mlookup)
        model = build_mirt_model(
            data, K=3, loading_prior="normal", human_order=HUMAN_ORDER,
            lineage=lin, time_t=release_time_covariate(data.mlookup, lin))
        np.testing.assert_allclose(
            total_logp(model), -162461.273670, rtol=self.RTOL)

    def test_mirt_k3_signed_theta_t(self, data):
        # Config lock: at the initial point every t cell is 0 and re-centering
        # is a no-op, so the likelihood term is identical to the plain signed
        # twin. The gap is exactly the n_models*3 Student-t(4) log-densities at
        # 0 (-1.0083 each) minus the ZeroSumNormal terms the block replaces.
        np.testing.assert_allclose(
            total_logp(build_mirt_model(data, K=3, loading_prior="signed",
                                        theta_t_cells=True)),
            -76416.725543, rtol=self.RTOL)

    def test_mirt_k3_normal_theta_t_full(self, data):
        # Same lock with both theta-structure priors on: the t block must cover
        # the unstructured rows PLUS the human-root / chain-founder slices and
        # nothing else, so the gap to its no-flag twin counts only those rows,
        # not every model.
        model = build_mirt_model(
            data, K=3, loading_prior="normal", human_order=HUMAN_ORDER,
            lineage=build_lineage_structure(data.mlookup), theta_t_cells=True)
        np.testing.assert_allclose(
            total_logp(model), -162528.152695, rtol=self.RTOL)

    def test_mirt_k3_normal_theta_pos(self, data):
        # Formula lock: at the initial point theta = 0 but softplus(0) = log 2,
        # so with normal loadings (A != 0 there) eta shifts on every
        # observation and this value moves if the positive link's arithmetic
        # changes. test_mirt_k3_normal is the flag-off twin.
        np.testing.assert_allclose(
            total_logp(build_mirt_model(data, K=3, loading_prior="normal",
                                        theta_pos=True)),
            -141221.873016, rtol=self.RTOL)

    def test_mirt_k3_normal_loglog(self, data):
        # Formula lock: at the initial point theta = 0 and logA_mix_z = 0
        # (ZeroSumNormal), so log A collapses to 0 on every cell and
        # eta = alpha_b * logsumexp_k(0, ..., 0) = alpha_b * log(K) — this
        # value moves if the LSE/alpha arithmetic changes. test_mirt_k3_normal
        # is the flag-off twin (loading_prior="normal" is the only prior
        # link="loglog" accepts).
        np.testing.assert_allclose(
            total_logp(build_mirt_model(data, K=3, loading_prior="normal",
                                        link="loglog")),
            -106096.900112, rtol=self.RTOL)

    def test_mirt_k1_loglog(self, data):
        # K=1 degeneracy: the ZeroSumNormal over a size-1 latent axis is
        # exactly 0, so logA_mix_z contributes nothing and the model is the
        # plain 2PL reparameterized with alpha as discrimination.
        # test_mirt_k1_normal is the flag-off twin.
        np.testing.assert_allclose(
            total_logp(build_mirt_model(data, K=1, loading_prior="normal",
                                        link="loglog")),
            -74615.691564, rtol=self.RTOL)

    def test_mirt_k3_normal_known_se(self, data):
        # Config lock AND formula lock: n_eff enters the Beta precision
        # deterministically, so unlike the theta-side priors this value moves if
        # the split arithmetic changes. 1,138 of 4,265 cells carry a reported
        # stderr, the rest reduce to the per-benchmark noise exactly.
        model = build_mirt_model(
            data, K=3, loading_prior="normal", human_order=HUMAN_ORDER,
            lineage=build_lineage_structure(data.mlookup), known_se=True)
        np.testing.assert_allclose(
            total_logp(model), -149388.164271, rtol=self.RTOL)

    def test_mirt_k3_normal_pooled_knownse(self, data):
        # The production combo: pooled noise on top of the known-SE split. At the
        # initial point z_b = 0 and mu_s = log 0.05, so sigma_b is 0.05 on every
        # benchmark — the same value the fixed prior starts at, hence the same
        # likelihood term as its known-SE-only twin above. This value differs
        # only through the prior terms of mu_s, tau_s and the 90 z_b.
        model = build_mirt_model(
            data, K=3, loading_prior="normal", human_order=HUMAN_ORDER,
            lineage=build_lineage_structure(data.mlookup), known_se=True,
            pooled_noise=True)
        np.testing.assert_allclose(
            total_logp(model), -191022.646777, rtol=self.RTOL)

    def test_mirt_k3_signed_floors(self, data_all):
        from multiaxis_eci.data import clip_scores_to_floors, load_benchmark_floors
        floors = load_benchmark_floors(data_all)
        clipped = clip_scores_to_floors(data_all, floors)
        model = build_mirt_model(clipped, K=3, loading_prior="signed",
                                 floor_c=floors)
        np.testing.assert_allclose(
            total_logp(model), -88403.000212, rtol=self.RTOL)

    def test_mirt_k4_bifactor(self, data):
        np.testing.assert_allclose(
            total_logp(build_mirt_model(data, K=4, loading_prior="bifactor")),
            -77302.963028, rtol=self.RTOL)

    def test_mirt_k4_bifactor_full_options(self, data_all):
        # The live exploratory base, bifactor loadings: both theta priors,
        # Brownian lineage, 3PL floors, estimated noise ceiling.
        from multiaxis_eci.data import clip_scores_to_floors, load_benchmark_floors
        floors = load_benchmark_floors(data_all)
        clipped = clip_scores_to_floors(data_all, floors)
        model = build_mirt_model(
            clipped, K=4, loading_prior="bifactor", human_order=HUMAN_ORDER,
            lineage=build_lineage_structure(clipped.mlookup), lineage_bm=True,
            floor_c=floors, ceiling_noise=True)
        np.testing.assert_allclose(
            total_logp(model), -184060.334716, rtol=self.RTOL)

    def test_mirt_k3_anchored(self, data):
        bench = data.blookup["benchmark"].tolist()
        model = build_mirt_model(
            data, K=3, loading_prior="normal",
            anchors={bench[5]: 0, bench[10]: 1, bench[20]: 2})
        np.testing.assert_allclose(
            total_logp(model), -76217.967914, rtol=self.RTOL)

    def test_nc_k3_full(self, data_all):
        from multiaxis_eci.fits.fit_nc import build_qmatrix
        Q, _ = build_qmatrix(data_all, 3, "full")
        np.testing.assert_allclose(
            total_logp(build_mirt_nc_model(data_all, Q)),
            -112456.494004, rtol=self.RTOL)

    def test_sparse_k3(self, data):
        bench = data.blookup["benchmark"].tolist()
        model = build_mirt_sparse_model(
            data, anchors={bench[0]: 0, bench[1]: 1, bench[2]: 2}, K=3)
        np.testing.assert_allclose(
            total_logp(model), -98951.434841, rtol=self.RTOL)

    def test_interaction_k3(self, data):
        bench = data.blookup["benchmark"].tolist()
        model = build_mirt_interaction_model(
            data, plt_founders=[bench[5], bench[10], bench[20]], K=3)
        np.testing.assert_allclose(
            total_logp(model), -81587.183216, rtol=self.RTOL)

    def test_interaction_k3_pooled_gamma(self, data):
        bench = data.blookup["benchmark"].tolist()
        model = build_mirt_interaction_model(
            data, plt_founders=[bench[5], bench[10], bench[20]], K=3,
            gamma_pooling="pooled")
        np.testing.assert_allclose(
            total_logp(model), -81397.751673, rtol=self.RTOL)

    def test_interaction_k3_gamma_none(self, data):
        # gamma == 0 skips the interaction term entirely, so this value must NOT
        # move when the interaction is reworked: it is the lock proving the
        # compensatory baseline (the matched LOO control) is untouched.
        bench = data.blookup["benchmark"].tolist()
        model = build_mirt_interaction_model(
            data, plt_founders=[bench[5], bench[10], bench[20]], K=3,
            gamma_pooling="none")
        np.testing.assert_allclose(
            total_logp(model), -76268.958770, rtol=self.RTOL)

    def test_interaction_k3_normal_floors_pooled(self, data):
        # non-negative loadings (no founders) + fixed-c 3PL + pooled gamma —
        # the production semi-compensatory configuration.
        from multiaxis_eci.data import clip_scores_to_floors, load_benchmark_floors
        floors = load_benchmark_floors(data)
        clipped = clip_scores_to_floors(data, floors)
        model = build_mirt_interaction_model(
            clipped, plt_founders=None, K=3, gamma_pooling="pooled",
            loading_prior="normal", floor_c=floors)
        np.testing.assert_allclose(
            total_logp(model), -79346.344881, rtol=self.RTOL)


class TestSparseBuilder:
    def test_build_creates_expected_vars(self, data):
        bench = data.blookup["benchmark"].tolist()
        model = build_mirt_sparse_model(
            data, anchors={bench[0]: 0, bench[1]: 1, bench[2]: 2}, K=3)
        names = ({v.name for v in model.free_RVs}
                 | {d.name for d in model.deterministics})
        for v in ["theta", "c", "g", "sigma_b", "phi_b", "gate_tau"]:
            assert v in names, f"missing {v}"

    def test_anchor_cells_fixed(self, data):
        bench = data.blookup["benchmark"].tolist()
        anchors = {bench[0]: 0, bench[1]: 1, bench[2]: 2}
        model = build_mirt_sparse_model(data, anchors=anchors, K=3)
        g_draws = pm.draw(model["g"], draws=4, random_seed=0)
        for bname, ax in anchors.items():
            bi = bench.index(bname)
            assert np.all(g_draws[:, bi, ax] == 1.0), f"{bname} anchor gate not pinned"
            off = [k for k in range(3) if k != ax]
            assert np.all(g_draws[:, bi, off] == 0.0), f"{bname} off-axis gate leaked"

    def test_tiny_sample_runs(self, data):
        bench = data.blookup["benchmark"].tolist()
        with build_mirt_sparse_model(
                data, anchors={bench[0]: 0, bench[1]: 1, bench[2]: 2}, K=3):
            trace = pm.sample(draws=5, tune=5, chains=1, cores=1,
                              progressbar=False, random_seed=0,
                              return_inferencedata=True,
                              compute_convergence_checks=False)
        assert np.isfinite(trace.posterior["g"].values).all()


class TestFloorsBuilder:
    def test_bad_shape_and_range_raise(self, data):
        with pytest.raises(ValueError):
            build_mirt_model(data, K=3, loading_prior="signed",
                             floor_c=np.zeros(3))
        with pytest.raises(ValueError):
            build_mirt_model(data, K=3, loading_prior="signed",
                             floor_c=np.full(data.n_benchmarks, 1.0))

    def test_no_new_free_rvs(self, data):
        floors = np.full(data.n_benchmarks, 0.25)
        m0 = build_mirt_model(data, K=3, loading_prior="signed")
        m1 = build_mirt_model(data, K=3, loading_prior="signed", floor_c=floors)
        assert ({v.name for v in m0.free_RVs}
                == {v.name for v in m1.free_RVs}), "floor_c added a free RV"

    def test_default_path_unchanged(self, data):
        # floor_c=None must be byte-identical to the current 2PL model logp
        m0 = build_mirt_model(data, K=3, loading_prior="signed")
        m1 = build_mirt_model(data, K=3, loading_prior="signed", floor_c=None)
        np.testing.assert_allclose(total_logp(m0), total_logp(m1), rtol=0)


class TestProductToOneBuilder:
    """The pt1 loading prior: log A sum-to-zero over benchmarks, per axis.

    What must hold structurally (the golden logp cannot see any of it):
    every axis's loadings have product exactly 1, loadings are strictly
    positive, and there is NO free loading scale -- one would reinstate the
    very multiplicative ridge the constraint exists to remove.
    """

    def test_product_is_exactly_one_per_axis(self, data):
        for K in (1, 4):
            model = build_mirt_model(data, K=K, loading_prior="pt1")
            A = pm.draw(model["A"], draws=5, random_seed=0)
            gm = np.exp(np.log(A).mean(axis=1))
            np.testing.assert_allclose(gm, 1.0, rtol=1e-12,
                                       err_msg=f"K={K} product-to-one broken")

    def test_loadings_strictly_positive(self, data):
        model = build_mirt_model(data, K=4, loading_prior="pt1")
        A = pm.draw(model["A"], draws=20, random_seed=0)
        assert (A > 0).all(), "pt1 loadings are exp(...) and cannot reach 0"

    def test_no_free_loading_scale(self, data):
        # sigma_A (the log SPREAD) is sampled, as Barry's tau_alpha is. What
        # must NOT exist is a free loading SCALE: it would restore one
        # multiplicative ridge per axis, which is what the constraint removes.
        model = build_mirt_model(data, K=4, loading_prior="pt1")
        names = {v.name for v in model.free_RVs}
        assert {"logA_z", "sigma_A"} <= names
        assert not any(n.startswith("tau_A") for n in names), \
            "pt1 must have NO free loading scale"
        assert "A_z" not in names, "pt1 must not also build the HalfNormal cells"

    def test_removes_one_dimension_per_axis(self, data):
        # the constraint is exactly K hard constraints, one per axis, so pt1
        # sits K free dimensions below the softly-identified "normal" block
        for K in (1, 4):
            n = build_mirt_model(data, K=K, loading_prior="normal")
            p = build_mirt_model(data, K=K, loading_prior="pt1")
            size = lambda m: sum(int(np.asarray(v).size)
                                 for v in m.initial_point().values())
            assert size(n) - size(p) == K, \
                f"K={K}: expected {K} fewer free dims, got {size(n) - size(p)}"

    def test_rejects_anchors(self, data):
        bench = data.blookup["benchmark"].tolist()
        with pytest.raises(ValueError, match="never exactly 0"):
            build_mirt_model(data, K=3, loading_prior="pt1",
                             anchors={bench[0]: 0})


class TestBifactorBuilder:
    """The bifactor loading prior: axis1 dense general, axes 2..K sparse.

    What must hold structurally (the golden logp cannot see any of it):
    the general column is exempt from the sparsity block, every loading is
    non-negative, and the specifics share ONE horseshoe scale.
    """

    def test_general_column_is_dense_and_exempt(self, data):
        model = build_mirt_model(data, K=4, loading_prior="bifactor")
        names = {v.name for v in model.free_RVs}
        assert {"g_z", "A_s_z", "lam_hs", "c2_hs", "tau_g",
                "tau_hs_bifactor"} <= names
        A, g, tau, lam, a_s = pm.draw(
            [model["A"], model["g_z"], model["tau_A"],
             model["lam_hs"], model["A_s_z"]], draws=3, random_seed=0)
        # the sparsity block covers the SPECIFICS only — K-1 columns, not K
        assert lam.shape == (3, data.n_benchmarks, 3)
        assert a_s.shape == (3, data.n_benchmarks, 3)
        assert A.shape == (3, data.n_benchmarks, 4)
        # axis1 is exactly the dense block, untouched by any local scale
        np.testing.assert_allclose(A[:, :, 0], g * tau[:, 0:1], rtol=1e-10)

    def test_loadings_non_negative(self, data):
        model = build_mirt_model(data, K=4, loading_prior="bifactor")
        A = pm.draw(model["A"], draws=20, random_seed=0)
        assert (A >= 0).all(), "bifactor loadings must be non-negative"

    def test_specifics_share_one_scale(self, data):
        # per-axis scales tie and freeze the sampler (householder /
        # reghorseshoe lesson), so all specifics MUST read one shared tau
        model = build_mirt_model(data, K=4, loading_prior="bifactor")
        tau = pm.draw(model["tau_A"], draws=5, random_seed=0)
        assert np.allclose(np.ptp(tau[:, 1:], axis=1), 0.0), \
            "specifics must share ONE horseshoe scale"
        assert not np.allclose(tau[:, 0], tau[:, 1]), \
            "general column must carry its OWN scale, not the horseshoe one"

    def test_rejects_anchors_and_k1(self, data):
        bench = data.blookup["benchmark"].tolist()
        with pytest.raises(ValueError, match="mutually exclusive"):
            build_mirt_model(data, K=3, loading_prior="bifactor",
                             anchors={bench[0]: 0})
        with pytest.raises(ValueError, match="K >= 2"):
            build_mirt_model(data, K=1, loading_prior="bifactor")


class TestThetaTCells:
    """The cell-wise leptokurtic theta block.

    The golden logp is blind to the shape choices that matter here: at the
    initial point every cell is 0, where re-centering is a no-op and a per-ROW
    heavy tail would look much the same. A per-row shape is elliptical and
    identifies no rotation, so the per-cell layout, the kurtosis and the
    restored location pin each need their own guard.
    """

    def test_block_is_per_cell_not_per_row(self, data):
        model = build_mirt_model(data, K=3, loading_prior="signed",
                                 theta_t_cells=True)
        cells = pm.draw(model["theta_t_z"], draws=4, random_seed=0)
        assert cells.shape == (4, data.n_models, 3), \
            "the t block must be one independent draw per (model, axis) cell"
        # within a row the three axis draws must differ — a shared per-row value
        # is exactly the elliptical per-row shape this flag exists to avoid
        assert np.all(np.ptp(cells, axis=2) > 0)

    def test_recentering_keeps_the_zero_sum_location_pin(self, data):
        # theta reaches the likelihood only through A theta - D, so a constant
        # shift of a theta column is absorbed exactly by D and the sum-to-zero
        # constraint is what removes that flat direction. iid t cells do NOT
        # sum to zero on their own, so the block subtracts each column's mean.
        # The golden logp cannot see this: at the initial point every cell is
        # 0, so centering is a no-op there.
        model = build_mirt_model(data, K=3, loading_prior="signed",
                                 theta_t_cells=True)
        th = pm.draw(model["theta"], draws=8, random_seed=0)
        assert np.abs(th.sum(axis=1)).max() < 1e-8

    def test_marginals_are_leptokurtic(self, data):
        # nu = 4 has infinite kurtosis, so the sample excess kurtosis of the
        # mixed block must sit well above the Gaussian block's ~0.
        from scipy.stats import kurtosis
        m0 = build_mirt_model(data, K=3, loading_prior="signed")
        m1 = build_mirt_model(data, K=3, loading_prior="signed",
                              theta_t_cells=True)
        k0 = kurtosis(pm.draw(m0["theta"], draws=40, random_seed=1).ravel())
        k1 = kurtosis(pm.draw(m1["theta"], draws=40, random_seed=1).ravel())
        assert abs(k0) < 0.5, f"Gaussian theta block is not mesokurtic: {k0}"
        assert k1 > 3.0, f"t(4) theta block is not leptokurtic: {k1}"

    def test_covers_structured_bases_only(self, data):
        # The block spans the unstructured rows plus one base per human root and
        # per chain founder — never the ordered increments, which carry their own
        # non-Gaussian shape.
        from multiaxis_eci.models.mirt import _human_structure
        lin = build_lineage_structure(data.mlookup)
        names = data.mlookup["model"].tolist()
        rows_h, _, _, n_roots, _ = _human_structure(HUMAN_ORDER, names)
        model = build_mirt_model(
            data, K=3, loading_prior="normal", human_order=HUMAN_ORDER,
            lineage=lin, theta_t_cells=True)
        assert model["theta_t_z"].type.shape[0] == (
            data.n_models - len(rows_h) - len(lin.row_idx)
            + n_roots + lin.n_chains)


class TestThetaPos:
    """--theta-pos: the likelihood reads softplus(theta), raw theta stays the
    reported ability. The golden locks the eta arithmetic; what it cannot see
    is WHICH variable each side reads, so both get their own guard."""

    def test_positive_copy_is_softplus_of_raw(self, data):
        model = build_mirt_model(data, K=3, loading_prior="normal",
                                 theta_pos=True)
        th, tp = pm.draw([model["theta"], model["theta_pos"]], draws=4,
                         random_seed=0)
        assert np.all(tp > 0.0)
        np.testing.assert_allclose(tp, np.logaddexp(0.0, th), rtol=1e-10)

    def test_raw_theta_keeps_the_location_pin(self, data):
        # the reported ability must stay the ZeroSumNormal draw — the flag is a
        # monotone re-expression on the likelihood side only, no new free RVs
        m0 = build_mirt_model(data, K=3, loading_prior="normal")
        m1 = build_mirt_model(data, K=3, loading_prior="normal",
                              theta_pos=True)
        assert {v.name for v in m0.free_RVs} == {v.name for v in m1.free_RVs}
        th = pm.draw(m1["theta"], draws=8, random_seed=0)
        assert np.abs(th.sum(axis=1)).max() < 1e-8


class TestLogLogLink:
    """link="loglog": eta = alpha_b * logsumexp_k(theta_k + log A_bk), the
    log-logistic IRF. The golden logp locks the LSE/alpha arithmetic at the
    initial point; what it cannot see is the exact row-mean/mix split that
    makes D a Deterministic rather than a sampled RV, so that identity gets
    its own guard."""

    def test_theta_pos_is_exp_of_raw(self, data):
        model = build_mirt_model(data, K=3, loading_prior="normal", link="loglog")
        th, tp = pm.draw([model["theta"], model["theta_pos"]], draws=4,
                         random_seed=0)
        assert np.all(tp > 0.0)
        np.testing.assert_allclose(tp, np.exp(th), rtol=1e-10)

    def test_D_is_negative_alpha_times_row_mean_log_A(self, data):
        # locks the exact row-mean/difficulty identity: D is derived from A's
        # own row scale, never sampled.
        model = build_mirt_model(data, K=3, loading_prior="normal", link="loglog")
        A, D, alpha = pm.draw([model["A"], model["D"], model["alpha"]],
                              draws=5, random_seed=0)
        assert np.all(A > 0.0)
        mean_log_A = np.log(A).mean(axis=-1)
        np.testing.assert_allclose(D, -alpha * mean_log_A, rtol=1e-8)

    def test_k1_row_centered_mix_is_exactly_zero(self, data):
        # size-1 ZeroSumNormal is deterministically 0, so at K=1 the row mean
        # IS log A[:, 0] and D reduces to -alpha * log(A[:, 0]) exactly.
        model = build_mirt_model(data, K=1, loading_prior="normal", link="loglog")
        A, D, alpha = pm.draw([model["A"], model["D"], model["alpha"]],
                              draws=5, random_seed=0)
        np.testing.assert_allclose(D, -alpha * np.log(A[..., 0]), rtol=1e-8)

    def test_rejects_incompatible_options(self, data):
        bench = data.blookup["benchmark"].tolist()
        for loading_prior in ("signed", "pt1", "bifactor"):
            with pytest.raises(ValueError):
                build_mirt_model(data, K=3, loading_prior=loading_prior,
                                 link="loglog")
        with pytest.raises(ValueError):
            build_mirt_model(data, K=3, loading_prior="normal", link="loglog",
                             theta_pos=True)
        with pytest.raises(ValueError):
            build_mirt_model(data, K=3, loading_prior="normal", link="loglog",
                             pin_benchmark=bench[5])
        with pytest.raises(ValueError):
            build_mirt_model(data, K=3, loading_prior="normal", link="loglog",
                             anchors={bench[5]: [0]})

    def test_free_rv_set_swaps_the_loading_and_difficulty_blocks(self, data):
        # Guards that the theta stack (ZeroSumNormal, human/lineage structure)
        # is untouched: only the loading/difficulty RVs differ from the
        # linear-link twin.
        m_loglog = build_mirt_model(data, K=3, loading_prior="normal",
                                    link="loglog")
        m_linear = build_mirt_model(data, K=3, loading_prior="normal")
        names_loglog = {v.name for v in m_loglog.free_RVs}
        names_linear = {v.name for v in m_linear.free_RVs}
        assert names_loglog - names_linear == {
            "tau_A_loglog", "logA_row_z", "logA_mix_z", "alpha_z", "tau_alpha"}
        assert names_linear - names_loglog == {
            "tau_A_normal", "A_z", "D_z"}


class TestStreamedDraws:
    """--stream-draws: nutpie writes each draw into a zarr store as it lands, so
    a killed run keeps its finished prefix. Toy model — the mechanism is the
    sampler's, not the MIRT's."""

    def test_partial_store_reads_back_and_saves(self, tmp_path):
        from nutpie import zarr_store

        from multiaxis_eci.persistence import load_live_draws, save_trace
        store = tmp_path / "live_draws.zarr"
        with pm.Model(coords={"g": ["a", "b", "c"]}):
            x = pm.Normal("x", dims="g")
            pm.Normal("y", x[0], 1.0,
                      observed=np.random.default_rng(0).normal(size=40))
            idata = pm.sample(
                draws=100, tune=100, chains=2, cores=2, nuts_sampler="nutpie",
                progressbar=False, random_seed=1,
                nuts_sampler_kwargs={
                    "save_warmup": False,
                    "zarr_store": zarr_store.LocalStore(str(store), mkdir=True)})
        # The store IGNORES save_warmup and hands the warmup back too — the RAM
        # that OOM-killed two finished runs, so fit.py drops it and so does the
        # reader.
        assert "warmup_posterior" in idata.groups()
        np.testing.assert_allclose(load_live_draws(store).posterior.x.values,
                                   idata.posterior.x.values)
        assert "warmup_posterior" not in load_live_draws(store).groups()

        # An unfinished run: draws land in order, so the tail is still NaN.
        import zarr
        z = zarr.open(str(store), mode="a")
        z["sample_stats"]["energy"][1, 60:] = np.nan
        z["posterior"]["x"][1, 60:, :] = np.nan
        partial = load_live_draws(store)
        assert partial.posterior.sizes["draw"] == 60
        assert np.isfinite(partial.posterior.x.values).all()

        # The store path stamps a NESTED sampler_settings attr, which netCDF
        # cannot hold: saving must not raise.
        assert isinstance(idata.attrs["sampler_settings"], dict)
        save_trace(partial, tmp_path / "trace.nc")
        assert (tmp_path / "trace.nc").exists()


class TestHumanMergedOrder:
    """--human-merge: the multi-parent tiers dominate EVERY parent, and the
    merge buys that with paths, not with parameters."""

    MERGES = [("Domain Expert", "Skilled Generalist"),
              ("Domain Expert", "High School Qualifier"),
              ("Top Performer", "Domain Expert"),
              ("Top Performer", "High School Top Performer")]

    def _theta(self, data, order):
        model = build_mirt_model(data, K=3, loading_prior="normal",
                                 human_order=order)
        idx = {m: i for i, m in enumerate(data.mlookup["model"])}
        return pm.draw(model["theta"], draws=100, random_seed=0), idx

    def test_merged_tier_dominates_every_parent(self, data):
        theta, idx = self._theta(data, HUMAN_ORDER_MERGED)
        for child, parent in self.MERGES:
            gap = theta[:, idx[child], :] - theta[:, idx[parent], :]
            assert gap.min() > 0, f"{child} < {parent} on some draw/axis"

    def test_tree_order_leaves_the_branches_incomparable(self, data):
        # The guarantee must come from the merged map, not from the geometry:
        # under the plain partial order the cross-branch pairs are free to
        # invert, so the test above is testing the prior and not a tautology.
        theta, idx = self._theta(data, HUMAN_ORDER)
        crossed = [(c, p) for c, p in self.MERGES
                   if (theta[:, idx[c], :] - theta[:, idx[p], :]).min() < 0]
        assert crossed == [("Domain Expert", "High School Qualifier"),
                           ("Top Performer", "High School Top Performer")]

    def test_merge_adds_paths_not_parameters(self, data):
        names = data.mlookup["model"].tolist()
        tree, merged = (_human_structure(o, names)
                        for o in (HUMAN_ORDER, HUMAN_ORDER_MERGED))
        assert tree[2].shape[1] == merged[2].shape[1]      # one increment/tier
        assert merged[1].shape[0] > tree[1].shape[0]       # more paths
        assert all(len(g) == 1 for g in tree[4])           # tree: max drops out


class TestInteractionBuilder:
    def test_build_creates_expected_vars(self, data):
        bench = data.blookup["benchmark"].tolist()
        model = build_mirt_interaction_model(
            data, plt_founders=[bench[5], bench[10], bench[20]], K=3)
        names = ({v.name for v in model.free_RVs}
                 | {d.name for d in model.deterministics})
        for v in ["A", "theta", "D", "gamma", "sigma_b", "phi_b"]:
            assert v in names, f"missing {v}"

    def test_founder_interactions_masked(self, data):
        bench = data.blookup["benchmark"].tolist()
        founders = [bench[5], bench[10], bench[20]]
        model = build_mirt_interaction_model(data, plt_founders=founders, K=3)
        gamma = pm.draw(model["gamma"], draws=4, random_seed=0)   # (4, B, 3 pairs)
        # Axis-1 founder loads only axis 0 → no pair fits inside its triangle.
        fi = bench.index(founders[0])
        assert np.all(gamma[:, fi, :] == 0.0)
        # Axis-2 founder loads axes 0-1 → only the (0,1) pair survives.
        fi = bench.index(founders[1])
        assert np.all(gamma[:, fi, 1:] == 0.0)

    def test_tiny_sample_runs(self, data):
        bench = data.blookup["benchmark"].tolist()
        with build_mirt_interaction_model(
                data, plt_founders=[bench[5], bench[10], bench[20]], K=3):
            trace = pm.sample(draws=5, tune=5, chains=1, cores=1,
                              progressbar=False, random_seed=0,
                              return_inferencedata=True,
                              compute_convergence_checks=False)
        assert np.isfinite(trace.posterior["gamma"].values).all()

    def test_pooled_gamma_shapes(self, data):
        bench = data.blookup["benchmark"].tolist()
        model = build_mirt_interaction_model(
            data, plt_founders=[bench[5], bench[10], bench[20]], K=3,
            gamma_pooling="pooled")
        gp = pm.draw(model["gamma_pooled"], draws=2, random_seed=0)   # (2, 3 pairs)
        gamma = pm.draw(model["gamma"], draws=2, random_seed=0)       # (2, B, 3 pairs)
        assert gp.shape[-1] == 3
        assert gamma.shape[1:] == (data.n_benchmarks, 3)

    def test_gamma_none_is_zero(self, data):
        bench = data.blookup["benchmark"].tolist()
        model = build_mirt_interaction_model(
            data, plt_founders=[bench[5], bench[10], bench[20]], K=3,
            gamma_pooling="none")
        names = {v.name for v in model.free_RVs}
        assert "gamma_z" not in names        # no free interaction params
        assert np.all(pm.draw(model["gamma"], draws=3, random_seed=0) == 0.0)

    def test_invalid_gamma_pooling_raises(self, data):
        bench = data.blookup["benchmark"].tolist()
        with pytest.raises(ValueError, match="gamma_pooling"):
            build_mirt_interaction_model(
                data, plt_founders=[bench[5], bench[10], bench[20]], K=3,
                gamma_pooling="bogus")

    def test_normal_prior_rejects_founders(self, data):
        bench = data.blookup["benchmark"].tolist()
        with pytest.raises(ValueError, match="non-negativity"):
            build_mirt_interaction_model(
                data, plt_founders=[bench[5], bench[10], bench[20]], K=3,
                loading_prior="normal")

    def test_signed_prior_requires_founders(self, data):
        with pytest.raises(ValueError, match="founders"):
            build_mirt_interaction_model(data, plt_founders=None, K=3)

    def test_normal_prior_loadings_nonnegative(self, data):
        model = build_mirt_interaction_model(
            data, plt_founders=None, K=3, gamma_pooling="pooled",
            loading_prior="normal")
        A = pm.draw(model["A"], draws=4, random_seed=0)
        assert np.all(A >= 0.0)
        # no founder mask: every benchmark keeps every interaction pair
        gamma = pm.draw(model["gamma"], draws=2, random_seed=0)
        assert gamma.shape[1:] == (data.n_benchmarks, 3)

    def test_conjunction_is_one_sided(self, data):
        # The two halves of the semi-compensatory setup: the product runs on
        # strictly positive abilities and gamma cannot go negative.
        model = build_mirt_interaction_model(
            data, plt_founders=None, K=3, gamma_pooling="pooled",
            loading_prior="normal")
        tp = pm.draw(model["theta_pos"], draws=4, random_seed=0)
        assert np.all(tp > 0.0)
        gamma = pm.draw(model["gamma"], draws=8, random_seed=0)
        assert np.all(gamma >= 0.0)
        # softplus is monotone: raw-theta order must survive on the product
        # scale (the human/lineage order priors act in raw space)
        th = pm.draw(model["theta"], draws=1, random_seed=1)
        np.testing.assert_allclose(
            np.argsort(th, axis=0), np.argsort(np.logaddexp(0.0, th), axis=0))


class TestCapabilityDraws:
    def _idata(self, posterior, dims):
        import arviz as az
        return az.from_dict(posterior=posterior, dims=dims)

    def test_reads_C_from_1d_trace(self):
        from multiaxis_eci.analysis import capability_draws
        rng = np.random.default_rng(0)
        C = rng.normal(size=(2, 5, 7))
        idata = self._idata({"C": C}, {"C": ["model"]})
        out = capability_draws(idata)
        assert out.shape == (10, 7)
        np.testing.assert_allclose(out, C.reshape(-1, 7))

    def test_reads_theta_from_k1_trace(self):
        from multiaxis_eci.analysis import capability_draws
        rng = np.random.default_rng(1)
        theta = rng.normal(size=(2, 5, 7, 1))
        idata = self._idata({"theta": theta}, {"theta": ["model", "latent"]})
        out = capability_draws(idata)
        assert out.shape == (10, 7)
        np.testing.assert_allclose(out, theta[..., 0].reshape(-1, 7))

    def test_raises_on_multi_axis_trace(self):
        from multiaxis_eci.analysis import capability_draws
        theta = np.zeros((1, 3, 4, 2))
        idata = self._idata({"theta": theta}, {"theta": ["model", "latent"]})
        with pytest.raises(ValueError, match="K=2"):
            capability_draws(idata)


class TestPublicAPISurface:
    """Every cross-module name the entry scripts import, per module.

    When a reorganization commit deliberately deletes a name, remove it here
    in the same commit — never earlier.
    """
    SURFACE = {
        "multiaxis_eci.analysis": [
            "all_models_stats_df", "eci_transform", "flat_C", "forest_stats_df",
            "human_stats_df", "sota_stats_df", "timeline_stats_df", "post_stats",
            "apply_rotation", "factor_corr_df", "factor_scores_df",
            "loadings_table", "mirt_factors_from_trace", "mirt_identified_rhat",
            "promax_rotate", "tau_spectrum_df",
            "mirt_informed_mask", "mirt_model_timeline_df", "prepare_fit",
            "trace_loading_prior", "trace_axis_names", "trace_anchors",
            "_aligned_reproducibility", "crosschain_axis_reproducibility",
            "mirt_identified_rhat_nc", "mirt_identified_rhat_sparse",
            "mirt_identified_rhat_interaction", "alignment_report",
            "mirt_crossover_df", "mirt_frontier_forecast",
            "mirt_human_axis_stats", "_release_dates",
        ],
        "multiaxis_eci.viz": [
            "all_models_forest_fig", "capability_timeline_fig",
            "density_overlay_fig", "forest_fig", "hyperparams_fig",
            "pit_ecdf_fig", "pit_hist_fig", "pred_vs_obs_fig",
            "residuals_per_benchmark_fig", "save_fig", "sota_forest_fig",
            "alignment_methods_fig", "axes_frontier_fig",
            "axes_scatter_matrix_fig", "build_fit_figures",
            "factor_vs_1d_fig", "per_bench_r2_delta_fig", "pred_scatter_fig",
            "assemble_dashboard", "build_comparison",
            "ppca_spectrum_fig", "write_dashboard", "capability_forecast_fig",
            "forest_grid_fig", "loadings_grid_fig", "subplot_grid",
            "crossover_dotwhisker_fig", "exceedance_prob_fig",
        ],
        "multiaxis_eci.ppc": [
            "compute_gof", "posterior_predictive_mirt",
            "posterior_predictive_mirt_nc", "posterior_predictive_mirt_sparse",
            "posterior_predictive_mirt_interaction", "pit_values",
        ],
        "multiaxis_eci.persistence": [
            "load_trace", "save_df", "save_json", "save_pit", "save_summary",
            "save_trace",
        ],
        "multiaxis_eci.data": [
            "PROCESSED_FILE", "drop_model_observations", "drop_zero_scores",
            "load_eci_data", "find_model_idx",
            "drop_model_benchmark_cells", "load_excluded_benchmarks",
            "load_retired_benchmarks",
            "load_benchmark_floors", "clip_scores_to_floors",
            "release_time_covariate",
        ],
        "multiaxis_eci.lineage": ["build_lineage_structure", "LINEAGE_MAP"],
        "multiaxis_eci.models": ["build_mirt_model", "build_mirt_nc_model",
                   "build_mirt_sparse_model", "build_mirt_interaction_model",
                   "INTERACTION_SCALE", "QMATRIX_VARIANTS", "axes_as_list"],
    }

    def test_all_names_importable(self):
        import importlib
        missing = []
        for module_name, names in self.SURFACE.items():
            mod = importlib.import_module(module_name)
            for name in names:
                if not hasattr(mod, name):
                    missing.append(f"{module_name}.{name}")
        assert not missing, f"public API names lost: {missing}"


class TestCLISurface:
    def test_fit_help_exits_clean(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "2_fit.py"), "--help"],
            capture_output=True, text=True, timeout=120)
        assert result.returncode == 0
        assert "--loading-prior" in result.stdout
        assert "--preset" in result.stdout

    def test_fit_unknown_arg_fails(self):
        result = subprocess.run(
            [sys.executable, str(ROOT / "2_fit.py"), "--nonsense"],
            capture_output=True, text=True, timeout=120)
        assert result.returncode != 0
