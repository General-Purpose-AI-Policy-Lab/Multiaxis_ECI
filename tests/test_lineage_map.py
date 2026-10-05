"""Guards on the lineage alias map (1_curated/lineage_map.csv).

The lineage prior gives every (chain, node, variant) group ONE ability:
theta = psi[node] + offset[(node, variant)]. A new release filed under an older node
silently shares that node's ability (muse-spark-1.3 on the April muse-spark node, glm-5.3
on glm-5, grok-4.6 on grok-4: found 2026-10-01), and nothing in the fit flags it. These
tests fail on the data instead:

- the tracked map is what the builder produces from 0_input (no stale map after a sync,
  no hand edit a rebuild would erase), and covers every scored model;
- a (chain, node, variant) group holds one model unless the pooling is reviewed in
  1_curated/lineage_reviewed_exceptions.csv;
- a chained model's release date sits within a month of its node's date;
- a node carries one date and one parent;
- the builder reads versions off the names, so the next release gets its own node.
- no release scores far below the one it continues (ECI-H of the current canonical fit,
  `analysis.lineage_check`) unless that regression was investigated and listed in the
  exceptions file (`check` = regression): a big drop is how a misfiled model shows (Qwen3
  1.7B on the 7B chain, 2026-10-01).

When a sync adds models: rerun `python 1_curated/2_build_lineage_map.py`, then read any
failure here as a routing question, fixed in the builder rules, in
lineage_node_overrides.csv, or (for true aliases of one release) in the exceptions file.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
MAP = ROOT / "1_curated" / "lineage_map.csv"
EXCEPTIONS = ROOT / "1_curated" / "lineage_reviewed_exceptions.csv"
SCORES = ROOT / "0_input" / "all_scores_flat.csv"
# Release dates may trail the node's (the min over its aliases) by a few weeks: a GA after
# its preview, an API name after the open weights. Beyond this, the row is another release.
MAX_DATE_GAP_DAYS = 31


def _builder():
    spec = importlib.util.spec_from_file_location(
        "build_lineage_map", ROOT / "1_curated" / "2_build_lineage_map.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def builder():
    return _builder()


@pytest.fixture(scope="module")
def lineage_map():
    return pd.read_csv(MAP, keep_default_na=False, dtype=str)


@pytest.fixture(scope="module")
def chained(lineage_map):
    return lineage_map[lineage_map["in_chain"] == "yes"]


@pytest.fixture(scope="module")
def exceptions():
    return pd.read_csv(EXCEPTIONS, keep_default_na=False, dtype=str)


def test_map_is_the_builders_output(builder, lineage_map, tmp_path_factory):
    out = tmp_path_factory.mktemp("lineage") / "lineage_map.csv"
    builder.main(out_path=out, report=False)
    fresh = pd.read_csv(out, keep_default_na=False, dtype=str)
    key = ["raw_string"]
    a = lineage_map.sort_values(key).reset_index(drop=True)
    b = fresh.sort_values(key).reset_index(drop=True)
    assert list(a.columns) == list(b.columns)
    diff = a.merge(b, how="outer", indicator=True)
    diff = diff[diff["_merge"] != "both"]
    assert diff.empty, ("lineage_map.csv is not the builder's output on 0_input; rerun "
                        "1_curated/2_build_lineage_map.py (left_only = tracked, right_only = "
                        f"rebuilt):\n{diff.sort_values(key).to_string(index=False)}")


def test_every_scored_model_is_mapped(lineage_map):
    scored = set(pd.read_csv(SCORES, usecols=["model_version"])["model_version"])
    missing = sorted(scored - set(lineage_map["raw_string"]))
    assert not missing, f"models scored in 0_input but absent from the lineage map: {missing}"


def test_no_unreviewed_pooling(chained, exceptions):
    """Two models in one (chain, node, variant) group are ONE test-taker in the fit."""
    groups = (chained.groupby(["chain", "node", "variant"])["raw_string"]
              .agg(lambda s: "|".join(sorted(s))))
    pooled = groups[groups.str.contains("|", regex=False)]
    ok = exceptions[exceptions["check"] == "pool"].set_index(["chain", "node", "variant"])["raw_strings"]
    unreviewed = {k: v for k, v in pooled.items() if ok.get(k) != v}
    assert not unreviewed, (
        "models sharing one (chain, node, variant) group, i.e. one theta, with no review in "
        f"{EXCEPTIONS.name}:\n" + "\n".join(f"  {k}: {v}" for k, v in sorted(unreviewed.items())))
    stale = sorted(set(ok.index) - set(pooled.index))
    assert not stale, f"{EXCEPTIONS.name}: pool entries that no longer exist, drop them: {stale}"


def test_release_dates_match_their_node(chained, exceptions):
    raw = pd.read_csv(SCORES, usecols=["model_version", "release_date"])
    released = pd.to_datetime(raw.groupby("model_version")["release_date"].first(), errors="coerce")
    rows = chained.assign(rel=chained["raw_string"].map(released),
                          nd=pd.to_datetime(chained["node_date"]))
    rows = rows[rows["rel"].notna()]
    gap = (rows["rel"] - rows["nd"]).dt.days
    ok = set(exceptions.loc[exceptions["check"] == "date", "raw_strings"])
    off = rows[(gap.abs() > MAX_DATE_GAP_DAYS) & ~rows["raw_string"].isin(ok)]
    assert off.empty, (
        f"chained models released more than {MAX_DATE_GAP_DAYS} days from their node's date "
        "(another release filed under an existing node?):\n"
        + off.assign(gap_days=gap)[["raw_string", "chain", "node", "node_date", "rel", "gap_days"]]
        .to_string(index=False))


def test_node_carries_one_date_and_one_parent(chained):
    by_node = chained.groupby(["chain", "node"])
    bad = by_node.filter(lambda g: g["node_date"].nunique() > 1 or g["parent"].nunique() > 1)
    assert bad.empty, ("rows of one node disagree on its date or parent (lineage.py keeps the "
                       f"first row's):\n{bad[['raw_string', 'chain', 'node', 'parent', 'node_date']]}")
    nodes = set(zip(chained["chain"], chained["node"]))
    orphans = chained[(chained["parent"] != "")
                      & ~pd.Series(list(zip(chained["chain"], chained["parent"])),
                                   index=chained.index).isin(nodes)]
    assert orphans.empty, f"parents that are no node of their chain:\n{orphans}"


def test_a_product_sits_on_one_node(chained):
    """Every run of one product (`data.model_product`: a release's efforts, its pro mode apart)
    shares one (chain, node): an effort filed on another node splits the release's ability."""
    from multiaxis_eci.data import model_product
    nodes = chained.assign(product=chained["raw_string"].map(model_product)) \
        .groupby("product")[["chain", "node"]].apply(lambda g: g.drop_duplicates().shape[0])
    assert (nodes == 1).all(), nodes[nodes > 1]


# A future release must get a node of its own, read off its name. Each case is a name the
# former hand lists routed onto an older node (or a neighbour of one), plus names the new
# rules must keep where they were.
@pytest.mark.parametrize("name, org, chain, node", [
    ("muse-spark-1.4_max", "Meta AI", "muse-spark", "spark 1.4"),
    ("muse-spark", "Meta AI", "muse-spark", "spark 1"),
    ("glm-5.4", "Z.ai (Zhipu AI)", "glm", "5.4"),
    ("glm-6_max", "Z.ai (Zhipu AI)", "glm", "6"),
    ("glm-5.4-flash", "Z.ai (Zhipu AI)", "glm-flash", "flash 5.4"),
    ("glm-5p2", "Z.ai (Zhipu AI)", "glm", "5.2"),
    ("grok-4.8_high", "xAI", "grok", "4.8"),
    ("grok-5", "xAI", "grok", "5"),
    ("grok-4-0709", "xAI", "grok", "4"),
    ("grok-4-20", "xAI", "grok", "4.20"),
    ("grok-4.20-0309-reasoning", "xAI", "grok", "4.20"),
    ("claude-opus-5-6_max", "Anthropic", "opus", "Opus 5.6"),
    ("claude-opus-6", "Anthropic", "opus", "Opus 6"),
    ("claude-sonnet-5-1_high", "Anthropic", "sonnet", "Sonnet 5.1"),
    ("claude-opus-5", "Anthropic", "opus", "Opus 5"),
    ("claude-opus-4-5-20251101_32K", "Anthropic", "opus", "Opus 4.5"),
    ("claude-3-7-sonnet-20250219", "Anthropic", "sonnet", "Sonnet 3.7"),
    ("claude-fable-5-2", "Anthropic", "mythos-fable", "Fable 5.2"),
    ("claude-mythos-6", "Anthropic", "mythos-fable", "Mythos 6"),
    ("claude-mythos-preview-early", "Anthropic", "mythos-fable", "Mythos Preview Early"),
    ("deepseek-v4.2-flash_max", "DeepSeek", "deepseek-flash", "V4.2-flash"),
    ("deepseek-v4.1-pro-1020", "DeepSeek", "deepseek-v", "V4.1-pro-1020"),
    ("deepseek-v4-flash-0731_high", "DeepSeek", "deepseek-flash", "V4-flash-0731"),
    ("qwen3.9-max-1015_xhigh", "Alibaba", "qwen-max", "max 3.9-1015"),
    ("qwen3-max-2025-09-23", "Alibaba", "qwen-max", "max 3"),
    ("mistral-medium-2611", "Mistral AI", "mistral-medium", "medium-2611"),
    ("mistral-small-3.1-2503", "Mistral AI", "mistral-small", "small-2503"),
    ("MiniMax-M3.1_xhigh", "MiniMax", "minimax", "M3.1"),
    ("MiniMax-M1-80k", "MiniMax", "minimax", "M1"),
    ("gpt-5.6-sol_promax", "OpenAI", "pro", "pro 5.6"),       # pro mode: the pro chain
    ("gpt-5.6-sol_high", "OpenAI", "gpt-sol", "sol 5.6"),      # codenames: a line each
    ("qwen3-1.7b", "Alibaba", "qwen-open-1.7b", "3-1.7B"),   # every Qwen open size a line
    ("qwen3.8-27b_low", "Alibaba", "qwen-open-27b", "3.8-27B"),
    ("Qwen3-235B-A22B-Thinking-2507", "Alibaba", "qwen-open-235b", "3-235B 2507"),
    ("LLaMA-7B", "Meta", "llama-7b", "1-7B"),
    ("Llama-3.1-405B", "Meta", "llama-405b", "3.1-405B"),
    ("claude-2.1", "Anthropic", "claude", "2.1"),
    ("claude-instant-1.2", "Anthropic", "claude-instant", "instant 1.2"),
    ("magistral-small-2509", "Mistral AI", "magistral-small", "small 2509"),
    ("gemma-3-27b-it", "Google", "gemma-27b", "3-27B"),
    ("gpt-6.1-sol_max", "OpenAI", "gpt-sol", "sol 6.1"),
    ("gpt-6-luna_none", "OpenAI", "gpt-luna", "luna 6"),
    ("gpt-5.6-terra", "OpenAI", "gpt-terra", "terra 5.6"),
    ("gpt-6-astra_xhigh", "OpenAI", "gpt-astra", "astra 6"),
    ("gpt-7-astra_proxhigh", "OpenAI", "pro", "pro 7"),
    ("gpt-5.4-pro-2026-03-05_xhigh", "OpenAI", "pro", "pro 5.4"),
    ("gpt-5.1-codex-max", "OpenAI", "codex", "5.1-codex-max"),   # a model, not an effort
    ("gpt-5.1-codex", "OpenAI", "codex", "5.1-codex"),
    ("deepseek-chat", "DeepSeek", "deepseek-v", "V3.2"),
])
def test_builder_reads_versions_off_names(builder, name, org, chain, node):
    got = builder.classify(name, org)
    assert (got["chain"], got["node"]) == (chain, node), got


@pytest.mark.parametrize("name, org", [
    ("deepseek-v5", "DeepSeek"),          # a new generation is flagged, not filed under V4
])
def test_unknown_generation_is_unrouted(builder, name, org):
    assert builder.classify(name, org)["node"] == "UNROUTED"


@pytest.mark.parametrize("name, org, variant", [
    ("gpt-6-instant", "OpenAI", "instant"),
    ("gpt-6_cyber", "OpenAI", "cyber"),
    ("gpt-6_chat", "OpenAI", "chat"),
    ("DeepSeek-V3.2-Speciale", "DeepSeek", "speciale"),
    ("Qwen2.5-72B", "Alibaba", "base"),
    ("qwen2.5-72b-instruct", "Alibaba", "bare"),
    ("MiniMax-M3_xhigh", "MiniMax", "effort:xhigh"),
    ("claude-instant-1.2", "Anthropic", "bare"),         # the product line, not a run mode
    ("deepseek-chat", "DeepSeek", "non-thinking"),       # the non-thinking endpoint
    ("deepseek-llm-67b-chat", "DeepSeek", "bare"),       # a chat model, not the endpoint
    ("Qwen3-235B-A22B-Instruct-2507", "Alibaba", "non-thinking"),
    ("Qwen3-235B-A22B-Thinking-2507", "Alibaba", "thinking"),
    ("gpt-5.1-codex-max", "OpenAI", "bare"),
])
def test_run_variants_get_their_own_group(builder, name, org, variant):
    assert builder.classify(name, org)["variant"] == variant


@pytest.mark.parametrize("name, org", [
    ("deepseek-r1-0528-qwen3-8b", "DeepSeek"),   # a distill: its student is another model
    ("gemini-1.5-flash-8b-001", "Google DeepMind"),
    ("gemini-3-deep-think-preview", "Google DeepMind"),   # parallel-thinking mode, not a release
    ("devstral-small-2512", "Mistral AI"),
    ("CodeQwen1.5-7B", "Alibaba"),   # a code model, off the general 7B spine
    ("Llama-3.1-405B-Instruct_bio-llama", "Meta"),   # a fine-tune, not the release
    ("qwen3.8-2.4t-a95b_xhigh", "Alibaba"),          # a size in trillions
    ("gpt-5.4-2026-03-05_cyber", "OpenAI"),          # the cyber-permissive fine-tune
    ("gpt-5.5_cyber", "OpenAI"),
    ("DeepSeek-V3.2-Speciale", "DeepSeek"),          # a reasoning-only RL fine-tune
    ("computer-use-preview-2025-03-11", "OpenAI"),   # the computer-use agent model
    ("kimi-k2.7-code", "Moonshot"),                  # a code model, off the general spine
    ("kimi-k2.7-code_xhigh", "Moonshot"),
])
def test_other_models_stay_out_of_the_chain(builder, name, org):
    assert builder.classify(name, org)["in_chain"] == "no"


def test_no_unreviewed_regression(exceptions):
    """A child release far below its parent on the current canonical fit's ECI-H is either a
    filing mistake (fix the map) or a genuine regression, investigated and listed in the
    exceptions file with its reason (key: chain, child node, variant compared, parent|child run).
    Skipped until the data generation has its canonical fit."""
    from multiaxis_eci import config
    from multiaxis_eci.analysis.lineage_check import MIN_DROP, regressions
    fit = config.RESULTS_DIR / "canonical_humanmerge" / "all_models_eci.csv"
    if not fit.exists():
        pytest.skip(f"no canonical fit yet in {config.RESULTS_DIR.name}/")
    flagged = regressions(pd.read_csv(fit), pd.read_csv(MAP))
    ok = set(map(tuple, exceptions.loc[exceptions["check"] == "regression",
                                       ["chain", "node", "variant", "raw_strings"]].values))
    unreviewed = [r for r in flagged.itertuples()
                  if (r.chain, r.child, r.variant, f"{r.parent_run}|{r.child_run}") not in ok]
    assert not unreviewed, (
        f"releases at least {MIN_DROP:g} ECI-H below their parent ({fit.parent.parent.name}), "
        f"neither fixed in the map nor reviewed in {EXCEPTIONS.name} "
        "(python 4_diagnostics/lineage_regressions.py lists every edge):\n" + "\n".join(
            f"  {r.chain}: {r.parent} -> {r.child} [{r.variant}] {r.parent_run} {r.parent_eci:.1f} "
            f"-> {r.child_run} {r.child_eci:.1f} (n={r.n_parent}/{r.n_child})" for r in unreviewed))
