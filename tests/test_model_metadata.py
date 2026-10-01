"""Guards on what the code reads about each model besides its scores: organization and country,
the product a run stands for, its reasoning regime. Each rule was found broken on a model
added by a sync (2026-10-01), silently: frontier runs without organization or country left the
US group of the country frontier; GPT-5.6 Sol's pro mode stood for Sol on the frontier;
`_none` runs of reasoning families counted as reasoning; `gemini-2.5` caught a Flash-Lite with
thinking off; hybrid releases other than Claude kept their bare run, of unknown mode, in the
regime fits. These tests read the whole current models table, so the next sync trips them.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from multiaxis_eci import config
from multiaxis_eci.analysis.regimes import (
    OFF_LEVELS,
    OFF_VARIANTS,
    family_best,
    hybrid_families,
    is_reasoning,
    regime_ignored,
    regime_table,
    run_mode,
)
from multiaxis_eci.data import (
    MODE_VARIANTS,
    MODELS_FILE,
    ORGANIZATION_COUNTRIES_FILE,
    is_bare,
    model_family,
    model_product,
    models_table,
    read_scores,
)

ROOT = Path(__file__).resolve().parent.parent
REGIMES = ROOT / "1_curated" / "model_regimes.csv"


@pytest.fixture(scope="module")
def models():
    return models_table()


# ── organization and country ───────────────────────────────────────────────────────────
def test_every_model_has_an_organization_and_a_country(models):
    """A blank country is accepted only for an organization the curated file leaves blank on
    purpose (a scaffold over other developers' models)."""
    cur = pd.read_csv(ORGANIZATION_COUNTRIES_FILE, dtype=str).fillna("")
    no_country = set(cur.loc[cur["country"] == "", "organization"])
    missing = models[(models["organization"] == "")
                     | ((models["country"] == "") & ~models["organization"].isin(no_country))]
    assert missing.empty, (
        "models with no organization or country after completion (fill the release's other rows "
        f"upstream, or add the organization to {ORGANIZATION_COUNTRIES_FILE.name}):\n"
        + missing[["model_version", "base_model", "organization", "country"]].to_string(index=False))


def test_a_release_has_one_organization_and_one_country(models):
    per = models.groupby("base_model")[["organization", "country"]].nunique()
    split = per[(per > 1).any(axis=1)]
    assert split.empty, f"releases whose runs disagree on organization or country:\n{split}"


def test_curated_countries_are_only_for_organizations_the_table_leaves_blank():
    raw = pd.read_csv(MODELS_FILE, dtype=str).fillna("")
    cur = pd.read_csv(ORGANIZATION_COUNTRIES_FILE, dtype=str).fillna("")
    known = set(raw.loc[raw["country"] != "", "organization"])
    assert (cur["source"] != "").all(), "every row says why"
    stale = sorted(set(cur["organization"]) & known)
    assert not stale, (f"{ORGANIZATION_COUNTRIES_FILE.name}: the models table now gives a country "
                       f"for {stale}; drop the rows")
    unused = sorted(set(cur["organization"]) - set(raw["organization"]))
    assert not unused, f"{ORGANIZATION_COUNTRIES_FILE.name}: no model of {unused}; drop the rows"


def test_score_rows_carry_their_release_organization():
    df = read_scores()
    blank = df["organization"].isna() | (df["organization"].astype(str).str.strip() == "")
    assert not blank.any(), sorted(df.loc[blank, "model_version"].unique())[:10]


# ── product ────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("name, product", [
    ("gpt-5.6-sol_promax", "gpt-5.6-sol-pro"),          # pro mode: a product of its own
    ("gpt-5.6-sol_max", "gpt-5.6-sol"),
    ("gpt-6-astra_promax", "gpt-6-astra-pro"),
    ("gpt-5.4-2026-03-05_cyber", "gpt-5.4-cyber@2026-03-05"),
    ("gpt-5-2025-08-07_chat", "gpt-5@2025-08-07"),      # a mode of the release (user, c219e66)
    ("claude-opus-4-6_thinking", "claude-opus-4-6"),
    ("Average Human", "Average Human"),
])
def test_model_product(name, product):
    assert model_product(name) == product


def test_every_variant_is_a_mode_or_a_product(models):
    """A new run variant upstream is a product of its own by default; one that is a way of
    running the release must be added to data.MODE_VARIANTS on purpose."""
    for name, variant in zip(models["model_version"], models["variant"]):
        same = model_product(name) == model_family(name)
        assert same == (variant in MODE_VARIANTS), (name, variant)


def test_family_best_keeps_products_apart():
    tl = pd.DataFrame({"name": ["gpt-5.6-sol_max", "gpt-5.6-sol_promax", "gpt-5.6-sol_high"],
                       "mean": [1.0, 2.0, 0.5]})
    assert sorted(family_best(tl)["name"]) == ["gpt-5.6-sol_max", "gpt-5.6-sol_promax"]


# ── reasoning regime ───────────────────────────────────────────────────────────────────
def test_runs_without_thinking_are_never_reasoning(models):
    off = models[models["reasoning_level"].isin(OFF_LEVELS) | models["variant"].isin(OFF_VARIANTS)]
    wrong = [m for m in off["model_version"] if is_reasoning(m)]
    assert not wrong, f"runs with thinking off classified reasoning: {wrong}"


def test_runs_that_state_thinking_are_reasoning(models):
    on = models[((~models["reasoning_level"].isin(OFF_LEVELS) & (models["reasoning_level"] != ""))
                 | (models["reasoning_tokens"] != "") | (models["variant"] == "thinking"))
                & ~models["variant"].isin(OFF_VARIANTS)]
    on = on[on["model_version"].map(run_mode) == "on"]
    wrong = [m for m in on["model_version"] if not is_reasoning(m)]
    assert not wrong, wrong


def test_bare_runs_of_hybrid_releases_are_left_out_of_both_fits(models):
    """Whatever the vendor: a release that runs both with and without thinking has a bare run
    of unknown mode."""
    hybrid = hybrid_families()
    bare = [m for m in models["model_version"]
            if is_bare(m) and run_mode(m) == "default" and model_family(m) in hybrid]
    kept = [m for m in bare if not regime_ignored(m)]
    assert not kept, kept
    # A release with an off run is hybrid whatever its line (not only Claude).
    fam_off = {model_family(m) for m in models["model_version"] if run_mode(m) == "off"}
    assert fam_off <= hybrid
    assert any(not f.startswith("claude") for f in fam_off & hybrid)


def test_a_reasoning_only_release_keeps_its_bare_run():
    assert not regime_ignored("o3-2025-04-16") and is_reasoning("o3-2025-04-16")


def test_gemini_25_flash_lite_without_budget_is_not_reasoning():
    assert not is_reasoning("gemini-2.5-flash-lite-preview-09-2025")
    assert is_reasoning("gemini-2.5-pro") and is_reasoning("gemini-2.5-flash")


@pytest.mark.parametrize("patterns", ["REASONING_FAMILY_PATTERNS", "NON_REASONING_PATTERNS",
                                      "HYBRID_LINE_PATTERNS"])
def test_every_regime_pattern_matches_a_current_model(models, patterns):
    names = list(models["model_version"])
    dead = [p for p in getattr(config, patterns) if not any(re.search(p, n, re.I) for n in names)]
    assert not dead, f"config.{patterns}: patterns no current model matches, drop them: {dead}"


def test_regimes_match_the_reviewed_table():
    """The code's regime of every model equals 1_curated/model_regimes.csv. A sync that adds a
    model, or a rule change, fails here until the table is rebuilt
    (python 1_curated/4_build_model_regimes.py) and its diff read."""
    reviewed = pd.read_csv(REGIMES, dtype=str).set_index("model_version")["regime"]
    now = regime_table().set_index("model_version")["regime"]
    both = pd.concat({"reviewed": reviewed, "code": now}, axis=1)
    diff = both[both["reviewed"] != both["code"]]
    assert diff.empty, ("regimes differ from 1_curated/model_regimes.csv (NaN: a model new to "
                        "one side); review, then rebuild the table:\n" + diff.to_string())
