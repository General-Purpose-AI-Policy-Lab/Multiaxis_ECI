"""The pure pieces of the human placement on a fit without humans (analysis.human_placement)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from multiaxis_eci.analysis import human_placement as hp
from multiaxis_eci.config import HUMAN_ORDER_MERGED


def test_dominant_axis_is_the_largest_variance_share():
    A = np.array([[1.0, 0.9, 0.0], [0.1, 0.0, 0.2], [0.5, 0.5, 0.6]])
    assert hp.dominant_axis(A).tolist() == [0, 2, 2]


def test_content_file_parses_and_names_known_axes():
    rules = hp.content_rules()
    assert rules["Cyber Capabilities"]["Furniture Assembly"] is False
    assert rules["Cyber Capabilities"]["Cybench"] is True
    assert rules["Scientific Knowledge & Reasoning"]["InterCode-CTF"] is True
    # the flagships of 2026-10-19, without and with ceilings: each one's modes A and B
    flagship = {"Fluid Intelligence & Formal Sciences", "Scientific Knowledge & Reasoning",
                "Agentic & Cyber Capabilities", "Visual & Commonsense",
                "Agentic Capabilities & Programming", "Cyber Capabilities",
                "Commonsense & Factual Knowledge"}
    # an exploration fit's own titles (semicomp0 mode A, 2026-10-08) borrow the flagship's
    # judgments through 5_place_humans.RULE_TITLES
    exploration = {"Cyber Capabilities & Reasoning", "Fluid Intelligence & Mathematics",
                   "Scientific Knowledge",
                   # the geomin-rotated / pinned frame (2026-10-27) and the K=4 bifactor (10-28)
                   "Cyber & Expert-Level Tasks",
                   "Mixed (Visual, Trivia, Cyber)", "Reasoning & Programming",
                   # the K=4 bifactor fits renamed from collinearity (2026-10-29), g kept as General
                   # Capabilities with its notable parts
                   "Desk Research, Trivia & Commonsense",
                   "General Capabilities (Agentic, Cyber & SWE)",
                   "General Capabilities (Math, Science & Cyber)"}
    assert flagship <= set(rules) <= flagship | exploration


def test_a_tier_row_takes_one_tier_off_and_leaves_the_pair_judged():
    rules, off = hp.content_rules(), hp.tier_exclusions()
    assert rules["General Capabilities (Math, Science & Cyber)"]["MMLU"] is True
    assert ("MMLU", "Average Human") in off["General Capabilities (Math, Science & Cyber)"]
    assert "General Capabilities (Agentic, Cyber & SWE)" not in off


def test_drawn_tiers_own_score_or_sandwich():
    tiers = list(HUMAN_ORDER_MERGED)
    sc = pd.DataFrame({"tier": ["Skilled Generalist", "Committee of Domain Experts"],
                       "benchmark": ["b1", "b2"], "score": [0.7, 0.75]})
    drawn = hp.drawn_tiers(sc, tiers)
    # own scores, and the tiers between them in the order (an ancestor and a descendant scored)
    assert {"Skilled Generalist", "Committee of Domain Experts", "Domain Expert",
            "Committee of Skilled Generalists"} == set(drawn)
    assert "Average Human" not in drawn and "Top Performer" not in drawn


def test_a_ceiling_score_places_nothing_alone():
    tiers = list(HUMAN_ORDER_MERGED)
    sc = pd.DataFrame({"tier": ["Average Human", "Top Performer"], "benchmark": ["b", "b"],
                       "score": [0.53, 1.0]})
    drawn = hp.drawn_tiers(sc, tiers)
    assert drawn == ["Average Human"]


def test_a_score_within_noise_of_the_fitted_ceiling_bounds_from_below_only():
    """Top Performer's SimpleBench 0.954 against d_b 0.970 placed it at 6.4 [4.0, 9.5] on the
    flagship's axis 4: near d_b the curve is flat. Such a score is a ceiling score, draws no tier,
    and its likelihood is P(score >= observed), which never falls as ability rises."""
    flags = hp.at_ceiling([0.954, 0.80, 1.0], [213, 213, 0], [0.970, 0.970, 1.0], [300, 300, 300])
    assert flags.tolist() == [True, False, True]
    sc = pd.DataFrame({"tier": ["Average Human", "Top Performer"], "benchmark": ["b", "b"],
                       "score": [0.80, 0.954], "at_ceiling": [False, True]})
    assert hp.drawn_tiers(sc, list(HUMAN_ORDER_MERGED)) == ["Average Human"]

    import pytensor.tensor as pt
    mu = np.array([0.90, 0.95, 0.969])                        # rising ability, same score
    a, b = pt.as_tensor(mu * 300), pt.as_tensor((1 - mu) * 300)
    y, n = np.full(3, 0.954), np.full(3, 213.0)
    lp = hp._at_least_logp(y, a, b, n, y * n).eval()
    assert np.all(np.diff(lp) > 0) and lp[-1] < 0
    beta_rows = hp._at_least_logp(y, a, b, np.zeros(3), np.zeros(3)).eval()
    assert np.all(np.diff(beta_rows) > 0)


def test_flagship_is_fitted_without_humans_and_says_so_only_when_they_are_in():
    """Models only is the untagged default since 2026-10-21; humans in emit `_humans`, an order
    token implies them, and the legacy `_nohumans` still parses."""
    from multiaxis_eci.analysis import FLAGSHIP
    from multiaxis_eci.analysis.fitspec import FitSpec, _parse_tag
    assert FLAGSHIP.no_humans and "humans" not in FLAGSHIP.tag
    assert FitSpec(K=4, lineage_prior=True, no_humans=False).tag == "_lineageprior_humans"
    merged = FitSpec(K=4, human_merge=True, lineage_prior=True)
    assert merged.no_humans is False and merged.tag == "_lineageprior_humanmerge"
    assert _parse_tag("_humanmerge_lineageprior_lineagebm")["human_merge"] is True       # legacy
    assert _parse_tag("_lineageprior_lineagebm_nohumans")["no_humans"] is True
    assert _parse_tag("_lineageprior_humans")["no_humans"] is False
