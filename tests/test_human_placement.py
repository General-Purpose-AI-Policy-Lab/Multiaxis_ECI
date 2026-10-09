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
    flagship = {"Fluid Intelligence & Formal Sciences", "Scientific Knowledge & Reasoning",
                "Agentic Capabilities & Programming", "Cyber Capabilities"}
    # an exploration fit's own titles (semicomp0 mode A, 2026-10-08) borrow the flagship's
    # judgments through 5_place_humans.RULE_TITLES
    exploration = {"Cyber Capabilities & Reasoning", "Fluid Intelligence & Mathematics",
                   "Scientific Knowledge"}
    assert flagship <= set(rules) <= flagship | exploration


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


def test_flagship_folder_carries_no_humans():
    from multiaxis_eci.analysis import FLAGSHIP
    assert FLAGSHIP.no_humans and FLAGSHIP.tag.endswith("_nohumans")
