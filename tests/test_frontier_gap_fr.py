"""The French of the frontier-gap figures is written for every group pair, not for one cut: each
caption naming a group, built the way 4_diagnostics/2_plot_frontier_gap.py builds it, comes out
with no English left (the US vs China table once read "Mean lag of the last 12 mois' Chinese
modèles records")."""
import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "2_model"))

from multiaxis_eci.analysis.frontier_gap import GROUP_NOUNS, GROUP_PAIRS, GROUP_TITLES  # noqa: E402
from multiaxis_eci.viz.i18n import FR_GROUPS, translate_text  # noqa: E402

_spec = importlib.util.spec_from_file_location("plot_frontier_gap",
                                               ROOT / "4_diagnostics" / "2_plot_frontier_gap.py")
plot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plot)

ENGLISH = re.compile(r"\b(models?|months|lag|latest|frontier|trend|slope|gap|today|"
                     r"behind|first|above|Chinese|US|Closed|Open-weights|closed|open-weights|vs)\b")


def _captions(kind: str) -> list[str]:
    leader, follower = GROUP_PAIRS[kind]
    titles, nouns = GROUP_TITLES[kind], GROUP_NOUNS[kind]
    names = {"leader": titles[leader], "follower": titles[follower],
             "leader_n": nouns[leader][0], "follower_n": nouns[follower][0]}
    out = [label.format(**names) for _, _, label, _, _ in plot.TABLE_ROWS]
    out += [label.format(**names) for _, label in plot.SHOWN_ROWS]
    out += [f"Months behind the {nouns[leader][0]} frontier, median [80% interval]",
            f"Months behind the {nouns[leader][0]}<br>frontier (ECI-H)",
            f"Release date of the {nouns[follower][0]} record",
            f"{titles[leader]} vs {titles[follower]} (ECI-H), by benchmark access",
            f"first {nouns[leader][1]} above:"]
    for g in (leader, follower):
        out += [f"{titles[g]}: frontier record", f"{titles[g]}: trend (80% band)",
                f"{titles[g]} trend", titles[g]]
    return out


@pytest.mark.parametrize("kind", sorted(GROUP_PAIRS))
def test_every_group_caption_is_french(kind):
    leftover = {c: fr for c in _captions(kind)
                if ENGLISH.search(fr := translate_text(c).replace("ECI-H", ""))}
    assert not leftover, leftover


@pytest.mark.parametrize("kind", sorted(GROUP_PAIRS))
def test_french_names_the_right_groups(kind):
    leader, follower = GROUP_PAIRS[kind]
    nouns = GROUP_NOUNS[kind]
    title = translate_text(f"Months behind the {nouns[leader][0]} frontier, median [80% interval]")
    assert FR_GROUPS[leader][1] in title
    row = translate_text(f"Lag of the latest {nouns[follower][0]} record")
    assert row.endswith(FR_GROUPS[follower][1])
