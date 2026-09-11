"""Open-weights or closed, one label per model of 0_input/models.csv -> 1_curated/model_openness.csv.

A model is `open` when its weights can be downloaded (Epoch's "Open weights" classes, whatever
the licence), `closed` when it is reachable through an API or a hosted product only, or never
released. The state at release counts: grok-2 is closed although its weights came out a year
later. Sources, in order:

  1. Epoch's model metadata (`accessibility`), read from the pipeline checkout's
     0_input/feed/epoch/model_metadata.csv, matched on the pipeline's model_version and on the
     raw Epoch names of 2_database/model_names.csv;
  2. the family: a release and its reasoning efforts share one state (`data.model_family`), so
     `gpt-5-2025-08-07_thinking` reads off `gpt-5-2025-08-07`;
  3. `model_openness_overrides.csv`, hand-researched, for what Epoch does not cover (it also
     wins over Epoch where both have a row).

Anything still unlabelled is written as `unknown` and listed; the frontier diagnostics refuse
to run on an unknown candidate.

  python 1_curated/3_build_model_openness.py [--pipeline ../benchmark-data-pipeline]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "2_model"))

from multiaxis_eci.config import PIPELINE_DIR  # noqa: E402
from multiaxis_eci.data import MODELS_FILE, model_family  # noqa: E402

OUT = ROOT / "1_curated" / "model_openness.csv"
OVERRIDES = ROOT / "1_curated" / "model_openness_overrides.csv"
OPEN_CLASSES = ("Open weights (unrestricted)", "Open weights (restricted use)",
                "Open weights (non-commercial)")
CLOSED_CLASSES = ("API access", "Hosted access (no API)", "Unreleased")


def epoch_labels(pipeline: Path, models: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """model_version -> (label, source) from Epoch's accessibility column."""
    meta = pd.read_csv(pipeline / "0_input" / "feed" / "epoch" / "model_metadata.csv",
                       low_memory=False).dropna(subset=["accessibility"])
    names = pd.read_csv(pipeline / "2_database" / "model_names.csv")

    def label(acc: str) -> str | None:
        if acc in OPEN_CLASSES:
            return "open"
        if acc in CLOSED_CLASSES:
            return "closed"
        return None

    by_version = meta.drop_duplicates("model_version").set_index("model_version")["accessibility"]
    by_display = meta.drop_duplicates("display_name").set_index("display_name")["accessibility"]
    out = {}
    for m in models["model_version"]:
        acc = by_version.get(m)
        if isinstance(acc, str) and label(acc):
            out[m] = (label(acc), f"epoch:{acc}")
    raw = (names[names["feed"] == "epoch"]
           .merge(models[["model_id", "model_version"]], on="model_id"))
    raw["acc"] = raw["raw_name"].map(by_display).fillna(raw["raw_name"].map(by_version))
    for m, grp in raw.dropna(subset=["acc"]).groupby("model_version"):
        if m in out:
            continue
        labels = {label(a) for a in grp["acc"]} - {None}
        if len(labels) == 1:
            acc = grp["acc"].mode().iloc[0]
            out[m] = (labels.pop(), f"epoch:{acc}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline", default=str(PIPELINE_DIR),
                    help="benchmark-data-pipeline checkout (default: the sibling directory)")
    args = ap.parse_args()

    models = pd.read_csv(MODELS_FILE, dtype=str)
    labels = epoch_labels(Path(args.pipeline), models)
    over = pd.read_csv(OVERRIDES, dtype=str)
    bad = over[~over["openness"].isin(["open", "closed"])]
    if len(bad):
        raise ValueError(f"overrides must say open or closed: {bad['model_version'].tolist()}")
    for r in over.itertuples(index=False):
        labels[r.model_version] = (r.openness, "override")

    # The family carries its state to the members Epoch does not list (effort variants, run
    # variants); a family whose members disagree is an error to fix in the overrides.
    fam: dict[str, dict[str, str]] = {}
    for m, (lab, _) in labels.items():
        fam.setdefault(model_family(m), {})[m] = lab
    conflicts = {f: d for f, d in fam.items() if len(set(d.values())) > 1}
    if conflicts:
        raise ValueError(f"families with both labels: {conflicts}")

    rows = []
    for m in models["model_version"]:
        if m in labels:
            lab, src = labels[m]
        elif model_family(m) in fam:
            member = next(iter(fam[model_family(m)]))
            lab, src = fam[model_family(m)][member], f"family:{member}"
        else:
            lab, src = "unknown", ""
        rows.append({"model_version": m, "openness": lab, "source": src})
    out = pd.DataFrame(rows).sort_values("model_version").reset_index(drop=True)
    OUT.write_text(out.to_csv(index=False))
    counts = out["openness"].value_counts().to_dict()
    print(f"wrote {OUT}: {counts}")
    unknown = out[out["openness"] == "unknown"]
    if len(unknown):
        info = models.set_index("model_version").loc[unknown["model_version"],
                                                       ["organization", "publication_date"]]
        print("unknown (add to model_openness_overrides.csv):")
        print(info.to_string())


if __name__ == "__main__":
    main()
