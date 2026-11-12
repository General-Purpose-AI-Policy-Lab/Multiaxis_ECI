"""Table 1: the data, per source feed, plus the human baselines and the single-axis scope.

Reads
- the multi-axis fit's data, built without the trace by r1_runs/aligned_loadings.load_data()
  (fit.py's own parser and FitSpec for `--K 4 --lineage-prior --lineage-bm`);
- 0_input/all_scores_flat.csv (multiaxis_eci.data.PROCESSED_FILE), joined on (model, benchmark)
  for each observation's `feed` and `stderr`;
- 0_input/human_baselines.csv (multiaxis_eci.data.HUMAN_FILE), one row per human score;
- 5_outputs/data20261019/canonical_humanmerge/{pit.csv, human_groups_eci.csv, gof.json, summary.csv}
  and the human file the single-axis fit was run on, deliverables/canonical_sciencespo_20261022/source/0_input/.

Writes table1_data.csv (the per-feed table) and table1_data.md (all three blocks) beside this file.
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE.parent / "r1_runs"))
import aligned_loadings as al  # noqa: E402
from multiaxis_eci.data import HUMAN_FILE, PROCESSED_FILE  # noqa: E402

# The pipeline's `feed` column, named as the paper names the feeds.
FEEDS = {"epoch": "Epoch AI benchmark hub (ZIP)", "epoch_live": "Epoch AI internal evaluations (live feed)",
         "rand": "RAND RR-A3797-1", "seal": "Scale SEAL leaderboards", "kaggle": "Kaggle Open Benchmarks",
         "epoch_cyber": "Epoch AI cyber ECI", "gpai_runs": "GPAI Policy Lab local runs",
         "cybench_site": "Cybench leaderboard"}

data = al.load_data()
ml = data.mlookup.sort_values("model_idx")["model"].to_numpy()
bl = data.blookup.sort_values("benchmark_idx")["benchmark"].to_numpy()
obs = pd.DataFrame({"model_version": ml[data.model_idx], "benchmark": bl[data.bench_idx]})
flat = pd.read_csv(PROCESSED_FILE)
assert not flat.duplicated(["model_version", "benchmark"]).any()
obs = obs.merge(flat[["model_version", "benchmark", "feed", "source", "stderr"]],
                on=["model_version", "benchmark"], how="left", validate="one_to_one")
n_attr = int(obs["feed"].notna().sum())
obs["has_se"] = obs["stderr"].notna() & (obs["stderr"] > 0)


def counts(g):
    return pd.Series({"observations": len(g), "models": g["model_version"].nunique(),
                      "benchmarks": g["benchmark"].nunique(), "with reported SE": int(g["has_se"].sum())})


rows = obs.groupby("feed").apply(counts, include_groups=False)
rows = rows.sort_values("observations", ascending=False)
rows.index = [FEEDS[f] for f in rows.index]
rows.loc["Total"] = counts(obs)
rows = rows.astype(int)
rows.index.name = "feed"
rows.to_csv(HERE / "table1_data.csv")

# Human baselines: scores per tier, and how many sit on a benchmark of the multi-axis fit.
hum = pd.read_csv(HUMAN_FILE)
hum["in_fit_benchmarks"] = hum["benchmark"].isin(set(bl))
tiers = hum.groupby("group").agg(scores=("score", "size"), on_fit_benchmarks=("in_fit_benchmarks", "sum"),
                                 benchmarks=("benchmark", "nunique")).sort_values("scores", ascending=False)
tiers.index.name = "tier"
tiers.loc["Total"] = [len(hum), int(hum["in_fit_benchmarks"].sum()), hum["benchmark"].nunique()]

# Single-axis fit. pit.csv leaves out the scores of exactly 0 or 1, so its rows are fewer than gof.json's
# n_obs; the human scores of the fit come from the human file of the fit's own source snapshot.
K1 = REPO / "5_outputs/data20261019/canonical_humanmerge"
pit = pd.read_csv(K1 / "pit.csv")
is_h = pit["model"].isin(set(pd.read_csv(K1 / "human_groups_eci.csv")["name"]))
gof = json.loads((K1 / "gof.json").read_text())
n_D = int(pd.read_csv(K1 / "summary.csv").iloc[:, 0].str.startswith("D[").sum())
k1_src = pd.read_csv(REPO / "deliverables/canonical_sciencespo_20261022/source/0_input/human_baselines.csv")
k1_hum = k1_src[k1_src["benchmark"].isin(set(pit["benchmark"]))]
n_h = len(k1_hum)
n_h_bd = int(((k1_hum["score"] <= 0) | (k1_hum["score"] >= 1)).sum())
assert n_h - n_h_bd == is_h.sum(), "human rows of pit.csv do not match the fit's human file"

def to_md(df):
    """Markdown table without the optional tabulate dependency."""
    head = [df.index.name or ""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    return "\n".join(lines + ["| " + " | ".join([str(i)] + [str(v) for v in r]) + " |"
                               for i, r in zip(df.index, df.to_numpy())])


grouping = (obs.groupby("feed")["source"].agg(lambda s: f"{s.nunique()} source strings, e.g. "
                                                   + "; ".join(s.value_counts().index[:3]))
            .rename(index=FEEDS))
md = ["# Table 1. Data of the multi-axis model, by source feed", "",
      f"Observations attributed to a feed: {n_attr} / {data.n_obs} (join on model and benchmark).",
      f"Fit scope: {data.n_obs} observations, {data.n_models} models, {data.n_benchmarks} benchmarks; "
      f"processed table: {len(flat)} rows.", "",
      to_md(rows), "",
      "Models and benchmarks count those with at least one score from the feed; the feed rows do not sum "
      "to the total for these two columns. Reported SE: a positive `stderr` in the processed table.", "",
      "Feed grouping: the pipeline's own `feed` column of the processed table.", ""]
md += [f"- {k}: {v}" for k, v in grouping.items()]
md += ["", "## Human baselines (0_input/human_baselines.csv)", "",
       f"{len(hum)} scores, {hum['group'].nunique()} tiers, {hum['benchmark'].nunique()} benchmarks; "
       f"{int(hum['in_fit_benchmarks'].sum())} scores on a benchmark of the multi-axis fit. "
       "The 44 scores that place the tiers (README) need the fitted loadings and are not recomputed here.", "",
       to_md(tiers.astype(int)), "",
       "## Single-axis model scope (5_outputs/data20261019/canonical_humanmerge)", "",
       f"- With human rows: {gof['n_obs']} observations, {n_D} benchmarks (gof.json n_obs; difficulties in "
       f"summary.csv). Without: {gof['n_obs'] - n_h} observations.",
       f"- Human rows: {n_h} scores of {k1_hum['group'].nunique()} tiers on the {n_D} benchmarks, from the fit's "
       f"source human file ({len(k1_src)} rows; 0_input/human_baselines.csv has {len(hum)}).",
       f"- pit.csv: {len(pit)} rows = {int((~is_h).sum())} AI + {int(is_h.sum())} human, {pit['benchmark'].nunique()} "
       f"benchmarks; it omits the {gof['n_zero_score'] + gof['n_one_score']} scores of exactly 0 or 1 "
       f"({n_h_bd} of them human)."]
(HERE / "table1_data.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
