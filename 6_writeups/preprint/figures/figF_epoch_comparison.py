"""Figure F: our single-axis index (ECI-H) against Epoch's published ECI, one point per matched model.

Reads
- 5_outputs/data20261019/canonical_humanmerge/all_models_eci.csv: ours (posterior median, 95% central
  interval, n_obs) per versioned model id;
- deliverables/sciencespo_plan_20261021/epoch_eci.csv: Epoch's ECI (eci_scores.csv of 2026-10-21);
- the pipeline snapshot deliverables/sciencespo_refit_20261020/pipeline/: Epoch's own ECI input
  (0_input/feed/epoch/processed_data_for_eci.csv, which lists the model_version ids Epoch pools into
  each ECI row) and the pipeline's name table (2_database/20261020/model_names.csv, raw name to model
  identity, with models.csv for the identity's id).
Writes figF_epoch_comparison.pdf (vector, 160 mm) and .png (preview) beside this file, in the\nstyle of style.py, and ../tables/epoch_comparison_matches.csv.

Matching. epoch_eci.csv's model_versions column is empty, so the version ids come from Epoch's ECI
input. Each id is resolved to our id by the pipeline's name table (feeds epoch and epoch_live), or by
the same id string when the table does not list it. An Epoch row is matched when at least one resolved
id is in our index and no resolved id also belongs to another Epoch row. No name is matched by
similarity. Variant rule: Epoch pools all effort settings and snapshots of a model into one row; we
take, among our ids for that row, the one with the most observations (ties: the first id by name).
"""
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
from style import MID, PALETTE, TINTS, save  # noqa: E402  (this folder is on sys.path)

SNAP = REPO / "deliverables/sciencespo_refit_20261020/pipeline"
ours = pd.read_csv(REPO / "5_outputs/data20261019/canonical_humanmerge/all_models_eci.csv").set_index("name")
epoch = pd.read_csv(REPO / "deliverables/sciencespo_plan_20261021/epoch_eci.csv")
pooled = pd.read_csv(SNAP / "0_input/feed/epoch/processed_data_for_eci.csv")[["Model", "model_version"]]
names = pd.read_csv(SNAP / "2_database/20261020/model_names.csv").query("feed in ['epoch', 'epoch_live']")
version_of = pd.read_csv(SNAP / "2_database/20261020/models.csv").set_index("model_id")["model_version"]
names["ours"] = names["model_id"].map(version_of)
by_raw = names.groupby("raw_name")["ours"].agg(set)

pooled = pooled.drop_duplicates()
pooled["ours"] = [by_raw.get(v, {v}) for v in pooled["model_version"]]
cand = pooled.explode("ours").dropna(subset=["ours"])
cand = cand[cand["ours"].isin(ours.index)].drop_duplicates(["Model", "ours"])
shared = cand.groupby("ours")["Model"].nunique().loc[lambda s: s > 1].index
clash = set(cand.loc[cand["ours"].isin(shared), "Model"])
cand = cand[~cand["Model"].isin(clash)]
cand["n_obs"] = cand["ours"].map(ours["n_obs"])
pick = cand.sort_values(["n_obs", "ours"], ascending=[False, True]).groupby("Model").agg(
    ours=("ours", "first"), n_obs=("n_obs", "first"), our_variants=("ours", "size"),
    all_our_ids=("ours", lambda s: ";".join(sorted(s))))

m = epoch.merge(pick, left_on="Model", right_index=True, how="left")
m["status"] = "matched"
m.loc[m["ours"].isna(), "status"] = "unmatched: no id of ours"
m.loc[m["Model"].isin(clash), "status"] = "unmatched: an id of ours also in another Epoch row"
m.loc[~m["Model"].isin(set(pooled["Model"])), "status"] = "unmatched: absent from Epoch's ECI input"
m = m.join(ours[["mean", "hdi_low", "hdi_high"]].rename(columns=lambda c: f"ours_{c}"), on="ours")
m = m.rename(columns={"ours_mean": "ours_median", "ours": "our_id"})
m["difference"] = m["ours_median"] - m["eci"]
cols = ["Model", "status", "our_id", "n_obs", "our_variants", "all_our_ids", "eci", "eci_ci_low",
        "eci_ci_high", "ours_median", "ours_hdi_low", "ours_hdi_high", "difference", "date"]
m[cols].sort_values(["status", "eci"], ascending=[True, False]).to_csv(
    HERE.parent / "tables/epoch_comparison_matches.csv", index=False)

ok = m[m["status"] == "matched"]
rho, r = spearmanr(ok["eci"], ok["ours_median"])[0], pearsonr(ok["eci"], ok["ours_median"])[0]
print(m["status"].value_counts().to_string(), f"\nSpearman {rho:.3f}  Pearson {r:.3f}")
print(ok.reindex(ok["difference"].abs().sort_values(ascending=False).index)[
    ["Model", "our_id", "n_obs", "eci", "ours_median", "difference"]].head(8).to_string(index=False))

print(f"matched {len(ok)} / {len(m)} Epoch rows; Spearman {rho:.3f}, Pearson {r:.3f}")

lo = min(ok["eci_ci_low"].min(), ok["ours_hdi_low"].min()) - 3
hi = max(ok["eci_ci_high"].max(), ok["ours_hdi_high"].max()) + 3
bar = dict(type="data", symmetric=False, thickness=1, width=0, color=TINTS[0])
fig = go.Figure([
    go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="y = x",
               line=dict(color=MID, width=1, dash="dash")),
    go.Scatter(x=[None], y=[None], mode="lines", line=dict(color=TINTS[0], width=2),
               name="Interval: Epoch's published (horizontal), our 95% (vertical)"),
    go.Scatter(x=ok["eci"], y=ok["ours_median"], mode="markers", name="Model (median)",
               marker=dict(color=PALETTE[0]),
               error_x=dict(bar, array=ok["eci_ci_high"] - ok["eci"], arrayminus=ok["eci"] - ok["eci_ci_low"]),
               error_y=dict(bar, array=ok["ours_hdi_high"] - ok["ours_median"],
                            arrayminus=ok["ours_median"] - ok["ours_hdi_low"]))])
fig.update_xaxes(title_text="Epoch's ECI", range=[lo, hi], constrain="domain")
fig.update_yaxes(title_text="Our index (ECI-H)", range=[lo, hi], scaleanchor="x", scaleratio=1,
                 constrain="domain")
fig.update_layout(legend=dict(orientation="h", x=0, y=-0.1, xanchor="left", yanchor="top",
                              traceorder="reversed"))   # below the plot, clear of data and grid
save(fig, HERE / "figF_epoch_comparison", height=520)
