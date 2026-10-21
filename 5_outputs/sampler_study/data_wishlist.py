"""Which measurements would separate the flagship's posterior modes.

For every (model, benchmark) cell with no score, each mode's predictive distribution of the
score: mu = c + (d - c) * sigmoid(A . theta - D) on the mode's chains, its mean, and its variance
(posterior variance of mu plus the Beta noise mu(1 - mu) / (1 + phi)). For a pair of modes the
utility of a cell is the gap between their means in units of their pooled predictive sd,
z = |p1 - p2| / s; a score measured there moves the odds between the two by about exp(z^2 / 2).
A cell's utility is its best pair (reported beside it), and z_A its best pair involving mode A,
the one the deliverables read.

Reads the 2026-10-19 flagship (ceilings, skew-normal, 20 chains of seeds 42 and 43) and the
modes of its modes file. Writes data_wishlist.csv (the ten best models of every benchmark, among
all models, among Anthropic's and among open-weight ones) and data_wishlist.md beside this script.

    .venv/bin/python 5_outputs/sampler_study/data_wishlist.py
"""
import itertools
import json
import sys
from pathlib import Path

import arviz as az
import numpy as np
import pandas as pd
import xarray as xr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "2_model"))
from multiaxis_eci.analysis.fitspec import FitSpec  # noqa: E402
from multiaxis_eci.data import (_known_release_date_by_model, load_model_openness,  # noqa: E402
                                model_organization)

FIT = ROOT / "5_outputs/data20261019/mirt_lineageprior_lineagebm"
TRACE = FIT / "trace_mirt_k4_lineageprior_lineagebm.nc"
N_BENCH, N_MODELS, N_CSV, N_OWN = 15, 5, 10, 10

modes = json.loads((FIT / f"mirt_modes_{TRACE.stem}.json").read_text())
CHAINS = {m["label"]: m["chains"] for m in modes["modes"]}
DLOGP = {m["label"]: -m["delta_logp"] for m in modes["modes"]}
LABELS = sorted(CHAINS)

with xr.open_dataset(TRACE, group="posterior") as p:
    spec = FitSpec.from_trace(az.InferenceData(posterior=p.isel(draw=slice(0, 1))), TRACE)
data, floor_c, _ = spec.load_data()
bench = data.blookup.sort_values("benchmark_idx")["benchmark"].tolist()
names = data.mlookup.sort_values("model_idx")["model"].to_numpy()
c = floor_c if floor_c is not None else np.zeros(len(bench))
observed = np.zeros((len(names), len(bench)), bool)
observed[data.model_idx, data.bench_idx] = True

post = xr.open_dataset(TRACE, group="posterior")


def draws(var, chains):
    v = post[var].isel(chain=chains).values
    return v.reshape(-1, *v.shape[2:])


def aligned_mean_loadings(chains):
    """The mode's mean loadings, each chain's axes permuted onto the first chain's."""
    per_chain = post["A"].isel(chain=chains).mean("draw").values
    ref, out = per_chain[0], []
    for a in per_chain:
        pm = max(itertools.permutations(range(a.shape[1])),
                 key=lambda q: sum(np.corrcoef(ref[:, k], a[:, q[k]])[0, 1] for k in range(a.shape[1])))
        out.append(a[:, list(pm)])
    return np.mean(out, 0)


pred, axis = {}, {}
a_titles = {k: a["title_fr"] for k, a in json.loads((FIT / "axis_names.json").read_text())["axes"].items()}
a_load = pd.read_csv(FIT / "mirt_loadings.csv")
a_top = a_load.loc[a_load.groupby("benchmark")["axis_share"].idxmax()].set_index("benchmark")["axis"]
for mode, ch in CHAINS.items():
    th, A, D = draws("theta", ch), draws("A", ch), draws("D", ch)
    d, phi = draws("ceiling_d", ch), draws("phi_b", ch)
    mean = np.empty((len(names), len(bench)))
    var = np.empty_like(mean)
    for b in range(len(bench)):
        eta = np.einsum("smk,sk->sm", th, A[:, b]) - D[:, b, None]
        mu = c[b] + (d[:, b, None] - c[b]) / (1 + np.exp(-eta))
        mean[:, b] = mu.mean(0)
        var[:, b] = mu.var(0) + (mu * (1 - mu) / (1 + phi[:, b, None])).mean(0)
    pred[mode] = mean, var
    if mode == "A":           # the named axes of the deliverables
        axis[mode] = np.asarray([a_titles[a_top[b]] for b in bench])
    else:                     # an unnamed mode: each axis by its two most collinear benchmarks
        M = aligned_mean_loadings(ch)
        share = M ** 2 / (M ** 2).sum(1, keepdims=True)
        dom = share.argmax(1)
        lab = {k: " / ".join(np.asarray(bench)[np.argsort(-np.where(dom == k, share[:, k], -1))[:2]])
               for k in range(M.shape[1])}
        axis[mode] = np.asarray([lab[k] for k in dom])

pairs = list(itertools.combinations(LABELS, 2))
zs = {}
for m1, m2 in pairs:
    (p1, v1), (p2, v2) = pred[m1], pred[m2]
    zs[m1 + m2] = np.abs(p1 - p2) / np.sqrt((v1 + v2) / 2)
Z = np.stack([zs[k] for k in zs])
best = Z.argmax(0)
z = Z.max(0)
a_pairs = [i for i, k in enumerate(zs) if "A" in k]
z_A = Z[a_pairs].max(0)

openness = load_model_openness()
released = _known_release_date_by_model()
mi, bi = np.nonzero(~observed & ~data.is_human[:, None])
pair_names = np.asarray([f"{k[0]}-{k[1]}" for k in zs])
cells = pd.DataFrame({
    "benchmark": np.asarray(bench)[bi], "model": names[mi],
    "z": z[mi, bi], "pair": pair_names[best[mi, bi]], "z_A": z_A[mi, bi],
    **{f"p_{m}": pred[m][0][mi, bi] for m in LABELS},
    **{f"sd_{m}": np.sqrt(pred[m][1][mi, bi]) for m in LABELS},
    **{f"axis_{m}": axis[m][bi] for m in LABELS},
})
cells["openness"] = cells["model"].map(openness).fillna("unknown")
cells["organization"] = cells["model"].map(model_organization)
cells["released"] = cells["model"].map(released).dt.strftime("%Y-%m").fillna("")
cells = cells.sort_values("z", ascending=False).reset_index(drop=True)


def per_benchmark(df, n):
    """The n best models of each benchmark, benchmarks ranked by their best cell."""
    t = df.groupby("benchmark", sort=False).head(n).copy()
    t["benchmark_rank"] = t["benchmark"].map(
        t.groupby("benchmark", sort=False)["z"].max().rank(ascending=False, method="first")).astype(int)
    return t.sort_values(["benchmark_rank", "z"], ascending=[True, False])


groups = {"benchmark": cells,
          "anthropic": cells[cells["organization"] == "Anthropic"],
          "open": cells[cells["openness"] == "open"]}
tables = {q: per_benchmark(df, N_CSV).assign(question=q) for q, df in groups.items()}
cols = (["question", "benchmark_rank", "benchmark", "model", "organization", "openness", "released",
         "z", "pair", "z_A"] + [f"p_{m}" for m in LABELS] + [f"sd_{m}" for m in LABELS]
        + [f"axis_{m}" for m in LABELS])
pd.concat(tables.values())[cols].round(3).to_csv(HERE / "data_wishlist.csv", index=False)


# ---- markdown ----
def cell(r):
    extra = " (open)" if r.openness == "open" else ""
    preds = " ".join(f"{m} {getattr(r, 'p_' + m):.2f}" for m in LABELS)
    return f"{r.model}{extra} {r.z:.1f} {r.pair} [{preds}]"


def md_table(t, n_bench, axes=True):
    rows = []
    for rank, g in t.groupby("benchmark_rank"):
        if rank > n_bench:
            break
        r0 = g.iloc[0]
        ax = f" {r0.axis_A} |" if axes else ""
        rows.append(f"| {rank} | {r0.benchmark} | {r0.z:.1f} ({r0.pair}) |{ax} "
                    + " ; ".join(cell(r) for r in g.head(N_MODELS).itertuples()) + " |")
    head = ("| rang | benchmark | meilleur z (paire) |" + (" axe dans A |" if axes else "")
            + " modèles par ordre d'utilité (z, paire ; prédictions des modes) |\n|"
            + "---|" * (5 if axes else 4))
    return head + "\n" + "\n".join(rows)


def mode_axes(m):
    if m == "A":
        return " ; ".join(f"*{t}*" for t in a_titles.values())
    return " ; ".join(sorted(set(axis[m])))


n_cells = len(cells)


def a_vs(m, n=8):
    """The n best cells for telling A from mode m, one per model and benchmark."""
    zz = zs["A" + m][mi, bi]
    df = pd.DataFrame({"benchmark": np.asarray(bench)[bi], "model": names[mi], "z": zz,
                       "pA": pred["A"][0][mi, bi], "pm": pred[m][0][mi, bi]})
    df = df.sort_values("z", ascending=False).drop_duplicates("benchmark").head(n)
    rows = [f"| {r.benchmark} | {r.model} | {r.z:.1f} | {r.pA:.2f} | {r.pm:.2f} |" for r in df.itertuples()]
    return (f"| benchmark | modèle | z (A-{m}) | A | {m} |\n|---|---|---|---|---|\n" + "\n".join(rows))


a_sections = "\n\n".join(f"### A contre {m}\n\n{a_vs(m)}" for m in LABELS if m != "A")
pair_count = cells.head(2000)["pair"].value_counts()
modes_md = "\n".join(
    f"- **{m}** : {len(CHAINS[m])} chaînes ({', '.join(map(str, CHAINS[m]))}), "
    + (f"{DLOGP[m]:.1f} nats sous A".replace(".", ",") if m != "A" else "la meilleure log-densité")
    + f" ; axes : {mode_axes(m)}" for m in LABELS)

md = f"""# Wishlist : mesures pour départager les modes du flagship

Générée par `data_wishlist.py` à partir du flagship du 19 octobre (K = 4, sans humains, capacités skew-normal, plafonds estimés), 20 chaînes (graines 42 et 43) en {len(LABELS)} modes :

{modes_md}

Les livrables lisent A. PSIS-LOO ne classe pas les modes et la LOO intégrée ne séparait pas A de D sur les 10 premières chaînes (+3 ± 8 nats pour A) : seules de nouvelles mesures peuvent les départager.

**Comment lire.** Pour chaque case (modèle, benchmark) sans score, chaque mode donne une prédiction : sa moyenne et son écart-type, qui ajoute à l'incertitude du mode le bruit propre au benchmark. Pour deux modes, l'utilité *z* est l'écart entre leurs moyennes en écarts-types prédictifs ; une mesure tombée sur la prédiction d'un mode multiplie à peu près ses chances contre l'autre par e^(z²/2) (z = 1,5 : × 3 ; z = 2 : × 7). Une case vaut sa meilleure paire, indiquée à côté (`A-C` : la case départage A et C) ; le csv donne aussi `z_A`, sa meilleure paire avec A. Les benchmarks sont classés par leur meilleure case ; pour chacun, les modèles par ordre d'utilité : si un benchmark n'est pas accessible, prendre le suivant ; si un modèle ne l'est pas, le suivant sur la même ligne. Les axes de B, C et D, qui n'ont pas de noms, sont désignés par leurs deux benchmarks les plus colinéaires. La disponibilité en API n'a pas été vérifiée ; `open` = poids ouverts. {f'{n_cells:,}'.replace(',', ' ')} cases non observées ; `data_wishlist.csv` garde les dix meilleurs modèles de chaque benchmark, parmi tous les modèles (`question = benchmark`), parmi ceux d'Anthropic (`anthropic`) et parmi les poids ouverts (`open`).

**Paires départagées par les 2 000 meilleures cases** : {', '.join(f'{k} {v}' for k, v in pair_count.items())}.

**Prudence.** B et D ne reposent que sur 2 chaînes chacun.

## En bref (lecture du 21 octobre, écrite à la main)

1. **La plupart des meilleures cases opposent deux modes que les livrables ne lisent pas** (B-D, C-D, B-C : 1 109 des 2 000 meilleures). Pour les livrables, ce qui compte est de départager A de chacun des autres : voir la section « Départager A ».
2. **A contre C**, le rival le plus fréquent (7 chaînes, 27 nats plus bas), est le plus difficile à départager : les deux modes ont les mêmes axes agentique et cyber et ne diffèrent que par la frontière entre connaissances et raisonnement. Les meilleures cases sont à z ≈ 2 : gpt-5.4-nano (sans raisonnement) sur ARC AI2 (A 0,79 contre C 0,91) et OpenBookQA (A 0,72 contre C 0,87) ; Phi-3-small sur Lech Mazur Writing ; falcon-180B sur CyberGym.
3. **A contre D** (l'ancienne structure, 33 nats plus bas) se départage comme avant : petits modèles d'OpenAI et petits modèles ouverts sur les QCM scientifiques (gpt-5.4-nano sur GPQA Main Chimie, A 0,67 contre D 0,46 ; Qwen2.5-Coder 1,5B sur MATH Level 5, A 0,60 contre D 0,16) et sur le sens commun dans l'autre sens (gpt-5-nano sur OpenBookQA, A 0,67 contre D 0,88).
4. **A contre B** est le moins séparable (meilleure case z = 1,8 : gpt-6-sol sans raisonnement sur GPQA Main Chimie, A 0,78 contre B 0,62).
5. **Un petit lot teste A contre les trois** : gpt-5.4-nano à plusieurs efforts sur ARC AI2, OpenBookQA et GPQA Main Chimie. Côté Anthropic, Claude Opus 4.1 thinking sur GPQA Main Chimie (A 0,63 contre B 0,74 et D 0,75) départage A de B et de D ; les Claude récents sur ARC AI2 et OpenBookQA séparent surtout C de D.

## Benchmarks classés

{md_table(tables["benchmark"], N_BENCH)}

## Départager A de chacun des autres modes

La meilleure case de chaque benchmark pour chaque paire avec A.

{a_sections}

## Ce que nous pouvons mesurer nous-mêmes

### Modèles Anthropic (comme la campagne de mesures propres, protocole RAND)

{md_table(tables["anthropic"], N_OWN, axes=False)}

### Modèles à poids ouverts

{md_table(tables["open"], N_OWN, axes=False)}
"""
(HERE / "data_wishlist.md").write_text(md)
print(md)
