"""Which measurements would separate the flagship's two posterior modes.

For every (model, benchmark) cell with no score, each mode's predictive distribution of the
score: mu = c + (d - c) * sigmoid(A . theta - D) on the mode's chains, its mean, and its variance
(posterior variance of mu plus the Beta noise mu(1 - mu) / (1 + phi)). The utility of a cell is the
gap between the two modes' means in units of their pooled predictive sd, z = |pA - pB| / s; a
score measured there moves the odds between the modes by about exp(z^2 / 2). The raw gap is
reported beside it.

Reads the 2026-10-19 flagship (ceilings, skew-normal, 10 chains of seed 42), modes A and B of its
modes file. Writes data_wishlist.csv (the ten best models of every benchmark, among all models,
among Anthropic's and among open-weight ones) and data_wishlist.md beside this script.

    .venv/bin/python 5_outputs/sampler_study/data_wishlist.py
"""
import json
import subprocess
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
# the seed-42 trace, kept aside while seed 43 is merged into the folder's copy
TRACE = ROOT.parent / "traces_backup/trace_mirt_k4_lineageprior_lineagebm_seed42.nc"
MODES_FILE = "5_outputs/data20261019/mirt_lineageprior_lineagebm/mirt_modes_trace_mirt_k4_lineageprior_lineagebm.json"
N_BENCH, N_MODELS, N_CSV, N_OWN = 15, 5, 10, 10

# The refit sets the modes file aside, so it is read from the commit that published the modes.
modes = json.loads(subprocess.run(["git", "show", f"52f5d93:{MODES_FILE}"], cwd=ROOT,
                                  capture_output=True, text=True, check=True).stdout)
CHAINS = {m["label"]: m["chains"] for m in modes["modes"]}
assert set(CHAINS) == {"A", "B"}

with xr.open_dataset(TRACE, group="posterior") as p:
    # the spec is checked against the folder the trace was fitted in
    spec = FitSpec.from_trace(az.InferenceData(posterior=p.isel(draw=slice(0, 1))),
                              FIT / TRACE.name.replace("_seed42", ""))
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


pred = {}
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

pA, vA = pred["A"]
pB, vB = pred["B"]
gap = pA - pB
z = np.abs(gap) / np.sqrt((vA + vB) / 2)

# each benchmark's most collinear axis in each mode, on the mode's own aligned frame
def main_axis(folder):
    names = {k: a["title_fr"] for k, a in json.loads((folder / "axis_names.json").read_text())["axes"].items()}
    load = pd.read_csv(folder / "mirt_loadings.csv")
    top = load.loc[load.groupby("benchmark")["axis_share"].idxmax()].set_index("benchmark")["axis"]
    return np.asarray([names[top[b]] for b in bench])


axis = {"A": main_axis(FIT), "B": main_axis(FIT / "secondary_mode")}

openness = load_model_openness()
released = _known_release_date_by_model()
mi, bi = np.nonzero(~observed & ~data.is_human[:, None])
cells = pd.DataFrame({
    "benchmark": np.asarray(bench)[bi], "model": names[mi],
    "z": z[mi, bi], "gap_A_minus_B": gap[mi, bi], "p_A": pA[mi, bi], "p_B": pB[mi, bi],
    "sd_A": np.sqrt(vA[mi, bi]), "sd_B": np.sqrt(vB[mi, bi]),
    "axis_A": axis["A"][bi], "axis_B": axis["B"][bi],
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
cols = ["question", "benchmark_rank", "benchmark", "model", "organization", "openness", "released",
        "z", "gap_A_minus_B", "p_A", "p_B", "sd_A", "sd_B", "axis_A", "axis_B"]
pd.concat(tables.values())[cols].round(3).to_csv(HERE / "data_wishlist.csv", index=False)


# ---- markdown ----
def cell(r):
    extra = " (open)" if r.openness == "open" else ""
    return f"{r.model}{extra} {r.z:.1f} [A {r.p_A:.2f} vs B {r.p_B:.2f}]"


def md_table(t, n_bench, axes=True):
    rows = []
    for rank, g in t.groupby("benchmark_rank"):
        if rank > n_bench:
            break
        r0 = g.iloc[0]
        ax = f" {r0.axis_A} / {r0.axis_B} |" if axes else ""
        rows.append(f"| {rank} | {r0.benchmark} | {r0.z:.1f} | {r0.gap_A_minus_B:+.2f} |{ax} "
                    + " ; ".join(cell(r) for r in g.head(N_MODELS).itertuples()) + " |")
    head = ("| rang | benchmark | meilleur z | écart A − B |" + (" axe (A / B) |" if axes else "")
            + " modèles par ordre d'utilité (z ; prédictions des modes) |\n|"
            + "---|" * (6 if axes else 5))
    return head + "\n" + "\n".join(rows)


n_cells = len(cells)
dlogp = -modes["modes"][1]["delta_logp"]
md = f"""# Wishlist : mesures pour départager les modes A et B du flagship

Générée le 2026-10-21 par `data_wishlist.py`, à partir du flagship du 19 octobre (K = 4, sans humains, capacités skew-normal, plafonds estimés) : 10 chaînes de la graine 42, mode A = chaînes {", ".join(map(str, CHAINS["A"]))} (la meilleure log-densité, celui que lisent les livrables), mode B = chaînes {", ".join(map(str, CHAINS["B"]))} ({f'{dlogp:.1f}'.replace('.', ',')} nats plus bas). La LOO intégrée ne les départage pas (+3 ± 8 nats pour A) : seules de nouvelles mesures le peuvent. La version précédente (2026-10-12, cinq modes A–E de l'ancien flagship sans plafonds et à prior gaussien) est dans l'historique git.

**Ce qui oppose les deux modes.** A sépare l'agentique (*Capacités agentiques & programmation*) du cyber (*Capacités cyber*) et a un axe *Sens commun & connaissances factuelles* ; B fusionne agentique et cyber, met les maths et ARC-AGI dans *Intelligence fluide & sciences formelles* et a un axe *Visuel & sens commun*. Une mesure départage les modes là où ils prédisent des scores différents pour un même modèle.

**Comment lire.** Pour chaque case (modèle, benchmark) sans score, chaque mode donne une prédiction : sa moyenne et son écart-type, qui ajoute à l'incertitude du mode le bruit propre au benchmark. L'utilité *z* est l'écart entre les deux moyennes en écarts-types prédictifs ; une mesure tombée sur la prédiction d'un mode multiplie à peu près ses chances contre l'autre par e^(z²/2) (z = 1,5 : × 3 ; z = 2 : × 7). Elle est préférée à l'écart brut, qui avantage les benchmarks bruités et les modèles mal mesurés. Les benchmarks sont classés par leur meilleure case ; pour chacun, les modèles par ordre d'utilité : si un benchmark n'est pas accessible, prendre le suivant ; si un modèle ne l'est pas, le suivant sur la même ligne. Les colonnes « axe » donnent l'axe qui porte le benchmark dans chaque mode. La disponibilité en API n'a pas été vérifiée ; `open` = poids ouverts. {f'{n_cells:,}'.replace(',', ' ')} cases non observées ; `data_wishlist.csv` garde les dix meilleurs modèles de chaque benchmark, parmi tous les modèles (`question = benchmark`), parmi ceux d'Anthropic (`anthropic`) et parmi les poids ouverts (`open`).

**Prudence.** B ne repose que sur 2 chaînes (800 tirages). La graine 43 (10 chaînes) tourne depuis le 21 octobre à 12 h 14 ; à son arrivée, les modes et cette liste sont à refaire.

## En bref (lecture du 21 octobre, écrite à la main)

1. **Les modes ne se séparent pas là où leurs axes diffèrent le plus.** L'agentique et le cyber, fusionnés dans B et séparés dans A, ne donnent presque pas d'écart de prédiction : CyberGym arrive 15ᵉ (z = 1,4). Le désaccord porte sur la place des petits modèles entre sciences et sens commun.
2. **Deux familles de QCM, en sens opposés.**
   - Sur les QCM scientifiques (GPQA Main et Diamond Chimie, MMLU-Pro Biologie et Chimie, WMDP Chimie, MATH Level 5, OTIS AIME), A prédit les petits modèles (nano d'OpenAI, petits modèles ouverts) plus haut que B, de 0,1 à 0,4 point.
   - Sur les QCM de sens commun (OpenBookQA, ARC AI2, HellaSwag), A prédit ces mêmes modèles plus bas, de 0,07 à 0,18.
   - Le signe dépend du modèle : sur GPQA Main Chimie, B prédit Claude Opus 4 / 4.1 et Sonnet 4.5 plus haut qu'A (d'environ 0,1) ; sur OpenBookQA et ARC AI2, A prédit Claude Haiku 5.5 et Sonnet 5 plus haut que B ; sur MATH Level 5, B prédit text-davinci-002 plus haut.
3. **Les cases décisives sont bon marché.** Petits modèles (gpt-5-nano, gpt-5.4-nano, gpt-4.1-nano, Qwen2.5-Coder 1,5B à 14B, Baichuan-2-13B, gemma-3-4b) sur des benchmarks que nous savons déjà lancer (GPQA, WMDP, MMLU-Pro, protocole RAND). Une case à z = 2,3 multiplie les chances d'un mode par ~14.
4. **Une paire de mesures teste les deux sens.** gpt-5.4-nano (effort low) sur GPQA Main Chimie (A 0,66 contre B 0,46) et Claude Opus 4.1 thinking sur le même benchmark (A 0,63 contre B 0,75). On peut y ajouter gpt-5-nano (high) sur OpenBookQA (A 0,70 contre B 0,88), qui teste le sens commun.
5. Les vieux modèles Anthropic (Claude 1.3 à 3.7) ne sont sans doute plus servis par l'API ; parmi les récents, les cases utiles sont Claude Haiku 5.5 (sans raisonnement) sur OpenBookQA et ARC AI2 (z = 1,6 à 1,7) et Sonnet 4.6 (sans raisonnement) sur OTIS AIME (z = 1,2).

## Benchmarks classés

{md_table(tables["benchmark"], N_BENCH)}

## Ce que nous pouvons mesurer nous-mêmes

### Modèles Anthropic (comme la campagne de mesures propres, protocole RAND)

{md_table(tables["anthropic"], N_OWN, axes=False)}

### Modèles à poids ouverts

{md_table(tables["open"], N_OWN, axes=False)}
"""
(HERE / "data_wishlist.md").write_text(md)
print(md)
