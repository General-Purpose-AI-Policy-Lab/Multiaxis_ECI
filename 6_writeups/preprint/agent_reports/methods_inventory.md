# Methods inventory: the model as implemented (repo HEAD 8a66950, data generation 20261019)

Paths are relative to `/Users/yassineessifi/Desktop/ECI_Bayesian`. `mirt.py` = `2_model/multiaxis_eci/models/mirt.py`; `config.py`, `data.py`, `lineage.py` sit in `2_model/multiaxis_eci/`; `fitspec.py`, `human_placement.py`, `regimes.py`, `frontier_gap.py`, `timelines.py`, `rotation.py`, `axes.py`, `stats.py` sit in `2_model/multiaxis_eci/analysis/`.

**Flagship spec.** `FitSpec(K=4, loading_prior="normal", lineage_prior=True, lineage_bm=True, floors=True, theta_skew=True, ceiling_noise=True, no_humans=True)` (fitspec.py:452). The defaults it inherits are effort_means=True, pooled_noise=True, beta_binomial=True, sigma_b_floor=0.005, time_prior=False, known_se=False and apply_exclusions=False (fitspec.py:85-152). This is the full exploration scope without humans: 5,487 obs / 887 models / 107 benchmarks (comparisons/README.md; mirt_gof_k4.json n_obs=5487).

---

## 0. Data scope and preprocessing

- **Source.** `0_input/all_scores_flat.csv` is synced from benchmark-data-pipeline (data.py:1-24). Retired benchmarks (FrontierMath v1, AlgoTune, MindCube) are removed upstream and never reach the repo (data.py:19-20; README "Data"). *CLAUDE.md is stale here: it describes a `retired_benchmarks.txt` applied in data.py.*
- **Curated exclusions** (`1_curated/excluded_benchmarks.txt`) are applied to the canonical K=1 index only. The list holds 8 "easy for humans" benchmarks plus 6 mainly visual ones; SpireClimb stays in (data.py:795-799; excluded_benchmarks.txt). The K-axis fits keep every benchmark (`include_all_benchmarks=not apply_exclusions`, fitspec.py:368-369).
- **Isolated-family drop** is on by default. Every release family whose observations sit on a single benchmark is dropped, after the benchmark filters. The ECI anchors are protected and humans are not affected (data.py:873-883; docstring at data.py:734-747). The opt-out `--keep-isolated-families` emits `_keepiso`.
- **Below-floor clipping.** K-axis fits raise below-chance scores to $c_b$ from a reviewed list (`1_curated/benchmark_score_clips.csv`, data.py:292-360). This happens only when floors are on (fitspec.py:384-391). The canonical index applies no clip because it has no floors (see §5).
- **Item counts.** `1_curated/benchmark_n_items.csv` lists 114 rows: 89 `items`, 25 `continuous` (data.py:267-289). README says 23 continuous benchmarks are in the fit.

---

## 1. Likelihood

For observation $n$ (model $m_n$, benchmark $b_n$):

$$\eta_n=\sum_{k=1}^{K}A_{b_nk}\,\theta_{m_nk}-D_{b_n},\qquad \mu_n=c_{b_n}+(d_{b_n}-c_{b_n})\,\sigma(\eta_n)$$
(mirt.py:1307-1310, 1311-1368)

**Beta-binomial branch (item-scored benchmarks, the default).** With $N_b$ items and $k_n=y_nN_b$, where $k_n$ may be fractional (mirt.py:1396-1416; `_beta_binomial_logp` mirt.py:568-589):
$$p_n\sim\mathrm{Beta}(\mu_n\phi_n,(1-\mu_n)\phi_n),\quad k_n\mid p_n\sim\mathrm{Bin}(N_{b_n},p_n)$$
$$\log P(k_n)=\log\binom{N}{k_n}+\log B(k_n+\alpha_n,N-k_n+\beta_n)-\log B(\alpha_n,\beta_n)$$
The pmf is written with log-gamma functions, so a non-integer $k$ is allowed. Its normalisation is then approximate, which affects absolute LOO values only (docs/model_math.md §2.3). **Why:** the term is a probability, bounded by 1, so it removes the unbounded density gain as $\sigma_b\to0$. On 2026-10-01 a Beta-only K=4 fit froze 6 of 8 chains in a noise spike on Chess Puzzles (mirt.py:1397-1403; fitspec.py:129-134).

**Beta branch (continuous benchmarks).** $y_n\sim\mathrm{Beta}(\mu_n\phi_n,(1-\mu_n)\phi_n)$. Exact 0/1 scores are clipped to $[\epsilon,1-\epsilon]$ with $\epsilon=$`CONTINUOUS_EPS`$=5\times10^{-3}$ under the Beta-binomial default, and with `ECI_EPS`$=10^{-3}$ in the Beta-only option (mirt.py:1415, 1418-1419; config.py:66-72).

**Precision and noise.** $\phi_b=\frac{1}{4\sigma_b^2}-1$, so $\mathrm{Var}=4\sigma_b^2\mu(1-\mu)$ in the Beta branch (mirt.py:1376). A `noise_domain` Potential returns $-\infty$ if any $\phi_n\le0$ (mirt.py:1390-1394).
Pooled noise (default in K-axis fits):
$$\sigma_b=\sigma_{\min}+\exp(\mu_s+\tau_sz_b),\ \ \mu_s\sim\mathcal N(\log0.05,0.5),\ \tau_s\sim\mathrm{HalfNormal}(0.5),\ z_b\sim\mathcal N(0,1),\ \sigma_{\min}=0.005$$
(mirt.py:1074-1085; config.py:100,106). **Why the floor:** MAP optimisation found noise-collapse spikes 3,000-5,000 nats above the sampled modes. The floor caps $\phi_b\le9999$ and sits below mode A's smallest $\sigma_b\approx0.013$ (config.py:101-106; model_math §5.2). The unpooled option uses $\sigma_b\sim\mathrm{LogNormal}(\log0.05,0.5)$ (mirt.py:1086-1090; config.py:90).

**Floors $c_b$.** Fixed per benchmark and never estimated. They come from the pipeline's `lower_bound` (data.py:171-192). Fixing $c_b$ avoids the weak identification of the estimated-$c$ 3PL and adds no parameters. Used with the clip-to-floor rule, an at-chance score reads as uninformative-low ability (mirt.py:857-867, 1369-1374).

**Ceilings $d_b$ (default since 2026-10-21).** $d_b=1-e_b$, with
$$e_b=0.25\,u_b,\quad u_b\sim\mathrm{Beta}(1.43,4.67)\ \text{truncated to}\ e_b\le1-\mathrm{best}_b$$
- Implementation: a `Uniform(0, top)` variable plus a Potential that carries the Beta log-density, without renormalisation (mirt.py:1341-1366; config.py:107-116).
- $\mathrm{best}_b$ is the maximum score in the data, or in the human baselines whether or not humans are fitted (data.py:213-226). Where someone scored 1.0, $d_b=1$.
- Benchmarks with a pipeline-recorded ceiling keep it fixed (data.py:195-210; mirt.py:1353-1362).
- **Why:** the gap is absolute (a share $e$ of wrong keys), so it does not scale with $c$. The prior is benchmark-forecasting's posterior population of $1-L$ (mean 0.059). The cap 0.25 keeps $d_b\ge0.75>c_b$.
- Identification: $d$ enters through $\log(d_b-c_b)$, which $D_b$ absorbs. It is pinned only by frontier scores that pile up below 1 (mirt.py:1322-1329).
- Evidence: robust mixture LOO +56 ± 12 nats on 4,967 reliable scores. Only 8 of 85 ceilings are data-informed (posterior SD <60% of prior SD), 70 stay prior-dominated (investigation/README.md).

**known-se** (not in the flagship): $1/N_{\text{tot}}=1/n_{\text{eff}}+4\sigma_b^2$, $\phi_n=N_{\text{tot}}-1$ (mirt.py:1377-1386).
**Censored Beta** (`--censor-bounds`, not default): mirt.py:552-565, 1420-1437.

---

## 2. Linear predictor, loading prior, identification

**Loadings (flagship: `normal`).**
$$A_{bk}=\tau_A\,\tilde A_{bk},\quad\tilde A_{bk}\sim\mathrm{HalfNormal}(1),\quad\tau_A\sim\mathrm{LogNormal}(\log0.5,0.5)$$
$\tau_A$ is one scalar shared by all axes (mirt.py:1100-1103, 1257-1260; config.py:92).

**Difficulty.**
$$D_b=\tau_{CD}D^z_b,\quad D^z_b\sim\mathcal N(0,1),\quad\tau_{CD}\sim\mathrm{LogNormal}(\log3,1)$$
(mirt.py:1091, 1280-1290; config.py:91).

**K=4** is set in FitSpec. A K other than 4 gets a `_k<K>` folder (fitspec.py:47, 218-219).

**Identification argument (as implemented):**
1. *Location.* $\theta$ enters only through $A\theta-D$. A shift $\theta_{\cdot k}+c_k$ is absorbed by $D_b+\sum_kA_{bk}c_k$.
   - Gaussian block: pinned exactly by the ZeroSumNormal over unstructured models plus chain founders (mirt.py:113-130, 219-222).
   - Skew-normal block (flagship): pinned softly by the zero-mean standardised prior, SD $1/\sqrt n$ on the common shift. It is not re-centred (mirt.py:91-93).
2. *Per-axis scale.* $(A_{\cdot k},\theta_{\cdot k})\to(cA_{\cdot k},\theta_{\cdot k}/c)$ is broken softly: unit-scale $\theta$ against one shared $\tau_A$, with no per-axis scale (mirt.py:667-675 explains this for `pt1`, which breaks it hard).
3. *Rotation.*
   - Gaussian block: spherical, so only $A\ge0$ (plus the per-axis lineage drift and spread) restricts rotations. Chains settle in different corners of the positive cone (GEOMETRY_REPORT.md "Pourquoi les modes existent").
   - Skew-normal block: independent non-Gaussian columns break the invariance, as in ICA (mirt.py:80-98; config.py:301-311).
4. *Sign.* None needed, since $A\ge0$ (rotation.py:74-75).
5. *Permutation (label switching).* Resolved after sampling, draw by draw.
   - `_collinearity_permutations` matches the cosines of unit loading columns to an iterated reference over all $K!$ permutations (rotation.py:17-48).
   - Axes are then ordered by mean loading energy $\sum_bA_{bk}^2$, because $\tau_A$ is shared (rotation.py:51-92; factors.py:56-79).
   - A chain subset is mapped onto the fit's `mirt_loadings.csv` by Hungarian matching on loading correlation, with a minimum of 0.8 (axes.py:26, 129-170).
6. *Modes.* `diagnose_chains.py --write-modes` groups chains when their matched loading correlation exceeds 0.8 (diagnose_chains.py:94) and no gap between sorted chain log-densities exceeds 20 nats (modes json "basis").
   - Deliverables read `selected_chains`, else the best-logp mode (fitspec.py:477-500).
   - A `secondary_chains` mode can get its own frame (fitspec.py:503-524).
7. *Convergence* is judged on identified quantities only: $\eta$ on 400 observations, $D$, $\sigma_b$, $\tau_{CD}$ (convergence.py:93-104).

**Axis naming.** Done by hand in `axis_names.json` from the top benchmarks and a "signature" per axis. A name is kept on refit only while its signature benchmarks still define the axis (axes.py; README "Current results").
- On the current ceiling flagship the names are marked **provisional** (axis_names.json "how").
- The selected mode A (8 of 10 chains) has the axes Fluid & Formal Sciences, Agentic & Programming, Cyber, and Visual & Commonsense. The knowledge MCQs moved onto axis 1. **README "Current results" still describes the no-ceiling fit's axes.**

---

## 3. Ability prior

**Exchangeable block (flagship: skew-normal).** It covers the unstructured models plus the chain founders' bases (mirt.py:212-233):
$$\theta_{mk}\sim\mathrm{SN}(\xi_k,\omega_k,\alpha_k),\ \alpha_k\sim\mathcal N(0,4^2),\ \delta_k=\tfrac{\alpha_k}{\sqrt{1+\alpha_k^2}},\ \omega_k=(1-2\delta_k^2/\pi)^{-1/2},\ \xi_k=-\omega_k\delta_k\sqrt{2/\pi}$$
Each column has mean 0 and SD 1; only the asymmetry is learned (mirt.py:80-98; config.py:311).

**Why the skew-normal block:**
1. Rotation identification, the ICA argument.
2. It matches the observed left tail of old and weak models.
3. Sampler-study evidence (GEOMETRY_REPORT.md):
   - 14 of 30 flagship chains reach the best region, against 4 of 30 (Fisher p≈0.01).
   - LOO is better by 60-100 nats on each of three seeds, and held-out predictive density is better on the bench.
   - Learned shapes: $\alpha\approx-8$ to $-10$ on three axes and ≈0 on one.
   - On the 2026-10-19 refit, two modes instead of 5-7.
   - Stated limit: it fixes orientation but not the multiple completions of the 6%-filled matrix.

The builder default stays Gaussian ZeroSumNormal, so the canonical K=1 index uses the Gaussian block (mirt.py:209-210; config.py:301-303).

**Lineage prior, Brownian with drift** (`--lineage-prior --lineage-bm`, mirt.py:345-405):
$$\psi_{\text{node}}=\psi_{\text{founder}}+\sum_{e\in\text{path}}\Delta_e,\quad\Delta_{ek}=\mu_k\,dt_e+s_k\sqrt{dt_e}\,z_{ek},\quad z_{ek}\sim\mathcal N(0,1)$$
$$\mu_k\sim\mathrm{HalfNormal}(1.0)\ \text{(logits/yr)},\quad s_k\sim\mathrm{HalfNormal}(2.0)\ \text{(logits/}\sqrt{\text{yr}})$$
- $dt_e$ is the release gap in years from the reviewed map dates, and must be >0 (lineage.py:266-273; config.py:278-279).
- One drift per axis, shared by all vendors. A per-vendor rate shrank to a single value at 5.9× the runtime (mirt.py:366-370).
- **Why BM rather than iid steps:** the prior no longer depends on how many intermediate releases the map lists, and a rate in logits per year is comparable across release cadences (mirt.py:357-365).
- Steps are soft: a release can regress.

**Chains are trees.** An explicit `parent` column hangs side lines off their branch point (lineage.py:188-227). A single-node chain survives if it has ≥2 (node, variant) groups (lineage.py:~136-174). The frozen map lists 78 chains, 271 nodes and 766 rows before data filtering (`lineage_map_fitted.csv`). A copy of the map is frozen beside each trace (fit.py:731-736).

**Effort variants.** $\theta_m=\psi_{\text{node}(m)}+o_{g(m)}$ (mirt.py:403-405). With effort means (default):
$$o_{gk}=\big(m_{\ell(g)}+\beta_{\text{tok}}\log_2(b_g/16\text{k})\big)u_k+\tau_o z_{gk}$$
$$m_\ell\sim\mathcal N(0,1),\ \beta_{\text{tok}}\sim\mathcal N(0,0.5),\ u=v/\lVert v\rVert,\ v\sim\mathcal N(0,I_K),\ \tau_o\sim\mathrm{HalfNormal}(0.25)$$
- The pretrained `base` checkpoint gets a free mean vector $\sim\mathcal N(0,I)$ (mirt.py:408-490; config.py:225, 253-270).
- The reference level ($m=0$) is "reasoning off", so $\psi$ means the release without test-time reasoning (config.py:240-252).
- The levels are unordered.
- The sign of $(m,u)$ is fixed after sampling (mirt.py:493-511).
- **Why rank one:** a free per-(level, axis) mean slides between correlated loading columns (mirt.py:430-434).

**Time prior.** Implemented (mirt.py:145-165, 195-202) but **not used** in the flagship (time_prior=False).

**Human ordered block.** Not in the flagship, which is fitted on models only. It is used in the canonical index and in the placement (§4-5). Roots $\sim\mathcal N(0,3^2)$ outside the model population. Increments $\delta\sim\mathrm{HalfNormal}(2)$, a hard order. Merged DAG via $\max(\text{parents})+\delta$ (mirt.py:316-342; config.py:152-158, 187, 195).

**Sampling** (fit.py:145-300, 685-845):
- nutpie with a low-rank modified mass matrix (fit.py:218-220), target_accept 0.95.
- Exploration defaults to 2,000 draws and 2,000 tune (fit.py:697-698). The README flagship used 20 chains × 2,500 draws, seeds 42 and 43 merged.
- The ceiling flagship has 10 chains (seed 42 only).
- Every 5th draw is saved (config.py:348).

---

## 4. Human tiers: post-hoc "2b" placement

Code: `4_diagnostics/5_place_humans.py`, `human_placement.py`.

- **Inputs frozen.** The plug-in medians $\hat A,\hat D,\hat\phi$ over the selected chains, on the aligned display frame. The ceiling fit's median $\hat d_b$ is used when present (5_place_humans.py:63-88; human_placement.py:215-217).
- **Which benchmarks count on axis $k$:** $\mathcal B_k=\{b:\arg\max_j\hat A_{bj}^2=k\}$ (`dominant_axis`, human_placement.py:65-67, 232). The content filter then applies, from `1_curated/human_axis_content.csv`: 90 judgments of (axis_title, benchmark, keep) (human_placement.py:56-62, 233-236).
- **Default-keep rule.** A benchmark on the axis with no judgment for that axis title is **kept** (`judged.get(b, True)`, human_placement.py:236). Unjudged ones are listed in `unjudged_benchmarks`. Judgments are keyed on the axis title, so a renamed axis inherits none unless `RULE_TITLES` maps it (5_place_humans.py:47-51). This matters because the ceiling flagship's provisional names include "Agentic Capabilities & Programming" and "Cyber Capabilities".
- **Other axes** are set by regression on axis $k$ over the model medians: $\theta_j\approx\alpha_j+\gamma_j\theta_k$ (human_placement.py:104-109). The effective one-dimensional predictor is:
$$\eta_b(\theta_k)=\Big(\hat A_{bk}+\sum_{j\ne k}\hat A_{bj}\gamma_j\Big)\theta_k+\sum_{j\ne k}\hat A_{bj}\alpha_j-\hat D_b$$
$$\mu=c_b+(\hat d_b-c_b)\sigma(\eta)$$
The fit's Beta-binomial/Beta likelihood follows at $\hat\phi_b$ (human_placement.py:141-161).
- **Prior.** `HUMAN_ORDER_MERGED` with roots $\mathcal N(0,3^2)$ and increments HalfNormal(2), one axis at a time. NUTS runs 4 chains of 1,000+1,000 draws (human_placement.py:151-163).
- **Uncertainty propagated:** only the tiers' posterior given the plug-in benchmark side. The following are **not** propagated:
  - uncertainty in $A,D,\phi,d$;
  - uncertainty in the regression profile;
  - mode uncertainty.
  The deliverables README states this caveat (deliverables/flagship_refit_20261021/DELIVERY.md).
- **Display rule.** A tier is drawn on axis $k$ if it has its own score below `CEILING_SCORE`=0.99, or sits between two such tiers in the order (human_placement.py:46, 95-99). This uses a fixed 0.99 rather than the fit's $\hat d_b$. The investigation notes that human scores near the estimated ceiling (e.g. LMCA) pull the Domain Expert up, and lists censoring as an open decision (investigation/README.md "À décider").
- **Check.** `consistency` compares observed and predicted scores for each tier and score, plus the date of the first model to reach each score (human_placement.py:171-196).
- **Crossings** read the placement draws against the forecast lines (human_placement.py:321-330).
- **Why post hoc.** In joint fits the tiers compensate across axes: about −5 SD on one axis and +8 on another (Gaussian), and −6.4 SD on science with skew + 2026-10-19 data. No tier prior, fifth "human" axis or softmin link fixed it (human_placement.py:1-21; model_math §5.4, §8; `mirt_lineageprior_lineagebm_noceiling_humanmerge/README.md`).

---

## 5. Canonical K=1 ECI-H

Run with `fit.py --preset canonical --human-merge`; every analysis reads `canonical_humanmerge/` (fit.py:1-11, 308-470).

- **Scope.** Curated exclusions are applied and humans are inside as test-takers. Isolated families are dropped. Result: 4,410 obs on 93 benchmarks (README; canonical_humanmerge/gof.json).
- **Model.** K=1 with the `pt1` loading prior: $A_b=\exp(\sigma_A z_b)$, $z\sim\mathrm{ZeroSumNormal}$ over benchmarks so that $\prod_bA_b=1$, and $\sigma_A\sim\mathrm{LogNormal}(\log0.5,0.5)$ (mirt.py:1104-1125, 1249-1256). This is Epoch's sum-to-zero log-α gauge. It is chosen for fewer divergences at unchanged abilities (fit.py:449-451).
- **Abilities.** Gaussian ZeroSumNormal block. Release nodes have no links between releases (`lineage_links=False`): each release node is its own ZSN draw, with variant offsets and effort means (mirt.py:514-527; fit.py:459-466).
- **Human order.** `HUMAN_ORDER_MERGED` with roots $\mathcal N(0,3^2)$ and increments HalfNormal(2) (fit.py:334-335, 380-385).
- **Likelihood.** Beta-binomial default. **No floors, no ceilings, no pooled noise, no $\sigma_b$ floor** (`sample_mirt` defaults, fit.py:145-153, 461-466; README "Method": "which the canonical index leaves out").
- **Sampling.** 8 chains × 2,500 draws, 2,000 tune, target 0.95, seed 42 (config.py:353-362).
- **Anchors.** Per draw, $\mathrm{ECI\text{-}H}=a+b\theta$, with $b=20/(\theta_{\text{GPT-5 med}}-\theta_{\text{C3.5S}})$ and $a=130-b\,\theta_{\text{C3.5S}}$. The anchors are `claude-3-5-sonnet-20241022`=130 and `gpt-5-2025-08-07_medium`=150 (stats.py:66-89; config.py:369-370).
- **Sensitivity fits that exist:**
  - `canonical/` (no human order) on 20261019.
  - Access-class fits: `canonical_{public,semi_private,private}_humanmerge/`.
  - README Limitations: without the order, some chains find a second mode (about e^-14 mass).

---

## 6. Forecast (K-axis, per axis)

`regimes.py`; `config.FORECAST_KW = dict(fit_basis="regimes", top_k=2, hdi_prob=0.8)` (config.py:438).

1. **Candidates** (`candidate_mask`): dated, not human, with axis coverage ≥1. Coverage is $\sum_{b\in\text{obs}(m)}\text{share}_{bk}$, where $\text{share}_{bk}=\hat A_{bk}^2/\sum_j\hat A_{bj}^2$ (timelines.py:31-73; config.py:407).
2. **Reduction.** One effort per family, the best posterior median on the axis (`family_best`). Then one release per organisation per day (`one_per_org_day`) (regimes.py:168-186).
3. **Regimes.** Each run is classed reasoning or non-reasoning by `run_mode` and the family evidence. The bare runs of hybrid releases sit out of both fits (regimes.py:69-136; config.py:444-485; reviewed in `1_curated/model_regimes.csv`).
4. **Frontier points (top-2):** a release whose median is ≥ the 2nd-largest median released so far (`frontier_topk`, regimes.py:189-196).
   - Reasoning set: top-2 among reasoning families.
   - Non-reasoning set: the non-reasoning part of the top-2 frontier over all families (regimes.py:276-281).
   - At least 3 reasoning points are required.
5. **Line fit.** Bayesian linear regression on the posterior **medians** (`post_stats` returns the median in the `mean` column, stats.py:27-39). Noise variance is $\mathrm{sd}_i^2+s^2$, with $s$ marginalised over the grid {0, .05, .1, .2, .35, .5}. Priors are $\mathcal N(0,10^2)$ on intercept and slope. The solution is analytic and sampled 4,000 times (regimes.py:216-249).
   - Point-to-point posterior correlation is **not** propagated, a stated choice (regimes.py:28-30).
6. **Projection.** The reasoning line runs from the first reasoning release to 2030-01-01. The non-reasoning line covers only its own span. Bands are 80% central (regimes.py:290-304).
7. **Crossover with a tier.** Per sample, the tier draw is paired with line samples. The crossing date is the earliest of two events (regimes.py:313-386):
   - the reasoning line reaches the tier at or after the switch;
   - the non-reasoning line reaches it at or before its last release.

   Intervals are central at 50% and 80% (forecast.py:475-490). Status is "no_crossing" if <50% of samples define a crossing.
8. `fit_start` (2024-10-01) does not apply to the regime basis.

---

## 7. Frontier gap (open/closed, US/CN) on K=1 ECI-H

Code: `frontier_gap.py`, `4_diagnostics/1_frontier_gap.py`.

- **Groups.** Leader and follower are closed/open (`model_openness.csv`) or US/CN (pipeline country) (frontier_gap.py:158, 174-183).
- **Scopes.** All benchmarks, public, semi-private and private (canonical class fits).
- **Linking.** A class fit is linked onto all-benchmarks ECI-H per draw by mean-sigma linking. The links are the leader's OpenAI, Anthropic and Google candidates with ≥4 scores in the class (frontier_gap.py:155-156, 222-255).
- **Candidates** need ≥2 scores in scope (`--min-obs`). Records also need `RECORD_MIN_OBS`=2 (frontier_gap.py:129).
- **Frontier points.** `FRONTIER_K=2`: the running top-2 of posterior medians, after family-best and one-per-org-day reduction (frontier_gap.py:138, 268-310). *The module docstring still says k=1 (frontier_gap.py:15-17), which is stale.*
- **Per-draw envelope.** The running k-th max over the leader's candidates. Candidates with posterior SD >6 ECI-H are excluded (`LEADER_SD_CAP`, frontier_gap.py:119, 330-336).
- **Lag at a frontier point** (Ihle's backward-looking lag). Per draw $s$:
$$\mathrm{lag}_s=12\,(t_{\text{rec}}-t^{*}_s),\quad t^{*}_s=\min\{t_j: E^{\text{lead}}_{s,(k)}(t_j)\ge E_{s,\text{rec}}\}$$
Here $E^{\text{lead}}_{s,(k)}$ is the leader's running k-th best at its candidate dates (frontier_gap.py:351-466).
  - If the level was already exceeded on the leader's first measured day, the crossing is borrowed from the all-benchmarks fit (class scopes). Otherwise it is dated back at the leader's early rate over 1.5 years.
  - Points dated back in >2/3 of draws are dropped (frontier_gap.py:344-348).
  - In class scopes, follower points more than 5 ECI-H below the follower's all-benchmarks frontier are not counted (frontier_gap.py:113, 198-219).
  - Dots show the posterior mean of the lag; bars show 50%.
- **Lag on an average day.** `lag_function` gives the same lag month by month from the per-draw frontiers. Its trailing 12-month mean is a table row (frontier_gap.py:121, 478-531).
- **Line gap.** A weighted line per group through its frontier points since 2024-10-01. ECI-H grid $s\in\{0,..,15\}$, prior SD 200, SD floor 0.5. Lag = 12·gap/leader slope; the forward-looking lag uses the follower slope (frontier_gap.py:96-106, 533-549).
- **Crossings** of each group line with the human tiers (frontier_gap.py:576).

---

## 8. Model-comparison tooling

- **PSIS-LOO / LOO-PIT.** Observations with Pareto $k>0.7$ are left out of calibration summaries (config.py:524-528).
  - Flagship: 856 of 5,487 have $k>0.7$ (mirt_gof_k4.json `pit_n_pareto_k_high`).
  - Canonical: 163 of 4,410.
- **Integrated LOO** (`4_diagnostics/integrated_loo.py:1-46`). Follows Vehtari et al. 2016. Each score's latent block ($\psi$ of its release, or $\theta$ of an unchained model) is integrated out:
  - per draw, a Laplace fit of the block's conditional posterior;
  - the closed-form cavity Gaussian;
  - a 1-D Gauss-Hermite integral over $\eta_i$, with 40 nodes.
  The weights are then $1/p(y_i\mid y_{-i},\phi_s)$, Pareto-smoothed. It covers the skew-normal prior (exact gradient and curvature in the Laplace step) and the ceilings. Human tiers are not integrated. Each chain is scored alone, and modes are compared at equal draws. Validated against a 200,000-sample importance-sampling integral (+0.001 nats mean).
- **Robust LOO** (`4_diagnostics/robust_loo.py:1-80`). Silva & Zanella (2024) mixture importance sampling:
$$q(\theta)\propto p(\theta\mid y)\sum_jc_j/p(y_j\mid\theta)$$
  - The weights $c_j$ are first-pass PSIS estimates of $p(y_j\mid y_{-j})$. This weighting is the authors' own extension; `--plain` runs the paper's version.
  - The rebuilt model gets a `pm.Potential` and is restarted inside the mode under study.
  - It reports the per-observation ESS.
- **Pareto-k handling.** Comparisons are reported on all scores and on the "reliable" subset with $k<0.7$ in every compared fit (integrated_loo json; investigation tables).

---

## 9. Decisions needing justification or ablation

Notes:
- "Exists" = a comparison artefact on disk. `5_outputs/data20261019/` holds only three K=4 fits: ceiling (flagship), `_noceiling`, `_noceiling_humanmerge`. Traces of those are **not** in 5_outputs (no .nc found).
- Older traces sit in the stale root `results/` folder (pre-pipeline data, humans inside, Gaussian θ, no ceilings).
- The 2026-08 assumption ladder is in `6_writeups/blogpost/ladder/ladder_notes.md`.

| # | Decision | Status |
|---|---|---|
| 1 | **K=4** vs 3/5 | Pre-pipeline only. Ladder on 5,004 obs with humans: K=1, K=3+both priors (2 modes), K=4+both (1 mode). `results/` holds K=2/3/5 `since20230201` traces and the `pre_pipeline/comparisons` K=2 demo. **Needs a run** on the current spec (skew, ceilings, models-only) for K∈{3,5}, with integrated LOO. |
| 2 | Skew-normal vs Gaussian θ | Exists as a sampler study (`5_outputs/sampler_study/GEOMETRY_REPORT.md`, 3 paired seeds, pre-ceiling data). No current-data `_normaltheta` folder. **Needs** one paired refit with ceilings. |
| 3 | Ceilings on/off | **Exists**: `mirt_lineageprior_lineagebm` vs `_noceiling` (data20261019), robust and integrated LOO (`investigation/`). Caveat: one seed, 10 chains; the mode structure flips between the two fits. |
| 4 | Ceiling lower bound from human baselines (vs models only) | Flagged as an open decision (investigation/README.md). **Needs a run.** |
| 5 | Humans inside (joint) vs placed post hoc | **Exists**: `_noceiling_humanmerge` vs `_noceiling` + placement (README there, `compare_tiers.py`). Not yet with ceilings. |
| 6 | Placement choices: dominant-axis rule, content filter, default-keep, regression profile, plug-in medians, 0.99 display cutoff | No ablation. **Needs** a sensitivity check (e.g. share threshold, full-posterior propagation over draws of $A,D,\phi,d$, censoring ceiling scores). Secondary-mode placement exists (`secondary_mode/human_placement.json`). |
| 7 | Lineage prior on/off; BM vs iid steps | Pre-pipeline ladder only (K=4 no priors: 2 modes, 778 divergences; human-only: 2 modes). README cites 2026-09-08 "8 chains on 8 solutions" without lineage (deleted outputs). **Needs a run** on the models-only spec. No BM vs iid comparison found. |
| 8 | Effort means (rank-1) vs mean-zero offsets | Prior-tail audit cited in config.py:231-238. No folder. **Needs a run** (`--no-effort-means`). |
| 9 | Fixed floors on/off | Earlier evidence in memory only (r̂ 1.18→1.03, pre-pipeline). No current folder. **Needs a run** (`--no-floors`). |
| 10 | Beta-binomial vs Beta | Anecdote only (2026-10-01 noise spike, fitspec.py:129-134). **Needs a run** (`--beta-likelihood`), or a justification by construction. |
| 11 | $\sigma_b$ floor 0.005 and pooled noise | MAP study (model_math §5.2). Pooled vs unpooled: the CLAUDE.md "validated" note is pre-pipeline. **Needs a run** (`--no-pooled-noise`); the floor value is a sensitivity point. |
| 12 | Isolated-family drop | Rationale only (data.py:734-747). **Needs a run** (`--keep-isolated-families`). |
| 13 | Exclusion list (canonical only) and full scope for K-axis | No `canonical_allbenchmarks` on 20261019. **Needs a run**. Also justify the asymmetric scope between the canonical index and the K-axis fits. |
| 14 | Canonical index lacks floors, ceilings, pooled noise and skew that the K-axis fit has | Undocumented asymmetry. **Needs** justification or a canonical refit with the K-axis likelihood. |
| 15 | Nonnegative `normal` loadings vs `signed`, `pt1`, `bifactor` | Pre-pipeline only (`results/`, CLAUDE.md pt1 vs normal at K=1). **Needs a run** at K=4 if claimed. |
| 16 | Compensatory vs softmin/noncomp | Tested 2026-10-07 (model_math §5.4/§8: noncomp −62 ± 27 nats; semicomp ties). Folders deleted. Cite or re-run. |
| 17 | Mode selection (best logp vs LOO) | Integrated LOO compare exists (`investigation/integrated_loo_compare.csv`): with ceilings, A vs B 3.2 ± 8.3 nats on reliable scores, a tie. Report mode-aware results. |
| 18 | **Convergence of the current flagship** | The 5-chain × 12k refit (deliverables/flagship_refit_20261021/DELIVERY.md) **failed**: η r̂ 1.10, min bulk ESS ≈30, 1,612 / 60,000 divergences, 3 loading solutions. Comparisons table: η r̂ 1.165 on the 20-chain no-ceiling card. **Blocking for a paper.** |
| 19 | Forecast design: medians not draws, top-2, reasoning split, linear trend, 2030 horizon, regime classifier | No ablation folder. `fit_basis` alternatives (envelope, regression) exist in code (forecast.py). **Needs** a sensitivity table. |
| 20 | Frontier gap: FRONTIER_K=2 vs 1, LEADER_SD_CAP=6, RECORD_MIN_OBS=2, linking set, 5-point off-frontier tolerance | Investigation exists (`5_outputs/us_cn_frontier/investigation_2026-10-20/`; docstrings report k=1 vs 2 and min-obs 3 results). Partly covered. |
| 21 | ECI-H anchors | Invariant by construction (stats.py:66-74). No ablation needed. |
| 22 | Human order (merged DAG) in the canonical index | **Exists**: `canonical/` vs `canonical_humanmerge/` on 20261019. |

---

## Docs and code disagreements found

- **CLAUDE.md** describes the old layout. It is wrong on all of these points:
  - the retired list;
  - canonical draws (says 10,000; now 2,500);
  - the ceiling prior (says relative Beta(1,20); now absolute 0.25·Beta(1.43,4.67), default ON);
  - the flagship being `--human-merge` (now models-only);
  - the θ prior (says ZSN; now skew-normal);
  - scope counts;
  - the BM median gap (says 0.238; config says 0.285).
- **docs/model_math.md**:
  - §5.4 lineage writes $\psi_{\text{founder}}\sim\mathcal N(0,1)$, but founders are in the skew/ZSN block.
  - §8 says 56 human scores; the README says 57.
  - §8 "a score at the ceiling" is implemented as a fixed 0.99, not $\hat d_b$.
- **docs/plots.md** says the forecast is "gated on human tiers being in the fit". Placement replaced that.
- **README "Current results"** describes the no-ceiling fit's axes and modes, not the ceiling flagship's.
- **config.py**:
  - :225 says "per-axis tau_o[k]", but tau_o is one scalar in code (mirt.py:375-381, 450).
  - :290-292 describes the t-block as a scale mixture, but code uses the direct density (mirt.py:54-59).
- **frontier_gap.py** docstring says top-1 records, but the default is `FRONTIER_K=2`.
