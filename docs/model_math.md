# ECI Beta-MIRT — the math

The whole framework is one likelihood (a reparametrized Beta) sitting on top of
one linear predictor $\eta$. Three cases differ **only** in how $\eta$ maps to
the mean $\mu$:

- **1D** — the canonical index, $K=1$.
- **KD** — the $K$-axis compensatory MIRT.
- **KD 3PL** — KD with a fixed per-benchmark chance floor.

Everything else (the Beta likelihood, the priors on difficulty, noise, and
abilities) is shared. This document defines every symbol. By default the
item-scored benchmarks enter through the Beta-binomial of §2.3 and the
continuous ones through the Beta below; `--beta-likelihood` puts every score
on the Beta (with the clipping of §2.2).

---

## 1. Index sets and observed data

| Symbol | Meaning |
|---|---|
| $m \in \{1,\dots,M\}$ | test-taker (a model, or a human baseline). $M$ = number of models. |
| $b \in \{1,\dots,B\}$ | benchmark. $B$ = number of benchmarks. |
| $n \in \{1,\dots,N\}$ | one observation = one (model, benchmark) score. $N$ = number of rows. |
| $m_n$ | the model taking observation $n$. |
| $b_n$ | the benchmark of observation $n$. |
| $k \in \{1,\dots,K\}$ | latent ability axis. $K=1$ is the canonical index. |
| $y_n \in [0,1]$ | the observed normalised score of observation $n$. |

**Scores are on the raw $[0,1]$ scale**, "normalised" only in the unit sense
(percent → proportion). They are **not** chance-corrected: a 4-choice
multiple-choice benchmark still clusters near $0.25$.

---

## 2. The shared likelihood — reparametrized Beta

Every version uses the same observation model. The Beta is written in a
**mean / precision** parametrization instead of the usual $(\alpha,\beta)$:

$$
y_n \;\sim\; \mathrm{Beta}\big(\mu_n\,\phi_{b_n},\;\;(1-\mu_n)\,\phi_{b_n}\big)
$$

so that

$$
\mathbb{E}[y_n] = \mu_n,
\qquad
\mathrm{Var}[y_n] = \frac{\mu_n(1-\mu_n)}{\phi_{b_n}+1}.
$$

| Symbol | Meaning |
|---|---|
| $\mu_n \in (0,1)$ | the expected score for observation $n$. This is where 1D / KD / 3PL differ (Section 4). |
| $\phi_b > 0$ | per-benchmark **precision** — how tightly scores concentrate around $\mu$. Large $\phi$ = low noise. |
| $\alpha_n = \mu_n\phi_{b_n}$ | first Beta shape parameter (code: `a`). |
| $\beta_n = (1-\mu_n)\phi_{b_n}$ | second Beta shape parameter (code: `b`). |

### 2.1 Precision from a noise SD

Rather than give $\phi_b$ a prior directly, we parametrize a per-benchmark
noise standard deviation $\sigma_b$ and derive $\phi_b$:

$$
\boxed{\;\phi_b = \dfrac{1}{4\sigma_b^{2}} - 1\;}
$$

This makes $\sigma_b$ the score SD at the **maximum-variance point** $\mu=\tfrac12$:
at $\mu=\tfrac12$ the Beta variance equals $\sigma_b^2$. In general the variance
is $4\sigma_b^2\,\mu(1-\mu)$.

| Symbol | Meaning |
|---|---|
| $\sigma_b > 0$ | per-benchmark noise SD (the SD of scores at $\mu=\tfrac12$). |

**Validity constraint:** $\phi_b>0$ requires $\sigma_b < \tfrac12$; the prior
(Section 5) keeps $\sigma_b$ well below that, and the model enforces it (a `noise_domain`
term at $-\infty$ outside it, 2026-10-09). **Floor:** every fit since 2026-10-13 adds
$\sigma_{\min} = 0.005$ to $\sigma_b$ (§5.2), so $\phi_b \le 9999$.

### 2.2 Boundary clipping (Beta-scored observations)

The Beta has open support $(0,1)$, but a handful of scores are exactly $0$ or
$1$. Those (and only those) are clipped inward before being passed as the
observed values:

$$
\tilde y_n = \min\!\big(\max(y_n,\ \varepsilon),\ 1-\varepsilon\big),
\qquad \varepsilon = 10^{-3}.
$$

Interior scores are untouched. $\varepsilon$ is `ECI_EPS` in the code.

### 2.3 Beta-binomial on item-scored benchmarks (the default)

Since 2026-10-01 the default likelihood treats a score that is a share of a benchmark's $N_b$ items as $k_n = y_n N_b$ successes:

$$
p_n \sim \mathrm{Beta}\big(\mu_n\phi_{b_n},\,(1-\mu_n)\phi_{b_n}\big), \qquad k_n \mid p_n \sim \mathrm{Binomial}(N_{b_n}, p_n),
$$

$$
\log P(k_n) = \log\binom{N}{k_n} + \log B(k_n + \alpha_n,\, N - k_n + \beta_n) - \log B(\alpha_n, \beta_n).
$$

The variance is $\mu(1-\mu)\,\big[1 + (N-1)/(\phi+1)\big]/N$, so even with no excess noise ($\phi\to\infty$) a score keeps the binomial spread $\mu(1-\mu)/N$ of item sampling. Each term is a probability, at most 1: a benchmark fitted exactly earns nothing more, where the Beta density grows without bound as $\sigma_b \to 0$. That unbounded gain is what froze six of eight chains of the 2026-10-01 K=4 fit, with one axis on Chess Puzzles (57 of 100 scores at the floor) and its $\sigma_b$ at $4\times10^{-4}$.

$k_n$ may be fractional (a mean over several runs of the same items): the log-gamma form extends the pmf smoothly to real $k$, and its normalization is then approximate, which only touches absolute log-likelihoods (LOO), not the posterior. Scores of 0 and 1 enter as $k=0$ and $k=N$, with no clipping. $N_b$ and the `items` / `continuous` split come from `1_curated/benchmark_n_items.csv` (`data.load_binomial_n`). The 23 continuous benchmarks keep the Beta of Section 2, their exact zeros clipped to `CONTINUOUS_EPS` $=5\times10^{-3}$. `--beta-likelihood` restores the Beta on every score.

---

## 3. The shared linear predictor $\eta$

All three versions build the same **linear predictor** for observation $n$:

$$
\boxed{\;\eta_n \;=\; \sum_{k=1}^{K} A_{b_n,\,k}\,\theta_{m_n,\,k}\;-\;D_{b_n}\;}
$$

| Symbol | Shape | Meaning |
|---|---|---|
| $\theta_{m,k}$ | $M\times K$ | latent **ability** of model $m$ on axis $k$. Higher = more capable. |
| $A_{b,k}$ | $B\times K$ | **loading** (discrimination) of benchmark $b$ on axis $k$: how strongly benchmark $b$ responds to ability on axis $k$. |
| $D_b$ | $B$ | **difficulty** of benchmark $b$ (in logits). Larger $D_b$ shifts the whole score curve down. |
| $\eta_n$ | $N$ | the logit-scale "signal" for observation $n$. |

This is **compensatory**: abilities enter as a weighted *sum*, so strength on
one axis can offset weakness on another. (The non-compensatory / conjunctive
variants replace this sum with a product — out of scope here.)

---

## 4. The three mean structures

The only difference between the three versions is the link from $\eta_n$ to
$\mu_n$.

Let $\sigma(x) = \dfrac{1}{1+e^{-x}}$ be the logistic sigmoid.

### 4.1 1D (canonical index, $K=1$)

The sum collapses to a single term:

$$
\eta_n = A_{b_n}\,\theta_{m_n} - D_{b_n},
\qquad
\mu_n = \sigma(\eta_n).
$$

Here $\theta_{m}\in\mathbb{R}$ is the model's single overall capability and
$A_b>0$ is the benchmark's discrimination. This is the standard **2PL** item
model with a Beta response. (The retired standalone 1D model wrote the same
mean as $\alpha_b(C_m - D_b)$; that is an affine reparametrization of
$A_b\theta_m - D_b$, absorbed by the ECI-H anchoring in Section 6.)

### 4.2 KD (K-axis compensatory MIRT)

The general case, $K\ge 1$:

$$
\eta_n = \sum_{k=1}^{K} A_{b_n,k}\,\theta_{m_n,k} - D_{b_n},
\qquad
\mu_n = \sigma(\eta_n).
$$

Setting $K=1$ recovers 1D exactly. There is **no chance floor**: a
random guesser is pushed toward $\mu\to 0$, not toward the benchmark's
guessing rate.

### 4.3 KD 3PL (fixed chance floor)

Same $\eta_n$ as KD, but the mean is lifted so a random guesser lands at the
benchmark's **chance rate** $c_b$ instead of $0$:

$$
\boxed{\;\mu_n = c_{b_n} + \big(1 - c_{b_n}\big)\,\sigma(\eta_n)\;}
$$

| Symbol | Meaning |
|---|---|
| $c_b \in [0,1)$ | **fixed** lower asymptote (chance floor) of benchmark $b$ — e.g. $0.25$ for 4-choice MC. Read from the view's `lower_bound` column (the pipeline's `benchmarks.csv`). |

As $\eta_n\to-\infty$, $\mu_n\to c_b$; as $\eta_n\to+\infty$, $\mu_n\to 1$. This
is the classic **3PL** lower-asymptote guessing term, with one crucial
restriction:

- $c_b$ is **fixed** from the known guessing rate, **never estimated**. This
  sidesteps the notorious weak identifiability of the free-$c$ 3PL and adds
  **no new parameters**.
- Setting $c_b = 0$ for all $b$ makes this byte-identical to KD. So KD is the
  special case $c\equiv 0$.

It is intended together with **clip-to-floor** observed scores (each $y_n$
raised up to at least $c_{b_n}$): then an at-chance score reads as
"uninformative-low ability" rather than as a hard demand for very negative
$\eta$.

---

## 5. Priors

All priors are written in **non-centered** form: NUTS samples a unit-scale
auxiliary variable (suffix `_z`), and a deterministic node multiplies it by the
relevant scale $\tau$. This breaks the funnel between each scale and the vector
it controls.

### 5.1 Difficulty $D_b$

$$
\tau_{CD} \sim \mathrm{LogNormal}(\log 3,\;1),
\qquad
D^{z}_b \sim \mathcal{N}(0,1),
\qquad
D_b = D^{z}_b \cdot \tau_{CD}.
$$

| Symbol | Meaning |
|---|---|
| $\tau_{CD}$ | one shared scale for all difficulties. LogNormal keeps most $D_b$ within $\approx[-6,6]$ logits. |
| $D^{z}_b$ | unit-normal auxiliary, one per benchmark. |

The **location** of the difficulty scale is not pinned here; it is pinned on
the ability side by the zero-sum ability prior (Section 5.4). (Optional
alternative: `pin_benchmark` fixes one benchmark's $D_b\equiv 0$ as "sea level"
— used when substantive ability priors let a subgroup of abilities float.)

### 5.2 Benchmark noise $\sigma_b$

$$
\sigma_b \sim \mathrm{LogNormal}(\log 0.05,\;0.5),
\qquad
\phi_b = \frac{1}{4\sigma_b^{2}} - 1.
$$

Median $\sigma_b = 0.05$; 90% prior interval $\approx[0.022,\,0.114]$. LogNormal
forces $\sigma_b>0$; the range keeps $\sigma_b\ll\tfrac12$ so $\phi_b>0$.

That independent LogNormal is what `--preset canonical` fits. **Exploration
fits (the flagship included) default to the hierarchical (pooled) variant**,
where the population location and spread of the log noise are learned, so a
thin benchmark's $\sigma_b$ shrinks toward the shared median instead of
keeping a free scale the panel cannot pin:

$$
\sigma_b = \sigma_{\min} + \exp(\mu_s + \tau_s\, z_b),
\qquad
\mu_s \sim \mathcal{N}(\log 0.05,\;0.5),\quad
\tau_s \sim \mathrm{HalfNormal}(0.5),\quad
z_b \sim \mathcal{N}(0,1),
$$

(non-centered; constants `PRIOR_SIGMA_B_POOLED`). `--no-pooled-noise` opts back
into the independent LogNormal above as the sensitivity run (with the same floor added).

**Floor $\sigma_{\min} = 0.005$** (`config.SIGMA_B_FLOOR`, carried in each trace's spec as
`sigma_b_floor`; traces fitted before 2026-10-13 carry none and are rebuilt without it). Without
it the whole population can slide down together: $\mu_s \approx -5$, $\tau_s \approx 3$ cost
about 26 nats of prior, while the Beta rows' density grows without bound as $\phi_b\to\infty$.
Multi-start MAP optimisation of the flagship (sampler study, 2026-10-12) found such
noise-collapse spikes, $\sigma_b \le 10^{-3}$ on many benchmarks, 3,000 to 5,000 nats of log
posterior above the modes NUTS samples; NUTS from jittered starts never reaches them, NUTS
started from those optima stays in them. The floor sits below mode A's smallest $\sigma_b$
(about 0.013) and caps $\phi_b$ at 9999.

### 5.3 Loadings $A_{b,k}$

There are four loading priors. All expose one derived scale vector $\tau_A$
(length $K$) for the downstream readers, though `pt1` fixes it at 1.

**(a) `"normal"` — exploration default + confirmatory.** Non-negative
loadings, one learned shared scale (the flagship K=4 fit and the confirmatory
Q-matrix fits use this):

$$
\tau_A^{\text{scalar}} \sim \mathrm{LogNormal}(\log 0.5,\;0.5),
\qquad \tau_{A,k} = \tau_A^{\text{scalar}}\ \ \forall k,
$$
$$
A^{z}_{b,k} \sim \mathrm{HalfNormal}(1),
\qquad
A_{b,k} = A^{z}_{b,k}\;\cdot\;\text{mask}_{b,k}\;\cdot\;\tau_{A,k}.
$$

| Symbol | Meaning |
|---|---|
| $\tau_A^{\text{scalar}}$ | single shared loading scale; typical loadings $\approx 0.5$. |
| $A^{z}_{b,k}$ | non-negative (HalfNormal) auxiliary. |
| $\text{mask}_{b,k}\in\{0,1\}$ | optional **Q-matrix / anchor** mask: $1$ if benchmark $b$ is allowed to load on axis $k$, else $0$. Absent anchors, all ones. |

**(b) `"pt1"` — the canonical 1D prior.** `--preset canonical` fits this one.
Non-negative loadings under a **product-to-one** constraint per axis: the log
loadings of each axis are a zero-sum normal over benchmarks, so each column's
geometric mean is exactly 1 (the sum-to-zero log-$\alpha$ gauge of Epoch's
public ECI, generalized to $K$ axes):

$$
\sigma_A \sim \mathrm{LogNormal}(\log 0.5,\;0.5),
\qquad
\log A^{z}_{\cdot,k} \sim \mathrm{ZeroSumNormal}\ \text{(over benchmarks)},
\qquad
A_{b,k} = \exp\!\big(\sigma_A\, \log A^{z}_{b,k}\big).
$$

$\eta$ has $K$ exact multiplicative degeneracies ($A_{\cdot,k} \to c\,A_{\cdot,k}$,
$\theta_{\cdot,k} \to \theta_{\cdot,k}/c$); the gauge breaks all $K$ hard, so
there is **no free $\tau_A$** here ($\tau_{A,k} \equiv 1$ — a free scale would
reinstate one ridge per axis). Costs: loadings are lognormal, so no benchmark
can sit at exactly zero loading, and hard anchors are unavailable
($\exp$ is never 0).

**(c) `"signed"` — exploration alternative.** iid signed cells × one shared scale:

$$
\tau_A^{\text{scalar}} \sim \mathrm{LogNormal}(\log 0.5,\;0.5),
\qquad
A^{z}_{b,k} \sim \mathcal{N}(0,1),
\qquad
A_{b,k} = A^{z}_{b,k}\cdot \tau_{A,k}.
$$

With signed cells and a single spherical scale, prior and likelihood are
**exactly invariant** under rotating $(A,\theta)$ together; the axes are
identified *after* sampling, one draw at a time. This is the only prior that can
represent **contrast** axes (some benchmarks $+$, others $-$). Incompatible with
the anchor mask (hard zeros would break the rotation symmetry).

**(d) `"bifactor"`.** Axis 1 is a dense **general** column (the `"normal"`
block on one column: HalfNormal cells × its own LogNormal scale); axes
$2..K$ are **specifics** whose non-negative cells carry regularized-horseshoe
local scales under one shared global scale. A specific cell is either squeezed
to $\approx 0$ or escapes roughly unshrunk at a near-constant prior cost, so
among likelihood-equivalent configurations the posterior prefers "common part
in $g$, few large loadings per specific" — the bi-quartimin simplicity
criterion moved into the prior. Axis identity is structural ($g$ is column 1),
so nothing is rotated downstream; a specific the data does not need collapses
to an empty column. Mutually exclusive with anchors; needs $K \ge 2$.

### 5.4 Abilities $\theta_{m,k}$

**Default (canonical + most fits):** a zero-sum normal per axis, over models:

$$
\theta_{\cdot,k} \sim \mathrm{ZeroSumNormal}(\sigma=1)
\quad\text{for each axis }k.
$$

Unit scale fixes the $A/\theta$ scale trade-off; the sum-to-zero constraint
$\sum_m \theta_{m,k}=0$ **pins the location** of each axis inside the sampling
geometry (no flat translation direction, no post-hoc recentering needed). This
is the only thing anchoring the overall location, which is why the difficulty
prior can leave $D$'s location free.

**Skew-normal block (the K-axis fits' default since 2026-10-20; `--normal-theta` restores the Gaussian one; an option `--theta-skew` from 2026-10-16).** The same exchangeable block
(unstructured models plus the chain founders) becomes independent skew-normal per axis,
standardised to mean 0 and sd 1, with a learned shape per axis:

$$
\theta_{m,k} \sim \mathrm{SkewNormal}(\xi_k, \omega_k, \alpha_k),
\qquad
\alpha_k \sim \mathcal{N}(0, 4^2),
\qquad
\delta_k = \frac{\alpha_k}{\sqrt{1+\alpha_k^2}},\ \
\omega_k = \Big(1 - \tfrac{2\delta_k^2}{\pi}\Big)^{-1/2},\ \
\xi_k = -\omega_k \delta_k \sqrt{2/\pi}.
$$

The Gaussian block is spherical: a rotation of $(A, \theta)$ leaves its prior, and the
likelihood, unchanged, and only the non-negative loadings (with the lineage's per-axis drift and
spread) fix the axes' orientation, which leaves chains stranded in different orientations.
Independent non-Gaussian columns break that invariance (the ICA argument), and the skew-normal
matches the abilities' actual shape, a left tail of old and weak models below a ceiling. The
location is pinned softly by the zero mean (sd $1/\sqrt{n}$ on the common shift) rather than by a
sum-to-zero constraint. `config.PRIOR_THETA_SKEW_ALPHA`; exclusive with `--theta-t`. In the
sampler study (`5_outputs/sampler_study/GEOMETRY_REPORT.md`) the shapes came out at $-8$ to $-10$
on three axes and $\approx 0$ on one, every chain; on the flagship 14 of 30 chains reached the
log-likelihood criterion against 4 of 30, with a better LOO on each of three seeds; on the refit of 2026-10-19 the 20
flagship chains fell into two modes, 17 of them in the best, where the Gaussian block had left five to seven. The
builder's own default (`build_mirt_model(theta_skew=False)`) stays Gaussian, so the canonical K=1 index and the golden
log-probabilities are unchanged; `FitSpec.theta_skew` carries the default of the K-axis fits.

**Optional substantive $\theta$ structure** (composes with any loading prior;
both are priors on $\theta$ only, and the unstructured models keep the
ZeroSumNormal so the location stays pinned):

*Human tiers — hard partial order.* For human baselines arranged as a partial
order (each tier has a parent tier; roots have none):

$$
\theta_{\text{root},k} \sim \mathcal{N}(0,\sigma_{\text{root}}^2),
\qquad
\theta_{\text{child},k} = \theta_{\text{parent},k} + \delta_{e,k},
\qquad
\delta_{e,k} \sim \mathrm{HalfNormal}(\sigma_\delta).
$$

$\sigma_{\text{root}}$ and $\sigma_\delta$ are `config.PRIOR_HUMAN_ROOT` and `config.PRIOR_DELTA_HUMAN`; each fit records the values it was sampled with in its spec (`FitSpec.human_root_sd` and `human_delta_sd`, the trace's `mirt_spec`), and whatever reads a trace back (a rebuilt model, the human-tier display rule) takes those.

The roots are private bases outside the models' ZeroSumNormal (2026-10-05): humans are no draw of the model population, and inside it the unit prior held Average Human 4 to 5 sd from the models on its extreme axis of the 2026-10-03 flagship, about a third of the way back from where its scores put it. They stay on the models' scale and axes: their scores on the shared benchmarks place them.

Because each increment $\delta_{e,k}\ge 0$, every child tier is **$\ge$ its
parent on every axis** (hard monotone). Tiers on different branches share no
path, so the prior says nothing about their relative strength.

*`--human-merge` — the flagship's order.* The merged order
(`config.HUMAN_ORDER_MERGED`) lets a tier carry **several parents**: the High
School branch merges into the adult spine, and the committees are chained,
Committee of Average Humans < Committee of Skilled Generalists < Committee of
Domain Experts, each still above its own base tier (2026-10-07; fits saved
before keep the order recorded in their `mirt_human_order` attr). Its ability is
a max over its root-to-tier paths, each path priced with one HalfNormal
increment per edge:

$$
\theta_{\text{child},k} = \max_{p \in \text{parents}} \theta_{p,k} + \delta_{e,k}.
$$

The path-unrolled implementation is algebraically exact
($\max(a,b)+\delta = \max(a+\delta,\,b+\delta)$), so a merged tier dominates
**every** parent on every axis while adding paths, not parameters. The gaps
stay free, and unrelated branches stay incomparable, as in the tree order.

*Tested and dropped (2026-10-07): a fifth axis for the humans.* On the K=4
fit, the human tiers sit far outside the models' cloud on two axes (about
−5 on the cyber axis, +8 on the agentic one), one error compensating the
other. A K=5 fit whose last loading column was anchored a priori on the
curated "easy for humans" list (full HalfNormal scale there, 0.1 elsewhere;
models move on that axis too) was meant to give the humans' own advantage
a place to go. With the humans' roots at sd 3 (10 chains) it split into 7
modes, the best a single chain, and the compensation stayed (tiers at −6 to
−3 and +8); the fifth axis took HellaSwag, PIQA, TriviaQA and WinoGrande but
placed no tier. With the roots at sd 1 on the four capability axes and 3 on
the anchored one (5 chains) the tiers came back inside the cloud on axes
1–4, but only because the prior put them there, the whole human gap moved
to the fifth axis (+16 to +23), axes 3 to 5 lost their meaning, and every
chain found its own mode. The code was removed; the human tiers are placed
instead on a fit without them, axis by axis, with the benchmark side frozen
(the "2b" placement).

*Tested and dropped (2026-10-07): the non-compensatory link.* `--link noncomp`
fixes the softmin link at beta = 5 (`config.NONCOMP_BETA`), close to the
minimum over the axes: a weak axis caps every benchmark it loads. On the K=4
fit with the humans (5 chains) it found four modes, the best a single chain,
and predicted the models' scores worse than the compensatory link: −62 nats
(se 27) on the 4,231 model scores with Pareto k < 0.7 in both fits, −123 on
plain PSIS. The humans stayed far from the models (above the frontier on
three axes, below it on an academic-knowledge axis no recent model is limited
by). The semi-compensatory link (beta learned, about 0.9) ties the
compensatory one on the models' scores.

*Lineage — soft release chains.* For a vendor's release chain (founder →
successors), improvement is the *mean* step but a node may regress:

$$
\psi_{\text{founder},k} \sim \mathcal{N}(0,1),
\qquad
\text{step: } \Delta_{e,k} = \mu_k + s_k\,z_{e,k},
$$
$$
\mu_k \sim \mathrm{HalfNormal}(0.3),\quad
s_k \sim \mathrm{HalfNormal}(1.0),\quad
z_{e,k}\sim\mathcal{N}(0,1),
$$
$$
\psi_{\text{node},k} = \psi_{\text{founder},k} + \textstyle\sum_{e\in\text{path}}\Delta_{e,k},
\qquad
\theta_{m,k} = \psi_{\text{node}(m),k} + \tau_o\,o_{g(m),k},
$$
$$
\tau_o \sim \mathrm{HalfNormal}(0.25),\qquad o_{g,k}\sim\mathcal{N}(0,1).
$$

| Symbol | Meaning |
|---|---|
| $\psi_{\text{node},k}$ | chain ability of a release node on axis $k$. |
| $\mu_k$ | positive **drift** — the mean per-step improvement (`PRIOR_LINEAGE_DRIFT=0.3`). |
| $s_k$ | per-step **regression tolerance**: a step can be negative (`PRIOR_LINEAGE_DELTA=1.0`). |
| $\tau_o$ | tight scale of the per-variant offset $o$ (`PRIOR_LINEAGE_OFFSET=0.25`). Reasoning-off variants (`effort:none`) have a scale of their own, $\tau_{\text{none}} \sim \text{HalfNormal}(1)$ (`PRIOR_LINEAGE_OFFSET_NONE`), since turning reasoning off can cost far more than a lower effort. |
| $g(m)$ | the variant group of model $m$. |

*`--lineage-bm` — the flagship's chain prior.* With `--lineage-prior
--lineage-bm` the chain is indexed by **time**: each step's mean and variance
scale with the release gap $dt_e$ (in years),

$$
\Delta_{e,k} = \mu_k\, dt_e + s_k \sqrt{dt_e}\; z_{e,k},
\qquad
\mu_k \sim \mathrm{HalfNormal}(1.0),\quad
s_k \sim \mathrm{HalfNormal}(2.0),
$$

a Brownian motion with drift observed at the release dates, under **one shared
per-axis drift rate** (a per-vendor pooled rate was fitted 2026-07-27 and
removed: roughly one release per year per vendor cannot pin a rate). Constants:
`PRIOR_LINEAGE_DRIFT_BM = 1.0` (logits/year) and `PRIOR_LINEAGE_DELTA_BM = 2.0`
(logits/$\sqrt{\text{year}}$). The mean-zero variant offsets (`--no-effort-means` only, see below) are unchanged.

*Effort means — a mean per effort level (the default since 2026-10-03; `--no-effort-means` opts out).* The offsets above are mean-zero, but the
prior-tail audit of the 2026-10-01 flagship put every reasoning-off offset at $-3$ to $-4$ sd
on the reasoning axis in all eight chains, and the `effort:high`-and-above ones at $+0.4$ on
average: a systematic shift fitted as noise. With the effort means the offset of a group $g$
at effort level $\ell(g)$ is

$$
o_{g,k} = \big(m_{\ell(g)} + \beta_{\text{tok}}\,\log_2(b_g/16\text{k})\big)\,u_k + \tau_o\,z_{g,k},
\qquad m_\ell \sim \mathcal{N}(0, 1),\quad \beta_{\text{tok}} \sim \mathcal{N}(0, 0.5),
$$

with one direction $u = v/\lVert v\rVert$, $v \sim \mathcal{N}(0, I_K)$, shared by every level
(rank one), and one spread $\tau_o$ for every group: $\tau_{\text{none}}$ is dropped. The levels
(`config.EFFORT_LEVEL_OF`) are `minimal`, `low`, `medium`, `high`, `xhigh`, `max`, `thinking`
(with `reasoning`), `unspecified` and `token`, whose budget $b_g$ adds the per-doubling slope
$\beta_{\text{tok}}$. They are **unordered**.

The reference, $m = 0$, is **reasoning off**, the one setting every model has, so $\psi$ means
the same thing in every chain: a release's level without test-time reasoning. At the reference
sit the non-reasoning run modes (`config.EFFORT_OFF_VARIANTS`: `effort:none`, `non-reasoning`,
`non-thinking` for DeepSeek's chat endpoint and Qwen3's Instruct-2507 checkpoints, GPT's
`instant` and `chat` models) and the runs naming no effort (`bare`, `instruct`, `preview`,
`effort:unknown`) of a non-reasoning release. The same runs of a release that always reasons
take `thinking` (reasoning on, effort unspecified; such a bare run sits within 0.2 ECI-H of its
release's effort runs on average), those of a hybrid release `unspecified` (its mode unknown),
per the reviewed regime of `1_curated/model_regimes.csv` (`lineage.effort_level`). A bare
release kept for the reference had meant reasoning off in a 2023 model, default reasoning in o3
and an unknown mix in a hybrid, and 25 chains mixed the three, so a step across the reasoning
era (GPT-4.1 to GPT-5) carried the reasoning gain. The price: a release with no reasoning-off
run (o3, gpt-oss, Gemini 2.5 Pro) has its $\psi$ extrapolated through the shared effort means.

A release's pretrained checkpoint (`base`) gets a free mean vector of its own,
$\mu_{\text{base}} \sim \mathcal{N}(0, I_K)$, off the effort direction: pretraining against post-training trades knowledge for reasoning, which no effort
level does. Every $m_\ell \sim \mathcal{N}(0, 1)$. $(m, u)$ and $(-m, -u)$ give the same means, likelihood and prior density, so the sampler sees two mirror copies; the sign is fixed after sampling, draw by draw, by flipping $(m, u)$ wherever the mean of $m$ over the levels is negative (`models/mirt.align_effort_sign`), so $u$ points from reasoning off toward more effort. This is the folded prior a sign constraint would give, without its boundary: a HalfNormal on $m_{\text{high}}$, the gauge until 2026-10-05, trapped one chain of the 2026-10-03 flagship in the mirror half with $m_{\text{high}}$ at 0. Crossing between the copies would take every effort mean through zero at once, and no chain of that fit came near it, so the flip never splits a chain. At $K=1$ there is no direction ($u = 1$) and no mirror. A free mean per (level, axis) was set aside: it would slide between correlated loading
columns as a per-axis $\tau_o$ did. Extra parameters: 10 at $K=1$ (9 levels, the slope and the
base mean, less $\tau_{\text{none}}$), 17 at $K=4$ (adding the direction and the
base vector). Readouts: `effort_mu` (level x axis), `effort_dir`, `effort_token_slope`,
`variant_base_mu` (`models/mirt._variant_offsets`). The cyber fine-tunes,
DeepSeek-V3.2-Speciale, OpenAI's computer-use model and Kimi K2.7 Code are out of the chains
(free test-takers), and Claude Instant is a product line whose runs are `bare`
(`1_curated/2_build_lineage_map.py`).

**Release nodes (canonical $K=1$ index).** The canonical fit keeps the nodes and variant offsets of the lineage prior and drops its chains: each release node $n$ has its own level $\psi_n$, drawn in the shared ZeroSumNormal with the unstructured models, and a run gets $\theta_m = \psi_{n(m)} + o_{g(m)}$ with the same $\tau_o$ (and the same effort means). No release is placed after another; a run's effort variants share their release's level (`models/mirt._release_block`).

### 5.5 Rotation identification (signed family only)

Two mutually exclusive ways to remove the rotation freedom of the signed
priors, so no post-hoc alignment is needed:

- **Anchors / Q-matrix** (`"normal"` only): hard-zero the mask so each
  benchmark loads on a prescribed axis subset.
- **PLT founders** (`"signed"`): pick $K$ ordered "founder"
  benchmarks; founder $r$ (0-based) gets a **positive-lower-triangular** row,

$$
A_{\text{founder}_r,\,k} = 0 \ \ (k>r),
\qquad
A_{\text{founder}_r,\,r} > 0,
\qquad
A_{\text{founder}_r,\,k}\ \text{free}\ (k<r).
$$

  The diagonal positivity is enforced by folding the cell's sign ($|A|$ on the
  diagonal, which turns a signed Normal cell into a HalfNormal-shaped one); the
  above-diagonal zeros are set directly. No new random variables. Together the
  $K$ rows kill rotation, sign flips, and axis permutation in the sampler.

---

## 6. From ability to the ECI-H scale

The human-anchored index is written **ECI-H** to keep it distinct from
Epoch's published ECI; the two share the anchor convention below.

The reported index is an **affine** transform of the overall capability
(axis 1 of a $K$-axis fit, or $\theta$ of a 1D fit), applied **per posterior
draw**:

$$
\mathrm{ECI\text{-}H}_m = a + b\,\theta_{m,1}.
$$

$a$ and $b$ are fixed by two anchor models:

$$
b = \frac{150 - 130}{\theta_{\text{GPT-5}} - \theta_{\text{Claude 3.5 Sonnet}}},
\qquad
a = 130 - b\,\theta_{\text{Claude 3.5 Sonnet}},
$$

i.e.

$$
\mathrm{ECI\text{-}H}_{\text{Claude 3.5 Sonnet (2024-10-22)}} = 130,
\qquad
\mathrm{ECI\text{-}H}_{\text{GPT-5 (2025-08-07, medium)}} = 150.
$$

The choice of anchors is arbitrary (it matches the public Epoch dashboard for
comparability); differences and rankings are **invariant** to it.

---

## 7. One-line summary of the three cases

$$
\mu_n =
\begin{cases}
\sigma\!\big(A_{b_n}\theta_{m_n} - D_{b_n}\big) & \textbf{1D}\ (K=1)\\[4pt]
\sigma\!\big(\sum_k A_{b_n,k}\theta_{m_n,k} - D_{b_n}\big) & \textbf{KD}\\[4pt]
c_{b_n} + (1-c_{b_n})\,\sigma\!\big(\sum_k A_{b_n,k}\theta_{m_n,k} - D_{b_n}\big) & \textbf{KD 3PL}
\end{cases}
$$

with the same $y_n\sim\mathrm{Beta}(\mu_n\phi_{b_n},(1-\mu_n)\phi_{b_n})$,
$\phi_b = \tfrac{1}{4\sigma_b^2}-1$, boundary clipping at $\varepsilon=10^{-3}$ (or the Beta-binomial of §2.3 on item-scored benchmarks, the default),
and the priors of Section 5 throughout. 1D is KD at $K=1$; KD is KD 3PL at
$c\equiv 0$.

---

## 8. Placing the human tiers on the flagship (fitted without them)

Since 2026-10-08 the K=4 flagship is fitted on the models alone (`--no-humans`) and the human
tiers are placed on it afterwards, one axis at a time (`analysis.human_placement`,
`4_diagnostics/5_place_humans.py`). With the benchmark side frozen at the selected chains'
medians ($\hat A$, $\hat D$, $\hat\phi$), on axis $k$:

- **Benchmarks.** $\mathcal B_k = \{\,b : k = \arg\max_j \hat A_{b,j}^2\,\}$, the benchmarks
  whose largest share of loading variance is on $k$, minus the pairs judged off-topic for the
  axis's name in `1_curated/human_axis_content.csv`.
- **Other axes.** A tier's abilities off $k$ are those of a typical model at the same ability on
  $k$: a least-squares line over the models' posterior medians,
  $\theta_{j} \approx \alpha_j + \gamma_j\,\theta_k$ for $j\neq k$. Benchmark $b$ then sees the
  tier through
  $$\eta_b(\theta_k) = \Big(\hat A_{b,k} + \sum_{j\ne k}\hat A_{b,j}\gamma_j\Big)\,\theta_k
  + \sum_{j\ne k}\hat A_{b,j}\alpha_j - \hat D_b ,$$
  and its score follows the fit's own likelihood (fixed-c floor, Beta-binomial or Beta).
- **Tiers.** The nine tiers keep the ordered prior of §5.4 on this one axis
  (`config.HUMAN_ORDER_MERGED`, roots $\mathcal N(0, 3^2)$, increments
  $\mathrm{HalfNormal}(2)$), sampled with NUTS.
- **Display.** A tier is drawn where it has a score below the ceiling on $\mathcal B_k$, or
  sits between two such tiers in the order; a score at the ceiling bounds the tier from below
  and places nothing alone.

The placement is checked score by score (observed against predicted at the tier's draws) and
against the first model that reached each score on the benchmark itself, in
`human_placement.json`.

Why not fit them inside: with the 56 human scores spread over four axes, the compensatory sum
of §4.2 let the tiers trade one axis against another (about −5 on one, +8 on another, every
model in between), whatever the prior on the tiers (§5.4), an extra axis anchored on the
benchmarks easy for humans, or a softer link (the softmin links below).

**Softmin links (exploration, `--link semicomp` / `semicomp0` / `noncomp`).** With
$s_b=\sum_k A_{b,k}$ and $w_{b,k}=A_{b,k}/s_b$,
$$\eta_b = \sum_k A_{b,k}\theta_k - D_b - \frac{s_b}{\beta}\log\sum_k w_{b,k}\,
e^{-\beta(\theta_k-\bar\theta_b)},\qquad \bar\theta_b=\sum_k w_{b,k}\theta_k ,$$
the compensatory predictor minus a non-negative imbalance penalty, which vanishes as
$\beta\to0$ and makes the weakest loaded axis cap the score as $\beta$ grows; to second order it
is $\frac{\beta}{2 s_b}\sum_{j<k}A_{b,j}A_{b,k}(\theta_j-\theta_k)^2$. `semicomp` learns
$\beta$ ($\mathrm{LogNormal}(0,1)$) with a per-axis offset added to $\theta$, `semicomp0` the
same without the offset, `noncomp` fixes $\beta=5$. On the 2026-10-01 data, `semicomp`
learns $\beta\approx0.8$ and ties the compensatory fit on held-out model scores; without the
offset's freedom one axis no longer becomes the binding one on almost every benchmark, which
it did with it (95 of 98 benchmarks on the fit without humans).
