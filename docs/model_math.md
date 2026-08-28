# ECI Beta-MIRT — the math

The whole framework is one likelihood (a reparametrized Beta) sitting on top of
one linear predictor $\eta$. Three cases differ **only** in how $\eta$ maps to
the mean $\mu$:

- **1D** — the canonical index, $K=1$.
- **KD** — the $K$-axis compensatory MIRT.
- **KD 3PL** — KD with a fixed per-benchmark chance floor.

Everything else (the Beta likelihood, the priors on difficulty, noise, and
abilities) is shared. This document defines every symbol.

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
(Section 5) keeps $\sigma_b$ well below that.

### 2.2 Boundary clipping

The Beta has open support $(0,1)$, but a handful of scores are exactly $0$ or
$1$. Those (and only those) are clipped inward before being passed as the
observed values:

$$
\tilde y_n = \min\!\big(\max(y_n,\ \varepsilon),\ 1-\varepsilon\big),
\qquad \varepsilon = 10^{-3}.
$$

Interior scores are untouched. $\varepsilon$ is `ECI_EPS` in the code.

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
$A_b\theta_m - D_b$, absorbed by the ECI anchoring in Section 6.)

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
| $c_b \in [0,1)$ | **fixed** lower asymptote (chance floor) of benchmark $b$ — e.g. $0.25$ for 4-choice MC. Read from `benchmark_lower_bounds.csv`. |

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

### 5.3 Loadings $A_{b,k}$

There are three loading priors. All share one derived scale vector $\tau_A$
(length $K$).

**(a) `"normal"` — canonical + confirmatory.** Non-negative loadings, one
learned shared scale:

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

This is the prior used for the **1D canonical index** ($K=1$, mask all ones)
and for the confirmatory Q-matrix fits.

**(b) `"signed"` — exploration default.** iid signed cells × one shared scale:

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

**Optional substantive $\theta$ structure** (composes with any loading prior;
both are priors on $\theta$ only, and the unstructured models keep the
ZeroSumNormal so the location stays pinned):

*Human tiers — hard partial order.* For human baselines arranged as a partial
order (each tier has a parent tier; roots have none):

$$
\theta_{\text{root},k} \sim \mathcal{N}(0,1),
\qquad
\theta_{\text{child},k} = \theta_{\text{parent},k} + \delta_{e,k},
\qquad
\delta_{e,k} \sim \mathrm{HalfNormal}(\text{PRIOR\_DELTA\_HUMAN}=1).
$$

Because each increment $\delta_{e,k}\ge 0$, every child tier is **$\ge$ its
parent on every axis** (hard monotone). Tiers on different branches share no
path, so the prior says nothing about their relative strength.

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
| $\tau_o$ | tight scale of the per-variant offset $o$ (`PRIOR_LINEAGE_OFFSET=0.25`). |
| $g(m)$ | the variant group of model $m$. |

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

## 6. From ability to the ECI scale

The reported index is an **affine** transform of the overall capability
(axis 1 of a $K$-axis fit, or $\theta$ of a 1D fit), applied **per posterior
draw**:

$$
\mathrm{ECI}_m = a + b\,\theta_{m,1}.
$$

$a$ and $b$ are fixed by two anchor models:

$$
b = \frac{150 - 130}{\theta_{\text{GPT-5}} - \theta_{\text{Claude 3.5 Sonnet}}},
\qquad
a = 130 - b\,\theta_{\text{Claude 3.5 Sonnet}},
$$

i.e.

$$
\mathrm{ECI}_{\text{Claude 3.5 Sonnet (2024-10-22)}} = 130,
\qquad
\mathrm{ECI}_{\text{GPT-5 (2025-08-07, medium)}} = 150.
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
$\phi_b = \tfrac{1}{4\sigma_b^2}-1$, boundary clipping at $\varepsilon=10^{-3}$,
and the priors of Section 5 throughout. 1D is KD at $K=1$; KD is KD 3PL at
$c\equiv 0$.
