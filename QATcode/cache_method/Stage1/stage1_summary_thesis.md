# Stage 1 Summary (Thesis-Oriented Version)

## 1. Purpose of Stage 1

Stage 1 builds an **initial static cache scheduler** for diffusion inference. It does **not** perform Stage‑2‑style refinement, nor does it chase every local bump in the evidence curves; it produces a **structured, interpretable scaffold** that Stage 2 can refine.

Under the current implementation (commit: `b262d51a947a53fc5276ff8243676c942a0bbce5`), Stage 1 performs two major functions:

1. construct a global shared temporal partition (`shared_zones`), and  
2. determine a zone-wise reuse period `k` for each block under the shared partition.

The final outputs of Stage 1 are therefore:

- `shared_zones`
- `k_per_zone`
- `expanded_mask`

These outputs together define the initial static scheduler used as the entry point of Stage 2.

---

## 2. Formal Definition of the Stage 1 Pipeline

### 2.1 Input from Stage 0

Stage 1 takes the normalized tri-evidence outputs from Stage 0 as input:

- `l1_interval_norm.npy`
- `cosdist_interval_norm.npy`
- `svd_interval_norm.npy`
- `fid_w_qdiffae_clip.npy`

In addition, Stage 1 also reads:

- `block_names.npy`
- `axis_interval_def.npy`
- `t_curr_interval.npy`

The implementation **validates** that Stage 0’s interval layout matches Stage 1’s reused‑timestep mapping, so cross‑stage drift cannot pass silently.

---

### 2.2 Mapping interval-wise evidence to reused DDIM timesteps

Stage 0 supplies **interval-wise** evidence (one column per transition). Stage 1 maps column $j$ to the **reused DDIM timestep** $t$ at which that reuse is scored—formally, interval $(t{+}1 \!\to\! t)$ maps to index $t$.

**Equation (1)** (L1/Cos branch; **not** a similarity or stability score—only a combined **change magnitude**):

$$
I_{\mathrm{l1cos}}[b,t]
  = 0.7\,\mathrm{L1}_{\mathrm{norm}}[b,t] + 0.3\,\mathrm{Cos}_{\mathrm{norm}}[b,t].
$$

**Equation (2)** (cutting evidence):

$$
I_{\mathrm{cut}}[b,t]
  = \tfrac{4}{9}\, I_{\mathrm{l1cos}}[b,t] + \tfrac{5}{9}\,\mathrm{SVD}_{\mathrm{norm}}[b,t].
$$

The **last** DDIM step ($t = T{-}1$; e.g. $t=99$ when $T=100$) has no interval column; the code sets **Equation (3)**:

$$
I_{\mathrm{cut}}[b,T-1] = 0
$$

for cutting statistics, while the deployed scheduler still **forces** that timestep to full-compute ($F$).

---

### 2.3 Construction of the global cutting signal

For each timestep $t$, the block-wise cutting evidence is aggregated into a global cutting signal $G[t]$ using block-level FID weights $w_b$.

**Equation (4)** (normalized FID weights; **uniform** $1/B$ fallback if $\sum_b w_b \approx 0$, with a warning):

$$
G[t] = \sum_b \frac{w_b}{\sum_{b'} w_{b'}} \, I_{\mathrm{cut}}[b,t].
$$

Thus $G[t]$ is a **convex combination** of the per-block cutting traces at $t$, and remains well-defined when weights vanish.

---

### 2.4 Temporal smoothing and change-point extraction

The global cutting signal is first reordered into the DDIM processing order:

- step index $i = 0$ corresponds to DDIM timestep $t = T-1$
- step index $i = T-1$ corresponds to DDIM timestep $t = 0$

The reordered signal is denoted $G_{\mathrm{proc}}$. A moving average yields $G_{\mathrm{smooth}}$.

The adjacent-step variation is **Equation (5)**:

$$
\Delta[i] = \left|\, G_{\mathrm{smooth}}[i] - G_{\mathrm{smooth}}[i-1] \,\right|.
$$

The scheduler takes the **top‑$K$** largest values in $\Delta$ as **change points** and builds the initial **shared** partition over the full trajectory. A **short‑zone merge** then enforces a minimum zone length so singleton or near‑singleton segments do not dominate the scaffold.

---

## 3. Design Principle of Shared Zones

A key design choice of the current Stage 1 implementation is that the temporal partition is shared across all blocks.

This means that Stage 1 does not assign independent temporal zones to each block.  
Instead, it first constructs a single global temporal scaffold from the aggregated evidence, and only then determines the block-wise reuse period within each zone.

This design has two implications.

First, the goal of Stage 1 is not to maximize the fit to any individual block’s SVD curve.  
Second, the purpose of Stage 1 is not to explain every local fluctuation, but rather to produce a scheduler scaffold that is:

- structurally meaningful,
- not overly coarse,
- not overly fragmented,
- still capable of supporting meaningful $k$ selection within each zone, and
- suitable for subsequent refinement in Stage 2.

---

## 4. Zone-wise Reuse Decision

After the shared temporal zones are defined, Stage 1 selects a reuse period $k$ for each pair $(b,z)$.

For block $b$, zone $z$, and candidate $k$, the objective is **Equation (6)**:

$$
J(b,z,k)
  = w_b \cdot \frac{1}{L_z} \sum_{t \in \mathcal{R}} I_{\mathrm{cut}}[b,t]
    \;+\; \lambda \cdot \frac{|\mathcal{F}|}{L_z},
$$

where:

- $w_b$: FID-based weight of block $b$
- $L_z$: length of zone $z$
- $\mathcal{R}$: set of **reuse** timesteps in the zone
- $\mathcal{F}$: set of **full-compute** timesteps in the zone; $|\mathcal{F}|$ counts them
- $\lambda$: trade-off between reuse risk and compute penalty

The first term measures reuse risk; the second penalizes a high full-compute fraction.

For each $(b,z)$, the code enumerates candidate $k$, drops equivalent F/R patterns via `unique_k_representatives(...)`, and picks the minimizer of $J(b,z,k)$.

---

## 5. Beyond “Following SVD Fluctuations” When Choosing $K$

SVD drift is often larger in early and late parts of the trajectory, which **supports using more than one temporal regime**—i.e., not merging the whole run into a single coarse zone.

That does **not** imply that **more** change points are always better.

The reason is that Stage 1 is not only a segmentation procedure.  
It is a scheduler synthesis stage.  
Therefore, zone quality also depends on whether each zone still affords a **non-degenerate** choice of $k$.

When $K$ is too large:

1. shared zones fragment,
2. many zones shrink to length $2$ or $3$,
3. distinct F/R patterns per zone collapse, and
4. $J(b,z,k)$ is driven by **local** noise rather than **zone-level** behavior.

Thus, larger $K$ helps only up to a point; beyond it, the scaffold is a weaker Stage‑2 prior.

---

## 6. Baseline Selection Strategy

The baseline should be selected according to the actual structure of the Stage 1 implementation.

### 6.1 Step 1: choose $K$ and the smoothing window first

- $K$ and the smoothing window fix the **shared scaffold**;
- $\lambda$ only shifts zone-wise $k$ **after** that scaffold is fixed.

Rank candidates by scaffold quality first, then sweep $\lambda$.

---

### 6.2 Hard criteria for Stage-2-ready shared zones

Let $L_z$ be the length of zone $z$, and let $N_{\mathrm{zones}}$ be the zone count (empirical thresholds for $T{=}100$ sweeps).

The scaffold passes only if **all** of the following hold:

$$
8 \le N_{\mathrm{zones}} \le 12, \quad
\mathrm{median}_z(L_z) \ge 3, \quad
\max_z L_z \le 0.55\,T,
$$
$$
\mathrm{frac}(L_z = 2) \le 0.40, \quad
\mathrm{frac}(L_z \le 3) \le 0.60.
$$

These rules reject both **over-merging** (too few, too long zones) and **over-splitting** (many zones stuck at the minimum length).

---

### 6.3 Hard criteria for candidate-$k$ validity

Let $C_z$ be the number of **distinct** F/R patterns after deduplication in zone $z$.

Require:

$$
\overline{C}_z \ge 3.0, \qquad \mathrm{frac}(C_z \le 2) \le 0.40.
$$

Because `unique_k_representatives` merges equivalent $k$’s, a long list of nominal $k$’s can be illusory when zones are short—these checks ensure the candidate set still **matters** for $J(b,z,k)$.

---

### 6.4 Ranking after the hard filtering stage

Among survivors, rank by:

1. lower mean full-compute fraction $F_{\mathrm{frac\,mean}}$;
2. tie-break: smaller $\max_z L_z$;
3. tie-break: prefer $\lambda = 1.0$.

This balances **reuse** against **interpretable** zone geometry.

---

## 7. Final Selected Baseline

Based on the current Stage 1 sweep results, the selected baseline is:

**`sweep_K16_sw3_lam0.5_kmax4`**

Hyperparameters (selected run): $K = 16$, $\lambda = 1.0$, $k_{\min} = 1$, $k_{\max} = 4$, smooth\_window $= 3$, min\_zone\_len $= 2$.

---

## 8. Why $K = 16$ over $K \in \{20,25\}$

Preferring $K=16$ is **not** a claim that $K=20$ or $K=25$ is “wrong”—it reflects Stage‑1’s job: a **Stage‑2‑friendly** scaffold, not maximal segmentation sensitivity.

- Versus **small** $K$ (e.g. $K=8$): $K=16$ resolves structure better and shortens oversized zones.
- Versus **large** $K$ (e.g. $K\in\{20,25\}$): $K=16$ keeps zones interpretable, preserves diverse candidate $k$ per zone, and leaves headroom for Stage‑2 edits.

So $K=16$ is a pragmatic compromise between **expressiveness** and **usability**.

---

## 9. Why $k_{\max} = 4$ in the baseline

Here $k_{\max}$ caps the **intra-zone** reuse stride; it does **not** set the global zone boundaries.

When zones are already short, raising $k_{\max}$ may add **no** new distinct F/R patterns—many $k$’s collapse under `unique_k_representatives`. Tweaking $k_{\max}$ alongside the scaffold also **confounds** partition quality with search-space width.

The baseline therefore fixes $k_{\max}=4$ and defers $\{3,4,5\}$ as a **controlled ablation** after initial Stage‑2 feedback.

---

## 10. Final Conclusion of Stage 1

The pipeline in order:

1. Build per-block cutting traces $I_{\mathrm{cut}}[b,t]$ from normalized L1/Cos/SVD.
2. Form $G[t]$ via FID-normalized mixing; reorder to DDIM order; smooth; form $\Delta$; take top‑$K$ change points; merge short zones $\Rightarrow$ `shared_zones`.
3. For each $(b,z)$, dedupe candidate $k$, minimize $J(b,z,k)$, expand to `expanded_mask`.
4. Ship **`shared_zones`, `k_per_zone`, `expanded_mask`** as the Stage‑2 entry artifact.

Under this formulation, the selected baseline is:

**`sweep_K16_sw3_lam0.5_kmax4`**

This baseline is considered the most appropriate entry point to Stage 2 because it provides a scheduler scaffold that is:

- sufficiently expressive in time,
- not overly fragmented,
- still meaningful for zone-wise $k$ selection, and
- amenable to subsequent local refinement.

---

## 11. Connection to Stage 2

Stage 2 runs **cache‑aware refinement** on this baseline: no full rescaffold; instead, local edits to $k$, zone boundaries, and validation (feature error, FID, compute).

**Stage‑1 in one line:** a **structured, interpretable, refinement‑ready** static scheduler for Stage‑2 to build on.
