# v_final Cache Pipeline

## Overall goal
Design a static per-block cache scheduler for CNN-based diffusion using data-driven tri-evidence analysis.

Important scope note:
- The scheduler is currently used on GPU/CPU side.
- CIM only handles integer conv/linear execution.
- The scheduler itself is not constrained by CIM implementation at this stage.

---

## Stage 0: tri-evidence analysis
For each block b and timestep t:

1. Similarity-based stability
   - Derived from neighboring timestep feature comparisons
   - Metrics include L1 / L2 / Cosine-related evidence
   - Larger stability score => more cacheable

2. SVD-drift-based stability
   - Based on principal subspace drift across timesteps
   - Larger stability score => more cacheable

3. FID sensitivity
   - Block-wise ablation gives FID sensitivity score
   - Larger FID sensitivity => higher risk
   - Safety score can be defined from this for scheduler synthesis

Also maintain normalized drift curves per block across timesteps.

Stage 0 outputs conceptually include:
- per-block per-timestep similarity stability
- per-block per-timestep SVD stability / drift-related values
- block-wise FID sensitivity / safety score
- normalized drift curves

Status: already implemented.

---

## Stage 1: offline scheduler synthesis
Input comes from Stage 0 outputs.

### 1. Global drift and zone segmentation
- Use FID sensitivity as block risk weight
- Form FID-weighted global drift curve across timesteps
- Smooth the curve
- Compute timestep-to-timestep change magnitude
- Choose top-K or threshold-based change points
- Create shared time zones

Important meaning:
- global drift is only used to define shared time structure
- it is NOT a direct step-level "must recompute here" trigger

### 2. Zone-level tri-evidence aggregation
For each block b and zone z:
- aggregate similarity evidence inside zone
- aggregate SVD evidence inside zone
- use block safety score from FID-based evidence

Combine them into a tri-evidence cacheability score:
- larger score => more cacheable

### 3. Map score to k[b,z]
- map tri-evidence score to an initial per-block per-zone cache period
- also compute zone-level risk ceiling from global risk
- final k[b,z] should respect that ceiling
- optionally apply temporal smoothing / regularization on neighboring zones

### 4. Output
Stage 1 output is a static scheduler config:
- shared zones
- per-block k_per_zone
- enough information to expand into per-timestep recompute mask later

Status: already implemented.

---

## Stage 2: cache-run refinement
This stage is NOT yet fully discussed.
Therefore, it is NOT ready for implementation.

Only high-level intent is known:
- run cache mode with Stage 1 scheduler
- compare against baseline
- observe global quality degradation and local feature error
- locally adjust k and possibly zone boundaries
- obtain refined final scheduler config

But the exact operational definition is not finalized yet.

Codex must not invent the full Stage 2 algorithm.

---

## What Codex may do now
Codex may:
1. locate current Stage 0/1 scripts, modules, configs, outputs
2. summarize their relations
3. build output inspection tools
4. write alignment checklist docs
5. help produce Stage 0/1 result summary files

Codex may NOT:
1. implement Stage 2 logic
2. silently modify Stage 1 formulas
3. rewrite the project design without instruction
