# AGENTS.md

## Repository purpose
This repository contains two major workstreams:

1. Q-DiffAE / Diff-AE quantization training optimization
2. v_final cache pipeline implementation for diffusion model feature reuse

The user is currently using ChatGPT web discussion as the main place for method design.
Codex in IDE should follow the current written project status files, not invent missing specs.

---

## Current project policy

### Workstream A: Diff-AE training optimization
Status: NOT ready for direct implementation.

Codex should NOT directly modify core training logic, quantization logic, loss design, or experiment flow for this workstream unless the user explicitly gives a concrete implementation task.

At this stage, this workstream is still in discussion / planning mode.

### Workstream B: v_final cache pipeline
Status:
- Stage 0: implemented, results exist
- Stage 1: implemented, results exist
- Stage 2: NOT yet fully discussed, NOT ready for implementation

Codex MAY help with:
- reading and mapping Stage 0/1 related code
- reading and summarizing Stage 0/1 outputs
- building inspection / validation utilities
- writing documentation
- checking whether current implementation aligns with the written v_final design

Codex should NOT implement Stage 2 logic yet.

---

## Mandatory reading order before doing work
Before editing any code, read these files in order:

1. docs/project_current_status.md
2. docs/cache_pipeline_v_final.md
3. docs/qdiffae_training_optimization_status.md
4. docs/codex_immediate_tasks.md

After reading, first provide:
- a concise Chinese summary of understanding
- the relevant code paths / folders you found
- a proposed execution plan

Do not jump into large code edits before this summary.

---

## Editing rules
1. Prefer additive changes over destructive refactors.
2. Do not rename existing files or directories unless explicitly asked.
3. Preserve existing result directories and file naming conventions.
4. Put new documentation under `docs/`.
5. Put new lightweight utilities under `tools/` or `scripts/` if such directories already exist; otherwise create `tools/`.
6. If a design detail is missing, write it as TODO / assumption in docs instead of silently deciding.
7. Reply in Chinese when summarizing project understanding.
8. Keep code comments concise and professional.
9. For any Stage 0/1 verification task, first inspect existing files before proposing rewrites.

---

## Current immediate priority
Highest priority right now:
1. Build a clean understanding of Stage 0/1 current implementation and outputs.
2. Create supporting docs/tools for Stage 0/1 result inspection and alignment checking.
3. Do not start Stage 2 implementation.
4. Do not start Diff-AE training optimization implementation unless a later task explicitly asks for it.
