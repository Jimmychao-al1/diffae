# Codex Immediate Tasks

## Allowed immediate tasks
These tasks are safe to start now.

### Task 1: Stage 0/1 code map
Goal:
- identify the code paths related to Stage 0 and Stage 1
- identify where inputs come from
- identify what outputs are generated
- identify result directories / file names / schemas

Expected output:
- `docs/code_map_stage0_stage1.md`

Content should include:
- entry scripts
- core modules
- helper scripts
- config files
- expected outputs
- brief explanation of each file's role

---

### Task 2: Stage 0/1 output inspection utility
Goal:
Create a lightweight utility that inspects current Stage 0/1 output files.

Suggested output file:
- `tools/inspect_cache_stage01_outputs.py`

Suggested capabilities:
- check existence of expected files
- print basic schema / keys
- print simple stats such as length / shape / range where possible
- optionally dump a concise markdown summary

Important:
- this is an inspection utility, not a rewriting utility
- it should not modify result files

---

### Task 3: v_final alignment checklist
Goal:
Compare current Stage 0/1 implementation against the written v_final design.

Expected output:
- `docs/v_final_alignment_checklist.md`

The checklist should include:
- Stage 0 evidence availability
- normalization availability
- FID weighting availability
- global drift segmentation logic
- zone construction logic
- tri-evidence aggregation logic
- k mapping logic
- scheduler config output structure

For each item, mark:
- implemented
- partially implemented
- unclear / needs inspection
- missing

---

## Not allowed right now
1. implementing Stage 2
2. changing Stage 0/1 formulas without explicit request
3. modifying Diff-AE training optimization logic
