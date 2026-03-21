# Project Current Status

## Workstream A: Diff-AE training optimization
Current status: discussion only, not yet ready for Codex-led implementation.

Known discussion points from prior analysis:
- Current setting under discussion included:
  - decay = 0.0
  - activation quantization off
  - a_w detach
- Layer 3 currently tends to keep `(True, False)`.
- For Layer 3, `(False, False)` only lowered loss by about `0.000002`, so it is not considered meaningfully better.
- Layer 144 currently tends to choose `(False, False)`.
- Logged "total loss" and "distillation loss" are both averages over the full 100 denoising steps.
- Actual batch size is always 6; logged batch_size value has a bug and should not be trusted.
- Logging every 20 steps records average loss, so sudden loss rise in a segment is meaningful.
- Noise is currently ignored in decision-making.

What is still missing:
- next-round experiment order
- exact ablation plan
- success criteria
- which parameters / toggles should be changed first

Conclusion:
This workstream should remain in discussion mode for now.
Codex should not directly modify training logic unless later instructed with a concrete task.

---

## Workstream B: v_final cache pipeline
Current status:
- Stage 0 implemented
- Stage 1 implemented
- results exist for Stage 0 and Stage 1
- Stage 2 not yet discussed in enough detail

What is needed now:
1. inspect current Stage 0/1 related code paths
2. inspect output files and schemas
3. produce a clean code map / result map
4. verify whether current implementation matches written v_final design
5. prepare the repo so future Stage 2 discussion and implementation can proceed cleanly

Conclusion:
This workstream is ready for Codex-assisted documentation, inspection, and alignment work.
It is NOT ready for Stage 2 implementation.
