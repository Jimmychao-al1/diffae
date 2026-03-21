# AutoResearch minimal harness for Q-DiffAE / quantize_ver2

## Goal
Run small, controlled experiments on `QATcode/quantize_ver2` using a fixed training/evaluation entrypoint and a small search space.

## Scope for v1
Only allow tuning of:
- `lora_factor`
- `weight_lr`
- `activation_lr`

Do **not** change:
- loss definition
- backward flow
- optimizer type
- validation logic
- dataset
- model architecture

## Success rule
A candidate run is better only if it improves the primary metric under the same evaluation setting.

Suggested primary metric for now:
- T=100 generation quality metric (the one you currently trust most)

Suggested secondary diagnostics:
- `loss/running_mean_ddim_loss`
- tail-related summaries
- per-timestep loss summaries if already logged

## Working rule
- One experiment = one small config change
- Keep logs and outputs outside git-tracked source directories
- Record every run in `results.tsv`
- Only promote a change after comparing against the current baseline
