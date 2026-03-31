#!/usr/bin/env bash
# Minimal examples for cache safety experiments (same Stage1 scheduler, different runtime unions).
# Usage: set SCHED and OUT_BASE, then uncomment/adapt.
set -euo pipefail

SCHED="${SCHED:-QATcode/cache_method/Stage1/stage1_output/sweep_K16_sw3_lam0.5_kmax4/scheduler_config.json}"
OUT_BASE="${OUT_BASE:-QATcode/cache_method/Stage2/stage2_output/safety_runs}"

PY=(python QATcode/cache_method/Stage2/stage2_runtime_refine.py)

# 1) Baseline (no runtime override; same as before defaults)
"${PY[@]}" --scheduler_config "$SCHED" --output_dir "$OUT_BASE/baseline" --seed 0

# 2) Prefix-only N ∈ {5,10,15}
for N in 5 10 15; do
  "${PY[@]}" --scheduler_config "$SCHED" --output_dir "$OUT_BASE/prefix_${N}" --seed 0 \
    --force-full-prefix-steps "$N"
done

# 3) First input block only (canonical: encoder_layer_0)
"${PY[@]}" --scheduler_config "$SCHED" --output_dir "$OUT_BASE/first_input_only" --seed 0 \
  --safety-first-input-block

# 4) Combined: prefix N + first input block
for N in 5 10 15; do
  "${PY[@]}" --scheduler_config "$SCHED" --output_dir "$OUT_BASE/combined_${N}" --seed 0 \
    --force-full-prefix-steps "$N" --safety-first-input-block
done

echo "Done. See stage2_runtime_diagnostics.json -> cache_scheduler_runtime_overrides in each output dir."
