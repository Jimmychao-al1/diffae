#!/usr/bin/env bash
set -euo pipefail

SCHED="${SCHED:-QATcode/cache_method/Stage1/stage1_output/sweep_K16_sw3/scheduler_config.json}"
OUT_BASE="${OUT_BASE:-QATcode/cache_method/Stage2/stage2_output/plan1_K16_sw3/10_variants_blockwise}"
THRESH="${THRESH:-QATcode/cache_method/Stage2/stage2_output/plan1_K16_sw3/01_blockwise_threshold/stage2_thresholds_blockwise.json}"

PY=(python QATcode/cache_method/Stage2/stage2_runtime_refine.py)

"${PY[@]}" --scheduler_config "$SCHED" --output_dir "$OUT_BASE/baseline" --seed 0 \
  --zone_l1_threshold 0.02 --peak_l1_threshold 0.08 \
  --threshold-config "$THRESH" \
  --eval-num-images 8 \
  --eval-chunk-size 2

for N in 5 10 15; do
  "${PY[@]}" --scheduler_config "$SCHED" --output_dir "$OUT_BASE/prefix_${N}" --seed 0 \
    --zone_l1_threshold 0.02 --peak_l1_threshold 0.08 \
    --threshold-config "$THRESH" \
    --force-full-prefix-steps "$N" \
    --eval-num-images 8 \
    --eval-chunk-size 2
done

"${PY[@]}" --scheduler_config "$SCHED" --output_dir "$OUT_BASE/first_input_only" --seed 0 \
  --zone_l1_threshold 0.02 --peak_l1_threshold 0.08 \
  --threshold-config "$THRESH" \
  --safety-first-input-block \
  --eval-num-images 8 \
  --eval-chunk-size 2

for N in 5 10 15; do
  "${PY[@]}" --scheduler_config "$SCHED" --output_dir "$OUT_BASE/combined_${N}" --seed 0 \
    --zone_l1_threshold 0.02 --peak_l1_threshold 0.08 \
    --threshold-config "$THRESH" \
    --force-full-prefix-steps "$N" \
    --safety-first-input-block \
    --eval-num-images 8 \
    --eval-chunk-size 2
done