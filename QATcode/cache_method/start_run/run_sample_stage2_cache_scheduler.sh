#!/usr/bin/env bash
set -euo pipefail

python QATcode/cache_method/start_run/sample_stage2_cache_scheduler.py \
  --mode float \
  --num_steps  100 \
  --eval_samples 5000 \
  --seed 0 \
  --quant-state tt \
  --use_cache_scheduler \
  --cache_scheduler_json QATcode/cache_method/Stage2/stage2_output/run_per_baseline/stage2_refined_scheduler_config.json
