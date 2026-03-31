#!/usr/bin/env bash
set -euo pipefail

# Run FID@5K sampling for multiple Stage2 refined schedulers.
#
# Output layout:
#   QATcode/cache_method/results/stage2_scheduler_runs/
#     runs_index.jsonl                      ← append-only index (one line per run)
#     YYYYMMDD/
#       YYYYMMDD__<scheduler_name>/
#         run_manifest.json
#         summary.json
#         detail_stats.json
#         scheduler_config.snapshot.json
#         run.log
#
# Execution order (fixed):
#   baseline, prefix_5, prefix_10, prefix_15,
#   first_input_only, combined_5, combined_10, combined_15

STAGE2_BASE="QATcode/cache_method/Stage2/stage2_output/plan1_K16_sw3"
RESULTS_ROOT="QATcode/cache_method/results/stage2_scheduler_runs"
RUNS_INDEX="${RESULTS_ROOT}/runs_index.jsonl"

mkdir -p "${RESULTS_ROOT}"

run_exp() {
  local name="$1"
  local sched_json="$2"

  local date_str
  date_str="$(date +%Y%m%d)"
  local run_dir="${RESULTS_ROOT}/${date_str}/${date_str}__${name}"
  mkdir -p "${run_dir}"

  echo "===== Running scheduler: ${name} | output: ${run_dir} ====="

  python QATcode/cache_method/start_run/sample_stage2_cache_scheduler.py \
    --mode float \
    --num_steps 100 \
    --eval_samples 5000 \
    --seed 0 \
    --quant-state tt \
    --use_cache_scheduler \
    --cache_scheduler_json "${sched_json}" \
    --scheduler-name "${name}" \
    --run-output-dir "${run_dir}" \
    --runs-index-path "${RUNS_INDEX}" \
    --log_file "${run_dir}/run.log"

  echo "  → done. Summary: ${run_dir}/summary.json"
}

run_exp "baseline"         "${STAGE2_BASE}/baseline/stage2_refined_scheduler_config.json"
run_exp "prefix_5"         "${STAGE2_BASE}/prefix_5/stage2_refined_scheduler_config.json"
run_exp "prefix_10"        "${STAGE2_BASE}/prefix_10/stage2_refined_scheduler_config.json"
run_exp "prefix_15"        "${STAGE2_BASE}/prefix_15/stage2_refined_scheduler_config.json"
run_exp "first_input_only" "${STAGE2_BASE}/first_input_only/stage2_refined_scheduler_config.json"
run_exp "combined_5"       "${STAGE2_BASE}/combined_5/stage2_refined_scheduler_config.json"
run_exp "combined_10"      "${STAGE2_BASE}/combined_10/stage2_refined_scheduler_config.json"
run_exp "combined_15"      "${STAGE2_BASE}/combined_15/stage2_refined_scheduler_config.json"

echo ""
echo "All runs complete."
echo "Index: ${RUNS_INDEX}"
