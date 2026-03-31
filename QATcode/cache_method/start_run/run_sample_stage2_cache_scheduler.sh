#!/usr/bin/env bash
set -euo pipefail

# Run FID sampling for multiple Stage2 refined schedulers and aggregate results.
# Execution order (fixed):
#   baseline, prefix_5, prefix_10, prefix_15, first_input_only, combined_5, combined_10, combined_15

STAGE2_BASE="QATcode/cache_method/Stage2/stage2_output/plan1_K16_sw3"
LOG_DIR="QATcode/cache_method/start_run/log"
RESULT_JSON="QATcode/cache_method/start_run/stage2_cache_scheduler_fid_results.json"

mkdir -p "$LOG_DIR"

# JSON array as a string; we grow this via a tiny Python helper.
results='[]'

run_exp() {
  local name="$1"
  local sched_json="$2"

  echo "===== Running scheduler: ${name} ====="

  local log_file="${LOG_DIR}/sample_stage2_cache_scheduler_${name}.log"

  SECONDS=0
  python QATcode/cache_method/start_run/sample_stage2_cache_scheduler.py \
    --mode float \
    --num_steps 100 \
    --eval_samples 5000 \
    --seed 0 \
    --quant-state tt \
    --use_cache_scheduler \
    --cache_scheduler_json "${sched_json}" \
    --log_file "${log_file}"
  local elapsed="${SECONDS}"

  # Parse FID from log: expect line like "FID@N T=100 score: X"
  local fid
  fid="$(grep 'FID@' "${log_file}" | tail -n 1 | awk '{print $NF}')"
  if [[ -z "${fid:-}" ]]; then
    echo "ERROR: Failed to parse FID score from log ${log_file}" >&2
    exit 1
  fi

  # Append this experiment to the JSON array.
  results="$(python - "$results" "$name" "$fid" "$elapsed" << 'PY'
import json
import sys

arr = json.loads(sys.argv[1])
name = sys.argv[2]
fid = float(sys.argv[3])
elapsed = float(sys.argv[4])

arr.append(
    {
        "scheduler": name,
        "fid": fid,
        "time_seconds": elapsed,
    }
)

print(json.dumps(arr, indent=2, ensure_ascii=False))
PY
)"
}

run_exp "baseline"          "${STAGE2_BASE}/baseline/stage2_refined_scheduler_config.json"
run_exp "prefix_5"          "${STAGE2_BASE}/prefix_5/stage2_refined_scheduler_config.json"
run_exp "prefix_10"         "${STAGE2_BASE}/prefix_10/stage2_refined_scheduler_config.json"
run_exp "prefix_15"         "${STAGE2_BASE}/prefix_15/stage2_refined_scheduler_config.json"
run_exp "first_input_only"  "${STAGE2_BASE}/first_input_only/stage2_refined_scheduler_config.json"
run_exp "combined_5"        "${STAGE2_BASE}/combined_5/stage2_refined_scheduler_config.json"
run_exp "combined_10"       "${STAGE2_BASE}/combined_10/stage2_refined_scheduler_config.json"
run_exp "combined_15"       "${STAGE2_BASE}/combined_15/stage2_refined_scheduler_config.json"

echo "${results}" > "${RESULT_JSON}"
echo "Wrote aggregated FID results to ${RESULT_JSON}"

