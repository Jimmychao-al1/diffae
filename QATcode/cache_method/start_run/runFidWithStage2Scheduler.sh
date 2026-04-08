#!/usr/bin/env bash
# FID 採樣：對齊 Stage2「六組第二趟」refined scheduler，依序跑 6 次 sample_stage2_cache_scheduler.py。
# 須先完成 QATcode/cache_method/Stage2/run_stage2_full_experiments.sh（或同等產物）。
#
# 用法（於任意目錄）：
#   bash QATcode/cache_method/start_run/runFidWithStage2Scheduler.sh
#
# 可覆寫：EXP_K16、EXP_K25、RESULTS_ROOT、NUM_STEPS、EVAL_SAMPLES、SEED、QUANT_STATE
# 只跑子集：RUN_K16_FID=0 略過 K16 兩筆；RUN_K25_FID=0 略過 K25 四筆
#
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

OUT_BASE="${OUT_BASE:-QATcode/cache_method/Stage2/stage2_output}"
EXP_K16="${EXP_K16:-${OUT_BASE}/fullExperimentsK16sw3}"
EXP_K25="${EXP_K25:-${OUT_BASE}/fullExperimentsK25sw2}"

RESULTS_ROOT="${RESULTS_ROOT:-QATcode/cache_method/results/fid_fullExperiments6}"
NUM_STEPS="${NUM_STEPS:-100}"
EVAL_SAMPLES="${EVAL_SAMPLES:-5000}"
SEED="${SEED:-0}"
QUANT_STATE="${QUANT_STATE:-tt}"

RUN_K16_FID="${RUN_K16_FID:-1}"
RUN_K25_FID="${RUN_K25_FID:-1}"

PY=(python QATcode/cache_method/start_run/sample_stage2_cache_scheduler.py)
RUNS_INDEX="${RESULTS_ROOT}/runs_index.jsonl"

mkdir -p "${RESULTS_ROOT}"

run_one() {
  local name="$1"
  local sched_json="$2"
  shift 2

  local date_str time_str run_dir
  date_str="$(date +%Y%m%d)"
  time_str="$(date +%m%d_%H)"
  run_dir="${RESULTS_ROOT}/${date_str}/${name}/${time_str}_${name}"
  mkdir -p "${run_dir}"

  echo "===== FID | ${name} → ${run_dir} ====="

  "${PY[@]}" \
    --mode float \
    --num_steps "${NUM_STEPS}" \
    --eval_samples "${EVAL_SAMPLES}" \
    --seed "${SEED}" \
    --quant-state "${QUANT_STATE}" \
    --use_cache_scheduler \
    --cache_scheduler_json "${sched_json}" \
    --scheduler-name "${name}" \
    --run-output-dir "${run_dir}" \
    --runs-index-path "${RUNS_INDEX}" \
    --log_file "${run_dir}/run.log" \
    "$@"

  echo "  → summary: ${run_dir}/summary.json"
}

echo "================================================================"
echo "FID × 6 | EXP_K16=${EXP_K16}"
echo "         | EXP_K25=${EXP_K25}"
echo "         | RESULTS_ROOT=${RESULTS_ROOT}"
echo "================================================================"

# --- K16_sw3：2 組（original blockwise threshold）---
#if [[ "${RUN_K16_FID}" == "1" ]]; then
#  run_one "k16_sw3_baseline" "${EXP_K16}/baseline/stage2_refined_scheduler_config.json"
#  run_one "k16_sw3_prefix_15" "${EXP_K16}/prefix_15/stage2_refined_scheduler_config.json" \
#    --force-full-prefix-steps 15
#else
#  echo "(skip K16 FID: RUN_K16_FID=${RUN_K16_FID})"
#fi

# --- K25_sw2：4 組（original ×2 + q90/q80/min1.3 ×2；目錄 min130）---
if [[ "${RUN_K25_FID}" == "1" ]]; then
  #run_one "k25_sw2_baseline" "${EXP_K25}/baseline/stage2_refined_scheduler_config.json"
  #run_one "k25_sw2_prefix_15" "${EXP_K25}/prefix_15/stage2_refined_scheduler_config.json" \
  #  --force-full-prefix-steps 15
  #run_one "k25_sw2_baseline_908030" "${EXP_K25}/baseline_908030/stage2_refined_scheduler_config.json"
  #run_one "k25_sw2_prefix_15_q90q80min130" "${EXP_K25}/prefix_15_q90q80min130/stage2_refined_scheduler_config.json" \
  #  --force-full-prefix-steps 15
  run_one "baseline_908030_prefix_15" "${EXP_K25}/baseline_908030/stage2_refined_scheduler_config.json" \
    --force-full-prefix-steps 15
else
  echo "(skip K25 FID: RUN_K25_FID=${RUN_K25_FID})"
fi

echo ""
echo "Done. runs_index: ${RUNS_INDEX}"
