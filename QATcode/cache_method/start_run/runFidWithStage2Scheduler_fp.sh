#!/usr/bin/env bash
# FP Diff-AE FID：三組選定 Stage2 baseline refined scheduler。
#
# 須先完成：
#   bash QATcode/cache_method/Stage2/run_stage2_fp_experiments.sh
#
# 用法：
#   bash QATcode/cache_method/start_run/runFidWithStage2Scheduler_fp.sh
#   EVAL_SAMPLES=5000 bash ...   # 5K 快速驗證
#   EVAL_SAMPLES=50000 bash ...  # 正式 50K
#   RUN_BASELINE=0 bash ...      # 略過 no-cache baseline
#
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON="${PYTHON:-/home/jimmy/anaconda3/envs/diffae_bw/bin/python}"
OUT_BASE="${OUT_BASE:-QATcode/cache_method/Stage2/stage2_output_fp}"
RESULTS_ROOT="${RESULTS_ROOT:-QATcode/cache_method/start_run/results/fid_fp}"
NUM_STEPS="${NUM_STEPS:-100}"
EVAL_SAMPLES="${EVAL_SAMPLES:-5000}"
SEED="${SEED:-0}"
RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_ONLY="${RUN_ONLY:-}"

PY=("${PYTHON}" QATcode/cache_method/start_run/sample_stage2_cache_scheduler.py)
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

  echo "===== FP FID | ${name} → ${run_dir} ====="

  local -a extra=(--use_cache_scheduler --cache_scheduler_json "${sched_json}")
  if [[ -z "${sched_json}" ]]; then
    extra=()
  fi

  "${PY[@]}" \
    --fp \
    --num_steps "${NUM_STEPS}" \
    --eval_samples "${EVAL_SAMPLES}" \
    --seed "${SEED}" \
    "${extra[@]}" \
    --scheduler-name "${name}" \
    --run-output-dir "${run_dir}" \
    --runs-index-path "${RUNS_INDEX}" \
    --log_file "${run_dir}/run.log" \
    "$@"

  echo "  → summary: ${run_dir}/summary.json"
}

echo "================================================================"
echo "FP FID × 3 cache configs (+ optional baseline)"
echo "  OUT_BASE=${OUT_BASE}"
echo "  RESULTS_ROOT=${RESULTS_ROOT}"
echo "  EVAL_SAMPLES=${EVAL_SAMPLES}"
echo "  RUN_ONLY=${RUN_ONLY:-<all cache configs>}"
echo "================================================================"

should_run() {
  local name="$1"
  [[ -z "${RUN_ONLY}" || "${RUN_ONLY}" == "${name}" ]]
}

if [[ "${RUN_BASELINE}" == "1" ]] && should_run "fp_baseline_no_cache"; then
  run_one "fp_baseline_no_cache" ""
fi

if should_run "fp_K25_sw2_lam0.50_baseline"; then
  run_one "fp_K25_sw2_lam0.50_baseline" \
    "${OUT_BASE}/K25_sw2_lam0.5/baseline/stage2_refined_scheduler_config.json"
fi

if should_run "fp_K16_sw3_lam0.50_baseline"; then
  run_one "fp_K16_sw3_lam0.50_baseline" \
    "${OUT_BASE}/K16_sw3_lam0.5/baseline/stage2_refined_scheduler_config.json"
fi

if should_run "fp_K25_sw3_lam0.50_baseline"; then
  run_one "fp_K25_sw3_lam0.50_baseline" \
    "${OUT_BASE}/K25_sw3_lam0.5/baseline/stage2_refined_scheduler_config.json"
fi

echo ""
echo "Done. runs_index: ${RUNS_INDEX}"
