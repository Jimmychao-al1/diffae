#!/usr/bin/env bash
# FID@50K：Q-DiffAE baseline（full compute）+ k25_sw2_baseline_908030（S3-Cache）
#
# 用法（repo root 或任意目錄）：
#   bash QATcode/cache_method/start_run/runFid50K_baseline_and_908030.sh
#
# 可覆寫：RESULTS_ROOT, NUM_STEPS, EVAL_SAMPLES, SEED, QUANT_STATE, EXP_K25
# 只跑其一：RUN_BASELINE=0 或 RUN_CACHE_908030=0
#
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

OUT_BASE="${OUT_BASE:-QATcode/cache_method/Stage2/stage2_output}"
EXP_K25="${EXP_K25:-${OUT_BASE}/fullExperimentsK25sw2}"

RESULTS_ROOT="${RESULTS_ROOT:-QATcode/cache_method/results/fid_fullExperiments6}"
NUM_STEPS="${NUM_STEPS:-100}"
EVAL_SAMPLES="${EVAL_SAMPLES:-50000}"
SEED="${SEED:-0}"
QUANT_STATE="${QUANT_STATE:-tt}"

RUN_BASELINE="${RUN_BASELINE:-1}"
RUN_CACHE_908030="${RUN_CACHE_908030:-1}"

PY=(python QATcode/cache_method/start_run/sample_stage2_cache_scheduler.py)
RUNS_INDEX="${RESULTS_ROOT}/runs_index.jsonl"

mkdir -p "${RESULTS_ROOT}"

run_baseline() {
  local name="qat_baseline_no_cache"
  local date_str time_str run_dir
  date_str="$(date +%Y%m%d)"
  time_str="$(date +%m%d_%H)"
  run_dir="${RESULTS_ROOT}/${date_str}/${name}/${time_str}_${name}"
  mkdir -p "${run_dir}"

  echo "===== FID@50K baseline (v2 full compute) → ${run_dir} ====="

  "${PY[@]}" \
    --mode float \
    --num_steps "${NUM_STEPS}" \
    --eval_samples "${EVAL_SAMPLES}" \
    --seed "${SEED}" \
    --quant-state "${QUANT_STATE}" \
    --scheduler-name "${name}" \
    --run-output-dir "${run_dir}" \
    --runs-index-path "${RUNS_INDEX}" \
    --log_file "${run_dir}/run.log"

  echo "  → summary: ${run_dir}/summary.json"
}

run_cache_908030() {
  local name="k25_sw2_baseline_908030"
  local sched_json="${EXP_K25}/baseline_908030/stage2_refined_scheduler_config.json"
  local date_str time_str run_dir
  date_str="$(date +%Y%m%d)"
  time_str="$(date +%m%d_%H)"
  run_dir="${RESULTS_ROOT}/${date_str}/${name}/${time_str}_${name}"
  mkdir -p "${run_dir}"

  if [[ ! -f "${sched_json}" ]]; then
    echo "ERROR: missing Stage2 scheduler: ${sched_json}" >&2
    exit 1
  fi

  echo "===== FID@50K cache ${name} → ${run_dir} ====="

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
    --log_file "${run_dir}/run.log"

  echo "  → summary: ${run_dir}/summary.json"
}

echo "================================================================"
echo "FID@50K × 2 | RESULTS_ROOT=${RESULTS_ROOT}"
echo "            | T=${NUM_STEPS} eval=${EVAL_SAMPLES} seed=${SEED}"
echo "            | runs_index=${RUNS_INDEX}"
echo "================================================================"

if [[ "${RUN_BASELINE}" == "1" ]]; then
  run_baseline
else
  echo "(skip baseline: RUN_BASELINE=${RUN_BASELINE})"
fi

if [[ "${RUN_CACHE_908030}" == "1" ]]; then
  run_cache_908030
else
  echo "(skip cache: RUN_CACHE_908030=${RUN_CACHE_908030})"
fi

echo ""
echo "Done. runs_index: ${RUNS_INDEX}"
