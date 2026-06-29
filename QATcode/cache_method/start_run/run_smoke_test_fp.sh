#!/usr/bin/env bash
# FP Diff-AE + S3-Cache smoke test（少量圖確認 cache 路徑不 crash）。
#
# 用法：
#   bash QATcode/cache_method/start_run/run_smoke_test_fp.sh
#   SCHED_JSON=... bash QATcode/cache_method/start_run/run_smoke_test_fp.sh
#
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON="${PYTHON:-/home/jimmy/anaconda3/envs/diffae_bw/bin/python}"
OUT_BASE="${OUT_BASE:-QATcode/cache_method/Stage2/stage2_output_fp}"
SCHED_JSON="${SCHED_JSON:-${OUT_BASE}/K25_sw2_lam0.5/baseline/stage2_refined_scheduler_config.json}"
RESULTS_ROOT="${RESULTS_ROOT:-QATcode/cache_method/start_run/results/fp_smoke}"
NUM_STEPS="${NUM_STEPS:-100}"
EVAL_SAMPLES="${EVAL_SAMPLES:-100}"
SEED="${SEED:-0}"
NAME="${NAME:-fp_smoke_k25_sw2_baseline}"

if [[ ! -f "${SCHED_JSON}" ]]; then
  echo "ERROR: refined scheduler not found: ${SCHED_JSON}" >&2
  echo "Run Stage2 first: bash QATcode/cache_method/Stage2/run_stage2_fp_experiments.sh" >&2
  exit 1
fi

date_str="$(date +%Y%m%d)"
time_str="$(date +%m%d_%H)"
run_dir="${RESULTS_ROOT}/${date_str}/${NAME}/${time_str}_${NAME}"
mkdir -p "${run_dir}"

echo "================================================================"
echo "FP smoke test"
echo "  scheduler: ${SCHED_JSON}"
echo "  eval_samples: ${EVAL_SAMPLES}"
echo "  output: ${run_dir}"
echo "================================================================"

"${PYTHON}" QATcode/cache_method/start_run/sample_stage2_cache_scheduler.py \
  --fp \
  --num_steps "${NUM_STEPS}" \
  --eval_samples "${EVAL_SAMPLES}" \
  --seed "${SEED}" \
  --use_cache_scheduler \
  --cache_scheduler_json "${SCHED_JSON}" \
  --scheduler-name "${NAME}" \
  --run-output-dir "${run_dir}" \
  --runs-index-path "${RESULTS_ROOT}/runs_index.jsonl" \
  --log_file "${run_dir}/run.log"

echo ""
echo "✅ Smoke test OK"
echo "   summary: ${run_dir}/summary.json"
if [[ -f "${run_dir}/summary.json" ]]; then
  cat "${run_dir}/summary.json"
fi
