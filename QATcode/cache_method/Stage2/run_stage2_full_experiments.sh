#!/usr/bin/env bash
# Stage2：依序完成 6 組實驗的「第二趟」refined scheduler（blockwise threshold 後的變體）。
#
# 實驗設計（Stage1 輸出固定）：
#   K16_sw3: global refine → blockwise(original) → baseline & prefix_15
#   K25_sw2: 同上 → original → baseline & prefix_15
#   K25_sw2: 同上 → q_zone=0.90, q_peak=0.80, peak_over_zone_min=1.3（慣称 908030）→ baseline_908030 & prefix_15（第二組閾值／目錄）
#
# 用法（於 repo 根目錄）：
#   bash QATcode/cache_method/Stage2/run_stage2_full_experiments.sh
#
# 可覆寫環境變數見下方 DEFAULTS，或只跑子集：RUN_K16=1 RUN_K25=0 ...
# 舊版曾用檔名 *q90_q80_min30*／目錄 *prefix_15_q90q80min30* 易與 peak_min=0.3 混淆；預設已改為 peak_over_zone_min=1.3 → *min130*。
#
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

# ── defaults（相對於 repo 根）────────────────────────────────────────────
STAGE1_ROOT="${STAGE1_ROOT:-QATcode/cache_method/Stage1/stage1_output}"
SCHED_K16="${SCHED_K16:-${STAGE1_ROOT}/sweep_K16_sw3_lam1.0_kmax4/scheduler_config.json}"
SCHED_K25="${SCHED_K25:-${STAGE1_ROOT}/sweep_K25_sw2_lam0.5_kmax4/scheduler_config.json}"

OUT_BASE="${OUT_BASE:-QATcode/cache_method/Stage2/stage2_output}"
EXP_K16="${EXP_K16:-${OUT_BASE}/fullExperimentsK16sw3}"
EXP_K25="${EXP_K25:-${OUT_BASE}/fullExperimentsK25sw2}"

SEED="${SEED:-0}"
ZONE="${ZONE:-0.02}"
PEAK="${PEAK:-0.08}"

# Stage2 診斷用影像數（較大較穩、較慢）
EVAL_NUM_IMAGES="${EVAL_NUM_IMAGES:-8}"
EVAL_CHUNK_SIZE="${EVAL_CHUNK_SIZE:-1}"

# build_blockwise_thresholds：original（預設與 build_blockwise_thresholds.py 一致）
Q_ZONE_ORIG="${Q_ZONE_ORIG:-0.75}"
Q_PEAK_ORIG="${Q_PEAK_ORIG:-0.95}"
PEAK_OVER_ZONE_MIN_ORIG="${PEAK_OVER_ZONE_MIN_ORIG:-1.5}"

# 「908030」：q_zone=0.90, q_peak=0.80, peak_over_zone_ratio_min=1.3（檔名用 min130）
Q_ZONE_9080="${Q_ZONE_9080:-0.90}"
Q_PEAK_9080="${Q_PEAK_9080:-0.80}"
PEAK_OVER_ZONE_MIN_9080="${PEAK_OVER_ZONE_MIN_9080:-1.3}"

PY_REFINE=(python QATcode/cache_method/Stage2/stage2_runtime_refine.py)
PY_THRESH=(python QATcode/cache_method/Stage2/build_blockwise_thresholds.py)

RUN_K16="${RUN_K16:-1}"
RUN_K25="${RUN_K25:-1}"

run_pass1_global_refine() {
  local sched="$1"
  local out_global="$2"
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Stage2 pass 1 (global thresholds, diagnostics) → ${out_global}"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  mkdir -p "${out_global}"
  "${PY_REFINE[@]}" \
    --scheduler_config "${sched}" \
    --output_dir "${out_global}" \
    --seed "${SEED}" \
    --zone_l1_threshold "${ZONE}" \
    --peak_l1_threshold "${PEAK}" \
    --eval-num-images "${EVAL_NUM_IMAGES}" \
    --eval-chunk-size "${EVAL_CHUNK_SIZE}"
}

build_thresh_orig() {
  local diag="$1"
  local out_json="$2"
  echo ""
  echo "→ build_blockwise_thresholds (original quantiles) → ${out_json}"
  mkdir -p "$(dirname "${out_json}")"
  "${PY_THRESH[@]}" \
    --diagnostics "${diag}" \
    --output "${out_json}" \
    --q_zone "${Q_ZONE_ORIG}" \
    --q_peak "${Q_PEAK_ORIG}" \
    --peak_over_zone_ratio_min "${PEAK_OVER_ZONE_MIN_ORIG}"
}

build_thresh_q90q80() {
  local diag="$1"
  local out_json="$2"
  echo ""
  echo "→ build_blockwise_thresholds (q90/q80/min1.3) → ${out_json}"
  mkdir -p "$(dirname "${out_json}")"
  "${PY_THRESH[@]}" \
    --diagnostics "${diag}" \
    --output "${out_json}" \
    --q_zone "${Q_ZONE_9080}" \
    --q_peak "${Q_PEAK_9080}" \
    --peak_over_zone_ratio_min "${PEAK_OVER_ZONE_MIN_9080}"
}

run_pass2_blockwise() {
  local sched="$1"
  local out_dir="$2"
  local thresh="$3"
  shift 3
  echo ""
  echo "Stage2 pass 2 (blockwise) → ${out_dir}"
  mkdir -p "${out_dir}"
  "${PY_REFINE[@]}" \
    --scheduler_config "${sched}" \
    --output_dir "${out_dir}" \
    --seed "${SEED}" \
    --zone_l1_threshold "${ZONE}" \
    --peak_l1_threshold "${PEAK}" \
    --threshold-config "${thresh}" \
    --eval-num-images "${EVAL_NUM_IMAGES}" \
    --eval-chunk-size "${EVAL_CHUNK_SIZE}" \
    "$@"
}

# ── K16_sw3：2 個第二趟變體 ───────────────────────────────────────────────
#run_k16() {
#  local g="${EXP_K16}/00_global_refine"
#  local th_dir="${EXP_K16}/01_blockwise_threshold"
#  local th_orig="${th_dir}/stage2_thresholds_blockwise.json"
#
#  run_pass1_global_refine "${SCHED_K16}" "${g}"
#  build_thresh_orig "${g}/stage2_runtime_diagnostics.json" "${th_orig}"
#
#  run_pass2_blockwise "${SCHED_K16}" "${EXP_K16}/baseline" "${th_orig}"
#  run_pass2_blockwise "${SCHED_K16}" "${EXP_K16}/prefix_15" "${th_orig}" \
#    --force-full-prefix-steps 15
#}

# ── K25_sw2：4 個第二趟變體（共用同一個 pass1 diagnostics）────────────────
run_k25() {
  local g="${EXP_K25}/00_global_refine"
  local th_dir="${EXP_K25}/01_blockwise_threshold"
  local th_orig="${th_dir}/stage2_thresholds_blockwise.json"
  local th_9080="${th_dir}/stage2_thresholds_blockwise_q90_q80_min130.json"

  #run_pass1_global_refine "${SCHED_K25}" "${g}"
  #build_thresh_orig "${g}/stage2_runtime_diagnostics.json" "${th_orig}"
  build_thresh_q90q80 "${g}/stage2_runtime_diagnostics.json" "${th_9080}"

  #run_pass2_blockwise "${SCHED_K25}" "${EXP_K25}/baseline" "${th_orig}"
  #run_pass2_blockwise "${SCHED_K25}" "${EXP_K25}/prefix_15" "${th_orig}" \
  #  --force-full-prefix-steps 15

  run_pass2_blockwise "${SCHED_K25}" "${EXP_K25}/baseline_908030" "${th_9080}"
  run_pass2_blockwise "${SCHED_K25}" "${EXP_K25}/prefix_15_q90q80min130" "${th_9080}" \
    --force-full-prefix-steps 15
}

echo "================================================================"
echo "Stage2 full experiments (6 second-pass refined schedulers)"
echo "  SCHED_K16=${SCHED_K16}"
echo "  SCHED_K25=${SCHED_K25}"
echo "  EXP_K16=${EXP_K16}"
echo "  EXP_K25=${EXP_K25}"
echo "  EVAL_NUM_IMAGES=${EVAL_NUM_IMAGES} EVAL_CHUNK_SIZE=${EVAL_CHUNK_SIZE}"
echo "================================================================"

if [[ ! -f "${SCHED_K16}" ]]; then
  echo "ERROR: missing Stage1 scheduler: ${SCHED_K16}" >&2
  exit 1
fi
if [[ ! -f "${SCHED_K25}" ]]; then
  echo "ERROR: missing Stage1 scheduler: ${SCHED_K25}" >&2
  exit 1
fi

#if [[ "${RUN_K16}" == "1" ]]; then
#  run_k16
#else
#  echo "(skip K16: RUN_K16=${RUN_K16})"
#fi

if [[ "${RUN_K25}" == "1" ]]; then
  run_k25
else
  echo "(skip K25: RUN_K25=${RUN_K25})"
fi

echo ""
echo "================================================================"
echo "✅ 完成。各變體目錄內應有 stage2_refined_scheduler_config.json"
echo "   K16: ${EXP_K16}/{baseline,prefix_15}"
echo "   K25: ${EXP_K25}/{baseline,prefix_15,baseline_908030,prefix_15_q90q80min130}"
echo "詳見 QATcode/cache_method/Stage2/stage2ExperimentsGuide.md"
echo "================================================================"
