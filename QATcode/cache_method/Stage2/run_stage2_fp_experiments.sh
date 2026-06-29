#!/usr/bin/env bash
# FP Diff-AE Stage2：三組選定 sweep 的完整兩趟 refine（global → blockwise baseline）。
#
# 對應 Stage1 run_id：
#   fp_K25_sw2_lam0.50  首選（與 Q-DiffAE K25_sw2 對稱）
#   fp_K16_sw3_lam0.50  激進（最低 ρ）
#   fp_K25_sw3_lam0.50  中間
#
# 用法（repo 根目錄）：
#   bash QATcode/cache_method/Stage2/run_stage2_fp_experiments.sh
#
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON="${PYTHON:-/home/jimmy/anaconda3/envs/diffae_bw/bin/python}"
STAGE1_ROOT="${STAGE1_ROOT:-QATcode/cache_method/Stage1/stage1_output_fp}"
OUT_BASE="${OUT_BASE:-QATcode/cache_method/Stage2/stage2_output_fp}"

SEED="${SEED:-0}"
ZONE="${ZONE:-0.02}"
PEAK="${PEAK:-0.08}"
EVAL_NUM_IMAGES="${EVAL_NUM_IMAGES:-8}"
EVAL_CHUNK_SIZE="${EVAL_CHUNK_SIZE:-1}"

Q_ZONE_ORIG="${Q_ZONE_ORIG:-0.75}"
Q_PEAK_ORIG="${Q_PEAK_ORIG:-0.95}"
PEAK_OVER_ZONE_MIN_ORIG="${PEAK_OVER_ZONE_MIN_ORIG:-1.5}"

PY_REFINE=("${PYTHON}" QATcode/cache_method/Stage2/stage2_runtime_refine.py)
PY_THRESH=("${PYTHON}" QATcode/cache_method/Stage2/build_blockwise_thresholds.py)
PY_VERIFY=("${PYTHON}" QATcode/cache_method/Stage2/verify_stage2.py)

# name | Stage1 scheduler_config relative to STAGE1_ROOT
RUNS=(
  "K25_sw2_lam0.5|sweep_K25_sw2_lam0.5_kmax4/scheduler_config.json"
  "K16_sw3_lam0.5|sweep_K16_sw3_lam0.5_kmax4/scheduler_config.json"
  "K25_sw3_lam0.5|sweep_K25_sw3_lam0.5_kmax4/scheduler_config.json"
)

run_one_experiment() {
  local tag="$1"
  local sched_rel="$2"
  local sched="${STAGE1_ROOT}/${sched_rel}"
  local exp_root="${OUT_BASE}/${tag}"
  local g="${exp_root}/00_global_refine"
  local th_dir="${exp_root}/01_blockwise_threshold"
  local th_orig="${th_dir}/stage2_thresholds_blockwise.json"
  local baseline="${exp_root}/baseline"

  if [[ ! -f "${sched}" ]]; then
    echo "ERROR: missing ${sched}" >&2
    exit 1
  fi

  echo ""
  echo "================================================================"
  echo "FP Stage2 | ${tag}"
  echo "  scheduler: ${sched}"
  echo "  output:    ${exp_root}"
  echo "================================================================"

  echo "→ pass 1 (global thresholds)"
  mkdir -p "${g}"
  "${PY_REFINE[@]}" \
    --fp \
    --scheduler_config "${sched}" \
    --output_dir "${g}" \
    --seed "${SEED}" \
    --zone_l1_threshold "${ZONE}" \
    --peak_l1_threshold "${PEAK}" \
    --eval-num-images "${EVAL_NUM_IMAGES}" \
    --eval-chunk-size "${EVAL_CHUNK_SIZE}"

  echo "→ build_blockwise_thresholds (original quantiles)"
  mkdir -p "${th_dir}"
  "${PY_THRESH[@]}" \
    --diagnostics "${g}/stage2_runtime_diagnostics.json" \
    --output "${th_orig}" \
    --q_zone "${Q_ZONE_ORIG}" \
    --q_peak "${Q_PEAK_ORIG}" \
    --peak_over_zone_ratio_min "${PEAK_OVER_ZONE_MIN_ORIG}"

  echo "→ pass 2 (blockwise baseline)"
  mkdir -p "${baseline}"
  "${PY_REFINE[@]}" \
    --fp \
    --scheduler_config "${sched}" \
    --output_dir "${baseline}" \
    --seed "${SEED}" \
    --zone_l1_threshold "${ZONE}" \
    --peak_l1_threshold "${PEAK}" \
    --threshold-config "${th_orig}" \
    --eval-num-images "${EVAL_NUM_IMAGES}" \
    --eval-chunk-size "${EVAL_CHUNK_SIZE}"

  echo "→ verify refined scheduler"
  "${PY_VERIFY[@]}" "${baseline}/stage2_refined_scheduler_config.json"

  echo "✅ ${tag} → ${baseline}/stage2_refined_scheduler_config.json"
}

echo "================================================================"
echo "FP Stage2 full experiments (3 × baseline refined scheduler)"
echo "  STAGE1_ROOT=${STAGE1_ROOT}"
echo "  OUT_BASE=${OUT_BASE}"
echo "  EVAL_NUM_IMAGES=${EVAL_NUM_IMAGES}"
echo "================================================================"

for entry in "${RUNS[@]}"; do
  tag="${entry%%|*}"
  sched_rel="${entry#*|}"
  run_one_experiment "${tag}" "${sched_rel}"
done

echo ""
echo "================================================================"
echo "✅ FP Stage2 完成。Refined JSON："
echo "   ${OUT_BASE}/K25_sw2_lam0.5/baseline/stage2_refined_scheduler_config.json"
echo "   ${OUT_BASE}/K16_sw3_lam0.5/baseline/stage2_refined_scheduler_config.json"
echo "   ${OUT_BASE}/K25_sw3_lam0.5/baseline/stage2_refined_scheduler_config.json"
echo "================================================================"
