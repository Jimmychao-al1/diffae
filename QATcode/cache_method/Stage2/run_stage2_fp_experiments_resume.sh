#!/usr/bin/env bash
# 續跑尚未完成的 FP Stage2（跳過已有 baseline refined JSON 的組合）。
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
  local refined="${baseline}/stage2_refined_scheduler_config.json"

  if [[ -f "${refined}" ]]; then
    echo "⏭ skip ${tag} (already has ${refined})"
    "${PY_VERIFY[@]}" "${refined}" || true
    return 0
  fi

  echo "▶ FP Stage2 | ${tag}"

  mkdir -p "${g}"
  "${PY_REFINE[@]}" --fp --scheduler_config "${sched}" --output_dir "${g}" \
    --seed "${SEED}" --zone_l1_threshold "${ZONE}" --peak_l1_threshold "${PEAK}" \
    --eval-num-images "${EVAL_NUM_IMAGES}" --eval-chunk-size "${EVAL_CHUNK_SIZE}"

  mkdir -p "${th_dir}"
  "${PY_THRESH[@]}" --diagnostics "${g}/stage2_runtime_diagnostics.json" --output "${th_orig}" \
    --q_zone "${Q_ZONE_ORIG}" --q_peak "${Q_PEAK_ORIG}" \
    --peak_over_zone_ratio_min "${PEAK_OVER_ZONE_MIN_ORIG}"

  mkdir -p "${baseline}"
  "${PY_REFINE[@]}" --fp --scheduler_config "${sched}" --output_dir "${baseline}" \
    --seed "${SEED}" --zone_l1_threshold "${ZONE}" --peak_l1_threshold "${PEAK}" \
    --threshold-config "${th_orig}" \
    --eval-num-images "${EVAL_NUM_IMAGES}" --eval-chunk-size "${EVAL_CHUNK_SIZE}"

  "${PY_VERIFY[@]}" "${refined}"
  echo "✅ ${tag}"
}

for entry in "${RUNS[@]}"; do
  run_one_experiment "${entry%%|*}" "${entry#*|}"
done

echo "Done."
