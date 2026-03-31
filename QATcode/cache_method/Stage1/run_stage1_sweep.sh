#!/usr/bin/env bash
# Stage-1 baseline sweep：K（change points 個數）、smooth_window、lambda、k_max
#
# 用法：
#   bash run_stage1_sweep.sh
# 或覆寫陣列（在腳本内改 K_LIST / SW_LIST / LAM_LIST / KMAX_LIST），或：
#   STAGE0_DIR=... BASE_OUT=... bash run_stage1_sweep.sh
#
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

STAGE0_DIR="${STAGE0_DIR:-QATcode/cache_method/Stage0/stage0e_output}"
BASE_OUT="${BASE_OUT:-QATcode/cache_method/Stage1/stage1_output}"
BASE_FIG="${BASE_FIG:-QATcode/cache_method/Stage1/stage1_figures}"
SCHEDULER="QATcode/cache_method/Stage1/stage1_scheduler.py"
VISUALIZE="QATcode/cache_method/Stage1/visualize_stage1.py"
VERIFY="QATcode/cache_method/Stage1/verify_scheduler.py"

# 預設掃描範圍（可自行改小以縮短時間）
K_LIST=(12 16)
SW_LIST=(3 5)
LAM_LIST=(0.25 0.5 1.0 2.0)
KMAX_LIST=(4)

echo "================================================================"
echo "Stage-1 baseline sweep"
echo "STAGE0_DIR=${STAGE0_DIR}"
echo "K_LIST=( ${K_LIST[*]} )"
echo "SW_LIST=( ${SW_LIST[*]} )"
echo "LAM_LIST=( ${LAM_LIST[*]} )"
echo "KMAX_LIST=( ${KMAX_LIST[*]} )"
echo "================================================================"

for K in "${K_LIST[@]}"; do
  for SW in "${SW_LIST[@]}"; do
    for LAM in "${LAM_LIST[@]}"; do
      for KMAX in "${KMAX_LIST[@]}"; do
        TAG="K${K}_sw${SW}_lam${LAM}_kmax${KMAX}"
        OUT_DIR="${BASE_OUT}/sweep_${TAG}"
        FIG_DIR="${BASE_FIG}/sweep_${TAG}"
        mkdir -p "${OUT_DIR}" "${FIG_DIR}"

        echo "────────────────────────────────────────"
        echo "▶ ${TAG}"
        echo "  output : ${OUT_DIR}"
        echo "────────────────────────────────────────"

        python3 "${SCHEDULER}" \
          --stage0_dir "${STAGE0_DIR}" \
          --output_dir "${OUT_DIR}" \
          --K "${K}" \
          --smooth_window "${SW}" \
          --lambda "${LAM}" \
          --k_max "${KMAX}"

        python3 "${VERIFY}" --config "${OUT_DIR}/scheduler_config.json"
        python3 "${VISUALIZE}" --stage1_output_dir "${OUT_DIR}" --output_dir "${FIG_DIR}"
        echo ""
      done
    done
  done
done

echo "================================================================"
echo "✅ sweep 完成。結果：${BASE_OUT}/sweep_*"
echo "================================================================"
