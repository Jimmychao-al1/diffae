#!/usr/bin/env bash
# Stage-1 FP sweep：K、smooth_window、lambda、k_max
#
# 用法：
#   bash QATcode/cache_method/Stage1/run_stage1_sweep_fp.sh
# 或：
#   STAGE0_DIR=... BASE_OUT=... bash QATcode/cache_method/Stage1/run_stage1_sweep_fp.sh
#
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON="${PYTHON:-/home/jimmy/anaconda3/envs/diffae_bw/bin/python}"
STAGE0_DIR="${STAGE0_DIR:-QATcode/cache_method/Stage0/stage0e_output_fp}"
BASE_OUT="${BASE_OUT:-QATcode/cache_method/Stage1/stage1_output_fp}"
BASE_FIG="${BASE_FIG:-QATcode/cache_method/Stage1/stage1_figures_fp}"
SCHEDULER="QATcode/cache_method/Stage1/stage1_scheduler.py"
VISUALIZE="QATcode/cache_method/Stage1/visualize_stage1.py"
VERIFY="QATcode/cache_method/Stage1/verify_scheduler.py"
SUMMARIZE="QATcode/cache_method/Stage1/summarize_stage1_sweep.py"

K_LIST=(16 25)
SW_LIST=(2 3)
LAM_LIST=(0.25 0.5 1.0 2.0)
KMAX_LIST=(4)

echo "================================================================"
echo "Stage-1 FP sweep"
echo "STAGE0_DIR=${STAGE0_DIR}"
echo "BASE_OUT=${BASE_OUT}"
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

        "$PYTHON" "${SCHEDULER}" \
          --stage0_dir "${STAGE0_DIR}" \
          --output_dir "${OUT_DIR}" \
          --K "${K}" \
          --smooth_window "${SW}" \
          --lambda "${LAM}" \
          --k_max "${KMAX}"

        "$PYTHON" "${VERIFY}" --config "${OUT_DIR}/scheduler_config.json"
        "$PYTHON" "${VISUALIZE}" --stage1_output_dir "${OUT_DIR}" --output_dir "${FIG_DIR}"
        echo ""
      done
    done
  done
done

"$PYTHON" "${SUMMARIZE}" \
  --base_out "${BASE_OUT}" \
  --output "${BASE_OUT}/stage1_sweep_summary_fp.csv" \
  --prefix "fp_"

echo "================================================================"
echo "✅ FP sweep 完成。"
echo "   結果目錄：${BASE_OUT}/sweep_*"
echo "   總結 CSV：${BASE_OUT}/stage1_sweep_summary_fp.csv"
echo "================================================================"
