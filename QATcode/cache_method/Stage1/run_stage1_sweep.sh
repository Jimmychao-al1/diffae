#!/usr/bin/env bash
# ============================================================
# Stage-1 topK sweep
#
# 用法：
#   bash run_stage1_sweep.sh              # 跑預設 topK 列表
#   bash run_stage1_sweep.sh 4 6 8 10     # 自訂 topK 值
# ============================================================

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"   # cd 到 repo root

STAGE0_DIR="QATcode/cache_method/Stage0/stage0e_output"
BASE_OUT="QATcode/cache_method/Stage1/stage1_output"
BASE_FIG="QATcode/cache_method/Stage1/stage1_figures"
SCHEDULER="QATcode/cache_method/Stage1/stage1_scheduler.py"
VISUALIZE="QATcode/cache_method/Stage1/visualize_stage1.py"
VERIFY="QATcode/cache_method/Stage1/verify_scheduler.py"

# 如果有命令列引數就用，否則用預設列表
if [ $# -gt 0 ]; then
    TOPK_LIST=("$@")
else
    TOPK_LIST=(4 6 8 10 15 20 25 30 35 40 45 50)
fi

echo "================================================================"
echo "Stage-1 topK sweep"
echo "topK values: ${TOPK_LIST[*]}"
echo "================================================================"
echo ""

for K in "${TOPK_LIST[@]}"; do
    OUT_DIR="${BASE_OUT}/topk_${K}"
    FIG_DIR="${BASE_FIG}/topk_${K}"

    echo "────────────────────────────────────────"
    echo "▶ topK = ${K}"
    echo "  output : ${OUT_DIR}"
    echo "  figures: ${FIG_DIR}"
    echo "────────────────────────────────────────"

    # 1) 執行 Stage-1
    python3 "${SCHEDULER}" \
        --stage0_dir "${STAGE0_DIR}" \
        --output_dir "${OUT_DIR}" \
        --cp_method topk \
        --cp_topk "${K}" \
        --eta 0.8

    # 2) 驗證
    python3 "${VERIFY}" \
        --config "${OUT_DIR}/scheduler_config.json"

    # 3) 可視化
    python3 "${VISUALIZE}" \
        --stage1_output_dir "${OUT_DIR}" \
        --output_dir "${FIG_DIR}"

    echo ""
done

echo "================================================================"
echo "✅  全部完成！結果結構："
echo ""
for K in "${TOPK_LIST[@]}"; do
    echo "  topk_${K}/"
    echo "    config : ${BASE_OUT}/topk_${K}/scheduler_config.json"
    echo "    diag   : ${BASE_OUT}/topk_${K}/scheduler_diagnostics.json"
    echo "    figures: ${BASE_FIG}/topk_${K}/"
done
echo "================================================================"
