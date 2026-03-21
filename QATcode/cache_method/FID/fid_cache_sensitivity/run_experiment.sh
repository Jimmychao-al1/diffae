#!/bin/bash
# FID Cache Sensitivity Analysis - 完整實驗腳本
# 使用方法: bash run_experiment.sh

set -e

SCRIPT="QATcode/fid_cache_sensitivity/fid_cache_sensitivity.py"
LOG_DIR="QATcode/fid_cache_sensitivity"

echo "=========================================="
echo "FID Cache Sensitivity - 完整實驗"
echo "T = 20, 100"
echo "k = 3, 4, 5"
echo "總計: 2 baselines + 186 個實驗"
echo "=========================================="
echo ""

# 遍歷所有 T 和 k 組合
for T in 20 100; do
    LOG_FILE="${LOG_DIR}/fid_sensitivity_T${T}.log"
    
    echo "=========================================="
    echo "開始 T=$T 實驗"
    echo "=========================================="
    
    # 1. 跑 baseline
    echo "[T=$T] 執行 Baseline..."
    python3 $SCRIPT --num_steps $T --baseline --log_file "$LOG_FILE" 2>&1 | tee -a "$LOG_FILE"
    
    # 2. 遍歷所有 k 值
    for K in 3 4 5; do
        echo ""
        echo "[T=$T, k=$K] 開始實驗..."
        python3 $SCRIPT --num_steps $T --k $K --log_file "$LOG_FILE" 2>&1 | tee -a "$LOG_FILE"
        echo "[T=$T, k=$K] 完成!"
    done
    
    echo ""
    echo "✅ T=$T 所有實驗完成"
    echo ""
done

echo "=========================================="
echo "🎉 全部實驗完成！"
echo "結果: ${LOG_DIR}/fid_sensitivity_results.json"
echo "=========================================="