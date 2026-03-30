#!/bin/bash
# FID Cache Sensitivity — 完整實驗（單機順序）或分機模式（--mode A / B）
#
# 單機（舊行為）:  bash run_experiment.sh
# 主機 1 模式 A:  bash run_experiment.sh A
# 主機 2 模式 B:  bash run_experiment.sh B
#
# 兩台請使用相同 repo；JSON 預設皆寫入本機
#   QATcode/cache_method/FID/fid_cache_sensitivity/fid_sensitivity_results.json
# 跑完後自行合併兩份 JSON 的 results.T100.k* 欄位（T20 僅 A 有）。
#
# 建議先各自寫好 baseline（或先執行）:
#   python3 QATcode/cache_method/FID/fid_cache_sensitivity/fid_cache_sensitivity.py --num_steps 20 --baseline
#   python3 QATcode/cache_method/FID/fid_cache_sensitivity/fid_cache_sensitivity.py --num_steps 100 --baseline
#
set -e

SCRIPT="QATcode/cache_method/FID/fid_cache_sensitivity/fid_cache_sensitivity.py"
LOG_DIR="QATcode/cache_method/FID/fid_cache_sensitivity"
MODE="${1:-}"

if [[ -n "$MODE" ]]; then
  if [[ "$MODE" != "A" && "$MODE" != "B" ]]; then
    echo "用法: $0 [A|B]"
    echo "  無參數 = 單機完整實驗 (T=20,100 各 baseline + 全部 k)"
    echo "  A      = 分機: T=20 全實驗 + T=100 前 26 個任務"
    echo "  B      = 分機: T=100 第 27～93 個任務"
    exit 1
  fi
  LOG_FILE="${LOG_DIR}/fid_sensitivity_mode_${MODE}.log"
  echo "=========================================="
  echo "FID Cache Sensitivity — 分片模式 ${MODE}"
  echo "Log: ${LOG_FILE}"
  echo "JSON: ${LOG_DIR}/fid_sensitivity_results.json"
  echo "=========================================="
  python3 "$SCRIPT" --mode "$MODE" --log_file "$LOG_FILE" 2>&1 | tee -a "$LOG_FILE"
  echo "✅ 模式 ${MODE} 結束"
  exit 0
fi

echo "=========================================="
echo "FID Cache Sensitivity - 完整實驗（單機）"
echo "T = 20, 100"
echo "k = 3, 4, 5"
echo "總計: 2 baselines + 186 個實驗"
echo "分機請改用: bash $0 A   或   bash $0 B"
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
