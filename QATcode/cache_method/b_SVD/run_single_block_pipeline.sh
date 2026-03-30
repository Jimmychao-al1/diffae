#!/usr/bin/env bash
# 對單一 block 執行 SVD 完整流程（A → B → C）
# 用法：bash QATcode/cache_method/SVD/run_single_block_pipeline.sh "model.output_blocks.11" [N]
#   $1 = block 名稱（必填）
#   $2 = 樣本數 N（選填，預設 32）
# 說明：
#   - 低磁碟模式：不寫 svd_features/*.pt，A 收集後直接在記憶體跑 B/C
#   - similarity NPZ 正式路徑：QATcode/cache_method/a_L1_L2_cosine/T_100/v2_latest/result_npz

set -e

if [[ -z "$1" ]]; then
  echo "用法: $0 <block_name> [target_N]"
  echo "範例: $0 model.output_blocks.11 16"
  exit 1
fi

BLOCK="$1"
TARGET_N="${2:-32}"
SAFE_NAME=$(echo "$BLOCK" | tr '.' '_')
LOG_DIR="QATcode/cache_method/SVD/logs"
SIM_NPZ="QATcode/cache_method/a_L1_L2_cosine/T_100/v2_latest/result_npz/${SAFE_NAME}.npz"
mkdir -p "$LOG_DIR"

echo "=============================="
echo "SVD 完整流程（單一 block, in-memory）"
echo "  Block: $BLOCK"
echo "  N: $TARGET_N"
echo "  Similarity NPZ: $SIM_NPZ"
echo "=============================="

if [[ ! -f "$SIM_NPZ" ]]; then
  echo "錯誤：找不到 similarity npz：$SIM_NPZ"
  echo "請先確認 a_L1_L2_cosine 的結果存在。"
  exit 1
fi

echo ""
echo "[階段 A->B->C] in-memory 執行中..."
python QATcode/cache_method/SVD/collect_features_for_svd.py \
  --num_steps 100 \
  --svd_target_block "$BLOCK" \
  --svd_target_N "$TARGET_N" \
  --svd_output_root "QATcode/cache_method/SVD" \
  --in_memory_pipeline \
  --representative-t -1 \
  --energy-threshold 0.98 \
  --similarity_npz "$SIM_NPZ" \
  --log_file "$LOG_DIR/svd_feature_${SAFE_NAME}.log"

echo ""
echo "=============================="
echo "完成！輸出位置："
echo "  Features:    不落地（in-memory，無 .pt）"
echo "  SVD JSON:    svd_metrics/${SAFE_NAME}.json"
echo "  Correlation: correlation/${SAFE_NAME}.json"
echo "  Figures:     correlation/figures/${SAFE_NAME}_*.png"
echo "=============================="
