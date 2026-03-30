#!/usr/bin/env bash
# 逐一執行所有 block 的單 block 流程（A -> B -> C）
# 預設低磁碟模式：每個 block 不寫 svd_features/*.pt（in-memory 直算）
# similarity NPZ 路徑以 run_single_block_pipeline.sh 的正式設定為準：
# QATcode/cache_method/a_L1_L2_cosine/T_100/v2_latest/result_npz
#
# 用法：
#   bash QATcode/cache_method/SVD/run_svd_all_blocks_single_pipeline.sh [target_N] [start_from_block]
# 參數：
#   $1 = target_N（選填，預設 32）
#   $2 = start_from_block（選填，從該 block 開始跑，用於續跑）
#
# 範例：
#   bash QATcode/cache_method/SVD/run_svd_all_blocks_single_pipeline.sh
#   bash QATcode/cache_method/SVD/run_svd_all_blocks_single_pipeline.sh 32 model.output_blocks.7

set -e

TARGET_N="${1:-32}"
START_FROM_BLOCK="${2:-}"
RUN_SINGLE_SCRIPT="QATcode/cache_method/SVD/run_single_block_pipeline.sh"

# GPU 記憶體較喫緊的 block，降低樣本數
declare -A OVERRIDE_N=(
  ["model.output_blocks.11"]=16
)

# 31 個 block（與 similarity 一致）
BLOCKS=(
  "model.input_blocks.0"
  "model.input_blocks.1"
  "model.input_blocks.2"
  "model.input_blocks.3"
  "model.input_blocks.4"
  "model.input_blocks.5"
  "model.input_blocks.6"
  "model.input_blocks.7"
  "model.input_blocks.8"
  "model.input_blocks.9"
  "model.input_blocks.10"
  "model.input_blocks.11"
  "model.input_blocks.12"
  "model.input_blocks.13"
  "model.input_blocks.14"
  "model.middle_block"
  "model.output_blocks.0"
  "model.output_blocks.1"
  "model.output_blocks.2"
  "model.output_blocks.3"
  "model.output_blocks.4"
  "model.output_blocks.5"
  "model.output_blocks.6"
  "model.output_blocks.7"
  "model.output_blocks.8"
  "model.output_blocks.9"
  "model.output_blocks.10"
  "model.output_blocks.11"
  "model.output_blocks.12"
  "model.output_blocks.13"
  "model.output_blocks.14"
)

if [[ ! -f "$RUN_SINGLE_SCRIPT" ]]; then
  echo "錯誤：找不到 $RUN_SINGLE_SCRIPT"
  exit 1
fi

if [[ -n "$START_FROM_BLOCK" ]]; then
  FOUND_START=0
  for BLOCK in "${BLOCKS[@]}"; do
    if [[ "$BLOCK" == "$START_FROM_BLOCK" ]]; then
      FOUND_START=1
      break
    fi
  done
  if [[ "$FOUND_START" != "1" ]]; then
    echo "錯誤：start_from_block 不在 block 清單中：$START_FROM_BLOCK"
    exit 1
  fi
fi

echo "=============================="
echo "SVD 全 block 單流程執行"
echo "  Total blocks: ${#BLOCKS[@]}"
echo "  Default target_N: $TARGET_N"
echo "  Feature mode: in-memory (no .pt files)"
if [[ -n "$START_FROM_BLOCK" ]]; then
  echo "  Start from: $START_FROM_BLOCK"
fi
echo "=============================="

STARTED=0
DONE_COUNT=0

for BLOCK in "${BLOCKS[@]}"; do
  if [[ -n "$START_FROM_BLOCK" && "$STARTED" != "1" ]]; then
    if [[ "$BLOCK" == "$START_FROM_BLOCK" ]]; then
      STARTED=1
    else
      continue
    fi
  fi

  N="${OVERRIDE_N[$BLOCK]:-$TARGET_N}"
  echo ""
  echo "--------------------------------------------------"
  echo "Running block: $BLOCK  (N=$N)"
  echo "--------------------------------------------------"

  bash "$RUN_SINGLE_SCRIPT" "$BLOCK" "$N"
  DONE_COUNT=$((DONE_COUNT + 1))
done

echo ""
echo "=============================="
echo "全部完成"
echo "  已執行 blocks: $DONE_COUNT"
echo "=============================="
