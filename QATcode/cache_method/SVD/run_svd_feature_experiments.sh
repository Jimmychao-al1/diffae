#!/usr/bin/env bash
set -e

PYTHON=python3
SCRIPT="QATcode/cache_method/SVD/collect_features_for_svd.py"

# 基本設定
NUM_STEPS=100          # T=100
TARGET_N=32            # 每個 timestep 收集的樣本數（可改為 128）
LOG_DIR="QATcode/cache_method/SVD/logs"
mkdir -p "$LOG_DIR"

# GPU 記憶體較吃緊的 block，降低樣本數
declare -A OVERRIDE_N=(
  ["model.output_blocks.11"]=16
)

# 31 個 block 名稱（與 similarity 一致）
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

echo "=============================="
echo "SVD Feature 收集實驗"
echo "Target N: $TARGET_N"
echo "Timesteps: $NUM_STEPS"
echo "Total blocks: ${#BLOCKS[@]}"
echo "=============================="

for BLOCK in "${BLOCKS[@]}"; do
  SAFE_NAME=$(echo "$BLOCK" | tr '.' '_')
  LOG_FILE="$LOG_DIR/svd_feature_${SAFE_NAME}.log"

  # 若有 override 就用，否則用預設 TARGET_N
  N=${OVERRIDE_N[$BLOCK]:-$TARGET_N}

  echo ""
  echo "=============================="
  echo "Running SVD feature for block: $BLOCK  (N=$N)"
  echo "Log: $LOG_FILE"
  echo "=============================="

  $PYTHON "$SCRIPT" \
    --num_steps "$NUM_STEPS" \
    --svd_target_block "$BLOCK" \
    --svd_target_N "$N" \
    --svd_output_root "QATcode/cache_method/SVD" \
    --log_file "$LOG_FILE" \
    2>&1 | tee -a "$LOG_FILE"
  
  echo "✅ $BLOCK completed"
done

echo ""
echo "=============================="
echo "All SVD feature experiments finished."
echo "=============================="
