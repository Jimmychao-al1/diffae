#!/usr/bin/env bash
set -e

PYTHON=python3
SCRIPT_V2="QATcode/cache_method/L1_L2_cosine/similarity_calculation.py"
SCRIPT_V1="QATcode/cache_method/L1_L2_cosine/similarity_calculation_v1.py"

# 基本設定
NUM_STEPS=100          # T=100
SAMPLES=128             # 生成樣本總數 (batch size = 32)
COLLECT_SAMPLES=20    # 實際收集用於相似度計算的樣本數 (每個batch)
SAMPLE_STRATEGY="random"  # 採樣策略: first=取前N個(最快), random=隨機選擇(增加多樣性), uniform=均勻分佈
LOG_ROOT="QATcode/cache_method/L1_L2_cosine/logs"
LOG_DIR_V2="$LOG_ROOT/v2"
LOG_DIR_V1="$LOG_ROOT/v1"
mkdir -p "$LOG_DIR_V2" "$LOG_DIR_V1"

# 31 個 block 名稱（與 model.named_modules() 一致）
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

for BLOCK in "${BLOCKS[@]}"; do
  SAFE_NAME=$(echo "$BLOCK" | tr '.' '_')
  LOG_FILE_V2="$LOG_DIR_V2/similarity_${SAFE_NAME}.log"
  LOG_FILE_V1="$LOG_DIR_V1/similarity_${SAFE_NAME}.log"

  echo "=============================="
  echo "Running v2 similarity for block: $BLOCK"
  echo "Log: $LOG_FILE_V2"
  echo "=============================="

  $PYTHON "$SCRIPT_V2" \
    --mode float \
    --num_steps "$NUM_STEPS" \
    --enable_similarity \
    --similarity_samples "$SAMPLES" \
    --similarity_collect_samples "$COLLECT_SAMPLES" \
    --similarity_sample_strategy "$SAMPLE_STRATEGY" \
    --similarity_target_block "$BLOCK" \
    --similarity_output_root "QATcode/cache_method/L1_L2_cosine" \
    --similarity_dtype float16 \
    --log_file "$LOG_FILE_V2" \
    2>&1 | tee -a "$LOG_FILE_V2"

  #echo "=============================="
  #echo "Running v1 similarity for block: $BLOCK"
  #echo "Log: $LOG_FILE_V1"
  #echo "=============================="
#
  #$PYTHON "$SCRIPT_V1" \
  #  --mode float \
  #  --num_steps "$NUM_STEPS" \
  #  --enable_similarity \
  #  --similarity_samples "$SAMPLES" \
  #  --similarity_collect_samples "$COLLECT_SAMPLES" \
  #  --similarity_sample_strategy "$SAMPLE_STRATEGY" \
  #  --similarity_target_block "$BLOCK" \
  #  --similarity_output_root "QATcode/cache_method/L1_L2_cosine" \
  #  --similarity_dtype float16 \
  #  --log_file "$LOG_FILE_V1" \
  #  2>&1 | tee -a "$LOG_FILE_V1"
done

echo "All similarity experiments finished."