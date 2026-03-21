#!/usr/bin/env bash
# 一鍵執行 SVD 完整流程：31 個 block
# 1. 階段 A：收集 feature（31 次）
# 2. 階段 B：計算 SVD 指標（所有 block）
# 3. 階段 C：SVD vs similarity 相關性分析（所有 block）
# 前置：需先跑過 run_similarity_experiments.sh，產生 T_100/Res/result_npz/*.npz

set -e

PYTHON=python3
NUM_STEPS=100
TARGET_N=32
LOG_DIR="QATcode/cache_method/SVD/logs"
mkdir -p "$LOG_DIR"

# GPU 記憶體較吃緊的 block，降低樣本數
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

echo "=============================="
echo "SVD 完整流程（31 blocks）"
echo "  A: 收集 feature (N=$TARGET_N, T=$NUM_STEPS)"
echo "  B: SVD 指標"
echo "  C: SVD vs similarity 相關性"
echo "=============================="

# ---------- 階段 A：收集 feature ----------
echo ""
echo "[階段 A] 收集 Feature（31 個 block）..."
for BLOCK in "${BLOCKS[@]}"; do
  SAFE_NAME=$(echo "$BLOCK" | tr '.' '_')
  LOG_FILE="$LOG_DIR/svd_feature_${SAFE_NAME}.log"
  N=${OVERRIDE_N[$BLOCK]:-$TARGET_N}
  echo "  → $BLOCK  (N=$N)"
  $PYTHON QATcode/cache_method/SVD/collect_features_for_svd.py \
    --num_steps "$NUM_STEPS" \
    --svd_target_block "$BLOCK" \
    --svd_target_N "$N" \
    --svd_output_root "QATcode/cache_method/SVD" \
    --log_file "$LOG_FILE" \
    2>&1 | tee -a "$LOG_FILE"
done
echo "✅ 階段 A 完成"

# ---------- 階段 B：SVD 指標 ----------
echo ""
echo "[階段 B] 計算 SVD 指標（所有 block）..."
$PYTHON QATcode/cache_method/SVD/svd_metrics.py \
  --all \
  --feature_root "QATcode/cache_method/SVD/svd_features" \
  --output_root "QATcode/cache_method/SVD/svd_metrics" \
  --representative-t -1 \
  --energy-threshold 0.98
echo "✅ 階段 B 完成"

# ---------- 階段 C：相關性分析 ----------
echo ""
echo "[階段 C] SVD vs similarity 相關性（所有 block）..."
SIM_NPZ_DIR="QATcode/cache_method/L1_L2_cosine/T_100/Res/result_npz"
if [[ ! -d "$SIM_NPZ_DIR" ]]; then
  echo "警告：$SIM_NPZ_DIR 不存在，請先執行 run_similarity_experiments.sh"
  echo "階段 C 將跳過缺少 npz 的 block"
fi
$PYTHON QATcode/cache_method/SVD/correlate_svd_similarity.py \
  --all \
  --svd_metrics_dir "QATcode/cache_method/SVD/svd_metrics" \
  --similarity_npz_dir "$SIM_NPZ_DIR" \
  --output_root "QATcode/cache_method/SVD/correlation" \
  --plot
echo "✅ 階段 C 完成"

echo ""
echo "=============================="
echo "SVD 完整流程結束"
echo "  輸出：svd_features/, svd_metrics/, correlation/"
echo "=============================="
