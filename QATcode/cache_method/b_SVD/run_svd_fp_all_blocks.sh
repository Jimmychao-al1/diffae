#!/usr/bin/env bash
# FP Diff-AE SVD evidence for all 31 blocks (Stage 0b)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON="${PYTHON:-/home/jimmy/anaconda3/envs/diffae_bw/bin/python}"
TARGET_N="${1:-32}"
START_FROM_BLOCK="${2:-}"
SIM_NPZ_ROOT="QATcode/cache_method/a_L1_L2_cosine/T_100/fp_v1/result_npz"
LOG_DIR="QATcode/cache_method/b_SVD/logs/fp_v1"
mkdir -p "$LOG_DIR"

declare -A OVERRIDE_N=(
  ["model.output_blocks.11"]=16
)

BLOCKS=(
  "model.input_blocks.0" "model.input_blocks.1" "model.input_blocks.2"
  "model.input_blocks.3" "model.input_blocks.4" "model.input_blocks.5"
  "model.input_blocks.6" "model.input_blocks.7" "model.input_blocks.8"
  "model.input_blocks.9" "model.input_blocks.10" "model.input_blocks.11"
  "model.input_blocks.12" "model.input_blocks.13" "model.input_blocks.14"
  "model.middle_block"
  "model.output_blocks.0" "model.output_blocks.1" "model.output_blocks.2"
  "model.output_blocks.3" "model.output_blocks.4" "model.output_blocks.5"
  "model.output_blocks.6" "model.output_blocks.7" "model.output_blocks.8"
  "model.output_blocks.9" "model.output_blocks.10" "model.output_blocks.11"
  "model.output_blocks.12" "model.output_blocks.13" "model.output_blocks.14"
)

echo "================================================================"
echo "FP SVD — all blocks (Stage 0b)"
echo "Output: QATcode/cache_method/b_SVD/svd_metrics_fp/"
echo "================================================================"

STARTED=0
for BLOCK in "${BLOCKS[@]}"; do
  if [[ -n "$START_FROM_BLOCK" && "$STARTED" != "1" ]]; then
    if [[ "$BLOCK" == "$START_FROM_BLOCK" ]]; then
      STARTED=1
    else
      continue
    fi
  fi

  SAFE_NAME=$(echo "$BLOCK" | tr '.' '_')
  N="${OVERRIDE_N[$BLOCK]:-$TARGET_N}"
  SIM_NPZ="${SIM_NPZ_ROOT}/${SAFE_NAME}.npz"

  if [[ ! -f "$SIM_NPZ" ]]; then
    echo "錯誤：找不到 similarity npz：$SIM_NPZ"
    exit 1
  fi

  echo "--------------------------------------------------"
  echo "Block: $BLOCK  N=$N"
  echo "--------------------------------------------------"

  "$PYTHON" QATcode/cache_method/b_SVD/collect_features_for_svd.py \
    --fp \
    --num_steps 100 \
    --svd_target_block "$BLOCK" \
    --svd_target_N "$N" \
    --svd_output_root "QATcode/cache_method/b_SVD" \
    --in_memory_pipeline \
    --representative-t -1 \
    --energy-threshold 0.98 \
    --similarity_npz "$SIM_NPZ" \
    --skip_correlation \
    --log_file "$LOG_DIR/svd_fp_${SAFE_NAME}.log"
done

echo "✅ FP SVD all-blocks finished."
