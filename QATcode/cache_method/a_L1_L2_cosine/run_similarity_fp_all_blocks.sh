#!/usr/bin/env bash
# FP Diff-AE similarity evidence for all 31 blocks (Stage 0a)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON="${PYTHON:-/home/jimmy/anaconda3/envs/diffae_bw/bin/python}"
SCRIPT="QATcode/cache_method/a_L1_L2_cosine/similarity_calculation_baseline.py"
LOG_ROOT="QATcode/cache_method/a_L1_L2_cosine/logs/fp_v1"
mkdir -p "$LOG_ROOT" QATcode/cache_method/a_L1_L2_cosine/log

NUM_STEPS=100
SAMPLES=128
COLLECT_SAMPLES=20
SAMPLE_STRATEGY="random"

echo "================================================================"
echo "FP Similarity — all blocks (Stage 0a)"
echo "Output: QATcode/cache_method/a_L1_L2_cosine/T_${NUM_STEPS}/fp_v1/result_npz"
echo "================================================================"

$PYTHON "$SCRIPT" \
  --num_steps "$NUM_STEPS" \
  --enable_similarity \
  --run_all_blocks \
  --similarity_samples "$SAMPLES" \
  --similarity_collect_samples "$COLLECT_SAMPLES" \
  --similarity_sample_strategy "$SAMPLE_STRATEGY" \
  --similarity_output_root "QATcode/cache_method/a_L1_L2_cosine" \
  --similarity_version "fp_v1" \
  --similarity_dtype float16 \
  --log_file "$LOG_ROOT/similarity_fp_all.log" \
  2>&1 | tee -a "$LOG_ROOT/similarity_fp_all.log"

echo "✅ FP similarity all-blocks finished."
