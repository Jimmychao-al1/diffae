#!/usr/bin/env bash
# Master driver: FP Stage 0a -> 0b -> 0e -> Stage 1 sweep
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON="${PYTHON:-/home/jimmy/anaconda3/envs/diffae_bw/bin/python}"
export PYTHON
LOG_DIR="QATcode/cache_method/logs/fp_pipeline"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/fp_pipeline_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$MASTER_LOG") 2>&1

echo "================================================================"
echo "FP Diff-AE S3-Cache pipeline start: $(date)"
echo "Log: $MASTER_LOG"
echo "================================================================"

echo ""
echo ">>> Stage 0a: FP Similarity (31 blocks)"
bash QATcode/cache_method/a_L1_L2_cosine/run_similarity_fp_all_blocks.sh

echo ""
echo ">>> Stage 0b: FP SVD (31 blocks)"
bash QATcode/cache_method/b_SVD/run_svd_fp_all_blocks.sh

echo ""
echo ">>> Stage 0e: normalization"
bash QATcode/cache_method/Stage0/run_stage0e_fp.sh

echo ""
echo ">>> Stage 1: sweep + CSV summary"
bash QATcode/cache_method/Stage1/run_stage1_sweep_fp.sh

echo ""
echo "================================================================"
echo "FP pipeline complete: $(date)"
echo "CSV: QATcode/cache_method/Stage1/stage1_output_fp/stage1_sweep_summary_fp.csv"
echo "================================================================"
