#!/usr/bin/env bash
# Resume FP pipeline from Stage 0b (similarity already done)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON="${PYTHON:-/home/jimmy/anaconda3/envs/diffae_bw/bin/python}"
export PYTHON
LOG_DIR="QATcode/cache_method/logs/fp_pipeline"
mkdir -p "$LOG_DIR"
MASTER_LOG="$LOG_DIR/fp_pipeline_resume_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$MASTER_LOG") 2>&1

echo ">>> Stage 0b: FP SVD (31 blocks)"
bash QATcode/cache_method/b_SVD/run_svd_fp_all_blocks.sh

echo ">>> Stage 0e: normalization"
bash QATcode/cache_method/Stage0/run_stage0e_fp.sh

echo ">>> Stage 1: sweep + CSV summary"
bash QATcode/cache_method/Stage1/run_stage1_sweep_fp.sh

echo "Done. CSV: QATcode/cache_method/Stage1/stage1_output_fp/stage1_sweep_summary_fp.csv"
