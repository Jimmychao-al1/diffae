#!/usr/bin/env bash
# Stage 0e: FP evidence normalization (c_FID reuses Q-DiffAE)
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

PYTHON="${PYTHON:-/home/jimmy/anaconda3/envs/diffae_bw/bin/python}"

"$PYTHON" QATcode/cache_method/Stage0/stage0e_normalization.py \
  --l1_cos_dir QATcode/cache_method/a_L1_L2_cosine/T_100/fp_v1/result_npz \
  --svd_dir QATcode/cache_method/b_SVD/svd_metrics_fp \
  --fid_json_path QATcode/cache_method/c_FID/fid_cache_sensitivity/fid_sensitivity_results.json \
  --output_dir QATcode/cache_method/Stage0/stage0e_output_fp

echo "✅ Stage 0e FP complete: Stage0/stage0e_output_fp"
