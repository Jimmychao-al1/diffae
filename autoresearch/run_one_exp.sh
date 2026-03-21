#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash autoresearch/run_one_exp.sh <run_id> <lora_factor> <weight_lr> <activation_lr>

if [ "$#" -ne 4 ]; then
  echo "Usage: bash autoresearch/run_one_exp.sh <run_id> <lora_factor> <weight_lr> <activation_lr>"
  exit 1
fi

RUN_ID="$1"
LORA_FACTOR="$2"
WEIGHT_LR="$3"
ACTIVATION_LR="$4"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORKDIR="$REPO_ROOT/QATcode/quantize_ver2"
OUTDIR="$REPO_ROOT/artifacts/autoresearch/$RUN_ID"
mkdir -p "$OUTDIR"

cd "$WORKDIR"

echo "[INFO] run_id=$RUN_ID" | tee "$OUTDIR/meta.txt"
echo "[INFO] lora_factor=$LORA_FACTOR" | tee -a "$OUTDIR/meta.txt"
echo "[INFO] weight_lr=$WEIGHT_LR" | tee -a "$OUTDIR/meta.txt"
echo "[INFO] activation_lr=$ACTIVATION_LR" | tee -a "$OUTDIR/meta.txt"

# Replace the command below with your real training command.
# Keep the entrypoint fixed; only vary the allowed hyperparameters.
python quantize_diffae_step6_train.py \
  --lora_factor "$LORA_FACTOR" \
  --weight_lr "$WEIGHT_LR" \
  --activation_lr "$ACTIVATION_LR" \
  2>&1 | tee "$OUTDIR/train.log"

# Optional: append a row manually or via a parser after evaluation.
echo "[INFO] Finished $RUN_ID"
