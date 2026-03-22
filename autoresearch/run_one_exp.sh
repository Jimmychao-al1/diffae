#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# ===== fixed experiment flags =====
BASE_CMD=(
  python -m QATcode.quantize_ver2.quantize_diffae_step6_train_v2
  --teacher-autocast-match
  --debug-timestep-grad-conflict
  --debug-timestep-grad-steps 0,80,99
  --debug-timestep-grad-interval 8
)

# ===== optional hyperparameters =====
# 注意：
# 下面三個參數名稱請改成你程式真正 argparse 支援的名稱。
# 如果目前程式還沒有這些 CLI 參數，先不要加進去。
EXTRA_ARGS=()

if [[ -n "${LORA_FACTOR:-}" ]]; then
  EXTRA_ARGS+=(--lora-factor "${LORA_FACTOR}")
fi

if [[ -n "${WEIGHT_LR:-}" ]]; then
  EXTRA_ARGS+=(--weight-lr "${WEIGHT_LR}")
fi

if [[ -n "${ACTIVATION_LR:-}" ]]; then
  EXTRA_ARGS+=(--activation-lr "${ACTIVATION_LR}")
fi

if [[ -n "${LOG_SUFFIX:-}" ]]; then
  EXTRA_ARGS+=(--log-suffix "${LOG_SUFFIX}")
else
  EXTRA_ARGS+=(--log-suffix autoresearch_run)
fi

echo "Running command:"
printf ' %q' "${BASE_CMD[@]}" "${EXTRA_ARGS[@]}"
echo

"${BASE_CMD[@]}" "${EXTRA_ARGS[@]}"
