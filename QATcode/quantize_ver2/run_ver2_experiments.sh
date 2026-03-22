#!/usr/bin/env bash
# 第二版量化訓練（ver2）— 依 SECOND_VERSION_TRAINING_PLAN.md 自動跑完整實驗序列
#
# 使用方式（請在 repo 根目錄執行）：
#   bash QATcode/quantize_ver2/run_ver2_experiments.sh
#
# 環境變數（可選）：
#   PHASE3_LORA_FACTOR   Phase 3 固定之 lora_factor（預設 2500，請依 Phase 2 結果修改）
#   SKIP_PHASE1=1        跳過 Phase 1 baseline
#   SKIP_PHASE2=1        跳過 Phase 2（lora_factor 掃描）
#   SKIP_PHASE3=1        跳過 Phase 3（act_quant_lr 掃描）
#   PYTHON               預設 python3
#
# 產生之 log：QATcode/quantize_ver2/log/step6_train_<log-suffix>.log
#
# 所有實驗皆附：--teacher-autocast-match（對應 train_v2 之 store_true，不需再寫 True）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# SCRIPT_DIR = .../QATcode/quantize_ver2  →  repo root = ../..
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRAIN_PY="${ROOT}/QATcode/quantize_ver2/quantize_diffae_step6_train_v2.py"
PYTHON="${PYTHON:-python3}"

SKIP_PHASE1="${SKIP_PHASE1:-0}"
SKIP_PHASE2="${SKIP_PHASE2:-0}"
SKIP_PHASE3="${SKIP_PHASE3:-0}"
# Phase 3：固定此 lora_factor 後掃 act_quant_lr（請在跑 Phase 3 前依 Phase 2 結果設定）
PHASE3_LORA_FACTOR="${PHASE3_LORA_FACTOR:-2500}"

if [[ ! -f "$TRAIN_PY" ]]; then
  echo "ERROR: 找不到訓練腳本: $TRAIN_PY"
  exit 1
fi

cd "$ROOT"

run_train() {
  local log_suffix="$1"
  shift
  echo ""
  echo "================================================================================"
  echo "RUN  --log-suffix ${log_suffix}"
  echo "     $*"
  echo "================================================================================"
  "$PYTHON" "$TRAIN_PY" --log-suffix "$log_suffix" --teacher-autocast-match --tensorboard "$@"
}

echo "Repo root: $ROOT"
echo "Train script: $TRAIN_PY"
echo "PHASE3_LORA_FACTOR=${PHASE3_LORA_FACTOR} (export 可覆寫)"
echo ""

# ------------------------------------------------------------------------------
# Phase 1：正式 baseline（lora_factor=2500, act_quant_lr=5e-4）
# ------------------------------------------------------------------------------
if [[ "$SKIP_PHASE1" != "1" ]]; then
  run_train "ver2_p1_baseline_lf2500_aq5e4" \
    --lora-factor 2500 --act-quant-lr 5e-4
else
  echo "[SKIP] Phase 1 baseline"
fi

# ------------------------------------------------------------------------------
# Phase 2：固定 act_quant_lr=5e-4，掃 lora_factor = 1800 / 2500 / 3500
# （2500 與 Phase 1 相同；若已跑過 Phase 1 可設 SKIP_PHASE2=1 並手動只跑 1800/3500，或保留完整重跑）
# ------------------------------------------------------------------------------
if [[ "$SKIP_PHASE2" != "1" ]]; then
  run_train "ver2_p2_lf1800_aq5e4" \
    --lora-factor 1800 --act-quant-lr 5e-4

  run_train "ver2_p2_lf2500_aq5e4" \
    --lora-factor 2500 --act-quant-lr 5e-4

  run_train "ver2_p2_lf3500_aq5e4" \
    --lora-factor 3500 --act-quant-lr 5e-4
else
  echo "[SKIP] Phase 2 lora_factor sweep"
fi

# ------------------------------------------------------------------------------
# Phase 3：固定 PHASE3_LORA_FACTOR，掃 act_quant_lr = 2e-4 / 5e-4 / 1e-3
# ------------------------------------------------------------------------------
if [[ "$SKIP_PHASE3" != "1" ]]; then
  LF="${PHASE3_LORA_FACTOR}"
  run_train "ver2_p3_lf${LF}_aq2e4" \
    --lora-factor "$LF" --act-quant-lr 2e-4

  run_train "ver2_p3_lf${LF}_aq5e4" \
    --lora-factor "$LF" --act-quant-lr 5e-4

  run_train "ver2_p3_lf${LF}_aq1e3" \
    --lora-factor "$LF" --act-quant-lr 1e-3
else
  echo "[SKIP] Phase 3 act_quant_lr sweep"
fi

echo ""
echo "================================================================================"
echo "全部排程中的實驗已跑完（若有略過請檢查 SKIP_* 環境變數）。"
echo "Log 目錄: ${ROOT}/QATcode/quantize_ver2/log/"
echo "================================================================================"
