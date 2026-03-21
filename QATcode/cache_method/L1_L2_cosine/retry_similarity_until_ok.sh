#!/usr/bin/env bash
set -u

# ========= 可調參數 =========
GPU_ID=0
NEED_MB=20000           # 預估要 20GB
BUFFER_MB=1500          # 建議留 buffer，避免剛好卡邊界 OOM（可調 1000~3000）
SLEEP_SEC=$((5*60))    # 5 分鐘

# ========= 你的目標指令 =========
CMD='bash QATcode/cache_method/L1_L2_cosine/run_similarity_experments.sh'

# ========= 自動定位專案根目錄 =========
# retry.sh 放在：QATcode/cache_method/L1_L2_cosine/ 底下
# 往上 3 層就是專案根目錄
WORKDIR="$(cd "$(dirname "$0")/../../.." && pwd)"

# ========= log 位置 =========
LOG_DIR="${WORKDIR}/QATcode/cache_method/L1_L2_cosine/retry_logs_similarity"
mkdir -p "$LOG_DIR"

# ========= 工具函式 =========
get_free_mb () {
  nvidia-smi --id="$GPU_ID" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' '
}

timestamp () {
  date +"%Y-%m-%d_%H-%M-%S"
}

# ========= 主迴圈 =========
while true; do
  FREE_MB=$(get_free_mb)
  REQ_MB=$((NEED_MB + BUFFER_MB))

  echo "[`date`] GPU${GPU_ID} free=${FREE_MB}MB, required=${REQ_MB}MB"

  # 1) 記憶體不夠就等
  if [ "$FREE_MB" -lt "$REQ_MB" ]; then
    echo "[`date`] Not enough GPU memory. Sleep 5 min..."
    sleep "$SLEEP_SEC"
    continue
  fi

  # 2) 確認目標腳本存在（避免路徑打錯）
  TARGET_SH="${WORKDIR}/QATcode/cache_method/L1_L2_cosine/run_similarity_experments.sh"
  if [ ! -f "$TARGET_SH" ]; then
    echo "[`date`] ❌ Script not found: ${TARGET_SH}"
    echo "[`date`] Please check filename/path."
    exit 1
  fi

  # 3) 夠就執行，並寫 log
  RUN_LOG="${LOG_DIR}/run_$(timestamp).log"
  echo "[`date`] Enough memory. Start running..."
  echo "[`date`] WORKDIR=${WORKDIR}"
  echo "[`date`] CMD=${CMD}"
  echo "[`date`] Log -> ${RUN_LOG}"

  set +e
  cd "$WORKDIR" || exit 1
  eval "$CMD" > "$RUN_LOG" 2>&1
  EXIT_CODE=$?
  set -e

  # 4) 成功就結束
  if [ "$EXIT_CODE" -eq 0 ]; then
    echo "[`date`] ✅ Job finished successfully!"
    echo "[`date`] Log: ${RUN_LOG}"
    exit 0
  fi

  # 5) OOM 就睡 5 分鐘再重試
  if grep -qiE "CUDA out of memory|out of memory" "$RUN_LOG"; then
    echo "[`date`] ⚠️ CUDA OOM detected. Sleep 5 min and retry..."
    sleep "$SLEEP_SEC"
    continue
  fi

  # 6) 其他錯誤直接停
  echo "[`date`] ❌ Job failed (NOT OOM). Exit code=${EXIT_CODE}"
  echo "[`date`] Please check log: ${RUN_LOG}"
  exit "$EXIT_CODE"
done
