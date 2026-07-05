#!/usr/bin/env bash
set -euo pipefail
cd /home/jimmy/diffae
PYTHON=/home/jimmy/anaconda3/envs/diffae_bw/bin/python
ROOT=/home/jimmy/diffae/QATcode/cache_method/resweep_output
STAGE_A2="$ROOT/stage_a2_stage2"
STAGE_B="$ROOT/stage_b_fid5k"
RUNS_DIR="$STAGE_B/runs"
LOG_DIR=/home/jimmy/diffae/QATcode/cache_method/logs
SCRIPT_TS="$(date +%Y%m%d_%H%M%S)"
SCRIPT_LOG="$LOG_DIR/stage_b_reverse_${SCRIPT_TS}.log"
RUNS_INDEX="$STAGE_B/runs_index.jsonl"
FID_REF="/home/jimmy/diffae/mycache/eval_images/ffhqlmdb256_size128_5000_5000"
CURRENT_LOCK=""
mkdir -p "$RUNS_DIR" "$LOG_DIR"
exec > >(tee -a "$SCRIPT_LOG") 2>&1
cleanup() {
  if [[ -n "${CURRENT_LOCK:-}" && -f "$CURRENT_LOCK" ]]; then
    rm -f "$CURRENT_LOCK"
    echo "[cleanup] removed lockfile: $CURRENT_LOCK"
  fi
}
trap cleanup INT TERM
CONFIGS=(
  K25_sw5_lam2.0_kmax4
  K25_sw5_lam1.0_kmax4
  K25_sw5_lam0.5_kmax4
  K25_sw5_lam0.25_kmax4
  K25_sw3_lam2.0_kmax4
  K25_sw3_lam1.0_kmax4
  K25_sw3_lam0.5_kmax4
  K25_sw3_lam0.25_kmax4
  K25_sw2_lam2.0_kmax4
  K25_sw2_lam1.0_kmax4
  K25_sw2_lam0.5_kmax4
  K25_sw2_lam0.25_kmax4
  K20_sw5_lam2.0_kmax4
  K20_sw5_lam1.0_kmax4
  K20_sw5_lam0.5_kmax4
  K20_sw5_lam0.25_kmax4
  K20_sw3_lam2.0_kmax4
  K20_sw3_lam1.0_kmax4
  K20_sw3_lam0.5_kmax4
  K20_sw3_lam0.25_kmax4
  K20_sw2_lam2.0_kmax4
  K20_sw2_lam1.0_kmax4
  K20_sw2_lam0.5_kmax4
  K20_sw2_lam0.25_kmax4
  K16_sw5_lam2.0_kmax4
  K16_sw5_lam1.0_kmax4
)
echo "[start] Stage B reverse FID@5K"
echo "[log] $SCRIPT_LOG"
echo "[fid_ref implicit] $FID_REF"
echo "[configs] ${#CONFIGS[@]}"
for CID in "${CONFIGS[@]}"; do
  SCHED="$STAGE_A2/$CID/baseline/stage2_refined_scheduler_config.json"
  RUN_DIR="$RUNS_DIR/$CID"
  SUMMARY="$RUN_DIR/summary.json"
  LOCK="$RUN_DIR/.running"
  SAMPLE_LOG="$STAGE_B/$CID.log"
  if [[ -f "$SUMMARY" ]]; then
    echo "[skip] $CID already has completion marker: $SUMMARY"
    continue
  fi
  if [[ ! -f "$SCHED" ]]; then
    echo "[fail] missing Stage A2 refined scheduler: $SCHED"
    exit 1
  fi
  mkdir -p "$RUN_DIR"
  if [[ -f "$LOCK" ]]; then
    echo "[skip] $CID has lockfile: $LOCK"
    continue
  fi
  if pgrep -af "sample_stage2_cache_scheduler.py" | grep -F -- "$CID" >/dev/null 2>&1; then
    echo "[skip] $CID appears to be running in another process"
    pgrep -af "sample_stage2_cache_scheduler.py" | grep -F -- "$CID" || true
    continue
  fi
  CURRENT_LOCK="$LOCK"
  echo "$$ $(date --iso-8601=seconds) $CID" > "$LOCK"
  START_SEC="$(date +%s)"
  echo "[run] $CID"
  echo "[scheduler] $SCHED"
  echo "[run_dir] $RUN_DIR"
  "$PYTHON" QATcode/cache_method/start_run/sample_stage2_cache_scheduler.py \
    --mode float \
    --num_steps 100 \
    --eval_samples 5000 \
    --seed 0 \
    --quant-state tt \
    --use_cache_scheduler \
    --cache_scheduler_json "$SCHED" \
    --scheduler-name "$CID" \
    --run-output-dir "$RUN_DIR" \
    --runs-index-path "$RUNS_INDEX" \
    --log_file "$SAMPLE_LOG" \
    --generate-dir-override /home/jimmy/diffae/mycache_rev/gen_images_reverse
  if [[ ! -f "$SUMMARY" ]]; then
    echo "[fail] missing completion marker after run: $SUMMARY"
    exit 1
  fi
  END_SEC="$(date +%s)"
  ELAPSED="$((END_SEC - START_SEC))"
  echo "[done] $CID elapsed=${ELAPSED}s summary=$SUMMARY"
  rm -f "$LOCK"
  CURRENT_LOCK=""
done
echo "[complete] Stage B reverse FID@5K done"
