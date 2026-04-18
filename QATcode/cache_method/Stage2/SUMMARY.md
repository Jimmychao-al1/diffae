# Stage 2：實驗指令與觀察

Stage2 採用**兩趟 refine 流程**：Pass 1 產生 diagnostics → 由診斷產生 per-block quantile 門檻 → Pass 2 使用 blockwise 門檻輸出最終 refined scheduler。

---

## 實驗設定

### 六組實驗對照

| # | Stage1 來源 | Blockwise 門檻 | 輸出子目錄 |
|---|-------------|----------------|------------|
| 1 | `sweep_K16_sw3_lam0.5_kmax4` | original（q75/q95/min1.5） | `baseline` |
| 2 | 同上 | 同上 | `prefix_15`（`--force-full-prefix-steps 15`） |
| 3 | `sweep_K25_sw2_lam0.5_kmax4` | original | `baseline` |
| 4 | 同上 | 同上 | `prefix_15` |
| 5 | 同上 | q90/q80/peak_min **1.3**（慣稱 908030） | `baseline_908030` |
| 6 | 同上 | 同上 | `prefix_15_q90q80min130` |

預設實驗根目錄：
- K16：`QATcode/cache_method/Stage2/stage2_output/fullExperimentsK16sw3/`
- K25：`QATcode/cache_method/Stage2/stage2_output/fullExperimentsK25sw2/`

K25 的 Pass 1 與 original / q90q80 兩份門檻檔共用同一份 `00_global_refine/stage2_runtime_diagnostics.json`。

### 環境變數（覆寫預設）

| 變數 | 預設 | 說明 |
|------|------|------|
| `STAGE1_ROOT` | `QATcode/cache_method/Stage1/stage1_output` | Stage1 sweep 根目錄 |
| `OUT_BASE` | `QATcode/cache_method/Stage2/stage2_output` | Stage2 輸出上層 |
| `EVAL_NUM_IMAGES` | `8` | 診斷用影像數 |
| `EVAL_CHUNK_SIZE` | `2` | 診斷 chunk，降 RAM 尖峰 |
| `Q_ZONE_ORIG` / `Q_PEAK_ORIG` / `PEAK_OVER_ZONE_MIN_ORIG` | `0.75` / `0.95` / `1.5` | original blockwise |
| `Q_ZONE_9080` / `Q_PEAK_9080` / `PEAK_OVER_ZONE_MIN_9080` | `0.90` / `0.80` / `1.3` | 908030 變體 |
| `RUN_K16` / `RUN_K25` | `1` | 設 `0` 可略過其中一條線 |

---

## 執行指令

### 一次跑滿六組（推薦）

```bash
bash QATcode/cache_method/Stage2/run_stage2_full_experiments.sh
```

### 手動逐步（單一實驗線）

以下路徑請自行替換 `OUT`、`SCHED`。

**Pass 1（global threshold）：**

```bash
python QATcode/cache_method/Stage2/stage2_runtime_refine.py \
  --scheduler_config "${SCHED}" \
  --output_dir "${OUT}/00_global_refine" \
  --seed 0 \
  --zone_l1_threshold 0.02 \
  --peak_l1_threshold 0.08 \
  --eval-num-images 8 \
  --eval-chunk-size 2
```

**產生 blockwise 門檻：**

```bash
python QATcode/cache_method/Stage2/build_blockwise_thresholds.py \
  --diagnostics "${OUT}/00_global_refine/stage2_runtime_diagnostics.json" \
  --output "${OUT}/01_blockwise_threshold/stage2_thresholds_blockwise.json" \
  --q_zone 0.75 --q_peak 0.95 --peak_over_zone_ratio_min 1.5
```

**Pass 2（blockwise threshold）：**

```bash
python QATcode/cache_method/Stage2/stage2_runtime_refine.py \
  --scheduler_config "${SCHED}" \
  --output_dir "${OUT}/baseline" \
  --seed 0 \
  --zone_l1_threshold 0.02 \
  --peak_l1_threshold 0.08 \
  --threshold-config "${OUT}/01_blockwise_threshold/stage2_thresholds_blockwise.json" \
  --eval-num-images 8 \
  --eval-chunk-size 2
```

**驗證：**

```bash
python QATcode/cache_method/Stage2/verify_stage2.py \
  "${OUT}/baseline/stage2_refined_scheduler_config.json"
```

---

## 觀察與設計決策

### 為何不建議只依賴單一 global threshold

`stage2_runtime_diagnostics.json` 的 `per_block_step_error` 顯示，不同 UNet block 的 L1 尺度差異很大。用同一組全域門檻去套所有 block，尺度較小的 block 幾乎永遠不觸發，較大的 block 則過敏。**Per-block threshold** 才能讓 refinement 對每層公平。

### 「908030」命名約定

**908030** 在本 repo 固定指：`q_zone=0.90`、`q_peak=0.80`、`peak_over_zone_ratio_min=1.3`（非 0.30）。對應門檻檔名為 `stage2_thresholds_blockwise_q90_q80_min130.json`（**min130** 對應 **1.3**）。

### Reuse 步誤差的正確收集

先前若只在 forward 路徑掛 hook，reuse 時不進入該層 forward，逐點誤差會漏掉。修正後 `model/unet_autoenc.py` 在 recompute 與 reuse 兩條路都呼叫 `cache_debug_collector`，Stage 2 可在**所有 timestep（含 reuse）** 取得正確的 baseline vs cache 比較。

### FID 採樣

Stage 2 只產 scheduler JSON；FID 採樣請用 `start_run/runFidWithStage2Scheduler.sh`（詳見 `QATcode/cache_method/start_run/README.md`）。
