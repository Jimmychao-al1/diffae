# stage2ExperimentsGuide

本文件說明 **Stage2** 與 **FID 採樣** 的完整指令，方便複製、修改或接到排程。  
（Stage1 固定輸出請見 `Stage1/stage1_output/sweep_*`。）

## 1. 管線概念（兩趟 refine）

| 步驟 | 目的 | 主要指令 |
|------|------|----------|
| Pass 1 | 用 **global** `zone_l1` / `peak_l1` 跑診斷，產生 `stage2_runtime_diagnostics.json` | `stage2_runtime_refine.py` **不加** `--threshold-config` |
| 門檻 | 由診斷算 **per-block quantile** 門檻 | `build_blockwise_thresholds.py` |
| Pass 2 | 載入 blockwise JSON，寫出最終 `stage2_refined_scheduler_config.json` | `stage2_runtime_refine.py` **加** `--threshold-config` |

第二趟若要做 **prefix 強制 full-compute**，在 Pass 2 加 `--force-full-prefix-steps N`（僅影響診斷用的 cache pass union，語意見 `README.md`）。

## 2. 一次跑滿六組實驗（推薦）

於 **repository 根目錄**：

```bash
bash QATcode/cache_method/Stage2/run_stage2_full_experiments.sh
```

### 2.1 六組實驗對照

| # | Stage1 來源 | Blockwise 門檻 | 輸出子目錄（相對於各實驗根） |
|---|-------------|----------------|-----------------------------|
| 1 | `sweep_K16_sw3_…`（見腳本 `SCHED_K16`） | original（q75/q95/min1.5） | `baseline` |
| 2 | 同上 | 同上 | `prefix_15`（`--force-full-prefix-steps 15`） |
| 3 | `sweep_K25_sw2_lam0.5_kmax4` | original | `baseline` |
| 4 | 同上 | 同上 | `prefix_15`（`--force-full-prefix-steps 15`） |
| 5 | 同上 | q90 / q80 / peak_min **1.3**（慣称 908030） | `baseline_908030` |
| 6 | 同上 | 同上 | `prefix_15_q90q80min130`（`--force-full-prefix-steps 15`） |

預設實驗根目錄：

- K16：`QATcode/cache_method/Stage2/stage2_output/fullExperimentsK16sw3/`（預設，見 `EXP_K16`）
- K25：`QATcode/cache_method/Stage2/stage2_output/fullExperimentsK25sw2/`

K25 的 **Pass 1** 與 **original / q90q80** 兩份門檻檔共用同一份 `00_global_refine/stage2_runtime_diagnostics.json`。

### 2.2 常用環境變數（覆寫預設）

| 變數 | 預設 | 說明 |
|------|------|------|
| `STAGE1_ROOT` | `QATcode/cache_method/Stage1/stage1_output` | Stage1 sweep 根目錄 |
| `SCHED_K16` / `SCHED_K25` | 見腳本內 | 明確指定 `scheduler_config.json` |
| `OUT_BASE` | `QATcode/cache_method/Stage2/stage2_output` | Stage2 輸出上層目錄 |
| `EVAL_NUM_IMAGES` | `8` | 診斷用影像數 |
| `EVAL_CHUNK_SIZE` | `2` | 診斷 chunk，降 RAM 尖峰 |
| `Q_ZONE_ORIG` / `Q_PEAK_ORIG` / `PEAK_OVER_ZONE_MIN_ORIG` | `0.75` / `0.95` / `1.5` | original blockwise |
| `Q_ZONE_9080` / `Q_PEAK_9080` / `PEAK_OVER_ZONE_MIN_9080` | `0.90` / `0.80` / **`1.3`** | `peak_over_zone_ratio_min`；門檻檔名 `*_q90_q80_min130.json` |
| `RUN_K16` / `RUN_K25` | `1` | 設 `0` 可略過其中一條線 |

### 2.3 「908030」與 `q90_q80_min130`

**908030** 在本 repo 預設指：**`q_zone=0.90`、`q_peak=0.80`、`peak_over_zone_ratio_min=1.3`**（非 0.30）。

`build_blockwise_thresholds.py` 第三個參數為 **`--peak_over_zone_ratio_min`**（約束 `peak_l1_threshold >= ratio * zone_l1_threshold`）。  
產生的門檻檔名為 `stage2_thresholds_blockwise_q90_q80_min130.json`（**min130** 對應 **1.3**）。若要覆寫：

```bash
export PEAK_OVER_ZONE_MIN_9080=1.3
bash QATcode/cache_method/Stage2/run_stage2_full_experiments.sh
```

## 3. 手動逐步指令（單一實驗線）

以下路徑請自行替換 `OUT`、`SCHED`。

### 3.1 Pass 1（global）

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

### 3.2 產生 blockwise 門檻

```bash
python QATcode/cache_method/Stage2/build_blockwise_thresholds.py \
  --diagnostics "${OUT}/00_global_refine/stage2_runtime_diagnostics.json" \
  --output "${OUT}/01_blockwise_threshold/stage2_thresholds_blockwise.json" \
  --q_zone 0.75 --q_peak 0.95 --peak_over_zone_ratio_min 1.5
```

### 3.3 Pass 2（baseline 範例）

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

### 3.4 驗證

```bash
python QATcode/cache_method/Stage2/verify_stage2.py "${OUT}/baseline/stage2_refined_scheduler_config.json"
python QATcode/cache_method/Stage2/verify_stage2.py --threshold-config "${OUT}/01_blockwise_threshold/stage2_thresholds_blockwise.json"
```

## 4. FID 採樣（Stage2 之後）

Stage2 只產 **scheduler JSON**；要算 FID 請用採樣腳本，例如：

```bash
bash QATcode/cache_method/start_run/runFidWithStage2Scheduler.sh
```

請先將該腳本內的 `run_one` 列或環境變數 `STAGE2_BASE`、`RESULTS_ROOT` 指到本實驗產生的目錄，或自行用：

```bash
python QATcode/cache_method/start_run/sample_stage2_cache_scheduler.py \
  --use_cache_scheduler \
  --cache_scheduler_json "<path>/stage2_refined_scheduler_config.json" \
  --scheduler-name "<name>" \
  --run-output-dir "<results_dir>" \
  ...
```

採樣入口腳本為 **`runFidWithStage2Scheduler.sh`**（與 Stage2 refine 職責不同）；詳見 `QATcode/cache_method/start_run/sampleStage2FidGuide.md`。

## 5. CLI 說明

`stage2_runtime_refine.py --help` 已將參數分組（Stage1 輸入、門檻、模型、診斷、safety），並附兩趟流程提示。

## 6. 相關文件

- `QATcode/cache_method/Stage2/README.md`：Stage2 語意、時間軸、輸出檔案
- `QATcode/cache_method/Stage2/run_stage2_safety_override_examples.sh`：safety / prefix 範例
