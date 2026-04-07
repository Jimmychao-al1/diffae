# sampleStage2FidGuide

在 **Stage2 已產出 `stage2_refined_scheduler_config.json`** 之後，用本流程做 **FFHQ 採樣 + FID**，並可選擇寫入 **run 產物** 與 **`runs_index.jsonl`** 索引。

## 前置條件

- 已完成 Stage2 refine（見 `QATcode/cache_method/Stage2/stage2ExperimentsGuide.md`）。
- 準備好要評分的 JSON 路徑（與 `run_stage2_full_experiments.sh` 產物對齊），例如：
  - `.../fullExperimentsK16sw3/baseline/stage2_refined_scheduler_config.json`
  - `.../fullExperimentsK25sw2/prefix_15_q90q80min130/stage2_refined_scheduler_config.json`

## 直接呼叫 Python（單次 run）

於 **repository 根目錄**：

```bash
python QATcode/cache_method/start_run/sample_stage2_cache_scheduler.py \
  --mode float \
  --num_steps 100 \
  --eval_samples 5000 \
  --seed 0 \
  --quant-state tt \
  --use_cache_scheduler \
  --cache_scheduler_json "<PATH>/stage2_refined_scheduler_config.json" \
  --scheduler-name "<顯示名稱>" \
  --run-output-dir "QATcode/cache_method/results/<你的實驗目錄>" \
  --runs-index-path "QATcode/cache_method/results/<你的實驗目錄>/runs_index.jsonl" \
  --log_file "QATcode/cache_method/results/<你的實驗目錄>/run.log"
```

- **不加** `--use_cache_scheduler` 時走 **baseline**（不讀 Stage2 JSON）；加則讀入 `--cache_scheduler_json`。
- 若 Stage2 變體在 refine 時使用 **`--force-full-prefix-steps N`**，FID 採樣應帶 **相同** `--force-full-prefix-steps N`，否則診斷與實際 cache union 不一致。
- `--eval_samples`：`5000` 與 `50000` 對應不同 FID 量級；`runs_index.jsonl` 內會用 `FID@5K` / `FID@50K` 等鍵（見腳本說明）。
- 完整 CLI 分組說明：`python QATcode/cache_method/start_run/sample_stage2_cache_scheduler.py --help`

## 批次腳本：`runFidWithStage2Scheduler.sh`

預設 **依序跑 6 組 FID**（對齊 Stage2 六個第二趟變體），`prefix_*` 變體會帶 **`--force-full-prefix-steps 15`**，與 Stage2 refine 一致。K25 的 q90/q80/**peak_min=1.3** 變體目錄為 **`prefix_15_q90q80min130`**（對應門檔 `*_min130.json`）。

| # | `scheduler-name`（runs_index） | `stage2_refined_scheduler_config.json` 來源 |
|---|-------------------------------|-----------------------------------------------|
| 1 | `k16_sw3_baseline` | `${EXP_K16}/baseline/...` |
| 2 | `k16_sw3_prefix_15` | `${EXP_K16}/prefix_15/...` |
| 3 | `k25_sw2_baseline` | `${EXP_K25}/baseline/...` |
| 4 | `k25_sw2_prefix_15` | `${EXP_K25}/prefix_15/...` |
| 5 | `k25_sw2_baseline_908030` | `${EXP_K25}/baseline_908030/...` |
| 6 | `k25_sw2_prefix_15_q90q80min130` | `${EXP_K25}/prefix_15_q90q80min130/...` |

```bash
bash QATcode/cache_method/start_run/runFidWithStage2Scheduler.sh
```

### 常用環境變數（覆寫預設）

| 變數 | 預設 | 說明 |
|------|------|------|
| `OUT_BASE` | `QATcode/cache_method/Stage2/stage2_output` | 與 Stage2 腳本一致之上層目錄 |
| `EXP_K16` | `${OUT_BASE}/fullExperimentsK16sw3` | K16 Stage2 實驗根 |
| `EXP_K25` | `${OUT_BASE}/fullExperimentsK25sw2` | K25 Stage2 實驗根 |
| `RESULTS_ROOT` | `.../results/fid_fullExperiments6` | FID 輸出與 `runs_index.jsonl` |
| `NUM_STEPS` | `100` | DDIM 步數 |
| `EVAL_SAMPLES` | `5000` | 生成張數 / FID 樣本數 |
| `SEED` | `0` | |
| `QUANT_STATE` | `tt` | `tt` / `ff` / `tf` / `ft` |
| `RUN_K16_FID` | `1` | 設 `0` 略過 K16 兩筆 |
| `RUN_K25_FID` | `1` | 設 `0` 略過 K25 四筆 |

若要增刪實驗，請編輯 `runFidWithStage2Scheduler.sh` 內對應 **`run_one`** 列。

## 輸出內容（當有 `--run-output-dir`）

每個 run 目錄內通常包含：

- `run_manifest.json`：指令與 git commit 等
- `summary.json`：FID、scheduler 統計等
- `detail_stats.json`
- `scheduler_config.snapshot.json`
- `run.log`

`--runs-index-path` 指向的 **JSONL** 每行一筆精簡紀錄（鍵名見 `--help`）。

## 與 Stage2 的關係

| 階段 | 產物 | 本流程 |
|------|------|--------|
| Stage2 refine | `stage2_refined_scheduler_config.json` | 輸入 |
| 本腳本 | FID、結果目錄、可選 `runs_index.jsonl` | 輸出 |

Stage2 管線指令不寫在本檔；請以 `stage2ExperimentsGuide.md` 為準。
