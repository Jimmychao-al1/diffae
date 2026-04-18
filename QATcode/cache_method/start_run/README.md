# start_run：FID 採樣與結果管理入口

此目錄為 Stage 2 之後的 **FID 採樣與結果管理入口**，提供採樣腳本與批次執行工具，不負責 Stage 2 refine 本身。

---

## 目錄結構

```
start_run/
├── sample_stage2_cache_scheduler.py    # FID 採樣主程式（支援 cache / baseline 兩種模式）
├── runFidWithStage2Scheduler.sh        # 批次跑六組 FID（與 Stage2 六組實驗對齊）
└── README.md                           # 本檔
```

---

## 前置條件

1. 已完成 Stage 2 refine，產出 `stage2_refined_scheduler_config.json`（見 `Stage2/README.md`）。
2. 確認要評分的 JSON 路徑，例如：
   - `QATcode/cache_method/Stage2/stage2_output/fullExperimentsK16sw3/baseline/stage2_refined_scheduler_config.json`
   - `QATcode/cache_method/Stage2/stage2_output/fullExperimentsK25sw2/prefix_15_q90q80min130/stage2_refined_scheduler_config.json`

---

## Quick Start

### 單次 FID 採樣

```bash
cd /home/jimmy/diffae

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

- 不加 `--use_cache_scheduler` 時走 **baseline**（不讀 Stage 2 JSON）
- 若 Stage 2 使用了 `--force-full-prefix-steps N`，採樣時需帶**相同** `--force-full-prefix-steps N`

### 批次跑六組 FID

```bash
bash QATcode/cache_method/start_run/runFidWithStage2Scheduler.sh
```

預設依序執行以下六組（與 Stage 2 六個第二趟變體對齊）：

| # | `scheduler-name` | Stage2 JSON 來源 |
|---|-----------------|------------------|
| 1 | `k16_sw3_baseline` | `fullExperimentsK16sw3/baseline/...` |
| 2 | `k16_sw3_prefix_15` | `fullExperimentsK16sw3/prefix_15/...` |
| 3 | `k25_sw2_baseline` | `fullExperimentsK25sw2/baseline/...` |
| 4 | `k25_sw2_prefix_15` | `fullExperimentsK25sw2/prefix_15/...` |
| 5 | `k25_sw2_baseline_908030` | `fullExperimentsK25sw2/baseline_908030/...` |
| 6 | `k25_sw2_prefix_15_q90q80min130` | `fullExperimentsK25sw2/prefix_15_q90q80min130/...` |

---

## 常用環境變數

| 變數 | 預設 | 說明 |
|------|------|------|
| `OUT_BASE` | `QATcode/cache_method/Stage2/stage2_output` | Stage2 輸出上層目錄 |
| `EXP_K16` | `${OUT_BASE}/fullExperimentsK16sw3` | K16 Stage2 實驗根 |
| `EXP_K25` | `${OUT_BASE}/fullExperimentsK25sw2` | K25 Stage2 實驗根 |
| `RESULTS_ROOT` | `QATcode/cache_method/results/fid_fullExperiments6` | FID 輸出與 `runs_index.jsonl` |
| `NUM_STEPS` | `100` | DDIM 步數 |
| `EVAL_SAMPLES` | `5000` | 生成張數 / FID 樣本數 |
| `SEED` | `0` | 隨機種子 |
| `QUANT_STATE` | `tt` | `tt` / `ff` / `tf` / `ft` |
| `RUN_K16_FID` | `1` | 設 `0` 略過 K16 兩筆 |
| `RUN_K25_FID` | `1` | 設 `0` 略過 K25 四筆 |

若要增刪實驗，請編輯 `runFidWithStage2Scheduler.sh` 內對應 `run_one` 列。

---

## 輸出內容

每個 run 目錄（`--run-output-dir`）通常包含：

| 檔案 | 說明 |
|------|------|
| `run_manifest.json` | 指令、git commit 等執行資訊 |
| `summary.json` | FID、scheduler 統計等 |
| `detail_stats.json` | 詳細統計 |
| `scheduler_config.snapshot.json` | 使用的 scheduler 快照 |
| `run.log` | 執行 log |

`--runs-index-path` 指向的 JSONL 每行一筆精簡紀錄，`eval_samples` 決定鍵名（`FID@5K` / `FID@50K` 等）。

---

## 與 Stage2 的關係

| 階段 | 產物 | 本目錄的角色 |
|------|------|-------------|
| Stage2 refine | `stage2_refined_scheduler_config.json` | 輸入 |
| 本腳本 | FID、結果目錄、`runs_index.jsonl` | 輸出 |

Stage2 管線指令（refine、diagnostics、blockwise threshold）請見 `Stage2/README.md` 與 `Stage2/SUMMARY.md`。
