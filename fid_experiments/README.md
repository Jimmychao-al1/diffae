# FID Experiments

這個資料夾包含所有 FID 實驗相關的腳本和工具。

## 檔案說明

### 核心腳本

- **`run_experiments.py`** - FID 實驗主腳本
  - 自動執行 `run_ffhq128.py`（baseline）和 `QATcode/sample_lora_intmodel.py`（QAT）
  - 支援不同 T（timesteps）和 k（eval_samples，以千為單位）參數
  - 自動解析 FID 結果並存入 CSV/JSONL
  - 所有結果和日誌存在 `../experiment_results/`

### 輔助工具

- **`reparse.py`** - 重新解析日誌檔案
  - 用於修正之前因 FID pattern 不匹配導致的空結果
  - 讀取 `../experiment_results/logs/` 中的所有日誌
  - 自動備份舊的 CSV/JSONL，然後更新

- **`summary.py`** - 顯示結果摘要
  - 格式化顯示所有實驗結果
  - 提供 Baseline vs QAT 的對比分析

- **`test_fix.py`** - 驗證程式碼修改
  - 檢查 `experiment.py` 和 `run_ffhq128.py` 的關鍵修改
  - 確保 `eval_num_images` 和 `ckpt_path` 等參數正確

## 使用方式

### 1. 執行完整實驗（預設：T=20/100, k=5/50）

```bash
cd fid_experiments
python run_experiments.py
```

### 2. 自訂參數

```bash
# 只跑 T=20, k=5
python run_experiments.py --steps 20 --k 5

# 跑 QAT int 模式
python run_experiments.py --qat-mode int

# 啟用快取
python run_experiments.py --enable-cache --cache-method Res --cache-threshold 0.03

# 只跑 baseline（跳過 QAT）
python run_experiments.py --skip-qat

# 測試命令（不真正執行）
python run_experiments.py --dry-run
```

### 3. 查看結果摘要

```bash
python summary.py
```

### 4. 重新解析舊日誌

```bash
python reparse.py
```

### 5. 驗證程式碼修改

```bash
python test_fix.py
```

## 結果輸出

所有實驗結果存放在 `../experiment_results/`：

- **`fid_results.csv`** - CSV 格式的結果表
- **`fid_results.jsonl`** - JSONL 格式（每行一個 JSON）
- **`logs/`** - 所有執行日誌
  - 格式：`YYYYMMDD_HHMMSS_<mode>_T<steps>_k<k>.log`

## 已修正的問題

1. **`experiment.py`**
   - `eval_num_images = 1` → `eval_num_images = EVAL_SAMPLES`
   - `resume_from_checkpoint` → `ckpt_path`（PyTorch Lightning 2.0+）

2. **FID 解析**
   - 新增正則表達式支援 `fid (rank): value` 格式
   - 支援 `FID@50000 100 steps score: 11.09` 格式

3. **CSV 處理**
   - 處理缺少 header 的 CSV 檔案
   - 自動備份和更新機制

## 注意事項

- 從 `fid_experiments/` 資料夾執行腳本
- `experiment_results/` 保留在 DiffAE root（不在此資料夾內）
- 所有路徑自動處理相對位置（`ROOT = Path(__file__).resolve().parent.parent`）
