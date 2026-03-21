# FID Experiments 重組總結

## 變更概要

已將所有 FID 實驗相關的腳本從 DiffAE 根目錄移至新資料夾 `fid_experiments/`，並重新命名以簡化識別。

## 檔案變更清單

### 舊位置 → 新位置

| 舊檔名 | 新位置 | 新檔名 |
|--------|---------|---------|
| `/home/jimmy/diffae/run_fid_experiments.py` | `fid_experiments/` | `run_experiments.py` |
| `/home/jimmy/diffae/reparse_fid_results.py` | `fid_experiments/` | `reparse.py` |
| `/home/jimmy/diffae/show_fid_summary.py` | `fid_experiments/` | `summary.py` |
| `/home/jimmy/diffae/test_fid_fix.py` | `fid_experiments/` | `test_fix.py` |
| （新增） | `fid_experiments/` | `README.md` |

### 程式碼調整

所有移動的檔案都已更新路徑引用：

```python
# 舊版本（在 DiffAE root）
ROOT = Path(__file__).resolve().parent

# 新版本（在 fid_experiments/）
ROOT = Path(__file__).resolve().parent.parent
```

這確保所有腳本正確指向：
- `../experiment_results/` - 實驗結果存放位置
- `../run_ffhq128.py` - baseline 腳本
- `../QATcode/sample_lora_intmodel.py` - QAT 腳本
- `../experiment.py` - 核心實驗模組

## 目錄結構

```
/home/jimmy/diffae/
├── fid_experiments/               # 新建資料夾
│   ├── README.md                  # 使用說明
│   ├── run_experiments.py         # 主要實驗腳本
│   ├── reparse.py                 # 重新解析日誌
│   ├── summary.py                 # 結果摘要顯示
│   └── test_fix.py                # 驗證修改
├── experiment_results/            # 保持在 root（不移動）
│   ├── logs/                      # 所有執行日誌
│   ├── fid_results.csv            # CSV 格式結果
│   └── fid_results.jsonl          # JSONL 格式結果
└── ... (其他 DiffAE 檔案)
```

## 使用方式變更

### 之前（在 root 執行）

```bash
cd /home/jimmy/diffae
python run_fid_experiments.py
python show_fid_summary.py
python reparse_fid_results.py
python test_fid_fix.py
```

### 現在（在 fid_experiments/ 執行）

```bash
cd /home/jimmy/diffae/fid_experiments
python run_experiments.py
python summary.py
python reparse.py
python test_fix.py
```

## 檔案權限

所有 Python 腳本已設定為可執行：

```bash
chmod +x fid_experiments/*.py
```

可直接執行：

```bash
cd fid_experiments
./run_experiments.py --help
./summary.py
```

## 驗證結果

執行 `test_fix.py` 驗證所有必要修改：

```bash
cd fid_experiments
python test_fix.py
```

✅ 預期輸出：
- `experiment.py: eval_num_images 已正確修改為 EVAL_SAMPLES`
- `experiment.py: resume_from_checkpoint 已正確改為 ckpt_path`
- `experiment.py: EVAL_SAMPLES 全域變數存在`
- `run_ffhq128.py: 正確傳遞 eval_samples 參數`

## 相關修正

在移動過程中，同時修正了一個遺漏的 PyTorch Lightning 2.0+ API 問題：

### experiment.py (Line 989)

```python
# 修正前
resume_from_checkpoint=resume,

# 修正後
ckpt_path=resume,
```

這是之前測試時發現但未修正的最後一個 `resume_from_checkpoint` 實例。

## 測試建議

1. **測試腳本路徑**
   ```bash
   cd fid_experiments
   python run_experiments.py --dry-run
   ```

2. **查看現有結果**
   ```bash
   cd fid_experiments
   python summary.py
   ```

3. **驗證程式碼修改**
   ```bash
   cd fid_experiments
   python test_fix.py
   ```

## 優點

1. **集中管理**：所有 FID 實驗相關工具都在同一資料夾
2. **簡潔命名**：去除重複前綴（`fid_` / `_fid_`）
3. **清晰分離**：實驗腳本與核心 DiffAE 程式碼分離
4. **易於選擇**：在 IDE 中更容易定位和選擇相關檔案
5. **完整文檔**：包含 README.md 說明使用方式

## 注意事項

- `experiment_results/` 保留在 DiffAE root，不移入 `fid_experiments/`
- 所有路徑都使用相對路徑（`parent.parent`），確保可移植性
- 舊檔案已刪除，避免混淆
- 所有腳本都已測試路徑正確性

---

**完成時間**: 2026-01-15  
**變更內容**: 重組 FID 實驗腳本至 `fid_experiments/` 資料夾
