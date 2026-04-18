# FID Cache Sensitivity Analysis

此模組測試 Diff-AE / Q-DiffAE UNet 中各 block 對 FID 的 cache 敏感度，以找出哪些 block 適合（或不適合）被 cache。

---

## 目錄結構

```
c_FID/fid_cache_sensitivity/
├── fid_cache_sensitivity.py        # 主程式
└── fid_sensitivity_results.json    # 實驗結果（自動累積，支援斷點續跑）
```

---

## 實作架構

- `get_all_layer_names()`：回傳所有 31 個 UNet layer 名稱（encoder × 15、middle × 1、decoder × 15）
- `create_simple_cache_config(layer_name, k, total_steps)`：為單一 layer 動態生成 cache 配置，其他 layer 全部設為每步計算
- `evaluate_fid_with_cache()`：使用指定 cache 配置生成圖片並計算 FID（呼叫 `metrics.evaluate_fid()`）
- **硬碟最佳化**：使用固定臨時資料夾，每次實驗前自動清空，FID 計算完成後立即刪除圖片（需求 ~2GB）
- **斷點續跑**：自動載入/保存 JSON 結果，跳過已完成的實驗

---

## Quick Start

### 1. 跑 Baseline（無 cache）

```bash
# T=20
python QATcode/cache_method/c_FID/fid_cache_sensitivity/fid_cache_sensitivity.py \
    --num_steps 20 --baseline

# T=100
python QATcode/cache_method/c_FID/fid_cache_sensitivity/fid_cache_sensitivity.py \
    --num_steps 100 --baseline
```

### 2. 測試單一 Layer

```bash
python QATcode/cache_method/c_FID/fid_cache_sensitivity/fid_cache_sensitivity.py \
    --num_steps 20 --layer encoder_layer_5 --k 3
```

### 3. 測試所有 Layers（完整實驗）

```bash
# 單一 k 值
python QATcode/cache_method/c_FID/fid_cache_sensitivity/fid_cache_sensitivity.py \
    --num_steps 20 --k 3

# 全部 k 值（3, 4, 5）
for k in 3 4 5; do
    python QATcode/cache_method/c_FID/fid_cache_sensitivity/fid_cache_sensitivity.py \
        --num_steps 20 --k $k
done
```

### 背景執行（長時間實驗）

```bash
nohup python QATcode/cache_method/c_FID/fid_cache_sensitivity/fid_cache_sensitivity.py \
    --num_steps 20 --k 3 > nohup_k3.out 2>&1 &
```

---

## CLI 參數說明

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--num_steps` | Diffusion steps（20 或 100） | 20 |
| `--eval_samples` | FID 評估樣本數 | 5000 |
| `--baseline` | 只跑 baseline FID（無 cache） | False |
| `--layer` | 指定要測試的單一 layer；不指定則測試所有 layer | None |
| `--k` | Cache frequency（3、4 或 5） | 必須指定 |
| `--output_json` | 結果 JSON 檔案名稱 | `fid_sensitivity_results.json` |
| `--log_file` | Log 檔案路徑 | `fid_sensitivity.log` |

---

## Layer 名稱列表

共 31 個 layer：

- **Encoder**：`encoder_layer_0` ~ `encoder_layer_14`（15 個）
- **Middle**：`middle_layer`（1 個）
- **Decoder**：`decoder_layer_0` ~ `decoder_layer_14`（15 個）

---

## 輸出格式

結果保存於 `QATcode/cache_method/c_FID/fid_cache_sensitivity/fid_sensitivity_results.json`：

```json
{
  "config": { "eval_samples": 5000 },
  "results": {
    "T20": {
      "baseline_fid": 15.23,
      "k3": {
        "encoder_layer_0": {"fid": 15.34, "delta": 0.11},
        "encoder_layer_1": {"fid": 15.56, "delta": 0.33}
      },
      "k4": { "..." : "..." },
      "k5": { "..." : "..." }
    },
    "T100": { "..." : "..." }
  },
  "last_updated": "2026-01-15 10:30:00"
}
```

---

## 實驗規模

| 設定 | 數量 | 預估時間 |
|------|------|---------|
| T=20，所有 layer，k=3,4,5 | 31 × 3 = 93 次 | ~15–23 小時 |
| T=100，所有 layer，k=3,4,5 | 31 × 3 = 93 次 | ~15–23 小時 |
| **合計** | 186 次 + 2 次 baseline | ~30–50 小時 |

---

## 注意事項

1. GPU 記憶體建議 ≥ 16GB
2. 硬碟空間需求：~2GB 暫存（圖片不保留）
3. 可安全中斷並重新執行，已完成的實驗會自動跳過
