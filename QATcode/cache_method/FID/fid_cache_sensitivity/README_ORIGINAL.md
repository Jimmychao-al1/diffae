# FID Cache Sensitivity Analysis - 原本 Diff-AE

此腳本用於測試**原本 Diff-AE（非量化版本）**中每個 UNet block 對 FID 的 cache 敏感度。

## 與量化版本的差異

| 項目 | 量化版本 | 原本版本 |
|------|----------|----------|
| 模型類型 | QuantModel_DiffAE_LoRA | BeatGANsAutoencModel |
| Checkpoint | `diffae_step6_lora_best.pth` | `checkpoints/ffhq128_autoenc_latent/last.ckpt` |
| 需要校準資料 | ✅ | ❌ |
| 需要量化初始化 | ✅ | ❌ |
| Cache 機制 | ✅ 相同 | ✅ 相同 |
| FID 計算 | ✅ 相同 | ✅ 相同 |

## 使用方法

### 模式選擇

腳本支援兩種模式：

1. **預設模式**：載入模型後直接使用 `evaluate_fid`
2. **train() 模式**：使用 `train()` 的配置方式（類似 `run_ffhq128.py`），使用 `--use_train` 參數

### 1. 跑 Baseline（無 cache）

```bash
# 預設模式
python3 QATcode/fid_cache_sensitivity/fid_cache_sensitivity_original.py \
    --num_steps 20 --baseline

# train() 模式（類似 run_ffhq128.py）
python3 QATcode/fid_cache_sensitivity/fid_cache_sensitivity_original.py \
    --num_steps 20 --baseline --use_train
```

### 2. 測試單一 Layer

```bash
# 預設模式
python3 QATcode/fid_cache_sensitivity/fid_cache_sensitivity_original.py \
    --num_steps 20 --layer encoder_layer_5 --k 3

# train() 模式
python3 QATcode/fid_cache_sensitivity/fid_cache_sensitivity_original.py \
    --num_steps 20 --layer encoder_layer_5 --k 3 --use_train
```

### 3. 測試所有 Layers

```bash
# 預設模式
python3 QATcode/fid_cache_sensitivity/fid_cache_sensitivity_original.py \
    --num_steps 20 --k 3

# train() 模式
python3 QATcode/fid_cache_sensitivity/fid_cache_sensitivity_original.py \
    --num_steps 20 --k 3 --use_train
```

### 4. 完整實驗（使用腳本）

```bash
# 執行完整實驗：T=20,100 × k=3,4,5
bash QATcode/fid_cache_sensitivity/run_experiment_original.sh
```

## 參數說明

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--num_steps` | Diffusion steps (20 or 100) | 20 |
| `--eval_samples` | FID 評估樣本數 | 5000 |
| `--baseline` | 只跑 baseline FID (無 cache) | False |
| `--layer` | 指定要測試的 layer | None |
| `--k` | Cache frequency (3, 4, or 5) | 必須指定 |
| `--output_json` | 結果 JSON 檔案名稱 | fid_sensitivity_original_results.json |
| `--log_file` | Log 檔案路徑 | fid_sensitivity_original.log |
| `--use_train` | 使用 train() 函數方式（類似 run_ffhq128.py） | False |
| `--gpus` | GPU 列表（逗號分隔，例如: 0,1） | 0 |

## 輸出結果

結果會保存在 `QATcode/fid_cache_sensitivity_original/fid_sensitivity_original_results.json`：

```json
{
  "config": {
    "eval_samples": 5000,
    "model_type": "original_diffae"
  },
  "results": {
    "T20": {
      "baseline_fid": 12.34,
      "k3": {
        "encoder_layer_0": {"fid": 12.45, "delta": 0.11},
        ...
      },
      "k4": { ... },
      "k5": { ... }
    },
    "T100": { ... }
  }
}
```

## 檔案結構

```
QATcode/fid_cache_sensitivity_original/
├── fid_cache_sensitivity_original.py    # 主程式
├── fid_sensitivity_original_results.json # 結果檔案
├── fid_sensitivity_original_T20.log      # T=20 的 log
└── fid_sensitivity_original_T100.log     # T=100 的 log
```

## 實驗規模

- **T=20, FID@5k**: 31 layers × 3 k值 = 93 次實驗
- **T=100, FID@5k**: 31 layers × 3 k值 = 93 次實驗
- **總計**: 186 次實驗 + 2 次 baseline

預估時間：~30-50 小時（假設每次 FID@5k 需要 10-15 分鐘）

## 硬碟空間優化

腳本已優化為**不保留生成的圖片**：
- 使用固定的臨時資料夾（位於 `{conf.generate_dir}_temp_T{num_steps}`）
- 每次實驗前自動清空該資料夾
- FID 計算完成後立即刪除所有圖片
- 只需要 5k 張圖片的暫存空間（約 1-2GB）

## 注意事項

1. 確保原本的 checkpoint 存在：`checkpoints/ffhq128_autoenc_latent/last.ckpt`
2. 確保有足夠的 GPU 記憶體（建議 ≥16GB）
3. 硬碟空間需求：只需要 ~2GB 暫存空間
4. 可以使用 `nohup` 或 `screen` 在背景執行長時間實驗

## 與量化版本比較

執行完兩個版本的實驗後，可以比較：
- 量化版本 vs 原本版本的 cache sensitivity 差異
- 哪些 layer 在量化後對 cache 更敏感
- 量化對 cache 策略的影響
