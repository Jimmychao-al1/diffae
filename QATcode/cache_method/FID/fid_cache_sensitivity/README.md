# FID Cache Sensitivity Analysis

此腳本用於測試 Diff-AE UNet 中每個 block 對 FID 的 cache 敏感度。

## 實驗目的

測量當對單一 layer 進行 cache 時，對 FID 分數的影響程度，以找出：
- 哪些 layer 對 cache 最不敏感（適合 cache）
- 哪些 layer 對 cache 最敏感（不適合 cache）

## 使用方法

### 1. 跑 Baseline（無 cache）

```bash
# 20 steps baseline
python QATcode/fid_cache_sensitivity/fid_cache_sensitivity.py --num_steps 20 --baseline

# 100 steps baseline
python QATcode/fid_cache_sensitivity/fid_cache_sensitivity.py --num_steps 100 --baseline
```

### 2. 測試單一 Layer

```bash
# 測試 encoder_layer_5，cache frequency k=3, 20 steps
python QATcode/fid_cache_sensitivity/fid_cache_sensitivity.py \
    --num_steps 20 \
    --layer encoder_layer_5 \
    --k 3

# 測試 middle_layer，k=4, 100 steps
python QATcode/fid_cache_sensitivity/fid_cache_sensitivity.py \
    --num_steps 100 \
    --layer middle_layer \
    --k 4
```

### 3. 測試所有 Layers

```bash
# 測試所有 31 個 layers，k=3, 20 steps
python QATcode/fid_cache_sensitivity/fid_cache_sensitivity.py \
    --num_steps 20 \
    --k 3

# 完整實驗：所有 k 值 (3, 4, 5)
for k in 3 4 5; do
    python QATcode/fid_cache_sensitivity/fid_cache_sensitivity.py \
        --num_steps 20 --k $k
done
```

## 參數說明

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--num_steps` | Diffusion steps (20 or 100) | 20 |
| `--eval_samples` | FID 評估樣本數 | 5000 |
| `--baseline` | 只跑 baseline FID (無 cache) | False |
| `--layer` | 指定要測試的 layer (不指定則測試所有 layer) | None |
| `--k` | Cache frequency (3, 4, or 5) | 必須指定 |
| `--output_json` | 結果 JSON 檔案名稱 | fid_sensitivity_results.json |
| `--log_file` | Log 檔案路徑 | fid_sensitivity.log |

## 輸出結果

結果會保存在 `QATcode/fid_cache_sensitivity/fid_sensitivity_results.json`：

```json
{
  "config": {
    "eval_samples": 5000
  },
  "results": {
    "T20": {
      "baseline_fid": 15.23,
      "k3": {
        "encoder_layer_0": {"fid": 15.34, "delta": 0.11},
        "encoder_layer_1": {"fid": 15.56, "delta": 0.33},
        ...
      },
      "k4": { ... },
      "k5": { ... }
    },
    "T100": {
      "baseline_fid": 12.34,
      "k3": { ... },
      ...
    }
  },
  "last_updated": "2026-01-15 10:30:00"
}
```

## Layer 名稱列表

總共 31 個 layers：

- Encoder: `encoder_layer_0` ~ `encoder_layer_14` (15 個)
- Middle: `middle_layer` (1 個)
- Decoder: `decoder_layer_0` ~ `decoder_layer_14` (15 個)

## 實驗規模

- **T=20, FID@5k**: 31 layers × 3 k值 = 93 次實驗
- **T=100, FID@5k**: 31 layers × 3 k值 = 93 次實驗
- **總計**: 186 次實驗 + 2 次 baseline

預估時間：~30-50 小時（假設每次 FID@5k 需要 10-15 分鐘）

## 斷點續跑

腳本會自動檢查已完成的實驗並跳過，可以安全地中斷並重新執行。

## 硬碟空間優化

腳本已優化為**不保留生成的圖片**：
- 使用固定的臨時資料夾（位於 `{conf.generate_dir}_temp_T{num_steps}`）
  - 例如：`checkpoints/ffhq128_autoenc_latent/mycache/gen_images_temp_T20`
- 每次實驗前自動清空該資料夾
- FID 計算完成後立即刪除所有圖片
- 只需要 5k 張圖片的暫存空間（約 1-2GB）

路徑選擇說明：使用 `conf.generate_dir` 作為基礎路徑，確保與 FID 計算的 cache 機制相容。

## 注意事項

1. 確保有足夠的 GPU 記憶體（建議 ≥16GB）
2. 硬碟空間需求大幅降低（只需要 ~2GB 暫存空間）
3. 可以使用 `nohup` 或 `screen` 在背景執行長時間實驗

```bash
# 使用 nohup 在背景執行
nohup python QATcode/fid_cache_sensitivity/fid_cache_sensitivity.py \
    --num_steps 20 --k 3 > nohup_k3.out 2>&1 &
```
