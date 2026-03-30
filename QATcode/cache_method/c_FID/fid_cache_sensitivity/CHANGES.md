# FID Cache Sensitivity Analysis - 修改總結

## 修改日期
2026-01-15

## 修改內容

### 1. 檔案結構
```
QATcode/fid_cache_sensitivity/
├── fid_cache_sensitivity.py    # 主程式（已大幅重構）
├── README.md                    # 使用說明（新增）
└── CHANGES.md                   # 本文件（新增）
```

### 2. 主要變更

#### A. 清理移除的內容
- ✅ 移除 cache 分析相關 import（`SimpleBlockCollector`, `EmbOutputCollector` 等）
- ✅ 移除 `main_int_model()` 函數（INT 量化相關）
- ✅ 簡化 `TrainingConfig` → `ExperimentConfig`
- ✅ 移除舊的 cache scheduler 載入邏輯

#### B. 新增的核心函數

1. **`get_all_layer_names()`**
   - 返回所有 31 個 UNet layer 的名稱列表
   - encoder_layer_0~14, middle_layer, decoder_layer_0~14

2. **`create_simple_cache_config(layer_name, k, total_steps)`**
   - 為單一 layer 生成 cache 配置
   - k: cache frequency (每 k 個 timestep 重新計算一次)
   - 其他 layer 全部設為「每個 timestep 都計算」

3. **`load_results(json_path)`** 和 **`save_results(results, json_path)`**
   - JSON 格式的結果管理
   - 支援增量更新和斷點續跑

4. **`should_skip_experiment(results, step_config, k, layer)`**
   - 檢查實驗是否已完成
   - 避免重複執行

5. **`evaluate_fid_with_cache(base_model, cache_scheduler, num_steps)`**
   - 使用指定的 cache 配置生成圖片並計算 FID
   - 直接調用 `metrics.evaluate_fid()`

#### C. 重構的主函數

**`main_float_model(target_layer, k_value, baseline_only)`**
- 完全重寫為專注於 FID sensitivity 實驗
- 支援三種模式：
  1. Baseline only（無 cache）
  2. 單一 layer 測試
  3. 所有 layer 測試（自動迭代）
- 自動載入/保存結果
- 自動跳過已完成的實驗

#### D. 重新設計的 CLI

```bash
# 新的參數設計
--num_steps       # Diffusion steps (20 or 100)
--eval_samples    # FID 評估樣本數（預設 5000）
--baseline        # 只跑 baseline
--layer           # 指定要測試的 layer
--k               # Cache frequency (3, 4, or 5)
--output_json     # 結果檔案名稱
--log_file        # Log 檔案路徑
```

舊的參數（已移除）：
- `--enable_cache`
- `--cache_method`
- `--cache_threshold`
- `--enable_quantitative_analysis`
- `--analysis_num_samples`
- `--mode`（不再需要 float/int 切換）

### 3. 結果檔案格式

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
    "T100": { ... }
  },
  "last_updated": "2026-01-15 10:30:00"
}
```

### 4. 使用範例

```bash
# 1. 跑 baseline
python QATcode/fid_cache_sensitivity/fid_cache_sensitivity.py \
    --num_steps 20 --baseline

# 2. 測試單一 layer
python QATcode/fid_cache_sensitivity/fid_cache_sensitivity.py \
    --num_steps 20 --layer encoder_layer_5 --k 3

# 3. 測試所有 layers
python QATcode/fid_cache_sensitivity/fid_cache_sensitivity.py \
    --num_steps 20 --k 3

# 4. 完整實驗（所有 k 值）
for k in 3 4 5; do
    python QATcode/fid_cache_sensitivity/fid_cache_sensitivity.py \
        --num_steps 20 --k $k
done
```

### 5. 與原始程式碼的差異

| 項目 | 原始版本 | 新版本 |
|------|----------|--------|
| 用途 | LoRA + INT 量化 + Cache 分析 | FID Cache Sensitivity |
| Cache 配置 | 從 CSV 載入離線分析結果 | 動態生成簡單配置 |
| 實驗模式 | Float vs INT | 只有 Float（量化） |
| 主函數 | `main_float_model()` + `main_int_model()` | 只有 `main_float_model()` |
| 結果格式 | 多個 log 檔案 | 單一 JSON 檔案 |
| 斷點續跑 | 不支援 | 完整支援 |

### 6. 硬碟空間最佳化

為了節省硬碟空間，腳本已最佳化為：
- ✅ 使用固定的臨時資料夾（所有實驗共用）
- ✅ 每次實驗前自動清空資料夾
- ✅ FID 計算完成後立即刪除圖片
- ✅ 硬碟需求從 ~200GB 降低到 ~2GB

### 7. 未來可能的改進

1. **平行化**: 使用多 GPU 同時測試不同 layer
2. **可視化**: 自動生成 FID vs layer 的圖表
3. **統計分析**: 計算 delta 的分佈、找出最佳 cache 策略
4. **更多 k 值**: 支援任意 k 值（目前固定 3, 4, 5）

### 8. 驗證狀態

- ✅ Python 語法檢查通過
- ✅ 核心函數邏輯正確
- ✅ 硬碟空間最佳化已實作
- ⏳ 需要在實際環境中測試完整流程（需要 PyTorch + GPU）

## 預期實驗時間

- 單次 FID@5k: ~10-15 分鐘
- T20 全部實驗: 31 layers × 3 k = 93 次，約 15-23 小時
- T100 全部實驗: 93 次，約 15-23 小時
- **總計**: ~30-50 小時

## 注意事項

1. 確保 checkpoint 存在：
   - `QATcode/diffae_step6_lora_best_20steps.pth` (for T=20)
   - `QATcode/diffae_step6_lora_best.pth` (for T=100)

2. GPU 記憶體需求：建議 ≥16GB

3. 硬碟空間：只需要 ~2GB 暫存空間（圖片不保留）

4. 可使用 `nohup` 或 `screen` 在背景執行長時間實驗
