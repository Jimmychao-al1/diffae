# Stage2：Runtime 對齊與單輪 Refinement

## 定位

Stage2 **不**重新設計 scheduler，也不沿用舊版「A[b,z]→k」線性映射敘事。正式輸入是 **Stage1 產生的 `scheduler_config.json`**（已含 `shared_zones`、`k_per_zone`、每 block 的 `expanded_mask`）。

Stage2 負責：

1. 讀取並驗證 Stage1 設定；
2. 轉成 diffusion runtime 使用的 `cache_scheduler`（`diffusion/base.py` + `model/unet_autoenc.py` 既有 `cached_data` / `cached_scheduler` 路徑）；
3. 以 **同一組隨機種子** 各跑一次 **full-compute baseline**（無 cache）與 **cache 模式**；
4. 在 31 個 UNet block 上收集特徵，計算 baseline vs cache 的 L1 / L2 / cosine，並依 **block、DDIM timestep、shared zone** 聚合；
5. 執行 **單輪、保守** refinement（不迭代），輸出 refined 設定與診斷 JSON。

## 與 Stage1 的關係

| 項目 | Stage1 | Stage2 |
|------|--------|--------|
| 產出 `scheduler_config.json` | ✓（搜尋/最佳化 shared zones、k、mask） | 讀入 Stage1，必要時微調 `k_per_zone` 與 `expanded_mask` |
| Runtime `cache_scheduler` | 不直接產出 | **由 `expanded_mask` 推導**（每層一組 recompute 的 DDIM timestep 集合） |

## 時間軸（務必與程式註解一致）

- DDIM 迴圈變數 **i** 由 **T−1 遞減到 0**（見 `diffusion/base.py` 的 `indices = range(num_timesteps)[::-1]`）。
- Stage1 的 **`expanded_mask[b, step_idx]`**：
  - **`step_idx = 0`** → 第一步 → **DDIM `i = T−1`**；
  - **`step_idx = T−1`** → 最後一步 → **DDIM `i = 0`**。
- 換算：**`step_idx = (T−1) − i`**。
- **`True`** = 該步該 block **full compute（recompute）**；**`False`** = reuse cache。

`cache_scheduler` 的語意與 FID 實驗腳本一致：**若 DDIM timestep `i` 落在 `cache_scheduler[layer]` 集合內，則該層本步 recompute（`cached_scheduler=1`）**。

## Block 名稱對應（Stage1 JSON ↔ runtime）

| Stage1 `blocks[].name` | Runtime（`cache_scheduler` key） |
|------------------------|----------------------------------|
| `model.input_blocks.{i}` | `encoder_layer_{i}` |
| `model.middle_block` | `middle_layer` |
| `model.output_blocks.{i}` | `decoder_layer_{i}` |

## 模組說明

| 檔案 | 用途 |
|------|------|
| `stage2_scheduler_adapter.py` | `load_stage1_scheduler_config`、`validate_stage1_scheduler_config`、`stage1_mask_to_runtime_cache_scheduler`、`stage1_block_to_runtime_block` |
| `stage2_error_collector.py` | 透過 UNet 可選 `cache_debug_collector` 回呼，baseline/cache 兩趟收集（含 **reuse 步** 從 cache 取出的 tensor），輸出 `per_block_step_error`、`per_block_zone_error`、`global_summary` |
| `stage2_runtime_refine.py` | 載入模型、兩趟採樣、診斷、單輪 refinement、寫入三個 JSON |
| `verify_stage2.py` | `time_order`、31 blocks、`k` 長度、`expanded_mask[0]`、**mask ⊇ rebuild(k)**（允許 peak 多開） |

## Refinement 規則（第一版，單輪）

1. **Zone 層級**：若某 **(block, zone)** 的 **mean L1** **>** `zone_l1_threshold`，則該 block 的 **`k_per_zone[zone_id]` 減 1**（下限 1），接著用 **`rebuild_expanded_mask_from_shared_zones_and_k_per_zone`** 依 **新 k** 重建 **`expanded_mask`**（與 Stage1 時間軸一致）。  
   → Runtime 實際吃的是由 **`expanded_mask` 推導的 `cache_scheduler`**，因此必須在改 k 後重建 mask，調整才會生效。
2. **Peak 層級**：在 **診斷已含 reuse 步** 的前提下，若某 **(block, DDIM timestep i)** 的 **L1** **>** `peak_l1_threshold`，則將 **`expanded_mask[(T−1)−i]`** 設為 **`True`**（含「原本是 reuse、被強制改為 recompute」的情況）。`stage2_refinement_summary.json` 會記錄 **`was_reuse_before_peak_repair`**。
3. **強制第一步 full compute**：**`expanded_mask[0] == True`**（**`step_idx=0` ↔ DDIM `i=T−1`**）。

**最終 refined `scheduler_config`**：`k_per_zone`、**`expanded_mask`**（zone 基底 ∪ peak 額外開啟）、以及診斷內 **`refined_cache_scheduler`**（由最終 mask 推導）三者一致對齊。

## 輸出檔案（`stage2_runtime_refine.py`）

在 `--output_dir` 下**僅此三個 JSON**（不寫 `.pt` / `.npy` / 圖片等大檔）：

- `stage2_refined_scheduler_config.json`：refined 設定（`version` 標為 `stage2_refined_v1`）；
- `stage2_runtime_diagnostics.json`：特徵誤差、輸入 `cache_scheduler`，以及 **`refined_cache_scheduler`**（與最終 mask 對齊）；
- `stage2_refinement_summary.json`：threshold、k/mask 調整紀錄、依 DDIM timestep 聚合的 L1 摘要。

## 如何執行

於 **repo 根目錄**，並已能 import `torch` 與專案依賴：

```bash
python QATcode/cache_method/Stage2/stage2_runtime_refine.py \
  --scheduler_config QATcode/cache_method/Stage1/stage1_output/sweep_K16_sw5_lam1.0_kmax4/scheduler_config.json \
  --output_dir QATcode/cache_method/Stage2/stage2_output/run1 \
  --seed 0 \
  --zone_l1_threshold 0.02 \
  --peak_l1_threshold 0.08
```

驗證 refined 設定：

```bash
python QATcode/cache_method/Stage2/verify_stage2.py QATcode/cache_method/Stage2/stage2_output/run1/stage2_refined_scheduler_config.json
```

## 依賴與注意

- 模型載入流程對齊 `sample_lora_intmodel_v2` + `similarity_calculation._load_quant_and_ema_from_ckpt`（QAT checkpoint）。
- 採樣沿用 **`renderer.render_uncondition`** → `sampler.sample(..., cache_scheduler=...)`，**不重寫 sampler**。
- 預設 **batch=1**（`eval_num_images=1`）以降低 31×T 層特徵的記憶體壓力；若需更穩定統計可之後改為多張圖平均（非本版範圍）。
- `run_stage2_refine` 的 `device` 會用於量化模型與 calibration；結束時（含中途失敗）會在 `finally` 清除 `cache_debug_collector`、清空 collector 暫存並可選呼叫 `torch.cuda.empty_cache()`。

### Reuse 步的誤差比較

先前若只在 **forward** 路徑掛 hook，**reuse** 時不會進入該層 forward，逐點誤差會漏掉。  
修正後：`model/unet_autoenc.py` 在 **recompute 與 reuse** 兩條路都會呼叫可選的 **`cache_debug_collector`**（預設 `None`，不影響既有 sampling）。Stage2 用此回呼取得 **baseline 特徵** 與 **cache 路徑上取出的 `cached_data` 特徵**，在 **所有 timestep（含 reuse）** 對齊比較。

### 層覆蓋範圍

診斷對齊 UNet 31 個 runtime 層（`encoder_layer_0..14`、`middle_layer`、`decoder_layer_0..14`），不含影像 encoder。
