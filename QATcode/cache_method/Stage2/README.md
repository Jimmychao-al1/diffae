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
| `stage2_error_collector.py` | 31 層 hook，baseline/cache 兩趟收集，輸出 `per_block_step_error`、`per_block_zone_error`、`global_summary` |
| `stage2_runtime_refine.py` | 載入模型、兩趟採樣、診斷、單輪 refinement、寫入三個 JSON |
| `verify_stage2.py` | 檢查 refined config：`T`、zones 分割、`expanded_mask` 維度、第一步 full、`k≥1` |

## Refinement 規則（第一版，單輪）

1. **Zone 層級**：若某 **(block, zone)** 的 **mean L1**（在該 zone 涵蓋的 DDIM timestep 上平均）**>** `zone_l1_threshold`，則該 block 的 **`k_per_zone[zone_id]` 減 1**，且 **不得小於 1**。
2. **Peak 層級**：若某 **(block, DDIM timestep i)** 的 **L1** **>** `peak_l1_threshold`，則將該 block 的 **`expanded_mask[(T−1)−i]`** 強制設為 **`True`**。
3. 最後 **強制第一步 full compute**：若 **`expanded_mask[0]`** 仍為 `False`，改為 `True`（**`step_idx=0` ↔ DDIM `i=T−1`**）。

## 輸出檔案（`stage2_runtime_refine.py`）

在 `--output_dir` 下：

- `stage2_refined_scheduler_config.json`：refined 設定（`version` 標為 `stage2_refined_v1`）；
- `stage2_runtime_diagnostics.json`：特徵誤差與 `cache_scheduler`（list 形式）；
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

### 特徵誤差統計的限制（第一版）

在 **cache 模式**下，若某層在某 DDIM 步為 **reuse**（`cached_scheduler=0`），`model/unet_autoenc.py` 會 **跳過該層 forward**，因此 `TimestepEmbedSequential` 的 **hook 不會執行**。  
本版 `stage2_error_collector.py` 只在 **baseline 與 cache 兩邊都實際 forward 到該層** 的 `(block, timestep)` 上計算 L1/L2/cosine；reuse 步不納入逐點誤差。若要對 reuse 步比較「快取值 vs 全算值」，需改動 UNet（例如在讀取 `cached_data` 時取樣）或另行設計，**不在本版範圍**。

### Hook 掃描

`stage2_error_collector` 會排除 **`model.encoder.input_blocks.*`**（影像 encoder），只鎖定 UNet 的 `model.input_blocks.*` / `model.middle_block` / `model.output_blocks.*`。
