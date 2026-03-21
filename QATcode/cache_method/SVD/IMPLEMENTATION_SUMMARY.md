# SVD Feature Analysis - 實作總結

已完成 SVD 特徵分析的完整管線實作，包含三個階段。

---

## 已實作內容

### ✅ 階段 A：Feature 收集

**檔案**：
- `collect_features_for_svd.py`：主程式（SvdFeatureCollector + 主流程）
- `run_svd_feature_experiments.sh`：31 個 block 批次執行腳本

**功能**：
- 複用 `similarity_calculation.py` 的模型載入與 `evaluate_fid` 流程
- `SvdFeatureCollector` 類：
  - 對單一 block 註冊 forward hook
  - 在每個 timestep 累積 feature（保持在 GPU）
  - Finalize 時 concat + 截斷到 target_N + 寫 `t_{t}.pt`
  - 輸出 `meta.json`（block、N、T、C、H、W）
- GPU 優化：hook 內只 detach + append，finalize 才搬 CPU

**關鍵設計**：
- 每個 t 累積固定 N 個樣本（所有 timestep 共用同一組 sample）
- `eval_num_images = ceil(target_N / 32) * 32`，確保每個 t 都能累積到 target_N
- Block 名稱：CLI 用 `model.input_blocks.0`，slug 用 `model_input_blocks_0`

**CLI 參數**：
- `--svd_target_block`：目標 block（必填）
- `--svd_target_N`：每個 timestep 的樣本數（預設 32）
- `--num_steps`：T（預設 100）
- `--svd_output_root`：輸出根目錄
- `--log_file`：log 檔案路徑

---

### ✅ 階段 B：SVD 指標計算

**檔案**：
- `svd_metrics.py`：SVD 分析主程式

**功能**：
- `load_features()`：載入某 block 的所有 `t_{t}.pt` 與 `meta.json`
- `compute_covariance_eigen()`：對 `(N,C,H,W)` reshape 成 `(C,M)`，計算 covariance `Σ = (X @ X^T) / M`，eigen decomposition
- `compute_rank_r()`：用 cumulative energy 定 rank r
- `compute_subspace_distance()`：計算 `d(t,t-1) = 1 - (||U_t^{(r)T} U_{t-1}^{(r)}||_F^2 / r)`
- `compute_energy_ratios()`：計算 `E_t(k)` 對 k ∈ {4, 8, 16, 32, 64}
- `process_single_block()`：完整流程 + 寫 JSON

**輸出 JSON 格式**：
- `block`、`T`、`C`、`N`、`H`、`W`
- `rank_r`、`representative_t`、`energy_threshold`、`actual_energy_at_r`
- `timesteps`：[0, 1, ..., 99]
- `subspace_dist`：長度 T（t=0 為 0.0）
- `energy_ratio`：{"k4": [...], "k8": [...], ...}

**CLI 參數**：
- `--feature_dir`：單一 block 的 feature 目錄
- `--feature_root` + `--all`：批次處理所有 block
- `--representative-t`：代表 timestep（-1 表示 T-1，預設）
- `--energy-threshold`：cumulative energy 門檻（預設 0.98）
- `--output_root`：輸出根目錄

---

### ✅ 階段 C：相關性分析

**檔案**：
- `correlate_svd_similarity.py`：相關性分析與視覺化

**功能**：
- `load_svd_metrics()`：載入 B 的 JSON
- `load_similarity_npz()`：載入 similarity NPZ（`l1_step_mean`, `cos_step_mean` 等）
- `compute_correlations()`：計算 Pearson / Spearman（含 p-value）
- `plot_alignment()`：對齊曲線圖（dual y-axis）
- `plot_scatter()`：散點圖（L1 vs SVD、CosDist vs SVD）

**相關性計算**：
- 對 t=1..T-1（跳過 t=0）：
  - `L1[t] = l1_step_mean[t]`
  - `L2[t] = l2_step_mean[t]`
  - `CosDist[t] = 1 - cos_step_mean[t]`
  - `SVDdist[t] = subspace_dist[t]`
- 計算：
  - L1 vs SVD（Pearson、Spearman）
  - L2 vs SVD（Pearson、Spearman）
  - CosDist vs SVD（Pearson、Spearman）

**輸出**：
- `correlation/<block_slug>.json`：相關性統計
- `correlation/figures/<block_slug>_alignment.png`：對齊曲線圖
- `correlation/figures/<block_slug>_scatter.png`：散點圖

**CLI 參數**：
- `--svd_metrics` + `--similarity_npz`：單一 block
- `--svd_metrics_dir` + `--similarity_npz_dir` + `--all`：批次處理
- `--plot`：生成圖表（預設開啟）
- `--output_root`：輸出根目錄

---

## 資料流

```
階段 A（Feature 收集）
  ↓ 輸出：svd_features/<block_slug>/t_{0..99}.pt + meta.json
階段 B（SVD 指標）
  ↓ 輸出：svd_metrics/<block_slug>.json
階段 C（相關性分析）
  ↓ 輸入：B 的 JSON + similarity NPZ
  ↓ 輸出：correlation/<block_slug>.json + figures/*.png
```

---

## 與 similarity_calculation.py 的關係

- **複用**：
  - 模型載入（`load_diffae_model`）
  - 量化流程（`create_float_quantized_model`、校準資料）
  - Hook 架構（pre-hook 更新 step、block hook 抓輸出）
  - `evaluate_fid` 驅動生成流程
  
- **不複用**：
  - L1/L2/cosine 計算與累積
  - NPZ/CSV/PNG 輸出
  - Multi-batch 折線圖

- **Finalize 差異**：
  - Similarity：計算統計、寫 NPZ/CSV、畫圖
  - SVD：concat + 截斷、寫 `.pt`、寫 `meta.json`

---

## 下一步

1. **測試階段 A**：先對單一 block（如 `model.input_blocks.0`）跑一次，確認 feature 正確收集
2. **測試階段 B**：對測試 block 計算 SVD 指標，檢查 rank r 與 subspace_dist 是否合理
3. **測試階段 C**：確認相關性數值與圖表
4. **批次執行**：若單一 block 測試通過，再跑 31 個 block

若有任何需要調整的參數或邏輯，修改對應 Python 腳本的 CLI 預設值或實作即可。
