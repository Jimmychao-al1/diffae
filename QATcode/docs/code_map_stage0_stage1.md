# Stage 0/1 Code Map

## 目標
整理目前 `v_final` cache pipeline 的 Stage 0 / Stage 1 程式結構、資料流與輸出位置，作為後續檢查與文件化基準。

---

## Pipeline 總覽（Stage 0 -> Stage 1）

1. 上游證據產生（L1/Cosine、SVD、FID sensitivity）
2. Stage 0 正規化整合（輸出統一 `.npy`）
3. Stage 1 離線 scheduler 合成（輸出 `scheduler_config.json`）
4. Stage 1 驗證與可視化

---

## Entry Scripts

- `QATcode/cache_method/Stage0/stage0e_normalization.py`
  - Stage 0 主入口：讀取三種證據來源，產生正規化指標與 FID 權重。
- `QATcode/cache_method/Stage0/verify_stage0e_output.sh`
  - Stage 0 輸出檢查腳本（shape/range/基本統計）。
- `QATcode/cache_method/Stage1/stage1_scheduler.py`
  - Stage 1 主入口：zone segmentation + tri-evidence aggregation + `k[b,z]` 映射。
- `QATcode/cache_method/Stage1/verify_scheduler.py`
  - 驗證 scheduler config（zone coverage、k 範圍、recompute mask）。
- `QATcode/cache_method/Stage1/visualize_stage1.py`
  - 生成 Stage 1 結果可視化圖。
- `QATcode/cache_method/Stage1/run_stage1_sweep.sh`
  - 掃描多組 `cp_topk` 設定，批次產生/驗證/繪圖。

---

## Core Modules

- `QATcode/cache_method/Stage0/stage0e_normalization.py`
  - 整合來源：
    - L1/Cosine：`QATcode/cache_method/L1_L2_cosine/T_100/Res/result_npz/*.npz`
    - SVD：`QATcode/cache_method/SVD/svd_metrics/*.json`
    - FID：`QATcode/cache_method/FID/fid_cache_sensitivity/fid_sensitivity_results.json`
  - 核心產物：`block_names`、interval-wise 指標、FID weights。

- `QATcode/cache_method/Stage1/stage1_scheduler.py`
  - 讀取 Stage 0 `.npy` 結果。
  - 計算 global drift / change points / zones。
  - 計算 tri-evidence score 並映射為 `k_per_zone`。
  - 輸出 scheduler config 與 diagnostics。

---

## Helper Scripts / Upstream Evidence

- `QATcode/cache_method/SVD/svd_metrics.py`
  - 從 feature tensors 計算 SVD 子空間距離與相關指標，輸出 JSON 給 Stage 0。
- `QATcode/cache_method/FID/fid_cache_sensitivity/fid_cache_sensitivity.py`
  - 產生 block-level FID sensitivity 結果 JSON，供 Stage 0 權重計算。
- `QATcode/cache_method/L1_L2_cosine/similarity_calculation.py`
  - 早期功能：繪製 similarity 相關結果圖（歷史分析用途）。
  - 在目前 Stage 0/1 主鏈路中，Stage 0 實際依賴的是 `T_100/Res/result_npz/*.npz` 既有輸出。

---

## Config-Related Files

- Stage 0 文件：
  - `QATcode/cache_method/Stage0/README.md`
- Stage 1 文件：
  - `QATcode/cache_method/Stage1/README.md`
  - `QATcode/cache_method/Stage1/QUICKSTART.md`
- Stage 1 參數來源：
  - `stage1_scheduler.py` CLI 參數（`alpha/beta/gamma`, `k_min/k_max`, `cp_method/cp_topk`, `regularize`）
  - `run_stage1_sweep.sh` 批次參數（topK 列表）

---

## Expected Outputs

### Stage 0 輸出（`QATcode/cache_method/Stage0/stage0e_output/`）

- `block_names.npy`
- `l1_interval_norm.npy`
- `cosdist_interval_norm.npy`
- `svd_interval_norm.npy`
- `fid_w_qdiffae_clip.npy`
- `fid_w_qdiffae_rank.npy`

### Stage 1 輸出（`QATcode/cache_method/Stage1/stage1_output/topk_*/`）

- `scheduler_config.json`
- `scheduler_diagnostics.json`

### Stage 1 圖表（`QATcode/cache_method/Stage1/stage1_figures/topk_*/`）

- drift + zones 圖
- k heatmap/histogram
- zone risk 圖

---

## 檔案角色對照（精簡）

- Stage 0：將三種證據轉為可比較、可聚合的統一正規化數據格式。
- Stage 1：把 Stage 0 指標合成靜態 scheduler（zones + 每 block 的 `k_per_zone`）。
- Verify/Visualize：確保 config 合法並提供人工檢視依據。
- Sweep：在不改公式的前提下比較不同切分強度（`cp_topk`）下的 scheduler 結果。

---

## 現況邊界

- Stage 2 尚未進入可實作狀態，本文件不含 Stage 2 實作細節。
- Diff-AE training optimization 屬討論中，本文件不涉及訓練邏輯修改。
