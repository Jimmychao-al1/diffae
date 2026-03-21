# v_final Alignment Checklist (Stage 0/1)

本文件比對：

- 設計基準：`docs/cache_pipeline_v_final.md`
- 目前實作：`QATcode/cache_method/Stage0`、`QATcode/cache_method/Stage1` 與現有輸出

狀態標記：

- `implemented`
- `partially implemented`
- `unclear / needs inspection`
- `missing`

---

## 一覽表


| 檢查項目                              | 狀態                    | 判定摘要                                                      |
| --------------------------------- | --------------------- | --------------------------------------------------------- |
| Stage 0 evidence availability     | partially implemented | 三類證據主體存在，但 Stage0 主輸出未保留 L2 指標，且採 interval `(T-1)` 表示     |
| normalization availability        | implemented           | Stage0 對 L1/Cos/SVD 與 FID 權重皆有正規化/檢查流程                    |
| FID weighting availability        | implemented           | 已由 FID sensitivity 建立 block 風險權重並輸出                       |
| global drift segmentation logic   | implemented           | Stage1 具 FID-weighted global drift、平滑、變化量與 change points  |
| zone construction logic           | implemented           | Stage1 由 change points 形成 shared zones，並檢查 coverage       |
| tri-evidence aggregation logic    | partially implemented | 已聚合 `S_sim + S_svd + S_fid`，但 `S_sim` 目前用 L1（未融合 Cos/L2）。Known deviation from intended final design; currently retained as ablation baseline. |
| k mapping logic                   | implemented           | 已有 `A -> k_raw`、zone risk ceiling、regularization          |
| scheduler config output structure | implemented           | 已輸出 zones + per-block `k_per_zone` + params + diagnostics |


---

## 逐項對照與證據

### 1) Stage 0 evidence availability

- 狀態：`partially implemented`
- 設計期望：
  - Similarity（L1/L2/Cos 相關）
  - SVD drift/stability
  - FID sensitivity/safety
- 實作證據：
  - Stage0 讀取來源：
    - L1/Cos：`QATcode/cache_method/L1_L2_cosine/T_100/Res/result_npz/*.npz`
    - SVD：`QATcode/cache_method/SVD/svd_metrics/*.json`
    - FID：`QATcode/cache_method/FID/fid_cache_sensitivity/fid_sensitivity_results.json`
    - 來源：`QATcode/cache_method/Stage0/stage0e_normalization.py`
  - Stage0 輸出：
    - `l1_interval_norm.npy`
    - `cosdist_interval_norm.npy`
    - `svd_interval_norm.npy`
    - `fid_w_qdiffae_clip.npy`
    - `fid_w_qdiffae_rank.npy`
    - `block_names.npy`
- 判定原因：
  - 三大證據主體齊備，但 Stage0 統一輸出中未見 L2 對應陣列（僅 L1/Cos/SVD/FID）。
  - 資料語義以 interval `(B, 99)` 為主，而非直接 per-timestep `(B, 100)`。

### 2) normalization availability

- 狀態：`implemented`
- 實作證據：
  - `normalize_minmax(...)`：`QATcode/cache_method/Stage0/stage0e_normalization.py`
  - Stage0 對 L1/Cos/SVD 各自 min-max 正規化後輸出。
  - FID 權重含 noise 修正、quantile clipping、正規化與 rank-based 權重。
  - `verify_stage0e_output.sh` 會檢查 range/shape/statistics。

### 3) FID weighting availability

- 狀態：`implemented`
- 實作證據：
  - 讀取 `fid_sensitivity_results.json` 並抽取 T=100 的 delta-FID：
    - `load_delta_fid_qdiffae(...)`
  - 計算 block 權重：
    - `compute_fid_weights(...)`
  - 輸出：
    - `fid_w_qdiffae_clip.npy`
    - `fid_w_qdiffae_rank.npy`
  - 位置：`QATcode/cache_method/Stage0/stage0e_normalization.py`

### 4) global drift segmentation logic

- 狀態：`implemented`
- 實作證據（Stage1）：
  - FID 加權 global drift：
    - `compute_global_drift(...)`
  - 曲線平滑：
    - `moving_average(...)`
  - change magnitude / points：
    - `find_zones(...)`（支援 `topk` 或 `threshold`）
  - 輸出診斷：
    - `D_global`, `D_smooth`, `Delta`, `change_points`
  - 位置：`QATcode/cache_method/Stage1/stage1_scheduler.py`

### 5) zone construction logic

- 狀態：`implemented`
- 實作證據：
  - `find_zones(...)` 根據 change points 建立 shared zones。
  - 產出 `zones: [{id,t_start,t_end}, ...]`。
  - 內建 coverage 驗證（`0..T-1`）。
  - 驗證腳本：`QATcode/cache_method/Stage1/verify_scheduler.py`

### 6) tri-evidence aggregation logic

- 狀態：`partially implemented`
- 設計期望：
  - zone 內聚合 similarity evidence + SVD evidence + FID safety evidence
- 實作證據：
  - `compute_zone_evidence(...)` 使用：
    - `S_sim = 1 - l1_interval_norm`
    - `S_svd = 1 - svd_interval_norm`
    - `S_fid = 1 - FID_sens`
    - `A = alpha*S_sim + beta*S_svd + gamma*S_fid`
  - 位置：`QATcode/cache_method/Stage1/stage1_scheduler.py`
- 判定原因：
  - 已有 tri-evidence 聚合框架；
  - 但目前 `S_sim` 取自 L1，未直接把 Cos/L2 納入 Stage1 聚合公式。
  - Known deviation from intended final design; currently retained as ablation baseline.

### 7) k mapping logic

- 狀態：`implemented`
- 實作證據：
  - `map_to_k_raw(...)`：`A[b,z] -> k_raw[b,z]`
  - `compute_zone_risk_ceiling(...)` + `apply_zone_ceiling(...)`
  - `regularize_k(...)`（`delta1/nondecreasing/none`）
  - 最終輸出 `k_per_zone`
  - 位置：`QATcode/cache_method/Stage1/stage1_scheduler.py`

### 8) scheduler config output structure

- 狀態：`implemented`
- 實作證據：
  - `scheduler_config.json` 具：
    - `version`, `T`, `t_order`, `params`, `zones`, `blocks[].k_per_zone`
  - `scheduler_diagnostics.json` 具：
    - `D_global`, `D_smooth`, `Delta`, `change_points`, `R_z`, `k_max_z` 等
  - 既有結果：
    - `QATcode/cache_method/Stage1/stage1_output/topk_*/scheduler_config.json`
    - `QATcode/cache_method/Stage1/stage1_output/topk_*/scheduler_diagnostics.json`

---

## 目前結論（僅 Stage 0/1）

- Stage 0/1 整體對 v_final 設計屬「可用且主體對齊」。
- 主要差異集中在：
  1. Stage0 輸出層沒有獨立 L2 指標檔。
  2. Stage1 tri-evidence 的 similarity 聚合目前以 L1 為主，尚未融合 Cos/L2。Known deviation from intended final design; currently retained as ablation baseline.
- Stage 2 不在本次檢查與實作範圍內。
