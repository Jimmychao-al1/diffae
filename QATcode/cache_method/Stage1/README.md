# Stage-1: Offline Scheduler Synthesis (T=100)

## 目的

從 Stage-0 的 tri-evidence（L1/SVD similarity + FID sensitivity）合成**靜態 cache scheduler**：
- **Zones**: 將 **analysis axis** 上的點索引 **0..99** 分成若干個 zone（shared across all blocks）
- **k[b,z]**: 每個 block 在每個 zone 的 cache frequency（每隔 k 個 **axis 索引** recompute）

### 時間軸慣例（與 Stage 0、L1_L2_cosine .npz 一致）

- **analysis axis** `axis_idx ∈ [0,99]`：與 similarity 收集的 step_counter、圖橫軸左→右一致。
- **DDIM** 進模型的 timestep：**`t_ddim = 99 - axis_idx`**（採樣順序為 t_ddim 從 99→0）。
- **Interval 陣列**長度 99：第 `j` 欄 = interval j（axis j 與 j+1 之間）⇔ DDIM **(99−j)→(98−j)**。
- 詳細推導與 audit：**`QATcode/docs/cache_time_axis_audit.md`**

## 演算法流程

### 1. 載入 Stage-0E 輸出
- **S_sim[b,t]**: L1-based stability（= 1 - L1_interval_norm）
- **S_svd[b,t]**: SVD-based stability（= 1 - SVD_interval_norm）
- **d_norm[b,t]**: SVD drift（= SVD_interval_norm）
- **FID_sens[b]**: FID sensitivity weight

語義：
- S_sim, S_svd: 越大越穩定/越可 cache
- d_norm: 越大越不穩定
- FID_sens: 越大越敏感（對 FID 影響大）

### 2. FID-weighted Global Drift
```
D_global[t] = sum_b (w_b * d_norm[b,t]) / sum_b w_b
D_smooth[t] = moving_avg(D_global, window=5)
```
- 用 FID sensitivity 加權，計算全域的 drift
- 平滑後用於找 change points

### 3. Zone Segmentation
```
Δ[t] = |D_smooth[t] - D_smooth[t-1]|
```
- 取 Δ 最大的 K=6 個點作為 change points（預設）
- 切分成 zones（shared across blocks）

### 4. Zone-level Tri-evidence Score
```
A[b,z] = α*S_sim[b,z] + β*S_svd[b,z] + γ*(1-FID_sens[b])
```
- S_sim[b,z] = zone z 內的平均 L1 stability
- S_svd[b,z] = zone z 內的平均 SVD stability
- 1-FID_sens[b] = FID-safe score（越大越可 cache）
- 預設 α=β=γ=1/3

### 5. Map to k_raw
```
k_raw[b,z] = k_min + round(A[b,z] * (k_max - k_min))
```
- A[b,z] 越大（越穩定）→ k 越大（越久才重算）
- 預設 k_min=1, k_max=8

### 6. Zone-level Risk Ceiling
```
R_z = mean_{t in zone z} D_smooth[t]
k_max_z = k_min + round((1 - R_z) * (k_max - k_min))
k[b,z] = min(k_raw[b,z], k_max_z)
```
- Zone 越危險（R_z 越大）→ k_max ceiling 越低
- 防止在高風險 zone 使用太大的 k

### 7. Regularization
- **delta1** (預設): 保證 `|k[b,z] - k[b,z-1]| <= 1`（相鄰 zone 不跳太多）
- **nondecreasing**: 保證 `k[b,z] >= k[b,z-1]`（遞增）
- **none**: 不正規化

## 使用方式

### 基本執行
```bash
cd /home/jimmy/diffae

python3 QATcode/cache_method/Stage1/stage1_scheduler.py \
  --stage0_dir QATcode/cache_method/Stage0/stage0e_output \
  --output_dir QATcode/cache_method/Stage1/stage1_output
```

### 調整參數
```bash
python3 QATcode/cache_method/Stage1/stage1_scheduler.py \
  --stage0_dir QATcode/cache_method/Stage0/stage0e_output \
  --output_dir QATcode/cache_method/Stage1/stage1_output \
  --alpha 0.4 --beta 0.4 --gamma 0.2 \
  --k_min 1 --k_max 10 \
  --smooth_window 7 \
  --cp_method topk --cp_topk 8 \
  --regularize nondecreasing
```

### Self-test（用假資料測試）
```bash
python3 QATcode/cache_method/Stage1/stage1_scheduler.py --self_test
```

## 輸出檔案

### 1. `scheduler_config.json`
主要配置檔，包含：
- **zones**: 所有 zones 的定義（**axis_start, axis_end**；舊檔可能僅有 t_start/t_end，語意相同）
- **axis_convention / ddim_timestep_formula / analysis_axis_order**：與 Stage 0 對齊的元資料
- **blocks**: 每個 block 的 k_per_zone 列表
- **params**: 所有演算法參數

格式：
```json
{
  "version": "v_final_stage1",
  "T": 100,
  "t_order": "analysis_axis_0_to_99_inclusive",
  "analysis_axis_order": "analysis_axis_0_to_99_inclusive",
  "axis_convention": "analysis_axis",
  "ddim_timestep_formula": "t_ddim = 99 - axis_idx",
  "params": {
    "alpha": 0.333,
    "beta": 0.333,
    "gamma": 0.334,
    "k_min": 1,
    "k_max": 8,
    "smooth_window": 5,
    "cp_method": "topk",
    "cp_topk": 6,
    "regularize": "delta1"
  },
  "zones": [
    {"id": 0, "axis_start": 0, "axis_end": 62, "t_start": 0, "t_end": 62},
    {"id": 1, "axis_start": 63, "axis_end": 67, "t_start": 63, "t_end": 67},
    ...
  ],
  "blocks": [
    {
      "id": 0,
      "name": "model.input_blocks.0",
      "k_per_zone": [6, 5, 6, 5, 5, 5, 5]
    },
    ...
  ]
}
```

### 2. `scheduler_diagnostics.json`
診斷資料，包含：
- **D_global**, **D_smooth**: 全域 drift 曲線
- **Delta**: 變化幅度
- **change_points**: 找到的切分點（timestep）
- **R_z**, **k_max_z**: zone-level risk 和 ceiling
- **A_stats**, **k_raw_stats**, **k_final_stats**: 統計摘要

## 可視化

生成 Stage-1 結果的圖表：
```bash
python3 QATcode/cache_method/Stage1/visualize_stage1.py
```

輸出（存至 `stage1_figures/`）：
1. **1_drift_and_zones.png**: D_global/D_smooth 曲線 + zone 切分 + change points
2. **2_k_heatmap.png**: K 分佈熱圖（B × Z）
3. **3_k_histogram.png**: K 值直方圖
4. **4_zone_risk.png**: Zone risk R_z + k_max ceiling

## 數值檢查

當前結果（T=100, B=31）：
- **Zones**: 7 個（Zone 0 很長 [0..62]，後期 [63..99] 分成 6 個小 zone）
- **K 範圍**: [3, 8]，mean=7.16，median=7
- **K 分佈**: 主要集中在 7-8（高 cache frequency）
- **Zone risk**: Zone 0-4 較低（<0.07），Zone 5-6 較高（~0.2）
- **Recompute 比例**: 以 Block 0 為例，100 個 timestep 中只需 recompute 20 次（節省 80%）

## 參數建議

### Tri-evidence 權重（α, β, γ）
- **預設**: α=β=γ=1/3（平衡）
- **強調 FID**: α=0.2, β=0.2, γ=0.6（優先考慮 FID-sensitive blocks）
- **強調 temporal stability**: α=0.5, β=0.4, γ=0.1（優先考慮時序穩定性）

### k 範圍（k_min, k_max）
- **預設**: [1, 8]
- **更激進 cache**: [2, 10]（節省更多計算，但可能略降品質）
- **更保守**: [1, 5]（品質優先）

### Change point 檢測
- **topk=6** (預設): 適中的 zone 數量（5-8 個）
- **topk=4**: 較少 zone（大區塊，k 變化較少）
- **topk=10**: 較多 zone（細粒度，k 更 adaptive）

### Regularization
- **delta1** (預設): 平滑但不強制遞增（允許 k 略降）
- **nondecreasing**: 強制 k 遞增（保守，適合需要單調性的場景）
- **none**: 不限制（可能出現大跳躍）

## 後續使用

Stage-2 將使用 `scheduler_config.json` 來：
1. 實作 runtime cache scheduler
2. 在 inference 時決定哪些 timestep recompute、哪些用 cache
3. 驗證 FID/quality 和 speed-up

## 檔案結構

```
QATcode/cache_method/Stage1/
├── __init__.py
├── stage1_scheduler.py          # 主程式
├── visualize_stage1.py           # 可視化腳本
├── README.md                     # 本文件
├── stage1_output/                # 輸出目錄
│   ├── scheduler_config.json
│   └── scheduler_diagnostics.json
└── stage1_figures/               # 圖表
    ├── 1_drift_and_zones.png
    ├── 2_k_heatmap.png
    ├── 3_k_histogram.png
    └── 4_zone_risk.png
```

## 設計特點

1. **Data-driven**: 完全基於 Stage-0 的量化指標，無 hand-crafted heuristics
2. **Zone-based**: Shared zones 讓所有 blocks 共用時序切分，簡化 scheduling
3. **Risk-aware**: Zone-level ceiling 防止在高風險區域過度 cache
4. **Smooth**: Regularization 避免 k 劇烈跳躍，保持穩定性
5. **可調**: 所有參數可透過 CLI 調整，支援不同策略（激進 vs 保守）

## 限制與改進方向

### 當前限制
1. **固定 T=100**: 只支援 DDIM-100，未泛化到其他 T
2. **Interval-wise**: 使用 (T-1) 個 interval，最後一個 timestep 沒有 transition
3. **單一策略**: 所有 blocks 用相同的 tri-evidence formula

### 可能改進
1. **支援多種 T**: T=20, T=50, T=100 的統一 pipeline
2. **Per-block 策略**: 某些 blocks 可能更重視 FID，某些更重視 similarity
3. **Dynamic adjustment**: Runtime 根據前幾步的實際 drift 微調 k
4. **更細粒度**: 支援 per-timestep k（不一定需要 zones）

---

**實作日期**: 2026-02-10  
**版本**: v_final_stage1
