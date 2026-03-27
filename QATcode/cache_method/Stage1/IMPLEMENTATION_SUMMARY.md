# Stage-1 Implementation Summary

## 實作日期
2026-02-10

## 目標
從 Stage-0 的 tri-evidence 合成靜態 cache scheduler（zones + k[b,z]）

## 完成項目

### 1. 核心模組 (`stage1_scheduler.py`)

✅ **資料載入器**
- 自動偵測 Stage-0E 輸出目錄
- 語義轉換：變化量 → 穩定性分數
- Shape & range 驗證
- 支援 (B=31, T-1=99) 的 interval-wise 資料

✅ **FID-weighted Global Drift**
- 權重計算：w_b = FID_sens[b]
- D_global[t] = Σ(w_b × d_norm[b,t]) / Σw_b
- Moving average 平滑（window=5）

✅ **Zone Segmentation**
- Δ[t] = |D_smooth[t] - D_smooth[t-1]|
- Change point 檢測：topk (預設 K=6) 或 threshold
- 生成 shared zones（覆蓋 t=0..99）

✅ **Tri-evidence Score**
- A[b,z] = α×S_sim[b,z] + β×S_svd[b,z] + γ×(1-FID_sens[b])
- Zone-level 聚合（mean over zone timesteps）
- 預設權重：α=β=γ=1/3

✅ **k 對應映射**
- k_raw[b,z] = k_min + round(A[b,z] × (k_max - k_min))
- 範圍：[1, 8]（可調）

✅ **Zone-level Risk Ceiling**
- R_z = mean_{t in zone z} D_smooth[t]
- k_max_z = k_min + round((1 - R_z) × (k_max - k_min))
- k_ceiling[b,z] = min(k_raw[b,z], k_max_z)

✅ **Regularization**
- **delta1**: |k[b,z] - k[b,z-1]| ≤ 1
- **nondecreasing**: k[b,z] ≥ k[b,z-1]
- **none**: 不限制

✅ **輸出格式**
- `scheduler_config.json`: zones + blocks + k_per_zone + params
- `scheduler_diagnostics.json`: D_global, D_smooth, Delta, R_z, k_max_z, 統計

✅ **Helper 函數**
- `build_recompute_mask(T, zones, k_per_zone)`: 轉成 per-timestep bool mask
- CLI 支援（argparse）
- Self-test（用假資料驗證邏輯）

### 2. 可視化模組 (`visualize_stage1.py`)

✅ **4 張圖表**
1. **Drift and Zones**: D_global/D_smooth 曲線 + Delta + zone backgrounds
2. **K Heatmap**: (B × Z) 熱圖，顯示每個 block/zone 的 k
3. **K Histogram**: 整體 k 分佈
4. **Zone Risk**: R_z bar chart + k_max_z bar chart

✅ **自動化**
- 一鍵生成所有圖表
- 存至 `stage1_figures/`

### 3. 驗證模組 (`verify_scheduler.py`)

✅ **完整性檢查**
- Zone coverage：是否完整覆蓋 0..T-1
- K range：是否在 [k_min, k_max]
- Zone start recompute：每個 zone 起點是否 True
- Regularization：是否符合設定的模式

✅ **統計分析**
- Recompute count per block（min/max/mean/median）
- Cache 節省率
- Sample recompute patterns

### 4. 文檔

✅ **README.md**
- 演算法流程詳解
- 使用方式
- 參數建議
- 輸出格式
- 設計特點與限制

✅ **QUICKSTART.md**
- 一鍵執行指令
- 關鍵結果摘要
- 調整策略範例

✅ **IMPLEMENTATION_SUMMARY.md** (本文件)

## 關鍵資料（T=100, B=31）

### Zones
```
Zone 0: t=0..62   (len=63)  - 長期穩定區
Zone 1: t=63..67  (len=5)   - 過渡區
Zone 2: t=68..87  (len=20)  - 中等穩定區
Zone 3: t=88..92  (len=5)   - 過渡區
Zone 4: t=93..95  (len=3)   - 短期不穩定區
Zone 5: t=96..96  (len=1)   - 極短不穩定區
Zone 6: t=97..99  (len=3)   - 最終區
```

### K 分佈
- **範圍**: [3, 8]
- **平均**: 7.16
- **中位數**: 7
- **分佈**: 主要集中在 7-8（約 75%）

### Cache 節省
- **平均 recompute 比例**: 16.6%（只需重算 16-22 timesteps）
- **平均節省**: 83.4%
- **最好情況**: 16/100 (Block 2)
- **最差情況**: 22/100 (Block 0)

### Zone Risk
```
Zone 0-2: R_z < 0.04  (低風險)
Zone 3-4: R_z ~ 0.07  (中等風險)
Zone 5-6: R_z ~ 0.20  (高風險，k_max ceiling 降到 7)
```

## 驗證結果

✅ **所有檢查通過**
- Zone coverage: 完整覆蓋 0..99
- K range: 全在 [1, 8]
- Zone start recompute: 所有 zone 起點都會 recompute
- Regularization (delta1): 無違規

## 設計決策

### 1. Interval-wise vs Timestep-wise
- **選擇**: Interval-wise（T-1 個 transition）
- **理由**: 更符合 Stage-0 的 drift 語義（t → t+1 的變化）
- **Trade-off**: 最後一個 timestep (t=99) 沒有 forward transition

### 2. Shared Zones vs Per-block Zones
- **選擇**: Shared zones（所有 blocks 共用）
- **理由**: 簡化 scheduling，減少狀態管理
- **Trade-off**: 無法針對個別 block 做極致最佳化

### 3. Zone-level Ceiling vs Global Ceiling
- **選擇**: Zone-level（每個 zone 有獨立的 k_max_z）
- **理由**: 高風險 zone 應該更保守
- **Trade-off**: 增加複雜度，但更 data-driven

### 4. Regularization 預設 delta1
- **選擇**: delta1（允許 ±1）而非 nondecreasing
- **理由**: 後期 zone 的 risk 可能略降，k 略增是合理的
- **Trade-off**: 可能在某些 block 出現 k 下降（但幅度 ≤1）

### 5. Alpha/Beta/Gamma 均等權重
- **選擇**: α=β=γ=1/3
- **理由**: 先不偏好任何指標，等 Stage-2 實驗後再調整
- **Trade-off**: 可能不是最優，但是最公平的起點

## 潛在改進方向

### 短期
1. **Multi-T support**: 支援 T=20, T=50（需調整 loader 和 zone 數量）
2. **Per-block weights**: 某些 blocks 可能需要不同的 α/β/γ
3. **Adaptive smoothing**: 根據 drift variance 動態調整 smooth_window

### 中期
1. **Dynamic scheduler**: Runtime 根據前幾步的 drift 微調 k
2. **Quality-aware ceiling**: 加入 FID budget，動態調整 k_max
3. **Zone merging**: 自動合併相似的 adjacent zones

### 長期
1. **Learned scheduler**: 用 RL/GD 學習最優 k[b,z]
2. **Cross-T generalization**: 訓練一個可用於任意 T 的 meta-scheduler
3. **Hardware-aware**: 考慮 cache size/memory bandwidth 等硬體限制

## 技術債務

### 無（目前）
- 程式碼乾淨，文檔完整
- 所有驗證通過
- 無 hard-coded magic numbers（除了預設參數，都可調）

### 待觀察
- Zone 5 只有 1 個 timestep（len=1），雖然合法但可能過於細粒度
- Change point 在 t=96, 97 很接近，可能可以合併

## 與 Stage-0 / Stage-2 的介面

### 輸入（from Stage-0E）
```
QATcode/cache_method/Stage0/stage0e_output/
├── block_names.npy              # (B,)
├── l1_interval_norm.npy         # (B, T-1)
├── cosdist_interval_norm.npy    # (B, T-1)
├── svd_interval_norm.npy        # (B, T-1)
└── fid_w_qdiffae_clip.npy       # (B,)
```

### 輸出（to Stage-2）
```
QATcode/cache_method/Stage1/stage1_output/
├── scheduler_config.json        ← 主要介面
└── scheduler_diagnostics.json   ← 可選（debug 用）
```

### scheduler_config.json 格式
- **version**: "v_final_stage1"
- **T**: 100
- **t_order / analysis_axis_order**: 與 Stage 0 圖橫軸一致（analysis axis 0→99）
- **axis_convention**, **ddim_timestep_formula**: 元資料（例如 `t_ddim = 99 - axis_idx`）
- **zones**: List[{id, **axis_start**, **axis_end**, （可選舊鍵 t_start/t_end 同值）}]
- **blocks**: List[{id, name, k_per_zone: List[int]}]
- **params**: Dict（紀錄所有參數）

### Stage-2 使用方式
1. 讀取 `scheduler_config.json`
2. 對每個 block，建立 **`recompute_mask[axis_idx]`**（索引為 **analysis axis**，長度 T=100）；若 API 要傳 DDIM 張量 timestep，用 **`t_ddim = 99 - axis_idx`**
3. 在 forward pass 時：
   - `if recompute_mask[axis_idx]`: full compute
   - `else`: use cache
4. 記錄 FID / speed / memory

## 執行時間

- **Self-test**: ~0.5s
- **Real data (B=31, T=100)**: ~0.4s
- **Visualization**: ~1.5s
- **Total**: ~2.5s（非常快）

## 依賴

- **Python**: 3.8+
- **NumPy**: 1.20+
- **Matplotlib**: 3.3+（可選，僅用於可視化）
- **標準庫**: json, logging, pathlib, dataclasses, argparse

## 檔案結構

```
QATcode/cache_method/Stage1/
├── __init__.py                    # 模組初始化
├── stage1_scheduler.py            # 主程式（933 行）
├── visualize_stage1.py            # 可視化（242 行）
├── verify_scheduler.py            # 驗證（271 行）
├── README.md                      # 完整文檔（370 行）
├── QUICKSTART.md                  # 快速開始（90 行）
├── IMPLEMENTATION_SUMMARY.md      # 本文件（370 行）
├── stage1_output/                 # 輸出目錄
│   ├── scheduler_config.json
│   └── scheduler_diagnostics.json
└── stage1_figures/                # 圖表
    ├── 1_drift_and_zones.png
    ├── 2_k_heatmap.png
    ├── 3_k_histogram.png
    └── 4_zone_risk.png
```

**總代碼量**: ~2,300 行（含文檔）

## 結論

Stage-1 成功實作了一個完整的、data-driven 的 offline scheduler synthesis pipeline。

**核心成果**：
1. ✅ 從 tri-evidence 合成 zones + k[b,z]
2. ✅ 平均節省 83.4% 計算
3. ✅ 所有驗證通過
4. ✅ 完整文檔與可視化
5. ✅ 可調參數，支援不同策略

**下一步**：
- Stage-2 將實作 runtime cache scheduler
- 驗證 FID degradation 和 speed-up
- 根據實驗結果回來調整 Stage-1 參數（α/β/γ, k_min/k_max, cp_topk）

---

**實作者**: AI Assistant (Claude Sonnet 4.5)  
**日期**: 2026-02-10  
**狀態**: ✅ **完成並驗證**
