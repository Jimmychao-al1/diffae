# Stage-1 快速開始

## 一鍵執行

```bash
cd /home/jimmy/diffae

# 1. 執行 Stage-1（生成 scheduler config）
python3 QATcode/cache_method/Stage1/stage1_scheduler.py

# 2. 驗證結果
python3 QATcode/cache_method/Stage1/verify_scheduler.py

# 3. 生成可視化
python3 QATcode/cache_method/Stage1/visualize_stage1.py
```

## 輸出位置

```
QATcode/cache_method/Stage1/
├── stage1_output/
│   ├── scheduler_config.json       ← 主要輸出（供 Stage-2 使用）
│   └── scheduler_diagnostics.json  ← 診斷資料
└── stage1_figures/
    ├── 1_drift_and_zones.png       ← Drift 曲線 + zone 切分
    ├── 2_k_heatmap.png             ← K 分佈熱圖
    ├── 3_k_histogram.png           ← K 直方圖
    └── 4_zone_risk.png             ← Zone risk 分析
```

## 關鍵結果（T=100）

- **Zones**: 7 個
  - Zone 0: t=0..62（大區塊，穩定期）
  - Zone 1-6: t=63..99（小區塊，不穩定期）

- **K 統計**:
  - 範圍: [3, 8]
  - 平均: 7.16
  - 中位數: 7

- **Cache 節省**: 平均 **83.4%**
  - 只需 recompute 16.6% 的 timesteps
  - 每個 block 平均只需重算 16-22 次（out of 100）

## 調整策略

### 更激進的 cache（節省更多計算）

```bash
python3 QATcode/cache_method/Stage1/stage1_scheduler.py \
  --k_min 2 --k_max 10 \
  --cp_topk 4  # 較少 zones，更大的 k
```

### 更保守的策略（品質優先）

```bash
python3 QATcode/cache_method/Stage1/stage1_scheduler.py \
  --k_min 1 --k_max 5 \
  --cp_topk 10  # 較多 zones，更小的 k
```

### 強調 FID sensitivity

```bash
python3 QATcode/cache_method/Stage1/stage1_scheduler.py \
  --alpha 0.2 --beta 0.2 --gamma 0.6  # 60% 權重給 FID
```

## 下一步：Stage-2

Stage-2 將使用 `scheduler_config.json` 來實作 runtime cache scheduler，並在實際 inference 中驗證：
1. FID degradation
2. Inference speed-up
3. Memory usage

---

**實作完成時間**: 2026-02-10  
**執行環境**: T=100 DDIM, FFHQ-128, Q-DiffAE
