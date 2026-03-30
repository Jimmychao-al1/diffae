# SVD Feature Analysis Pipeline

SVD 特徵分析管線，用於研究 Diff-AE / Q-DiffAE 模型中各 UNet block 的 channel 子空間變化，並分析其與 similarity 指標的相關性。

---

## 目錄結構

```
SVD/
├── collect_features_for_svd.py      # 階段 A：Feature 收集
├── svd_metrics.py                   # 階段 B：SVD 指標計算
├── correlate_svd_similarity.py      # 階段 C：相關性分析
├── run_svd_all_blocks_single_pipeline.sh   # Shell：31 個 block 低磁碟 in-memory 批次流程
├── svd_features/                    # 階段 A 輸出
│   └── model_input_blocks_0/
│       ├── t_0.pt ... t_99.pt       # 每個 timestep 的 (N,C,H,W) tensor
│       └── meta.json                # block 資訊（N, T, C, H, W）
├── svd_metrics/                     # 階段 B 輸出
│   └── model_input_blocks_0.json    # SVD 指標（rank_r, subspace_dist, energy_ratio）
├── correlation/                     # 階段 C 輸出
│   ├── model_input_blocks_0.json    # 相關性統計
│   └── figures/                     # 對齊曲線圖、散點圖
└── logs/                            # 執行 log
```

---

## 使用流程

### 階段 A：收集 Feature

**目的**：對每個 block 在 T=100 個 timestep 收集 feature，每個 timestep 累積 N 個樣本的 `(N, C, H, W)` tensor。

**單一 block：**
```bash
python QATcode/cache_method/SVD/collect_features_for_svd.py \
  --num_steps 100 \
  --svd_target_block model.input_blocks.0 \
  --svd_target_N 32 \
  --svd_output_root QATcode/cache_method/SVD \
  --log_file QATcode/cache_method/SVD/logs/feature_model_input_blocks_0.log
```

**批次執行 31 個 block（in-memory，主流程）:**
```bash
cd /home/jimmy/diffae
bash QATcode/cache_method/SVD/run_svd_all_blocks_single_pipeline.sh
```

**輸出**：`svd_features/model_input_blocks_0/t_{0..99}.pt` + `meta.json`

---

### 階段 B：計算 SVD 指標

**目的**：對每個 block 的 feature 做 per-timestep SVD，計算 eigenvalues、子空間距離、energy ratio。

**單一 block：**
```bash
python QATcode/cache_method/SVD/svd_metrics.py \
  --feature_dir QATcode/cache_method/SVD/svd_features/model_input_blocks_0 \
  --output_root QATcode/cache_method/SVD/svd_metrics \
  --representative-t -1 \
  --energy-threshold 0.98
```

**批次處理所有 block：**
```bash
python QATcode/cache_method/SVD/svd_metrics.py \
  --feature_root QATcode/cache_method/SVD/svd_features \
  --output_root QATcode/cache_method/SVD/svd_metrics \
  --representative-t -1 \
  --energy-threshold 0.98 \
  --all
```

**參數說明：**
- `--representative-t`：代表 timestep（-1 表示 T-1，即最後一步），用於定 rank r
- `--energy-threshold`：cumulative energy 門檻（預設 0.98）
- `--compute-energy`：計算 energy ratio 曲線（預設開啟）

**輸出**：`svd_metrics/model_input_blocks_0.json`

---

### 階段 C：相關性分析

**目的**：分析 SVD 子空間距離與 similarity（L1、cosine distance）在 interval-wise 維度上的相關性。

**單一 block：**
```bash
python QATcode/cache_method/SVD/correlate_svd_similarity.py \
  --svd_metrics QATcode/cache_method/SVD/svd_metrics/model_input_blocks_0.json \
  --similarity_npz QATcode/cache_method/a_L1_L2_cosine/T_100/v2_latest/result_npz/model_input_blocks_0.npz \
  --output_root QATcode/cache_method/SVD/correlation \
  --plot
```

**批次處理所有 block：**
```bash
python QATcode/cache_method/SVD/correlate_svd_similarity.py \
  --svd_metrics_dir QATcode/cache_method/SVD/svd_metrics \
  --similarity_npz_dir QATcode/cache_method/a_L1_L2_cosine/T_100/v2_latest/result_npz \
  --output_root QATcode/cache_method/SVD/correlation \
  --plot \
  --all
```

**輸出**：
- `correlation/model_input_blocks_0.json`（相關性統計）
- `correlation/figures/model_input_blocks_0_alignment.png`（對齊曲線圖）
- `correlation/figures/model_input_blocks_0_scatter.png`（散點圖）

---

## 關鍵設計

### 1. Feature 收集（階段 A）

- **與 similarity 的差異**：
  - Similarity：累積後計算 L1/L1_rel/cosine，寫 NPZ/CSV/PNG
  - SVD：只累積 raw feature，寫 `t_{t}.pt` + `meta.json`
- **GPU 最佳化**：
  - Hook 內立即做 `detach().cpu()`，避免 GPU OOM
  - buffer 與後續 concat 以 CPU tensor 為主
- **target_N**：每個 t 最終寫出 `(target_N, C, H, W)`，所有 timestep 共用同一組 N 個 sample

### 2. SVD 指標（階段 B）

- **Second moment（uncentered）**：對 `(N, C, H, W)` reshape 成 `(C, M)`（M=N×H×W），計算 `Σ = (X @ X^T) / M`
- **Rank r**：用代表 timestep（預設 T-1）的 eigenvalues 做 cumulative energy，找最小 r 使得 `sum(λ[:r]) / sum(λ) >= 0.98`
- **子空間距離**：`d(t, t-1) = 1 - (||U_t^{(r)T} U_{t-1}^{(r)}||_F^2 / r)`

### 3. 相關性分析（階段 C）

- **Similarity 來源**：直接讀 **NPZ**（`l1_step_mean`, `cos_step_mean`），不需要額外 JSON
- **序列對齊**：以 interval-wise 對齊（SVD 使用 `subspace_dist[1:]`；similarity 使用 step mean）
- **相關性**：Pearson / Spearman（L1 vs SVD、CosDist vs SVD）
- **視覺化**：對齊曲線圖（dual y-axis）+ 散點圖（含相關係數標註）

---

## 注意事項

1. **Block 名稱一致性**：
   - Feature 收集時用 `model.input_blocks.0`
   - 寫檔時 slug 為 `model_input_blocks_0`（`.` → `_`）
   - Similarity NPZ 檔名也是 `model_input_blocks_0.npz`
   - 階段 C 依此對應

2. **依賴**：
   - 階段 A：依賴 `similarity_calculation.py` 的工具函數（`load_diffae_model` 等）
   - 階段 B：只依賴 A 的輸出
   - 階段 C：依賴 B 的輸出 + similarity 的 NPZ

3. **T=100**：所有實驗固定 T=100

4. **計算資源**：
   - 階段 A 需要 GPU（生成圖片與 feature 收集）
   - 階段 B、C 可在 CPU 上執行（讀檔、SVD、相關性計算）

---

## 快速開始

```bash
# 1. 以 in-memory 主流程依序處理所有 block（A->B->C）
bash QATcode/cache_method/SVD/run_svd_all_blocks_single_pipeline.sh

# 2. 計算所有 block 的 SVD 指標
python QATcode/cache_method/SVD/svd_metrics.py --all

# 3. 分析所有 block 的相關性（L1 + cosine distance）
python QATcode/cache_method/SVD/correlate_svd_similarity.py --all --plot

# 4. 查看結果
ls QATcode/cache_method/SVD/correlation/*.json
ls QATcode/cache_method/SVD/correlation/figures/*.png
```

### 低磁碟模式（單一 block，完全不寫 .pt）

```bash
# A 收集後直接在記憶體跑 B/C，不落地 svd_features/t_{t}.pt
bash QATcode/cache_method/SVD/run_single_block_pipeline.sh "model.output_blocks.11" 16
```

### 低磁碟模式（全部 block 依序跑）

```bash
# 預設：N=32，in-memory（不寫 .pt）
bash QATcode/cache_method/SVD/run_svd_all_blocks_single_pipeline.sh

# 指定 N
bash QATcode/cache_method/SVD/run_svd_all_blocks_single_pipeline.sh 32

# 中斷後從某個 block 繼續
bash QATcode/cache_method/SVD/run_svd_all_blocks_single_pipeline.sh 32 model.output_blocks.7
```

---

## 進階使用

### 調整 target_N

直接在主流程指定 target_N：

```bash
bash QATcode/cache_method/SVD/run_svd_all_blocks_single_pipeline.sh 128
```

### 調整 energy threshold

```bash
python QATcode/cache_method/SVD/svd_metrics.py --all --energy-threshold 0.95
```

### 使用不同代表 timestep

```bash
python QATcode/cache_method/SVD/svd_metrics.py --all --representative-t 50  # 使用 t=50
```

---

## 輸出格式

### meta.json（階段 A）
```json
{
  "block": "model_input_blocks_0",
  "target_block_name": "model.input_blocks.0",
  "N": 32,
  "T": 100,
  "C": 256,
  "H": 32,
  "W": 32
}
```

### SVD 指標 JSON（階段 B）
```json
{
  "block": "model_input_blocks_0",
  "T": 100,
  "C": 256,
  "rank_r": 45,
  "representative_t": 99,
  "energy_threshold": 0.98,
  "actual_energy_at_r": 0.9802,
  "timesteps": [0, 1, ..., 99],
  "subspace_dist": [0.0, 0.123, ...],
  "energy_ratio": {
    "k4": [0.45, 0.46, ...],
    "k8": [0.62, 0.63, ...],
    ...
  }
}
```

### 相關性 JSON（階段 C）
```json
{
  "block": "model_input_blocks_0",
  "T_svd": 100,
  "interval_length_used": 99,
  "rank_r": 45,
  "x_axis_def": "interval-wise t_curr (left=noise side, right=clear side)",
  "correlation": {
    "L1_vs_SVD": {
      "pearson": 0.856,
      "spearman": 0.823,
      "pearson_pvalue": 1.23e-25,
      "spearman_pvalue": 3.45e-22
    },
    "CosDist_vs_SVD": {...}
  }
}
```
