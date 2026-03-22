# Stage-0E: Loader + Normalization

此模組用於將 Stage-0 的原始實驗數據（T=100）轉換為正規化的 interval-wise 指標，供 Stage-1 cache scheduler 使用。

## 功能

1. **載入並對齊三種 interval-wise 指標**：
   - L1rel_rate：`||F[t] - F[t-1]||_1 / ||F[t-1]||_1`
   - Cosine distance：`1 - cosine_similarity(F[t], F[t-1])`
   - SVD subspace drift：子空間距離，基於 leading eigenvectors

2. **Min-max 正規化**：
   - 三種指標獨立正規化到 [0, 1]
   - 處理 NaN/Inf
   - 過濾極端值

3. **FID-based block weights**：
   - 從 Q-DiffAE T=100 的 Delta-FID 計算每個 block 的敏感度權重
   - Noise 修正 + Quantile clipping + Min-max 正規化
   - 同時產生 rank-based weights（用於 ablation）

## 資料來源

```
L1/Cosine: QATcode/cache_method/L1_L2_cosine/T_100/Res/result_npz/*.npz
SVD:       QATcode/cache_method/SVD/svd_metrics/*.json
FID:       QATcode/cache_method/FID/fid_cache_sensitivity/fid_sensitivity_results.json
```

## 使用方式

### 方法 1：直接執行（使用預設路徑）

```bash
cd /home/jimmy/diffae
python3 QATcode/cache_method/Stage0/stage0e_normalization.py
```

### 方法 2：以模組方式匯入並自訂路徑

```python
from QATcode.cache_method.Stage0.stage0e_normalization import run_stage0e

run_stage0e(
    l1_cos_dir="path/to/L1_L2_cosine/T_100/Res/result_npz",
    svd_dir="path/to/SVD/svd_metrics",
    fid_json_path="path/to/fid_sensitivity_results.json",
    output_dir="path/to/output",
    eps_noise=0.5,    # FID noise 修正閾值
    quantile=0.95     # FID quantile clipping
)
```

## 輸出檔案

所有輸出儲存為 `.npy` 格式，預設位於 `QATcode/cache_method/Stage0/stage0e_output/`：

| 檔案名稱 | Shape | 描述 |
|---------|-------|------|
| `block_names.npy` | (31,) | Block 名稱列表（object array） |
| `l1_interval_norm.npy` | (31, 99) | 正規化的 L1rel_rate，**欄 j** = analysis 上 **interval j**（見下方） |
| `cosdist_interval_norm.npy` | (31, 99) | 正規化的 cosine distance |
| `svd_interval_norm.npy` | (31, 99) | 正規化的 SVD 子空間距離 |
| `fid_w_qdiffae_clip.npy` | (31,) | FID weights (quantile-clipped) |
| `fid_w_qdiffae_rank.npy` | (31,) | FID weights (rank-based，用於 ablation) |

## Interval Mapping 規則

**Interval index** `j ∈ [0, T-2]`：與 L1_L2_cosine `.npz` 的 `l1_rate_step_mean[j]` 同欄位語意。

- 表 **analysis axis** 上 **點 j 與 j+1 之間**的變化（圖橫軸由左到右 j 遞增）。
- 與 **DDIM** 進模型的 timestep：**該區間對應 t_ddim 由 (99−j) 變到 (98−j)**（T=100）。**不要**把 j 直接當成「DDIM 張量裡的 t=j」。

範例 `j=50`：
- `l1_interval_norm[b, 50]` 等三檔同欄
- DDIM 上對應的是 **t_ddim：49→48** 這一步的特徵差（不是 t=50→51）

### 原始數據的索引對應

| 指標 | 原始格式 | Interval i 的來源 |
|------|---------|------------------|
| L1 | `l1_rate_step_mean[i]` | 直接使用（前向：i→i+1） |
| Cosine | `1.0 - cos_step_mean[i]` | 轉換後使用（前向：i→i+1） |
| SVD | `subspace_dist[i+1]` | 因為 `subspace_dist[t]` 測量 (t-1)→t，所以 interval i 用 `subspace_dist[i+1]` |

## 數值檢查

程式會自動檢查：
- ✅ 無 NaN/Inf
- ✅ 數值範圍 [0, 1]
- ✅ 統計資訊（min/max/mean/std）

## 注意事項

1. **Block 名稱格式**：
   - Metric 檔案使用：`model_input_blocks_0`
   - 內部轉換為：`model.input_blocks.0`
   - FID 使用：`encoder_layer_0`（自動 mapping）

2. **FID 數據要求**：
   - 必須包含 `"T100"` 或 `"100steps"` 的實驗結果
   - 需要 k=3, 4, 5 的數據
   - 如果某些 block 缺少 FID 數據，對應權重設為 0

3. **Cosine distance 修正**：
   - 原始 `cos_step_mean` 是 **similarity**
   - 自動轉換為 **distance**：`1.0 - cos_step_mean`

## 驗證輸出

```python
import numpy as np

# 載入數據
names = np.load('stage0e_output/block_names.npy', allow_pickle=True)
l1 = np.load('stage0e_output/l1_interval_norm.npy')
cos = np.load('stage0e_output/cosdist_interval_norm.npy')
svd = np.load('stage0e_output/svd_interval_norm.npy')
w_clip = np.load('stage0e_output/fid_w_qdiffae_clip.npy')

# 檢查
print(f"Blocks: {len(names)}")
print(f"L1 shape: {l1.shape}, range: [{l1.min():.4f}, {l1.max():.4f}]")
print(f"FID weights 非零數: {np.sum(w_clip > 0)}/{len(names)}")

# 查看特定 block 在特定 interval 的指標
block_idx = 0  # model.input_blocks.0
interval_idx = 50  # analysis interval j=50（DDIM: t_ddim 49→48）
print(f"\nBlock {names[block_idx]}, interval {interval_idx}:")
print(f"  L1: {l1[block_idx, interval_idx]:.4f}")
print(f"  Cosine: {cos[block_idx, interval_idx]:.4f}")
print(f"  SVD: {svd[block_idx, interval_idx]:.4f}")
print(f"  FID weight: {w_clip[block_idx]:.4f}")
```

## 下一步：Stage-1

這些正規化後的指標可直接用於 Stage-1 的 cache scheduler，透過加權融合產生最終的 cache decision signal。
