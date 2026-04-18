# b_SVD：SVD Feature Analysis Pipeline

SVD 特徵分析管線，用於研究 Diff-AE / Q-DiffAE 模型中各 UNet block 的 channel 子空間變化，並分析其與 similarity 指標（L1、Cosine distance）的相關性。分析結果作為 tri-evidence 架構中 SVD evidence 的產出依據。

---

## 目錄結構

```
b_SVD/
├── collect_features_for_svd.py             # 階段 A：Feature 收集
├── svd_metrics.py                          # 階段 B：SVD 指標計算
├── correlate_svd_similarity.py             # 階段 C：相關性分析
├── run_svd_all_blocks_single_pipeline.sh   # 批次：31 個 block，in-memory 低磁碟流程
├── run_single_block_pipeline.sh            # 單一 block A→B→C in-memory 流程
├── svd_features/                           # 階段 A 輸出（可略過，in-memory 模式不寫磁碟）
│   └── model_input_blocks_0/
│       ├── t_0.pt ... t_99.pt              # 每個 timestep 的 (N,C,H,W) tensor
│       └── meta.json                       # block 資訊（N, T, C, H, W）
├── svd_metrics/                            # 階段 B 輸出
│   └── model_input_blocks_0.json           # SVD 指標（rank_r, subspace_dist, energy_ratio）
├── correlation/                            # 階段 C 輸出
│   ├── model_input_blocks_0.json           # 相關性統計
│   └── figures/                            # 對齊曲線圖、散點圖
└── logs/                                   # 執行 log
```

---

## 實作架構

### 階段 A：Feature 收集（`collect_features_for_svd.py`）

- 複用 `similarity_calculation.py` 的量化模型載入與 `evaluate_fid` 驅動流程
- `SvdFeatureCollector` 類：對指定 block 註冊 forward hook，逐 timestep 累積 `(N, C, H, W)` 特徵
- **GPU 最佳化**：hook 內立即執行 `output.detach().cpu()`，避免 GPU OOM；所有 buffer 以 CPU tensor 儲存
- 每個 timestep 最多保留 `target_N` 個樣本，所有 timestep 共用同一組 sample
- 輸出：`svd_features/<block_slug>/t_{0..99}.pt` + `meta.json`

**CLI 參數：**

| 參數 | 預設 | 說明 |
|------|------|------|
| `--svd_target_block` | 必填 | 目標 block（如 `model.input_blocks.0`） |
| `--svd_target_N` | `32` | 每個 timestep 的樣本數 |
| `--num_steps` | `100` | T（固定 100） |
| `--svd_output_root` | `QATcode/cache_method/b_SVD` | 輸出根目錄 |
| `--in_memory_pipeline` | — | 不寫 `.pt`，直接串接階段 B/C |
| `--similarity_npz` | — | in-memory 模式下指定 NPZ 路徑，串接階段 C |

---

### 階段 B：SVD 指標計算（`svd_metrics.py`）

- 對每個 timestep 的 feature `X`（shape `(N,C,H,W)`）計算 **channel 二階矩陣**：  
  `Σ = (X_reshaped @ X_reshaped^T) / M`（M = N×H×W，未中心化）
- `torch.linalg.eigh` 分解得 eigenvalues / eigenvectors，按遞減排序
- **Rank r** 定義：用代表 timestep（預設 T-1）的累積能量門檻（預設 0.98）確定最小 r
- **子空間距離**：  
  `d(t, t-1) = 1 - (||U_t^{(r)T} U_{t-1}^{(r)}||_F^2 / r)`
- 可選輸出 energy ratio 曲線（k ∈ {4, 8, 16, 32, 64}）

**CLI 參數：**

| 參數 | 預設 | 說明 |
|------|------|------|
| `--feature_dir` | — | 單一 block 的 feature 目錄（階段 A 輸出） |
| `--feature_root` + `--all` | — | 批次處理所有 block |
| `--representative-t` | `-1`（即 T-1） | 代表 timestep，用於定 rank r |
| `--energy-threshold` | `0.98` | 累積能量門檻 |
| `--output_root` | `QATcode/cache_method/b_SVD/svd_metrics` | JSON 輸出目錄 |

---

### 階段 C：相關性分析（`correlate_svd_similarity.py`）

- 讀取階段 B 的 SVD JSON 與 `a_L1_L2_cosine` 的 similarity NPZ（`l1_step_mean`, `cos_step_mean`）
- **Cosine distance** = `1 - cos_step_mean`（由 NPZ 讀取後轉換，不重新計算）
- **序列對齊**：SVD 使用 `subspace_dist[1:]`；similarity 使用 interval-wise step mean；取最短長度 L 對齊
- 計算 **Pearson / Spearman 相關性**（L1 vs SVD、CosDist vs SVD，含 p-value）
- 輸出對齊曲線圖（dual y-axis）與散點圖

---

## Quick Start

### 低磁碟模式：單一 block 完整流程（A→B→C，不寫 `.pt`）

```bash
cd /home/jimmy/diffae
bash QATcode/cache_method/b_SVD/run_single_block_pipeline.sh "model.input_blocks.0" 32
```

### 低磁碟模式：所有 31 個 block 依序執行

```bash
# 預設 N=32
bash QATcode/cache_method/b_SVD/run_svd_all_blocks_single_pipeline.sh

# 指定 N
bash QATcode/cache_method/b_SVD/run_svd_all_blocks_single_pipeline.sh 64

# 從指定 block 繼續（中斷後接續）
bash QATcode/cache_method/b_SVD/run_svd_all_blocks_single_pipeline.sh 32 model.output_blocks.7
```

### 單獨跑階段 B（已有 `.pt`）

```bash
# 單一 block
python QATcode/cache_method/b_SVD/svd_metrics.py \
  --feature_dir QATcode/cache_method/b_SVD/svd_features/model_input_blocks_0 \
  --output_root QATcode/cache_method/b_SVD/svd_metrics \
  --representative-t -1 \
  --energy-threshold 0.98

# 所有 block
python QATcode/cache_method/b_SVD/svd_metrics.py \
  --feature_root QATcode/cache_method/b_SVD/svd_features \
  --output_root QATcode/cache_method/b_SVD/svd_metrics \
  --representative-t -1 \
  --energy-threshold 0.98 \
  --all
```

### 單獨跑階段 C

```bash
# 單一 block
python QATcode/cache_method/b_SVD/correlate_svd_similarity.py \
  --svd_metrics QATcode/cache_method/b_SVD/svd_metrics/model_input_blocks_0.json \
  --similarity_npz QATcode/cache_method/a_L1_L2_cosine/T_100/v2_latest/result_npz/model_input_blocks_0.npz \
  --output_root QATcode/cache_method/b_SVD/correlation \
  --plot

# 所有 block
python QATcode/cache_method/b_SVD/correlate_svd_similarity.py \
  --svd_metrics_dir QATcode/cache_method/b_SVD/svd_metrics \
  --similarity_npz_dir QATcode/cache_method/a_L1_L2_cosine/T_100/v2_latest/result_npz \
  --output_root QATcode/cache_method/b_SVD/correlation \
  --plot --all
```

---

## 輸出格式

### meta.json（階段 A）

```json
{
  "block": "model_input_blocks_0",
  "target_block_name": "model.input_blocks.0",
  "N": 32, "T": 100, "C": 256, "H": 32, "W": 32
}
```

### SVD 指標 JSON（階段 B）

```json
{
  "block": "model_input_blocks_0",
  "T": 100, "C": 256, "rank_r": 45,
  "representative_t": 99,
  "energy_threshold": 0.98,
  "actual_energy_at_r": 0.9802,
  "timesteps": [0, 1, "...", 99],
  "subspace_dist": [0.0, 0.123, "..."],
  "energy_ratio": {"k4": [...], "k8": [...]}
}
```

### 相關性 JSON（階段 C）

```json
{
  "block": "model_input_blocks_0",
  "T_svd": 100, "interval_length_used": 99, "rank_r": 45,
  "x_axis_def": "interval-wise t_curr (left=noise side, right=clear side)",
  "correlation": {
    "L1_vs_SVD": {"pearson": 0.856, "spearman": 0.823, ...},
    "CosDist_vs_SVD": {...}
  }
}
```

---

## 注意事項

1. **Block 名稱一致性**：CLI 輸入用 `model.input_blocks.0`；目錄/檔名 slug 用 `model_input_blocks_0`（`.` → `_`）；similarity NPZ 檔名亦同。
2. **依賴**：階段 A 需要 GPU（驅動 diffusion 生成）；階段 B、C 可在 CPU 上執行。
3. **Similarity NPZ 來源**：正式路徑為 `QATcode/cache_method/a_L1_L2_cosine/T_100/v2_latest/result_npz/`；需先執行 a_L1_L2_cosine 管線。
4. **T=100**：所有實驗固定 T=100。
