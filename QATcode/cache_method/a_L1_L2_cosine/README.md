# a_L1_L2_cosine：L1 / L2 / Cosine Similarity 計算管線

此模組針對 Diff-AE / Q-DiffAE 推理過程，對 UNet 各 block 的輸出特徵，在相鄰 DDIM timestep 之間計算 L1、L1-relative、Cosine Similarity，並輸出供後續 Stage 0 使用的 NPZ/CSV/PNG 結果。

---

## 目錄結構

```
a_L1_L2_cosine/
├── similarity_calculation.py           # 主程式（v2，使用 quantize_ver2 模組）
├── similarity_calculation_baseline.py  # 舊版（v1，僅供對照參考）
├── run_similarity_experiments.sh       # 批次對 31 個 block 依序執行 v2 版本
├── retry_similarity_until_ok.sh        # 自動重試包裝腳本
├── logs/                               # 執行 log（logs/v2/ 子目錄）
└── T_100/
    └── v2_latest/
        ├── result_npz/                 # 每個 block 的 NPZ（Stage 0 正式來源）
        ├── L1/                         # L1 曲線圖與 CSV
        └── cosine/                     # Cosine 熱圖
```

---

## 實作架構

### 核心類：`SimilarityCollector`

透過 forward hook 掛載到指定 UNet block，逐 timestep 收集輸出，並依以下三個指標計算：

| 指標 | 計算方式 |
|------|---------|
| **L1 (relative)** | `mean(|t1-t2|) / mean(|t1|, |t2|)` 對稱版本，逐樣本計算後平均 |
| **L1 rate** | `sum(|t1-t2|) / sum(|t1|)` 非對稱版本（以 t1 為參考） |
| **Cosine Similarity** | 特徵展平後取 dot product / norm 乘積，矩陣運算版本 |

`_calc_metrics_batch()` 對每一對 `(t_prev, t_curr)` 同時回傳 l1、l1_rate、l2、cos 四個值；主流程僅累加 **L1 / L1_rel / Cosine**（不輸出 L2）。

### 兩種輸出維度

- **點對點矩陣（point-wise）**：`T×T` 全對矩陣，記錄所有 timestep pair 的 L1 / cosine，輸出為 `l1rel.csv` 與 `cosine_heatmap.png`。
- **interval-wise step 曲線**：相鄰步 `t-1→t` 的均值，長度 `T-1`，輸出為 NPZ 中的 `l1_step_mean` / `cos_step_mean`（Stage 0 正式讀取欄位）。

### 時間軸約定

- analysis index `i = 0..T-1`（noise→clear 收集順序）
- display t_curr = `(T-1) - i`
- interval index `j = 0..T-2`，display t_curr = `(T-2) - j`（寫入 NPZ 的 `t_curr_interval` 欄位，Stage 0 校驗用）

---

## Quick Start

### 對單一 block 執行一次

```bash
cd /home/jimmy/diffae

python3 QATcode/cache_method/a_L1_L2_cosine/similarity_calculation.py \
  --mode float \
  --num_steps 100 \
  --enable_similarity \
  --similarity_samples 128 \
  --similarity_collect_samples 20 \
  --similarity_sample_strategy random \
  --similarity_target_block model.input_blocks.0 \
  --similarity_output_root QATcode/cache_method/a_L1_L2_cosine \
  --similarity_dtype float16
```

### 批次對所有 31 個 block

```bash
bash QATcode/cache_method/a_L1_L2_cosine/run_similarity_experiments.sh
```

腳本會依序對 `model.input_blocks.{0..14}`、`model.middle_block`、`model.output_blocks.{0..14}` 各執行一次，log 寫入 `logs/v2/`。

### 失敗後自動重試

```bash
bash QATcode/cache_method/a_L1_L2_cosine/retry_similarity_until_ok.sh
```

---

## CLI 參數說明

| 參數 | 預設 | 說明 |
|------|------|------|
| `--mode` | `float` | 模型模式（`float` / `int`） |
| `--num_steps` | `100` | DDIM 步數 T |
| `--enable_similarity` | — | 啟用 Similarity Analysis（必須加此旗標） |
| `--similarity_samples` | `64` | 生成圖片總數（驅動 evaluate_fid 迴圈） |
| `--similarity_collect_samples` | `15` | 實際用於計算的樣本數（每 batch 上限） |
| `--similarity_sample_strategy` | `first` | 採樣策略：`first` / `random` / `uniform` |
| `--similarity_target_block` | `None` | 指定單一 block；不加則走批次模式 |
| `--similarity_output_root` | `QATcode/cache_method/a_L1_L2_cosine` | 輸出根目錄 |
| `--similarity_dtype` | `float16` | NPZ 儲存精度（`float16` / `float32`） |

---

## 輸出格式

### NPZ（`result_npz/<block_slug>.npz`，Stage 0 正式來源）

| 欄位 | shape | 說明 |
|------|-------|------|
| `l1_step_mean` | `(T-1,)` | interval-wise L1 均值（Stage 0 正式指標） |
| `cos_step_mean` | `(T-1,)` | interval-wise cosine similarity 均值 |
| `l1_rate_step_mean` | `(T-1,)` | interval-wise L1 rate 均值（上游殘留欄位） |
| `t_curr_interval` | `(T-1,)` | 顯示用 t_curr 軸（Stage 0 格式校驗用） |
| `l1rel` | `(T, T)` | 點對點 L1 rel 矩陣 |
| `cosine` | `(T, T)` | 點對點 cosine 矩陣 |

---

## 依賴

- 量化模組：`QATcode/quantize_ver2/quant_model_lora_v2.py`、`quant_layer_v2.py` 等
- Checkpoint：`QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best.pth`
- Base Diff-AE：`checkpoints/ffhq128_autoenc_latent/last.ckpt`
