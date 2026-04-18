# Stage 1：Offline Cache Scheduler（baseline v1）

本目錄實作 **Stage 1 baseline v1**：在 diffusion 推理時間軸上，從 Stage 0 的 interval-wise 證據合成 **全域共用的時間分段（shared zones）**，再對 **每個 block、每個 zone** 以代價函數選出 reuse 週期 \(k\)，最後展開成逐步的 **full-compute / reuse（F/R）** 排程。  
Stage 1 **不**做 inference 端最終 refinement；該步留給 Stage 2（cache-run 微調）。

方法敘述、符號與 baseline 選取討論見同目錄 `SUMMARY.md`。

---

## 1. 在整條 pipeline 裡的角色

| 階段 | 角色 |
|------|------|
| **Stage 0** | 產出正規化後的 L1 / Cos / SVD interval 證據、FID 權重、時間軸 metadata（`.npy`）。 |
| **Stage 1（本目錄）** | 讀入上述輸出 → 建 **shared_zones** → 每 \((b,z)\) 選 **\(k\)** → 輸出 **`expanded_mask [B,T]`** 與 JSON 設定。 |
| **Stage 2（未於此實作）** | 以 Stage 1 為起點，依實際 cache 行為 refinement。 |

設計上 **所有 block 共用同一套 zones**（global temporal skeleton），**不**為每個 block 各切一套 zone；block 之間只差 **每 zone 內的 \(k_{b,z}\)**。

---

## 2. 時間軸與 mask 約定（與程式一致）

- **DDIM 採樣順序**：\(99 \to 0\)（\(t=99\) 為第一步，\(t=0\) 為最後一步；對 **T=100**）。
- **`expanded_mask[b, i]`** 的索引 **`i` 為步序**（processing order）：
  - **`i = 0`** → DDIM **`t = T-1`**（例如 99）
  - **`i = T-1`** → **`t = 0`**
  - 對應式：**`t = (T-1) - i`**（程式內 `ddim_t_to_step_index` / `step_index_to_ddim_t`）。
- **`True`** = full compute（**F**），**`False`** = reuse（**R**）。合成結束後強制 **`expanded[:, 0] = True`**（對應 **\(t=T-1\)** 必為 F）。

### Interval → reused timestep

Stage 0 陣列為 **interval 欄** \(j=0,\ldots,T-2\)。Stage 1 將 **interval \((t{+}1 \!\to\! t)\)** 的證據對應到 **reused timestep `t`**：

- **`t = (T-2) - j`**（`interval_j_to_reused_ddim_t`）
- **`t = T-1`** 無 interval 欄：在 **`I_cut`** 裡置 **0**，排程上仍強制該步 F。

載入時會校驗 **`t_curr_interval.npy`** 必須與 **`np.arange(T-2, -1, -1)`** 完全一致，否則 **`ValueError`**（避免與 Stage 0 約定漂移）。**`axis_interval_def`** 若為空僅 **warning**，不中斷。

---

## 3. 演算法流程（對應 `stage1_scheduler.py`）

以下為實際呼叫順序的濃縮版；係數以程式常數為準（`W_L1_BRANCH=0.7`、`ALPHA_ICUT=4/9` 等）。

1. **`load_stage0_formal`**  
   讀取七個正式檔（見 §5），shape / `t_curr` 檢查、數值 clip 至 \([0,1]\)（必要時 warning）。

2. **`build_I_l1cos_I_cut_per_ddim_t`**  
   - \(I_{\mathrm{l1cos}}[b,t] = 0.7\,\mathrm{L1\_norm} + 0.3\,\mathrm{Cos\_norm}\)（**變化量**分支，**非** similarity score；診斷鍵 **`I_l1cos_stats`**）。  
   - \(I_{\mathrm{cut}}[b,t] = \frac{4}{9} I_{\mathrm{l1cos}} + \frac{5}{9} \mathrm{SVD\_norm}\)。

3. **`global_cutting_signal_G`**  
   \(G[t] = \sum_b \tilde w_b\, I_{\mathrm{cut}}[b,t]\)，\(\tilde w\) 為 **FID 權重正規化**；若 **`fid_w` 全接近 0** → **warning** + **均匀 \(1/B\)**（避免 G 不定）。

4. **`processing_order_series` → `moving_average` → `delta_adjacent`**  
   將 \(G\) 依步序 \(i=0\ldots T-1\) 排列 → **smooth_window** 移動平均 → 鄰步差分得到 **\(\Delta\)**。

5. **`topk_change_point_indices`**（候選邊界 **\(i=1\ldots T-1\)**）→ **`zones_from_step_boundaries`** → **`merge_short_zones_step`**（**min_zone_len**，短 zone **merge right**，最後一段 **merge left**）。

6. **`step_zone_to_ddim_zone`** 寫入 **`shared_zones`**（`t_start`, `t_end`, `length`，且 **\(t_{\mathrm{start}} \ge t_{\mathrm{end}}\)**）。

7. **`validate_shared_zones_ddim`（fail-fast）**  
   無 overlap、完整覆蓋 \(t\in[0,T-1]\)、長度和為 **T**。

8. 每個 zone：**`unique_k_representatives`** 在 \([k_{\min},k_{\max}]\) 內對 **F/R pattern 去重**；對每個 \((b,z)\)、每個候選 **\(k\)** 算 **`cost_J_for_k`**（\(J\)：**reuse 項乘 `fid_w[b]`**，**\(\lambda |F|/L_z\)** 不加 FID），取 **argmin**。

9. **`expand_zone_mask_ddim`** 合併成 **`expanded_mask`**；**`or_expanded_with_zone_mask`** 在 OR 前檢查 **步序是否重疊**（異常則 **raise**）。

10. 寫入三個 JSON（§6）；可選在 diagnostics 中記錄 **lambda sweep** 下若重選的 **k** 對照。

---

## 4. 目標函數 \(J(b,z,k)\)（與程式一致）

對 zone 長度 \(L_z\)、reuse 集 \(\mathcal{R}\)、full-compute 集 \(\mathcal{F}\)（由 zone 內 \(k\) 展開）：

\[
J = w_b \cdot \frac{1}{L_z}\sum_{t\in\mathcal{R}} I_{\mathrm{cut}}[b,t]
  + \lambda \cdot \frac{|\mathcal{F}|}{L_z}.
\]

**\(k=1\)** 時若無第二項，reuse 風險項常為 0，易過度選 **全 F**；**\(\lambda |F|/L_z\)** 提供 **full-compute 比例懲罰**（不加 FID 權重）。

---

## 5. 輸入（Stage 0 目錄）

以下檔案須置於 **`--stage0_dir`**：

| 檔案 | 用途 |
|------|------|
| `block_names.npy` | Block 名稱，\((B,)\) |
| `l1_interval_norm.npy` | L1 interval 正規化變化量，\((B, T-1)\) |
| `cosdist_interval_norm.npy` | Cos 距離，\((B, T-1)\) |
| `svd_interval_norm.npy` | SVD interval，\((B, T-1)\) |
| `fid_w_qdiffae_clip.npy` | FID 權重，\((B,)\) |
| `axis_interval_def.npy` | 文字說明（空則 warning） |
| `t_curr_interval.npy` | 長度 \(T-1\)，且須等於 **`arange(T-2,-1,-1)`** |

預設 **T=100**、**\(T-1=99\)** interval 欄；若 shape 不符會 **raise**。

---

## 6. 輸出（`--output_dir`）

| 檔案 | 內容摘要 |
|------|----------|
| **`scheduler_config.json`** | `version`（如 `stage1_baseline_v1`）、`T`、`time_order: "ddim_99_to_0"`、`stage1_baseline_params`、`shared_zones`（`id`, `t_start`, `t_end`, `length`）、`blocks`（`id`, `name`, `k_per_zone`, `expanded_mask`） |
| **`scheduler_diagnostics.json`** | `I_l1cos_stats` / `I_cut_stats`、`G_ddim`、\(G_{\mathrm{proc}}\)、\(G_{\mathrm{smooth}}\)、\(\Delta\)、change points、step zones 合併前後、每 zone **候選 k**、**cost_tables_per_zone**、baseline **\(\lambda\)**、**lambda_sweep** 對照等 |
| **`verification_summary.json`** | zone 邊界、每 block **k**、**#F/#R**、**total cost**、每 zone 候選 k 與選後 **J** 摘要 |

`stage1_baseline_params` 內含 **`K_effective_used`**（與 CLI 的 `--K` 取 **min(K, T-1)** 等一致）、**`W_L1_branch` / `W_COS_branch`**（舊名 `W_L1_sim` 已廢除，若下游仍解析舊鍵需更新）。

---

## 7. 命令列與工具

### 7.1 主程式 `stage1_scheduler.py`

```bash
python3 QATcode/cache_method/Stage1/stage1_scheduler.py \
  --stage0_dir QATcode/cache_method/Stage0/stage0e_output \
  --output_dir QATcode/cache_method/Stage1/stage1_output \
  --K 16 \
  --smooth_window 5 \
  --lambda 1.0 \
  --k_min 1 \
  --k_max 4 \
  --min_zone_len 2 \
  --lambda_sweep "0.25,0.5,1.0,2.0"
```

| 參數 | 預設 | 說明 |
|------|------|------|
| `--stage0_dir` | `Stage0/stage0e_output` | Stage 0 輸出目錄 |
| `--output_dir` | `Stage1/stage1_output` | 三個 JSON 寫入處 |
| `--K` | 10 | top‑\(\Delta\) 的 change point 個數（內部 cap） |
| `--smooth_window` | 5 | 對 \(G_{\mathrm{proc}}\) 的移動平均視窗 |
| `--lambda` | 1.0 | \(J\) 中 compute penalty 係數 |
| `--lambda_sweep` | `0.25,0.5,1.0,2.0` | 逗號分隔，寫入 diagnostics 的 sweep 對照 |
| `--k_min` / `--k_max` | 1 / 4 | 候選 \(k\) 範圍（zone 內 pattern 去重） |
| `--min_zone_len` | 2 | 短 zone merge 門檻 |
| `--self_test` | — | 用暫存假資料跑內建檢查（不依賴 Stage 0） |

### 7.2 `verify_scheduler.py`

檢查 **`shared_zones`** 是否完整分割 DDIM \(t\)、**`k_per_zone`** 範圍、**`i=0` 全 F**、各 zone 起點 F、以及 **`expanded_mask`** 與 **zone + k 重建** 是否一致。

```bash
python3 QATcode/cache_method/Stage1/verify_scheduler.py \
  --config QATcode/cache_method/Stage1/stage1_output/scheduler_config.json
```

### 7.3 `visualize_stage1.py`

讀取 **`scheduler_config.json`** + **`scheduler_diagnostics.json`**，輸出四張圖（預設檔名）：

1. `1_global_cutting_and_zones.png` — \(G\)、平滑、change points、zones  
2. `2_k_zone_heatmap.png` — 每 block × zone 的 \(k\)  
3. `3_expanded_mask_heatmap.png` — **expanded_mask**  
4. `4_candidate_selected_k.png` — 每 zone 選中 \(k\) 摘要  

```bash
python3 QATcode/cache_method/Stage1/visualize_stage1.py \
  --stage1_output_dir QATcode/cache_method/Stage1/stage1_output \
  --output_dir QATcode/cache_method/Stage1/stage1_figures
```

依賴：**matplotlib**。

### 7.4 `run_stage1_sweep.sh`

對 **`K`、`smooth_window`、`lambda`、`k_max`** 做巢狀迴圈（可改腳本內 `K_LIST` 等，或設 **`STAGE0_DIR`** / **`BASE_OUT`**）。每次執行會跑 scheduler → verify → visualize。

---

## 8. 健壯性與除錯要點

- **Stage 0 / Stage 1 約定**：`t_curr_interval` 不符即 **fail**；訊息含 got/expected 前幾項。  
- **FID 全零**：`G` 用 **均匀權重**，log **warning**。  
- **Zones**：合成後 **`validate_shared_zones_ddim`**，錯誤即 **raise**。  
- **Mask 合併**：`or_expanded_with_zone_mask` 偵測 **步序重疊**（理論上不應發生）。  
- Python API：**`rebuild_expanded_mask_from_config(config)`** 可從 JSON 重建 **`[B,T]`**，邏輯與合成路徑一致。

---

## 9. 依賴

- **Python 3**  
- **NumPy**（scheduler、verify）  
- **Matplotlib**（visualize）

---

## 10. 目錄結構（精簡）

```
QATcode/cache_method/Stage1/
├── stage1_scheduler.py      # 主流程
├── verify_scheduler.py
├── visualize_stage1.py
├── run_stage1_sweep.sh
├── README.md                 # 本檔（含 Quick Start）
├── SUMMARY.md                # 方法設計、baseline 選取、實驗觀察
├── stage1_output/            # 執行輸出（可 gitignore）
└── stage1_figures/
```

---

## 11. 與舊版 Stage 1 的差異（概念）

舊版 **\(A[b,z]\to k\)** 線性映射、zone risk ceiling、舊 regularization **主路徑已移除**。現行 baseline 為：**全域 \(G\) → smoothing → top‑K → merge** 得 **shared zones**，再以 **\(J(b,z,k)\)** 選 **\(k\)**，主輸出為 **DDIM 步序上的 `expanded_mask`**，而非以 analysis axis 為主的舊 mask 敘述。

**版本字串**：見 `scheduler_config.json` 的 **`version`**（目前 **`stage1_baseline_v1`**）。

---

## 12. Quick Start

**時間軸**：對外輸出以 **DDIM 99→0** 為準；`expanded_mask[b,i]` 的 **i=0 為 t=99**。interval ↔ reused timestep 對應見 §2 與 `scheduler_diagnostics.json` 的 `mapping_note`。

**前置**：`t_curr_interval.npy` 須與 Stage 0 正式定義 `arange(T-2,-1,-1)` 一致，否則排程器直接報錯。**FID 全零**時 `G[t]` 改為均匀加權（log warning）。

### 一鍵執行

```bash
cd /home/jimmy/diffae

# 執行 Stage 1（預設參數）
python3 QATcode/cache_method/Stage1/stage1_scheduler.py \
  --stage0_dir QATcode/cache_method/Stage0/stage0e_output \
  --output_dir QATcode/cache_method/Stage1/stage1_output

# 驗證輸出
python3 QATcode/cache_method/Stage1/verify_scheduler.py \
  --config QATcode/cache_method/Stage1/stage1_output/scheduler_config.json

# 視覺化
python3 QATcode/cache_method/Stage1/visualize_stage1.py \
  --stage1_output_dir QATcode/cache_method/Stage1/stage1_output \
  --output_dir QATcode/cache_method/Stage1/stage1_figures
```

### 輸出位置

```
stage1_output/
├── scheduler_config.json
├── scheduler_diagnostics.json
└── verification_summary.json

stage1_figures/
├── 1_global_cutting_and_zones.png
├── 2_k_zone_heatmap.png
├── 3_expanded_mask_heatmap.png
└── 4_candidate_selected_k.png
```

### 常用參數

- `--K`：top-K change points（內部會與 T 一齊 cap）
- `--smooth_window`：對 `G`（步序）的 moving average 視窗
- `--lambda`：`J(b,z,k)` 中 compute penalty 係數（預設 1.0）
- `--k_min` / `--k_max`：候選 k 範圍（pattern 去重後評估）

Sweep 範例：`bash QATcode/cache_method/Stage1/run_stage1_sweep.sh`（編輯腳本內 `K_LIST` 等陣列）。

### Self-test

```bash
python3 QATcode/cache_method/Stage1/stage1_scheduler.py --self_test
```
