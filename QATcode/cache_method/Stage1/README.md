# Stage-1：Offline Scheduler（baseline v1）

## 定位（新版 baseline）

Stage-1 **不**負責最終 refinement；角色為：

1. 讀入 Stage 0 正式輸出（`l1_interval_norm.npy`、`cosdist_interval_norm.npy`、`svd_interval_norm.npy`、`fid_w_qdiffae_clip.npy`、`block_names.npy`、`axis_interval_def.npy`、`t_curr_interval.npy`）。
2. 建立 **global / shared temporal skeleton**（同一套 `shared_zones` 給所有 block）。
3. 在每個 zone 上為每個 block 選初始 `k_{b,z}`（**cost-based**，見下）。
4. 展開成步級 `F/R` schedule，輸出 **`expanded_mask.shape = [B, T]`**（`True=F`，`False=R`）。

**相較舊版**：已廢除 `A[b,z] → k` 線性映射、zone risk ceiling、舊 regularization 主路徑；改為 **FID 加權的全域 cutting 信號 `G` → moving average → top-K change points → merge 短 zone**，以及 **每 (block, zone) 對候選 k 最小化 `J(b,z,k)`**。

### 時間軸（唯一正式）

- **DDIM 順序 99 → 0**：`expanded_mask[b, i]` 的 **i=0 對應 DDIM t=99（第一步）**，**i=T-1 對應 t=0**。
- Stage 0 為 **interval-wise**；正式對應：**interval (t+1 → t)** 的證據算在 **reused timestep t** 上（見程式內註解與 `scheduler_diagnostics.json` 的 `mapping_note`）。
- **載入檢查**：`t_curr_interval.npy` 必須與 `np.arange(T-2,-1,-1)` 完全一致，否則 **raise ValueError**（避免與 Stage 0 欄位語義漂移）。`axis_interval_def` 若為空則 **warning** 但不中止。
- **合成期 fail-fast**：`shared_zones` 建立後會驗證 DDIM `t` 完整分割、無 overlap、長度和為 T；展開 `expanded_mask` 時若 zone 在步序上重疊則 **raise**（理論上不應發生）。

### 核心公式（摘要）

- `I_l1cos = 0.7*L1 + 0.3*Cos`（**變化量**分支加權；**不是** similarity / stability 分數；診斷鍵名 `I_l1cos_stats`）
- `I_cut = (4/9)*I_l1cos + (5/9)*SVD`
- `G[t]`：對 `I_cut[b,t]` 用 `fid_w_qdiffae_clip[b]` 做 **block 加權平均**；若 **fid_w 全為 0** 則 **warning** 並改為 **uniform 1/B**（仍寫入同一 `G_ddim` 結構）
- Zone：`G` 沿 **處理步序**（i=0 為 t=99）做 moving average → 鄰步差分 → **top-K** change points → **min_zone_len=2**，短 zone **預設 merge right**（最後一段則 merge left）
- `J(b,z,k) = w_b * (1/L_z) * sum_{t in R} I_cut[b,t] + lambda * (|F|/L_z)`，其中 **僅第一項乘 `w_b`**（FID），`lambda` 預設 1.0；候選 k 在 zone 內對 **F/R pattern 去重**

## 使用方式

```bash
cd /home/jimmy/diffae

python3 QATcode/cache_method/Stage1/stage1_scheduler.py \
  --stage0_dir QATcode/cache_method/Stage0/stage0e_output \
  --output_dir QATcode/cache_method/Stage1/stage1_output \
  --K 10 \
  --smooth_window 5 \
  --lambda 1.0 \
  --k_min 1 \
  --k_max 4
```

- `--lambda_sweep`：逗號分隔，寫入 `scheduler_diagnostics.json`（各 λ 下若重選 k 的對照）。

```bash
python3 QATcode/cache_method/Stage1/verify_scheduler.py \
  --config QATcode/cache_method/Stage1/stage1_output/scheduler_config.json

python3 QATcode/cache_method/Stage1/visualize_stage1.py \
  --stage1_output_dir QATcode/cache_method/Stage1/stage1_output \
  --output_dir QATcode/cache_method/Stage1/stage1_figures
```

```bash
bash QATcode/cache_method/Stage1/run_stage1_sweep.sh
```

## 輸出檔案

| 檔案 | 說明 |
|------|------|
| `scheduler_config.json` | `version`, `T`, `time_order=ddim_99_to_0`, `stage1_baseline_params`, `shared_zones`, `blocks`（含 `k_per_zone`, `expanded_mask`） |
| `scheduler_diagnostics.json` | `I_l1cos_stats` / `I_cut_stats`、`G_ddim`、平滑曲線、change points、zones、候選 k、cost 表、`lambda_sweep` 對照等 |
| `verification_summary.json` | zone 邊界、每 block 的 k、`#F`/`#R`、total cost、每 zone 候選 k 與選後摘要 |

## 檔案結構

```
QATcode/cache_method/Stage1/
├── stage1_scheduler.py
├── verify_scheduler.py
├── visualize_stage1.py
├── run_stage1_sweep.sh
├── README.md
├── stage1_output/
└── stage1_figures/
```

**版本**：`stage1_baseline_v1`（見 `scheduler_config.json` 的 `version`）
