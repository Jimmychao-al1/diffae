# Stage-0E: Loader + Normalization

此模組將 Stage-0 的原始資料（T=100）轉為可直接供 Stage-1 使用的正規化 interval-wise 指標。

## 正式指標定義

1. **L1 step mean (interval-wise)**  
   正式來源為 similarity npz 的 `l1_step_mean`。
2. **Cosine distance (interval-wise)**  
   由 `1.0 - cos_step_mean` 取得。
3. **SVD interval distance**  
   由 `subspace_dist[1:]` 對齊 interval。

## 資料來源（正式路徑）

```text
Similarity: QATcode/cache_method/L1_L2_cosine/T_100/v2_latest/result_npz/*.npz
SVD:        QATcode/cache_method/SVD/svd_metrics/*.json
FID:        QATcode/cache_method/FID/fid_cache_sensitivity/fid_sensitivity_results.json
```

> `T_100/Res/result_npz` 僅為歷史路徑，不是目前正式路徑。

## 相容性（legacy fallback）

- Stage0 會先讀 `l1_step_mean`（正式）。
- 若單一 npz 缺少 `l1_step_mean`，才 fallback 使用 `l1_rate_step_mean`。
- fallback 會在 log 發出 warning，並在 `stage0e_metadata.json` 記錄 `l1_source_key`。

## 正規化與 clipping 說明

- **Evidence normalization（L1/Cos/SVD）**：是全域 min-max normalization，不含 quantile clipping。
- **FID weight 分支**：會做 quantile clipping，之後使用 `w = S_clip / max(S_clip)`（max-normalization）。

## 時間軸定義

- 內部資料順序維持 analysis axis interval index `j=0..T-2`。
- 輸出顯示使用 `t_curr=(T-2)-j`，並寫入：
  - `t_curr_interval.npy`
  - `axis_interval_def.npy`

## 使用方式

```bash
cd /home/jimmy/diffae
python3 QATcode/cache_method/Stage0/stage0e_normalization.py
```

或以模組方式：

```python
from QATcode.cache_method.Stage0.stage0e_normalization import run_stage0e

run_stage0e(
    l1_cos_dir="QATcode/cache_method/L1_L2_cosine/T_100/v2_latest/result_npz",
    svd_dir="QATcode/cache_method/SVD/svd_metrics",
    fid_json_path="QATcode/cache_method/FID/fid_cache_sensitivity/fid_sensitivity_results.json",
    output_dir="QATcode/cache_method/Stage0/stage0e_output",
    eps_noise=0.5,
    quantile=0.95,
    strict=False,
)
```

## 輸出檔案

預設輸出到 `QATcode/cache_method/Stage0/stage0e_output/`：

| 檔案名稱 | 描述 |
|---|---|
| `block_names.npy` | block 名稱列表（正式） |
| `l1_interval_norm.npy` | L1 step mean (interval-wise) 的正規化結果 |
| `cosdist_interval_norm.npy` | cosine distance 的正規化結果 |
| `svd_interval_norm.npy` | SVD interval distance 的正規化結果 |
| `fid_w_qdiffae_clip.npy` | FID weight（quantile-clipped + max-normalized） |
| `fid_w_qdiffae_rank.npy` | FID rank-based weight |
| `t_curr_interval.npy` | 顯示用 t_curr 軸 |
| `axis_interval_def.npy` | t_curr 軸定義文字 |
| `stage0e_metadata.json` | 指標來源、路徑、失敗 block 摘要 |

向下相容別名（不影響正式檔名）：

- `block_names_metric.npy`
- `fid_weights.npy`
- `delta_fid.npy`

## 驗證

```bash
bash QATcode/cache_method/Stage0/verify_stage0e_output.sh
```

此腳本會檢查必要檔案、shape、一致性、值域與時間軸資訊，失敗時回傳非 0 exit code。
