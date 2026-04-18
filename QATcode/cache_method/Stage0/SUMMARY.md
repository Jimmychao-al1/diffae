# Stage-0E：正規化設計決策與觀察

---

## 實驗設定

- **資料規模**：31 個 UNet block，T=100 步，interval-wise（長度 99）
- **三路 evidence**：L1（來自 similarity NPZ）、Cosine distance（`1 - cos_step_mean`）、SVD（`subspace_dist[1:]`）
- **FID weight**：來自 `c_FID/fid_cache_sensitivity` 的 delta FID 結果

---

## 觀察結果

### Evidence 正規化方式

採用**全域 min-max normalization**（不含 quantile clipping）：

- 優點：保留 block 間的相對尺度差異，不強行壓縮極端值
- 問題：若存在少數極端 block，可能使其他 block 的變化被壓縮到很小的數值範圍

因此正規化後建議搭配 `stage0e_metadata.json` 確認各 block 的指標來源與失敗狀況。

### FID Weight 分支設計

FID weight 分支選擇先做 **quantile clipping（q=0.95）** 再做 max-normalization：

- 原因：FID delta 分布本身不均勻，少數 block 的 delta 遠高於其他，直接 min-max 會使多數 block 的權重趨近於 0
- Quantile clipping 讓多數 block 能獲得有意義的非零權重，再由 max-normalization 確保值域一致

### Legacy Fallback 設計

設計 `l1_step_mean → l1_rate_step_mean` fallback 的原因：

- 部分舊版 NPZ 缺少 `l1_step_mean` 欄位（上游 v1 格式）
- Fallback 機制讓 Stage0 在上游 NPZ 尚未完全重跑時仍可執行
- 所有 fallback 狀況均記錄在 `stage0e_metadata.json` 的 `l1_source_key` 欄位，方便後續追蹤

### 時間軸對齊

- 內部維持 analysis index `j = 0..T-2`（noise→clear 順序）
- 對外輸出 `t_curr_interval.npy`，語意為 `(T-2) - j`（左大右小）
- Stage 1 會校驗此陣列必須等於 `arange(T-2, -1, -1)`，確保 Stage0/Stage1 間不發生靜默錯位

---

## 結論

1. Stage0E 的設計以「提供給 Stage 1 一個乾淨、對齊的 evidence 矩陣」為核心目標，不做超出正規化與對齊之外的計算。
2. Evidence normalization（L1/Cos/SVD）與 FID weight 分支採用不同的正規化策略，是刻意設計的，反映了兩者用途不同（前者描述 block-timestep 變化量，後者描述 block 優先權重）。
3. `t_curr_interval.npy` 的嚴格校驗是防止時間軸漂移的關鍵設計，應在任何修改上游管線後重新驗證。
