# Stage 1：方法設計、Baseline 選取與觀察

---

## 實驗設定

- **輸入**：Stage 0 正式輸出（31 blocks，T=100，interval 長度 99）
- **固定設定**：`T=100`、`k_max=4`、`min_zone_len=2`
- **Sweep 範圍**：`K ∈ {8, 10, 12, 16, 20, 25}`，`smooth_window ∈ {3, 5}`，`λ ∈ {0.25, 0.5, 1.0, 2.0}`

---

## 方法設計

### Stage 1 的定位

Stage 1 目標是建立一個**初始 static cache scheduler**，供 Stage 2 在此之上做 refinement。不追求最終調參完成，也不擬合每一個局部尖峰；而是先給出**結構清楚、可解釋、可再修正**的時間骨架。

### 核心流程

1. **`I_l1cos[b,t]`**：L1 / Cos 加權（`0.7 × L1_norm + 0.3 × Cos_norm`）
2. **`I_cut[b,t]`**：cutting evidence（`4/9 × I_l1cos + 5/9 × SVD_norm`）
3. **`G[t]`**：全域訊號（各 block 以 FID 權重加權聚合）
4. **Shared zones**：`G` 依步序排列 → smoothing → top-K change points → merge
5. **`k` 選擇**：每 `(b,z)` 最小化 `J(b,z,k) = w_b × mean(I_cut[R]) + λ × |F|/L_z`
6. **`expanded_mask`**：展開成逐步的 F/R mask，強制 `mask[0] = True`

### Stage-2-ready 評估門檻（T=100）

**Zone scaffold 品質：**
- `8 ≤ N_zones ≤ 12`
- `median(L_z) ≥ 3`
- `max(L_z) ≤ 0.55T`
- `frac(L_z = 2) ≤ 0.40`

**Candidate k 有效性：**
- 平均相異 F/R pattern 個數 `≥ 3.0`
- `frac(C_z ≤ 2) ≤ 0.40`（避免短 zone 下 pattern collapse）

---

## 觀察結果

### 為何選 K=16 而非 K=20、K=25

- K=16 在「時間表達」與「不過碎」之間較合適：比 K=8 更能切出結構，比 K=20/25 更能保留 zone 內 k 的有效空間與後續 refine 餘地。
- K 過大時 shared zones 過碎，大量 zone 長度只剩 2、3，候選 k 去重後有效選項減少，`J(b,z,k)` 更易被局部雜訊主導。

### 為何先固定 k_max=4

- `k_max` 不決定全域骨架，只限制 zone 內允許的最大 reuse 間隔。
- 在已有多個短 zone 時，單純加大 k_max 也未必增加相異 pattern。
- 建議待 Stage 2 首輪後，再將 `k_max ∈ {3, 4, 5}` 做獨立 ablation。

### SVD 局部起伏的影響

- SVD 在前後段常較「吵」，支持「時間軸不要只切一刀」。
- 但 **不** 支持「change point 越多越好」——Stage 1 是 scheduler 合成而非純 segmentation。

---

## 結論：選定 Baseline

**`sweep_K16_sw3_lam0.5_kmax4`**

對應：`K=16`、`smooth_window=3`、`λ=0.5`、`k_min=1`、`k_max=4`、`min_zone_len=2`。

此 baseline 的選取邏輯：
1. 先評 zone 結構是否通過 Stage-2-ready 門檻；
2. 結構通過後，比較 `λ` 對 zone 內 k 選擇的影響；
3. `F_frac_mean` 較小者優先；仍接近則優先 `λ = 1.0`（此次 `λ=0.5` 表現較佳）。

---

## 與 Stage 2 的銜接

Stage 2 以此 baseline 為起點做 refinement：
- 重點是 feature error、局部調 k、必要時調 boundary、FID / 算力驗證
- **不** 重新從頭定義全域 scaffold
- Stage 1 輸出（`shared_zones + k_per_zone + expanded_mask`）三者一併作為 Stage 2 起點
