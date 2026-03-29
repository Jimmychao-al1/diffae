# Stage 1 總結（中文版）

## 1. Stage 1 的定位

Stage 1 的目標是建立一個 **初始 static cache scheduler**，供 Stage 2 在此之上做 refinement。  
它**不**追求最終調參完成，也**不**擬合證據曲線上每一個局部尖峰；而是先給出 **結構清楚、可解釋、可再修正** 的時間骨架。

在目前實作（commit: `b262d51a947a53fc5276ff8243676c942a0bbce5`）中，Stage 1 做兩件事：

1. 由全域證據切出 **全 block 共用** 的 `shared_zones`；
2. 在每個 $(b,z)$ 上選定 reuse 週期 $k$，並展開成逐步的 $F/R$ mask。

輸出為 `shared_zones`、`k_per_zone`、`expanded_mask`，三者一併作為 Stage 2 的起點。

---

## 2. Stage 1 的流程定義

### 2.1 輸入資料

讀入 Stage 0 正式檔案（略舉檔名）：`block_names.npy`、`l1_interval_norm.npy`、`cosdist_interval_norm.npy`、`svd_interval_norm.npy`、`fid_w_qdiffae_clip.npy`、`axis_interval_def.npy`、`t_curr_interval.npy`。

程式會 **校驗** `t_curr_interval` 是否與「interval → reused timestep」約定一致，避免 Stage 0 / Stage 1 **靜默錯位**。

---

### 2.2 將 interval-wise evidence 映射到 reused DDIM timestep

Stage 0 以 **interval** 為欄；Stage 1 將欄位對應到 **reuse 發生時所評估的 DDIM timestep** $t$（語義上：interval $(t{+}1\!\to\!t)$ 的風險落在 $t$）。

**式 (1)**（L1 / Cos **變化量**加權；**不是** stability / similarity 分數）：

$$
I_{\mathrm{l1cos}}[b,t]
  = 0.7\,\mathrm{L1}_{\mathrm{norm}}[b,t] + 0.3\,\mathrm{Cos}_{\mathrm{norm}}[b,t].
$$

**式 (2)**（cutting evidence）：

$$
I_{\mathrm{cut}}[b,t]
  = \tfrac{4}{9}\, I_{\mathrm{l1cos}}[b,t] + \tfrac{5}{9}\,\mathrm{SVD}_{\mathrm{norm}}[b,t].
$$

最後一步 DDIM（$t = T{-}1$，例如 $T{=}100$ 時 $t{=}99$）沒有對應 interval 欄，程式令 **式 (3)**：

$$
I_{\mathrm{cut}}[b, T-1] = 0
$$

（僅作為 cutting 統計；實際排程仍 **強制該步為 full-compute**）。

---

## 3. Global cutting signal $G[t]$

每個 timestep $t$，將各 block 的 $I_{\mathrm{cut}}[b,t]$ 以 FID 權重 $w_b$ 聚合為全域訊號 $G[t]$。

**式 (4)**（權重正規化；若 $\sum_b w_b \approx 0$ 則改 **均匀** $1/B$ 並 **warning**，與程式一致）：

$$
G[t] = \sum_b \frac{w_b}{\sum_{b'} w_{b'}} \, I_{\mathrm{cut}}[b,t].
$$

---

## 4. 時間分段：由 $G$ 得到 shared zones

先將 $G[t]$ 依 **DDIM 採樣順序**重排（步序 $i=0$ 對應 $t=T{-}1$，$i=T{-}1$ 對應 $t=0$），得 $G_{\mathrm{proc}}$，再 smoothing 得 $G_{\mathrm{smooth}}$。

**式 (5)**（鄰步差分）：

$$
\Delta[i] = \left|\, G_{\mathrm{smooth}}[i] - G_{\mathrm{smooth}}[i-1] \,\right|.
$$

在 $\Delta$ 上取 **top‑$K$** change points 得到初始 shared partition；再對過短 zone 做 **merge**，避免結構被大量短區段支配。

---

## 5. 為什麼不能只跟著 SVD 局部起伏加 $K$

SVD 在前後段常較「吵」，這支持「時間軸**不要**只切一刀」— 但 **不** 支持「change point **越多越好**」。

Stage 1 不是純 segmentation，而是 **scheduler 合成**：zones 還要保留合理的 $k$ 決策空間，並留給 Stage 2 **局部微調**。若 $K$ 過大：

1. shared zones 過碎；
2. 大量 zone 長度只剩 $2$、$3$；
3. 候選 $k$ 經 pattern 去重後 **有效** 選項變少；
4. $J(b,z,k)$ 更易被 **局部雜訊** 主導，而非區段層級行為。

因此 Stage 1 追求的是「可當 prior 的骨架」，不是最大化對曲線的局部擬合。

---

## 6. Zone 內的 $k$：目標函數

在 shared zones 固定後，對每個 $(b,z)$ 選 $k$。**式 (6)**：

$$
J(b,z,k)
  = w_b \cdot \frac{1}{L_z} \sum_{t \in \mathcal{R}} I_{\mathrm{cut}}[b,t]
    \;+\; \lambda \cdot \frac{|\mathcal{F}|}{L_z},
$$

其中 $L_z$ 為 zone 長度；$\mathcal{R}$、$\mathcal{F}$ 分別為該 zone 內 reuse / full-compute 的 timestep 集合；$\lambda$ 權衡 reuse 風險與計算懲罰。

實作上先以 `unique_k_representatives(...)` **去掉等價 F/R pattern 的冗餘 $k$**，再取最小 $J$ 對應的 $k$。

---

## 7. Baseline 怎麼挑

### 7.1 先選 $K$ 與平滑視窗，再選 $\lambda$

- $K$ 與平滑視窗決定 **shared_zones**；
- $\lambda$ 只改 zone **內** $k$ 的選法，**不**改骨架。

故應先評 **zone 結構**，再比 $\lambda$。

---

## 8. Stage‑2‑ready 的經驗門檻（$T{=}100$ 等設定）

預設：$T=100$，$k_{\max}=4$，min\_zone\_len $=2$。

### 8.1 Shared zone scaffold

令 $L_z$ 為各 zone 長度，$N_{\mathrm{zones}}$ 為 zone 數。要求：

$$
8 \le N_{\mathrm{zones}} \le 12, \quad
\mathrm{median}_z(L_z) \ge 3, \quad
\max_z L_z \le 0.55\,T,
$$
$$
\mathrm{frac}(L_z = 2) \le 0.40, \quad
\mathrm{frac}(L_z \le 3) \le 0.60.
$$

意在同時排除「切太粗」與「切太碎」。

### 8.2 Candidate $k$ 有效性

令 $C_z$ 為 zone $z$ 去重後 **相異 F/R pattern** 個數。要求：

$$
\overline{C}_z \ge 3.0, \qquad \mathrm{frac}(C_z \le 2) \le 0.40.
$$

避免短 zone 下名義上很多 $k$、實際上 pattern 早已 collapse。

### 8.3 硬篩後排序

1. $F_{\mathrm{frac\,mean}}$ 較小者；
2. 接近則 $\max_z L_z$ 較小者；
3. 仍接近則優先 $\lambda = 1.0$。

---

## 9. 本文件採用的 baseline

依目前 sweep，選定：

**`sweep_K16_sw5_lam1.0_kmax4`**

對應：$K=16$，$\lambda=1.0$，$k_{\min}=1$，$k_{\max}=4$，smooth\_window $=5$，min\_zone\_len $=2$。

---

## 10. 為何選 $K=16$ 而非 $K=20$、$K=25$

並非說 $K=20/25$「錯」，而是就 **Stage 2 起點** 而言，$K=16$ 在「時間表達」與「不過碎」之間較合適：比 $K=8$ 更能切出結構，比 $K=20/25$ 更能保留 zone 內 $k$ 的有效空間與後續 refine 餘地。

---

## 11. 為何先固定 $k_{\max}=4$

$k_{\max}$ **不**決定全域骨架，只限制 zone 內允許的最大 reuse 間隔；在已有多個短 zone 時，單純加大 $k_{\max}$ 也未必增加 **相異** pattern。若同時調 $k_{\max}$ 與切分策略，會混淆兩種效應。建議待 Stage 2 首輪後，再將 $k_{\max}\in\{3,4,5\}$ 做獨立 ablation。

---

## 12. 小結

流程可概括為：由 L1/Cos/SVD 建 $I_{\mathrm{cut}}$ → FID 加權得 $G$ → smoothing 與 top‑$K$ 得 shared zones → 以 $J(b,z,k)$ 選 $k$ → 輸出 `shared_zones + k_per_zone + expanded_mask`。

選定 baseline：**`sweep_K16_sw5_lam1.0_kmax4`**。

---

## 13. 與 Stage 2 的銜接

Stage 2 以此為起點做 cache-run refinement：重點是 **feature error、局部調 $k$、必要時調 boundary、FID/算力驗證**，而非重新從頭定義全域 scaffold。

**一句話：** Stage 1 提供 **結構清楚、可解釋、可再 refinement** 的初始 static scheduler。
