# b_SVD：SVD Correlation 實驗結果摘要

---

## 實驗設定

- **分析對象**：Diff-AE / Q-DiffAE UNet 共 31 個 block（`model.input_blocks.{0..14}`、`model.middle_block`、`model.output_blocks.{0..14}`）
- **T=100**：固定 DDIM 步數 100
- **SVD 樣本數**：每個 block 每個 timestep 收集 N=32 個樣本
- **Similarity 來源**：`a_L1_L2_cosine/T_100/v2_latest/result_npz/*.npz`，正式指標為 `l1_step_mean`（非 l1_rate）
- **對齊方式**：interval-wise（SVD 使用 `subspace_dist[1:]`，similarity 使用 `l1_step_mean` / `cos_step_mean`），x 軸為 `t_curr`（左側 noise，右側 clear）
- **相關性統計**：Pearson + Spearman，各附 p-value

---

## 觀察結果

### 對齊與時間軸

- SVD 與 similarity 的 interval-wise 對齊方式整體正確，無明顯長度錯位或 timestep 方向顛倒問題。
- x 軸語意明確：左側對應較 noisy 的區段，右側對應較 clear 的區段；可延續到後續 Stage 0 / Stage 1 分析。

### 正相關趨勢

- 多數 block 中，SVD subspace distance 與 similarity evidence（L1、cosine distance）呈現明顯**正相關**。
- 當特定 timestep 區段的特徵變化較大時，L1 與 cosine distance 往往同步提高；SVD drift 並非難以解釋的孤立指標。

### Block 間的異質性

不同 block 的相關型態可分為兩類：

**類型 A：整段趨勢一致的 block**
- SVD、L1、cosine distance 的曲線輪廓相對接近，不只局部尖峰同步，整體趨勢也具一定一致性。
- 適合作為 tri-evidence 合理性的代表案例，可用於論文或簡報的正向示意圖。

**類型 B：局部區段主導的 block**
- 整體仍有正相關，但一致性主要集中在特定區段（局部尖峰、端點區域、少數大變化步）。
- 並不表示結果錯誤，但解讀時應更為保守，不宜視為「全段高度一致」的代表。

---

## 結論

1. **SVD pipeline 與 similarity pipeline 的對齊方式合理**，可作為 tri-evidence 的前置驗證。
2. **正式 similarity evidence 採用 L1（非 L1 rate）** 是一致且可解釋的選擇。
3. **SVD subspace distance** 可作為 tri-evidence 中具有解釋性與互補性的組成部分，但它與 similarity 之間的關係並非完全等價，而是從不同角度描述同一類 timestep 結構變化。
4. **不宜只依賴單一相關係數**：有相關不等於整段穩定一致；應搭配 alignment 圖的曲線形態，區分是全段趨勢一致還是局部共振。

精確表述：

> SVD subspace distance 與 similarity evidence（尤其是 L1）在多數 block 上具有明顯的正向對應關係，且整體時間軸對齊方式正確。不同 block 的相關型態存在差異；部分 block 呈現全段趨勢一致性，另一些 block 則更容易受到局部區段影響。因此，SVD evidence 可作為 tri-evidence 中具有解釋性與互補性的組成部分，但後續 Stage 0 / Stage 1 中仍應搭配曲線形態與多種統計觀察共同判讀。

---

## 對 Stage 0 / Stage 1 的意義

- 本結果支持 **interval-wise `t_curr` 軸定義** 延續到後續分析。
- SVD evidence 可納入 tri-evidence，但不應被過度解讀為與 similarity 完全等價。
- 建議在論文或簡報中強調「互補」而非「完全一致」，這更符合目前結果，也更容易與 Stage 1 / Stage 2 的設計邏輯銜接。
- 不同 block 的相關型態差異本身也是重要資訊，可作為 scheduler 設計的背景依據之一。
