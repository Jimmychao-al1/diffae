# SVD Correlation 結果整理

## 1. 整體結論

本次 SVD correlation 結果，從資料結構、時間軸對齊方式與整體趨勢觀察來看，已可作為後續 Stage 0 / Stage 1 分析的有效基礎。

目前可得到的核心結論如下：

- SVD 與 similarity 的對齊方式整體正確，未見明顯的長度錯位或 timestep 方向顛倒問題。
- 本次正式使用的 similarity 指標為 **L1**，而非 **L1 rate**。
- 目前分析採用的是 **interval-wise** 對齊，而不是直接以 100 個 point-wise timestep 做逐點比較。
- 多數 block 中，SVD subspace distance 與 similarity evidence（L1、cosine distance）可觀察到明顯的趨勢一致性。
- 不同 block 的相關型態並不完全相同；部分 block 呈現較穩定的全段對齊，另一些 block 則較容易受到局部尖峰或特定區段變化所主導。

整體而言，這批結果支持將 **SVD evidence** 納入 tri-evidence 架構中使用；但在解讀時，仍需同時考慮數值統計與曲線形態，而不宜只依賴單一相關係數。

---

## 2. 結果結構與對齊檢查

### 2.1 基本結果結構

本次結果涵蓋 **31 個 UNet block**，並已產出對應的：

- `svd_metrics/*.json`
- `correlation/*.json`
- `correlation/figures/*.png`
- `logs/*.log`

從目前整理結果來看，各 block 的輸出格式與分析流程皆一致，未見明顯 schema 錯誤或缺檔情況。

### 2.2 interval-wise 分析設定

本次 SVD 與 similarity 的對齊不是直接比較 100 個 timestep 的 point-wise 值，而是採用 **interval-wise** 定義：

- SVD 使用 `subspace_dist[1:]`
- similarity 使用 interval 對應的 step-level 指標
- x 軸對應的是 **current timestep `t_curr`**

這代表本次分析真正比較的是：

> 相鄰 timestep 之間的變化量，是否在 SVD 與 similarity 兩條 evidence 中呈現一致趨勢。

這種設計在方法上是合理的，因為 SVD subspace distance 本質上就是描述 `t` 與 `t-1` 之間的變化，而不是單點狀態本身。

### 2.3 時間軸定義

目前結果已明確建立 x 軸的語義：

- 採用 interval-wise 的 `t_curr`
- 左側對應較 noisy 的區段
- 右側對應較 clear 的區段

因此，目前 Stage C 的圖與統計結果在時間軸解讀上是清楚的，可直接作為後續 Stage 0 / Stage 1 的輸入與說明依據。

---

## 3. 整體觀察

### 3.1 多數 block 呈現正相關

從目前整理結果來看，SVD 與 similarity evidence 在多數 block 上呈現 **正相關**。  
這表示：

- 當某些 timestep 區段的 feature 變化較大時，
- L1 與 cosine distance 往往也會同步提高，
- 顯示 SVD drift 並非完全獨立、難以解釋的指標。

換言之，SVD evidence 與 similarity evidence 並不是彼此脫節的兩套訊號，而是對同一類 timestep 結構變化，提供不同角度的描述。

### 3.2 block 間存在異質性

雖然整體趨勢偏正向，但不同 block 的相關型態仍有明顯差異。  
大致可分為兩類：

#### 類型 A：整段趨勢較一致的 block

這類 block 中，SVD、L1、cosine distance 的曲線輪廓相對接近，  
不只是局部尖峰同步，而是整體趨勢也具有一定程度的一致性。

這類 block 較適合作為：

- tri-evidence 合理性的代表案例
- Stage 0 / Stage 1 的示意圖
- 論文或簡報中的正向例子

#### 類型 B：局部區段主導的 block

這類 block 的特徵通常是：

- 整體看起來仍有正相關
- 但一致性主要集中在某些區段
- 尤其可能由局部尖峰、端點區域，或少數大變化區段主導

這類 block 並不表示結果錯誤，  
但在解讀上應更加保守，不宜直接將其當作「全段 timestep 都高度一致」的代表。

---

## 4. 對結果的解讀

### 4.1 為何這份結果是有價值的？

本次 correlation 結果的價值，不在於證明 SVD 與 similarity 完全等價，  
而在於顯示：

> SVD subspace distance 可以作為一個具有可解釋性的補充 evidence，與 L1 / cosine distance 一起描述 diffusion 過程中的 block-level timestep 變化。

這件事對後續 tri-evidence pipeline 很重要，因為它代表：

- SVD evidence 不是任意加入的額外指標
- 它與 similarity 之間存在可觀察的一致性
- 但又保留了與 similarity 不完全相同的資訊量，因此具有互補意義

### 4.2 為何不能只看單一相關係數？

目前結果也提醒了一個重要事實：

> 「有相關」不等於「整段都穩定一致」。

即使某些 block 的整體相關係數看起來不錯，也可能只是因為：

- 特定 timestep 區段變化特別大
- 多條 evidence 在該區段剛好同步上升
- 因此拉高了整體相關程度

所以在後續 Stage 0 / Stage 1 使用時，更合理的做法是：

- 同時參考相關性統計
- 觀察 alignment 圖的曲線形態
- 分辨是全段一致，還是局部共振

---

## 5. 對 Stage 0 / Stage 1 的意義

### 5.1 目前可支持的判斷

本次結果已足以支持以下幾點：

1. **SVD pipeline 與 similarity pipeline 的對齊方式是合理的**
2. **目前正式 similarity evidence 採用 L1 是一致且可解釋的**
3. **interval-wise 的 `t_curr` 軸定義可以延續到後續分析**
4. **SVD evidence 可作為 tri-evidence 中具有方法意義的一部分**
5. **多數 block 中，SVD 與 similarity 之間確實存在可觀察的對應關係**

因此，這批結果可視為：

- SVD evidence 的 sanity check
- tri-evidence 合理性的支撐材料
- Stage 0 / Stage 1 使用 SVD 指標的前置驗證

### 5.2 使用時的注意事項

雖然整體結果可用，但後續在 Stage 0 / Stage 1 中仍應注意：

#### (1) 不宜只依賴單一統計量
不建議只根據單一相關係數判定某個 block 的 evidence 是否穩定。

#### (2) 需要結合圖形判讀
alignment 圖的形態可以幫助區分：

- 整段趨勢一致
- 局部尖峰同步
- 端點或尾段主導

#### (3) block 間差異本身就是重要資訊
不同 block 的相關型態不同，並不一定表示方法失敗；  
反而說明不同 block 在 diffusion 過程中的角色與敏感性可能本來就不同。  
這種異質性本身，也可作為後續 scheduler 設計的背景依據之一。

---

## 6. 結論

綜合本次結果，可得到以下判斷：

### 6.1 結果是否可用？
**可以。**

### 6.2 是否需要因為目前觀察結果而重跑？
**目前沒有必要僅因為 correlation 結果本身而重跑。**

### 6.3 最合適的總結表述

較精確的說法可寫為：

> 本次 SVD correlation 結果顯示，SVD subspace distance 與 similarity evidence（尤其是 L1）在多數 block 上具有明顯的正向對應關係，且整體時間軸對齊方式正確。  
> 然而，不同 block 的相關型態存在差異；部分 block 呈現較穩定的全段趨勢一致性，另一些 block 則更容易受到局部區段或尖峰變化影響。  
> 因此，SVD evidence 可作為 tri-evidence 中具有解釋性與互補性的組成部分，但在後續 Stage 0 / Stage 1 中，仍應搭配曲線形態與多種統計觀察共同判讀，而不宜簡化為單一相關係數的結論。

---

## 7. 後續建議

1. **挑選代表性 block 作為展示案例**
   - 一類是整體對齊較佳的正向案例
   - 另一類是局部區段主導的限制案例

2. **在 Stage 0 / Stage 1 中延續目前的解讀方式**
   - 保留 interval-wise `t_curr` 定義
   - 正式 similarity evidence 維持以 L1 為主
   - SVD 作為可解釋且互補的 evidence 使用

3. **將本結果定位為方法合理性驗證，而非最終判決工具**
   - 它適合支撐「SVD evidence 為何可納入 tri-evidence」
   - 但不應被過度解讀為「SVD 與 similarity 完全等價」

4. **在論文或簡報中強調“互補”而非“完全一致”**
   - 這樣的表述更符合目前結果
   - 也更容易與後續 Stage 1 / Stage 2 的設計邏輯銜接