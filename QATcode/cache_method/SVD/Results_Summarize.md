# SVD Correlation 結果觀察整理

## 1. 整體結論

本次 SVD correlation 結果，**就資料結構、對齊方式、以及數值趨勢而言，整體是可用的**。  
在不考慮先前已修正的圖上殘留命名問題下，目前結果可支持後續 Stage 0 / Stage 1 的分析與使用。

整體判斷如下：

- SVD 與 similarity 的輸出對齊方式正確
- 正式分析指標已採用 **L1**，而非 **L1 rate**
- interval-wise 的 x 軸定義明確且一致
- 多數 block 可觀察到 **SVD 與 L1 / cosine distance 的正相關**
- 少數 block 雖然 Pearson 高，但 Spearman 弱，顯示其相關性主要可能由局部尖峰主導，而非整段 timestep 都穩定一致

---

## 2. 結果結構與對齊檢查

### 2.1 基本結構

本次結果共包含 **31 個 block** 的 correlation 結果。  
每個 block 的 correlation JSON 在結構上皆一致，未見明顯 schema 錯誤。

### 2.2 interval 長度

所有 block 的：

- `interval_length_used = 99`

這表示目前 correlation 分析確實是在做 **interval-wise** 對齊，而不是 point-wise 的 100 點直接對比。

### 2.3 x 軸定義

所有 block 的 `t_curr_interval` 皆為：

- `98 → 0`

同時 `x_axis_def` 也一致定義為：

- `interval-wise t_curr (left=noise side, right=clear side)`

這代表目前使用的對齊方式是：

- SVD 使用 `subspace_dist[1:]`
- similarity 使用 interval-wise 的 `l1_step_mean` / `cos_step_mean`
- x 軸對應的是 **current timestep `t_curr`**

此結果說明目前 Stage C 的時間軸定義是清楚且正確的，沒有出現長度錯位或 timestep 方向相反的問題。

---

## 3. 整體數值觀察

### 3.1 整體趨勢

從 31 個 block 的 correlation 結果來看，整體呈現：

- `L1 vs SVD`：整體正相關
- `CosDist vs SVD`：整體也呈現正相關

其中，整體統計上可觀察到：

#### L1 vs SVD
- Pearson 平均約 **0.852**
- Spearman 平均約 **0.564**
- Spearman 中位數約 **0.657**

#### CosDist vs SVD
- Pearson 平均約 **0.830**
- Spearman 平均約 **0.526**
- Spearman 中位數約 **0.578**

### 3.2 對數值的解讀

上述現象代表：

- 多數 block 中，SVD 與 similarity 確實存在明顯正相關
- 但許多 block 的 **Pearson 明顯高於 Spearman**
- 這表示兩者之間常見的是：
  - 整體趨勢相近
  - 可能被少數大 spike 或局部區段共同變化所主導
  - 但未必在整段 timestep 上都具有強單調一致性

換言之，這些結果較接近：

> **SVD 與 similarity 在多數 block 上具有可觀察的整體趨勢一致性，但不是所有 block 都能表現出強而穩定的全段單調同步。**

這種現象在 diffusion timestep analysis 中是合理的，並不直接表示方法失敗；但後續在 Stage 0 / Stage 1 使用時，應避免只依賴 Pearson 來判斷 evidence 是否穩定。

---

## 4. block 類型觀察

從相關性與 alignment 圖來看，目前可大致將 block 分成兩類。

---

### 4.1 類型一：整體具有中高程度對齊的 block

這類 block 中，SVD、L1、cosine distance 的曲線整體輪廓相對一致，  
不只是局部尖峰對齊，而是整體趨勢上具有一定同步性。

這類 block 較適合作為：

- tri-evidence 合理性的展示例
- Stage 0 / Stage 1 的代表案例
- 論文或簡報中用來說明「SVD drift 與 similarity evidence 具有一致性」的例子

較具代表性的 block 包括：

- `model_input_blocks_9`
- `model_input_blocks_7`
- `model_input_blocks_8`
- `model_input_blocks_1`
- `model_input_blocks_10`
- `model_input_blocks_11`
- `model_input_blocks_14`
- `model_output_blocks_9`

其中，**`model_input_blocks_9`** 是相對漂亮且具代表性的正面案例。

其特徵為：

- noise side 偏高
- 中段下降
- clear side 再度上升
- SVD / L1 / cosine 的整體輪廓相近

這種 block 非常適合之後用於：

- 方法合理性的展示
- tri-evidence 可解釋性的說明
- Stage 0 / Stage 1 報告中的代表圖

---

### 4.2 類型二：Pearson 高，但 Spearman 弱的 block

這類 block 的特徵是：

- 線性相關（Pearson）可能很高
- 但排序一致性（Spearman）很弱
- 通常表示整體趨勢不是穩定同步，而是少數區段、尤其是尾端尖峰，主導了 correlation

這類 block 不代表「錯」，但解讀時要小心。  
它們較像是：

> **局部共振型 / endpoint spike dominated 的 block**

較需注意的 block 包括：

- `model_output_blocks_11`
- `model_input_blocks_3`
- `model_input_blocks_4`
- `model_output_blocks_12`
- `model_output_blocks_13`

其中最明顯的是 **`model_output_blocks_11`**。

其現象為：

- `L1_spearman` 幾乎接近 0
- `Cos_spearman` 也偏弱
- 但 Pearson 卻很高

這表示它更像是：

- 大部分 timestep 區段變化不具穩定排序關係
- 但接近 clear side 時，SVD 與 similarity 同時出現顯著尖峰
- 因此線性上看似相近，卻不代表整段都具有良好的一致性

此類 block 不適合拿來當作「SVD 與 similarity 在整段 timestep 上一致」的主要證據。

---

## 5. 幾個代表性 block 的具體解讀

### 5.1 `model_input_blocks_0`

此 block 的觀察重點為：

- `L1_spearman` 很高
- `Cos_spearman` 中等
- `Cos_pearson` 相對較低

從 alignment 圖可觀察到：

- SVD 存在數個明顯尖峰
- L1 整體較接近平滑上升
- cosine distance 的變化則較像稀疏台階

這代表：

> 在此 block 中，L1 對 timestep drift 的反應較穩定，  
> cosine distance 對此 block 的結構變化捕捉程度則相對較弱。

此觀察反而支持目前選擇 **L1 作為正式 similarity evidence** 的決策。

---

### 5.2 `model_input_blocks_4`

此 block 的現象是：

- Pearson 很高
- Spearman 很弱

從圖形來看，其合理解釋為：

- clear side 右端的共同爆點大幅拉高 Pearson
- 但 noise side 與中段的排序與起伏並未穩定對齊

這說明：

> 若只看 Pearson，容易高估此 block 的整體一致性。  
> Spearman 在這裡具有重要的輔助判讀價值。

因此在後續 tri-evidence 分析中，不建議僅以 Pearson 決定 evidence 是否可靠。

---

### 5.3 `model_input_blocks_9`

此 block 是本次結果中相當理想的案例之一。

其特徵為：

- noise side 較高
- 中段降低
- clear side 再次上升
- SVD / L1 / cosine 三者在整體曲線輪廓上都具有相當不錯的一致性

這代表：

> 對於某些 block，SVD drift 並非獨立且難以解釋的指標，  
> 而是與 similarity evidence 具有相當明確的共同結構。

因此此 block 很適合作為後續：

- 論文示意圖
- 簡報中的正向代表案例
- tri-evidence 有效性的視覺化說明

---

## 6. 對 Stage 0 / Stage 1 的意義

### 6.1 可以接受的部分

目前結果已可支持以下判斷：

1. **SVD pipeline 與 similarity pipeline 的對齊方式正確**
2. **正式分析已採用 L1，而非 L1 rate**
3. **interval-wise 的 `t_curr` 軸定義已經明確建立**
4. **大多數 block 的 SVD 與 similarity 之間確實存在正相關**
5. **SVD evidence 並非無法解釋或與 similarity 完全脫節**

因此，這份結果可作為後續 Stage 0 / Stage 1 的基礎輸入。

---

### 6.2 後續使用時需要注意的部分

雖然結果整體可用，但後續使用時應特別注意以下幾點：

#### (1) 不要只看 Pearson
目前多個 block 出現：

- Pearson 高
- Spearman 弱

這說明有些 block 的相關性主要來自局部尖峰，而不是整段都一致。

因此在後續 Stage 0 / Stage 1 中，若要判斷某個 block 的 tri-evidence 是否穩定，不建議只用 Pearson 作為依據。

#### (2) Spearman 與圖形形態同樣重要
除了數值本身之外，alignment 圖的形態也很重要。  
特別是要辨別：

- 是整段曲線整體一致
- 還是只是 clear side / noise side 局部共振

#### (3) endpoint spike dominated 的 block 要保守解讀
像 `model_output_blocks_11` 這類型的 block，不能直接拿來當作「SVD 與 similarity 全程一致」的證據。  
這類 block 在後續分析中較適合作為：

- 補充觀察
- 異常類型說明
- 反例或限制討論

而不是主要方法有效性的代表案例。

---

## 7. 總結判斷

綜合本次結果，可得出以下整體判斷：

### 7.1 結果是否可用？
**可以。**

### 7.2 是否需要因數值問題而重跑？
**目前沒有必要僅因為數值結果而重跑。**

從 JSON 與圖的整體觀察來看：

- 沒有明顯長度錯位
- 沒有明顯 schema 錯誤
- 沒有 evidence 完全失效的跡象
- 大多數 block 的 SVD 與 similarity 的確存在合理對應關係

### 7.3 最合適的結論表述

較精確的說法應為：

> 本次 SVD correlation 結果顯示，SVD subspace distance 與 similarity evidence（L1、cosine distance）在多數 block 上具有明顯正相關，且對齊方式正確。  
> 然而，不同 block 之間的相關型態存在差異；部分 block 呈現較穩定的全段趨勢一致性，另一些 block 則較受局部尖峰影響。  
> 因此，SVD evidence 可作為 tri-evidence 中具有解釋性的組成部分，但在後續 Stage 0 / Stage 1 中，仍應搭配 Spearman 與圖形形態一起判讀，而不宜只依賴 Pearson。

---

## 8. 後續建議

接下來較建議的方向為：

1. **挑選代表性 block**
   - 正面案例：如 `model_input_blocks_9`
   - 保留一到兩個局部尖峰主導型 block，作為限制說明

2. **在 Stage 0 / Stage 1 使用時保留多種判讀方式**
   - 不只看 Pearson
   - 同時觀察 Spearman 與 alignment 圖形態

3. **將這批結果作為 SVD evidence 的 sanity check 與方法合理性支撐**
   - 可用於論文方法章節的定性說明
   - 可用於簡報中說明 tri-evidence 並非彼此獨立、而是存在一致性與互補性