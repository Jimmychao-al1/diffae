# Activation Distribution Analysis

## 分析目的
在 **相同 `w + lora` effective weights** 的前提下，比較：

- **FF**：weight quant off, act quant off
- **FT**：weight quant off, act quant on

並確認：

> **activation quantization 到底改變了哪些 logical Conv2d / Linear layer 的輸入分布，進而可能影響最終 FID。**

---

## 實驗設定
- Sampling steps：**T = 100**
- Seed：**0**
- 生成樣本數：**32**
- 比較模式：**FF vs FT**
- 權重路徑：**`w + lora`**
- 分析對象：從 quant wrapper modules 收集的 **logical layer input activations**

---

## 分析層覆蓋情況
- 總共選取 logical layers：**142 層**
- Conv2d layers：**72 層**
- Linear layers：**70 層**

Wrapper 類型分布：
- `QuantModule_DiffAE_LoRA`：**140 層**
- `QuantModule`：**2 層**
  - `model.input_blocks.0.0`
  - `model.out.2`

---

## activation_distribution_analysis.py 輸出欄位與分析意義

本分析程式的目的是：  
在 **相同 `w + lora` 權重** 的前提下，比較 **FF**（weight quant off, act quant off）與 **FT**（weight quant off, act quant on）兩種模式下，各個 logical Conv2d / Linear layer 的 **forward input activation 分布** 是否發生改變。

換句話說，這裡分析的是：

> **某一層在真正進入 Conv2d / Linear 運算之前，所接收到的 activation 數值分布長什麼樣子。**

這些統計量的意義不是只看「數值大小」，而是用來判斷：

- activation quant 是否改變了該 layer 的輸入尺度
- 是否改變了分布的 tail 行為
- 是否讓分布更集中、更加稀疏、或更加偏斜
- 這些變化是否集中在特定 layer / block / timestep

---

### 1. 基本統計量

#### `numel`
- 定義：該 layer 在該 timestep 下，用於統計的 activation 總元素數量。
- 意義：
  - 用來表示本次統計的樣本量。
  - `numel` 越大，統計通常越穩定。
- 注意：
  - 不同 layer 因 feature map shape 不同，`numel` 可能差很多，因此不同 layer 間不適合直接比較原始統計值的絕對大小，而應更重視 **FF vs FT 的相對變化**。

---

#### `mean`
- 定義：activation 平均值。
- 意義：
  - 反映整體分布中心是否偏正或偏負。
  - 若 `mean` 接近 0，代表正負大致平衡。
- 在本研究中的用途：
  - 用來觀察 activation quant 是否讓某層輸入整體偏移。
  - 若 FF 與 FT 的 `mean` 差異很小，表示 quant 沒有明顯改變整體中心位置。

---

#### `std`
- 定義：standard deviation，標準差。
- 意義：
  - 描述 activation 的整體離散程度。
  - `std` 越大，表示數值分布越分散、波動越大。
- 在本研究中的用途：
  - 是判斷 activation quant 是否改變某層輸入尺度的核心指標之一。
  - 若 `std_ratio = std_FT / std_FF > 1`，表示 FT 下該層輸入更分散。
  - 若 `< 1`，表示 FT 下分布較收斂。

---

#### `min` / `max`
- 定義：該 activation 分布中的最小值與最大值。
- 意義：
  - 描述最極端的負值與正值。
- 在本研究中的用途：
  - 只能作為輔助觀察，因為它們對單一極端值非常敏感。
  - 若要看 tail 行為，通常比 `min/max` 更適合看 `q99`、`q999`、`abs_max`。

---

#### `abs_mean`
- 定義：所有 activation 絕對值的平均。
- 意義：
  - 反映數值整體幅度，不考慮正負號。
- 在本研究中的用途：
  - 如果 `mean` 接近 0，但 `abs_mean` 很大，表示 activation 雖然正負平衡，但整體強度不小。
  - 可用來輔助判斷 quant 是否改變某層輸入能量。

---

#### `abs_max`
- 定義：絕對值最大值，即 `max(|x|)`。
- 意義：
  - 反映極端值幅度。
- 在本研究中的用途：
  - 是判斷 activation quant 是否影響極端值的重要指標。
  - 若 `abs_max_ratio > 1`，代表 FT 下該層出現更大的極端 activation。
  - 若 `< 1`，則可解讀為 FT 對極端值有抑制效果。

---

### 2. 分位數統計量（Quantiles）

分位數的好處是：  
比 `min/max` 更穩健，較不容易被單一 outlier 主導。

---

#### `q001`
- 定義：0.1% quantile。
- 意義：
  - 代表最左側極小值區域的位置。
  - 只有最小的 0.1% 數值會低於它。
- 在本研究中的用途：
  - 用來觀察負向 tail 是否改變。
  - 若 FF 與 FT 的 `q001` 差異很大，表示 quant 可能影響到極端負值區域。

---

#### `q01`
- 定義：1% quantile。
- 意義：
  - 代表左側尾巴的較穩定估計。
- 在本研究中的用途：
  - 與 `q001` 類似，但比 `q001` 穩定一些。
  - 可用來判斷負向 tail 是否整體位移。

---

#### `q05`
- 定義：5% quantile。
- 意義：
  - 代表左側較低值區域。
- 在本研究中的用途：
  - 比極端 tail 更接近主要分布主體。
  - 可用來看 quant 是否改變較低值區段的形狀。

---

#### `q50`
- 定義：median，中位數。
- 意義：
  - 表示分布中心位置，比 `mean` 更不受極端值影響。
- 在本研究中的用途：
  - 可用來確認 activation 的中心是否穩定。
  - 若 `q50` 幾乎不變，但 `q99/q999` 改變，表示中心沒變，但 tail 改變了。

---

#### `q95`
- 定義：95% quantile。
- 意義：
  - 代表右側高值區域，但還不算極端 tail。
- 在本研究中的用途：
  - 用來看高值區是否整體偏移。
  - 若 `q95` 已變化明顯，通常代表不只是最極端值在改變，而是整個高值區都在變。

---

#### `q99`
- 定義：99% quantile。
- 意義：
  - 代表右側尾巴位置。
  - 比 `q95` 更接近 tail，但比 `q999` 更穩定。
- 在本研究中的用途：
  - 是本研究中很重要的 tail 指標之一。
  - 若 `q99_ratio > 1`，可解讀為 FT 下高值尾巴略為擴大。
  - 若 `< 1`，可解讀為 FT 下高值尾巴被壓縮。
- 為什麼重要：
  - FID 的改善不一定來自整體均勻變化，可能只來自少數關鍵 layer 的高值 tail 改變。
  - `q99` 可以用來抓這種現象。

---

#### `q999`
- 定義：99.9% quantile。
- 意義：
  - 更接近極端高值 tail。
  - 比 `max` 穩定，但比 `q99` 更敏感於極端區域。
- 在本研究中的用途：
  - 是分析 activation tail 行為的核心指標之一。
  - 若 `q999_ratio` 顯著偏離 1，表示 FT 對極高 activation 區域有實質影響。
- 解讀方式：
  - `q999_ratio > 1`：FT 下極高尾巴更重
  - `q999_ratio < 1`：FT 下極高尾巴被抑制
- 為什麼比 `max` 更好：
  - `max` 容易受單一值影響；`q999` 仍然聚焦極端區，但穩定性較高。

---

### 3. 符號比例與稀疏性相關指標

#### `zero_ratio`
- 定義：activation 中「剛好等於 0」的比例。
- 意義：
  - 反映該 layer input 是否呈現明顯稀疏性。
- 在本研究中的用途：
  - 若 FT 使 `zero_ratio` 上升，可能表示 activation quant 讓更多小值被推到 0 附近，分布更稀疏。
  - 若幾乎不變，表示 quant 對 sparsity 沒有明顯影響。
- 注意：
  - 對某些 activation 分布來說，`zero_ratio = 0` 並不奇怪，尤其在沒有 ReLU hard-threshold 的情況下。
  - 因此它是輔助指標，不一定是主導指標。

---

#### `pos_ratio`
- 定義：activation 中大於 0 的比例。
- 意義：
  - 反映分布正值區的佔比。
- 在本研究中的用途：
  - 用來看 quant 是否改變正負值平衡。
  - 若 `pos_ratio` 明顯上升，代表 FT 可能讓分布往正方向偏。
  - 若幾乎維持 0.5，表示正負平衡大致不變。

---

#### `neg_ratio`
- 定義：activation 中小於 0 的比例。
- 意義：
  - 與 `pos_ratio` 互補，反映負值區比例。
- 在本研究中的用途：
  - 可和 `pos_ratio` 一起解讀分布偏斜方向。
- 注意：
  - `pos_ratio + neg_ratio + zero_ratio ≈ 1`

---

### 4. 高階分布形狀指標

#### `kurtosis`
- 定義：峰度（kurtosis），用來描述分布尾巴厚度與尖峰程度。
- 意義：
  - `kurtosis` 高：表示分布具有較重尾（heavy tail），或者大量值集中在中心、但伴隨少數極端值。
  - `kurtosis` 低：表示分布較平滑、尾巴較輕。
- 在本研究中的用途：
  - 用來判斷 activation quant 是否改變 heavy-tail 性質。
  - 若 FT 的 `kurtosis` 顯著增加，可能表示：
    - 大多數值更集中，但少數極端值更突出
    - 或 tail 結構變得更重
  - 若顯著降低，則可能表示 tail 被削弱。
- 注意：
  - `kurtosis` 很容易受 tail 影響，因此應搭配 `q99/q999/abs_max` 一起解讀，而不要單獨下結論。

---

#### `skewness`
- 定義：偏度（skewness），用來描述分布是否向某一側偏斜。
- 意義：
  - `skewness > 0`：右偏，正向 tail 較長或較重
  - `skewness < 0`：左偏，負向 tail 較長或較重
  - 接近 0：較對稱
- 在本研究中的用途：
  - 若 FT 下 `skewness` 增大，表示高值正向 tail 可能變得更突出。
  - 若變小或轉負，可能表示分布型態往另一側偏移。
- 為什麼重要：
  - 若 FID 改善與某些 layer 的高值 tail 重分配有關，`skewness` 可以幫助辨識這種「偏向哪一側發生改變」。

---

### 5. 分位數估計樣本量

#### `sample_numel_for_quantiles`
- 定義：實際用來估計 quantiles 的元素數量。
- 意義：
  - 有些分析流程為了節省記憶體或加速，quantile 可能不是對全部元素直接計算，而是用抽樣後的元素估計。
- 在本研究中的用途：
  - 用來確認 quantile 統計是建立在多少樣本之上。
  - 若這個數值偏小，則 `q999` 這類極端 quantile 的穩定性也會相對下降。
- 注意：
  - 在解讀 `q999` 時，應同時考慮 `sample_numel_for_quantiles` 是否足夠大。

---

## 在本研究中，哪些指標最重要？

雖然程式輸出了很多統計量，但在本研究脈絡下，較關鍵的指標可分為三類：

### A. 整體尺度變化
- `std`
- `abs_mean`
- `abs_max`

用來回答：

> activation quant 是否改變了該 layer input 的整體幅度與波動範圍？

---

### B. Tail 行為變化
- `q99`
- `q999`
- `kurtosis`
- `skewness`

用來回答：

> activation quant 是否改變了高值 tail / heavy-tail / 偏斜結構？

這一類指標特別重要，因為目前結果顯示：

> FID 改善不太像是全域平均壓縮，而更可能是某些局部 layer 的 tail 結構被改變。

---

### C. 分布符號與稀疏性
- `zero_ratio`
- `pos_ratio`
- `neg_ratio`

用來回答：

> activation quant 是否改變了 activation 的符號平衡或稀疏程度？

這類指標通常不是主結論來源，但能幫助補充：
- 是否有更多值被壓到 0
- 是否分布偏向正值或負值

---

## FF vs FT 比較時的解讀方式

本分析不只看單一模式的統計值，更重要的是比較：

- `delta = FT - FF`
- `ratio = FT / FF`

例如：

- `std_ratio > 1`
  - FT 下該 layer 輸入更分散
- `q999_ratio > 1`
  - FT 下極高 tail 更重
- `abs_max_ratio < 1`
  - FT 下極端值被抑制
- `zero_ratio_delta > 0`
  - FT 下 activation 更稀疏

因此，最終重點不在於某個 layer 的 `q999` 是否大，而在於：

> **同一個 layer 在 FT 相對 FF 是否發生一致、可解釋的統計變化。**

---

## 解讀時的注意事項

1. **不要只看單一指標**
   - 例如 `q999` 增加，不代表整個分布都變大。
   - 必須搭配 `std`、`abs_max`、`kurtosis`、`skewness` 一起看。

2. **不要只看單一 layer**
   - 本研究更重視：
     - 哪一類 layer 變化較明顯
     - 哪些 block 集中出現變化
     - 這些變化是否有結構性

3. **不要把 activation 統計直接等同於 FID 原因**
   - activation analysis 的定位是：
     - 找出 quant effect 作用在哪裡
   - 真正要連到 FID，還需要後續的：
     - `pred_xstart`
     - trajectory
     - timestep-level 行為分析

---

## 本段總結
`activation_distribution_analysis.py` 所輸出的這些統計量，並不是單純為了描述某層 activation 的數值大小，而是為了從不同角度刻畫：

- 分布中心是否偏移
- 分布整體尺度是否改變
- 高值 / 低值 tail 是否重分配
- 稀疏性與正負平衡是否改變
- 這些變化是否集中在特定 layer / block / timestep

其中，在本研究中最重要的觀察重點是：

> **Conv2d layer input 的 `std / q99 / q999 / abs_max / kurtosis / skewness` 是否在 FT 相對 FF 下發生局部而一致的改變。**

因為這些局部統計結構的變化，才最有可能是後續影響 `pred_xstart` trajectory 與最終 FID 的來源。

---

## 主要觀察結果

### 1. 整體影響幅度不大，activation quant 並沒有均勻地改變整個網路
從 142 層的整體統計來看：

- 主要標籤為 **`minimal_change`**：**99 / 142 層**
- **`mixed_effect`**：**43 / 142 層**

這代表：

> activation quant 並不是對整個網路造成強烈、全面性的擾動，  
> 而是只影響到其中一部分 layer。

也就是說，目前看到的 effect 是 **局部性的（localized）**，不是全域性的（global）。

---

### 2. Linear layers 幾乎沒有變化
Linear layer 的輸入分布在 FF 與 FT 之間幾乎一致：

- mean `std_ratio`：**約 1.000**
- mean `abs_max_ratio`：**約 1.000**
- mean `q999_ratio`：**約 1.008**
- `zero_ratio_delta`：**0.0**

這代表：

> 目前 activation quant 的主要 effect **不是來自 embedding / conditioning 的 Linear input 改變**。

因此，就目前證據來看：

> **Linear layers 並不是解釋 FID 改善的主要來源。**

---

### 3. 主要差異集中在 Conv2d layers，尤其部分 `out_layers.3`
真正有明顯變化的是 Conv2d layers：

- mean `std_ratio`：**1.022**
- mean `abs_max_ratio`：**1.022**
- mean `q999_ratio`：**1.039**
- 約 **26.4%** 的 Conv2d layer 具有 mean `q999_ratio > 1.05`

變化最明顯的 layer 主要集中在少數 decoder / output-side Conv 路徑，例如：

- `model.output_blocks.12.0.out_layers.3`
- `model.output_blocks.2.1.out_layers.3`
- `model.output_blocks.2.0.out_layers.3`
- `model.output_blocks.3.0.out_layers.3`
- `model.output_blocks.5.2.out_layers.3`

這代表：

> activation quant 的主要 effect 並不是平均分佈在所有卷積層，  
> 而是集中在某些特定 Conv2d layer，尤其是 decoder / output-side 的 `out_layers.3`。

---

### 4. 變化最集中的 block 多數位於 `output_blocks`
以 mean `q999_ratio` 來看，變化最明顯的 block 為：

1. `model.output_blocks.12`
2. `model.output_blocks.2`
3. `model.output_blocks.3`
4. `model.output_blocks.5`
5. `model.output_blocks.9`

這代表：

> activation quant 的 effect 並不是以 embedding path 為中心，  
> 而是更偏向 **decoder / output-side 的 feature transformation**。

也就是說，FID 的改善若真的來自 activation quant，其影響更可能是透過：

> **後段特徵轉換路徑的變化**

而不是 time embedding 或 early input path。

---

### 5. 目前結果**不支持**「單純 tail compression」這種簡化說法
一個重要現象是：

在變化最明顯的 Conv layers 中，通常都有：

- `q999_ratio > 1`
- `std_ratio > 1`
- 很多情況下 `abs_max_ratio > 1`

也就是說，在 FT 下，這些 layer 的輸入分布往往呈現：

> **高尾部分稍大、variance 稍大，並不是單純被壓縮。**

因此，目前 activation analysis **不支持** 這種過度簡化的敘事：

> 「activation quant 讓 FID 變好，是因為它把 activation tail 全面壓小了」

更準確的說法應該是：

> activation quant **大多數 layer 幾乎不變**，  
> 但會對一部分 Conv pathway（尤其 decoder-side 的 `out_layers.3`）造成 **局部性的 feature statistics 重分配（redistribution）**。

---

### 6. 兩個非 LoRA 的 `QuantModule` 並不是主要來源
這次分析中有兩個特殊的 non-LoRA wrapper layer：

- `model.input_blocks.0.0`
- `model.out.2`

它們的變化幅度都不大：

- `model.input_blocks.0.0`：`q999_ratio ≈ 1.001`
- `model.out.2`：`q999_ratio ≈ 1.024`

這表示：

> 這兩個特殊 layer 並不是目前 activation effect 的主要來源。

主要訊號仍然來自：

> **網路主體中的 LoRA-related logical Conv layers**

---

## 整體解讀
這一輪 activation analysis 已經提供了一個清楚的第一層圖像：

1. 先前消融中觀察到的 FID 改善訊號，主要確實與 **activation quant** 有關。
2. 但從 activation input 分布來看，這種 effect **不是全域性的**。
3. 真正明顯的改變集中在一部分 **Conv2d layers**，尤其是 decoder / output-side 的 `out_layers.3`。
4. **Linear layers 幾乎不變**，因此不太可能是主要原因。
5. 目前證據 **不支持**「整體 activation tail 被壓縮」這種單一敘事；比較合理的解釋是：
   - activation quant 對少數關鍵 Conv pathway 造成了 **局部的 feature distribution 重分配**

---

## 這個階段可以回答什麼、不能回答什麼

### 目前可以回答的問題
- 真正有變化的是哪一類 layer：**Conv2d > Linear**
- 變化集中在哪裡：**decoder / output-side blocks**
- 這個 effect 是 **局部性的**，不是整個網路一起變
- 目前不適合用「uniform clipping / compression」來描述這個現象

---

## 本階段限制
- 本次樣本數為 **32**，因此這一輪更適合被視為：
  - **機制定位（mechanism localization）**
  - 而不是最終統計結論
- 目前 ratio 是建立在 activation summary statistics 上，而不是直接的 feature semantics
- 若要建立更強的「FID 改善因果敘事」，仍需要 trajectory-level 的證據

---

## 本階段總結
這一輪 activation analysis 的核心結論是：

> activation quant 帶來的差異，並不是均勻地改變整個網路，而是局部地作用在一部分 Conv2d pathway，尤其是 decoder / output-side 的 `out_layers.3`。  
> 由於這類 layer-level evidence 仍不足以直接解釋最終生成分布為何改善，下一步應進入 **Pred-xstart / Trajectory Analysis**，檢查這些局部差異是否真的轉化成較佳的 denoising trajectory 與最終生成行為。