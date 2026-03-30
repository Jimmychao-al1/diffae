# FID Analysis

## 分析目的

本文件用於固定目前 baseline / quant ablation 的 FID 結果，並釐清以下問題：

> 為何在目前的 Q-DiffAE / LoRA + Quant pipeline 中，某些量化設定的 FID 反而比 float control 更好？

本文件的角色是：

1. 固定目前已知數值結果
2. 明確定義各設定的意義
3. 做出第一層因果判讀
4. 作為後續 activation / trajectory / generation distribution 分析的起點

---

## 重要前提

本次比較中，各設定皆使用相同的 effective weight：

- **所有設定皆使用 `w + lora`**

因此本次 ablation 不是在比較「有沒有 LoRA」，而是在比較：

- 同一個 `w + lora` 模型
- 在不同 quant path 下進行推論
- 對 FID 的影響為何

這點非常重要，因為它使本次實驗可以更直接回答：

> FID 的改善，究竟是來自 LoRA/adaptation，還是 quantization path 本身？

---

## 設定定義

### `FF`
- weight quant: off
- activation quant: off

意義：
- 可視為目前 `w + lora` 模型的 **float control**

---

### `tf`
- weight quant: on
- activation quant: off

意義：
- 觀察 **weight quant 單獨開啟** 時的影響

---

### `ft`
- weight quant: off
- activation quant: on

意義：
- 觀察 **activation quant 單獨開啟** 時的影響

---

### `tt`
- weight quant: on
- activation quant: on

意義：
- 觀察 **weight + activation quant 同時開啟** 時的影響

---

## FID 結果

### T = 20

- `FF` = 21.01409339904785
- `tf` = 20.515295028686523
- `ft` = 19.3142147064209
- `tt` = 19.86904716491

---

### T = 100

- `FF` = 17.103330612182617
- `tf` = 17.05845069885254
- `ft` = 14.580471992492676
- `tt` = 14.947220802307129

---

## 第一層觀察

## 1. `FF` 是目前最重要的 float control

因為所有設定都固定使用 `w + lora`，所以 `FF` 不是原始 baseline `w`，而是：

- 同一個 `w + lora` 模型
- 關掉 weight quant 與 activation quant
- 以 float path 進行推論

因此 `FF` 可以視為：

> 目前訓練後模型在不啟用 quant path 時的控制組

---

## 2. weight quant 單獨開啟時，影響很小

### T=20
- `FF` → `tf` 改善約 0.50

### T=100
- `FF` → `tf` 改善約 0.045

尤其在 T=100 下，`tf` 幾乎等同於 `FF`。

這表示：

> **weight quantization 單獨作用時，對 FID 幾乎沒有明顯影響。**

---

## 3. activation quant 單獨開啟時，改善最明顯

### T=20
- `FF` → `ft` 改善約 1.70

### T=100
- `FF` → `ft` 改善約 2.52

在兩個步數設定下，`ft` 都是四組中最好的結果。

這表示：

> **目前觀察到的 FID 改善，主要與 activation quantization 有關。**

---

## 4. `tt` 雖然仍有改善，但通常略差於 `ft`

### T=20
- `tt` 優於 `FF`
- 但略差於 `ft`

### T=100
- `tt` 優於 `FF`
- 但仍差於 `ft`

這表示：

> 當 activation quant 已帶來主要增益時，額外加入 weight quant 並沒有進一步改善，甚至可能吃掉部分增益。

---

## 第一層中間結論

根據目前 ablation，可得到以下中間結論：

### 結論 1
目前的 FID 改善，**不是單純由 LoRA 本身造成**。

原因是：
- 四組設定都固定使用相同的 `w + lora`
- 真正造成明顯差異的是 quant path 是否開啟，尤其是 activation quant path

---

### 結論 2
目前的 FID 改善，**不是由 weight quantization 主導**。

原因是：
- `tf` 與 `FF` 幾乎相同，尤其在 `T=100` 下差異極小

---

### 結論 3
目前的 FID 改善，**主要來自 activation quantization path**。

原因是：
- `ft` 在 `T=20` 與 `T=100` 下皆為最佳
- `tt` 雖然也有改善，但通常略差於 `ft`

---

## 目前最合理的第一層解釋

目前最合理的方向有兩種，後續需靠進一步分析驗證：

### 假說 A：train-test path consistency
若模型在訓練 / QAT 過程中已適應 activation quant path，
則在推論時關掉 activation quant，反而可能造成內部特徵分布 mismatch。

換句話說：

- `FF` 較差，不代表 float path 本身一定較差
- 而可能代表：
  - 目前的 `w + lora` 模型
  - 已經更適合在 activation quant 開啟的路徑下推論

---

### 假說 B：activation quant 帶來穩定化 / 正則化效果
activation quant 可能透過以下方式改變內部特徵：

- 壓縮動態範圍
- 抑制極端值
- 降低 heavy-tail 現象
- 改變 variance
- 使某些 timestep / block 的 activation 更穩定

若這些變化有助於 diffusion trajectory 更穩定，
則有可能進一步改善最終生成分布，並使 FID 下降。

---

## 後續分析主軸

根據目前結果，後續不應平均分析所有設定，
而應優先聚焦在：

- **`FF` vs `ft`**
- **`T = 100`**

原因：

1. `ft` 是目前最佳設定
2. 主要效應來自 activation quant
3. `T=100` 的改善更明顯
4. `T=100` 也更接近原本訓練設定

---

## 後續分析優先順序

### Step 1：Activation Distribution Analysis
目標：
- 分析 `FF` 與 `ft` 在各 block / timestep 的 activation 分布差異
- 驗證 activation quant 是否造成：
  - tail 壓縮
  - 極值抑制
  - variance 穩定化
  - 某些關鍵層的分布修正

---

### Step 2：Pred-xstart / Trajectory Analysis
目標：
- 檢查 activation 分布的差異，是否進一步影響 diffusion trajectory
- 分析 `pred_xstart` 或中間特徵統計

---

### Step 3：Generation Metrics Analysis
目標：
- 用 FID 以外的指標，分析最終生成分布
- 補充判斷 FID 改善來自：
  - precision 提升
  - recall 提升
  - 或更保守的生成模式

---

## 目前版本摘要

### 已確認事實
- 所有設定皆使用 `w + lora`
- `FF` 是 float control
- `tf` 幾乎沒有影響
- `ft` 改善最大
- `tt` 仍有效，但略差於 `ft`

### 目前最重要結論
> **FID 改善的主要來源是 activation quantization，而不是 weight quantization。**

### 後續最優先任務
> **進行 `FF vs ft`, `T=100` 的 activation distribution analysis。**

---

## 備註

本文件的角色不是直接給出最終機制結論，
而是固定目前已知的數值結果與第一層判讀。

真正要回答「為何 FID 改善」，
仍需配合後續的 activation / trajectory / generation distribution 分析。