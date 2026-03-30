# FID Analysis

## 分析目的

本文件用於固定目前 baseline / quant ablation 的 FID 結果，回答這個核心問題：

> 在同一個 `w + lora` 權重下，為何某些 quant path 的 FID 反而比 float control 更好？

本文件定位：
- 固定目前可重現的數值
- 統一定義比較設定
- 給出第一層判讀（observation + hypothesis）
- 作為後續 activation / trajectory / generation metrics 分析的起點

---

## Experiment Protocol（固定實驗條件）

> 這段只記錄目前結果對應的條件；若條件改變，需在此處新增新小節，不覆蓋舊資料。

- effective weight：所有設定都固定為 **`w + lora`**
- 比較維度：只切換 `set_quant_state(weight_quant, act_quant)`
- metric：FID（數值來源見 `fid_results/fid_results.jsonl`）
- 主要觀察：`T=20`、`T=100`
- 命名規則（統一）：`FF/TF/FT/TT`
  - `FF` = (False, False)
  - `TF` = (True,  False)
  - `FT` = (False, True)
  - `TT` = (True,  True)

---

## 設定定義

### `FF`
- weight quant: off
- activation quant: off
- 意義：`w + lora` 的 float control

### `TF`
- weight quant: on
- activation quant: off
- 意義：觀察 weight quant 單獨效果

### `FT`
- weight quant: off
- activation quant: on
- 意義：觀察 activation quant 單獨效果

### `TT`
- weight quant: on
- activation quant: on
- 意義：觀察 weight + activation quant 同時作用

---

## FID 結果（目前記錄）

### T = 20
- `FF` = 21.01409339904785
- `TF` = 20.515295028686523
- `FT` = 19.3142147064209
- `TT` = 19.86904716491

### T = 100
- `FF` = 17.103330612182617
- `TF` = 17.05845069885254
- `FT` = 14.580471992492676
- `TT` = 14.947220802307129

---

## 第一層觀察（Observation）

1. `FF` 是目前最重要的控制組  
   因為四組都固定同一個 `w + lora`，`FF` 代表「同一模型關掉 quant path」的基準。

2. `TF` 幾乎貼近 `FF`（尤其在 `T=100`）  
   表示 weight quant 單獨作用下，FID 變化很小。

3. `FT` 在 `T=20` 與 `T=100` 都是最佳  
   目前可觀察到的主要改善訊號集中在 activation quant path。

4. `TT` 雖優於 `FF`，但通常略差於 `FT`  
   代表 activation quant 的正向效應可能被部分 weight quant 效應抵銷。

---

## 中間結論（當前條件下）

> 以下結論限定在「目前記錄條件」下成立，非最終機制定論。

- 結論 A：目前改善訊號不是「是否有 LoRA」造成  
  因為四組都固定使用相同 `w + lora`。
- 結論 B：目前改善訊號不是由 weight quant 主導  
  因為 `TF` 與 `FF` 差異極小。
- 結論 C：目前最強訊號來自 activation quant path  
  因為 `FT` 在兩個 T 都最佳。

---

## 合理解釋（Hypothesis）

### Hypothesis A：train-test path consistency
模型在 QAT 過程已適應 activation quant path，若推論時關掉 activation quant（`FF`），可能造成內部分布 mismatch。

### Hypothesis B：activation quant 的穩定化效果
activation quant 可能抑制 tail / 極端值、改變 variance，進而讓 diffusion trajectory 更穩，最後反映在 FID。

---

## 接下來要驗證什麼

優先主軸：
- 比較對象：`FF` vs `FT`
- 主要設定：`T=100`

優先順序：
1. Activation Distribution Analysis（已建立第一版程式）
2. Pred-xstart / Trajectory Analysis
3. Generation Metrics（precision/recall/diversity）

---

## 備註

本文件是「目前結果的固定點」，不是最終因果證明。  
真正回答「為何 FID 改善」，需要把 activation / trajectory / generation-level 證據串起來。