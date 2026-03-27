# 第二版量化訓練（ver2）正式實驗計畫與代辦事項

---

## 1. 文件目的

本文件用於規劃 **第二版量化訓練（ver2）** 的正式實驗順序、待辦事項與觀察重點，方便依序執行與追蹤。

**原則**：

- 在**正式掃參前**，第二版的核心量化方法（quant flow、clipping、fake-quant 等）**先固定**，避免在 baseline 與掃參過程中混入「方法變動」與「超參數變動」兩種因素，導致結果無法歸因。
- 本文件以 **實驗執行順序** 與 **可勾選代辦** 為主，不作泛泛的介紹。

---

## 2. 第二版目前固定的方法定義

以下為目前 ver2 已採用且**本計畫期間不變更**的核心定義（僅作為實驗共識，細節以程式為準）：

| 項目 | 說明 |
|------|------|
| **有效權重** | `weight_eff = org_weight + lora_weight`（先合併 LoRA，再進入量化路徑） |
| **權重量化** | 對 `weight_eff` 做 **normalized fake-quant**（第二版正規化假量化流程） |
| **激活量化** | 使用 **per-timestep（per-step）的 `scale_list`**，依擴散步對應更新 |
| **Quant flow** | 第二版**不修改**整體 quant 流程（僅在固定方法下調超參數） |
| **Clipping** | 第二版**不修改** clipping 寫法（避免與掃參混淆） |
| **實驗策略** | **先固定程式方法**，再進行 **lora_factor / act_quant_lr** 等超參數掃描 |

---

## 3. 第二版訓練的正式實驗順序

### Phase 0：方法固定

**目的**：後續實驗結果只反映超參數與訓練動態，不混入「方法改版」。

- [ ] 不再更動 **quant flow**（含 forward 與 fake-quant 路徑）
- [ ] 不再更動 **clipping / fake-quant** 邏輯
- [ ] 不再更動 **EMA / loss** 主流程（除非本文件另有明確放行）
- [ ] 確認與訓練入口相關的 ver2 腳本（例如 `quantize_diffae_step6_train_v2.py`）**僅透過 CLI 調整已支援之超參**，不順手改核心

---

### Phase 1：建立 baseline

**第一組正式 baseline 數值**：

| 超參數 | 數值 |
|--------|------|
| `lora_factor` | **2500** |
| `act_quant_lr` | **5e-4** |

**意義**：

- 在第二版方法**定義不變**的前提下，此組為 **ver2 第一個正式基準點**。
- 後續所有掃參與比較，**一律以本 baseline 為參考**（loss 曲線、tail 指標、生成圖、scale_list 行為等）。

**執行提醒**：專案中已支援 `--lora-factor` 與 `--act-quant-lr` 時，應以此組數值啟動訓練並留存完整 log。

---

### Phase 2：只掃 `lora_factor`

**固定**：`act_quant_lr = 5e-4`（與 baseline 相同）

**建議實驗組**：

| 編號 | `lora_factor` |
|------|----------------|
| A | 1800 |
| B | 2500（baseline） |
| C | 3500 |

**為什麼先掃 `lora_factor`（第一優先）**：

- 第二版先合併 `org_weight + lora_weight` 再對 **weight_eff** 做 normalized fake-quant。
- **LoRA 的更新幅度**會直接改變「量化後權重」是否落在格點邊界附近、是否容易跨格點。
- 因此 **`lora_factor`（透過 `lora_lr = avg_weight_scale / lora_factor` 控制 LoRA 有效學習率）** 應列為**第一個**掃描軸，**固定** activation 側學習率後再比較。

---

### Phase 3：選定較佳 `lora_factor` 後，再掃 `act_quant_lr`

在 Phase 2 **選定一個較佳的 `lora_factor`** 後，**固定該值**，只掃 `act_quant_lr`：

**建議掃描值**：

| 值 |
|----|
| 2e-4 |
| 5e-4 |
| 1e-3 |

**為什麼 `act_quant_lr` 放在第二順位**：

- 先讓**權重側（LoRA + fake-quant）**的更新幅度與穩定性落在可解讀區間。
- 再微調 **activation 量化參數（`scale_list`）** 的學習速度，避免權重與激活兩側同時亂飄、難以歸因。

---

## 4. 每個階段要觀察的重點

建議每個實驗（含 baseline）都至少記錄與比對：

- **Training loss**：整體是否下降、是否異常抖動或發散。
- **Distill loss**：與 teacher 對齊的狀態是否一致惡化或僅局部問題。
- **Per-timestep mean loss**：是否集中在特定 timestep（尤其尾段）。
- **Tail ratio / tail outlier ratio**（若 log 或診斷有輸出）：尾段是否異常偏高。
- **驗證生成圖**：固定種子或固定驗證步驟下，視覺穩定性與品質。
- **Quant update diagnostics**（若有啟用量化診斷）：有效量化更新比例、層級是否長期「不更新」。
- **`scale_list` 是否過度震盪**：數值或梯度是否劇烈跳動。
- **LoRA 是否幾乎不更新 / 更新過大**：搭配 loss 與判讀規則（見第 5 節）調整 `lora_factor`。

---

## 5. `lora_factor` 的判讀規則

`lora_factor` 越大 → **每層 `lora_lr = avg_weight_scale / lora_factor` 越小** → LoRA 更新越**保守**。

### 若出現下列現象（代表 LoRA 可能「太激進」）

- Loss **很抖**、不易收斂
- **Tail ratio** 偏高
- **尾段 timestep loss** 偏高
- **生成圖不穩**、偽影或品質崩壞

**建議**：把 **`lora_factor` 調大**（例如 2500 → 3500 → 5000），降低 LoRA 有效學習率。

### 若出現下列現象（代表 LoRA 可能「太保守」）

- Loss **幾乎不動**、長期平臺
- **有效量化更新比例太低**（若診斷有輸出）
- **訓練曲線很平**但沒有實質改善

**建議**：把 **`lora_factor` 調小**（例如 2500 → 1800 → 1200），提高 LoRA 有效學習率。

---

## 6. `act_quant_lr` 的判讀規則

在固定 `lora_factor` 的前提下調整 `act_quant_lr`（`scale_list` 參數羣組的學習率）。

### 若出現下列現象（代表激活量化側可能「太激進」）

- **`scale_list` 波動太大**
- **Loss 震盪更嚴重**
- **Tail ratio 惡化**

**建議**：**降低** `act_quant_lr`（例如 5e-4 → 2e-4 → 1e-4）。

### 若出現下列現象（代表激活量化適應「太弱」）

- **`scale_list` 幾乎不動**
- 懷疑 activation 量化對訓練訊號反應不足

**建議**：**提高** `act_quant_lr`（例如 5e-4 → 7e-4 → 1e-3）。

---

## 7. 目前先不要動的項目

以下項目在**完成第二版正式 baseline 與超參數掃描前**，**不建議**一併修改：

| 項目 | 原因 |
|------|------|
| **Clipping 寫法** | 與 fake-quant 動態範圍直接相關，一改就難與舊實驗對照 |
| **Quant flow** | 核心路徑變動會使所有掃參失去可比性 |
| **Loss 定義** | 與 teacher / distill 對齊方式綁定，改動後無法單獨解讀超參 |
| **Tail repair 策略** | 屬額外訓練策略，易與 `lora_factor` / `act_quant_lr` 效果混淆 |
| **EMA 主流程** | 影響 checkpoint 與推理對齊，應與超參掃描分開 |
| **大幅修改 warmup** | 改變有效學習率排程，與「只改兩個超參」的目標衝突 |
| **其他會讓結果難以歸因的方法變動** | 同時動方法與超參數 → 無法判斷改善來源 |

**總結**：第二版現階段重點是 **完成正式 baseline 與 lora_factor / act_quant_lr 掃描**；**方法與訓練設定同時變動**會造成結果**無法解讀**，應避免。

---

## 8. 代辦事項清單（可打勾）

- [ ] 確認第二版程式方法已固定，不再修改 quant flow
- [ ] 跑 baseline：`lora_factor=2500`，`act_quant_lr=5e-4`
- [ ] 整理 baseline 的 training log、loss、tail ratio、生成圖
- [ ] 固定 `act_quant_lr=5e-4`，掃描 `lora_factor` = 1800 / 2500 / 3500
- [ ] 比較不同 `lora_factor` 的 loss 與生成品質
- [ ] 選定最佳 `lora_factor`
- [ ] 在最佳 `lora_factor` 下掃描 `act_quant_lr` = 2e-4 / 5e-4 / 1e-3
- [ ] 比較 `act_quant_lr` 對 `scale_list` 與尾段 loss 的影響
- [ ] 選定第二版正式訓練設定（並記錄於實驗筆記）
- [ ] 整理第二版結果，準備後續 FID 與 cache pipeline 實驗

---

## 9. 目前建議的第一個正式實驗設定

**建議第一個正式實驗（正式 baseline）**：

| 參數 | 值 |
|------|-----|
| `lora_factor` | **2500** |
| `act_quant_lr` | **5e-4** |

**說明**：此組為第二版在**方法固定**前提下，**最合理的起點**——先以該組作為**正式 baseline**，完成記錄與再現後，再依 **Phase 2 → Phase 3** 展開掃描，避免無基準比較。

---

*文件路徑：`QATcode/quantize_ver2/SECOND_VERSION_TRAINING_PLAN.md`*
