# Cache pipeline：時間軸語意 Audit（Stage 0 / Stage 1 vs DDIM）

**日期**：依 repo 現況整理，供 Stage 2 與文件對齊用。

## 1. 結論摘要

| 問題 | 結論 |
|------|------|
| `.npz` 的 `l1_rate_step_mean[j]` 是 analysis axis interval 還是 DDIM 的「step j→j+1」當成 t_ddim？ | **等同 analysis axis 的 interval j**（見 §2）。**不是**「DDIM 張量 t=j 與 t=j+1」那種以 t_ddim 編號的 j。 |
| Stage 0 / Stage 1 數值是否一致？ | **一致**：Stage 0 直接堆疊 npz 欄位；Stage 1 對 `(B,99)` 做聚合，**未**做 `99-x` 重排。 |
| 是否需要對陣列做數值翻轉？ | **不需要**（在「analysis axis = similarity 的 step_counter 索引」此一定義下）。若誤把 zone 標籤當成 **t_ddim**，才會覺得「反了」——那是命名問題，不是現有陣列要重排。 |

## 2. `.npz` 欄位語意（源頭：`similarity_calculation.py`）

- Hook 以 **`_step_counter` ∈ {0,…,99}** 存 `outputs[k]`，且 **`current_step_idx = max_timesteps - 1 - _step_counter`**（見 `_create_step_pre_hook`）。
- 因此 **`outputs[0]`** 對應 **DDIM 進模型的 t_ddim = 99**（採樣鏈第一步），**`outputs[99]`** 對應 **t_ddim = 0**。
- **Step-change** 累加使用 `for s in range(1, T): t_prev=outputs[s-1], t_curr=outputs[s]`，寫入 **`l1_rate` 的 index `s-1`**。
- 故 **`l1_rate_step_mean[j]`** = **step_counter j → j+1** 的相對變化 = **DDIM 上由 t_ddim (99−j) 走到 (98−j)**。

與專案約定對齊：

- **analysis axis 點索引** `axis_idx ∈ [0,99]`：**與 step_counter / `outputs` 下標一致**；**t_ddim = 99 − axis_idx**。
- **interval j**（長度 99）：**axis j 與 axis j+1 之間**，對應 **t_ddim：(99−j) → (98−j)**。與 **`l1_rate_step_mean[j]`** 同一欄位語意。

## 3. Stage 0（`stage0e_normalization.py`）

- 讀入的 `L1_interval` 等 **不重新排序**；與 npz **欄位順序一致**。
- 舊版 README 若寫「step 50→51」而未註明是 **step_counter** 或 **t_ddim**，易誤解成 **DDIM t=50→51**。正確對照應寫：**interval j** 或 **step_counter j→j+1**，並可註 **t_ddim (99−j)→(98−j)**。

## 4. Stage 1（`stage1_scheduler.py`）

- `d_norm`、`S_*` 之 **第 j 欄** = Stage 0 **第 j 欄** = npz **interval j** 語意。
- `D_global`、`D_smooth`、`Delta` 長度 99，對應 **interval 軸**。
- **Zones** 的邊界在實作裡是 **0..99 的整數區間**，與 **analysis axis 上的「點」索引**一致（與 `outputs[k]` 的 k、與 **mask[k]** 的 k 一致）；**不是**未經轉換的 DDIM `t` 張量值當索引。

## 5. 誤導來源（僅命名／文件，非數值反轉）

- 將 **「timestep」** 一詞同時用在 **DDIM t_ddim** 與 **圖軸 / step_counter**，未加前綴。
- `t_order: "0_to_99"` 未說明是 **analysis axis**。
- Stage 0 README 範例若暗示 **「t=50」等於 DDIM 的 50**，與本 audit 不符。

## 6. Stage 2 接線提醒

- 讀取 `scheduler_config` 的 zone／mask 時，索引 **k** 為 **analysis axis_idx**；若 API 需要 **DDIM 的 tensor timestep**：**t_ddim = 99 − k**（T=100）。
