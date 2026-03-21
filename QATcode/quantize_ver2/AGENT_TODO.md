# Quantize Ver2 Agent Todo

最後更新：2026-03-04  
目標：在不改既有 `lora_factor=700` 與主要 LR 設定前提下，找出可穩定降低 Step6 loss 的訓練策略。

## 0) 固定前提（不要動）

- `runtime_mode`: 維持目前版本
- `quant_state`: 維持 `True,False`（已驗證有效）
- `lora_factor`: 固定 `700`
- 其他 LR 設定：先不改

## 1) P0 / P1 實作理念

### P0：Timestep-weighted Distillation

- 定義：對不同 diffusion step 的蒸餾損失給不同權重（通常低噪聲步權重大）。
- 目的：把更新重點放在對最終品質較敏感的 timestep。
- 在本專案的注意事項：
  - Step6 是逐 step 訓練（每個 loop 一個 `t`），若用 batch mean normalization，權重會退化成 1。
  - 因此需使用「schedule-level normalization」避免單步退化。
- 判斷是否有效：
  - `Distill loss(raw)` 與 `Distill loss(weighted)` 不應長期完全重疊
  - `best_raw_loss` 與 `last10_mean_raw_loss` 需有可觀下降

### P1：Gradient Accumulation Across Timesteps

- 定義：累積多個 timestep 的梯度後再做一次 `optimizer.step()`（例如 `accum_steps=4`）。
- 目的：
  - 降低每 step 更新噪聲
  - 讓跨 timestep 的目標更接近同一參數點上的平均梯度
- 在本專案的注意事項：
  - 目前邏輯是每個 timestep 立即 `zero_grad/backward/step`，需重構更新節奏。
  - `lr_scheduler.step()` 與 `ema.update()` 建議跟 `optimizer.step()` 同步。
- 判斷是否有效：
  - `last10_std_raw_loss` 下降（曲線更穩）
  - `best_raw_loss` 或 `last10_mean_raw_loss` 有顯著改善

## 2) 目前基線（已完成）

- [x] `A0` Baseline（`True,False`, factor=700）
  - best raw loss（160 epoch）: `0.000535`
  - 作為後續所有實驗比較標準

- [x] `P0-only`（目前版）
  - 結果：與 baseline 幾乎同級，無顯著改善
  - 註記：P0 已修正為 schedule-level normalization，但單獨效果仍有限

## 3) 下一輪主實驗矩陣（60 epoch 篩選）

### 必跑

- [ ] `A1` P1 only（Gradient Accumulation）
  - 核心設定：`accum_steps=4`

- [ ] `A2` P1 + P0（目前 t-weight）
  - 用於驗證：P1 下 P0 是否開始有效

- [ ] `A3` P1 + SNR-weighted loss
  - 用於驗證：SNR 權重是否優於目前 P0

### 可選（前面有改善才跑）

- [ ] `A4` P1 + SNR-weighted + P0
  - 檢查 SNR 與 P0 是否互補

- [ ] `B1` P1 + EMA update only on optimizer.step
  - accumulation 後讓 EMA 更新節奏與 step 節奏一致

- [ ] `B2` P1 + clip_grad_norm 小掃描（`0.5` vs `1.0`）
  - 僅在 A1/A3 顯示正向時做

## 4) 評估規則（每組先跑 60 epoch）

每組統一記錄以下指標：

- `best_raw_loss@<=60`
- `last10_mean_raw_loss@60`
- `last10_std_raw_loss@60`
- `time_to_best_epoch`

淘汰規則（60 epoch）：

- 若相較 `A0` 改善 `< 2%`，直接淘汰，不跑滿 160
- 若改善 `>= 5%`，列為候選，進入 160 epoch 完整驗證

## 5) 決策門檻

- 目標 1（最低門檻）：`best_raw_loss < 0.000535`
- 目標 2（穩定門檻）：`last10_mean_raw_loss` 同時下降，且 `last10_std_raw_loss` 不惡化
- 若只出現單點更低、但均值與波動更差，判定為「不通過」

## 6) 執行命令模板（先保留，待 P1/SNR CLI 就位後填滿）

```bash
# A1: P1 only
python QATcode/quantize_ver2/quantize_diffae_step6_train_v2.py \
  --log-suffix a1_p1_only \
  --quant-diag-interval 4 \
  --quant-diag-topk 20

# A2: P1 + P0
python QATcode/quantize_ver2/quantize_diffae_step6_train_v2.py \
  --log-suffix a2_p1_p0 \
  --enable-t-weight \
  --t-weight-power 1.5 \
  --t-weight-min 0.2 \
  --t-weight-max 1.0 \
  --quant-diag-interval 4 \
  --quant-diag-topk 20

# A3: P1 + SNR-weighted
python QATcode/quantize_ver2/quantize_diffae_step6_train_v2.py \
  --log-suffix a3_p1_snr \
  --quant-diag-interval 4 \
  --quant-diag-topk 20
```

## 7) Agent 工作順序（照這順序執行）

1. [ ] 先實作 `A1` 所需程式（P1 only，最小改動）
2. [ ] 跑 `A1` 到 60 epoch，回報四個指標
3. [ ] 再實作/開啟 `A3`（SNR weighting）
4. [ ] 跑 `A3` 到 60 epoch，比較 A1/A3
5. [ ] 若 A3 有效，再補 `A2` / `A4` / `B1` / `B2`
6. [ ] 選 1~2 組最佳候選跑滿 160 epoch

---

備註：  
這份檔案是固定入口；之後只要打開 `QATcode/quantize_ver2/AGENT_TODO.md` 就能延續進度與決策依據。
