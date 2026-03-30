# Pred-xstart trajectory analysis — auto draft

此檔為依統計自動產生的**保守**摘要，不宣稱訓練或量化之因果機制。

## 1. 本次分析驗證了什麼

- 在 T=100 的 DDIM progressive 軌跡上，對 pred_xstart 做 per-timestep 分佈與跨模型對齊度量。
- 若已產生 `divergence_to_baseline.json`，可讀取量化模型與 BASELINE 在同一 noise level 的張量差異（L1/L2/cosine）。
- 若已產生 `high_noise_regime_summary.json` 與 `trajectory_regularization_summary.json`，可比較高噪聲區段的標的統計與軌跡平滑度/距離終點。

## 2. 目前可支持的中間結論（僅陳述觀測）

- 請以數值為準：比較各 `summary` JSON 內 mean / zone mean，而非僅看圖。
- 若某模型在 high-noise zone 的 `saturation_ratio_abs_ge_099` 或 `std` 明顯較高，代表該區段 pred_xstart 分佈較分散或較常觸及飽和；**這不自動等於**生成品質較差。

## 3. 目前仍不能直接下的結論

- 「為何量化後比原作更好」涉及感知指標、資料與訓練目標；本管線**不**量測 FID/LPIPS 或重建誤差。
- pred_xstart 統計差異**不**等同於 latent 或權重空間的因果解釋。

## 4. 建議的下一步

- 將此處 high-noise / divergence 指標與實際樣本視覺與下游指標對照。
- 檢查：`high_noise_regime_summary.json`、`trajectory_regularization_summary.json`。

## 附註：self trajectory delta 的 NaN

- first_step_or_last_step_has_no_adjacent_comparison；summary 已忽略合法 NaN。

### Baseline divergence 摘要（mean L1 over t）

- ff: 0.11468002436682583
- ft: 0.058060541674494745
- tt: 0.05410272595472634
