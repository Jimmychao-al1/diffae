# Overall Results 說明

## 資料夾：`weight_hist/overall_normalized/`

| 檔案名稱 | 用處 |
|---|---|
| `overall_w_norm_vs_q_norm.png` | 所有 140 層 normalized weight 的整體分布。pre-quant 用填色 histogram，post-quant 用 PMF stem plot。用來確認量化 level 的分配是否合理覆蓋原始分布。**適合投影片。** |
| `overall_residual_norm.png` | 所有層的 LSB 殘差（normalized）分布，疊加理論均勻分布 U(-0.5, 0.5) 作為基準。用來驗證 rounding quantization 的殘差行為是否正常。 |
| `overall_ecdf_norm.png` | 所有層合併後的 ECDF overlay，標示 KS distance（= 0.0086）。是全模型量化失真的單一數字摘要，學術說服力強。**適合論文。** |
| `overall_level_occupancy.png` | 140 × 255 的 heatmap，顯示每個 layer 在各量化 level k ∈ [-127, 127] 的使用頻率（log scale）。可直接觀察是否有 layer 發生 saturation（k=±127 集中）或 level 浪費。**適合論文。** |
| `overall_level_pmf.png` | 所有層合計的 aggregate PMF，顯示全模型 255 個 level 的整體使用分布。屬於探索性分析，資訊量不如 heatmap，通常不需放入論文或投影片。 |
| `overall_normalized_stats.json` | 上述分析的數值摘要（mean、std、rel_L2 等），供程式讀取或手動查閱。 |

## 資料夾：`weight_hist/overall_raw/`

| 檔案名稱 | 用處 |
|---|---|
| `overall_w_lora_vs_q.png` | 所有層 raw weight（未 normalize）的整體分布 overlay。注意：不同層的 dynamic range 差異很大，此圖為探索性用途，不適合跨層比較。 |
| `overall_residual_raw.png` | 所有層 raw 殘差分布。同上，混合了不同 scale 的層，僅供探索性參考。 |
| `overall_raw_stats.json` | Raw 分析的數值摘要。 |

---

**使用建議**

- 論文：`overall_ecdf_norm.png` + `overall_level_occupancy.png`
- 投影片：`overall_w_norm_vs_q_norm.png`
- Raw 資料夾的結果：僅供內部探索，不建議對外呈現