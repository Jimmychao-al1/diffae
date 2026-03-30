# Baseline Quant Analysis

## 目的

本資料夾用於分析以下研究問題：

> 為何在目前的 Q-DiffAE / LoRA + Quant pipeline 中，某些量化設定的 FID 反而比 float control 或原始 baseline 更好？

目前已知的關鍵現象不是「整體量化都能改善 FID」，而是更精確地：

- 在相同 `w + lora` 權重下，
- **activation quant path** 對 FID 改善最明顯，
- **weight quant path** 單獨影響很小，
- 因此後續分析主軸應聚焦在：
  - activation quant 如何改變內部特徵分布
  - 此變化是否進一步影響 diffusion trajectory
  - 最終是否使生成分布更接近真實分布

---

## 命名規則（統一）

- `FF`：weight quant off, act quant off
- `TF`：weight quant on,  act quant off
- `FT`：weight quant off, act quant on
- `TT`：weight quant on,  act quant on

註：本資料夾一律使用大寫命名；歷史檔案若有 `tf/ft/tt` 視為同義。

---

## 目前分析主線

本資料夾的分析分為三條主線：

### 1. Activation Distribution Analysis
目的：
- 比較 `FF` 與 `FT` 的 activation 分布差異
- 找出 activation quant 是否帶來：
  - 極值壓縮
  - heavy-tail 抑制
  - variance 穩定化
  - 某些 block / timestep 的特徵分布修正

核心問題：
> activation quant 到底改變了哪些內部數值特性，並可能因此改善 FID？

---

### 2. Pred-xstart / Trajectory Analysis
目的：
- 比較 `FF` 與 `FT` 在 diffusion 過程中的 `pred_xstart` / feature trajectory 差異
- 觀察 activation 分布的改變，是否真的反映到生成路徑上

核心問題：
> activation quant 所造成的內部分布變化，是否真的改變了 diffusion trajectory？

---

### 3. Generation Metrics Analysis
目的：
- 以 FID 之外的指標補充生成分布分析
- 例如 precision / recall / diversity 類指標

核心問題：
> FID 改善是因為生成品質更高、覆蓋更廣，還是模型更保守？

---

## 目前優先順序

目前最優先的是：

1. 固定並記錄現有 FID 消融結論
2. 進行 **Activation Distribution Analysis**
3. 再進行 Pred-xstart / Trajectory Analysis
4. 最後再補 Generation Metrics Analysis

目前第一優先分析設定為：

- **比較對象**：`FF` vs `FT`
- **步數**：`T = 100`
- **原因**：
  - `T=100` 的改善幅度更明顯
  - `FT` 是目前最佳設定
  - 現有消融已指出主因在 activation quant path，而非 weight quant

---

## 目前已知 FID 結論

詳細數值請見：

- `fid_results/fid_results.jsonl`
- `fid_results/FID_ANALYSIS.md`

目前已知重點如下：

- 所有設定皆使用相同 effective weight：`w + lora`
- `FF` 可視為 float control
- `TF`（只開 weight quant）改善極小
- `FT`（只開 activation quant）改善最大
- `TT`（weight + activation quant）仍有改善，但通常略差於 `FT`

因此目前的中間結論是：

> FID 改善的主要來源是 activation quantization，而非 LoRA 本身，也非 weight quantization 單獨作用。

---

## 資料夾規劃

### `fid_results/`
存放 FID 相關結果與文字分析。

- `fid_results.jsonl`
- `FID_ANALYSIS.md`

---

### `activation_results/`
存放 activation distribution 分析結果。

目前已落地第一版（`T_100/official/seed0`）：
- `FF/`：FF 模式單獨統計（json/npz/plots）
- `FT/`：FT 模式單獨統計（json/npz/plots）
- `FF_vs_FT/`：比較摘要與比較圖
- `selected_layers.json`

分析主程式：
- `activation_distribution_analysis.py`

分析摘要模板：
- `activation_results/ACTIVATION_ANALYSIS.md`

---

### `pred_xstart_results/`
存放 `pred_xstart` / trajectory 分析結果。

主程式：
- `pred_xstart_quantile_analysis.py`（Pred-xstart / Trajectory Analysis）

#### 目錄結構（每輪：`T_<T>/<images_mode>/seed<N>/`）

- `models/BASELINE|FF|FT|TT/`：單一模式統計
  - `pred_xstart_stats.json` / `.npz`
  - `trajectory_self_delta.json` / `.npz`（相鄰步 delta；邊界步無相鄰可比時為 NaN，見檔內 `boundary_nan_policy`）
  - `distance_to_final.json` / `.npz`（若 `--enable_distance_to_final`）
  - `trajectory_summary.json`：整合 self-delta 與 distance-to-final 的摘要
- `comparisons/<a>_vs_<b>/`：pairwise 比較（小寫，例如 `ff_vs_ft`、`baseline_vs_ff`）
  - `pred_xstart_compare.json` / `.npz`、`cross_model_same_t_delta.*`
  - `distance_to_final_compare.*`（若啟用 distance-to-final）
  - `baseline_vs_*` 另含 `divergence_to_baseline.json`（相對 BASELINE 的 L1/L2/cosine 曲線與 summary）
- `comparisons/baseline_divergence_summary.json`：FF/FT/TT 相對 BASELINE 的 divergence 摘要彙總
- `comparisons/high_noise_regime_summary.json`：各模型高噪聲區（t 上段三分之一）之 mean 指標與 pair 差分
- `comparisons/trajectory_regularization_summary.json`：各模式 `trajectory_summary.json` 彙總
- `plots/summary/`：跨模型摘要圖（baseline divergence overlay、high-noise 指標、trajectory regularization overlay）
- `plots/pairwise/`：各 pair 的 overlay / same-t / self-delta / distance 圖與 quantile band overlay
- `analysis_summary.md`：依統計自動產生的保守文字稿（可手改）

**向後相容**：執行後若 symlink 建立成功，run 根目錄下仍可有 `FF` → `models/FF`、`FF_vs_FT` → `comparisons/ff_vs_ft`（僅 Unix；失敗時請直接用新路徑）。

#### 輸出內容重點
- per-timestep 分布統計：`mean/std/min/max/abs_mean/abs_max/q001~q999`
- 邊界飽和比率：`|x|>=0.95`、`|x|>=0.99`
- same-t cross-model delta（任意已跑之 pair）：L1 / L2 / cosine
- self trajectory delta：相鄰 timestep；合法 NaN 已於 json 與 summary 中忽略
- distance-to-final（可選）：各 t 與最終 `pred_xstart`（t=0）之 L1 / L2 / cosine

#### 此分析能回答什麼 / 不能回答什麼
- **能**：量化模型與 BASELINE 或彼此在同一 noise level 的 pred_xstart 張量差異；高噪聲區分佈形狀（std、尾端、飽和、正負比）；軌跡平滑度與距離終點的相對比較。
- **不能**：直接證明「為何量化後比原作更好」的因果；不取代 FID/LPIPS/主觀視覺；pred_xstart 統計不等於權重或 activation 的完整故事。

#### 指標意義（簡述）
- `q50`：中心趨勢；`q99`：尾端；`abs_max`：極值；`std`：離散度
- same-t delta：兩模型在同一 t 的 pred_xstart 差異
- self trajectory delta：單一模型相鄰步變化（邊界步可為 NaN）
- distance-to-final：各步與最終 pred_xstart 的距離，用於看是否較早朝終點收斂
- high-noise regime：以 t 上段三分之一近似高噪聲區（與 `high_noise_regime_summary.json` meta 一致）

---

### `generation_metrics_results/`
存放除 FID 之外的生成分布分析結果。

建議內容：
- precision / recall
- diversity 類指標
- 補充性質的比較圖與分析摘要

---

## 命名原則

### 設定縮寫
- `FF`：weight quant off, act quant off
- `TF`：weight quant on, act quant off
- `FT`：weight quant off, act quant on
- `TT`：weight quant on, act quant on

### 模型比較主軸
目前主要使用：
- `comparisons/ff_vs_ft`（或舊 symlink `FF_vs_FT`）
- `T100`
- `official`
- `seed0`

若日後增加多 seed、多 checkpoint 或不同資料組，再往下擴充資料夾層級。

---

## 分析原則

1. **先固定現象，再分析原因**
   - 先用 md 寫清楚目前已知結論
   - 避免之後反覆依賴聊天歷史回顧

2. **先看主要效應，再做全域補充**
   - 既然目前主因明顯在 activation quant，就優先分析 activation
   - 不平均分配時間在所有設定上

3. **優先分析 `FF vs FT`**
   - 這組最能對照 activation quant 與軌跡差異（不等同於已解釋 FID 因果）

4. **先做 T=100，再考慮 T=20**
   - 因為 T=100 的現象更穩、更接近訓練設定

---

## 後續待完成事項

### A. FID 結論固定
- [x] 完整化 `fid_results/FID_ANALYSIS.md`
- [ ] 補 protocol 欄位（seed / eval_samples / command / commit）到每輪結果

### B. Activation 分析
- [x] 建立 activation 統計蒐集程式（`activation_distribution_analysis.py`）
- [x] 完成首輪 `FF vs FT`, `T=100`（含 `selected_layers.json`、各模式 json、比較 json、曲線圖）
- [ ] 決定最終固定的 layer 子集（目前可用 `--target-name-patterns`、`--max-layers`）
- [ ] 撰寫 `ACTIVATION_ANALYSIS.md` 的機制解讀版本

### C. Pred-xstart 分析
- [x] 整理 `pred_xstart_quantile_analysis.py` 輸出至 `models/`、`comparisons/`、`plots/summary`、`plots/pairwise`
- [ ] 比較 `FF vs FT`, `T=100`（依實驗跑官方結果）
- [x] 自動產生 `analysis_summary.md` 與各 summary json（仍可依實驗補手寫結論）

### D. 全域生成指標
- [ ] 規劃 precision / recall / diversity 類分析
- [ ] 補到最終分析鏈中

---

## 備註

本資料夾中的 md 檔案設計目標是：

- 讓未來可以直接從 repo 了解目前進度與中間結論
- 降低對聊天歷史的依賴
- 使後續論文撰寫、簡報整理、與教授討論時，能直接引用既有整理內容