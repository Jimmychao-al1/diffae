# Q-DiffAE 訓練優化狀態（quantize_ver2 程式追蹤）

## 1. Current status（目前狀態）

- 本文件基於 `QATcode/quantize_ver2/` 現有程式與既有 log/產物進行「程式追蹤」整理。
- 目前定位：**以現況理解為主、暫不變更訓練邏輯與量化行為**。
- `QATcode/docs/qdiffae_training_optimization_status.md` 原先內容的有效觀察（如 layer 3 / layer 144 行為、loss 記錄解讀）已保留並移至後段「優化討論狀態」。

來源指向：
- 文檔位置：`QATcode/docs/qdiffae_training_optimization_status.md`
- 討論紀錄來源：`QATcode/quantize_ver2/AGENT_TODO.md`
- 既有執行證據：`QATcode/quantize_ver2/log/step4.log`、`QATcode/quantize_ver2/log/step5.log`、`QATcode/quantize_ver2/log/step6_train_*.log`

---

## 2. Purpose of `QATcode/quantize_ver2`

`quantize_ver2` 目標可從程式看成三段：

1. Step4：建立 Diff-AE UNet 的選擇性量化版本並做初始化校準。
2. Step6：在 LoRA + ver2 normalized fake-quant 架構下做蒸餾式 QAT 訓練。
3. Sample/評估：載入訓練後權重做推論評估與（可選）快取分析。

補充：Step5（整數轉換）程式存在，但從程式與 log 看起來不是目前穩定主線（見第 10 節）。

來源指向：
- Step4 入口腳本：`QATcode/quantize_ver2/quantize_diffae_step4_selective_v2.py`（`main`）
- Step5 入口腳本：`QATcode/quantize_ver2/convert_diffae_step5_intmodel_v2.py`（`main`）
- Step6 入口腳本：`QATcode/quantize_ver2/quantize_diffae_step6_train_v2.py`（`main`）
- 推論評估腳本：`QATcode/quantize_ver2/sample_lora_intmodel_v2.py`（`main_float_model`）

---

## 3. Directory / file role summary

來源指向：
- 核心量化模組：`QATcode/quantize_ver2/quant_layer_v2.py`
- 模型包裝（Step4）：`QATcode/quantize_ver2/quant_model_selective_v2.py`（`SelectiveQuantModel`）
- 模型包裝（Step6）：`QATcode/quantize_ver2/quant_model_lora_v2.py`（`QuantModel_DiffAE_LoRA`、`QuantModule_DiffAE_LoRA`）
- 訓練器：`QATcode/quantize_ver2/diffae_trainer_v2.py`（`SpacedDiffusionBeatGans_Trainer`）
- 校準資料集：`QATcode/quantize_ver2/quant_dataset_v2.py`（`DiffusionInputDataset`）

### 3.1 主要入口腳本
- `QATcode/quantize_ver2/quantize_diffae_step4_selective_v2.py`
- `QATcode/quantize_ver2/convert_diffae_step5_intmodel_v2.py`
- `QATcode/quantize_ver2/quantize_diffae_step6_train_v2.py`
- `QATcode/quantize_ver2/sample_lora_intmodel_v2.py`

### 3.2 核心模型與量化模組
- `QATcode/quantize_ver2/quant_layer_v2.py`
  - `normalized_fake_quant`
  - `UniformAffineQuantizer`
  - `TemporalActivationQuantizer`
  - `QuantModule`
- `QATcode/quantize_ver2/quant_model_selective_v2.py`
  - `SelectiveQuantModel`（Step4）
- `QATcode/quantize_ver2/quant_model_lora_v2.py`
  - `QuantModule_DiffAE_LoRA`
  - `QuantModel_DiffAE_LoRA`（Step6）
- `QATcode/quantize_ver2/diffae_trainer_v2.py`
  - `SpacedDiffusionBeatGans_Trainer`
  - `create_diffae_trainer`

### 3.3 資料與輔助
- `QATcode/quantize_ver2/quant_dataset_v2.py`（`DiffusionInputDataset`）
- `QATcode/quantize_ver2/globalvar.py`（全域暫存工具）
- `QATcode/quantize_ver2/quant_model_v2.py`（較舊的通用量化模型實作）
- `QATcode/quantize_ver2/downsample_talsq_ckpt.py`（checkpoint 下採樣工具）

### 3.4 參考/討論文件
- `QATcode/quantize_ver2/IMPLEMENTATION_SUMMARY.md`
- `QATcode/quantize_ver2/AGENT_TODO.md`

---

## 4. Entry scripts and execution flow

來源指向：
- Step4 腳本流程：`quantize_diffae_step4_selective_v2.py`（`main`、`create_quantized_model`、`calibrate_quantized_model`、`save_quantized_model`）
- Step5 腳本流程：`convert_diffae_step5_intmodel_v2.py`（`main`、`load_quantized_model`、`convert_to_integer_weights`）
- Step6 腳本流程：`quantize_diffae_step6_train_v2.py`（`main`、`load_diffae_model`、`create_quantized_model`、`setup_optimizer_with_dynamic_lr`）
- Sample 腳本流程：`sample_lora_intmodel_v2.py`（`main_float_model`）

### 4.1 Step4（選擇性量化初始化）
入口：`quantize_diffae_step4_selective_v2.py`

流程：
1. 載入 Diff-AE `last.ckpt`。
2. 載入 `calibration_diffae.pth`，取樣本。
3. 建立 `SelectiveQuantModel`，將 UNet 核心層替換成 `QuantModule`。
4. 設定首尾特殊層為 8-bit（首 3 層 + 最後 1 層）。
5. 量化開啟下做一次 forward（single-forward calibration）。
6. 輸出量化模型與 config。

### 4.2 Step5（轉整數模型）
入口：`convert_diffae_step5_intmodel_v2.py`

- 腳本保留「假量化權重轉整數」相關函式。
- 但主程式目前偏向直接存 `state_dict`；且既有 `step5.log` 出現 `delta is None` 相關錯誤紀錄。
- 因此目前主線更像是 Step4 產出的 `..._intmodel.pth` 被 Step6 直接載入（仍為 ver2 fake-quant 流程語境）。

### 4.3 Step6（QAT 訓練）
入口：`quantize_diffae_step6_train_v2.py`

流程：
1. 載入 FP Diff-AE（teacher）與量化學生模型骨架（LoRA 版）。
2. 載入 Step4 產物 `diffae_unet_quantw8a8_intmodel.pth`。
3. 設定可訓練參數（LoRA + `scale_list`）。
4. 建立 optimizer/scheduler。
5. 使用 `diffae_trainer_v2.py` 的蒸餾訓練器，跑 inference-style DDIM training loop。
6. 依 loss 儲存 checkpoint，定期做 EMA 驗證生成圖。

### 4.4 Sample/評估
入口：`sample_lora_intmodel_v2.py`

- 主要做推論/評估（FID）與可選 cache 分析。
- 不是 Step6 訓練主迴圈本體。

---

## 5. Implemented pipeline trace（主體）

> 本節僅描述「程式已實作」與可直接對應到程式/日誌的事實。

來源指向：
- Step4 追蹤主來源：`quantize_diffae_step4_selective_v2.py`
- Step6 追蹤主來源：`quantize_diffae_step6_train_v2.py`、`diffae_trainer_v2.py`
- 模型替換邏輯：`quant_model_selective_v2.py`、`quant_model_lora_v2.py`
- 日誌佐證：`log/step4.log`、`log/step6_train_*.log`

### 5.1 Step4：從 FP 模型到 selective quant checkpoint

1. 建立 `LitModel(conf)` 後讀取 `conf.logdir/last.ckpt`。
2. 校準資料來源為 `QATcode/quantize_ver2/calibration_diffae.pth`，以 `DiffusionInputDataset` 展平成 `(x_t, t, y)`。
3. `SelectiveQuantModel` 只量化 `time_embed / input_blocks / middle_block / output_blocks / out` 內的 `Conv2d/Linear`。
4. `set_first_last_layers()` 把前 3 層與最後 1 層設 8-bit 並 `ignore_reconstruction=True`。
5. calibration 階段為「量化開啟 + 單次 forward」。
6. 輸出：
   - `QATcode/quantize_ver2/diffae_unet_quantw8a8_selective.pth`
   - `QATcode/quantize_ver2/diffae_unet_quantw8a8_intmodel.pth`（Step6 相容命名）
   - `QATcode/quantize_ver2/diffae_unet_quantw8a8_selective_config.pth`

程式來源：
- 入口：`quantize_diffae_step4_selective_v2.py` `main`
- 模型包裝：`quant_model_selective_v2.py` `SelectiveQuantModel`
- 量化狀態設定：`quant_model_selective_v2.py` `SelectiveQuantModel.set_quant_state`
- 校準函式：`quantize_diffae_step4_selective_v2.py` `calibrate_quantized_model`
- 輸出函式：`quantize_diffae_step4_selective_v2.py` `save_quantized_model`

### 5.2 Step6：從 Step4 產物進入 LoRA+QAT 訓練

1. 載入 FP 模型（teacher）與其 deep copy（`fp_model`）作蒸餾對照。
2. 將學生模型替換為 `QuantModel_DiffAE_LoRA`：
   - 大部分層替換成 `QuantModule_DiffAE_LoRA`。
   - 特殊層（索引清單）保留 `QuantModule`。
3. 讀入 Step4 權重到量化學生模型。
4. 先用校準資料做一次 forward，初始化 temporal activation quantizer 狀態。
5. 凍結策略：只開 `name` 含 `lora` 或 `scale_list` 的參數為 trainable。
6. 訓練核心呼叫：
   - `distill_trainer.training_losses_with_inference_distillation(...)`
   - 內部是 DDIM 100 steps 的 inference-style 逐步蒸餾。
7. 每個 diffusion step 內部在 trainer 中執行：
   - teacher 前向（no_grad）
   - student 前向
   - distill loss（可加 timestep 權重）
   - `backward -> clip_grad_norm -> optimizer.step -> lr_scheduler.step -> EMA.update`
8. 每個 epoch 匯總平均 loss；若改善則存檔；每 8 epoch 做一次驗證生成並輸出圖網格。

程式來源：
- 入口：`quantize_diffae_step6_train_v2.py` `main`
- 建模：`quantize_diffae_step6_train_v2.py` `create_quantized_model`
- LoRA 量化模組：`quant_model_lora_v2.py` `QuantModel_DiffAE_LoRA`、`QuantModule_DiffAE_LoRA`
- 訓練呼叫：`quantize_diffae_step6_train_v2.py` `distill_trainer.training_losses_with_inference_distillation(...)`
- 蒸餾實作：`diffae_trainer_v2.py` `training_losses_with_inference_distillation`、`ddim_sample_with_training`、`p_mean_variance_with_distillation`
- 儲存：`quantize_diffae_step6_train_v2.py` `save_checkpoint`

### 5.3 驗證與可視化輸出行為

- 驗證採樣時會取 EMA 模型，並重新設定 quant 狀態與 runtime mode。
- 產生 FP 與 quant 結果拼圖，輸出到 `QATcode/quantize_ver2/training_samples/epoch_{k}_grid.png`。

程式來源：
- EMA 模型取得：`diffae_trainer_v2.py` `get_ema_model`
- 驗證採樣器：`diffae_trainer_v2.py` `SpacedDiffusionBeatGans_Sampler.sample`
- 驗證輸出圖：`quantize_diffae_step6_train_v2.py` `main`（epoch 驗證段）

---

## 6. Quantization-related forward path

來源指向：
- 核心量化算子：`quant_layer_v2.py` `normalized_fake_quant`
- 一般量化層前向：`quant_layer_v2.py` `QuantModule.forward`
- LoRA 量化層前向：`quant_model_lora_v2.py` `QuantModule_DiffAE_LoRA.forward`
- 時間步 activation 量化器：`quant_layer_v2.py` `TemporalActivationQuantizer.forward`

### 6.1 ver2 核心：normalized fake-quant
在 `quant_layer_v2.py`：

- `normalized_fake_quant(x, scale)`：
  - 以 absmax scale 正規化
  - 映射至 int8 格點（對稱範圍）
  - 回到 normalized 浮點域

### 6.2 `QuantModule` 前向（非 LoRA）
`QuantModule.forward` 依 `set_quant_state` 有 4 種路徑：

1. `weight=True, act=True`：x 與 w 都 fake-quant，之後以 `(a_x * a_w)` rescale，最後加 FP32 bias。  
2. `weight=True, act=False`：只量化權重（含 rescale）。  
3. `weight=False, act=True`：只量化 activation（含 rescale）。  
4. `weight=False, act=False`：走 `org_weight/org_bias` FP 路徑。

### 6.3 `TemporalActivationQuantizer`

- 每層持有 `scale_list`（每 timestep 一個可訓練參數）。
- `current_step` 預設從最後步開始，forward 後會自動遞減（循環）。

### 6.4 Step6 當前主要量化開關

- 入口腳本中主設定為 `quant_model.set_quant_state(True, False)`（註解標示 GPT test1）。
- 另有 `QuantModel_DiffAE_LoRA.set_quant_state()` 內對特殊層（如 layer 1/2/3/144）做覆寫策略。

程式來源：
- 腳本設定：`quantize_diffae_step6_train_v2.py` `main`
- 模型覆寫：`quant_model_lora_v2.py` `QuantModel_DiffAE_LoRA.set_quant_state`

---

## 7. LoRA / module interaction

來源指向：
- LoRA 層定義：`quant_model_lora_v2.py` `QuantModule_DiffAE_LoRA.__init__`
- LoRA 權重合成與前向：`quant_model_lora_v2.py` `QuantModule_DiffAE_LoRA.forward`
- 權重尺度策略：`quant_model_lora_v2.py` `_compute_a_w`、`_get_a_w`
- runtime mode 傳遞：`quant_model_lora_v2.py` `QuantModel_DiffAE_LoRA.set_runtime_mode`

### 7.1 LoRA 權重注入位置
在 `QuantModule_DiffAE_LoRA.forward`：

- Linear：以 `loraA/loraB` 生成低秩更新，與 `org_weight` 相加得 `weight_eff`。
- Conv2d：以對應張量運算生成 LoRA 權重後加到 `org_weight`。

### 7.2 LoRA 與量化交互

- `weight_eff` 才是後續量化或非量化分支的權重基礎。
- `a_w` 由 `weight_eff` 動態計算（可由 runtime mode 控制是否 cache）。

### 7.3 可訓練參數範圍（Step6）

- 實作上僅放行 `lora*` 與 `scale_list`。
- 其餘權重皆凍結。

---

## 8. Config / checkpoint / log / output locations

來源指向：
- 路徑常數：`quantize_diffae_step6_train_v2.py` `TrainingConfig`
- Step4 輸出路徑：`quantize_diffae_step4_selective_v2.py` `save_quantized_model`
- Step6 儲存路徑：`quantize_diffae_step6_train_v2.py` `save_checkpoint`
- log 目錄建立：`quantize_diffae_step4_selective_v2.py` `setup_log_file`、`convert_diffae_step5_intmodel_v2.py` `setup_log_file`、`quantize_diffae_step6_train_v2.py` `TrainingConfig.setup_environment`

### 8.1 主要 checkpoint/資料
- 基礎模型：`checkpoints/ffhq128_autoenc_latent/last.ckpt`
- 校準資料：`QATcode/quantize_ver2/calibration_diffae.pth`
- Step4 輸出：
  - `QATcode/quantize_ver2/diffae_unet_quantw8a8_selective.pth`
  - `QATcode/quantize_ver2/diffae_unet_quantw8a8_selective_config.pth`
  - `QATcode/quantize_ver2/diffae_unet_quantw8a8_intmodel.pth`

### 8.2 Step6 訓練輸出（目前程式行為）
- log：`QATcode/quantize_ver2/log/step6_train*.log`
- 測試權重（目前 `save_checkpoint()` 實作固定覆寫）：
  - `QATcode/quantize_ver2/test_ckpt/test.pth`
- 驗證圖：`QATcode/quantize_ver2/training_samples/epoch_*_grid.png`
- 量化診斷目錄：`QATcode/quantize_ver2/quant_diag_runs/`（目前工作樹中未見已落地 json/圖檔）

### 8.3 路徑命名差異（事實記錄）
- `TrainingConfig` 內有 `SAVE_BEST_PATH` / `SAVE_FINAL_PATH` 指向 `.../checkpoints/...`，
  但 `save_checkpoint()` 內最終又指定為 `TEST_PATH`。
- 因此當前實際保存位置以 `test_ckpt/test.pth` 為準。

---

## 9. Current optimization discussion status（獨立於已實作流程）

> 本節是「討論中/未定案」內容，不視為既有 pipeline 已落地事實。

來源指向：
- 討論議題主來源：`QATcode/quantize_ver2/AGENT_TODO.md`
- 討論相關旗標：`quantize_diffae_step6_train_v2.py` `_get_cli_args`（例如 `--enable-t-weight` 等）
- 討論結果觀察載體：`QATcode/quantize_ver2/log/step6_train_*.log`

### 9.1 既有討論觀察（保留原文件有效內容）

- 目前討論過的設定包含：`decay=0.0`、activation quant off、`a_w detach`。
- Layer 3 曾觀察偏好 `(True, False)`。
- Layer 144 曾觀察偏向 `(False, False)`。
- `(False, False)` 對 Layer 3 的改善量級曾被記錄為極小（約 `0.000002`）。
- loss 記錄解讀：每 20 step 的記錄是區間平均，尖峰區域有意義。
- 討論中曾指出 batch_size 顯示值與實際值可能不一致（需再核對對應實驗版本）。

### 9.2 目前未定案議題

1. 下一輪實驗優先順序。
2. 修改執行順序與 ablation 矩陣。
3. 成功判準（best / mean / std / time-to-best）。
4. 哪些參數固定、哪些持續掃描。
5. 是否新增實驗輔助腳本。

---

## 10. Known unclear points / TODO for later confirmation

> 以下均明確標示為「待確認」，不做設計推論。

來源指向：
- Step5 不一致：`convert_diffae_step5_intmodel_v2.py`（`main` 與 `convert_to_integer_weights`）+ `log/step5.log`
- 層數與特殊層策略：`quant_model_lora_v2.py`（`special_module_count_list`、`total_count`、`set_quant_state`）
- timestep 對齊：`quant_layer_v2.py` `TemporalActivationQuantizer.forward`、`quant_model_lora_v2.py` `set_quant_step`
- checkpoint 寫入：`quantize_diffae_step6_train_v2.py` `TrainingConfig`、`save_checkpoint`
- 診斷輸出：`quantize_diffae_step6_train_v2.py` `log_quant_update_diagnostics`、`_append_jsonl`、`_plot_quant_diag_curves`

1. **Step5 實際定位待確認**  
   程式有整數轉換函式，但主流程與 log 呈現不一致；目前看起來不是穩定主線。

2. **量化層總數描述存在差異，待確認**  
   Step4 log 有 `quantized 142 modules`；`QuantModel_DiffAE_LoRA` 內含 `total_count=144` 與特殊層索引邏輯。

3. **`set_quant_state` 特殊層覆寫語義待確認**  
   `QuantModel_DiffAE_LoRA.set_quant_state()` 中對 layer 3/144 的設定與 log 文字描述存在可讀性落差（例如 log 文案與實際設定值不完全一致）。

4. **timestep 對齊策略待確認**  
   `TemporalActivationQuantizer.current_step` 有自動遞減機制；`set_quant_step()` 雖存在，但在目前 trainer 主路徑中未直接呼叫。

5. **checkpoint 路徑策略待確認**  
   `SAVE_BEST_PATH/SAVE_FINAL_PATH` 與 `TEST_PATH` 的最終落地邏輯需確認是否為暫時測試策略。

6. **quant diag 輸出落地狀態待確認**  
   程式有輸出路徑與 json/曲線寫入流程，但目前工作樹未見對應輸出檔。

7. **Sample 腳本中的路徑/模式分支待確認是否為現行標準**  
   `sample_lora_intmodel_v2.py` 同時含 float/int 與 cache 分析分支，且部分路徑命名與 `quantize_ver2` 主線不同。
