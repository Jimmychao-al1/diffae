# quantize_ver2：Normalized Fake-Quantization 實作

此目錄實作 Diff-AE 的 **normalized fake-quantization** 量化訓練管線（QAT），採用對稱 int8 + per-timestep activation scale，取代原始的 affine quantization（0..255）。

---

## 目錄結構

```
quantize_ver2/
├── quant_layer_v2.py                       # 核心量化層（UniformAffineQuantizer、TemporalActivationQuantizer）
├── quant_model_lora_v2.py                  # 量化模型（QuantModel_DiffAE_LoRA、QuantModule_DiffAE_LoRA）
├── quant_dataset_v2.py                     # 校準資料集
├── diffae_trainer_v2.py                    # QAT 訓練器
├── quantize_diffae_step4_selective_v2.py   # Step 4：校準初始化
├── quantize_diffae_step6_train_v2.py       # Step 6：QAT 訓練（LoRA + activation scale）
├── sample_lora_intmodel_v2.py              # 推理採樣（Float 量化模型）
├── checkpoints/                            # 訓練產出 checkpoint
│   ├── diffae_step6_lora_best.pth          # T=100 最佳 checkpoint（正式）
│   └── diffae_step6_lora_best_20steps.pth  # T=20 最佳 checkpoint
├── calibration_diffae.pth                  # 校準資料
└── log/                                    # 訓練 log
```

---

## 實作架構

### 核心量化設計

**量化範圍**：`[-128, 127]`（symmetric int8）

**核心函數 `normalized_fake_quant(x, scale)`**：

```
x_norm = clip(x / scale, -1, 1)      # 歸一化到 [-1, 1]
x_int  = round(x_norm × 127)         # 量化
x_int  = clip(x_int, -128, 127)
x_q    = x_int / 127.0               # 輸出 normalized fake-quant 結果
```

### Activation Scale（`a_x[k]`）

- 類型：可訓練 `nn.Parameter`，維度 `[num_steps]`（per-timestep, per-layer）
- 初始化：通過 calibration 計算每層每步的 absmax
- 訓練：Step 6 中與 LoRA 一起最佳化

### Weight Scale（`a_w`）

- 類型：動態 absmax（每次 forward 計算）
- 維度：Per-out-channel（Conv2d: `[Cout,1,1,1]`；Linear: `[Cout,1]`）
- 動態計算確保 LoRA 修改後的有效權重 `weight_eff = org_weight + lora_weight` 始終正確對應

### Forward Pass（Conv2d 範例）

```
a_x = x.abs().max()                         # scalar
x_norm = normalized_fake_quant(x, a_x)      # [-1, 1]

a_w = w_eff.abs().amax(dim=(1,2,3), keepdim=True)   # per-channel
w_norm = normalized_fake_quant(w_eff, a_w)  # [-1, 1]

y_norm = F.conv2d(x_norm, w_norm, bias=None, ...)
y = y_norm * (a_x * a_w)                    # 還原尺度
y += bias_fp32                              # FP32 bias
```

---

## 訓練管線（Step 4 → Step 6）

### Step 4：校準初始化

```bash
cd /home/jimmy/diffae
python QATcode/quantize_ver2/quantize_diffae_step4_selective_v2.py
```

- 啟用量化後執行一次 forward 校準
- 輸出初始 checkpoint（`diffae_unet_quantw8a8_selective.pth`）

### Step 6：QAT 訓練

```bash
python QATcode/quantize_ver2/quantize_diffae_step6_train_v2.py
```

- 載入 Step 4 checkpoint，訓練 LoRA + `a_x[k]`
- 使用 `set_quant_step(i)` 在每個 DDIM step 設置 `TemporalActivationQuantizer.current_step`
- 輸出：`checkpoints/diffae_step6_lora_best.pth`

### 推理採樣

```bash
python QATcode/quantize_ver2/sample_lora_intmodel_v2.py \
  --mode float --num_steps 100 --eval_samples 5000
```

---

## 注意事項

- **Step 5 未使用**：`convert_diffae_step5_intmodel_v2.py` 保留但不在 pipeline 中
- **首尾層固定 8-bit**：按原邏輯處理
- **TemporalActivationQuantizer**：需確保 `current_step` 初始值正確（通常為 `num_steps-1`）
- **新版 checkpoint 與舊版不相容**：需重新執行 Step 4 校準

---

## 與原版差異

| 項目 | 原版（affine quant） | Ver2（normalized fake-quant） |
|------|---------------------|-------------------------------|
| 整數範圍 | `[0, 255]` | `[-128, 127]` |
| Activation scale | `delta/zero_point`（可訓練） | `a_x[k]`（可訓練，absmax-based） |
| Weight scale | `delta/zero_point`（可訓練） | `a_w`（動態 absmax） |
| Bias 處理 | 與 conv/linear 一起計算 | FP32，rescale 後再加 |
