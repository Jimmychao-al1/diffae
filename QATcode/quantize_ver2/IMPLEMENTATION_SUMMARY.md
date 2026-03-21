# Quantize Ver2 實作總結

## 概述

本次實作完成了 **normalized fake-quantization** 流程，替代原本的 affine quantization (0..255)。

核心改動：`absmax → normalize to [-1,1] → multiply by 127 → round+clip to [-128,127] → divide by 127 → rescale → FP32 bias`

---

## 核心設計

### 1. 量化範圍
- **整數範圍**: `[-128, 127]` (symmetric int8)
- **歸一化範圍**: `[-1, 1]` (normalized fake-quant 輸出)

### 2. Scale 參數

#### Activation Scale (a_x[k])
- **類型**: 可訓練 `nn.Parameter`
- **維度**: `[num_steps]` (per-timestep, per-layer)
- **初始化**: 通過 calibration 計算每層每步的 absmax
- **訓練**: Step6 中與 LoRA 一起微調

#### Weight Scale (a_w)
- **類型**: 動態 absmax (每次 forward 計算)
- **維度**: Per-out-channel (`[Cout,1,1,1]` for Conv2d, `[Cout,1]` for Linear)
- **原因**: 
  - LoRA 動態修改權重 (`weight_eff = org_weight + lora_weight`)
  - 動態計算確保 scale 始終對應當前有效權重
  - 避免 scale 與實際權重分佈脫節

---

## 修改文件清單

### 1. `quant_layer_v2.py` ✅

#### 新增函數
```python
def normalized_fake_quant(x, scale, eps=1e-8):
    """核心量化函數: normalize → round+clip → denormalize"""
    scale = torch.clamp(scale, min=eps)
    x_norm = torch.clamp(x / scale, -1.0, 1.0)
    x_int = round_ste(x_norm * 127.0)
    x_int = torch.clamp(x_int, -128.0, 127.0)
    x_q = x_int / 127.0
    return x_q
```

#### `UniformAffineQuantizer` 修改
- 移除 `delta/zero_point` 初始化邏輯
- `forward()`: 動態計算 absmax → 調用 `normalized_fake_quant`
- 支援 per-tensor (activation) 和 per-channel (weight)

#### `TemporalActivationQuantizer` 重寫
- **參數**: `scale_list = nn.Parameter(torch.ones(num_steps) * 0.1, requires_grad=True)`
- **校準**: `calibrate_step(x, step)` - 記錄每步的 absmax
- **Forward**: 使用 `scale_list[current_step]` → 調用 `normalized_fake_quant`
- **自動遞減**: `current_step--` (兼容 sampling loop)

#### `QuantModule.forward()` 修改
- **Both quant 路徑**:
  1. 計算 `a_x = input.abs().max()`
  2. `x_norm = normalized_fake_quant(input, a_x)`
  3. 計算 `a_w` (per-channel absmax)
  4. `w_norm = normalized_fake_quant(weight, a_w)`
  5. `y_norm = conv/linear(x_norm, w_norm, bias=None)`
  6. **Rescale**: `y = y_norm * (a_x * a_w)`
  7. **Add FP32 bias**: `y += bias_fp32`

---

### 2. `quant_model_lora_v2.py` ✅

#### `QuantModule_DiffAE_LoRA.forward()` 修改
- **LoRA 權重合成**: `weight_eff = org_weight + lora_weight`
- **Normalized quant 流程**:
  1. 從 `act_quantizer` 取得 `a_x[current_step]`
  2. 調用 `act_quantizer(input)` → 返回 `x_norm`
  3. 動態計算 `a_w` (per-channel absmax of `weight_eff`)
  4. `w_norm = normalized_fake_quant(weight_eff, a_w)`
  5. `y_norm = conv/linear(x_norm, w_norm, bias=None)`
  6. **Rescale**: `y = y_norm * (a_x * a_w)`
  7. **Add FP32 bias**: `y += bias_fp32`

#### `QuantModel_DiffAE_LoRA` 新增方法
```python
def set_quant_step(self, step: int):
    """設置所有 TemporalActivationQuantizer 的 current_step"""
    for m in self.model.modules():
        if hasattr(m, 'act_quantizer') and hasattr(m.act_quantizer, 'current_step'):
            m.act_quantizer.current_step = step
```

---

### 3. `quantize_diffae_step4_selective_v2.py` ✅

#### `calibrate_quantized_model()` 重寫
- **目標**: 以最小流程完成 Step4 初始化
- **方法**: 啟用量化後執行一次 forward（`x/t/cond`）
- **說明**: per-timestep `a_x[k]` 的主要更新在 Step6 訓練期進行

---

### 4. `diffae_trainer_v2.py` ✅

#### `ddim_sample_with_training()` 修改
- **在訓練循環中設置 timestep**:
```python
for i in indices:
    t = torch.tensor([i] * batch_size, device=device)
    
    # 設置 TemporalActivationQuantizer 的 current_step
    if hasattr(self.quant_model, 'set_quant_step'):
        self.quant_model.set_quant_step(i)
    
    # ... (原本的訓練邏輯)
```

---

## Forward Pass 計算流程詳解

### Normalized Fake-Quant (單值)
```
x = 3.5, a_x = 5.0
→ x_norm = clip(3.5/5.0, -1, 1) = 0.7
→ x_int = round(0.7 * 127) = round(88.9) = 89
→ x_int = clip(89, -128, 127) = 89
→ x_q = 89 / 127 ≈ 0.7008  (normalized output)
```

### 完整 Forward (Conv2d 範例)
```
Input: x [B, C_in, H, W]
Weight: w [C_out, C_in, K, K]

1. a_x = x.abs().max()  → scalar
2. x_norm = normalized_fake_quant(x, a_x)  → [B, C_in, H, W], range ≈[-1,1]

3. a_w = w.abs().amax(dim=(1,2,3), keepdim=True)  → [C_out, 1, 1, 1]
4. w_norm = normalized_fake_quant(w, a_w)  → [C_out, C_in, K, K], range ≈[-1,1]

5. y_norm = F.conv2d(x_norm, w_norm, bias=None, ...)  → [B, C_out, H', W']
   (normalized output, small magnitude)

6. scale_factor = (a_x * a_w).view(1, C_out, 1, 1)  → [1, C_out, 1, 1]
7. y = y_norm * scale_factor  → [B, C_out, H', W'] (restored scale)

8. y += bias_fp32.view(1, C_out, 1, 1)  → final output
```

---

## Step4/Step6/Sample 一致性

### Step4 (校準)
- 目標: 產出 ver2 初始化 checkpoint
- 方法: 單次 forward 校準（輕量初始化）
- 輸出:
  - `diffae_unet_quantw8a8_selective.pth`
  - `diffae_unet_quantw8a8_intmodel.pth`（Step6 相容命名）

### Step6 (訓練)
- 載入 Step4 checkpoint (帶初始 `a_x[k]`)
- 訓練 LoRA + `a_x[k]` (同步優化)
- 每步通過 `set_quant_step(i)` 設置 timestep
- 輸出: `diffae_step6_lora_best.pth`

### Sample (推理)
- 載入 Step6 checkpoint
- Sampling loop 自動遞減 `current_step` (或手動設置)
- 使用訓練後的 `a_x[k]` 進行量化推理

---

## 與原版差異對比

| 項目 | 原版 (affine quant) | Ver2 (normalized fake-quant) |
|------|---------------------|------------------------------|
| 整數範圍 | `[0, 255]` | `[-128, 127]` |
| Activation scale | `delta/zero_point` (可訓練) | `a_x[k]` (可訓練, absmax-based) |
| Weight scale | `delta/zero_point` (可訓練) | `a_w` (動態 absmax) |
| Bias 處理 | 與 conv/linear 一起計算 | FP32, rescale 後再加 |
| Forward 流程 | `x_q = (round(x/delta) + zp) * delta` | `x_norm → conv → rescale → +bias` |
| 校準方式 | 計算 `delta/zp` 範圍 | 記錄 absmax |

---

## 優勢

1. **對稱性**: `[-128, 127]` 更接近對稱，減少 zero-point 偏移影響
2. **動態 a_w**: LoRA 權重變化時，scale 自動對應，避免分佈脫節
3. **FP32 bias**: 保持 bias 精度，減少量化誤差累積
4. **Rescale 機制**: 明確分離 normalized 計算與尺度還原，數值更穩定

---

## 測試建議

1. **Step4 校準**: 
   ```bash
   cd /home/jimmy/diffae
   python QATcode/quantize_ver2/quantize_diffae_step4_selective_v2.py
   ```
   檢查輸出的 scale 統計，確認範圍合理。

2. **Step6 訓練**:
   ```bash
   python QATcode/quantize_ver2/quantize_diffae_step6_train_v2.py
   ```
   監控 distillation loss 收斂。

3. **Sample 推理**:
   ```bash
   python QATcode/quantize_ver2/sample_lora_intmodel_v2.py --mode float --T 20 --eval_samples 5000
   ```
   比較 FID 與原版。

---

## 注意事項

- **Step5 未使用**: `convert_diffae_step5_intmodel_v2.py` 保留但不在 pipeline 中
- **首尾層**: 仍保持 8-bit，按原邏輯處理
- **TemporalActivationQuantizer**: 需確保 `current_step` 初始值正確 (通常為 `num_steps-1`)
- **兼容性**: 新版 checkpoint 與舊版不兼容，需重新 Step4 校準

---

## 實作完成日期
2026-01-15

## 貢獻者
AI Assistant (Claude Sonnet 4.5)
