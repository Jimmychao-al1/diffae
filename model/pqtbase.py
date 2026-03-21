import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import torch

class quant_one(Function):
    @staticmethod
    def forward(ctx, x, scale=128):
        x_scaled = x * scale
        x_rounded = torch.round(x_scaled)
        return x_rounded / scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None
    
def normalize_model_asymmetric(model, eps: float = 1e-6):
    """
    對 model 裡所有可訓練的參數做非對稱仿射映射，將 [p_min, p_max] 線性映射到 [-1,1]，
    並直接修改 model，最後回傳這個已被歸一化的 model。
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            print(f"Normalizing parameter: {name}, shape: {param.shape}")
            parts = name.split('.')
            if not param.requires_grad or (('weight' or 'bias') not in parts):
                continue

            # 原始最小/最大值
            p_min = float(param.data.min().item())
            p_max = float(param.data.max().item())

            # 避免 p_max==p_min
            if abs(p_max - p_min) < eps:
                # 全部參數相同時，不做變動
                continue

            # 仿射 mapping 參數： scale = (p_max - p_min)/2， zero_point = (p_max + p_min)/2
            scale      = (p_max - p_min) / 2.0
            zero_point = (p_max + p_min) / 2.0

            # (p - zero_point) / scale → 𝑝′ ∈ [-1,1]
            param.data.sub_(zero_point).div_(scale)

    return model

def apply_quantization_hooks(model, bits: int = 8, eps: float = 1e-6):
    """
    在所有指定層的 forward_pre_hook 和 forward_hook 中，
    對輸入和輸出做對稱量化：映射到 [-1,1] → 放大至 ±Q → 四捨五入 → 還原至浮點。
    """
    Q = 2 ** (bits - 1)  # 8-bit signed: Q=128

    CLAMP_MODULES = (
        nn.Conv1d, nn.Conv2d, nn.Conv3d,
        nn.Linear,
        nn.GroupNorm,
        nn.SiLU,
    )

    def _quantize_tensor(x: torch.Tensor):
        max = x.abs().max()
        x = x/max.clamp(min=eps)  # 避免除以零
        return quant_one.apply(x, Q).clamp(-1, 1)

    def _pre_hook(module, inputs):
        # 只對 tensor 類型 inputs 做量化
        new_inputs = []
        for x in inputs:
            if torch.is_tensor(x):
                new_inputs.append(_quantize_tensor(x))
            else:
                new_inputs.append(x)
        return tuple(new_inputs)

    def _post_hook(module, inputs, output):
        # 對 output 做同樣量化
        if torch.is_tensor(output):
            return _quantize_tensor(output)
        elif isinstance(output, (list, tuple)):
            return type(output)(
                _quantize_tensor(o) if torch.is_tensor(o) else o
                for o in output
            )
        else:
            return output

    # 註冊 hooks
    for module in model.modules():
        if isinstance(module, CLAMP_MODULES):
            module.register_forward_pre_hook(_pre_hook)
            #module.register_forward_hook(_post_hook)

    return model