import logging
import math
from typing import Union

from sympy.core.facts import FactKB
import torch
import torch.nn as nn
import torch.nn.functional as F
from QATcode.quantize_ver2.quant_layer_v2 import (
    QuantModule,
    SimpleDequantizer,
    StraightThrough,
    TemporalActivationQuantizer,
)
# 避免循環相依：不要在此處 import model.unet_autoenc
from model.blocks import *
logger = logging.getLogger(__name__)

class QuantModule_DiffAE_LoRA(nn.Module):
    """
    支援 LoRA 微調的量化模組（ver2 float fake-quant 路徑）
    核心流程：normalized fake-quant + rescale + FP32 bias
    """
    def __init__(self, 
                 org_module : Union[nn.Conv2d, nn.Linear], 
                 weight_quant_params, 
                 act_quant_params, 
                 num_steps=100, 
                 lora_rank=32,
                 mode='train',
                 target_modules=None):
        super(QuantModule_DiffAE_LoRA, self).__init__()
        
        # 基本設定
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                 dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
            
        # 權重處理（ver2 主路徑使用 org_weight + LoRA）
        self.org_weight = org_module.weight.data.clone().cuda()
        self.ori_shape = org_module.weight.shape
        self.size_scale = int(8 // weight_quant_params['n_bits'])
        self.mode = mode
        self.register_buffer('weight', torch.randn(size=[self.ori_shape[0]//self.size_scale]+list(self.ori_shape[1:])))
        
        # Bias 處理
        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone().cuda()
        else:
            self.bias = None
            self.org_bias = None
        # 量化器初始化
        self.use_weight_quant = False
        self.use_act_quant = False
        self.intn_dequantizer = None  # 保留欄位給 legacy/int 路徑，float 主路徑不使用
        
        # 初始化權重量化器 (修正：不能是 None)
        from QATcode.quantize_ver2.quant_layer_v2 import UniformAffineQuantizer
        self.weight_quantizer = UniformAffineQuantizer(
            **weight_quant_params, 
            weight_tensor=org_module.weight
        )
        self.act_quantizer = TemporalActivationQuantizer(**act_quant_params, num_steps=num_steps)
        
        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False  # False = 使用 LoRA, True = 普通量化
        # runtime_mode:
        # - train: dynamic a_w every forward
        # - infer: optionally use cached a_w
        self.runtime_mode = 'train'
        self.use_cached_aw = False
        self.cached_a_w = None
        self.use_original_weight = False

        # LoRA 參數設定
        self.lora_rank = lora_rank
        lora_dropout = 0.0
        if lora_dropout > 0.0:
            self.lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout_layer = nn.Identity()
        # LoRA 層初始化
        if isinstance(org_module, nn.Linear) and self.weight_quantizer.n_bits <= 8:
            self.loraA = nn.Linear(org_module.in_features, lora_rank, bias=False)
            self.loraB = nn.Linear(lora_rank, org_module.out_features, bias=False)
            nn.init.kaiming_uniform_(self.loraA.weight, a=math.sqrt(5))
            nn.init.zeros_(self.loraB.weight)
        elif isinstance(org_module, nn.Conv2d) and self.weight_quantizer.n_bits <= 8:
            self.loraA = nn.Conv2d(org_module.in_channels, lora_rank, org_module.kernel_size, 
                                 org_module.stride, org_module.padding, org_module.dilation, 
                                 org_module.groups, bias=False)
            self.loraB = nn.Conv2d(lora_rank, org_module.out_channels, 1, bias=False)
            nn.init.kaiming_uniform_(self.loraA.weight, a=math.sqrt(5))
            nn.init.zeros_(self.loraB.weight)

    def _compute_a_w(self, weight_eff: torch.Tensor):
        if len(weight_eff.shape) == 4:  # Conv2d: [Cout, Cin, H, W]
            return weight_eff.abs().amax(dim=(1, 2, 3), keepdim=True) + 1e-8
        if len(weight_eff.shape) == 2:  # Linear: [Cout, Cin]
            return weight_eff.abs().amax(dim=1, keepdim=True) + 1e-8
        return weight_eff.abs().max() + 1e-8

    def _get_a_w(self, weight_eff: torch.Tensor):
        if self.runtime_mode == 'infer' and self.use_cached_aw:
            if self.cached_a_w is None:
                self.cached_a_w = self._compute_a_w(weight_eff).detach()
            return self.cached_a_w.to(weight_eff.device)
        return self._compute_a_w(weight_eff)
    
    def forward(self, input: torch.Tensor):
        """
        Forward pass with LoRA + normalized fake-quant + rescale + FP32 bias.
        
        Flow:
        1. Compute effective weight: weight_eff = org_weight + lora_weight
        2. Quantize activation to x_norm (using TemporalActivationQuantizer with a_x[k])
        3. Quantize weight_eff to w_norm (dynamic absmax per-channel)
        4. Compute y_norm = conv/linear(x_norm, w_norm, bias=None)
        5. Rescale: y = y_norm * (a_x[k] * a_w)
        6. Add FP32 bias: y += bias_fp32
        """
        # Keep original bias in FP32
        bias_fp32 = self.bias if self.bias is not None else None
        
        # Step 1: Compute effective weight (org_weight + lora_weight)
        if self.fwd_func is F.linear:
            E = torch.eye(self.org_weight.shape[1], device=input.device)
            lora_weight = self.loraB(self.loraA(self.lora_dropout_layer(E)))
            lora_weight = lora_weight.T
            weight_eff = self.org_weight + lora_weight.to(self.org_weight.device)
        elif self.fwd_func is F.conv2d:
            lora_weight = self.loraB.weight.squeeze(-1).squeeze(-1) @ self.loraA.weight.permute(2,3,0,1)
            lora_weight = lora_weight.permute(2,3,0,1)
            weight_eff = self.org_weight + lora_weight.to(self.org_weight.device)
        else:
            weight_eff = self.org_weight

        if self.use_original_weight:
            weight_eff = self.org_weight

        # Ensure weight is on the same device as input.
        #weight_eff = weight_eff.to(input.device)
        
        if self.use_weight_quant and self.use_act_quant:
            # Both quant enabled: normalized quant + rescale
            # Step 2: Quantize activation using TemporalActivationQuantizer (has internal a_x[k])
            # The act_quantizer returns normalized x_norm and stores scale internally
            step_idx = max(0, min(self.act_quantizer.current_step, self.act_quantizer.total_steps - 1))
            a_x = self.act_quantizer.scale_list[step_idx].clamp(min=1e-8)
            x_norm = self.act_quantizer(input)  # Returns normalized quantized input
            
            # Step 3: Compute weight scale (a_w)
            a_w = self._get_a_w(weight_eff).detach()
            
            # Step 4: Quantize weight_eff to normalized range
            from QATcode.quantize_ver2.quant_layer_v2 import normalized_fake_quant
            w_norm = normalized_fake_quant(weight_eff, a_w, eps=1e-8)
            
            # Step 5: Compute normalized output (bias=None)
            y_norm = self.fwd_func(x_norm, w_norm, None, **self.fwd_kwargs)
            
            # Step 6: Rescale to original scale
            if len(weight_eff.shape) == 4:
                scale_factor = (a_x * a_w).view(1, -1, 1, 1)
            else:
                scale_factor = (a_x * a_w).view(1, -1)
            
            out = y_norm * scale_factor
            
            # Step 7: Add FP32 bias (if exists)
            if bias_fp32 is not None:
                if len(weight_eff.shape) == 4:
                    out = out + bias_fp32.view(1, -1, 1, 1)
                else:
                    out = out + bias_fp32.view(1, -1)
        
        elif self.use_weight_quant and not self.use_act_quant:
            # Only weight quant:
            # keep activation in FP32, but apply fake-quant + dequant on weight.
            # This preserves output scale while simulating quantization noise.
            from QATcode.quantize_ver2.quant_layer_v2 import normalized_fake_quant
            a_w = self._get_a_w(weight_eff).detach()
            w_norm = normalized_fake_quant(weight_eff, a_w, eps=1e-8)
            w_q = w_norm * a_w
            out = self.fwd_func(input, w_q, bias_fp32, **self.fwd_kwargs)
        
        elif not self.use_weight_quant and self.use_act_quant:
            # Only act quant:
            # quantize activation to normalized domain, then dequant back to FP32 scale.
            step_idx = max(0, min(self.act_quantizer.current_step, self.act_quantizer.total_steps - 1))
            a_x = self.act_quantizer.scale_list[step_idx].clamp(min=1e-8)
            x_norm = self.act_quantizer(input)
            x_q = x_norm * a_x
            out = self.fwd_func(x_q, weight_eff, bias_fp32, **self.fwd_kwargs)
        
        else:
            # No quantization
            bias = self.org_bias if self.org_bias is not None else None
            out = self.fwd_func(input, weight_eff, bias, **self.fwd_kwargs)
        
        return out
    
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def set_runtime_mode(self, mode: str = 'train', use_cached_aw: bool = False, clear_cached_aw: bool = False):
        """
        Set runtime behavior for a_w strategy.
        """
        if mode not in ('train', 'infer'):
            raise ValueError(f"Unsupported runtime mode: {mode}")
        self.runtime_mode = mode
        self.use_cached_aw = use_cached_aw
        if clear_cached_aw:
            self.cached_a_w = None


class QuantModel_DiffAE_LoRA(nn.Module):
    """
    Diff-AE 專用的 LoRA 量化模型 - Step 6 使用
    """
    def __init__(self, 
                 model: nn.Module, 
                 weight_quant_params, 
                 act_quant_params, 
                 num_steps=100, 
                 lora_rank=32,
                 quantize_skip_connections=False,
                 target_modules=None,
                 mode='train'):
        super().__init__()
        self.model = model
        self.num_steps = num_steps
        self.lora_rank = lora_rank
        self.count = 0
        self.mode = mode
        self.special_module_count_list = [1,2,3,144]  # Diff-AE 應為 142 層
        self.total_count = 144
        self.quantize_skip_connections = quantize_skip_connections
        
        # 選擇性量化 - 只處理 UNet 核心模組
        self.target_modules = ['time_embed', 'input_blocks', 'middle_block', 'output_blocks', 'out']
        self.quant_unet_modules_with_lora(weight_quant_params, act_quant_params)
        
    def quant_unet_modules_with_lora(self, 
                                    weight_quant_params, 
                                    act_quant_params):
        """
        對 UNet 核心模組進行 LoRA 量化改造
        """
        logger.info("=== 開始 UNet LoRA 量化改造 ===")
        
        for module_name in self.target_modules:
            if hasattr(self.model, module_name):
                module = getattr(self.model, module_name)
                logger.info("改造模組: %s", module_name)
                
                # 遞歸替換量化層
                self.quant_module_refactor_with_lora(
                    module, weight_quant_params, act_quant_params
                )
            else:
                logger.warning("模組 %s 不存在", module_name)
    
    def quant_module_refactor_with_lora(self, 
                                        module, 
                                        weight_quant_params, 
                                        act_quant_params,
                                        target_modules=None):
        """
        遞歸替換 Conv2d 和 Linear 層為 LoRA 量化層
        """
        for name, child_module in module.named_children():
            if isinstance(child_module, (nn.Conv2d, nn.Linear)):
                
                should_skip = False
                if not self.quantize_skip_connections:
                    if 'skip' in name.lower():
                        should_skip = True
                        logger.info("  跳過量化 skip connection: %s", name)
                
                # 替換為 LoRA 量化模組
                if not should_skip:
                    self.count += 1
                    if self.count in self.special_module_count_list:
                        logger.info("  跳過量化特殊模組: %s", name)
                        setattr(module, name, QuantModule(child_module, 
                        weight_quant_params, act_quant_params, need_init=False))
                    else:
                        logger.info("  替換層 %s -> QuantModule_DiffAE_LoRA", name)
                        setattr(module, name, QuantModule_DiffAE_LoRA(
                            child_module, 
                            weight_quant_params, 
                            act_quant_params,
                            num_steps=self.num_steps,
                            lora_rank=self.lora_rank,
                            mode=self.mode
                        ))

            
            elif isinstance(child_module, StraightThrough):
                continue
            
            else:
                # 遞歸處理子模組
                self.quant_module_refactor_with_lora(
                    child_module, weight_quant_params, act_quant_params
                )
    
    
    
    def set_quant_state(self, 
                        weight_quant: bool = False, 
                        act_quant: bool = False,
                        target_modules=None):
        """設置量化狀態"""
        #for m in self.model.modules():
        #    if isinstance(m, (QuantModule_DiffAE_LoRA, QuantModule)):
        #        m.set_quant_state(weight_quant, act_quant)
        count = 0
        #for m in self.model.modules():
        #    if isinstance(m, QuantModule_DiffAE_LoRA):
        #        count += 1
        #        m.set_quant_state(weight_quant, act_quant)
        #    elif isinstance(m, QuantModule):
        #        count += 1
        #        if count in (1,2,144):
        #            m.set_quant_state(True, True)
        #            logger.info("set quant state to True, True for layer %d", count)
        #        else:
        #            m.set_quant_state(True, False)
        #            logger.info("set quant state to True, False for layer %d", count)
        
        # time_embed 不量化
        for m in self.model.modules():
            if isinstance(m, QuantModule_DiffAE_LoRA):
                count += 1
                m.set_quant_state(weight_quant, act_quant)
            elif isinstance(m, QuantModule):
                # QuantModule : layer 1,2,144 權重、輸入都量化 ；layer 3 只進行權重量化
                #if weight_quant and act_quant:
                    count += 1
                    if count in [1,2]:
                        m.set_quant_state(True, True)
                        logger.info("set quant state to True, True for layer %d", count)
                    elif count == 3:
                        # 測試 layer 3 進行完整量化
                        #m.set_quant_state(True, True)
                        #logger.info("[Test] set quant state to True, True for layer %d", count)

                        # 測試 layer 3 不進行量化
                        #m.set_quant_state(False, False)
                        #logger.info("[Test] set quant state to False, False for layer %d", count)

                        # layer 3 權重量化
                        m.set_quant_state(True, False)
                        logger.info("set quant state to True, False for layer %d", count)
                    elif count == 144:
                        m.set_quant_state(False, False)
                        logger.info("set quant state to False, False for layer %d", count)
                #else:
                #    count += 1
                #    if count in (1,2,3,144):
                #        m.set_quant_state(False, False)
                #        logger.info("set quant state to False, False for layer %d", count)


            elif isinstance(m, QuantModule_DiffAE_LoRA):
                count += 1
                m.set_quant_state(weight_quant, act_quant)
    
    def set_quant_step(self, step: int):
        """
        設置所有 TemporalActivationQuantizer 的 current_step
        
        用於訓練/推理時手動控制 timestep 對應的 activation scale
        
        Args:
            step: 當前 diffusion timestep (0..num_steps-1)
        """
        for m in self.model.modules():
            if isinstance(m, (QuantModule_DiffAE_LoRA, QuantModule)):
                if hasattr(m, 'act_quantizer') and hasattr(m.act_quantizer, 'current_step'):
                    m.act_quantizer.current_step = step

    def set_runtime_mode(self, mode: str = 'train', use_cached_aw: bool = False, clear_cached_aw: bool = False):
        """
        Propagate runtime_mode to all quantized modules.
        """
        for m in self.model.modules():
            if isinstance(m, (QuantModule_DiffAE_LoRA, QuantModule)):
                if hasattr(m, 'set_runtime_mode'):
                    m.set_runtime_mode(mode=mode, use_cached_aw=use_cached_aw, clear_cached_aw=clear_cached_aw)
    
    def set_first_last_layer_to_8bit(self):
        """設置首尾層為 8-bit (按 EfficientDM 邏輯)"""
        logger.info("=" * 50)
        logger.info("5. Setting first and last layers to 8-bit...")
        
        quant_modules_list = []
        for name, module in self.named_modules():
            if isinstance(module, (QuantModule_DiffAE_LoRA, QuantModule)):
                quant_modules_list.append(module)
        
        logger.info("找到 %d 個量化模組", len(quant_modules_list))
        
        # 設定前 1 層和最後 1 層
        special_indices = [0,1,2,-1]
        
        for idx in special_indices:
            quant_modules_list[idx].weight_quantizer.bitwidth_refactor(8)
            quant_modules_list[idx].act_quantizer.bitwidth_refactor(8)
            quant_modules_list[idx].ignore_reconstruction = True
        
        logger.info("✅ 首尾層 8-bit 設定完成")

    
    def forward(self, *args, **kwargs):
        """前向傳播"""
        return self.model(*args, **kwargs)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        """
        重參數化 (reparameterization) 採樣
        """
        return self.model.reparameterize(mu, logvar)

    def sample_z(self, n: int, device):
        """
        生成 z
        """
        return self.model.sample_z(n, device)

    def noise_to_cond(self, noise: torch.Tensor):
        return self.model.noise_to_cond(noise)

    def encode(self, x):
        cond = self.model.encoder.forward(x)
        return {'cond': cond}
    
    def encode_stylespace(self, x, return_vector: bool = True):
        """
        將圖片編碼到風格空間
        """
        return self.model.encode_stylespace(x, return_vector)
    
    @property
    def stylespace_sizes(self):
        return self.model.stylespace_sizes
    
    @property
    def dtype(self):
        return self.model.dtype
    
    @property
    def device(self):
        return self.model.device
    
    @property
    def conf(self):
        return self.model.conf
    
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


#===============================================================
# INT_QuantModule_DiffAE_LoRA
#===============================================================

class INT_QuantModule_DiffAE_LoRA(nn.Module):
    """
    True-int forward: same structure as QuantModule_DiffAE_LoRA so checkpoints
    load identically, but the full-quant branch uses int32 accumulation.
    """
    def __init__(self, 
                 org_module : Union[nn.Conv2d, nn.Linear], 
                 weight_quant_params, 
                 act_quant_params, 
                 num_steps=100, 
                 lora_rank=32,
                 mode='train',
                 target_modules=None):
        super(INT_QuantModule_DiffAE_LoRA, self).__init__()

        # 基本設定
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                   dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear

        # 權重處理（org_weight 跟隨 org_module 的 device，由呼叫端負責搬到 GPU）
        self.org_weight = org_module.weight.data.clone().cuda()
        self.ori_shape  = org_module.weight.shape
        self.size_scale = int(8 // weight_quant_params['n_bits'])
        self.mode = mode
        self.register_buffer('weight', torch.randn(
            size=[self.ori_shape[0] // self.size_scale] + list(self.ori_shape[1:])))

        # Bias 處理
        if org_module.bias is not None:
            self.bias     = org_module.bias
            self.org_bias = org_module.bias.data.clone().cuda()
        else:
            self.bias     = None
            self.org_bias = None

        # 量化器初始化
        self.use_weight_quant  = False
        self.use_act_quant     = False
        self.intn_dequantizer  = None
        
        # 初始化權重量化器 (修正：不能是 None)
        from QATcode.quantize_ver2.quant_layer_v2 import UniformAffineQuantizer
        self.weight_quantizer = UniformAffineQuantizer(
            **weight_quant_params, 
            weight_tensor=org_module.weight
        )
        self.act_quantizer = TemporalActivationQuantizer(**act_quant_params, num_steps=num_steps)
        
        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False  # False = 使用 LoRA, True = 普通量化
        # runtime_mode:
        # - train: dynamic a_w every forward
        # - infer: optionally use cached a_w
        self.runtime_mode = 'train'
        self.use_cached_aw = False
        self.cached_a_w = None
        self.use_original_weight = False

        # LoRA 參數設定
        self.lora_rank = lora_rank
        lora_dropout = 0.0
        if lora_dropout > 0.0:
            self.lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout_layer = nn.Identity()
        # LoRA 層初始化
        if isinstance(org_module, nn.Linear) and self.weight_quantizer.n_bits <= 8:
            self.loraA = nn.Linear(org_module.in_features, lora_rank, bias=False)
            self.loraB = nn.Linear(lora_rank, org_module.out_features, bias=False)
            nn.init.kaiming_uniform_(self.loraA.weight, a=math.sqrt(5))
            nn.init.zeros_(self.loraB.weight)
        elif isinstance(org_module, nn.Conv2d) and self.weight_quantizer.n_bits <= 8:
            self.loraA = nn.Conv2d(org_module.in_channels, lora_rank, org_module.kernel_size, 
                                 org_module.stride, org_module.padding, org_module.dilation, 
                                 org_module.groups, bias=False)
            self.loraB = nn.Conv2d(lora_rank, org_module.out_channels, 1, bias=False)
            nn.init.kaiming_uniform_(self.loraA.weight, a=math.sqrt(5))
            nn.init.zeros_(self.loraB.weight)

    def _compute_a_w(self, weight_eff: torch.Tensor):
        if len(weight_eff.shape) == 4:  # Conv2d: [Cout, Cin, H, W]
            return weight_eff.abs().amax(dim=(1, 2, 3), keepdim=True) + 1e-8
        if len(weight_eff.shape) == 2:  # Linear: [Cout, Cin]
            return weight_eff.abs().amax(dim=1, keepdim=True) + 1e-8
        return weight_eff.abs().max() + 1e-8

    def _get_a_w(self, weight_eff: torch.Tensor):
        if self.runtime_mode == 'infer' and self.use_cached_aw:
            if self.cached_a_w is None:
                self.cached_a_w = self._compute_a_w(weight_eff).detach()
            return self.cached_a_w.to(weight_eff.device)
        return self._compute_a_w(weight_eff)

    def forward(self, input: torch.Tensor):
        """
        INT32-ACCUM forward (full-quant branch).

        full-quant branch dtype chain:
          input         : float32
          x_norm        : float32  (TemporalActivationQuantizer output)
          x_int         : int32    (_norm_to_int_code, values in [-127,127])
          w_norm        : float32  (normalized_fake_quant output)
          w_int         : int32    (_norm_to_int_code, values in [-127,127])
          y_accum       : int32    (_int_linear_accum / _int_conv2d_accum via torch._int_mm)
          y_norm_int    : float32  (y_accum.float() / 127^2)
          out           : float32  (y_norm_int * scale_factor + bias_fp32)

        weight-only / act-only / no-quant branches: float path unchanged.
        """
        from QATcode.quantize_ver2.quant_layer_v2 import (
            normalized_fake_quant,
            _norm_to_int_code,
            _int_linear_accum,
            _int_conv2d_accum,
        )

        bias_fp32 = self.bias if self.bias is not None else None

        # Step 1: effective weight = org_weight + lora_weight
        if self.fwd_func is F.linear:
            E = torch.eye(self.org_weight.shape[1], device=input.device)
            lora_weight = self.loraB(self.loraA(self.lora_dropout_layer(E)))
            lora_weight = lora_weight.T
            weight_eff = self.org_weight + lora_weight.to(self.org_weight.device)
        elif self.fwd_func is F.conv2d:
            lora_weight = (self.loraB.weight.squeeze(-1).squeeze(-1)
                           @ self.loraA.weight.permute(2, 3, 0, 1))
            lora_weight = lora_weight.permute(2, 3, 0, 1)
            weight_eff = self.org_weight + lora_weight.to(self.org_weight.device)
        else:
            weight_eff = self.org_weight

        if self.use_weight_quant and self.use_act_quant:
            # ---- INT32 ACCUMULATION PATH ----
            step_idx = max(0, min(self.act_quantizer.current_step,
                                  self.act_quantizer.total_steps - 1))
            a_x    = self.act_quantizer.scale_list[step_idx].clamp(min=1e-8)
            x_norm = self.act_quantizer(input)            # float32

            a_w    = self._get_a_w(weight_eff).detach()
            w_norm = normalized_fake_quant(weight_eff, a_w, eps=1e-8)  # float32

            x_int = _norm_to_int_code(x_norm)            # int32
            w_int = _norm_to_int_code(w_norm)            # int32

            if self.fwd_func is F.linear:
                y_accum = _int_linear_accum(x_int, w_int)   # int32
            else:
                y_accum = _int_conv2d_accum(                 # int32
                    x_int, w_int,
                    stride=self.fwd_kwargs['stride'],
                    padding=self.fwd_kwargs['padding'],
                    dilation=self.fwd_kwargs['dilation'],
                    groups=self.fwd_kwargs['groups'],
                )
            assert y_accum.dtype == torch.int32

            # int32 → float32 for scale + bias
            y_norm_int = y_accum.float() / (127.0 * 127.0)  # float32

            if len(weight_eff.shape) == 4:
                scale_factor = (a_x * a_w).view(1, -1, 1, 1)
            else:
                scale_factor = (a_x * a_w).view(1, -1)

            out = y_norm_int * scale_factor

            if bias_fp32 is not None:
                if len(weight_eff.shape) == 4:
                    out = out + bias_fp32.view(1, -1, 1, 1)
                else:
                    out = out + bias_fp32.view(1, -1)

        elif self.use_weight_quant and not self.use_act_quant:
            a_w    = self._get_a_w(weight_eff).detach()
            w_norm = normalized_fake_quant(weight_eff, a_w, eps=1e-8)
            w_q    = w_norm * a_w
            out    = self.fwd_func(input, w_q, bias_fp32, **self.fwd_kwargs)

        elif not self.use_weight_quant and self.use_act_quant:
            step_idx = max(0, min(self.act_quantizer.current_step,
                                  self.act_quantizer.total_steps - 1))
            a_x    = self.act_quantizer.scale_list[step_idx].clamp(min=1e-8)
            x_norm = self.act_quantizer(input)
            x_q    = x_norm * a_x
            out    = self.fwd_func(x_q, weight_eff, bias_fp32, **self.fwd_kwargs)

        else:
            bias = self.org_bias if self.org_bias is not None else None
            out  = self.fwd_func(input, weight_eff, bias, **self.fwd_kwargs)

        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

    def set_runtime_mode(self, mode: str = 'train', use_cached_aw: bool = False,
                         clear_cached_aw: bool = False):
        if mode not in ('train', 'infer'):
            raise ValueError(f"Unsupported runtime mode: {mode}")
        self.runtime_mode = mode
        self.use_cached_aw = use_cached_aw
        if clear_cached_aw:
            self.cached_a_w = None

from QATcode.quantize_ver2.quant_layer_v2 import QuantModule
class INT_QuantModel_DiffAE_LoRA(nn.Module):
    """
    INT 推論用 QuantModel。
    非特殊層替換成 INT_QuantModule_DiffAE_LoRA（true-int forward）。
    特殊層（1,2,3,144）使用 QuantModule（float fake-quant）。
    結構與 QuantModel_DiffAE_LoRA 完全一致，checkpoint 可直接載入。
    """
    def __init__(self,
                 model: nn.Module,
                 weight_quant_params,
                 act_quant_params,
                 num_steps=100,
                 lora_rank=32,
                 quantize_skip_connections=False,
                 target_modules=None,
                 mode='train'):
        super().__init__()
        self.model = model
        self.num_steps = num_steps
        self.lora_rank = lora_rank
        self.count = 0
        self.mode = mode
        self.special_module_count_list = [1, 2, 3, 144]
        self.total_count = 144
        self.quantize_skip_connections = quantize_skip_connections
        self.target_modules = ['time_embed', 'input_blocks', 'middle_block',
                               'output_blocks', 'out']
        self._quant_unet_modules(weight_quant_params, act_quant_params)

    # ------------------------------------------------------------------
    def _quant_unet_modules(self, weight_quant_params, act_quant_params):
        logger.info("=== INT_QuantModel: 開始替換模組 ===")
        for module_name in self.target_modules:
            if hasattr(self.model, module_name):
                module = getattr(self.model, module_name)
                logger.info("改造模組: %s", module_name)
                self._refactor(module, weight_quant_params, act_quant_params)
            else:
                logger.warning("模組 %s 不存在", module_name)

    def _refactor(self, module, weight_quant_params, act_quant_params):
        for name, child in module.named_children():
            if isinstance(child, (nn.Conv2d, nn.Linear)):
                if not self.quantize_skip_connections and 'skip' in name.lower():
                    logger.info("  skip %s", name)
                    continue
                self.count += 1
                if self.count in self.special_module_count_list:
                    logger.info("  特殊層 %s (#%d) -> QuantModule", name, self.count)
                    setattr(module, name, QuantModule(
                        child, weight_quant_params, act_quant_params, need_init=False))
                else:
                    logger.info("  %s (#%d) -> INT_QuantModule_DiffAE_LoRA", name, self.count)
                    setattr(module, name, INT_QuantModule_DiffAE_LoRA(
                        child,
                        weight_quant_params,
                        act_quant_params,
                        num_steps=self.num_steps,
                        lora_rank=self.lora_rank,
                        mode=self.mode,
                    ))
            elif isinstance(child, StraightThrough):
                continue
            else:
                self._refactor(child, weight_quant_params, act_quant_params)

    # ------------------------------------------------------------------
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False,
                        target_modules=None):
        count = 0
        for m in self.model.modules():
            if isinstance(m, INT_QuantModule_DiffAE_LoRA):
                count += 1
                m.set_quant_state(weight_quant, act_quant)
            elif isinstance(m, QuantModule):
                count += 1
                if count in [1, 2]:
                    m.set_quant_state(True, True)
                    logger.info("set quant state to True, True for layer %d", count)
                elif count == 3:
                    m.set_quant_state(True, False)
                    logger.info("set quant state to True, False for layer %d", count)
                elif count == 144:
                    m.set_quant_state(False, False)
                    logger.info("set quant state to False, False for layer %d", count)


            elif isinstance(m, QuantModule_DiffAE_LoRA):
                count += 1
                m.set_quant_state(weight_quant, act_quant)
    
    def set_quant_step(self, step: int):
        for m in self.model.modules():
            if isinstance(m, (INT_QuantModule_DiffAE_LoRA, QuantModule)):
                if hasattr(m, 'act_quantizer') and hasattr(m.act_quantizer, 'current_step'):
                    m.act_quantizer.current_step = step

    def set_runtime_mode(self, mode: str = 'train', use_cached_aw: bool = False,
                         clear_cached_aw: bool = False):
        for m in self.model.modules():
            if isinstance(m, (INT_QuantModule_DiffAE_LoRA, QuantModule)):
                if hasattr(m, 'set_runtime_mode'):
                    m.set_runtime_mode(mode=mode, use_cached_aw=use_cached_aw,
                                       clear_cached_aw=clear_cached_aw)

    def set_first_last_layer_to_8bit(self):
        logger.info("Setting first/last layers to 8-bit (INT model)...")
        quant_modules_list = [
            m for _, m in self.named_modules()
            if isinstance(m, (INT_QuantModule_DiffAE_LoRA, QuantModule))
        ]
        logger.info("找到 %d 個量化模組", len(quant_modules_list))
        for idx in [0, 1, 2, -1]:
            quant_modules_list[idx].weight_quantizer.bitwidth_refactor(8)
            quant_modules_list[idx].act_quantizer.bitwidth_refactor(8)
            quant_modules_list[idx].ignore_reconstruction = True
        logger.info("首尾層 8-bit 設定完成")

    # ------------------------------------------------------------------
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor):
        return self.model.reparameterize(mu, logvar)

    def sample_z(self, n: int, device):
        return self.model.sample_z(n, device)

    def noise_to_cond(self, noise: torch.Tensor):
        return self.model.noise_to_cond(noise)

    def encode(self, x):
        return {'cond': self.model.encoder.forward(x)}

    def encode_stylespace(self, x, return_vector: bool = True):
        return self.model.encode_stylespace(x, return_vector)

    @property
    def stylespace_sizes(self):
        return self.model.stylespace_sizes

    @property
    def dtype(self):
        return self.model.dtype

    @property
    def device(self):
        return self.model.device

    @property
    def conf(self):
        return self.model.conf

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)
