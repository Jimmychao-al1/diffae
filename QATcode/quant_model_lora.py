import logging
import math
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from QATcode.quant_layer import (
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
    支援 LoRA 微調的量化模組 - 專為 Diff-AE Step 6 設計
    結合整數量化、LoRA 微調和 TALSQ
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
            
        # 權重處理 - 用於整數量化
        self.org_weight = org_module.weight.data.clone().cuda()
        self.ori_shape = org_module.weight.shape
        self.size_scale = int(8 // weight_quant_params['n_bits'])
        self.mode = mode
        self.register_buffer('weight', torch.randn(size=[self.ori_shape[0]//self.size_scale]+list(self.ori_shape[1:])))
        
        # Bias 處理
        if org_module.bias is not None:
            self.bias = org_module.bias
            self.org_bias = org_module.bias.data.clone()
        else:
            self.bias = None
            self.org_bias = None
        # 量化器初始化
        self.use_weight_quant = False
        self.use_act_quant = False
        self.intn_dequantizer = None  # 將在訓練時初始化
        
        # 初始化權重量化器 (修正：不能是 None)
        from QATcode.quant_layer import UniformAffineQuantizer
        self.weight_quantizer = UniformAffineQuantizer(
            **weight_quant_params, 
            weight_tensor=org_module.weight
        )
        self.act_quantizer = TemporalActivationQuantizer(**act_quant_params, num_steps=num_steps)
        
        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False  # False = 使用 LoRA, True = 普通量化
        
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
    
    def forward(self, input: torch.Tensor):
        #orig_weight = self.intn_dequantizer(self.weight)
        
        # 改成使用 self.org_weight + lora_weight 來計算權重
        if self.fwd_func is F.linear:
            E = torch.eye(self.org_weight.shape[1], device=input.device)
            lora_weight = self.loraB(self.loraA(self.lora_dropout_layer(E)))
            lora_weight = lora_weight.T
            #weight = orig_weight + lora_weight
            weight = self.org_weight + lora_weight.to(self.org_weight.device)
        elif self.fwd_func is F.conv2d:
            lora_weight = self.loraB.weight.squeeze(-1).squeeze(-1) @ self.loraA.weight.permute(2,3,0,1)  ## (cout, r) @　(3, 3, r, cin)
            lora_weight = lora_weight.permute(2,3,0,1)
            #weight = orig_weight + lora_weight
            weight = self.org_weight + lora_weight.to(self.org_weight.device)
        else:
            weight = self.org_weight
        if self.use_weight_quant:
            # 使用 self.weight_quantizer 來量化權重: self.org_weight + lora_weight
            weight = self.weight_quantizer(weight)
            bias = self.bias
        else:
            weight = self.org_weight
            bias = self.org_bias

            
        if self.use_act_quant:
            input = self.act_quantizer(input)

        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        
        return out
    
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant


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
        for m in self.model.modules():
            if isinstance(m, QuantModule_DiffAE_LoRA):
                count += 1
                m.set_quant_state(weight_quant, act_quant)
            elif isinstance(m, QuantModule):
                count += 1
                if count in (1,2,144):
                    m.set_quant_state(True, True)
                    logger.info("set quant state to True, True for layer %d", count)
                else:
                    m.set_quant_state(True, False)
                    logger.info("set quant state to True, False for layer %d", count)
    
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

    def get_trainable_parameters(self):
        """
        獲取可訓練參數：LoRA 參數 + 量化參數
        """
        lora_params = []
        quant_weight_params = []
        quant_act_params = []
        
        for name, param in self.named_parameters():
            if 'lora' in name:
                lora_params.append(param)
                param.requires_grad = True
            elif 'delta' in name and 'list' not in name:
                # 權重量化參數
                quant_weight_params.append(param)
                param.requires_grad = True
            elif 'delta_list' in name or 'zp_list' in name:
                # 激活量化參數 (TALSQ)
                quant_act_params.append(param)
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        return {
            'lora': lora_params,
            'quant_weight': quant_weight_params, 
            'quant_act': quant_act_params
        }
    
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
    支援 LoRA 微調的量化模組 - 專為 Diff-AE Step 6 設計
    結合整數量化、LoRA 微調和 TALSQ
    """
    def __init__(self, 
                 org_module : Union[nn.Conv2d, nn.Linear], 
                 weight_quant_params : dict = {}, 
                 act_quant_params : dict = {}, 
                 num_steps=100):
        super(INT_QuantModule_DiffAE_LoRA, self).__init__()
        
        # 基本設定
        if isinstance(org_module, nn.Conv2d):
            self.fwd_kwargs = dict(stride=org_module.stride, padding=org_module.padding,
                                 dilation=org_module.dilation, groups=org_module.groups)
            self.fwd_func = F.conv2d
        else:
            self.fwd_kwargs = dict()
            self.fwd_func = F.linear
            
        # 權重處理 - 用於整數量化
        self.ori_shape = org_module.weight.shape
        self.size_scale = int(8 // weight_quant_params['n_bits'])
        self.register_buffer('weight', torch.zeros(
            size=[self.ori_shape[0]//self.size_scale] + list(self.ori_shape[1:]),
            dtype=torch.uint8
        ))
        
        # Bias 處理
        if org_module.bias is not None:
            self.bias = org_module.bias
        else:
            self.bias = None
        # 量化器初始化
        self.use_weight_quant = True
        self.use_act_quant = False
        
        # 初始化權重量化器 (修正：不能是 None)
        from QATcode.quant_layer import INT_UniformAffineQuantizer
        self.weight_quantizer = INT_UniformAffineQuantizer(
            **weight_quant_params, 
            weight_tensor=org_module.weight
        )
        self.act_quantizer = TemporalActivationQuantizer(**act_quant_params, num_steps=num_steps)
        
        self.activation_function = StraightThrough()
        self.ignore_reconstruction = False  # INT 版本不使用 LoRA，但保留此標記用於其他邏輯
        
    
    def forward(self, input: torch.Tensor):

        if self.use_weight_quant:
            # 使用 self.weight_quantizer 來量化權重: self.weight
            weight = self.weight_quantizer(self.weight)
            bias = self.bias
        else:
            raise RuntimeError("INT_QuantModule_DiffAE_LoRA 必須啟用 weight quantization")

            
        if self.use_act_quant:
            input = self.act_quantizer(input)

        out = self.fwd_func(input, weight, bias, **self.fwd_kwargs)
        
        return out
    
    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

from QATcode.quant_layer import INT_QuantModule
class INT_QuantModel_DiffAE_LoRA(nn.Module):
    def __init__(self, 
                 model: nn.Module, 
                 weight_quant_params : dict = {}, 
                 act_quant_params : dict = {}, 
                 num_steps=100, 
                 quantize_skip_connections=False,
                 target_modules=None):
        super().__init__()
        self.model = model
        self.num_steps = num_steps
        self.count = 0
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
                                        act_quant_params):
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
                        setattr(module, name, INT_QuantModule(child_module, 
                        weight_quant_params, act_quant_params, need_init=False))
                    else:
                        logger.info("  替換層 %s -> INT_QuantModule_DiffAE_LoRA", name)
                        setattr(module, name, INT_QuantModule_DiffAE_LoRA(
                            child_module, 
                            weight_quant_params, 
                            act_quant_params,
                            num_steps=self.num_steps,
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
                        ):
        """設置量化狀態"""
        #for m in self.model.modules():
        #    if isinstance(m, (QuantModule_DiffAE_LoRA, QuantModule)):
        #        m.set_quant_state(weight_quant, act_quant)
        count = 0
        for m in self.model.modules():
            if isinstance(m, INT_QuantModule_DiffAE_LoRA):
                count += 1
                m.set_quant_state(weight_quant, act_quant)
            elif isinstance(m, INT_QuantModule):
                count += 1
                if count in (1,2,144):
                    m.set_quant_state(True, True)
                    logger.info("set quant state to True, True for layer %d", count)
                else:
                    m.set_quant_state(True, False)
                    logger.info("set quant state to True, False for layer %d", count)
    
    def set_first_last_layer_to_8bit(self):
        """設置首尾層為 8-bit (按 EfficientDM 邏輯)"""
        logger.info("=" * 50)
        logger.info("5. Setting first and last layers to 8-bit...")
        
        quant_modules_list = []
        for name, module in self.named_modules():
            if isinstance(module, (INT_QuantModule_DiffAE_LoRA, INT_QuantModule)):
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