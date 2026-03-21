"""
Q-DiffAE Quantitative Analysis Module
用于分析模型大小、参数量、MACs、生成时间和 Data Transfer
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
import time
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class QuantitativeAnalyzer:
    """定量分析器"""
    
    def __init__(self, log_file: str = "QATcode/quantitative_analysis.log"):
        self.log_file = log_file
        self.results = {}
        
        # 设置日志
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    def analyze_model_size_and_params(self, model: nn.Module, model_name: str) -> Dict:
        """
        分析模型大小和参数量
        
        Returns:
            {
                'total_params': int,
                'trainable_params': int,
                'model_size_mb': float,
                'checkpoint_size_mb': float,  # 如果可能的话
                'layer_breakdown': Dict[str, int]  # 各层参数量
            }
        """
        total_params = 0
        trainable_params = 0
        layer_breakdown = {}
        
        # 计算参数量
        for name, param in model.named_parameters():
            num_params = param.numel()
            total_params += num_params
            if param.requires_grad:
                trainable_params += num_params
            layer_breakdown[name] = num_params
        
        # 计算模型大小（假设 FP32）
        model_size_mb = total_params * 4 / (1024 ** 2)  # FP32 = 4 bytes
        
        # 对于量化模型，计算实际存储大小
        quantized_size_mb = 0
        quantized_params = 0
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                if isinstance(module.weight, torch.Tensor):
                    if module.weight.dtype == torch.uint8:
                        # INT8 量化权重
                        quantized_size_mb += module.weight.numel() * 1 / (1024 ** 2)
                        quantized_params += module.weight.numel()
                    else:
                        # FP32 权重
                        quantized_size_mb += module.weight.numel() * 4 / (1024 ** 2)
                        quantized_params += module.weight.numel()
        
        # 如果没有量化权重，使用原始大小
        if quantized_size_mb == 0:
            quantized_size_mb = model_size_mb
        
        result = {
            'model_name': model_name,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': model_size_mb,
            'quantized_size_mb': quantized_size_mb,
            'quantized_params': quantized_params if quantized_params > 0 else total_params,
            'layer_breakdown': layer_breakdown
        }
        
        logger.info(f"=== {model_name} Model Analysis ===")
        logger.info(f"Total params: {total_params:,} ({total_params/1e6:.2f}M)")
        logger.info(f"Trainable params: {trainable_params:,}")
        logger.info(f"Model size (FP32): {model_size_mb:.2f} MB")
        if quantized_size_mb != model_size_mb:
            logger.info(f"Quantized size: {quantized_size_mb:.2f} MB")
            logger.info(f"Compression ratio: {model_size_mb/quantized_size_mb:.2f}x")
        
        return result
    
    def calculate_macs(self, model: nn.Module, input_shape: Tuple, 
                      timestep: int = 0, use_cache: bool = False,
                      cache_scheduler: Optional[Dict] = None) -> Dict:
        """
        计算 MACs (Multiply-Accumulate Operations)
        
        Args:
            model: 模型
            input_shape: 输入形状 (C, H, W)
            timestep: 时间步
            use_cache: 是否使用 cache
            cache_scheduler: cache scheduler 字典
            
        Returns:
            {
                'total_macs': int,
                'macs_per_timestep': int,
                'total_macs_all_timesteps': int,
                'cache_reduction_ratio': float,  # 如果使用 cache
                'layer_macs': Dict[str, int]
            }
        """
        try:
            from fvcore.nn import FlopCountMode, flop_count
            USE_FVCORE = True
        except ImportError:
            logger.warning("fvcore not installed, using manual MACs calculation")
            USE_FVCORE = False
        
        device = next(model.parameters()).device
        batch_size = 1
        
        # 创建输入
        x = torch.randn(batch_size, *input_shape, device=device)
        t = torch.tensor([timestep] * batch_size, device=device)
        
        # 准备 model_kwargs
        model_kwargs = {}
        if use_cache and cache_scheduler is not None:
            model_kwargs['cached_data'] = {}
            model_kwargs['cached_scheduler'] = [1] * len(cache_scheduler)  # 假设全部重新计算
        
        if USE_FVCORE:
            try:
                # 计算单步 MACs
                flops_dict, _ = flop_count(model, (x, t), **model_kwargs)
                total_macs = sum(flops_dict.values())
                layer_macs = {k: int(v) for k, v in flops_dict.items()}
            except Exception as e:
                logger.warning(f"fvcore calculation failed: {e}, using manual method")
                USE_FVCORE = False
        
        if not USE_FVCORE:
            # 手动计算 MACs
            total_macs, layer_macs = self._manual_macs_calculation(model, input_shape)
        
        # 计算所有 timestep 的 MACs
        num_timesteps = 100 if timestep == 0 else timestep + 1  # 假设
        total_macs_all = total_macs * num_timesteps
        
        # 如果使用 cache，计算减少的 MACs
        cache_reduction_ratio = 0.0
        if use_cache and cache_scheduler is not None:
            # 计算实际需要计算的 layer 数量
            total_layers = len(cache_scheduler)
            recompute_layers = sum(1 for layer_timesteps in cache_scheduler.values() 
                                 if timestep in layer_timesteps)
            if total_layers > 0:
                cache_reduction_ratio = 1.0 - (recompute_layers / total_layers)
                total_macs_all = int(total_macs_all * (1 - cache_reduction_ratio))
        
        result = {
            'total_macs': int(total_macs),
            'macs_per_timestep': int(total_macs),
            'total_macs_all_timesteps': int(total_macs_all),
            'cache_reduction_ratio': cache_reduction_ratio,
            'layer_macs': layer_macs,
            'num_timesteps': num_timesteps
        }
        
        logger.info(f"=== MACs Analysis ===")
        logger.info(f"MACs per timestep: {total_macs/1e9:.2f} G")
        logger.info(f"Total MACs ({num_timesteps} steps): {total_macs_all/1e9:.2f} G")
        if use_cache:
            logger.info(f"Cache reduction ratio: {cache_reduction_ratio*100:.1f}%")
        
        return result
    
    def _manual_macs_calculation(self, model: nn.Module, input_shape: Tuple) -> Tuple[int, Dict]:
        """手动计算 MACs（简化版本）"""
        total_macs = 0
        layer_macs = {}
        
        # 遍历模型计算各层 MACs
        for name, module in model.named_modules():
            macs = 0
            if isinstance(module, nn.Conv2d):
                # Conv2d: output_size * kernel_size * in_channels
                # 简化：假设 output_size = input_size (stride=1, padding=same)
                if hasattr(module, 'weight') and module.weight is not None:
                    kernel_size = module.kernel_size[0] * module.kernel_size[1] if isinstance(module.kernel_size, tuple) else module.kernel_size ** 2
                    in_channels = module.in_channels
                    out_channels = module.out_channels
                    # 假设 output size = input size (简化)
                    output_size = input_shape[1] * input_shape[2]  # H * W
                    macs = output_size * kernel_size * in_channels * out_channels
            elif isinstance(module, nn.Linear):
                macs = module.in_features * module.out_features
            
            if macs > 0:
                layer_macs[name] = int(macs)
                total_macs += macs
        
        return int(total_macs), layer_macs
    
    def measure_generation_time(self, sampler, model, conf, num_samples: int = 10,
                                image_size: Tuple[int, int] = (128, 128),
                                device=None, latent_sampler=None,
                                conds_mean=None, conds_std=None) -> Dict:
        """
        测量生成时间
        
        Returns:
            {
                'avg_time_per_image': float,
                'total_time': float,
                'std_time': float,
                'min_time': float,
                'max_time': float
            }
        """
        if device is None:
            device = next(model.parameters()).device
        
        times = []
        
        logger.info(f"Measuring generation time for {num_samples} samples...")
        
        # 准备数据
        from renderer import render_uncondition
        
        for i in range(num_samples):
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            # 生成一张图片
            with torch.no_grad():
                try:
                    # 创建随机噪声
                    x_T = torch.randn(1, 3, *image_size, device=device)
                    
                    # 使用 render_uncondition 生成图片
                    _ = render_uncondition(
                        conf=conf,
                        model=model,
                        x_T=x_T,
                        sampler=sampler,
                        latent_sampler=latent_sampler,
                        conds_mean=conds_mean,
                        conds_std=conds_std,
                        clip_latent_noise=False
                    )
                except Exception as e:
                    logger.warning(f"Generation failed at sample {i+1}: {e}")
                    import traceback
                    logger.warning(traceback.format_exc())
                    continue
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            elapsed = time.time() - start_time
            times.append(elapsed)
            
            if (i + 1) % 5 == 0:
                logger.info(f"Generated {i+1}/{num_samples} samples (avg: {np.mean(times):.3f}s)")
        
        if len(times) == 0:
            logger.error("No successful generations!")
            return {
                'avg_time_per_image': 0.0,
                'total_time': 0.0,
                'std_time': 0.0,
                'min_time': 0.0,
                'max_time': 0.0,
                'num_samples': 0
            }
        
        times = np.array(times)
        result = {
            'avg_time_per_image': float(np.mean(times)),
            'total_time': float(np.sum(times)),
            'std_time': float(np.std(times)),
            'min_time': float(np.min(times)),
            'max_time': float(np.max(times)),
            'num_samples': len(times)
        }
        
        logger.info(f"=== Generation Time Analysis ===")
        logger.info(f"Average time per image: {result['avg_time_per_image']:.3f}s")
        logger.info(f"Std: {result['std_time']:.3f}s")
        logger.info(f"Min: {result['min_time']:.3f}s, Max: {result['max_time']:.3f}s")
        
        return result
    
    def analyze_data_transfer(self, model: nn.Module, input_shape: Tuple[int, int, int],
                             batch_size: int = 1, cali_images=None, cali_t=None, cali_y=None) -> Dict:
        """
        分析各层的 Data Transfer（输入输出大小）
        
        Args:
            model: 模型
            input_shape: 输入形状 (C, H, W)
            batch_size: batch size
            cali_images: 校正数据 images（如果提供，使用校正数据做 forward pass）
            cali_t: 校正数据 timesteps
            cali_y: 校正数据 conditions
        
        Returns:
            {
                'total_input_size_mb': float,
                'total_output_size_mb': float,
                'total_transfer_mb': float,
                'layer_breakdown': Dict[str, Dict]  # 每层的输入输出大小
            }
        """
        device = next(model.parameters()).device
        
        layer_transfers = {}
        hooks = []
        
        def hook_fn(name):
            def hook(module, input, output):
                input_size = sum(i.numel() * 4 for i in input if isinstance(i, torch.Tensor))  # FP32 = 4 bytes
                output_size = output.numel() * 4 if isinstance(output, torch.Tensor) else 0
                
                layer_transfers[name] = {
                    'input_size_mb': input_size / (1024 ** 2),
                    'output_size_mb': output_size / (1024 ** 2),
                    'total_size_mb': (input_size + output_size) / (1024 ** 2),
                    'layer_type': type(module).__name__
                }
            return hook
        
        # 注册 hooks（只针对量化后的 Conv2d 和 Linear）
        for name, module in model.named_modules():
            # 检查是否是量化模块中的 Conv2d 或 Linear
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Forward pass - 使用校正数据（参考量化模型建立时的做法）
        try:
            with torch.no_grad():
                if cali_images is not None and cali_t is not None and cali_y is not None:
                    # 使用校正数据（参考 sample_lora_intmodel.py 中的做法）
                    _ = model(x=cali_images[:batch_size].to(device), 
                             t=cali_t[:batch_size].to(device), 
                             cond=cali_y[:batch_size].to(device))
                else:
                    # 如果没有提供校正数据，使用随机数据
                    x = torch.randn(batch_size, *input_shape, device=device)
                    t = torch.tensor([0] * batch_size, device=device)
                    # 尝试使用 cond 参数
                    try:
                        # 生成随机 cond（如果是 latent diffusion）
                        cond = torch.randn(batch_size, 512, device=device)  # 假设 latent dim = 512
                        _ = model(x=x, t=t, cond=cond)
                    except:
                        try:
                            _ = model(x=x, t=t)
                        except:
                            _ = model(x)
        except Exception as e:
            logger.warning(f"Forward pass failed: {e}")
            import traceback
            logger.warning(traceback.format_exc())
            # 移除 hooks
            for hook in hooks:
                hook.remove()
            return {
                'total_input_size_mb': 0.0,
                'total_output_size_mb': 0.0,
                'total_transfer_mb': 0.0,
                'layer_breakdown': {}
            }
        
        # 移除 hooks
        for hook in hooks:
            hook.remove()
        
        # 计算总和
        total_input = sum(info['input_size_mb'] for info in layer_transfers.values())
        total_output = sum(info['output_size_mb'] for info in layer_transfers.values())
        total_transfer = total_input + total_output
        
        result = {
            'total_input_size_mb': total_input,
            'total_output_size_mb': total_output,
            'total_transfer_mb': total_transfer,
            'layer_breakdown': layer_transfers
        }
        
        logger.info(f"=== Data Transfer Analysis ===")
        logger.info(f"Total input size: {total_input:.2f} MB")
        logger.info(f"Total output size: {total_output:.2f} MB")
        logger.info(f"Total transfer: {total_transfer:.2f} MB")
        
        return result
    
    def analyze_timestep_embed_sequential_transfer(self, model: nn.Module, input_shape: Tuple[int, int, int],
                                                   batch_size: int = 1, cali_images=None, cali_t=None, cali_y=None) -> Dict:
        """
        分析所有 TimestepEmbedSequential 的 Data Transfer（輸入輸出大小）
        
        Args:
            model: 模型
            input_shape: 輸入形狀 (C, H, W)
            batch_size: batch size
            cali_images: 校正數據 images
            cali_t: 校正數據 timesteps
            cali_y: 校正數據 conditions
        
        Returns:
            {
                'total_input_size_mb': float,
                'total_output_size_mb': float,
                'total_transfer_mb': float,
                'layer_breakdown': Dict[str, Dict]  # 每層的輸入輸出大小
            }
        """
        from model.blocks import TimestepEmbedSequential
        
        device = next(model.parameters()).device
        layer_transfers = {}
        hooks = []
        
        def hook_fn(name):
            def hook(module, input, output):
                # input 是一個 tuple，第一個元素是 x
                input_size = 0
                if len(input) > 0 and isinstance(input[0], torch.Tensor):
                    input_size = input[0].numel() * 4  # FP32 = 4 bytes
                
                output_size = output.numel() * 4 if isinstance(output, torch.Tensor) else 0
                
                layer_transfers[name] = {
                    'input_size_mb': input_size / (1024 ** 2),
                    'output_size_mb': output_size / (1024 ** 2),
                    'total_size_mb': (input_size + output_size) / (1024 ** 2),
                    'layer_type': 'TimestepEmbedSequential'
                }
            return hook
        
        # 註冊 hooks（只針對 TimestepEmbedSequential）
        for name, module in model.named_modules():
            if isinstance(module, TimestepEmbedSequential):
                # 跳過 encoder 相關的
                if 'encoder' in name.lower():
                    continue
                hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Forward pass
        try:
            with torch.no_grad():
                if cali_images is not None and cali_t is not None and cali_y is not None:
                    _ = model(x=cali_images[:batch_size].to(device), 
                             t=cali_t[:batch_size].to(device), 
                             cond=cali_y[:batch_size].to(device))
                else:
                    x = torch.randn(batch_size, *input_shape, device=device)
                    t = torch.tensor([0] * batch_size, device=device)
                    try:
                        cond = torch.randn(batch_size, 512, device=device)
                        _ = model(x=x, t=t, cond=cond)
                    except:
                        try:
                            _ = model(x=x, t=t)
                        except:
                            _ = model(x)
        except Exception as e:
            logger.warning(f"Forward pass failed: {e}")
            import traceback
            logger.warning(traceback.format_exc())
            for hook in hooks:
                hook.remove()
            return {
                'total_input_size_mb': 0.0,
                'total_output_size_mb': 0.0,
                'total_transfer_mb': 0.0,
                'layer_breakdown': {}
            }
        
        # 移除 hooks
        for hook in hooks:
            hook.remove()
        
        # 計算總和
        total_input = sum(info['input_size_mb'] for info in layer_transfers.values())
        total_output = sum(info['output_size_mb'] for info in layer_transfers.values())
        total_transfer = total_input + total_output
        
        result = {
            'total_input_size_mb': total_input,
            'total_output_size_mb': total_output,
            'total_transfer_mb': total_transfer,
            'layer_breakdown': layer_transfers
        }
        
        logger.info(f"=== TimestepEmbedSequential Data Transfer Analysis ===")
        logger.info(f"Total input size: {total_input:.2f} MB")
        logger.info(f"Total output size: {total_output:.2f} MB")
        logger.info(f"Total transfer: {total_transfer:.2f} MB")
        logger.info(f"Number of blocks: {len(layer_transfers)}")
        
        return result
    
    def analyze_quantized_layers_transfer(self, model: nn.Module, input_shape: Tuple[int, int, int],
                                         batch_size: int = 1, num_steps: int = 20,
                                         cali_images=None, cali_t=None, cali_y=None) -> Dict:
        """
        分析所有被量化的 Conv2d/Linear 的 Data Transfer（輸入輸出大小）
        只追蹤 QuantModule_DiffAE_LoRA 和 QuantModule
        
        Args:
            model: 模型
            input_shape: 輸入形狀 (C, H, W)
            batch_size: batch size
            num_steps: 擴散步數，用於計算所有 timestep 的總和
            cali_images: 校正數據 images
            cali_t: 校正數據 timesteps
            cali_y: 校正數據 conditions
        
        Returns:
            {
                'total_input_size_mb': float,  # 所有 timestep 的總和
                'total_output_size_mb': float,  # 所有 timestep 的總和
                'total_transfer_mb': float,     # 所有 timestep 的總和
                'single_timestep_input_mb': float,  # 單一 timestep
                'single_timestep_output_mb': float, # 單一 timestep
                'single_timestep_transfer_mb': float, # 單一 timestep
                'num_steps': int,
                'layer_breakdown': Dict[str, Dict]  # 每層的輸入輸出大小（所有 timestep 總和）
            }
        """
        try:
            from QATcode.quant_model_lora import QuantModule_DiffAE_LoRA, INT_QuantModule_DiffAE_LoRA
            from QATcode.quant_layer import QuantModule, INT_QuantModule
        except ImportError:
            logger.warning("Cannot import quant modules")
            return {
                'total_input_size_mb': 0.0,
                'total_output_size_mb': 0.0,
                'total_transfer_mb': 0.0,
                'single_timestep_input_mb': 0.0,
                'single_timestep_output_mb': 0.0,
                'single_timestep_transfer_mb': 0.0,
                'num_steps': num_steps,
                'layer_breakdown': {}
            }
        
        device = next(model.parameters()).device
        layer_transfers = {}
        hooks = []
        
        def hook_fn(name):
            def hook(module, input, output):
                # 跳過 LoRA 層（只追蹤原始的 Conv2d/Linear）
                if 'lora' in name.lower() and ('loraA' in name or 'loraB' in name):
                    return
                
                input_size = sum(i.numel() * 4 for i in input if isinstance(i, torch.Tensor))  # FP32 = 4 bytes
                output_size = output.numel() * 4 if isinstance(output, torch.Tensor) else 0
                
                layer_transfers[name] = {
                    'input_size_mb': input_size / (1024 ** 2),
                    'output_size_mb': output_size / (1024 ** 2),
                    'total_size_mb': (input_size + output_size) / (1024 ** 2),
                    'layer_type': type(module).__name__
                }
            return hook
        
        # 註冊 hooks（只針對被量化的模組，排除 LoRA 層）
        for name, module in model.named_modules():
            if isinstance(module, (QuantModule_DiffAE_LoRA, QuantModule, INT_QuantModule_DiffAE_LoRA, INT_QuantModule)):
                # 跳過 LoRA 層
                if 'lora' not in name.lower() or ('loraA' not in name and 'loraB' not in name):
                    hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Forward pass
        try:
            with torch.no_grad():
                if cali_images is not None and cali_t is not None and cali_y is not None:
                    _ = model(x=cali_images[:batch_size].to(device), 
                             t=cali_t[:batch_size].to(device), 
                             cond=cali_y[:batch_size].to(device))
                else:
                    x = torch.randn(batch_size, *input_shape, device=device)
                    t = torch.tensor([0] * batch_size, device=device)
                    try:
                        cond = torch.randn(batch_size, 512, device=device)
                        _ = model(x=x, t=t, cond=cond)
                    except:
                        try:
                            _ = model(x=x, t=t)
                        except:
                            _ = model(x)
        except Exception as e:
            logger.warning(f"Forward pass failed: {e}")
            import traceback
            logger.warning(traceback.format_exc())
            for hook in hooks:
                hook.remove()
            return {
                'total_input_size_mb': 0.0,
                'total_output_size_mb': 0.0,
                'total_transfer_mb': 0.0,
                'layer_breakdown': {}
            }
        
        # 移除 hooks
        for hook in hooks:
            hook.remove()
        
        # 計算單一 timestep 的總和
        single_timestep_input = sum(info['input_size_mb'] for info in layer_transfers.values())
        single_timestep_output = sum(info['output_size_mb'] for info in layer_transfers.values())
        single_timestep_transfer = single_timestep_input + single_timestep_output
        
        # 計算所有 timestep 的總和
        total_input = single_timestep_input * num_steps
        total_output = single_timestep_output * num_steps
        total_transfer = single_timestep_transfer * num_steps
        
        # 更新 layer_breakdown 為所有 timestep 的總和
        layer_breakdown_all_steps = {}
        for layer_name, layer_info in layer_transfers.items():
            layer_breakdown_all_steps[layer_name] = {
                'input_size_mb': layer_info['input_size_mb'] * num_steps,
                'output_size_mb': layer_info['output_size_mb'] * num_steps,
                'total_size_mb': layer_info['total_size_mb'] * num_steps,
                'layer_type': layer_info['layer_type'],
                'single_timestep_input_mb': layer_info['input_size_mb'],
                'single_timestep_output_mb': layer_info['output_size_mb'],
                'single_timestep_total_mb': layer_info['total_size_mb']
            }
        
        result = {
            'total_input_size_mb': total_input,
            'total_output_size_mb': total_output,
            'total_transfer_mb': total_transfer,
            'single_timestep_input_mb': single_timestep_input,
            'single_timestep_output_mb': single_timestep_output,
            'single_timestep_transfer_mb': single_timestep_transfer,
            'num_steps': num_steps,
            'layer_breakdown': layer_breakdown_all_steps
        }
        
        logger.info(f"=== Quantized Layers Data Transfer Analysis ===")
        logger.info(f"Single timestep - Input: {single_timestep_input:.2f} MB, Output: {single_timestep_output:.2f} MB, Total: {single_timestep_transfer:.2f} MB")
        logger.info(f"All {num_steps} timesteps - Input: {total_input:.2f} MB, Output: {total_output:.2f} MB, Total: {total_transfer:.2f} MB")
        logger.info(f"Number of quantized layers: {len(layer_transfers)}")
        
        return result
    
    def analyze_quantized_layers_transfer_no_skip(self, model: nn.Module, input_shape: Tuple[int, int, int],
                                                 batch_size: int = 1, num_steps: int = 20,
                                                 cali_images=None, cali_t=None, cali_y=None) -> Dict:
        """
        分析所有被量化的 Conv2d/Linear 的 Data Transfer（排除 skip_connection）
        只追蹤 QuantModule_DiffAE_LoRA 和 QuantModule，但排除 skip_connection
        
        Args:
            model: 模型
            input_shape: 輸入形狀 (C, H, W)
            batch_size: batch size
            num_steps: 擴散步數，用於計算所有 timestep 的總和
            cali_images: 校正數據 images
            cali_t: 校正數據 timesteps
            cali_y: 校正數據 conditions
        
        Returns:
            {
                'total_input_size_mb': float,  # 所有 timestep 的總和（排除 skip_connection）
                'total_output_size_mb': float,
                'total_transfer_mb': float,
                'single_timestep_input_mb': float,
                'single_timestep_output_mb': float,
                'single_timestep_transfer_mb': float,
                'num_steps': int,
                'layer_breakdown': Dict[str, Dict]
            }
        """
        try:
            from QATcode.quant_model_lora import QuantModule_DiffAE_LoRA, INT_QuantModule_DiffAE_LoRA
            from QATcode.quant_layer import QuantModule, INT_QuantModule
        except ImportError:
            logger.warning("Cannot import quant modules")
            return {
                'total_input_size_mb': 0.0,
                'total_output_size_mb': 0.0,
                'total_transfer_mb': 0.0,
                'single_timestep_input_mb': 0.0,
                'single_timestep_output_mb': 0.0,
                'single_timestep_transfer_mb': 0.0,
                'num_steps': num_steps,
                'layer_breakdown': {}
            }
        
        device = next(model.parameters()).device
        layer_transfers = {}
        hooks = []
        
        def hook_fn(name):
            def hook(module, input, output):
                # 跳過 skip_connection 和 LoRA 層
                if 'skip_connection' in name:
                    return
                if 'lora' in name.lower() and ('loraA' in name or 'loraB' in name):
                    return
                
                input_size = sum(i.numel() * 4 for i in input if isinstance(i, torch.Tensor))  # FP32 = 4 bytes
                output_size = output.numel() * 4 if isinstance(output, torch.Tensor) else 0
                
                layer_transfers[name] = {
                    'input_size_mb': input_size / (1024 ** 2),
                    'output_size_mb': output_size / (1024 ** 2),
                    'total_size_mb': (input_size + output_size) / (1024 ** 2),
                    'layer_type': type(module).__name__
                }
            return hook
        
        # 註冊 hooks（只針對被量化的模組，排除 skip_connection 和 LoRA）
        for name, module in model.named_modules():
            if isinstance(module, (QuantModule_DiffAE_LoRA, QuantModule, INT_QuantModule_DiffAE_LoRA, INT_QuantModule)):
                # 排除 skip_connection 和 LoRA 層
                if 'skip_connection' not in name:
                    if 'lora' not in name.lower() or ('loraA' not in name and 'loraB' not in name):
                        hooks.append(module.register_forward_hook(hook_fn(name)))
        
        # Forward pass
        try:
            with torch.no_grad():
                if cali_images is not None and cali_t is not None and cali_y is not None:
                    _ = model(x=cali_images[:batch_size].to(device), 
                             t=cali_t[:batch_size].to(device), 
                             cond=cali_y[:batch_size].to(device))
                else:
                    x = torch.randn(batch_size, *input_shape, device=device)
                    t = torch.tensor([0] * batch_size, device=device)
                    try:
                        cond = torch.randn(batch_size, 512, device=device)
                        _ = model(x=x, t=t, cond=cond)
                    except:
                        try:
                            _ = model(x=x, t=t)
                        except:
                            _ = model(x)
        except Exception as e:
            logger.warning(f"Forward pass failed: {e}")
            import traceback
            logger.warning(traceback.format_exc())
            for hook in hooks:
                hook.remove()
            return {
                'total_input_size_mb': 0.0,
                'total_output_size_mb': 0.0,
                'total_transfer_mb': 0.0,
                'single_timestep_input_mb': 0.0,
                'single_timestep_output_mb': 0.0,
                'single_timestep_transfer_mb': 0.0,
                'num_steps': num_steps,
                'layer_breakdown': {}
            }
        
        # 移除 hooks
        for hook in hooks:
            hook.remove()
        
        # 計算單一 timestep 的總和（排除 skip_connection）
        single_timestep_input = sum(info['input_size_mb'] for info in layer_transfers.values())
        single_timestep_output = sum(info['output_size_mb'] for info in layer_transfers.values())
        single_timestep_transfer = single_timestep_input + single_timestep_output
        
        # 計算所有 timestep 的總和
        total_input = single_timestep_input * num_steps
        total_output = single_timestep_output * num_steps
        total_transfer = single_timestep_transfer * num_steps
        
        # 更新 layer_breakdown 為所有 timestep 的總和
        layer_breakdown_all_steps = {}
        for layer_name, layer_info in layer_transfers.items():
            layer_breakdown_all_steps[layer_name] = {
                'input_size_mb': layer_info['input_size_mb'] * num_steps,
                'output_size_mb': layer_info['output_size_mb'] * num_steps,
                'total_size_mb': layer_info['total_size_mb'] * num_steps,
                'layer_type': layer_info['layer_type'],
                'single_timestep_input_mb': layer_info['input_size_mb'],
                'single_timestep_output_mb': layer_info['output_size_mb'],
                'single_timestep_total_mb': layer_info['total_size_mb']
            }
        
        result = {
            'total_input_size_mb': total_input,
            'total_output_size_mb': total_output,
            'total_transfer_mb': total_transfer,
            'single_timestep_input_mb': single_timestep_input,
            'single_timestep_output_mb': single_timestep_output,
            'single_timestep_transfer_mb': single_timestep_transfer,
            'num_steps': num_steps,
            'layer_breakdown': layer_breakdown_all_steps
        }
        
        logger.info(f"=== Quantized Layers Data Transfer (No Skip Connection) Analysis ===")
        logger.info(f"Single timestep - Input: {single_timestep_input:.2f} MB, Output: {single_timestep_output:.2f} MB, Total: {single_timestep_transfer:.2f} MB")
        logger.info(f"All {num_steps} timesteps - Input: {total_input:.2f} MB, Output: {total_output:.2f} MB, Total: {total_transfer:.2f} MB")
        logger.info(f"Number of quantized layers (excluding skip_connection): {len(layer_transfers)}")
        
        return result
    
    def analyze_cached_quantized_layers_transfer(self, model: nn.Module, input_shape: Tuple[int, int, int],
                                                 cache_scheduler: Optional[Dict], num_steps: int,
                                                 sampler=None, conf=None, latent_sampler=None,
                                                 conds_mean=None, conds_std=None,
                                                 batch_size: int = 1, cali_images=None, cali_t=None, cali_y=None) -> Dict:
        """
        分析使用 cache 後，有執行計算的量化層的 Data Transfer（輸入輸出大小）
        需要模擬整個擴散過程，追蹤每個 timestep 哪些層有執行計算
        
        Args:
            model: 模型
            input_shape: 輸入形狀 (C, H, W)
            cache_scheduler: cache scheduler 字典，格式為 {layer_name: Set[int]}，表示哪些 timestep 需要重新計算
            num_steps: 擴散步數
            batch_size: batch size
            cali_images: 校正數據 images
            cali_t: 校正數據 timesteps
            cali_y: 校正數據 conditions
        
        Returns:
            {
                'total_input_size_mb': float,
                'total_output_size_mb': float,
                'total_transfer_mb': float,
                'layer_breakdown': Dict[str, Dict]  # 每層的輸入輸出大小（累積所有 timestep）
                'timestep_breakdown': Dict[int, Dict]  # 每個 timestep 的統計
            }
        """
        try:
            from QATcode.quant_model_lora import QuantModule_DiffAE_LoRA
            from QATcode.quant_layer import QuantModule
        except ImportError:
            logger.warning("Cannot import QuantModule_DiffAE_LoRA or QuantModule")
            return {
                'total_input_size_mb': 0.0,
                'total_output_size_mb': 0.0,
                'total_transfer_mb': 0.0,
                'layer_breakdown': {},
                'timestep_breakdown': {}
            }
        
        device = next(model.parameters()).device
        
        # 建立 layer name 到 module 的映射，以及 layer name 到 cache scheduler key 的映射
        layer_name_to_module = {}
        layer_name_to_cache_key = {}
        
        # 首先找到所有量化層
        for name, module in model.named_modules():
            if isinstance(module, (QuantModule_DiffAE_LoRA, QuantModule)):
                layer_name_to_module[name] = module
        
        # 解析 cache_scheduler（可能是字符串格式的集合）
        if cache_scheduler is None:
            logger.warning("cache_scheduler is None, returning empty result")
            return {
                'total_input_size_mb': 0.0,
                'total_output_size_mb': 0.0,
                'total_transfer_mb': 0.0,
                'layer_breakdown': {},
                'timestep_breakdown': {}
            }
        
        # 轉換 cache_scheduler 格式：如果是字符串，解析為 Set
        parsed_cache_scheduler = {}
        for key, value in cache_scheduler.items():
            if isinstance(value, str):
                # 解析字符串格式的集合，如 "{0, 1, 2, 3}"
                import ast
                try:
                    parsed_cache_scheduler[key] = ast.literal_eval(value)
                except:
                    # 如果解析失敗，嘗試手動解析
                    value = value.strip('{}')
                    parsed_cache_scheduler[key] = set(int(x.strip()) for x in value.split(',') if x.strip())
            elif isinstance(value, (set, list)):
                parsed_cache_scheduler[key] = set(value) if isinstance(value, list) else value
            else:
                parsed_cache_scheduler[key] = value
        
        cache_scheduler = parsed_cache_scheduler
        
        # 簡化：假設 cache_scheduler 的 key 對應到 TimestepEmbedSequential
        # 我們需要追蹤每個 TimestepEmbedSequential 內部的量化層
        from model.blocks import TimestepEmbedSequential
        
        # 建立 TimestepEmbedSequential 到其內部量化層的映射
        sequential_to_quantized_layers = {}
        for seq_name, seq_module in model.named_modules():
            if isinstance(seq_module, TimestepEmbedSequential):
                if 'encoder' in seq_name.lower():
                    continue
                quantized_in_seq = []
                for layer_name, layer_module in model.named_modules():
                    if layer_name.startswith(seq_name) and isinstance(layer_module, (QuantModule_DiffAE_LoRA, QuantModule)):
                        quantized_in_seq.append(layer_name)
                sequential_to_quantized_layers[seq_name] = quantized_in_seq
        
        # 建立 cache scheduler key 到 TimestepEmbedSequential name 的映射
        # 這需要根據實際的命名規則來匹配
        cache_key_to_sequential = {}
        for cache_key in cache_scheduler.keys():
            # 嘗試匹配 TimestepEmbedSequential
            # 例如 "encoder_layer_0" 可能對應到 "model.input_blocks.0" 等
            # 這裡需要根據實際的模型結構來調整
            matched_sequential = None
            for seq_name in sequential_to_quantized_layers.keys():
                # 簡化匹配邏輯：根據 cache_key 和 seq_name 的結構來匹配
                # 實際實現可能需要更複雜的邏輯
                if 'input_blocks' in seq_name or 'output_blocks' in seq_name or 'middle_block' in seq_name:
                    # 提取索引
                    try:
                        if 'encoder_layer' in cache_key:
                            idx = int(cache_key.split('_')[-1])
                            if 'input_blocks' in seq_name:
                                # 需要根據實際結構來匹配
                                matched_sequential = seq_name
                        elif 'decoder_layer' in cache_key:
                            idx = int(cache_key.split('_')[-1])
                            if 'output_blocks' in seq_name:
                                matched_sequential = seq_name
                        elif 'middle_layer' in cache_key:
                            if 'middle_block' in seq_name:
                                matched_sequential = seq_name
                    except:
                        pass
            if matched_sequential:
                cache_key_to_sequential[cache_key] = matched_sequential
        
        # 如果無法建立映射，使用簡化方法：追蹤所有量化層，但只在對應的 timestep 記錄
        # 這裡我們使用一個更簡單的方法：追蹤所有量化層，並在 forward pass 時檢查是否應該記錄
        
        layer_transfers = {}  # {layer_name: {'input_size_mb': float, 'output_size_mb': float, ...}}
        timestep_transfers = {}  # {timestep: {'layers': int, 'input_size_mb': float, ...}}
        
        hooks = []
        current_timestep = [0]  # 使用 list 以便在 hook 中修改
        
        def hook_fn(name, cache_key=None):
            def hook(module, input, output):
                # 檢查當前 timestep 是否需要記錄這個層
                should_record = True
                if cache_key is not None and cache_scheduler is not None:
                    # 檢查這個 cache_key 在當前 timestep 是否需要重新計算
                    if cache_key in cache_scheduler:
                        recompute_timesteps = cache_scheduler[cache_key]
                        # 轉換 timestep：擴散過程是從 T-1 到 0，但 cache_scheduler 可能是正序
                        # 需要根據實際情況調整
                        if current_timestep[0] not in recompute_timesteps:
                            should_record = False
                
                if should_record:
                    input_size = sum(i.numel() * 4 for i in input if isinstance(i, torch.Tensor))
                    output_size = output.numel() * 4 if isinstance(output, torch.Tensor) else 0
                    
                    if name not in layer_transfers:
                        layer_transfers[name] = {
                            'input_size_mb': 0.0,
                            'output_size_mb': 0.0,
                            'total_size_mb': 0.0,
                            'layer_type': type(module).__name__,
                            'execution_count': 0
                        }
                    
                    layer_transfers[name]['input_size_mb'] += input_size / (1024 ** 2)
                    layer_transfers[name]['output_size_mb'] += output_size / (1024 ** 2)
                    layer_transfers[name]['total_size_mb'] += (input_size + output_size) / (1024 ** 2)
                    layer_transfers[name]['execution_count'] += 1
                    
                    # 記錄 timestep 統計
                    t = current_timestep[0]
                    if t not in timestep_transfers:
                        timestep_transfers[t] = {
                            'layers_executed': 0,
                            'input_size_mb': 0.0,
                            'output_size_mb': 0.0
                        }
                    timestep_transfers[t]['layers_executed'] += 1
                    timestep_transfers[t]['input_size_mb'] += input_size / (1024 ** 2)
                    timestep_transfers[t]['output_size_mb'] += output_size / (1024 ** 2)
            return hook
        
        # 註冊 hooks
        for name, module in model.named_modules():
            if isinstance(module, (QuantModule_DiffAE_LoRA, QuantModule)):
                # 嘗試找到對應的 cache_key
                cache_key = None
                for seq_name, quantized_layers in sequential_to_quantized_layers.items():
                    if name in quantized_layers:
                        for ck, seq in cache_key_to_sequential.items():
                            if seq == seq_name:
                                cache_key = ck
                                break
                        break
                hooks.append(module.register_forward_hook(hook_fn(name, cache_key)))
        
        # 模擬整個擴散過程
        logger.info("Simulating diffusion process with cache scheduler...")
        
        # 使用 sampler 運行完整的擴散過程
        try:
            with torch.no_grad():
                if sampler is not None and conf is not None:
                    # 使用實際的擴散過程
                    from renderer import render_uncondition
                    
                    # 準備輸入
                    if cali_images is not None:
                        x_T = cali_images[:batch_size].to(device)
                    else:
                        x_T = torch.randn(batch_size, *input_shape, device=device)
                    
                    # 包裝模型的 forward 方法來追蹤 timestep
                    original_forward = model.forward
                    
                    def wrapped_forward(*args, **kwargs):
                        # 從參數中提取 timestep
                        timestep = None
                        if len(args) >= 2:
                            timestep = args[1]
                        elif 't' in kwargs:
                            timestep = kwargs['t']
                        
                        if timestep is not None:
                            if torch.is_tensor(timestep):
                                if timestep.numel() > 1:
                                    timestep_val = timestep[0].item()
                                else:
                                    timestep_val = timestep.item()
                            else:
                                timestep_val = timestep
                            
                            # 轉換 timestep：擴散過程是從 T-1 到 0
                            # cache_scheduler 中的 timestep 是正序（0 到 T-1）
                            # 需要轉換
                            current_timestep[0] = timestep_val
                        
                        return original_forward(*args, **kwargs)
                    
                    # 臨時替換 forward 方法
                    model.forward = wrapped_forward
                    
                    try:
                        # 運行擴散過程
                        _ = render_uncondition(
                            conf=conf,
                            model=model,
                            x_T=x_T,
                            sampler=sampler,
                            latent_sampler=latent_sampler,
                            conds_mean=conds_mean,
                            conds_std=conds_std,
                            clip_latent_noise=False
                        )
                    finally:
                        # 恢復原始 forward 方法
                        model.forward = original_forward
                else:
                    # 簡化實現：運行單次 forward pass
                    logger.warning("sampler or conf not provided, using simplified single forward pass")
                    if cali_images is not None and cali_t is not None and cali_y is not None:
                        current_timestep[0] = cali_t[0].item() if torch.is_tensor(cali_t[0]) else cali_t[0]
                        _ = model(x=cali_images[:batch_size].to(device), 
                                 t=cali_t[:batch_size].to(device), 
                                 cond=cali_y[:batch_size].to(device))
                    else:
                        x = torch.randn(batch_size, *input_shape, device=device)
                        t_tensor = torch.tensor([0] * batch_size, device=device)
                        current_timestep[0] = 0
                        try:
                            cond = torch.randn(batch_size, 512, device=device)
                            _ = model(x=x, t=t_tensor, cond=cond)
                        except:
                            try:
                                _ = model(x=x, t=t_tensor)
                            except:
                                _ = model(x)
        except Exception as e:
            logger.warning(f"Forward pass failed: {e}")
            import traceback
            logger.warning(traceback.format_exc())
            for hook in hooks:
                hook.remove()
            return {
                'total_input_size_mb': 0.0,
                'total_output_size_mb': 0.0,
                'total_transfer_mb': 0.0,
                'layer_breakdown': {},
                'timestep_breakdown': {}
            }
        
        # 移除 hooks
        for hook in hooks:
            hook.remove()
        
        # 計算總和
        total_input = sum(info['input_size_mb'] for info in layer_transfers.values())
        total_output = sum(info['output_size_mb'] for info in layer_transfers.values())
        total_transfer = total_input + total_output
        
        result = {
            'total_input_size_mb': total_input,
            'total_output_size_mb': total_output,
            'total_transfer_mb': total_transfer,
            'layer_breakdown': layer_transfers,
            'timestep_breakdown': timestep_transfers
        }
        
        logger.info(f"=== Cached Quantized Layers Data Transfer Analysis ===")
        logger.info(f"Total input size: {total_input:.2f} MB")
        logger.info(f"Total output size: {total_output:.2f} MB")
        logger.info(f"Total transfer: {total_transfer:.2f} MB")
        logger.info(f"Number of quantized layers executed: {len([k for k, v in layer_transfers.items() if v['execution_count'] > 0])}")
        
        return result
    
    def run_full_analysis(self, original_model: Optional[nn.Module], quantized_model: nn.Module,
                         sampler, conf, config: Dict, latent_sampler=None,
                         conds_mean=None, conds_std=None, 
                         skip_original_time_and_transfer: bool = False,
                         cali_images=None, cali_t=None, cali_y=None) -> Dict:
        """
        运行完整分析
        
        Args:
            original_model: 原始 FP32 模型（可以为 None，如果跳过 original 分析）
            quantized_model: 量化模型
            sampler: 采样器
            conf: 配置对象
            config: 配置字典，包含 steps, cache_method, threshold 等
            skip_original_time_and_transfer: 如果为 True，跳过原始模型的生成时间和 data transfer 分析
                                            （用于同一组 step+method 只执行一次）
            cali_images: 校正数据 images（用于 data transfer 分析）
            cali_t: 校正数据 timesteps
            cali_y: 校正数据 conditions
        """
        logger.info("=" * 80)
        logger.info("Starting Full Quantitative Analysis")
        logger.info(f"Config: {json.dumps(config, indent=2, default=str)}")
        logger.info("=" * 80)
        
        results = {
            'config': config,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 1. 模型大小和参数量
        logger.info("\n[1/4] Analyzing model size and parameters...")
        if original_model is not None:
            results['original_model'] = self.analyze_model_size_and_params(
                original_model, 'Original Diff-AE'
            )
        else:
            # 如果跳过 original 分析，使用占位符
            results['original_model'] = {'model_size_mb': 0.0, 'quantized_size_mb': 0.0}
        results['quantized_model'] = self.analyze_model_size_and_params(
            quantized_model, 'Q-DiffAE'
        )
        
        # 计算压缩比
        if original_model is not None and results['original_model']['model_size_mb'] > 0:
            compression_ratio = results['original_model']['model_size_mb'] / results['quantized_model']['quantized_size_mb']
            results['compression_ratio'] = compression_ratio
            logger.info(f"Compression ratio: {compression_ratio:.2f}x")
        else:
            results['compression_ratio'] = 0.0
        
        # 2. MACs 分析
        logger.info("\n[2/4] Analyzing MACs...")
        input_shape = (3, 128, 128)  # 根据实际调整
        if original_model is not None:
            results['macs_original'] = self.calculate_macs(
                original_model, input_shape, timestep=0, use_cache=False
            )
        else:
            results['macs_original'] = {'total_macs': 0}
        results['macs_quantized'] = self.calculate_macs(
            quantized_model, input_shape, timestep=0, use_cache=False
        )
        
        if config.get('enable_cache', False):
            cache_scheduler = config.get('cache_scheduler', None)
            results['macs_quantized_cache'] = self.calculate_macs(
                quantized_model, input_shape, timestep=0,
                use_cache=True, cache_scheduler=cache_scheduler
            )
        
        # 3. 生成时间
        logger.info("\n[3/4] Measuring generation time...")
        num_samples = config.get('num_samples', 10)
        
        # 只在需要时执行原始模型的生成时间分析
        if not skip_original_time_and_transfer and original_model is not None:
            results['time_original'] = self.measure_generation_time(
                sampler, original_model, conf, num_samples=num_samples, device=config.get('device'),
                latent_sampler=latent_sampler, conds_mean=conds_mean, conds_std=conds_std
            )
        else:
            if skip_original_time_and_transfer:
                logger.info("Skipping original model generation time (already measured for this step+method)")
            else:
                logger.info("Skipping original model generation time (original_model is None)")
            results['time_original'] = None
        
        results['time_quantized'] = self.measure_generation_time(
            sampler, quantized_model, conf, num_samples=num_samples, device=config.get('device'),
            latent_sampler=latent_sampler, conds_mean=conds_mean, conds_std=conds_std
        )
        
        if config.get('enable_cache', False):
            results['time_quantized_cache'] = self.measure_generation_time(
                sampler, quantized_model, conf, num_samples=num_samples, device=config.get('device'),
                latent_sampler=latent_sampler, conds_mean=conds_mean, conds_std=conds_std
            )
        
        # 4. Data Transfer
        logger.info("\n[4/4] Analyzing data transfer...")
        
        # 只在需要时执行原始模型的 data transfer 分析
        if not skip_original_time_and_transfer and original_model is not None:
            results['data_transfer_original'] = self.analyze_data_transfer(
                original_model, input_shape, 
                cali_images=cali_images, cali_t=cali_t, cali_y=cali_y
            )
        else:
            if skip_original_time_and_transfer:
                logger.info("Skipping original model data transfer (already measured for this step+method)")
            else:
                logger.info("Skipping original model data transfer (original_model is None)")
            results['data_transfer_original'] = None
        
        results['data_transfer_quantized'] = self.analyze_data_transfer(
            quantized_model, input_shape,
            cali_images=cali_images, cali_t=cali_t, cali_y=cali_y
        )
        
        # 5. 新的 Data Transfer 分析
        logger.info("\n[5/7] Analyzing TimestepEmbedSequential data transfer...")
        results['data_transfer_timestep_embed_sequential'] = self.analyze_timestep_embed_sequential_transfer(
            quantized_model, input_shape,
            cali_images=cali_images, cali_t=cali_t, cali_y=cali_y
        )
        
        logger.info("\n[6/7] Analyzing quantized layers data transfer...")
        num_steps = config.get('num_steps', 20)
        results['data_transfer_quantized_layers'] = self.analyze_quantized_layers_transfer(
            quantized_model, input_shape, num_steps=num_steps,
            cali_images=cali_images, cali_t=cali_t, cali_y=cali_y
        )
        
        logger.info("\n[6.5/7] Analyzing quantized layers data transfer (no skip connection)...")
        results['data_transfer_quantized_layers_no_skip'] = self.analyze_quantized_layers_transfer_no_skip(
            quantized_model, input_shape, num_steps=num_steps,
            cali_images=cali_images, cali_t=cali_t, cali_y=cali_y
        )
        
        if config.get('enable_cache', False):
            logger.info("\n[7/7] Analyzing cached quantized layers data transfer...")
            cache_scheduler = config.get('cache_scheduler', None)
            num_steps = config.get('num_steps', 100)
            results['data_transfer_cached_quantized_layers'] = self.analyze_cached_quantized_layers_transfer(
                quantized_model, input_shape, cache_scheduler, num_steps,
                sampler=sampler, conf=conf, latent_sampler=latent_sampler,
                conds_mean=conds_mean, conds_std=conds_std,
                cali_images=cali_images, cali_t=cali_t, cali_y=cali_y
            )
        else:
            results['data_transfer_cached_quantized_layers'] = None
        
        # 保存结果
        self.results = results
        self.save_results(config)
        
        logger.info("\n" + "=" * 80)
        logger.info("Analysis Complete!")
        logger.info("=" * 80)
        
        return results
    
    def save_results(self, config: Dict):
        """保存结果到 JSON"""
        output_file = self.log_file.replace('.log', '.json')
        # 确保目录存在
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        
        # 转换不可序列化的对象
        def convert_to_serializable(obj):
            if isinstance(obj, (torch.Tensor, np.ndarray)):
                return obj.tolist() if hasattr(obj, 'tolist') else str(obj)
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            else:
                return str(obj)
        
        serializable_results = convert_to_serializable(self.results)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        logger.info(f"Results saved to {output_file}")

