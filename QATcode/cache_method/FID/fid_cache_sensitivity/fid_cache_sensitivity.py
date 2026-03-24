"""
Diff-AE FID Cache Sensitivity Analysis
用於測試每個 UNet block 對 FID 的 cache 敏感度
"""

from copy import deepcopy
import os
import sys
import time
import logging
# === DEBUG/DIAG ===
import math
import shutil
from contextlib import contextmanager
from dataclasses import dataclass

import random
from typing import Tuple, List, Dict, Any, Optional, Union, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# 添加專案路徑
sys.path.append(".")
sys.path.append("./model")

from QATcode.quantize_ver2.quant_model_lora_v2 import QuantModel_DiffAE_LoRA
from QATcode.quantize_ver2.quant_model_lora_v2 import QuantModule_DiffAE_LoRA, INT_QuantModel_DiffAE_LoRA, INT_QuantModule_DiffAE_LoRA
from QATcode.quantize_ver2.quant_layer_v2 import QuantModule, SimpleDequantizer
from QATcode.quantize_ver2.quant_dataset_v2 import DiffusionInputDataset
#rom QATcode.quantize_ver2.diffae_trainer_v2 import *
from diffusion.diffusion import _WrappedModel
from model.unet_autoenc import BeatGANsAutoencModel
from experiment import *
from templates import *
from templates_latent import *

from torchvision.utils import make_grid, save_image
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
#from thop import profile, clever_format
from model.blocks import QKVAttention, QKVAttentionLegacy
from model.nn import timestep_embedding

# JSON 處理
import json
from datetime import datetime

# matrix calculation
#torch.set_float32_matmul_precision('highest')
#torch.backends.cudnn.benchmark = True
#=============================================================================
# 配置與常量
#=============================================================================

# 配置
class ExperimentConfig:
    """FID Cache Sensitivity 實驗配置"""
    # 硬體設定
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    GPU_ID = '0'
    SEED = 0
    
    # 實驗參數
    NUM_DIFFUSION_STEPS = 20  # 預設 20 steps
    EVAL_SAMPLES = 5_000  # FID@5k
    BATCH_SIZE = 12
    LORA_RANK = 32
    MODE = 'train'
    
    # 量化參數
    N_BITS_W = 8
    N_BITS_A = 8
    
    # 文件路徑
    MODEL_PATH = "checkpoints/ffhq128_autoenc_latent/last.ckpt"
    BEST_CKPT_PATH = "QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best.pth"
    CALIB_DATA_PATH = "QATcode/quantize_ver2/calibration_diffae.pth"
    CALIB_SAMPLES = 1024
    
    # 輸出路徑
    OUTPUT_DIR = "QATcode/cache_method/FID/fid_cache_sensitivity"
    RESULTS_JSON = "fid_sensitivity_results.json"
    LOG_FILE = 'QATcode/cache_method/FID/fid_cache_sensitivity/fid_sensitivity.log'

    
    

class AverageMeter:
    def __init__(self):
        self.reset()
    def reset(self):
        self.sum = 0.0
        self.cnt = 0
    @property
    def avg(self):
        return self.sum / max(1, self.cnt)
    def update(self, val, n=1):
        self.sum += float(val) * n
        self.cnt += n


# 初始化全局配置
CONFIG = ExperimentConfig()
LOGGER = logging.getLogger("FIDSensitivity")

#=============================================================================
# 工具函數
#=============================================================================

def _seed_all(seed: int) -> None:
    """統一設定 random/numpy/torch 的隨機種子。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def print_trainable_parameters(model: nn.Module) -> Tuple[int, int]:
    """
    打印可訓練參數統計
    
    Args:
        model: 要分析的模型
        
    Returns:
        (trainable_params, all_param): 可訓練參數數量和總參數數量
    """
    trainable_params = 0
    all_param = 0
    
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            
    LOGGER.info(f"可訓練參數: {trainable_params:,} || "
          f"總參數: {all_param:,} || "
          f"可訓練比例: {100 * trainable_params / all_param:.2f}%")
    
    return trainable_params, all_param



def time_operation(func: Callable) -> Callable:
    """
    用於測量函數執行時間的裝飾器
    
    Args:
        func: 要測量的函數
        
    Returns:
        包裝後的函數，會打印執行時間
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start_time
        LOGGER.info(f"執行 '{func.__name__}' 完成，耗時: {elapsed:.2f} 秒")
        return result
    return wrapper

#=============================================================================
# 模型載入與創建函數
#=============================================================================

@time_operation
def load_diffae_model(model_path: str = CONFIG.MODEL_PATH) -> LitModel:
    """
    載入預訓練的 Diff-AE 模型
    
    Args:
        model_path: 檢查點路徑
        
    Returns:
        LitModel: 加載的擴散模型
    """
    LOGGER.info(f"載入 Diff-AE 模型: {model_path}")
    
    # 載入檢查點
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    conf = ffhq128_autoenc_latent()
    model = LitModel(conf)
    
    # 載入權重
    model.load_state_dict(ckpt['state_dict'], strict=False)
    LOGGER.info("Diff-AE 模型載入完成")
    
    return model

@time_operation
def create_float_quantized_model(
    diffusion_model: BeatGANsAutoencModel,
    num_steps: int = CONFIG.NUM_DIFFUSION_STEPS, 
    lora_rank: int = CONFIG.LORA_RANK,
    mode: str = 'train'
) -> QuantModel_DiffAE_LoRA:
    """
    創建量化模型並載入 Step 5 的整數權重
    
    Args:
        diffusion_model: 基礎擴散模型
        num_steps: 時間步數量，用於 TALSQ
        lora_rank: LoRA 低秩適應的秩
        
    Returns:
        QuantModel_DiffAE_LoRA: 量化後的模型
    """
    LOGGER.info("=== 創建 LoRA 量化模型 ===")
    
    # 量化參數設定
    wq_params = {
        'n_bits': CONFIG.N_BITS_W, 
        'channel_wise': True, 
        'scale_method': 'mse' 
    }
    aq_params = {
        'n_bits': CONFIG.N_BITS_A, 
        'channel_wise': False, 
        'scale_method': 'max',  
        'leaf_param': True
    }
    
    # 創建量化模型
    quant_model = QuantModel_DiffAE_LoRA(
        model=diffusion_model,
        weight_quant_params=wq_params,
        act_quant_params=aq_params,
        num_steps=num_steps,
        lora_rank=lora_rank,
        mode=mode
    )

    quant_model.eval()
    
    return quant_model

def create_int_quantized_model(
    diffusion_model: BeatGANsAutoencModel,
    num_steps: int = CONFIG.NUM_DIFFUSION_STEPS, 
    lora_rank: int = CONFIG.LORA_RANK,
    mode: str = 'train'
) -> INT_QuantModel_DiffAE_LoRA:
    """
    創建量化模型並載入 Step 5 的整數權重
    
    Args:
        diffusion_model: 基礎擴散模型
        num_steps: 時間步數量，用於 TALSQ
        lora_rank: LoRA 低秩適應的秩
        
    Returns:
        QuantModel_DiffAE_LoRA: 量化後的模型
    """
    LOGGER.info("=== 創建 LoRA 量化模型 ===")
    
    # 量化參數設定
    wq_params = {
        'n_bits': CONFIG.N_BITS_W, 
        'channel_wise': True, 
        'scale_method': 'mse' 
    }
    aq_params = {
        'n_bits': CONFIG.N_BITS_A, 
        'channel_wise': False, 
        'scale_method': 'max',  
        'leaf_param': True
    }
    
    # 創建量化模型
    quant_model = INT_QuantModel_DiffAE_LoRA(
        model=diffusion_model,
        weight_quant_params=wq_params,
        act_quant_params=aq_params,
        num_steps=num_steps,
    )

    quant_model.eval()
    
    return quant_model

#=============================================================================
# 資料處理函數
#=============================================================================

def get_train_samples(
    train_loader: DataLoader, 
    num_samples: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    從資料載入器中獲取指定數量樣本
    
    Args:
        train_loader: 資料載入器
        num_samples: 需要獲取的樣本數量
        
    Returns:
        (image_tensor, t_tensor, y_tensor): 批次資料，時間步，條件標籤
    """
    image_data, t_data, y_data = [], [], []
    for (image, t, y) in train_loader:
        image_data.append(image)
        t_data.append(t)
        y_data.append(y)
        if len(image_data) >= num_samples:
            break
    
    return (
        torch.cat(image_data, dim=0)[:num_samples], 
        torch.cat(t_data, dim=0)[:num_samples], 
        torch.cat(y_data, dim=0)[:num_samples]
    )

def load_calibration_data() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    載入或生成校準資料
    
    Returns:
        (images, timesteps, conditions): 校準資料張量
    """
    LOGGER.info(f"載入校準資料...")
    
    try:
        dataset = DiffusionInputDataset(CONFIG.CALIB_DATA_PATH)
        data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
        cali_images, cali_t, cali_y = get_train_samples(
            data_loader, 
            num_samples=CONFIG.CALIB_SAMPLES
        )
        LOGGER.info("✅ 載入真實校準資料成功")
    except Exception as e:
        LOGGER.warning(f"⚠️ 載入校準資料失敗: {e}")
        LOGGER.info("使用合成校準資料")
        cali_images = torch.randn(CONFIG.CALIB_SAMPLES, 3, 128, 128)
        cali_t = torch.randint(0, CONFIG.NUM_DIFFUSION_STEPS, (CONFIG.CALIB_SAMPLES,))
        cali_y = torch.randint(0, 1000, (CONFIG.CALIB_SAMPLES,))
    
    return cali_images, cali_t, cali_y

@torch.no_grad()
def sync_ema_once(base_model: LitModel):
    model = base_model.model
    ema_model = copy.deepcopy(model)
    setattr(base_model, 'ema_model', ema_model)
    return base_model

def make_state_dict(m: torch.nn.Module, drop_uint8: bool = True):
    """輸出乾淨的 state_dict；預設移除 uint8 權重（你之後另行導出 INT8 時再存）。"""
    out = {}
    for k, v in m.state_dict().items():
        if drop_uint8 and getattr(v, "dtype", None) == torch.uint8:
            continue
        out[k] = v.detach().cpu()
    return out

def remap_keys(sd, drop_prefix=None, add_prefix=None):
    out = {}
    for k, v in sd.items():
        if drop_prefix and k.startswith(drop_prefix):
            k = k[len(drop_prefix):]
        if add_prefix:
            k = add_prefix + k
        out[k] = v
    return out

#=============================================================================
# FID Cache Sensitivity 核心函數
#=============================================================================

def get_all_layer_names() -> List[str]:
    """
    返回所有 UNet layer 的名稱列表
    
    Returns:
        List[str]: 包含 31 個 layer 名稱的列表
    """
    layers = []
    # Encoder layers: 0-14
    for i in range(15):
        layers.append(f'encoder_layer_{i}')
    # Middle layer
    layers.append('middle_layer')
    # Decoder layers: 0-14
    for i in range(15):
        layers.append(f'decoder_layer_{i}')
    return layers


def create_simple_cache_config(layer_name: str, k: int, total_steps: int) -> dict:
    """
    為單一 layer 生成簡單的 cache 配置
    
    Args:
        layer_name: 要 cache 的 layer 名稱 (e.g., 'encoder_layer_5')
        k: cache frequency (每 k 個 timestep 重新計算一次)
        total_steps: 總 timestep 數 (20 或 100)
    
    Returns:
        dict: cache_scheduler 配置
              格式: {layer_name: set of timesteps to recompute}
    """
    # 生成要重新計算的 timestep (0, k, 2k, 3k, ...)
    recompute_steps = set(range(0, total_steps, k))
    
    # DDIM 是倒序執行的（從 total_steps-1 到 0）
    # 必須確保第一個執行的 timestep（total_steps-1）也要重新計算
    # 否則會嘗試讀取不存在的 cache
    recompute_steps.add(total_steps - 1)
    
    # 獲取所有 layer
    all_layers = get_all_layer_names()
    
    cache_scheduler = {}
    for layer in all_layers:
        if layer == layer_name:
            # 目標 layer: 只在指定的 timestep 重新計算，其他使用 cache
            cache_scheduler[layer] = recompute_steps
        else:
            # 其他 layer: 每個 timestep 都計算 (不 cache)
            cache_scheduler[layer] = set(range(total_steps))
    
    return cache_scheduler


def load_results(json_path: str) -> dict:
    """
    載入已有的實驗結果
    
    Args:
        json_path: JSON 檔案路徑
    
    Returns:
        dict: 實驗結果字典
    """
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    else:
        return {
            "config": {
                "eval_samples": CONFIG.EVAL_SAMPLES,
            },
            "results": {}
        }


def save_results(results: dict, json_path: str):
    """
    保存實驗結果 (支援增量更新)
    
    Args:
        results: 實驗結果字典
        json_path: JSON 檔案路徑
    """
    results["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    LOGGER.info(f"✅ 結果已保存至: {json_path}")


def should_skip_experiment(results: dict, step_config: str, k: int, layer: str) -> bool:
    """
    檢查該實驗是否已經跑過
    
    Args:
        results: 當前結果字典
        step_config: 'T20' 或 'T100'
        k: cache frequency
        layer: layer 名稱
    
    Returns:
        bool: True 表示應該跳過
    """
    if step_config not in results.get("results", {}):
        return False
    if f'k{k}' not in results["results"][step_config]:
        return False
    return layer in results["results"][step_config][f'k{k}']


@torch.no_grad()
def evaluate_fid_with_cache(
    base_model: LitModel,
    cache_scheduler: Optional[dict] = None,
    num_steps: int = None
) -> float:
    """
    生成圖片並計算 FID
    
    Args:
        base_model: Diff-AE 模型
        cache_scheduler: cache 配置 (None 表示不使用 cache)
        num_steps: diffusion steps (如果為 None 則使用 CONFIG.NUM_DIFFUSION_STEPS)
    
    Returns:
        float: FID 分數
    """
    if num_steps is None:
        num_steps = CONFIG.NUM_DIFFUSION_STEPS
    
    # 克隆配置以避免影響原始模型
    conf = base_model.conf.clone()
    
    # 設定 cache scheduler
    if cache_scheduler is not None:
        conf.cache_scheduler = cache_scheduler
        LOGGER.info("✅ Cache scheduler 已設定")
    else:
        conf.cache_scheduler = None
        LOGGER.info("🚫 不使用 cache (baseline)")
    
    # 設定評估樣本數
    conf.eval_num_images = CONFIG.EVAL_SAMPLES
    #conf.eval_num_images = 128

    # 設定 batch_size_eval
    #conf.batch_size_eval = 64
    
    # 創建 sampler
    sampler = conf._make_diffusion_conf(T=num_steps).make_sampler()
    latent_sampler = conf._make_latent_diffusion_conf(T=num_steps).make_sampler()
    
    # 使用固定的輸出目錄（所有實驗共用）
    output_dir = f'{conf.generate_dir}_temp_T{num_steps}'
    
    # 清空輸出目錄（刪除舊圖片）
    if os.path.exists(output_dir):
        LOGGER.info(f"🗑️  清空輸出目錄: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    LOGGER.info(f"開始生成 {CONFIG.EVAL_SAMPLES} 張圖片並計算 FID...")
    LOGGER.info(f"輸出目錄: {output_dir} (實驗結束後會清空)")
    
    # 使用 metrics.evaluate_fid 進行完整的 FID 評估
    from metrics import evaluate_fid
    fid_score = evaluate_fid(
        sampler=sampler,
        model=base_model.ema_model,
        conf=conf,
        device=CONFIG.DEVICE,
        train_data=base_model.train_data,
        val_data=base_model.val_data,
        latent_sampler=latent_sampler,
        conds_mean=base_model.conds_mean,
        conds_std=base_model.conds_std,
        remove_cache=False,
        clip_latent_noise=False,
        T=num_steps,
        output_dir=output_dir
    )
    
    LOGGER.info(f"📊 FID Score: {fid_score:.4f}")
    
    # FID 計算完成後清空圖片（節省空間）
    LOGGER.info(f"🗑️  清空臨時圖片: {output_dir}")
    shutil.rmtree(output_dir)
    
    return fid_score


#=============================================================================
# 主生成流程
#=============================================================================

@time_operation
def main_float_model(
    target_layer: Optional[str] = None,
    k_value: Optional[int] = None,
    baseline_only: bool = False
):
    """
    FID Cache Sensitivity 實驗主流程
    
    Args:
        target_layer: 指定要測試的 layer (None = 測試所有 layer)
        k_value: cache frequency (3, 4, 或 5)
        baseline_only: 是否只跑 baseline (無 cache)
    
    流程:
    1. 載入模型
    2. 如果 baseline_only，只跑 baseline FID
    3. 否則，對指定 layer (或所有 layer) 進行 cache sensitivity 測試
    """
    LOGGER.info("=" * 50)
    LOGGER.info("FID Cache Sensitivity Analysis")
    LOGGER.info("=" * 50)
    
    LOGGER.info(f"使用設備: {CONFIG.DEVICE}")
    LOGGER.info(f"Diffusion steps: {CONFIG.NUM_DIFFUSION_STEPS}")
    LOGGER.info(f"FID evaluation samples: {CONFIG.EVAL_SAMPLES}")
    
    try:
        # 1. 載入模型和 checkpoint
        base_model: LitModel = load_diffae_model()
        LOGGER.info("✅ Diff-AE 模型載入成功")
        
        diffusion_model = base_model.ema_model
        
        # 2. 創建量化模型
        quant_model: QuantModel_DiffAE_LoRA = create_float_quantized_model(
            diffusion_model,
            num_steps=CONFIG.NUM_DIFFUSION_STEPS,
            lora_rank=CONFIG.LORA_RANK,
            mode=CONFIG.MODE
        )
        quant_model.to(CONFIG.DEVICE)
        quant_model.eval()
        
        
        
        # 4. 載入校準資料並初始化
        cali_images, cali_t, cali_y = load_calibration_data()
        quant_model.set_first_last_layer_to_8bit()
        quant_model.set_quant_state(True, True)
        
        # 初始化量化器
        with torch.no_grad():
            _ = quant_model(
                x=cali_images[:4].to(CONFIG.DEVICE),
                t=cali_t[:4].to(CONFIG.DEVICE),
                cond=cali_y[:4].to(CONFIG.DEVICE)
            )
        
        # 轉換權重為 uint8
        for name, module in quant_model.named_modules():
            if isinstance(module, QuantModule_DiffAE_LoRA) and module.ignore_reconstruction is False:
                device = module.weight.data.device
                with torch.no_grad():
                    weight_cpu = module.weight.data.detach().cpu()
                    weight_uint8 = weight_cpu.to(torch.uint8)
                    module.weight.data = weight_uint8.to(device)
        
        # 5. 載入訓練好的 checkpoint
        ckpt = torch.load(CONFIG.BEST_CKPT_PATH, map_location='cpu', weights_only=False)
        from QATcode.cache_method.L1_L2_cosine.similarity_calculation import _load_quant_and_ema_from_ckpt
        _load_quant_and_ema_from_ckpt(base_model, quant_model, ckpt)
        
        base_model.to(CONFIG.DEVICE)
        base_model.eval()
        base_model.setup()
        LOGGER.info("✅ 模型載入與設定完成")
        
        # 6. 載入或初始化結果字典
        step_config = f"T{CONFIG.NUM_DIFFUSION_STEPS}"
        results_path = os.path.join(CONFIG.OUTPUT_DIR, CONFIG.RESULTS_JSON)
        results = load_results(results_path)
        
        # 確保結果字典有正確的結構
        if step_config not in results["results"]:
            results["results"][step_config] = {}
        
        # 7. 如果是 baseline only，只跑 baseline
        if baseline_only:
            LOGGER.info("=" * 50)
            LOGGER.info("執行 Baseline FID (無 cache)")
            LOGGER.info("=" * 50)
            
            if "baseline_fid" not in results["results"][step_config]:
                baseline_fid = evaluate_fid_with_cache(
                    base_model,
                    cache_scheduler=None,
                    num_steps=CONFIG.NUM_DIFFUSION_STEPS
                )
                results["results"][step_config]["baseline_fid"] = baseline_fid
                save_results(results, results_path)
            else:
                baseline_fid = results["results"][step_config]["baseline_fid"]
                LOGGER.info(f"📊 Baseline FID (已存在): {baseline_fid:.4f}")
            
            return
        
        # 8. 執行 cache sensitivity 實驗
        if k_value is None:
            LOGGER.error("❌ 必須指定 k_value (3, 4, 或 5)")
            return
        
        # 確保 baseline 已經跑過
        if "baseline_fid" not in results["results"][step_config]:
            LOGGER.info("⚠️ Baseline FID 尚未計算，先執行 baseline...")
            baseline_fid = evaluate_fid_with_cache(
                base_model,
                cache_scheduler=None,
                num_steps=CONFIG.NUM_DIFFUSION_STEPS
            )
            results["results"][step_config]["baseline_fid"] = baseline_fid
            save_results(results, results_path)
        else:
            baseline_fid = results["results"][step_config]["baseline_fid"]
        
        LOGGER.info(f"📊 Baseline FID: {baseline_fid:.4f}")
        
        # 確保 k{k_value} 子字典存在
        k_key = f"k{k_value}"
        if k_key not in results["results"][step_config]:
            results["results"][step_config][k_key] = {}
        
        # 決定要測試哪些 layer
        if target_layer is not None:
            layers_to_test = [target_layer]
            LOGGER.info(f"🎯 測試單一 layer: {target_layer}")
        else:
            layers_to_test = get_all_layer_names()
            LOGGER.info(f"🎯 測試所有 {len(layers_to_test)} 個 layers")
        
        # 9. 對每個 layer 進行測試
        for i, layer in enumerate(layers_to_test, 1):
            # 檢查是否已經跑過
            if should_skip_experiment(results, step_config, k_value, layer):
                LOGGER.info(f"⏭️  [{i}/{len(layers_to_test)}] {layer} (k={k_value}) - 已完成，跳過")
                continue
            
            LOGGER.info("=" * 50)
            LOGGER.info(f"🔬 [{i}/{len(layers_to_test)}] 測試 {layer} (k={k_value})")
            LOGGER.info("=" * 50)
            
            # 創建 cache config
            cache_config = create_simple_cache_config(
                layer_name=layer,
                k=k_value,
                total_steps=CONFIG.NUM_DIFFUSION_STEPS
            )
            
            # 計算 FID
            fid = evaluate_fid_with_cache(
                base_model,
                cache_scheduler=cache_config,
                num_steps=CONFIG.NUM_DIFFUSION_STEPS
            )
            
            delta = fid - baseline_fid
            
            # 保存結果
            results["results"][step_config][k_key][layer] = {
                "fid": fid,
                "delta": delta
            }
            save_results(results, results_path)
            
            LOGGER.info(f"✅ {layer}: FID={fid:.4f}, Δ={delta:+.4f}")
        
        LOGGER.info("=" * 50)
        LOGGER.info("🎉 實驗完成！")
        LOGGER.info(f"📁 結果已保存至: {results_path}")
        LOGGER.info("=" * 50)

    except Exception as e:
        LOGGER.error(f"❌ Error: {e}")
        import traceback
        LOGGER.error(traceback.format_exc())
        raise e


# main_int_model() 已移除 - 此腳本專注於 FID sensitivity 分析

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='FID Cache Sensitivity Analysis for Diff-AE'
    )
    
    # 基本參數
    parser.add_argument('--num_steps', '--n', type=int, default=20,
                        help='Diffusion steps (20 or 100)')
    parser.add_argument('--eval_samples', '--es', type=int, default=1000,
                        help='Number of samples for FID evaluation')
    
    # 實驗控制
    parser.add_argument('--baseline', action='store_true',
                        help='只跑 baseline FID (無 cache)')
    parser.add_argument('--layer', type=str, default=None,
                        help='指定要測試的 layer (例如: encoder_layer_5)。不指定則測試所有 layer')
    parser.add_argument('--k', type=int, default=None, choices=[3, 4, 5],
                        help='Cache frequency (3, 4, or 5). baseline 模式下不需要')
    
    # 輸出設定
    parser.add_argument('--output_json', type=str, default=None,
                        help='結果 JSON 檔案名稱 (預設: fid_sensitivity_results.json)')
    parser.add_argument('--log_file', '--lf', type=str, default=None,
                        help='Log 檔案路徑')
    
    args = parser.parse_args()
    
    # 更新配置
    CONFIG.NUM_DIFFUSION_STEPS = args.num_steps
    CONFIG.EVAL_SAMPLES = args.eval_samples
    
    if args.output_json is not None:
        CONFIG.RESULTS_JSON = args.output_json
    
    if args.log_file is not None:
        CONFIG.LOG_FILE = args.log_file
    
    # 設定 checkpoint 路徑
    if CONFIG.NUM_DIFFUSION_STEPS == 100:
        CONFIG.BEST_CKPT_PATH = "QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best.pth"
    elif CONFIG.NUM_DIFFUSION_STEPS == 20:
        CONFIG.BEST_CKPT_PATH = "QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best_20steps.pth"
    else:
        print(f"警告: 不支援的 num_steps={CONFIG.NUM_DIFFUSION_STEPS}，使用預設 checkpoint")
        CONFIG.BEST_CKPT_PATH = "QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best.pth"
    
    # 設置環境
    os.makedirs(CONFIG.OUTPUT_DIR, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = CONFIG.GPU_ID
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(CONFIG.LOG_FILE)
        ],
        force=True
    )
    
    # 輸出配置資訊
    LOGGER.info("=" * 50)
    LOGGER.info("FID Cache Sensitivity Analysis")
    LOGGER.info("=" * 50)
    LOGGER.info(f"Diffusion steps: {CONFIG.NUM_DIFFUSION_STEPS}")
    LOGGER.info(f"FID evaluation samples: {CONFIG.EVAL_SAMPLES}")
    LOGGER.info(f"Checkpoint: {CONFIG.BEST_CKPT_PATH}")
    LOGGER.info(f"Output directory: {CONFIG.OUTPUT_DIR}")
    LOGGER.info(f"Results JSON: {CONFIG.RESULTS_JSON}")
    LOGGER.info(f"Log file: {CONFIG.LOG_FILE}")
    
    # 參數驗證
    if not args.baseline and args.k is None:
        LOGGER.error("❌ 必須指定 --k (3, 4, 或 5) 或使用 --baseline 模式")
        exit(1)
    
    if args.baseline and args.k is not None:
        LOGGER.warning("⚠️ Baseline 模式下會忽略 --k 參數")
    
    if args.layer is not None:
        all_layers = get_all_layer_names()
        if args.layer not in all_layers:
            LOGGER.error(f"❌ 無效的 layer 名稱: {args.layer}")
            LOGGER.info(f"有效的 layer 名稱: {', '.join(all_layers[:5])} ... (共 {len(all_layers)} 個)")
            exit(1)
    
    LOGGER.info("=" * 50)
    
    # 執行實驗
    main_float_model(
        target_layer=args.layer,
        k_value=args.k,
        baseline_only=args.baseline
    )
