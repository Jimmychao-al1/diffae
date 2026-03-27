"""
Diff-AE EfficientDM Step 6: 量化感知微調 (QAT)
實現 LoRA 微調 + TALSQ 時間感知激活量化
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

from QATcode.quant_model_lora import QuantModel_DiffAE_LoRA
from QATcode.quant_model_lora import QuantModule_DiffAE_LoRA
from QATcode.quant_layer import QuantModule, SimpleDequantizer
from QATcode.quant_dataset import DiffusionInputDataset
from QATcode.diffae_trainer import *
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

#=============================================================================
# 配置與常量
#=============================================================================

# 訓練配置
class TrainingConfig:
    """訓練與量化相關配置 - EfficientDM 風格"""
    # 硬體設定
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    GPU_ID = '0'
    SEED = 0
    
    # 訓練參數 - 按 EfficientDM 設定
    BATCH_SIZE = 16     # 適中的 batch size
    NUM_EPOCHS = 240   # 按原作 EfficientDM 設定
    LEARNING_RATE = 1e-4
    LORA_RANK = 32
    NUM_DIFFUSION_STEPS = 100  # 對齊原作的 ddim_steps=100
    CLIP_GRAD_NORM = 1.0
    
    # 量化參數
    N_BITS_W = 8  # 權重量化位元數
    N_BITS_A = 8  # 激活量化位元數
    
    # 文件路徑
    MODEL_PATH = "checkpoints/ffhq128_autoenc_latent/last.ckpt"
    QUANT_CKPT_PATH = "QATcode/diffae_unet_quantw8a8_intmodel.pth"
    CALIB_DATA_PATH = "QATcode/calibration_diffae.pth"
    SAVE_BEST_PATH = "QATcode/diffae_step6_lora_best.pth"
    SAVE_FINAL_PATH = "QATcode/diffae_step6_lora_final.pth"

    TEST_PATH = "QATcode/test_ckpt/test.pth"
    
    # 其他設定 - 調整為 EfficientDM 風格
    LOG_INTERVAL = 5   # 每隔幾個批次記錄一次
    CALIB_SAMPLES = 1024
    TRAIN_BATCHES_PER_EPOCH = 2  # 增加每個 epoch 的批次數，因為不再依賴真實資料
    CURRENT_LAYER_IDX = 4
    
    @classmethod
    def setup_environment(cls) -> None:
        """設置執行環境"""
        os.environ['CUDA_VISIBLE_DEVICES'] = cls.GPU_ID
        torch.cuda.manual_seed(cls.SEED)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[logging.StreamHandler(), logging.FileHandler('QATcode/training.log')],
        )

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
CONFIG = TrainingConfig()
LOGGER = logging.getLogger("QuantTraining")

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


def _get_cli_args():
    """解析可選 CLI 參數（預設不改行為）。"""
    import argparse

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--pause', action='store_true', default=False, help='分析後是否暫停等待 Enter')
    parser.add_argument('--dry-run', action='store_true', default=False, help='僅跑最小路徑，不做 optimizer.step')
    parser.add_argument('--max-steps', type=int, default=10, help='dry-run 模式下的最大步數')
    
    # 新增功能旗標（預設關閉 = zero-diff）
    parser.add_argument('--enable-layer-by-layer', action='store_true', default=False, 
                        help='啟用 layer-by-layer 逐層訓練')
    parser.add_argument('--enable-proxy-validation', action='store_true', default=False,
                        help='啟用代理指標驗證')
    parser.add_argument('--exp-dir', type=str, default='runs/default',
                        help='實驗目錄路徑')
    
    try:
        args, _ = parser.parse_known_args()
    except SystemExit:
        class _Args:
            pause = False
            dry_run = False
            max_steps = 10
            enable_layer_by_layer = False
            enable_proxy_validation = False
            exp_dir = 'runs/default'

        args = _Args()
    return args



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

def log_trainable_parameters_details(model: nn.Module) -> None:
    """
    詳細記錄所有 requires_grad=True 的參數信息
    
    Args:
        model: 要分析的模型
    """
    LOGGER.info("=== 詳細可訓練參數信息 ===")
    
    trainable_count = 0
    total_count = 0
    
    for name, param in model.named_parameters():
        total_count += 1
        if param.requires_grad:
            trainable_count += 1
            LOGGER.info(f"可訓練參數 {trainable_count}: {name}")
            LOGGER.info(f"  形狀: {param.shape}")
            LOGGER.info(f"  資料類型: {param.dtype}")
            LOGGER.info(f"  設備: {param.device}")
            LOGGER.info(f"  參數數量: {param.numel():,}")
            LOGGER.info(f"  數值範圍: [{param.min().item():.6f}, {param.max().item():.6f}]")
            LOGGER.info(f"  均值: {param.mean().item():.6f}")
            LOGGER.info(f"  標準差: {param.std().item():.6f}")
            LOGGER.info("")
    
    LOGGER.info(f"總計: {trainable_count}/{total_count} 個參數可訓練")

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
def create_quantized_model(
    diffusion_model: BeatGANsAutoencModel,
    num_steps: int = CONFIG.NUM_DIFFUSION_STEPS, 
    lora_rank: int = CONFIG.LORA_RANK
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
        'scale_method': 'mse'  # 與 Step4/Step5 對齊
    }
    aq_params = {
        'n_bits': CONFIG.N_BITS_A, 
        'channel_wise': False, 
        'scale_method': 'mse',  # 與 Step4 對齊
        'leaf_param': True
    }
    
    # 創建量化模型
    quant_model = QuantModel_DiffAE_LoRA(
        model=diffusion_model,
        weight_quant_params=wq_params,
        act_quant_params=aq_params,
        num_steps=num_steps,
        lora_rank=lora_rank
    )
    quant_model = quant_model.to(CONFIG.DEVICE)
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





#=============================================================================
# 量化模型準備函數
#=============================================================================



def get_adaptive_learning_rates(base_model: LitModel, model_type: str = "diffae") -> Dict[str, float]:
    """
    根據量化分析資料確定自適應學習率
    
    Args:
        base_model: 基礎模型
        model_type: "diffae" 或 "efficientdm"
    
    Returns:
        包含三種學習率的字典
    """
    # 收集權重 delta 統計
    weight_deltas = []
    high_error_layers = 0
    total_layers = 0
    
    for name, module in base_model.named_modules():
        if isinstance(module, QuantModule_DiffAE_LoRA):
            total_layers += 1
            
            # 收集權重 delta
            if hasattr(module, 'weight_quantizer') and hasattr(module.weight_quantizer, 'delta'):
                delta = module.weight_quantizer.delta
                if torch.is_tensor(delta):
                    avg_delta = delta.mean().item()
                    weight_deltas.append(avg_delta)
            
            # 檢查激活量化誤差（需要從之前的分析中估算）
            # 此處使用簡化的啟發式方法
            if hasattr(module, 'act_quantizer'):
                # 假設高誤差層的比例
                if model_type == "diffae" and total_layers % 3 == 0:  # 基於觀察到的高 pct_clipped
                    high_error_layers += 1
    
    if not weight_deltas:
        weight_deltas = [0.002]  # 預設值
    
    avg_weight_delta = np.mean(weight_deltas)
    max_weight_delta = np.max(weight_deltas)
    error_ratio = high_error_layers / max(total_layers, 1)
    
    # 基於模型類型和分析資料調整學習率
    if model_type == "diffae":
        # Diff-AE 需要更保守的學習率，因為：
        # 1. 權重變化幅度大
        # 2. 激活量化誤差高
        # 3. 有複雜的 latent space
        
        lora_factor = 2200  # 2500 -> 3500-4500
        #lora_factor = 3500
        #weight_quant_lr = 5e-7 * (1 + avg_weight_delta / 0.002)  # 更保守
        weight_quant_lr = 2e-5
        #act_quant_lr = 3e-4 * (1 - error_ratio * 0.3)  # 根據誤差率調整
        act_quant_lr = 2e-5
        
    else:  # efficientdm
        # EfficientDM 可以使用相對激進的學習率
        lora_factor = 2000 - int(500 * error_ratio)  # 2500 -> 1500-2000
        weight_quant_lr = 1.5e-6 * (1 + avg_weight_delta / 0.001)
        act_quant_lr = 6e-4 * (1 - error_ratio * 0.2)
    
    return {
        'lora_factor': lora_factor,
        'weight_quant_lr': weight_quant_lr,
        'act_quant_lr': act_quant_lr,
        'stats': {
            'avg_weight_delta': avg_weight_delta,
            'max_weight_delta': max_weight_delta,
            'error_ratio': error_ratio,
            'total_layers': total_layers
        }
    }

def setup_optimizer_with_dynamic_lr(
    base_model: LitModel,
    ddim_steps: int,
    model_type: str = "diffae"
) -> Tuple[torch.optim.Optimizer, Any]:
    """
    基於量化分析資料的動態學習率設定
    """
    from transformers import get_linear_schedule_with_warmup
    
    # 獲取自適應學習率
    lr_config = get_adaptive_learning_rates(base_model, model_type)
    
    lora_factor = lr_config['lora_factor']
    weight_quant_lr = lr_config['weight_quant_lr']
    act_quant_lr = lr_config['act_quant_lr']
    
    print(f"=== 自適應學習率設定 ({model_type.upper()}) ===")
    print(f"LoRA 因子: {lora_factor}")
    print(f"權重量化 LR: {weight_quant_lr:.2e}")
    print(f"激活量化 LR: {act_quant_lr:.2e}")
    print(f"統計信息: {lr_config['stats']}")
    
    firstone = True
    optimizer = None
    
    for name, module in base_model.named_modules():
        if isinstance(module, QuantModule_DiffAE_LoRA):
            if len(module.act_quantizer.delta_list) != ddim_steps:
                raise ValueError('Wrong act_quantizer.delta_list length')
            
            avg_delta = torch.sum(module.weight_quantizer.delta) / torch.numel(module.weight_quantizer.delta)
            
            # LoRA 參數 - 使用自適應因子
            params = [param for name, param in module.named_parameters() 
                      if 'lora' in name and param.requires_grad]
            lora_lr = avg_delta / lora_factor
            if firstone:
                optimizer = torch.optim.AdamW(params, lr=lora_lr, foreach=False)
                firstone = False
            else:
                optimizer.add_param_group({'params': params, 'lr': lora_lr})
            
            # 權重量化器參數 - 使用自適應學習率
            params = [param for name, param in module.named_parameters() 
                      if 'delta' in name and 'list' not in name and param.requires_grad]
            if firstone:
                optimizer = torch.optim.AdamW(params, lr=weight_quant_lr, foreach=False)
                firstone = False
            else:
                optimizer.add_param_group({'params': params, 'lr': weight_quant_lr})
            
            # 激活量化器參數 - 使用自適應學習率
            params = [param for name, param in module.named_parameters() 
                      if ('delta_list' in name or 'zp_list' in name) and param.requires_grad]
            if firstone:
                optimizer = torch.optim.AdamW(params, lr=act_quant_lr, foreach=False)
                firstone = False
            else:
                optimizer.add_param_group({'params': params, 'lr': act_quant_lr})
    
    # 創建學習率調度器
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(CONFIG.NUM_EPOCHS * ddim_steps),
    )
    
    return optimizer, lr_scheduler

def setup_optimizer_for_current_layer(
    base_model: LitModel,
    layer_manager,
    ddim_steps: int,
    model_type: str = "diffae"
) -> Tuple[torch.optim.Optimizer, Any]:
    """
    僅為「當前層」建立最佳化器，確保 Optimizer 只包含當前層參數。
    """
    from transformers import get_linear_schedule_with_warmup

    # 取得當前層模組
    current_module = layer_manager.get_current_layer_module()
    if current_module is None:
        raise ValueError("未找到當前層模組，請檢查 Layer-by-Layer 設定")

    # 自適應學習率（全域統計 + 當前層 avg_delta）
    lr_config = get_adaptive_learning_rates(base_model, model_type)
    lora_factor = lr_config['lora_factor']
    weight_quant_lr = lr_config['weight_quant_lr']
    act_quant_lr = lr_config['act_quant_lr']

    # 當前層的 avg_delta
    if hasattr(current_module, 'weight_quantizer') and hasattr(current_module.weight_quantizer, 'delta'):
        delta = current_module.weight_quantizer.delta
        if torch.is_tensor(delta):
            avg_delta = torch.sum(delta) / torch.numel(delta)
        else:
            avg_delta = torch.tensor(0.002, device=delta.device if hasattr(delta, 'device') else 'cpu')
    else:
        avg_delta = torch.tensor(0.002)
    lora_lr = float(avg_delta) / float(lora_factor)

    # 收集當前層參數 - 與 layer_manager.get_current_layer_params() 保持一致
    all_current_params = []
    lora_params = []
    wq_params = []
    aq_params = []
    other_params = []  # 新增：處理未分類的參數
    
    for name, param in current_module.named_parameters():
        if param.requires_grad:
            all_current_params.append(param)
            if 'lora' in name:
                lora_params.append(param)
            elif 'delta' in name and 'list' not in name:
                wq_params.append(param)
            elif ('delta_list' in name) or ('zp_list' in name):
                aq_params.append(param)
            else:
                other_params.append(param)  # 未分類的參數

    # 調試信息
    LOGGER.info(f"當前層參數統計:")
    LOGGER.info(f"  總參數: {len(all_current_params)}")
    LOGGER.info(f"  LoRA 參數: {len(lora_params)}")
    LOGGER.info(f"  權重量化參數: {len(wq_params)}")
    LOGGER.info(f"  激活量化參數: {len(aq_params)}")
    LOGGER.info(f"  其他參數: {len(other_params)}")

    # 打印其他參數的名稱
    if len(other_params) > 0:
        LOGGER.info("其他參數名稱:")
        for name, param in current_module.named_parameters():
            if param.requires_grad:
                is_classified = False
                if 'lora' in name:
                    is_classified = True
                elif 'delta' in name and 'list' not in name:
                    is_classified = True
                elif ('delta_list' in name) or ('zp_list' in name):
                    is_classified = True
                
                if not is_classified:
                    LOGGER.info(f"  {name}")

    if len(all_current_params) == 0:
        raise ValueError("當前層沒有可訓練參數（requires_grad=True），請檢查層狀態設定")

    # 建立最佳化器（包含所有參數）
    param_groups = []
    if len(lora_params) > 0:
        param_groups.append({'params': lora_params, 'lr': lora_lr})
    if len(wq_params) > 0:
        param_groups.append({'params': wq_params, 'lr': weight_quant_lr})
    if len(aq_params) > 0:
        param_groups.append({'params': aq_params, 'lr': act_quant_lr})
    if len(other_params) > 0:
        # 其他參數使用權重量化學習率
        param_groups.append({'params': other_params, 'lr': weight_quant_lr})

    optimizer = torch.optim.AdamW(param_groups, foreach=False)

    # 學習率調度器
    from transformers import get_linear_schedule_with_warmup
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=(CONFIG.NUM_EPOCHS * ddim_steps),
    )

    return optimizer, lr_scheduler

#=============================================================================
# 訓練循環函數
#=============================================================================



def save_checkpoint(
    base_model: LitModel,
    qnn: QuantModel_DiffAE_LoRA,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    ema_helper=None,
    is_best: bool = False
) -> None:
    """
    保存訓練檢查點
    
    Args:
        base_model: 基礎模型
        qnn: 量化模型
        optimizer: 最佳化器
        epoch: 當前訓練輪數
        loss: 當前損失值
        ema_helper: EMA 輔助器
        is_best: 是否為最佳模型
    """
    if is_best:
        save_path = CONFIG.SAVE_BEST_PATH
    else:
        save_path = CONFIG.SAVE_FINAL_PATH
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 1) raw student 權重（當前訓練中的 qnn）
    raw_model_state = {k: v.detach().cpu().clone() for k, v in qnn.state_dict().items()}

    # 2) EMA student 權重（由 ema_helper.shadow 套到 raw state）
    shadow = getattr(ema_helper, 'shadow', {}) if ema_helper is not None else {}
    ema_model_state = {}
    for k, v in raw_model_state.items():
        if k in shadow and torch.is_tensor(shadow[k]):
            ema_model_state[k] = shadow[k].detach().cpu().clone()
        else:
            ema_model_state[k] = v.detach().cpu().clone() if torch.is_tensor(v) else v

    # 3) 以 base_model 全量 state_dict 為基底，覆蓋 model/ema_model 分支
    #    確保「整個 base_model」可被直接 load_state_dict。
    base_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}
    for k, v in raw_model_state.items():
        base_state[f"model.{k}"] = v
    for k, v in ema_model_state.items():
        base_state[f"ema_model.{k}"] = v

    # 4) 保存 checkpoint（含 resume 所需欄位 + 可直接 load 的平鋪鍵）
    ckpt = {
        "format_version": "qat_step6_raw_ema_v1",
        "epoch": int(epoch),
        "loss": float(loss),
        "raw_model_state_dict": raw_model_state,
        "ema_model_state_dict": ema_model_state,
        "ema_shadow": ema_helper.state_dict() if ema_helper is not None else {},
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else {},
    }
    ckpt.update(base_state)

    torch.save(ckpt, save_path)
    LOGGER.info(
        f"模型已保存: {save_path} "
        f"(base={len(base_state)} keys, raw={len(raw_model_state)} keys, ema={len(ema_model_state)} keys)"
    )







#=============================================================================
# 主訓練流程
#=============================================================================

@time_operation
def main():
    """
    Diff-AE EfficientDM Step 6 訓練主流程
    
    流程概要:
    1. 載入預訓練模型
    2. 創建並設定量化模型
    3. 設定最佳化器與學習率
    4. 訓練與評估
    """
    LOGGER.info("=" * 50)
    LOGGER.info("Diff-AE EfficientDM Step 6: 量化感知微調 (QAT)")
    LOGGER.info("=" * 50)
    
    # 設置執行環境
    CONFIG.setup_environment()
    _seed_all(CONFIG.SEED)
    args = _get_cli_args()
    LOGGER.info(f"使用設備: {CONFIG.DEVICE}")
    
    # 記錄新功能狀態
    LOGGER.info("=== 新功能狀態 ===")
    LOGGER.info(f"Layer-by-Layer 訓練: {'啟用' if args.enable_layer_by_layer else '關閉'}")
    LOGGER.info(f"代理指標驗證: {'啟用' if args.enable_proxy_validation else '關閉'}")
    LOGGER.info(f"實驗目錄: {args.exp_dir}")
    
    try:
        # 1. 載入基礎模型
        base_model : LitModel = load_diffae_model()
        #base_model.ema_model.cpu()
        #base_model = base_model.to(CONFIG.DEVICE)
        LOGGER.info("✅ Diff-AE 模型載入成功")
        
        fp_model = deepcopy(base_model.ema_model).to(CONFIG.DEVICE)
        assert fp_model is not None, "ema_model is None"
        fp_model.cuda()
        LOGGER.info("✅ FP Diff-AE 模型載入成功")
        
        # 從LitModel獲取關鍵組件
        LOGGER.info(f'base_model.conf.train_mode: {base_model.conf.train_mode}')
        diffusion_model = base_model.ema_model
        assert diffusion_model is not None, "ema_model is None"
        diffusion_model.cuda()
        diffusion_model.eval()
        
        


        
        # 2. 創建量化模型與準備組件
        quant_model : QuantModel_DiffAE_LoRA = create_quantized_model(
            diffusion_model, 
            num_steps=CONFIG.NUM_DIFFUSION_STEPS,
            lora_rank=CONFIG.LORA_RANK
        )
        quant_model.cuda()
        quant_model.eval()
        
        # 載入校準資料
        cali_images, cali_t, cali_y = load_calibration_data()
        
        # 設定量化組件
        quant_model.set_first_last_layer_to_8bit()
        device = next(quant_model.parameters()).device

        LOGGER.info("✅ 量化模型創建成功")
        quant_model.set_quant_state(True, True)

        #for name, module in quant_model.named_modules():
        #    if isinstance(module, QuantModule_DiffAE_LoRA) and module.ignore_reconstruction is False:
        #        module.intn_dequantizer = SimpleDequantizer(uaq=module.weight_quantizer, weight=module.weight)
        ckpt = torch.load(CONFIG.QUANT_CKPT_PATH, map_location='cpu', weights_only=False)
        quant_model.load_state_dict(ckpt, strict=False) # no lora weight in ckpt

        for name, module in quant_model.named_modules():
            if isinstance(module, QuantModule_DiffAE_LoRA) and module.ignore_reconstruction is False:
                module.weight.data = module.weight.data.byte()
                #module.intn_dequantizer.delta.data = module.weight_quantizer.delta ## share the same delta and zp
                #module.intn_dequantizer.zero_point.data = module.weight_quantizer.zero_point

        print('First run to init the model') ## need run to init emporal act quantizer
        with torch.no_grad():
            _ = quant_model(x=cali_images[:4].to(device), t=cali_t[:4].to(device), cond=cali_y[:4].to(device))

        #quant_model.set_quant_state(False, False)
        #quant_model.set_quant_state(True, True)
        setattr(base_model, 'model', quant_model)

        LOGGER.info("✅ 量化模型載入成功")

        # 初始化新功能模組
        LOGGER.info("=== 初始化新功能模組 ===")
        
        

        # 設定可訓練參數
        for name, param in base_model.named_parameters():
            if 'lora' in name or 'delta' in name or 'zp_list' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        for name, param in fp_model.named_parameters():
            param.requires_grad = False
        
        # 3. 統計參數
        LOGGER.info("\n=== 模型參數統計 ===")
        print_trainable_parameters(quant_model)
        #count_lora_parameters(quant_model)
        
        # 4. 創建最佳化器 (按原作邏輯 + Layer-by-Layer 支援)
        ddim_steps = CONFIG.NUM_DIFFUSION_STEPS  # 對應原作
        
        optimizer, lr_scheduler = setup_optimizer_with_dynamic_lr(base_model, ddim_steps)
        
        
        # 5. 準備基礎模型設定
        base_model.eval()
        base_model.setup()
        LOGGER.info("✅ 基礎模型設定完成")
        
        # 6. 創建 Diff-AE + EfficientDM 整合的訓練器
        LOGGER.info("=== 創建整合訓練器 ===")
        base_model.eval()
        # 獲取訓練 dataloader
        base_model.train_dataloader()
        train_dataloader = base_model._train_dataloader()
        qnn = base_model.model
        #print(qnn)

        #sampler = base_model.sampler
        sampler = base_model.conf._make_diffusion_conf(T=CONFIG.NUM_DIFFUSION_STEPS).make_sampler()
        
        # 使用模組化的訓練器，加入條件標準化參數
        distill_trainer = create_diffae_trainer(
            base_sampler=sampler,
            fp_model=fp_model,
            quant_model=qnn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            conds_mean=base_model.conds_mean,  # 使用原有的條件標準化參數
            conds_std=base_model.conds_std,
            conf=base_model.conf.clone()
        )

        # 驗證訓練器
        validation_sampler = base_model.sampler
        validation_trainer = create_diffae_trainer(
            base_sampler=validation_sampler,
            fp_model=fp_model,
            quant_model=qnn,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            conds_mean=base_model.conds_mean,
            conds_std=base_model.conds_std,
            conf=base_model.conf.clone()
        )
        eval_sample = base_model.conf._make_diffusion_conf(T=CONFIG.NUM_DIFFUSION_STEPS).make_sampler()
        eval_sampler = SpacedDiffusionBeatGans_Sampler(
            base_sampler=eval_sample,
            quant_model=qnn,
            conds_mean=base_model.conds_mean,  # 加入條件標準化參數
            conds_std=base_model.conds_std,
            conf=base_model.conf.clone(),
            ema_helper=distill_trainer.ema  # 新增 EMA 輔助器
        )
        
        # 8. 訓練循環 - 使用 EfficientDM 風格的 inference 流程訓練
        LOGGER.info(f"\n=== 開始 EfficientDM 風格訓練 (共 {CONFIG.NUM_EPOCHS} epoch) ===")

        # 記錄詳細的可訓練參數信息
        #log_trainable_parameters_details(qnn)

        global start_time, best_loss
        start_time = time.time()
        best_loss = float('inf')
        fp_model.eval()
        best_epoch = -1

        
        # 使用新的單步 Teacher-Student 訓練方法
        for epoch in range(CONFIG.NUM_EPOCHS):
            _seed_all(CONFIG.SEED + epoch)
            epoch_start = time.time()
            qnn.train()
            
            LOGGER.info(f"\n--- Epoch {epoch+1}/{CONFIG.NUM_EPOCHS} ---")
            
            
            
            # 統計變數
            epoch_losses = []
            epoch_distill_losses = []
            epoch_noise_losses = []
            timestep_counts = {}  # 統計時間步分佈
            
            '''Diff-AE inference distillation training'''
            # Diff-AE inference distillation training
            
            loss_dict = distill_trainer.training_losses_with_inference_distillation(
                batch_size=CONFIG.BATCH_SIZE,
                shape=(CONFIG.BATCH_SIZE, 3, 128, 128),
                conf=base_model.conf.clone(),
                device=CONFIG.DEVICE
            )

            # 統計損失
            epoch_losses.append(float(loss_dict['loss']))
            epoch_distill_losses.append(float(loss_dict['distill_loss']))
            #timestep_counts[loss_dict['timestep_k']] = timestep_counts.get(loss_dict['timestep_k'], 0) + 1
            
            """
            '''Diff-AE random timestep t training'''
            
            # 遍歷 dataloader 進行單步訓練
            total_batches = len(train_dataloader)
            #selected_batches = random.sample(range(total_batches), max_batches)
            max_batches = 200
            with tqdm(total=max_batches, desc=f"Epoch {epoch+1}/{CONFIG.NUM_EPOCHS}") as pbar:
                for batch_idx, batch in enumerate(train_dataloader):
                    if batch_idx >= max_batches:
                        break
                    # 如果 batch 是 tuple/list，取第一個元素作為圖像
                    #print('batch:', batch['img'])
                    if isinstance(batch, (tuple, list)):
                        x_start = batch['img']
                    else:
                        x_start = batch['img']
                    
                    #x_start = x_start.to(CONFIG.DEVICE)
                    x_start = x_start.cuda()
                    
                    # 執行單步 TS 訓練
                    loss_dict = distill_trainer.training_step_ts(x_start)
                    
                    # 統計損失
                    epoch_losses.append(loss_dict['total_loss'])
                    epoch_distill_losses.append(loss_dict['distill_loss'])
                    epoch_noise_losses.append(loss_dict['noise_loss'])
                    
                    # 統計時間步分佈
                    k = loss_dict['timestep_k']
                    timestep_counts[k] = timestep_counts.get(k, 0) + 1
                    
                    # 更新進度條
                    pbar.set_postfix({
                        'Loss': f"{loss_dict['total_loss']:.4f}",
                        'LR': f"{loss_dict['learning_rate']:.2e}",
                        'k': k
                    })
                    pbar.update(1)
                    
                    # 定期記錄詳細統計
                    #if batch_idx % CONFIG.LOG_INTERVAL == 0:
                    #    LOGGER.info(f"Batch {batch_idx}/{total_batches}: "
                    #              f"Loss={loss_dict['total_loss']:.6f}, "
                    #              f"Distill={loss_dict['distill_loss']:.6f}, "
                    #              f"Noise={loss_dict['noise_loss']:.6f}, "
                    #              f"t={loss_dict['timestep_t']}, k={k}")
            """
            # 計算 epoch 統計
            avg_loss = np.mean(epoch_losses)
            avg_distill_loss = np.mean(epoch_distill_losses)
            epoch_time = time.time() - epoch_start
            
            LOGGER.info(f"\nEpoch {epoch+1}/{CONFIG.NUM_EPOCHS} 完成:")
            LOGGER.info(f"  總損失: {avg_loss:.6f}")
            LOGGER.info(f"  蒸餾損失: {avg_distill_loss:.6f}")
            LOGGER.info(f"  時間: {epoch_time:.2f}秒")
            
            # 記錄時間步分佈
            #if epoch % 4 == 0:  # 每 4 個 epoch 記錄一次
            #    LOGGER.info("時間步分佈統計:")
            #    for k in sorted(timestep_counts.keys())[:10]:  # 顯示前 10 個
            #        LOGGER.info(f"  k={k}: {timestep_counts[k]} 次")
            
            # 代理指標驗證（每層結束時）
            proxy_results = {}
            
                

            # 保存最佳模型
            improved = avg_loss < best_loss
            if improved:
                best_loss = avg_loss
                best_epoch = epoch+1
                
                save_checkpoint(base_model, qnn, optimizer, best_epoch, best_loss, 
                                  ema_helper=distill_trainer.ema, is_best=True)
                
                LOGGER.info(f"  ✅ 新的最佳模型已保存 (Loss: {best_loss:.6f})")
            
            

            
            
            # 定期驗證生成（使用 DDIM 100 步，不進行參數更新）
            if epoch % 8 == 0:  # 減少頻率，節省時間
                LOGGER.info(f"\n=== Epoch {epoch+1} 驗證生成 (使用 EMA 模型) ===")
                with torch.no_grad():
                    qnn.eval()
                    _seed_all(CONFIG.SEED)
                    
                    # 小批量驗證生成
                    val_batch_size = 8
                    x_T = torch.randn(
                        (val_batch_size, 3, 128, 128),
                        device=CONFIG.DEVICE)

                    # 執行驗證生成（使用 EMA 模型）
                    start_time = time.time()
                    ema_model = distill_trainer.get_ema_model()
                    # 驗證生成時：QuantModule 關閉量化 (False, False)，LoRA 量化模組打開 (True, True)
                    ema_model.set_quant_state(True, True)
                    for name, module in ema_model.named_modules():
                        if hasattr(module, 'act_quantizer') and hasattr(module.act_quantizer, 'current_step'):
                            module.act_quantizer.current_step = CONFIG.NUM_DIFFUSION_STEPS - 1
                            #LOGGER.info(f"驗證前量化器 {name} current_step: {module.act_quantizer.current_step}")
                            #break
                    batch_images = eval_sampler.sample(
                        ema_model=ema_model,
                        fp_model=fp_model,
                        x_T=x_T,
                        noise=None,
                        cond=None,
                        x_start=None,
                        model_kwargs=None,
                        progress=False
                    ).cpu()

                    fp_images = eval_sampler.sample(
                        ema_model=fp_model,
                        fp_model=fp_model,
                        x_T=x_T,
                        noise=None,
                        cond=None,
                        x_start=None,
                        model_kwargs=None,
                        progress=False
                    ).cpu()

                    # 檢查生成圖像的數值範圍
                    LOGGER.info(f"生成圖像數值範圍: [{batch_images.min():.4f}, {batch_images.max():.4f}]")
                    LOGGER.info(f"生成圖像均值: {batch_images.mean():.4f}")
                    LOGGER.info(f"生成圖像標準差: {batch_images.std():.4f}")

                    gen_time = time.time() - start_time

                    # 正規化圖像到 [0, 1]
                    batch_images_normalized = (batch_images + 1) / 2
                    fp_images_normalized = (fp_images + 1) / 2
                    batch_images_normalized = torch.clamp(batch_images_normalized, 0, 1)

                    LOGGER.info(f"正規化後數值範圍: [{batch_images_normalized.min():.4f}, {batch_images_normalized.max():.4f}]")
                    
                    # 保存生成圖像
                    #os.makedirs('QATcode/training_samples', exist_ok=True)
                    #for j in range(len(batch_images)):
                    #    img_name = f'{epoch}_eval_{j}.png'
                    #    torchvision.utils.save_image(
                    #        batch_images[j],
                    #        os.path.join('QATcode/training_samples', f'{img_name}.png'))
                    
                    # 創建網格圖像
                    combined_images = torch.cat([fp_images_normalized, batch_images_normalized ], dim=0)
                    grid_image = torchvision.utils.make_grid(combined_images, nrow=val_batch_size, padding=2)
                    torchvision.utils.save_image(
                        grid_image, 
                        os.path.join('QATcode/training_samples', f'epoch_{epoch}_grid.png'))
                    
                    LOGGER.info(f"✅ Epoch {epoch+1} EMA 模型驗證生成完成:")
                    LOGGER.info(f"  生成時間: {gen_time:.2f}秒")
                    LOGGER.info(f"  圖像範圍: [{batch_images.min():.3f}, {batch_images.max():.3f}]")
                    LOGGER.info(f"  保存路徑: QATcode/training_samples/")
                    
                    # 檢查量化器狀態是否正確重置
                    final_quant_step = None
                    for name, module in qnn.named_modules():
                        if hasattr(module, 'act_quantizer') and hasattr(module.act_quantizer, 'current_step'):
                            final_quant_step = module.act_quantizer.current_step
                            break
                    LOGGER.info(f"驗證後量化器 current_step: {final_quant_step}")
                    
                    # 恢復訓練模式
                    qnn.train()

                    # 清空 GPU 快取
                    torch.cuda.empty_cache()
            # ================================================
        # 9. 訓練完成
        total_time = time.time() - start_time
        LOGGER.info(f"\n=== 整合訓練完成 ===")
        LOGGER.info(f"總訓練時間: {total_time:.2f}秒")
        LOGGER.info(f"最佳損失: {best_loss:.6f}")
        LOGGER.info(f"平均每個 epoch: {total_time/CONFIG.NUM_EPOCHS:.2f}秒")
        
        # 11. 保存最終模型
        save_checkpoint(base_model, qnn, optimizer, CONFIG.NUM_EPOCHS-1, avg_loss, 
                          ema_helper=distill_trainer.ema, is_best=False)
        
        # 12. 模型測試 - 使用 Diff-AE 的測試邏輯
        LOGGER.info(f"\n=== 模型測試 ===")
        qnn.eval()
        
    except Exception as e:
        LOGGER.error(f"❌ 訓練過程錯誤: {e}")
        import traceback
        LOGGER.error(traceback.format_exc())
    
    LOGGER.info("\n🎉 Step 6 量化感知微調完成！")

if __name__ == "__main__":
    main()
