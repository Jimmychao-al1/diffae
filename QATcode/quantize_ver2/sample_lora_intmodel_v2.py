"""
Diff-AE EfficientDM Step 6: 量化感知微調 (QAT)
實現 LoRA 微調 + ver2 normalized fake-quant 推論/評估
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
from QATcode.quantize_ver2.quant_layer_v2 import QuantModule
from QATcode.quantize_ver2.quant_dataset_v2 import DiffusionInputDataset
from QATcode.quantize_ver2.diffae_trainer_v2 import *
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

# 添加快取分析功能
try:
    from cache_analysis.simple_collector import SimpleBlockCollector
    CACHE_ANALYSIS_AVAILABLE = True
except ImportError:
    CACHE_ANALYSIS_AVAILABLE = False
    print("⚠️ 快取分析功能不可用，請檢查 cache_analysis 模組")

# 添加方向2分析功能
try:
    from cache_analysis.emb_output_collector import EmbOutputCollector
    from cache_analysis.correlation_analyzer import CorrelationAnalyzer
    DIRECTION2_AVAILABLE = True
except ImportError:
    DIRECTION2_AVAILABLE = False
    print("⚠️ 方向2分析功能不可用，請檢查 cache_analysis 模組")

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
    MODE = 'train'
    CACHE_METHOD = 'Res' #Res or Att
    # 訓練參數 - 按 EfficientDM 設定
    BATCH_SIZE = 12     # 適中的 batch size
    LORA_RANK = 32
    NUM_DIFFUSION_STEPS = 100  # 對齊原作的 ddim_steps=100
    
    # 快取分析參數
    ENABLE_CACHE_ANALYSIS = False   # 是否啟用快取分析 (方向1)
    CACHE_ANALYSIS_SAMPLES = 5     # 快取分析樣本數
    
    # 方向2分析參數
    ENABLE_DIRECTION2_ANALYSIS = False  # 是否啟用方向2分析
    DIRECTION2_SAMPLES = 10            # 方向2分析樣本數 (較少樣本以加快分析)
    
    # Cache Scheduler 參數
    ENABLE_CACHE = False  # 是否啟用 cache scheduler
    CACHE_THRESHOLD = 0.1  # L1rel 閾值
    
    # 定量分析參數
    ENABLE_QUANTITATIVE_ANALYSIS = False  # 是否啟用定量分析
    ANALYSIS_NUM_SAMPLES = 10  # 生成時間測試的樣本數
    
    # 量化參數
    N_BITS_W = 8  # 權重量化位元數
    N_BITS_A = 8  # 激活量化位元數
    
    # 文件路徑
    MODEL_PATH = "checkpoints/ffhq128_autoenc_latent/last.ckpt"
    BEST_CKPT_PATH = "QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best.pth" 

    CALIB_DATA_PATH = "QATcode/quantize_ver2/calibration_diffae.pth"

    EVAL_SAMPLES = 50_000
    CALIB_SAMPLES = 1024
    
    LOG_FILE = 'QATcode/quantize_ver2/log/sample_lora_intmodel_v2.log'  # 預設 log 檔案路徑

    
    

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
    創建 ver2 float 量化模型（normalized fake-quant）
    
    Args:
        diffusion_model: 基礎擴散模型
        num_steps: 時間步數量（對應 per-step activation scale）
        lora_rank: LoRA 低秩適應的秩
        
    Returns:
        QuantModel_DiffAE_LoRA: 量化後的模型
    """
    LOGGER.info("=== 創建 LoRA 量化模型 ===")
    
    # 量化參數設定
    wq_params = {
        'n_bits': CONFIG.N_BITS_W, 
        'channel_wise': True, 
        'scale_method': 'absmax'
    }
    aq_params = {
        'n_bits': CONFIG.N_BITS_A, 
        'channel_wise': False, 
        'scale_method': 'absmax',
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
    創建 int 路徑量化模型（legacy/相容用途）
    
    Args:
        diffusion_model: 基礎擴散模型
        num_steps: 時間步數量
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
# 主生成流程
#=============================================================================

@time_operation
def main_float_model():
    """
    Diff-AE EfficientDM Step 6 訓練主流程
    
    流程概要:
    1. 載入預訓練模型
    2. 創建並設定量化模型
    3. 設定優化器與學習率
    4. 訓練與評估
    """
    LOGGER.info("=" * 50)
    LOGGER.info("Diff-AE EfficientDM Step 7 : 生成圖片")
    LOGGER.info("=" * 50)
    
    # 設置運行環境（已在參數解析後設置，此處不再重複調用）
    # CONFIG.setup_environment()
    #_seed_all(CONFIG.SEED)

    LOGGER.info(f"使用設備: {CONFIG.DEVICE}")
    
    # 記錄新功能狀態
    LOGGER.info("=== 新功能狀態 ===")

    
    try:
        # 1. 載入基礎模型
        base_model : LitModel = load_diffae_model()
        LOGGER.info("✅ Diff-AE 模型載入成功")

        # 從LitModel獲取關鍵組件
        LOGGER.info(f'base_model.conf.train_mode: {base_model.conf.train_mode}')
        diffusion_model = base_model.ema_model
        #LOGGER.info(base_model)
        # 2. 創建量化模型與準備組件
        quant_model : QuantModel_DiffAE_LoRA = create_float_quantized_model(
            diffusion_model, 
            num_steps=CONFIG.NUM_DIFFUSION_STEPS,
            lora_rank=CONFIG.LORA_RANK,
            mode=CONFIG.MODE
        )
        #
        ## 創建後立刻搬到 GPU
        quant_model.to(CONFIG.DEVICE)
        quant_model.eval()

        # 載入校準資料
        cali_images, cali_t, cali_y = load_calibration_data()
        
        # 設定量化組件
        quant_model.set_first_last_layer_to_8bit()
        device = next(quant_model.parameters()).device

        LOGGER.info("✅ 量化模型創建成功")
        quant_model.set_quant_state(True, True)
        if hasattr(quant_model, 'set_runtime_mode'):
            # warmup/calibration stage keeps dynamic a_w
            quant_model.set_runtime_mode(mode='train', use_cached_aw=False, clear_cached_aw=True)


        
        #print('First run to init the model') ## need run to init emporal act quantizer
        with torch.no_grad():
            _ = quant_model(x=cali_images[:32].to(device), t=cali_t[:32].to(device), cond=cali_y[:32].to(device))

        #quant_model.set_quant_state(False, False)
        
        

        ckpt = torch.load(CONFIG.BEST_CKPT_PATH, map_location='cpu', weights_only=False)
        from QATcode.cache_method.L1_L2_cosine.similarity_calculation import _load_quant_and_ema_from_ckpt
        _load_quant_and_ema_from_ckpt(base_model, quant_model, ckpt)
        if hasattr(base_model.ema_model, 'set_runtime_mode'):
            # final sampling stage can reuse cached a_w
            base_model.ema_model.set_runtime_mode(mode='infer', use_cached_aw=True, clear_cached_aw=True)


        
        # 在這邊對 base_model 做 EMA 同步並儲存，以及對 base_model 中的 model 做儲存，以方便未來做模型大小統計
        LOGGER.info("✅ 量化模型載入成功")

        # 初始化新功能模組
        LOGGER.info("=== 初始化新功能模組 ===")
        
        # 4. 創建優化器 (按原作邏輯 + Layer-by-Layer 支援)
        ddim_steps = CONFIG.NUM_DIFFUSION_STEPS  # 對應原作
        
        
        
        # 5. 準備基礎模型設定
        base_model.to(CONFIG.DEVICE)
        base_model.eval()
        base_model.setup()
        LOGGER.info("✅ 基礎模型設定完成")
        


        T = CONFIG.NUM_DIFFUSION_STEPS
        T_latent = CONFIG.NUM_DIFFUSION_STEPS
        base_model.train_dataloader()
        sampler = base_model.conf._make_diffusion_conf(T=T).make_sampler()
        latent_sampler = base_model.conf._make_latent_diffusion_conf(T=T_latent).make_sampler()
        conf = base_model.conf.clone()
        
        # === 快取分析功能整合 ===
        collector = None
        if CONFIG.ENABLE_CACHE_ANALYSIS:
            LOGGER.info("=" * 50)
            LOGGER.info("啟用快取分析功能")
            LOGGER.info(f"分析樣本數: {CONFIG.CACHE_ANALYSIS_SAMPLES}")
            LOGGER.info(f"擴散步數: {T}")
            LOGGER.info("=" * 50)
            
            # 創建收集器
            collector = SimpleBlockCollector(
                save_dir=f"cache_analysis/collected_outputs/T_{T}/{CONFIG.CACHE_METHOD}",
                max_batch_collect=CONFIG.CACHE_ANALYSIS_SAMPLES,
                cache_method=CONFIG.CACHE_METHOD ,#Res or Att
                max_timesteps=T
            )
            
            # 註冊鉤子
            collector.register_hooks(base_model.ema_model)
            collector._step_counter = -1
            collector.current_step = None
            
            # 修改樣本數以進行分析
            conf.eval_num_images = 32
            
            LOGGER.info(f"開始收集 {CONFIG.CACHE_ANALYSIS_SAMPLES} 個樣本的 TimestepEmbedSequential 輸出...")
            score = evaluate_fid(
                sampler,
                base_model.ema_model,
                conf,
                device=CONFIG.DEVICE,
                train_data=base_model.train_data,
                val_data=base_model.val_data,
                latent_sampler=latent_sampler,
                conds_mean=base_model.conds_mean,
                conds_std=base_model.conds_std,
                remove_cache=False,
                clip_latent_noise=False,
                T=T,
            )
        else:
            conf.eval_num_images = CONFIG.EVAL_SAMPLES
            score = evaluate_fid(
                sampler,
                base_model.ema_model,
                conf,
                device=CONFIG.DEVICE,
                train_data=base_model.train_data,
                val_data=base_model.val_data,
                latent_sampler=latent_sampler,
                conds_mean=base_model.conds_mean,
                conds_std=base_model.conds_std,
                remove_cache=False,
                clip_latent_noise=False,
                T=T,
                output_dir=f'{conf.generate_dir}_QAT_T{T}'
            )
            LOGGER.info(f'FID@{CONFIG.EVAL_SAMPLES} {T} steps score: {score}')
            LOGGER.info('=' * 50)
            LOGGER.info('=' * 50)
        collector = None
        # === 快取分析後處理 ===
        if collector is not None:
            LOGGER.info("=" * 50)
            LOGGER.info("處理收集到的數據...")
            
            # 計算並保存 L1rel 矩陣
            LOGGER.info("計算 L1 相對差異矩陣...")
            collector.calculate_and_save_l1rel_matrices()
            
            # 打印統計信息
            stats = collector.get_stats()
            LOGGER.info("收集統計信息:")
            LOGGER.info(f"  總 Block 數量: {stats['total_blocks']}")
            
            for block_name, block_stats in stats['blocks'].items():
                LOGGER.info(f"  {block_name}:")
                LOGGER.info(f"    時間步數: {block_stats['timesteps']}")
                LOGGER.info(f"    時間步範圍: {block_stats['timestep_range']}")
                LOGGER.info(f"    平均樣本數/步: {block_stats['avg_samples_per_timestep']:.1f}")
            
            # 清理鉤子
            collector.remove_hooks()
            
            LOGGER.info("=" * 50)
            LOGGER.info("✅ 快取分析完成！")
            LOGGER.info("📁 L1rel 矩陣已保存到: cache_analysis/collected_outputs/")
            LOGGER.info("📋 下一步: 使用 CSV 文件進行快取策略分析")
            LOGGER.info("=" * 50)
        
        
        
        global start_time
        start_time = time.time()
        

    except Exception as e:
        LOGGER.error(f"Error: {e}")
        raise e


        
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_steps','--n', type=int, default=100)
    parser.add_argument('--samples', '--s', type=int, default=5)
    parser.add_argument('--eval_samples', '--es', type=int, default=50000)
    parser.add_argument('--mode', '--m', type=str, default='float')
    parser.add_argument('--enable_cache', action='store_true', help='啟用 cache scheduler')
    parser.add_argument('--cache_method', type=str, default='Res', choices=['Res', 'Att'], 
                        help='Cache 方法：Res (TimestepEmbedSequential) 或 Att (AttentionBlock)')
    parser.add_argument('--cache_threshold', type=float, default=0.1, 
                        help='Cache scheduler 的 L1rel 閾值')
    parser.add_argument('--enable_quantitative_analysis', action='store_true',
                        help='啟用定量分析')
    parser.add_argument('--analysis_num_samples', type=int, default=10,
                        help='生成時間測試的樣本數')
    parser.add_argument('--log_file','--lf', type=str, default=None,
                        help='指定 log 檔案路徑（預設: QATcode/quantize_ver2/log/sample_lora_intmodel_v2.log）')
    args = parser.parse_args()
    CONFIG.NUM_DIFFUSION_STEPS = args.num_steps
    CONFIG.CACHE_ANALYSIS_SAMPLES = args.samples
    CONFIG.EVAL_SAMPLES = args.eval_samples
    CONFIG.ENABLE_CACHE = args.enable_cache
    CONFIG.CACHE_METHOD = args.cache_method
    CONFIG.CACHE_THRESHOLD = args.cache_threshold
    CONFIG.ENABLE_QUANTITATIVE_ANALYSIS = args.enable_quantitative_analysis
    CONFIG.ANALYSIS_NUM_SAMPLES = args.analysis_num_samples
    # 如果指定了 log_file，則使用指定的路徑；否則使用預設值
    if args.log_file is not None:
        CONFIG.LOG_FILE = args.log_file
        print(f"[設定] Log 檔案已設為: {CONFIG.LOG_FILE}", flush=True)
    
    def setup_environment(cls) -> None:
        """設置運行環境"""
        os.environ['CUDA_VISIBLE_DEVICES'] = cls.GPU_ID
        #torch.cuda.manual_seed(cls.SEED)
        log_dir = os.path.dirname(cls.LOG_FILE)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        
        # 重新配置 logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[logging.StreamHandler(), logging.FileHandler(cls.LOG_FILE)],
            force=True  # Python 3.8+ 強制重新配置
        )
    
    setup_environment(CONFIG)
    
    LOGGER.info(f"Using {CONFIG.NUM_DIFFUSION_STEPS} steps")
    LOGGER.info(f"Using {CONFIG.EVAL_SAMPLES} evaluation samples")
    LOGGER.info(f"Cache enabled: {CONFIG.ENABLE_CACHE}, method: {CONFIG.CACHE_METHOD}, threshold: {CONFIG.CACHE_THRESHOLD}")
    LOGGER.info(f"Quantitative analysis enabled: {CONFIG.ENABLE_QUANTITATIVE_ANALYSIS}, samples: {CONFIG.ANALYSIS_NUM_SAMPLES}")
    LOGGER.info(f"Log file: {CONFIG.LOG_FILE}")
    if args.mode == 'float':
        CONFIG.BEST_CKPT_PATH = "QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best.pth" if CONFIG.NUM_DIFFUSION_STEPS == 100 else "QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best_20steps.pth"
        main_float_model()
    elif args.mode == 'int':
        CONFIG.BEST_CKPT_PATH = "QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best_int.pth" if CONFIG.NUM_DIFFUSION_STEPS == 100 else "QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best_int_20steps.pth"
        pass
    elif args.mode == 'final':
        CONFIG.BEST_CKPT_PATH = "QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best_final.pth"
        main_float_model()
    else:
        LOGGER.error(f"Invalid mode: {args.mode}")
        exit(1)
