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

from QATcode.quantize_ver2.quant_model_lora_v2 import QuantModel_DiffAE_LoRA
from QATcode.quantize_ver2.quant_model_lora_v2 import QuantModule_DiffAE_LoRA, INT_QuantModel_DiffAE_LoRA, INT_QuantModule_DiffAE_LoRA
from QATcode.quantize_ver2.quant_layer_v2 import QuantModule, SimpleDequantizer
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
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
#from thop import profile, clever_format
from model.blocks import QKVAttention, QKVAttentionLegacy
from model.nn import timestep_embedding
import seaborn as sns
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

    # Similarity 分析參數（L1rel/L2rel/Cosine）
    ENABLE_SIMILARITY_ANALYSIS = False
    SIMILARITY_SAMPLES = 64  # FID 評估用的總樣本數
    SIMILARITY_COLLECT_SAMPLES = 15  # 實際收集用於相似度計算的樣本數（建議 10-15）
    SIMILARITY_TARGET_BLOCK = None  # 例如 "model.input_blocks.0"
    SIMILARITY_OUTPUT_ROOT = "QATcode/cache_method/L1_L2_cosine"
    SIMILARITY_SAVE_DTYPE = "float16"  # npz 儲存精度：float16 / float32
    SIMILARITY_PLOT_COSINE_STEP = False  # cosine 只做 heatmap 時可關閉
    
    # 量化參數
    N_BITS_W = 8  # 權重量化位元數
    N_BITS_A = 8  # 激活量化位元數
    
    # 文件路徑
    MODEL_PATH = "checkpoints/ffhq128_autoenc_latent/last.ckpt"
    BEST_CKPT_PATH = "QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best.pth" 

    CALIB_DATA_PATH = "QATcode/quantize_ver2/calibration_diffae.pth"

    EVAL_SAMPLES = 50_000
    CALIB_SAMPLES = 1024
    
    LOG_FILE = 'QATcode/cache_method/L1_L2_cosine/log/similarity_calculation.log'  # 預設 log 檔案路徑

    
    

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
# Similarity Analysis (L1rel / L2rel / Cosine)
#=============================================================================

class SimilarityCollector:
    """收集 TimestepEmbedSequential 輸出並計算 L1/L2/Cosine 矩陣與 step-change 曲線。"""

    def __init__(
        self,
        save_root: str,
        max_timesteps: int,
        num_samples: int,
        target_block: Optional[str] = None,
        dtype: str = "float16",
        save_cosine_step_plot: bool = False,
        base_image_size: int = 128,
        device: Optional[torch.device] = None,
        sample_strategy: str = "first",  # "first", "random", "uniform"
    ):
        self.save_root = Path(save_root)
        self.max_timesteps = int(max_timesteps)
        self.num_samples = int(num_samples)
        self.target_block = target_block
        self.save_cosine_step_plot = save_cosine_step_plot
        self.save_dtype = np.float16 if dtype == "float16" else np.float32
        self.base_image_size = base_image_size
        # 設置設備：優先使用傳入的 device，否則自動檢測
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        # 採樣策略：用於增加多樣性
        # "first": 取前 N 個（默認，最快）
        # "random": 隨機選擇 N 個（增加多樣性）
        # "uniform": 均勻分佈選擇（例如 batch_size=32, collect=15，選擇 0, 2, 4, 6, ...）
        self.sample_strategy = sample_strategy
        self._sample_indices_cache = {}  # 緩存採樣索引，確保同一 batch 內所有 timestep 使用相同的索引

        self.result_npz_dir = self.save_root / "result_npz"
        self.l1_dir = self.save_root / "L1"
        self.l1_change_dir = self.save_root / "L1_change"
        self.l2_dir = self.save_root / "L2"
        self.cos_dir = self.save_root / "cosine"
        self.tier_plot_dir = self.save_root / "tier_plots"
        for d in [self.result_npz_dir, self.l1_dir, self.cos_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.hooks = []
        self._step_counter = -1
        self.current_step_idx = None
        self.current_mapped_t = None
        self.step_idx_list = list(range(self.max_timesteps))
        self.mapped_t_list = None

        # per-batch storage: {block_name: {step_idx: tensor(B,C,H,W)}}
        self._batch_outputs: Dict[str, Dict[int, torch.Tensor]] = {}
        self._batch_size = None
        
        # 样本计数器：追踪已收集的样本数
        self._collected_samples = 0
        self._collection_active = True
        self._batch_collect_limit = None  # 每個 batch 要收集的樣本數（會在 register_hooks 時設定）

        # per-block accumulators
        self.block_sums = {}
        self.block_counts = {}  # 用於 L1/L2（累加樣本數）
        self.block_counts_cos = {}  # 獨立用於 cosine（累加 batch 數）
        self.block_step_sums = {}
        self.block_step_sumsq = {}
        self.block_step_counts = {}
        self.block_shapes = {}

        # tier accumulators
        self.tier_step_sums = {}
        self.tier_step_sumsq = {}
        self.tier_step_counts = {}
        
        # per-batch 數據收集（用於多組折線圖）
        # 格式：{block_name: [batch_0_data, batch_1_data, ...]}
        # 每個 batch_data 是 {metric: {sums, sumsq, counts}}
        self.batch_step_data = {}  # {block_name: [{l1: {...}, l2: {...}, cos: {...}}, ...]}
        self.current_batch_idx = 0  # 追蹤當前是第幾個 batch
        self.current_batch_collected = 0  # 當前 batch 已收集的樣本數
        # 當前 batch 的累加器（用於累積當前 batch 的數據，直到達到 collect_samples）
        self.current_batch_accumulator = {}  # {block_name: {l1: {...}, l2: {...}, cos: {...}}}

    def register_hooks(self, model: nn.Module, sampler):
        """註冊 block hook + model step hook。"""
        from model.blocks import TimestepEmbedSequential

        self._step_counter = -1
        self.current_step_idx = None
        self.current_mapped_t = None
        self.mapped_t_list = (
            list(getattr(sampler, "timestep_map", []))
            if getattr(sampler, "timestep_map", None) is not None
            else None
        )

        for name, module in model.named_modules():
            if not isinstance(module, TimestepEmbedSequential):
                continue
            if "encoder" in name:
                continue
            if self.target_block and name != self.target_block:
                continue
            self.hooks.append(module.register_forward_hook(self._create_block_hook(name)))
            LOGGER.info(f"[Similarity] 註冊 block hook: {name}")

        # model pre-hook: 更新 step index
        self.hooks.append(
            model.register_forward_pre_hook(self._create_step_pre_hook(), with_kwargs=True)
        )
        # model post-hook: 在 step_idx==0 時結算 batch
        self.hooks.append(model.register_forward_hook(self._create_model_post_hook()))
        LOGGER.info("[Similarity] 註冊 model step hook 完成")

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()

    def _create_step_pre_hook(self):
        def pre_hook(module, args, kwargs):
            self._step_counter = (self._step_counter + 1) % self.max_timesteps
            self.current_step_idx = self.max_timesteps - 1 - self._step_counter
            if self.mapped_t_list and 0 <= self.current_step_idx < len(self.mapped_t_list):
                self.current_mapped_t = int(self.mapped_t_list[self.current_step_idx])
            else:
                self.current_mapped_t = None
        return pre_hook

    def _create_model_post_hook(self):
        def post_hook(module, input, output):
            if self.current_step_idx == 0:
                self._finalize_current_batch()
                self._step_counter = -1
                # 清除當前 batch 的採樣索引緩存，為下一個 batch 做準備
                # 只清除當前 batch 的緩存，保留其他 block 的緩存（如果有的話）
                keys_to_remove = [k for k in self._sample_indices_cache.keys() if f"_batch_{self.current_batch_idx}" in k]
                for k in keys_to_remove:
                    del self._sample_indices_cache[k]
        return post_hook

    def _create_block_hook(self, block_name: str):
        def hook_fn(module, input, output):
            # 如果已收集足够样本，直接跳过
            if not self._collection_active:
                return
            if self._step_counter is None or self._step_counter < 0:
                return
            # 使用 _step_counter (0→99) 作为索引，统一表示方法
            step_idx = int(self._step_counter)
            if not (0 <= step_idx < self.max_timesteps):
                return
            
            # 计算还需要收集多少样本
            remaining = self.num_samples - self._collected_samples
            if remaining <= 0:
                self._collection_active = False
                return
            
            # 先保持在 GPU，避免阻塞；dtype 转换也在 GPU 上做
            out = output.detach()
            out = out.to(dtype=torch.float16 if self.save_dtype == np.float16 else torch.float32)
            
            # 只取需要的样本数（batch 内截断）
            batch_size = out.shape[0]
            n_collect = min(batch_size, remaining)
            
            # 根據採樣策略選擇樣本
            if n_collect < batch_size:
                # 需要從 batch 中選擇樣本
                if self.sample_strategy == "first":
                    # 策略1：取前 N 個（最快，但可能缺乏多樣性）
                    indices = torch.arange(n_collect, device=out.device)
                elif self.sample_strategy == "random":
                    # 策略2：隨機選擇 N 個（增加多樣性）
                    # 使用 block_name 和當前 batch 作為 key，確保同一 batch 內所有 timestep 使用相同的索引
                    cache_key = f"{block_name}_batch_{self.current_batch_idx}"
                    if cache_key not in self._sample_indices_cache:
                        # 為當前 batch 生成隨機索引
                        indices = torch.randperm(batch_size, device=out.device)[:n_collect]
                        self._sample_indices_cache[cache_key] = indices
                    else:
                        indices = self._sample_indices_cache[cache_key]
                elif self.sample_strategy == "uniform":
                    # 策略3：均勻分佈選擇（例如 batch_size=32, collect=15，選擇 0, 2, 4, 6, ...）
                    cache_key = f"{block_name}_batch_{self.current_batch_idx}"
                    if cache_key not in self._sample_indices_cache:
                        # 計算均勻分佈的步長
                        step = batch_size / n_collect
                        indices = torch.tensor([int(i * step) for i in range(n_collect)], device=out.device, dtype=torch.long)
                        self._sample_indices_cache[cache_key] = indices
                    else:
                        indices = self._sample_indices_cache[cache_key]
                else:
                    # 默認：取前 N 個
                    indices = torch.arange(n_collect, device=out.device)
                
                out = out[indices]
            else:
                # 不需要選擇，使用全部樣本
                indices = torch.arange(batch_size, device=out.device)
            
            self._batch_outputs.setdefault(block_name, {})[step_idx] = out
            if self._batch_size is None:
                self._batch_size = n_collect
        return hook_fn

    def _ensure_block_accumulators(self, block_name: str):
        if block_name in self.block_sums:
            return
        T = self.max_timesteps
        # 使用 GPU tensor 累加器（計算時在 GPU，最後轉移到 CPU）
        self.block_sums[block_name] = {
            "l1": torch.zeros((T, T), dtype=torch.float64, device=self.device),
            "l1_rate": torch.zeros((T, T), dtype=torch.float64, device=self.device),
            "l2": torch.zeros((T, T), dtype=torch.float64, device=self.device),
            "cos": torch.zeros((T, T), dtype=torch.float64, device=self.device),
        }
        self.block_counts[block_name] = torch.zeros((T, T), dtype=torch.int64, device=self.device)  # 用於 L1/L2
        self.block_counts_cos[block_name] = torch.zeros((T, T), dtype=torch.int64, device=self.device)  # 用於 cosine
        self.block_step_sums[block_name] = {
            "l1": torch.zeros((T - 1,), dtype=torch.float64, device=self.device),
            "l1_rate": torch.zeros((T - 1,), dtype=torch.float64, device=self.device),
            "l2": torch.zeros((T - 1,), dtype=torch.float64, device=self.device),
            "cos": torch.zeros((T - 1,), dtype=torch.float64, device=self.device),
        }
        self.block_step_sumsq[block_name] = {
            "l1": torch.zeros((T - 1,), dtype=torch.float64, device=self.device),
            "l1_rate": torch.zeros((T - 1,), dtype=torch.float64, device=self.device),
            "l2": torch.zeros((T - 1,), dtype=torch.float64, device=self.device),
            "cos": torch.zeros((T - 1,), dtype=torch.float64, device=self.device),
        }
        self.block_step_counts[block_name] = torch.zeros((T - 1,), dtype=torch.int64, device=self.device)

    def _assign_tier(self, block_name: str, h: int, w: int, base: int):
        if h == base and w == base:
            return "full"
        if h == base // 2 and w == base // 2:
            return "half"
        if h == base // 4 and w == base // 4:
            return "quarter"
        if h == base // 8 and w == base // 8:
            return "eighth"
        return None

    def _init_tier_accumulators(self, tier: str):
        if tier in self.tier_step_sums:
            return
        T = self.max_timesteps
        # 使用 GPU tensor 累加器
        self.tier_step_sums[tier] = {
            "l1": torch.zeros((T - 1,), dtype=torch.float64, device=self.device),
            "l1_rate": torch.zeros((T - 1,), dtype=torch.float64, device=self.device),
            "l2": torch.zeros((T - 1,), dtype=torch.float64, device=self.device),
            "cos": torch.zeros((T - 1,), dtype=torch.float64, device=self.device),
        }
        self.tier_step_sumsq[tier] = {
            "l1": torch.zeros((T - 1,), dtype=torch.float64, device=self.device),
            "l1_rate": torch.zeros((T - 1,), dtype=torch.float64, device=self.device),
            "l2": torch.zeros((T - 1,), dtype=torch.float64, device=self.device),
            "cos": torch.zeros((T - 1,), dtype=torch.float64, device=self.device),
        }
        self.tier_step_counts[tier] = torch.zeros((T - 1,), dtype=torch.int64, device=self.device)

    def _finalize_current_batch(self):
        if not self._batch_outputs:
            return
        
        T = self.max_timesteps
        for block_name, step_dict in self._batch_outputs.items():
            if len(step_dict) != T:
                LOGGER.warning(f"[Similarity] block {block_name} timesteps 不完整: {len(step_dict)}/{T}")
                continue

            self._ensure_block_accumulators(block_name)
            # 保持在 GPU 上，不轉移到 CPU（所有計算在 GPU 上進行）
            outputs = [step_dict[i] for i in range(T)]
            
            # 更新已收集样本数（基于实际收集的样本）
            actual_batch_size = outputs[0].shape[0]
            
            # 計算當前 batch 實際應該收集多少樣本
            # 考慮全局限制和每個 batch 的限制
            if self._collection_active and self._batch_collect_limit is not None:
                # 計算當前 batch 還需要多少樣本才能達到 batch 限制
                remaining_for_batch = self._batch_collect_limit - self.current_batch_collected
                # 計算全局還剩多少樣本
                remaining_global = self.num_samples - self._collected_samples
                
                # 實際收集的樣本數 = min(實際 batch 大小, batch 剩餘, 全局剩餘)
                actual_collected = min(actual_batch_size, remaining_for_batch, remaining_global)
                
                # 如果實際 batch 大小超過需要收集的數量，只處理前 actual_collected 個樣本
                if actual_batch_size > actual_collected:
                    outputs = [out[:actual_collected] for out in outputs]
                    actual_batch_size = actual_collected
            else:
                # 如果沒有 batch 限制或已停止收集，使用實際 batch 大小
                remaining_global = self.num_samples - self._collected_samples
                actual_collected = min(actual_batch_size, remaining_global) if self._collection_active else 0
                if actual_batch_size > actual_collected:
                    outputs = [out[:actual_collected] for out in outputs]
                    actual_batch_size = actual_collected
            
            self._collected_samples += actual_collected
            
            # 只有當還在收集時，才累加到當前 batch 和累加器
            if self._collection_active:
                self.current_batch_collected += actual_collected
                
                # 检查是否已收集足够样本（全局）
                # 注意：即使達到全局限制，也要先保存當前 batch 的數據，然後再停止
                if self._collected_samples >= self.num_samples:
                    LOGGER.info(f"[Similarity] 已收集 {self._collected_samples} 个样本，停止数据收集（目标: {self.num_samples}）")
                    # 不立即設置 _collection_active = False，讓後續邏輯有機會保存當前 batch

            # 記錄 block 空間尺寸，用於 tier 分組
            if block_name not in self.block_shapes:
                _, _, h, w = outputs[0].shape
                self.block_shapes[block_name] = (h, w)

            # 計算 L1/L2 矩陣（per-sample 累加）
            for i in range(T):
                t1 = outputs[i]
                for j in range(i, T):
                    t2 = outputs[j]
                    l1_vals, l1_rate_vals, l2_vals, _ = self._calc_metrics_batch(t1, t2)
                    self.block_sums[block_name]["l1"][i, j] += l1_vals.sum()
                    self.block_sums[block_name]["l1_rate"][i, j] += l1_rate_vals.sum()
                    self.block_sums[block_name]["l2"][i, j] += l2_vals.sum()
                    self.block_counts[block_name][i, j] += l1_vals.numel()
                    if i != j:
                        self.block_sums[block_name]["l1"][j, i] += l1_vals.sum()
                        # l1_rate 使用對稱值：以 t1 為參考計算 ||t1-t2||/||t1||，[t2][t1] 填入相同值
                        self.block_sums[block_name]["l1_rate"][j, i] += l1_rate_vals.sum()
                        self.block_sums[block_name]["l2"][j, i] += l2_vals.sum()
                        self.block_counts[block_name][j, i] += l1_vals.numel()
            
            # 計算 Cosine 相似度矩陣（使用矩陣運算，參考 metrics.py）
            # 對每個樣本分別計算 cosine 矩陣，然後平均
            # outputs[i] shape: (batch_size, C, H, W)
            batch_size = outputs[0].shape[0]
            cosine_matrices = []
            
            for b in range(batch_size):
                # 對每個樣本，提取所有 timestep 的特徵
                features_list = []
                for i in range(T):
                    feat = outputs[i][b].flatten().to(torch.float64)  # shape: (C*H*W,)
                    features_list.append(feat)
                
                # 堆疊所有 timestep 的特徵
                features = torch.stack(features_list)  # shape: (T, C*H*W)
                
                # 計算 cosine similarity（參考 metrics.py 的實現）
                norms = torch.norm(features, p=2, dim=1, keepdim=True)  # shape: (T, 1)
                normalized_features = features / (norms + 1e-8)  # 避免除零，shape: (T, C*H*W)
                cosine_matrix = torch.mm(normalized_features, normalized_features.T)  # shape: (T, T)
                cosine_matrices.append(cosine_matrix)
            
            # 對所有樣本求平均
            cosine_matrix_avg = torch.stack(cosine_matrices).mean(dim=0)  # shape: (T, T)，在 GPU 上
            
            # 累加到累加器（對角線自動是 1），直接在 GPU 上累加
            self.block_sums[block_name]["cos"] += cosine_matrix_avg
            # counts：每個 batch 貢獻 1（因為是矩陣運算，每個 batch 計算一次）
            # 使用獨立的 counts 矩陣，避免與 L1/L2 的 counts 混淆
            # L1/L2 的 counts 累加的是樣本數，cosine 的 counts 累加的是 batch 數
            self.block_counts_cos[block_name] += torch.ones(cosine_matrix_avg.shape, dtype=torch.int64, device=self.device)

            # 計算 step-change：outputs[s-1]→outputs[s]，s 為 _step_counter（0→99）。
            # 寫入 l1_rate 的 index [s-1]；與 npz「l1_rate_step_mean」欄位一致。
            # 語意（見 QATcode/docs/cache_time_axis_audit.md）：index j = analysis 上 interval j
            # （axis j↔j+1），對應 DDIM t_ddim (99−j)→(98−j)；**不是**「DDIM 張量 t=j→j+1」。
            # 初始化當前 batch 的累加器（如果還沒有），使用 GPU tensor
            if block_name not in self.current_batch_accumulator:
                self.current_batch_accumulator[block_name] = {
                    "l1": {"sums": torch.zeros(T - 1, dtype=torch.float64, device=self.device), "sumsq": torch.zeros(T - 1, dtype=torch.float64, device=self.device), "counts": torch.zeros(T - 1, dtype=torch.int64, device=self.device)},
                    "l1_rate": {"sums": torch.zeros(T - 1, dtype=torch.float64, device=self.device), "sumsq": torch.zeros(T - 1, dtype=torch.float64, device=self.device), "counts": torch.zeros(T - 1, dtype=torch.int64, device=self.device)},
                    "l2": {"sums": torch.zeros(T - 1, dtype=torch.float64, device=self.device), "sumsq": torch.zeros(T - 1, dtype=torch.float64, device=self.device), "counts": torch.zeros(T - 1, dtype=torch.int64, device=self.device)},
                    "cos": {"sums": torch.zeros(T - 1, dtype=torch.float64, device=self.device), "sumsq": torch.zeros(T - 1, dtype=torch.float64, device=self.device), "counts": torch.zeros(T - 1, dtype=torch.int64, device=self.device)},
                }
                if block_name not in self.batch_step_data:
                    self.batch_step_data[block_name] = []
            
            for s in range(1, T):
                t_prev = outputs[s - 1]  # _step_counter = s-1
                t_curr = outputs[s]      # _step_counter = s
                l1_vals, l1_rate_vals, l2_vals, cos_vals = self._calc_metrics_batch(t_prev, t_curr)
                
                # 累加到全局累加器
                self.block_step_sums[block_name]["l1"][s - 1] += l1_vals.sum()
                self.block_step_sums[block_name]["l1_rate"][s - 1] += l1_rate_vals.sum()
                self.block_step_sums[block_name]["l2"][s - 1] += l2_vals.sum()
                self.block_step_sums[block_name]["cos"][s - 1] += cos_vals.sum()
                self.block_step_sumsq[block_name]["l1"][s - 1] += (l1_vals ** 2).sum()
                self.block_step_sumsq[block_name]["l1_rate"][s - 1] += (l1_rate_vals ** 2).sum()
                self.block_step_sumsq[block_name]["l2"][s - 1] += (l2_vals ** 2).sum()
                self.block_step_sumsq[block_name]["cos"][s - 1] += (cos_vals ** 2).sum()
                self.block_step_counts[block_name][s - 1] += l1_vals.numel()
                
                # 只有當還在收集時，才累加到當前 batch 的累加器
                if self._collection_active:
                    self.current_batch_accumulator[block_name]["l1"]["sums"][s - 1] += l1_vals.sum()
                    self.current_batch_accumulator[block_name]["l1"]["sumsq"][s - 1] += (l1_vals ** 2).sum()
                    self.current_batch_accumulator[block_name]["l1"]["counts"][s - 1] += l1_vals.numel()
                    self.current_batch_accumulator[block_name]["l1_rate"]["sums"][s - 1] += l1_rate_vals.sum()
                    self.current_batch_accumulator[block_name]["l1_rate"]["sumsq"][s - 1] += (l1_rate_vals ** 2).sum()
                    self.current_batch_accumulator[block_name]["l1_rate"]["counts"][s - 1] += l1_rate_vals.numel()
                    self.current_batch_accumulator[block_name]["l2"]["sums"][s - 1] += l2_vals.sum()
                    self.current_batch_accumulator[block_name]["l2"]["sumsq"][s - 1] += (l2_vals ** 2).sum()
                    self.current_batch_accumulator[block_name]["l2"]["counts"][s - 1] += l2_vals.numel()
                    self.current_batch_accumulator[block_name]["cos"]["sums"][s - 1] += cos_vals.sum()
                    self.current_batch_accumulator[block_name]["cos"]["sumsq"][s - 1] += (cos_vals ** 2).sum()
                    self.current_batch_accumulator[block_name]["cos"]["counts"][s - 1] += cos_vals.numel()
            
            # 檢查當前 batch 是否收集完足夠的樣本（每個 batch 收集 _batch_collect_limit 個樣本）
            # _batch_collect_limit 在 register_hooks 時設定為 similarity_collect_samples
            # 或者已經達到全局限制（需要保存最後一個 batch）
            should_save_batch = False
            if self._batch_collect_limit is not None and self.current_batch_collected >= self._batch_collect_limit:
                should_save_batch = True
            elif self._collected_samples >= self.num_samples and self.current_batch_collected > 0:
                # 達到全局限制，但當前 batch 有數據，也要保存
                should_save_batch = True
            
            if should_save_batch:
                # 當前 batch 已收集足夠的樣本，保存數據並開始新 batch
                # 轉移到 CPU 並轉換為 numpy（用於保存和後續繪圖）
                batch_data = {
                    "l1": {k: v.cpu().clone().numpy() if isinstance(v, torch.Tensor) else v.copy() for k, v in self.current_batch_accumulator[block_name]["l1"].items()},
                    "l1_rate": {k: v.cpu().clone().numpy() if isinstance(v, torch.Tensor) else v.copy() for k, v in self.current_batch_accumulator[block_name]["l1_rate"].items()},
                    "l2": {k: v.cpu().clone().numpy() if isinstance(v, torch.Tensor) else v.copy() for k, v in self.current_batch_accumulator[block_name]["l2"].items()},
                    "cos": {k: v.cpu().clone().numpy() if isinstance(v, torch.Tensor) else v.copy() for k, v in self.current_batch_accumulator[block_name]["cos"].items()},
                }
                self.batch_step_data[block_name].append(batch_data)
                LOGGER.info(f"[Similarity] Batch {self.current_batch_idx} 完成，已收集 {self.current_batch_collected} 個樣本")
                
                # 如果已達到全局限制，停止收集
                if self._collected_samples >= self.num_samples:
                    self._collection_active = False
                
                # 重置當前 batch 的累加器（GPU tensor）
                T = self.max_timesteps
                self.current_batch_accumulator[block_name] = {
                    "l1": {"sums": torch.zeros(T - 1, dtype=torch.float64, device=self.device), "sumsq": torch.zeros(T - 1, dtype=torch.float64, device=self.device), "counts": torch.zeros(T - 1, dtype=torch.int64, device=self.device)},
                    "l1_rate": {"sums": torch.zeros(T - 1, dtype=torch.float64, device=self.device), "sumsq": torch.zeros(T - 1, dtype=torch.float64, device=self.device), "counts": torch.zeros(T - 1, dtype=torch.int64, device=self.device)},
                    "l2": {"sums": torch.zeros(T - 1, dtype=torch.float64, device=self.device), "sumsq": torch.zeros(T - 1, dtype=torch.float64, device=self.device), "counts": torch.zeros(T - 1, dtype=torch.int64, device=self.device)},
                    "cos": {"sums": torch.zeros(T - 1, dtype=torch.float64, device=self.device), "sumsq": torch.zeros(T - 1, dtype=torch.float64, device=self.device), "counts": torch.zeros(T - 1, dtype=torch.int64, device=self.device)},
                }
                self.current_batch_collected = 0
                self.current_batch_idx += 1

            # tier 聚合（使用 step-change），使用 _step_counter 順序 (0→99)
            h, w = self.block_shapes[block_name]
            tier = self._assign_tier(block_name, h, w, self.base_image_size)
            if tier:
                self._init_tier_accumulators(tier)
                for s in range(1, T):
                    t_prev = outputs[s - 1]  # _step_counter = s-1
                    t_curr = outputs[s]      # _step_counter = s
                    l1_vals, l1_rate_vals, l2_vals, cos_vals = self._calc_metrics_batch(t_prev, t_curr)
                    self.tier_step_sums[tier]["l1"][s - 1] += l1_vals.sum()
                    self.tier_step_sums[tier]["l1_rate"][s - 1] += l1_rate_vals.sum()
                    self.tier_step_sums[tier]["l2"][s - 1] += l2_vals.sum()
                    self.tier_step_sums[tier]["cos"][s - 1] += cos_vals.sum()
                    self.tier_step_sumsq[tier]["l1"][s - 1] += (l1_vals ** 2).sum()
                    self.tier_step_sumsq[tier]["l1_rate"][s - 1] += (l1_rate_vals ** 2).sum()
                    self.tier_step_sumsq[tier]["l2"][s - 1] += (l2_vals ** 2).sum()
                    self.tier_step_sumsq[tier]["cos"][s - 1] += (cos_vals ** 2).sum()
                    self.tier_step_counts[tier][s - 1] += l1_vals.numel()

        self._batch_outputs = {}
        self._batch_size = None

    def _calc_metrics_batch(self, t1: torch.Tensor, t2: torch.Tensor, eps: float = 1e-6):
        # L1rel: mean(|x-y|) / mean(|x|,|y|) - 對稱版本
        diff = torch.abs(t1 - t2)
        l1_diff = diff.mean(dim=(1, 2, 3))
        l1_ref = (torch.abs(t1).mean(dim=(1, 2, 3)) + torch.abs(t2).mean(dim=(1, 2, 3))) / 2.0 + eps
        l1_vals = l1_diff / l1_ref

        # L1rel 變化率: ||x-y||_1 / ||x||_1 - 非對稱版本，使用 L1 norm（不取 mean）
        # ||x||_1 = sum(|x|)，即所有元素的絕對值之和
        l1_rate_diff = diff.sum(dim=(1, 2, 3))  # sum(|t1-t2|)
        l1_rate_ref = torch.abs(t1).sum(dim=(1, 2, 3)) + eps  # sum(|t1|)
        l1_rate_vals = l1_rate_diff / l1_rate_ref

        # L2rel: ||x-y||2 / (||x||2 + ||y||2)/2
        diff_sq = (t1 - t2).pow(2).mean(dim=(1, 2, 3)).sqrt()
        l2_ref = (t1.pow(2).mean(dim=(1, 2, 3)).sqrt() + t2.pow(2).mean(dim=(1, 2, 3)).sqrt()) / 2.0 + eps
        l2_vals = diff_sq / l2_ref

        # Cosine similarity
        t1_flat = t1.view(t1.size(0), -1)
        t2_flat = t2.view(t2.size(0), -1)
        dot = (t1_flat * t2_flat).sum(dim=1)
        denom = t1_flat.norm(dim=1) * t2_flat.norm(dim=1) + eps
        cos_vals = dot / denom

        return l1_vals, l1_rate_vals, l2_vals, cos_vals

    def finalize(self):
        """只輸出：result_npz / L1 / cosine"""
        for block_name in self.block_sums.keys():
            # 在 GPU 上計算 mean，然後轉移到 CPU
            l1_mean_tensor = self._safe_div(self.block_sums[block_name]["l1"], self.block_counts[block_name])
            l1_rate_mean_tensor = self._safe_div(self.block_sums[block_name]["l1_rate"], self.block_counts[block_name])
            l2_mean_tensor = self._safe_div(self.block_sums[block_name]["l2"], self.block_counts[block_name])
            # cosine 使用獨立的 counts，因為累加方式不同（batch 數 vs 樣本數）
            cos_mean_tensor = self._safe_div(self.block_sums[block_name]["cos"], self.block_counts_cos[block_name])
            
            # 轉移到 CPU 並轉換為 numpy
            l1_mean = l1_mean_tensor.cpu().numpy() if isinstance(l1_mean_tensor, torch.Tensor) else l1_mean_tensor
            l1_rate_mean = l1_rate_mean_tensor.cpu().numpy() if isinstance(l1_rate_mean_tensor, torch.Tensor) else l1_rate_mean_tensor
            l2_mean = l2_mean_tensor.cpu().numpy() if isinstance(l2_mean_tensor, torch.Tensor) else l2_mean_tensor
            cos_mean = cos_mean_tensor.cpu().numpy() if isinstance(cos_mean_tensor, torch.Tensor) else cos_mean_tensor
            
            # 調試信息：檢查 cosine 矩陣的統計信息
            cos_min, cos_max = cos_mean.min(), cos_mean.max()
            cos_diag = np.diag(cos_mean)
            cos_off_diag = cos_mean[~np.eye(cos_mean.shape[0], dtype=bool)]
            LOGGER.info(f"[Similarity] {block_name} cosine: min={cos_min:.6f}, max={cos_max:.6f}, "
                       f"diag_mean={cos_diag.mean():.6f}, off_diag_mean={cos_off_diag.mean():.6f}, "
                       f"off_diag_std={cos_off_diag.std():.6f}")

            l1_step_mean, l1_step_std = self._step_mean_std(block_name, "l1")
            cos_step_mean, cos_step_std = self._step_mean_std(block_name, "cos")

            npz_path = self.result_npz_dir / f"{block_name.replace('.', '_')}.npz"
            mapped_t = (
                np.array(self.mapped_t_list, dtype=np.int32)
                if self.mapped_t_list is not None
                else np.array([-1] * len(self.step_idx_list), dtype=np.int32)
            )
            np.savez_compressed(
                npz_path,
                l1rel=l1_mean.astype(self.save_dtype),
                cosine=cos_mean.astype(self.save_dtype),
                step_idx=np.array(self.step_idx_list, dtype=np.int32),
                mapped_t=mapped_t,
                l1_step_mean=l1_step_mean.astype(self.save_dtype),
                l1_step_std=l1_step_std.astype(self.save_dtype),
                cos_step_mean=cos_step_mean.astype(self.save_dtype),
                cos_step_std=cos_step_std.astype(self.save_dtype),
            )

            block_slug = block_name.replace(".", "_")
            l1_block_dir = self.l1_dir / block_slug
            cos_block_dir = self.cos_dir / block_slug
            for d in [l1_block_dir, cos_block_dir]:
                d.mkdir(parents=True, exist_ok=True)

            self._save_matrix_csv(l1_mean, l1_block_dir / f"{block_slug}_l1rel.csv")

            # 繪製單組折線圖（原有功能）
            self._plot_step_curve(l1_step_mean, l1_step_std, l1_block_dir / f"{block_slug}_l1.png", "L1rel")
            
            # 如果有 per-batch 數據，繪製多組折線圖
            if block_name in self.batch_step_data and len(self.batch_step_data[block_name]) > 0:
                self._plot_multi_batch_curves(
                    block_name, "l1", l1_block_dir / f"{block_slug}_l1_multi_batch.png", "L1rel (Multi-Batch)"
                )
            self._plot_heatmap(cos_mean, cos_block_dir / f"{block_slug}_cosine_heatmap.png", "Cosine similarity")
        # 不輸出 tier 聚合圖

    def _plot_tiers(self):
        for tier in ["full", "half", "quarter", "eighth"]:
            if tier not in self.tier_step_sums:
                continue
            for metric in ["l1", "l1_rate", "l2", "cos"]:
                mean, std = self._tier_mean_std(tier, metric)
                title = f"{tier} ({metric})"
                fname = f"{tier}_{metric}.png"
                out = self.tier_plot_dir / fname
                self._plot_step_curve(mean, std, out, title)

    def _step_mean_std(self, block_name: str, metric: str):
        sums = self.block_step_sums[block_name][metric]
        sums_sq = self.block_step_sumsq[block_name][metric]
        counts = self.block_step_counts[block_name]
        mean = self._safe_div(sums, counts)
        var = self._safe_div(sums_sq, counts) - mean ** 2
        # 支持 GPU tensor 和 numpy array
        if isinstance(var, torch.Tensor):
            std = torch.sqrt(torch.clamp(var, min=0.0))
            # 轉移到 CPU 並轉換為 numpy
            return mean.cpu().numpy(), std.cpu().numpy()
        else:
            std = np.sqrt(np.maximum(var, 0.0))
            return mean, std

    def _tier_mean_std(self, tier: str, metric: str):
        sums = self.tier_step_sums[tier][metric]
        sums_sq = self.tier_step_sumsq[tier][metric]
        counts = self.tier_step_counts[tier]
        mean = self._safe_div(sums, counts)
        var = self._safe_div(sums_sq, counts) - mean ** 2
        # 支持 GPU tensor 和 numpy array
        if isinstance(var, torch.Tensor):
            std = torch.sqrt(torch.clamp(var, min=0.0))
            # 轉移到 CPU 並轉換為 numpy
            return mean.cpu().numpy(), std.cpu().numpy()
        else:
            std = np.sqrt(np.maximum(var, 0.0))
            return mean, std
    
    def _batch_mean_std(self, batch_data: dict, metric: str):
        """計算單個 batch 的 mean 和 std"""
        sums = batch_data[metric]["sums"]
        sums_sq = batch_data[metric]["sumsq"]
        counts = batch_data[metric]["counts"]
        mean = self._safe_div(sums, counts)
        var = self._safe_div(sums_sq, counts) - mean ** 2
        std = np.sqrt(np.maximum(var, 0.0))
        return mean, std

    def _plot_step_curve(self, mean: np.ndarray, std: np.ndarray, out_path: Path, title: str):
        # X 軸：從 t0→t1, t1→t2, ..., t98→t99，對應 step_counter 0→99
        # x[i] 表示從 step_counter=i 到 step_counter=i+1 的變化
        x = np.arange(len(mean))
        plt.figure(figsize=(8, 4))
        plt.plot(x, mean, color="blue", linewidth=2)
        plt.fill_between(x, mean - std, mean + std, color="blue", alpha=0.2)
        plt.title(title)
        plt.xlabel("Denoising steps")
        plt.ylabel("Relative metric")
        # X 軸標籤：0, 10, 20, 30, ..., 100
        # 生成 0, 10, 20, ..., 100 的標籤位置（每 10 個 timestep）
        tick_positions = []
        tick_labels = []
        for i in range(0, len(mean), 10):
            tick_positions.append(i)
            tick_labels.append(str(i))
        # 添加最後一個點（如果不在列表中）
        if len(mean) - 1 not in tick_positions:
            tick_positions.append(len(mean) - 1)
            tick_labels.append(str(len(mean) - 1))
        plt.xticks(tick_positions, tick_labels)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
    
    def _plot_multi_batch_curves(self, block_name: str, metric: str, out_path: Path, title: str):
        """繪製多組 batch 的折線圖（4 組數據在同一張圖上）"""
        if block_name not in self.batch_step_data or len(self.batch_step_data[block_name]) == 0:
            return
        
        batch_data_list = self.batch_step_data[block_name]
        num_batches = len(batch_data_list)
        
        # 計算每組的 mean 和 std
        means_list = []
        stds_list = []
        for batch_data in batch_data_list:
            mean, std = self._batch_mean_std(batch_data, metric)
            means_list.append(mean)
            stds_list.append(std)
        
        # X 軸：從 t0→t1, t1→t2, ..., t98→t99
        x = np.arange(len(means_list[0]))
        plt.figure(figsize=(10, 6))
        
        # 繪製每組的線條和陰影（相同顏色，alpha 疊加會讓重疊區域變深）
        line_color = "blue"
        fill_color = "blue"
        line_width = 1.0  # 統一調細
        
        for mean, std in zip(means_list, stds_list):
            # 繪製線條（相同顏色、相同粗細）
            plt.plot(x, mean, color=line_color, linewidth=line_width, alpha=0.8)
            # 繪製陰影（相同顏色，alpha 疊加）
            plt.fill_between(x, mean - std, mean + std, color=fill_color, alpha=0.15)
        
        plt.title(title)
        plt.xlabel("Denoising steps")
        plt.ylabel("Relative metric")
        
        # X 軸標籤：0, 10, 20, 30, ..., 100
        tick_positions = []
        tick_labels = []
        for i in range(0, len(means_list[0]), 10):
            tick_positions.append(i)
            tick_labels.append(str(i))
        if len(means_list[0]) - 1 not in tick_positions:
            tick_positions.append(len(means_list[0]) - 1)
            tick_labels.append(str(len(means_list[0]) - 1))
        plt.xticks(tick_positions, tick_labels)
        
        # 添加圖例說明有多少組數據
        plt.text(0.02, 0.98, f"{num_batches} batches", 
                transform=plt.gca().transAxes, 
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()

    def _plot_heatmap(self, matrix: np.ndarray, out_path: Path, title: str):
        # 參考 metrics.py 的實現
        
        # 1) 轉成 float，避免 object / weird dtype
        m = np.array(matrix, dtype=np.float64)

        # 2) 找出 NaN / Inf
        bad = ~np.isfinite(m)
        bad_ratio = bad.mean()

        # 如果全部都是 NaN/Inf，直接保底成 0（避免整張空白）
        if bad_ratio >= 1.0:
            m = np.zeros_like(m, dtype=np.float64)
            bad = np.zeros_like(m, dtype=bool)
            bad_ratio = 0.0

        # 3) Cosine similarity 範圍 [0, 1]（實驗確認結果都 > 0）
        if "cosine" in title.lower():
            # 確保值在 [0, 1] 範圍內（避免數值誤差）
            m = np.clip(m, 0.0, 1.0)
            
            # 檢查實際值域，如果範圍很小（比如都在 0.9-1.0），可以放大顯示
            actual_min, actual_max = np.nanmin(m), np.nanmax(m)
            if actual_max - actual_min < 0.1 and actual_min > 0.8:
                # 值域很小且接近 1，放大顯示以突出差異
                vmin = max(actual_min - 0.05, 0.0)
                vmax = min(actual_max + 0.05, 1.0)
            else:
                vmin, vmax = 0.0, 1.0  # 使用完整範圍
            
            # 使用 sequential colormap，讓顏色漸變更清晰易讀
            # plasma: 紫→粉→黃，對比度高，適合顯示細微差異
            # inferno: 黑→紫→紅→黃，也很適合
            cmap = "plasma"  # 或 "inferno", "viridis"
        else:
            # 用 nanmin/nanmax 取值域
            vmin = np.nanmin(m)
            vmax = np.nanmax(m)
            # 如果值域太窄，給一個合理顯示範圍
            if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or abs(vmax - vmin) < 1e-6:
                vmin, vmax = 0.0, vmax if vmax > 0 else 1.0
            cmap = "viridis"

        # 4) 把 NaN/Inf 用 masked array 表示
        mm = np.ma.array(m, mask=bad)

        # 5) 使用 seaborn heatmap（參考 metrics.py）
        # 增大圖片尺寸以容納更多標籤
        plt.figure(figsize=(10, 8))
        
        # 設定 x/y 軸標籤：根據矩陣大小動態調整間隔
        # 如果矩陣很大（>50），每 5 個顯示一次；否則每 2 個顯示一次
        if m.shape[0] > 50:
            tick_interval = 5
        else:
            tick_interval = 2
        
        tick_positions = list(range(0, m.shape[0], tick_interval))
        tick_labels = [str(i) for i in tick_positions]
        # 添加最後一個點
        if m.shape[0] - 1 not in tick_positions:
            tick_positions.append(m.shape[0] - 1)
            tick_labels.append(str(m.shape[0] - 1))
        
        # 使用 seaborn heatmap，但不直接設定 ticklabels（會導致擠在一起）
        # 先繪製 heatmap，然後手動設定 ticks
        ax = sns.heatmap(mm, 
                        cmap=cmap, 
                        annot=False, 
                        vmin=vmin, 
                        vmax=vmax,
                        linewidths=0.1,
                        xticklabels=False,  # 先不顯示標籤
                        yticklabels=False,  # 先不顯示標籤
                        cbar_kws={'label': 'Similarity'})
        
        # 手動設定 x 軸 ticks 和標籤（旋轉 45 度避免重疊）
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)
        
        # 手動設定 y 軸 ticks 和標籤
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels, rotation=0, fontsize=9)
        
        plt.title(title, fontsize=12)
        plt.xlabel("Denoising steps (step_counter)", fontsize=10)
        plt.ylabel("Denoising steps (step_counter)", fontsize=10)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

    def _save_matrix_csv(self, matrix: np.ndarray, out_path: Path):
        labels = [f"t_{t}" for t in self.step_idx_list]
        df = pd.DataFrame(matrix, index=labels, columns=labels)
        df.to_csv(out_path)

    def _safe_div(self, num, denom):
        """安全除法，支持 numpy array 和 torch tensor"""
        if isinstance(num, torch.Tensor):
            # GPU tensor 版本
            # 確保輸出 dtype 是 float64，即使輸入是其他類型
            out = torch.zeros_like(num, dtype=torch.float64)
            mask = denom > 0
            out[mask] = num[mask].to(torch.float64) / denom[mask].to(torch.float64)
            return out
        else:
            # numpy array 版本
            out = np.zeros_like(num, dtype=np.float64)
            mask = denom > 0
            out[mask] = num[mask] / denom[mask]
            return out

def _collect_prefixed_state_dict(
    ckpt: Dict[str, Any],
    prefix: str,
    add_prefix: str = "",
) -> Dict[str, torch.Tensor]:
    """從扁平 checkpoint 中抓取指定 prefix 的權重，移除 prefix 後可選擇補新前綴。"""
    out: Dict[str, torch.Tensor] = {}
    for k, v in ckpt.items():
        if isinstance(k, str) and k.startswith(prefix) and torch.is_tensor(v):
            new_k = k[len(prefix):]
            if add_prefix:
                new_k = add_prefix + new_k
            out[new_k] = v
    return out

def _load_quant_and_ema_from_ckpt(base_model: LitModel, quant_model: nn.Module, ckpt: Dict[str, Any]) -> None:
    """
    載入策略（嚴格符合需求）:
    1) base_model.ema_model 架構 = deepcopy(quant_model)
    2) base_model.ema_model 權重只吃 `ema_model.model.*`
    3) base_model.model 權重吃 `model.model.*`（若存在）
    """
    setattr(base_model, 'model', quant_model)
    base_model.ema_model = deepcopy(quant_model)

    # base_model.ema_model(=quant_model) 的 state_dict key 以 `model.` 開頭
    # ckpt 來源是 `ema_model.model.*`，移除後需補回 `model.` 才能正確對位
    ema_sd = _collect_prefixed_state_dict(ckpt, "ema_model.model.", add_prefix="model.")
    if len(ema_sd) == 0:
        raise KeyError(
            "Checkpoint 不包含 `ema_model.model.*` 權重；無法依指定規則載入 EMA 生成模型。"
        )
    ema_msg = base_model.ema_model.load_state_dict(ema_sd, strict=False)
    ema_effective_loaded = len(ema_sd) - len(ema_msg.unexpected_keys)
    LOGGER.info(
        "EMA(load from ema_model.model.*): provided=%d effective_loaded=%d missing=%d unexpected=%d",
        len(ema_sd), ema_effective_loaded, len(ema_msg.missing_keys), len(ema_msg.unexpected_keys)
    )
    if len(ema_msg.missing_keys) > 0:
        LOGGER.warning("EMA missing keys (前10): %s", ema_msg.missing_keys[:10])
    if len(ema_msg.unexpected_keys) > 0:
        LOGGER.warning("EMA unexpected keys (前10): %s", ema_msg.unexpected_keys[:10])

    model_sd = _collect_prefixed_state_dict(ckpt, "model.model.", add_prefix="model.")
    if len(model_sd) > 0:
        model_msg = base_model.model.load_state_dict(model_sd, strict=False)
        model_effective_loaded = len(model_sd) - len(model_msg.unexpected_keys)
        LOGGER.info(
            "MODEL(load from model.model.*): provided=%d effective_loaded=%d missing=%d unexpected=%d",
            len(model_sd), model_effective_loaded, len(model_msg.missing_keys), len(model_msg.unexpected_keys)
        )
    else:
        LOGGER.warning("Checkpoint 未找到 `model.model.*`，base_model.model 保持目前權重。")

#=============================================================================
# pred_xstart quantile analysis pipeline (independent)
#=============================================================================

PRED_XSTART_Q_LIST = np.array([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99], dtype=np.float64)


def _pred_xstart_images_count(images_mode: str) -> int:
    if images_mode == "debug":
        return 32
    if images_mode == "v1":
        return 64
    if images_mode == "official":
        return 128
    raise ValueError(f"Invalid images_mode: {images_mode}")


def _resolve_best_ckpt_path(num_diffusion_steps: int) -> str:
    # 對齊現有程式邏輯：20steps 對應 _20steps checkpoint
    if num_diffusion_steps == 100:
        return "QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best.pth"
    return "QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best_20steps.pth"


def _make_noise_banks(
    *,
    num_images: int,
    chunk_batch: int,
    img_size: int,
    style_ch: int,
    seed: int,
    device: torch.device,
):
    """
    baseline/v2 必須共用同一份 inputs：
    - x_T (diffusion noise) 連續張量銀行
    - latent_noise (latent condition noise) 連續張量銀行
    """
    # 用獨立 generator，避免後續模型 forward 造成全域 RNG 進位而影響噪聲一致性
    cpu_gen = torch.Generator(device="cpu")
    cpu_gen.manual_seed(seed)

    x_T = torch.randn(
        (num_images, 3, img_size, img_size),
        generator=cpu_gen,
        device="cpu",
        dtype=torch.float32,
    )
    latent_noise = torch.randn(
        (num_images, style_ch),
        generator=cpu_gen,
        device="cpu",
        dtype=torch.float32,
    )

    # 可選：檢查切塊數量是否對齊 chunk_batch
    assert num_images % chunk_batch == 0, "num_images must be divisible by chunk_batch"
    return x_T.to(device), latent_noise.to(device)


@torch.no_grad()
def _compute_latent_cond(
    *,
    conf: Any,
    latent_sampler,
    latent_net,
    latent_noise_chunk: torch.Tensor,
    conds_mean: torch.Tensor,
    conds_std: torch.Tensor,
    device: torch.device,
):
    # 直接照 render_uncondition 的 latent diffusion 邏輯複製，確保語意一致
    cond = latent_sampler.sample(
        model=latent_net,
        noise=latent_noise_chunk,
        clip_denoised=conf.latent_clip_sample,
    )

    if getattr(conf, "latent_znormalize", False):
        cond = cond * conds_std.to(device) + conds_mean.to(device)
    return cond


@torch.no_grad()
def _collect_pred_xstart_histograms(
    *,
    model: nn.Module,
    sampler,
    latent_sampler,
    conf: Any,
    x_T_bank: torch.Tensor,
    latent_noise_bank: torch.Tensor,
    conds_mean: torch.Tensor,
    conds_std: torch.Tensor,
    T: int,
    pred_key: np.ndarray,
    clip_denoised: bool,
    hist_range_min: float,
    hist_range_max: float,
    hist_bins: int,
    chunk_batch: int,
    device: torch.device,
):
    """
    以 progressive loop 取得每個 timestep 的 out['pred_xstart']，
    依 t_idx 直接寫入 hist_counts[t_idx, :].
    """
    hist_counts = torch.zeros(
        (T, hist_bins),
        device=device,
        dtype=torch.float64,
    )

    # 以 chunk 為單位避免一次搬太多 noise/latent_noise
    num_images = x_T_bank.shape[0]
    assert num_images == latent_noise_bank.shape[0]
    num_chunks = num_images // chunk_batch

    for ci in range(num_chunks):
        b0 = ci * chunk_batch
        b1 = (ci + 1) * chunk_batch

        x_T_chunk = x_T_bank[b0:b1].to(device)
        latent_noise_chunk = latent_noise_bank[b0:b1].to(device)

        # latent condition
        if conf.train_mode.is_latent_diffusion():
            cond_chunk = _compute_latent_cond(
                conf=conf,
                latent_sampler=latent_sampler,
                latent_net=model.latent_net,
                latent_noise_chunk=latent_noise_chunk,
                conds_mean=conds_mean,
                conds_std=conds_std,
                device=device,
            )
            model_kwargs = {"cond": cond_chunk, "x_start": None}
        else:
            model_kwargs = None

        for out in sampler.ddim_sample_loop_progressive(
            model=model,
            noise=x_T_chunk,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
            device=device,
            progress=False,
            eta=0.0,
            cache_scheduler=None,
        ):
            # out['t'] 在 ddim_sample_loop_progressive 中是 t tensor（值域 0..T-1 的 index）
            t_idx = int(out["t"][0].item())
            pred_flat = out["pred_xstart"].reshape(-1)

            # 強制對齊 histogram range（雖然 clip_denoised=True 通常已 clamp 到 [-1, 1]）
            pred_flat = pred_flat.clamp(hist_range_min, hist_range_max)

            counts = torch.histc(
                pred_flat,
                bins=hist_bins,
                min=hist_range_min,
                max=hist_range_max,
            ).to(dtype=torch.float64)
            hist_counts[t_idx] += counts

    hist_counts_np = hist_counts.cpu().numpy()
    return hist_counts_np


def _hist_counts_to_quantiles(
    *,
    hist_counts: np.ndarray,
    bin_edges: np.ndarray,
    q_list: np.ndarray,
):
    """
    使用 cumulative counts 反推 quantiles。
    每個 quant 位點落在 bin 內做線性插值。
    """
    T, bins = hist_counts.shape
    assert bin_edges.shape == (bins + 1,)

    quantiles = np.zeros((T, len(q_list)), dtype=np.float64)
    cumulative = np.cumsum(hist_counts, axis=1)
    total = cumulative[:, -1]

    for t in range(T):
        if total[t] <= 0:
            quantiles[t, :] = np.nan
            continue

        for qi, q in enumerate(q_list):
            target = q * total[t]
            idx = int(np.searchsorted(cumulative[t], target, side="left"))

            # idx 對應落在哪個 bin
            if idx <= 0:
                left_edge = bin_edges[0]
                right_edge = bin_edges[1]
                prev_cdf = 0.0
                count_in_bin = hist_counts[t, 0]
            elif idx >= bins:
                left_edge = bin_edges[-2]
                right_edge = bin_edges[-1]
                prev_cdf = cumulative[t, bins - 2] if bins >= 2 else 0.0
                count_in_bin = hist_counts[t, bins - 1]
            else:
                left_edge = bin_edges[idx]
                right_edge = bin_edges[idx + 1]
                prev_cdf = cumulative[t, idx - 1]
                count_in_bin = hist_counts[t, idx]

            if count_in_bin <= 0:
                quantiles[t, qi] = left_edge
            else:
                frac = (target - prev_cdf) / count_in_bin
                frac = float(np.clip(frac, 0.0, 1.0))
                quantiles[t, qi] = left_edge + frac * (right_edge - left_edge)

    return quantiles


def _save_pred_xstart_quantile_npz(
    *,
    out_npz_path: Path,
    hist_counts: np.ndarray,
    bin_edges: np.ndarray,
    quantiles: np.ndarray,
    t: np.ndarray,
    metadata: Dict[str, Any],
):
    out_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz_path,
        hist_counts=hist_counts,
        bin_edges=bin_edges,
        quantiles=quantiles,
        t=t,
        **metadata,
    )


def plot_pred_xstart_quantile_overlay(
    *,
    npz_baseline: Path,
    npz_v2: Path,
    out_png: Path,
):
    """
    baseline vs v2 直接疊圖：
    - outer band: q01 ~ q99
    - inner band: q25 ~ q75
    - median: q50
    y-axis 固定 [-1, 1]
    """
    import matplotlib.pyplot as plt

    nb = np.load(npz_baseline, allow_pickle=True)
    nv = np.load(npz_v2, allow_pickle=True)

    q_b = nb["quantiles"]  # (T,7)
    q_v = nv["quantiles"]
    t = nb["t"]  # (T,)

    def bands(q_arr: np.ndarray):
        q01 = q_arr[:, 0]
        q25 = q_arr[:, 2]
        q50 = q_arr[:, 3]
        q75 = q_arr[:, 4]
        q99 = q_arr[:, 6]
        return q01, q25, q50, q75, q99

    b01, b25, b50, b75, b99 = bands(q_b)
    v01, v25, v50, v75, v99 = bands(q_v)

    plt.figure(figsize=(10, 5))
    plt.fill_between(t, b01, b99, color="tab:blue", alpha=0.15, label="baseline q01-q99")
    plt.fill_between(t, b25, b75, color="tab:blue", alpha=0.25, label="baseline q25-q75")
    plt.plot(t, b50, color="tab:blue", linewidth=2.0, label="baseline q50")

    plt.fill_between(t, v01, v99, color="tab:orange", alpha=0.15, label="v2 q01-q99")
    plt.fill_between(t, v25, v75, color="tab:orange", alpha=0.25, label="v2 q25-q75")
    plt.plot(t, v50, color="tab:orange", linewidth=2.0, label="v2 q50")

    plt.ylim(-1.0, 1.0)
    plt.xlim(0, len(t) - 1)
    plt.xlabel("t_idx (out['t'] index, ascending 0..T-1)")
    plt.ylabel("pred_xstart value")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


@time_operation
def main_pred_xstart_quantile_analysis(
    *,
    images_mode: str = "official",
    chunk_batch: int = 32,
    hist_range_min: float = -1.0,
    hist_range_max: float = 1.0,
    hist_bins: int = 4096,
    clip_denoised: bool = True,
    seed: int = 0,
    T: Optional[int] = None,
    pred_output_root: str = "QATcode/cache_method/pred_xstart_quantile_analysis",
    skip_plot: bool = False,
):
    """
    baseline / quantize_ver2:
    - 共用 x_T 與 latent_noise
    - 用 progressive loop 累積 hist_counts[t_idx, :]
    - 從 cumulative counts 推 quantiles，輸出可直接疊圖的 npz
    """
    _seed_all(seed)
    device = CONFIG.DEVICE
    if T is None:
        T = CONFIG.NUM_DIFFUSION_STEPS
    num_images = _pred_xstart_images_count(images_mode)

    # baseline 先載入一次（取得 conf、conds_mean/std、以及 samplers）
    LOGGER.info(f"[pred_xstart_quantile] images_mode={images_mode}, num_images={num_images}, T={T}, chunk_batch={chunk_batch}")
    LOGGER.info(f"[pred_xstart_quantile] hist_range=[{hist_range_min},{hist_range_max}], bins={hist_bins}, clip_denoised={clip_denoised}")

    base_model_baseline: LitModel = load_diffae_model(CONFIG.MODEL_PATH)
    base_model_baseline.to(device)
    base_model_baseline.eval()
    base_model_baseline.setup()
    base_model_baseline.train_dataloader()

    conf = base_model_baseline.conf.clone()

    sampler = conf._make_diffusion_conf(T=T).make_sampler()
    latent_sampler = conf._make_latent_diffusion_conf(T=T).make_sampler()

    # sampler timestep map (optional)
    sampler_timestep_map_exists = hasattr(sampler, "timestep_map") and getattr(sampler, "timestep_map", None)
    mapped_t = (
        np.array(getattr(sampler, "timestep_map"), dtype=np.int32)
        if sampler_timestep_map_exists
        else None
    )

    x_T_bank, latent_noise_bank = _make_noise_banks(
        num_images=num_images,
        chunk_batch=chunk_batch,
        img_size=conf.img_size,
        style_ch=conf.style_ch,
        seed=seed,
        device=device,
    )

    hist_out_root = Path(pred_output_root) / f"T_{T}" / ("debug" if images_mode == "debug" else images_mode)
    hist_out_root.mkdir(parents=True, exist_ok=True)

    bin_edges = np.linspace(hist_range_min, hist_range_max, hist_bins + 1, dtype=np.float64)
    t = np.arange(T, dtype=np.int32)
    pred_key = np.array("pred_xstart")

    def collect_for(tag: str, model: nn.Module, ckpt_path: str):
        LOGGER.info(f"[pred_xstart_quantile] collecting: {tag}")
        hist_counts = _collect_pred_xstart_histograms(
            model=model,
            sampler=sampler,
            latent_sampler=latent_sampler,
            conf=conf,
            x_T_bank=x_T_bank,
            latent_noise_bank=latent_noise_bank,
            conds_mean=base_model_baseline.conds_mean,
            conds_std=base_model_baseline.conds_std,
            T=T,
            pred_key=pred_key,
            clip_denoised=clip_denoised,
            hist_range_min=hist_range_min,
            hist_range_max=hist_range_max,
            hist_bins=hist_bins,
            chunk_batch=chunk_batch,
            device=device,
        )

        quantiles = _hist_counts_to_quantiles(
            hist_counts=hist_counts,
            bin_edges=bin_edges,
            q_list=PRED_XSTART_Q_LIST,
        )

        out_npz = hist_out_root / f"pred_xstart_quantiles_{tag}_seed{seed}_N{num_images}_bins{hist_bins}.npz"

        metadata: Dict[str, Any] = {
            "pred_key": pred_key,
            "clip_denoised": np.array(clip_denoised, dtype=np.bool_),
            "hist_range_min": np.array(hist_range_min, dtype=np.float64),
            "hist_range_max": np.array(hist_range_max, dtype=np.float64),
            "hist_bins": np.array(hist_bins, dtype=np.int32),
            "q_list": PRED_XSTART_Q_LIST,
            "x_axis_def": np.array("out[t] (timestep index), ascending via stats[t_idx]"),
            "t_total": np.array(T, dtype=np.int32),
            "model_tag": np.array(tag),
            "ckpt_path": np.array(ckpt_path),
            "seed": np.array(seed, dtype=np.int32),
            "num_images": np.array(num_images, dtype=np.int32),
            "chunk_batch": np.array(chunk_batch, dtype=np.int32),
            "sampler_timestep_map_exists": np.array(bool(sampler_timestep_map_exists), dtype=np.bool_),
        }
        if mapped_t is not None:
            metadata["mapped_t"] = mapped_t

        _save_pred_xstart_quantile_npz(
            out_npz_path=out_npz,
            hist_counts=hist_counts,
            bin_edges=bin_edges,
            quantiles=quantiles,
            t=t,
            metadata=metadata,
        )
        return out_npz

    # baseline: 用 baseline 的 ema_model（不做 quant）
    npz_baseline = collect_for("baseline", base_model_baseline.ema_model, CONFIG.MODEL_PATH)

    # v2: 載入一次獨立 base_model，再掛上 quant checkpoint（避免覆寫 baseline）
    base_model_v2: LitModel = load_diffae_model(CONFIG.MODEL_PATH)
    base_model_v2.to(device)
    base_model_v2.eval()
    base_model_v2.setup()

    diffusion_model = base_model_v2.ema_model
    quant_model: QuantModel_DiffAE_LoRA = create_float_quantized_model(
        diffusion_model,
        num_steps=T,
        lora_rank=CONFIG.LORA_RANK,
        mode=CONFIG.MODE,
    )
    quant_model.to(device)
    quant_model.eval()

    # 動態掛上子模組時，需要初始化 dequantizer（對齊原有 main_float_model）
    for _, module in quant_model.named_modules():
        if isinstance(module, QuantModule_DiffAE_LoRA) and getattr(module, "ignore_reconstruction", False) is False:
            module.intn_dequantizer = SimpleDequantizer(
                uaq=module.weight_quantizer,
                weight=module.weight,
            ).to(device)

    # 初始化量化參數（TemporalActivationQuantizer 需要先跑一小段 forward）
    cali_images, cali_t, cali_y = load_calibration_data()
    quant_model.set_first_last_layer_to_8bit()
    quant_model.set_quant_state(True, True)
    with torch.no_grad():
        _ = quant_model(x=cali_images[:4].to(device), t=cali_t[:4].to(device), cond=cali_y[:4].to(device))

    ckpt_path_v2 = _resolve_best_ckpt_path(T)
    ckpt = torch.load(ckpt_path_v2, map_location="cpu", weights_only=False)
    _load_quant_and_ema_from_ckpt(base_model_v2, quant_model, ckpt)

    # v2 的前向用 base_model_v2.ema_model（已被替換成量化版本）
    npz_v2 = collect_for("v2_latest", base_model_v2.ema_model, ckpt_path_v2)

    if not skip_plot:
        out_png = hist_out_root / f"pred_xstart_quantiles_overlay_{images_mode}_T{T}_seed{seed}.png"
        plot_pred_xstart_quantile_overlay(
            npz_baseline=npz_baseline,
            npz_v2=npz_v2,
            out_png=out_png,
        )
        LOGGER.info(f"[pred_xstart_quantile] plot saved: {out_png}")

    LOGGER.info("[pred_xstart_quantile] done.")
    return npz_baseline, npz_v2

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
    _seed_all(CONFIG.SEED)

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

        # 動態掛上子模組時，要記得 .to(device)
        for name, module in quant_model.named_modules():
            if isinstance(module, QuantModule_DiffAE_LoRA) and module.ignore_reconstruction is False:
                module.intn_dequantizer = SimpleDequantizer(uaq=module.weight_quantizer, weight=module.weight).to(CONFIG.DEVICE)

        # 若你之後覆寫 buffer，請用 copy_ 並對齊裝置
        #for name, module in quant_model.named_modules():
        #    if isinstance(module, QuantModule_DiffAE_LoRA) and module.ignore_reconstruction is False:
        #        module.intn_dequantizer.delta.data.copy_(module.weight_quantizer.delta.to(CONFIG.DEVICE))
        #        module.intn_dequantizer.zero_point.data.copy_(module.weight_quantizer.zero_point.to(CONFIG.DEVICE))

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

        #for name, module in quant_model.named_modules():
        #    if isinstance(module, QuantModule_DiffAE_LoRA) and module.ignore_reconstruction is False:
        #        # 強制在 CPU 上轉換權重，避免 sm_120 不支援的 CUDA kernel
        #        device = module.weight.data.device
        #        with torch.no_grad():
        #            weight_cpu = module.weight.data.detach().cpu()
        #            weight_uint8 = weight_cpu.to(torch.uint8)
        #            module.weight.data = weight_uint8.to(device)
        
        #print('First run to init the model') ## need run to init emporal act quantizer
        with torch.no_grad():
            _ = quant_model(x=cali_images[:4].to(device), t=cali_t[:4].to(device), cond=cali_y[:4].to(device))

        #quant_model.set_quant_state(False, False)
        

        ckpt = torch.load(CONFIG.BEST_CKPT_PATH, map_location='cpu', weights_only=False)
        _load_quant_and_ema_from_ckpt(base_model, quant_model, ckpt)

        
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
        if CONFIG.ENABLE_CACHE_ANALYSIS and CONFIG.ENABLE_SIMILARITY_ANALYSIS:
            LOGGER.warning("同時啟用 Cache Analysis 與 Similarity Analysis，優先執行 Cache Analysis")
        
        if CONFIG.ENABLE_CACHE_ANALYSIS:
            pass
        elif CONFIG.ENABLE_SIMILARITY_ANALYSIS:
            LOGGER.info("=" * 50)
            LOGGER.info("啟用 Similarity Analysis (L1/L2/Cosine)")
            LOGGER.info(f"FID 評估樣本數: {CONFIG.SIMILARITY_SAMPLES}")
            LOGGER.info(f"實際收集樣本數: {CONFIG.SIMILARITY_COLLECT_SAMPLES} (用於加速)")
            LOGGER.info(f"擴散步數: {T}")
            LOGGER.info("=" * 50)

            similarity_root = f"{CONFIG.SIMILARITY_OUTPUT_ROOT}/T_{T}/v2_latest"
            # 計算總共要收集的樣本數：每個 batch 收集 similarity_collect_samples 個
            # 如果 similarity_samples = 128, batch_size = 32，會有 4 個 batch
            # 每個 batch 收集 15 個，總共收集 60 個（但實際上會收集 4 組數據，每組 15 個）
            total_collect_samples = CONFIG.SIMILARITY_COLLECT_SAMPLES * (CONFIG.SIMILARITY_SAMPLES // 32)  # 假設 batch_size = 32
            similarity_collector = SimilarityCollector(
                save_root=similarity_root,
                max_timesteps=T,
                num_samples=total_collect_samples,  # 總共要收集的樣本數
                target_block=CONFIG.SIMILARITY_TARGET_BLOCK,
                dtype=CONFIG.SIMILARITY_SAVE_DTYPE,
                save_cosine_step_plot=CONFIG.SIMILARITY_PLOT_COSINE_STEP,
                base_image_size=conf.img_size,
                device=CONFIG.DEVICE,  # 傳入設備，用於 GPU 計算
                sample_strategy=CONFIG.SIMILARITY_SAMPLE_STRATEGY,  # 採樣策略
            )
            # 設定每個 batch 的收集限制（用於多組折線圖）
            similarity_collector._batch_collect_limit = CONFIG.SIMILARITY_COLLECT_SAMPLES
            similarity_collector.register_hooks(base_model.ema_model, sampler)

            conf.eval_num_images = CONFIG.SIMILARITY_SAMPLES
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
                remove_cache=True,
                clip_latent_noise=False,
                T=T,
                output_dir=f'{conf.generate_dir}_QAT_T{T}_similarity'
            )
            LOGGER.info(f'[Similarity] FID@{CONFIG.SIMILARITY_SAMPLES} {T} steps score: {score}')

            similarity_collector.remove_hooks()
            similarity_collector.finalize()
            LOGGER.info("✅ Similarity Analysis 完成")
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
        
        
        
        torch.cuda.empty_cache()
        global start_time
        start_time = time.time()

    except Exception as e:
        LOGGER.error(f"Error: {e}")
        raise e



        
        

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_steps','--n', type=int, default=100)
    parser.add_argument('--eval_samples', '--es', type=int, default=50000)
    parser.add_argument('--mode', '--m', type=str, default='float')
    parser.add_argument('--enable_similarity', action='store_true',
                        help='啟用 Similarity Analysis (L1/L2/Cosine)')
    parser.add_argument('--run_pred_xstart_quantile_analysis', action='store_true',
                        help='只跑 pred_xstart 每 timestep 分布的 histogram+quantile 分析（baseline vs v2）')
    parser.add_argument('--images_mode', type=str, default='official', choices=['debug', 'v1', 'official'],
                        help='debug=32 images, v1=64 images, official=128 images')
    parser.add_argument('--chunk_batch', type=int, default=32,
                        help='hist 分布收集時的 chunk batch（規格固定 32；用於 64->2 chunks, 128->4 chunks）')
    parser.add_argument('--hist_bins', type=int, default=4096,
                        help='histogram bins（規格固定 4096）')
    parser.add_argument('--seed', type=int, default=0,
                        help='noise/latent_noise 的 seed（寫入 npz metadata）')
    parser.add_argument('--pred_output_root', type=str, default="QATcode/cache_method/pred_xstart_quantile_analysis",
                        help='pred_xstart quantile analysis 輸出根目錄')
    parser.add_argument('--skip_plot', action='store_true',
                        help='只輸出 npz，不產生疊圖')
    parser.add_argument('--similarity_samples', type=int, default=64,
                        help='Similarity FID 評估樣本數（總生成數量）')
    parser.add_argument('--similarity_collect_samples', type=int, default=15,
                        help='實際收集用於相似度計算的樣本數（建議 10-15，用於加速）')
    parser.add_argument('--similarity_target_block', type=str, default=None,
                        help='只分析單一 block (例如 model.input_blocks.0)')
    parser.add_argument('--similarity_output_root', type=str,
                        default="QATcode/cache_method/L1_L2_cosine",
                        help='Similarity 結果輸出根目錄')
    parser.add_argument('--similarity_dtype', type=str, default="float16",
                        choices=["float16", "float32"],
                        help='npz 儲存精度')
    parser.add_argument('--similarity_cosine_step_plot', action='store_true',
                        help='Cosine 也輸出 step-change 折線圖')
    parser.add_argument('--similarity_sample_strategy', type=str, default="first",
                        choices=["first", "random", "uniform"],
                        help='樣本採樣策略: first=取前N個(最快), random=隨機選擇(增加多樣性), uniform=均勻分佈')
    parser.add_argument('--log_file','--lf', type=str, default=None,
                        help='指定 log 檔案路徑（預設: QATcode/cache_method/L1_L2_cosine/log/similarity_calculation.log）')
    parser.add_argument('--run_all_blocks', action='store_true',
                        help='自動運行所有 31 個 block 的實驗（類似 run_similarity_experiments.sh）')
    args = parser.parse_args()
    CONFIG.NUM_DIFFUSION_STEPS = args.num_steps
    CONFIG.EVAL_SAMPLES = args.eval_samples
    CONFIG.ENABLE_SIMILARITY_ANALYSIS = args.enable_similarity
    CONFIG.SEED = args.seed
    CONFIG.SIMILARITY_SAMPLES = args.similarity_samples
    CONFIG.SIMILARITY_COLLECT_SAMPLES = args.similarity_collect_samples
    CONFIG.SIMILARITY_TARGET_BLOCK = args.similarity_target_block
    CONFIG.SIMILARITY_OUTPUT_ROOT = args.similarity_output_root
    CONFIG.SIMILARITY_SAVE_DTYPE = args.similarity_dtype
    CONFIG.SIMILARITY_PLOT_COSINE_STEP = args.similarity_cosine_step_plot
    CONFIG.SIMILARITY_SAMPLE_STRATEGY = args.similarity_sample_strategy
    # 如果指定了 log_file，則使用指定的路徑；否則使用預設值
    if args.log_file is not None:
        CONFIG.LOG_FILE = args.log_file
        print(f"[設定] Log 檔案已設為: {CONFIG.LOG_FILE}", flush=True)
    
    def setup_environment(cls) -> None:
        """設置運行環境"""
        os.environ['CUDA_VISIBLE_DEVICES'] = cls.GPU_ID
        #torch.cuda.manual_seed(cls.SEED)
        
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
    LOGGER.info(f"Log file: {CONFIG.LOG_FILE}")

    # ------------------------------------------------------------
    # pred_xstart quantile analysis (independent)
    # ------------------------------------------------------------
    if args.run_pred_xstart_quantile_analysis:
        npz_baseline, npz_v2 = main_pred_xstart_quantile_analysis(
            images_mode=args.images_mode,
            chunk_batch=args.chunk_batch,
            hist_bins=args.hist_bins,
            clip_denoised=True,
            seed=args.seed,
            T=CONFIG.NUM_DIFFUSION_STEPS,
            pred_output_root=args.pred_output_root,
            skip_plot=args.skip_plot,
        )
        LOGGER.info(f"[pred_xstart_quantile] baseline npz: {npz_baseline}")
        LOGGER.info(f"[pred_xstart_quantile] v2 npz: {npz_v2}")
        raise SystemExit(0)
    
    # 定義所有 31 個 block 名稱（與 run_similarity_experiments.sh 一致）
    ALL_BLOCKS = [
        "model.input_blocks.0",
        "model.input_blocks.1",
        "model.input_blocks.2",
        "model.input_blocks.3",
        "model.input_blocks.4",
        "model.input_blocks.5",
        "model.input_blocks.6",
        "model.input_blocks.7",
        "model.input_blocks.8",
        "model.input_blocks.9",
        "model.input_blocks.10",
        "model.input_blocks.11",
        "model.input_blocks.12",
        "model.input_blocks.13",
        "model.input_blocks.14",
        "model.middle_block",
        "model.output_blocks.0",
        "model.output_blocks.1",
        "model.output_blocks.2",
        "model.output_blocks.3",
        "model.output_blocks.4",
        "model.output_blocks.5",
        "model.output_blocks.6",
        "model.output_blocks.7",
        "model.output_blocks.8",
        "model.output_blocks.9",
        "model.output_blocks.10",
        "model.output_blocks.11",
        "model.output_blocks.12",
        "model.output_blocks.13",
        "model.output_blocks.14",
    ]
    
    # 如果啟用 run_all_blocks，循環處理所有 block
    if args.run_all_blocks and args.enable_similarity:
        LOGGER.info("=" * 50)
        LOGGER.info("啟用自動運行所有 block 實驗")
        LOGGER.info(f"總共 {len(ALL_BLOCKS)} 個 block")
        LOGGER.info("=" * 50)
        
        # 設置 log 目錄
        log_dir = Path(CONFIG.SIMILARITY_OUTPUT_ROOT) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存原始參數
        original_log_file = CONFIG.LOG_FILE
        original_target_block = CONFIG.SIMILARITY_TARGET_BLOCK
        
        # 循環處理每個 block
        for idx, block in enumerate(ALL_BLOCKS, 1):
            safe_name = block.replace('.', '_')
            log_file = log_dir / f"similarity_{safe_name}.log"
            
            LOGGER.info("=" * 50)
            LOGGER.info(f"[{idx}/{len(ALL_BLOCKS)}] Running similarity for block: {block}")
            LOGGER.info(f"Log: {log_file}")
            LOGGER.info("=" * 50)
            
            # 更新配置
            CONFIG.SIMILARITY_TARGET_BLOCK = block
            CONFIG.LOG_FILE = str(log_file)
            
            # 重新配置 logging（為每個 block 使用獨立的 log 文件）
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s [%(levelname)s] %(message)s',
                handlers=[logging.StreamHandler(), logging.FileHandler(CONFIG.LOG_FILE)],
                force=True
            )
            
            try:
                if args.mode == 'float':
                    CONFIG.BEST_CKPT_PATH = "QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best.pth" if CONFIG.NUM_DIFFUSION_STEPS == 100 else "QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best_20steps.pth"
                    main_float_model()
                elif args.mode == 'int':
                    CONFIG.BEST_CKPT_PATH = "QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best.pth" if CONFIG.NUM_DIFFUSION_STEPS == 100 else "QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best_20steps.pth"
                    #main_int_model()
                    pass
                else:
                    LOGGER.error(f"Invalid mode: {args.mode}")
                    exit(1)
                
                LOGGER.info(f"✅ Block {block} 完成")
            except Exception as e:
                LOGGER.error(f"❌ Block {block} 失敗: {e}")
                import traceback
                LOGGER.error(traceback.format_exc())
                # 繼續處理下一個 block，不中斷整個流程
                continue
        
        # 恢復原始配置
        CONFIG.SIMILARITY_TARGET_BLOCK = original_target_block
        CONFIG.LOG_FILE = original_log_file
        
        LOGGER.info("=" * 50)
        LOGGER.info("✅ 所有 block 實驗完成")
        LOGGER.info("=" * 50)
    else:
        # 單個 block 模式（原有邏輯）
        if args.mode == 'float':
            CONFIG.BEST_CKPT_PATH = "QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best.pth" if CONFIG.NUM_DIFFUSION_STEPS == 100 else "QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best_20steps.pth"
            main_float_model()
        elif args.mode == 'int':
            CONFIG.BEST_CKPT_PATH = "QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best.pth" if CONFIG.NUM_DIFFUSION_STEPS == 100 else "QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best_20steps.pth"
            #main_int_model()
            pass
        else:
            LOGGER.error(f"Invalid mode: {args.mode}")
            exit(1)
