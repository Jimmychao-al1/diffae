"""
Diff-AE Original Model FID Cache Sensitivity Analysis
用於測試原本 Diff-AE（非量化版本）每個 UNet block 對 FID 的 cache 敏感度
"""

import os
import sys
import time
import logging
import shutil
import copy

import random
from typing import Tuple, List, Dict, Any, Optional, Callable

import numpy as np
import torch
import torch.nn as nn

# 添加專案路徑
sys.path.append(".")
sys.path.append("./model")

from diffusion.diffusion import _WrappedModel
from model.unet_autoenc import BeatGANsAutoencModel
from experiment import *
from templates import *
from templates_latent import *

from tqdm.auto import tqdm

# JSON 處理
import json
from datetime import datetime

#=============================================================================
# 配置與常量
#=============================================================================

class ExperimentConfig:
    """FID Cache Sensitivity 實驗配置（原本 Diff-AE）"""
    # 硬體設定
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    GPU_ID = '0'
    SEED = 0
    
    # 實驗參數
    NUM_DIFFUSION_STEPS = 20  # 預設 20 steps
    EVAL_SAMPLES = 5_000  # FID@5k
    BATCH_SIZE = 12
    
    # 文件路徑
    MODEL_PATH = "checkpoints/ffhq128_autoenc_latent/last.ckpt"
    
    # 輸出路徑
    OUTPUT_DIR = "QATcode/fid_cache_sensitivity_original"
    RESULTS_JSON = "fid_sensitivity_original_results.json"
    LOG_FILE = 'QATcode/fid_cache_sensitivity_original/fid_sensitivity_original.log'


# 初始化全局配置
CONFIG = ExperimentConfig()
LOGGER = logging.getLogger("FIDSensitivityOriginal")

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
# 模型載入函數
#=============================================================================

@time_operation
def load_diffae_model(model_path: str = CONFIG.MODEL_PATH) -> LitModel:
    """
    載入預訓練的原本 Diff-AE 模型
    
    Args:
        model_path: 檢查點路徑
        
    Returns:
        LitModel: 加載的擴散模型
    """
    LOGGER.info(f"載入原本 Diff-AE 模型: {model_path}")
    
    # 載入檢查點
    ckpt = torch.load(model_path, map_location='cpu', weights_only=False)
    conf = ffhq128_autoenc_latent()
    model = LitModel(conf)
    
    # 載入權重
    model.load_state_dict(ckpt['state_dict'], strict=False)
    LOGGER.info("✅ 原本 Diff-AE 模型載入完成")
    
    return model

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
                "model_type": "original_diffae"
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


def evaluate_fid_with_train(
    cache_scheduler: Optional[dict] = None,
    num_steps: int = None,
    gpus: List[int] = None
) -> float:
    """
    使用 train() 函數來執行 FID 評估（完全整合 run_ffhq128.py 的方式）
    
    使用 trainer.test() 來獲取結果，與 run_ffhq128.py 完全一致
    
    Args:
        cache_scheduler: cache 配置 (None 表示不使用 cache)
        num_steps: diffusion steps (如果為 None 則使用 CONFIG.NUM_DIFFUSION_STEPS)
        gpus: GPU 列表，預設為 [0]
    
    Returns:
        float: FID 分數（從 trainer.test() 的返回值中提取）
    """
    if num_steps is None:
        num_steps = CONFIG.NUM_DIFFUSION_STEPS
    
    if gpus is None:
        gpus = [0]
    
    # 創建配置（類似 run_ffhq128.py）
    conf = ffhq128_autoenc_latent()
    
    # 設定 cache scheduler
    if cache_scheduler is not None:
        conf.cache_scheduler = cache_scheduler
        LOGGER.info("✅ Cache scheduler 已設定")
    else:
        conf.cache_scheduler = None
        LOGGER.info("🚫 不使用 cache (baseline)")
    
    # 設定 eval_programs（類似 run_ffhq128.py:35）
    conf.eval_programs = [f'fid({num_steps},{num_steps})']
    
    # 使用固定的輸出目錄（所有實驗共用）
    output_dir = f'{conf.generate_dir}_temp_T{num_steps}'
    
    # 清空輸出目錄（刪除舊圖片）
    if os.path.exists(output_dir):
        LOGGER.info(f"🗑️  清空輸出目錄: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    LOGGER.info(f"開始生成 {CONFIG.EVAL_SAMPLES} 張圖片並計算 FID...")
    LOGGER.info(f"輸出目錄: {output_dir} (實驗結束後會清空)")
    LOGGER.info("使用 train() 函數執行評估（完全對齊 run_ffhq128.py）")
    
    # 複製 train() 函數的邏輯來創建 trainer 並執行 test（完全對齊 experiment.py）
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning import loggers as pl_loggers
    from torch.utils.data import DataLoader, TensorDataset
    
    # 設定全局 EVAL_SAMPLES（類似 train() 函數內部，line 940-941）
    import experiment
    experiment.EVAL_SAMPLES = CONFIG.EVAL_SAMPLES
    
    # 創建模型（類似 train() 函數內部，line 946）
    model = LitModel(conf)
    
    # 創建 checkpoint 和 trainer（類似 train() 函數內部，line 951-1004）
    if not os.path.exists(conf.logdir):
        os.makedirs(conf.logdir)
    
    checkpoint = ModelCheckpoint(
        dirpath=f'{conf.logdir}',
        save_last=True,
        save_top_k=1,
        every_n_train_steps=conf.save_every_samples // conf.batch_size_effective
    )
    checkpoint_path = f'{conf.logdir}/last.ckpt'
    
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=conf.logdir,
        name=None,
        version=''
    )
    
    # 設定 accelerator 和 plugins（類似 train() 函數內部，line 977-985）
    plugins = []
    if len(gpus) == 1:
        accelerator = None
    else:
        accelerator = 'ddp'
        from pytorch_lightning.plugins import DDPPlugin
        plugins.append(DDPPlugin(find_unused_parameters=False))
    
    trainer = pl.Trainer(
        max_steps=conf.total_samples // conf.batch_size_effective,
        resume_from_checkpoint=checkpoint_path if os.path.exists(checkpoint_path) else None,
        gpus=gpus,  # 使用 gpus 參數（對齊 experiment.py line 990）
        num_nodes=1,
        accelerator=accelerator,
        precision=16 if conf.fp16 else 32,
        callbacks=[checkpoint, LearningRateMonitor()],
        replace_sampler_ddp=True,
        logger=tb_logger,
        accumulate_grad_batches=conf.accum_batches,
        plugins=plugins,
    )
    
    # 載入 checkpoint（類似 train() 函數內部，mode='eval'，line 1027-1032）
    eval_path = getattr(conf, 'eval_path', None) or checkpoint_path
    if os.path.exists(eval_path):
        state = torch.load(eval_path, map_location='cpu', weights_only=False)
        model.load_state_dict(state['state_dict'])
        LOGGER.info(f"✅ 載入 checkpoint: {eval_path} (step: {state.get('global_step', 'unknown')})")
    else:
        LOGGER.warning(f"⚠️  Checkpoint 不存在: {eval_path}")
    
    # 創建 dummy dataloader（類似 train() 函數內部，line 1025-1026）
    dummy = DataLoader(
        TensorDataset(torch.tensor([0.] * conf.batch_size)),
        batch_size=conf.batch_size
    )
    
    # 執行 trainer.test()（類似 train() 函數內部，line 1063）
    out = trainer.test(model, dataloaders=dummy)
    
    # 提取結果（類似 train() 函數內部，line 1065-1066）
    out = out[0]  # first (and only) loader
    
    # 從結果字典中提取 FID 分數
    # 對於 fid(T,T) 格式，FID 會被記錄為 f'fid_ema_T{T}'（experiment.py line 851）
    fid_key = f'fid_ema_T{num_steps}'
    
    if fid_key in out:
        fid_score = float(out[fid_key])
        LOGGER.info(f"📊 FID Score (從 trainer.test()): {fid_score:.4f}")
    else:
        # 如果找不到，嘗試其他可能的 key
        possible_keys = [k for k in out.keys() if 'fid' in k.lower()]
        if possible_keys:
            fid_key = possible_keys[0]
            fid_score = float(out[fid_key])
            LOGGER.info(f"📊 FID Score (從 {fid_key}): {fid_score:.4f}")
        else:
            LOGGER.warning(f"⚠️  無法從 trainer.test() 結果中找到 FID 分數")
            LOGGER.warning(f"可用的 keys: {list(out.keys())}")
            fid_score = 0.0
    
    # FID 計算完成後清空圖片（節省空間）
    LOGGER.info(f"🗑️  清空臨時圖片: {output_dir}")
    shutil.rmtree(output_dir)
    
    return fid_score


@torch.no_grad()
def evaluate_fid_with_cache(
    base_model: LitModel,
    cache_scheduler: Optional[dict] = None,
    num_steps: int = None
) -> float:
    """
    生成圖片並計算 FID（原本的方式，保留作為備選）
    
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
    
    # 創建 sampler
    sampler = conf._make_diffusion_conf(T=num_steps).make_sampler()
    latent_sampler = conf._make_latent_diffusion_conf(T=num_steps).make_sampler()
    
    # 使用固定的輸出目錄（所有實驗共用）
    output_dir = f'{conf.generate_dir}_temp_original_T{num_steps}'
    
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
def main_original_model(
    target_layer: Optional[str] = None,
    k_value: Optional[int] = None,
    baseline_only: bool = False,
    use_train_mode: bool = False,
    gpus: List[int] = None
):
    """
    FID Cache Sensitivity 實驗主流程（原本 Diff-AE）
    
    Args:
        target_layer: 指定要測試的 layer (None = 測試所有 layer)
        k_value: cache frequency (3, 4, 或 5)
        baseline_only: 是否只跑 baseline (無 cache)
        use_train_mode: 是否使用 train() 函數方式（類似 run_ffhq128.py）
        gpus: GPU 列表（use_train_mode=True 時使用）
    
    流程:
    1. 如果 use_train_mode=True，使用 train() 函數方式（不需要載入模型）
    2. 否則，載入原本的 Diff-AE 模型並使用 evaluate_fid_with_cache
    3. 如果 baseline_only，只跑 baseline FID
    4. 否則，對指定 layer (或所有 layer) 進行 cache sensitivity 測試
    """
    LOGGER.info("=" * 50)
    LOGGER.info("FID Cache Sensitivity Analysis (Original Diff-AE)")
    LOGGER.info("=" * 50)
    
    LOGGER.info(f"使用設備: {CONFIG.DEVICE}")
    LOGGER.info(f"Diffusion steps: {CONFIG.NUM_DIFFUSION_STEPS}")
    LOGGER.info(f"FID evaluation samples: {CONFIG.EVAL_SAMPLES}")
    LOGGER.info(f"使用 train() 模式: {use_train_mode}")
    
    try:
        # 載入或初始化結果字典
        step_config = f"T{CONFIG.NUM_DIFFUSION_STEPS}"
        results_path = os.path.join(CONFIG.OUTPUT_DIR, CONFIG.RESULTS_JSON)
        results = load_results(results_path)
        
        # 確保結果字典有正確的結構
        if step_config not in results["results"]:
            results["results"][step_config] = {}
        
        # 如果使用 train() 模式，不需要載入模型
        if use_train_mode:
            if gpus is None:
                gpus = [0]
            
            # 4. 如果是 baseline only，只跑 baseline
            if baseline_only:
                LOGGER.info("=" * 50)
                LOGGER.info("執行 Baseline FID (無 cache) - 使用 train() 模式")
                LOGGER.info("=" * 50)
                
                if "baseline_fid" not in results["results"][step_config]:
                    baseline_fid = evaluate_fid_with_train(
                        cache_scheduler=None,
                        num_steps=CONFIG.NUM_DIFFUSION_STEPS,
                        gpus=gpus
                    )
                    results["results"][step_config]["baseline_fid"] = baseline_fid
                    save_results(results, results_path)
                else:
                    baseline_fid = results["results"][step_config]["baseline_fid"]
                    LOGGER.info(f"📊 Baseline FID (已存在): {baseline_fid:.4f}")
                
                return
            
            # 5. 執行 cache sensitivity 實驗
            if k_value is None:
                LOGGER.error("❌ 必須指定 k_value (3, 4, 或 5)")
                return
            
            # 確保 baseline 已經跑過
            if "baseline_fid" not in results["results"][step_config]:
                LOGGER.info("⚠️ Baseline FID 尚未計算，先執行 baseline...")
                baseline_fid = evaluate_fid_with_train(
                    cache_scheduler=None,
                    num_steps=CONFIG.NUM_DIFFUSION_STEPS,
                    gpus=gpus
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
            
            # 6. 對每個 layer 進行測試
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
                
                # 計算 FID（使用 train() 模式）
                fid = evaluate_fid_with_train(
                    cache_scheduler=cache_config,
                    num_steps=CONFIG.NUM_DIFFUSION_STEPS,
                    gpus=gpus
                )
                
                delta = fid - baseline_fid
                
                # 保存結果
                results["results"][step_config][k_key][layer] = {
                    "fid": fid,
                    "delta": delta
                }
                save_results(results, results_path)
                
                LOGGER.info(f"✅ {layer}: FID={fid:.4f}, Δ={delta:+.4f}")
        
        else:
            # 原本的方式：載入模型並使用 evaluate_fid_with_cache
            # 1. 載入原本的 Diff-AE 模型
            base_model: LitModel = load_diffae_model()
            LOGGER.info("✅ 原本 Diff-AE 模型載入成功")
            
            # 2. 設定模型（不需要量化步驟）
            base_model.to(CONFIG.DEVICE)
            base_model.eval()
            base_model.setup()
            LOGGER.info("✅ 模型設定完成")
            
            # 4. 如果是 baseline only，只跑 baseline
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
            
            # 5. 執行 cache sensitivity 實驗
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
            
            # 6. 對每個 layer 進行測試
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


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='FID Cache Sensitivity Analysis for Original Diff-AE'
    )
    
    # 基本參數
    parser.add_argument('--num_steps', '--n', type=int, default=20,
                        help='Diffusion steps (20 or 100)')
    parser.add_argument('--eval_samples', '--es', type=int, default=5000,
                        help='Number of samples for FID evaluation')
    
    # 實驗控制
    parser.add_argument('--baseline', action='store_true',
                        help='只跑 baseline FID (無 cache)')
    parser.add_argument('--layer', type=str, default=None,
                        help='指定要測試的 layer (例如: encoder_layer_5)。不指定則測試所有 layer')
    parser.add_argument('--k', type=int, default=None, choices=[3, 4, 5],
                        help='Cache frequency (3, 4, or 5). baseline 模式下不需要')
    parser.add_argument('--use_train', action='store_true',
                        help='使用 train() 函數方式（類似 run_ffhq128.py），不需要預先載入模型')
    parser.add_argument('--gpus', type=str, default='0',
                        help='GPU 列表（逗號分隔，例如: 0,1），僅在 --use_train 模式下使用')
    
    # 輸出設定
    parser.add_argument('--output_json', type=str, default=None,
                        help='結果 JSON 檔案名稱 (預設: fid_sensitivity_original_results.json)')
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
    LOGGER.info("FID Cache Sensitivity Analysis (Original Diff-AE)")
    LOGGER.info("=" * 50)
    LOGGER.info(f"Diffusion steps: {CONFIG.NUM_DIFFUSION_STEPS}")
    LOGGER.info(f"FID evaluation samples: {CONFIG.EVAL_SAMPLES}")
    LOGGER.info(f"Model checkpoint: {CONFIG.MODEL_PATH}")
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
    
    # 解析 GPU 列表
    gpus = [int(x.strip()) for x in args.gpus.split(',')]
    
    # 執行實驗
    main_original_model(
        target_layer=args.layer,
        k_value=args.k,
        baseline_only=args.baseline,
        use_train_mode=args.use_train,
        gpus=gpus
    )
