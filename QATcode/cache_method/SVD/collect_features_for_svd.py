"""
SVD Feature 收集腳本

功能：
- 對單一 block 在 T=100 個 timestep 收集 feature
- 每個 timestep 累積 target_N 個樣本的 (N, C, H, W) tensor
- 輸出：
  - 預設模式：svd_features/<block_slug>/t_{0..99}.pt + meta.json
  - in-memory pipeline：不寫 t_{t}.pt，直接串接 svd_metrics / correlate_svd_similarity

參考：similarity_calculation.py 的 hook 架構與 evaluate_fid 呼叫方式
"""

import os
import sys
import json
import copy
import time
import logging
from pathlib import Path
from typing import Dict, Optional, List

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange

# 添加專案根目錄到 Python 路徑
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from config import *
from templates_latent import *
from experiment import LitModel
from metrics import evaluate_fid
from model.blocks import TimestepEmbedSequential

# 複用 similarity_calculation.py 的工具函數
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../L1_L2_cosine')))
from QATcode.cache_method.L1_L2_cosine.similarity_calculation import (
    load_diffae_model,
    create_float_quantized_model,
    load_calibration_data,
    SimpleDequantizer,
)
from QATcode.quantize_ver2.quant_model_lora_v2 import QuantModule_DiffAE_LoRA

# ==================== 設定 ====================
class Config:
    """全局配置（與 similarity_calculation.py 一致，否則 load_state_dict 會 size mismatch）"""
    # 模型路徑（先載入 last.ckpt，再 overlay .pth）
    MODEL_PATH = 'checkpoints/ffhq128_autoenc_latent/last.ckpt'
    BEST_CKPT_PATH = 'QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best.pth'
    
    # 基本設定
    DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    GPU_ID = '0'
    SEED = 0
    
    # 量化設定（LORA_RANK / MODE / NUM_DIFFUSION_STEPS 須與 .pth 訓練時一致）
    MODE = 'train'
    LORA_RANK = 32
    
    # SVD Feature 收集設定
    SVD_TARGET_BLOCK = None  # 從 CLI 指定
    SVD_TARGET_N = 32  # 每個 timestep 收集的樣本數
    SVD_OUTPUT_ROOT = 'QATcode/cache_method/SVD'
    NUM_DIFFUSION_STEPS = 100  # T=100
    
    # FID 設定（用於驅動生成流程）
    EVAL_SAMPLES = 50000  # evaluate_fid 用，實際由 eval_num_images 控制
    
    # Log 設定
    LOG_FILE = None

    # 低磁碟模式：A 收集後直接在記憶體跑 B/C，不寫 t_{t}.pt
    IN_MEMORY_PIPELINE = False
    SVD_REPRESENTATIVE_T = -1
    SVD_ENERGY_THRESHOLD = 0.98
    SVD_COMPUTE_ENERGY = True
    SIMILARITY_NPZ = None
    SKIP_CORRELATION = False

CONFIG = Config()

# ==================== Logger ====================
def setup_logger(log_file: Optional[str] = None) -> logging.Logger:
    """設定 logger"""
    logger = logging.getLogger('SVD_Feature_Collector')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] [%(name)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file).resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), mode='a', encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

LOGGER = setup_logger()

# ==================== SVD Feature Collector ====================
class SvdFeatureCollector:
    """
    SVD Feature 收集器
    
    對單一 block 在每個 timestep 收集 feature，最終寫出 (target_N, C, H, W) 的 tensor
    """
    
    def __init__(
        self,
        save_root: str,
        max_timesteps: int,
        target_N: int,
        target_block: str,
        device: torch.device
    ):
        self.save_root = Path(save_root)
        self.max_timesteps = int(max_timesteps)
        self.target_N = int(target_N)
        self.target_block = target_block
        self.device = device
        
        # Block slug（用於目錄名）
        self.block_slug = target_block.replace(".", "_")
        
        # Hook 列表
        self.hooks = []
        
        # Step 追蹤
        self._step_counter = -1
        self.current_step_idx = None
        
        # 每個 timestep 的 feature buffer：{t: [tensor1, tensor2, ...]}
        # 每個 tensor 為 (B, C, H, W)，會依序 concat
        self.feature_buffers: Dict[int, List[torch.Tensor]] = {t: [] for t in range(max_timesteps)}
        self.feature_counts: Dict[int, int] = {t: 0 for t in range(max_timesteps)}
        
        # 記錄 shape（從第一個 tensor 取得）
        self.C = None
        self.H = None
        self.W = None
        
        LOGGER.info(f"[SVD] 初始化 SvdFeatureCollector: block={target_block}, target_N={target_N}, T={max_timesteps}")
    
    def register_hooks(self, model: nn.Module, sampler):
        """註冊 hook"""
        # 對 target_block 註冊 forward hook
        registered = False
        for name, module in model.named_modules():
            if not isinstance(module, TimestepEmbedSequential):
                continue
            if "encoder" in name:
                continue
            # 對應 block 名稱（可能需要去掉或加上 "model." 前綴）
            if name == self.target_block or f"model.{name}" == self.target_block or name == self.target_block.replace("model.", ""):
                self.hooks.append(module.register_forward_hook(self._create_block_hook(name)))
                LOGGER.info(f"[SVD] 註冊 block hook: {name}")
                registered = True
                break
        
        if not registered:
            LOGGER.warning(f"[SVD] 未找到 block: {self.target_block}，請檢查名稱")
        
        # 註冊 model pre-hook（更新 step index）
        self.hooks.append(
            model.register_forward_pre_hook(self._create_step_pre_hook(), with_kwargs=True)
        )
        LOGGER.info("[SVD] 註冊 model step hook 完成")
    
    def remove_hooks(self):
        """移除 hook"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        LOGGER.info("[SVD] Hook 已移除")
    
    def _create_step_pre_hook(self):
        """創建 step pre-hook"""
        def pre_hook(module, args, kwargs):
            self._step_counter = (self._step_counter + 1) % self.max_timesteps
            self.current_step_idx = self.max_timesteps - 1 - self._step_counter
        return pre_hook
    
    def _create_block_hook(self, block_name: str):
        """創建 block forward hook"""
        def hook_fn(module, input, output):
            if self._step_counter is None or self._step_counter < 0:
                return
            
            step_idx = int(self._step_counter)
            if not (0 <= step_idx < self.max_timesteps):
                return

            # 每個 timestep 只保留前 target_N 個樣本，避免記憶體暴增
            if self.feature_counts[step_idx] >= self.target_N:
                return
            
            # Detach 並立即移到 CPU（避免 GPU OOM）
            out = output.detach().cpu()

            remain = self.target_N - self.feature_counts[step_idx]
            if out.shape[0] > remain:
                out = out[:remain]
            
            # 記錄 shape（第一次）
            if self.C is None:
                _, self.C, self.H, self.W = out.shape
                LOGGER.info(f"[SVD] Feature shape: C={self.C}, H={self.H}, W={self.W}")
            
            # Append 到 buffer（CPU）
            self.feature_buffers[step_idx].append(out)
            self.feature_counts[step_idx] += int(out.shape[0])
        
        return hook_fn
    
    def finalize(self):
        """
        完成收集並落地寫出 t_{t}.pt 與 meta.json（僅非 in-memory 流程使用）
        
        與 similarity 的 finalize 完全不同：
        - 不計算 L1/L2/cosine
        - 只做 concat + 截斷到 target_N + 寫 .pt + 寫 meta.json
        """
        LOGGER.info(f"[SVD] 開始 finalize，目標目錄: {self.save_root / self.block_slug}")

        features, meta = self.build_feature_tensors()

        # 創建輸出目錄
        output_dir = self.save_root / self.block_slug
        output_dir.mkdir(parents=True, exist_ok=True)

        for t, tensor_t in enumerate(tqdm(features, desc="Writing features")):
            torch.save(tensor_t, output_dir / f"t_{t}.pt")

        with open(output_dir / "meta.json", 'w') as f:
            json.dump(meta, f, indent=2)

        LOGGER.info(
            f"[SVD] Finalize 完成：N={meta['N']}, T={meta['T']}, C={meta['C']}, H={meta['H']}, W={meta['W']}"
        )
        LOGGER.info(f"[SVD] 輸出目錄: {output_dir}")

    def build_feature_tensors(self):
        """
        將 buffer 整理成可直接計算 SVD 的 features 與 meta（不落地）
        """
        features: List[torch.Tensor] = []
        actual_N = None

        for t in tqdm(range(self.max_timesteps), desc="Preparing features"):
            if len(self.feature_buffers[t]) == 0:
                raise RuntimeError(f"[SVD] t={t} 無資料，無法進行後續 SVD 計算")

            tensor_t = torch.cat(self.feature_buffers[t], dim=0)  # (total_samples, C, H, W)

            if tensor_t.shape[0] < self.target_N:
                LOGGER.warning(
                    f"[SVD] t={t} 只有 {tensor_t.shape[0]} 個樣本，少於 target_N={self.target_N}"
                )
                actual_N_t = tensor_t.shape[0]
            else:
                tensor_t = tensor_t[:self.target_N]
                actual_N_t = self.target_N

            if actual_N is None:
                actual_N = actual_N_t

            features.append(tensor_t)

        meta = {
            "block": self.block_slug,
            "target_block_name": self.target_block,
            "N": actual_N,
            "T": self.max_timesteps,
            "C": self.C,
            "H": self.H,
            "W": self.W
        }
        return features, meta

# ==================== 主流程 ====================
def main():
    """主流程：複用 similarity_calculation.py 的架構"""
    
    LOGGER.info("=" * 80)
    LOGGER.info("SVD Feature 收集開始")
    LOGGER.info("=" * 80)
    
    try:
        # 1. 載入基礎模型
        base_model: LitModel = load_diffae_model(CONFIG.MODEL_PATH)
        LOGGER.info("✅ Diff-AE 模型載入成功")
        
        diffusion_model = base_model.ema_model
        
        # 2. 創建量化模型
        quant_model = create_float_quantized_model(
            diffusion_model,
            num_steps=CONFIG.NUM_DIFFUSION_STEPS,
            lora_rank=CONFIG.LORA_RANK,
            mode=CONFIG.MODE
        )
        quant_model.to(CONFIG.DEVICE)
        quant_model.eval()
        
        # 動態掛子模組
        for name, module in quant_model.named_modules():
            if isinstance(module, QuantModule_DiffAE_LoRA) and module.ignore_reconstruction is False:
                module.intn_dequantizer = SimpleDequantizer(uaq=module.weight_quantizer, weight=module.weight).to(CONFIG.DEVICE)
        
        #for name, module in quant_model.named_modules():
        #    if isinstance(module, QuantModule_DiffAE_LoRA) and module.ignore_reconstruction is False:
        #        module.intn_dequantizer.delta.data.copy_(module.weight_quantizer.delta.to(CONFIG.DEVICE))
        #        module.intn_dequantizer.zero_point.data.copy_(module.weight_quantizer.zero_point.to(CONFIG.DEVICE))
        
        # 載入校準資料
        cali_images, cali_t, cali_y = load_calibration_data()
        
        # 設定量化
        quant_model.set_first_last_layer_to_8bit()
        device = CONFIG.DEVICE
        quant_model.set_quant_state(True, True)
        
        #for name, module in quant_model.named_modules():
        #    if isinstance(module, QuantModule_DiffAE_LoRA) and module.ignore_reconstruction is False:
        #        module.intn_dequantizer = SimpleDequantizer(uaq=module.weight_quantizer, weight=module.weight)
        
        
        # First run to init
        with torch.no_grad():
            _ = quant_model(x=cali_images[:4].to(device), t=cali_t[:4].to(device), cond=cali_y[:4].to(device))
        
        # 載入 checkpoint
        ckpt = torch.load(CONFIG.BEST_CKPT_PATH, map_location='cpu', weights_only=False)
        from QATcode.cache_method.L1_L2_cosine.similarity_calculation import _load_quant_and_ema_from_ckpt
        _load_quant_and_ema_from_ckpt(base_model, quant_model, ckpt)
        
        LOGGER.info("✅ 量化模型載入成功")
        
        # 3. 準備基礎設定
        base_model.to(CONFIG.DEVICE)
        base_model.eval()
        base_model.setup()
        LOGGER.info("✅ 基礎模型設定完成")
        
        # 4. 創建 sampler
        T = CONFIG.NUM_DIFFUSION_STEPS
        T_latent = CONFIG.NUM_DIFFUSION_STEPS
        base_model.train_dataloader()
        sampler = base_model.conf._make_diffusion_conf(T=T).make_sampler()
        latent_sampler = base_model.conf._make_latent_diffusion_conf(T=T_latent).make_sampler()
        conf = base_model.conf.clone()
        
        # 5. 創建 SvdFeatureCollector
        LOGGER.info("=" * 50)
        LOGGER.info(f"啟用 SVD Feature 收集")
        LOGGER.info(f"目標 Block: {CONFIG.SVD_TARGET_BLOCK}")
        LOGGER.info(f"Target N: {CONFIG.SVD_TARGET_N}")
        LOGGER.info(f"擴散步數: {T}")
        LOGGER.info("=" * 50)
        
        svd_root = f"{CONFIG.SVD_OUTPUT_ROOT}/svd_features"
        svd_collector = SvdFeatureCollector(
            save_root=svd_root,
            max_timesteps=T,
            target_N=CONFIG.SVD_TARGET_N,
            target_block=CONFIG.SVD_TARGET_BLOCK,
            device=CONFIG.DEVICE
        )
        
        # 6. 註冊 hook
        svd_collector.register_hooks(base_model.ema_model, sampler)
        
        # 7. 設定 eval_num_images（確保每個 t 都能累積到 target_N）
        # batch_size_eval = 32，需要 ceil(target_N / 32) 個 batch
        import math
        num_batches_needed = math.ceil(CONFIG.SVD_TARGET_N / 32)
        conf.eval_num_images = num_batches_needed * 32
        LOGGER.info(f"[SVD] eval_num_images 設為 {conf.eval_num_images}（{num_batches_needed} batches）")
        
        # 8. 呼叫 evaluate_fid（驅動生成流程）
        LOGGER.info("[SVD] 開始生成與收集...")
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
            output_dir=f'{conf.generate_dir}_SVD_T{T}_feature'
        )
        LOGGER.info(f'[SVD] FID@{conf.eval_num_images} {T} steps score: {score}')
        
        # 9. 移除 hook
        svd_collector.remove_hooks()

        if CONFIG.IN_MEMORY_PIPELINE:
            LOGGER.info("[SVD] 啟用 in-memory pipeline：略過 t_{t}.pt 寫檔，直接計算 B/C")
            meta = {
                "block": svd_collector.block_slug,
                "target_block_name": svd_collector.target_block,
                "T": T,
                "C": svd_collector.C,
                "H": svd_collector.H,
                "W": svd_collector.W
            }

            from QATcode.cache_method.SVD.svd_metrics import process_feature_buffers_in_memory
            svd_metrics_dir = Path(CONFIG.SVD_OUTPUT_ROOT) / "svd_metrics"
            svd_result = process_feature_buffers_in_memory(
                feature_buffers=svd_collector.feature_buffers,
                meta=meta,
                target_N=CONFIG.SVD_TARGET_N,
                output_dir=svd_metrics_dir,
                representative_t=CONFIG.SVD_REPRESENTATIVE_T,
                energy_threshold=CONFIG.SVD_ENERGY_THRESHOLD,
                compute_energy=CONFIG.SVD_COMPUTE_ENERGY
            )
            LOGGER.info("[SVD] 階段 B 完成（in-memory）")

            if not CONFIG.SKIP_CORRELATION:
                if CONFIG.SIMILARITY_NPZ is None:
                    raise ValueError("in-memory pipeline 需要提供 --similarity_npz 才能執行階段 C")
                from QATcode.cache_method.SVD.correlate_svd_similarity import process_single_correlation
                svd_json_path = svd_metrics_dir / f"{svd_result['block']}.json"
                corr_output_dir = Path(CONFIG.SVD_OUTPUT_ROOT) / "correlation"
                process_single_correlation(
                    svd_json_path=svd_json_path,
                    similarity_npz_path=Path(CONFIG.SIMILARITY_NPZ),
                    output_dir=corr_output_dir,
                    plot_figures=True
                )
                LOGGER.info("[SVD] 階段 C 完成（in-memory）")
        else:
            # 10. Finalize（寫檔）
            svd_collector.finalize()
            LOGGER.info("✅ SVD Feature 收集完成")
        
        torch.cuda.empty_cache()
    
    except Exception as e:
        LOGGER.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        raise e

# ==================== CLI ====================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SVD Feature Collection")
    parser.add_argument('--num_steps', '--n', type=int, default=100, help='擴散步數 T')
    parser.add_argument('--svd_target_block', type=str, required=True, help='目標 block 名稱（如 model.input_blocks.0）')
    parser.add_argument('--svd_target_N', type=int, default=32, help='每個 timestep 收集的樣本數')
    parser.add_argument('--svd_output_root', type=str, default='QATcode/cache_method/SVD', help='輸出根目錄')
    parser.add_argument('--log_file', '--lf', type=str, default=None, help='Log 檔案路徑')
    parser.add_argument('--in_memory_pipeline', action='store_true',
                        help='A 收集後直接在記憶體執行 B/C，不寫 t_{t}.pt')
    parser.add_argument('--representative-t', type=int, default=-1,
                        help='B 階段 rank 估計代表 timestep，-1 表示最後一步')
    parser.add_argument('--energy-threshold', type=float, default=0.98,
                        help='B 階段 cumulative energy 門檻')
    parser.add_argument('--no-compute-energy', action='store_true',
                        help='B 階段不計算 energy ratio')
    parser.add_argument('--similarity_npz', type=str, default=None,
                        help='C 階段相依的 similarity npz 路徑（in-memory pipeline 需要）')
    parser.add_argument('--skip_correlation', action='store_true',
                        help='in-memory pipeline 只跑 A->B，略過 C')
    
    args = parser.parse_args()
    
    # 更新 CONFIG
    CONFIG.NUM_DIFFUSION_STEPS = args.num_steps
    CONFIG.SVD_TARGET_BLOCK = args.svd_target_block
    CONFIG.SVD_TARGET_N = args.svd_target_N
    CONFIG.SVD_OUTPUT_ROOT = args.svd_output_root
    CONFIG.LOG_FILE = args.log_file
    CONFIG.IN_MEMORY_PIPELINE = args.in_memory_pipeline
    CONFIG.SVD_REPRESENTATIVE_T = args.representative_t
    CONFIG.SVD_ENERGY_THRESHOLD = args.energy_threshold
    CONFIG.SVD_COMPUTE_ENERGY = not args.no_compute_energy
    CONFIG.SIMILARITY_NPZ = args.similarity_npz
    CONFIG.SKIP_CORRELATION = args.skip_correlation
    
    # 重設 logger（加上 log_file）
    if CONFIG.LOG_FILE:
        LOGGER = setup_logger(CONFIG.LOG_FILE)
        LOGGER.info(f"[設定] Log 檔案: {CONFIG.LOG_FILE}")
    
    # 設定環境
    os.environ['CUDA_VISIBLE_DEVICES'] = CONFIG.GPU_ID
    
    # 執行主流程
    main()
