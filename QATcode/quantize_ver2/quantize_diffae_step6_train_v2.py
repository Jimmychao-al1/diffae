"""
Diff-AE EfficientDM Step 6: 量化感知微調 (QAT)
實現 LoRA 微調 + ver2 normalized fake-quant（per-step activation scale）
"""

from copy import deepcopy
import os
import sys
import time
import logging
import json
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
from QATcode.quantize_ver2.quant_model_lora_v2 import QuantModule_DiffAE_LoRA
from QATcode.quantize_ver2.quant_layer_v2 import QuantModule, normalized_fake_quant
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
    BATCH_SIZE = 6     # 適中的 batch size
    NUM_EPOCHS = 160   # 按原作 EfficientDM 設定
    LEARNING_RATE = 1e-4
    LORA_RANK = 32
    NUM_DIFFUSION_STEPS = 100  # 對齊原作的 ddim_steps=100
    CLIP_GRAD_NORM = 1.0
    
    # 量化參數
    N_BITS_W = 8  # 權重量化位元數
    N_BITS_A = 8  # 激活量化位元數
    
    # 文件路徑
    MODEL_PATH = "checkpoints/ffhq128_autoenc_latent/last.ckpt"
    QUANT_CKPT_PATH = "QATcode/quantize_ver2/diffae_unet_quantw8a8_intmodel.pth"
    CALIB_DATA_PATH = "QATcode/quantize_ver2/calibration_diffae.pth"
    SAVE_BEST_PATH = "QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best.pth"
    SAVE_FINAL_PATH = "QATcode/quantize_ver2/checkpoints/diffae_step6_lora_final.pth"

    TEST_PATH = "QATcode/quantize_ver2/test_ckpt/test.pth"
    
    # 其他設定 - 調整為 EfficientDM 風格
    LOG_INTERVAL = 5   # 每隔幾個批次記錄一次
    CALIB_SAMPLES = 1024
    TRAIN_BATCHES_PER_EPOCH = 2  # 增加每個 epoch 的批次數，因為不再依賴真實數據
    CURRENT_LAYER_IDX = 4
    
    @classmethod
    def setup_environment(cls, log_name: str = "step6_train.log") -> str:
        """設置運行環境並初始化 logger。"""
        os.environ['CUDA_VISIBLE_DEVICES'] = cls.GPU_ID
        torch.cuda.manual_seed(cls.SEED)
        log_dir = os.path.join("QATcode", "quantize_ver2", "log")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_name)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[logging.StreamHandler(), logging.FileHandler(log_path, mode='w')],
            force=True,
        )
        return log_path

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
    parser.add_argument('--quant-diag-interval', type=int, default=8,
                        help='每隔多少個 epoch 記錄一次「每層有效量化更新比例」(<=0 表示關閉)')
    parser.add_argument('--quant-diag-topk', type=int, default=12,
                        help='每次診斷時額外列出前 Top-K 層（依有效量化更新比例排序）')
    parser.add_argument('--quant-diag-root', type=str, default='QATcode/quantize_ver2/quant_diag_runs',
                        help='量化更新診斷輸出根目錄（每次 run 自動建立新子資料夾）')
    parser.add_argument('--quant-diag-save-full-layers', action='store_true', default=False,
                        help='是否在 jsonl 中保存每次診斷的完整 layers 資訊（檔案會較大）')
    parser.add_argument('--log-suffix', type=str, default='',
                        help='log 檔名後綴，例如 1 -> step6_train_1.log')
    parser.add_argument('--auto-log', action='store_true', default=False,
                        help='自動使用 timestamp 產生唯一 log 檔名')
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='啟用 TensorBoard 記錄（需安裝 torch.utils.tensorboard）')
    parser.add_argument('--tb-dir', type=str, default='QATcode/quantize_ver2/tb_runs',
                        help='TensorBoard SummaryWriter 輸出根目錄')
    parser.add_argument('--loss-chunk-size', type=int, default=20,
                        help='training-step loss chunk 大小（以 optimizer/global step 為單位）')
    parser.add_argument('--teacher-autocast-match', action='store_true', default=False,
                        help='讓 teacher 分支也在與 quant 分支相同 autocast 條件下 forward（A/B 驗證用）')
    parser.add_argument('--debug-scale-list-update', action='store_true', default=False,
                        help='輸出 scale_list 參數 grad 與 step 前後變化（最小診斷，預設關閉）')
    parser.add_argument('--debug-scale-list-interval', type=int, default=100,
                        help='scale_list 診斷記錄間隔（以 optimizer/global step 計）')
    parser.add_argument('--debug-timestep-grad-conflict', action='store_true', default=False,
                        help='啟用 timestep gradient conflict 診斷（最小侵入，預設關閉）')
    parser.add_argument('--debug-timestep-grad-steps', type=str, default='0,80,99',
                        help='要診斷的 timestep，逗號分隔，例如 0,80,99')
    parser.add_argument('--debug-timestep-grad-interval', type=int, default=0,
                        help='每隔多少個 epoch 觸發一次 grad conflict 診斷（<=0 關閉）')
    # Step-2：rollout 結尾 tail-repair（預設關閉）
    parser.add_argument('--tail-repair-enable', action='store_true', default=False,
                        help='在每次 DDIM rollout 結束後做額外小步更新（實驗用，預設關閉）')
    parser.add_argument('--tail-repair-steps', type=int, default=1, choices=[1, 2],
                        help='tail-repair 外層重複次數（1 或 2）')
    parser.add_argument('--tail-repair-t-range', type=str, default='0',
                        help='要補強的 timestep，逗號分隔，例如 0 或 0,1,2,3,4')
    parser.add_argument('--tail-repair-lr-scale', type=float, default=0.25,
                        help='tail-repair 區塊內暫時將 optimizer 各 param_group 的 lr 乘上此係數（結束後還原）')

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
            quant_diag_interval = 8
            quant_diag_topk = 12
            quant_diag_root = 'QATcode/quantize_ver2/quant_diag_runs'
            quant_diag_save_full_layers = False
            log_suffix = ''
            auto_log = False
            tensorboard = False
            tb_dir = 'QATcode/quantize_ver2/tb_runs'
            loss_chunk_size = 20
            teacher_autocast_match = False
            debug_scale_list_update = False
            debug_scale_list_interval = 100
            debug_timestep_grad_conflict = False
            debug_timestep_grad_steps = '0,80,99'
            debug_timestep_grad_interval = 0
            tail_repair_enable = False
            tail_repair_steps = 1
            tail_repair_t_range = '0'
            tail_repair_lr_scale = 0.25

        args = _Args()
    return args


def _quantize_to_int_codes(x: torch.Tensor, scale: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    將權重張量量化到 int code（對齊 ver2 normalized fake-quant 的離散格點）。
    """
    s = scale.clamp(min=eps)
    x_scaled = torch.clamp(x / s, -1.0, 1.0)
    return torch.round(x_scaled * 127.0).clamp(-128.0, 127.0)


def _compute_effective_weight(module: QuantModule_DiffAE_LoRA) -> Optional[torch.Tensor]:
    """
    依照 QuantModule_DiffAE_LoRA.forward 的權重合成邏輯，計算當前 effective weight。
    """
    if not hasattr(module, 'org_weight') or (module.org_weight is None):
        return None

    org_weight = module.org_weight.detach()
    device = org_weight.device
    dtype = org_weight.dtype

    if not (hasattr(module, 'loraA') and hasattr(module, 'loraB')):
        return org_weight

    if module.fwd_func is F.linear:
        in_features = org_weight.shape[1]
        E = torch.eye(in_features, device=device, dtype=dtype)
        lora_weight = module.loraB(module.loraA(module.lora_dropout_layer(E))).T
        return org_weight + lora_weight.to(device=device, dtype=dtype)

    if module.fwd_func is F.conv2d:
        lora_weight = module.loraB.weight.squeeze(-1).squeeze(-1) @ module.loraA.weight.permute(2, 3, 0, 1)
        lora_weight = lora_weight.permute(2, 3, 0, 1)
        return org_weight + lora_weight.to(device=device, dtype=dtype)

    return org_weight


def collect_quant_update_diagnostics(
    qnn: nn.Module,
    topk: int = 12,
) -> Dict[str, Any]:
    """
    計算每層「有效量化更新比例」：
    - changed_ratio_dynamic: 使用各自動態 a_w（最接近實際 forward）
    - changed_ratio_same_scale: 固定使用 base a_w（觀察 LoRA 是否跨越既有量化台階）
    """
    rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        for name, module in qnn.named_modules():
            if not isinstance(module, QuantModule_DiffAE_LoRA):
                continue

            if not hasattr(module, '_compute_a_w'):
                continue

            base_w = module.org_weight.detach()
            eff_w = _compute_effective_weight(module)
            if eff_w is None:
                continue
            eff_w = eff_w.detach().to(base_w.device, dtype=base_w.dtype)

            if eff_w.shape != base_w.shape:
                LOGGER.warning(f"[QuantUpdateDiag] skip layer={name}, shape mismatch {tuple(base_w.shape)} vs {tuple(eff_w.shape)}")
                continue

            a_w_base = module._compute_a_w(base_w)
            a_w_eff = module._compute_a_w(eff_w)

            q_base_int = _quantize_to_int_codes(base_w, a_w_base)
            q_eff_int_dynamic = _quantize_to_int_codes(eff_w, a_w_eff)
            q_eff_int_same_scale = _quantize_to_int_codes(eff_w, a_w_base)

            changed_ratio_dynamic = (q_eff_int_dynamic != q_base_int).float().mean().item()
            changed_ratio_same_scale = (q_eff_int_same_scale != q_base_int).float().mean().item()

            q_base_norm = normalized_fake_quant(base_w, a_w_base, eps=1e-8)
            q_eff_norm_dynamic = normalized_fake_quant(eff_w, a_w_eff, eps=1e-8)

            rows.append({
                "name": name,
                "numel": int(base_w.numel()),
                "changed_ratio_dynamic": float(changed_ratio_dynamic),
                "changed_ratio_same_scale": float(changed_ratio_same_scale),
                "mean_abs_delta_fp": float((eff_w - base_w).abs().mean().item()),
                "mean_abs_delta_qnorm": float((q_eff_norm_dynamic - q_base_norm).abs().mean().item()),
            })

    if not rows:
        return {"summary": None, "layers": [], "top_layers": []}

    total_numel = max(1, sum(r["numel"] for r in rows))
    weighted_changed_ratio_dynamic = sum(r["changed_ratio_dynamic"] * r["numel"] for r in rows) / total_numel
    weighted_changed_ratio_same_scale = sum(r["changed_ratio_same_scale"] * r["numel"] for r in rows) / total_numel
    mean_abs_delta_fp = float(np.mean([r["mean_abs_delta_fp"] for r in rows]))
    mean_abs_delta_qnorm = float(np.mean([r["mean_abs_delta_qnorm"] for r in rows]))

    rows_sorted = sorted(rows, key=lambda x: x["changed_ratio_dynamic"], reverse=True)
    return {
        "summary": {
            "layer_count": len(rows),
            "weighted_changed_ratio_dynamic": float(weighted_changed_ratio_dynamic),
            "weighted_changed_ratio_same_scale": float(weighted_changed_ratio_same_scale),
            "mean_abs_delta_fp": mean_abs_delta_fp,
            "mean_abs_delta_qnorm": mean_abs_delta_qnorm,
        },
        "layers": rows_sorted,
        "top_layers": rows_sorted[:max(0, topk)],
    }


def log_quant_update_diagnostics(
    qnn: nn.Module,
    epoch: int,
    total_epochs: int,
    topk: int = 12,
) -> Dict[str, Any]:
    diag = collect_quant_update_diagnostics(qnn=qnn, topk=topk)
    if diag["summary"] is None:
        LOGGER.warning("[QuantUpdateDiag] 無可用層（未找到 QuantModule_DiffAE_LoRA）")
        return diag

    s = diag["summary"]
    LOGGER.info(
        "[QuantUpdateDiag] Epoch %d/%d | layers=%d | weighted_changed_ratio(dynamic)=%.6f | weighted_changed_ratio(same_scale)=%.6f | mean|ΔW_fp|=%.6e | mean|ΔW_qnorm|=%.6e",
        epoch, total_epochs, s["layer_count"],
        s["weighted_changed_ratio_dynamic"], s["weighted_changed_ratio_same_scale"],
        s["mean_abs_delta_fp"], s["mean_abs_delta_qnorm"],
    )

    LOGGER.info("[QuantUpdateDiag] Top-%d layers by changed_ratio_dynamic:", max(0, topk))
    for row in diag["top_layers"]:
        LOGGER.info(
            "  layer=%s | ratio_dyn=%.6f | ratio_same_scale=%.6f | mean|ΔW_fp|=%.6e | mean|ΔW_qnorm|=%.6e | numel=%d",
            row["name"], row["changed_ratio_dynamic"], row["changed_ratio_same_scale"],
            row["mean_abs_delta_fp"], row["mean_abs_delta_qnorm"], row["numel"]
        )
    return diag


def _make_quant_diag_run_dir(root_dir: str) -> str:
    os.makedirs(root_dir, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(root_dir, run_id)
    suffix = 1
    while os.path.exists(run_dir):
        run_dir = os.path.join(root_dir, f"{run_id}_{suffix}")
        suffix += 1
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def _append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _plot_quant_diag_curves(history: List[Dict[str, Any]], run_dir: str) -> None:
    if len(history) == 0:
        return

    epochs = [h["epoch"] for h in history]
    ratio_dyn = [h["weighted_changed_ratio_dynamic"] for h in history]
    ratio_same = [h["weighted_changed_ratio_same_scale"] for h in history]
    delta_fp = [h["mean_abs_delta_fp"] for h in history]
    delta_qnorm = [h["mean_abs_delta_qnorm"] for h in history]

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, ratio_dyn, marker="o", linewidth=1.8, label="weighted_changed_ratio_dynamic")
    plt.plot(epochs, ratio_same, marker="s", linewidth=1.8, label="weighted_changed_ratio_same_scale")
    plt.xlabel("Epoch")
    plt.ylabel("Ratio")
    plt.title("Quant Update Effective Ratio")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "curve_weighted_changed_ratio.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, delta_fp, marker="o", linewidth=1.8, label="mean_abs_delta_fp")
    plt.plot(epochs, delta_qnorm, marker="s", linewidth=1.8, label="mean_abs_delta_qnorm")
    plt.xlabel("Epoch")
    plt.ylabel("Magnitude")
    plt.title("Quant Update Delta Magnitude")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, "curve_delta_magnitude.png"), dpi=160)
    plt.close()



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
            LOGGER.info(f"  數據類型: {param.dtype}")
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
    創建 ver2 量化模型（normalized fake-quant + LoRA）
    
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
        lora_rank=lora_rank
    )
    quant_model = quant_model.to(CONFIG.DEVICE)
    quant_model.eval()
    if hasattr(quant_model, 'set_runtime_mode'):
        quant_model.set_runtime_mode(mode='train', use_cached_aw=False, clear_cached_aw=True)
    
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
    根據量化分析數據確定自適應學習率
    
    Args:
        base_model: 基礎模型
        model_type: "diffae" 或 "efficientdm"
    
    Returns:
        包含三種學習率的字典
    """
    # 收集權重尺度統計（ver2: dynamic absmax，無 delta/zp）
    weight_scales = []
    high_error_layers = 0
    total_layers = 0
    
    for name, module in base_model.named_modules():
        if isinstance(module, QuantModule_DiffAE_LoRA):
            total_layers += 1
            
            # 收集權重尺度（以 org_weight 的平均絕對值作為穩定 proxy）
            if hasattr(module, 'org_weight') and torch.is_tensor(module.org_weight):
                avg_scale = module.org_weight.detach().abs().mean().item()
                weight_scales.append(max(avg_scale, 1e-8))
            
            # 檢查激活量化誤差（需要從之前的分析中估算）
            # 這裡使用簡化的啟發式方法
            if hasattr(module, 'act_quantizer'):
                # 假設高誤差層的比例
                if model_type == "diffae" and total_layers % 3 == 0:  # 基於觀察到的高 pct_clipped
                    high_error_layers += 1
    
    if not weight_scales:
        weight_scales = [0.002]  # 預設值
    
    avg_weight_scale = np.mean(weight_scales)
    max_weight_scale = np.max(weight_scales)
    error_ratio = high_error_layers / max(total_layers, 1)
    
    # 基於模型類型和分析數據調整學習率
    if model_type == "diffae":
        # Diff-AE 需要更保守的學習率，因為：
        # 1. 權重變化幅度大
        # 2. 激活量化誤差高
        # 3. 有複雜的 latent space
        
        #lora_factor = 600 # default1
        #lora_factor = 2200  # default2
        lora_factor = 700 # train_3


        weight_quant_lr = 1e-6 #train_3

        #act_quant_lr = 2e-5 # default
        act_quant_lr = 5e-4 # train_3
        #act_quant_lr = 5e-4 # GPT test2

        
    else:  # efficientdm
        # EfficientDM 可以使用相對激進的學習率
        lora_factor = 2000  # 2500 -> 1500-2000
        weight_quant_lr = 1.5e-6 
        act_quant_lr = 6e-4 
    
    return {
        'lora_factor': lora_factor,
        'weight_quant_lr': weight_quant_lr,
        'act_quant_lr': act_quant_lr,
        'stats': {
            'avg_weight_scale': avg_weight_scale,
            'max_weight_scale': max_weight_scale,
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
    基於量化分析數據的動態學習率設定
    """
    from transformers import get_linear_schedule_with_warmup
    
    # 獲取自適應學習率
    lr_config = get_adaptive_learning_rates(base_model, model_type)
    
    lora_factor = lr_config['lora_factor']
    weight_quant_lr = lr_config['weight_quant_lr']
    act_quant_lr = lr_config['act_quant_lr']
    
    LOGGER.info("=== 自適應學習率設定 (%s) ===", model_type.upper())
    LOGGER.info("LoRA 因子: %s", lora_factor)
    LOGGER.info("權重量化 LR: %.2e", weight_quant_lr)
    LOGGER.info("激活量化 LR: %.2e", act_quant_lr)
    LOGGER.info("統計信息: %s", lr_config['stats'])
    
    param_groups = []
    lora_group_count = 0
    aq_group_count = 0

    # -----------------------------------------------------------------------------
    # [Reference: first-run style (train_1)]
    # min_lora_lr = None
    # max_lora_lr = None
    # -----------------------------------------------------------------------------
    # [Reference: first-run style train_2]
    # Keep per-layer adaptive LR, but avoid overly small early-stage updates.
    # min_lora_lr = 1e-6
    # max_lora_lr = 3e-5
    
    
    for name, module in base_model.named_modules():
        if isinstance(module, QuantModule_DiffAE_LoRA):
            if not hasattr(module, 'act_quantizer') or not hasattr(module.act_quantizer, 'scale_list'):
                raise ValueError(f'Layer {name} missing act_quantizer.scale_list')
            if len(module.act_quantizer.scale_list) != ddim_steps:
                raise ValueError('Wrong act_quantizer.scale_list length')

            # ver2: LoRA LR 以權重尺度估計（取 org_weight abs-mean）
            if hasattr(module, 'org_weight') and torch.is_tensor(module.org_weight):
                avg_weight_scale = module.org_weight.detach().abs().mean().clamp(min=1e-8)
            else:
                avg_weight_scale = torch.tensor(0.002, device=next(module.parameters()).device)
            
            # -----------------------------------------------------------------------------
            # [Reference: first-run style train_1]
            # lora_lr = float(avg_weight_scale) / float(max(lora_factor, 1))
            # -----------------------------------------------------------------------------
            # [Reference: first-run style train_2]
            # lora_lr = float(avg_weight_scale) / float(max(lora_factor, 1))
            # if (min_lora_lr is not None) and (max_lora_lr is not None):
            #     lora_lr = max(min_lora_lr, min(lora_lr, max_lora_lr))
            # -----------------------------------------------------------------------------
            # [Reference: first-run style train_3]
            lora_lr = float(avg_weight_scale) / float(max(lora_factor, 1))
            # -----------------------------------------------------------------------------

            # LoRA 參數
            lora_params = [
                param for p_name, param in module.named_parameters()
                if 'lora' in p_name and param.requires_grad
            ]
            if len(lora_params) > 0:
                # -----------------------------------------------------------------------------
                # [Reference: first-run style train_1]
                # param_groups.append({'params': lora_params, 'lr': lora_lr})
                # -----------------------------------------------------------------------------
                # [Reference: first-run style train_2]
                #param_groups.append({
                #    'params': lora_params,
                #    'lr': lora_lr,
                #    # LoRA is low-rank adaptation; strong decay tends to underfit.
                #    'weight_decay': 0.0,
                #})
                # -----------------------------------------------------------------------------
                # [Reference: first-run style train_3]
                # param_groups.append({'params': lora_params, 'lr': lora_lr})
                # -----------------------------------------------------------------------------
                # GPT test1
                param_groups.append({'params': lora_params, 'lr': lora_lr, 'weight_decay': 0.0})
                lora_group_count += 1

            # Activation quant 參數（ver2: scale_list）
            aq_params = [
                param for p_name, param in module.named_parameters()
                if ('scale_list' in p_name) and param.requires_grad
            ]
            if len(aq_params) > 0:
                # -----------------------------------------------------------------------------
                # [Reference: first-run style train_1]
                # param_groups.append({'params': aq_params, 'lr': act_quant_lr})
                # -----------------------------------------------------------------------------
                # [Reference: first-run style train_2]
                # # scale_list is a scale parameter; avoid decay shrinkage.
                # param_groups.append({
                #     'params': aq_params,
                #     'lr': act_quant_lr,
                #     'weight_decay': 0.0,
                # })
                # -----------------------------------------------------------------------------
                # [Reference: first-run style train_3]
                # param_groups.append({'params': aq_params, 'lr': act_quant_lr})
                # -----------------------------------------------------------------------------
                # GPT test1
                param_groups.append({'params': aq_params, 'lr': act_quant_lr, 'weight_decay': 0.0})
                # -----------------------------------------------------------------------------
                aq_group_count += 1

    if len(param_groups) == 0:
        raise ValueError("No trainable params found for optimizer (expected LoRA/scale_list)")

    LOGGER.info(
        "Optimizer param groups - LoRA: %d, Act-quant(scale_list): %d",
        lora_group_count,
        aq_group_count,
    )
    optimizer = torch.optim.AdamW(param_groups, foreach=False)
    
    # 創建學習率調度器
    total_steps = CONFIG.NUM_EPOCHS * ddim_steps
    
    # -----------------------------------------------------------------------------
    # [Reference: first-run style train_1]
    # warmup_steps = 0
    # -----------------------------------------------------------------------------
    # train_2
    # Previous 3% warmup is too long for this setup and slows early convergence.
    # warmup_steps = min(100, max(20, int(total_steps * 0.005)))
    # ------------------------------------------------------------
    # train_3
    warmup_steps = 100
    # ------------------------------------------------------------
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
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
        optimizer: 優化器
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
    3. 設定優化器與學習率
    4. 訓練與評估
    """
    args = _get_cli_args()
    if args.auto_log:
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_name = f"step6_train_{ts}.log"
    elif args.log_suffix:
        log_name = f"step6_train_{args.log_suffix}.log"
    else:
        log_name = "step6_train.log"

    # 設置運行環境
    log_path = CONFIG.setup_environment(log_name=log_name)
    _seed_all(CONFIG.SEED)

    # TensorBoard writer（可選）
    tb_writer = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter as _SW
            _tb_run_name = log_name.replace(".log", "")
            _tb_run_dir = os.path.join(args.tb_dir, _tb_run_name)
            os.makedirs(_tb_run_dir, exist_ok=True)
            tb_writer = _SW(log_dir=_tb_run_dir)
            LOGGER.info(f"TensorBoard 已啟用，輸出目錄: {_tb_run_dir}")
        except ImportError:
            LOGGER.warning("torch.utils.tensorboard 未安裝，TensorBoard 已停用。")

    LOGGER.info("=" * 50)
    LOGGER.info("Diff-AE EfficientDM Step 6: 量化感知微調 (QAT)")
    LOGGER.info("=" * 50)
    LOGGER.info(f"訓練 log 檔案: {log_path}")
    LOGGER.info(f"使用設備: {CONFIG.DEVICE}")
    
    # 記錄新功能狀態
    LOGGER.info("=== 新功能狀態 ===")
    LOGGER.info(f"量化更新診斷 interval: {args.quant_diag_interval}")
    LOGGER.info(f"量化更新診斷 topk: {args.quant_diag_topk}")
    LOGGER.info(f"量化更新診斷 root: {args.quant_diag_root}")
    LOGGER.info(f"Loss chunk size(optimizer/global step): {args.loss_chunk_size}")
    LOGGER.info(
        "Teacher autocast match (A/B): %s | Scale-list debug: %s (interval=%d)",
        args.teacher_autocast_match,
        args.debug_scale_list_update,
        args.debug_scale_list_interval,
    )
    LOGGER.info(
        "Timestep grad conflict debug: %s | steps=%s | interval=%d(epoch)",
        args.debug_timestep_grad_conflict,
        args.debug_timestep_grad_steps,
        args.debug_timestep_grad_interval,
    )
    LOGGER.info(
        "Tail repair: %s | outer_steps=%d | t_range=%s | lr_scale=%.6f",
        args.tail_repair_enable,
        args.tail_repair_steps,
        args.tail_repair_t_range,
        args.tail_repair_lr_scale,
    )
    
    try:
        quant_diag_run_dir = None
        quant_diag_jsonl = None
        quant_diag_summary_json = None
        quant_diag_history: List[Dict[str, Any]] = []
        if args.quant_diag_interval > 0:
            quant_diag_run_dir = _make_quant_diag_run_dir(args.quant_diag_root)
            quant_diag_jsonl = os.path.join(quant_diag_run_dir, "quant_update_diag.jsonl")
            quant_diag_summary_json = os.path.join(quant_diag_run_dir, "quant_update_diag_summary.json")
            LOGGER.info(f"量化更新診斷輸出目錄: {quant_diag_run_dir}")

        # 1. 載入基礎模型
        base_model : LitModel = load_diffae_model()
        LOGGER.info("✅ Diff-AE 模型載入成功")
        
        fp_model = deepcopy(base_model.ema_model).to(CONFIG.DEVICE)
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
        #quant_model.set_quant_state(True, False)  # Test 1
        quant_model.set_quant_state(True, True)  # Test 2
        if hasattr(quant_model, 'set_runtime_mode'):
            quant_model.set_runtime_mode(mode='train', use_cached_aw=False, clear_cached_aw=True)

        # ver2 float pipeline: no int dequantizer path here
        ckpt = torch.load(CONFIG.QUANT_CKPT_PATH, map_location='cpu')
        quant_model.load_state_dict(ckpt, strict=False) # no lora weight in ckpt

        

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
            # ver2: train LoRA + per-step activation scale_list
            if 'lora' in name or 'scale_list' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        for name, param in fp_model.named_parameters():
            param.requires_grad = False
        
        # 3. 統計參數
        LOGGER.info("\n=== 模型參數統計 ===")
        print_trainable_parameters(quant_model)
        #count_lora_parameters(quant_model)
        
        # 4. 創建優化器 (按原作邏輯 + Layer-by-Layer 支援)
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
            conf=base_model.conf.clone(),
        )
        distill_trainer.set_loss_chunk_size(args.loss_chunk_size)
        if hasattr(distill_trainer, 'set_teacher_autocast_match'):
            distill_trainer.set_teacher_autocast_match(args.teacher_autocast_match)
        if hasattr(distill_trainer, 'set_scale_list_update_debug'):
            distill_trainer.set_scale_list_update_debug(
                enabled=args.debug_scale_list_update,
                interval=args.debug_scale_list_interval,
            )
        if hasattr(distill_trainer, 'set_timestep_grad_conflict_debug'):
            _grad_steps = []
            for _s in str(args.debug_timestep_grad_steps).split(','):
                _s = _s.strip()
                if len(_s) == 0:
                    continue
                try:
                    _grad_steps.append(int(_s))
                except ValueError:
                    LOGGER.warning("忽略非法 timestep 字串: %s", _s)
            distill_trainer.set_timestep_grad_conflict_debug(
                enabled=args.debug_timestep_grad_conflict,
                steps=_grad_steps,
                interval=args.debug_timestep_grad_interval,
            )
        if hasattr(distill_trainer, 'set_tail_repair'):
            _tr_range: List[int] = []
            for _s in str(args.tail_repair_t_range).split(','):
                _s = _s.strip()
                if len(_s) == 0:
                    continue
                try:
                    _tr_range.append(int(_s))
                except ValueError:
                    LOGGER.warning("忽略非法 tail-repair timestep: %s", _s)
            distill_trainer.set_tail_repair(
                enabled=args.tail_repair_enable,
                steps=int(args.tail_repair_steps),
                t_range=_tr_range,
                lr_scale=float(args.tail_repair_lr_scale),
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
            conf=base_model.conf.clone(),
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
        LOGGER.info("DDIM timestep 定義: t=99 為高噪聲端，t=0 為接近清晰圖端。")

        # 記錄詳細的可訓練參數信息
        #log_trainable_parameters_details(qnn)

        global start_time, best_loss
        training_start_time = time.time()
        start_time = training_start_time
        best_loss = float('inf')
        fp_model.eval()
        best_epoch = -1
        final_epoch_loss = float('nan')

        
        # 使用新的單步 Teacher-Student 訓練方法
        for epoch in range(CONFIG.NUM_EPOCHS):
            _seed_all(CONFIG.SEED + epoch)
            epoch_start = time.time()
            qnn.train()
            
            LOGGER.info(f"\n--- Epoch {epoch+1}/{CONFIG.NUM_EPOCHS} ---")
            
            
            
            '''Diff-AE inference distillation training'''
            # Diff-AE inference distillation training
            
            loss_dict = distill_trainer.training_losses_with_inference_distillation(
                batch_size=CONFIG.BATCH_SIZE,
                shape=(CONFIG.BATCH_SIZE, 3, 128, 128),
                conf=base_model.conf.clone(),
                device=CONFIG.DEVICE,
                debug_epoch=epoch + 1,
            )

            # 舊版 loss 讀法（mean MSE）保留註解：
            epoch_distill_loss = float(loss_dict['distill_loss'])
            avg_loss = float(loss_dict['loss'])

            # 新版：使用 EfficientDM 對齊的 distill_loss_eff
            #epoch_distill_loss_eff = float(loss_dict.get('distill_loss_eff', loss_dict['distill_loss_eff']))
            #avg_loss = epoch_distill_loss_eff
            final_epoch_loss = avg_loss
            epoch_time = time.time() - epoch_start
            
            LOGGER.info(f"\nEpoch {epoch+1}/{CONFIG.NUM_EPOCHS} 完成:")
            LOGGER.info(f"  平均蒸餾損失(DDIM全步平均): {epoch_distill_loss:.6f}")
            LOGGER.info(f"  平均損失(目前等同蒸餾損失): {avg_loss:.6f}")
            #LOGGER.info(f"  平均蒸餾損失(DDIM全步平均,eff): {epoch_distill_loss_eff:.6f}")
            #LOGGER.info(f"  平均損失(目前等同蒸餾損失,eff): {avg_loss:.6f}")
            LOGGER.info(f"  時間: {epoch_time:.2f}秒")

            _tail_rep = loss_dict.get("tail_repair")
            if isinstance(_tail_rep, dict) and int(_tail_rep.get("n_updates", 0)) > 0:
                LOGGER.info(
                    "[TailRepair] epoch=%d summary | n_updates=%d mean_loss=%.6f",
                    epoch + 1,
                    int(_tail_rep["n_updates"]),
                    float(_tail_rep["mean_loss"]),
                )

            # TensorBoard scalar logging
            if tb_writer is not None:
                _global_step = epoch + 1
                tb_writer.add_scalar("train/loss", avg_loss, _global_step)
                tb_writer.add_scalar("train/distill_loss", epoch_distill_loss, _global_step)
                #tb_writer.add_scalar("train/loss_eff", avg_loss, _global_step)
                #tb_writer.add_scalar("train/distill_loss_eff", epoch_distill_loss_eff, _global_step)
                tb_writer.add_scalar("train/epoch_time_sec", epoch_time, _global_step)
                # 取 optimizer 第一個 param group 的 LR 作代表
                _cur_lr = optimizer.param_groups[0]["lr"]
                tb_writer.add_scalar("train/lr", _cur_lr, _global_step)
                if isinstance(_tail_rep, dict) and int(_tail_rep.get("n_updates", 0)) > 0:
                    tb_writer.add_scalar(
                        "train/tail_repair_mean_loss",
                        float(_tail_rep.get("mean_loss", float("nan"))),
                        _global_step,
                    )
                    tb_writer.add_scalar(
                        "train/tail_repair_n_updates",
                        int(_tail_rep.get("n_updates", 0)),
                        _global_step,
                    )

            # 單步 DDIM loss / 累積平均 DDIM loss（x 軸：global optimizer step）
            rollout_step_records = loss_dict.get('rollout_step_records', [])
            for rec in rollout_step_records:
                _step_x = int(rec["global_step"])
                if tb_writer is not None:
                    tb_writer.add_scalar("loss/current_ddim_step_loss", float(rec["current_step_loss"]), _step_x)
                    tb_writer.add_scalar("loss/running_mean_ddim_loss", float(rec["running_mean_ddim_loss"]), _step_x)
                    #tb_writer.add_scalar("loss/current_ddim_step_loss_eff", float(rec["current_step_loss_eff"]), _step_x)
                    #tb_writer.add_scalar("loss/running_mean_ddim_loss_eff", float(rec["running_mean_ddim_loss_eff"]), _step_x)

            # 每個 timestep 的平均 loss 分佈（epoch 粒度）
            per_timestep_loss_sum = np.zeros(CONFIG.NUM_DIFFUSION_STEPS, dtype=np.float64)
            per_timestep_loss_count = np.zeros(CONFIG.NUM_DIFFUSION_STEPS, dtype=np.int64)
            for rec in rollout_step_records:
                _t = int(rec["ddim_t"])
                if 0 <= _t < CONFIG.NUM_DIFFUSION_STEPS:
                    per_timestep_loss_sum[_t] += float(rec["current_step_loss"])
                    #per_timestep_loss_sum[_t] += float(rec["current_step_loss_eff"])
                    per_timestep_loss_count[_t] += 1

            per_timestep_mean_loss = np.full(CONFIG.NUM_DIFFUSION_STEPS, np.nan, dtype=np.float64)
            _valid = per_timestep_loss_count > 0
            per_timestep_mean_loss[_valid] = per_timestep_loss_sum[_valid] / per_timestep_loss_count[_valid]

            if tb_writer is not None:
                _epoch_step = epoch + 1
                for _t in range(CONFIG.NUM_DIFFUSION_STEPS):
                    if not np.isnan(per_timestep_mean_loss[_t]):
                        tb_writer.add_scalar(
                            f"loss_timestep_mean/t_{_t:03d}",
                            float(per_timestep_mean_loss[_t]),
                            _epoch_step,
                        )
                        tb_writer.add_scalar(
                            f"loss_timestep/t_{_t:03d}",
                            float(per_timestep_mean_loss[_t]),
                            _epoch_step,
                        )

            # 尾段主導指標（尾段定義：t=0..19，接近清晰圖）
            _tail_n = min(20, CONFIG.NUM_DIFFUSION_STEPS)
            _tail = per_timestep_mean_loss[:_tail_n]
            _all = per_timestep_mean_loss
            _head = per_timestep_mean_loss[-_tail_n:]
            _tail_valid = _tail[~np.isnan(_tail)]
            _all_valid = _all[~np.isnan(_all)]
            _head_valid = _head[~np.isnan(_head)]

            tail_ratio = float("nan")
            tail_outlier_ratio = float("nan")
            if len(_tail_valid) > 0 and len(_all_valid) > 0:
                tail_ratio = float(_tail_valid.mean() / max(_all_valid.mean(), 1e-12))
                tail_outlier_ratio = float(_tail_valid.max() / max(_tail_valid.mean(), 1e-12))
                if tb_writer is not None:
                    _epoch_step = epoch + 1
                    tb_writer.add_scalar("summary/tail_mean_ratio", tail_ratio, _epoch_step)
                    tb_writer.add_scalar("summary/tail_max_ratio", tail_outlier_ratio, _epoch_step)
                    tb_writer.add_scalar("summary/tail_ratio", tail_ratio, _epoch_step)
                    tb_writer.add_scalar("summary/tail_outlier_ratio", tail_outlier_ratio, _epoch_step)

            _tail_mean = float(_tail_valid.mean()) if len(_tail_valid) > 0 else float("nan")
            _head_mean = float(_head_valid.mean()) if len(_head_valid) > 0 else float("nan")
            _first5 = ", ".join(
                f"t{t:03d}:{per_timestep_mean_loss[t]:.3e}"
                for t in range(min(5, CONFIG.NUM_DIFFUSION_STEPS))
                if not np.isnan(per_timestep_mean_loss[t])
            )
            _last5 = ", ".join(
                f"t{t:03d}:{per_timestep_mean_loss[t]:.3e}"
                for t in range(max(0, CONFIG.NUM_DIFFUSION_STEPS - 5), CONFIG.NUM_DIFFUSION_STEPS)
                if not np.isnan(per_timestep_mean_loss[t])
            )
            LOGGER.info(
                "[LossSummary] tail_ratio=%.6f | tail_outlier_ratio=%.6f | mean(t=0..%d)=%.6e | mean(t=%d..%d)=%.6e",
                tail_ratio,
                tail_outlier_ratio,
                _tail_n - 1,
                _tail_mean,
                CONFIG.NUM_DIFFUSION_STEPS - _tail_n,
                CONFIG.NUM_DIFFUSION_STEPS - 1,
                _head_mean,
            )
            LOGGER.info("[LossSummary] per-timestep mean (first5): %s", _first5 if _first5 else "N/A")
            LOGGER.info("[LossSummary] per-timestep mean (last5): %s", _last5 if _last5 else "N/A")

            # training-step (optimizer/global step) non-overlap chunk summary
            chunk_summaries = loss_dict.get('loss_chunk_summaries', [])
            for chunk in chunk_summaries:
                chunk_end_step = int(chunk["chunk_end_global_step"])
                if tb_writer is not None:
                    tb_writer.add_scalar("chunk/mean_total_loss", chunk["mean_total_loss"], chunk_end_step)
                    tb_writer.add_scalar("chunk/std_total_loss", chunk["std_total_loss"], chunk_end_step)
                    tb_writer.add_scalar("chunk/max_total_loss", chunk["max_total_loss"], chunk_end_step)
                    tb_writer.add_scalar("chunk/min_total_loss", chunk["min_total_loss"], chunk_end_step)
                    if "mean_distill_loss" in chunk:
                        tb_writer.add_scalar("chunk/mean_distill_loss", chunk["mean_distill_loss"], chunk_end_step)
                        tb_writer.add_scalar("chunk/std_distill_loss", chunk["std_distill_loss"], chunk_end_step)
                        tb_writer.add_scalar("chunk/max_distill_loss", chunk["max_distill_loss"], chunk_end_step)
                        tb_writer.add_scalar("chunk/min_distill_loss", chunk["min_distill_loss"], chunk_end_step)

                LOGGER.info(
                    "[LossChunk] steps=%d..%d size=%d | total(mean=%.6f std=%.6f max=%.6f min=%.6f)%s",
                    int(chunk["chunk_start_global_step"]),
                    chunk_end_step,
                    int(chunk["chunk_size"]),
                    float(chunk["mean_total_loss"]),
                    float(chunk["std_total_loss"]),
                    float(chunk["max_total_loss"]),
                    float(chunk["min_total_loss"]),
                    (
                        " | distill(mean=%.6f std=%.6f max=%.6f min=%.6f)" % (
                            float(chunk["mean_distill_loss"]),
                            float(chunk["std_distill_loss"]),
                            float(chunk["max_distill_loss"]),
                            float(chunk["min_distill_loss"]),
                        )
                    ) if "mean_distill_loss" in chunk else "",
                )

            # 每層有效量化更新比例診斷
            if args.quant_diag_interval > 0:
                do_diag = ((epoch + 1) == 1) or (((epoch + 1) % args.quant_diag_interval) == 0)
                if do_diag:
                    diag = log_quant_update_diagnostics(
                        qnn=qnn,
                        epoch=epoch + 1,
                        total_epochs=CONFIG.NUM_EPOCHS,
                        topk=args.quant_diag_topk,
                    )
                    if (quant_diag_run_dir is not None) and (diag.get("summary") is not None):
                        entry = {
                            "epoch": epoch + 1,
                            "total_epochs": CONFIG.NUM_EPOCHS,
                            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "summary": diag["summary"],
                            "top_layers": diag.get("top_layers", []),
                        }
                        if args.quant_diag_save_full_layers:
                            entry["layers"] = diag.get("layers", [])

                        if quant_diag_jsonl is not None:
                            _append_jsonl(quant_diag_jsonl, entry)

                        s = diag["summary"]
                        quant_diag_history.append({
                            "epoch": int(epoch + 1),
                            "weighted_changed_ratio_dynamic": float(s["weighted_changed_ratio_dynamic"]),
                            "weighted_changed_ratio_same_scale": float(s["weighted_changed_ratio_same_scale"]),
                            "mean_abs_delta_fp": float(s["mean_abs_delta_fp"]),
                            "mean_abs_delta_qnorm": float(s["mean_abs_delta_qnorm"]),
                        })
                        _plot_quant_diag_curves(quant_diag_history, quant_diag_run_dir)
                        if quant_diag_summary_json is not None:
                            _write_json(
                                quant_diag_summary_json,
                                {
                                    "run_dir": quant_diag_run_dir,
                                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "diag_interval": args.quant_diag_interval,
                                    "diag_topk": args.quant_diag_topk,
                                    "history": quant_diag_history,
                                    "last_entry": entry,
                                },
                            )

            
                

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
                    validation_start_time = time.time()
                    ema_model = distill_trainer.get_ema_model()
                    # 驗證生成時：QuantModule 關閉量化 (False, False)，LoRA 量化模組打開 (True, True)
                    ema_model.set_quant_state(True, True) # GPT test2
                    #ema_model.set_quant_state(True, False) # GPT test1
                    if hasattr(ema_model, 'set_runtime_mode'):
                        ema_model.set_runtime_mode(mode='infer', use_cached_aw=True, clear_cached_aw=True)
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

                    gen_time = time.time() - validation_start_time

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
                        os.path.join('QATcode/quantize_ver2/training_samples', f'epoch_{epoch}_grid.png'))

                    # TensorBoard image logging（grid_image shape: [C,H,W], float 0~1）
                    if tb_writer is not None:
                        tb_writer.add_image("val/image_grid", grid_image, epoch + 1)

                    LOGGER.info(f"✅ Epoch {epoch+1} EMA 模型驗證生成完成:")
                    LOGGER.info(f"  生成時間: {gen_time:.2f}秒")
                    LOGGER.info(f"  圖像範圍: [{batch_images.min():.3f}, {batch_images.max():.3f}]")
                    LOGGER.info(f"  保存路徑: QATcode/quantize_ver2/training_samples/")
                    
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
        total_time = time.time() - training_start_time
        LOGGER.info(f"\n=== 整合訓練完成 ===")
        LOGGER.info(f"總訓練時間: {total_time:.2f}秒")
        LOGGER.info(f"最佳損失: {best_loss:.6f}")
        LOGGER.info(f"平均每個 epoch: {total_time/CONFIG.NUM_EPOCHS:.2f}秒")
        
        # 11. 保存最終模型
        save_checkpoint(base_model, qnn, optimizer, CONFIG.NUM_EPOCHS-1, final_epoch_loss, 
                          ema_helper=distill_trainer.ema, is_best=False)
        
        # 12. 模型測試 - 使用 Diff-AE 的測試邏輯
        LOGGER.info(f"\n=== 模型測試 ===")
        qnn.eval()
        
    except Exception as e:
        LOGGER.error(f"❌ 訓練過程錯誤: {e}")
        import traceback
        LOGGER.error(traceback.format_exc())
    finally:
        if tb_writer is not None:
            tb_writer.close()
            LOGGER.info("TensorBoard writer 已關閉。")

    LOGGER.info("\n🎉 Step 6 量化感知微調完成！")

if __name__ == "__main__":
    main()
