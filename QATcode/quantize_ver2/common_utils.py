"""Shared utility helpers for quantization and cache workflows."""

import logging
import random
import time
from typing import Any, Callable, Dict, Optional, Tuple, Type

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

DEFAULT_CALIB_BATCH_SIZE = 32
DEFAULT_IMAGE_SHAPE = (3, 128, 128)
DEFAULT_NUM_CLASSES = 1000


def seed_all(seed: int) -> None:
    """Set random seeds for random/numpy/torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_trainable_parameters(
    model: nn.Module,
    logger: logging.Logger,
) -> Tuple[int, int]:
    """
    Log trainable/total parameter counts with a provided logger.
    """
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    logger.info(
        "可訓練參數: %s || 總參數: %s || 可訓練比例: %.2f%%",
        f"{trainable_params:,}",
        f"{all_param:,}",
        100 * trainable_params / all_param if all_param > 0 else 0.0,
    )
    return trainable_params, all_param


def make_time_operation(logger: logging.Logger) -> Callable:
    """
    Decorator factory for logging function execution time.
    """

    def time_operation(func: Callable) -> Callable:
        """Public function time_operation."""

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Public function wrapper."""
            start_time = time.time()
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info("執行 '%s' 完成，耗時: %.2f 秒", func.__name__, elapsed)
            return result

        return wrapper

    return time_operation


def get_train_samples(
    train_loader: DataLoader,
    num_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collect batch tensors from dataloader and truncate to num_samples.
    """
    image_data, t_data, y_data = [], [], []
    for image, t, y in train_loader:
        image_data.append(image)
        t_data.append(t)
        y_data.append(y)
        if len(image_data) >= num_samples:
            break

    return (
        torch.cat(image_data, dim=0)[:num_samples],
        torch.cat(t_data, dim=0)[:num_samples],
        torch.cat(y_data, dim=0)[:num_samples],
    )


def load_calibration_data(
    *,
    calib_data_path: str,
    calib_samples: int,
    num_diffusion_steps: int,
    dataset_cls: Type,
    logger: logging.Logger,
    batch_size: int = DEFAULT_CALIB_BATCH_SIZE,
    image_shape: Tuple[int, int, int] = DEFAULT_IMAGE_SHAPE,
    num_classes: int = DEFAULT_NUM_CLASSES,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load calibration tensors; fall back to synthetic tensors on failure.
    """
    logger.info("載入校準資料...")
    try:
        dataset = dataset_cls(calib_data_path)
        data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        cali_images, cali_t, cali_y = get_train_samples(
            data_loader,
            num_samples=calib_samples,
        )
        logger.info("✅ 載入真實校準資料成功")
    except Exception as e:
        logger.warning("⚠️ 載入校準資料失敗: %s", e)
        logger.info("使用合成校準資料")
        cali_images = torch.randn(calib_samples, *image_shape)
        cali_t = torch.randint(0, num_diffusion_steps, (calib_samples,))
        cali_y = torch.randint(0, num_classes, (calib_samples,))

    return cali_images, cali_t, cali_y


@torch.no_grad()
def sync_ema_once(base_model: Any) -> Any:
    """
    Copy current base_model.model into base_model.ema_model once.
    """
    model = base_model.model
    # Use deepcopy on demand to preserve existing behavior.
    import copy as _copy

    base_model.ema_model = _copy.deepcopy(model)
    return base_model


def make_state_dict(m: torch.nn.Module, drop_uint8: bool = True) -> Any:
    """
    Export state_dict on CPU; optionally drop uint8 tensors.
    """
    out = {}
    for k, v in m.state_dict().items():
        if drop_uint8 and getattr(v, "dtype", None) == torch.uint8:
            continue
        out[k] = v.detach().cpu()
    return out


def remap_keys(
    sd: Dict[str, Any], drop_prefix: Optional[str] = None, add_prefix: Optional[str] = None
) -> Dict[str, Any]:
    """
    Remap checkpoint keys by dropping/adding prefixes.
    """
    out = {}
    for k, v in sd.items():
        if drop_prefix and k.startswith(drop_prefix):
            k = k[len(drop_prefix) :]
        if add_prefix:
            k = add_prefix + k
        out[k] = v
    return out


def load_diffae_model(
    model_path: str,
    logger: logging.Logger,
    *,
    log_on_start: str = "載入 Diff-AE 模型: %s",
    log_on_done: str = "Diff-AE 模型載入完成",
) -> Any:
    """
    Load pretrained Diff-AE (LitModel) from a checkpoint.
    Lazy-imports experiment / templates_latent to reduce import cycles.
    """
    from experiment import LitModel
    from templates_latent import ffhq128_autoenc_latent

    logger.info(log_on_start, model_path)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    conf = ffhq128_autoenc_latent()
    model = LitModel(conf)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    logger.info(log_on_done)
    return model
