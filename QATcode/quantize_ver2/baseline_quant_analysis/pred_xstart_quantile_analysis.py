"""
Pred-xstart per-timestep quantile analysis (baseline vs quantize_ver2).

Core spec:
- Progressive DDIM loop: sampler.ddim_sample_loop_progressive(...)
- Collect out["pred_xstart"] distribution at each out["t"] (t_idx)
- Histogram accumulation: hist_range=[-1,1], bins=4096
- Quantile extraction via cumulative histogram counts (bin-wise linear interp)
- Outputs two npz (baseline / v2) with identical schema, plus overlay plot
"""

from __future__ import annotations

import copy
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import matplotlib.pyplot as plt

sys.path.append(".")
sys.path.append("./model")

from QATcode.quantize_ver2.quant_dataset_v2 import DiffusionInputDataset
from QATcode.quantize_ver2.quant_model_lora_v2 import (
    QuantModel_DiffAE_LoRA,
    QuantModule_DiffAE_LoRA,
    INT_QuantModel_DiffAE_LoRA,
)
from QATcode.quantize_ver2.quant_layer_v2 import SimpleDequantizer
from QATcode.quantize_ver2.common_utils import (
    seed_all as _seed_all,
    make_time_operation,
    get_train_samples as _common_get_train_samples,
    load_calibration_data as _common_load_calibration_data,
    load_diffae_model as _common_load_diffae_model,
)

from experiment import LitModel


LOGGER = logging.getLogger("pred_xstart_quantile_analysis")


time_operation = make_time_operation(LOGGER)


@dataclass
class PredXStartConfig:
    """Public class PredXStartConfig."""

    # baseline model checkpoint
    MODEL_PATH: str = "checkpoints/ffhq128_autoenc_latent/last.ckpt"
    # quantize ver2 checkpoint
    BEST_CKPT_PATH_100: str = "QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best.pth"
    BEST_CKPT_PATH_20: str = "QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best_20steps.pth"

    # LoRA / quant params for constructing QuantModel_DiffAE_LoRA
    LORA_RANK: int = 32
    N_BITS_W: int = 8
    N_BITS_A: int = 8

    # calibration for TemporalActivationQuantizer
    CALIB_DATA_PATH: str = "QATcode/quantize_ver2/calibration_diffae.pth"
    CALIB_SAMPLES: int = 1024

    # default diffusion steps; overriden by CLI --num_steps
    NUM_DIFFUSION_STEPS: int = 100

    # runtime device
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CONFIG = PredXStartConfig()


def load_diffae_model(model_path: str = CONFIG.MODEL_PATH) -> LitModel:
    """Public function load_diffae_model."""
    return _common_load_diffae_model(model_path, LOGGER)


def create_float_quantized_model(
    diffusion_model: nn.Module,
    num_steps: int = CONFIG.NUM_DIFFUSION_STEPS,
    lora_rank: int = CONFIG.LORA_RANK,
    mode: str = "train",
) -> QuantModel_DiffAE_LoRA:
    """Public function create_float_quantized_model."""
    LOGGER.info("=== 創建 LoRA Float Fake-Quant 模型 ===")
    wq_params = {
        "n_bits": CONFIG.N_BITS_W,
        "channel_wise": True,
        "scale_method": "absmax",
    }
    aq_params = {
        "n_bits": CONFIG.N_BITS_A,
        "channel_wise": False,
        "scale_method": "absmax",
        "leaf_param": True,
    }

    quant_model = QuantModel_DiffAE_LoRA(
        model=diffusion_model,
        weight_quant_params=wq_params,
        act_quant_params=aq_params,
        num_steps=num_steps,
        lora_rank=lora_rank,
        mode=mode,
    )
    quant_model.eval()
    return quant_model


def create_int_quantized_model(
    diffusion_model: nn.Module,
    num_steps: int = CONFIG.NUM_DIFFUSION_STEPS,
    lora_rank: int = CONFIG.LORA_RANK,
    mode: str = "train",
) -> INT_QuantModel_DiffAE_LoRA:
    """Public function create_int_quantized_model."""
    LOGGER.info("=== 創建 LoRA True-INT 模型(目前分析預設不使用) ===")
    wq_params = {
        "n_bits": CONFIG.N_BITS_W,
        "channel_wise": True,
        "scale_method": "mse",
    }
    aq_params = {
        "n_bits": CONFIG.N_BITS_A,
        "channel_wise": False,
        "scale_method": "max",
        "leaf_param": True,
    }

    quant_model = INT_QuantModel_DiffAE_LoRA(
        model=diffusion_model,
        weight_quant_params=wq_params,
        act_quant_params=aq_params,
        num_steps=num_steps,
    )
    quant_model.eval()
    return quant_model


def _get_train_samples(
    train_loader: DataLoader, num_samples: int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return _common_get_train_samples(train_loader, num_samples)


def load_calibration_data() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Public function load_calibration_data."""
    return _common_load_calibration_data(
        calib_data_path=CONFIG.CALIB_DATA_PATH,
        calib_samples=CONFIG.CALIB_SAMPLES,
        num_diffusion_steps=CONFIG.NUM_DIFFUSION_STEPS,
        dataset_cls=DiffusionInputDataset,
        logger=LOGGER,
    )


def _collect_prefixed_state_dict(
    ckpt: Dict[str, Any],
    prefix: str,
    add_prefix: str = "",
) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    for k, v in ckpt.items():
        if isinstance(k, str) and k.startswith(prefix) and torch.is_tensor(v):
            new_k = k[len(prefix) :]
            if add_prefix:
                new_k = add_prefix + new_k
            out[new_k] = v
    return out


def _load_quant_and_ema_from_ckpt(
    base_model: LitModel, quant_model: nn.Module, ckpt: Dict[str, Any]
) -> None:
    """
    Load strategy (consistent with the existing codebase logic):
    - base_model.ema_model 架構 = deepcopy(quant_model)
    - base_model.ema_model 權重只喫 `ema_model.model.*` (mapped to `model.*` keys)
    - base_model.model 權重喫 `model.model.*` (if exists)
    """
    setattr(base_model, "model", quant_model)
    base_model.ema_model = copy.deepcopy(quant_model)

    ema_sd = _collect_prefixed_state_dict(ckpt, "ema_model.model.", add_prefix="model.")
    if len(ema_sd) == 0:
        raise KeyError("Checkpoint 不包含 `ema_model.model.*` 權重；無法依指定規則載入 EMA 生成模型。")
    ema_msg = base_model.ema_model.load_state_dict(ema_sd, strict=False)
    LOGGER.info(
        "EMA(load from ema_model.model.*): provided=%d effective_loaded=%d missing=%d unexpected=%d",
        len(ema_sd),
        len(ema_sd) - len(ema_msg.unexpected_keys),
        len(ema_msg.missing_keys),
        len(ema_msg.unexpected_keys),
    )

    model_sd = _collect_prefixed_state_dict(ckpt, "model.model.", add_prefix="model.")
    if len(model_sd) > 0:
        model_msg = base_model.model.load_state_dict(model_sd, strict=False)
        LOGGER.info(
            "MODEL(load from model.model.*): provided=%d effective_loaded=%d missing=%d unexpected=%d",
            len(model_sd),
            len(model_sd) - len(model_msg.unexpected_keys),
            len(model_msg.missing_keys),
            len(model_msg.unexpected_keys),
        )


PRED_XSTART_Q_LIST = np.array([0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99], dtype=np.float64)
PRED_XSTART_Q_LIST_FULL = np.array(
    [0.001, 0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99, 0.999], dtype=np.float64
)


def _images_mode_to_num_images(images_mode: str) -> int:
    if images_mode == "debug":
        return 32
    if images_mode == "v1":
        return 64
    if images_mode == "official":
        return 128
    raise ValueError(f"Invalid images_mode: {images_mode}")


def _resolve_best_ckpt_path(num_diffusion_steps: int) -> str:
    # align with existing assumption: 100 steps uses _100 checkpoint, otherwise use _20.
    if num_diffusion_steps == 100:
        return CONFIG.BEST_CKPT_PATH_100
    return CONFIG.BEST_CKPT_PATH_20


def _resolve_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


SELF_DELTA_BOUNDARY_META = {
    "boundary_nan_policy": "first_step_or_last_step_has_no_adjacent_comparison",
    "nan_when_adjacent_count_zero": True,
    "note": "Adjacent delta compares pred_xstart at t with previous yielded timestep; the first yielded step per chunk has no neighbor.",
}


def _layout_run_paths(out_root: Path) -> Dict[str, Path]:
    """Canonical layout: models/, comparisons/, plots/summary/, plots/pairwise/."""
    return {
        "root": out_root,
        "models": out_root / "models",
        "comparisons": out_root / "comparisons",
        "plots_summary": out_root / "plots" / "summary",
        "plots_pairwise": out_root / "plots" / "pairwise",
    }


def _pairwise_comparison_dir(comparisons_root: Path, a: str, b: str) -> Path:
    """a_vs_b in lowercase (e.g. ff_vs_ft)."""
    return comparisons_root / f"{a.lower()}_vs_{b.lower()}"


def _timestep_zone_masks(T: int) -> Dict[str, np.ndarray]:
    """Split t in {0..T-1} into high / middle / low noise by index thirds (high t = more noise)."""
    t = np.arange(T, dtype=np.int32)
    hi = int(np.ceil(2 * T / 3))
    low = int(np.floor(T / 3))
    return {
        "high_noise": t >= hi,
        "middle_noise": (t >= low) & (t < hi),
        "low_noise": t < low,
    }


def _nanmean_ignore(a: np.ndarray) -> float:
    v = np.asarray(a, dtype=np.float64)
    if v.size == 0:
        return float("nan")
    m = np.nanmean(v)
    return float(m)


def _nanmax_ignore(a: np.ndarray) -> Tuple[float, int]:
    v = np.asarray(a, dtype=np.float64)
    if v.size == 0:
        return float("nan"), -1
    mask = np.isfinite(v)
    if not np.any(mask):
        return float("nan"), -1
    idx = int(np.nanargmax(v))
    return float(v[idx]), idx


def _nanmin_ignore(a: np.ndarray) -> Tuple[float, int]:
    v = np.asarray(a, dtype=np.float64)
    if v.size == 0:
        return float("nan"), -1
    mask = np.isfinite(v)
    if not np.any(mask):
        return float("nan"), -1
    idx = int(np.nanargmin(v))
    return float(v[idx]), idx


def _ensure_legacy_symlinks(out_root: Path, models_root: Path) -> None:
    """Backward compat: out_root/FF -> models/FF, out_root/FF_vs_FT -> comparisons/ff_vs_ft."""
    legacy_map = [
        ("BASELINE", "BASELINE"),
        ("FF", "FF"),
        ("FT", "FT"),
        ("TT", "TT"),
    ]
    for legacy_name, model_name in legacy_map:
        src = models_root / model_name
        dst = out_root / legacy_name
        if not src.exists():
            continue
        if dst.exists() or dst.is_symlink():
            continue
        try:
            dst.symlink_to(f"models/{model_name}", target_is_directory=True)
        except OSError:
            LOGGER.warning("Could not create symlink %s -> models/%s", dst, model_name)

    cmp_src = out_root / "comparisons" / "ff_vs_ft"
    cmp_dst = out_root / "FF_vs_FT"
    if cmp_src.exists() and not cmp_dst.exists() and not cmp_dst.is_symlink():
        try:
            cmp_dst.symlink_to("comparisons/ff_vs_ft", target_is_directory=True)
        except OSError:
            LOGGER.warning("Could not create symlink FF_vs_FT -> comparisons/ff_vs_ft")


def _make_noise_banks(
    *,
    num_images: int,
    chunk_batch: int,
    img_size: int,
    style_ch: int,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    baseline / v2 must share the same inputs:
    - x_T bank for DDIM noise
    - latent_noise bank for latent condition generation
    """
    assert num_images % chunk_batch == 0, "num_images must be divisible by chunk_batch"

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
    return x_T.to(device), latent_noise.to(device)


@dataclass
class _DistAccumulator:
    hist_bins: int
    hist_min: float
    hist_max: float
    # running stats
    count: int = 0
    sum_v: float = 0.0
    sum_sq: float = 0.0
    min_v: float = float("inf")
    max_v: float = float("-inf")
    abs_sum: float = 0.0
    abs_max: float = 0.0
    pos_count: int = 0
    neg_count: int = 0
    zero_count: int = 0
    sat95_count: int = 0
    sat99_count: int = 0
    hist_counts: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.hist_counts = torch.zeros((self.hist_bins,), dtype=torch.float64)

    def update(self, x_flat: torch.Tensor) -> None:
        """Public function update."""
        if x_flat.numel() == 0:
            return
        x = x_flat.detach().to(dtype=torch.float32)
        self.count += int(x.numel())
        self.sum_v += float(x.sum().item())
        self.sum_sq += float((x * x).sum().item())
        self.min_v = min(self.min_v, float(x.min().item()))
        self.max_v = max(self.max_v, float(x.max().item()))
        x_abs = x.abs()
        self.abs_sum += float(x_abs.sum().item())
        self.abs_max = max(self.abs_max, float(x_abs.max().item()))
        self.pos_count += int((x > 0).sum().item())
        self.neg_count += int((x < 0).sum().item())
        self.zero_count += int((x == 0).sum().item())
        self.sat95_count += int((x_abs >= 0.95).sum().item())
        self.sat99_count += int((x_abs >= 0.99).sum().item())
        xc = x.clamp(self.hist_min, self.hist_max)
        hc = torch.histc(xc, bins=self.hist_bins, min=self.hist_min, max=self.hist_max).to(
            torch.float64
        )
        self.hist_counts += hc.cpu()

    def summary(self, *, q_list: np.ndarray, bin_edges: np.ndarray) -> Dict[str, float]:
        """Public function summary."""
        if self.count <= 0:
            return {}
        mean = self.sum_v / self.count
        var = max(0.0, self.sum_sq / self.count - mean * mean)
        std = float(np.sqrt(var))
        qvals = _hist_counts_to_quantiles_single(self.hist_counts.numpy(), bin_edges, q_list)
        q_names = ["q001", "q01", "q05", "q25", "q50", "q75", "q95", "q99", "q999"]
        q_out: Dict[str, float] = {k: float(v) for k, v in zip(q_names, qvals)}
        return {
            "numel": int(self.count),
            "mean": float(mean),
            "std": float(std),
            "min": float(self.min_v),
            "max": float(self.max_v),
            "abs_mean": float(self.abs_sum / self.count),
            "abs_max": float(self.abs_max),
            **q_out,
            "positive_ratio": float(self.pos_count / self.count),
            "negative_ratio": float(self.neg_count / self.count),
            "zero_ratio": float(self.zero_count / self.count),
            "saturation_ratio_abs_ge_095": float(self.sat95_count / self.count),
            "saturation_ratio_abs_ge_099": float(self.sat99_count / self.count),
        }


@dataclass
class _MetricAccumulator:
    sum_l1: float = 0.0
    sum_l2: float = 0.0
    sum_cos: float = 0.0
    count: int = 0

    def update(self, a: torch.Tensor, b: torch.Tensor) -> None:
        """Public function update."""
        da = a.detach().reshape(a.shape[0], -1).to(torch.float32)
        db = b.detach().reshape(b.shape[0], -1).to(torch.float32)
        if da.shape != db.shape:
            raise RuntimeError(f"metric shape mismatch: {da.shape} vs {db.shape}")
        diff = da - db
        l1 = diff.abs().mean(dim=1)
        l2 = torch.sqrt((diff * diff).mean(dim=1) + 1e-12)
        cos = torch.nn.functional.cosine_similarity(da, db, dim=1, eps=1e-8)
        self.sum_l1 += float(l1.sum().item())
        self.sum_l2 += float(l2.sum().item())
        self.sum_cos += float(cos.sum().item())
        self.count += int(da.shape[0])

    def summary(self) -> Dict[str, float]:
        """Public function summary."""
        if self.count <= 0:
            return {"count": 0, "l1": float("nan"), "l2": float("nan"), "cosine": float("nan")}
        return {
            "count": int(self.count),
            "l1": float(self.sum_l1 / self.count),
            "l2": float(self.sum_l2 / self.count),
            "cosine": float(self.sum_cos / self.count),
        }


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
) -> torch.Tensor:
    cond = latent_sampler.sample(
        model=latent_net,
        noise=latent_noise_chunk,
        clip_denoised=conf.latent_clip_sample,
    )
    if conf.latent_znormalize:
        cond = cond * conds_std.to(device) + conds_mean.to(device)
    return cond


def _hist_counts_to_quantiles_single(
    hist_counts_1d: np.ndarray, bin_edges: np.ndarray, q_list: np.ndarray
) -> np.ndarray:
    cumulative = np.cumsum(hist_counts_1d)
    total = cumulative[-1] if cumulative.size > 0 else 0.0
    out = np.zeros((len(q_list),), dtype=np.float64)
    if total <= 0:
        out[:] = np.nan
        return out
    bins = hist_counts_1d.shape[0]
    for qi, q in enumerate(q_list):
        target = q * total
        idx = int(np.searchsorted(cumulative, target, side="left"))
        if idx >= bins:
            idx = bins - 1
        left_edge = bin_edges[idx]
        right_edge = bin_edges[idx + 1]
        prev_cdf = cumulative[idx - 1] if idx > 0 else 0.0
        count_in_bin = hist_counts_1d[idx]
        if count_in_bin <= 0:
            out[qi] = left_edge
        else:
            frac = float(np.clip((target - prev_cdf) / count_in_bin, 0.0, 1.0))
            out[qi] = left_edge + frac * (right_edge - left_edge)
    return out


def _hist_counts_to_quantiles(
    *,
    hist_counts: np.ndarray,  # (T, bins)
    bin_edges: np.ndarray,  # (bins+1,)
    q_list: np.ndarray,  # (7,)
) -> np.ndarray:
    """
    cumulative counts => quantiles via bin-wise linear interpolation.
    """
    T = hist_counts.shape[0]
    quantiles = np.zeros((T, len(q_list)), dtype=np.float64)
    for t in range(T):
        quantiles[t, :] = _hist_counts_to_quantiles_single(hist_counts[t], bin_edges, q_list)
    return quantiles


def _save_pred_xstart_quantile_npz(
    *,
    out_npz_path: Path,
    hist_counts: np.ndarray,
    bin_edges: np.ndarray,
    quantiles: np.ndarray,
    t: np.ndarray,
    metadata: Dict[str, Any],
) -> None:
    out_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz_path,
        hist_counts=hist_counts,
        bin_edges=bin_edges,
        quantiles=quantiles,
        t=t,
        **metadata,
    )


def _configure_model_quant_mode(model: nn.Module, mode_tag: str) -> None:
    mode_u = mode_tag.upper()
    if mode_u == "BASELINE":
        return
    if mode_u == "FF":
        model.set_quant_state(False, False)
    elif mode_u == "FT":
        model.set_quant_state(False, True)
    elif mode_u == "TF":
        model.set_quant_state(True, False)
    elif mode_u == "TT":
        model.set_quant_state(True, True)
    else:
        raise ValueError(f"Unsupported mode tag: {mode_tag}")


def _build_eval_model_with_w_plus_lora(
    *,
    model_path: str,
    ckpt_path: str,
    num_steps: int,
    device: torch.device,
) -> Tuple[LitModel, nn.Module]:
    base_model = load_diffae_model(model_path)
    base_model.to(device)
    base_model.eval()
    base_model.setup()
    base_model.train_dataloader()
    diffusion_model = base_model.ema_model
    quant_model = create_float_quantized_model(
        diffusion_model, num_steps=num_steps, lora_rank=CONFIG.LORA_RANK, mode="train"
    )
    quant_model.to(device)
    quant_model.eval()
    for _, module in quant_model.named_modules():
        if (
            isinstance(module, QuantModule_DiffAE_LoRA)
            and getattr(module, "ignore_reconstruction", False) is False
        ):
            module.intn_dequantizer = SimpleDequantizer(
                uaq=module.weight_quantizer, weight=module.weight
            ).to(device)
    cali_images, cali_t, cali_y = load_calibration_data()
    quant_model.set_first_last_layer_to_8bit()
    quant_model.set_quant_state(True, True)
    with torch.no_grad():
        _ = quant_model(
            x=cali_images[:4].to(device), t=cali_t[:4].to(device), cond=cali_y[:4].to(device)
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _load_quant_and_ema_from_ckpt(base_model, quant_model, ckpt)
    if hasattr(base_model.ema_model, "set_runtime_mode"):
        base_model.ema_model.set_runtime_mode(
            mode="infer", use_cached_aw=True, clear_cached_aw=True
        )
    return base_model, base_model.ema_model


@torch.no_grad()
def _collect_pred_xstart_streaming_stats(
    *,
    model: nn.Module,
    mode_tag: str,
    sampler,
    latent_sampler,
    conf: Any,
    x_T_bank: torch.Tensor,
    latent_noise_bank: torch.Tensor,
    conds_mean: torch.Tensor,
    conds_std: torch.Tensor,
    T: int,
    clip_denoised: bool,
    hist_min: float,
    hist_max: float,
    hist_bins: int,
    chunk_batch: int,
    device: torch.device,
) -> Dict[str, Any]:
    _configure_model_quant_mode(model, mode_tag)
    per_t = [
        _DistAccumulator(hist_bins=hist_bins, hist_min=hist_min, hist_max=hist_max)
        for _ in range(T)
    ]
    self_delta = [_MetricAccumulator() for _ in range(T)]
    counts_t = np.zeros((T,), dtype=np.int64)
    num_images = x_T_bank.shape[0]
    num_chunks = num_images // chunk_batch
    cache_scheduler = getattr(conf, "cache_scheduler", None)
    for ci in range(num_chunks):
        b0, b1 = ci * chunk_batch, (ci + 1) * chunk_batch
        x_T_chunk = x_T_bank[b0:b1].to(device)
        latent_noise_chunk = latent_noise_bank[b0:b1].to(device)
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
            model_kwargs = {"cond": cond_chunk}
        else:
            model_kwargs = None
        prev_pred: Optional[torch.Tensor] = None
        prev_t_idx: Optional[int] = None
        for out in sampler.ddim_sample_loop_progressive(
            model=model,
            noise=x_T_chunk,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
            device=device,
            progress=False,
            eta=0.0,
            cache_scheduler=cache_scheduler,
        ):
            t_idx = int(out["t"][0].item())
            pred = out["pred_xstart"].detach()
            if t_idx == 0:
                LOGGER.debug("pred_xstart shape at t=0: %s", tuple(pred.shape))
                per_t[t_idx].update(pred.reshape(-1))
                counts_t[t_idx] += int(pred.numel())
                if prev_pred is not None and prev_t_idx is not None:
                    # adjacent step delta anchored on current t
                    self_delta[t_idx].update(pred, prev_pred)
                prev_pred = pred
                prev_t_idx = t_idx
    bin_edges = np.linspace(hist_min, hist_max, hist_bins + 1, dtype=np.float64)
    stats_by_t = {}
    hist_counts = np.zeros((T, hist_bins), dtype=np.float64)
    for t in range(T):
        s = per_t[t].summary(q_list=PRED_XSTART_Q_LIST_FULL, bin_edges=bin_edges)
        d = self_delta[t].summary()
        if s:
            s["adjacent_l1"] = d["l1"]
            s["adjacent_l2"] = d["l2"]
            s["adjacent_cosine"] = d["cosine"]
            s["adjacent_count"] = d["count"]
            stats_by_t[str(t)] = s
            hist_counts[t, :] = per_t[t].hist_counts.numpy()
    return {
        "stats_by_t": stats_by_t,
        "hist_counts": hist_counts,
        "bin_edges": bin_edges,
        "counts_t": counts_t,
    }


@torch.no_grad()
def _collect_cross_model_pair_delta(
    *,
    model_a: nn.Module,
    model_b: nn.Module,
    mode_a: str,
    mode_b: str,
    latent_net: nn.Module,
    sampler,
    latent_sampler,
    conf: Any,
    x_T_bank: torch.Tensor,
    latent_noise_bank: torch.Tensor,
    conds_mean: torch.Tensor,
    conds_std: torch.Tensor,
    T: int,
    clip_denoised: bool,
    chunk_batch: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Same-t pred_xstart L1/L2/cosine between two models (aligned progressive loops)."""
    _configure_model_quant_mode(model_a, mode_a)
    _configure_model_quant_mode(model_b, mode_b)
    same_t = [_MetricAccumulator() for _ in range(T)]
    num_images = x_T_bank.shape[0]
    num_chunks = num_images // chunk_batch
    cache_scheduler = getattr(conf, "cache_scheduler", None)
    for ci in range(num_chunks):
        b0, b1 = ci * chunk_batch, (ci + 1) * chunk_batch
        x_T_chunk = x_T_bank[b0:b1].to(device)
        latent_noise_chunk = latent_noise_bank[b0:b1].to(device)
        if conf.train_mode.is_latent_diffusion():
            cond_chunk = _compute_latent_cond(
                conf=conf,
                latent_sampler=latent_sampler,
                latent_net=latent_net,
                latent_noise_chunk=latent_noise_chunk,
                conds_mean=conds_mean,
                conds_std=conds_std,
                device=device,
            )
            model_kwargs = {"cond": cond_chunk}
        else:
            model_kwargs = None
        gen_a = sampler.ddim_sample_loop_progressive(
            model=model_a,
            noise=x_T_chunk,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
            device=device,
            progress=False,
            eta=0.0,
            cache_scheduler=cache_scheduler,
        )
        gen_b = sampler.ddim_sample_loop_progressive(
            model=model_b,
            noise=x_T_chunk,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
            device=device,
            progress=False,
            eta=0.0,
            cache_scheduler=cache_scheduler,
        )
        for out_a, out_b in zip(gen_a, gen_b):
            t_a = int(out_a["t"][0].item())
            t_b = int(out_b["t"][0].item())
            if t_a != t_b:
                raise RuntimeError(
                    f"timestep mismatch in progressive loop ({mode_a}/{mode_b}): {t_a} vs {t_b}"
                )
            same_t[t_a].update(out_a["pred_xstart"], out_b["pred_xstart"])
    by_t = {str(t): same_t[t].summary() for t in range(T)}
    return {"by_t": by_t, "meta": {"mode_a": mode_a, "mode_b": mode_b, "T": T}}


@torch.no_grad()
def _collect_distance_to_final(
    *,
    model: nn.Module,
    mode_tag: str,
    sampler,
    latent_sampler,
    conf: Any,
    x_T_bank: torch.Tensor,
    latent_noise_bank: torch.Tensor,
    conds_mean: torch.Tensor,
    conds_std: torch.Tensor,
    T: int,
    clip_denoised: bool,
    chunk_batch: int,
    device: torch.device,
) -> Dict[str, Any]:
    _configure_model_quant_mode(model, mode_tag)
    dist = [_MetricAccumulator() for _ in range(T)]
    num_images = x_T_bank.shape[0]
    num_chunks = num_images // chunk_batch
    cache_scheduler = getattr(conf, "cache_scheduler", None)
    for ci in range(num_chunks):
        b0, b1 = ci * chunk_batch, (ci + 1) * chunk_batch
        x_T_chunk = x_T_bank[b0:b1].to(device)
        latent_noise_chunk = latent_noise_bank[b0:b1].to(device)
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
            model_kwargs = {"cond": cond_chunk}
        else:
            model_kwargs = None
        per_t_pred: Dict[int, torch.Tensor] = {}
        final_pred: Optional[torch.Tensor] = None
        for out in sampler.ddim_sample_loop_progressive(
            model=model,
            noise=x_T_chunk,
            clip_denoised=clip_denoised,
            model_kwargs=model_kwargs,
            device=device,
            progress=False,
            eta=0.0,
            cache_scheduler=cache_scheduler,
        ):
            t_idx = int(out["t"][0].item())
            pred = out["pred_xstart"].detach()
            per_t_pred[t_idx] = pred
            if t_idx == 0:
                final_pred = pred
        if final_pred is None:
            LOGGER.warning(
                "mode=%s chunk=%d final timestep (t=0) missing; skip distance-to-final for this chunk",
                mode_tag,
                ci,
            )
            continue
        for t_idx, pred in per_t_pred.items():
            dist[t_idx].update(pred, final_pred)
    by_t = {str(t): dist[t].summary() for t in range(T)}
    return {"by_t": by_t}


def _save_stats_outputs(
    *,
    mode_dir: Path,
    stats: Dict[str, Any],
    T: int,
    mode: str,
    seed: int,
    num_images: int,
    hist_bins: int,
    hist_min: float,
    hist_max: float,
) -> None:
    mode_dir.mkdir(parents=True, exist_ok=True)
    stats_json = {
        "meta": {
            "mode": mode,
            "T": T,
            "seed": seed,
            "num_images": num_images,
            "hist_bins": hist_bins,
            "hist_min": hist_min,
            "hist_max": hist_max,
        },
        "by_t": stats["stats_by_t"],
        "counts_t": {str(i): int(v) for i, v in enumerate(stats["counts_t"].tolist())},
    }
    with open(mode_dir / "pred_xstart_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats_json, f, indent=2, ensure_ascii=False)
    np.savez_compressed(
        mode_dir / "pred_xstart_stats.npz",
        hist_counts=stats["hist_counts"],
        bin_edges=stats["bin_edges"],
        counts_t=stats["counts_t"],
        t=np.arange(T, dtype=np.int32),
    )
    # self trajectory delta
    by_t = stats["stats_by_t"]
    self_delta_json = {
        "meta": {
            **stats_json["meta"],
            **SELF_DELTA_BOUNDARY_META,
            "nan_explanation": "Timesteps with adjacent_count==0 have no valid adjacent comparison (boundary of the DDIM chain per chunk).",
        },
        "by_t": {
            str(t): {
                "l1": by_t[str(t)].get("adjacent_l1", np.nan),
                "l2": by_t[str(t)].get("adjacent_l2", np.nan),
                "cosine": by_t[str(t)].get("adjacent_cosine", np.nan),
                "adjacent_count": int(by_t[str(t)].get("adjacent_count", 0)),
            }
            for t in range(T)
            if str(t) in by_t
        },
    }
    with open(mode_dir / "trajectory_self_delta.json", "w", encoding="utf-8") as f:
        json.dump(self_delta_json, f, indent=2, ensure_ascii=False)
    np.savez_compressed(
        mode_dir / "trajectory_self_delta.npz",
        t=np.arange(T, dtype=np.int32),
        l1=np.array(
            [self_delta_json["by_t"].get(str(t), {}).get("l1", np.nan) for t in range(T)],
            dtype=np.float64,
        ),
        l2=np.array(
            [self_delta_json["by_t"].get(str(t), {}).get("l2", np.nan) for t in range(T)],
            dtype=np.float64,
        ),
        cosine=np.array(
            [self_delta_json["by_t"].get(str(t), {}).get("cosine", np.nan) for t in range(T)],
            dtype=np.float64,
        ),
        adjacent_count=np.array(
            [self_delta_json["by_t"].get(str(t), {}).get("adjacent_count", 0) for t in range(T)],
            dtype=np.int64,
        ),
    )


def _t_series_from_stats(stats_json: Dict[str, Any], key: str, T: int) -> np.ndarray:
    by_t = stats_json.get("by_t", {})
    return np.array(
        [float(by_t.get(str(t), {}).get(key, np.nan)) for t in range(T)], dtype=np.float64
    )


def _plot_overlay_two_curves(
    *,
    t: np.ndarray,
    y_a: np.ndarray,
    y_b: np.ndarray,
    label_a: str,
    label_b: str,
    ylabel: str,
    title: str,
    out_png: Path,
) -> None:
    plt.figure(figsize=(8.5, 4.2))
    plt.plot(t, y_a, label=label_a, linewidth=1.8)
    plt.plot(t, y_b, label=label_b, linewidth=1.8)
    plt.gca().invert_xaxis()
    plt.grid(True, alpha=0.25)
    plt.xlabel("DDIM timestep t (left=noisy, right=clear)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def _plot_overlay_curve(
    *,
    t: np.ndarray,
    ff: np.ndarray,
    ft: np.ndarray,
    ylabel: str,
    title: str,
    out_png: Path,
) -> None:
    _plot_overlay_two_curves(
        t=t, y_a=ff, y_b=ft, label_a="FF", label_b="FT", ylabel=ylabel, title=title, out_png=out_png
    )


def _plot_quantile_band_overlay_from_stats(
    *,
    ref_stats_json: Dict[str, Any],
    target_stats_json: Dict[str, Any],
    ref_label: str,
    target_label: str,
    T: int,
    out_png: Path,
) -> None:
    def _q_series(stats_json: Dict[str, Any], key: str) -> np.ndarray:
        by_t = stats_json.get("by_t", {})
        return np.array(
            [float(by_t.get(str(t), {}).get(key, np.nan)) for t in range(T)], dtype=np.float64
        )

    t = np.arange(T, dtype=np.int32)
    r01 = _q_series(ref_stats_json, "q01")
    r25 = _q_series(ref_stats_json, "q25")
    r50 = _q_series(ref_stats_json, "q50")
    r75 = _q_series(ref_stats_json, "q75")
    r99 = _q_series(ref_stats_json, "q99")
    x01 = _q_series(target_stats_json, "q01")
    x25 = _q_series(target_stats_json, "q25")
    x50 = _q_series(target_stats_json, "q50")
    x75 = _q_series(target_stats_json, "q75")
    x99 = _q_series(target_stats_json, "q99")

    plt.figure(figsize=(10, 5))
    plt.fill_between(t, r01, r99, color="tab:blue", alpha=0.15, label=f"{ref_label} q01-q99")
    plt.fill_between(t, r25, r75, color="tab:blue", alpha=0.25, label=f"{ref_label} q25-q75")
    plt.plot(t, r50, color="tab:blue", linewidth=2.0, label=f"{ref_label} q50")

    plt.fill_between(t, x01, x99, color="tab:orange", alpha=0.15, label=f"{target_label} q01-q99")
    plt.fill_between(t, x25, x75, color="tab:orange", alpha=0.25, label=f"{target_label} q25-q75")
    plt.plot(t, x50, color="tab:orange", linewidth=2.0, label=f"{target_label} q50")

    plt.xlim(0, max(0, T - 1))
    plt.gca().invert_xaxis()
    plt.grid(True, alpha=0.25)
    plt.xlabel("DDIM timestep t")
    plt.ylabel("Predicted x_start value")
    plt.title(f"Pred-xstart Quantile Bands: {ref_label} vs {target_label}")
    plt.legend(loc="best")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _mode_key_short(mode: str) -> str:
    u = mode.upper()
    if u == "BASELINE":
        return "baseline"
    return u.lower()


def _save_distance_to_final_json(mode_dir: Path, dist: Dict[str, Any], T: int, mode: str) -> None:
    mode_dir.mkdir(parents=True, exist_ok=True)
    payload = {"meta": {"mode": mode, "T": T}, "by_t": dist["by_t"]}
    with open(mode_dir / "distance_to_final.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    l1 = np.array(
        [dist["by_t"].get(str(t), {}).get("l1", np.nan) for t in range(T)], dtype=np.float64
    )
    l2 = np.array(
        [dist["by_t"].get(str(t), {}).get("l2", np.nan) for t in range(T)], dtype=np.float64
    )
    cos = np.array(
        [dist["by_t"].get(str(t), {}).get("cosine", np.nan) for t in range(T)], dtype=np.float64
    )
    np.savez_compressed(
        mode_dir / "distance_to_final.npz", t=np.arange(T, dtype=np.int32), l1=l1, l2=l2, cosine=cos
    )


def _divergence_stats_from_series(
    l1: np.ndarray, l2: np.ndarray, cos: np.ndarray, zones: Dict[str, np.ndarray]
) -> Dict[str, Any]:
    """Aggregate stats; ignores NaNs (boundary / missing)."""
    out: Dict[str, Any] = {
        "l1": {
            "mean_over_t": _nanmean_ignore(l1),
            "max_over_t": _nanmax_ignore(l1)[0],
            "argmax_timestep": _nanmax_ignore(l1)[1],
        },
        "l2": {
            "mean_over_t": _nanmean_ignore(l2),
            "max_over_t": _nanmax_ignore(l2)[0],
            "argmax_timestep": _nanmax_ignore(l2)[1],
        },
        "cosine": {
            "mean_over_t": _nanmean_ignore(cos),
            "min_over_t": _nanmin_ignore(cos)[0],
            "argmin_timestep": _nanmin_ignore(cos)[1],
        },
        "zones": {},
    }
    for zn, mask in zones.items():
        m = np.asarray(mask, dtype=bool)
        if m.size != l1.size:
            continue
        out["zones"][zn] = {
            "l1_mean": _nanmean_ignore(np.where(m, l1, np.nan)),
            "l2_mean": _nanmean_ignore(np.where(m, l2, np.nan)),
            "cosine_mean": _nanmean_ignore(np.where(m, cos, np.nan)),
        }
    return out


def _write_divergence_to_baseline_file(
    *,
    out_path: Path,
    cross: Dict[str, Any],
    T: int,
    target_mode: str,
) -> Dict[str, Any]:
    """cross = same-t delta between BASELINE and target M."""
    l1 = np.array(
        [cross["by_t"].get(str(t), {}).get("l1", np.nan) for t in range(T)], dtype=np.float64
    )
    l2 = np.array(
        [cross["by_t"].get(str(t), {}).get("l2", np.nan) for t in range(T)], dtype=np.float64
    )
    cos = np.array(
        [cross["by_t"].get(str(t), {}).get("cosine", np.nan) for t in range(T)], dtype=np.float64
    )
    zones = _timestep_zone_masks(T)
    summary = _divergence_stats_from_series(l1, l2, cos, zones)
    payload = {
        "meta": {
            "reference": "BASELINE",
            "target": target_mode,
            "T": T,
            "description": "Per-timestep ||pred_xstart_baseline - pred_xstart_target|| metrics (batch-mean L1/L2, cosine).",
        },
        "series": {
            "l1": l1.tolist(),
            "l2": l2.tolist(),
            "cosine": cos.tolist(),
            "t": list(range(T)),
        },
        "summary": summary,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def _series_from_stats_key(stats_json: Dict[str, Any], key: str, T: int) -> np.ndarray:
    return _t_series_from_stats(stats_json, key, T)


def _high_noise_zone_metrics(stats_json: Dict[str, Any], T: int) -> Dict[str, float]:
    """Mean over high-noise timesteps of selected scalar stats."""
    zones = _timestep_zone_masks(T)
    hi = zones["high_noise"]
    keys = [
        "q50",
        "std",
        "abs_max",
        "q99",
        "saturation_ratio_abs_ge_099",
        "positive_ratio",
        "negative_ratio",
    ]
    out: Dict[str, float] = {}
    for k in keys:
        s = _series_from_stats_key(stats_json, k, T)
        out[f"mean_{k}_high_noise_zone"] = float(_nanmean_ignore(np.where(hi, s, np.nan)))
    return out


def _pairwise_high_noise_diff(
    stats_a: Dict[str, Any], stats_b: Dict[str, Any], T: int, pair_name: str
) -> Dict[str, Any]:
    ma = _high_noise_zone_metrics(stats_a, T)
    mb = _high_noise_zone_metrics(stats_b, T)
    out: Dict[str, Any] = {"pair": pair_name}
    key_map = {
        "mean_q50_high_noise_zone": "mean_diff_q50_high_noise",
        "mean_std_high_noise_zone": "mean_diff_std_high_noise",
        "mean_abs_max_high_noise_zone": "mean_diff_abs_max_high_noise",
        "mean_q99_high_noise_zone": "mean_diff_q99_high_noise",
        "mean_saturation_ratio_abs_ge_099_high_noise_zone": "mean_diff_sat099_high_noise",
        "mean_positive_ratio_high_noise_zone": "mean_diff_pos_ratio_high_noise",
        "mean_negative_ratio_high_noise_zone": "mean_diff_neg_ratio_high_noise",
    }
    for src, dst in key_map.items():
        if src in ma and src in mb:
            out[dst] = float(mb[src] - ma[src])
    return out


def _self_delta_series_from_stats(
    stats_json: Dict[str, Any], T: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    l1 = _series_from_stats_key(stats_json, "adjacent_l1", T)
    l2 = _series_from_stats_key(stats_json, "adjacent_l2", T)
    cos = _series_from_stats_key(stats_json, "adjacent_cosine", T)
    return l1, l2, cos


def _trajectory_self_summary(
    l1: np.ndarray, l2: np.ndarray, cos: np.ndarray, T: int
) -> Dict[str, Any]:
    zones = _timestep_zone_masks(T)
    hi = zones["high_noise"]
    return {
        "self_trajectory_delta": {
            "mean_L1_all_t": _nanmean_ignore(l1),
            "mean_L2_all_t": _nanmean_ignore(l2),
            "mean_cosine_all_t": _nanmean_ignore(cos),
            "mean_L1_high_noise": _nanmean_ignore(np.where(hi, l1, np.nan)),
            "mean_L2_high_noise": _nanmean_ignore(np.where(hi, l2, np.nan)),
            "mean_cosine_high_noise": _nanmean_ignore(np.where(hi, cos, np.nan)),
            "max_L1": _nanmax_ignore(l1)[0],
            "argmax_L1_timestep": _nanmax_ignore(l1)[1],
            "max_L2": _nanmax_ignore(l2)[0],
            "argmax_L2_timestep": _nanmax_ignore(l2)[1],
            "min_cosine": _nanmin_ignore(cos)[0],
            "argmin_cosine_timestep": _nanmin_ignore(cos)[1],
        }
    }


def _distance_to_final_summary_from_by_t(by_t: Dict[str, Any], T: int) -> Dict[str, Any]:
    l1 = np.array([by_t.get(str(t), {}).get("l1", np.nan) for t in range(T)], dtype=np.float64)
    l2 = np.array([by_t.get(str(t), {}).get("l2", np.nan) for t in range(T)], dtype=np.float64)
    cos = np.array([by_t.get(str(t), {}).get("cosine", np.nan) for t in range(T)], dtype=np.float64)
    zones = _timestep_zone_masks(T)
    return {
        "distance_to_final": {
            "mean_L1_all_t": _nanmean_ignore(l1),
            "mean_L2_all_t": _nanmean_ignore(l2),
            "mean_cosine_all_t": _nanmean_ignore(cos),
            "mean_L1_high_noise": _nanmean_ignore(np.where(zones["high_noise"], l1, np.nan)),
            "mean_L1_middle": _nanmean_ignore(np.where(zones["middle_noise"], l1, np.nan)),
            "mean_L1_low_noise": _nanmean_ignore(np.where(zones["low_noise"], l1, np.nan)),
            "mean_L2_high_noise": _nanmean_ignore(np.where(zones["high_noise"], l2, np.nan)),
            "mean_L2_middle": _nanmean_ignore(np.where(zones["middle_noise"], l2, np.nan)),
            "mean_L2_low_noise": _nanmean_ignore(np.where(zones["low_noise"], l2, np.nan)),
            "mean_cosine_high_noise": _nanmean_ignore(np.where(zones["high_noise"], cos, np.nan)),
            "mean_cosine_middle": _nanmean_ignore(np.where(zones["middle_noise"], cos, np.nan)),
            "mean_cosine_low_noise": _nanmean_ignore(np.where(zones["low_noise"], cos, np.nan)),
        }
    }


def _write_model_trajectory_summary_json(
    *,
    mode_dir: Path,
    stats_json: Dict[str, Any],
    dist_by_t: Optional[Dict[str, Any]],
    T: int,
    mode: str,
) -> Dict[str, Any]:
    l1, l2, cos = _self_delta_series_from_stats(stats_json, T)
    payload: Dict[str, Any] = {
        "meta": {"mode": mode, "T": T, **SELF_DELTA_BOUNDARY_META},
        **_trajectory_self_summary(l1, l2, cos, T),
    }
    if dist_by_t is not None:
        payload.update(_distance_to_final_summary_from_by_t(dist_by_t, T))
    mode_dir.mkdir(parents=True, exist_ok=True)
    with open(mode_dir / "trajectory_summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def _plot_baseline_divergence_overlays(
    *,
    div_payloads: Dict[str, Dict[str, Any]],
    plots_summary: Path,
    T: int,
) -> None:
    """div_payloads keys: ff, ft, tt — each has series l1, cosine."""
    t = np.arange(T)
    plots_summary.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8.5, 4.2))
    for label, key in [("FF", "ff"), ("FT", "ft"), ("TT", "tt")]:
        if key in div_payloads:
            s = div_payloads[key]["series"]["l1"]
            plt.plot(
                t, np.asarray(s, dtype=np.float64), label=f"BASELINE vs {label}", linewidth=1.8
            )
    plt.gca().invert_xaxis()
    plt.grid(True, alpha=0.25)
    plt.xlabel("DDIM timestep t (left=noisy, right=clear)")
    plt.ylabel("L1")
    plt.title("Divergence to baseline (L1)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(plots_summary / "baseline_divergence_l1_overlay.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.5, 4.2))
    for label, key in [("FF", "ff"), ("FT", "ft"), ("TT", "tt")]:
        if key in div_payloads:
            s = div_payloads[key]["series"]["cosine"]
            plt.plot(
                t, np.asarray(s, dtype=np.float64), label=f"BASELINE vs {label}", linewidth=1.8
            )
    plt.gca().invert_xaxis()
    plt.grid(True, alpha=0.25)
    plt.xlabel("DDIM timestep t (left=noisy, right=clear)")
    plt.ylabel("cosine")
    plt.title("Divergence to baseline (cosine)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(plots_summary / "baseline_divergence_cos_overlay.png", dpi=180)
    plt.close()


def _plot_high_noise_model_comparison(
    *,
    stats_by_mode: Dict[str, Dict[str, Any]],
    plots_summary: Path,
    T: int,
) -> None:
    """One compact 2x2 of std / q99 / sat099 / pos_ratio for BASELINE, FF, FT, TT when present."""
    t = np.arange(T)
    modes_order = ["BASELINE", "FF", "FT", "TT"]
    present = [m for m in modes_order if m in stats_by_mode]
    if not present:
        return
    plots_summary.mkdir(parents=True, exist_ok=True)

    def series(m: str, k: str) -> np.ndarray:
        """Public function series."""
        return _series_from_stats_key(stats_by_mode[m], k, T)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True)
    pairs = [
        (axes[0, 0], "std", "std"),
        (axes[0, 1], "q99", "q99"),
        (axes[1, 0], "saturation_ratio_abs_ge_099", "sat(|x|>=0.99)"),
        (axes[1, 1], "positive_ratio", "positive ratio"),
    ]
    for ax, key, ylab in pairs:
        for m in present:
            ax.plot(t, series(m, key), label=m, linewidth=1.6)
        ax.invert_xaxis()
        ax.grid(True, alpha=0.25)
        ax.set_ylabel(ylab)
        ax.legend(loc="best", fontsize=8)
    for ax in axes[-1, :]:
        ax.set_xlabel("DDIM timestep t (left=noisy, right=clear)")
    fig.suptitle("High-noise-relevant pred_xstart stats (full trajectory)")
    plt.tight_layout()
    plt.savefig(plots_summary / "high_noise_metrics_models_overlay.png", dpi=180)
    plt.close()


def _plot_trajectory_regularization_overlays(
    *,
    stats_by_mode: Dict[str, Dict[str, Any]],
    dist_by_mode: Dict[str, Optional[Dict[str, Any]]],
    plots_summary: Path,
    T: int,
) -> None:
    t = np.arange(T)
    plots_summary.mkdir(parents=True, exist_ok=True)
    order = ["BASELINE", "FF", "FT", "TT"]
    present = [m for m in order if m in stats_by_mode]
    plt.figure(figsize=(8.5, 4.2))
    for m in present:
        l1, _, _ = _self_delta_series_from_stats(stats_by_mode[m], T)
        plt.plot(t, l1, label=m, linewidth=1.6)
    plt.gca().invert_xaxis()
    plt.grid(True, alpha=0.25)
    plt.xlabel("DDIM timestep t (left=noisy, right=clear)")
    plt.ylabel("L1")
    plt.title("Self trajectory delta (adjacent L1)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(plots_summary / "trajectory_self_delta_l1_overlay.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8.5, 4.2))
    for m in present:
        d = dist_by_mode.get(m)
        if d is None:
            continue
        by_t = d["by_t"]
        l1 = np.array(
            [by_t.get(str(ti), {}).get("l1", np.nan) for ti in range(T)], dtype=np.float64
        )
        plt.plot(t, l1, label=m, linewidth=1.6)
    plt.gca().invert_xaxis()
    plt.grid(True, alpha=0.25)
    plt.xlabel("DDIM timestep t (left=noisy, right=clear)")
    plt.ylabel("L1 to final pred_xstart")
    plt.title("Distance to final (L1)")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(plots_summary / "trajectory_distance_to_final_l1_overlay.png", dpi=180)
    plt.close()


def _write_analysis_summary_md(
    *,
    out_path: Path,
    T: int,
    baseline_div: Dict[str, Dict[str, Any]],
    high_noise_path: Path,
    traj_path: Path,
) -> None:
    lines = [
        "# Pred-xstart trajectory analysis — auto draft",
        "",
        "此檔為依統計自動產生的**保守**摘要，不宣稱訓練或量化之因果機制。",
        "",
        "## 1. 本次分析驗證了什麼",
        "",
        f"- 在 T={T} 的 DDIM progressive 軌跡上，對 pred_xstart 做 per-timestep 分佈與跨模型對齊度量。",
        "- 若已產生 `divergence_to_baseline.json`，可讀取量化模型與 BASELINE 在同一 noise level 的張量差異（L1/L2/cosine）。",
        "- 若已產生 `high_noise_regime_summary.json` 與 `trajectory_regularization_summary.json`，可比較高噪聲區段的標的統計與軌跡平滑度/距離終點。",
        "",
        "## 2. 目前可支持的中間結論（僅陳述觀測）",
        "",
        "- 請以數值為準：比較各 `summary` JSON 內 mean / zone mean，而非僅看圖。",
        "- 若某模型在 high-noise zone 的 `saturation_ratio_abs_ge_099` 或 `std` 明顯較高，代表該區段 pred_xstart 分佈較分散或較常觸及飽和；**這不自動等於**生成品質較差。",
        "",
        "## 3. 目前仍不能直接下的結論",
        "",
        "- 「為何量化後比原作更好」涉及感知指標、資料與訓練目標；本管線**不**量測 FID/LPIPS 或重建誤差。",
        "- pred_xstart 統計差異**不**等同於 latent 或權重空間的因果解釋。",
        "",
        "## 4. 建議的下一步",
        "",
        "- 將此處 high-noise / divergence 指標與實際樣本視覺與下游指標對照。",
        f"- 檢查：`{high_noise_path.name}`、`{traj_path.name}`。",
        "",
        "## 附註：self trajectory delta 的 NaN",
        "",
        f"- {SELF_DELTA_BOUNDARY_META['boundary_nan_policy']}；summary 已忽略合法 NaN。",
        "",
    ]
    if baseline_div:
        lines.append("### Baseline divergence 摘要（mean L1 over t）")
        lines.append("")
        for k, pl in baseline_div.items():
            s = pl.get("summary", {}).get("l1", {}).get("mean_over_t", float("nan"))
            lines.append(f"- {k}: {s}")
        lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")


def _save_pairwise_compare_outputs(
    *,
    compare_dir: Path,
    plots_pairwise: Path,
    name_a: str,
    name_b: str,
    stats_a: Dict[str, Any],
    stats_b: Dict[str, Any],
    cross_same_t: Dict[str, Any],
    dist_a_final: Optional[Dict[str, Any]],
    dist_b_final: Optional[Dict[str, Any]],
    T: int,
    skip_plots: bool,
    plot_file_prefix: str,
) -> None:
    """Generic FF/FT-style compare; diff_* = second model minus first (name_b - name_a)."""
    compare_dir.mkdir(parents=True, exist_ok=True)
    plots_pairwise.mkdir(parents=True, exist_ok=True)
    ka = _mode_key_short(name_a)
    kb = _mode_key_short(name_b)

    def q50(s: Dict[str, Any]) -> np.ndarray:
        """Public function q50."""
        return _t_series_from_stats(s, "q50", T)

    def std(s: Dict[str, Any]) -> np.ndarray:
        """Public function std."""
        return _t_series_from_stats(s, "std", T)

    def absmax(s: Dict[str, Any]) -> np.ndarray:
        """Public function absmax."""
        return _t_series_from_stats(s, "abs_max", T)

    def q99(s: Dict[str, Any]) -> np.ndarray:
        """Public function q99."""
        return _t_series_from_stats(s, "q99", T)

    a_q50, b_q50 = q50(stats_a), q50(stats_b)
    a_std, b_std = std(stats_a), std(stats_b)
    a_absmax, b_absmax = absmax(stats_a), absmax(stats_b)
    a_q99, b_q99 = q99(stats_a), q99(stats_b)
    same_t_l1 = np.array(
        [cross_same_t["by_t"].get(str(t), {}).get("l1", np.nan) for t in range(T)], dtype=np.float64
    )
    same_t_l2 = np.array(
        [cross_same_t["by_t"].get(str(t), {}).get("l2", np.nan) for t in range(T)], dtype=np.float64
    )
    same_t_cos = np.array(
        [cross_same_t["by_t"].get(str(t), {}).get("cosine", np.nan) for t in range(T)],
        dtype=np.float64,
    )
    a_self_l1 = _t_series_from_stats(stats_a, "adjacent_l1", T)
    b_self_l1 = _t_series_from_stats(stats_b, "adjacent_l1", T)
    a_self_l2 = _t_series_from_stats(stats_a, "adjacent_l2", T)
    b_self_l2 = _t_series_from_stats(stats_b, "adjacent_l2", T)
    a_self_cos = _t_series_from_stats(stats_a, "adjacent_cosine", T)
    b_self_cos = _t_series_from_stats(stats_b, "adjacent_cosine", T)
    cmp_json = {
        "meta": {"T": T, "mode_a": name_a, "mode_b": name_b, "diff_definition": f"{kb}_minus_{ka}"},
        "series": {
            f"{ka}_q50": a_q50.tolist(),
            f"{kb}_q50": b_q50.tolist(),
            f"{ka}_std": a_std.tolist(),
            f"{kb}_std": b_std.tolist(),
            f"{ka}_abs_max": a_absmax.tolist(),
            f"{kb}_abs_max": b_absmax.tolist(),
            f"{ka}_q99": a_q99.tolist(),
            f"{kb}_q99": b_q99.tolist(),
            "diff_q50": (b_q50 - a_q50).tolist(),
            "diff_std": (b_std - a_std).tolist(),
            "diff_abs_max": (b_absmax - a_absmax).tolist(),
            "diff_q99": (b_q99 - a_q99).tolist(),
            "same_t_l1": same_t_l1.tolist(),
            "same_t_l2": same_t_l2.tolist(),
            "same_t_cosine": same_t_cos.tolist(),
            f"{ka}_self_delta_l1": a_self_l1.tolist(),
            f"{kb}_self_delta_l1": b_self_l1.tolist(),
            f"{ka}_self_delta_l2": a_self_l2.tolist(),
            f"{kb}_self_delta_l2": b_self_l2.tolist(),
            f"{ka}_self_delta_cosine": a_self_cos.tolist(),
            f"{kb}_self_delta_cosine": b_self_cos.tolist(),
        },
    }
    with open(compare_dir / "pred_xstart_compare.json", "w", encoding="utf-8") as f:
        json.dump(cmp_json, f, indent=2, ensure_ascii=False)
    np.savez_compressed(
        compare_dir / "pred_xstart_compare.npz",
        **{k: np.array(v) for k, v in cmp_json["series"].items()},
        t=np.arange(T),
    )
    with open(compare_dir / "cross_model_same_t_delta.json", "w", encoding="utf-8") as f:
        json.dump({"meta": {"T": T}, "by_t": cross_same_t["by_t"]}, f, indent=2, ensure_ascii=False)
    np.savez_compressed(
        compare_dir / "cross_model_same_t_delta.npz",
        t=np.arange(T),
        l1=same_t_l1,
        l2=same_t_l2,
        cosine=same_t_cos,
    )
    if dist_a_final is not None and dist_b_final is not None:
        a_fd_l1 = np.array(
            [dist_a_final["by_t"].get(str(t), {}).get("l1", np.nan) for t in range(T)],
            dtype=np.float64,
        )
        a_fd_l2 = np.array(
            [dist_a_final["by_t"].get(str(t), {}).get("l2", np.nan) for t in range(T)],
            dtype=np.float64,
        )
        a_fd_cos = np.array(
            [dist_a_final["by_t"].get(str(t), {}).get("cosine", np.nan) for t in range(T)],
            dtype=np.float64,
        )
        b_fd_l1 = np.array(
            [dist_b_final["by_t"].get(str(t), {}).get("l1", np.nan) for t in range(T)],
            dtype=np.float64,
        )
        b_fd_l2 = np.array(
            [dist_b_final["by_t"].get(str(t), {}).get("l2", np.nan) for t in range(T)],
            dtype=np.float64,
        )
        b_fd_cos = np.array(
            [dist_b_final["by_t"].get(str(t), {}).get("cosine", np.nan) for t in range(T)],
            dtype=np.float64,
        )
        with open(compare_dir / "distance_to_final_compare.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "meta": {"T": T, "mode_a": name_a, "mode_b": name_b},
                    "series": {
                        f"{ka}_l1": a_fd_l1.tolist(),
                        f"{ka}_l2": a_fd_l2.tolist(),
                        f"{ka}_cosine": a_fd_cos.tolist(),
                        f"{kb}_l1": b_fd_l1.tolist(),
                        f"{kb}_l2": b_fd_l2.tolist(),
                        f"{kb}_cosine": b_fd_cos.tolist(),
                        "diff_l1": (b_fd_l1 - a_fd_l1).tolist(),
                        "diff_l2": (b_fd_l2 - a_fd_l2).tolist(),
                        "diff_cosine": (b_fd_cos - a_fd_cos).tolist(),
                    },
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        np.savez_compressed(
            compare_dir / "distance_to_final_compare.npz",
            t=np.arange(T),
            **{
                f"{ka}_l1": a_fd_l1,
                f"{ka}_l2": a_fd_l2,
                f"{ka}_cosine": a_fd_cos,
                f"{kb}_l1": b_fd_l1,
                f"{kb}_l2": b_fd_l2,
                f"{kb}_cosine": b_fd_cos,
            },
        )
    if skip_plots:
        return
    t = np.arange(T)
    pf = plot_file_prefix
    _plot_overlay_two_curves(
        t=t,
        y_a=a_q50,
        y_b=b_q50,
        label_a=name_a,
        label_b=name_b,
        ylabel="q50",
        title=f"{pf} pred_xstart q50",
        out_png=plots_pairwise / f"{pf}_pred_xstart_q50_overlay.png",
    )
    _plot_overlay_two_curves(
        t=t,
        y_a=a_std,
        y_b=b_std,
        label_a=name_a,
        label_b=name_b,
        ylabel="std",
        title=f"{pf} pred_xstart std",
        out_png=plots_pairwise / f"{pf}_pred_xstart_std_overlay.png",
    )
    _plot_overlay_two_curves(
        t=t,
        y_a=a_absmax,
        y_b=b_absmax,
        label_a=name_a,
        label_b=name_b,
        ylabel="abs_max",
        title=f"{pf} pred_xstart abs_max",
        out_png=plots_pairwise / f"{pf}_pred_xstart_absmax_overlay.png",
    )
    _plot_overlay_two_curves(
        t=t,
        y_a=a_q99,
        y_b=b_q99,
        label_a=name_a,
        label_b=name_b,
        ylabel="q99",
        title=f"{pf} pred_xstart q99",
        out_png=plots_pairwise / f"{pf}_pred_xstart_q99_overlay.png",
    )
    plt.figure(figsize=(8.5, 4.2))
    plt.plot(t, same_t_l1, linewidth=1.8)
    plt.gca().invert_xaxis()
    plt.grid(True, alpha=0.25)
    plt.xlabel("DDIM timestep t (left=noisy, right=clear)")
    plt.ylabel("L1")
    plt.title(f"{pf} same-t L1")
    plt.tight_layout()
    plt.savefig(plots_pairwise / f"{pf}_same_t_l1_curve.png", dpi=180)
    plt.close()
    plt.figure(figsize=(8.5, 4.2))
    plt.plot(t, same_t_l2, linewidth=1.8)
    plt.gca().invert_xaxis()
    plt.grid(True, alpha=0.25)
    plt.xlabel("DDIM timestep t (left=noisy, right=clear)")
    plt.ylabel("L2")
    plt.title(f"{pf} same-t L2")
    plt.tight_layout()
    plt.savefig(plots_pairwise / f"{pf}_same_t_l2_curve.png", dpi=180)
    plt.close()
    plt.figure(figsize=(8.5, 4.2))
    plt.plot(t, same_t_cos, linewidth=1.8)
    plt.gca().invert_xaxis()
    plt.grid(True, alpha=0.25)
    plt.xlabel("DDIM timestep t (left=noisy, right=clear)")
    plt.ylabel("cosine")
    plt.title(f"{pf} same-t cosine")
    plt.tight_layout()
    plt.savefig(plots_pairwise / f"{pf}_same_t_cos_curve.png", dpi=180)
    plt.close()
    _plot_overlay_two_curves(
        t=t,
        y_a=a_self_l1,
        y_b=b_self_l1,
        label_a=name_a,
        label_b=name_b,
        ylabel="L1",
        title=f"{pf} self trajectory adjacent L1",
        out_png=plots_pairwise / f"{pf}_self_delta_l1_compare.png",
    )
    _plot_overlay_two_curves(
        t=t,
        y_a=a_self_l2,
        y_b=b_self_l2,
        label_a=name_a,
        label_b=name_b,
        ylabel="L2",
        title=f"{pf} self trajectory adjacent L2",
        out_png=plots_pairwise / f"{pf}_self_delta_l2_compare.png",
    )
    _plot_overlay_two_curves(
        t=t,
        y_a=a_self_cos,
        y_b=b_self_cos,
        label_a=name_a,
        label_b=name_b,
        ylabel="cosine",
        title=f"{pf} self trajectory adjacent cosine",
        out_png=plots_pairwise / f"{pf}_self_delta_cos_compare.png",
    )
    dtf = compare_dir / "distance_to_final_compare.npz"
    if dtf.exists():
        fd = np.load(dtf)
        _plot_overlay_two_curves(
            t=t,
            y_a=fd[f"{ka}_l1"],
            y_b=fd[f"{kb}_l1"],
            label_a=name_a,
            label_b=name_b,
            ylabel="L1",
            title=f"{pf} distance-to-final L1",
            out_png=plots_pairwise / f"{pf}_final_distance_l1_compare.png",
        )
        _plot_overlay_two_curves(
            t=t,
            y_a=fd[f"{ka}_l2"],
            y_b=fd[f"{kb}_l2"],
            label_a=name_a,
            label_b=name_b,
            ylabel="L2",
            title=f"{pf} distance-to-final L2",
            out_png=plots_pairwise / f"{pf}_final_distance_l2_compare.png",
        )
        _plot_overlay_two_curves(
            t=t,
            y_a=fd[f"{ka}_cosine"],
            y_b=fd[f"{kb}_cosine"],
            label_a=name_a,
            label_b=name_b,
            ylabel="cosine",
            title=f"{pf} distance-to-final cosine",
            out_png=plots_pairwise / f"{pf}_final_distance_cos_compare.png",
        )


def _save_compare_outputs(
    *,
    compare_dir: Path,
    ff_stats_json: Dict[str, Any],
    ft_stats_json: Dict[str, Any],
    cross_same_t: Dict[str, Any],
    dist_ff_final: Optional[Dict[str, Any]],
    dist_ft_final: Optional[Dict[str, Any]],
    T: int,
    skip_plots: bool,
    plots_pairwise: Optional[Path] = None,
) -> None:
    """Backward-compatible wrapper: FF vs FT pairwise compare."""
    pw = plots_pairwise if plots_pairwise is not None else compare_dir / "plots"
    _save_pairwise_compare_outputs(
        compare_dir=compare_dir,
        plots_pairwise=pw,
        name_a="FF",
        name_b="FT",
        stats_a=ff_stats_json,
        stats_b=ft_stats_json,
        cross_same_t=cross_same_t,
        dist_a_final=dist_ff_final,
        dist_b_final=dist_ft_final,
        T=T,
        skip_plots=skip_plots,
        plot_file_prefix="ff_vs_ft",
    )


@time_operation
def run_pred_xstart_trajectory_analysis(
    *,
    images_mode: str,
    chunk_batch: int,
    hist_bins: int,
    hist_min: float,
    hist_max: float,
    seed: int,
    pred_output_root: str,
    skip_plots: bool,
    num_steps: int,
    run_ff: bool,
    run_ft: bool,
    run_tt: bool,
    run_baseline: bool,
    run_compare: bool,
    enable_distance_to_final: bool,
    save_quantile_band_overlay: bool,
    overlay_reference_mode: str,
    overlay_target_modes: str,
) -> Dict[str, Any]:
    """Public function run_pred_xstart_trajectory_analysis."""
    _seed_all(seed)
    device = CONFIG.DEVICE
    T = int(num_steps)
    num_images = _images_mode_to_num_images(images_mode)
    LOGGER.info(
        "[pred_xstart_trajectory] images_mode=%s num_images=%d T=%d chunk_batch=%d bins=%d seed=%d run_ff=%s run_ft=%s run_compare=%s",
        images_mode,
        num_images,
        T,
        chunk_batch,
        hist_bins,
        seed,
        run_ff,
        run_ft,
        run_compare,
    )
    if num_images % chunk_batch != 0:
        raise ValueError(
            f"num_images({num_images}) must be divisible by chunk_batch({chunk_batch})"
        )
    ckpt_path = _resolve_best_ckpt_path(T)
    # baseline model (original Diff-AE path)
    base_model_baseline: Optional[LitModel] = None
    model_baseline: Optional[nn.Module] = None
    if run_baseline:
        base_model_baseline = load_diffae_model(CONFIG.MODEL_PATH)
    base_model_baseline.to(device)
    base_model_baseline.eval()
    base_model_baseline.setup()
    base_model_baseline.train_dataloader()
    model_baseline = base_model_baseline.ema_model

    base_model_ff, model_ff = _build_eval_model_with_w_plus_lora(
        model_path=CONFIG.MODEL_PATH, ckpt_path=ckpt_path, num_steps=T, device=device
    )
    # build second model for strict same-t compare
    base_model_ft, model_ft = _build_eval_model_with_w_plus_lora(
        model_path=CONFIG.MODEL_PATH, ckpt_path=ckpt_path, num_steps=T, device=device
    )
    conf = base_model_ff.conf.clone()
    sampler = conf._make_diffusion_conf(T=T).make_sampler()
    latent_sampler = conf._make_latent_diffusion_conf(T=T).make_sampler()
    x_T_bank, latent_noise_bank = _make_noise_banks(
        num_images=num_images,
        chunk_batch=chunk_batch,
        img_size=conf.img_size,
        style_ch=conf.style_ch,
        seed=seed,
        device=device,
    )
    out_root = Path(pred_output_root) / f"T_{T}" / images_mode / f"seed{seed}"
    paths = _layout_run_paths(out_root)
    models_root = paths["models"]
    comparisons_root = paths["comparisons"]
    plots_summary = paths["plots_summary"]
    plots_pairwise = paths["plots_pairwise"]
    for _p in (models_root, comparisons_root, plots_summary, plots_pairwise):
        _p.mkdir(parents=True, exist_ok=True)

    ff_dir = models_root / "FF"
    ft_dir = models_root / "FT"
    tt_dir = models_root / "TT"
    baseline_dir = models_root / "BASELINE"
    cmp_dir_ff_ft = comparisons_root / "ff_vs_ft"

    model_ft_clone: Optional[nn.Module] = None
    if run_compare and run_ft and run_tt:
        model_ft_clone = copy.deepcopy(model_ft)

    mode_stats_cache: Dict[str, Dict[str, Any]] = {}
    ff_stats_json: Optional[Dict[str, Any]] = None
    ft_stats_json: Optional[Dict[str, Any]] = None
    dist_by_mode: Dict[str, Optional[Dict[str, Any]]] = {
        "BASELINE": None,
        "FF": None,
        "FT": None,
        "TT": None,
    }
    if run_ff:
        ff_stats = _collect_pred_xstart_streaming_stats(
            model=model_ff,
            mode_tag="FF",
            sampler=sampler,
            latent_sampler=latent_sampler,
            conf=conf,
            x_T_bank=x_T_bank,
            latent_noise_bank=latent_noise_bank,
            conds_mean=base_model_ff.conds_mean,
            conds_std=base_model_ff.conds_std,
            T=T,
            clip_denoised=True,
            hist_min=hist_min,
            hist_max=hist_max,
            hist_bins=hist_bins,
            chunk_batch=chunk_batch,
            device=device,
        )
        _save_stats_outputs(
            mode_dir=ff_dir,
            stats=ff_stats,
            T=T,
            mode="FF",
            seed=seed,
            num_images=num_images,
            hist_bins=hist_bins,
            hist_min=hist_min,
            hist_max=hist_max,
        )
        ff_stats_json = {
            "meta": {"mode": "FF", "T": T, "seed": seed, "num_images": num_images},
            "by_t": ff_stats["stats_by_t"],
        }
        mode_stats_cache["FF"] = ff_stats_json
        ff_missing = [t for t in range(T) if str(t) not in ff_stats["stats_by_t"]]
        LOGGER.info(
            "FF done: non-empty timesteps=%d missing=%d sample_values_per_t[min=%d max=%d]",
            len(ff_stats["stats_by_t"]),
            len(ff_missing),
            int(np.min(ff_stats["counts_t"])) if ff_stats["counts_t"].size else 0,
            int(np.max(ff_stats["counts_t"])) if ff_stats["counts_t"].size else 0,
        )
    if run_ft:
        ft_stats = _collect_pred_xstart_streaming_stats(
            model=model_ft,
            mode_tag="FT",
            sampler=sampler,
            latent_sampler=latent_sampler,
            conf=conf,
            x_T_bank=x_T_bank,
            latent_noise_bank=latent_noise_bank,
            conds_mean=base_model_ft.conds_mean,
            conds_std=base_model_ft.conds_std,
            T=T,
            clip_denoised=True,
            hist_min=hist_min,
            hist_max=hist_max,
            hist_bins=hist_bins,
            chunk_batch=chunk_batch,
            device=device,
        )
        _save_stats_outputs(
            mode_dir=ft_dir,
            stats=ft_stats,
            T=T,
            mode="FT",
            seed=seed,
            num_images=num_images,
            hist_bins=hist_bins,
            hist_min=hist_min,
            hist_max=hist_max,
        )
        ft_stats_json = {
            "meta": {"mode": "FT", "T": T, "seed": seed, "num_images": num_images},
            "by_t": ft_stats["stats_by_t"],
        }
        mode_stats_cache["FT"] = ft_stats_json
        ft_missing = [t for t in range(T) if str(t) not in ft_stats["stats_by_t"]]
        LOGGER.info(
            "FT done: non-empty timesteps=%d missing=%d sample_values_per_t[min=%d max=%d]",
            len(ft_stats["stats_by_t"]),
            len(ft_missing),
            int(np.min(ft_stats["counts_t"])) if ft_stats["counts_t"].size else 0,
            int(np.max(ft_stats["counts_t"])) if ft_stats["counts_t"].size else 0,
        )
    if run_tt:
        tt_stats = _collect_pred_xstart_streaming_stats(
            model=model_ft,
            mode_tag="TT",
            sampler=sampler,
            latent_sampler=latent_sampler,
            conf=conf,
            x_T_bank=x_T_bank,
            latent_noise_bank=latent_noise_bank,
            conds_mean=base_model_ft.conds_mean,
            conds_std=base_model_ft.conds_std,
            T=T,
            clip_denoised=True,
            hist_min=hist_min,
            hist_max=hist_max,
            hist_bins=hist_bins,
            chunk_batch=chunk_batch,
            device=device,
        )
        _save_stats_outputs(
            mode_dir=tt_dir,
            stats=tt_stats,
            T=T,
            mode="TT",
            seed=seed,
            num_images=num_images,
            hist_bins=hist_bins,
            hist_min=hist_min,
            hist_max=hist_max,
        )
        mode_stats_cache["TT"] = {
            "meta": {"mode": "TT", "T": T, "seed": seed, "num_images": num_images},
            "by_t": tt_stats["stats_by_t"],
        }
        LOGGER.info("TT done: non-empty timesteps=%d", len(tt_stats["stats_by_t"]))
    if run_baseline and model_baseline is not None and base_model_baseline is not None:
        b_stats = _collect_pred_xstart_streaming_stats(
            model=model_baseline,
            mode_tag="BASELINE",
            sampler=sampler,
            latent_sampler=latent_sampler,
            conf=conf,
            x_T_bank=x_T_bank,
            latent_noise_bank=latent_noise_bank,
            conds_mean=base_model_baseline.conds_mean,
            conds_std=base_model_baseline.conds_std,
            T=T,
            clip_denoised=True,
            hist_min=hist_min,
            hist_max=hist_max,
            hist_bins=hist_bins,
            chunk_batch=chunk_batch,
            device=device,
        )
        _save_stats_outputs(
            mode_dir=baseline_dir,
            stats=b_stats,
            T=T,
            mode="BASELINE",
            seed=seed,
            num_images=num_images,
            hist_bins=hist_bins,
            hist_min=hist_min,
            hist_max=hist_max,
        )
        mode_stats_cache["BASELINE"] = {
            "meta": {"mode": "BASELINE", "T": T, "seed": seed, "num_images": num_images},
            "by_t": b_stats["stats_by_t"],
        }
        LOGGER.info("BASELINE done: non-empty timesteps=%d", len(b_stats["stats_by_t"]))

    if enable_distance_to_final:
        if run_baseline and model_baseline is not None and base_model_baseline is not None:
            dist_by_mode["BASELINE"] = _collect_distance_to_final(
                model=model_baseline,
                mode_tag="BASELINE",
                sampler=sampler,
                latent_sampler=latent_sampler,
                conf=conf,
                x_T_bank=x_T_bank,
                latent_noise_bank=latent_noise_bank,
                conds_mean=base_model_baseline.conds_mean,
                conds_std=base_model_baseline.conds_std,
                T=T,
                clip_denoised=True,
                chunk_batch=chunk_batch,
                device=device,
            )
            _save_distance_to_final_json(baseline_dir, dist_by_mode["BASELINE"], T, "BASELINE")
        if run_ff:
            dist_by_mode["FF"] = _collect_distance_to_final(
                model=model_ff,
                mode_tag="FF",
                sampler=sampler,
                latent_sampler=latent_sampler,
                conf=conf,
                x_T_bank=x_T_bank,
                latent_noise_bank=latent_noise_bank,
                conds_mean=base_model_ff.conds_mean,
                conds_std=base_model_ff.conds_std,
                T=T,
                clip_denoised=True,
                chunk_batch=chunk_batch,
                device=device,
            )
            _save_distance_to_final_json(ff_dir, dist_by_mode["FF"], T, "FF")
        if run_ft:
            dist_by_mode["FT"] = _collect_distance_to_final(
                model=model_ft,
                mode_tag="FT",
                sampler=sampler,
                latent_sampler=latent_sampler,
                conf=conf,
                x_T_bank=x_T_bank,
                latent_noise_bank=latent_noise_bank,
                conds_mean=base_model_ft.conds_mean,
                conds_std=base_model_ft.conds_std,
                T=T,
                clip_denoised=True,
                chunk_batch=chunk_batch,
                device=device,
            )
            _save_distance_to_final_json(ft_dir, dist_by_mode["FT"], T, "FT")
        if run_tt:
            dist_by_mode["TT"] = _collect_distance_to_final(
                model=model_ft,
                mode_tag="TT",
                sampler=sampler,
                latent_sampler=latent_sampler,
                conf=conf,
                x_T_bank=x_T_bank,
                latent_noise_bank=latent_noise_bank,
                conds_mean=base_model_ft.conds_mean,
                conds_std=base_model_ft.conds_std,
                T=T,
                clip_denoised=True,
                chunk_batch=chunk_batch,
                device=device,
            )
            _save_distance_to_final_json(tt_dir, dist_by_mode["TT"], T, "TT")

    def _load_pred_stats(mode: str, mdir: Path) -> Optional[Dict[str, Any]]:
        p = mdir / "pred_xstart_stats.json"
        if not p.exists():
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    for _mode, _mdir in [
        ("BASELINE", baseline_dir),
        ("FF", ff_dir),
        ("FT", ft_dir),
        ("TT", tt_dir),
    ]:
        sj = _load_pred_stats(_mode, _mdir)
        if sj is None:
            continue
        d = dist_by_mode.get(_mode)
        _write_model_trajectory_summary_json(
            mode_dir=_mdir,
            stats_json=sj,
            dist_by_t=d["by_t"] if d is not None else None,
            T=T,
            mode=_mode,
        )

    baseline_div_payloads: Dict[str, Dict[str, Any]] = {}

    def _stats_cached(mode: str) -> Optional[Dict[str, Any]]:
        if mode in mode_stats_cache:
            return mode_stats_cache[mode]
        p = models_root / mode / "pred_xstart_stats.json"
        if not p.exists():
            return None
        with open(p, "r", encoding="utf-8") as f:
            o = json.load(f)
            mode_stats_cache[mode] = o
            return o

    if run_compare:
        if ff_stats_json is None:
            with open(ff_dir / "pred_xstart_stats.json", "r", encoding="utf-8") as f:
                ff_stats_json = json.load(f)
        if ft_stats_json is None:
            with open(ft_dir / "pred_xstart_stats.json", "r", encoding="utf-8") as f:
                ft_stats_json = json.load(f)

        cross_ff_ft = _collect_cross_model_pair_delta(
            model_a=model_ff,
            model_b=model_ft,
            mode_a="FF",
            mode_b="FT",
            latent_net=model_ff.latent_net,
            sampler=sampler,
            latent_sampler=latent_sampler,
            conf=conf,
            x_T_bank=x_T_bank,
            latent_noise_bank=latent_noise_bank,
            conds_mean=base_model_ff.conds_mean,
            conds_std=base_model_ff.conds_std,
            T=T,
            clip_denoised=True,
            chunk_batch=chunk_batch,
            device=device,
        )
        _save_pairwise_compare_outputs(
            compare_dir=cmp_dir_ff_ft,
            plots_pairwise=plots_pairwise,
            name_a="FF",
            name_b="FT",
            stats_a=ff_stats_json,
            stats_b=ft_stats_json,
            cross_same_t=cross_ff_ft,
            dist_a_final=dist_by_mode.get("FF") if enable_distance_to_final else None,
            dist_b_final=dist_by_mode.get("FT") if enable_distance_to_final else None,
            T=T,
            skip_plots=skip_plots,
            plot_file_prefix="ff_vs_ft",
        )

        if run_baseline and model_baseline is not None and base_model_baseline is not None:
            bj = _stats_cached("BASELINE")
            for tgt_name, tgt_model in [("FF", model_ff), ("FT", model_ft), ("TT", model_ft)]:
                sj = _stats_cached(tgt_name)
                if bj is None or sj is None:
                    continue
                cross_b = _collect_cross_model_pair_delta(
                    model_a=model_baseline,
                    model_b=tgt_model,
                    mode_a="BASELINE",
                    mode_b=tgt_name,
                    latent_net=model_baseline.latent_net,
                    sampler=sampler,
                    latent_sampler=latent_sampler,
                    conf=conf,
                    x_T_bank=x_T_bank,
                    latent_noise_bank=latent_noise_bank,
                    conds_mean=base_model_baseline.conds_mean,
                    conds_std=base_model_baseline.conds_std,
                    T=T,
                    clip_denoised=True,
                    chunk_batch=chunk_batch,
                    device=device,
                )
                dname = f"baseline_vs_{tgt_name.lower()}"
                cd = comparisons_root / dname
                _save_pairwise_compare_outputs(
                    compare_dir=cd,
                    plots_pairwise=plots_pairwise,
                    name_a="BASELINE",
                    name_b=tgt_name,
                    stats_a=bj,
                    stats_b=sj,
                    cross_same_t=cross_b,
                    dist_a_final=dist_by_mode.get("BASELINE") if enable_distance_to_final else None,
                    dist_b_final=dist_by_mode.get(tgt_name) if enable_distance_to_final else None,
                    T=T,
                    skip_plots=skip_plots,
                    plot_file_prefix=dname,
                )
                pl = _write_divergence_to_baseline_file(
                    out_path=cd / "divergence_to_baseline.json",
                    cross=cross_b,
                    T=T,
                    target_mode=tgt_name,
                )
                baseline_div_payloads[tgt_name.lower()] = pl

        if baseline_div_payloads:
            with open(
                comparisons_root / "baseline_divergence_summary.json", "w", encoding="utf-8"
            ) as f:
                json.dump(
                    {
                        "meta": {"T": T, "reference": "BASELINE"},
                        "targets": {
                            k: v.get("summary", {}) for k, v in baseline_div_payloads.items()
                        },
                    },
                    f,
                    indent=2,
                    ensure_ascii=False,
                )

        if run_tt:
            ff_sj = _stats_cached("FF")
            tt_sj = _stats_cached("TT")
            if ff_sj is not None and tt_sj is not None:
                cross_ftt = _collect_cross_model_pair_delta(
                    model_a=model_ff,
                    model_b=model_ft,
                    mode_a="FF",
                    mode_b="TT",
                    latent_net=model_ff.latent_net,
                    sampler=sampler,
                    latent_sampler=latent_sampler,
                    conf=conf,
                    x_T_bank=x_T_bank,
                    latent_noise_bank=latent_noise_bank,
                    conds_mean=base_model_ff.conds_mean,
                    conds_std=base_model_ff.conds_std,
                    T=T,
                    clip_denoised=True,
                    chunk_batch=chunk_batch,
                    device=device,
                )
                _save_pairwise_compare_outputs(
                    compare_dir=comparisons_root / "ff_vs_tt",
                    plots_pairwise=plots_pairwise,
                    name_a="FF",
                    name_b="TT",
                    stats_a=ff_sj,
                    stats_b=tt_sj,
                    cross_same_t=cross_ftt,
                    dist_a_final=dist_by_mode.get("FF") if enable_distance_to_final else None,
                    dist_b_final=dist_by_mode.get("TT") if enable_distance_to_final else None,
                    T=T,
                    skip_plots=skip_plots,
                    plot_file_prefix="ff_vs_tt",
                )

        if run_tt and run_ft and model_ft_clone is not None:
            ft_sj = _stats_cached("FT")
            tt_sj = _stats_cached("TT")
            if ft_sj is not None and tt_sj is not None:
                cross_ttp = _collect_cross_model_pair_delta(
                    model_a=model_ft,
                    model_b=model_ft_clone,
                    mode_a="FT",
                    mode_b="TT",
                    latent_net=model_ft.latent_net,
                    sampler=sampler,
                    latent_sampler=latent_sampler,
                    conf=conf,
                    x_T_bank=x_T_bank,
                    latent_noise_bank=latent_noise_bank,
                    conds_mean=base_model_ft.conds_mean,
                    conds_std=base_model_ft.conds_std,
                    T=T,
                    clip_denoised=True,
                    chunk_batch=chunk_batch,
                    device=device,
                )
                _save_pairwise_compare_outputs(
                    compare_dir=comparisons_root / "ft_vs_tt",
                    plots_pairwise=plots_pairwise,
                    name_a="FT",
                    name_b="TT",
                    stats_a=ft_sj,
                    stats_b=tt_sj,
                    cross_same_t=cross_ttp,
                    dist_a_final=dist_by_mode.get("FT") if enable_distance_to_final else None,
                    dist_b_final=dist_by_mode.get("TT") if enable_distance_to_final else None,
                    T=T,
                    skip_plots=skip_plots,
                    plot_file_prefix="ft_vs_tt",
                )

        missing_ff = [t for t in range(T) if str(t) not in ff_stats_json["by_t"]]
        missing_ft = [t for t in range(T) if str(t) not in ft_stats_json["by_t"]]
        same_counts = np.array(
            [int(cross_ff_ft["by_t"].get(str(t), {}).get("count", 0)) for t in range(T)],
            dtype=np.int64,
        )
        LOGGER.info(
            "Compare done: missing timesteps FF=%s FT=%s | same-t aligned sample_count[min=%d max=%d]",
            missing_ff[:10],
            missing_ft[:10],
            int(np.min(same_counts)) if same_counts.size else 0,
            int(np.max(same_counts)) if same_counts.size else 0,
        )

    per_model_hn: Dict[str, Any] = {}
    for m in ["BASELINE", "FF", "FT", "TT"]:
        sj = _stats_cached(m)
        if sj is not None:
            per_model_hn[m] = _high_noise_zone_metrics(sj, T)

    pairwise_hn: Dict[str, Any] = {}
    for pname, a, b in [
        ("baseline_vs_ff", "BASELINE", "FF"),
        ("baseline_vs_ft", "BASELINE", "FT"),
        ("baseline_vs_tt", "BASELINE", "TT"),
        ("ff_vs_ft", "FF", "FT"),
        ("ff_vs_tt", "FF", "TT"),
        ("ft_vs_tt", "FT", "TT"),
    ]:
        sa, sb = _stats_cached(a), _stats_cached(b)
        if sa is None or sb is None:
            continue
        pairwise_hn[pname] = _pairwise_high_noise_diff(sa, sb, T, pname)

    with open(comparisons_root / "high_noise_regime_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "T": T,
                    "high_noise_zone": "upper third of timestep indices t (t >= ceil(2T/3)); higher t = more noise in this DDIM ordering",
                },
                "per_model_high_noise_means": per_model_hn,
                "pairwise_mean_diffs": pairwise_hn,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    traj_reg: Dict[str, Any] = {}
    for m in ["BASELINE", "FF", "FT", "TT"]:
        tp = models_root / m / "trajectory_summary.json"
        if tp.exists():
            with open(tp, "r", encoding="utf-8") as f:
                traj_reg[m] = json.load(f)
    with open(
        comparisons_root / "trajectory_regularization_summary.json", "w", encoding="utf-8"
    ) as f:
        json.dump({"meta": {"T": T}, "by_mode": traj_reg}, f, indent=2, ensure_ascii=False)

    if not skip_plots:
        if baseline_div_payloads:
            _plot_baseline_divergence_overlays(
                div_payloads=baseline_div_payloads,
                plots_summary=plots_summary,
                T=T,
            )
        sbm: Dict[str, Dict[str, Any]] = {}
        for m in ["BASELINE", "FF", "FT", "TT"]:
            sj = _stats_cached(m)
            if sj is not None:
                sbm[m] = sj
        if sbm:
            _plot_high_noise_model_comparison(stats_by_mode=sbm, plots_summary=plots_summary, T=T)
            _plot_trajectory_regularization_overlays(
                stats_by_mode=sbm,
                dist_by_mode={m: dist_by_mode.get(m) for m in sbm},
                plots_summary=plots_summary,
                T=T,
            )

    if save_quantile_band_overlay:
        ref_mode = overlay_reference_mode.strip().upper()
        target_modes = [m.strip().upper() for m in overlay_target_modes.split(",") if m.strip()]
        if ref_mode not in mode_stats_cache:
            ref_path = models_root / ref_mode / "pred_xstart_stats.json"
            if ref_path.exists():
                with open(ref_path, "r", encoding="utf-8") as f:
                    mode_stats_cache[ref_mode] = json.load(f)
        ref_stats = mode_stats_cache.get(ref_mode)
        if ref_stats is None:
            LOGGER.warning("overlay skipped: reference mode stats missing (%s)", ref_mode)
        else:
            for tm in target_modes:
                if tm == ref_mode:
                    continue
                tgt_stats = mode_stats_cache.get(tm)
                if tgt_stats is None:
                    tgt_path = models_root / tm / "pred_xstart_stats.json"
                    if tgt_path.exists():
                        with open(tgt_path, "r", encoding="utf-8") as f:
                            tgt_stats = json.load(f)
                            mode_stats_cache[tm] = tgt_stats
                if tgt_stats is None:
                    LOGGER.warning("overlay target missing: %s (skip)", tm)
                    continue
                _plot_quantile_band_overlay_from_stats(
                    ref_stats_json=ref_stats,
                    target_stats_json=tgt_stats,
                    ref_label=ref_mode,
                    target_label=tm,
                    T=T,
                    out_png=plots_pairwise
                    / f"pred_xstart_quantiles_overlay_{ref_mode.lower()}_vs_{tm.lower()}.png",
                )

    _ensure_legacy_symlinks(out_root, models_root)
    _write_analysis_summary_md(
        out_path=out_root / "analysis_summary.md",
        T=T,
        baseline_div=baseline_div_payloads,
        high_noise_path=comparisons_root / "high_noise_regime_summary.json",
        traj_path=comparisons_root / "trajectory_regularization_summary.json",
    )
    LOGGER.info(
        "Summary: processed_images=%d, T=%d, output_root=%s",
        num_images,
        T,
        out_root,
    )
    return {"output_root": str(out_root), "T": T, "num_images": num_images}


def plot_pred_xstart_quantile_overlay(*, npz_baseline: Path, npz_v2: Path, out_png: Path) -> None:
    """Public function plot_pred_xstart_quantile_overlay."""
    nb = np.load(npz_baseline, allow_pickle=True)
    nv = np.load(npz_v2, allow_pickle=True)

    q_b = nb["quantiles"]  # (T,7)
    q_v = nv["quantiles"]
    t = nb["t"]  # (T,)

    def bands(q_arr: np.ndarray) -> "Any":
        """Public function bands."""
        q01 = q_arr[:, 0]
        q25 = q_arr[:, 2]
        q50 = q_arr[:, 3]
        q75 = q_arr[:, 4]
        q99 = q_arr[:, 6]
        return q01, q25, q50, q75, q99

    b01, b25, b50, b75, b99 = bands(q_b)
    v01, v25, v50, v75, v99 = bands(q_v)

    plt.figure(figsize=(10, 5))
    # baseline
    plt.fill_between(t, b01, b99, color="tab:blue", alpha=0.15, label="baseline q01-q99")
    plt.fill_between(t, b25, b75, color="tab:blue", alpha=0.25, label="baseline q25-q75")
    plt.plot(t, b50, color="tab:blue", linewidth=2.0, label="baseline q50")
    # v2
    plt.fill_between(t, v01, v99, color="tab:orange", alpha=0.15, label="v2 q01-q99")
    plt.fill_between(t, v25, v75, color="tab:orange", alpha=0.25, label="v2 q25-q75")
    plt.plot(t, v50, color="tab:orange", linewidth=2.0, label="v2 q50")

    plt.ylim(-1.0, 1.0)
    plt.xlim(0, len(t) - 1)
    # 依照你統一約定：左邊最雜端 t=T-1，右邊最清晰端 t=0
    plt.gca().invert_xaxis()
    plt.xlabel("DDIM timestep t")  # noise (T-1) -> clear (0) 放在論文圖中說明
    plt.ylabel("Predicted x_start value")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="best")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


@time_operation
def main_pred_xstart_quantile_analysis(
    *,
    images_mode: str,
    chunk_batch: int,
    hist_bins: int,
    seed: int,
    pred_output_root: str,
    skip_plot: bool,
    num_steps: int,
) -> Tuple[Path, Path]:
    """Public function main_pred_xstart_quantile_analysis."""
    # Legacy compatibility wrapper: keep old API name but route to the new formal trajectory pipeline.
    res = run_pred_xstart_trajectory_analysis(
        images_mode=images_mode,
        chunk_batch=chunk_batch,
        hist_bins=hist_bins,
        hist_min=-1.0,
        hist_max=1.0,
        seed=seed,
        pred_output_root=pred_output_root,
        skip_plots=skip_plot,
        num_steps=num_steps,
        run_ff=True,
        run_ft=True,
        run_tt=True,
        run_baseline=True,
        run_compare=True,
        enable_distance_to_final=True,
    )
    root = Path(res["output_root"])
    return (
        root / "models" / "FF" / "pred_xstart_stats.npz",
        root / "models" / "FT" / "pred_xstart_stats.npz",
    )


def _setup_logging(log_file: str) -> None:
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
        force=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", action="store_true", help="run full pred-xstart / trajectory analysis"
    )
    parser.add_argument(
        "--run_pred_xstart_quantile_analysis", action="store_true", help="legacy alias of --run"
    )
    parser.add_argument(
        "--images_mode", type=str, default="official", choices=["debug", "v1", "official"]
    )
    parser.add_argument("--chunk_batch", type=int, default=32)
    parser.add_argument("--hist_bins", type=int, default=4096)
    parser.add_argument("--hist_min", type=float, default=-1.0)
    parser.add_argument("--hist_max", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pred_output_root",
        type=str,
        default="QATcode/quantize_ver2/baseline_quant_analysis/pred_xstart_results",
    )
    parser.add_argument("--skip_plots", action="store_true")
    parser.add_argument("--num_steps", "--n", type=int, default=100)
    parser.add_argument("--run_ff", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--run_ft", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--run_tt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="collect TT under models/TT and ff_vs_tt; required with --run_baseline for comparisons/baseline_vs_tt",
    )
    parser.add_argument(
        "--run_baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="collect BASELINE and pairwise baseline_vs_ff|ft|tt (needs each target mode's stats, e.g. --run_tt for TT)",
    )
    parser.add_argument("--run_compare", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--enable_distance_to_final", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--save_quantile_band_overlay", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--overlay_reference_mode", type=str, default="BASELINE")
    parser.add_argument("--overlay_target_modes", type=str, default="TT,FT,FF")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if not args.run and not args.run_pred_xstart_quantile_analysis:
        parser.print_help()
        raise SystemExit(0)

    CONFIG.NUM_DIFFUSION_STEPS = int(args.num_steps)
    CONFIG.DEVICE = _resolve_device(args.device)
    log_file = str(Path(args.pred_output_root) / "logs" / "pred_xstart_trajectory_analysis.log")
    _setup_logging(log_file)
    LOGGER.info("Using device: %s", CONFIG.DEVICE)

    res = run_pred_xstart_trajectory_analysis(
        images_mode=args.images_mode,
        chunk_batch=int(args.chunk_batch),
        hist_bins=int(args.hist_bins),
        hist_min=float(args.hist_min),
        hist_max=float(args.hist_max),
        seed=int(args.seed),
        pred_output_root=args.pred_output_root,
        skip_plots=bool(args.skip_plots),
        num_steps=int(args.num_steps),
        run_ff=bool(args.run_ff),
        run_ft=bool(args.run_ft),
        run_tt=bool(args.run_tt),
        run_baseline=bool(args.run_baseline),
        run_compare=bool(args.run_compare),
        enable_distance_to_final=bool(args.enable_distance_to_final),
        save_quantile_band_overlay=bool(args.save_quantile_band_overlay),
        overlay_reference_mode=str(args.overlay_reference_mode),
        overlay_target_modes=str(args.overlay_target_modes),
    )
    LOGGER.info("Done: %s", res)
