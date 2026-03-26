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
import logging
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys

sys.path.append(".")
sys.path.append("./model")

from QATcode.quantize_ver2.quant_dataset_v2 import DiffusionInputDataset
from QATcode.quantize_ver2.quant_model_lora_v2 import (
    QuantModel_DiffAE_LoRA,
    QuantModule_DiffAE_LoRA,
    INT_QuantModel_DiffAE_LoRA,
)
from QATcode.quantize_ver2.quant_layer_v2 import SimpleDequantizer

from experiment import LitModel
from templates_latent import ffhq128_autoenc_latent


LOGGER = logging.getLogger("pred_xstart_quantile_analysis")


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def time_operation(func: Callable) -> Callable:
    def wrapper(*args, **kwargs):
        start = time.time()
        out = func(*args, **kwargs)
        elapsed = time.time() - start
        LOGGER.info("執行 '%s' 完成，耗時: %.2f 秒", func.__name__, elapsed)
        return out

    return wrapper


@dataclass
class PredXStartConfig:
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
    LOGGER.info("載入 Diff-AE 模型: %s", model_path)
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    conf = ffhq128_autoenc_latent()
    model = LitModel(conf)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    LOGGER.info("Diff-AE 模型載入完成")
    return model


def create_float_quantized_model(
    diffusion_model: nn.Module,
    num_steps: int = CONFIG.NUM_DIFFUSION_STEPS,
    lora_rank: int = CONFIG.LORA_RANK,
    mode: str = "train",
) -> QuantModel_DiffAE_LoRA:
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


def _get_train_samples(train_loader: DataLoader, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        torch.cat(y_data, dim=0)[:num_samples],
    )


def load_calibration_data() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    LOGGER.info("載入校準資料...")
    try:
        dataset = DiffusionInputDataset(CONFIG.CALIB_DATA_PATH)
        data_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)
        cali_images, cali_t, cali_y = _get_train_samples(data_loader, num_samples=CONFIG.CALIB_SAMPLES)
        LOGGER.info("✅ 載入真實校準資料成功")
    except Exception as e:
        LOGGER.warning("⚠️ 載入校準資料失敗: %s", e)
        LOGGER.info("使用合成校準資料")
        # fallback shapes match dataset expectations for calibration forward
        cali_images = torch.randn(CONFIG.CALIB_SAMPLES, 3, 128, 128)
        cali_t = torch.randint(0, CONFIG.NUM_DIFFUSION_STEPS, (CONFIG.CALIB_SAMPLES,))
        cali_y = torch.randint(0, 1000, (CONFIG.CALIB_SAMPLES,))
    return cali_images, cali_t, cali_y


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


def _load_quant_and_ema_from_ckpt(base_model: LitModel, quant_model: nn.Module, ckpt: Dict[str, Any]) -> None:
    """
    Load strategy (consistent with the existing codebase logic):
    - base_model.ema_model 架構 = deepcopy(quant_model)
    - base_model.ema_model 權重只吃 `ema_model.model.*` (mapped to `model.*` keys)
    - base_model.model 權重吃 `model.model.*` (if exists)
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
    if conf.latent_znormalize :
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
    clip_denoised: bool,
    hist_range_min: float,
    hist_range_max: float,
    hist_bins: int,
    chunk_batch: int,
    device: torch.device,
) -> np.ndarray:
    """
    Progressive loop:
    for each yielded out:
      t_idx = int(out["t"][0].item())
      pred_flat = out["pred_xstart"].reshape(-1)
      hist_counts[t_idx] += histogram(pred_flat, range, bins)
    """
    hist_counts = torch.zeros((T, hist_bins), device=device, dtype=torch.float64)

    num_images = x_T_bank.shape[0]
    assert num_images == latent_noise_bank.shape[0]
    num_chunks = num_images // chunk_batch

    cache_scheduler = getattr(conf, "cache_scheduler", None)

    for ci in range(num_chunks):
        b0 = ci * chunk_batch
        b1 = (ci + 1) * chunk_batch

        x_T_chunk = x_T_bank[b0:b1].to(device)
        latent_noise_chunk = latent_noise_bank[b0:b1].to(device)

        # latent condition (renderer.py semantic alignment)
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
            pred_flat = out["pred_xstart"].reshape(-1)
            pred_flat = pred_flat.clamp(hist_range_min, hist_range_max)
            counts = torch.histc(
                pred_flat,
                bins=hist_bins,
                min=hist_range_min,
                max=hist_range_max,
            ).to(dtype=torch.float64)
            hist_counts[t_idx] += counts

    return hist_counts.cpu().numpy()


def _hist_counts_to_quantiles(
    *,
    hist_counts: np.ndarray,  # (T, bins)
    bin_edges: np.ndarray,  # (bins+1,)
    q_list: np.ndarray,  # (7,)
) -> np.ndarray:
    """
    cumulative counts => quantiles via bin-wise linear interpolation.
    """
    T, bins = hist_counts.shape
    cumulative = np.cumsum(hist_counts, axis=1)
    total = cumulative[:, -1]

    quantiles = np.zeros((T, len(q_list)), dtype=np.float64)
    for t in range(T):
        if total[t] <= 0:
            quantiles[t, :] = np.nan
            continue
        for qi, q in enumerate(q_list):
            target = q * total[t]
            idx = int(np.searchsorted(cumulative[t], target, side="left"))
            if idx >= bins:
                idx = bins - 1

            left_edge = bin_edges[idx]
            right_edge = bin_edges[idx + 1]
            prev_cdf = cumulative[t, idx - 1] if idx > 0 else 0.0
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


def plot_pred_xstart_quantile_overlay(*, npz_baseline: Path, npz_v2: Path, out_png: Path) -> None:
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
    images_mode: str,
    chunk_batch: int,
    hist_bins: int,
    seed: int,
    pred_output_root: str,
    skip_plot: bool,
    num_steps: int,
) -> Tuple[Path, Path]:
    clip_denoised = True  # fixed spec

    _seed_all(seed)
    device = CONFIG.DEVICE

    T = int(num_steps)
    num_images = _images_mode_to_num_images(images_mode)
    hist_range_min, hist_range_max = -1.0, 1.0

    LOGGER.info(
        "[pred_xstart_quantile] images_mode=%s num_images=%d T=%d chunk_batch=%d bins=%d seed=%d",
        images_mode,
        num_images,
        T,
        chunk_batch,
        hist_bins,
        seed,
    )

    # 1) baseline: load model, compute conds_mean/std, build sampler
    base_model_baseline: LitModel = load_diffae_model(CONFIG.MODEL_PATH)
    base_model_baseline.to(device)
    base_model_baseline.eval()
    base_model_baseline.setup()
    base_model_baseline.train_dataloader()

    conf = base_model_baseline.conf.clone()
    sampler = conf._make_diffusion_conf(T=T).make_sampler()
    latent_sampler = conf._make_latent_diffusion_conf(T=T).make_sampler()

    sampler_timestep_map_exists = bool(hasattr(sampler, "timestep_map") and getattr(sampler, "timestep_map", None))
    mapped_t = (
        np.array(getattr(sampler, "timestep_map"), dtype=np.int32)
        if sampler_timestep_map_exists
        else None
    )

    # 2) shared input banks (baseline/v2 MUST share)
    x_T_bank, latent_noise_bank = _make_noise_banks(
        num_images=num_images,
        chunk_batch=chunk_batch,
        img_size=conf.img_size,
        style_ch=conf.style_ch,
        seed=seed,
        device=device,
    )

    # output paths
    out_root = Path(pred_output_root) / f"T_{T}" / images_mode
    out_root.mkdir(parents=True, exist_ok=True)
    bin_edges = np.linspace(hist_range_min, hist_range_max, hist_bins + 1, dtype=np.float64)
    t = np.arange(T, dtype=np.int32)
    pred_key = np.array("pred_xstart")

    def collect_for(tag: str, model: nn.Module, ckpt_path: str) -> Path:
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

        out_npz = out_root / f"pred_xstart_quantiles_{tag}_seed{seed}_N{num_images}_bins{hist_bins}.npz"

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
            "sampler_timestep_map_exists": np.array(sampler_timestep_map_exists, dtype=np.bool_),
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

    # baseline
    npz_baseline = collect_for("baseline", base_model_baseline.ema_model, CONFIG.MODEL_PATH)

    # 3) v2 quantize_ver2: rebuild model and attach quant checkpoint
    base_model_v2: LitModel = load_diffae_model(CONFIG.MODEL_PATH)
    base_model_v2.to(device)
    base_model_v2.eval()

    diffusion_model = base_model_v2.ema_model
    quant_model = create_float_quantized_model(
        diffusion_model,
        num_steps=T,
        lora_rank=CONFIG.LORA_RANK,
        mode="train",
    )
    quant_model.to(device)
    quant_model.eval()

    # init dequantizer for dynamic module usage (matches previous logic)
    for _, module in quant_model.named_modules():
        if isinstance(module, QuantModule_DiffAE_LoRA) and getattr(module, "ignore_reconstruction", False) is False:
            module.intn_dequantizer = SimpleDequantizer(uaq=module.weight_quantizer, weight=module.weight).to(device)

    # temporal act quant needs a calibration forward
    cali_images, cali_t, cali_y = load_calibration_data()
    quant_model.set_first_last_layer_to_8bit()
    quant_model.set_quant_state(True, True)
    with torch.no_grad():
        _ = quant_model(
            x=cali_images[:4].to(device),
            t=cali_t[:4].to(device),
            cond=cali_y[:4].to(device),
        )

    ckpt_path_v2 = _resolve_best_ckpt_path(T)
    ckpt = torch.load(ckpt_path_v2, map_location="cpu", weights_only=False)
    _load_quant_and_ema_from_ckpt(base_model_v2, quant_model, ckpt)

    npz_v2 = collect_for("v2_latest", base_model_v2.ema_model, ckpt_path_v2)

    if not skip_plot:
        out_png = out_root / f"pred_xstart_quantiles_overlay_{images_mode}_T{T}_seed{seed}.png"
        plot_pred_xstart_quantile_overlay(npz_baseline=npz_baseline, npz_v2=npz_v2, out_png=out_png)
        LOGGER.info("疊圖輸出: %s", out_png)

    LOGGER.info("[pred_xstart_quantile] done.")
    return npz_baseline, npz_v2


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
    parser.add_argument("--run_pred_xstart_quantile_analysis", action="store_true")
    parser.add_argument("--images_mode", type=str, default="official", choices=["debug", "v1", "official"])
    parser.add_argument("--chunk_batch", type=int, default=32)
    parser.add_argument("--hist_bins", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pred_output_root", type=str, default="QATcode/cache_method/pred_xstart_quantile_analysis")
    parser.add_argument("--skip_plot", action="store_true")
    parser.add_argument("--num_steps", "--n", type=int, default=100)
    args = parser.parse_args()

    if not args.run_pred_xstart_quantile_analysis:
        parser.print_help()
        raise SystemExit(0)

    CONFIG.NUM_DIFFUSION_STEPS = int(args.num_steps)
    log_file = str(Path(args.pred_output_root) / "log" / "pred_xstart_quantile_analysis.log")
    _setup_logging(log_file)

    npz_baseline, npz_v2 = main_pred_xstart_quantile_analysis(
        images_mode=args.images_mode,
        chunk_batch=int(args.chunk_batch),
        hist_bins=int(args.hist_bins),
        seed=int(args.seed),
        pred_output_root=args.pred_output_root,
        skip_plot=bool(args.skip_plot),
        num_steps=int(args.num_steps),
    )
    LOGGER.info("baseline npz: %s", npz_baseline)
    LOGGER.info("v2 npz: %s", npz_v2)


