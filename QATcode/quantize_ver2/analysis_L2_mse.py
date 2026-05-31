#!/usr/bin/env python3
"""Per-timestep L2/MSE analysis for Diff-AE FP32 vs Q-DiffAE W8A8.

The script compares DDIM trajectories from the FP EMA model and the QAT EMA
model with identical ``x_T`` and ``latent_noise`` for each run. It uses the
progressive DDIM generator from ``diffusion/base.py`` so every intermediate
``x_t`` and ``pred_xstart`` can be measured.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path("/home/jimmy/diffae")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from QATcode.cache_method.a_L1_L2_cosine.similarity_calculation import (
    _load_quant_and_ema_from_ckpt,
)
from QATcode.quantize_ver2 import sample_lora_intmodel_v2 as qv2
from QATcode.quantize_ver2.common_utils import load_diffae_model, seed_all
from templates_latent import ffhq128_autoenc_latent


DEVICE = torch.device("cuda:0")
SEED = 42
NUM_SAMPLES = 16
NUM_DIFFUSION_STEPS = 100
FP_MODEL_PATH = "checkpoints/ffhq128_autoenc_latent/last.ckpt"
QAT_CKPT_PATH = "QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best.pth"
OUTPUT_DIR = "QATcode/quantize_ver2/analysis_L2_results"

LOGGER = logging.getLogger("L2MSEAnalysis")


def _resolve_repo_path(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def _setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "analysis_L2_mse.log", encoding="utf-8"),
        ],
        force=True,
    )


def load_fp_model(model_path: str = FP_MODEL_PATH) -> Tuple[Any, Any]:
    """Load FP32 Diff-AE and return ``(base_model, conf)`` for inference."""
    base_model = load_diffae_model(model_path, LOGGER)
    conf = ffhq128_autoenc_latent()
    base_model.eval()
    base_model.to(DEVICE)
    base_model.setup()
    return base_model, conf


def load_quant_model(
    fp_model_path: str = FP_MODEL_PATH,
    qat_ckpt_path: str = QAT_CKPT_PATH,
    num_steps: int = NUM_DIFFUSION_STEPS,
) -> Tuple[Any, Any]:
    """Load Q-DiffAE W8A8 and return ``(base_model, conf)`` for inference."""
    qv2.CONFIG.DEVICE = DEVICE
    qv2.CONFIG.NUM_DIFFUSION_STEPS = num_steps
    qv2.CONFIG.BEST_CKPT_PATH = qat_ckpt_path
    qv2.CONFIG.QUANT_STATE_WEIGHT = True
    qv2.CONFIG.QUANT_STATE_ACT = True

    base_model = load_diffae_model(fp_model_path, LOGGER)
    diffusion_model = base_model.ema_model

    quant_model = qv2.create_float_quantized_model(
        diffusion_model,
        num_steps=num_steps,
        lora_rank=qv2.CONFIG.LORA_RANK,
        mode=qv2.CONFIG.MODE,
    )
    quant_model.to(DEVICE)
    quant_model.eval()

    # Match sample_lora_intmodel_v2.py: initialize quantizers before loading QAT EMA.
    cali_images, cali_t, cali_y = qv2.load_calibration_data()
    quant_model.set_first_last_layer_to_8bit()
    quant_model.set_quant_state(True, True)
    if hasattr(quant_model, "set_runtime_mode"):
        quant_model.set_runtime_mode(mode="train", use_cached_aw=False, clear_cached_aw=True)

    with torch.no_grad():
        _ = quant_model(
            x=cali_images[:32].to(DEVICE),
            t=cali_t[:32].to(DEVICE),
            cond=cali_y[:32].to(DEVICE),
        )

    ckpt = torch.load(qat_ckpt_path, map_location="cpu", weights_only=False)
    _load_quant_and_ema_from_ckpt(base_model, quant_model, ckpt)

    base_model.ema_model.set_quant_state(True, True)
    if hasattr(base_model.ema_model, "set_runtime_mode"):
        base_model.ema_model.set_runtime_mode(
            mode="infer",
            use_cached_aw=True,
            clear_cached_aw=True,
        )

    base_model.to(DEVICE)
    base_model.eval()
    base_model.setup()
    conf = ffhq128_autoenc_latent()
    return base_model, conf


def make_samplers(conf: Any, T: int = NUM_DIFFUSION_STEPS) -> Tuple[Any, Any]:
    """Create image sampler and latent sampler."""
    sampler = conf._make_diffusion_conf(T=T).make_sampler()
    latent_sampler = conf._make_latent_diffusion_conf(T=T).make_sampler()
    return sampler, latent_sampler


def _normalize_cond(base_model: Any, cond: torch.Tensor, conf: Any) -> torch.Tensor:
    if conf.latent_znormalize:
        cond = cond * base_model.conds_std.to(cond.device) + base_model.conds_mean.to(cond.device)
    return cond


@torch.no_grad()
def generate_cond(
    base_model: Any,
    latent_sampler: Any,
    latent_noise: torch.Tensor,
    conf: Any,
) -> torch.Tensor:
    """Generate semantic latent condition using the model's own latent_net."""
    cond = latent_sampler.sample(
        model=base_model.ema_model.latent_net,
        noise=latent_noise,
        clip_denoised=conf.latent_clip_sample,
    )
    return _normalize_cond(base_model, cond, conf)


@torch.no_grad()
def collect_fp_trajectory(
    model: Any,
    sampler: Any,
    cond: torch.Tensor,
    x_T: torch.Tensor,
    device: torch.device,
) -> Dict[str, Any]:
    """Run FP DDIM progressive loop and keep every step on CPU."""
    model_kwargs = {"x_start": None, "cond": cond}
    gen = sampler.ddim_sample_loop_progressive(
        model=model,
        shape=x_T.shape,
        noise=x_T,
        clip_denoised=True,
        model_kwargs=model_kwargs,
        device=device,
        progress=False,
        eta=0.0,
    )

    samples: List[torch.Tensor] = []
    pred_xstarts: List[torch.Tensor] = []
    timesteps: List[int] = []

    for out in gen:
        samples.append(out["sample"].detach().cpu())
        pred_xstarts.append(out["pred_xstart"].detach().cpu())
        timesteps.append(int(out["t"][0].detach().cpu().item()))

    return {
        "samples": samples,
        "pred_xstarts": pred_xstarts,
        "timesteps": timesteps,
        "cond_fp": cond.detach().cpu(),
    }


@torch.no_grad()
def compare_quant_to_fp(
    fp_traj: Dict[str, Any],
    quant_model: Any,
    sampler_q: Any,
    cond_q: torch.Tensor,
    x_T: torch.Tensor,
    device: torch.device,
) -> Dict[str, List[float]]:
    """Run quant DDIM progressive loop and compare each step with FP trajectory."""
    model_kwargs_q = {"x_start": None, "cond": cond_q}
    gen_q = sampler_q.ddim_sample_loop_progressive(
        model=quant_model,
        shape=x_T.shape,
        noise=x_T,
        clip_denoised=True,
        model_kwargs=model_kwargs_q,
        device=device,
        progress=False,
        eta=0.0,
    )

    results: Dict[str, List[float]] = {
        "timesteps": [],
        "mse_xt": [],
        "l2_norm_xt": [],
        "mse_pred_xstart": [],
    }

    total_steps = len(fp_traj["timesteps"])
    for step_idx, out_q in enumerate(gen_q):
        xt_fp = fp_traj["samples"][step_idx].to(device)
        xstart_fp = fp_traj["pred_xstarts"][step_idx].to(device)
        xt_q = out_q["sample"]
        xstart_q = out_q["pred_xstart"]

        diff_xt = xt_q - xt_fp
        mse_xt = (diff_xt**2).mean().item()
        l2_mean = diff_xt.reshape(diff_xt.shape[0], -1).norm(dim=1).mean().item()

        diff_xstart = xstart_q - xstart_fp
        mse_xstart = (diff_xstart**2).mean().item()

        t_value = int(fp_traj["timesteps"][step_idx])
        results["timesteps"].append(t_value)
        results["mse_xt"].append(mse_xt)
        results["l2_norm_xt"].append(l2_mean)
        results["mse_pred_xstart"].append(mse_xstart)

        if step_idx % 10 == 0:
            print(
                f"  Step {step_idx}/{total_steps}, t={t_value}, "
                f"MSE(x_t)={mse_xt:.6f}, L2={l2_mean:.4f}, "
                f"MSE(x0_hat)={mse_xstart:.6f}"
            )

    if len(results["timesteps"]) != total_steps:
        raise RuntimeError(
            f"Quant trajectory length {len(results['timesteps'])} != FP length {total_steps}"
        )

    return results


def _free_model(model: Any) -> None:
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_fp_phase(
    args: argparse.Namespace,
    traj_dir: Path,
) -> None:
    LOGGER.info("Loading FP Diff-AE model")
    fp_base, fp_conf = load_fp_model(args.fp_model_path)
    sampler_fp, latent_sampler_fp = make_samplers(fp_conf, args.num_steps)

    for run_idx in range(args.num_runs):
        seed = args.seed + run_idx
        LOGGER.info("[FP] run %d/%d seed=%d", run_idx + 1, args.num_runs, seed)
        seed_all(seed)
        x_T = torch.randn(
            args.num_samples,
            3,
            fp_conf.img_size,
            fp_conf.img_size,
            device=DEVICE,
        )
        latent_noise = torch.randn(args.num_samples, fp_conf.style_ch, device=DEVICE)
        cond_fp = generate_cond(fp_base, latent_sampler_fp, latent_noise, fp_conf)
        fp_traj = collect_fp_trajectory(fp_base.ema_model, sampler_fp, cond_fp, x_T, DEVICE)
        torch.save(fp_traj, traj_dir / f"fp_traj_run{run_idx:03d}.pt")

    _free_model(fp_base)


def run_quant_phase(
    args: argparse.Namespace,
    traj_dir: Path,
) -> List[Dict[str, List[float]]]:
    LOGGER.info("Loading Q-DiffAE W8A8 model")
    q_base, q_conf = load_quant_model(args.fp_model_path, args.qat_ckpt_path, args.num_steps)
    sampler_q, latent_sampler_q = make_samplers(q_conf, args.num_steps)

    all_results: List[Dict[str, List[float]]] = []
    for run_idx in range(args.num_runs):
        seed = args.seed + run_idx
        LOGGER.info("[Q] run %d/%d seed=%d cond_mode=%s", run_idx + 1, args.num_runs, seed, args.cond_mode)
        seed_all(seed)
        x_T = torch.randn(
            args.num_samples,
            3,
            q_conf.img_size,
            q_conf.img_size,
            device=DEVICE,
        )
        latent_noise = torch.randn(args.num_samples, q_conf.style_ch, device=DEVICE)

        fp_traj_path = traj_dir / f"fp_traj_run{run_idx:03d}.pt"
        fp_traj = torch.load(fp_traj_path, map_location="cpu", weights_only=False)

        if args.cond_mode == "shared":
            cond_q = fp_traj["cond_fp"].to(DEVICE)
        else:
            cond_q = generate_cond(q_base, latent_sampler_q, latent_noise, q_conf)

        results = compare_quant_to_fp(fp_traj, q_base.ema_model, sampler_q, cond_q, x_T, DEVICE)
        all_results.append(results)

        del fp_traj
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _free_model(q_base)
    return all_results


def aggregate_results(all_results: List[Dict[str, List[float]]]) -> Dict[str, Any]:
    if not all_results:
        raise ValueError("No results to aggregate")

    timesteps = all_results[0]["timesteps"]
    metrics = ["mse_xt", "l2_norm_xt", "mse_pred_xstart"]
    agg: Dict[str, Any] = {
        "timesteps": timesteps,
        "step_indices": list(range(len(timesteps))),
        "num_runs": len(all_results),
        "runs": all_results,
    }

    for metric in metrics:
        arr = np.asarray([r[metric] for r in all_results], dtype=np.float64)
        agg[f"{metric}_mean"] = arr.mean(axis=0).tolist()
        agg[f"{metric}_std"] = arr.std(axis=0).tolist()

    return agg


def _plot_metric(
    aggregate: Dict[str, Any],
    mean_key: str,
    std_key: str,
    ylabel: str,
    title: str,
    out_prefix: Path,
) -> None:
    x = np.asarray(aggregate["step_indices"], dtype=np.int64)
    y = np.asarray(aggregate[mean_key], dtype=np.float64)
    y_std = np.asarray(aggregate[std_key], dtype=np.float64)

    plt.figure(figsize=(12, 5))
    plt.plot(x, y, label="mean", linewidth=2)
    if aggregate["num_runs"] > 1:
        plt.fill_between(x, y - y_std, y + y_std, alpha=0.25, label="mean ± std")

    peak_idx = int(np.argmax(y))
    if peak_idx >= len(x) // 3 and y[peak_idx] > max(y[0], 1e-12) * 2:
        plt.annotate(
            f"peak step={peak_idx}",
            xy=(x[peak_idx], y[peak_idx]),
            xytext=(x[peak_idx], y[peak_idx] * 1.1),
            arrowprops={"arrowstyle": "->"},
        )

    plt.xlabel("DDIM step index (0 = highest noise, last = t=0)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.savefig(out_prefix.with_suffix(".pdf"), dpi=150, bbox_inches="tight")
    plt.close()


def save_outputs(args: argparse.Namespace, aggregate: Dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "config": {
            "device": str(DEVICE),
            "seed": args.seed,
            "num_samples": args.num_samples,
            "num_runs": args.num_runs,
            "num_steps": args.num_steps,
            "cond_mode": args.cond_mode,
            "fp_model_path": args.fp_model_path,
            "qat_ckpt_path": args.qat_ckpt_path,
        },
        "results": aggregate,
    }
    result_path = output_dir / "l2_mse_results.json"
    with result_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    LOGGER.info("Saved numeric results to %s", result_path)

    _plot_metric(
        aggregate,
        "mse_xt_mean",
        "mse_xt_std",
        "MSE(x_t)",
        "Per-Timestep MSE of x_t: Diff-AE (FP32) vs Q-DiffAE (W8A8)",
        output_dir / "fig_mse_xt",
    )
    _plot_metric(
        aggregate,
        "l2_norm_xt_mean",
        "l2_norm_xt_std",
        r"$||x_t^{quant} - x_t^{fp}||_2$ (batch mean)",
        "Error Accumulation: L2 Norm of x_t Difference",
        output_dir / "fig_l2_accumulation",
    )
    _plot_metric(
        aggregate,
        "mse_pred_xstart_mean",
        "mse_pred_xstart_std",
        r"MSE($\hat{x}_0$)",
        "Per-Timestep MSE of Predicted x_0",
        output_dir / "fig_mse_pred_xstart",
    )
    LOGGER.info("Saved figures to %s", output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="L2: Per-timestep MSE analysis")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    parser.add_argument("--num-runs", type=int, default=5, help="不同 seed 跑幾次")
    parser.add_argument("--num-steps", type=int, default=NUM_DIFFUSION_STEPS)
    parser.add_argument(
        "--cond-mode",
        choices=["own", "shared"],
        default="own",
        help="own=各用自己latent_net產cond; shared=都用FP的cond",
    )
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--fp-model-path", type=str, default=FP_MODEL_PATH)
    parser.add_argument("--qat-ckpt-path", type=str, default=QAT_CKPT_PATH)
    parser.add_argument(
        "--keep-trajectories",
        action="store_true",
        help="保留暫存的 FP trajectory .pt 檔",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    global DEVICE
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if DEVICE.type != "cuda":
        raise RuntimeError("CUDA is required for this analysis")

    args.output_dir = str(_resolve_repo_path(args.output_dir))
    args.fp_model_path = str(_resolve_repo_path(args.fp_model_path))
    args.qat_ckpt_path = str(_resolve_repo_path(args.qat_ckpt_path))

    output_dir = Path(args.output_dir)
    _setup_logging(output_dir)
    LOGGER.info("args=%s", args)

    traj_dir = output_dir / "_fp_trajectories"
    if traj_dir.exists():
        shutil.rmtree(traj_dir)
    traj_dir.mkdir(parents=True, exist_ok=True)

    try:
        run_fp_phase(args, traj_dir)
        all_results = run_quant_phase(args, traj_dir)
        aggregate = aggregate_results(all_results)
        save_outputs(args, aggregate, output_dir)
    finally:
        if not args.keep_trajectories and traj_dir.exists():
            shutil.rmtree(traj_dir)
            LOGGER.info("Removed temporary trajectories: %s", traj_dir)


if __name__ == "__main__":
    main()
