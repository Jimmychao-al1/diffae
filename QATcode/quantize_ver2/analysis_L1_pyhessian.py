#!/usr/bin/env python3
"""Layer 1: PyHessian sharpness comparison for Diff-AE vs Q-DiffAE.

The Hessian is computed on the image UNet epsilon-prediction MSE loss from
``GaussianDiffusion.training_losses`` using fixed FFHQ images, timesteps, and
Gaussian noise for a fair FP32/QAT comparison.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import gc
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from pyhessian import hessian
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset


REPO_ROOT = Path("/home/jimmy/diffae")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(str(REPO_ROOT))

from dataset import FFHQlmdb  # noqa: E402
from QATcode.quantize_ver2 import analysis_L2_mse as l2_loader  # noqa: E402
from QATcode.quantize_ver2.quant_model_lora_v2 import QuantModule_DiffAE_LoRA  # noqa: E402


FP_MODEL_PATH = "checkpoints/ffhq128_autoenc_latent/last.ckpt"
QAT_CKPT_PATH = "QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best.pth"
OUTPUT_DIR = "QATcode/quantize_ver2/analysis_L1_results"
OUTPUT_JSON = "l1_hessian_results.json"
DATASET_PATH = "datasets/ffhq256_lmdb"
IMAGE_SIZE = 128
DIFFUSION_T = 1000

LOGGER = logging.getLogger("Layer1PyHessian")


def resolve_repo_path(path: str | Path) -> Path:
    """Resolve a path relative to the Diff-AE repository root."""
    p = Path(path)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def setup_logging(output_dir: Path) -> None:
    """Configure console and file logging."""
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(output_dir / "analysis_L1_pyhessian.log", encoding="utf-8"),
        ],
        force=True,
    )


class FixedHessianDataset(Dataset):
    """Return fixed ``x_start``, ``noise``, and ``t`` packed for PyHessian."""

    def __init__(
        self,
        image_dataset: Dataset,
        timesteps: torch.Tensor,
        noises: torch.Tensor,
    ) -> None:
        if len(image_dataset) != len(timesteps) or len(image_dataset) != len(noises):
            raise ValueError(
                "image_dataset, timesteps, and noises must have the same length: "
                f"{len(image_dataset)}, {len(timesteps)}, {len(noises)}"
            )
        self.image_dataset = image_dataset
        self.timesteps = timesteps.cpu().long()
        self.noises = noises.cpu().float()

    def __len__(self) -> int:
        return len(self.image_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        item = self.image_dataset[idx]
        if not isinstance(item, dict) or "img" not in item:
            raise TypeError("Expected FFHQlmdb item to be a dict containing 'img'")

        x_start = item["img"].float()
        noise = self.noises[idx]
        t_plane = torch.full(
            (1, x_start.shape[1], x_start.shape[2]),
            float(self.timesteps[idx].item()),
            dtype=x_start.dtype,
        )
        packed = torch.cat([x_start, noise, t_plane], dim=0)
        dummy_target = torch.tensor(0.0, dtype=x_start.dtype)
        return packed, dummy_target


class EpsilonMSEHessianWrapper(nn.Module):
    """Wrap Diff-AE denoiser + GaussianDiffusion into PyHessian's model API."""

    def __init__(self, model: nn.Module, sampler: Any, suppress_loss_stdout: bool = True) -> None:
        super().__init__()
        self.model = model
        self.sampler = sampler
        self.suppress_loss_stdout = suppress_loss_stdout

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        x_start = packed[:, :3].contiguous()
        noise = packed[:, 3:6].contiguous()
        t = packed[:, 6, 0, 0].round().long()

        if self.suppress_loss_stdout:
            # diffusion/base.py prints every sampled t; suppress it during HVP loops.
            with contextlib.redirect_stdout(io.StringIO()):
                terms = self.sampler.training_losses(
                    model=self.model,
                    x_start=x_start,
                    t=t,
                    noise=noise,
                )
        else:
            terms = self.sampler.training_losses(
                model=self.model,
                x_start=x_start,
                t=t,
                noise=noise,
            )
        return terms["loss"].mean()


class ScalarLossCriterion(nn.Module):
    """Criterion adapter: the wrapper already returns a scalar loss."""

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        del target
        return output


def seed_all(seed: int) -> None:
    """Seed Python-side RNGs used by torch and numpy."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloader(
    *,
    n_samples: int,
    batch_size: int,
    num_workers: int,
    seed: int,
    dataset_path: str,
    pin_memory: bool,
) -> Tuple[DataLoader, torch.Tensor, torch.Tensor]:
    """Create deterministic FFHQ image dataloader with fixed timesteps/noise."""
    dataset = FFHQlmdb(
        path=str(resolve_repo_path(dataset_path)),
        image_size=IMAGE_SIZE,
        original_resolution=256,
        do_augment=False,
    )
    subset = Subset(dataset, list(range(n_samples)))

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    t_fixed = torch.randint(0, DIFFUSION_T, (n_samples,), generator=gen)
    noise_fixed = torch.randn(
        n_samples,
        3,
        IMAGE_SIZE,
        IMAGE_SIZE,
        generator=gen,
    )

    hessian_dataset = FixedHessianDataset(subset, t_fixed, noise_fixed)
    dataloader = DataLoader(
        hessian_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return dataloader, t_fixed, noise_fixed


def load_fp32(device: torch.device, fp_model_path: str) -> Tuple[Any, nn.Module, Any]:
    """Load FP32 Diff-AE via the Layer 2 loader."""
    l2_loader.DEVICE = device
    base_model, _ = l2_loader.load_fp_model(str(resolve_repo_path(fp_model_path)))
    base_model.to(device)
    base_model.eval()
    ema_model = base_model.ema_model
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad_(True)
    return base_model, ema_model, base_model.sampler


def load_qdiffae(
    device: torch.device,
    fp_model_path: str,
    qat_ckpt_path: str,
) -> Tuple[Any, nn.Module, Any]:
    """Load Q-DiffAE W8A8 via the exact Layer 2 quantized-model loader."""
    l2_loader.DEVICE = device
    base_model, _ = l2_loader.load_quant_model(
        fp_model_path=str(resolve_repo_path(fp_model_path)),
        qat_ckpt_path=str(resolve_repo_path(qat_ckpt_path)),
        num_steps=100,
    )
    base_model.to(device)
    base_model.eval()
    ema_model = base_model.ema_model
    ema_model.eval()
    if hasattr(ema_model, "set_quant_state"):
        ema_model.set_quant_state(True, True)
    if hasattr(ema_model, "set_runtime_mode"):
        ema_model.set_runtime_mode(mode="infer", use_cached_aw=True, clear_cached_aw=True)
    for param in ema_model.parameters():
        param.requires_grad_(True)
    return base_model, ema_model, base_model.sampler


def collect_lora_matched_weight_names(qat_ema_model: nn.Module) -> List[str]:
    """Map Q-DiffAE LoRA module locations back to FP32 module weight names."""
    matched_names: List[str] = []
    for module_name, module in qat_ema_model.named_modules():
        if not isinstance(module, QuantModule_DiffAE_LoRA):
            continue

        # QAT ema_model is QuantModel_DiffAE_LoRA(model=<BeatGANsAutoencModel>),
        # so module names are prefixed with "model.".  FP32 ema_model has the
        # same underlying path without that wrapper prefix.
        fp_module_name = module_name.removeprefix("model.")
        matched_names.append(f"{fp_module_name}.weight")

    return sorted(set(matched_names))


def freeze_fp32_to_matched_weights(
    fp_ema_model: nn.Module,
    matched_weight_names: List[str],
) -> Dict[str, Any]:
    """Freeze FP32 parameters except the original weights matching QAT LoRA locations."""
    matched_set = set(matched_weight_names)
    found_names: List[str] = []
    missing_names = sorted(matched_set)

    for param in fp_ema_model.parameters():
        param.requires_grad_(False)

    missing_set = set(missing_names)
    for name, param in fp_ema_model.named_parameters():
        if name in matched_set:
            param.requires_grad_(True)
            found_names.append(name)
            missing_set.discard(name)

    matched_params = sum(
        p.numel() for name, p in fp_ema_model.named_parameters() if name in set(found_names)
    )
    return {
        "matched_layers": found_names,
        "missing_layers": sorted(missing_set),
        "matched_count": len(found_names),
        "matched_params": int(matched_params),
    }


def validate_lora_matched_param_count(
    matched_params: int,
    qat_params: int,
    allow_mismatch: bool = False,
) -> None:
    """Stop before Hessian if the matched FP32 parameter count is clearly inconsistent."""
    if qat_params <= 0:
        LOGGER.warning("Cannot validate matched parameter count because qat_params=%d", qat_params)
        return

    ratio = matched_params / float(qat_params)
    LOGGER.info(
        "LoRA-matched FP32 params=%d vs QAT grad params=%d (ratio=%.4f)",
        matched_params,
        qat_params,
        ratio,
    )
    if ratio > 2.0 or ratio < 0.5:
        message = (
            "LoRA-matched FP32 parameter count differs from QAT by more than 2x: "
            f"matched={matched_params}, qat={qat_params}, ratio={ratio:.4f}. "
            "This is allowed only when explicitly requested."
        )
        if allow_mismatch:
            LOGGER.warning(message)
        else:
            raise RuntimeError(message)


def prepare_sampler_for_hessian(sampler: Any, disable_fp16_loss: bool) -> None:
    """Use full precision loss evaluation for stable second-order derivatives."""
    if disable_fp16_loss and hasattr(sampler, "conf") and hasattr(sampler.conf, "fp16"):
        sampler.conf.fp16 = False


def count_trainable_params(model: nn.Module) -> int:
    """Count parameters included by PyHessian."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def disable_unused_parameters(
    wrapper_model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[int, int]:
    """Disable params that do not receive gradients for the wrapped epsilon-MSE loss."""
    wrapper_model.to(device)
    wrapper_model.eval()
    wrapper_model.zero_grad(set_to_none=True)

    inputs, targets = next(iter(dataloader))
    del targets
    loss = wrapper_model(inputs.to(device))
    loss.backward()

    disabled = 0
    disabled_params = 0
    for _, param in wrapper_model.named_parameters():
        if param.requires_grad and param.grad is None:
            param.requires_grad_(False)
            disabled += 1
            disabled_params += param.numel()

    wrapper_model.zero_grad(set_to_none=True)
    return disabled, disabled_params


def compute_hessian_metrics(
    *,
    wrapper_model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    eig_iter: int,
    eig_tol: float,
    trace_iter: int,
    trace_tol: float,
) -> Dict[str, float]:
    """Compute top eigenvalue and Hutchinson trace estimate with PyHessian."""
    wrapper_model.to(device)
    wrapper_model.eval()
    criterion = ScalarLossCriterion()
    disabled_tensors, disabled_params = disable_unused_parameters(
        wrapper_model,
        dataloader,
        device,
    )
    if disabled_tensors:
        LOGGER.info(
            "Disabled %d unused parameter tensors (%d scalars) before PyHessian",
            disabled_tensors,
            disabled_params,
        )

    hessian_comp = hessian(
        wrapper_model,
        criterion=criterion,
        dataloader=dataloader,
        cuda=device.type == "cuda",
    )
    top_eigenvalues, _ = hessian_comp.eigenvalues(
        top_n=1,
        maxIter=eig_iter,
        tol=eig_tol,
    )
    trace_samples = hessian_comp.trace(maxIter=trace_iter, tol=trace_tol)

    return {
        "top_eigenvalue": float(top_eigenvalues[0]),
        "trace": float(np.mean(trace_samples)),
        "trace_std": float(np.std(trace_samples)),
        "trace_num_samples": int(len(trace_samples)),
        "n_params_grad": int(count_trainable_params(wrapper_model)),
        "n_param_tensors_disabled_unused": int(disabled_tensors),
        "n_params_disabled_unused": int(disabled_params),
    }


def safe_ratio(numerator: float, denominator: float) -> float:
    """Return numerator / denominator, preserving NaN on zero denominator."""
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def print_results(fp32_results: Dict[str, float], qat_results: Dict[str, float]) -> None:
    """Print terminal comparison table."""
    ratio_eig = safe_ratio(qat_results["top_eigenvalue"], fp32_results["top_eigenvalue"])
    ratio_trace = safe_ratio(qat_results["trace"], fp32_results["trace"])

    print("\n=== PyHessian Sharpness Results ===")
    print(f"{'Model':<20} | {'Top Eigenvalue':>15} | {'Trace':>15}")
    print("-" * 56)
    print(
        f"{'Diff-AE (FP32)':<20} | {fp32_results['top_eigenvalue']:>15.4f} | "
        f"{fp32_results['trace']:>15.4f}"
    )
    print(
        f"{'Q-DiffAE (W8A8)':<20} | {qat_results['top_eigenvalue']:>15.4f} | "
        f"{qat_results['trace']:>15.4f}"
    )
    print(f"{'Ratio (Q/F)':<20} | {ratio_eig:>15.4f} | {ratio_trace:>15.4f}")
    print("\nInterpretation: ratio < 1.0 -> Q-DiffAE has flatter loss landscape")


def print_lora_matched_results(results: Dict[str, Any]) -> None:
    """Print terminal comparison table with the LoRA-matched FP32 control."""
    fp32 = results["diffae_fp32"]
    matched = results["diffae_fp32_lora_matched"]
    qat = results["qdiffae_w8a8"]
    ratio = results["ratio_lora_matched"]

    print("\n=== PyHessian Sharpness Results (with LoRA-Matched Control) ===")
    print(f"{'Model':<30} | {'n_params_grad':>13} | {'Top Eigenvalue':>15} | {'Trace':>15}")
    print("-" * 86)
    print(
        f"{'Diff-AE FP32 (all params)':<30} | {int(fp32['n_params_grad']):>13} | "
        f"{float(fp32['top_eigenvalue']):>15.4f} | {float(fp32['trace']):>15.4f}"
    )
    print(
        f"{'Diff-AE FP32 (LoRA-matched)':<30} | {int(matched['n_params_grad']):>13} | "
        f"{float(matched['top_eigenvalue']):>15.4f} | {float(matched['trace']):>15.4f}"
    )
    print(
        f"{'Q-DiffAE W8A8 (LoRA)':<30} | {int(qat['n_params_grad']):>13} | "
        f"{float(qat['top_eigenvalue']):>15.4f} | {float(qat['trace']):>15.4f}"
    )
    print("-" * 86)
    print(
        f"{'Ratio (Q / F_matched)':<30} | {'':>13} | "
        f"{float(ratio['top_eigenvalue_qdiffae_over_diffae_matched']):>15.4f} | "
        f"{float(ratio['trace_qdiffae_over_diffae_matched']):>15.4f}"
    )


def load_existing_results(output_path: Path) -> Dict[str, Any]:
    """Load the main Layer 1 JSON that will receive the matched-control results."""
    if not output_path.exists():
        raise FileNotFoundError(
            f"Main result JSON not found: {output_path}. "
            "Run the main FP32/QAT PyHessian experiment first."
        )
    with output_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def run_lora_matched_control(args: argparse.Namespace, output_dir: Path, device: torch.device) -> None:
    """Compute FP32 Hessian on original weights corresponding to QAT LoRA module positions."""
    output_path = output_dir / args.output_name
    results = load_existing_results(output_path)
    if "qdiffae_w8a8" not in results:
        raise KeyError("Existing JSON does not contain 'qdiffae_w8a8'")

    dataloader, t_fixed, noise_fixed = build_dataloader(
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        dataset_path=args.dataset_path,
        pin_memory=device.type == "cuda",
    )
    LOGGER.info(
        "Fixed Hessian data prepared for LoRA-matched control: "
        "n=%d, batch_size=%d, t_range=[%d, %d], noise_shape=%s",
        args.n_samples,
        args.batch_size,
        int(t_fixed.min().item()),
        int(t_fixed.max().item()),
        tuple(noise_fixed.shape),
    )

    LOGGER.info("=== Loading Q-DiffAE W8A8 for LoRA location mapping ===")
    qat_base, qat_ema, qat_sampler = load_qdiffae(device, args.fp_model_path, args.qat_ckpt_path)
    del qat_sampler
    matched_weight_names = collect_lora_matched_weight_names(qat_ema)
    qat_grad_params = int(results["qdiffae_w8a8"].get("n_params_grad", 0))
    LOGGER.info("Collected %d LoRA-matched FP32 weight names", len(matched_weight_names))
    for name in matched_weight_names[:20]:
        LOGGER.info("  matched candidate: %s", name)
    if len(matched_weight_names) > 20:
        LOGGER.info("  ... %d more matched candidates", len(matched_weight_names) - 20)

    del qat_ema, qat_base
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    LOGGER.info("=== Loading Diff-AE FP32 for LoRA-matched Hessian ===")
    fp_base, fp_ema, fp_sampler = load_fp32(device, args.fp_model_path)
    mapping_info = freeze_fp32_to_matched_weights(fp_ema, matched_weight_names)
    if mapping_info["missing_layers"]:
        LOGGER.warning(
            "Missing %d mapped FP32 weights; first missing entries: %s",
            len(mapping_info["missing_layers"]),
            mapping_info["missing_layers"][:20],
        )

    LOGGER.info(
        "Matched FP32 weights: %d tensors, %d scalars",
        mapping_info["matched_count"],
        mapping_info["matched_params"],
    )

    if args.lora_matched_dry_run:
        ratio = (
            float(mapping_info["matched_params"]) / float(qat_grad_params)
            if qat_grad_params > 0
            else float("nan")
        )
        LOGGER.info("Dry run requested; not computing PyHessian.")
        print(
            "LoRA-matched dry run: "
            f"{mapping_info['matched_count']} tensors, {mapping_info['matched_params']} params; "
            f"QAT grad params={qat_grad_params}; ratio={ratio:.4f}"
        )
        if qat_grad_params > 0 and (ratio > 2.0 or ratio < 0.5):
            print(
                "WARNING: matched FP32 parameter count differs from QAT by more than 2x. "
                "Inspect mapping before running PyHessian."
            )
        return

    validate_lora_matched_param_count(
        int(mapping_info["matched_params"]),
        qat_grad_params,
        allow_mismatch=args.allow_lora_matched_param_mismatch,
    )

    prepare_sampler_for_hessian(fp_sampler, disable_fp16_loss=not args.keep_fp16_loss)
    fp_wrapper = EpsilonMSEHessianWrapper(
        fp_ema,
        fp_sampler,
        suppress_loss_stdout=not args.show_loss_timesteps,
    )
    matched_results = compute_hessian_metrics(
        wrapper_model=fp_wrapper,
        dataloader=dataloader,
        device=device,
        eig_iter=args.eig_iter,
        eig_tol=args.eig_tol,
        trace_iter=args.trace_iter,
        trace_tol=args.trace_tol,
    )
    matched_results["matched_layers"] = mapping_info["matched_layers"]
    matched_results["missing_layers"] = mapping_info["missing_layers"]
    matched_results["pre_hessian_matched_count"] = int(mapping_info["matched_count"])
    matched_results["pre_hessian_matched_params"] = int(mapping_info["matched_params"])

    qat_results = results["qdiffae_w8a8"]
    results["diffae_fp32_lora_matched"] = matched_results
    results["ratio_lora_matched"] = {
        "top_eigenvalue_qdiffae_over_diffae_matched": round(
            safe_ratio(float(qat_results["top_eigenvalue"]), matched_results["top_eigenvalue"]),
            4,
        ),
        "trace_qdiffae_over_diffae_matched": round(
            safe_ratio(float(qat_results["trace"]), matched_results["trace"]),
            4,
        ),
    }
    results.setdefault("config", {})["lora_matched_control"] = {
        "mapping": "QAT QuantModule_DiffAE_LoRA module path -> FP32 same path .weight",
        "n_candidates_from_qat_lora_modules": len(matched_weight_names),
        "n_matched_fp32_weight_tensors": int(mapping_info["matched_count"]),
        "n_matched_fp32_weight_params": int(mapping_info["matched_params"]),
        "n_missing_fp32_weight_tensors": len(mapping_info["missing_layers"]),
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")

    print_lora_matched_results(results)
    print(f"Updated results saved to {output_path}")

    del fp_wrapper, fp_ema, fp_base, fp_sampler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Layer 1 PyHessian sharpness comparison for Diff-AE vs Q-DiffAE"
    )
    parser.add_argument("--n-samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--eig-iter", type=int, default=100)
    parser.add_argument("--eig-tol", type=float, default=1e-3)
    parser.add_argument("--trace-iter", type=int, default=100)
    parser.add_argument("--trace-tol", type=float, default=1e-3)
    parser.add_argument("--fp-model-path", type=str, default=FP_MODEL_PATH)
    parser.add_argument("--qat-ckpt-path", type=str, default=QAT_CKPT_PATH)
    parser.add_argument("--dataset-path", type=str, default=DATASET_PATH)
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--output-name", type=str, default=OUTPUT_JSON)
    parser.add_argument(
        "--keep-fp16-loss",
        action="store_true",
        help="Keep sampler.conf.fp16 during Hessian loss evaluation.",
    )
    parser.add_argument(
        "--show-loss-timesteps",
        action="store_true",
        help="Do not suppress the timestep prints emitted by training_losses().",
    )
    parser.add_argument(
        "--lora-matched-control",
        action="store_true",
        help="Append FP32 LoRA-matched Hessian control to the existing result JSON.",
    )
    parser.add_argument(
        "--lora-matched-dry-run",
        action="store_true",
        help="Only report LoRA-matched FP32 parameter mapping/counts; no Hessian.",
    )
    parser.add_argument(
        "--allow-lora-matched-param-mismatch",
        action="store_true",
        help="Allow FP32 matched full-weight params to differ from QAT LoRA params by >2x.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = resolve_repo_path(args.output_dir)
    setup_logging(output_dir)
    seed_all(args.seed)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")

    LOGGER.info("args=%s", vars(args))
    if args.lora_matched_control or args.lora_matched_dry_run:
        run_lora_matched_control(args, output_dir, device)
        return

    dataloader, t_fixed, noise_fixed = build_dataloader(
        n_samples=args.n_samples,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        dataset_path=args.dataset_path,
        pin_memory=device.type == "cuda",
    )
    LOGGER.info(
        "Fixed Hessian data prepared: n=%d, batch_size=%d, t_range=[%d, %d], noise_shape=%s",
        args.n_samples,
        args.batch_size,
        int(t_fixed.min().item()),
        int(t_fixed.max().item()),
        tuple(noise_fixed.shape),
    )

    LOGGER.info("=== Loading Diff-AE FP32 ===")
    fp_base, fp_ema, fp_sampler = load_fp32(device, args.fp_model_path)
    prepare_sampler_for_hessian(fp_sampler, disable_fp16_loss=not args.keep_fp16_loss)
    fp_wrapper = EpsilonMSEHessianWrapper(
        fp_ema,
        fp_sampler,
        suppress_loss_stdout=not args.show_loss_timesteps,
    )
    fp32_results = compute_hessian_metrics(
        wrapper_model=fp_wrapper,
        dataloader=dataloader,
        device=device,
        eig_iter=args.eig_iter,
        eig_tol=args.eig_tol,
        trace_iter=args.trace_iter,
        trace_tol=args.trace_tol,
    )
    LOGGER.info(
        "FP32: top_eig=%.6f, trace=%.6f",
        fp32_results["top_eigenvalue"],
        fp32_results["trace"],
    )

    del fp_wrapper, fp_ema, fp_base, fp_sampler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    LOGGER.info("=== Loading Q-DiffAE W8A8 ===")
    qat_base, qat_ema, qat_sampler = load_qdiffae(device, args.fp_model_path, args.qat_ckpt_path)
    prepare_sampler_for_hessian(qat_sampler, disable_fp16_loss=not args.keep_fp16_loss)
    qat_wrapper = EpsilonMSEHessianWrapper(
        qat_ema,
        qat_sampler,
        suppress_loss_stdout=not args.show_loss_timesteps,
    )
    qat_results = compute_hessian_metrics(
        wrapper_model=qat_wrapper,
        dataloader=dataloader,
        device=device,
        eig_iter=args.eig_iter,
        eig_tol=args.eig_tol,
        trace_iter=args.trace_iter,
        trace_tol=args.trace_tol,
    )
    LOGGER.info(
        "QAT: top_eig=%.6f, trace=%.6f",
        qat_results["top_eigenvalue"],
        qat_results["trace"],
    )

    ratio_eig = safe_ratio(qat_results["top_eigenvalue"], fp32_results["top_eigenvalue"])
    ratio_trace = safe_ratio(qat_results["trace"], fp32_results["trace"])
    output_path = output_dir / args.output_name

    results = {
        "experiment": "Layer1_PyHessian_Sharpness",
        "date": datetime.now().strftime("%Y-%m-%d"),
        "config": {
            "n_samples": args.n_samples,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "loss": "epsilon_prediction_MSE",
            "target": "ema_model",
            "T": DIFFUSION_T,
            "dataset": str(resolve_repo_path(args.dataset_path)),
            "image_size": IMAGE_SIZE,
            "do_augment": False,
            "pixel_range": "[-1, 1]",
            "eig_iter": args.eig_iter,
            "eig_tol": args.eig_tol,
            "trace_iter": args.trace_iter,
            "trace_tol": args.trace_tol,
            "fp16_loss_enabled": bool(args.keep_fp16_loss),
            "fp_model_path": str(resolve_repo_path(args.fp_model_path)),
            "qat_ckpt_path": str(resolve_repo_path(args.qat_ckpt_path)),
        },
        "diffae_fp32": fp32_results,
        "qdiffae_w8a8": qat_results,
        "ratio": {
            "top_eigenvalue_qdiffae_over_diffae": round(ratio_eig, 4),
            "trace_qdiffae_over_diffae": round(ratio_trace, 4),
        },
        "interpretation": "ratio < 1.0 supports flat minima hypothesis",
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        f.write("\n")

    print_results(fp32_results, qat_results)
    print(f"Results saved to {output_path}")

    del qat_wrapper, qat_ema, qat_base, qat_sampler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
