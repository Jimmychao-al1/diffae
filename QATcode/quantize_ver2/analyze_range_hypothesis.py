#!/usr/bin/env python3
"""
analyze_range_hypothesis.py — Weight / Activation Range Hypothesis Analysis

Research question: Does INT8 quantisation precision suffice to represent the
effective weight distribution of Q-DiffAE layers without meaningfully
changing its structure?

Modes
-----
  weight      Static per-layer weight range statistics              [implemented]
  activation  Per-timestep activation range + quant error via hooks [implemented]
  weight-error Round-trip quantisation error analysis               [placeholder]
  crossref    Correlate range stats with Stage-0 FID sensitivity    [placeholder]

Usage
-----
    cd /home/jimmy/diffae
    python -m QATcode.quantize_ver2.analyze_range_hypothesis \\
        --mode weight \\
        --ckpt_path QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best.pth \\
        --output_dir QATcode/quantize_ver2/range_analysis_results \\
        --device cuda:0
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import kurtosis as scipy_kurtosis

# ---------------------------------------------------------------------------
# Path bootstrap (same pattern as analyze_weight_histogram.py)
# ---------------------------------------------------------------------------
sys.path.append(".")
sys.path.append("./model")

from QATcode.quantize_ver2.quant_layer_v2 import TemporalActivationQuantizer
from QATcode.quantize_ver2.quant_model_lora_v2 import QuantModule_DiffAE_LoRA
from QATcode.quantize_ver2.baseline_quant_analysis.pred_xstart_quantile_analysis import (
    CONFIG,
    _compute_latent_cond,
    _load_quant_and_ema_from_ckpt,
    _make_noise_banks,
    create_float_quantized_model,
    load_calibration_data,
    load_diffae_model,
)

LOGGER = logging.getLogger("analyze_range_hypothesis")

EXPECTED_LAYERS = 140

# ---------------------------------------------------------------------------
# Weight helpers  (mirrored from analyze_weight_histogram.py)
# ---------------------------------------------------------------------------


@torch.no_grad()
def _compute_a_w(w: torch.Tensor) -> torch.Tensor:
    """Per-channel absmax scale — mirrors QuantModule_DiffAE_LoRA._compute_a_w().

    Returns a tensor of shape [Cout, 1, 1, 1] (Conv) or [Cout, 1] (Linear)
    with a small epsilon for numerical stability.
    """
    if w.dim() == 4:  # Conv  [Cout, Cin, kH, kW]
        return w.abs().amax(dim=(1, 2, 3), keepdim=True) + 1e-8
    elif w.dim() == 2:  # Linear [Cout, Cin]
        return w.abs().amax(dim=1, keepdim=True) + 1e-8
    else:
        return w.abs().amax().unsqueeze(0) + 1e-8  # fallback


@torch.no_grad()
def _compute_lora_weight(mod: QuantModule_DiffAE_LoRA) -> torch.Tensor:
    """Reconstruct LoRA delta using the same identity-trick as forward().

    Mirrors the logic in analyze_weight_histogram.py::_compute_lora_weight().
    """
    device = mod.org_weight.device
    if mod.fwd_func is F.linear:
        E = torch.eye(mod.org_weight.shape[1], device=device)
        lora_weight = mod.loraB(mod.loraA(E)).T  # [Cout, Cin]
    else:
        lora_weight = (
            mod.loraB.weight.squeeze(-1).squeeze(-1)  # [Cout, rank]
            @ mod.loraA.weight.permute(2, 3, 0, 1)   # [kH, kW, rank, Cin]
        ).permute(2, 3, 0, 1)  # [Cout, Cin, kH, kW]
    return lora_weight.to(device)


@torch.no_grad()
def _get_effective_weight(mod: QuantModule_DiffAE_LoRA) -> torch.Tensor:
    """Return float32 effective weight  w_eff = org_weight + LoRA_delta."""
    w = mod.org_weight.detach().float()
    lora = _compute_lora_weight(mod)
    return (w + lora).float()


# ---------------------------------------------------------------------------
# Per-layer statistics
# ---------------------------------------------------------------------------


def _per_channel_stats(
    w_eff: torch.Tensor,
) -> Tuple[float, float, float]:
    """Compute per-channel delta_int8 / std statistics.

    Weight quantiser uses channel_wise=True (absmax per output channel).
    Returns (pch_delta_int8_mean, pch_delta_over_std_mean, pch_delta_over_std_max).
    """
    # a_w shape: [Cout, 1, ...] — squeeze to 1-D [Cout]
    a_w = _compute_a_w(w_eff).squeeze()
    if a_w.dim() == 0:
        a_w = a_w.unsqueeze(0)

    # Per-channel range = 2 * absmax  (symmetric quantisation)
    pch_delta_int8 = (2.0 * a_w / 254.0).cpu().numpy()  # [Cout]

    # Per-channel std: std of each output-channel slice
    out_ch = w_eff.shape[0]
    w_flat_per_ch = w_eff.reshape(out_ch, -1).cpu().float().numpy()
    pch_std = w_flat_per_ch.std(axis=1) + 1e-12  # [Cout]

    pch_dos = pch_delta_int8 / pch_std  # [Cout]

    return (
        float(pch_delta_int8.mean()),
        float(pch_dos.mean()),
        float(pch_dos.max()),
    )


def compute_layer_stats(
    idx: int,
    name: str,
    mod: QuantModule_DiffAE_LoRA,
) -> Optional[Dict]:
    """Compute all per-layer statistics for one QuantModule_DiffAE_LoRA.

    Returns a dict of scalar values, or None if extraction fails.
    """
    try:
        w_eff = _get_effective_weight(mod)
    except Exception as exc:
        LOGGER.warning("[%3d] %s — could not extract effective weight: %s", idx + 1, name, exc)
        return None

    w_np = w_eff.cpu().float().numpy().flatten()

    w_min = float(w_np.min())
    w_max = float(w_np.max())
    w_range = w_max - w_min
    w_mean = float(w_np.mean())
    w_std = float(w_np.std()) + 1e-12
    w_kurt = float(scipy_kurtosis(w_np, bias=True))  # excess kurtosis

    # Per-tensor INT8 step  (symmetric, 254 levels)
    delta_int8 = w_range / 254.0
    delta_over_std = delta_int8 / w_std

    # Precision utilisation metrics
    # precision_ratio: INT8 effective bits (≈8) as fraction of FP32 mantissa bits (23)
    precision_ratio = math.log2(max(w_range / delta_int8, 1e-12)) / 23.0
    # fp32_precision: approximate FP32 ULP at this magnitude
    fp32_precision = w_range * 2.0 ** -23
    # int8_over_fp32: how many FP32 steps fit in one INT8 step
    int8_over_fp32 = delta_int8 / (fp32_precision + 1e-30)

    # Per-channel stats (channel_wise=True in weight quantiser)
    pch_delta_int8_mean, pch_delta_over_std_mean, pch_delta_over_std_max = _per_channel_stats(w_eff)

    row = {
        "layer_name": name,
        "layer_idx": idx,
        "numel": int(w_np.size),
        "w_min": w_min,
        "w_max": w_max,
        "w_range": w_range,
        "w_mean": w_mean,
        "w_std": float(w_np.std()),
        "w_kurtosis": w_kurt,
        "delta_int8": delta_int8,
        "delta_over_std": delta_over_std,
        "precision_ratio": precision_ratio,
        "fp32_precision": fp32_precision,
        "int8_over_fp32": int8_over_fp32,
        # per-channel extension
        "pch_delta_int8_mean": pch_delta_int8_mean,
        "pch_delta_over_std_mean": pch_delta_over_std_mean,
        "pch_delta_over_std_max": pch_delta_over_std_max,
    }

    LOGGER.debug(
        "[%3d/%d] %-60s  range=%.4f  Δ/σ=%.4f",
        idx + 1, EXPECTED_LAYERS, name, w_range, delta_over_std,
    )
    print(
        f"[{idx + 1:3d}/{EXPECTED_LAYERS}] {name:<60s}  "
        f"range={w_range:.4f}  Δ/σ={delta_over_std:.4f}"
    )
    return row


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

_CSV_FIELDS = [
    "layer_idx", "layer_name", "numel",
    "w_min", "w_max", "w_range", "w_mean", "w_std", "w_kurtosis",
    "delta_int8", "delta_over_std",
    "precision_ratio", "fp32_precision", "int8_over_fp32",
    "pch_delta_int8_mean", "pch_delta_over_std_mean", "pch_delta_over_std_max",
]


def save_csv(rows: List[Dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: r["layer_idx"]):
            writer.writerow({k: row[k] for k in _CSV_FIELDS})
    LOGGER.info("CSV saved → %s", path)


def save_json_summary(rows: List[Dict], path: Path) -> None:
    dos = np.array([r["delta_over_std"] for r in rows])
    w_ranges = np.array([r["w_range"] for r in rows])
    w_stds = np.array([r["w_std"] for r in rows])
    total_params = int(sum(r["numel"] for r in rows))

    summary = {
        "total_layers": len(rows),
        "total_params": total_params,
        "delta_over_std": {
            "mean": float(dos.mean()),
            "median": float(np.median(dos)),
            "min": float(dos.min()),
            "max": float(dos.max()),
            "pct_below_0.05": float((dos < 0.05).mean() * 100),
            "pct_below_0.10": float((dos < 0.10).mean() * 100),
        },
        "w_range": {
            "mean": float(w_ranges.mean()),
            "median": float(np.median(w_ranges)),
            "min": float(w_ranges.min()),
            "max": float(w_ranges.max()),
        },
        "w_std": {
            "mean": float(w_stds.mean()),
            "median": float(np.median(w_stds)),
            "min": float(w_stds.min()),
            "max": float(w_stds.max()),
        },
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    LOGGER.info("JSON summary saved → %s", path)


def print_console_summary(rows: List[Dict]) -> None:
    dos = np.array([r["delta_over_std"] for r in rows])
    n = len(rows)

    print()
    print("=" * 70)
    print(f"  Weight Range Analysis — {n} layers confirmed")
    print("=" * 70)
    print(f"  delta_over_std  (Δ/σ = INT8_step / w_std)")
    print(f"    mean   : {dos.mean():.4f}")
    print(f"    median : {np.median(dos):.4f}")
    print(f"    min    : {dos.min():.4f}")
    print(f"    max    : {dos.max():.4f}")
    print(f"    < 0.05 : {(dos < 0.05).mean() * 100:.1f}%  of layers")
    print(f"    < 0.10 : {(dos < 0.10).mean() * 100:.1f}%  of layers")
    print()

    sorted_by_range = sorted(rows, key=lambda r: r["w_range"], reverse=True)
    print("  Top-5 widest-range layers:")
    for r in sorted_by_range[:5]:
        print(f"    [{r['layer_idx']:3d}] {r['layer_name']:<55s}  range={r['w_range']:.4f}")
    print()
    print("  Top-5 narrowest-range layers:")
    for r in sorted_by_range[-5:]:
        print(f"    [{r['layer_idx']:3d}] {r['layer_name']:<55s}  range={r['w_range']:.4f}")
    print("=" * 70)
    print()


# ---------------------------------------------------------------------------
# Model loading  (exact pattern from analyze_weight_histogram.py)
# ---------------------------------------------------------------------------


def _load_model(ckpt_path: str, device: torch.device):
    """Load QAT model and return analysis_model ready for weight inspection.

    Follows the exact 5-step pattern from analyze_weight_histogram.py:
      1. Load base Diff-AE model
      2. Create float fake-quant model (wraps diffusion model with LoRA)
      3. Calibrate TemporalActivationQuantizer scales (single forward pass)
      4. Load QAT TT checkpoint via _load_quant_and_ema_from_ckpt
      5. Return base_model.ema_model with set_runtime_mode("infer")
    """
    LOGGER.info("Loading base Diff-AE model …")
    base_model = load_diffae_model()
    diffusion_model = base_model.ema_model

    quant_model = create_float_quantized_model(
        diffusion_model,
        num_steps=CONFIG.NUM_DIFFUSION_STEPS,
        lora_rank=32,
        mode="train",
    )
    quant_model.to(device).eval()

    LOGGER.info("Loading calibration data …")
    cali_images, cali_t, cali_y = load_calibration_data()

    quant_model.set_first_last_layer_to_8bit()
    quant_model.set_quant_state(True, True)

    if hasattr(quant_model, "set_runtime_mode"):
        quant_model.set_runtime_mode(mode="train", use_cached_aw=False, clear_cached_aw=True)

    LOGGER.info("Running calibration forward pass (32 samples) …")
    with torch.no_grad():
        _ = quant_model(
            x=cali_images[:32].to(device),
            t=cali_t[:32].to(device),
            cond=cali_y[:32].to(device),
        )

    resolved_ckpt = ckpt_path or CONFIG.BEST_CKPT_PATH_100
    LOGGER.info("Loading QAT checkpoint: %s", resolved_ckpt)
    ckpt = torch.load(resolved_ckpt, map_location="cpu", weights_only=False)
    _load_quant_and_ema_from_ckpt(base_model, quant_model, ckpt)
    LOGGER.info("Checkpoint loaded successfully")

    analysis_model = base_model.ema_model
    analysis_model.to(device).eval()

    if hasattr(analysis_model, "set_runtime_mode"):
        analysis_model.set_runtime_mode(mode="infer", use_cached_aw=True, clear_cached_aw=True)

    return analysis_model


def _load_model_for_inference(
    ckpt_path: Optional[str],
    device: torch.device,
    num_steps: int = 100,
):
    """Load QAT model for inference (activation analysis).

    Extended version of _load_model that also calls base_model.setup() and
    base_model.train_dataloader() to initialise conds_mean / conds_std, which
    are required by _compute_latent_cond for latent-diffusion conditioning.
    Returns (base_model, analysis_model).
    """
    LOGGER.info("Loading base Diff-AE model (inference mode) …")
    base_model = load_diffae_model()
    base_model.to(device)
    base_model.eval()

    try:
        base_model.setup()
        base_model.train_dataloader()
        LOGGER.info("base_model.setup() + train_dataloader() done")
    except Exception as exc:
        LOGGER.warning(
            "setup()/train_dataloader() failed (%s) — conds_mean/std will default to 0/1", exc
        )

    diffusion_model = base_model.ema_model
    quant_model = create_float_quantized_model(
        diffusion_model,
        num_steps=num_steps,
        lora_rank=32,
        mode="train",
    )
    quant_model.to(device).eval()

    LOGGER.info("Loading calibration data …")
    cali_images, cali_t, cali_y = load_calibration_data()

    quant_model.set_first_last_layer_to_8bit()
    quant_model.set_quant_state(True, True)

    if hasattr(quant_model, "set_runtime_mode"):
        quant_model.set_runtime_mode(mode="train", use_cached_aw=False, clear_cached_aw=True)

    LOGGER.info("Running calibration forward pass …")
    with torch.no_grad():
        _ = quant_model(
            x=cali_images[:32].to(device),
            t=cali_t[:32].to(device),
            cond=cali_y[:32].to(device),
        )

    resolved_ckpt = ckpt_path or CONFIG.BEST_CKPT_PATH_100
    LOGGER.info("Loading QAT checkpoint: %s", resolved_ckpt)
    ckpt = torch.load(resolved_ckpt, map_location="cpu", weights_only=False)
    _load_quant_and_ema_from_ckpt(base_model, quant_model, ckpt)
    LOGGER.info("Checkpoint loaded successfully")

    analysis_model = base_model.ema_model
    analysis_model.to(device).eval()

    if hasattr(analysis_model, "set_runtime_mode"):
        analysis_model.set_runtime_mode(mode="infer", use_cached_aw=True, clear_cached_aw=True)

    return base_model, analysis_model


def _patch_t_tracking(model: torch.nn.Module, t_ref: Dict) -> object:
    """Wrap model.forward to intercept the 't' argument and store its value.

    Replicated from per_timestep_act_quant_analysis._patch_t_tracking to avoid
    importing that module (which triggers matplotlib.use("Agg")).
    Returns the original forward callable for cleanup.
    """
    orig_forward = model.forward

    def _wrapped(*args, **kwargs):
        t_val = kwargs.get("t", None)
        if t_val is None and len(args) >= 2:
            t_val = args[1]
        if torch.is_tensor(t_val) and t_val.numel() > 0:
            t_ref["t"] = int(t_val.reshape(-1)[0].item())
        return orig_forward(*args, **kwargs)

    model.forward = _wrapped  # type: ignore[method-assign]
    return orig_forward


# ---------------------------------------------------------------------------
# Activation mode helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_activation_stats(x: torch.Tensor) -> Dict:
    """Compute summary statistics for a pre-quant activation tensor."""
    x_flat = x.detach().float().flatten()
    return {
        "act_min": float(x_flat.min().item()),
        "act_max": float(x_flat.max().item()),
        "act_range": float((x_flat.max() - x_flat.min()).item()),
        "act_mean": float(x_flat.mean().item()),
        "act_std": float(x_flat.std().item()),
        "act_absmax": float(x_flat.abs().max().item()),
        "numel": int(x_flat.numel()),
    }


@torch.no_grad()
def compute_error_stats(x_pre: torch.Tensor, x_dequant: torch.Tensor) -> Dict:
    """Compute round-trip quantisation error statistics."""
    x_pre_f = x_pre.detach().float()
    x_dq_f = x_dequant.detach().float()
    diff = x_pre_f - x_dq_f
    abs_diff = diff.abs()
    x_abs = x_pre_f.abs()
    rel_diff = abs_diff / (x_abs + 1e-10)
    return {
        "err_rmse": float(diff.pow(2).mean().sqrt().item()),
        "err_mae": float(abs_diff.mean().item()),
        "err_max": float(abs_diff.max().item()),
        "err_rel_mean": float(rel_diff.mean().item()),
        "err_rel_l2": float(
            (diff.pow(2).sum() / (x_pre_f.pow(2).sum() + 1e-10)).sqrt().item()
        ),
    }


def _make_act_hooks(
    layer_name: str,
    acc_act: Dict,
    acc_err: Dict,
    scale_map: Dict,
):
    """Factory for (pre_hook, post_hook) pair for one QuantModule_DiffAE_LoRA.

    Uses a shared dict `pre_store` as a cross-hook scratch pad.
    The pre_hook runs before TemporalActivationQuantizer.forward() (current_step
    not yet decremented), the post_hook runs after (current_step already
    decremented).  We therefore save step_idx in the pre_hook and reuse it in
    the post_hook.
    """
    pre_store: Dict = {}

    def _pre_hook(module: torch.nn.Module, inputs: Tuple) -> None:
        if not inputs or not torch.is_tensor(inputs[0]):
            return
        x = inputs[0].detach()
        # Read step_idx BEFORE forward() decrements it
        step_idx = int(max(0, min(module.current_step, module.total_steps - 1)))

        stats = compute_activation_stats(x)
        acc_act.setdefault(layer_name, {}).setdefault(step_idx, []).append(stats)

        # Record scale from scale_list (same value for all samples at same step)
        scale_val = float(module.scale_list[step_idx].item())
        scale_map.setdefault(layer_name, {})[step_idx] = scale_val

        # Stash pre-quant tensor + step_idx for the post_hook
        pre_store["x"] = x
        pre_store["step_idx"] = step_idx

    def _post_hook(module: torch.nn.Module, inputs: Tuple, output: torch.Tensor) -> None:
        x_pre = pre_store.get("x")
        step_idx = pre_store.get("step_idx")
        if x_pre is None or step_idx is None:
            return
        # output is normalized fake-quant ∈ [-1, 1]; dequant = output * scale
        scale = module.scale_list[step_idx].detach().clamp(min=1e-8)
        x_dequant = output.detach() * scale
        err_stats = compute_error_stats(x_pre, x_dequant)
        acc_err.setdefault(layer_name, {}).setdefault(step_idx, []).append(err_stats)
        pre_store.clear()

    return _pre_hook, _post_hook


def _aggregate_act_stats(
    lora_modules: List[Tuple[str, object]],
    acc_act: Dict,
    acc_err: Dict,
    scale_map: Dict,
) -> List[Dict]:
    """Aggregate per-sample statistics into one row per (layer, step)."""
    rows: List[Dict] = []
    for layer_idx, (name, _mod) in enumerate(lora_modules):
        step_data = acc_act.get(name, {})
        err_data = acc_err.get(name, {})
        scales = scale_map.get(name, {})
        for step_idx in sorted(step_data.keys()):
            stat_list = step_data[step_idx]
            err_list = err_data.get(step_idx, [])

            # Aggregation rules per prompt specification
            act_min = min(s["act_min"] for s in stat_list)
            act_max = max(s["act_max"] for s in stat_list)
            act_range = float(np.mean([s["act_range"] for s in stat_list]))
            act_mean = float(np.mean([s["act_mean"] for s in stat_list]))
            act_std = float(np.mean([s["act_std"] for s in stat_list]))
            act_absmax = max(s["act_absmax"] for s in stat_list)

            err_rmse = float(np.mean([e["err_rmse"] for e in err_list])) if err_list else float("nan")
            err_mae = float(np.mean([e["err_mae"] for e in err_list])) if err_list else float("nan")
            err_rel_l2 = float(np.mean([e["err_rel_l2"] for e in err_list])) if err_list else float("nan")

            # scale for this (layer, step): same across samples (learned parameter)
            scale_val = scales.get(step_idx, float("nan"))
            # activation INT8 step = scale / 127  (symmetric: range [-scale, scale], 254 levels)
            delta_int8 = scale_val / 127.0 if not math.isnan(scale_val) else float("nan")
            act_std_safe = act_std if act_std > 1e-12 else float("nan")
            delta_over_std = delta_int8 / act_std_safe if (
                not math.isnan(delta_int8) and not math.isnan(act_std_safe)
            ) else float("nan")

            rows.append({
                "layer_idx": layer_idx,
                "layer_name": name,
                "step": step_idx,
                "act_min": act_min,
                "act_max": act_max,
                "act_range": act_range,
                "act_mean": act_mean,
                "act_std": act_std,
                "act_absmax": act_absmax,
                "delta_int8": delta_int8,
                "delta_over_std": delta_over_std,
                "err_rmse": err_rmse,
                "err_mae": err_mae,
                "err_rel_l2": err_rel_l2,
            })

    # Sort: layer_idx ASC, step DESC (step 99=high-noise first, step 0=clean last)
    rows.sort(key=lambda r: (r["layer_idx"], -r["step"]))
    return rows


_ACT_CSV_FIELDS = [
    "layer_idx", "layer_name", "step",
    "act_min", "act_max", "act_range", "act_mean", "act_std", "act_absmax",
    "delta_int8", "delta_over_std",
    "err_rmse", "err_mae", "err_rel_l2",
]


def _save_activation_outputs(rows: List[Dict], output_dir: Path, args: argparse.Namespace) -> None:
    """Write main CSV, per-layer CSV, per-step CSV, and JSON summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 6.1  Main 140×100 CSV ────────────────────────────────────────────────
    main_csv = output_dir / "activation_range_stats.csv"
    with open(main_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_ACT_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in _ACT_CSV_FIELDS})
    LOGGER.info("Main CSV saved → %s  (%d rows)", main_csv, len(rows))

    # ── 6.2  Per-layer summary (140 rows) ────────────────────────────────────
    by_layer: Dict[str, List[Dict]] = {}
    for row in rows:
        by_layer.setdefault(row["layer_name"], []).append(row)

    layer_rows: List[Dict] = []
    for layer_name, lrows in by_layer.items():
        layer_idx = lrows[0]["layer_idx"]
        stds = [r["act_std"] for r in lrows]
        dos = [r["delta_over_std"] for r in lrows if not math.isnan(r["delta_over_std"])]
        rel_l2s = [r["err_rel_l2"] for r in lrows if not math.isnan(r["err_rel_l2"])]
        act_std_mean = float(np.mean(stds))
        act_std_cv = float(np.std(stds) / (act_std_mean + 1e-12))
        layer_rows.append({
            "layer_idx": layer_idx,
            "layer_name": layer_name,
            "act_std_mean": act_std_mean,
            "act_std_cv": act_std_cv,
            "delta_over_std_mean": float(np.mean(dos)) if dos else float("nan"),
            "delta_over_std_max": float(np.max(dos)) if dos else float("nan"),
            "err_rel_l2_mean": float(np.mean(rel_l2s)) if rel_l2s else float("nan"),
            "err_rel_l2_max": float(np.max(rel_l2s)) if rel_l2s else float("nan"),
        })
    layer_rows.sort(key=lambda r: r["layer_idx"])

    layer_csv = output_dir / "activation_range_per_layer.csv"
    layer_fields = [
        "layer_idx", "layer_name",
        "act_std_mean", "act_std_cv",
        "delta_over_std_mean", "delta_over_std_max",
        "err_rel_l2_mean", "err_rel_l2_max",
    ]
    with open(layer_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=layer_fields)
        writer.writeheader()
        writer.writerows(layer_rows)
    LOGGER.info("Per-layer CSV saved → %s", layer_csv)

    # ── 6.3  Per-step summary (100 rows) ─────────────────────────────────────
    by_step: Dict[int, List[Dict]] = {}
    for row in rows:
        by_step.setdefault(row["step"], []).append(row)

    step_rows: List[Dict] = []
    for step_idx, srows in by_step.items():
        dos = [r["delta_over_std"] for r in srows if not math.isnan(r["delta_over_std"])]
        rel_l2s = [r["err_rel_l2"] for r in srows if not math.isnan(r["err_rel_l2"])]
        step_rows.append({
            "step": step_idx,
            "delta_over_std_mean_across_layers": float(np.mean(dos)) if dos else float("nan"),
            "err_rel_l2_mean_across_layers": float(np.mean(rel_l2s)) if rel_l2s else float("nan"),
        })
    step_rows.sort(key=lambda r: -r["step"])  # step 99 (high noise) first

    step_csv = output_dir / "activation_range_per_step.csv"
    step_fields = ["step", "delta_over_std_mean_across_layers", "err_rel_l2_mean_across_layers"]
    with open(step_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=step_fields)
        writer.writeheader()
        writer.writerows(step_rows)
    LOGGER.info("Per-step CSV saved → %s", step_csv)

    # ── 6.4  JSON summary ────────────────────────────────────────────────────
    all_dos = [r["delta_over_std"] for r in rows if not math.isnan(r["delta_over_std"])]
    all_rel_l2 = [r["err_rel_l2"] for r in rows if not math.isnan(r["err_rel_l2"])]

    worst_dos_row = max(rows, key=lambda r: r["delta_over_std"] if not math.isnan(r["delta_over_std"]) else -1)
    worst_rel_l2_row = max(rows, key=lambda r: r["err_rel_l2"] if not math.isnan(r["err_rel_l2"]) else -1)

    cvs = [lr["act_std_cv"] for lr in layer_rows]
    most_unstable = layer_rows[int(np.argmax(cvs))] if cvs else {}

    summary = {
        "n_samples": int(args.n_samples),
        "ddim_steps": int(args.ddim_steps),
        "total_layers": len(by_layer),
        "total_datapoints": len(rows),
        "delta_over_std": {
            "global_mean": float(np.mean(all_dos)) if all_dos else float("nan"),
            "global_median": float(np.median(all_dos)) if all_dos else float("nan"),
            "pct_below_0.05": float((np.array(all_dos) < 0.05).mean() * 100) if all_dos else float("nan"),
            "pct_below_0.10": float((np.array(all_dos) < 0.10).mean() * 100) if all_dos else float("nan"),
            "worst_layer": worst_dos_row.get("layer_name", ""),
            "worst_layer_value": worst_dos_row.get("delta_over_std", float("nan")),
        },
        "err_rel_l2": {
            "global_mean": float(np.mean(all_rel_l2)) if all_rel_l2 else float("nan"),
            "global_max": float(np.max(all_rel_l2)) if all_rel_l2 else float("nan"),
            "worst_layer": worst_rel_l2_row.get("layer_name", ""),
            "worst_step": int(worst_rel_l2_row.get("step", -1)),
        },
        "act_std_temporal_stability": {
            "cv_mean_across_layers": float(np.mean(cvs)) if cvs else float("nan"),
            "cv_max": float(np.max(cvs)) if cvs else float("nan"),
            "most_unstable_layer": most_unstable.get("layer_name", ""),
        },
    }

    json_path = output_dir / "activation_range_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    LOGGER.info("JSON summary saved → %s", json_path)


def _print_activation_summary(rows: List[Dict], n_samples: int, ddim_steps: int) -> None:
    all_dos = [r["delta_over_std"] for r in rows if not math.isnan(r["delta_over_std"])]
    all_rel_l2 = [r["err_rel_l2"] for r in rows if not math.isnan(r["err_rel_l2"])]
    worst_rel_l2_row = max(rows, key=lambda r: r["err_rel_l2"] if not math.isnan(r["err_rel_l2"]) else -1) if all_rel_l2 else {}

    dos_arr = np.array(all_dos)
    print()
    print("=" * 70)
    print(f"  Activation Range Analysis — {n_samples} samples × {ddim_steps} steps")
    print("=" * 70)
    print(f"  Activation delta_over_std  (Δ/σ = INT8_step / act_std)")
    print(f"    mean   : {dos_arr.mean():.4f}")
    print(f"    median : {np.median(dos_arr):.4f}")
    print(f"    < 0.05 : {(dos_arr < 0.05).mean() * 100:.1f}%  of (layer, step) pairs")
    print(f"    < 0.10 : {(dos_arr < 0.10).mean() * 100:.1f}%  of (layer, step) pairs")
    if all_rel_l2:
        print(f"  Quant error rel_L2:")
        print(f"    mean   : {np.mean(all_rel_l2):.6f}")
        print(f"    max    : {np.max(all_rel_l2):.6f}  "
              f"(layer={worst_rel_l2_row.get('layer_name','?')}, step={worst_rel_l2_row.get('step','?')})")
    print("=" * 70)
    print()


# ---------------------------------------------------------------------------
# Mode implementations
# ---------------------------------------------------------------------------


def run_weight_mode(args: argparse.Namespace) -> None:
    """--mode weight: static per-layer weight range analysis (no forward pass)."""
    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    analysis_model = _load_model(args.ckpt_path, device)

    lora_modules = [
        (name, mod)
        for name, mod in analysis_model.named_modules()
        if isinstance(mod, QuantModule_DiffAE_LoRA)
    ]
    assert len(lora_modules) == EXPECTED_LAYERS, (
        f"Expected {EXPECTED_LAYERS} QuantModule_DiffAE_LoRA layers, got {len(lora_modules)}"
    )
    LOGGER.info("Found %d QuantModule_DiffAE_LoRA layers — starting analysis …", len(lora_modules))

    rows: List[Dict] = []

    with torch.no_grad():
        for idx, (name, mod) in enumerate(lora_modules):
            row = compute_layer_stats(idx, name, mod)
            if row is not None:
                rows.append(row)

    if not rows:
        LOGGER.error("No layers produced valid statistics — aborting.")
        sys.exit(1)

    csv_path = output_dir / "weight_range_stats.csv"
    json_path = output_dir / "weight_range_summary.json"

    save_csv(rows, csv_path)
    save_json_summary(rows, json_path)
    print_console_summary(rows)

    print(f"  CSV  → {csv_path}")
    print(f"  JSON → {json_path}")


def run_activation_mode(args: argparse.Namespace) -> None:
    """--mode activation: per-timestep activation range + quant error via hooks.

    Runs DDIM sampling for n_samples images, with forward pre+post hooks on
    every TemporalActivationQuantizer in the 140 QuantModule_DiffAE_LoRA layers.
    Collects activation stats (pre-quant) and round-trip error stats (post-quant)
    for each (layer, timestep-step) pair, then writes 4 output files.
    """
    device = torch.device(
        args.device if (torch.cuda.is_available() or "cpu" in args.device) else "cpu"
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_samples: int = int(args.n_samples)
    ddim_steps: int = int(args.ddim_steps)

    # ── 1. Load model (inference variant with conds_mean / conds_std) ─────────
    base_model, analysis_model = _load_model_for_inference(
        args.ckpt_path, device, num_steps=ddim_steps
    )

    # ── 2. Build samplers from config ─────────────────────────────────────────
    conf = base_model.conf.clone()
    sampler = conf._make_diffusion_conf(T=ddim_steps).make_sampler()
    latent_sampler = conf._make_latent_diffusion_conf(T=ddim_steps).make_sampler()

    # Resolve conds_mean / conds_std (may not exist if setup() failed)
    style_ch: int = int(conf.style_ch)
    conds_mean: torch.Tensor = getattr(
        base_model, "conds_mean", torch.zeros(style_ch, device=device)
    )
    conds_std: torch.Tensor = getattr(
        base_model, "conds_std", torch.ones(style_ch, device=device)
    )

    # ── 3. Collect all 140 QuantModule_DiffAE_LoRA layers ─────────────────────
    lora_modules: List[Tuple[str, QuantModule_DiffAE_LoRA]] = [
        (name, mod)
        for name, mod in analysis_model.named_modules()
        if isinstance(mod, QuantModule_DiffAE_LoRA)
    ]
    assert len(lora_modules) == EXPECTED_LAYERS, (
        f"Expected {EXPECTED_LAYERS} QuantModule_DiffAE_LoRA layers, got {len(lora_modules)}"
    )
    LOGGER.info("Registered hooks on %d layers, running %d samples …", len(lora_modules), n_samples)

    # ── 4. Accumulation containers ────────────────────────────────────────────
    # acc_act[layer_name][step_idx] = list of stat dicts  (one per sample)
    # acc_err[layer_name][step_idx] = list of error dicts (one per sample)
    # scale_map[layer_name][step_idx] = float scale (from scale_list; same for all samples)
    acc_act: Dict = {}
    acc_err: Dict = {}
    scale_map: Dict = {}
    t_ref: Dict = {"t": -1}

    # ── 5. Register hooks ─────────────────────────────────────────────────────
    handles: List = []
    for layer_name, mod in lora_modules:
        aq = mod.act_quantizer  # TemporalActivationQuantizer
        pre_h, post_h = _make_act_hooks(layer_name, acc_act, acc_err, scale_map)
        handles.append(aq.register_forward_pre_hook(pre_h))
        handles.append(aq.register_forward_hook(post_h))

    # ── 6. Patch t-tracking + run sampling ───────────────────────────────────
    orig_forward = _patch_t_tracking(analysis_model, t_ref)
    try:
        analysis_model.set_quant_state(True, True)

        for s in range(n_samples):
            seed = 42 + s

            # Per-sample noise (batch_size=1 for simplicity; matches the hook
            # accumulation design where each sample contributes one entry per step)
            x_T_bank, latent_noise_bank = _make_noise_banks(
                num_images=1,
                chunk_batch=1,
                img_size=128,
                style_ch=style_ch,
                seed=seed,
                device=device,
            )

            # Reset TemporalActivationQuantizer current_step for this sample
            for m in analysis_model.modules():
                if isinstance(m, TemporalActivationQuantizer):
                    m.current_step = m.total_steps - 1

            t_ref["t"] = -1  # reset t tracker

            # Compute latent conditioning
            if conf.train_mode.is_latent_diffusion():
                cond = _compute_latent_cond(
                    conf=conf,
                    latent_sampler=latent_sampler,
                    latent_net=analysis_model.latent_net,
                    latent_noise_chunk=latent_noise_bank,
                    conds_mean=conds_mean,
                    conds_std=conds_std,
                    device=device,
                )
                model_kwargs: Optional[Dict] = {"cond": cond}
            else:
                model_kwargs = None

            # DDIM progressive loop — hooks fire at every UNet call
            cache_scheduler = getattr(conf, "cache_scheduler", None)
            for _out in sampler.ddim_sample_loop_progressive(
                model=analysis_model,
                noise=x_T_bank,
                clip_denoised=True,
                model_kwargs=model_kwargs,
                device=device,
                progress=False,
                eta=0.0,
                cache_scheduler=cache_scheduler,
            ):
                pass

            print(f"[Inference] Sample {s + 1}/{n_samples} done  (seed={seed})")

    finally:
        analysis_model.forward = orig_forward  # type: ignore[method-assign]
        for h in handles:
            try:
                h.remove()
            except Exception as exc:
                LOGGER.warning("Failed to remove hook: %s", exc)

    # ── 7. Aggregate, save, print ─────────────────────────────────────────────
    LOGGER.info("Aggregating statistics …")
    rows = _aggregate_act_stats(lora_modules, acc_act, acc_err, scale_map)
    LOGGER.info("Total (layer, step) data points: %d", len(rows))

    _save_activation_outputs(rows, output_dir, args)
    _print_activation_summary(rows, n_samples, ddim_steps)

    print(f"  activation_range_stats.csv   → {output_dir / 'activation_range_stats.csv'}")
    print(f"  activation_range_per_layer.csv → {output_dir / 'activation_range_per_layer.csv'}")
    print(f"  activation_range_per_step.csv  → {output_dir / 'activation_range_per_step.csv'}")
    print(f"  activation_range_summary.json  → {output_dir / 'activation_range_summary.json'}")


def run_weight_error_mode(args: argparse.Namespace) -> None:
    """--mode weight-error: round-trip quantisation error analysis (placeholder)."""
    raise NotImplementedError(
        "--mode weight-error is not yet implemented. "
        "It will compute pre/post-quant residuals and report MSE / SNR per layer."
    )


def run_crossref_mode(args: argparse.Namespace) -> None:
    """--mode crossref: correlate range stats with Stage-0 FID sensitivity (placeholder)."""
    raise NotImplementedError(
        "--mode crossref is not yet implemented. "
        "It will load stage0e_output/fid_w_qdiffae_clip.npy and correlate "
        "per-block FID sensitivity with per-layer range statistics."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )


_MODE_HANDLERS = {
    "weight": run_weight_mode,
    "activation": run_activation_mode,
    "weight-error": run_weight_error_mode,
    "crossref": run_crossref_mode,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Range Hypothesis Analysis for Q-DiffAE quantised layers. "
            "Analyse weight / activation ranges to test whether INT8 precision "
            "suffices to represent the effective weight distribution."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=list(_MODE_HANDLERS.keys()),
        required=True,
        help="Analysis mode to run.",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help=(
            "Path to the QAT TT checkpoint (.pth). "
            f"Defaults to CONFIG.BEST_CKPT_PATH_100 ({CONFIG.BEST_CKPT_PATH_100!r})."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="QATcode/quantize_ver2/range_analysis_results",
        help="Directory to write output CSV / JSON files.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="PyTorch device string (e.g. cuda:0, cpu).",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )

    # ── Activation mode arguments ───────────────────────────────────────────
    act_grp = parser.add_argument_group(
        "activation mode",
        "Arguments used only when --mode activation is selected.",
    )
    act_grp.add_argument(
        "--n_samples",
        type=int,
        default=8,
        help=(
            "Number of independent DDIM samples to run. "
            "Each sample uses seed 42, 43, … (42 + n_samples - 1). "
            "Statistics are aggregated across samples."
        ),
    )
    act_grp.add_argument(
        "--ddim_steps",
        type=int,
        default=100,
        help=(
            "Number of DDIM denoising steps per sample. "
            "Must match the number of steps the TemporalActivationQuantizer "
            "was trained with (default: 100)."
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    _setup_logging(args.log_level)

    handler = _MODE_HANDLERS[args.mode]
    handler(args)


if __name__ == "__main__":
    main()
