"""
Weight histogram analysis for all QuantModule_DiffAE_LoRA layers.

For each layer, extracts and plots two-panel histograms:
  Panel 1 (top):    w  vs  w_q          – original weight vs its quantized version
  Panel 2 (bottom): w+LoRA  vs  w+LoRA_q – effective weight vs its quantized version
  Both panels share the same x-axis range for easy comparison.

Summary figure (two subplots):
  Top:    per-layer LoRA delta norm  ‖loraB.weight @ loraA.weight‖₂
  Bottom: per-layer quant-error ratio  ‖w_eff − w_eff_q‖₂ / ‖w − w_q‖₂
          (horizontal dashed line at y=1; below 1 means LoRA reduced quant error)

Per-layer histograms → weight_hist/per_layer/<idx>_<layer>.png
Summary figure       → weight_hist/summary_quant_error.png

Usage:
    python -m QATcode.quantize_ver2.baseline_quant_analysis.analyze_weight_histogram
    python -m QATcode.quantize_ver2.baseline_quant_analysis.analyze_weight_histogram \
        --ckpt QATcode/quantize_ver2/checkpoints/diffae_step6_lora_best.pth
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(".")
sys.path.append("./model")

from QATcode.quantize_ver2.baseline_quant_analysis.pred_xstart_quantile_analysis import (
    CONFIG,
    _load_quant_and_ema_from_ckpt,
    create_float_quantized_model,
    load_calibration_data,
    load_diffae_model,
)
from QATcode.quantize_ver2.quant_model_lora_v2 import QuantModule_DiffAE_LoRA

LOGGER = logging.getLogger("analyze_weight_histogram")

OUT_ROOT = Path("QATcode/quantize_ver2/baseline_quant_analysis/weight_hist")


# ---------------------------------------------------------------------------
# Weight extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def _compute_lora_weight(mod: QuantModule_DiffAE_LoRA) -> torch.Tensor:
    """Reconstruct the LoRA delta weight using the same logic as forward()."""
    device = mod.org_weight.device
    if mod.fwd_func is F.linear:
        # loraA: Linear(in, rank), loraB: Linear(rank, out)
        # Passing identity gives the full weight delta
        E = torch.eye(mod.org_weight.shape[1], device=device)
        lora_weight = mod.loraB(mod.loraA(E))  # [in_features, out_features]
        lora_weight = lora_weight.T             # [out_features, in_features]
    else:
        # loraA: Conv2d(in, rank, kH, kW), loraB: Conv2d(rank, out, 1, 1)
        lora_weight = (
            mod.loraB.weight.squeeze(-1).squeeze(-1)   # [out, rank]
            @ mod.loraA.weight.permute(2, 3, 0, 1)     # [kH, kW, rank, in]
        )                                               # → [out, kH, kW, in] broadcast
        lora_weight = lora_weight.permute(2, 3, 0, 1)  # [out, in, kH, kW]
    return lora_weight.to(device)


@torch.no_grad()
def _extract_four_weights(
    mod: QuantModule_DiffAE_LoRA,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return (w, w_q, w_lora, w_lora_q) as flattened CPU float32 tensors.

      w       = org_weight
      w_q     = weight_quantizer(w)
      w_lora  = org_weight + lora_weight   (effective weight)
      w_lora_q= weight_quantizer(w_lora)
    """
    w = mod.org_weight.detach().float()
    w_q = mod.weight_quantizer(w.clone())

    lora = _compute_lora_weight(mod)
    w_lora = (w + lora).float()
    w_lora_q = mod.weight_quantizer(w_lora.clone())

    return (
        w.cpu().flatten(),
        w_q.cpu().flatten(),
        w_lora.cpu().flatten(),
        w_lora_q.cpu().flatten(),
    )


@torch.no_grad()
def _compute_lora_delta_norm(mod: QuantModule_DiffAE_LoRA) -> float:
    """‖lora_weight‖₂  (the LoRA delta reconstructed from loraA / loraB)."""
    lora = _compute_lora_weight(mod)
    return float(lora.norm())


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_layer_histogram(
    layer_name: str,
    w: torch.Tensor,
    w_q: torch.Tensor,
    w_lora: torch.Tensor,
    w_lora_q: torch.Tensor,
    out_path: Path,
    bins: int = 300,
) -> None:
    """
    Two-panel histogram for one layer.
      Top panel:    w  vs  w_q
      Bottom panel: w+LoRA  vs  w+LoRA_q
    Both panels share the same x-axis range (union of all four distributions).
    """
    # Shared x range: union of all four distributions
    all_vals = torch.cat([w, w_q, w_lora, w_lora_q])
    lo, hi = float(all_vals.min()), float(all_vals.max())
    edges = np.linspace(lo, hi, bins + 1)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True,
        gridspec_kw={"hspace": 0.35},
    )
    fig.suptitle(layer_name, fontsize=8, y=1.01)

    hist_kw = dict(bins=edges, density=True, histtype="step", linewidth=1.3)

    # ── Top: original weight vs its quantized version ─────────────────────
    ax_top.hist(w.numpy(),   label="w  (original)", color="steelblue",  **hist_kw)
    ax_top.hist(w_q.numpy(), label="w_q  (quant)",  color="darkorange", linestyle="--", **hist_kw)
    ax_top.set_ylabel("density")
    ax_top.set_title("original weight", fontsize=8)
    ax_top.legend(fontsize=7)

    # ── Bottom: effective weight (w+LoRA) vs its quantized version ────────
    ax_bot.hist(w_lora.numpy(),   label="w+LoRA  (effective)",   color="seagreen", **hist_kw)
    ax_bot.hist(w_lora_q.numpy(), label="w+LoRA_q (post-quant)", color="crimson",  linestyle="--", **hist_kw)
    ax_bot.set_xlabel("weight value")
    ax_bot.set_ylabel("density")
    ax_bot.set_title("LoRA-effective weight", fontsize=8)
    ax_bot.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    LOGGER.debug("Saved histogram → %s", out_path)


def _plot_summary(
    layer_names: List[str],
    lora_norms: List[float],
    err_base: List[float],
    err_lora: List[float],
    out_path: Path,
) -> None:
    """
    Two-subplot summary figure.
      Top:    per-layer LoRA delta norm  ‖loraB @ loraA‖₂
      Bottom: per-layer quant-error ratio  ‖w_eff − w_eff_q‖₂ / ‖w − w_q‖₂
              (dashed y=1 baseline; below 1 = LoRA reduced quantization error)
    """
    n = len(layer_names)
    fig_w = max(10, n * 0.28)
    x = np.arange(n)

    # Guard against division-by-zero for any layer where base error is 0
    ratios = [
        (el / eb) if eb > 0.0 else float("nan")
        for eb, el in zip(err_base, err_lora)
    ]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(fig_w, 8),
        gridspec_kw={"hspace": 0.55},
    )

    # ── Top: LoRA delta norm ───────────────────────────────────────────────
    ax_top.bar(x, lora_norms, color="mediumpurple", alpha=0.85)
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(layer_names, rotation=90, fontsize=5)
    ax_top.set_ylabel("‖LoRA delta‖₂")
    ax_top.set_title("Per-layer LoRA delta norm  ‖loraB @ loraA‖₂")

    # ── Bottom: quant-error ratio ──────────────────────────────────────────
    ax_bot.bar(x, ratios, color="steelblue", alpha=0.85)
    ax_bot.axhline(y=1.0, color="crimson", linestyle="--", linewidth=1.2, label="ratio = 1  (no change)")
    ax_bot.set_xticks(x)
    ax_bot.set_xticklabels(layer_names, rotation=90, fontsize=5)
    ax_bot.set_ylabel("‖w_eff − w_eff_q‖₂  /  ‖w − w_q‖₂")
    ax_bot.set_title("Per-layer quant-error ratio  (< 1 = LoRA reduced quant error)")
    ax_bot.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved summary plot → %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )


def main(args: argparse.Namespace) -> None:
    _setup_logging()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info("Device: %s", device)

    # ── 1. Build model ─────────────────────────────────────────────────────
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

    # ── 2. Calibrate (init TemporalActivationQuantizer scales) ────────────
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

    # ── 3. Load checkpoint ─────────────────────────────────────────────────
    ckpt_path = args.ckpt or CONFIG.BEST_CKPT_PATH_100
    LOGGER.info("Loading QAT checkpoint: %s", ckpt_path)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _load_quant_and_ema_from_ckpt(base_model, quant_model, ckpt)
    LOGGER.info("Checkpoint loaded successfully")

    # Use EMA model for analysis (consistent with inference path)
    analysis_model = base_model.ema_model
    analysis_model.to(device).eval()

    if hasattr(analysis_model, "set_runtime_mode"):
        analysis_model.set_runtime_mode(mode="infer", use_cached_aw=True, clear_cached_aw=True)

    # ── 4. Collect all QuantModule_DiffAE_LoRA layers ─────────────────────
    lora_modules = [
        (name, mod)
        for name, mod in analysis_model.named_modules()
        if isinstance(mod, QuantModule_DiffAE_LoRA)
    ]
    LOGGER.info("Found %d QuantModule_DiffAE_LoRA layers", len(lora_modules))

    # ── 5. Per-layer histograms ────────────────────────────────────────────
    per_layer_dir = OUT_ROOT / "per_layer"
    per_layer_dir.mkdir(parents=True, exist_ok=True)

    layer_names: List[str] = []
    lora_norms: List[float] = []
    err_base: List[float] = []
    err_lora: List[float] = []

    for idx, (name, mod) in enumerate(lora_modules):
        LOGGER.info("[%d/%d] %s", idx + 1, len(lora_modules), name)

        w, w_q, w_lora, w_lora_q = _extract_four_weights(mod)

        safe = name.replace(".", "_")
        png_path = per_layer_dir / f"{idx:03d}_{safe}.png"
        _plot_layer_histogram(name, w, w_q, w_lora, w_lora_q, png_path)

        err_base.append(float((w - w_q).norm()))
        err_lora.append(float((w_lora - w_lora_q).norm()))
        lora_norms.append(_compute_lora_delta_norm(mod))
        # Short label for the summary x-axis: last two path segments
        parts = name.split(".")
        short = ".".join(parts[-2:]) if len(parts) >= 2 else name
        layer_names.append(f"{idx}:{short}")

    LOGGER.info("Saved %d per-layer PNGs → %s", len(lora_modules), per_layer_dir)

    # ── 6. Summary plot ────────────────────────────────────────────────────
    summary_png = OUT_ROOT / "summary_quant_error.png"
    _plot_summary(layer_names, lora_norms, err_base, err_lora, summary_png)

    LOGGER.info("Done.  Output root: %s", OUT_ROOT)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Weight histogram analysis for all QuantModule_DiffAE_LoRA layers"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help=(
            "Path to QAT checkpoint .pth file. "
            "Defaults to CONFIG.BEST_CKPT_PATH_100 "
            f"({CONFIG.BEST_CKPT_PATH_100!r})"
        ),
    )
    args = parser.parse_args()
    main(args)
